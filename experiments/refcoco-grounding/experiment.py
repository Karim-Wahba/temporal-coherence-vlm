"""
experiment.py
-------------
Core runner for the RefCOCO grounding experiment.

For each (image, expression) item it:
  1. Runs RefCOCORunner.run_with_tam() — one forward pass for both the
     predicted bbox and the TAM heatmap of the label tokens.
  2. Finds which generated tokens form the "label" value in the JSON output.
  3. Averages TAM heatmaps across multi-token labels.
     For a single image, each tam_map has shape (1, H_tam, W_tam);
     the 2D heatmap is tam_map[0].
  4. Computes IoU, mass-in-GT, mass-in-pred, attention entropy.
  5. Saves a two-column visualisation figure per item (sampled only).
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from refcoco_loader import RefCOCOItem, RefCOCOLoader
from refcoco_runner import RefCOCORunner, _find_label_token_indices
from metrics import (
    compute_iou,
    compute_mass_in_gt,
    compute_attention_entropy,
    compute_mass_accuracy_correlation,
    accuracy_at_threshold,
)
from visualizer import save_item_figure, save_scatter_plots


class RefCOCOGroundingExperiment:
    """
    Parameters
    ----------
    model, processor : loaded Qwen model and processor
    refcoco_root     : path to the RefCOCO data root
    save_dir         : output directory; visualisations go in save_dir/visualizations/
    dataset          : "refcoco" | "refcoco+" | "refcocog"
    split            : "train" | "val" | "testA" | "testB"
    max_new_tokens   : generation budget
    vis_every        : save a visualisation every N items (0 = never)
    """

    def __init__(
        self,
        model,
        processor,
        refcoco_root: str,
        save_dir: str,
        dataset: str = "refcoco",
        split: str = "val",
        max_new_tokens: int = 256,
        vis_every: int = 50,
    ):
        self.save_dir = Path(save_dir)
        self.vis_dir  = self.save_dir / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.runner = RefCOCORunner(model, processor, max_new_tokens=max_new_tokens)
        self.dataset = dataset
        self.split   = split
        self.vis_every = vis_every

    # ── per-item ──────────────────────────────────────────────────────────────

    def run_item(self, item: RefCOCOItem, item_idx: int) -> dict:
        """Run the full pipeline for one (image, expression) item."""
        print(f"  [{item_idx}] img={item.image_id} sent={item.sent_id}"
              f" | \"{item.expression[:60]}\"")

        # 1. Inference + TAM
        try:
            pred_box, raw_text, tam_result = self.runner.run_with_tam(
                item.image_pil, item.expression
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            traceback.print_exc()
            return {
                "ref_id": item.ref_id, "sent_id": item.sent_id,
                "image_id": item.image_id, "expression": item.expression,
                "error": str(e),
            }

        gen_tokens = tam_result["gen_tokens"]
        tam_maps   = tam_result["tam_maps"]    # List[(1,H_tam,W_tam) or None]
        # vision_shape = (1, H_tam, W_tam) for a single image

        # 2. Find label token indices in the generated output
        label_tok_idxs = _find_label_token_indices(gen_tokens)

        # 3. Average TAM heatmaps over label tokens → one 2D heatmap
        valid_idxs = [
            i for i in label_tok_idxs
            if i < len(tam_maps)
            and tam_maps[i] is not None
            and tam_maps[i].ndim == 3
            and tam_maps[i].shape[0] >= 1
        ]

        heatmap: Optional[np.ndarray] = None
        label_tokens: List[str] = []
        if valid_idxs:
            slices = [tam_maps[i][0].astype(np.float32) for i in valid_idxs]
            avg = np.mean(slices, axis=0)
            max_val = avg.max()
            if max_val > 0:
                avg /= max_val
            heatmap = avg
            label_tokens = [gen_tokens[i] for i in valid_idxs]

        # 4. Metrics
        H, W = item.image_size()
        iou = compute_iou(pred_box, item.gt_box)

        mass_in_gt   = compute_mass_in_gt(heatmap, item.gt_box,  H, W) if heatmap is not None else None
        mass_in_pred = compute_mass_in_gt(heatmap, pred_box,     H, W) if heatmap is not None else None
        entropy      = compute_attention_entropy(heatmap) if heatmap is not None else None

        # 5. Visualisation (sampled)
        vis_path = None
        if self.vis_every > 0 and item_idx % self.vis_every == 0:
            vis_path = str(self.vis_dir / f"img{item.image_id}_sent{item.sent_id}.png")
            try:
                save_item_figure(
                    image_pil=item.image_pil,
                    expression=item.expression,
                    pred_box=pred_box,
                    gt_box=item.gt_box,
                    heatmap=heatmap,
                    label_tokens=label_tokens,
                    save_path=vis_path,
                    image_id=item.image_id,
                )
            except Exception as e:
                print(f"  [WARN] visualisation failed: {e}")

        return {
            "ref_id":      item.ref_id,
            "sent_id":     item.sent_id,
            "image_id":    item.image_id,
            "ann_id":      item.ann_id,
            "expression":  item.expression,
            "split":       item.split,
            "iou":         iou,
            "pred_box":    list(pred_box) if pred_box else None,
            "gt_box":      list(item.gt_box),
            "mass_in_gt":  mass_in_gt,
            "mass_in_pred": mass_in_pred,
            "entropy":     entropy,
            "has_heatmap": heatmap is not None,
            "label_tokens": label_tokens,
            "gen_text":    tam_result["gen_text"],
            "raw_text":    raw_text,
            "vis_path":    vis_path,
        }

    # ── batch ─────────────────────────────────────────────────────────────────

    def run_all(
        self,
        refcoco_root: str,
        max_items: Optional[int] = None,
        sents_per_ref: Optional[int] = 1,
    ) -> List[dict]:
        loader = RefCOCOLoader(
            refcoco_root=refcoco_root,
            dataset=self.dataset,
            split=self.split,
            max_items=max_items,
            sents_per_ref=sents_per_ref,
        )
        print(f"Dataset summary: {loader.summary()}")
        print(f"Running RefCOCO grounding on {len(loader)} items")

        all_results = []
        for idx, item in enumerate(loader):
            result = self.run_item(item, idx)
            all_results.append(result)

            # Incremental save so results survive partial runs
            out_path = self.save_dir / "results.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        return all_results

    # ── summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[dict]) -> dict:
        valid = [r for r in results if "error" not in r]

        ious         = [r["iou"] for r in valid]
        mass_gt_all  = [r["mass_in_gt"]   for r in valid if r.get("mass_in_gt")   is not None]
        mass_pred_all= [r["mass_in_pred"] for r in valid if r.get("mass_in_pred") is not None]
        entropy_all  = [r["entropy"]      for r in valid if r.get("entropy")       is not None]

        # Correlation: mass-in-GT vs IoU
        iou_for_mgt  = [r["iou"] for r in valid if r.get("mass_in_gt")  is not None]
        iou_for_mpred= [r["iou"] for r in valid if r.get("mass_in_pred")is not None]
        mass_acc_corr      = compute_mass_accuracy_correlation(mass_gt_all,   iou_for_mgt)
        mass_pred_acc_corr = compute_mass_accuracy_correlation(mass_pred_all, iou_for_mpred)

        return {
            "n_items":  len(valid),
            "n_errors": len(results) - len(valid),
            # Q1
            "mean_iou":       float(np.mean(ious)) if ious else None,
            "acc@0.25":       accuracy_at_threshold(ious, 0.25),
            "acc@0.5":        accuracy_at_threshold(ious, 0.5),
            "acc@0.75":       accuracy_at_threshold(ious, 0.75),
            # Q2
            "mean_mass_in_gt":   float(np.mean(mass_gt_all))   if mass_gt_all   else None,
            "mean_mass_in_pred": float(np.mean(mass_pred_all)) if mass_pred_all else None,
            "mean_entropy":      float(np.mean(entropy_all))   if entropy_all   else None,
            "frac_with_heatmap": float(sum(r["has_heatmap"] for r in valid) / len(valid))
                                 if valid else None,
            # Q3
            "mass_accuracy_correlation":      mass_acc_corr,
            "mass_pred_accuracy_correlation": mass_pred_acc_corr,
        }
