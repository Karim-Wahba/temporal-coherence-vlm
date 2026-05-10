"""
experiment.py
-------------
Grounding-stability experiment for Ref-YouTube-VOS.

Identical pipeline to grounding-stability/experiment.py; the only
differences are:
  • Uses RefYouTubeVOSLoader (masks → bboxes) instead of DAVISVOTLoader.
  • fps defaults to 6.0 (YouTube-VOS annotates every 5th frame of 30 fps).

For each (video, expression) pair it:
  1. Runs QwenVOTRunner.run_with_tam() — one forward pass.
  2. Parses JSON output to find label tokens per frame.
  3. Averages TAM heatmaps over multi-token labels.
  4. Computes IoU, TAM instability, mass-in-GT, mass-in-pred, entropy,
     and Pearson/Spearman correlations.
  5. Saves per-sequence visualisation figures.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_GS      = _HERE.parent / "grounding-stability"
_REFDAVIS= _HERE.parent / "Ref-DAVIS"

sys.path.insert(0, str(_REFDAVIS))
sys.path.insert(0, str(_GS))          # token_parser, metrics, visualizer
sys.path.insert(0, str(_HERE))        # loader (must shadow nothing in _GS)

from benchmark.qwen_vot_runner import QwenVOTRunner
from loader import RefYouTubeVOSLoader, RefYouTubeVOSItem, ANNOTATED_FPS

# import shared utilities from grounding-stability
from token_parser import parse_frame_labels, find_label_token_indices
from metrics import (
    frame_iou_series,
    compute_mass_in_gt,
    compute_tam_instability,
    compute_correlations,
    compute_attention_entropy,
    compute_mass_accuracy_correlation,
)
from visualizer import save_sequence_figure


class RefYouTubeVOSGroundingExperiment:
    """
    Parameters
    ----------
    model, processor : loaded Qwen model and processor
    data_root        : Ref-YouTube-VOS root (contains train/ and valid/)
    save_dir         : output directory
    video_mode       : True → single video block (3D RoPE)
    sample_rate      : send every Nth *annotated* frame to the model
    max_new_tokens   : generation budget
    split            : "valid" or "train"
    """

    def __init__(
        self,
        model,
        processor,
        data_root: str,
        save_dir: str,
        video_mode: bool = False,
        sample_rate: int = 2,
        max_new_tokens: int = 4096,
        split: str = "valid",
    ):
        self.save_dir = Path(save_dir)
        self.vis_dir  = self.save_dir / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.runner = QwenVOTRunner(
            model, processor,
            max_new_tokens=max_new_tokens,
            fps=ANNOTATED_FPS,
            sample_rate=sample_rate,
            video_mode=video_mode,
        )
        self.loader = RefYouTubeVOSLoader(data_root, split=split)
        self.sample_rate = sample_rate
        self.video_mode  = video_mode
        self.fps         = self.runner.fps

    # ── per-sequence ──────────────────────────────────────────────────────────

    def run_sequence(self, item: RefYouTubeVOSItem) -> dict:
        prefix = f"{item.seq_name}_exp{item.exp_id}"
        print(f"  Running {prefix} | \"{item.expression[:60]}\"")

        # 1. Inference + TAM
        try:
            boxes, raw_text, tam_result = self.runner.run_with_tam(
                item.frames_pil, item.expression
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            traceback.print_exc()
            return {"seq_name": item.seq_name, "exp_id": item.exp_id,
                    "expression": item.expression, "error": str(e)}

        gen_tokens = tam_result["gen_tokens"]
        tam_maps   = tam_result["tam_maps"]
        vision_T   = tam_result["vision_shape"][0]

        # 2. Parse label tokens per frame
        parsed_entries = parse_frame_labels(
            tam_result["gen_text"],
            fps=self.fps,
            sample_rate=self.sample_rate,
        )
        label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

        # 3. Build per-frame averaged heatmaps
        frame_heatmaps: Dict[int, np.ndarray] = {}
        frame_label_tokens: Dict[int, List[str]] = {}

        for sampled_t, tok_idxs in label_token_map:
            if sampled_t >= vision_T:
                continue
            valid_idxs = [
                i for i in tok_idxs
                if i < len(tam_maps)
                and tam_maps[i] is not None
                and tam_maps[i].ndim == 3
                and sampled_t < tam_maps[i].shape[0]
            ]
            if not valid_idxs:
                continue
            slices  = [tam_maps[i][sampled_t].astype(np.float32) for i in valid_idxs]
            avg_map = np.mean(slices, axis=0)
            max_val = avg_map.max()
            if max_val > 0:
                avg_map /= max_val
            frame_heatmaps[sampled_t]     = avg_map
            frame_label_tokens[sampled_t] = [gen_tokens[i] for i in valid_idxs]

        # 4. IoU per annotated frame
        H, W = item.frame_size()
        iou_series     = frame_iou_series(boxes, item.gt_boxes)
        sampled_indices= list(range(0, item.num_frames, self.sample_rate))
        sampled_iou    = iou_series[sampled_indices] if sampled_indices else iou_series

        # 5. Mass-in-GT / mass-in-pred
        mass_in_gt:   Dict[int, Optional[float]] = {}
        mass_in_pred: Dict[int, Optional[float]] = {}
        for sampled_t, hmap in frame_heatmaps.items():
            orig_t   = sampled_t * self.sample_rate
            gt_box   = item.gt_boxes[orig_t] if orig_t < item.num_frames else None
            pred_box = boxes[orig_t]          if orig_t < len(boxes)      else None
            mass_in_gt[sampled_t]   = compute_mass_in_gt(hmap, gt_box,   H, W)
            mass_in_pred[sampled_t] = compute_mass_in_gt(hmap, pred_box, H, W)

        # 6. TAM instability between consecutive detected frames
        instability = compute_tam_instability(frame_heatmaps)

        # 7. Instability vs IoU failure correlation
        inst_vals, failure_vals = [], []
        for (t, t1), inst_val in instability.items():
            orig_t    = t * self.sample_rate
            iou_fail  = 1.0 - float(iou_series[orig_t]) if orig_t < len(iou_series) else 1.0
            inst_vals.append(inst_val)
            failure_vals.append(iou_fail)
        correlations = compute_correlations(inst_vals, failure_vals)

        # 8. Attention entropy
        entropy_per_frame = {
            t: compute_attention_entropy(hm) for t, hm in frame_heatmaps.items()
        }
        mean_entropy = float(np.mean(list(entropy_per_frame.values()))) \
                       if entropy_per_frame else None

        # 9. Attention accuracy: mass-in-GT vs IoU, mass-in-pred vs IoU
        mass_acc_vals, iou_acc_vals   = [], []
        mass_pred_vals, iou_pred_vals = [], []
        for sampled_t in sorted(frame_heatmaps.keys()):
            orig_t = sampled_t * self.sample_rate
            if orig_t >= len(iou_series):
                continue
            iou_val = float(iou_series[orig_t])
            m_gt    = mass_in_gt.get(sampled_t)
            m_pred  = mass_in_pred.get(sampled_t)
            if m_gt   is not None: mass_acc_vals.append(m_gt);   iou_acc_vals.append(iou_val)
            if m_pred is not None: mass_pred_vals.append(m_pred); iou_pred_vals.append(iou_val)

        mass_accuracy_corr      = compute_mass_accuracy_correlation(mass_acc_vals,  iou_acc_vals)
        mass_pred_accuracy_corr = compute_mass_accuracy_correlation(mass_pred_vals, iou_pred_vals)

        # 10. Visualisation
        detected_sampled = sorted(frame_heatmaps.keys())
        pred_boxes_dict  = {t: boxes[t] for t in range(item.num_frames)}
        vis_path         = str(self.vis_dir / f"{prefix}.png")

        try:
            save_sequence_figure(
                seq_name=item.seq_name, expression=item.expression,
                frames_pil=item.frames_pil,
                detected_sampled_frames=detected_sampled,
                pred_boxes=pred_boxes_dict, gt_boxes=item.gt_boxes,
                frame_heatmaps=frame_heatmaps, label_tokens=frame_label_tokens,
                sample_rate=self.sample_rate, save_path=vis_path,
            )
        except Exception as e:
            print(f"  [WARN] visualisation failed: {e}")

        return {
            "seq_name": item.seq_name, "exp_id": item.exp_id,
            "expression": item.expression,
            "num_frames": item.num_frames,
            "num_detected_frames": len(detected_sampled),
            "sample_rate": self.sample_rate,
            "fps": self.fps,
            "mean_iou":      float(sampled_iou.mean()),
            "iou_variance":  float(sampled_iou.var()),
            "iou_per_frame": iou_series.tolist(),
            "mass_in_gt":    {str(k): v for k, v in mass_in_gt.items()},
            "mean_mass_in_gt": float(np.nanmean([v for v in mass_in_gt.values() if v is not None]))
                               if mass_in_gt else None,
            "mass_in_pred":  {str(k): v for k, v in mass_in_pred.items()},
            "mean_mass_in_pred": float(np.nanmean([v for v in mass_in_pred.values() if v is not None]))
                                 if mass_in_pred else None,
            "instability":   {f"{t}-{t1}": v for (t, t1), v in instability.items()},
            "mean_instability": float(np.mean(list(instability.values()))) if instability else None,
            "entropy_per_frame": {str(k): v for k, v in entropy_per_frame.items()},
            "mean_entropy":  mean_entropy,
            "correlations":  correlations,
            "mass_accuracy_correlation":      mass_accuracy_corr,
            "mass_pred_accuracy_correlation": mass_pred_accuracy_corr,
            "gen_text":     tam_result["gen_text"],
            "vis_path":     vis_path,
            "label_tokens": {str(k): v for k, v in frame_label_tokens.items()},
        }

    # ── batch ─────────────────────────────────────────────────────────────────

    def run_all(
        self,
        sequences: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        expressions_per_seq: Optional[int] = 1,
        resume: bool = True,
        retry_errors: bool = False,
    ) -> List[dict]:
        """
        Parameters
        ----------
        resume       : If True (default) and {save_dir}/results.json exists,
                       load it and skip items already present, so a killed
                       SLURM job can be re-submitted and pick up where it
                       stopped.
        retry_errors : If True, re-run items previously recorded with an
                       "error" key (transient OOM / timeout etc.).
        """
        items = list(self.loader)
        if sequences:
            items = [it for it in items if it.seq_name in sequences]
        if expressions_per_seq is not None:
            seen: Dict[str, int] = {}
            filtered = []
            for it in items:
                c = seen.get(it.seq_name, 0)
                if c < expressions_per_seq:
                    filtered.append(it)
                    seen[it.seq_name] = c + 1
            items = filtered
        if max_sequences is not None:
            unique: List[str] = []
            filtered = []
            for it in items:
                if it.seq_name not in unique:
                    if len(unique) >= max_sequences:
                        continue
                    unique.append(it.seq_name)
                filtered.append(it)
            items = filtered

        # ── Resume from existing results.json ────────────────────────────
        out_path = self.save_dir / "results.json"
        all_results: List[dict] = []
        completed_keys: set = set()

        if resume and out_path.exists():
            try:
                with open(out_path) as f:
                    all_results = json.load(f)
                if retry_errors:
                    # Drop errored entries so they get re-run and re-appended
                    all_results = [r for r in all_results if "error" not in r]
                for r in all_results:
                    completed_keys.add((r.get("seq_name"), str(r.get("exp_id"))))
                print(
                    f"Resume: loaded {len(all_results)} existing results"
                    f" ({len(completed_keys)} items already done)"
                )
            except Exception as e:
                print(f"[WARN] failed to load {out_path} ({e}); starting fresh")
                all_results = []
                completed_keys = set()

        items_to_run = [
            it for it in items
            if (it.seq_name, str(it.exp_id)) not in completed_keys
        ]
        skipped = len(items) - len(items_to_run)
        print(f"Running Ref-YouTube-VOS grounding-stability on {len(items_to_run)} "
              f"items (skipped {skipped} already done)")

        for idx, item in enumerate(items_to_run):
            print(f"\n[{idx+1}/{len(items_to_run)}] {item.seq_name}")
            result = self.run_sequence(item)
            all_results.append(result)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        return all_results

    # ── summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[dict]) -> dict:
        # Reuse identical logic from grounding-stability; import at call time
        # to avoid a circular path issue.
        sys.path.insert(0, str(Path(__file__).parent.parent / "grounding-stability"))
        from experiment import GroundingStabilityExperiment
        return GroundingStabilityExperiment.summarize(results)
