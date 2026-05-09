"""
experiment.py
-------------
Per-(sequence, expression) inference for the expression-variance experiment.

For every DAVIS (seq, exp) item we run one forward pass with TAM, then record:
  - mean IoU across frames
  - mean attention mass-in-GT across detected frames
  - mean attention mass-in-pred across detected frames

Heatmaps for the mass metrics are built by averaging all label-noun token
positions per frame (the 'none' strategy from grounding-stability-max).

The output `results.json` is a flat list of one row per (seq, exp). Grouping
by (seq, obj_id) and aggregating is done downstream by analyze.py.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE       = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
_REF_DAVIS  = _EXPERIMENTS / "Ref-DAVIS"
_GS_MAX     = _EXPERIMENTS / "grounding-stability-max"

for p in [str(_REF_DAVIS), str(_REF_DAVIS / "benchmark"), str(_GS_MAX)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem  # noqa: E402
from benchmark.qwen_vot_runner import QwenVOTRunner                  # noqa: E402

# Reuse parsing + metrics from grounding-stability-max.
from token_parser import parse_frame_labels, find_label_token_indices  # noqa: E402
from metrics import frame_iou_series, compute_mass_in_gt              # noqa: E402


class ExpressionVarianceExperiment:
    def __init__(
        self,
        model,
        processor,
        davis_root: str,
        save_dir: str,
        video_mode: bool = True,
        sample_rate: int = 8,
        max_new_tokens: int = 4096,
        split: str = "valid",
        seed: int = 0,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runner = QwenVOTRunner(
            model, processor,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            video_mode=video_mode,
            seed=seed,
        )
        self.loader = DAVISVOTLoader(davis_root, split=split)
        self.sample_rate = sample_rate
        self.fps = self.runner.fps

    # ── per-item ──────────────────────────────────────────────────────────────

    def run_item(self, item: DAVISVOTItem) -> dict:
        prefix = f"{item.seq_name}_obj{item.obj_id}_exp{item.exp_id}"
        print(f"  Running {prefix} | \"{item.expression[:60]}\"")

        try:
            boxes, raw_text, tam_result = self.runner.run_with_tam(
                item.frames_pil, item.expression
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            traceback.print_exc()
            return {
                "seq_name": item.seq_name,
                "obj_id":   item.obj_id,
                "exp_id":   item.exp_id,
                "expression": item.expression,
                "error":    str(e),
            }

        gen_tokens = tam_result["gen_tokens"]
        tam_maps   = tam_result["tam_maps"]
        vision_T   = tam_result["vision_shape"][0]

        parsed_entries  = parse_frame_labels(
            tam_result["gen_text"],
            fps=self.fps,
            sample_rate=self.sample_rate,
        )
        label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

        # Build per-frame heatmap by averaging all label tokens for that frame.
        frame_heatmaps = {}
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
            mx = avg_map.max()
            if mx > 0:
                avg_map /= mx
            frame_heatmaps[sampled_t] = avg_map

        # IoU per original frame, averaged over sampled positions.
        H, W = item.frame_size()
        iou_series      = frame_iou_series(boxes, item.gt_boxes)
        sampled_indices = list(range(0, item.num_frames, self.sample_rate))
        sampled_iou     = iou_series[sampled_indices] if sampled_indices else iou_series

        # Mass-in-GT, mass-in-pred per detected frame.
        m_gt_vals: List[float]   = []
        m_pred_vals: List[float] = []
        for sampled_t, hmap in frame_heatmaps.items():
            orig_t   = sampled_t * self.sample_rate
            gt_box   = item.gt_boxes[orig_t] if orig_t < item.num_frames else None
            pred_box = boxes[orig_t]          if orig_t < len(boxes)     else None
            mg = compute_mass_in_gt(hmap, gt_box,   H, W)
            mp = compute_mass_in_gt(hmap, pred_box, H, W)
            if mg is not None: m_gt_vals.append(float(mg))
            if mp is not None: m_pred_vals.append(float(mp))

        return {
            "seq_name":           item.seq_name,
            "obj_id":             int(item.obj_id),
            "exp_id":             item.exp_id,
            "expression":         item.expression,
            "num_frames":         item.num_frames,
            "num_detected_frames": len(frame_heatmaps),
            "mean_iou":           float(sampled_iou.mean()) if len(sampled_iou) else None,
            "mean_mass_in_gt":    float(np.mean(m_gt_vals))  if m_gt_vals  else None,
            "mean_mass_in_pred":  float(np.mean(m_pred_vals)) if m_pred_vals else None,
        }

    # ── full sweep ────────────────────────────────────────────────────────────

    def run_all(
        self,
        sequences: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        expressions_per_seq: Optional[int] = None,
    ) -> List[dict]:
        items = list(self.loader)
        if sequences:
            items = [it for it in items if it.seq_name in sequences]
        if expressions_per_seq is not None:
            from collections import Counter
            kept, count = [], Counter()
            for it in items:
                key = (it.seq_name, it.obj_id)
                if count[key] < expressions_per_seq:
                    kept.append(it)
                    count[key] += 1
            items = kept
        if max_sequences is not None:
            seen = []
            kept = []
            for it in items:
                if it.seq_name not in seen:
                    if len(seen) >= max_sequences:
                        continue
                    seen.append(it.seq_name)
                kept.append(it)
            items = kept

        print(f"Running expression-variance on {len(items)} items "
              f"({len({(it.seq_name, it.obj_id) for it in items})} (seq,obj) groups)")

        results = []
        out_path = self.save_dir / "results.json"
        for i, item in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {item.seq_name} obj{item.obj_id} exp{item.exp_id}")
            row = self.run_item(item)
            results.append(row)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        return results
