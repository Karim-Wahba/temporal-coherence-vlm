"""
experiment.py
-------------
Layer-ablation runner. For each (sequence, expression):

  1. Run Qwen3-VL once (single forward pass with output_hidden_states=True).
  2. Parse predicted boxes + label-noun token positions per frame.
  3. For each requested variant (per-layer or cumavg-of-last-K), call TAM
     using that variant's logit-lens, then build per-frame heatmaps by
     averaging the label-noun tokens of that frame (mode: "none" / nouns).
  4. Compute mass-in-GT and mass-in-pred per detected frame per variant.
  5. Save per-sequence layer-grid heatmap figures + per-sequence metrics JSON.

Aggregation across sequences is done by visualizer.save_layer_curves().
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_GS_MAX = _HERE.parent / "grounding-stability-max"
_REF_DAVIS = _HERE.parent / "Ref-DAVIS"

sys.path.insert(0, str(_REF_DAVIS))
sys.path.insert(0, str(_REF_DAVIS / "benchmark"))
sys.path.insert(0, str(_GS_MAX))   # token_parser, metrics
sys.path.insert(0, str(_HERE))

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem
from benchmark.qwen_vot_runner import QwenVOTRunner

from token_parser import parse_frame_labels, find_label_token_indices
from metrics import compute_mass_in_gt, frame_iou_series

from layer_tam import extract_layer_tam
from visualizer import save_layer_grid_figure


def _avg_label_token_heatmap(
    label_token_map: List[Tuple[int, List[int]]],
    tam_maps: list,
    vision_T: int,
) -> Dict[int, np.ndarray]:
    """
    Build per-frame heatmap by averaging the label-noun tokens for each frame.
    Mirrors the `none` branch of grounding-stability-max/experiment.py.
    Returns {sampled_t: H_tam x W_tam float32, peak-normalised to [0,1]}.
    """
    out: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in label_token_map:
        if sampled_t >= vision_T:
            continue
        valid = [
            i for i in tok_idxs
            if i < len(tam_maps)
            and tam_maps[i] is not None
            and tam_maps[i].ndim == 3
            and sampled_t < tam_maps[i].shape[0]
        ]
        if not valid:
            continue
        slices = [tam_maps[i][sampled_t].astype(np.float32) for i in valid]
        avg = np.mean(slices, axis=0)
        m = avg.max()
        if m > 0:
            avg = avg / m
        out[sampled_t] = avg
    return out


class GroundingStabilityLayersExperiment:
    """
    Parameters
    ----------
    model, processor : loaded Qwen3-VL model + processor
    davis_root       : DAVIS root (with Annotations_bbox)
    save_dir         : output directory
    video_mode       : True → 3D-RoPE single-video block; False → image-mode
    sample_rate      : send every Nth frame to model
    max_new_tokens   : generation budget
    split            : "valid" / "train"
    layer_indices    : negative ints — single-layer variants to ablate
    cumavg_Ks        : positive ints — cumavg-over-last-K variants
    apply_norm       : apply final RMSNorm before LM head (logit-lens recipe)
    save_heatmap_grid: save per-sequence layer-grid PNGs (heavy; off in metric-only runs)
    """

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
        layer_indices: Optional[List[int]] = None,
        cumavg_Ks: Optional[List[int]] = None,
        apply_norm: bool = True,
        save_heatmap_grid: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.vis_dir = self.save_dir / "visualizations"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.runner = QwenVOTRunner(
            model, processor,
            max_new_tokens=max_new_tokens,
            sample_rate=sample_rate,
            video_mode=video_mode,
        )
        self.loader = DAVISVOTLoader(davis_root, split=split)
        self.model = model
        self.processor = processor
        self.sample_rate = sample_rate
        self.video_mode = video_mode
        self.fps = self.runner.fps
        self.layer_indices = layer_indices or list(range(-1, -11, -1))
        self.cumavg_Ks = cumavg_Ks or list(range(1, 11))
        self.apply_norm = apply_norm
        self.save_heatmap_grid = save_heatmap_grid

    # ── per-sequence ──────────────────────────────────────────────────────────

    def run_sequence(self, item: DAVISVOTItem) -> dict:
        prefix = f"{item.seq_name}_exp{item.exp_id}"
        print(f"  Running {prefix} | \"{item.expression[:60]}\"")

        # 1. One generation pass — reuse QwenVOTRunner internals
        try:
            sampled_frames = item.frames_pil[::self.sample_rate]
            messages, is_video = self.runner._build_messages(sampled_frames, item.expression)
            raw, inputs, outputs = self.runner._generate(
                messages, is_video=is_video, return_hidden_states=True
            )
        except Exception as e:
            print(f"  [ERROR] inference failed: {e}")
            traceback.print_exc()
            return {"seq_name": item.seq_name, "exp_id": item.exp_id,
                    "expression": item.expression, "error": str(e)}

        # 2. Parse boxes
        from benchmark.qwen_vot_runner import (
            _parse_frame_detections, _parse_time_detections,
            _map_frame_detections_to_frames, _map_detections_to_frames,
        )
        N = item.num_frames
        W, H = item.frames_pil[0].size
        if self.video_mode:
            detections = _parse_frame_detections(raw)
            boxes = _map_frame_detections_to_frames(detections, N, W, H, self.sample_rate)
        else:
            detections = _parse_time_detections(raw)
            boxes = _map_detections_to_frames(detections, N, W, H,
                                              fps=self.fps, sample_rate=self.sample_rate)

        # 3. Multi-variant TAM
        try:
            multi = extract_layer_tam(
                inputs, outputs, sampled_frames,
                self.model, self.processor,
                layer_indices=self.layer_indices,
                cumavg_Ks=self.cumavg_Ks,
                apply_norm=self.apply_norm,
                verbose=True,
            )
        except Exception as e:
            print(f"  [ERROR] layer-tam failed: {e}")
            traceback.print_exc()
            return {"seq_name": item.seq_name, "exp_id": item.exp_id,
                    "expression": item.expression, "error": str(e)}

        gen_tokens = multi["gen_tokens"]
        gen_text = multi["gen_text"]
        vision_T = multi["vision_shape"][0]

        # 4. Label-noun token positions per detected frame
        parsed_entries = parse_frame_labels(
            gen_text, fps=self.fps, sample_rate=self.sample_rate
        )
        label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

        # 5. IoU per original frame
        iou_series = frame_iou_series(boxes, item.gt_boxes)

        # 6. Per-variant per-frame heatmap + masses
        H_orig, W_orig = item.frame_size()
        per_variant: Dict[str, dict] = {}
        for vkey, vresult in multi["variants"].items():
            heatmaps = _avg_label_token_heatmap(
                label_token_map, vresult["tam_maps"], vision_T,
            )
            mass_gt: Dict[int, Optional[float]] = {}
            mass_pred: Dict[int, Optional[float]] = {}
            for sampled_t, hmap in heatmaps.items():
                orig_t = sampled_t * self.sample_rate
                gt_box = item.gt_boxes[orig_t] if orig_t < item.num_frames else None
                pred_box = boxes[orig_t] if orig_t < len(boxes) else None
                mass_gt[sampled_t] = compute_mass_in_gt(hmap, gt_box, H_orig, W_orig)
                mass_pred[sampled_t] = compute_mass_in_gt(hmap, pred_box, H_orig, W_orig)

            mg = [v for v in mass_gt.values() if v is not None]
            mp = [v for v in mass_pred.values() if v is not None]
            per_variant[vkey] = {
                "mean_mass_in_gt": float(np.mean(mg)) if mg else None,
                "mean_mass_in_pred": float(np.mean(mp)) if mp else None,
                "mass_in_gt": {str(k): v for k, v in mass_gt.items()},
                "mass_in_pred": {str(k): v for k, v in mass_pred.items()},
                # heatmaps held only transiently for figure rendering
                "_heatmaps": heatmaps,
            }

        # 7. Per-sequence layer-grid figures
        detected_sampled = sorted({
            t for v in per_variant.values() for t in v["_heatmaps"].keys()
        })
        layer_keys = [f"layer_{L}" for L in self.layer_indices]
        cumavg_keys = [f"cumavg_{K}" for K in self.cumavg_Ks]

        if self.save_heatmap_grid and detected_sampled:
            for grid_name, keys in [("per_layer", layer_keys),
                                    ("cumavg", cumavg_keys)]:
                fig_path = self.vis_dir / f"{prefix}_{grid_name}.png"
                try:
                    save_layer_grid_figure(
                        seq_name=item.seq_name,
                        expression=item.expression,
                        grid_label=grid_name,
                        frames_pil=item.frames_pil,
                        detected_sampled_frames=detected_sampled,
                        sample_rate=self.sample_rate,
                        gt_boxes=item.gt_boxes,
                        pred_boxes={t: boxes[t] for t in range(item.num_frames)},
                        variant_keys=keys,
                        per_variant_heatmaps={
                            k: per_variant[k]["_heatmaps"] for k in keys
                        },
                        save_path=str(fig_path),
                    )
                except Exception as e:
                    print(f"  [WARN] {grid_name} grid figure failed: {e}")
                    traceback.print_exc()

        # Drop heatmaps before serialising
        for v in per_variant.values():
            v.pop("_heatmaps", None)

        return {
            "seq_name": item.seq_name,
            "exp_id": item.exp_id,
            "expression": item.expression,
            "num_frames": item.num_frames,
            "num_detected_frames": len(detected_sampled),
            "sample_rate": self.sample_rate,
            "iou_per_frame": iou_series.tolist(),
            "gen_text": gen_text,
            "layer_indices": self.layer_indices,
            "cumavg_Ks": self.cumavg_Ks,
            "norm_applied": multi["norm_applied"],
            "per_variant": per_variant,
        }

    # ── batch ─────────────────────────────────────────────────────────────────

    def run_all(
        self,
        sequences: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        expressions_per_seq: Optional[int] = 1,
    ) -> List[dict]:
        items = list(self.loader)
        if sequences:
            items = [it for it in items if it.seq_name in sequences]
        if expressions_per_seq is not None:
            seen: Dict[str, int] = {}
            filtered = []
            for it in items:
                count = seen.get(it.seq_name, 0)
                if count < expressions_per_seq:
                    filtered.append(it)
                    seen[it.seq_name] = count + 1
            items = filtered
        if max_sequences is not None:
            unique_seqs: List[str] = []
            filtered = []
            for it in items:
                if it.seq_name not in unique_seqs:
                    if len(unique_seqs) >= max_sequences:
                        continue
                    unique_seqs.append(it.seq_name)
                filtered.append(it)
            items = filtered

        results_path = self.save_dir / "results.json"
        all_results: List[dict] = []
        done_keys: set = set()
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            done_keys = {(r.get("seq_name"), str(r.get("exp_id"))) for r in all_results}
            print(f"[resume] loaded {len(all_results)} prior results from {results_path}")

        print(f"Running grounding-stability-layers on {len(items)} items "
              f"[layers={self.layer_indices}, cumavg_Ks={self.cumavg_Ks}, "
              f"apply_norm={self.apply_norm}]")
        for idx, item in enumerate(items):
            key = (item.seq_name, str(item.exp_id))
            if key in done_keys:
                print(f"\n[{idx+1}/{len(items)}] {item.seq_name}_exp{item.exp_id} — skip (done)")
                continue
            print(f"\n[{idx+1}/{len(items)}] {item.seq_name}")
            result = self.run_sequence(item)
            all_results.append(result)
            tmp_path = results_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            os.replace(tmp_path, results_path)
        return all_results

    # ── summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[dict]) -> dict:
        valid = [r for r in results if "error" not in r]
        if not valid:
            return {"n_sequences": 0, "n_errors": len(results)}

        # Aggregate per-variant means across sequences
        variant_keys = list(valid[0]["per_variant"].keys())
        agg = {}
        for vk in variant_keys:
            mg, mp = [], []
            for r in valid:
                v = r["per_variant"].get(vk)
                if v is None:
                    continue
                if v.get("mean_mass_in_gt") is not None:
                    mg.append(v["mean_mass_in_gt"])
                if v.get("mean_mass_in_pred") is not None:
                    mp.append(v["mean_mass_in_pred"])
            agg[vk] = {
                "mean_mass_in_gt": float(np.mean(mg)) if mg else None,
                "mean_mass_in_pred": float(np.mean(mp)) if mp else None,
                "n_seq_gt": len(mg),
                "n_seq_pred": len(mp),
            }

        return {
            "n_sequences": len(valid),
            "n_errors": len(results) - len(valid),
            "layer_indices": valid[0].get("layer_indices"),
            "cumavg_Ks": valid[0].get("cumavg_Ks"),
            "per_variant": agg,
        }
