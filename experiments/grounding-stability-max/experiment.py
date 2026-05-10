"""
experiment.py
-------------
Core runner for the grounding-stability-max experiment.

For each (sequence, expression) pair it:
  1. Runs QwenVOTRunner.run_with_tam() — one forward pass.
  2. Parses the JSON output to find which generated tokens are the label nouns
     for each detected frame.
  3. Builds per-frame heatmaps using one of three strategies selected by
     `token_selection`:
       "none"      – average all label tokens (original behaviour)
       "best_gt"   – use the single token position whose avg mass-in-GT is
                     highest across the sequence
       "best_pred" – same but maximising avg mass-in-predicted-box
  4. Computes IoU per frame, TAM instability, mass-in-GT, mass-in-pred,
     Pearson/Spearman correlations.
  5. Saves a two-row visualisation figure per sequence.
  6. When token_selection != "none", also stores per-position mass values so
     the caller can produce variance plots.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REF_DAVIS = _HERE.parent / "Ref-DAVIS"
sys.path.insert(0, str(_REF_DAVIS))
sys.path.insert(0, str(_REF_DAVIS / "benchmark"))
sys.path.insert(0, str(_HERE))  # must come last to shadow benchmark/metrics.py

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem
from benchmark.qwen_vot_runner import QwenVOTRunner

from token_parser import parse_frame_labels, find_label_token_indices
from metrics import (
    frame_iou_series,
    compute_mass_in_gt,
    compute_tam_instability,
    compute_correlations,
    compute_attention_entropy,
    compute_flow_correlation,
    compute_mass_accuracy_correlation,
)
from visualizer import save_sequence_figure, save_flow_figure


class GroundingStabilityExperiment:
    """
    Parameters
    ----------
    model, processor : loaded Qwen model and processor
    davis_root       : path to DAVIS dataset root
    save_dir         : output directory
    video_mode       : True → single video block (3D RoPE), False → interleaved images
    sample_rate      : send every Nth frame to the model
    max_new_tokens   : generation budget
    split            : "valid" or "train"
    token_selection  : "none" | "best_gt" | "best_pred"
                       Controls which per-label-token heatmap to use per frame.
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
        token_selection: str = "none",
    ):
        assert token_selection in ("none", "best_gt", "best_pred"), \
            f"token_selection must be 'none', 'best_gt', or 'best_pred', got {token_selection!r}"

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
        self.sample_rate = sample_rate
        self.video_mode = video_mode
        self.fps = self.runner.fps
        self.token_selection = token_selection

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_per_position_heatmaps(
        self,
        label_token_map: List[Tuple[int, List[int]]],
        tam_maps: list,
        vision_T: int,
        gen_tokens: List[str],
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, str]]]:
        """
        For each (sampled_t, tok_idxs) entry, split out the individual token
        positions instead of averaging them together.

        Returns
        -------
        per_pos_hmaps  : {pos_idx: {sampled_t: heatmap_float32_normalised}}
        per_pos_tokens : {pos_idx: {sampled_t: token_string}}
        """
        per_pos_hmaps: Dict[int, Dict[int, np.ndarray]] = {}
        per_pos_tokens: Dict[int, Dict[int, str]] = {}

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
            for pos, tok_idx in enumerate(valid_idxs):
                hmap = tam_maps[tok_idx][sampled_t].astype(np.float32)
                max_val = hmap.max()
                if max_val > 0:
                    hmap = hmap / max_val
                per_pos_hmaps.setdefault(pos, {})[sampled_t] = hmap
                per_pos_tokens.setdefault(pos, {})[sampled_t] = gen_tokens[tok_idx]

        return per_pos_hmaps, per_pos_tokens

    def _select_best_position(
        self,
        per_pos_hmaps: Dict[int, Dict[int, np.ndarray]],
        gt_boxes: list,
        pred_boxes: list,
        H: int,
        W: int,
    ) -> Tuple[
        int,
        Dict[int, Dict[int, float]],
        Dict[int, Dict[int, float]],
        Dict[int, float],
        Dict[int, float],
    ]:
        """
        For each token position, compute mass-in-GT and mass-in-pred at every
        detected frame, then average across frames.  Select the position whose
        average is highest according to `self.token_selection`.

        Returns
        -------
        best_pos              : int
        per_pos_masses_gt     : {pos: {sampled_t: mass}}
        per_pos_masses_pred   : {pos: {sampled_t: mass}}
        per_pos_avg_mass_gt   : {pos: float}
        per_pos_avg_mass_pred : {pos: float}
        """
        per_pos_masses_gt:   Dict[int, Dict[int, float]] = {}
        per_pos_masses_pred: Dict[int, Dict[int, float]] = {}
        per_pos_avg_mass_gt:   Dict[int, float] = {}
        per_pos_avg_mass_pred: Dict[int, float] = {}

        for pos, frame_hmaps in per_pos_hmaps.items():
            gt_masses:   Dict[int, float] = {}
            pred_masses: Dict[int, float] = {}
            for sampled_t, hmap in frame_hmaps.items():
                orig_t = sampled_t * self.sample_rate
                gt_box   = gt_boxes[orig_t]   if orig_t < len(gt_boxes)   else None
                pred_box = pred_boxes[orig_t] if orig_t < len(pred_boxes) else None
                m_gt   = compute_mass_in_gt(hmap, gt_box,   H, W)
                m_pred = compute_mass_in_gt(hmap, pred_box, H, W)
                if m_gt is not None:
                    gt_masses[sampled_t]   = float(m_gt)
                if m_pred is not None:
                    pred_masses[sampled_t] = float(m_pred)
            per_pos_masses_gt[pos]   = gt_masses
            per_pos_masses_pred[pos] = pred_masses
            per_pos_avg_mass_gt[pos]   = float(np.mean(list(gt_masses.values())))   if gt_masses   else 0.0
            per_pos_avg_mass_pred[pos] = float(np.mean(list(pred_masses.values()))) if pred_masses else 0.0

        if not per_pos_hmaps:
            return 0, {}, {}, {}, {}

        if self.token_selection == "best_gt":
            best_pos = max(per_pos_avg_mass_gt,   key=lambda p: per_pos_avg_mass_gt[p])
        else:  # best_pred
            best_pos = max(per_pos_avg_mass_pred, key=lambda p: per_pos_avg_mass_pred[p])

        return best_pos, per_pos_masses_gt, per_pos_masses_pred, per_pos_avg_mass_gt, per_pos_avg_mass_pred

    # ── per-sequence ──────────────────────────────────────────────────────────

    def run_sequence(self, item: DAVISVOTItem) -> dict:
        """Run the full pipeline for one (sequence, expression) item."""
        prefix = f"{item.seq_name}_exp{item.exp_id}"
        print(f"  Running {prefix} | \"{item.expression[:60]}\"")

        # 1. Inference + TAM in one forward pass
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

        # 2. Parse which gen_tokens belong to each frame's label
        parsed_entries  = parse_frame_labels(
            tam_result["gen_text"],
            fps=self.fps,
            sample_rate=self.sample_rate,
        )
        label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

        # 3. Build per-frame heatmaps
        frame_heatmaps:     Dict[int, np.ndarray] = {}
        frame_label_tokens: Dict[int, List[str]]  = {}

        # Extra fields populated only when token_selection != "none"
        selected_token_pos:    Optional[int]              = None
        per_pos_masses_gt:     Dict[int, Dict[int, float]] = {}
        per_pos_masses_pred:   Dict[int, Dict[int, float]] = {}
        per_pos_avg_mass_gt:   Dict[int, float]            = {}
        per_pos_avg_mass_pred: Dict[int, float]            = {}

        if self.token_selection == "none":
            # Original: average all label tokens for each frame
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
                slices = [tam_maps[i][sampled_t].astype(np.float32) for i in valid_idxs]
                avg_map = np.mean(slices, axis=0)
                max_val = avg_map.max()
                if max_val > 0:
                    avg_map /= max_val
                frame_heatmaps[sampled_t]     = avg_map
                frame_label_tokens[sampled_t] = [gen_tokens[i] for i in valid_idxs]

        else:
            # Build per-position heatmaps and select the best one
            H, W = item.frame_size()
            per_pos_hmaps, per_pos_tok_strs = self._build_per_position_heatmaps(
                label_token_map, tam_maps, vision_T, gen_tokens
            )
            (
                selected_token_pos,
                per_pos_masses_gt,
                per_pos_masses_pred,
                per_pos_avg_mass_gt,
                per_pos_avg_mass_pred,
            ) = self._select_best_position(per_pos_hmaps, item.gt_boxes, boxes, H, W)

            best_hmaps  = per_pos_hmaps.get(selected_token_pos, {})
            best_tokens = per_pos_tok_strs.get(selected_token_pos, {})
            for sampled_t, hmap in best_hmaps.items():
                frame_heatmaps[sampled_t]     = hmap
                tok_str = best_tokens.get(sampled_t, "")
                frame_label_tokens[sampled_t] = [tok_str] if tok_str else []

        # 4. IoU per original frame
        H, W = item.frame_size()
        iou_series      = frame_iou_series(boxes, item.gt_boxes)
        sampled_indices = list(range(0, item.num_frames, self.sample_rate))
        sampled_iou     = iou_series[sampled_indices] if sampled_indices else iou_series

        # 5. Mass-in-GT and mass-in-pred for each detected frame
        mass_in_gt:   Dict[int, Optional[float]] = {}
        mass_in_pred: Dict[int, Optional[float]] = {}
        for sampled_t, hmap in frame_heatmaps.items():
            orig_t   = sampled_t * self.sample_rate
            gt_box   = item.gt_boxes[orig_t] if orig_t < item.num_frames else None
            pred_box = boxes[orig_t]          if orig_t < len(boxes)     else None
            mass_in_gt[sampled_t]   = compute_mass_in_gt(hmap, gt_box,   H, W)
            mass_in_pred[sampled_t] = compute_mass_in_gt(hmap, pred_box, H, W)

        # 6. TAM instability between consecutive detected frames
        instability = compute_tam_instability(frame_heatmaps)

        # 7. Correlations: instability (t→t+1) vs IoU failure at frame t
        inst_vals, failure_vals = [], []
        for (t, t1), inst_val in instability.items():
            orig_t   = t * self.sample_rate
            iou_fail = 1.0 - float(iou_series[orig_t]) if orig_t < len(iou_series) else 1.0
            inst_vals.append(inst_val)
            failure_vals.append(iou_fail)
        correlations = compute_correlations(inst_vals, failure_vals)

        # 8. Attention entropy per detected frame
        entropy_per_frame = {
            t: compute_attention_entropy(hm) for t, hm in frame_heatmaps.items()
        }
        mean_entropy = float(np.mean(list(entropy_per_frame.values()))) \
            if entropy_per_frame else None

        # 9. Optical flow correlation
        frame_grays: Dict[int, np.ndarray] = {}
        for sampled_t in frame_heatmaps:
            orig_t = min(sampled_t * self.sample_rate, len(item.frames_pil) - 1)
            rgb    = np.array(item.frames_pil[orig_t].convert("RGB"))
            frame_grays[sampled_t] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        flow_corr = compute_flow_correlation(frame_heatmaps, frame_grays)

        # 10. Attention accuracy: mass-in-GT vs IoU, mass-in-pred vs IoU
        mass_acc_vals, iou_acc_vals   = [], []
        mass_pred_vals, iou_pred_vals = [], []
        for sampled_t in sorted(frame_heatmaps.keys()):
            orig_t = sampled_t * self.sample_rate
            if orig_t >= len(iou_series):
                continue
            iou_val = float(iou_series[orig_t])
            m_gt    = mass_in_gt.get(sampled_t)
            m_pred  = mass_in_pred.get(sampled_t)
            if m_gt is not None:
                mass_acc_vals.append(m_gt)
                iou_acc_vals.append(iou_val)
            if m_pred is not None:
                mass_pred_vals.append(m_pred)
                iou_pred_vals.append(iou_val)
        mass_accuracy_corr      = compute_mass_accuracy_correlation(mass_acc_vals,  iou_acc_vals)
        mass_pred_accuracy_corr = compute_mass_accuracy_correlation(mass_pred_vals, iou_pred_vals)

        # 11. Visualisation
        detected_sampled = sorted(frame_heatmaps.keys())
        pred_boxes_dict  = {t: boxes[t] for t in range(item.num_frames)}
        vis_path  = str(self.vis_dir / f"{prefix}.png")

        try:
            save_sequence_figure(
                seq_name=item.seq_name,
                expression=item.expression,
                frames_pil=item.frames_pil,
                detected_sampled_frames=detected_sampled,
                pred_boxes=pred_boxes_dict,
                gt_boxes=item.gt_boxes,
                frame_heatmaps=frame_heatmaps,
                label_tokens=frame_label_tokens,
                sample_rate=self.sample_rate,
                save_path=vis_path,
            )
        except Exception as e:
            print(f"  [WARN] visualisation failed: {e}")

        flow_path       = str(self.vis_dir / f"{prefix}_flow.png")
        flow_pair_stats = {f"{t}-{t1}": v for (t, t1), v in flow_corr["per_pair"].items()}
        try:
            save_flow_figure(
                seq_name=item.seq_name,
                expression=item.expression,
                frames_pil=item.frames_pil,
                detected_sampled_frames=detected_sampled,
                frame_heatmaps=frame_heatmaps,
                sample_rate=self.sample_rate,
                save_path=flow_path,
                flow_pair_stats=flow_pair_stats,
            )
        except Exception as e:
            print(f"  [WARN] flow figure failed: {e}")

        result = {
            "seq_name": item.seq_name,
            "exp_id": item.exp_id,
            "expression": item.expression,
            "num_frames": item.num_frames,
            "num_detected_frames": len(detected_sampled),
            "sample_rate": self.sample_rate,
            "token_selection": self.token_selection,
            # Q1
            "mean_iou": float(sampled_iou.mean()),
            "iou_variance": float(sampled_iou.var()),
            "iou_per_frame": iou_series.tolist(),
            # Q2
            "mass_in_gt":   {str(k): v for k, v in mass_in_gt.items()},
            "mean_mass_in_gt": float(np.nanmean([v for v in mass_in_gt.values() if v is not None]))
                               if mass_in_gt else None,
            "mass_in_pred": {str(k): v for k, v in mass_in_pred.items()},
            "mean_mass_in_pred": float(np.nanmean([v for v in mass_in_pred.values() if v is not None]))
                                 if mass_in_pred else None,
            "instability": {f"{t}-{t1}": v for (t, t1), v in instability.items()},
            "mean_instability": float(np.mean(list(instability.values()))) if instability else None,
            "entropy_per_frame": {str(k): v for k, v in entropy_per_frame.items()},
            "mean_entropy": mean_entropy,
            # Q3
            "correlations": correlations,
            # Q4
            "mass_accuracy_correlation": mass_accuracy_corr,
            "mass_pred_accuracy_correlation": mass_pred_accuracy_corr,
            # Q5
            "flow_correlation": {
                "per_pair": flow_pair_stats,
                "mean_r": flow_corr["mean_r"],
                "aggregate": flow_corr["aggregate"],
            },
            # metadata
            "gen_text": tam_result["gen_text"],
            "vis_path": vis_path,
            "flow_path": flow_path,
            "label_tokens": {str(k): v for k, v in frame_label_tokens.items()},
        }

        # Token-selection extras (only when not using the averaging baseline)
        if self.token_selection != "none":
            result["selected_token_pos"] = selected_token_pos
            result["per_token_avg_mass_gt"]   = {str(p): v for p, v in per_pos_avg_mass_gt.items()}
            result["per_token_avg_mass_pred"]  = {str(p): v for p, v in per_pos_avg_mass_pred.items()}
            result["per_token_masses_gt"]  = {
                str(p): {str(t): v for t, v in masses.items()}
                for p, masses in per_pos_masses_gt.items()
            }
            result["per_token_masses_pred"] = {
                str(p): {str(t): v for t, v in masses.items()}
                for p, masses in per_pos_masses_pred.items()
            }

        return result

    # ── batch ─────────────────────────────────────────────────────────────────

    def run_all(
        self,
        sequences: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        expressions_per_seq: Optional[int] = 1,
    ) -> List[dict]:
        """
        Run the experiment on multiple sequences.

        sequences           : whitelist of sequence names; None = all
        max_sequences       : cap on unique sequences
        expressions_per_seq : cap on expressions per sequence
        """
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

        print(f"Running grounding-stability-max on {len(items)} items "
              f"[token_selection={self.token_selection}]")
        all_results = []
        for idx, item in enumerate(items):
            print(f"\n[{idx+1}/{len(items)}] {item.seq_name}")
            result = self.run_sequence(item)
            all_results.append(result)

            out_path = self.save_dir / "results.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        return all_results

    # ── summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[dict]) -> dict:
        """Aggregate per-sequence results into dataset-level statistics."""
        valid = [r for r in results if "error" not in r]

        mean_ious      = [r["mean_iou"]          for r in valid]
        iou_variances  = [r["iou_variance"]       for r in valid]
        mean_mass      = [r["mean_mass_in_gt"]    for r in valid if r.get("mean_mass_in_gt")   is not None]
        mean_mass_pred = [r["mean_mass_in_pred"]  for r in valid if r.get("mean_mass_in_pred") is not None]
        mean_instab    = [r["mean_instability"]   for r in valid if r.get("mean_instability")  is not None]
        mean_entropies = [r["mean_entropy"]       for r in valid if r.get("mean_entropy")      is not None]

        all_inst, all_fail = [], []
        for r in valid:
            iou_list = r.get("iou_per_frame", [])
            sr = r.get("sample_rate", 1)
            for key, inst_val in r.get("instability", {}).items():
                t      = int(key.split("-")[0])
                orig_t = t * sr
                if orig_t < len(iou_list):
                    all_inst.append(inst_val)
                    all_fail.append(1.0 - iou_list[orig_t])
        global_corr = compute_correlations(all_inst, all_fail)

        per_seq_corrs_p = [r["correlations"]["pearson_r"]  for r in valid
                           if r["correlations"].get("pearson_r")  is not None]
        per_seq_corrs_s = [r["correlations"]["spearman_r"] for r in valid
                           if r["correlations"].get("spearman_r") is not None]

        all_mass_acc,  all_iou_acc  = [], []
        all_mass_pred, all_iou_pred = [], []
        for r in valid:
            iou_list = r.get("iou_per_frame", [])
            sr = r.get("sample_rate", 1)
            for k, mass_val in r.get("mass_in_gt", {}).items():
                if mass_val is None:
                    continue
                orig_t = int(k) * sr
                if orig_t < len(iou_list):
                    all_mass_acc.append(mass_val)
                    all_iou_acc.append(iou_list[orig_t])
            for k, mass_val in r.get("mass_in_pred", {}).items():
                if mass_val is None:
                    continue
                orig_t = int(k) * sr
                if orig_t < len(iou_list):
                    all_mass_pred.append(mass_val)
                    all_iou_pred.append(iou_list[orig_t])
        mass_accuracy_global_corr      = compute_mass_accuracy_correlation(all_mass_acc,  all_iou_acc)
        mass_pred_accuracy_global_corr = compute_mass_accuracy_correlation(all_mass_pred, all_iou_pred)

        all_flow_r: List[float] = []
        all_img_flow_means: List[float] = []
        all_hm_flow_means:  List[float] = []
        for r in valid:
            for v in r.get("flow_correlation", {}).get("per_pair", {}).values():
                if isinstance(v, dict) and v.get("r") is not None:
                    all_flow_r.append(v["r"])
                    all_img_flow_means.append(v["img_flow_mean"])
                    all_hm_flow_means.append(v["hm_flow_mean"])
        flow_global_corr = compute_mass_accuracy_correlation(all_img_flow_means, all_hm_flow_means)

        return {
            "n_sequences":   len(valid),
            "n_errors":      len(results) - len(valid),
            "mean_iou":            float(np.mean(mean_ious))      if mean_ious      else None,
            "mean_iou_variance":   float(np.mean(iou_variances))  if iou_variances  else None,
            "mean_mass_in_gt":     float(np.mean(mean_mass))      if mean_mass      else None,
            "mean_mass_in_pred":   float(np.mean(mean_mass_pred)) if mean_mass_pred else None,
            "mean_instability":    float(np.mean(mean_instab))    if mean_instab    else None,
            "mean_entropy":        float(np.mean(mean_entropies)) if mean_entropies else None,
            "global_correlation":              global_corr,
            "mean_per_seq_pearson_r":          float(np.mean(per_seq_corrs_p)) if per_seq_corrs_p else None,
            "mean_per_seq_spearman_r":         float(np.mean(per_seq_corrs_s)) if per_seq_corrs_s else None,
            "mass_accuracy_global_correlation":      mass_accuracy_global_corr,
            "mass_pred_accuracy_global_correlation": mass_pred_accuracy_global_corr,
            "mean_flow_correlation_r":         float(np.mean(all_flow_r)) if all_flow_r else None,
            "flow_global_correlation":         flow_global_corr,
        }
