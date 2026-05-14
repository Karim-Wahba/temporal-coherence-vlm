"""
metrics/grounding_metrics.py
----------------------------
Aggregates per-clip grounding metrics from a GroundingOutput.

Returns a GroundingMetrics object with:
  mean_iou
  mean_mass_in_gt       attention mass inside GT box
  mean_mass_in_pred     attention mass inside predicted box
  token_category_breakdown
      For each POS category (noun, adj, verb, other), the mean mass-in-GT of
      heat maps built from only that category's tokens.

The token-category breakdown extends what the prior pos-dominance experiment
measured (which token classes the model attends to inside the GT). Reusing
the pos_tagger from pos-dominance keeps tagging consistent across experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import _paths  # noqa: F401

import numpy as np

# Both functions come from grounding-stability-max/metrics.py (added to sys.path
# by _paths). benchmark/metrics.py has different helpers (jaccard, bbox_iou).
from metrics import frame_iou_series, compute_mass_in_gt  # type: ignore

try:
    from pos_tagger import build_pos_map  # type: ignore
    _HAS_POS = True
except ImportError:
    _HAS_POS = False


# ── helpers ────────────────────────────────────────────────────────────────────

_POS_TO_CAT = {
    "label_noun":      "noun",
    "label_adj":       "adj",
    "label_verb":      "verb",
    "label_adv":       "other",
    "label_other_pos": "other",
}


def _categorize_words(expression: str) -> Dict[str, str]:
    """Word -> 'noun'|'adj'|'verb'|'other'."""
    if not _HAS_POS:
        words = re.findall(r"[A-Za-z']+", expression.lower())
        return {w: "noun" if i == len(words) - 1 else "other"
                for i, w in enumerate(words)}
    pos_map = build_pos_map(expression)
    return {w: _POS_TO_CAT.get(tag, "other") for w, tag in pos_map.items()}


# ── output type ────────────────────────────────────────────────────────────────

@dataclass
class GroundingMetrics:
    expression: str
    mean_iou: Optional[float]
    mean_mass_in_gt: Optional[float]
    mean_mass_in_pred: Optional[float]
    token_category_breakdown: Dict[str, Optional[float]] = field(default_factory=dict)
    num_detected_frames: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "mean_iou":   self.mean_iou,
            "mean_mass_in_gt":   self.mean_mass_in_gt,
            "mean_mass_in_pred": self.mean_mass_in_pred,
            "token_category_breakdown": self.token_category_breakdown,
            "num_detected_frames": self.num_detected_frames,
            "error": self.error,
        }


# ── core computation ──────────────────────────────────────────────────────────

def compute_metrics(
    grounding,                # GroundingOutput from grounding.qwen3vl_runner
    clip,                     # Clip from data.ref_davis_loader
    sample_rate: int,
    fps: float,
    compute_category_breakdown: bool = True,
) -> GroundingMetrics:
    """Compute IoU + mass-in-box + per-POS-category breakdown for one clip.

    If grounding.tam_result is None (skip_tam fast path), only mean_iou is
    computed and all mass metrics are set to None.
    """
    boxes      = grounding.boxes
    tam_result = grounding.tam_result

    H, W = clip.frame_size()

    # ── 1. Per-frame IoU ────────────────────────────────────────────────────
    iou_series = frame_iou_series(boxes, clip.gt_boxes)
    sampled    = list(range(0, clip.num_frames, sample_rate))
    mean_iou   = float(iou_series[sampled].mean()) if sampled else float(iou_series.mean())

    # ── Fast path: no TAM data → IoU only ──────────────────────────────────
    if tam_result is None:
        return GroundingMetrics(
            expression=grounding.expression,
            mean_iou=round(mean_iou, 4) if mean_iou is not None else None,
            mean_mass_in_gt=None,
            mean_mass_in_pred=None,
            token_category_breakdown={"noun": None, "adj": None, "verb": None, "other": None},
            num_detected_frames=0,
        )

    # ── TAM-dependent metrics from here on ─────────────────────────────────
    from attribution.tam import build_frame_heatmaps, extract_label_token_map

    gen_tokens = tam_result["gen_tokens"]
    tam_maps   = tam_result["tam_maps"]
    vision_T   = tam_result["vision_shape"][0]

    # ── 2. Default heat maps (avg of all label tokens) ─────────────────────
    frame_heatmaps = build_frame_heatmaps(tam_result, fps=fps, sample_rate=sample_rate)

    m_gt_vals: List[float]   = []
    m_pred_vals: List[float] = []
    for sampled_t, hmap in frame_heatmaps.items():
        orig_t   = sampled_t * sample_rate
        gt_box   = clip.gt_boxes[orig_t] if orig_t < clip.num_frames else None
        pred_box = boxes[orig_t]          if orig_t < len(boxes)     else None
        mg = compute_mass_in_gt(hmap, gt_box,   H, W)
        mp = compute_mass_in_gt(hmap, pred_box, H, W)
        if mg is not None: m_gt_vals.append(float(mg))
        if mp is not None: m_pred_vals.append(float(mp))

    mean_gt   = float(np.mean(m_gt_vals))   if m_gt_vals   else None
    mean_pred = float(np.mean(m_pred_vals)) if m_pred_vals else None

    # ── 3. Token-category breakdown of mass-in-GT ───────────────────────────
    cat_breakdown: Dict[str, Optional[float]] = {"noun": None, "adj": None, "verb": None, "other": None}
    if compute_category_breakdown:
        cat_breakdown = _compute_category_breakdown(
            tam_result=tam_result,
            boxes=boxes,
            clip=clip,
            sample_rate=sample_rate,
            fps=fps,
        )

    return GroundingMetrics(
        expression=grounding.expression,
        mean_iou=round(mean_iou, 4) if mean_iou is not None else None,
        mean_mass_in_gt=round(mean_gt, 4) if mean_gt is not None else None,
        mean_mass_in_pred=round(mean_pred, 4) if mean_pred is not None else None,
        token_category_breakdown={k: (round(v, 4) if v is not None else None)
                                  for k, v in cat_breakdown.items()},
        num_detected_frames=len(frame_heatmaps),
    )


def _compute_category_breakdown(
    tam_result: dict,
    boxes,
    clip,
    sample_rate: int,
    fps: float,
) -> Dict[str, Optional[float]]:
    """Mean mass-in-GT computed separately per POS category of label tokens."""
    from attribution.tam import extract_label_token_map

    tam_maps = tam_result["tam_maps"]
    vision_T = tam_result["vision_shape"][0]
    gen_tokens = tam_result["gen_tokens"]
    H, W = clip.frame_size()

    cat_map = _categorize_words(clip.seed_expression)  # word -> category
    # Get per-token text strings (gen_tokens are already decoded strings)
    token_strs = [t.strip().lower().replace("ġ", "").replace(" ", "")
                  for t in gen_tokens]

    def token_category(tok_str: str) -> str:
        for word, cat in cat_map.items():
            if word and word in tok_str:
                return cat
        return "other"

    label_token_map = extract_label_token_map(tam_result, fps=fps, sample_rate=sample_rate)

    # Per category: collect (sampled_t -> averaged heatmap of just that category's tokens)
    per_cat_vals: Dict[str, List[float]] = {"noun": [], "adj": [], "verb": [], "other": []}

    for sampled_t, tok_idxs in label_token_map:
        if sampled_t >= vision_T:
            continue
        for cat in per_cat_vals.keys():
            valid = [
                i for i in tok_idxs
                if i < len(tam_maps)
                and tam_maps[i] is not None
                and tam_maps[i].ndim == 3
                and sampled_t < tam_maps[i].shape[0]
                and i < len(token_strs)
                and token_category(token_strs[i]) == cat
            ]
            if not valid:
                continue
            slices = [tam_maps[i][sampled_t].astype(np.float32) for i in valid]
            hmap = np.mean(slices, axis=0)
            mx = hmap.max()
            if mx > 0:
                hmap /= mx
            orig_t = sampled_t * sample_rate
            gt_box = clip.gt_boxes[orig_t] if orig_t < clip.num_frames else None
            mg = compute_mass_in_gt(hmap, gt_box, H, W)
            if mg is not None:
                per_cat_vals[cat].append(float(mg))

    return {
        cat: float(np.mean(vals)) if vals else None
        for cat, vals in per_cat_vals.items()
    }
