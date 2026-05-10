"""
metrics.py
----------
Metrics for the RefCOCO grounding experiment (single-image setting).

Q1: Does grounding work?     → IoU of predicted box vs GT
Q2: Where does the model look? → mass-in-GT, mass-in-pred, attention entropy
Q3: Does attention predict accuracy? → Pearson / Spearman (mass-in-GT vs IoU)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ── Q1: Grounding accuracy ────────────────────────────────────────────────────

def compute_iou(pred_box, gt_box) -> float:
    """IoU between two (x1, y1, x2, y2) boxes. Returns 0.0 if either is None."""
    if pred_box is None or gt_box is None:
        return 0.0
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = max(0, px2 - px1) * max(0, py2 - py1)
    area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = area_p + area_g - inter
    return float(inter / union) if union > 0 else 0.0


# ── Q2: Attention localisation ────────────────────────────────────────────────

def compute_mass_in_gt(
    heatmap: np.ndarray,
    gt_box,
    frame_H: int,
    frame_W: int,
) -> Optional[float]:
    """
    Fraction of heatmap activation inside the GT bounding box.

    heatmap : (H_tam, W_tam) float32 in [0, 1]
    gt_box  : (x1, y1, x2, y2) in original pixel space
    """
    if gt_box is None or heatmap is None:
        return None
    hmap = heatmap.astype(np.float32)
    total = hmap.sum()
    if total == 0:
        return 0.0
    H_tam, W_tam = hmap.shape
    x1, y1, x2, y2 = gt_box
    hx1 = max(0, int(x1 * W_tam / frame_W))
    hx2 = min(W_tam, int(x2 * W_tam / frame_W))
    hy1 = max(0, int(y1 * H_tam / frame_H))
    hy2 = min(H_tam, int(y2 * H_tam / frame_H))
    inside = hmap[hy1:hy2, hx1:hx2].sum()
    return float(inside / total)


def compute_attention_entropy(heatmap: np.ndarray) -> float:
    """
    Shannon entropy of the normalised attention heatmap.
    Higher = more diffuse; lower = more focused.
    """
    h = heatmap.astype(np.float64)
    total = h.sum()
    if total == 0:
        return 0.0
    p = (h / total).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Q3: Correlation ───────────────────────────────────────────────────────────

def _correlation_dict(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n": int(len(x))}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r":  float(pr), "pearson_p":  float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "n": int(len(x)),
    }


def compute_mass_accuracy_correlation(
    mass_vals: List[float],
    iou_vals: List[float],
) -> dict:
    """Pearson + Spearman correlation between attention mass-in-GT and IoU."""
    x = np.array(mass_vals, dtype=np.float64)
    y = np.array(iou_vals,  dtype=np.float64)
    return _correlation_dict(x, y)


# ── Accuracy@k ────────────────────────────────────────────────────────────────

def accuracy_at_threshold(iou_vals: List[float], threshold: float = 0.5) -> float:
    """Fraction of items with IoU ≥ threshold (standard grounding metric)."""
    if not iou_vals:
        return 0.0
    return float(sum(v >= threshold for v in iou_vals) / len(iou_vals))
