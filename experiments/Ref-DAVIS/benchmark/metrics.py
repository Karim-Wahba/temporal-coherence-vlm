"""
metrics.py
----------
Standard VOS metrics:
  - J  : Region similarity (IoU) per frame
  - F  : Boundary F-measure per frame
  - J&F: Primary Ref-DAVIS metric (mean of mean-J and mean-F)

Temporal coherence metrics (new):
  - J-decay    : Linear slope of per-frame J over time (negative = losing track)
  - J-variance : Std of per-frame J (high = unstable tracking)
  - J-first    : J on first frame
  - J-last     : J on last frame
  - success_rate : fraction of frames with J > threshold

All functions operate on lists/arrays of binary masks (H,W) uint8.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


# ─── Region Similarity (J / IoU) ─────────────────────────────────────────────

def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU between two binary masks.
    Returns 0.0 if both masks are empty (no object in frame).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any() and not pred.any():
        return 1.0  # both empty = correct
    if not gt.any() or not pred.any():
        return 0.0
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def batch_jaccard(preds: list, gts: list) -> np.ndarray:
    """Returns per-frame J scores, shape (T,)."""
    return np.array([jaccard(p, g) for p, g in zip(preds, gts)])


# ─── Boundary F-measure (F) ──────────────────────────────────────────────────

def _get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Extract boundary pixels via morphological dilation."""
    mask = mask.astype(bool)
    h, w = mask.shape
    dil = max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask, structure=struct, iterations=dil)
    eroded = binary_erosion(mask, structure=struct, iterations=dil)
    boundary = dilated ^ eroded
    return boundary.astype(np.uint8)


def f_measure(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """
    Boundary F-measure between predicted and GT masks.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    pred_b = _get_boundary(pred, dilation_ratio)
    gt_b = _get_boundary(gt, dilation_ratio)

    precision_num = (pred_b & gt_b).sum()
    precision = precision_num / (pred_b.sum() + 1e-8)
    recall = precision_num / (gt_b.sum() + 1e-8)

    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def batch_f_measure(preds: list, gts: list) -> np.ndarray:
    """Returns per-frame F scores, shape (T,)."""
    return np.array([f_measure(p, g) for p, g in zip(preds, gts)])


# ─── Temporal Coherence Metrics ───────────────────────────────────────────────

def j_decay(j_scores: np.ndarray) -> float:
    """
    Linear slope of J over frame index (frames as x-axis).
    Negative slope = model loses track over time.
    Normalised by sequence length so values are comparable across sequences.
    """
    T = len(j_scores)
    if T < 2:
        return 0.0
    x = np.arange(T, dtype=float) / (T - 1)  # normalised 0..1
    slope = np.polyfit(x, j_scores, 1)[0]
    return float(slope)


def j_variance(j_scores: np.ndarray) -> float:
    """Std of per-frame J. High = unstable tracking."""
    return float(np.std(j_scores))


def success_rate(j_scores: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of frames with J > threshold."""
    return float((j_scores > threshold).mean())


# ─── Per-Sequence Summary ─────────────────────────────────────────────────────

def compute_sequence_metrics(
    preds: list,       # list of (H,W) uint8 binary masks, length T
    gts: list,         # list of (H,W) uint8 binary masks, length T
    dilation_ratio: float = 0.02,
) -> dict:
    """
    Compute all metrics for one sequence × expression.

    Returns
    -------
    dict with keys:
        J_per_frame, F_per_frame,
        mean_J, mean_F, JF,
        J_decay, J_variance, J_first, J_last,
        success_rate_50, success_rate_75
    """
    assert len(preds) == len(gts), "pred and gt must have same length"

    J = batch_jaccard(preds, gts)
    F = batch_f_measure(preds, gts)

    return {
        # Per-frame arrays (for plotting)
        "J_per_frame": J.tolist(),
        "F_per_frame": F.tolist(),
        # Scalar metrics
        "mean_J": float(J.mean()),
        "mean_F": float(F.mean()),
        "JF": float((J.mean() + F.mean()) / 2),
        # Temporal coherence metrics
        "J_decay": j_decay(J),
        "J_variance": j_variance(J),
        "J_first": float(J[0]),
        "J_last": float(J[-1]),
        "success_rate_50": success_rate(J, 0.5),
        "success_rate_75": success_rate(J, 0.75),
        "num_frames": len(J),
    }


# ─── VOT Metrics ─────────────────────────────────────────────────────────────

def bbox_iou(pred: tuple, gt: tuple) -> float:
    """
    IoU between two (x1, y1, x2, y2) bounding boxes.
    Returns 0.0 if either box is None or has zero area.
    """
    if pred is None or gt is None:
        return 0.0
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = max(0, px2 - px1) * max(0, py2 - py1)
    area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = area_p + area_g - inter
    return float(inter) / float(union) if union > 0 else 0.0


def center_error(pred: tuple, gt: tuple) -> float:
    """Euclidean distance between box centers. Returns inf if either is None."""
    if pred is None or gt is None:
        return float("inf")
    cx_p = (pred[0] + pred[2]) / 2.0
    cy_p = (pred[1] + pred[3]) / 2.0
    cx_g = (gt[0] + gt[2]) / 2.0
    cy_g = (gt[1] + gt[3]) / 2.0
    return float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))


def compute_vot_sequence_metrics(
    pred_boxes: list,   # list of Box (x1,y1,x2,y2) or None, length T
    gt_boxes: list,     # list of Box (x1,y1,x2,y2) or None, length T
) -> dict:
    """
    Compute per-sequence VOT metrics.

    Returns
    -------
    dict with keys:
        iou_per_frame, center_error_per_frame,
        mean_iou, success_rate_50, success_rate_75,
        iou_decay, iou_variance, iou_first, iou_last,
        mean_center_error, precision_20 (fraction of frames with center_error < 20px),
        num_frames
    """
    assert len(pred_boxes) == len(gt_boxes)

    iou_scores = np.array([bbox_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)])
    ce_scores = np.array([center_error(p, g) for p, g in zip(pred_boxes, gt_boxes)])
    # Replace inf with large number for averaging; track separately
    ce_finite = np.where(np.isfinite(ce_scores), ce_scores, np.nan)

    T = len(iou_scores)
    x = np.arange(T, dtype=float) / max(T - 1, 1)
    iou_slope = float(np.polyfit(x, iou_scores, 1)[0]) if T >= 2 else 0.0

    return {
        "iou_per_frame": iou_scores.tolist(),
        "center_error_per_frame": ce_scores.tolist(),
        "mean_iou": float(iou_scores.mean()),
        "success_rate_50": float((iou_scores > 0.5).mean()),
        "success_rate_75": float((iou_scores > 0.75).mean()),
        "iou_decay": iou_slope,
        "iou_variance": float(np.std(iou_scores)),
        "iou_first": float(iou_scores[0]),
        "iou_last": float(iou_scores[-1]),
        "mean_center_error": float(np.nanmean(ce_finite)) if not np.all(np.isnan(ce_finite)) else float("inf"),
        "precision_20": float(np.sum(ce_scores < 20) / T),
        "num_frames": T,
    }


def aggregate_vot_metrics(per_sequence: list) -> dict:
    """
    Average scalar VOT metrics across sequences.

    Parameters
    ----------
    per_sequence : list of dicts from compute_vot_sequence_metrics()
    """
    scalar_keys = [
        "mean_iou", "success_rate_50", "success_rate_75",
        "iou_decay", "iou_variance", "iou_first", "iou_last",
        "mean_center_error", "precision_20",
    ]
    agg = {}
    for k in scalar_keys:
        vals = np.array([r[k] for r in per_sequence if k in r and np.isfinite(r[k])])
        if len(vals) > 0:
            agg[k] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std())
        else:
            agg[k] = float("nan")
            agg[f"{k}_std"] = float("nan")
    agg["num_sequences"] = len(per_sequence)
    return agg


def aggregate_metrics(per_sequence: list) -> dict:
    """
    Average scalar metrics across all sequence×expression entries.

    Parameters
    ----------
    per_sequence : list of dicts from compute_sequence_metrics()

    Returns
    -------
    dict with mean of each scalar metric, plus std for key metrics
    """
    scalar_keys = [
        "mean_J", "mean_F", "JF",
        "J_decay", "J_variance", "J_first", "J_last",
        "success_rate_50", "success_rate_75",
    ]
    agg = {}
    for k in scalar_keys:
        vals = np.array([r[k] for r in per_sequence if k in r])
        agg[k] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std())
    agg["num_sequences"] = len(per_sequence)
    return agg
