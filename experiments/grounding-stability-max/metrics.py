"""
metrics.py
----------
Q1: Does grounding break?       → IoU per frame, IoU variance over time
Q2: Is TAM stable?              → TAM instability, mass-in-GT, mass-in-pred, attention entropy
Q3: Prediction of failure?      → Pearson / Spearman correlation (instability vs IoU failure)
Q4: Attention accuracy?         → mass-in-GT vs IoU, mass-in-pred vs IoU correlation
Q5: Flow alignment?             → optical flow correlation (image vs heatmap)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats


# ── Q1: Grounding ─────────────────────────────────────────────────────────────

def compute_iou(pred_box, gt_box) -> float:
    """
    IoU between two (x1, y1, x2, y2) boxes in the same coordinate space.
    Returns 0.0 if either box is None or degenerate.
    """
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


def frame_iou_series(
    pred_boxes: List,
    gt_boxes: List,
) -> np.ndarray:
    """
    Compute IoU for every frame. pred_boxes / gt_boxes are parallel lists
    (elements may be None). Returns float32 array of length N.
    """
    return np.array([compute_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)],
                    dtype=np.float32)


# ── Q2: TAM stability ─────────────────────────────────────────────────────────

def compute_mass_in_gt(
    heatmap: np.ndarray,
    gt_box,
    frame_H: int,
    frame_W: int,
) -> Optional[float]:
    """
    Fraction of heatmap activation that falls inside the GT bounding box.

    heatmap  : (H_tam, W_tam) float or uint8 – the per-frame TAM slice
    gt_box   : (x1, y1, x2, y2) in original (frame_H, frame_W) pixel space
    Returns None when gt_box is None or heatmap is empty.
    """
    if gt_box is None or heatmap is None:
        return None

    hmap = heatmap.astype(np.float32)
    total = hmap.sum()
    if total == 0:
        return 0.0

    H_tam, W_tam = hmap.shape
    x1, y1, x2, y2 = gt_box
    # Scale to heatmap coordinates
    hx1 = max(0, int(x1 * W_tam / frame_W))
    hx2 = min(W_tam, int(x2 * W_tam / frame_W))
    hy1 = max(0, int(y1 * H_tam / frame_H))
    hy2 = min(H_tam, int(y2 * H_tam / frame_H))

    inside = hmap[hy1:hy2, hx1:hx2].sum()
    return float(inside / total)


def compute_tam_instability(
    frame_heatmaps: Dict[int, np.ndarray],
) -> Dict[Tuple[int, int], float]:
    """
    Mean absolute pixel difference between the per-frame TAM slices of
    consecutive detected frames.

    frame_heatmaps : {sampled_frame_idx: (H_tam, W_tam) float heatmap [0,1]}
    Returns        : {(t, t+1): instability_score} for consecutive pairs only.
    """
    sorted_frames = sorted(frame_heatmaps.keys())
    instability: Dict[Tuple[int, int], float] = {}
    for i in range(len(sorted_frames) - 1):
        t, t1 = sorted_frames[i], sorted_frames[i + 1]
        if t1 != t + 1:
            continue  # non-consecutive sampled frames – skip
        h0 = frame_heatmaps[t].astype(np.float32)
        h1 = frame_heatmaps[t1].astype(np.float32)
        instability[(t, t1)] = float(np.abs(h0 - h1).mean())
    return instability


# ── Q2b: Attention entropy ────────────────────────────────────────────────────

def compute_attention_entropy(heatmap: np.ndarray) -> float:
    """
    Shannon entropy of the normalised attention heatmap.
    Higher = more diffuse attention; lower = more focused.
    """
    h = heatmap.astype(np.float64)
    total = h.sum()
    if total == 0:
        return 0.0
    p = (h / total).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Q3: Correlation ───────────────────────────────────────────────────────────

def _correlation_dict(x: np.ndarray, y: np.ndarray) -> Dict:
    """Compute Pearson and Spearman r/p between two equal-length arrays."""
    if len(x) < 3:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n": int(len(x))}
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "n": int(len(x)),
    }


def compute_correlations(
    instability_vals: List[float],
    iou_failure_vals: List[float],
) -> Dict:
    """
    Pearson and Spearman correlation between TAM instability and IoU failure
    (1 - IoU) at the same frame transition.
    """
    x = np.array(instability_vals, dtype=np.float64)
    y = np.array(iou_failure_vals, dtype=np.float64)
    return _correlation_dict(x, y)


# ── Q4: Attention accuracy correlation ───────────────────────────────────────

def compute_mass_accuracy_correlation(
    mass_vals: List[float],
    iou_vals: List[float],
) -> Dict:
    """
    Pearson and Spearman correlation between attention mass-in-GT (x) and IoU (y).
    Tests whether attention localisation predicts grounding quality.
    """
    x = np.array(mass_vals, dtype=np.float64)
    y = np.array(iou_vals, dtype=np.float64)
    return _correlation_dict(x, y)


# ── Q5: Flow alignment ────────────────────────────────────────────────────────

def compute_flow_correlation(
    frame_heatmaps: Dict[int, np.ndarray],
    frame_grays: Dict[int, np.ndarray],
) -> Dict:
    """
    For each consecutive detected frame pair, compute Farneback optical flow
    on both the RGB image (grayscale) and the TAM heatmap, then measure the
    Pearson correlation between their per-pixel flow magnitudes.

    frame_heatmaps : {sampled_t: (H_tam, W_tam) float32 [0, 1]}
    frame_grays    : {sampled_t: (H, W) uint8 grayscale}

    Returns
    -------
    dict:
        per_pair  : {(t, t+1): {"r": float, "p": float,
                                 "img_flow_mean": float, "hm_flow_mean": float}}
        mean_r    : float — mean per-pair Pearson r (nan if no pairs)
        aggregate : _correlation_dict-style result correlating per-pair
                    mean image-flow magnitude with per-pair mean heatmap-flow magnitude
    """
    sorted_frames = sorted(set(frame_heatmaps.keys()) & set(frame_grays.keys()))
    per_pair: Dict[Tuple[int, int], Dict] = {}
    img_means: List[float] = []
    hm_means: List[float] = []

    for i in range(len(sorted_frames) - 1):
        t, t1 = sorted_frames[i], sorted_frames[i + 1]
        if t1 != t + 1:
            continue

        gray0, gray1 = frame_grays[t], frame_grays[t1]
        hm0, hm1 = frame_heatmaps[t], frame_heatmaps[t1]
        H, W = gray0.shape[:2]

        flow_img = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag_img, _ = cv2.cartToPolar(flow_img[..., 0], flow_img[..., 1])

        hm0_u8 = cv2.resize(
            (hm0 * 255).clip(0, 255).astype(np.uint8), (W, H))
        hm1_u8 = cv2.resize(
            (hm1 * 255).clip(0, 255).astype(np.uint8), (W, H))
        flow_hm = cv2.calcOpticalFlowFarneback(
            hm0_u8, hm1_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag_hm, _ = cv2.cartToPolar(flow_hm[..., 0], flow_hm[..., 1])

        x = mag_img.ravel().astype(np.float64)
        y = mag_hm.ravel().astype(np.float64)
        corr = _correlation_dict(x, y)

        if corr["pearson_r"] is not None:
            per_pair[(t, t1)] = {
                "r": corr["pearson_r"],
                "p": corr["pearson_p"],
                "img_flow_mean": float(mag_img.mean()),
                "hm_flow_mean":  float(mag_hm.mean()),
            }
            img_means.append(float(mag_img.mean()))
            hm_means.append(float(mag_hm.mean()))

    aggregate = _correlation_dict(
        np.array(img_means, dtype=np.float64),
        np.array(hm_means,  dtype=np.float64),
    )
    mean_r = float(np.mean([v["r"] for v in per_pair.values()])) if per_pair else float("nan")

    return {"per_pair": per_pair, "mean_r": mean_r, "aggregate": aggregate}
