"""
visualizer.py
-------------
Two-row figure per sequence:
  Row 0 – frame image with GT bbox (lime) and predicted bbox (red)
  Row 1 – TAM heatmap blended on the frame, same bboxes overlaid
Columns = detected sampled frames, in temporal order.
Column title = sampled frame index + label tokens used for TAM.
Figure title  = sequence name and expression.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

from metrics import compute_iou


def _to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def _draw_box(ax, box, color: str, lw: float = 2.0):
    if box is None:
        return
    x1, y1, x2, y2 = box
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)


def _blend_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Resize heatmap to frame size, apply JET colormap, blend 50/50."""
    H, W = frame_rgb.shape[:2]
    hmap_u8 = (heatmap.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    hmap_resized = cv2.resize(hmap_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hmap_color = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    hmap_rgb = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    blended = (0.5 * frame_rgb + 0.5 * hmap_rgb).clip(0, 255).astype(np.uint8)
    return blended


def save_sequence_figure(
    seq_name: str,
    expression: str,
    frames_pil: List[Image.Image],
    detected_sampled_frames: List[int],
    pred_boxes: Dict[int, Optional[Tuple]],    # original frame idx → box
    gt_boxes: List[Optional[Tuple]],           # indexed by original frame idx
    frame_heatmaps: Dict[int, np.ndarray],     # sampled frame idx → (H_tam, W_tam) [0,1]
    label_tokens: Dict[int, List[str]],        # sampled frame idx → token strings
    sample_rate: int,
    save_path: str,
    max_cols: int = 16,
):
    """
    Save the two-row diagnostic figure for one (sequence, expression) pair.

    detected_sampled_frames : 0-indexed sampled frame indices that the model
                              emitted a detection for, in temporal order.
    pred_boxes              : keyed by *original* frame index.
    gt_boxes                : list indexed by original frame index.
    frame_heatmaps          : keyed by *sampled* frame index.
    label_tokens            : keyed by *sampled* frame index.
    """
    cols = detected_sampled_frames[:max_cols]
    n_cols = len(cols)
    if n_cols == 0:
        return

    col_w = 2.8
    fig, axes = plt.subplots(2, n_cols, figsize=(col_w * n_cols, 5.5))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"{seq_name}  |  \"{expression}\"",
        fontsize=9, fontweight="bold", y=1.02,
    )

    for col_idx, sampled_t in enumerate(cols):
        orig_t = sampled_t * sample_rate
        frame_rgb = _to_rgb(frames_pil[min(orig_t, len(frames_pil) - 1)])
        pred_box = pred_boxes.get(orig_t)
        gt_box = gt_boxes[orig_t] if orig_t < len(gt_boxes) else None
        heatmap = frame_heatmaps.get(sampled_t)
        toks = label_tokens.get(sampled_t, [])

        iou = compute_iou(pred_box, gt_box)

        # ── Row 0: bbox overlay ──────────────────────────────────────────
        ax0 = axes[0, col_idx]
        ax0.imshow(frame_rgb)
        ax0.axis("off")
        _draw_box(ax0, gt_box, color="lime")
        _draw_box(ax0, pred_box, color="red")

        # Clean token strings for display (strip BPE markers)
        tok_display = [t.replace("▁", " ").replace("Ġ", " ").strip() for t in toks]
        ax0.set_title(
            f"t={orig_t}  IoU={iou:.2f}\n[{', '.join(tok_display)}]",
            fontsize=6, pad=2,
        )

        # ── Row 1: TAM heatmap overlay ───────────────────────────────────
        ax1 = axes[1, col_idx]
        if heatmap is not None:
            blended = _blend_heatmap(frame_rgb, heatmap)
            ax1.imshow(blended)
        else:
            ax1.imshow(frame_rgb)
            H, W = frame_rgb.shape[:2]
            ax1.text(W // 2, H // 2, "no TAM",
                     ha="center", va="center", color="white",
                     fontsize=8, bbox=dict(boxstyle="round", fc="black", alpha=0.5))
        ax1.axis("off")
        _draw_box(ax1, gt_box, color="lime")
        _draw_box(ax1, pred_box, color="red")

        mass = None
        if heatmap is not None and gt_box is not None:
            from metrics import compute_mass_in_gt
            H_orig, W_orig = frame_rgb.shape[:2]
            mass = compute_mass_in_gt(heatmap, gt_box, H_orig, W_orig)
        ax1.set_title(
            f"mass-in-GT: {mass:.2f}" if mass is not None else "mass-in-GT: –",
            fontsize=6, pad=2,
        )

    # Row labels on the leftmost column
    axes[0, 0].set_ylabel("Boxes\n(lime=GT  red=pred)", fontsize=6, labelpad=3)
    axes[1, 0].set_ylabel("TAM heatmap", fontsize=6, labelpad=3)

    # Legend
    legend_elements = [
        mpatches.Patch(edgecolor="lime", facecolor="none", label="GT"),
        mpatches.Patch(edgecolor="red",  facecolor="none", label="Pred"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=7,
               framealpha=0.8, ncol=2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Optical flow ──────────────────────────────────────────────────────────────

def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Encode optical flow (H, W, 2) as an HSV image: hue=direction, value=magnitude."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_flow_figure(
    seq_name: str,
    expression: str,
    frames_pil: List[Image.Image],
    detected_sampled_frames: List[int],
    frame_heatmaps: Dict[int, np.ndarray],
    sample_rate: int,
    save_path: str,
    max_cols: int = 15,
    flow_pair_stats: Optional[Dict[str, dict]] = None,
):
    """
    Two-row figure of optical flow between consecutive detected sampled frames:
      Row 0 – dense optical flow of the RGB image (Farneback)
      Row 1 – dense optical flow of the TAM heatmap

    flow_pair_stats : optional {"t-t1": {"r": float, ...}} from compute_flow_correlation;
                      when provided, the Pearson r is shown in each column title.
    """
    pairs = [
        (detected_sampled_frames[i], detected_sampled_frames[i + 1])
        for i in range(len(detected_sampled_frames) - 1)
    ][:max_cols]

    if not pairs:
        return

    n_cols = len(pairs)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.5))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"{seq_name}  |  \"{expression}\"  – Optical Flow",
        fontsize=9, fontweight="bold", y=1.02,
    )

    for col_idx, (t0, t1) in enumerate(pairs):
        orig_t0 = min(t0 * sample_rate, len(frames_pil) - 1)
        orig_t1 = min(t1 * sample_rate, len(frames_pil) - 1)

        frame0 = _to_rgb(frames_pil[orig_t0])
        frame1 = _to_rgb(frames_pil[orig_t1])
        H, W = frame0.shape[:2]

        # Image flow
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        flow_img = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        pair_r = None
        if flow_pair_stats:
            pair_info = flow_pair_stats.get(f"{t0}-{t1}", {})
            pair_r = pair_info.get("r") if isinstance(pair_info, dict) else None

        ax0 = axes[0, col_idx]
        ax0.imshow(_flow_to_rgb(flow_img))
        ax0.axis("off")
        title = f"t={orig_t0}→{orig_t1}"
        if pair_r is not None:
            title += f"\nr={pair_r:.3f}"
        ax0.set_title(title, fontsize=6, pad=2)

        # Heatmap flow
        ax1 = axes[1, col_idx]
        hmap0 = frame_heatmaps.get(t0)
        hmap1 = frame_heatmaps.get(t1)

        if hmap0 is not None and hmap1 is not None:
            hm0 = cv2.resize((hmap0 * 255).clip(0, 255).astype(np.uint8), (W, H))
            hm1 = cv2.resize((hmap1 * 255).clip(0, 255).astype(np.uint8), (W, H))
            flow_hm = cv2.calcOpticalFlowFarneback(hm0, hm1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ax1.imshow(_flow_to_rgb(flow_hm))
        else:
            ax1.text(0.5, 0.5, "no heatmap", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=8)
        ax1.axis("off")

    axes[0, 0].set_ylabel("Image flow", fontsize=6, labelpad=3)
    axes[1, 0].set_ylabel("Heatmap flow", fontsize=6, labelpad=3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Correlation plots ─────────────────────────────────────────────────────────

def _scatter_with_regression(ax, xs, ys, color, label, all_xs, all_ys):
    """Plot per-sequence points and accumulate into global lists."""
    if xs:
        ax.scatter(xs, ys, color=color, alpha=0.6, s=20)
        all_xs.extend(xs)
        all_ys.extend(ys)


def _add_regression_line(ax, all_xs, all_ys, loc="upper left"):
    """Overlay OLS regression line with r/p annotation."""
    if len(all_xs) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(all_xs, all_ys)
        x_line = np.linspace(min(all_xs), max(all_xs), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5)
        ax.text(0.05, 0.95,
                f"r={r_val:.3f}  p={p_val:.3f}  n={len(all_xs)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def save_correlation_plots(results: List[dict], save_dir: str) -> str:
    """
    3×2 grid of scatter plots aggregated across all sequences:
      (0,0) IoU vs TAM Instability        — one point per consecutive frame pair
      (0,1) IoU vs Mass-in-GT             — one point per sampled frame
      (1,0) Attention Accuracy vs IoU     — mass-in-GT (x) predicting IoU (y)
      (1,1) Image Flow vs Heatmap Flow    — per-pair mean flow magnitudes
      (2,0) Mass-in-Pred vs IoU           — attention inside predicted bbox vs IoU
      (2,1) Mass-in-GT vs Mass-in-Pred    — GT-box attention vs pred-box attention
    Each panel has an OLS regression line and r/p annotation.
    """
    valid = [r for r in results if "error" not in r]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    cmap = plt.get_cmap("tab20")
    seq_names = sorted({r["seq_name"] for r in valid})
    color_for = {name: cmap(i % 20) for i, name in enumerate(seq_names)}

    # ── (0,0) IoU vs TAM Instability ────────────────────────────────────
    ax = axes[0, 0]
    all_iou_inst, all_inst = [], []
    for r in valid:
        iou_list = r.get("iou_per_frame", [])
        sr = r.get("sample_rate", 1)
        xs, ys = [], []
        for key, inst_val in r.get("instability", {}).items():
            t = int(key.split("-")[0])
            orig_t = t * sr
            if orig_t < len(iou_list):
                xs.append(iou_list[orig_t])
                ys.append(inst_val)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_iou_inst, all_inst)
    ax.set_xlabel("IoU")
    ax.set_ylabel("TAM Instability")
    ax.set_title("IoU vs TAM Instability")
    _add_regression_line(ax, all_iou_inst, all_inst, loc="upper right")

    # ── (0,1) IoU vs Mass-in-GT ──────────────────────────────────────────
    ax = axes[0, 1]
    all_iou_mass, all_mass = [], []
    for r in valid:
        iou_list = r.get("iou_per_frame", [])
        sr = r.get("sample_rate", 1)
        xs, ys = [], []
        for k, mass in r.get("mass_in_gt", {}).items():
            if mass is None:
                continue
            orig_t = int(k) * sr
            if orig_t < len(iou_list):
                xs.append(iou_list[orig_t])
                ys.append(mass)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_iou_mass, all_mass)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Mass in GT")
    ax.set_title("IoU vs Mass-in-GT")
    _add_regression_line(ax, all_iou_mass, all_mass, loc="upper left")

    # ── (1,0) Attention Accuracy: mass-in-GT (x) → IoU (y) ──────────────
    ax = axes[1, 0]
    all_mass_acc, all_iou_acc = [], []
    for r in valid:
        iou_list = r.get("iou_per_frame", [])
        sr = r.get("sample_rate", 1)
        xs, ys = [], []
        for k, mass in r.get("mass_in_gt", {}).items():
            if mass is None:
                continue
            orig_t = int(k) * sr
            if orig_t < len(iou_list):
                xs.append(mass)
                ys.append(iou_list[orig_t])
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_mass_acc, all_iou_acc)
    ax.set_xlabel("Attention Accuracy (Mass-in-GT)")
    ax.set_ylabel("IoU")
    ax.set_title("Attention Accuracy vs IoU")
    _add_regression_line(ax, all_mass_acc, all_iou_acc, loc="upper left")

    # ── (1,1) Flow Alignment: image flow mean mag (x) → heatmap flow mean mag (y) ──
    ax = axes[1, 1]
    all_img_flow, all_hm_flow = [], []
    for r in valid:
        xs, ys = [], []
        for v in r.get("flow_correlation", {}).get("per_pair", {}).values():
            if isinstance(v, dict) and v.get("img_flow_mean") is not None:
                xs.append(v["img_flow_mean"])
                ys.append(v["hm_flow_mean"])
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_img_flow, all_hm_flow)
    ax.set_xlabel("Image Flow Magnitude")
    ax.set_ylabel("Heatmap Flow Magnitude")
    ax.set_title("Flow Alignment: Image vs Heatmap")
    _add_regression_line(ax, all_img_flow, all_hm_flow, loc="upper left")

    # ── (2,0) Mass-in-Pred vs IoU ────────────────────────────────────────
    ax = axes[2, 0]
    all_iou_pred, all_mass_pred = [], []
    for r in valid:
        iou_list = r.get("iou_per_frame", [])
        sr = r.get("sample_rate", 1)
        xs, ys = [], []
        for k, mass in r.get("mass_in_pred", {}).items():
            if mass is None:
                continue
            orig_t = int(k) * sr
            if orig_t < len(iou_list):
                xs.append(iou_list[orig_t])
                ys.append(mass)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_iou_pred, all_mass_pred)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Mass in Pred Box")
    ax.set_title("IoU vs Mass-in-Pred")
    _add_regression_line(ax, all_iou_pred, all_mass_pred, loc="upper left")

    # ── (2,1) Mass-in-GT vs Mass-in-Pred ────────────────────────────────
    ax = axes[2, 1]
    all_mgt, all_mpred = [], []
    for r in valid:
        xs, ys = [], []
        mass_gt_dict   = r.get("mass_in_gt",   {})
        mass_pred_dict = r.get("mass_in_pred", {})
        for k in mass_gt_dict:
            m_gt   = mass_gt_dict.get(k)
            m_pred = mass_pred_dict.get(k)
            if m_gt is not None and m_pred is not None:
                xs.append(m_gt)
                ys.append(m_pred)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_mgt, all_mpred)
    ax.set_xlabel("Mass in GT Box")
    ax.set_ylabel("Mass in Pred Box")
    ax.set_title("Mass-in-GT vs Mass-in-Pred")
    _add_regression_line(ax, all_mgt, all_mpred, loc="upper left")

    plt.tight_layout()
    save_path = Path(save_dir) / "correlation_plots.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
