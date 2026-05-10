"""
visualizer.py
-------------
Visualisations for the RefCOCO grounding experiment.

save_item_figure   – two-column figure: image+boxes | TAM heatmap+boxes
save_scatter_plots – correlation scatter plots across the full result set
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
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
    ))


def _blend_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    hmap_u8 = (heatmap.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    hmap_resized = cv2.resize(hmap_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hmap_color = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    hmap_rgb = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    return (0.5 * frame_rgb + 0.5 * hmap_rgb).clip(0, 255).astype(np.uint8)


def save_item_figure(
    image_pil: Image.Image,
    expression: str,
    pred_box: Optional[Tuple],
    gt_box: Optional[Tuple],
    heatmap: Optional[np.ndarray],  # (H_tam, W_tam) float32 [0,1]
    label_tokens: List[str],
    save_path: str,
    image_id: int = 0,
):
    """
    Two-column figure for one (image, expression) pair.
      Col 0 – image with GT (lime) and predicted (red) bounding boxes
      Col 1 – TAM heatmap blended on image, same boxes overlaid
    """
    frame_rgb = _to_rgb(image_pil)
    iou = compute_iou(pred_box, gt_box)
    tok_display = [t.replace("▁", " ").replace("Ġ", " ").strip() for t in label_tokens]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        f'image {image_id}  |  "{expression}"',
        fontsize=9, fontweight="bold", y=1.02,
    )

    # Col 0: boxes
    axes[0].imshow(frame_rgb)
    axes[0].axis("off")
    _draw_box(axes[0], gt_box,   color="lime")
    _draw_box(axes[0], pred_box, color="red")
    axes[0].set_title(f"IoU = {iou:.3f}", fontsize=8, pad=3)

    # Col 1: heatmap
    if heatmap is not None:
        axes[1].imshow(_blend_heatmap(frame_rgb, heatmap))
    else:
        axes[1].imshow(frame_rgb)
        H, W = frame_rgb.shape[:2]
        axes[1].text(W // 2, H // 2, "no TAM", ha="center", va="center",
                     color="white", fontsize=8,
                     bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    axes[1].axis("off")
    _draw_box(axes[1], gt_box,   color="lime")
    _draw_box(axes[1], pred_box, color="red")
    axes[1].set_title(
        f"label tokens: {tok_display}" if tok_display else "label tokens: –",
        fontsize=7, pad=3,
    )

    legend = [
        mpatches.Patch(edgecolor="lime", facecolor="none", label="GT"),
        mpatches.Patch(edgecolor="red",  facecolor="none", label="Pred"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=7, framealpha=0.8, ncol=2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plots(results: List[dict], save_dir: str) -> str:
    """
    2×2 scatter plot grid aggregated across all items:
      (0,0) mass-in-GT vs IoU       – attention localisation predicting accuracy
      (0,1) mass-in-pred vs IoU     – self-consistency of prediction + attention
      (1,0) entropy vs IoU          – focused attention → better grounding?
      (1,1) mass-in-GT vs mass-in-pred – GT-box vs pred-box attention alignment
    Each panel has an OLS regression line with r and p annotations.
    """
    valid = [r for r in results if "error" not in r]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def _scatter(ax, xs, ys, xlabel, ylabel, title):
        if xs:
            ax.scatter(xs, ys, alpha=0.4, s=15, color="steelblue")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if len(xs) >= 3:
            slope, intercept, r, p, _ = stats.linregress(xs, ys)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5)
            ax.text(0.05, 0.95, f"r={r:.3f}  p={p:.3f}  n={len(xs)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    mass_gt   = [r["mass_in_gt"]   for r in valid if r.get("mass_in_gt")   is not None]
    mass_pred = [r["mass_in_pred"] for r in valid if r.get("mass_in_pred") is not None]
    iou_mgt   = [r["iou"] for r in valid if r.get("mass_in_gt")   is not None]
    iou_mpred = [r["iou"] for r in valid if r.get("mass_in_pred") is not None]
    entropy   = [r["entropy"] for r in valid if r.get("entropy") is not None]
    iou_ent   = [r["iou"] for r in valid if r.get("entropy") is not None]

    _scatter(axes[0, 0], mass_gt,   iou_mgt,   "Mass-in-GT",   "IoU", "Attention Accuracy vs IoU")
    _scatter(axes[0, 1], mass_pred, iou_mpred, "Mass-in-Pred", "IoU", "Mass-in-Pred vs IoU")
    _scatter(axes[1, 0], entropy,   iou_ent,   "Attention Entropy", "IoU", "Entropy vs IoU")

    # mass-in-GT vs mass-in-pred (must have both)
    both = [(r["mass_in_gt"], r["mass_in_pred"]) for r in valid
            if r.get("mass_in_gt") is not None and r.get("mass_in_pred") is not None]
    xs_b = [v[0] for v in both]
    ys_b = [v[1] for v in both]
    _scatter(axes[1, 1], xs_b, ys_b, "Mass-in-GT", "Mass-in-Pred", "GT vs Pred Attention Alignment")

    plt.tight_layout()
    save_path = Path(save_dir) / "scatter_plots.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
