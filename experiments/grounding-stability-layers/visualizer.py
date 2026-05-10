"""
visualizer.py
-------------
Per-sequence layer-grid figure (rows = detected sampled frames, cols = variants)
and aggregate mass-vs-layer / mass-vs-cumavg curves.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def _draw_box(ax, box, color: str, lw: float = 1.5):
    if box is None:
        return
    x1, y1, x2, y2 = box
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)


def _blend_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    hmap_u8 = (heatmap.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    hmap_resized = cv2.resize(hmap_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hmap_color = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    hmap_rgb = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    return (0.5 * frame_rgb + 0.5 * hmap_rgb).clip(0, 255).astype(np.uint8)


def save_layer_grid_figure(
    seq_name: str,
    expression: str,
    grid_label: str,
    frames_pil: List[Image.Image],
    detected_sampled_frames: List[int],
    sample_rate: int,
    gt_boxes: List[Optional[Tuple]],
    pred_boxes: Dict[int, Optional[Tuple]],
    variant_keys: List[str],
    per_variant_heatmaps: Dict[str, Dict[int, np.ndarray]],
    save_path: str,
    max_frames: int = 8,
):
    """
    Grid figure: rows = detected frames (capped at max_frames),
                 cols = variant_keys (one heatmap each).

    Top label row holds the variant name; first column shows the raw frame
    with GT (lime) / pred (red) bbox.
    """
    rows = detected_sampled_frames[:max_frames]
    n_rows = len(rows)
    n_vars = len(variant_keys)
    n_cols = n_vars + 1  # leftmost column: raw frame with bboxes
    if n_rows == 0 or n_vars == 0:
        return

    col_w, row_h = 2.6, 2.4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(col_w * n_cols, row_h * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"{seq_name} | \"{expression}\"  ({grid_label})",
        fontsize=11, fontweight="bold", y=1.005,
    )

    for r, sampled_t in enumerate(rows):
        orig_t = sampled_t * sample_rate
        frame_rgb = _to_rgb(frames_pil[min(orig_t, len(frames_pil) - 1)])
        gt_box = gt_boxes[orig_t] if orig_t < len(gt_boxes) else None
        pred_box = pred_boxes.get(orig_t)

        ax_raw = axes[r, 0]
        ax_raw.imshow(frame_rgb)
        ax_raw.axis("off")
        _draw_box(ax_raw, gt_box, "lime")
        _draw_box(ax_raw, pred_box, "red")
        ax_raw.set_ylabel(f"t={orig_t}", fontsize=8)
        if r == 0:
            ax_raw.set_title("frame", fontsize=8)

        for c, vkey in enumerate(variant_keys, start=1):
            ax = axes[r, c]
            hmap = per_variant_heatmaps.get(vkey, {}).get(sampled_t)
            if hmap is not None:
                ax.imshow(_blend_heatmap(frame_rgb, hmap))
            else:
                ax.imshow(frame_rgb)
                H, W = frame_rgb.shape[:2]
                ax.text(W // 2, H // 2, "—",
                        ha="center", va="center", color="white",
                        fontsize=10,
                        bbox=dict(boxstyle="round", fc="black", alpha=0.5))
            ax.axis("off")
            _draw_box(ax, gt_box, "lime")
            _draw_box(ax, pred_box, "red")
            if r == 0:
                ax.set_title(vkey, fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── aggregate plots ──────────────────────────────────────────────────────────

def _collect_curve(results, variant_keys):
    """Return (xs, mean_mass_gt, mean_mass_pred, n_per_variant)."""
    mg, mp, n_seq = [], [], []
    for vk in variant_keys:
        gts, preds = [], []
        for r in results:
            v = r.get("per_variant", {}).get(vk)
            if v is None:
                continue
            if v.get("mean_mass_in_gt") is not None:
                gts.append(v["mean_mass_in_gt"])
            if v.get("mean_mass_in_pred") is not None:
                preds.append(v["mean_mass_in_pred"])
        mg.append(float(np.mean(gts)) if gts else np.nan)
        mp.append(float(np.mean(preds)) if preds else np.nan)
        n_seq.append(min(len(gts), len(preds)))
    return mg, mp, n_seq


def save_layer_curves(
    results: List[dict],
    save_dir: str,
    layer_indices: List[int],
    cumavg_Ks: List[int],
):
    """
    Two figures:
      mass_vs_layer.png  — x = layer index, y = mean mass-in-GT / mass-in-pred
      mass_vs_cumavg.png — x = K, y = same two metrics
    """
    save_dir = Path(save_dir)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None, None

    layer_keys = [f"layer_{L}" for L in layer_indices]
    cumavg_keys = [f"cumavg_{K}" for K in cumavg_Ks]

    layer_mg, layer_mp, layer_n = _collect_curve(valid, layer_keys)
    cum_mg, cum_mp, cum_n = _collect_curve(valid, cumavg_keys)

    layer_path = save_dir / "mass_vs_layer.png"
    cum_path = save_dir / "mass_vs_cumavg.png"

    # Layer figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = layer_indices
    ax.plot(xs, layer_mg, marker="o", label="mean mass-in-GT", color="tab:green")
    ax.plot(xs, layer_mp, marker="s", label="mean mass-in-pred", color="tab:red")
    ax.set_xlabel("Layer index (negative from final)")
    ax.set_ylabel("Mean mass (averaged over sequences × detected frames)")
    ax.set_title(f"Per-layer logit-lens TAM  (n_seq={len(valid)})")
    ax.invert_xaxis()  # show -1 on the right
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(layer_path, dpi=140)
    plt.close(fig)

    # Cumavg figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = cumavg_Ks
    ax.plot(xs, cum_mg, marker="o", label="mean mass-in-GT", color="tab:green")
    ax.plot(xs, cum_mp, marker="s", label="mean mass-in-pred", color="tab:red")
    ax.set_xlabel("K  (avg of last K layers)")
    ax.set_ylabel("Mean mass (averaged over sequences × detected frames)")
    ax.set_title(f"Cumulative-avg logit-lens TAM  (n_seq={len(valid)})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cum_path, dpi=140)
    plt.close(fig)

    return str(layer_path), str(cum_path)


# ── per-sequence mass-in-GT curves ───────────────────────────────────────────

def _collect_per_seq_curve(results, variant_keys, metric="mean_mass_in_gt"):
    """
    Returns (curves, seq_names) where curves is shape (n_seq, n_variants),
    NaN where the metric is missing.
    """
    curves: List[np.ndarray] = []
    names:  List[str] = []
    for r in results:
        per_var = r.get("per_variant", {})
        row = []
        for vk in variant_keys:
            v = per_var.get(vk)
            if v is None or v.get(metric) is None:
                row.append(np.nan)
            else:
                row.append(float(v[metric]))
        curves.append(np.array(row, dtype=np.float64))
        names.append(f"{r.get('seq_name','?')}_exp{r.get('exp_id','?')}")
    return np.vstack(curves), names


def _plot_per_seq_curves(
    ax,
    xs: List[int],
    curves: np.ndarray,
    seq_names: List[str],
    line_color: str = "tab:green",
    seq_alpha: float = 0.35,
    seq_lw: float = 0.8,
):
    """Thin per-sequence lines + bold mean ± std band."""
    n_seq = curves.shape[0]
    for i in range(n_seq):
        ax.plot(xs, curves[i], color="tab:gray", alpha=seq_alpha,
                linewidth=seq_lw)

    mean = np.nanmean(curves, axis=0)
    std  = np.nanstd(curves, axis=0)
    ax.fill_between(xs, mean - std, mean + std, color=line_color,
                    alpha=0.18, label="±1 std")
    ax.plot(xs, mean, marker="o", color=line_color, linewidth=2.4,
            label=f"mean (n={n_seq})")


def save_mass_in_gt_layer_comparison(
    results: List[dict],
    save_dir: str,
    layer_indices: List[int],
):
    """
    Per-sequence + mean mass-in-GT vs layer index.
    Each sequence is a faint grey line; the mean (with ±1 std band) is bold.
    """
    save_dir = Path(save_dir)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    layer_keys = [f"layer_{L}" for L in layer_indices]
    curves, names = _collect_per_seq_curve(valid, layer_keys, "mean_mass_in_gt")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_per_seq_curves(ax, layer_indices, curves, names, line_color="tab:green")
    ax.set_xlabel("Layer index (negative from final)")
    ax.set_ylabel("Mass-in-GT  (per-sequence mean)")
    ax.set_title(f"Per-layer mass-in-GT — per sequence  (n={len(valid)})")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    out = save_dir / "mass_in_gt_layer_comparison.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def save_mass_in_gt_cumavg_comparison(
    results: List[dict],
    save_dir: str,
    cumavg_Ks: List[int],
):
    """
    Per-sequence + mean mass-in-GT vs K (cumulative avg of last K layers).
    """
    save_dir = Path(save_dir)
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    cumavg_keys = [f"cumavg_{K}" for K in cumavg_Ks]
    curves, names = _collect_per_seq_curve(valid, cumavg_keys, "mean_mass_in_gt")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_per_seq_curves(ax, cumavg_Ks, curves, names, line_color="tab:blue")
    ax.set_xlabel("K  (avg of last K layers)")
    ax.set_ylabel("Mass-in-GT  (per-sequence mean)")
    ax.set_title(f"Cumulative-avg mass-in-GT — per sequence  (n={len(valid)})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    out = save_dir / "mass_in_gt_cumavg_comparison.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)
