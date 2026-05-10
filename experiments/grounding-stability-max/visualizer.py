"""
visualizer.py
-------------
Two-row figure per sequence:
  Row 0 – frame image with GT bbox (lime) and predicted bbox (red)
  Row 1 – TAM heatmap blended on the frame, same bboxes overlaid
Columns = detected sampled frames, in temporal order.

Additional plots:
  save_token_variance_figure – distribution of mass-in-GT or mass-in-pred
                               across token positions, for all sequences
  save_iou_per_frame_figure  – IoU vs normalised frame position across all
                               sequences, to surface temporal drift
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
    hmap_u8      = (heatmap.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    hmap_resized = cv2.resize(hmap_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hmap_color   = cv2.applyColorMap(hmap_resized, cv2.COLORMAP_JET)
    hmap_rgb     = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    blended      = (0.5 * frame_rgb + 0.5 * hmap_rgb).clip(0, 255).astype(np.uint8)
    return blended


def save_sequence_figure(
    seq_name: str,
    expression: str,
    frames_pil: List[Image.Image],
    detected_sampled_frames: List[int],
    pred_boxes: Dict[int, Optional[Tuple]],
    gt_boxes: List[Optional[Tuple]],
    frame_heatmaps: Dict[int, np.ndarray],
    label_tokens: Dict[int, List[str]],
    sample_rate: int,
    save_path: str,
    max_cols: int = 16,
):
    cols   = detected_sampled_frames[:max_cols]
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
        orig_t    = sampled_t * sample_rate
        frame_rgb = _to_rgb(frames_pil[min(orig_t, len(frames_pil) - 1)])
        pred_box  = pred_boxes.get(orig_t)
        gt_box    = gt_boxes[orig_t] if orig_t < len(gt_boxes) else None
        heatmap   = frame_heatmaps.get(sampled_t)
        toks      = label_tokens.get(sampled_t, [])
        iou       = compute_iou(pred_box, gt_box)

        ax0 = axes[0, col_idx]
        ax0.imshow(frame_rgb)
        ax0.axis("off")
        _draw_box(ax0, gt_box,   color="lime")
        _draw_box(ax0, pred_box, color="red")
        tok_display = [t.replace("▁", " ").replace("Ġ", " ").strip() for t in toks]
        ax0.set_title(
            f"t={orig_t}  IoU={iou:.2f}\n[{', '.join(tok_display)}]",
            fontsize=6, pad=2,
        )

        ax1 = axes[1, col_idx]
        if heatmap is not None:
            ax1.imshow(_blend_heatmap(frame_rgb, heatmap))
        else:
            ax1.imshow(frame_rgb)
            H, W = frame_rgb.shape[:2]
            ax1.text(W // 2, H // 2, "no TAM",
                     ha="center", va="center", color="white",
                     fontsize=8, bbox=dict(boxstyle="round", fc="black", alpha=0.5))
        ax1.axis("off")
        _draw_box(ax1, gt_box,   color="lime")
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

    axes[0, 0].set_ylabel("Boxes\n(lime=GT  red=pred)", fontsize=6, labelpad=3)
    axes[1, 0].set_ylabel("TAM heatmap", fontsize=6, labelpad=3)

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
    """Encode optical flow (H, W, 2) as HSV: hue=direction, value=magnitude."""
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
        frame0  = _to_rgb(frames_pil[orig_t0])
        frame1  = _to_rgb(frames_pil[orig_t1])
        H, W    = frame0.shape[:2]

        gray0    = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1    = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        flow_img = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        pair_r = None
        if flow_pair_stats:
            pair_info = flow_pair_stats.get(f"{t0}-{t1}", {})
            pair_r    = pair_info.get("r") if isinstance(pair_info, dict) else None

        ax0 = axes[0, col_idx]
        ax0.imshow(_flow_to_rgb(flow_img))
        ax0.axis("off")
        title = f"t={orig_t0}→{orig_t1}"
        if pair_r is not None:
            title += f"\nr={pair_r:.3f}"
        ax0.set_title(title, fontsize=6, pad=2)

        ax1   = axes[1, col_idx]
        hmap0 = frame_heatmaps.get(t0)
        hmap1 = frame_heatmaps.get(t1)
        if hmap0 is not None and hmap1 is not None:
            hm0     = cv2.resize((hmap0 * 255).clip(0, 255).astype(np.uint8), (W, H))
            hm1     = cv2.resize((hmap1 * 255).clip(0, 255).astype(np.uint8), (W, H))
            flow_hm = cv2.calcOpticalFlowFarneback(hm0, hm1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ax1.imshow(_flow_to_rgb(flow_hm))
        else:
            ax1.text(0.5, 0.5, "no heatmap", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=8)
        ax1.axis("off")

    axes[0, 0].set_ylabel("Image flow",   fontsize=6, labelpad=3)
    axes[1, 0].set_ylabel("Heatmap flow", fontsize=6, labelpad=3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Correlation plots (unchanged from grounding-stability) ────────────────────

def _scatter_with_regression(ax, xs, ys, color, label, all_xs, all_ys):
    if xs:
        ax.scatter(xs, ys, color=color, alpha=0.6, s=20)
        all_xs.extend(xs)
        all_ys.extend(ys)


def _add_regression_line(ax, all_xs, all_ys, loc="upper left"):
    if len(all_xs) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(all_xs, all_ys)
        x_line = np.linspace(min(all_xs), max(all_xs), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5)
        ax.text(0.05, 0.95,
                f"r={r_val:.3f}  p={p_val:.3f}  n={len(all_xs)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


def save_correlation_plots(results: List[dict], save_dir: str) -> str:
    valid = [r for r in results if "error" not in r]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    cmap      = plt.get_cmap("tab20")
    seq_names = sorted({r["seq_name"] for r in valid})
    color_for = {name: cmap(i % 20) for i, name in enumerate(seq_names)}

    ax = axes[0, 0]
    all_iou_inst, all_inst = [], []
    for r in valid:
        iou_list = r.get("iou_per_frame", [])
        sr = r.get("sample_rate", 1)
        xs, ys = [], []
        for key, inst_val in r.get("instability", {}).items():
            t      = int(key.split("-")[0])
            orig_t = t * sr
            if orig_t < len(iou_list):
                xs.append(iou_list[orig_t])
                ys.append(inst_val)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_iou_inst, all_inst)
    ax.set_xlabel("IoU"); ax.set_ylabel("TAM Instability")
    ax.set_title("IoU vs TAM Instability")
    _add_regression_line(ax, all_iou_inst, all_inst, loc="upper right")

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
    ax.set_xlabel("IoU"); ax.set_ylabel("Mass in GT")
    ax.set_title("IoU vs Mass-in-GT")
    _add_regression_line(ax, all_iou_mass, all_mass, loc="upper left")

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
    ax.set_xlabel("Attention Accuracy (Mass-in-GT)"); ax.set_ylabel("IoU")
    ax.set_title("Attention Accuracy vs IoU")
    _add_regression_line(ax, all_mass_acc, all_iou_acc, loc="upper left")

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
    ax.set_xlabel("Image Flow Magnitude"); ax.set_ylabel("Heatmap Flow Magnitude")
    ax.set_title("Flow Alignment: Image vs Heatmap")
    _add_regression_line(ax, all_img_flow, all_hm_flow, loc="upper left")

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
    ax.set_xlabel("IoU"); ax.set_ylabel("Mass in Pred Box")
    ax.set_title("IoU vs Mass-in-Pred")
    _add_regression_line(ax, all_iou_pred, all_mass_pred, loc="upper left")

    ax = axes[2, 1]
    all_mgt, all_mpred = [], []
    for r in valid:
        mass_gt_dict   = r.get("mass_in_gt",   {})
        mass_pred_dict = r.get("mass_in_pred", {})
        xs, ys = [], []
        for k in mass_gt_dict:
            m_gt   = mass_gt_dict.get(k)
            m_pred = mass_pred_dict.get(k)
            if m_gt is not None and m_pred is not None:
                xs.append(m_gt)
                ys.append(m_pred)
        _scatter_with_regression(ax, xs, ys, color_for[r["seq_name"]], r["seq_name"],
                                 all_mgt, all_mpred)
    ax.set_xlabel("Mass in GT Box"); ax.set_ylabel("Mass in Pred Box")
    ax.set_title("Mass-in-GT vs Mass-in-Pred")
    _add_regression_line(ax, all_mgt, all_mpred, loc="upper left")

    plt.tight_layout()
    save_path = Path(save_dir) / "correlation_plots.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ── New: token variance plot ──────────────────────────────────────────────────

def save_token_variance_figure(
    results: List[dict],
    save_dir: str,
    mode: str,
) -> str:
    """
    Bar chart showing how mass (GT or pred) varies across token positions.

    For each token position (0 = first label token, 1 = second, …):
      - bar height  : mean mass across all sequences × frames
      - error bar   : std across sequences (each sequence contributes its
                      per-position average mass)
      - scatter dots: individual per-sequence averages

    Also marks the selected position for each sequence and shows the overall
    best position (highest mean) with a star.

    mode : "best_gt" → use mass-in-GT; "best_pred" → use mass-in-pred
    """
    mass_key = "per_token_avg_mass_gt" if mode == "best_gt" else "per_token_avg_mass_pred"
    ylabel   = "Mean mass-in-GT" if mode == "best_gt" else "Mean mass-in-pred"
    title    = (
        "Token-position mass-in-GT across sequences\n(bar = mean ± std over sequences)"
        if mode == "best_gt"
        else
        "Token-position mass-in-pred across sequences\n(bar = mean ± std over sequences)"
    )

    valid = [r for r in results if "error" not in r and mass_key in r]
    if not valid:
        return ""

    # Collect {pos: [avg_mass_per_sequence]}
    pos_to_seq_avgs: Dict[int, List[float]] = {}
    # Also collect {pos: [all individual frame masses across sequences]}
    pos_to_all_masses: Dict[int, List[float]] = {}
    # Which position was selected per sequence
    selected_positions: List[int] = []

    all_masses_key = "per_token_masses_gt" if mode == "best_gt" else "per_token_masses_pred"

    for r in valid:
        avg_dict    = r.get(mass_key, {})
        masses_dict = r.get(all_masses_key, {})
        sel_pos     = r.get("selected_token_pos")
        if sel_pos is not None:
            selected_positions.append(int(sel_pos))

        for pos_str, avg_val in avg_dict.items():
            pos = int(pos_str)
            pos_to_seq_avgs.setdefault(pos, []).append(float(avg_val))

        for pos_str, frame_masses in masses_dict.items():
            pos = int(pos_str)
            flat = [float(v) for v in frame_masses.values()]
            pos_to_all_masses.setdefault(pos, []).extend(flat)

    if not pos_to_seq_avgs:
        return ""

    positions    = sorted(pos_to_seq_avgs.keys())
    means        = [float(np.mean(pos_to_seq_avgs[p]))  for p in positions]
    stds         = [float(np.std(pos_to_seq_avgs[p]))   for p in positions]
    best_pos_idx = int(np.argmax(means))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # ── Left: bar chart with scatter overlay ─────────────────────────────────
    ax = axes[0]
    bar_colors = ["#2196F3"] * len(positions)
    bar_colors[best_pos_idx] = "#FF5722"

    bars = ax.bar(positions, means, yerr=stds, color=bar_colors,
                  capsize=5, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Scatter individual sequence averages per position
    rng = np.random.default_rng(42)
    for pos in positions:
        seq_vals = pos_to_seq_avgs[pos]
        jitter   = rng.uniform(-0.15, 0.15, size=len(seq_vals))
        ax.scatter(pos + jitter, seq_vals, color="black", s=18, alpha=0.6, zorder=3)

    ax.set_xlabel("Token position within label (0 = first token)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"pos {p}" for p in positions], rotation=45, ha="right")
    ax.set_ylim(0, 1)

    # Annotate best position
    ax.text(positions[best_pos_idx], means[best_pos_idx] + stds[best_pos_idx] + 0.02,
            "★ best", ha="center", va="bottom", fontsize=9, color="#FF5722", fontweight="bold")

    legend_patches = [
        mpatches.Patch(facecolor="#FF5722", alpha=0.7, label="Best overall position"),
        mpatches.Patch(facecolor="#2196F3", alpha=0.7, label="Other positions"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_title("Per-position mean ± std (dots = per-sequence averages)")

    # ── Right: box plot of all individual frame masses per position ───────────
    ax2 = axes[1]
    box_data = [pos_to_all_masses.get(p, []) for p in positions]
    bp = ax2.boxplot(box_data, positions=positions, widths=0.4, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#FF5722" if i == best_pos_idx else "#90CAF9")
        patch.set_alpha(0.7)

    ax2.set_xlabel("Token position within label")
    ax2.set_ylabel(ylabel)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"pos {p}" for p in positions], rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_title("Distribution of per-frame masses across all sequences")

    # Variance annotation per position
    for pos in positions:
        vals = pos_to_all_masses.get(pos, [])
        if vals:
            v = float(np.var(vals))
            ax2.text(pos, 0.02, f"var={v:.3f}", ha="center", va="bottom",
                     fontsize=6.5, color="dimgray")

    plt.tight_layout()
    fname     = f"token_variance_{mode}.png"
    save_path = Path(save_dir) / fname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ── New: IoU per frame (temporal drift) ──────────────────────────────────────

def save_iou_per_frame_figure(results: List[dict], save_dir: str) -> str:
    """
    Scatter plot of IoU vs normalised frame position (frame_idx / num_frames)
    across all sequences.

    Each sequence contributes one dot per frame, coloured distinctly.
    An OLS trend line is overlaid on the pooled data to reveal whether IoU
    systematically drops as the sequence progresses.
    """
    valid = [r for r in results if "error" not in r and r.get("iou_per_frame")]
    if not valid:
        return ""

    cmap      = plt.get_cmap("tab20")
    seq_names = sorted({r["seq_name"] for r in valid})
    color_for = {name: cmap(i % 20) for i, name in enumerate(seq_names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "IoU per frame across sequences\n"
        "(x = normalised frame position,  trend line = OLS over pooled data)",
        fontsize=11, fontweight="bold",
    )

    all_pos_norm: List[float] = []
    all_iou:      List[float] = []

    # ── Left: scatter coloured by sequence ───────────────────────────────────
    ax = axes[0]
    for r in valid:
        iou_list  = r["iou_per_frame"]
        n         = len(iou_list)
        if n == 0:
            continue
        xs = [i / max(n - 1, 1) for i in range(n)]
        ax.scatter(xs, iou_list,
                   color=color_for[r["seq_name"]], alpha=0.35, s=8,
                   label=r["seq_name"])
        all_pos_norm.extend(xs)
        all_iou.extend(iou_list)

    # Trend line
    if len(all_pos_norm) >= 3:
        slope, intercept, r_val, p_val, _ = stats.linregress(all_pos_norm, all_iou)
        x_line = np.linspace(0, 1, 200)
        ax.plot(x_line, slope * x_line + intercept, "k-", lw=2.5, label="OLS trend")
        ax.text(0.05, 0.05,
                f"r={r_val:.3f}  p={p_val:.3f}  n={len(all_iou)}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Normalised frame position  (0 = start,  1 = end)")
    ax.set_ylabel("IoU")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Scatter (coloured by sequence)")
    # Legend only if not too many sequences
    if len(seq_names) <= 20:
        handles, labels = ax.get_legend_handles_labels()
        seq_handles = [(h, l) for h, l in zip(handles, labels) if l != "OLS trend"]
        trend_handle = [(h, l) for h, l in zip(handles, labels) if l == "OLS trend"]
        ordered = seq_handles + trend_handle
        if ordered:
            ax.legend(*zip(*ordered), fontsize=6, markerscale=2,
                      ncol=2, loc="lower left")

    # ── Right: per-sequence mean IoU vs normalised quartile bins ─────────────
    ax2  = axes[1]
    bins = np.linspace(0, 1, 6)   # 5 equal-width bins: [0-0.2], [0.2-0.4], …
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    # Pool all (pos, iou) pairs and bin
    binned_iou: Dict[int, List[float]] = {i: [] for i in range(len(bins) - 1)}
    for r in valid:
        iou_list = r["iou_per_frame"]
        n        = len(iou_list)
        if n == 0:
            continue
        for i, iou_val in enumerate(iou_list):
            pos_norm = i / max(n - 1, 1)
            bin_idx  = min(int(pos_norm * (len(bins) - 1)), len(bins) - 2)
            binned_iou[bin_idx].append(iou_val)

    bin_means = [float(np.mean(binned_iou[i])) if binned_iou[i] else np.nan
                 for i in range(len(bins) - 1)]
    bin_stds  = [float(np.std(binned_iou[i]))  if binned_iou[i] else np.nan
                 for i in range(len(bins) - 1)]

    ax2.bar(bin_centres, bin_means, width=0.18, yerr=bin_stds,
            color="#42A5F5", alpha=0.7, capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Normalised frame position bin")
    ax2.set_ylabel("Mean IoU")
    ax2.set_xticks(bin_centres)
    ax2.set_xticklabels([f"{b:.1f}–{bins[i+1]:.1f}" for i, b in enumerate(bins[:-1])],
                        rotation=30, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_title("Mean IoU ± std per temporal bin (all sequences pooled)")

    # Annotate bar tops with count
    for i, (mean, std) in enumerate(zip(bin_means, bin_stds)):
        n_bin = len(binned_iou[i])
        if not np.isnan(mean):
            ax2.text(bin_centres[i], mean + (std if not np.isnan(std) else 0) + 0.01,
                     f"n={n_bin}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    save_path = Path(save_dir) / "iou_per_frame.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)
