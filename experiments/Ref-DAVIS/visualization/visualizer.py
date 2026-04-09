"""
visualizer.py
-------------
All visualization functions for the benchmark and TAM diagnostic results.

Functions
---------
  plot_j_curves()              - per-sequence J over time
  plot_tam_centroids()         - TAM centroid trajectory vs GT centroid
  plot_frame_mass_heatmap()    - (tokens x frames) attention mass heatmap
  plot_failure_gallery()       - visual grid of failure cases with overlays
  plot_aggregate_summary()     - bar charts, distributions, metric table
  plot_temporal_binding()      - Experiment 5: binding scores by prompt
  save_failure_case()          - save annotated frames for one failure
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ─── Colours ─────────────────────────────────────────────────────────────────
CMAP_QUAL = plt.cm.tab10.colors
FAILURE_COLORS = {
    "SUCCESS":           "#2ecc71",
    "NEVER_FOUND":       "#e74c3c",
    "LOST_TRACK":        "#e67e22",
    "PARTIAL_TRACK":     "#f1c40f",
    "IDENTITY_SWAP":     "#9b59b6",
    "TEMPORAL_COLLAPSE": "#3498db",
    "ATTENTION_DRIFT":   "#1abc9c",
    "OCCLUSION_FAIL":    "#e91e63",
    "SCALE_FAILURE":     "#795548",
    "UNSTABLE":          "#607d8b",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _overlay_mask(frame_arr: np.ndarray, mask: np.ndarray,
                  color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Overlay binary mask on RGB frame array."""
    out = frame_arr.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + v * alpha,
                                out[:, :, c])
    return out.astype(np.uint8)


def _overlay_box(frame_arr: np.ndarray, box: Tuple, color=(255, 0, 0),
                 thickness=2) -> np.ndarray:
    """Draw bounding box on frame array."""
    out = frame_arr.copy()
    if box is None:
        return out
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _overlay_tam(frame_arr: np.ndarray, tam_map: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay TAM heatmap (2D) on frame array."""
    h, w = frame_arr.shape[:2]
    smooth = cv2.resize(tam_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    smooth = (smooth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(smooth, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (frame_arr * (1 - alpha) + colored * alpha).astype(np.uint8)


# ─── Plot 1: J Curves ─────────────────────────────────────────────────────────

def plot_j_curves(
    results: List[dict],
    save_path: str,
    title: str = "Per-Frame J (IoU) over Time",
    max_seqs: int = 20,
):
    """
    Plot J over frame index for each sequence.
    Color-coded by primary failure mode.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0
    for r in results[:max_seqs]:
        j = np.array(r["metrics"]["J_per_frame"])
        mode = r.get("failure", {}).get("primary_failure", "UNKNOWN")
        color = FAILURE_COLORS.get(mode, "#aaaaaa")
        ax.plot(j, color=color, alpha=0.6, linewidth=1.2,
                label=f"{r['seq_name']}[{r['exp_id']}]")
        plotted += 1

    # Mean J across all
    all_j = [r["metrics"]["J_per_frame"] for r in results if "metrics" in r]
    if all_j:
        min_len = min(len(j) for j in all_j)
        mean_j = np.mean([j[:min_len] for j in all_j], axis=0)
        ax.plot(mean_j, "k--", linewidth=2.5, label="Mean J", zorder=10)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="J=0.5 threshold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Jaccard (IoU)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Legend for failure modes
    handles = [mpatches.Patch(color=c, label=m) for m, c in FAILURE_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ─── Plot 2: TAM Centroid Trajectories ───────────────────────────────────────

def plot_tam_centroids(
    drift_result: dict,
    frames_pil: List[Image.Image],
    save_path: str,
    seq_name: str = "",
):
    """
    Plot TAM attention centroid trajectory vs GT centroid over frames.
    """
    T = len(frames_pil)
    per_frame_centroid = drift_result.get("per_frame_centroid")  # (T, 2)
    gt_centroids = drift_result.get("gt_centroids")              # (T, 2) or None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: centroid paths
    ax = axes[0]
    if per_frame_centroid is not None:
        cx = per_frame_centroid[:, 0]
        cy = per_frame_centroid[:, 1]
        sc = ax.scatter(cx, cy, c=np.arange(T), cmap="viridis",
                        s=40, zorder=3, label="TAM centroid")
        ax.plot(cx, cy, "b-", alpha=0.4, linewidth=1)
        plt.colorbar(sc, ax=ax, label="Frame")

    if gt_centroids is not None:
        gx = gt_centroids[:, 0]
        gy = gt_centroids[:, 1]
        ax.scatter(gx, gy, c="red", marker="x", s=50, zorder=4, label="GT centroid")
        ax.plot(gx, gy, "r--", alpha=0.4, linewidth=1)

    ax.set_title(f"Centroid Trajectory — {seq_name}")
    ax.set_xlabel("x (TAM pixels)")
    ax.set_ylabel("y (TAM pixels)")
    ax.legend()
    ax.invert_yaxis()

    # Right: displacement error over time
    ax2 = axes[1]
    disp = drift_result.get("displacement_error")
    if disp is not None:
        ax2.plot(disp, "b-o", markersize=3, linewidth=1.5, label="L2 centroid error")
        ax2.axhline(disp.mean() if hasattr(disp, "mean") else np.mean(disp),
                    color="r", linestyle="--", label=f"Mean={np.nanmean(disp):.1f}px")
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Centroid Displacement Error (px)")
        ax2.set_title("Attention Drift Error Over Time")
        ax2.legend()
    else:
        velocity = drift_result.get("centroid_velocity")
        if velocity is not None:
            ax2.plot(velocity, "g-o", markersize=3, linewidth=1.5)
            ax2.set_xlabel("Frame Transition")
            ax2.set_ylabel("Centroid Velocity (px/frame)")
            ax2.set_title("Attention Centroid Velocity")

    fig.suptitle(f"TAM Attention Drift Analysis — {seq_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ─── Plot 3: Frame Mass Heatmap ──────────────────────────────────────────────

def plot_frame_mass_heatmap(
    tam_result: dict,
    save_path: str,
    seq_name: str = "",
    expression: str = "",
    max_tokens: int = 60,
):
    """
    Heatmap of (tokens x frames) frame attention mass.
    Reveals temporal collapse (all rows same) vs temporal diversity.
    """
    frame_mass = tam_result["frame_mass"]  # (num_tokens, T)
    T = tam_result["vision_shape"][0]
    gen_tokens = tam_result.get("gen_tokens", [])

    # Filter to active tokens
    active = [i for i, t in enumerate(gen_tokens)
              if "<|" not in t and t.strip() and i < frame_mass.shape[0]]
    if len(active) > max_tokens:
        active = active[:max_tokens]

    if not active:
        return

    mass_sub = frame_mass[active]  # (N_active, T)
    token_labels = [gen_tokens[i].replace(" ", "_")[:10] for i in active]

    fig, ax = plt.subplots(figsize=(max(8, T * 0.8), max(6, len(active) * 0.25)))
    im = ax.imshow(mass_sub, aspect="auto", cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Attention Mass")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Generated Token")
    ax.set_xticks(range(T))
    ax.set_xticklabels([str(t) for t in range(T)], fontsize=8)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(token_labels, fontsize=6)
    ax.set_title(
        f"Frame Attention Mass — {seq_name}\n\"{expression[:60]}\"",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Plot 4: Failure Gallery ─────────────────────────────────────────────────

def plot_failure_gallery(
    results: List[dict],
    save_path: str,
    frames_key: str = "frames_pil",
    max_seqs: int = 8,
    frames_per_seq: int = 4,
):
    """
    Grid: each row = one sequence, columns = sampled frames with
    GT mask (green) and predicted mask (red) overlaid.
    """
    valid = [r for r in results if frames_key in r and "metrics" in r][:max_seqs]
    if not valid:
        return

    n_rows = len(valid)
    n_cols = frames_per_seq
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(valid):
        frames = r[frames_key]
        gt_masks = r.get("gt_masks", [])
        pred_masks = r.get("pred_masks", [])
        T = len(frames)
        sample_idx = np.linspace(0, T - 1, n_cols, dtype=int)
        mode = r.get("failure", {}).get("primary_failure", "?")
        color = FAILURE_COLORS.get(mode, "#aaaaaa")

        for col, t in enumerate(sample_idx):
            ax = axes[row, col]
            frame_arr = np.array(frames[t])
            if t < len(gt_masks) and gt_masks[t] is not None:
                frame_arr = _overlay_mask(frame_arr, gt_masks[t], color=(0, 255, 0))
            if t < len(pred_masks) and pred_masks[t] is not None:
                frame_arr = _overlay_mask(frame_arr, pred_masks[t],
                                          color=(255, 50, 50), alpha=0.35)
            ax.imshow(frame_arr)
            ax.set_title(f"t={t}", fontsize=7)
            ax.axis("off")

            if col == 0:
                label = f"{r['seq_name']}\n[{mode}]\nJ={r['metrics']['mean_J']:.2f}"
                ax.set_ylabel(label, fontsize=7, color=color)

    fig.suptitle("Failure Gallery (Green=GT, Red=Pred)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─── Plot 5: Aggregate Summary ───────────────────────────────────────────────

def plot_aggregate_summary(
    agg_metrics: dict,
    failure_summary: dict,
    save_path: str,
    model_name: str = "Qwen3-VL-8B",
):
    """
    4-panel summary: J distribution, failure modes pie,
    temporal metrics bar, J&F comparison table.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark Summary — {model_name}", fontsize=14)

    # Panel 1: Failure mode distribution (pie)
    ax = axes[0, 0]
    dist = failure_summary.get("primary_distribution", {})
    labels = [k for k, v in dist.items() if v["count"] > 0]
    sizes = [dist[l]["count"] for l in labels]
    colors = [FAILURE_COLORS.get(l, "#aaaaaa") for l in labels]
    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               textprops={"fontsize": 8})
    ax.set_title("Primary Failure Mode Distribution")

    # Panel 2: Key metric bars
    ax = axes[0, 1]
    metric_keys = ["mean_J", "mean_F", "JF", "success_rate_50"]
    metric_labels = ["Mean J", "Mean F", "J&F", "Success@50"]
    values = [agg_metrics.get(k, 0) for k in metric_keys]
    bars = ax.bar(metric_labels, values, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Primary Benchmark Metrics")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: Temporal coherence metrics
    ax = axes[1, 0]
    tc_keys = ["J_decay", "J_variance", "J_first", "J_last"]
    tc_labels = ["J-Decay\n(neg=bad)", "J-Variance\n(high=bad)",
                 "J First Frame", "J Last Frame"]
    tc_values = [agg_metrics.get(k, 0) for k in tc_keys]
    tc_colors = ["#e74c3c" if v < 0 else "#3498db" for v in tc_values]
    bars2 = ax.bar(tc_labels, tc_values, color=tc_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Temporal Coherence Metrics")
    for bar, val in zip(bars2, tc_values):
        ypos = val + 0.01 if val >= 0 else val - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 4: Summary text table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["J&F", f"{agg_metrics.get('JF', 0):.4f}"],
        ["Mean J", f"{agg_metrics.get('mean_J', 0):.4f}"],
        ["Mean F", f"{agg_metrics.get('mean_F', 0):.4f}"],
        ["J-Decay", f"{agg_metrics.get('J_decay', 0):.4f}"],
        ["J-Variance", f"{agg_metrics.get('J_variance', 0):.4f}"],
        ["Success@0.5", f"{agg_metrics.get('success_rate_50', 0):.4f}"],
        ["Success@0.75", f"{agg_metrics.get('success_rate_75', 0):.4f}"],
        ["Num Sequences", str(agg_metrics.get("num_sequences", 0))],
        ["Collapse Rate", f"{agg_metrics.get('collapse_rate', 0):.4f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    ax.set_title(f"Summary Table — {model_name}", pad=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ─── Plot 6: Temporal Binding ─────────────────────────────────────────────────

def plot_temporal_binding(
    binding_result: dict,
    save_path: str,
    seq_name: str = "",
):
    """Bar chart of binding scores per prompt (Experiment 5)."""
    scores = binding_result.get("binding_scores", {})
    if not scores:
        return
    labels = list(scores.keys())
    values = [scores[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    colors = [CMAP_QUAL[i % len(CMAP_QUAL)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Attention Mass on Target Frames")
    ax.set_title(
        f"Temporal Binding Score by Prompt — {seq_name}\n"
        f"Steerable: {binding_result.get('is_steerable', False)}, "
        f"Mean: {binding_result.get('mean_binding', 0):.3f}"
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ─── Save Individual Failure Case ─────────────────────────────────────────────

def plot_iou_curves(
    results: List[dict],
    save_path: str,
    title: str = "Per-Frame IoU over Time",
    max_seqs: int = 20,
):
    """
    Plot IoU over frame index for each sequence (VOT equivalent of plot_j_curves).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results[:max_seqs]:
        iou = np.array(r["metrics"]["iou_per_frame"])
        ax.plot(iou, alpha=0.6, linewidth=1.2,
                label=f"{r['seq_name']}[{r['exp_id']}]")

    all_iou = [r["metrics"]["iou_per_frame"] for r in results if "metrics" in r]
    if all_iou:
        min_len = min(len(i) for i in all_iou)
        mean_iou = np.mean([i[:min_len] for i in all_iou], axis=0)
        ax.plot(mean_iou, "k--", linewidth=2.5, label="Mean IoU", zorder=10)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="IoU=0.5 threshold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_vot_result_case(
    result: dict,
    save_dir: str,
    num_columns: int = 5,
    sample_rate: int = 1,
):
    """
    Save a grid figure for a single VOT result showing sampled frames with GT
    bbox (green) and predicted bbox (red) overlaid.
    sample_rate controls which frames are shown (every Nth frame).
    """
    os.makedirs(save_dir, exist_ok=True)
    seq = result.get("seq_name", "seq")
    exp = result.get("exp_id", "0")

    frames = result.get("frames_pil", [])[::sample_rate]
    gt_boxes = result.get("gt_boxes", [])[::sample_rate]
    pred_boxes = result.get("boxes", [])[::sample_rate]
    iou_per_frame = result.get("metrics", {}).get("iou_per_frame", [])
    T = len(frames)
    if T == 0:
        return

    num_rows = math.ceil(T / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns,
                             figsize=(num_columns * 3, num_rows * 3))
    axes = np.array(axes).reshape(num_rows, num_columns)

    for t in range(num_rows * num_columns):
        ax = axes[t // num_columns, t % num_columns]
        if t < T:
            frame_arr = np.array(frames[t])
            gt_b = gt_boxes[t] if t < len(gt_boxes) else None
            pred_b = pred_boxes[t] if t < len(pred_boxes) else None
            frame_arr = _overlay_box(frame_arr, gt_b, color=(0, 200, 0), thickness=2)
            frame_arr = _overlay_box(frame_arr, pred_b, color=(255, 50, 50), thickness=2)
            iou_val = iou_per_frame[t] if t < len(iou_per_frame) else 0.0
            ax.imshow(frame_arr)
            ax.set_title(f"t={t * sample_rate} IoU={iou_val:.2f}", fontsize=7)
        ax.axis("off")

    exp_text = result.get("expression", "")[:60]
    mean_iou = result.get("metrics", {}).get("mean_iou", 0.0)
    mode = result.get("failure", {}).get("primary_failure", "UNKNOWN")
    color = FAILURE_COLORS.get(mode, "#000000")
    fig.suptitle(
        f"{seq} | exp={exp} | mean IoU={mean_iou:.3f} | {mode}\n\"{exp_text}\"",
        fontsize=10, color=color,
    )
    fig.tight_layout()

    fname = f"{seq}__exp{exp}__iou{mean_iou:.3f}__{mode}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_vot_aggregate_summary(
    agg_metrics: dict,
    save_path: str,
    model_name: str = "Qwen3-VL-8B",
):
    """
    2-panel summary for VOT: primary metrics bar + temporal coherence bar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"VOT Benchmark Summary — {model_name}", fontsize=14)

    # Panel 1: Primary metrics
    ax = axes[0]
    metric_keys = ["mean_iou", "success_rate_50", "success_rate_75", "precision_20"]
    metric_labels = ["Mean IoU", "Success@50", "Success@75", "Precision@20"]
    values = [agg_metrics.get(k, 0) for k in metric_keys]
    bars = ax.bar(metric_labels, values,
                  color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Primary VOT Metrics")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Temporal coherence metrics
    ax = axes[1]
    tc_keys = ["iou_decay", "iou_variance", "iou_first", "iou_last"]
    tc_labels = ["IoU-Decay\n(neg=bad)", "IoU-Variance\n(high=bad)",
                 "IoU First Frame", "IoU Last Frame"]
    tc_values = [agg_metrics.get(k, 0) for k in tc_keys]
    tc_colors = ["#e74c3c" if v < 0 else "#3498db" for v in tc_values]
    bars2 = ax.bar(tc_labels, tc_values, color=tc_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Temporal Coherence Metrics")
    for bar, val in zip(bars2, tc_values):
        ypos = val + 0.01 if val >= 0 else val - 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_failure_case(
    result: dict,
    save_dir: str,
    tam_result: Optional[dict] = None,
    max_frames: int = 8,
):
    """
    Save a multi-panel figure for a single failure case:
    top row: GT overlay, bottom row: TAM heatmap overlay (if available).
    """
    os.makedirs(save_dir, exist_ok=True)
    seq = result.get("seq_name", "seq")
    exp = result.get("exp_id", "0")
    mode = result.get("failure", {}).get("primary_failure", "UNKNOWN")

    frames = result.get("frames_pil", [])
    gt_masks = result.get("gt_masks", [])
    pred_masks = result.get("pred_masks", [])
    T = len(frames)
    if T == 0:
        return

    n_show = min(max_frames, T)
    sample_idx = np.linspace(0, T - 1, n_show, dtype=int)

    n_rows = 2 if tam_result else 1
    fig, axes = plt.subplots(n_rows, n_show, figsize=(n_show * 3, n_rows * 3 + 1))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for col, t in enumerate(sample_idx):
        frame_arr = np.array(frames[t])
        # Top row: GT + pred overlay
        ax = axes[0, col]
        disp = frame_arr.copy()
        if t < len(gt_masks) and gt_masks[t] is not None:
            disp = _overlay_mask(disp, gt_masks[t], color=(0, 255, 0))
        if t < len(pred_masks) and pred_masks[t] is not None:
            disp = _overlay_mask(disp, pred_masks[t], color=(255, 50, 50), alpha=0.4)
        j_val = result["metrics"]["J_per_frame"][t] if t < len(result["metrics"]["J_per_frame"]) else 0
        ax.imshow(disp)
        ax.set_title(f"t={t} J={j_val:.2f}", fontsize=7)
        ax.axis("off")

        # Bottom row: TAM overlay (mean map for this frame)
        if tam_result and n_rows == 2:
            ax2 = axes[1, col]
            mean_map = None
            active = [i for i, tok in enumerate(tam_result["gen_tokens"])
                      if "<|" not in tok and tam_result["tam_maps"][i] is not None]
            if active:
                maps = [tam_result["tam_maps"][i] for i in active
                        if tam_result["tam_maps"][i].shape[0] > t]
                if maps:
                    mean_map = np.mean([m[t] for m in maps], axis=0)
            if mean_map is not None:
                tam_disp = _overlay_tam(frame_arr, mean_map)
                ax2.imshow(tam_disp)
            else:
                ax2.imshow(frame_arr)
            ax2.set_title("TAM", fontsize=7)
            ax2.axis("off")

    exp_text = result.get("expression", "")[:60]
    color = FAILURE_COLORS.get(mode, "#000000")
    fig.suptitle(
        f"{seq} | exp={exp} | {mode}\n\"{exp_text}\"",
        fontsize=10, color=color
    )
    fig.tight_layout()

    fname = f"{seq}__exp{exp}__{mode}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=100, bbox_inches="tight")
    plt.close(fig)
