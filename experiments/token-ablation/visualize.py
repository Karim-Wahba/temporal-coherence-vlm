"""
visualize.py
------------
Plotting utilities for the token ablation study.

  plot_strategy_bar(results, save_path)
      Bar chart of mean GT mass per strategy with per-expression error bars.

  plot_heatmap_grid(frames, strategy_heatmaps, gt_boxes, save_path)
      Grid of attention heatmaps for a single representative frame,
      one column per strategy.

  plot_per_frame_curves(results, save_path)
      Per-frame GT mass curves for each strategy (one line per strategy).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── colour palette ────────────────────────────────────────────────────────────

_PALETTE = plt.get_cmap("tab20")


def _strategy_color(strategy_name: str, strategy_names: List[str]) -> tuple:
    idx = strategy_names.index(strategy_name)
    return _PALETTE(idx % 20)


# ── helpers ───────────────────────────────────────────────────────────────────

def _blend_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    hm_u8 = (heatmap.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    hm_resized = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return (0.5 * frame_rgb + 0.5 * hm_rgb).clip(0, 255).astype(np.uint8)


def _draw_box(ax, box, color="lime", lw=1.5):
    if box is None:
        return
    x1, y1, x2, y2 = box
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
    ))


# ── 1. bar chart ──────────────────────────────────────────────────────────────

def plot_strategy_bar(
    results: List[dict],
    strategy_names: List[str],
    save_path: str,
    title: str = "Token Selection Ablation",
):
    """
    Bar chart: mean GT-mass per strategy across expressions.
    Each bar is the mean over all (strategy, frame) pairs from all expressions.
    Error bars show std across per-expression means.
    """
    # per_strategy_per_expr[strategy] = [mean_mass_expr0, mean_mass_expr1, ...]
    per_strategy: Dict[str, List[float]] = {s: [] for s in strategy_names}

    for r in results:
        for sname in strategy_names:
            masses = r.get("strategy_masses", {}).get(sname, [])
            if masses:
                per_strategy[sname].append(float(np.mean(masses)))

    means = [np.mean(per_strategy[s]) if per_strategy[s] else 0.0 for s in strategy_names]
    stds  = [np.std(per_strategy[s])  if per_strategy[s] else 0.0 for s in strategy_names]

    x = np.arange(len(strategy_names))
    colors = [_strategy_color(s, strategy_names) for s in strategy_names]

    fig, ax = plt.subplots(figsize=(max(10, len(strategy_names) * 0.9), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 1.2})

    # annotate values on top
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=7, rotation=45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean GT Mass (attention inside GT bbox)")
    ax.set_title(title)
    ax.set_ylim(0, min(1.05, max(means) * 1.35 + 0.05))
    ax.axhline(means[strategy_names.index("all_tokens_mean")] if "all_tokens_mean" in strategy_names else 0,
               color="gray", linestyle=":", linewidth=1, label="all_tokens_mean baseline")
    ax.legend(fontsize=7)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  bar chart → {save_path}")


# ── 2. heatmap grid ───────────────────────────────────────────────────────────

def plot_heatmap_grid(
    frame_pil: Image.Image,
    gt_box: Optional[Tuple],
    strategy_heatmaps: Dict[str, Optional[np.ndarray]],  # strategy → (H_tam, W_tam)
    save_path: str,
    frame_label: str = "",
):
    """
    One row per strategy: frame image with GT box + attention heatmap overlay.
    All strategies shown for the same representative frame.
    """
    strategy_names = list(strategy_heatmaps.keys())
    n = len(strategy_names)
    if n == 0:
        return

    frame_rgb = np.array(frame_pil.convert("RGB"))
    H, W = frame_rgb.shape[:2]

    fig, axes = plt.subplots(n, 2, figsize=(6, 1.8 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(f"Attention heatmaps per strategy — {frame_label}", fontsize=9,
                 fontweight="bold", y=1.01)

    for row, sname in enumerate(strategy_names):
        hm = strategy_heatmaps[sname]

        # left: frame + GT box
        ax0 = axes[row, 0]
        ax0.imshow(frame_rgb)
        _draw_box(ax0, gt_box)
        ax0.axis("off")
        if row == 0:
            ax0.set_title("Frame + GT", fontsize=7)
        ax0.set_ylabel(sname, fontsize=7, rotation=0, ha="right", va="center", labelpad=50)

        # right: heatmap overlay
        ax1 = axes[row, 1]
        if hm is not None:
            blended = _blend_heatmap(frame_rgb, hm)
            ax1.imshow(blended)
            _draw_box(ax1, gt_box)
        else:
            ax1.imshow(frame_rgb)
            ax1.text(0.5, 0.5, "no heatmap", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=8, color="red")
        ax1.axis("off")
        if row == 0:
            ax1.set_title("TAM overlay", fontsize=7)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap grid → {save_path}")


# ── 3. per-frame curves ───────────────────────────────────────────────────────

def plot_per_frame_curves(
    results: List[dict],
    strategy_names: List[str],
    save_path: str,
    title: str = "Per-frame GT mass by strategy",
):
    """
    Per-frame GT-mass curve for each strategy, averaged across expressions.
    X = sampled frame index, Y = GT mass. One line per strategy.
    """
    # Accumulate per-frame masses per strategy
    # strategy → {sampled_t: [mass_from_expr0, mass_from_expr1, ...]}
    from collections import defaultdict
    per_strat_t: Dict[str, Dict[int, List[float]]] = {
        s: defaultdict(list) for s in strategy_names
    }

    for r in results:
        for sname in strategy_names:
            frame_masses = r.get("strategy_frame_masses", {}).get(sname, {})
            for t_str, mass in frame_masses.items():
                per_strat_t[sname][int(t_str)].append(mass)

    # Get union of all frame indices
    all_ts = sorted({
        t for s in strategy_names for t in per_strat_t[s]
    })
    if not all_ts:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, sname in enumerate(strategy_names):
        xs, ys, errs = [], [], []
        for t in all_ts:
            vals = per_strat_t[sname].get(t, [])
            if vals:
                xs.append(t)
                ys.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
        if xs:
            color = _strategy_color(sname, strategy_names)
            ax.plot(xs, ys, marker="o", markersize=3, label=sname, color=color)
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.1, color=color)

    ax.set_xlabel("Sampled frame index")
    ax.set_ylabel("GT mass")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  per-frame curves → {save_path}")


# ── 4. oracle token analysis ─────────────────────────────────────────────────

def plot_oracle_token_analysis(
    oracle_rows: List[dict],
    save_path: str,
    top_k: int = 20,
):
    """
    Two-panel figure from the output of analyze_oracle_tokens():

    Left  — frequency bar chart: how often each token_clean text wins the oracle
            (rank=1) across all frames and expressions.  Bars coloured by the
            mean GT mass achieved when that token wins.

    Right — per-frame heatmap of GT mass for the top-10 most-frequent oracle
            tokens.  Rows = token text, columns = sampled frame index.
            Colour = GT mass at that frame (white = token never won that frame).
    """
    winners = [r for r in oracle_rows if r["rank"] == 1]
    if not winners:
        print("  [WARN] no oracle winners to plot")
        return

    from collections import Counter, defaultdict

    # ── count wins and mean GT mass per token ─────────────────────────────
    win_count:   Counter = Counter()
    win_masses:  defaultdict = defaultdict(list)
    frame_mass:  defaultdict = defaultdict(dict)   # token_clean → {orig_t: gt_mass}

    for row in winners:
        tc = row["token_clean"] or repr(row["token"])
        win_count[tc] += 1
        win_masses[tc].append(row["gt_mass"])
        frame_mass[tc][row["orig_t"]] = row["gt_mass"]

    top_tokens = [tok for tok, _ in win_count.most_common(top_k)]
    counts     = [win_count[tok] for tok in top_tokens]
    mean_masses = [float(np.mean(win_masses[tok])) for tok in top_tokens]

    # ── layout ────────────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(14, max(5, top_k * 0.45)),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # Left: frequency bars coloured by mean GT mass
    norm = plt.Normalize(vmin=0, vmax=max(mean_masses) if mean_masses else 1)
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(norm(m)) for m in mean_masses]
    y = np.arange(len(top_tokens))
    bars = ax_left.barh(y, counts, color=colors, edgecolor="black", linewidth=0.4)
    for bar, m in zip(bars, mean_masses):
        ax_left.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{m:.3f}", va="center", fontsize=7,
        )
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([repr(t) for t in top_tokens], fontsize=8)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Win count (# frames oracle chose this token)")
    ax_left.set_title("Oracle token frequency\n(colour = mean GT mass when winning)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_left, label="mean GT mass", shrink=0.6)

    # Right: per-frame mass grid for top-10
    grid_tokens = top_tokens[:10]
    all_orig_ts = sorted({row["orig_t"] for row in winners})
    grid = np.full((len(grid_tokens), len(all_orig_ts)), np.nan)
    for row_i, tok in enumerate(grid_tokens):
        for col_i, t in enumerate(all_orig_ts):
            if t in frame_mass[tok]:
                grid[row_i, col_i] = frame_mass[tok][t]

    im = ax_right.imshow(
        grid, aspect="auto", vmin=0, vmax=1,
        cmap="YlOrRd", interpolation="nearest",
    )
    ax_right.set_xticks(range(len(all_orig_ts)))
    ax_right.set_xticklabels(all_orig_ts, rotation=90, fontsize=6)
    ax_right.set_yticks(range(len(grid_tokens)))
    ax_right.set_yticklabels([repr(t) for t in grid_tokens], fontsize=8)
    ax_right.set_xlabel("Original frame index")
    ax_right.set_title("GT mass per frame when oracle chose this token\n(blank = token didn't win that frame)")
    plt.colorbar(im, ax=ax_right, label="GT mass", shrink=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  oracle analysis → {save_path}")


# ── 5. token spotlight ────────────────────────────────────────────────────────

def plot_token_spotlight(
    token_scores: List[Tuple[str, float]],  # (token_text, mean_gt_mass)
    save_path: str,
    top_k: int = 20,
):
    """
    Horizontal bar chart of the top-k individual tokens by mean GT mass.
    Useful companion to identify what the oracle picks.
    """
    top = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]
    names = [t for t, _ in top]
    vals  = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.4)))
    y = np.arange(len(names))
    ax.barh(y, vals, color=plt.get_cmap("RdYlGn")(
        np.array(vals) / max(vals + [1e-8])
    ))
    ax.set_yticks(y)
    ax.set_yticklabels([repr(n) for n in names], fontsize=8)
    ax.set_xlabel("Mean GT mass")
    ax.set_title(f"Top-{top_k} tokens by mean GT mass — breakdance")
    ax.invert_yaxis()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  token spotlight → {save_path}")
