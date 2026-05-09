"""
visualizer.py
-------------
Four figures for the POS dominance experiment:

  temporal_profile.png   – stacked bar: oracle category fraction per normalised
                           frame bin (the primary result figure)
  category_summary.png   – two-panel: overall win rate + mean GT mass per category
  dominance_heatmap.png  – grid: sequences × bins, coloured by dominant category
  category_race.png      – line plot: mean GT mass per category over time
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from experiment import CATEGORY_ORDER, CATEGORY_COLORS, N_BINS


def _label(cat: str) -> str:
    """Short display name for a category."""
    return cat.replace("label_", "").replace("_pos", "/other")


# ── temporal profile ──────────────────────────────────────────────────────────

def save_temporal_profile(summary: dict, save_dir: str) -> str:
    """
    Stacked bar chart showing, for each normalised frame bin, what fraction
    of oracle winners belong to each POS category.
    This is the primary plot for the experiment.
    """
    bins = summary["temporal_bins"]
    cats_present = [c for c in CATEGORY_ORDER
                    if any(b["dominance"].get(c, 0) > 0 for b in bins)]

    x        = np.arange(N_BINS)
    bottoms  = np.zeros(N_BINS)
    fig, ax  = plt.subplots(figsize=(13, 5))

    for cat in cats_present:
        vals = np.array([b["dominance"].get(cat, 0.0) for b in bins])
        ax.bar(x, vals, 0.8, bottom=bottoms,
               color=CATEGORY_COLORS[cat], label=_label(cat), alpha=0.92)
        bottoms += vals

    # annotate frame counts along x-axis
    for b, bin_info in enumerate(bins):
        n = bin_info["n_frames"]
        if n:
            ax.text(b, -0.06, f"n={n}", ha="center", va="top", fontsize=6,
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([b["range"] for b in bins], rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Normalised frame position within sequence")
    ax.set_ylabel("Fraction of oracle winners")
    ax.set_ylim(0, 1.05)
    ax.set_title("Which token category has the highest GT-box activation at each point in the sequence?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="Category")

    plt.tight_layout()
    out = Path(save_dir) / "temporal_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  temporal_profile  → {out}")
    return str(out)


# ── overall category summary ──────────────────────────────────────────────────

def save_category_summary(summary: dict, save_dir: str) -> str:
    """
    Two-panel horizontal bar chart:
      Left  — overall fraction of oracle wins per category
      Right — mean GT mass (when winning) per category
    """
    od    = summary["overall_dominance"]
    cats  = [c for c in CATEGORY_ORDER if c in od]
    fracs  = [od[c]["fraction"]     for c in cats]
    masses = [od[c]["mean_gt_mass"] for c in cats]
    colors = [CATEGORY_COLORS[c]    for c in cats]
    labels = [_label(c)             for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(cats) * 0.55 + 2)))
    y = np.arange(len(cats))

    ax1.barh(y, fracs, color=colors, edgecolor="black", linewidth=0.4)
    for i, f in enumerate(fracs):
        ax1.text(f + 0.003, i, f"{f:.1%}", va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Fraction of oracle wins (all frames)")
    ax1.set_title("Overall dominance rate")
    ax1.set_xlim(0, max(fracs) * 1.25 if fracs else 1)

    ax2.barh(y, masses, color=colors, edgecolor="black", linewidth=0.4)
    for i, m in enumerate(masses):
        ax2.text(m + 0.005, i, f"{m:.3f}", va="center", fontsize=8)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean GT mass when winning")
    ax2.set_title("GT-box attention strength per category")
    ax2.set_xlim(0, 1)

    n_seq = summary.get("n_sequences", "?")
    n_fr  = summary.get("n_frames", "?")
    fig.suptitle(f"POS Dominance Summary  ({n_seq} sequences, {n_fr} oracle frames)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = Path(save_dir) / "category_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  category_summary  → {out}")
    return str(out)


# ── per-sequence dominance heatmap ────────────────────────────────────────────

def save_dominance_heatmap(results: List[dict], save_dir: str) -> str:
    """
    Grid where each cell is the dominant oracle category for that
    (sequence, normalised-bin) pair.  Rows sorted by primary_category,
    then sequence name, to cluster similar patterns visually.
    """
    valid = [r for r in results
             if "error" not in r and r.get("oracle_winners")]
    valid.sort(key=lambda r: (r.get("primary_category", "z"), r["seq_name"]))

    cat_to_int = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    n_seqs     = len(valid)

    # dominant category per (sequence, bin)
    grid = np.full((n_seqs, N_BINS), np.nan)
    for row_i, r in enumerate(valid):
        bin_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for w in r["oracle_winners"]:
            b = w.get("bin_idx", 0)
            bin_counts[b][w["category"]] += 1
        for b, counts in bin_counts.items():
            dominant = max(counts, key=counts.__getitem__)
            grid[row_i, b] = cat_to_int.get(dominant, len(CATEGORY_ORDER) - 1)

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap   = ListedColormap([CATEGORY_COLORS[c] for c in CATEGORY_ORDER])
    bounds = np.arange(-0.5, len(CATEGORY_ORDER) + 0.5)
    norm   = BoundaryNorm(bounds, cmap.N)

    fig_h = max(6, n_seqs * 0.17 + 2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels([f"{b*10}%" for b in range(N_BINS)], fontsize=7)
    ax.set_yticks(range(n_seqs))
    ax.set_yticklabels([r["seq_name"] for r in valid], fontsize=5)
    ax.set_xlabel("Normalised frame position")
    ax.set_title("Dominant oracle category per sequence × frame bin", fontsize=10)

    # legend — only categories that appear in the grid
    cats_in_grid = {
        CATEGORY_ORDER[int(v)]
        for row in grid for v in row
        if not np.isnan(v)
    }
    legend_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor="black",
                       linewidth=0.5, label=_label(c))
        for c in CATEGORY_ORDER if c in cats_in_grid
    ]
    ax.legend(handles=legend_patches, fontsize=6,
              bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    out = Path(save_dir) / "dominance_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  dominance_heatmap → {out}")
    return str(out)


# ── category race (mean mass over time) ───────────────────────────────────────

def save_category_race(results: List[dict], save_dir: str) -> str:
    """
    Line plot: for each category, the mean GT mass (when it is the oracle
    winner) across normalised frame bins.  Shows whether some categories
    are not only more frequent winners but also stronger ones at specific
    points in the sequence.
    """
    valid = [r for r in results if "error" not in r]

    bin_masses: Dict[str, List[List[float]]] = {
        cat: [[] for _ in range(N_BINS)] for cat in CATEGORY_ORDER
    }
    for r in valid:
        for w in r.get("oracle_winners", []):
            cat = w["category"]
            b   = w.get("bin_idx", 0)
            if cat in bin_masses and 0 <= b < N_BINS:
                bin_masses[cat][b].append(w["gt_mass"])

    cats_with_data = [
        cat for cat in CATEGORY_ORDER
        if any(bin_masses[cat][b] for b in range(N_BINS))
    ]

    x   = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(13, 5))

    for cat in cats_with_data:
        means = [
            float(np.mean(bin_masses[cat][b])) if bin_masses[cat][b] else np.nan
            for b in range(N_BINS)
        ]
        counts = [len(bin_masses[cat][b]) for b in range(N_BINS)]
        # marker size proportional to win count (capped for readability)
        ms = [min(12, max(3, c ** 0.5)) for c in counts]
        ax.plot(x, means, marker="o", label=_label(cat),
                color=CATEGORY_COLORS[cat], linewidth=2)
        for xi, (m, s) in enumerate(zip(means, ms)):
            if not np.isnan(m):
                ax.scatter(xi, m, s=s ** 2, color=CATEGORY_COLORS[cat],
                           zorder=5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b*10}%" for b in range(N_BINS)], fontsize=8)
    ax.set_xlabel("Normalised frame position")
    ax.set_ylabel("Mean GT mass when oracle winner")
    ax.set_ylim(0, 1)
    ax.set_title("GT-box activation strength per category over sequence time\n"
                 "(marker size ∝ number of oracle wins in that bin)",
                 fontsize=9)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="Category")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(save_dir) / "category_race.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  category_race     → {out}")
    return str(out)


# ── pairwise category matrix ──────────────────────────────────────────────────

def save_pairwise_matrix(pairwise: dict, top2: dict, save_dir: str) -> str:
    """
    Two-panel figure:

    Left  — Symmetric K×K combined-score matrix.
             Cell (i,j) = mean GT-mass when the best token from cat_i and
             the best token from cat_j are averaged (≈ (mass_i + mass_j)/2).
             Diagonal = single-category mean best mass.
             Shows which category COMBINATIONS are richest in GT-box signal.

    Right — Directed K×K top-2 co-occurrence matrix.
             Cell (i,j) = fraction of frames where rank-1 category = i
             AND rank-2 category = j.
             Shows which pairs the model's oracle typically selects together.
    """
    # ── determine which categories have data ──────────────────────────────────
    cats_in_pw  = set()
    for v in pairwise.values():
        cats_in_pw.add(v["cat_A"]); cats_in_pw.add(v["cat_B"])
    cats_in_top2 = set()
    for key in top2.get("counts", {}):
        a, b = key.split("|")
        cats_in_top2.add(a); cats_in_top2.add(b)

    cats = [c for c in CATEGORY_ORDER if c in (cats_in_pw | cats_in_top2)]
    K    = len(cats)
    if K == 0:
        print("  [WARN] no pairwise data to plot")
        return ""
    idx  = {c: i for i, c in enumerate(cats)}
    labs = [_label(c) for c in cats]

    # ── build matrices ────────────────────────────────────────────────────────
    score_mat = np.full((K, K), np.nan)
    count_mat = np.zeros((K, K), dtype=int)

    for key, v in pairwise.items():
        a, b = v["cat_A"], v["cat_B"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        score_mat[i, j] = score_mat[j, i] = v["mean_combined"]
        count_mat[i, j] = count_mat[j, i] = v["n_frames"]

    top2_mat  = np.zeros((K, K))
    total_t2  = top2.get("total_frames", 1) or 1
    for key, cnt in top2.get("counts", {}).items():
        a, b = key.split("|")
        if a in idx and b in idx:
            top2_mat[idx[a], idx[b]] = cnt / total_t2

    # ── figure ────────────────────────────────────────────────────────────────
    cell_size = max(1.0, 8.0 / K)
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(K * cell_size * 2 + 4, K * cell_size + 2))

    # ── Panel 1: combined score ───────────────────────────────────────────────
    vmin = float(np.nanmin(score_mat)) if not np.all(np.isnan(score_mat)) else 0
    vmax = float(np.nanmax(score_mat)) if not np.all(np.isnan(score_mat)) else 1
    im1  = ax1.imshow(score_mat, vmin=vmin, vmax=vmax, cmap="YlOrRd",
                      interpolation="nearest")
    for i in range(K):
        for j in range(K):
            if np.isnan(score_mat[i, j]):
                continue
            txt = f"{score_mat[i,j]:.3f}"
            if count_mat[i, j] > 0:
                txt += f"\n(n={count_mat[i,j]})"
            ax1.text(j, i, txt, ha="center", va="center",
                     fontsize=max(4, 7 - K // 3),
                     color="black" if score_mat[i, j] < (vmin + vmax) * 0.65 else "white")
    # highlight diagonal
    for k in range(K):
        ax1.add_patch(mpatches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="#333", linewidth=2))
    ax1.set_xticks(range(K)); ax1.set_xticklabels(labs, rotation=45, ha="right",
                                                    fontsize=max(6, 9 - K // 3))
    ax1.set_yticks(range(K)); ax1.set_yticklabels(labs, fontsize=max(6, 9 - K // 3))
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Mean GT mass")
    ax1.set_title("Combined score: (best_A + best_B) / 2\n"
                  "(diagonal = single-category score)", fontsize=9)

    # ── Panel 2: top-2 co-occurrence ─────────────────────────────────────────
    im2 = ax2.imshow(top2_mat, vmin=0, cmap="Blues", interpolation="nearest")
    for i in range(K):
        for j in range(K):
            if top2_mat[i, j] > 0:
                ax2.text(j, i, f"{top2_mat[i,j]:.2%}",
                         ha="center", va="center",
                         fontsize=max(4, 7 - K // 3),
                         color="black" if top2_mat[i, j] < top2_mat.max() * 0.6 else "white")
    ax2.set_xticks(range(K)); ax2.set_xticklabels(labs, rotation=45, ha="right",
                                                    fontsize=max(6, 9 - K // 3))
    ax2.set_yticks(range(K)); ax2.set_yticklabels(labs, fontsize=max(6, 9 - K // 3))
    ax2.set_xlabel("Rank-2 category")
    ax2.set_ylabel("Rank-1 category")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Fraction of frames")
    ax2.set_title("Top-2 oracle co-occurrence\n"
                  "(row = rank 1, col = rank 2)", fontsize=9)

    fig.suptitle("Category pair analysis — which combinations dominate?",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path(save_dir) / "pairwise_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  pairwise_matrix   → {out}")
    return str(out)


# ── adjacency matrix (directional) ────────────────────────────────────────────

def save_adjacency_matrix(adj_pairwise: dict, save_dir: str) -> str:
    """
    Asymmetric K×K heatmap of adjacency-based combined scores.

    Cell (i, j) = mean (mass_a + mass_b)/2 over all (frame, position) where
                  the token at position p has category cat_i AND the token at
                  position p+1 has category cat_j.

    Unlike save_pairwise_matrix this is **directional**: row=preceding,
    col=following. Rows / columns sum independently of the symmetric pairwise
    score.
    """
    cats_present: set = set()
    for v in adj_pairwise.values():
        cats_present.add(v["cat_A"])
        cats_present.add(v["cat_B"])
    cats = [c for c in CATEGORY_ORDER if c in cats_present]
    K = len(cats)
    if K == 0:
        print("  [WARN] no adjacency data to plot")
        return ""
    idx  = {c: i for i, c in enumerate(cats)}
    labs = [_label(c) for c in cats]

    score_mat = np.full((K, K), np.nan)
    count_mat = np.zeros((K, K), dtype=int)
    for key, v in adj_pairwise.items():
        a, b = v["cat_A"], v["cat_B"]
        if a in idx and b in idx:
            score_mat[idx[a], idx[b]] = v["mean_combined"]
            count_mat[idx[a], idx[b]] = v["n_pairs"]

    cell_size = max(1.0, 9.0 / K)
    fig, ax = plt.subplots(figsize=(K * cell_size + 2, K * cell_size + 1.5))

    vmin = float(np.nanmin(score_mat)) if not np.all(np.isnan(score_mat)) else 0
    vmax = float(np.nanmax(score_mat)) if not np.all(np.isnan(score_mat)) else 1
    im = ax.imshow(score_mat, vmin=vmin, vmax=vmax, cmap="YlOrRd",
                   interpolation="nearest")
    for i in range(K):
        for j in range(K):
            if np.isnan(score_mat[i, j]):
                continue
            txt = f"{score_mat[i, j]:.3f}"
            if count_mat[i, j] > 0:
                txt += f"\n(n={count_mat[i, j]})"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=max(4, 7 - K // 3),
                    color="black" if score_mat[i, j] < (vmin + vmax) * 0.65 else "white")

    ax.set_xticks(range(K)); ax.set_xticklabels(labs, rotation=45, ha="right",
                                                  fontsize=max(6, 9 - K // 3))
    ax.set_yticks(range(K)); ax.set_yticklabels(labs, fontsize=max(6, 9 - K // 3))
    ax.set_xlabel("Following token category (position p+1)")
    ax.set_ylabel("Preceding token category (position p)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean combined GT mass")
    ax.set_title(
        "Adjacency-based combined score: (mass_p + mass_{p+1}) / 2\n"
        "Row = category at p,  Column = category at p+1  (directional)",
        fontsize=10,
    )

    plt.tight_layout()
    out = Path(save_dir) / "adjacency_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  adjacency_matrix  → {out}")
    return str(out)


# ── unconditional per-category position profile ──────────────────────────────

def save_category_position_profile(profile: dict, save_dir: str) -> str:
    """
    Line plot of mean best-token GT mass per category across normalised frame
    position bins. Unlike category_race.png (which only counts wins) this
    includes EVERY frame the category appeared in — so each category curve
    shows how its grounding strength varies over time independent of whether
    it dominated.

    profile : dict from POSDominanceExperiment.compute_category_position_profile
    """
    cats_with_data = [
        cat for cat in CATEGORY_ORDER
        if cat in profile and any(m is not None for m in profile[cat]["bin_means"])
    ]
    if not cats_with_data:
        print("  [WARN] no per-category position data to plot")
        return ""

    x = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(13, 5))

    for cat in cats_with_data:
        means  = profile[cat]["bin_means"]
        counts = profile[cat]["bin_counts"]
        ys = [m if m is not None else np.nan for m in means]
        ax.plot(x, ys, marker="o", label=_label(cat),
                color=CATEGORY_COLORS[cat], linewidth=2)
        # marker size encodes per-bin sample count
        for xi, (m, c) in enumerate(zip(ys, counts)):
            if not np.isnan(m):
                size = min(12, max(3, c ** 0.5)) ** 2
                ax.scatter(xi, m, s=size, color=CATEGORY_COLORS[cat],
                           zorder=5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b*10}%" for b in range(N_BINS)], fontsize=8)
    ax.set_xlabel("Normalised frame position")
    ax.set_ylabel("Mean GT mass (best token of category, every frame it appears in)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Per-category GT-mass over time — UNCONDITIONAL\n"
        "(every frame the category produced a token, not just when it won)",
        fontsize=10,
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="Category")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(save_dir) / "category_position_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  category_position_profile → {out}")
    return str(out)
