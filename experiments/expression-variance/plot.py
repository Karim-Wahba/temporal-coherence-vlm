"""
plot.py
-------
Four figures for the expression-variance experiment.

Inputs : grouped_stats.json (from analyze.py)
Outputs: four PNGs in <out_dir>/figures/

Figure 1 — Range chart per (seq, obj). Three stacked subplots (IoU, mass-in-GT,
           mass-in-pred). Rows sorted by IoU range descending. Each row: a
           horizontal line min → max, a big dot at the mean, ticks at every
           expression value. Variance annotated on the right.

Figure 2 — Best-vs-worst scatter. One panel per metric. X = worst-expression
           value, Y = best-expression value, all points are above y=x by
           construction. Diagonal drawn for reference; median gap annotated.

Figure 3 — Coupling between attention and IoU within a group. For each
           (seq, obj), Spearman correlation between the group's expression-
           level mass-in-GT and IoU. Histogram with mean / fraction-positive
           annotated; we plot mass-in-pred vs IoU as a second panel.

Figure 4 — Strip plot per (seq, obj). One row per group, one dot per
           expression. Dot X position = IoU, dot color = mass-in-GT,
           dot size = mass-in-pred. Lets you see all individual expression
           outcomes in one frame and spot patterns.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import spearmanr

METRICS = [
    ("iou",          "Mean IoU",                "tab:blue"),
    ("mass_in_gt",   "Attention mass in GT",    "tab:green"),
    ("mass_in_pred", "Attention mass in pred",  "tab:orange"),
]


def _load(grouped_path: Path):
    grouped = json.load(open(grouped_path))
    rows = []
    for key, g in grouped.items():
        if not g["iou"]:
            continue
        rows.append({
            "key":          key,
            "seq_name":     g["seq_name"],
            "obj_id":       g["obj_id"],
            "label":        f"{g['seq_name']} (obj {g['obj_id']})",
            "expressions":  g["expressions"],
            "iou":          g["iou"],
            "mass_in_gt":   g["mass_in_gt"],
            "mass_in_pred": g["mass_in_pred"],
        })
    return rows


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def figure_1_range(rows, save_path: Path):
    rows_sorted = sorted(rows, key=lambda r: r["iou"]["range"], reverse=True)
    n = len(rows_sorted)
    labels = [r["label"] for r in rows_sorted]
    y = np.arange(n)[::-1]  # top-to-bottom = highest range first

    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.22 * n)), sharey=True)

    for ax, (key, title, color) in zip(axes, METRICS):
        for yi, r in zip(y, rows_sorted):
            s = r[key]
            ax.hlines(yi, s["min"], s["max"], color=color, alpha=0.5, linewidth=2)
            ax.scatter(s["values"], [yi] * len(s["values"]),
                       color=color, alpha=0.6, s=22, edgecolors="none", zorder=3)
            ax.scatter([s["mean"]], [yi], color="black", s=28, zorder=4,
                       marker="D", label="_nolegend_")
            ax.text(1.005, yi, f"σ²={s['variance']:.3f}", fontsize=6,
                    color="dimgray", va="center",
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.set_yticks(y)
        ax.grid(True, axis="x", alpha=0.25)

    axes[0].set_yticklabels(labels, fontsize=7)
    legend_elements = [
        mlines.Line2D([], [], color="gray", linewidth=2, label="min → max"),
        mlines.Line2D([], [], color="black", marker="D", linestyle="None",
                      markersize=6, label="mean"),
        mlines.Line2D([], [], color="tab:blue", marker="o", linestyle="None",
                      markersize=5, alpha=0.6, label="per-expression value"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=7,
                   framealpha=0.9)
    fig.suptitle("Per-expression range within each (sequence, object) group  "
                 "— sorted by IoU range",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def figure_2_best_vs_worst(rows, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, title, color) in zip(axes, METRICS):
        worst = np.array([r[key]["min"] for r in rows])
        best  = np.array([r[key]["max"] for r in rows])
        gaps  = best - worst

        ax.scatter(worst, best, s=42, color=color, alpha=0.7, edgecolors="white",
                   linewidth=0.5)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="y = x")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Worst expression — {title}")
        ax.set_ylabel(f"Best expression — {title}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        median_gap = float(np.median(gaps))
        mean_gap   = float(np.mean(gaps))
        ax.text(0.03, 0.97,
                f"median Δ = {median_gap:.3f}\nmean Δ = {mean_gap:.3f}\nn = {len(rows)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                          edgecolor="lightgray"))
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
    fig.suptitle("Best vs. worst expression per (sequence, object) — "
                 "vertical distance above y=x is the upside from picking a better expression",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def _spearman_safe(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    rho, _ = spearmanr(a, b)
    return float(rho)


def figure_3_coupling(rows, save_path: Path):
    rho_gt   = []
    rho_pred = []
    for r in rows:
        if r["iou"]["n"] < 3:
            continue
        rho_gt.append(_spearman_safe(r["mass_in_gt"]["values"], r["iou"]["values"]))
        rho_pred.append(_spearman_safe(r["mass_in_pred"]["values"], r["iou"]["values"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, vals, title, color in [
        (axes[0], rho_gt,   "Spearman(mass-in-GT, IoU) per group",   "tab:green"),
        (axes[1], rho_pred, "Spearman(mass-in-pred, IoU) per group", "tab:orange"),
    ]:
        v = np.array([x for x in vals if not np.isnan(x)])
        ax.hist(v, bins=np.linspace(-1, 1, 21), color=color, alpha=0.75,
                edgecolor="white")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        if v.size:
            mean_rho = float(v.mean())
            frac_pos = float((v > 0).mean())
            ax.axvline(mean_rho, color="black", linewidth=1.5)
            ax.text(0.02, 0.95,
                    f"mean ρ = {mean_rho:+.3f}\nfraction ρ > 0: {frac_pos:.0%}\n"
                    f"n groups = {v.size}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.85, edgecolor="lightgray"))
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ within group")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("# (seq, obj) groups")
    fig.suptitle("Within-group correlation between attention mass and IoU "
                 "— a positive distribution means attention can rank expressions",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def figure_4_strip(rows, save_path: Path):
    rows_sorted = sorted(rows, key=lambda r: r["iou"]["mean"], reverse=True)
    n = len(rows_sorted)
    y = np.arange(n)[::-1]
    labels = [r["label"] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.24 * n)))

    sizes_norm = []
    for r in rows_sorted:
        for v in r["mass_in_pred"]["values"]:
            sizes_norm.append(v)
    s_min = min(sizes_norm) if sizes_norm else 0.0
    s_max = max(sizes_norm) if sizes_norm else 1.0
    s_range = max(1e-6, s_max - s_min)

    for yi, r in zip(y, rows_sorted):
        ious   = r["iou"]["values"]
        m_gt   = r["mass_in_gt"]["values"]
        m_pred = r["mass_in_pred"]["values"]
        sizes  = [40 + 200 * (mp - s_min) / s_range for mp in m_pred]
        sc = ax.scatter(ious, [yi] * len(ious), c=m_gt,
                        cmap="viridis", vmin=0.0, vmax=1.0,
                        s=sizes, alpha=0.85, edgecolors="black", linewidths=0.4)
        ax.hlines(yi, min(ious), max(ious), color="lightgray", alpha=0.7,
                  linewidth=1, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean IoU (one dot per expression)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("All expressions per (seq, obj). "
                 "Color = mass-in-GT, size = mass-in-pred.")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mass-in-GT")

    size_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=np.sqrt(40 + 200 * 0.0), label="low mass-in-pred"),
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=np.sqrt(40 + 200 * 0.5), label="mid"),
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=np.sqrt(40 + 200 * 1.0), label="high mass-in-pred"),
    ]
    ax.legend(handles=size_handles, loc="lower right", fontsize=7,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", required=True,
                    help="Path to grouped_stats.json")
    ap.add_argument("--out_dir", default=None,
                    help="Directory for figures/ (default = grouped_stats parent)")
    args = ap.parse_args()

    grouped_path = Path(args.grouped)
    out_dir = Path(args.out_dir) if args.out_dir else grouped_path.parent
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(grouped_path)
    if not rows:
        print("[plot] no usable groups found")
        return

    figure_1_range          (rows, fig_dir / "fig1_range_per_group.png")
    figure_2_best_vs_worst  (rows, fig_dir / "fig2_best_vs_worst.png")
    figure_3_coupling       (rows, fig_dir / "fig3_attention_iou_coupling.png")
    figure_4_strip          (rows, fig_dir / "fig4_strip_per_group.png")

    print(f"Wrote 4 figures to {fig_dir}")


if __name__ == "__main__":
    main()
