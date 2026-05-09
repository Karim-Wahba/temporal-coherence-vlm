"""
plot_extras.py
--------------
Supplementary figures + CSV tables that quantify *where* the
expression-variance headroom lives.

Inputs : grouped_stats.json (from analyze.py)
Outputs:
  CSVs in <out_dir>/:
    iou_bucket_table.csv          — avg range / oracle-gain by mean-IoU bucket
    concentration_table.csv       — top-K groups → cumulative share of total gap
    range_distribution_table.csv  — histogram of within-group ranges
    top_range_groups.csv          — top highest-range groups, full per-expression rows

  Figures in <out_dir>/figures/:
    fig5_concentration_lorenz.png — cumulative gap vs. cumulative groups
    fig6_bucket_bars.png          — avg range + oracle gain by mean-IoU bucket
    fig7_meaniou_vs_range.png     — scatter, with regression + r annotation
    fig8_failure_examples.png     — top-K highest-range groups, per-expression bars
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import _load


IOU_BUCKETS = [(0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]
RANGE_BUCKETS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
TOP_K_CONCENTRATION = (5, 10, 15, 20, 30)
TOP_K_FAILURES = 8


# ── CSV writers ──────────────────────────────────────────────────────────────

def write_iou_bucket_csv(rows, path: Path):
    means  = np.array([r["iou"]["mean"]  for r in rows])
    maxs   = np.array([r["iou"]["max"]   for r in rows])
    ranges = np.array([r["iou"]["range"] for r in rows])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_mean_iou_bucket", "n_groups",
                    "avg_group_mean_iou",
                    "avg_range_best_minus_worst",
                    "avg_oracle_gain_best_minus_mean"])
        for lo, hi in IOU_BUCKETS:
            mask = (means >= lo) & (means < hi)
            n = int(mask.sum())
            if n == 0:
                w.writerow([f"[{lo:.2f}, {hi:.2f})", 0, "", "", ""])
                continue
            w.writerow([
                f"[{lo:.2f}, {hi:.2f})",
                n,
                f"{means[mask].mean():.3f}",
                f"{ranges[mask].mean():.3f}",
                f"{(maxs[mask] - means[mask]).mean():.3f}",
            ])


def write_concentration_csv(rows, path: Path):
    ranges = np.array([r["iou"]["range"] for r in rows])
    total = float(ranges.sum())
    order = np.argsort(ranges)[::-1]
    n_groups = len(rows)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["top_k_groups", "fraction_of_groups", "fraction_of_total_gap"])
        for k in TOP_K_CONCENTRATION:
            if k > n_groups:
                continue
            frac_groups = k / n_groups
            frac_gap = float(ranges[order[:k]].sum() / total) if total > 0 else 0.0
            w.writerow([k, f"{frac_groups:.3f}", f"{frac_gap:.3f}"])


def write_range_distribution_csv(rows, path: Path):
    ranges = np.array([r["iou"]["range"] for r in rows])
    n_groups = len(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["range_bucket", "n_groups", "fraction_of_groups"])
        for lo, hi in RANGE_BUCKETS:
            mask = (ranges >= lo) & (ranges < hi)
            n = int(mask.sum())
            w.writerow([f"[{lo:.2f}, {hi:.2f})", n, f"{n / n_groups:.2f}"])


def write_top_range_groups_csv(rows, path: Path, k: int = TOP_K_FAILURES):
    ranges = np.array([r["iou"]["range"] for r in rows])
    order = np.argsort(ranges)[::-1][:k]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "expression", "iou", "group_range", "group_mean"])
        for i in order:
            r = rows[i]
            for exp, v in zip(r["expressions"], r["iou"]["values"]):
                w.writerow([r["key"], exp, f"{v:.3f}",
                            f"{r['iou']['range']:.3f}", f"{r['iou']['mean']:.3f}"])


# ── Figures ──────────────────────────────────────────────────────────────────

def figure_5_concentration_lorenz(rows, save_path: Path):
    ranges = np.array([r["iou"]["range"] for r in rows])
    total = ranges.sum()
    if total <= 0:
        return
    sorted_desc = np.sort(ranges)[::-1]
    cum_gap = np.cumsum(sorted_desc) / total
    cum_groups = np.arange(1, len(sorted_desc) + 1) / len(sorted_desc)

    cum_gap = np.concatenate([[0.0], cum_gap])
    cum_groups = np.concatenate([[0.0], cum_groups])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cum_groups, cum_gap, color="tab:blue", linewidth=2.2,
            label="actual concentration")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1,
            label="uniform spread (reference)")
    ax.fill_between(cum_groups, cum_groups, cum_gap,
                    where=(cum_gap >= cum_groups), color="tab:blue", alpha=0.10)

    n = len(ranges)
    annotate_ks = [k for k in (5, 10, 15) if k <= n]
    for k in annotate_ks:
        gx = k / n
        gy = cum_gap[k]
        ax.scatter([gx], [gy], s=40, color="tab:blue", zorder=4)
        ax.annotate(f"top {k} groups\n({gx*100:.0f}% of groups → {gy*100:.0f}% of gap)",
                    xy=(gx, gy), xytext=(gx + 0.05, gy - 0.10),
                    fontsize=8, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="lightgray", linewidth=0.8))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cumulative fraction of groups (sorted by IoU range, desc)")
    ax.set_ylabel("Cumulative fraction of total best-vs-worst gap")
    ax.set_title("Concentration of rephrasing headroom across groups")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_6_bucket_bars(rows, save_path: Path):
    means  = np.array([r["iou"]["mean"]  for r in rows])
    maxs   = np.array([r["iou"]["max"]   for r in rows])
    ranges = np.array([r["iou"]["range"] for r in rows])

    labels, ns, avg_range, avg_gain = [], [], [], []
    for lo, hi in IOU_BUCKETS:
        mask = (means >= lo) & (means < hi)
        if mask.sum() == 0:
            continue
        labels.append(f"[{lo:.2f}, {hi:.2f})\nn={int(mask.sum())}")
        ns.append(int(mask.sum()))
        avg_range.append(float(ranges[mask].mean()))
        avg_gain.append(float((maxs[mask] - means[mask]).mean()))

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - width / 2, avg_range, width, color="tab:blue",
                label="avg best − worst range")
    b2 = ax.bar(x + width / 2, avg_gain,  width, color="tab:orange",
                label="avg oracle gain (best − mean)")

    for bars, vals in [(b1, avg_range), (b2, avg_gain)]:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Group-mean IoU bucket")
    ax.set_ylabel("IoU points")
    ax.set_title("Headroom is concentrated in low-IoU groups\n"
                 "(model is robust where it works, brittle where it fails)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(avg_range), max(avg_gain)) * 1.30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_7_meaniou_vs_range(rows, save_path: Path):
    means  = np.array([r["iou"]["mean"]  for r in rows])
    ranges = np.array([r["iou"]["range"] for r in rows])
    keys   = [r["key"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(means, ranges, s=46, c=ranges, cmap="Reds",
                    edgecolors="black", linewidths=0.4, alpha=0.85)

    if len(means) >= 2 and np.std(means) > 0:
        slope, intercept = np.polyfit(means, ranges, 1)
        xs = np.linspace(means.min(), means.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="black",
                linewidth=1.4, linestyle="--", label="linear fit")
        r = float(np.corrcoef(means, ranges)[0, 1])
        ax.text(0.97, 0.97,
                f"Pearson r = {r:+.3f}\nn = {len(means)}",
                transform=ax.transAxes, va="top", ha="right", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.9, edgecolor="lightgray"))

    order = np.argsort(ranges)[::-1][:5]
    for i in order:
        ax.annotate(keys[i], xy=(means[i], ranges[i]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color="dimgray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.25, ranges.max() * 1.05))
    ax.set_xlabel("Group-mean IoU")
    ax.set_ylabel("Best − worst IoU within group (range)")
    ax.set_title("Where rephrasing matters: failing groups are the brittle ones")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("range (best − worst)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_8_failure_examples(rows, save_path: Path, k: int = TOP_K_FAILURES):
    ranges = np.array([r["iou"]["range"] for r in rows])
    order = np.argsort(ranges)[::-1][:k]

    ncols = 2
    nrows = (len(order) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13, 2.4 * nrows + 0.5),
                             squeeze=False)

    for ax_idx, i in enumerate(order):
        r = rows[i]
        ax = axes[ax_idx // ncols][ax_idx % ncols]

        pairs = sorted(zip(r["iou"]["values"], r["expressions"]),
                       key=lambda x: x[0])
        ious = [p[0] for p in pairs]
        exps = [p[1] for p in pairs]
        # truncate long expression strings
        exp_disp = [(e if len(e) <= 55 else e[:52] + "…") for e in exps]

        colors = ["tab:red" if v < 0.10 else
                  "tab:orange" if v < 0.30 else
                  "tab:green" for v in ious]

        y = np.arange(len(ious))
        ax.barh(y, ious, color=colors, edgecolor="black", linewidth=0.4)
        for yi, v in zip(y, ious):
            ax.text(min(v + 0.012, 0.985), yi, f"{v:.3f}",
                    va="center", ha="left", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"'{e}'" for e in exp_disp], fontsize=7.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("IoU")
        ax.set_title(f"{r['key']}    range={r['iou']['range']:.2f}  "
                     f"mean={r['iou']['mean']:.2f}",
                     fontsize=9)
        ax.grid(True, axis="x", alpha=0.25)

    # blank out unused cells
    for j in range(len(order), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Top high-variance groups: same target, very different IoU per expression",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI / orchestration ──────────────────────────────────────────────────────

def render_all(grouped_path: Path, out_dir: Path):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(grouped_path)
    if not rows:
        print("[plot_extras] no usable groups found")
        return

    write_iou_bucket_csv         (rows, out_dir / "iou_bucket_table.csv")
    write_concentration_csv      (rows, out_dir / "concentration_table.csv")
    write_range_distribution_csv (rows, out_dir / "range_distribution_table.csv")
    write_top_range_groups_csv   (rows, out_dir / "top_range_groups.csv")

    figure_5_concentration_lorenz (rows, fig_dir / "fig5_concentration_lorenz.png")
    figure_6_bucket_bars          (rows, fig_dir / "fig6_bucket_bars.png")
    figure_7_meaniou_vs_range     (rows, fig_dir / "fig7_meaniou_vs_range.png")
    figure_8_failure_examples     (rows, fig_dir / "fig8_failure_examples.png")

    print(f"[plot_extras] wrote 4 CSVs to {out_dir}")
    print(f"[plot_extras] wrote 4 figures to {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", required=True, help="Path to grouped_stats.json")
    ap.add_argument("--out_dir", default=None,
                    help="Default = parent of grouped_stats.json")
    args = ap.parse_args()

    grouped_path = Path(args.grouped)
    out_dir = Path(args.out_dir) if args.out_dir else grouped_path.parent
    render_all(grouped_path, out_dir)


if __name__ == "__main__":
    main()
