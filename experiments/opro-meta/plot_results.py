"""
plot_results.py
---------------
Visualisations for opro-meta optimisation results.

Reads `clip_results.json` produced by main.py and writes:

  fig1_optimization_curves.png  Best IoU per iteration, one line per clip
                                (the "optimisation loss" / convergence trace)
  fig2_miou_progression.png     Mean IoU across ALL clips at each iteration
                                (running-best per clip, averaged)
  fig3_top_improvements.png     Top-N clips by IoU gain, seed vs best
                                expressions with delta annotation
  fig4_seed_vs_best.png         Per-clip seed IoU vs best IoU scatter
                                (above diagonal = optimizer helped)
  fig5_delta_distribution.png   Histogram of per-clip IoU deltas (how often
                                the optimizer helps / hurts / no-ops)
  summary.csv                   Per-clip seed / best / delta + expressions

Usage
-----
  python plot_results.py --results results/<run>/clip_results.json
                         --out_dir results/<run>/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Palette ────────────────────────────────────────────────────────────────────

_C_SEED   = "#4C72B0"   # blue   – seed (original)
_C_BEST   = "#55A868"   # green  – best LLM
_C_GAIN   = "#2a7a2a"   # dark green – positive delta
_C_LOSS   = "#b22222"   # dark red   – negative delta
_C_FLAT   = "#888888"   # grey       – no change


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def _clip_label(c: dict) -> str:
    return f"{c['seq_name']}__obj{c['obj_id']}"


def _short(e: str, n: int = 40) -> str:
    return e if len(e) <= n else e[: n - 1] + "…"


# ── Per-clip running-best trace ────────────────────────────────────────────────

def _running_best_per_iter(c: dict) -> Dict[int, float]:
    """Iteration index -> best IoU seen so far (None excluded)."""
    by_iter: Dict[int, List[float]] = {}
    for h in c["history"]:
        if h.get("mean_iou") is None:
            continue
        by_iter.setdefault(h["iteration"], []).append(h["mean_iou"])
    running: Dict[int, float] = {}
    best = -float("inf")
    for it in sorted(by_iter):
        best = max(best, max(by_iter[it]))
        running[it] = best
    return running


def _seed_iou(c: dict) -> Optional[float]:
    for h in c["history"]:
        if h["source"] == "seed" and h.get("mean_iou") is not None:
            return h["mean_iou"]
    return None


# ── Fig 1: per-clip convergence curves ────────────────────────────────────────

def fig1_optimization_curves(data: List[dict], out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_max_it = 0
    for gi, c in enumerate(data):
        rb = _running_best_per_iter(c)
        if not rb:
            continue
        xs = sorted(rb.keys())
        ys = [rb[x] for x in xs]
        all_max_it = max(all_max_it, max(xs))
        color = plt.cm.tab10(gi % 10)
        ax.plot(xs, ys, marker="o", ms=5, lw=1.5, alpha=0.8,
                color=color, label=_clip_label(c) if len(data) <= 10 else None)

    # Bold average over clips
    aggregated: Dict[int, List[float]] = {}
    for c in data:
        rb = _running_best_per_iter(c)
        for k, v in rb.items():
            aggregated.setdefault(k, []).append(v)
    xs = sorted(aggregated)
    if xs:
        ys = [np.mean(aggregated[x]) for x in xs]
        ax.plot(xs, ys, marker="s", ms=8, lw=3.0, color="black",
                label=f"Mean over {len(data)} clips", zorder=10)

    ax.set_xlabel("Iteration  (0 = seed)")
    ax.set_ylabel("Best IoU so far")
    ax.set_title("Per-clip optimisation curves\n(running-best IoU per iteration)", fontsize=11)
    ax.set_xticks(range(all_max_it + 1))
    ax.set_ylim(0, 1.05)
    ax.grid(lw=0.4, alpha=0.5)
    if len(data) <= 10:
        ax.legend(fontsize=7, loc="lower right", ncol=2)
    else:
        ax.legend(fontsize=8, loc="lower right")
    _save(fig, out)


# ── Fig 2: mIoU across clips over time ────────────────────────────────────────

def fig2_miou_progression(data: List[dict], out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    aggregated: Dict[int, List[float]] = {}
    for c in data:
        rb = _running_best_per_iter(c)
        for k, v in rb.items():
            aggregated.setdefault(k, []).append(v)

    if not aggregated:
        print("  No data for mIoU progression — skipping")
        plt.close(fig)
        return

    xs = sorted(aggregated)
    mean_ys = np.array([np.mean(aggregated[x]) for x in xs])
    std_ys  = np.array([np.std(aggregated[x])  for x in xs])
    n_ys    = np.array([len(aggregated[x])     for x in xs])

    ax.plot(xs, mean_ys, marker="o", ms=7, lw=2.5, color="#2a7a2a",
            label=f"mIoU (running best, n={len(data)} clips)")
    ax.fill_between(xs, mean_ys - std_ys, mean_ys + std_ys,
                    color="#2a7a2a", alpha=0.2, label="±1 std")

    # Annotate sample sizes
    for x, y, n in zip(xs, mean_ys, n_ys):
        ax.text(x, y + 0.015, f"n={n}", ha="center", fontsize=7, color="#555555")

    ax.set_xlabel("Iteration  (0 = seed)")
    ax.set_ylabel("mIoU across clips")
    ax.set_title("Mean IoU progression — optimisation 'loss' curve\n"
                 "(higher is better)", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(mean_ys.max() + 0.1, 1.05))
    ax.grid(lw=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, out)


# ── Fig 3: top improvements bar+text ──────────────────────────────────────────

def fig3_top_improvements(data: List[dict], out: Path, top_n: int = 10):
    rows = []
    for c in data:
        seed = _seed_iou(c)
        best = c.get("best_iou")
        if seed is None or best is None:
            continue
        rows.append({
            "label":  _clip_label(c),
            "seed_expr":   c["seed_expression"],
            "best_expr":   c["best_expression"],
            "seed_iou":    seed,
            "best_iou":    best,
            "delta":       best - seed,
        })

    if not rows:
        print("  No data for top improvements — skipping")
        return

    rows = sorted(rows, key=lambda r: r["delta"], reverse=True)[:top_n]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.85)))

    y = np.arange(n)[::-1]  # so largest delta sits at top
    deltas = [r["delta"] for r in rows]
    colors = [_C_GAIN if d > 0.01 else (_C_LOSS if d < -0.01 else _C_FLAT) for d in deltas]

    ax.barh(y, deltas, color=colors, edgecolor="white", height=0.65, alpha=0.85)

    for i, (yi, r, d, col) in enumerate(zip(y, rows, deltas, colors)):
        # Right of bar: delta + clip label
        sign = "+" if d >= 0 else ""
        ax.text(max(d, 0) + 0.005, yi, f"Δ={sign}{d:.3f}   {r['label']}",
                va="center", ha="left", fontsize=8, color=col, fontweight="bold")
        # Below the bar: expression transitions
        ax.text(min(d, 0) - 0.005, yi + 0.18,
                f'seed (IoU={r["seed_iou"]:.3f}):  "{_short(r["seed_expr"], 50)}"',
                va="center", ha="right", fontsize=7.5, color="#4C72B0")
        ax.text(min(d, 0) - 0.005, yi - 0.18,
                f'best (IoU={r["best_iou"]:.3f}):  "{_short(r["best_expr"], 50)}"',
                va="center", ha="right", fontsize=7.5, color="#2a7a2a")

    max_d = max(deltas + [0])
    min_d = min(deltas + [0])
    ax.set_xlim(min_d - 0.35, max_d + 0.25)
    ax.set_yticks([])
    ax.set_xlabel("IoU gain over seed", fontsize=9)
    ax.set_title(f"Top {n} clips by IoU improvement\n"
                 f"(green = optimiser helped, red = hurt)", fontsize=11)
    ax.axvline(0, color="black", lw=0.6)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    _save(fig, out)


# ── Fig 4: seed vs best scatter ───────────────────────────────────────────────

def fig4_seed_vs_best(data: List[dict], out: Path):
    pts = []
    for c in data:
        seed = _seed_iou(c)
        best = c.get("best_iou")
        if seed is None or best is None:
            continue
        pts.append((seed, best, _clip_label(c)))

    if not pts:
        print("  No data for seed-vs-best — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for seed, best, _lbl in pts:
        col = _C_GAIN if best - seed > 0.01 else (_C_LOSS if best - seed < -0.01 else _C_FLAT)
        ax.scatter(seed, best, s=55, color=col, alpha=0.7, edgecolors="white", linewidths=0.4)

    # Diagonal: no change
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.6, label="seed = best")

    # Counts
    n_up   = sum(1 for s, b, _ in pts if b - s >  0.01)
    n_flat = sum(1 for s, b, _ in pts if abs(b - s) <= 0.01)
    n_down = sum(1 for s, b, _ in pts if b - s < -0.01)
    ax.text(0.02, 0.98, f"  improved : {n_up}\n  no change: {n_flat}\n  hurt     : {n_down}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", lw=0.5))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Seed IoU")
    ax.set_ylabel("Best LLM IoU")
    ax.set_title(f"Seed vs best LLM IoU per clip  (n={len(pts)})\n"
                 "points above diagonal → optimiser helped", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(lw=0.4, alpha=0.4)

    legend = [
        mpatches.Patch(color=_C_GAIN, label="improved"),
        mpatches.Patch(color=_C_FLAT, label="no change (±0.01)"),
        mpatches.Patch(color=_C_LOSS, label="hurt"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    _save(fig, out)


# ── Fig 5: delta distribution ─────────────────────────────────────────────────

def fig5_delta_distribution(data: List[dict], out: Path):
    deltas = []
    for c in data:
        seed = _seed_iou(c)
        best = c.get("best_iou")
        if seed is None or best is None:
            continue
        deltas.append(best - seed)

    if not deltas:
        print("  No data for delta distribution — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(-max(abs(min(deltas)), abs(max(deltas))) - 0.05,
                       max(abs(min(deltas)), abs(max(deltas))) + 0.05,
                       21)
    n, edges, _ = ax.hist(deltas, bins=bins, color="#4C72B0", alpha=0.75, edgecolor="white")

    # Color positive vs negative bars
    for i, patch in enumerate(ax.patches):
        center = (edges[i] + edges[i + 1]) / 2
        patch.set_facecolor(_C_GAIN if center > 0.01 else (_C_LOSS if center < -0.01 else _C_FLAT))
        patch.set_alpha(0.8)

    ax.axvline(0, color="black", lw=0.8)
    mean_d = float(np.mean(deltas))
    ax.axvline(mean_d, color="#222222", lw=1.5, ls=":",
               label=f"mean Δ = {mean_d:+.3f}")

    ax.set_xlabel("IoU(best) − IoU(seed)  (per clip)")
    ax.set_ylabel("Number of clips")
    ax.set_title(f"Distribution of per-clip IoU improvement  (n={len(deltas)})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    _save(fig, out)


# ── Summary CSV ───────────────────────────────────────────────────────────────

def write_summary_csv(data: List[dict], out: Path):
    fields = ["seq_name", "obj_id", "seed_iou", "best_iou", "delta_iou",
              "seed_expression", "best_expression", "n_attempts"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in data:
            seed = _seed_iou(c)
            best = c.get("best_iou")
            delta = (best - seed) if (seed is not None and best is not None) else None
            w.writerow({
                "seq_name": c["seq_name"],
                "obj_id":   c["obj_id"],
                "seed_iou": f"{seed:.4f}" if seed is not None else "",
                "best_iou": f"{best:.4f}" if best is not None else "",
                "delta_iou": f"{delta:+.4f}" if delta is not None else "",
                "seed_expression": c["seed_expression"],
                "best_expression": c.get("best_expression") or "",
                "n_attempts": len(c["history"]),
            })
    print(f"  Wrote {out}")


# ── Driver ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="clip_results.json path")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--top_n", type=int, default=10, help="how many top clips in fig3")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(results_path.read_text())
    print(f"Loaded {len(data)} clip(s) from {results_path}")

    fig1_optimization_curves(data, out_dir / "fig1_optimization_curves.png")
    fig2_miou_progression(   data, out_dir / "fig2_miou_progression.png")
    fig3_top_improvements(   data, out_dir / "fig3_top_improvements.png", top_n=args.top_n)
    fig4_seed_vs_best(       data, out_dir / "fig4_seed_vs_best.png")
    fig5_delta_distribution( data, out_dir / "fig5_delta_distribution.png")
    write_summary_csv(       data, out_dir / "summary.csv")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
