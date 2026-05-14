"""
plot.py
-------
Visualisations for LLM-optimizer results.

Figures produced
----------------
  fig1_strip.png          Per-group expression score strip-plot
                          (original vs LLM, all IoU values)
  fig2_before_after.png   Worst/Mean/Best IoU before & after optimisation
  fig3_iou_vs_mass.png    IoU vs MassGT scatter — shows attention/box dissociation
  fig4_iterations.png     Best IoU found per iteration (convergence curve)
  fig5_failure_modes.png  Rule-based failure-mode breakdown of all expressions

Usage
-----
  python plot.py --results results/llm_optimizer/optimization_results.json \
                 --out_dir results/llm_optimizer/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_classifier import classify_rule_based


# ── Palette & helpers ──────────────────────────────────────────────────────────

_C_ORIG  = "#4C72B0"   # blue  – original seed expressions
_C_LLM1  = "#DD8452"   # orange – llm_iter_1
_C_LLM2  = "#55A868"   # green  – llm_iter_2
_C_BEST  = "#C44E52"   # red   – best expression marker

_SOURCE_COLOR = {
    "original": _C_ORIG,
    "llm_iter_1": _C_LLM1,
    "llm_iter_2": _C_LLM2,
}

def _source_color(src: str) -> str:
    if src in _SOURCE_COLOR:
        return _SOURCE_COLOR[src]
    # fallback for iter_3, iter_4, …
    return "#8172B2"

def _group_label(r: dict) -> str:
    return f"{r['seq_name']}\nobj{r['obj_id']}"

def _short_expr(e: str, n: int = 28) -> str:
    return e if len(e) <= n else e[:n-1] + "…"

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


# ── Fig 1: Strip plot ──────────────────────────────────────────────────────────

def fig1_strip(data: list[dict], out: Path):
    n = len(data)
    fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 5))

    for gi, r in enumerate(data):
        for h in r["history"]:
            iou = h["iou"]
            if iou is None:
                continue
            color = _source_color(h["source"])
            jitter = (hash(h["expression"]) % 100 - 50) / 600
            ax.scatter(gi + jitter, iou, color=color, s=55, zorder=3,
                       alpha=0.85, linewidths=0.4, edgecolors="white")

        # Mark best original
        if r["best_original_iou"] is not None:
            ax.scatter(gi, r["best_original_iou"], marker="*", s=180,
                       color=_C_BEST, zorder=5, linewidths=0.5, edgecolors="white")

    ax.set_xticks(range(n))
    ax.set_xticklabels([_group_label(r) for r in data], fontsize=8)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Expression IoU — original seed vs LLM-generated candidates", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    legend = [
        mpatches.Patch(color=_C_ORIG,  label="Original seed"),
        mpatches.Patch(color=_C_LLM1,  label="LLM iter 1"),
        mpatches.Patch(color=_C_LLM2,  label="LLM iter 2"),
        plt.scatter([], [], marker="*", color=_C_BEST, s=120, label="Best original"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")
    _save(fig, out)


# ── Fig 2: Before / After bar chart ───────────────────────────────────────────

def fig2_before_after(data: list[dict], out: Path):
    labels    = [_group_label(r) for r in data]
    worst_orig = []
    mean_orig  = []
    best_orig  = []
    best_llm   = []

    for r in data:
        orig_iou = [h["iou"] for h in r["history"] if h["source"] == "original" and h["iou"] is not None]
        llm_iou  = [h["iou"] for h in r["history"] if h["source"] != "original"  and h["iou"] is not None]
        worst_orig.append(min(orig_iou) if orig_iou else 0)
        mean_orig.append( float(np.mean(orig_iou)) if orig_iou else 0)
        best_orig.append( max(orig_iou) if orig_iou else 0)
        best_llm.append(  max(llm_iou)  if llm_iou  else 0)

    n  = len(data)
    x  = np.arange(n)
    w  = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n * 2.4), 5))
    ax.bar(x - 1.5*w, worst_orig, w, label="Orig worst",  color="#d6e4f0", edgecolor="grey", lw=0.6)
    ax.bar(x - 0.5*w, mean_orig,  w, label="Orig mean",   color=_C_ORIG,  edgecolor="grey", lw=0.6, alpha=0.8)
    ax.bar(x + 0.5*w, best_orig,  w, label="Orig best",   color="#1a4f8a", edgecolor="grey", lw=0.6)
    ax.bar(x + 1.5*w, best_llm,   w, label="Best LLM",    color=_C_LLM1,  edgecolor="grey", lw=0.6)

    # Annotate delta on best-llm bars: compare to worst original (floor improvement)
    for i, (wst, bl) in enumerate(zip(worst_orig, best_llm)):
        delta = bl - wst
        color = "#2a7a2a" if delta > 0.01 else ("#888888" if delta >= -0.01 else "#b22222")
        ax.text(i + 1.5*w, bl + 0.01, f"Δ={delta:+.2f}\nvs worst", ha="center", va="bottom",
                fontsize=6.5, color=color, fontweight="bold", linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Expression IoU: original distribution vs best LLM candidate\n"
                 "(Δ annotation = best LLM − worst original = floor improvement)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    _save(fig, out)


# ── Fig 3: IoU vs MassGT scatter ──────────────────────────────────────────────

def fig3_iou_vs_mass(data: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))

    group_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for gi, r in enumerate(data):
        marker = group_markers[gi % len(group_markers)]
        label_done = set()
        for h in r["history"]:
            if h["iou"] is None or h["mass_in_gt"] is None:
                continue
            color = _source_color(h["source"])
            lbl = None
            if color not in label_done:
                lbl = h["source"].replace("llm_", "LLM ").replace("_", " ")
                label_done.add(color)
            ax.scatter(h["iou"], h["mass_in_gt"], color=color, marker=marker,
                       s=60, alpha=0.8, zorder=3, linewidths=0.4, edgecolors="white")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, alpha=0.5, label="IoU = MassGT")

    # Per-group legend (shapes)
    shape_handles = [
        plt.scatter([], [], marker=group_markers[gi % len(group_markers)],
                    color="grey", s=50, label=_group_label(r))
        for gi, r in enumerate(data)
    ]
    color_handles = [
        mpatches.Patch(color=_C_ORIG,  label="Original"),
        mpatches.Patch(color=_C_LLM1,  label="LLM iter 1"),
        mpatches.Patch(color=_C_LLM2,  label="LLM iter 2"),
    ]
    ax.legend(handles=shape_handles + color_handles, fontsize=7,
              loc="upper left", ncol=2, framealpha=0.8)

    ax.set_xlabel("IoU  (box overlap with GT)")
    ax.set_ylabel("MassGT  (attention inside GT box)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.5)
    ax.set_title("IoU vs Attention Mass in GT\n"
                 "Points above diagonal → attention correct, box regression off", fontsize=10)
    ax.grid(lw=0.4, alpha=0.4)

    # Annotate quadrant interpretation
    ax.text(0.05, 0.45, "Attention\ncorrect,\nbox wrong", fontsize=7, color="grey",
            ha="left", va="top", style="italic")
    ax.text(0.85, 0.04, "Both\ngood", fontsize=7, color="grey",
            ha="right", va="bottom", style="italic")
    ax.text(0.05, 0.04, "Both\nwrong", fontsize=7, color="grey",
            ha="left", va="bottom", style="italic")

    _save(fig, out)


# ── Fig 4: Convergence curve ───────────────────────────────────────────────────

def fig4_iterations(data: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))

    for gi, r in enumerate(data):
        # Collect best IoU seen at end of each iteration step
        orig_iou = [h["iou"] for h in r["history"]
                    if h["source"] == "original" and h["iou"] is not None]
        if not orig_iou:
            continue
        running_best = [max(orig_iou)]

        for it_data in r["iterations"]:
            it_iou = [c["iou"] for c in it_data["candidates"] if c["iou"] is not None]
            running_best.append(max(running_best[-1], max(it_iou) if it_iou else running_best[-1]))

        xs = list(range(len(running_best)))
        color = plt.cm.tab10(gi / max(len(data), 1))
        ax.plot(xs, running_best, marker="o", ms=6, lw=1.8, color=color,
                label=_group_label(r))
        ax.axhline(max(orig_iou), color=color, lw=0.8, ls=":", alpha=0.5)

    ax.set_xticks(range(max(len(r["iterations"]) + 1 for r in data)))
    ax.set_xticklabels(["Seed best"] + [f"After iter {i+1}"
                        for i in range(max(len(r["iterations"]) for r in data))], fontsize=8)
    ax.set_ylabel("Best IoU so far")
    ax.set_ylim(0, 1.05)
    ax.set_title("Optimizer convergence — best IoU per iteration", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(lw=0.4, alpha=0.5)
    _save(fig, out)


# ── Fig 5: Failure-mode breakdown ─────────────────────────────────────────────

def fig5_failure_modes(data: list[dict], out: Path):
    from collections import Counter

    mode_counter_orig = Counter()
    mode_counter_llm  = Counter()

    for r in data:
        for h in r["history"]:
            mode = classify_rule_based(h["expression"]).primary_mode
            if h["source"] == "original":
                mode_counter_orig[mode] += 1
            else:
                mode_counter_llm[mode] += 1

    all_modes = sorted(set(list(mode_counter_orig) + list(mode_counter_llm)))

    orig_vals = [mode_counter_orig.get(m, 0) for m in all_modes]
    llm_vals  = [mode_counter_llm.get(m,  0) for m in all_modes]

    x = np.arange(len(all_modes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(x + w/2, orig_vals, w, label="Original", color=_C_ORIG,  alpha=0.85, edgecolor="white")
    ax.barh(x - w/2, llm_vals,  w, label="LLM",      color=_C_LLM1,  alpha=0.85, edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(all_modes, fontsize=9)
    ax.set_xlabel("Number of expressions")
    ax.set_title("Failure-mode distribution: original seed vs LLM-generated candidates", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", lw=0.4, alpha=0.5)

    # Annotate counts
    for i, (ov, lv) in enumerate(zip(orig_vals, llm_vals)):
        if ov: ax.text(ov + 0.05, i + w/2, str(ov), va="center", fontsize=8)
        if lv: ax.text(lv + 0.05, i - w/2, str(lv), va="center", fontsize=8)

    _save(fig, out)


# ── Driver ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="optimization_results.json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(results_path.read_text())
    print(f"Loaded {len(data)} groups from {results_path}")

    fig1_strip(        data, out_dir / "fig1_strip.png")
    fig2_before_after( data, out_dir / "fig2_before_after.png")
    fig3_iou_vs_mass(  data, out_dir / "fig3_iou_vs_mass.png")
    fig4_iterations(   data, out_dir / "fig4_iterations.png")
    fig5_failure_modes(data, out_dir / "fig5_failure_modes.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
