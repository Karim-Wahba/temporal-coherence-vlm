"""
plot_analysis.py
----------------
Visualisations for analysis.json produced by the analysis phase of poc.py.

Figures produced
----------------
  ana1_feature_correlation.png   Delta-IoU per structural feature (present vs absent)
  ana2_pos_patterns.png          Top POS patterns by mean IoU (bubble = count)
  ana3_feature_distributions.png IoU distribution when each feature is present vs absent
  ana4_length_vs_iou.png         Expression word count vs IoU (scatter + bin means)
  ana5_semantic_categories.png   Qwen-discovered failure/success categories with IoU ranges

Usage
-----
  python plot_analysis.py --analysis path/to/analysis.json --out_dir path/to/figures
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def _parse_iou_range(range_str: str):
    """Parse '0.40-0.55' or '0.40–0.55' into (lo, hi)."""
    cleaned = range_str.replace("–", "-").replace("—", "-")
    m = re.findall(r"[\d.]+", cleaned)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    return None, None


def _extract_semantic_categories(raw: str) -> tuple[list[dict], list[dict]]:
    """Pull complete category objects out of (possibly truncated) JSON string."""
    pattern = r'\{[^{}]*"name"[^{}]*"typical_iou_range"[^{}]*\}'
    matches = re.findall(pattern, raw, re.DOTALL)

    failures, successes = [], []
    # Heuristic: search for where success_patterns section starts in raw
    succ_start = raw.find('"success_patterns"')
    succ_char = succ_start if succ_start != -1 else len(raw)

    for m in matches:
        m_clean = m.replace("–", "-").replace("—", "-")
        m_clean = m_clean.replace("‘", "'").replace("’", "'")
        try:
            obj = json.loads(m_clean)
        except Exception:
            continue
        # Estimate which section this belongs to by position in raw
        pos = raw.find('"' + obj["name"] + '"')
        if pos != -1 and pos < succ_char:
            failures.append(obj)
        else:
            successes.append(obj)
    return failures, successes


# ── Fig 1: Feature correlation ─────────────────────────────────────────────────

def ana1_feature_correlation(syntactic: dict, out: Path):
    corr = syntactic["feature_correlations"]

    features  = list(corr.keys())
    deltas     = [corr[f]["delta"] for f in features]
    n_present  = [corr[f]["n_present"] for f in features]
    iou_pres   = [corr[f]["mean_iou_present"] or 0 for f in features]
    iou_abs    = [corr[f]["mean_iou_absent"]  or 0 for f in features]

    # Sort by delta
    order = np.argsort(deltas)
    features  = [features[i]  for i in order]
    deltas     = [deltas[i]    for i in order]
    n_present  = [n_present[i] for i in order]
    iou_pres   = [iou_pres[i]  for i in order]
    iou_abs    = [iou_abs[i]   for i in order]

    n = len(features)
    y = np.arange(n)
    colors = ["#C44E52" if d < 0 else "#55A868" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(y, deltas, color=colors, edgecolor="white", height=0.6)

    # Annotate: n_present and IoU values
    for i, (d, np_, ip, ia) in enumerate(zip(deltas, n_present, iou_pres, iou_abs)):
        sign = "+" if d >= 0 else ""
        ax.text(d + (0.002 if d >= 0 else -0.002),
                i, f"{sign}{d:.3f}  (n={np_:3d}  pres={ip:.3f} abs={ia:.3f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ mean IoU  (present − absent)", fontsize=9)
    ax.set_title("Feature impact on mean IoU\n"
                 "Green = feature present → higher IoU  |  Red = feature present → lower IoU",
                 fontsize=10)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.set_xlim(min(deltas) - 0.08, max(deltas) + 0.25)

    legend = [
        mpatches.Patch(color="#55A868", label="Positive effect"),
        mpatches.Patch(color="#C44E52", label="Negative effect"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    _save(fig, out)


# ── Fig 2: POS patterns ────────────────────────────────────────────────────────

def ana2_pos_patterns(syntactic: dict, out: Path):
    patterns = syntactic["pos_pattern_stats"]
    if not patterns:
        print("  No POS pattern data — skipping ana2")
        return

    # Show top 15 by mean IoU (already sorted)
    patterns = patterns[:15]
    labels    = [p["pattern"] for p in patterns]
    means     = [p["mean_iou"] for p in patterns]
    counts    = [p["count"]    for p in patterns]

    # Shorten long patterns for display
    def shorten(pat, max_len=50):
        return pat if len(pat) <= max_len else pat[:max_len - 1] + "…"

    short_labels = [shorten(l) for l in labels]

    n  = len(patterns)
    y  = np.arange(n)
    # Bar color: gradient from green (high IoU) to red (low IoU)
    norm = plt.Normalize(min(means), max(means))
    cmap = plt.cm.RdYlGn
    colors = [cmap(norm(m)) for m in means]

    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.55)))
    bars = ax.barh(y, means, color=colors, edgecolor="white", height=0.65)

    # Count labels on bars
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(m + 0.005, i, f"{m:.3f}  (n={c})", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Mean IoU", fontsize=9)
    ax.set_title("Top POS patterns by mean IoU\n(color: green=good, red=poor)", fontsize=10)
    ax.set_xlim(0, max(means) + 0.2)
    ax.axvline(np.mean(means), color="grey", lw=0.8, ls="--", alpha=0.6,
               label=f"Mean={np.mean(means):.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.4, label="Mean IoU")

    _save(fig, out)


# ── Fig 3: Feature distributions ──────────────────────────────────────────────

def ana3_feature_distributions(syntactic: dict, out: Path):
    per_expr = syntactic["per_expression"]
    features = [
        "has_verb", "multi_adj", "adj_heavy", "long_expr",
        "short_expr", "noun_only", "adj_noun", "has_adj",
    ]

    # Collect IoU lists
    data_present = {f: [] for f in features}
    data_absent  = {f: [] for f in features}
    for row in per_expr:
        iou  = row["iou"]
        feat = row["features"]
        for f in features:
            if feat.get(f):
                data_present[f].append(iou)
            else:
                data_absent[f].append(iou)

    n  = len(features)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes = axes.flatten()

    for ax, f in zip(axes, features):
        pres = data_present[f]
        abse = data_absent[f]

        parts = ax.violinplot([pres, abse], positions=[0, 1],
                              showmedians=True, showextrema=False)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        for i, (pc, col) in enumerate(zip(parts["bodies"], ["#55A868", "#4C72B0"])):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)

        # Mean markers
        ax.scatter([0, 1], [np.mean(pres) if pres else 0,
                            np.mean(abse) if abse else 0],
                   color="white", s=40, zorder=5, edgecolors="black", lw=1)

        delta = np.mean(pres) - np.mean(abse) if pres and abse else 0
        sign  = "+" if delta >= 0 else ""
        ax.set_title(f"{f}\nΔ={sign}{delta:.3f}  (n_pres={len(pres)})", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Present", "Absent"], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.set_ylabel("IoU" if ax in axes[::4] else "")

    present_patch = mpatches.Patch(color="#55A868", alpha=0.7, label="Feature present")
    absent_patch  = mpatches.Patch(color="#4C72B0", alpha=0.7, label="Feature absent")
    fig.legend(handles=[present_patch, absent_patch], fontsize=9,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("IoU distribution: feature present vs absent\n"
                 "(white dot = mean, line = median)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out)


# ── Fig 4: Length vs IoU ───────────────────────────────────────────────────────

def ana4_length_vs_iou(syntactic: dict, out: Path):
    per_expr = syntactic["per_expression"]
    lengths  = [row["features"]["n_words"] for row in per_expr]
    ious     = [row["iou"]                 for row in per_expr]

    lengths = np.array(lengths)
    ious    = np.array(ious)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter with transparency
    ax.scatter(lengths, ious, alpha=0.35, s=30, color="#4C72B0", zorder=3,
               linewidths=0, label="Expressions")

    # Bin means: 1–3, 4–5, 6–7, 8–9, 10+
    bins = [(1, 3), (4, 5), (6, 7), (8, 9), (10, 99)]
    bin_labels = ["1-3", "4-5", "6-7", "8-9", "10+"]
    bin_centers = [2, 4.5, 6.5, 8.5, 11]
    bin_means   = []
    bin_stds    = []
    bin_counts  = []
    for lo, hi in bins:
        mask = (lengths >= lo) & (lengths <= hi)
        vals = ious[mask]
        bin_means.append(np.mean(vals) if len(vals) else np.nan)
        bin_stds.append( np.std(vals)  if len(vals) else np.nan)
        bin_counts.append(len(vals))

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-",
                color="#C44E52", ms=8, lw=2, capsize=5, zorder=5, label="Bin mean ± std")

    # Annotate counts
    for cx, cm, cnt in zip(bin_centers, bin_means, bin_counts):
        if not np.isnan(cm):
            ax.text(cx, cm + 0.04, f"n={cnt}", ha="center", fontsize=8, color="#C44E52")

    ax.set_xlabel("Expression length (words)", fontsize=9)
    ax.set_ylabel("Mean IoU", fontsize=9)
    ax.set_title("Expression length vs IoU\n(red: bin means with ±1 std)", fontsize=10)
    ax.set_xticks(sorted(set(lengths)))
    ax.set_xticklabels(sorted(set(lengths)), fontsize=7, rotation=45)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(lw=0.4, alpha=0.4)
    _save(fig, out)


# ── Fig 5: Semantic categories ─────────────────────────────────────────────────

def ana5_semantic_categories(semantic: dict, out: Path):
    raw = semantic.get("raw_response", "")
    failures, successes = _extract_semantic_categories(raw)

    if not failures and not successes:
        print("  No semantic categories parsed — skipping ana5")
        return

    all_cats  = [(c, "failure") for c in failures] + [(c, "success") for c in successes]
    names     = [c["name"] for c, _ in all_cats]
    kinds     = [k          for _, k  in all_cats]
    lo_vals   = []
    hi_vals   = []
    mid_vals  = []
    for c, _ in all_cats:
        lo, hi = _parse_iou_range(c.get("typical_iou_range", ""))
        lo_vals.append(lo if lo is not None else 0.0)
        hi_vals.append(hi if hi is not None else 0.0)
        mid_vals.append(((lo or 0) + (hi or 0)) / 2)

    # Sort by mid IoU
    order  = np.argsort(mid_vals)
    names  = [names[i]  for i in order]
    kinds  = [kinds[i]  for i in order]
    lo_v   = [lo_vals[i]  for i in order]
    hi_v   = [hi_vals[i]  for i in order]
    mid_v  = [mid_vals[i] for i in order]

    n   = len(all_cats)
    y   = np.arange(n)
    colors = ["#55A868" if k == "success" else "#C44E52" for k in kinds]

    def wrap(s, w=40):
        words = s.split()
        lines, cur = [], []
        for word in words:
            if sum(len(c) + 1 for c in cur) + len(word) > w:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        lines.append(" ".join(cur))
        return "\n".join(lines)

    wrapped_names = [wrap(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.85)))
    for i, (lo, hi, mid, col) in enumerate(zip(lo_v, hi_v, mid_v, colors)):
        # Range bar
        ax.barh(i, hi - lo, left=lo, color=col, alpha=0.45, height=0.55,
                edgecolor=col, linewidth=1.2)
        # Mid point
        ax.scatter(mid, i, color=col, s=70, zorder=5, edgecolors="white", linewidth=1)
        # IoU label
        ax.text(hi + 0.01, i, f"{lo:.2f}–{hi:.2f}", va="center", fontsize=7.5, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_names, fontsize=8)
    ax.set_xlabel("Mean IoU range", fontsize=9)
    ax.set_title(
        "Qwen-discovered expression categories\n"
        "Green = success pattern  |  Red = failure mode  |  bar = IoU range, dot = midpoint",
        fontsize=10,
    )
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5, label="IoU=0.5")
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    legend = [
        mpatches.Patch(color="#55A868", alpha=0.6, label="Success pattern"),
        mpatches.Patch(color="#C44E52", alpha=0.6, label="Failure mode"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")

    # Key insight annotation if available
    ki = semantic.get("key_insight", "")
    if ki:
        fig.text(0.5, -0.02, f"Key insight: {ki}", ha="center", fontsize=8,
                 style="italic", color="#555555", wrap=True)

    fig.tight_layout()
    _save(fig, out)


# ── Fig 6: Head noun IoU distribution ─────────────────────────────────────────

def ana6_head_nouns(syntactic: dict, out: Path):
    from collections import defaultdict

    per_expr = syntactic["per_expression"]
    noun_groups: dict[str, list[float]] = defaultdict(list)
    for row in per_expr:
        hn = row["features"].get("head_noun", "")
        if hn:
            noun_groups[hn].append(row["iou"])

    # Keep nouns with at least 3 expressions
    noun_groups = {k: v for k, v in noun_groups.items() if len(v) >= 3}
    if not noun_groups:
        print("  No head noun groups — skipping ana6")
        return

    nouns  = sorted(noun_groups.keys(), key=lambda k: np.mean(noun_groups[k]))
    means  = [np.mean(noun_groups[n]) for n in nouns]
    counts = [len(noun_groups[n])     for n in nouns]

    n = len(nouns)
    y = np.arange(n)

    norm   = plt.Normalize(min(means), max(means))
    colors = [plt.cm.RdYlGn(norm(m)) for m in means]

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.42)))
    ax.barh(y, means, color=colors, edgecolor="white", height=0.65)

    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(m + 0.005, i, f"{m:.3f}  n={c}", va="center", fontsize=7.5)

    # Overlay individual points
    for i, noun in enumerate(nouns):
        vals = noun_groups[noun]
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(vals))
        ax.scatter(vals, [i + j for j in jitter],
                   color="white", s=18, zorder=4, alpha=0.7,
                   linewidths=0.4, edgecolors="grey")

    ax.set_yticks(y)
    ax.set_yticklabels(nouns, fontsize=8)
    ax.set_xlabel("Mean IoU", fontsize=9)
    ax.set_title("IoU by head noun  (min 3 expressions)\n"
                 "Sorted by mean IoU — dots = individual expressions", fontsize=10)
    ax.set_xlim(0, max(means) + 0.25)
    ax.axvline(np.mean(means), color="grey", lw=0.8, ls="--", alpha=0.6,
               label=f"Overall mean={np.mean(means):.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.35, label="Mean IoU")

    _save(fig, out)


# ── Driver ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="analysis.json path")
    ap.add_argument("--out_dir",  default=None)
    args = ap.parse_args()

    analysis_path = Path(args.analysis)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_path.parent / "figures_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(analysis_path.read_text())
    syntactic = data["syntactic"]
    semantic  = data.get("semantic", {})

    print(f"Loaded analysis: {syntactic['n_expressions']} expressions, "
          f"tagger={syntactic['tagger']}")

    ana1_feature_correlation(syntactic, out_dir / "ana1_feature_correlation.png")
    ana2_pos_patterns(        syntactic, out_dir / "ana2_pos_patterns.png")
    ana3_feature_distributions(syntactic, out_dir / "ana3_feature_distributions.png")
    ana4_length_vs_iou(       syntactic, out_dir / "ana4_length_vs_iou.png")
    ana5_semantic_categories( semantic,  out_dir / "ana5_semantic_categories.png")
    ana6_head_nouns(          syntactic, out_dir / "ana6_head_nouns.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
