"""
compare_models.py
-----------------
Generates a comparison report between two or more model benchmark runs.

Usage
-----
    python compare_models.py \
        --runs results/qwen3vl_8b results/qwen25vl_7b \
        --names "Qwen3-VL-8B" "Qwen2.5-VL-7B" \
        --save_dir results/comparison
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("JF",              "J&F ↑",            True),
    ("mean_J",          "Mean J ↑",         True),
    ("mean_F",          "Mean F ↑",         True),
    ("J_decay",         "J-Decay ↑",        True),   # less negative = better
    ("J_variance",      "J-Variance ↓",     False),
    ("success_rate_50", "Success@0.5 ↑",    True),
    ("success_rate_75", "Success@0.75 ↑",   True),
    ("J_first",         "J First Frame ↑",  True),
    ("J_last",          "J Last Frame ↑",   True),
]


def load_run(run_dir: str) -> dict:
    summary_path = os.path.join(run_dir, "summary.json")
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json in {run_dir}")
    with open(summary_path) as f:
        summary = json.load(f)
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    return {"summary": summary, "df": df, "run_dir": run_dir}


def print_comparison_table(runs: list, names: list):
    print(f"\n{'Metric':<22}", end="")
    for name in names:
        print(f"  {name[:14]:>14}", end="")
    print(f"  {'Delta(A-B)':>12}")
    print("─" * (22 + 16 * len(names) + 14))

    agg_list = [r["summary"]["aggregate_metrics"] for r in runs]

    for key, label, higher_better in METRICS:
        print(f"  {label:<20}", end="")
        values = [a.get(key, float("nan")) for a in agg_list]
        for val in values:
            print(f"  {val:>14.4f}", end="")
        if len(values) >= 2:
            delta = values[0] - values[1]
            winner = "✓" if (delta > 0) == higher_better else "✗"
            print(f"  {delta:>+11.4f} {winner}", end="")
        print()


def plot_comparison_bars(runs: list, names: list, save_path: str):
    """Grouped bar chart comparing all models across key metrics."""
    plot_metrics = [
        ("JF", "J&F"),
        ("mean_J", "Mean J"),
        ("mean_F", "Mean F"),
        ("success_rate_50", "Success@0.5"),
        ("J_variance", "J-Variance"),
    ]

    x = np.arange(len(plot_metrics))
    width = 0.8 / len(runs)
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (run, name) in enumerate(zip(runs, names)):
        agg = run["summary"]["aggregate_metrics"]
        values = [agg.get(k, 0) for k, _ in plot_metrics]
        offset = (i - len(runs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in plot_metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Ref-DAVIS Temporal Coherence Benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_j_decay_comparison(runs: list, names: list, save_path: str):
    """
    Per-sequence J-decay comparison: shows which sequences each model
    loses track on (sorted by baseline model's J-decay).
    """
    dfs = [r["df"] for r in runs if r["df"] is not None]
    if len(dfs) < 2:
        return

    # Merge on seq_name
    merged = dfs[0][["seq_name", "J_decay"]].copy().rename(columns={"J_decay": names[0]})
    for df, name in zip(dfs[1:], names[1:]):
        merged = merged.merge(
            df[["seq_name", "J_decay"]].rename(columns={"J_decay": name}),
            on="seq_name", how="inner"
        )
    merged = merged.sort_values(names[0])

    fig, ax = plt.subplots(figsize=(max(10, len(merged) * 0.4), 6))
    x = np.arange(len(merged))
    width = 0.8 / len(names)
    colors = plt.cm.Set1.colors

    for i, name in enumerate(names):
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar(x + offset, merged[name], width, label=name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["seq_name"].tolist(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("J-Decay (slope of IoU over time)")
    ax.set_title("Per-Sequence J-Decay Comparison\n(More negative = model loses track faster)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_failure_mode_comparison(runs: list, names: list, save_path: str):
    """Side-by-side failure mode distributions."""
    all_modes = [
        "SUCCESS", "NEVER_FOUND", "LOST_TRACK", "PARTIAL_TRACK",
        "IDENTITY_SWAP", "TEMPORAL_COLLAPSE", "ATTENTION_DRIFT",
        "UNSTABLE", "OCCLUSION_FAIL",
    ]
    COLORS = {
        "SUCCESS": "#2ecc71", "NEVER_FOUND": "#e74c3c", "LOST_TRACK": "#e67e22",
        "PARTIAL_TRACK": "#f1c40f", "IDENTITY_SWAP": "#9b59b6",
        "TEMPORAL_COLLAPSE": "#3498db", "ATTENTION_DRIFT": "#1abc9c",
        "UNSTABLE": "#607d8b", "OCCLUSION_FAIL": "#e91e63",
    }

    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6))
    if len(runs) == 1:
        axes = [axes]

    for ax, run, name in zip(axes, runs, names):
        dist = run["summary"].get("failure_summary", {}).get("primary_distribution", {})
        labels = [m for m in all_modes if m in dist and dist[m]["count"] > 0]
        sizes = [dist[m]["count"] for m in labels]
        colors = [COLORS.get(m, "#aaaaaa") for m in labels]
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                   textprops={"fontsize": 8})
        ax.set_title(name)

    fig.suptitle("Failure Mode Distribution Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    p = argparse.ArgumentParser("Compare Model Benchmark Runs")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Paths to result directories from benchmark.py")
    p.add_argument("--names", nargs="+", default=None,
                   help="Display names (defaults to directory names)")
    p.add_argument("--save_dir", default="results/comparison")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    names = args.names or [os.path.basename(r) for r in args.runs]
    assert len(names) == len(args.runs), "Number of names must match number of runs"

    runs = []
    for run_dir in args.runs:
        try:
            runs.append(load_run(run_dir))
        except Exception as e:
            print(f"Failed to load {run_dir}: {e}")
            return

    print_comparison_table(runs, names)

    plot_comparison_bars(
        runs, names,
        os.path.join(args.save_dir, "metric_comparison.png")
    )
    plot_j_decay_comparison(
        runs, names,
        os.path.join(args.save_dir, "j_decay_per_sequence.png")
    )
    plot_failure_mode_comparison(
        runs, names,
        os.path.join(args.save_dir, "failure_mode_comparison.png")
    )

    print(f"\nAll comparison outputs in: {args.save_dir}/")


if __name__ == "__main__":
    main()
