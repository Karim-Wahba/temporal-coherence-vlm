"""
run.py
------
End-to-end driver for the expression-variance experiment.

Steps:
  1. Run inference on every (seq, exp) item — saves results.json
  2. Group by (seq_name, obj_id) and compute {min, mean, max, variance}
     per metric — saves grouped_stats.json + summary.json
  3. Render four figures into figures/

Usage
-----
    python run.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --save_dir   results/expression_variance \\
        --split      valid \\
        --sample_rate 8

Subset flags:
    --max_sequences  N        cap on unique sequences
    --expressions_per_seq N   cap on expressions per (seq, obj)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import ExpressionVarianceExperiment
from analyze    import group_results, summary
from plot       import figure_1_range, figure_2_best_vs_worst, figure_3_coupling, figure_4_strip, _load
from plot_extras import render_all as render_extras


def parse_args():
    p = argparse.ArgumentParser("Expression-Variance Experiment")
    p.add_argument("--davis_root", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/expression_variance")
    p.add_argument("--split", default="valid", choices=["valid", "train"])
    p.add_argument("--sample_rate", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--expressions_per_seq", type=int, default=None,
                   help="None = use all expressions per (seq, obj)")
    p.add_argument("--image_mode", action="store_true")
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip inference and just re-run analyze + plot from existing results.json")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "results.json"
    grouped_path = save_dir / "grouped_stats.json"
    summary_path = save_dir / "summary.json"
    fig_dir      = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Inference ──────────────────────────────────────────────────────────
    if args.skip_inference:
        if not results_path.exists():
            sys.exit(f"--skip_inference set but {results_path} doesn't exist")
        results = json.load(open(results_path))
        print(f"Loaded {len(results)} existing rows from {results_path}")
    else:
        print(f"Loading model: {args.model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        model.eval()

        exp = ExpressionVarianceExperiment(
            model=model,
            processor=processor,
            davis_root=args.davis_root,
            save_dir=str(save_dir),
            video_mode=not args.image_mode,
            sample_rate=args.sample_rate,
            max_new_tokens=args.max_new_tokens,
            split=args.split,
            seed=args.seed,
        )
        results = exp.run_all(
            sequences=args.sequences,
            max_sequences=args.max_sequences,
            expressions_per_seq=args.expressions_per_seq,
        )
        print(f"Wrote {results_path}  ({len(results)} rows)")

    # ── 2. Group + summarize ──────────────────────────────────────────────────
    grouped = group_results(results)
    summ    = summary(grouped)
    grouped_path.write_text(json.dumps(grouped, indent=2))
    summary_path.write_text(json.dumps(summ,    indent=2))
    print(f"Wrote {grouped_path}")
    print(f"Wrote {summary_path}")

    print("\n=== Dataset-level summary ===")
    print(f"  groups: {summ['num_groups']}")
    for m in ("iou", "mass_in_gt", "mass_in_pred"):
        s = summ[m]
        print(f"  {m}:")
        print(f"    mean within-group std:    {s['mean_within_group_std']:.4f}")
        print(f"    mean within-group range:  {s['mean_within_group_range']:.4f}")
        print(f"    mean best - worst gap:    {s['mean_best_minus_worst']:.4f}")
        print(f"    mean(worst) → mean(best): {s['mean_worst']:.4f} → {s['mean_best']:.4f}")

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    rows = _load(grouped_path)
    figure_1_range         (rows, fig_dir / "fig1_range_per_group.png")
    figure_2_best_vs_worst (rows, fig_dir / "fig2_best_vs_worst.png")
    figure_3_coupling      (rows, fig_dir / "fig3_attention_iou_coupling.png")
    figure_4_strip         (rows, fig_dir / "fig4_strip_per_group.png")
    print(f"\nWrote 4 figures to {fig_dir}")

    render_extras(grouped_path, save_dir)


if __name__ == "__main__":
    main()
