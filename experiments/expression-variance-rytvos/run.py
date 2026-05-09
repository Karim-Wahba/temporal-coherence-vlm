"""
run.py
------
End-to-end driver for the expression-variance experiment on Ref-YouTube-VOS.

Steps:
  1. Run inference on every (seq, obj, exp) item — saves results.json
  2. Group by (seq_name, obj_id) and compute {min, mean, max, variance}
     per metric — saves grouped_stats.json + summary.json
  3. Render four figures into figures/

Usage
-----
    python run.py \\
        --data_root /home/wahba/git/data/ref-youtube-vos \\
        --save_dir  results/expression_variance \\
        --split     valid \\
        --sample_rate 2 \\
        --seed 0

Subset flags:
    --max_sequences  N        cap on unique sequences
    --expressions_per_seq N   cap on expressions per (seq, obj)
    --min_expressions_per_group N  drop (seq, obj) groups with fewer expressions

Determinism: --seed flows through QwenVOTRunner, which seeds python/numpy/
torch RNGs, sets torch.use_deterministic_algorithms(True), cudnn.deterministic,
CUBLAS_WORKSPACE_CONFIG, and re-seeds before every generate(). Greedy decoding
is the default. test_determinism.py in Ref-DAVIS/benchmark/ verifies bit-exact
runs.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import ExpressionVarianceRYTVOSExperiment

# Reuse the analysis + plotting pipeline from the DAVIS expression-variance
# experiment — the input shape (results.json, one row per (seq, obj, exp)) is
# identical, so the grouping/plotting code is dataset-agnostic.
_HERE        = Path(__file__).resolve().parent
_EXPR_VAR    = _HERE.parent / "expression-variance"
sys.path.insert(0, str(_EXPR_VAR))

from analyze import group_results, summary  # noqa: E402
from plot    import (  # noqa: E402
    figure_1_range, figure_2_best_vs_worst, figure_3_coupling, figure_4_strip,
    _load,
)


def parse_args():
    p = argparse.ArgumentParser("Expression-Variance Experiment (Ref-YouTube-VOS)")
    p.add_argument("--data_root", required=True,
                   help="Root of Ref-YouTube-VOS (contains valid/)")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/expression_variance")
    p.add_argument("--split", default="valid", choices=["valid"])
    p.add_argument("--sample_rate", type=int, default=2,
                   help="Send every Nth annotated frame to the model "
                        "(YouTube-VOS annotates ~6 fps, default 2 → ~3 fps)")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for python/numpy/torch RNGs (deterministic mode)")
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--expressions_per_seq", type=int, default=None,
                   help="None = use all expressions per (seq, obj)")
    p.add_argument("--min_expressions_per_group", type=int, default=2,
                   help="Drop (seq, obj) groups with fewer expressions than this "
                        "before running (they can't contribute to variance)")
    p.add_argument("--video_mode", action="store_true",
                   help="Use 3D RoPE video mode (default: interleaved image mode)")
    p.add_argument("--no_resume", action="store_true",
                   help="Start from scratch even if results.json exists in save_dir")
    p.add_argument("--retry_errors", action="store_true",
                   help="Re-run items previously recorded with an error")
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

        exp = ExpressionVarianceRYTVOSExperiment(
            model=model,
            processor=processor,
            data_root=args.data_root,
            save_dir=str(save_dir),
            video_mode=args.video_mode,
            sample_rate=args.sample_rate,
            max_new_tokens=args.max_new_tokens,
            split=args.split,
            seed=args.seed,
        )
        results = exp.run_all(
            sequences=args.sequences,
            max_sequences=args.max_sequences,
            expressions_per_seq=args.expressions_per_seq,
            min_expressions_per_group=args.min_expressions_per_group,
            resume=not args.no_resume,
            retry_errors=args.retry_errors,
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
        if s is None or s.get("mean_within_group_std") is None:
            print(f"  {m}: (no groups with >=2 expressions)")
            continue
        print(f"  {m}:")
        print(f"    mean within-group std:    {s['mean_within_group_std']:.4f}")
        print(f"    mean within-group range:  {s['mean_within_group_range']:.4f}")
        print(f"    mean best - worst gap:    {s['mean_best_minus_worst']:.4f}")
        print(f"    mean(worst) → mean(best): {s['mean_worst']:.4f} → {s['mean_best']:.4f}")

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    rows = _load(grouped_path)
    if not rows:
        print("[plot] no usable groups — skipping figures")
        return
    figure_1_range         (rows, fig_dir / "fig1_range_per_group.png")
    figure_2_best_vs_worst (rows, fig_dir / "fig2_best_vs_worst.png")
    figure_3_coupling      (rows, fig_dir / "fig3_attention_iou_coupling.png")
    figure_4_strip         (rows, fig_dir / "fig4_strip_per_group.png")
    print(f"\nWrote 4 figures to {fig_dir}")


if __name__ == "__main__":
    main()
