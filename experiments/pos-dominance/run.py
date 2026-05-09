"""
run.py
------
CLI entry point for the POS dominance experiment.

Usage
-----
    # Full run (236 DAVIS valid sequences):
    python run.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --model_id   Qwen/Qwen3-VL-8B-Instruct \\
        --save_dir   results/pos_dominance

    # Quick layout test (no GPU, synthetic TAM maps):
    python run.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --dry_run  --max_sequences 10

    # Restrict to specific sequences:
    python run.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --sequences  breakdance bike-packing

Outputs
-------
    {save_dir}/
        results.json          per-sequence oracle-winner records (incremental)
        summary.json          dataset-level aggregates
        temporal_profile.png  primary plot: category dominance over normalised time
        category_summary.png  overall win rate + mean GT mass per category
        dominance_heatmap.png sequences × bins grid coloured by dominant category
        category_race.png     mean GT mass per category over normalised time
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure our local modules shadow same-named files in sibling experiments
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiment import POSDominanceExperiment, _DryRunRunner
from pos_tagger import tagger_info
from visualizer import (
    save_temporal_profile,
    save_category_summary,
    save_dominance_heatmap,
    save_category_race,
    save_pairwise_matrix,
    save_adjacency_matrix,
    save_category_position_profile,
)


def parse_args():
    p = argparse.ArgumentParser("POS Dominance Experiment")
    p.add_argument("--davis_root", required=True,
                   help="Root of DAVIS dataset (must contain Annotations_bbox/)")
    p.add_argument("--model_id",     default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir",     default="results/pos_dominance")
    p.add_argument("--split",        default="valid", choices=["valid", "train"])
    p.add_argument("--sample_rate",  type=int, default=8,
                   help="Send every Nth frame to the model")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sequences",    nargs="*", default=None,
                   help="Restrict to these sequence names")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Cap on number of unique sequences")
    p.add_argument("--expressions_per_seq", type=int, default=1)
    p.add_argument("--image_mode",   action="store_true",
                   help="Interleaved image mode (default: video mode)")
    p.add_argument("--dry_run",      action="store_true",
                   help="Skip model load, use synthetic TAM maps (layout test)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print("Dry-run mode: using synthetic TAM maps (no GPU required).")
        runner = _DryRunRunner(sample_rate=args.sample_rate)
    else:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        # benchmark.qwen_vot_runner is on sys.path after experiment import
        from benchmark.qwen_vot_runner import QwenVOTRunner  # noqa: E402

        print(f"Loading model: {args.model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        model.eval()

        runner = QwenVOTRunner(
            model, processor,
            max_new_tokens=args.max_new_tokens,
            sample_rate=args.sample_rate,
            video_mode=not args.image_mode,
        )

    experiment = POSDominanceExperiment(
        runner=runner,
        davis_root=args.davis_root,
        save_dir=args.save_dir,
        split=args.split,
    )

    results = experiment.run_all(
        sequences=args.sequences,
        max_sequences=args.max_sequences,
        expressions_per_seq=args.expressions_per_seq,
    )

    summary    = POSDominanceExperiment.summarize(results)
    pairwise   = POSDominanceExperiment.compute_pairwise(results)
    top2_pairs = POSDominanceExperiment.compute_top2_pairs(results)
    adj_pairs  = POSDominanceExperiment.compute_adjacent_pairwise(results)
    cat_profile = POSDominanceExperiment.compute_category_position_profile(results)
    print(f"\nPOS tagger: {tagger_info()}")

    summary["pairwise"]                  = pairwise
    summary["top2_pairs"]                = top2_pairs
    summary["adjacent_pairwise"]         = adj_pairs
    summary["category_position_profile"] = cat_profile

    summary_path = Path(args.save_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Overall dominance ===")
    for cat, v in summary["overall_dominance"].items():
        print(f"  {cat:<20s}  {v['fraction']:5.1%}  "
              f"mean_mass={v['mean_gt_mass']:.4f}  n={v['count']}")

    print("\n=== Top-5 pairwise combined scores  (best-per-category, unordered) ===")
    sorted_pw = sorted(pairwise.items(),
                       key=lambda kv: kv[1]["mean_combined"], reverse=True)
    for key, v in sorted_pw[:5]:
        diag = key.split("|")[0] == key.split("|")[1]
        tag  = " [single]" if diag else ""
        print(f"  {key:<40s}  {v['mean_combined']:.4f}  n={v['n_frames']}{tag}")

    print("\n=== Top-5 adjacency-based combined scores  (cat_p → cat_p+1, ordered) ===")
    sorted_adj = sorted(adj_pairs.items(),
                        key=lambda kv: kv[1]["mean_combined"], reverse=True)
    for key, v in sorted_adj[:5]:
        diag = key.split("|")[0] == key.split("|")[1]
        tag  = " [run]" if diag else ""
        print(f"  {key:<40s}  {v['mean_combined']:.4f}  n_pairs={v['n_pairs']}{tag}")

    print("\nGenerating plots…")
    save_temporal_profile(summary,        args.save_dir)
    save_category_summary(summary,        args.save_dir)
    save_dominance_heatmap(results,       args.save_dir)
    save_category_race(results,           args.save_dir)
    save_pairwise_matrix(pairwise, top2_pairs, args.save_dir)
    save_adjacency_matrix(adj_pairs,      args.save_dir)
    save_category_position_profile(cat_profile, args.save_dir)

    print(f"\nResults  → {args.save_dir}/results.json")
    print(f"Summary  → {summary_path}")


if __name__ == "__main__":
    main()
