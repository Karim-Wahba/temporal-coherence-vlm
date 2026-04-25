"""
run.py
------
CLI entry point for the grounding-stability experiment.

Usage
-----
    python run.py \\
        --davis_root /path/to/DAVIS \\
        --model_id   Qwen/Qwen3-VL-8B-Instruct \\
        --save_dir   results/grounding_stability \\
        --split      valid \\
        --sample_rate 8 \\
        --max_sequences 10

Results are written to:
    {save_dir}/results.json          – per-sequence metrics
    {save_dir}/summary.json          – dataset-level aggregates
    {save_dir}/visualizations/*.png  – two-row figures per sequence
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import GroundingStabilityExperiment
from visualizer import save_correlation_plots


def parse_args():
    p = argparse.ArgumentParser("Grounding Stability Experiment")
    p.add_argument("--davis_root", required=True,
                   help="Root of DAVIS dataset (must contain Annotations_bbox/)")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/grounding_stability")
    p.add_argument("--split", default="valid", choices=["valid", "train"])
    p.add_argument("--sample_rate", type=int, default=8,
                   help="Send every Nth frame to the model")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Restrict to these sequence names")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Cap on unique sequences to process")
    p.add_argument("--expressions_per_seq", type=int, default=1)
    p.add_argument("--image_mode", action="store_true",
                   help="Use interleaved image mode instead of video mode")
    return p.parse_args()


def main():
    args = parse_args()
    video_mode = not args.image_mode

    print(f"Loading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    experiment = GroundingStabilityExperiment(
        model=model,
        processor=processor,
        davis_root=args.davis_root,
        save_dir=args.save_dir,
        video_mode=video_mode,
        sample_rate=args.sample_rate,
        max_new_tokens=args.max_new_tokens,
        split=args.split,
    )

    results = experiment.run_all(
        sequences=args.sequences,
        max_sequences=args.max_sequences,
        expressions_per_seq=args.expressions_per_seq,
    )

    summary = GroundingStabilityExperiment.summarize(results)

    summary_path = Path(args.save_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    corr_path = save_correlation_plots(results, args.save_dir)

    print(f"\nResults  → {args.save_dir}/results.json")
    print(f"Summary  → {summary_path}")
    print(f"Figures  → {args.save_dir}/visualizations/")
    print(f"Corr.    → {corr_path}")


if __name__ == "__main__":
    main()
