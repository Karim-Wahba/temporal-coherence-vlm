"""
run.py
------
CLI entry point for the Ref-YouTube-VOS grounding-stability experiment.

Usage
-----
    python run.py \\
        --data_root   /home/wahba/git/data/ref-youtube-vos \\
        --model_id    Qwen/Qwen3-VL-8B-Instruct \\
        --save_dir    results/rytvos_valid \\
        --split       valid \\
        --sample_rate 2 \\
        --max_sequences 50

YouTube-VOS annotates every 5th frame of 30 fps video (~6 fps effective).
sample_rate here subsamples the *annotated* frames (default 2 → sends every
other annotated frame to the model, i.e. effectively 3 fps).

Results are written to:
    {save_dir}/results.json          – per-sequence metrics
    {save_dir}/summary.json          – dataset-level aggregates
    {save_dir}/visualizations/*.png  – per-sequence figures
    {save_dir}/correlation_plots.png – pooled scatter plots
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import RefYouTubeVOSGroundingExperiment

# Correlation plots live in grounding-stability's visualizer
sys.path.insert(0, str(Path(__file__).parent.parent / "grounding-stability"))
from visualizer import save_correlation_plots


def parse_args():
    p = argparse.ArgumentParser("Ref-YouTube-VOS Grounding Stability Experiment")
    p.add_argument("--data_root", required=True,
                   help="Root of Ref-YouTube-VOS (contains train/ and valid/)")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/rytvos_valid")
    p.add_argument("--split", default="valid", choices=["train", "valid"])
    p.add_argument("--sample_rate", type=int, default=2,
                   help="Send every Nth annotated frame to the model")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Restrict to these video IDs")
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--expressions_per_seq", type=int, default=1)
    p.add_argument("--video_mode", action="store_true",
                   help="Use 3D RoPE video mode (default: interleaved image mode)")
    p.add_argument("--no_resume", action="store_true",
                   help="Start from scratch even if results.json exists in save_dir")
    p.add_argument("--retry_errors", action="store_true",
                   help="Re-run items previously recorded with an error")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    experiment = RefYouTubeVOSGroundingExperiment(
        model=model,
        processor=processor,
        data_root=args.data_root,
        save_dir=args.save_dir,
        video_mode=args.video_mode,
        sample_rate=args.sample_rate,
        max_new_tokens=args.max_new_tokens,
        split=args.split,
    )

    results = experiment.run_all(
        sequences=args.sequences,
        max_sequences=args.max_sequences,
        expressions_per_seq=args.expressions_per_seq,
        resume=not args.no_resume,
        retry_errors=args.retry_errors,
    )

    summary = RefYouTubeVOSGroundingExperiment.summarize(results)
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
