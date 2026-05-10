"""
run.py
------
CLI entry point for the RefCOCO grounding experiment.

Usage
-----
    python run.py \\
        --refcoco_root /home/wahba/git/data/refcoco \\
        --dataset      refcoco \\
        --split        val \\
        --model_id     Qwen/Qwen3-VL-8B-Instruct \\
        --save_dir     results/refcoco_val \\
        --max_items    500 \\
        --sents_per_ref 1

Results are written to:
    {save_dir}/results.json      – per-item metrics
    {save_dir}/summary.json      – dataset-level aggregates
    {save_dir}/scatter_plots.png – correlation scatter plots
    {save_dir}/visualizations/   – per-item figures (every --vis_every items)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import RefCOCOGroundingExperiment
from visualizer import save_scatter_plots


def parse_args():
    p = argparse.ArgumentParser("RefCOCO Grounding Experiment")
    p.add_argument("--refcoco_root", required=True,
                   help="Root of RefCOCO data (must contain images/ and annotations/)")
    p.add_argument("--dataset", default="refcoco",
                   choices=["refcoco", "refcoco+", "refcocog"])
    p.add_argument("--split", default="val",
                   choices=["train", "val", "testA", "testB", "test"])
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/refcoco_val")
    p.add_argument("--max_items", type=int, default=None,
                   help="Cap on total items to process")
    p.add_argument("--sents_per_ref", type=int, default=1,
                   help="Max expressions per ref object (1 = first only)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--vis_every", type=int, default=50,
                   help="Save a visualisation every N items (0 = never)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    experiment = RefCOCOGroundingExperiment(
        model=model,
        processor=processor,
        refcoco_root=args.refcoco_root,
        save_dir=args.save_dir,
        dataset=args.dataset,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        vis_every=args.vis_every,
    )

    results = experiment.run_all(
        refcoco_root=args.refcoco_root,
        max_items=args.max_items,
        sents_per_ref=args.sents_per_ref,
    )

    summary = RefCOCOGroundingExperiment.summarize(results)

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

    scatter_path = save_scatter_plots(results, args.save_dir)

    print(f"\nResults  → {args.save_dir}/results.json")
    print(f"Summary  → {summary_path}")
    print(f"Figures  → {args.save_dir}/visualizations/")
    print(f"Scatter  → {scatter_path}")


if __name__ == "__main__":
    main()
