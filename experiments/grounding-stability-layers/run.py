"""
run.py
------
CLI for the layer-ablation TAM experiment.

Usage
-----
  Dry run (1 seq × 1 expr, all 20 variants, full per-seq heatmap grids):
      python run.py --davis_root /path/to/DAVIS --dry_run

  Phase 1 (heatmaps): small set + full grids
      python run.py --davis_root ... --max_sequences 5 --expressions_per_seq 2

  Phase 2 (metric curves only): no per-seq grids
      python run.py --davis_root ... --max_sequences 30 \\
          --expressions_per_seq 4 --no_heatmap_grid

Outputs
-------
  {save_dir}/results.json         per-sequence per-variant metrics
  {save_dir}/summary.json         aggregate per-variant means
  {save_dir}/mass_vs_layer.png    layer ablation curve
  {save_dir}/mass_vs_cumavg.png   cumulative-avg ablation curve
  {save_dir}/visualizations/{seq}_per_layer.png   (when --save_heatmap_grid)
  {save_dir}/visualizations/{seq}_cumavg.png      (when --save_heatmap_grid)
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import GroundingStabilityLayersExperiment
from visualizer import (
    save_layer_curves,
    save_mass_in_gt_layer_comparison,
    save_mass_in_gt_cumavg_comparison,
)


def parse_args():
    p = argparse.ArgumentParser("Grounding Stability Layers Ablation")
    p.add_argument("--davis_root", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/grounding_stability_layers")
    p.add_argument("--split", default="valid", choices=["valid", "train"])
    p.add_argument("--sample_rate", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--expressions_per_seq", type=int, default=1)
    p.add_argument("--image_mode", action="store_true",
                   help="Interleaved image mode instead of video mode")

    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layer indices for per-layer variants (default -1..-10)")
    p.add_argument("--cumavg_ks", nargs="+", type=int, default=None,
                   help="K values for cumulative-avg variants (default 1..10)")
    p.add_argument("--no_norm", action="store_true",
                   help="Skip the final RMSNorm before LM head (debug only)")
    p.add_argument("--no_heatmap_grid", action="store_true",
                   help="Skip per-sequence heatmap-grid PNGs (use for large runs)")

    p.add_argument("--dry_run", action="store_true",
                   help="Restrict to 1 sequence × 1 expression with full grids; "
                        "useful for sanity-checking the pipeline.")
    return p.parse_args()


def main():
    args = parse_args()
    video_mode = not args.image_mode

    if args.dry_run:
        args.max_sequences = 1
        args.expressions_per_seq = 1
        args.no_heatmap_grid = False
        if args.save_dir == "results/grounding_stability_layers":
            args.save_dir = "results/dry_run"

    layer_indices = args.layers or list(range(-1, -11, -1))
    cumavg_Ks = args.cumavg_ks or list(range(1, 11))

    print(f"Layer indices : {layer_indices}")
    print(f"Cumavg Ks     : {cumavg_Ks}")
    print(f"video_mode    : {video_mode}")
    print(f"save_dir      : {args.save_dir}")
    print(f"dry_run       : {args.dry_run}")

    print(f"\nLoading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    experiment = GroundingStabilityLayersExperiment(
        model=model,
        processor=processor,
        davis_root=args.davis_root,
        save_dir=args.save_dir,
        video_mode=video_mode,
        sample_rate=args.sample_rate,
        max_new_tokens=args.max_new_tokens,
        split=args.split,
        layer_indices=layer_indices,
        cumavg_Ks=cumavg_Ks,
        apply_norm=not args.no_norm,
        save_heatmap_grid=not args.no_heatmap_grid,
    )

    results = experiment.run_all(
        sequences=args.sequences,
        max_sequences=args.max_sequences,
        expressions_per_seq=args.expressions_per_seq,
    )

    summary = GroundingStabilityLayersExperiment.summarize(results)
    summary_path = Path(args.save_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    layer_fig, cum_fig = save_layer_curves(
        results, args.save_dir, layer_indices, cumavg_Ks
    )
    layer_cmp = save_mass_in_gt_layer_comparison(
        results, args.save_dir, layer_indices
    )
    cumavg_cmp = save_mass_in_gt_cumavg_comparison(
        results, args.save_dir, cumavg_Ks
    )

    print(f"\nResults       → {args.save_dir}/results.json")
    print(f"Summary       → {summary_path}")
    print(f"Vis dir       → {args.save_dir}/visualizations/")
    if layer_fig:
        print(f"Layer curve   → {layer_fig}")
    if cum_fig:
        print(f"Cumavg curve  → {cum_fig}")
    if layer_cmp:
        print(f"Layer per-seq → {layer_cmp}")
    if cumavg_cmp:
        print(f"Cumavg per-seq→ {cumavg_cmp}")


if __name__ == "__main__":
    main()
