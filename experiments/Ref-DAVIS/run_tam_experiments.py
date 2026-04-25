"""
run_tam_experiments.py
----------------------
Runs the 5 TAM diagnostic experiments independently of the full benchmark loop.
Useful for deep-diving specific sequences or running ablations.

Usage
-----
    # Experiment 2: Temporal collapse on specific sequences
    python run_tam_experiments.py \
        --davis_root /path/to/davis \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --experiment collapse \
        --sequences blackswan camel \
        --save_dir results/tam_diagnostics

    # Experiment 5: Prompt temporal binding
    python run_tam_experiments.py \
        --davis_root /path/to/davis \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --experiment binding \
        --save_dir results/tam_diagnostics

    # All experiments on all sequences
    python run_tam_experiments.py \
        --davis_root /path/to/davis \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --experiment all \
        --save_dir results/tam_diagnostics
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diagnostics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visualization"))

from benchmark.ref_davis_loader import RefDAVISLoader
from diagnostics.tam_runner import TAMRunner
from diagnostics.tam_analyzer import (
    attention_drift,
    temporal_collapse,
    occlusion_recovery,
    prompt_temporal_binding,
)
from visualization.visualizer import (
    plot_tam_centroids,
    plot_frame_mass_heatmap,
    plot_temporal_binding,
)


# ─── Prompt sets for Experiment 5 ─────────────────────────────────────────────

def _build_binding_prompts(expression: str, T: int) -> dict:
    """Build 4 prompts targeting different temporal windows."""
    early_end = max(1, T // 4)
    late_start = min(T - 1, 3 * T // 4)
    mid_start = T // 4
    mid_end = 3 * T // 4

    prompts = {
        "generic": f"Describe the {expression} in this video.",
        "beginning": f"What is the {expression} doing at the beginning of the video?",
        "middle": f"What is the {expression} doing in the middle of the video?",
        "end": f"What is the {expression} doing at the end of the video?",
    }
    targets = {
        "generic": list(range(T)),
        "beginning": list(range(0, early_end + 1)),
        "middle": list(range(mid_start, mid_end + 1)),
        "end": list(range(late_start, T)),
    }
    return prompts, targets


# ─── Experiment Runners ───────────────────────────────────────────────────────

def run_exp_collapse(tam_runner, item, save_dir, prefix) -> dict:
    """Experiment 2: Temporal Attention Collapse."""
    tam_result = tam_runner.run(
        item.frames_pil,
        expression=f"Track the {item.expression} through all frames of this video.",
        max_new_tokens=128,
    )
    result = temporal_collapse(tam_result)

    plot_frame_mass_heatmap(
        tam_result,
        os.path.join(save_dir, f"{prefix}_frame_mass.png"),
        seq_name=item.seq_name,
        expression=item.expression,
    )
    return {
        "collapse_rate": result["collapse_rate"],
        "mean_entropy": result["mean_entropy"],
        "is_collapsed": result["is_collapsed"],
        "dominant_frame_hist": result["dominant_frame_hist"].tolist(),
    }


def run_exp_drift(tam_runner, item, save_dir, prefix) -> dict:
    """Experiment 1: Attention Drift."""
    H, W = item.frame_size()
    tam_result = tam_runner.run(
        item.frames_pil,
        expression=f"Where is the {item.expression} located in each frame?",
        max_new_tokens=128,
    )
    drift_result = attention_drift(
        tam_result,
        gt_masks=item.masks,
        frame_size=(H, W),
    )
    plot_tam_centroids(
        drift_result,
        item.frames_pil,
        os.path.join(save_dir, f"{prefix}_centroid.png"),
        seq_name=item.seq_name,
    )
    return {
        "mean_drift_error": drift_result.get("mean_drift_error"),
        "velocity_variance": drift_result.get("velocity_variance"),
    }


def run_exp_binding(tam_runner, item, save_dir, prefix) -> dict:
    """Experiment 5: Prompt Temporal Binding."""
    T = item.num_frames
    prompts, targets = _build_binding_prompts(item.expression, T)

    tam_results = {}
    for prompt_name, prompt_text in prompts.items():
        print(f"      Binding prompt: '{prompt_name}'")
        try:
            tam_results[prompt_name] = tam_runner.run(
                item.frames_pil,
                expression=prompt_text,
                max_new_tokens=64,
            )
            # Save frame mass heatmap per prompt
            plot_frame_mass_heatmap(
                tam_results[prompt_name],
                os.path.join(save_dir, f"{prefix}_binding_{prompt_name}.png"),
                seq_name=item.seq_name,
                expression=prompt_text,
            )
        except Exception as e:
            print(f"      Binding prompt '{prompt_name}' failed: {e}")

    if len(tam_results) < 2:
        return {"error": "insufficient prompt results"}

    binding_result = prompt_temporal_binding(tam_results, targets)
    plot_temporal_binding(
        binding_result,
        os.path.join(save_dir, f"{prefix}_binding_scores.png"),
        seq_name=item.seq_name,
    )
    return {
        "binding_scores": binding_result["binding_scores"],
        "mean_binding": binding_result["mean_binding"],
        "is_steerable": binding_result["is_steerable"],
        "binding_std": binding_result["binding_std"],
    }


def run_exp_occlusion(tam_runner, item, save_dir, prefix) -> dict:
    """
    Experiment 4: Occlusion Recovery.
    Auto-detect occlusion frames as frames where GT mask area < 10% of median.
    """
    H, W = item.frame_size()
    masks = item.masks
    areas = np.array([m.sum() for m in masks], dtype=float)
    median_area = np.median(areas[areas > 0]) if (areas > 0).any() else 1
    occ_frames = [t for t, a in enumerate(areas) if a < 0.1 * median_area]

    if not occ_frames:
        return {"skipped": "no occlusion frames detected"}

    tam_result = tam_runner.run(
        item.frames_pil,
        expression=f"Continuously track the {item.expression} even when it's hidden.",
        max_new_tokens=128,
    )
    from diagnostics.tam_analyzer import occlusion_recovery as _occ_rec
    occ_result = _occ_rec(
        tam_result,
        occlusion_frames=occ_frames,
        gt_masks=masks,
        frame_size=(H, W),
    )
    return {
        "occlusion_frames": occ_frames,
        "centroid_jump": occ_result.get("centroid_jump"),
        "frozen_prior_score": occ_result.get("frozen_prior_score"),
        "during_occ_entropy": occ_result.get("during_occ_entropy"),
        "outside_occ_entropy": occ_result.get("outside_occ_entropy"),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("TAM Diagnostic Experiments")
    p.add_argument("--davis_root", default="/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--experiment",
                   choices=["collapse", "drift", "binding", "occlusion", "all"],
                   default="all")
    p.add_argument("--save_dir", default="results/tam_diagnostics")
    p.add_argument("--split", default="valid")
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--max_sequences", type=int, default=10)
    p.add_argument("--tam_submodule_path", default=None)
    p.add_argument("--repeat_frames", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    loader = RefDAVISLoader(
        args.davis_root,
        split=args.split,
        expressions_per_seq=1,
        sequences=args.sequences,
    )
    items = list(loader)
    if args.max_sequences:
        seen = set()
        filtered = []
        for it in items:
            if it.seq_name not in seen:
                if len(seen) >= args.max_sequences:
                    break
                seen.add(it.seq_name)
            filtered.append(it)
        items = filtered

    print(f"Running experiment '{args.experiment}' on {len(items)} items")

    # Load model
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    tam_runner = TAMRunner(
        model, processor,
        tam_submodule_path=args.tam_submodule_path,
        repeat_frames=args.repeat_frames,
    )

    # Run experiments
    all_exp_results = []
    for item_idx, item in enumerate(items):
        print(f"\n[{item_idx+1}/{len(items)}] {item.seq_name} | \"{item.expression[:50]}\"")
        prefix = f"{item.seq_name}_exp{item.exp_id}"
        row = {
            "seq_name": item.seq_name,
            "exp_id": item.exp_id,
            "expression": item.expression,
            "num_frames": item.num_frames,
        }

        try:
            if args.experiment in ("collapse", "all"):
                print("  → Collapse")
                row["collapse"] = run_exp_collapse(tam_runner, item, args.save_dir, prefix)

            if args.experiment in ("drift", "all"):
                print("  → Drift")
                row["drift"] = run_exp_drift(tam_runner, item, args.save_dir, prefix)

            if args.experiment in ("binding", "all"):
                print("  → Binding")
                row["binding"] = run_exp_binding(tam_runner, item, args.save_dir, prefix)

            if args.experiment in ("occlusion", "all"):
                print("  → Occlusion")
                row["occlusion"] = run_exp_occlusion(tam_runner, item, args.save_dir, prefix)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            row["error"] = str(e)

        all_exp_results.append(row)

    # Save results JSON
    out_path = os.path.join(args.save_dir, f"exp_{args.experiment}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_exp_results, f, indent=2, default=str)

    # Summary statistics
    print(f"\n=== Experiment Summary ===")
    if args.experiment in ("collapse", "all"):
        rates = [r["collapse"].get("collapse_rate", 0)
                 for r in all_exp_results if "collapse" in r]
        entropies = [r["collapse"].get("mean_entropy", 0)
                     for r in all_exp_results if "collapse" in r]
        if rates:
            print(f"  Temporal Collapse Rate : {np.mean(rates):.3f} ± {np.std(rates):.3f}")
            print(f"  Mean Temporal Entropy  : {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
            print(f"  Sequences collapsed    : {sum(r['collapse'].get('is_collapsed', False) for r in all_exp_results if 'collapse' in r)}/{len(rates)}")

    if args.experiment in ("drift", "all"):
        errors = [r["drift"].get("mean_drift_error")
                  for r in all_exp_results if "drift" in r]
        errors = [e for e in errors if e is not None]
        if errors:
            print(f"  Mean Drift Error       : {np.mean(errors):.2f} ± {np.std(errors):.2f} px")

    if args.experiment in ("binding", "all"):
        steerability = [r["binding"].get("is_steerable", False)
                        for r in all_exp_results if "binding" in r]
        binding_stds = [r["binding"].get("binding_std", 0)
                        for r in all_exp_results if "binding" in r]
        if steerability:
            print(f"  Steerable sequences    : {sum(steerability)}/{len(steerability)}")
            print(f"  Mean Binding Std       : {np.mean(binding_stds):.3f}")

    print(f"\nResults saved to: {out_path}")
    print(f"Visualizations: {args.save_dir}/")


if __name__ == "__main__":
    main()
