"""
benchmark.py
------------
Main orchestrator for the Ref-DAVIS temporal coherence benchmark.

Usage
-----
    python benchmark.py \
        --davis_root /path/to/davis \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --save_dir results/qwen3vl_8b \
        --split valid \
        --strategy joint \
        --run_tam \
        --expressions_per_seq 1 \
        --max_sequences 30

    # Compare two models
    python benchmark.py \
        --davis_root /path/to/davis \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --save_dir results/qwen3vl_8b \
        --compare_csv results/qwen25vl_7b/metrics.csv

Output layout
-------------
    results/<model>/
    ├── metrics.csv              — per-sequence J, F, J&F + temporal metrics
    ├── failure_analysis.csv     — per-sequence failure mode classification
    ├── summary.json             — aggregate metrics + model config
    ├── plots/
    │   ├── j_curves.png
    │   ├── aggregate_summary.png
    │   └── ...
    ├── failure_cases/           — per-failure-type annotated frames
    └── tam/                     — per-sequence TAM diagnostics (if --run_tam)
        ├── <seq>_frame_mass.png
        └── <seq>_centroid.png
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── Local imports (adjust sys.path as needed) ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "diagnostics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "visualization"))

from benchmark.ref_davis_loader import RefDAVISLoader
from benchmark.metrics import compute_sequence_metrics, aggregate_metrics
from benchmark.qwen_vos_runner import QwenVOSRunner, box_to_mask
from diagnostics.failure_classifier import FailureClassifier
from visualization.visualizer import (
    plot_j_curves,
    plot_tam_centroids,
    plot_frame_mass_heatmap,
    plot_failure_gallery,
    plot_aggregate_summary,
    save_failure_case,
)


def load_model(model_id: str, device: str = "auto"):
    """Load Qwen model and processor."""
    print(f"Loading model: {model_id}")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.")
    return model, processor


def run_tam_diagnostics(
    model, processor, item, pred_masks, save_dir_tam: str,
    tam_submodule_path: Optional[str] = None,
) -> dict:
    """Run TAM diagnostics for one sequence item."""
    from diagnostics.tam_runner import TAMRunner
    from diagnostics.tam_analyzer import attention_drift, temporal_collapse

    tam_runner = TAMRunner(model, processor, tam_submodule_path=tam_submodule_path)

    # Use a simple describe-the-object prompt for TAM (not bbox prompt)
    tam_prompt = f"Describe where '{item.expression}' is located in each frame of this video."

    try:
        tam_result = tam_runner.run(
            frames_pil=item.frames_pil,
            expression=tam_prompt,
            max_new_tokens=128,
        )
    except Exception as e:
        print(f"    TAM failed: {e}")
        return {}

    H, W = item.frame_size()
    drift_result = attention_drift(
        tam_result,
        gt_masks=item.masks,
        frame_size=(H, W),
    )
    collapse_result = temporal_collapse(tam_result)

    # Save TAM plots
    seq_prefix = f"{item.seq_name}_exp{item.exp_id}"
    plot_frame_mass_heatmap(
        tam_result,
        os.path.join(save_dir_tam, f"{seq_prefix}_frame_mass.png"),
        seq_name=item.seq_name,
        expression=item.expression,
    )
    plot_tam_centroids(
        drift_result,
        item.frames_pil,
        os.path.join(save_dir_tam, f"{seq_prefix}_centroid.png"),
        seq_name=item.seq_name,
    )

    return {
        "tam_result": tam_result,
        "drift": drift_result,
        "collapse": collapse_result,
    }


def main():
    p = argparse.ArgumentParser("Ref-DAVIS Temporal Coherence Benchmark")
    p.add_argument("--davis_root", default="/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised", help="Root of DAVIS dataset")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir", default="results/benchmark")
    p.add_argument("--split", default="valid", choices=["train", "valid"])
    p.add_argument("--strategy", default="joint", choices=["joint", "per_frame"],
                   help="VOS inference strategy")
    p.add_argument("--run_tam", action="store_true",
                   help="Also run TAM diagnostics (slow, adds ~3x time per seq)")
    p.add_argument("--expressions_per_seq", type=int, default=1,
                   help="Number of expressions per sequence (1=fastest, 4=full)")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="Cap number of sequences (for debugging)")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--sample_rate", type=int, default=2,
                   help="Send every Nth frame to model in joint mode (1=all frames, tends to collapse)")
    p.add_argument("--tam_submodule_path", default=None,
                   help="Path to TAM submodule directory")
    p.add_argument("--compare_csv", default=None,
                   help="CSV from another model run for comparison table")
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Run only specific sequences (for debugging)")
    p.add_argument("--checkpoint_json", default=None,
                   help="Path to existing results JSON to resume from")
    args = p.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "failure_cases"), exist_ok=True)
    if args.run_tam:
        os.makedirs(os.path.join(args.save_dir, "tam"), exist_ok=True)

    print(f"=== Ref-DAVIS Temporal Coherence Benchmark ===")
    print(f"  Model   : {args.model_id}")
    print(f"  Dataset : {args.davis_root} ({args.split})")
    print(f"  Save    : {args.save_dir}")
    print(f"  TAM     : {args.run_tam}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    seq_filter = args.sequences
    loader = RefDAVISLoader(
        args.davis_root,
        split=args.split,
        expressions_per_seq=args.expressions_per_seq,
        sequences=seq_filter,
    )
    print(f"\nDataset: {len(loader)} items across {len(loader.sequence_names())} sequences")

    items = list(loader)
    if args.max_sequences:
        # Take first max_sequences unique sequences
        seen = set()
        filtered = []
        for it in items:
            if it.seq_name not in seen:
                seen.add(it.seq_name)
                if len(seen) > args.max_sequences:
                    break
            filtered.append(it)
        items = filtered

    # ── Load checkpoint if resuming ───────────────────────────────────────────
    completed_keys = set()
    all_results = []
    if args.checkpoint_json and os.path.exists(args.checkpoint_json):
        with open(args.checkpoint_json) as f:
            checkpoint = json.load(f)
        all_results = checkpoint.get("results", [])
        for r in all_results:
            completed_keys.add(f"{r['seq_name']}_{r['exp_id']}")
        print(f"Resuming: {len(completed_keys)} sequences already done")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model_id)
    runner = QwenVOSRunner(
        model, processor,
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        sample_rate=args.sample_rate,
    )
    classifier = FailureClassifier()

    # ── Main evaluation loop ──────────────────────────────────────────────────
    t_start = time.time()
    for item_idx, item in enumerate(items):
        key = f"{item.seq_name}_{item.exp_id}"
        if key in completed_keys:
            print(f"  [{item_idx+1}/{len(items)}] SKIP (done) {item.seq_name} exp={item.exp_id}")
            continue

        print(f"\n[{item_idx+1}/{len(items)}] {item.seq_name} exp={item.exp_id}: "
              f"\"{item.expression[:60]}\"  ({item.num_frames} frames)")

        try:
            H, W = item.frame_size()
            frames = item.frames_pil
            gt_masks = item.masks

            # ── Run VOS inference ─────────────────────────────────────────
            t0 = time.time()
            boxes, pred_masks = runner.run_and_get_masks(
                frames, item.expression, H, W
            )
            print(f"    Inference: {time.time() - t0:.1f}s  "
                  f"parsed {sum(b is not None for b in boxes)}/{len(boxes)} boxes")

            # ── Compute metrics ───────────────────────────────────────────
            metrics = compute_sequence_metrics(pred_masks, gt_masks)
            print(f"    J={metrics['mean_J']:.3f}  F={metrics['mean_F']:.3f}  "
                  f"J&F={metrics['JF']:.3f}  J-decay={metrics['J_decay']:.3f}")

            # ── TAM diagnostics ───────────────────────────────────────────
            tam_diagnostics = {}
            if args.run_tam:
                print("    Running TAM...")
                tam_diagnostics = run_tam_diagnostics(
                    model, processor, item, pred_masks,
                    save_dir_tam=os.path.join(args.save_dir, "tam"),
                    tam_submodule_path=args.tam_submodule_path,
                )

            # ── Classify failure ──────────────────────────────────────────
            failure = classifier.classify(
                seq_name=item.seq_name,
                exp_id=item.exp_id,
                expression=item.expression,
                metrics=metrics,
                tam_collapse=tam_diagnostics.get("collapse"),
                tam_drift=tam_diagnostics.get("drift"),
            )
            print(f"    Failure: {failure.primary_failure.value}  "
                  f"flags={[f.value for f in failure.secondary_flags]}")

            # ── Collect result ────────────────────────────────────────────
            result = {
                "seq_name": item.seq_name,
                "exp_id": item.exp_id,
                "expression": item.expression,
                "obj_id": item.obj_id,
                "num_frames": item.num_frames,
                "metrics": metrics,
                "failure": failure.to_dict(),
                "frames_pil": frames,
                "gt_masks": gt_masks,
                "pred_masks": pred_masks,
                "boxes": [list(b) if b else None for b in boxes],
            }
            all_results.append(result)

            # ── Save failure case visualization ───────────────────────────
            tam_result_for_vis = tam_diagnostics.get("tam_result") if args.run_tam else None
            save_failure_case(
                result,
                save_dir=os.path.join(args.save_dir, "failure_cases"),
                tam_result=tam_result_for_vis,
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            continue

        # ── Checkpoint ────────────────────────────────────────────────────
        checkpoint_path = os.path.join(args.save_dir, "checkpoint.json")
        serializable = [
            {k: v for k, v in r.items()
             if k not in ("frames_pil", "gt_masks", "pred_masks")}
            for r in all_results
        ]
        with open(checkpoint_path, "w") as f:
            json.dump({"results": serializable}, f, indent=2)

    print(f"\n=== Evaluation complete: {len(all_results)} sequences in "
          f"{time.time() - t_start:.0f}s ===")

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    metric_dicts = [r["metrics"] for r in all_results if "metrics" in r]
    failure_dicts = [r["failure"] for r in all_results if "failure" in r]

    agg = aggregate_metrics(metric_dicts)
    # Add collapse_rate to agg if TAM was run
    if args.run_tam:
        collapse_rates = [r.get("failure", {}).get("collapse_rate", 0)
                         for r in all_results]
        agg["collapse_rate"] = float(np.mean(collapse_rates)) if collapse_rates else 0.0

    failure_summary = classifier.summarize(
        [type('FR', (), r["failure"])() if False else
         type('FR', (), {**r["failure"],
                          "primary_failure": type('FM', (), {"value": r["failure"]["primary_failure"]})(),
                          "secondary_flags": []})()
         for r in all_results if "failure" in r]
    )

    # Simpler: just count directly
    failure_counts = {}
    for r in all_results:
        mode = r.get("failure", {}).get("primary_failure", "UNKNOWN")
        failure_counts[mode] = failure_counts.get(mode, 0) + 1
    total = len(all_results)
    failure_summary = {
        "total": total,
        "primary_distribution": {
            k: {"count": v, "pct": 100 * v / total}
            for k, v in failure_counts.items()
        },
        "success_rate": 100 * failure_counts.get("SUCCESS", 0) / total if total else 0,
    }

    print(f"\n=== Aggregate Results ===")
    print(f"  J&F          : {agg['JF']:.4f}")
    print(f"  Mean J       : {agg['mean_J']:.4f}")
    print(f"  Mean F       : {agg['mean_F']:.4f}")
    print(f"  J-Decay      : {agg['J_decay']:.4f}  (neg = losing track over time)")
    print(f"  J-Variance   : {agg['J_variance']:.4f}")
    print(f"  Success@0.5  : {agg['success_rate_50']:.4f}")
    print(f"\n=== Failure Mode Distribution ===")
    for mode, info in sorted(failure_summary["primary_distribution"].items(),
                              key=lambda x: -x[1]["count"]):
        print(f"  {mode:<22}: {info['count']:>3}  ({info['pct']:.1f}%)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        m = r.get("metrics", {})
        f = r.get("failure", {})
        rows.append({
            "seq_name": r["seq_name"],
            "exp_id": r["exp_id"],
            "expression": r["expression"],
            "num_frames": r["num_frames"],
            "mean_J": m.get("mean_J"),
            "mean_F": m.get("mean_F"),
            "JF": m.get("JF"),
            "J_decay": m.get("J_decay"),
            "J_variance": m.get("J_variance"),
            "J_first": m.get("J_first"),
            "J_last": m.get("J_last"),
            "success_rate_50": m.get("success_rate_50"),
            "success_rate_75": m.get("success_rate_75"),
            "primary_failure": f.get("primary_failure"),
            "secondary_flags": "|".join(f.get("secondary_flags", [])),
            "collapse_rate": f.get("collapse_rate"),
            "mean_drift_error": f.get("mean_drift_error"),
            "notes": f.get("notes"),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model_id": args.model_id,
        "split": args.split,
        "strategy": args.strategy,
        "expressions_per_seq": args.expressions_per_seq,
        "num_sequences": len(all_results),
        "aggregate_metrics": agg,
        "failure_summary": failure_summary,
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots...")
    plots_dir = os.path.join(args.save_dir, "plots")

    plot_j_curves(
        all_results,
        os.path.join(plots_dir, "j_curves.png"),
        title=f"Per-Frame J over Time — {args.model_id}",
    )
    plot_failure_gallery(
        all_results,
        os.path.join(plots_dir, "failure_gallery.png"),
    )
    plot_aggregate_summary(
        agg,
        failure_summary,
        os.path.join(plots_dir, "aggregate_summary.png"),
        model_name=args.model_id,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    if args.compare_csv:
        _print_comparison_table(csv_path, args.compare_csv, args.model_id)

    print(f"\nAll outputs in: {args.save_dir}/")


def _print_comparison_table(csv_a: str, csv_b: str, model_a_name: str):
    """Print a side-by-side comparison of two model CSVs."""
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    metrics = ["mean_J", "mean_F", "JF", "J_decay", "J_variance",
               "success_rate_50", "success_rate_75"]

    print(f"\n{'Metric':<22} {'Model A':>12} {'Model B':>12} {'Delta':>10}")
    print("-" * 60)
    for m in metrics:
        va = df_a[m].mean() if m in df_a else float("nan")
        vb = df_b[m].mean() if m in df_b else float("nan")
        delta = va - vb
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {m:<20} {va:>12.4f} {vb:>12.4f} {delta:>+9.4f} {arrow}")

    print(f"\n  Model A: {model_a_name}")
    print(f"  Model B: {csv_b}")


if __name__ == "__main__":
    main()
