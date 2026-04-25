"""
run_ablation.py
---------------
Token-selection ablation on the DAVIS 'breakdance' sequence.

Runs a single forward pass per expression, then evaluates 13 token-selection
strategies against each other by measuring how much GT-box attention mass
each strategy's heatmap places inside the GT bounding box.

Usage
-----
    python run_ablation.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --model_id   Qwen/Qwen3-VL-8B-Instruct \\
        --save_dir   results/token_ablation \\
        --sample_rate 8 \\
        [--sequence breakdance] \\
        [--expressions_per_seq 4] \\
        [--dry_run]   # skip model load, use random TAM maps for layout testing

Outputs
-------
    {save_dir}/
        results.json                   per-expression per-strategy scores
        summary.json                   aggregated mean ± std per strategy
        plots/
            bar_chart.png              strategy comparison bar chart
            per_frame_curves.png       GT mass vs frame index per strategy
            token_spotlight.png        top tokens by mean GT mass
            heatmaps/<seq>_<exp>_<t>.png  attention grids for sampled frames
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
_REF_DAVIS = _EXPERIMENTS / "Ref-DAVIS"
_GS = _EXPERIMENTS / "grounding-stability"

sys.path.insert(0, str(_REF_DAVIS))
sys.path.insert(0, str(_REF_DAVIS / "benchmark"))
sys.path.insert(0, str(_REF_DAVIS / "diagnostics"))
sys.path.insert(0, str(_GS))           # token_parser, metrics (grounding-stability versions)
sys.path.insert(0, str(_HERE))         # strategies, visualize

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem
from benchmark.qwen_vot_runner import QwenVOTRunner

# grounding-stability token_parser and metrics
from token_parser import parse_frame_labels, find_label_token_indices
from strategies import (
    STRATEGIES,
    StrategyContext,
    _compute_mass_in_gt_fast,
    analyze_oracle_tokens,
)
from visualize import (
    plot_strategy_bar,
    plot_heatmap_grid,
    plot_per_frame_curves,
    plot_token_spotlight,
    plot_oracle_token_analysis,
)
from categorize import apply_categories, plot_category_breakdown, print_summary


# ── per-token scoring ─────────────────────────────────────────────────────────

def score_all_tokens(
    tam_maps,
    gt_boxes: list,
    vision_T: int,
    frame_H: int,
    frame_W: int,
    sample_rate: int,
) -> List[Tuple[int, float]]:
    """
    For every token, compute its mean GT mass across all sampled frames
    that have a valid GT box. Returns [(tok_idx, mean_gt_mass), ...].
    """
    scores = []
    for i, tm in enumerate(tam_maps):
        if tm is None or tm.ndim != 3:
            continue
        masses = []
        for sampled_t in range(min(vision_T, tm.shape[0])):
            orig_t = sampled_t * sample_rate
            if orig_t >= len(gt_boxes) or gt_boxes[orig_t] is None:
                continue
            hm = tm[sampled_t].astype(np.float32)
            masses.append(_compute_mass_in_gt_fast(hm, gt_boxes[orig_t], frame_H, frame_W))
        if masses:
            scores.append((i, float(np.mean(masses))))
    return scores


# ── run one expression ────────────────────────────────────────────────────────

def run_expression(
    item: DAVISVOTItem,
    runner: QwenVOTRunner,
    strategy_names: List[str],
    save_dir: Path,
    dry_run: bool = False,
) -> dict:
    prefix = f"{item.seq_name}_exp{item.exp_id}"
    print(f"  [{prefix}] \"{item.expression[:70]}\"")

    H, W = item.frame_size()

    if dry_run:
        # Synthetic TAM maps for layout testing
        T_fake = max(1, item.num_frames // runner.sample_rate)
        H_tam, W_tam = 14, 14
        n_tok = 40
        gen_tokens = [f"tok{i}" for i in range(n_tok)]
        tam_maps = [np.random.rand(T_fake, H_tam, W_tam).astype(np.float32)
                    for _ in range(n_tok)]
        gen_text = json.dumps([
            {"frame": t, "bbox_2d": [100, 100, 400, 400], "label": "breakdancer"}
            for t in range(T_fake)
        ])
        boxes = item.gt_boxes[::runner.sample_rate] + [None] * T_fake
        boxes = boxes[:item.num_frames]
        vision_T = T_fake
    else:
        try:
            boxes, gen_text, tam_result = runner.run_with_tam(
                item.frames_pil, item.expression
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            return {"seq_name": item.seq_name, "exp_id": item.exp_id,
                    "expression": item.expression, "error": str(e)}

        gen_tokens  = tam_result["gen_tokens"]
        tam_maps    = tam_result["tam_maps"]
        vision_T    = tam_result["vision_shape"][0]

    # ── build label_token_map ─────────────────────────────────────────────
    fps = runner.fps
    parsed = parse_frame_labels(gen_text, fps=fps, sample_rate=runner.sample_rate)
    label_token_map = find_label_token_indices(gen_tokens, parsed)

    ctx = StrategyContext(
        gen_tokens      = gen_tokens,
        tam_maps        = tam_maps,
        vision_T        = vision_T,
        label_token_map = label_token_map,
        gen_text        = gen_text,
        gt_boxes        = item.gt_boxes,
        frame_H         = H,
        frame_W         = W,
        sample_rate     = runner.sample_rate,
        expression      = item.expression,
    )

    # ── apply every strategy ──────────────────────────────────────────────
    strategy_masses:       Dict[str, List[float]] = {}
    strategy_frame_masses: Dict[str, Dict[str, float]] = {}

    for sname in strategy_names:
        fn = STRATEGIES[sname]
        try:
            frame_heatmaps = fn(ctx)
        except Exception as e:
            print(f"    [WARN] strategy {sname} failed: {e}")
            strategy_masses[sname] = []
            strategy_frame_masses[sname] = {}
            continue

        masses: List[float] = []
        per_frame: Dict[str, float] = {}
        for sampled_t, hm in frame_heatmaps.items():
            orig_t = sampled_t * runner.sample_rate
            if orig_t >= len(item.gt_boxes) or item.gt_boxes[orig_t] is None:
                continue
            mass = _compute_mass_in_gt_fast(hm, item.gt_boxes[orig_t], H, W)
            masses.append(mass)
            per_frame[str(sampled_t)] = mass

        strategy_masses[sname]       = masses
        strategy_frame_masses[sname] = per_frame

    # ── per-token scores for spotlight ────────────────────────────────────
    token_scores = score_all_tokens(
        tam_maps, item.gt_boxes, vision_T, H, W, runner.sample_rate
    )
    token_score_list = [
        {"tok_idx": i, "token": gen_tokens[i], "mean_gt_mass": s}
        for i, s in sorted(token_scores, key=lambda x: -x[1])
    ]

    # ── oracle token analysis ─────────────────────────────────────────────
    oracle_rows = analyze_oracle_tokens(ctx)
    oracle_winners = [r for r in oracle_rows if r["rank"] == 1]
    print(f"    oracle winners per frame:")
    for row in oracle_winners:
        print(f"      t={row['orig_t']:3d}  tok={row['tok_idx']:3d}"
              f"  {row['token_clean']!r:<20s}  gt_mass={row['gt_mass']:.4f}")

    # ── heatmap grid for a representative frame ───────────────────────────
    _save_heatmap_grids(
        item, ctx, strategy_names, save_dir / "plots" / "heatmaps",
        prefix, runner.sample_rate
    )

    # Print summary
    print(f"    gen_text snippet: {gen_text[:120]!r}")
    print(f"    detected frames: {[t for t, _ in label_token_map]}")
    for sname in strategy_names:
        ms = strategy_masses.get(sname, [])
        mean = float(np.mean(ms)) if ms else float("nan")
        print(f"    {sname:<28s} mean_GT_mass={mean:.4f}  n={len(ms)}")

    return {
        "seq_name":             item.seq_name,
        "exp_id":               item.exp_id,
        "expression":           item.expression,
        "num_frames":           item.num_frames,
        "vision_T":             vision_T,
        "gen_text":             gen_text,
        "strategy_masses":      strategy_masses,
        "strategy_frame_masses": strategy_frame_masses,
        "token_scores":         token_score_list[:50],
        "oracle_rows":          oracle_rows,      # full per-frame ranked list
        "oracle_winners":       oracle_winners,   # rank-1 token per frame
    }


def _save_heatmap_grids(
    item: DAVISVOTItem,
    ctx: StrategyContext,
    strategy_names: List[str],
    out_dir: Path,
    prefix: str,
    sample_rate: int,
):
    """Save a heatmap-grid figure for each sampled frame that has a GT box."""
    for sampled_t in range(ctx.vision_T):
        orig_t = sampled_t * sample_rate
        if orig_t >= len(item.gt_boxes) or item.gt_boxes[orig_t] is None:
            continue
        gt_box = item.gt_boxes[orig_t]
        frame_pil = item.frames_pil[min(orig_t, len(item.frames_pil) - 1)]

        strat_hms: Dict[str, Optional[np.ndarray]] = {}
        for sname in strategy_names:
            fn = STRATEGIES[sname]
            try:
                hms = fn(ctx)
                strat_hms[sname] = hms.get(sampled_t)
            except Exception:
                strat_hms[sname] = None

        save_path = str(out_dir / f"{prefix}_t{orig_t:04d}.png")
        plot_heatmap_grid(
            frame_pil=frame_pil,
            gt_box=gt_box,
            strategy_heatmaps=strat_hms,
            save_path=save_path,
            frame_label=f"{prefix}  t={orig_t}",
        )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Token Selection Ablation")
    p.add_argument("--davis_root",
                   default="/mnt/lustre/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised")
    p.add_argument("--model_id",     default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir",     default="results/token_ablation")
    p.add_argument("--split",        default="valid")
    p.add_argument("--sequence",     default=None,
                   help="Restrict to one sequence name (default: all sequences)")
    p.add_argument("--expressions_per_seq", type=int, default=4)
    p.add_argument("--sample_rate",  type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--image_mode",   action="store_true",
                   help="Interleaved image mode (default: video mode)")
    p.add_argument("--dry_run",      action="store_true",
                   help="Skip model load, use synthetic TAM maps (layout test)")
    p.add_argument("--strategies",   nargs="*", default=None,
                   help="Subset of strategies to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    strategy_names = args.strategies or list(STRATEGIES.keys())
    print(f"Strategies ({len(strategy_names)}): {strategy_names}")

    # ── load data ─────────────────────────────────────────────────────────
    seq_label = args.sequence or "all sequences"
    loader = DAVISVOTLoader(
        davis_root=args.davis_root,
        split=args.split,
        sequences=[args.sequence] if args.sequence else None,
        expressions_per_seq=args.expressions_per_seq,
    )
    items = list(loader)
    print(f"Loaded {len(items)} expression(s) for '{seq_label}'")
    if not items:
        raise SystemExit(f"No items found for '{seq_label}'")

    # ── load model ────────────────────────────────────────────────────────
    if args.dry_run:
        model = processor = None
        runner = _DryRunRunner(sample_rate=args.sample_rate)
    else:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
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

    # ── run ablation ──────────────────────────────────────────────────────
    all_results = []
    for idx, item in enumerate(items):
        print(f"\n[{idx+1}/{len(items)}] {item.seq_name}")
        result = run_expression(
            item, runner, strategy_names, save_dir, dry_run=args.dry_run
        )
        all_results.append(result)

        # incremental save
        with open(save_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── aggregate summary ─────────────────────────────────────────────────
    valid = [r for r in all_results if "error" not in r]
    summary = {"sequence": args.sequence or "all", "n_expressions": len(valid), "strategies": {}}
    for sname in strategy_names:
        all_masses = []
        for r in valid:
            all_masses.extend(r.get("strategy_masses", {}).get(sname, []))
        per_expr_means = [
            float(np.mean(r.get("strategy_masses", {}).get(sname, [0.0])))
            for r in valid if r.get("strategy_masses", {}).get(sname)
        ]
        summary["strategies"][sname] = {
            "mean_gt_mass":  float(np.mean(all_masses))       if all_masses else None,
            "std_gt_mass":   float(np.std(all_masses))        if all_masses else None,
            "per_expr_mean": float(np.mean(per_expr_means))   if per_expr_means else None,
            "per_expr_std":  float(np.std(per_expr_means))    if per_expr_means else None,
            "n_frames":      len(all_masses),
        }

    print("\n=== Summary ===")
    ranked = sorted(
        summary["strategies"].items(),
        key=lambda kv: kv[1]["mean_gt_mass"] or 0.0,
        reverse=True,
    )
    for sname, stats in ranked:
        m = stats["mean_gt_mass"]
        s = stats["per_expr_std"]
        print(f"  {sname:<28s}  mean={m:.4f}  expr_std={s:.4f}" if m is not None else f"  {sname}  N/A")

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── plots ─────────────────────────────────────────────────────────────
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_strategy_bar(valid, strategy_names, str(plots_dir / "bar_chart.png"),
                      title=f"Token Selection Ablation — {seq_label}")
    plot_per_frame_curves(valid, strategy_names, str(plots_dir / "per_frame_curves.png"),
                          title=f"Per-frame GT mass by strategy — {seq_label}")

    # token spotlight: pool top tokens across all expressions
    tok_pool: Dict[str, List[float]] = {}
    for r in valid:
        for entry in r.get("token_scores", []):
            tok = entry["token"]
            tok_pool.setdefault(tok, []).append(entry["mean_gt_mass"])
    token_scores_agg = [
        (tok, float(np.mean(vals))) for tok, vals in tok_pool.items()
    ]
    plot_token_spotlight(token_scores_agg, str(plots_dir / "token_spotlight.png"), top_k=25)

    # pool all oracle rows across expressions, apply categories
    all_oracle_rows = []
    for r in valid:
        rows = r.get("oracle_rows", [])
        # reconstruct sparse gen_tokens list from token_scores index map
        idx_to_tok = {e["tok_idx"]: e["token"] for e in r.get("token_scores", [])}
        max_idx = max((row["tok_idx"] for row in rows), default=0)
        gen_tokens_sparse = [idx_to_tok.get(i, "") for i in range(max_idx + 1)]
        for row in rows:
            row["exp_id"] = r["exp_id"]
        apply_categories(rows, r["expression"], gen_tokens_sparse)
        all_oracle_rows.extend(rows)

    if all_oracle_rows:
        plot_oracle_token_analysis(
            all_oracle_rows,
            str(plots_dir / "oracle_token_analysis.png"),
        )
        plot_category_breakdown(
            all_oracle_rows,
            str(plots_dir / "category_breakdown.png"),
        )
        print_summary(all_oracle_rows)

    print(f"\nResults  → {save_dir}/results.json")
    print(f"Summary  → {save_dir}/summary.json")
    print(f"Plots    → {plots_dir}/")


# ── dry-run stub ──────────────────────────────────────────────────────────────

class _DryRunRunner:
    """Minimal runner stub used when --dry_run is set."""
    def __init__(self, sample_rate: int = 8):
        self.sample_rate = sample_rate
        self.fps = 24.0
        self.video_mode = True

    def run_with_tam(self, frames_pil, expression):
        import json
        T = max(1, len(frames_pil) // self.sample_rate)
        H_tam, W_tam = 14, 14
        n_tok = 50
        gen_tokens = (
            ["[", "{", '"frame"', ":", " 0", ",", '"bbox_2d"', ":", "[", "100", ",", "100",
             ",", "400", ",", "400", "]", ",", '"label"', ":", ' "breakdancer"', "}", "]"]
            + [f"tok{i}" for i in range(n_tok - 23)]
        )
        tam_maps = [
            np.random.rand(T, H_tam, W_tam).astype(np.float32)
            for _ in gen_tokens
        ]
        gen_text = json.dumps([
            {"frame": t, "bbox_2d": [100, 100, 400, 400], "label": "breakdancer"}
            for t in range(T)
        ])
        return (
            [None] * (len(frames_pil)),
            gen_text,
            {"gen_tokens": gen_tokens, "tam_maps": tam_maps,
             "vision_shape": (T, H_tam, W_tam), "gen_text": gen_text},
        )


if __name__ == "__main__":
    main()
