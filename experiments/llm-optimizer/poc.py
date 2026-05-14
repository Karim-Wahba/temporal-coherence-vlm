"""
poc.py
------
Proof-of-concept: LLM-as-optimizer for video grounding expressions.

Loads existing expression-variance results (no inference needed for the seed
data), runs the OPRO-style optimizer on the highest-variance groups, and
saves a full report.

Two evaluation modes
--------------------
  --eval mock   (default) Score suggestions with a rule-based heuristic
                that penalises known failure patterns. Fast, no GPU required.

  --eval real   Run full TAM inference on each suggestion. Requires
                --davis_root and a GPU. Results are identical in format to
                the expression-variance experiment.

Usage
-----
  # Analysis + mock scoring on top-5 high-variance groups
  python poc.py \
      --results_json  ../../expression-variance/results/expression_variance/results.json \
      --grouped_json  ../../expression-variance/results/expression_variance/grouped_stats.json \
      --model_id      Qwen/Qwen3-VL-8B-Instruct \
      --top_n_groups  5 \
      --n_candidates  3 \
      --n_iterations  2 \
      --eval          mock \
      --out_dir       results/llm_optimizer

  # Real evaluation (needs DAVIS + GPU)
  python poc.py \
      --results_json  ... \
      --grouped_json  ... \
      --model_id      Qwen/Qwen3-VL-8B-Instruct \
      --top_n_groups  5 \
      --eval          real \
      --davis_root    /path/to/DAVIS2017/unsupervised \
      --out_dir       results/llm_optimizer
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_text import QwenTextLLM
from optimizer import ExpressionOptimizer, ScoredExpression
from analysis import SyntacticAnalyzer, SemanticAnalyzer, build_system_prompt


# ── Mock scorer ────────────────────────────────────────────────────────────────

_ACTION_RE = re.compile(
    r"\b(jump|run|swim|mov|walk|hang|fly|go|gallop|leap|trot|roll|ride|spin|"
    r"fall|climb|skate|surf|kick|throw|catch|shoot|race|drift)\w*\b",
    re.IGNORECASE,
)
_SPATIAL_RE = re.compile(
    r"\b(at the end|at the top|at the bottom|on the top|on the bottom|"
    r"in the middle|top half|bottom half|left half|right half)\b",
    re.IGNORECASE,
)
_SUPER_RE = re.compile(
    r"\b(smallest|largest|biggest|tallest|shortest|closest|farthest|"
    r"most \w+|least \w+|the (first|second|third|last|middle))\b",
    re.IGNORECASE,
)
_RELATIONAL_RE = re.compile(
    r"\b(next to|beside|in front of|behind|to the (left|right) of|"
    r"hanging on|attached to|next to|which the)\b",
    re.IGNORECASE,
)
_COLOR_ADJ_RE = re.compile(
    r"\b(black|white|grey|gray|red|blue|green|yellow|orange|brown|"
    r"golden|silver|dark|light|bright|spotted|striped|patterned)\b",
    re.IGNORECASE,
)


def mock_score(expression: str, group_mean_iou: float) -> float:
    """
    Estimate IoU improvement potential for a generated expression.

    Penalises known failure patterns relative to the group's mean IoU.
    This is a heuristic proxy — NOT a real grounding score.
    """
    score = group_mean_iou  # start at group baseline

    # Penalties
    if _ACTION_RE.search(expression):
        score -= 0.15
    if _SPATIAL_RE.search(expression):
        score -= 0.20
    if _SUPER_RE.search(expression):
        score -= 0.12
    if _RELATIONAL_RE.search(expression):
        score -= 0.10

    n_attrs = len(_COLOR_ADJ_RE.findall(expression))
    words = expression.split()

    # Bonus for good structure: noun + 1-2 color/texture attributes, short
    if 3 <= len(words) <= 8:
        score += 0.05
    if 1 <= n_attrs <= 2:
        score += 0.04
    if n_attrs >= 4:
        score -= 0.08  # over-specific

    return round(max(0.0, min(1.0, score)), 4)


def make_mock_evaluator(group_mean_iou: float, group_mean_mass: float | None = None):
    mass_baseline = group_mean_mass if group_mean_mass is not None else group_mean_iou * 0.4

    def evaluate(seq_name: str, obj_id: int, expressions: list[str]) -> list[tuple]:
        results = []
        for e in expressions:
            iou = mock_score(e, group_mean_iou)
            # Mass roughly tracks IoU but with less sensitivity to box accuracy
            mass_gt = round(max(0.0, min(1.0, mass_baseline + (iou - group_mean_iou) * 0.6)), 4)
            results.append((iou, mass_gt, None))
        return results
    return evaluate


# ── Real evaluator (TAM inference) ────────────────────────────────────────────

def make_real_evaluator(model, processor, davis_root: str, sample_rate: int = 8):
    """Build an evaluator that runs TAM and returns mean IoU per expression."""
    import sys
    _EXP = Path(__file__).resolve().parent
    _EXPS = _EXP.parent
    _REF = _EXPS / "Ref-DAVIS"
    _GS  = _EXPS / "grounding-stability-max"
    for p in [str(_REF), str(_REF / "benchmark"), str(_GS)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from benchmark.davis_vot_loader import DAVISVOTLoader
    from benchmark.qwen_vot_runner import QwenVOTRunner
    from token_parser import parse_frame_labels, find_label_token_indices
    from metrics import frame_iou_series, compute_mass_in_gt
    import numpy as np

    loader = DAVISVOTLoader(davis_root, split="valid")
    runner = QwenVOTRunner(model, processor, sample_rate=sample_rate)

    # Build lookup: (seq_name, obj_id) -> first DAVISVOTItem for that pair
    item_map: dict[tuple, object] = {}
    for item in loader:
        key = (item.seq_name, item.obj_id)
        if key not in item_map:
            item_map[key] = item

    def evaluate(seq_name: str, obj_id: int, expressions: list[str]) -> list[tuple]:
        base_item = item_map.get((seq_name, obj_id))
        if base_item is None:
            print(f"    [warn] no DAVIS item for ({seq_name}, {obj_id})")
            return [(None, None, None)] * len(expressions)

        H, W = base_item.frame_size()
        out = []
        for expr in expressions:
            try:
                boxes, _, tam_result = runner.run_with_tam(base_item.frames_pil, expr)

                iou_series = frame_iou_series(boxes, base_item.gt_boxes)
                sampled = list(range(0, base_item.num_frames, sample_rate))
                mean_iou = float(iou_series[sampled].mean()) if sampled else float(iou_series.mean())

                # Compute attention mass metrics the same way as experiment.py
                gen_tokens = tam_result["gen_tokens"]
                tam_maps   = tam_result["tam_maps"]
                vision_T   = tam_result["vision_shape"][0]
                parsed_entries  = parse_frame_labels(tam_result["gen_text"], fps=runner.fps, sample_rate=sample_rate)
                label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

                m_gt_vals, m_pred_vals = [], []
                for sampled_t, tok_idxs in label_token_map:
                    if sampled_t >= vision_T:
                        continue
                    valid = [i for i in tok_idxs if i < len(tam_maps)
                             and tam_maps[i] is not None and tam_maps[i].ndim == 3
                             and sampled_t < tam_maps[i].shape[0]]
                    if not valid:
                        continue
                    avg_map = np.mean([tam_maps[i][sampled_t].astype(np.float32) for i in valid], axis=0)
                    mx = avg_map.max()
                    if mx > 0:
                        avg_map /= mx
                    orig_t   = sampled_t * sample_rate
                    gt_box   = base_item.gt_boxes[orig_t] if orig_t < base_item.num_frames else None
                    pred_box = boxes[orig_t] if orig_t < len(boxes) else None
                    mg = compute_mass_in_gt(avg_map, gt_box,   H, W)
                    mp = compute_mass_in_gt(avg_map, pred_box, H, W)
                    if mg is not None: m_gt_vals.append(float(mg))
                    if mp is not None: m_pred_vals.append(float(mp))

                mass_gt   = round(float(np.mean(m_gt_vals)),   4) if m_gt_vals   else None
                mass_pred = round(float(np.mean(m_pred_vals)), 4) if m_pred_vals else None
                out.append((round(mean_iou, 4), mass_gt, mass_pred))
            except Exception as e:
                print(f"    [error] {expr!r}: {e}")
                out.append((None, None, None))
        return out

    return evaluate


# ── Data loading ───────────────────────────────────────────────────────────────

def load_seed_data(results_json: Path, grouped_json: Path) -> dict[str, list[ScoredExpression]]:
    """Return {group_key: [ScoredExpression, ...]} sorted by group variance (desc)."""
    results = json.loads(results_json.read_text())
    grouped = json.loads(grouped_json.read_text())

    # Build quick lookup: (seq, obj, exp_id) -> (mean_iou, mean_mass_in_gt, mean_mass_in_pred)
    lookup: dict[tuple, tuple] = {}
    for r in results:
        if not r.get("error"):
            key = (r["seq_name"], int(r["obj_id"]), str(r["exp_id"]))
            lookup[key] = (
                r.get("mean_iou"),
                r.get("mean_mass_in_gt"),
                r.get("mean_mass_in_pred"),
            )

    seed: dict[str, list[ScoredExpression]] = {}
    for gkey, gdata in grouped.items():
        iou_stats = gdata.get("iou", {})
        if not iou_stats:
            continue
        seq = gdata["seq_name"]
        oid = int(gdata["obj_id"])
        exprs = gdata["expressions"]
        exp_ids = gdata["exp_ids"]

        scored = []
        for expr, eid in zip(exprs, exp_ids):
            iou, mass_gt, mass_pred = lookup.get((seq, oid, str(eid)), (None, None, None))
            scored.append(ScoredExpression(
                expression=expr,
                iou=iou,
                mass_in_gt=mass_gt,
                mass_in_pred=mass_pred,
                source="original",
            ))
        seed[gkey] = scored

    return seed


_SELECT_MODES = {
    # headroom: lowest best-IoU first → most room to improve
    "headroom": lambda s: s["best"],
    # mean: lowest mean-IoU first → worst overall performance
    "mean":     lambda s: s["mean"],
    # range: highest range first (negated so ascending sort works)
    "range":    lambda s: -s["range"],
}


def select_groups(
    seed: dict[str, list[ScoredExpression]],
    grouped: dict,
    top_n: int,
    select_by: str = "headroom",
) -> list[tuple[str, list[ScoredExpression], float]]:
    key_fn = _SELECT_MODES[select_by]
    rows = []
    for gkey, scored in seed.items():
        iou_stats = grouped[gkey].get("iou", {})
        stats = {
            "best":  iou_stats.get("max",  0.0),
            "mean":  iou_stats.get("mean", 0.0),
            "range": iou_stats.get("range", 0.0),
        }
        rows.append((gkey, scored, stats))
    rows.sort(key=lambda x: key_fn(x[2]))
    print(f"\nGroup selection ({select_by}):")
    for gkey, scored, stats in rows[:top_n]:
        print(f"  {gkey:<30}  best={stats['best']:.3f}  mean={stats['mean']:.3f}  range={stats['range']:.3f}")
    return [(gkey, scored, stats["mean"]) for gkey, scored, stats in rows[:top_n]]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("LLM Expression Optimizer — PoC")
    p.add_argument("--results_json", required=True,
                   help="Path to expression-variance results.json")
    p.add_argument("--grouped_json", required=True,
                   help="Path to grouped_stats.json")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--top_n_groups", type=int, default=5,
                   help="Number of groups to optimise")
    p.add_argument("--select_by", choices=["headroom", "mean", "range"], default="headroom",
                   help="headroom=lowest best-IoU (most room to improve), "
                        "mean=lowest average IoU, range=highest best-vs-worst spread")
    p.add_argument("--n_candidates", type=int, default=3,
                   help="Candidate expressions proposed per iteration")
    p.add_argument("--n_iterations", type=int, default=2,
                   help="OPRO iterations per group")
    p.add_argument("--eval", choices=["mock", "real"], default="mock",
                   help="mock=rule-based heuristic score, real=full TAM inference")
    p.add_argument("--davis_root", default=None,
                   help="Required when --eval real")
    p.add_argument("--sample_rate", type=int, default=8)
    p.add_argument("--use_llm_classify", action="store_true",
                   help="Also use Qwen for failure-mode classification (slower)")
    p.add_argument("--skip_analysis", action="store_true",
                   help="Reuse cached analysis from out_dir/analysis.json (skip re-running)")
    p.add_argument("--out_dir", default="results/llm_optimizer")
    return p.parse_args()


def main():
    args = parse_args()

    if args.eval == "real" and not args.davis_root:
        sys.exit("--eval real requires --davis_root")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = Path(args.results_json)
    grouped_json = Path(args.grouped_json)

    # ── Load seed data ─────────────────────────────────────────────────────────
    print("Loading seed data...")
    grouped_raw = json.loads(grouped_json.read_text())
    seed = load_seed_data(results_json, grouped_json)
    groups = select_groups(seed, grouped_raw, args.top_n_groups, args.select_by)
    print(f"Selected {len(groups)} groups ({args.select_by} selection)")

    # All (expression, iou) pairs across the full dataset — used for analysis
    all_pairs: list[tuple[str, float]] = []
    raw_results = json.loads(results_json.read_text())
    for r in raw_results:
        if not r.get("error") and r.get("mean_iou") is not None:
            all_pairs.append((r["expression"], r["mean_iou"]))

    # ── Load Qwen ──────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    llm = QwenTextLLM(model, processor, max_new_tokens=1024)
    print("Model loaded.")

    # ── Phase 1 & 2: Analysis → dynamic system prompt ─────────────────────────
    analysis_path = out_dir / "analysis.json"
    prompt_path   = out_dir / "system_prompt.txt"

    if args.skip_analysis and analysis_path.exists():
        print("\nLoading cached analysis...")
        cached = json.loads(analysis_path.read_text())
        syntactic_result = cached["syntactic"]
        semantic_result  = cached["semantic"]
    else:
        print(f"\nPhase 1: Syntactic analysis ({len(all_pairs)} expressions)...")
        syntactic_result = SyntacticAnalyzer().analyze(all_pairs)
        print(f"  Tagger: {syntactic_result['tagger']}")
        print(f"  Top feature by IoU impact: "
              f"{next(iter(syntactic_result['feature_correlations']))}")

        print("\nPhase 2: Semantic analysis (Qwen discovers failure taxonomy)...")
        semantic_result = SemanticAnalyzer().analyze(all_pairs, llm)
        n_cats = len(semantic_result.get("failure_categories", []))
        print(f"  Discovered {n_cats} failure categories")
        if semantic_result.get("key_insight"):
            print(f"  Key insight: {semantic_result['key_insight']}")

        analysis_path.write_text(json.dumps(
            {"syntactic": syntactic_result, "semantic": semantic_result}, indent=2
        ))
        print(f"  Saved analysis to {analysis_path}")

    system_prompt = build_system_prompt(syntactic_result, semantic_result)
    prompt_path.write_text(system_prompt)
    print(f"  System prompt written to {prompt_path}")

    # ── Build evaluator ────────────────────────────────────────────────────────
    if args.eval == "real":
        print("Building real TAM evaluator...")
        evaluator = make_real_evaluator(model, processor, args.davis_root, args.sample_rate)
    else:
        evaluator = None  # set per-group below

    # ── Run optimizer ──────────────────────────────────────────────────────────
    optimizer = ExpressionOptimizer(
        llm=llm,
        evaluator=evaluator,
        n_candidates=args.n_candidates,
        n_iterations=args.n_iterations,
        use_llm_classify=args.use_llm_classify,
        system_prompt=system_prompt,
    )

    all_results = []
    for gkey, scored, group_mean in groups:
        parts = gkey.rsplit("__obj", 1)
        seq_name = parts[0]
        obj_id = int(parts[1])

        print(f"\n{'='*60}")
        print(f"Group: {gkey}  (mean_iou={group_mean:.3f})")
        print(f"Seed expressions:")
        for s in sorted(scored, key=lambda x: x.iou or 0, reverse=True):
            iou_s  = f"{s.iou:.3f}"        if s.iou        is not None else " N/A"
            mass_s = f"{s.mass_in_gt:.3f}" if s.mass_in_gt is not None else " N/A"
            print(f"  IoU={iou_s}  MassGT={mass_s}  \"{s.expression}\"")

        # Swap in per-group mock evaluator if needed
        group_mean_mass = sum(s.mass_in_gt for s in scored if s.mass_in_gt is not None)
        group_mean_mass /= max(1, sum(1 for s in scored if s.mass_in_gt is not None))
        if args.eval == "mock":
            optimizer.evaluator = make_mock_evaluator(group_mean, group_mean_mass)

        result = optimizer.run(seq_name, obj_id, scored)
        all_results.append(optimizer.to_dict(result))

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = out_dir / "optimization_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}")
    print(f"Saved results to {out_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n=== Summary ===")
    hdr = f"{'Group':<30} {'Orig IoU':>9} {'LLM IoU':>9} {'Δ IoU':>7}  {'LLM MassGT':>11}  Best LLM Expression"
    print(hdr)
    print("-" * 110)
    for r in all_results:
        g     = f"{r['seq_name']}__obj{r['obj_id']}"
        orig  = r["best_original_iou"]
        best  = r["best_overall_iou"]
        delta = r["delta_iou"]
        expr  = r["best_overall_expression"] or ""
        # pull best overall mass from history
        best_mass = next(
            (h["mass_in_gt"] for h in r["history"] if h["expression"] == expr),
            None,
        )
        orig_s  = f"{orig:.3f}"   if orig  is not None else "  N/A"
        best_s  = f"{best:.3f}"   if best  is not None else "  N/A"
        delta_s = f"{delta:+.3f}" if delta is not None else "   N/A"
        mass_s  = f"{best_mass:.3f}" if best_mass is not None else "    N/A"
        print(f"{g:<30} {orig_s:>9} {best_s:>9} {delta_s:>7}  {mass_s:>11}  {expr[:45]}")

    # Save a CSV version too
    import csv
    csv_path = out_dir / "optimization_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group",
            "best_original_expression", "best_original_iou",
            "best_overall_expression",  "best_overall_iou",
            "best_overall_mass_in_gt",  "delta_iou",
        ])
        writer.writeheader()
        for r in all_results:
            expr = r["best_overall_expression"] or ""
            best_mass = next(
                (h["mass_in_gt"] for h in r["history"] if h["expression"] == expr),
                None,
            )
            writer.writerow({
                "group":                      f"{r['seq_name']}__obj{r['obj_id']}",
                "best_original_expression":   r["best_original_expression"],
                "best_original_iou":          r["best_original_iou"],
                "best_overall_expression":    r["best_overall_expression"],
                "best_overall_iou":           r["best_overall_iou"],
                "best_overall_mass_in_gt":    best_mass,
                "delta_iou":                  r["delta_iou"],
            })
    print(f"\nWrote summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
