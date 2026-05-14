"""
optimizer.py
------------
OPRO-style (Optimization by PROmpting) expression optimizer.

At each iteration the optimizer shows Qwen all previously tested
(expression, IoU) pairs sorted by score, then asks it to identify
the pattern of failure and propose new candidate expressions.

The loop:
  1. Seed with existing expressions + IoU scores from results.json
  2. Ask Qwen to analyse failures and propose N candidates
  3. Evaluate candidates (real evaluation via TAM, or mock scorer)
  4. Add evaluated candidates to the history
  5. Repeat for `n_iterations`

The evaluator is injected as a callable:
  evaluate(seq_name, obj_id, expressions) -> list[float | None]
This lets us swap between a real TAM evaluator and the mock scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from failure_classifier import classify_group, ClassificationResult, FAILURE_MODES
from qwen_text import QwenTextLLM


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ScoredExpression:
    expression: str
    iou: float | None
    mass_in_gt: float | None = None    # attention mass inside GT box (higher = better focus)
    mass_in_pred: float | None = None  # attention mass inside predicted box
    source: str = "original"           # "original" | "llm_iter_{n}"
    failure_mode: str | None = None


@dataclass
class OptimizationResult:
    seq_name: str
    obj_id: int
    history: list[ScoredExpression] = field(default_factory=list)
    best_original: ScoredExpression | None = None
    best_overall: ScoredExpression | None = None
    iterations: list[dict] = field(default_factory=list)

    def improvement(self) -> float | None:
        if self.best_original and self.best_overall:
            orig = self.best_original.iou
            best = self.best_overall.iou
            if orig is not None and best is not None:
                return best - orig
        return None


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert in video object grounding with vision-language models.
Your task: improve natural language expressions so that a model can better \
locate a specific object across all frames of a video.

Key findings from grounding experiments (244 expression/IoU pairs analysed):

METRICS (shown per expression):
  IoU        — spatial overlap of predicted box vs ground-truth box (0=miss, 1=perfect)
  MassGT     — fraction of model attention inside the GT box (higher = attention is focused
                on the right region, even when the box prediction is slightly off)
  MassPred   — fraction of attention inside the predicted box (consistency check)
  Both mass metrics come from TAM (Temporal Attention Maps) over the generated tokens.
  A high MassGT with low IoU means the model is attending the right region but the
  box coordinates are imprecise. A low MassGT means attention is scattered or wrong.

WHAT WORKS (token dominance in oracle attention):
  - Head noun (correct object category): 26 % of oracle frames
  - Static adjectives (color, texture, shape): 15 % of oracle frames
  - Short, discriminative descriptions → higher IoU and MassGT

WHAT HURTS:
  1. Action / motion language  ("jumping", "moving", "going to the right")
       ‣ verbs are oracle only 3 % of frames — they actively mislead
       ‣ example: "dog jumping for a snack" IoU=0.10 vs "white dog with patches" IoU=0.67
  2. Ordinal / spatial position ("at the end", "top half", "bottom half")
       ‣ object position changes each frame → completely unreliable (IoU=0 observed)
  3. Superlatives / ordinal rank ("smallest", "middle", "the first")
       ‣ relative size/rank changes as objects enter/leave frame
  4. Over-specific (≥ 4 adjectives / attributes)
       ‣ competes with the head noun for model attention
  5. Wrong head noun  → catastrophic failure
  6. Relational context ("hanging on", "next to the person")
       ‣ fails when the reference object is ambiguous or partially visible

EXPRESSION RULES:
  ✓ Use the correct, specific object category as the head noun
  ✓ Add 1-2 static discriminative attributes (color, texture, distinctive marking)
  ✓ Keep it concise (4-8 words ideal)
  ✗ No action verbs, no positional/ordinal/superlative terms, no relational clauses

When analysing, consider both IoU AND MassGT together:
  - Low IoU + low MassGT  → attention is wrong (expression misleads the model)
  - Low IoU + high MassGT → attention is correct but box regression is off
  - High IoU + high MassGT → ideal: good expression

Output ONLY valid JSON (no markdown fences)."""


# ── Prompts ────────────────────────────────────────────────────────────────────

def _build_prompt(
    seq_name: str,
    obj_id: int,
    scored: list[ScoredExpression],
    n_candidates: int,
    failure_modes: list[ClassificationResult],
) -> str:
    # Sort by IoU descending (best first), unknowns at end
    ranked = sorted(
        [(s, fm) for s, fm in zip(scored, failure_modes)],
        key=lambda x: (x[0].iou is not None, x[0].iou or 0),
        reverse=True,
    )

    lines = []
    for s, fm in ranked:
        iou_str    = f"{s.iou:.3f}"        if s.iou        is not None else " N/A"
        mass_str   = f"{s.mass_in_gt:.3f}" if s.mass_in_gt is not None else " N/A"
        mode_str   = f"  [{fm.final_mode}]" if fm.final_mode != "GOOD" else ""
        lines.append(f'  IoU={iou_str}  MassGT={mass_str}  "{s.expression}"{mode_str}')

    history_block = "\n".join(lines)

    best = ranked[0][0] if ranked else None
    best_note = ""
    if best and best.iou is not None:
        best_note = (
            f'\nThe current best expression is "{best.expression}" (IoU={best.iou:.3f}). '
            f"Your candidates should stay close to what makes it work — preserve its "
            f"core noun and key attributes, and only change what the failure analysis "
            f"suggests. Do NOT invent completely different descriptions."
        )

    prompt = (
        f'Object: obj{obj_id} in video sequence "{seq_name}"\n\n'
        f"All tested expressions (best → worst by IoU):\n{history_block}\n"
        f"{best_note}\n\n"
        f"Task:\n"
        f"1. Briefly identify the key failure patterns in the low-IoU / low-MassGT expressions.\n"
        f"2. Propose exactly {n_candidates} new candidate expressions that should "
        f"achieve higher IoU and MassGT by avoiding those patterns.\n"
        f"   Each candidate must be different from every expression listed above.\n\n"
        f'Output JSON: {{"analysis": "...", "candidates": ["...", ..., "..."]}}'
    )
    return prompt


# ── Optimizer ─────────────────────────────────────────────────────────────────

class ExpressionOptimizer:
    """
    OPRO-style optimizer that uses Qwen as both the failure analyst
    and the expression proposal engine.

    Parameters
    ----------
    llm          : QwenTextLLM wrapping the already-loaded Qwen model
    evaluator    : callable(seq_name, obj_id, [expr, ...]) -> [iou, ...]
                   Pass None to skip evaluation (dry-run / analysis only)
    n_candidates : candidates proposed per iteration
    n_iterations : optimization iterations
    use_llm_classify : also run Qwen for failure-mode classification
                       (slower but richer explanations)
    """

    def __init__(
        self,
        llm: QwenTextLLM,
        evaluator: Callable[[str, int, list[str]], list[float | None]] | None = None,
        n_candidates: int = 3,
        n_iterations: int = 2,
        use_llm_classify: bool = False,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.evaluator = evaluator
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.use_llm_classify = use_llm_classify
        # Use externally-built prompt (from analysis.py) if provided, else default
        self._system = system_prompt if system_prompt else _SYSTEM

    def _classify(self, scored: list[ScoredExpression]) -> list[ClassificationResult]:
        exprs = [s.expression for s in scored]
        llm = self.llm if self.use_llm_classify else None
        return classify_group(exprs, llm=llm)

    def _propose(
        self,
        seq_name: str,
        obj_id: int,
        scored: list[ScoredExpression],
        failure_modes: list[ClassificationResult],
    ) -> tuple[list[str], str]:
        """Ask Qwen for new candidates. Returns (candidates, analysis_text)."""
        prompt = _build_prompt(seq_name, obj_id, scored, self.n_candidates, failure_modes)
        parsed = self.llm.generate_json(prompt, system=self._system)

        candidates: list[str] = []
        analysis = ""

        if parsed and isinstance(parsed, dict):
            analysis = parsed.get("analysis", "")
            raw_cands = parsed.get("candidates", [])
            if isinstance(raw_cands, list):
                # Filter: non-empty strings not already tested
                tested = {s.expression.lower() for s in scored}
                for c in raw_cands:
                    if isinstance(c, str) and c.strip() and c.strip().lower() not in tested:
                        candidates.append(c.strip())

        return candidates[:self.n_candidates], analysis

    def run(self, seq_name: str, obj_id: int, seed_scored: list[ScoredExpression]) -> OptimizationResult:
        result = OptimizationResult(seq_name=seq_name, obj_id=obj_id)
        result.history = list(seed_scored)

        # Classify seed expressions
        failure_modes = self._classify(result.history)
        for s, fm in zip(result.history, failure_modes):
            s.failure_mode = fm.final_mode

        original_iou = [s.iou for s in result.history if s.iou is not None]
        result.best_original = max(result.history, key=lambda s: s.iou or 0)

        print(f"\n  Seed: {len(result.history)} expressions | "
              f"best={result.best_original.iou:.3f}  worst={min(original_iou):.3f}")

        for it in range(1, self.n_iterations + 1):
            print(f"\n  --- Iteration {it}/{self.n_iterations} ---")
            failure_modes = self._classify(result.history)

            candidates, analysis = self._propose(seq_name, obj_id, result.history, failure_modes)

            if not candidates:
                print("  Qwen returned no usable candidates — stopping early")
                break

            print(f"  Analysis: {analysis[:200]}")
            print(f"  Candidates ({len(candidates)}):")
            for c in candidates:
                print(f"    • {c}")

            # Evaluate candidates
            if self.evaluator is not None:
                iou_values = self.evaluator(seq_name, obj_id, candidates)
            else:
                iou_values = [None] * len(candidates)

            new_scored = []
            for expr, scores in zip(candidates, iou_values):
                iou, mass_gt, mass_pred = scores if isinstance(scores, tuple) else (scores, None, None)
                s = ScoredExpression(
                    expression=expr,
                    iou=iou,
                    mass_in_gt=mass_gt,
                    mass_in_pred=mass_pred,
                    source=f"llm_iter_{it}",
                )
                new_scored.append(s)
                result.history.append(s)
                iou_str  = f"{iou:.3f}"    if iou    is not None else "pending"
                mass_str = f"{mass_gt:.3f}" if mass_gt is not None else "pending"
                print(f"    IoU={iou_str}  MassGT={mass_str}  \"{expr}\"")

            result.iterations.append({
                "iteration": it,
                "analysis": analysis,
                "candidates": [
                    {
                        "expression":  s.expression,
                        "iou":         s.iou,
                        "mass_in_gt":  s.mass_in_gt,
                        "mass_in_pred": s.mass_in_pred,
                        "source":      s.source,
                    }
                    for s in new_scored
                ],
            })

        # Best overall (including LLM suggestions)
        scored_with_iou = [s for s in result.history if s.iou is not None]
        if scored_with_iou:
            result.best_overall = max(scored_with_iou, key=lambda s: s.iou)

        imp = result.improvement()
        if imp is not None:
            bo = result.best_overall
            mass_str = f"  MassGT={bo.mass_in_gt:.3f}" if bo.mass_in_gt is not None else ""
            print(f"\n  Best original: {result.best_original.iou:.3f}  "
                  f"→  Best overall: {bo.iou:.3f}{mass_str}  "
                  f"(Δ={imp:+.3f})")

        return result

    def to_dict(self, result: OptimizationResult) -> dict:
        return {
            "seq_name": result.seq_name,
            "obj_id": result.obj_id,
            "best_original_expression": result.best_original.expression if result.best_original else None,
            "best_original_iou": result.best_original.iou if result.best_original else None,
            "best_overall_expression": result.best_overall.expression if result.best_overall else None,
            "best_overall_iou": result.best_overall.iou if result.best_overall else None,
            "delta_iou": result.improvement(),
            "history": [
                {
                    "expression":   s.expression,
                    "iou":          s.iou,
                    "mass_in_gt":   s.mass_in_gt,
                    "mass_in_pred": s.mass_in_pred,
                    "source":       s.source,
                    "failure_mode": s.failure_mode,
                }
                for s in result.history
            ],
            "iterations": result.iterations,
        }
