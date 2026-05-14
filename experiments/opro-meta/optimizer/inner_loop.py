"""
optimizer/inner_loop.py
-----------------------
OPRO-style per-clip expression optimizer.

For one (seq, obj) clip:
  1. Score the seed expression with the grounder (IoU + mass-in-box + per-POS-cat).
  2. Show the LLM all attempts so far + their scores, ask for K diverse
     reformulations along syntactic + semantic axes.
  3. Score each candidate, append to history.
  4. Repeat for N iterations or until convergence.

Returns a ClipResult with the full trajectory and best expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from linguistic.syntactic import extract_structure
from opro_metrics.grounding_metrics import GroundingMetrics, compute_metrics
from optimizer.llm_client import _try_parse_json


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    expression: str
    iteration: int
    source: str                         # 'seed' | 'llm_iter_{n}'
    metrics: GroundingMetrics
    rationale: Optional[str] = None     # LLM explanation for proposing this candidate


@dataclass
class ClipResult:
    seq_name: str
    obj_id: int
    seed_expression: str
    history: List[Attempt] = field(default_factory=list)
    best: Optional[Attempt] = None

    def to_dict(self) -> dict:
        return {
            "seq_name": self.seq_name,
            "obj_id":   self.obj_id,
            "seed_expression": self.seed_expression,
            "best_expression": self.best.expression if self.best else None,
            "best_iou":        self.best.metrics.mean_iou if self.best else None,
            "history": [
                {
                    "iteration":  a.iteration,
                    "source":     a.source,
                    "rationale":  a.rationale,
                    **a.metrics.to_dict(),
                }
                for a in self.history
            ],
        }


# ── Prompt building ───────────────────────────────────────────────────────────

_INNER_SYSTEM = """\
You are an expert in video object grounding with vision-language models.

Your task: propose better natural-language referring expressions for a
specific object in a video clip. The grounding model uses Qwen3-VL and is
evaluated by IoU between predicted bounding boxes and ground-truth boxes.

You will see the history of expressions already tried, each with:
  IoU             box-overlap score (0 = miss, 1 = perfect)
  MassGT          attention mass inside the GT box (higher = right region)
  MassPred        attention mass inside the predicted box
  noun/adj/verb   mass-in-GT broken down by token POS category
  Structure       parsed dependency summary: head noun, adjective modifiers,
                  verb relations (e.g. riding->bike), prepositional phrases.
                  This lets you reason about *what* changed structurally
                  between high- and low-scoring expressions, not just word
                  choice. Compare e.g. "head=bike  adj=[black]" with
                  "head=man  verb=[riding->bike]" — the head changed.

Goals when proposing candidates:
  - Vary BOTH syntactic structure (change head noun, add/drop modifiers,
    swap verb phrase for noun phrase, etc.) AND semantic content
    (add/remove attributes, swap synonyms, change head noun specificity).
  - Avoid the structural patterns that score poorly in the history.
  - Stay anchored to what made the best entry work — preserve its head noun
    and any clearly helpful adjective modifiers unless the data says
    otherwise.

Output ONLY valid JSON (no markdown fences) of the form:
{
  "analysis": "1-2 sentences naming the structural failure pattern in the
               low-IoU entries and what structural change you'll try",
  "candidates": [
    {"expression": "<new candidate>", "kind": "syntactic|semantic|mixed",
     "rationale": "what this candidate tests structurally"},
    ...
  ]
}"""


def _render_attempt_row(a: Attempt) -> str:
    m = a.metrics
    iou = f"{m.mean_iou:.3f}" if m.mean_iou is not None else " N/A"
    mg  = f"{m.mean_mass_in_gt:.3f}" if m.mean_mass_in_gt is not None else " N/A"
    mp  = f"{m.mean_mass_in_pred:.3f}" if m.mean_mass_in_pred is not None else " N/A"
    cat = m.token_category_breakdown or {}
    def fmt(k):
        v = cat.get(k)
        return f"{v:.2f}" if isinstance(v, (int, float)) and v is not None else "—"
    cats = f"noun={fmt('noun')} adj={fmt('adj')} verb={fmt('verb')}"

    struct = extract_structure(a.expression)
    return (
        f'  IoU={iou} MassGT={mg} MassPred={mp} [{cats}]  "{a.expression}"\n'
        f'        Structure: {struct["summary"]}'
    )


def _format_history(
    history: List[Attempt],
    n_top: int = 5,
    n_bottom: int = 5,
) -> str:
    """Render the history block shown to the inner-loop LLM.

    Sorted best -> worst by IoU (None last). If the history fits in
    n_top + n_bottom rows, render every row; otherwise show only the top
    n_top + bottom n_bottom, with a marker line indicating how many
    intermediate attempts are hidden and the IoU band they cover.
    """
    ranked = sorted(
        history,
        key=lambda a: (a.metrics.mean_iou is not None, a.metrics.mean_iou or 0),
        reverse=True,
    )

    if n_top <= 0 and n_bottom <= 0:
        n_top, n_bottom = 5, 5

    if len(ranked) <= n_top + n_bottom:
        return "\n".join(_render_attempt_row(a) for a in ranked)

    top    = ranked[:n_top]
    bottom = ranked[-n_bottom:]
    middle = ranked[n_top:-n_bottom]

    mid_ious = [a.metrics.mean_iou for a in middle if a.metrics.mean_iou is not None]
    if mid_ious:
        lo, hi = min(mid_ious), max(mid_ious)
        marker = f"  [... {len(middle)} intermediate attempts omitted (IoU range {lo:.3f}-{hi:.3f}) ...]"
    else:
        marker = f"  [... {len(middle)} intermediate attempts omitted ...]"

    lines = [_render_attempt_row(a) for a in top]
    lines.append(marker)
    lines.extend(_render_attempt_row(a) for a in bottom)
    return "\n".join(lines)


def _build_prompt(
    seq_name: str,
    obj_id: int,
    history: List[Attempt],
    n_candidates: int,
    history_top_n: int = 5,
    history_bottom_n: int = 5,
) -> str:
    block = _format_history(history, n_top=history_top_n, n_bottom=history_bottom_n)
    best = max(
        (a for a in history if a.metrics.mean_iou is not None),
        key=lambda a: a.metrics.mean_iou,
        default=None,
    )
    best_note = ""
    if best is not None:
        best_note = (
            f'\nCurrent best: "{best.expression}" (IoU={best.metrics.mean_iou:.3f}). '
            f"Keep close to its core noun + any clearly helpful attributes."
        )
    return (
        f'Object: obj{obj_id} in video "{seq_name}"\n\n'
        f"Attempts so far (best -> worst by IoU; middle band may be omitted "
        f"once history grows past {history_top_n + history_bottom_n} entries):\n"
        f"{block}\n"
        f"{best_note}\n\n"
        f"Propose exactly {n_candidates} new candidate expressions. "
        f"Each candidate must be different from every expression above."
    )


# ── Loop ──────────────────────────────────────────────────────────────────────

@dataclass
class InnerLoopConfig:
    n_candidates: int = 5
    n_iterations: int = 10
    early_stop_no_improve: int = 3          # stop after N iters without best-IoU improvement
    system_prompt: Optional[str] = None     # override _INNER_SYSTEM (used by meta-prompt eval)
    history_top_n: int = 5                  # rows from the top of the IoU ranking shown to the LLM
    history_bottom_n: int = 5               # rows from the bottom; middle band is summarised


def run_clip(
    clip,                                   # data.ref_davis_loader.Clip
    grounder,                               # grounding.qwen3vl_runner.Qwen3VLGrounder
    llm_client,                             # optimizer.llm_client.* backend
    config: InnerLoopConfig,
    logger=None,                            # opro_logging.RunLogger | None
) -> ClipResult:
    result = ClipResult(
        seq_name=clip.seq_name,
        obj_id=clip.obj_id,
        seed_expression=clip.seed_expression,
    )
    system = config.system_prompt or _INNER_SYSTEM

    def evaluate(expr: str) -> GroundingMetrics:
        try:
            grounding = grounder.ground(clip.frames_pil, expr)
            return compute_metrics(
                grounding=grounding,
                clip=clip,
                sample_rate=grounder.sample_rate,
                fps=grounder.fps,
            )
        except Exception as e:
            return GroundingMetrics(
                expression=expr,
                mean_iou=None, mean_mass_in_gt=None, mean_mass_in_pred=None,
                error=str(e),
            )

    # ── seed ────────────────────────────────────────────────────────────────
    if logger:
        logger.log("clip_start", seq=clip.seq_name, obj=clip.obj_id,
                   seed_expression=clip.seed_expression,
                   iteration_budget=config.n_iterations)

    seed_metrics = evaluate(clip.seed_expression)
    seed_attempt = Attempt(
        expression=clip.seed_expression,
        iteration=0,
        source="seed",
        metrics=seed_metrics,
    )
    result.history.append(seed_attempt)
    if logger:
        # `expression` is already in seed_metrics.to_dict(), don't pass it twice.
        logger.log("candidate_evaluated", seq=clip.seq_name, obj=clip.obj_id,
                   iter=0, **seed_metrics.to_dict())

    print(f"  seed IoU={seed_metrics.mean_iou}  \"{clip.seed_expression}\"")

    # ── iterations ──────────────────────────────────────────────────────────
    last_improve_iter = 0
    best_iou = seed_metrics.mean_iou or -1.0

    for it in range(1, config.n_iterations + 1):
        prompt = _build_prompt(
            clip.seq_name, clip.obj_id, result.history, config.n_candidates,
            history_top_n=config.history_top_n,
            history_bottom_n=config.history_bottom_n,
        )

        # Two-step call so we can see and log the raw text on parse failure.
        raw = llm_client.generate(prompt, system=system)
        parsed = _try_parse_json(raw)

        if logger:
            logger.log("llm_raw_response", seq=clip.seq_name, obj=clip.obj_id,
                       iter=it, raw=raw[:4000], parsed_ok=bool(parsed))

        if not parsed or not isinstance(parsed, dict):
            snippet = raw[:400].replace("\n", " ")
            print(f"  iter {it}: LLM returned no parseable JSON — stopping")
            print(f"            raw[:400] = {snippet!r}")
            print(f"            (full raw saved to JSONL log as event=llm_raw_response)")
            break

        analysis = parsed.get("analysis", "")
        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            print(f"  iter {it}: LLM returned no candidates — stopping")
            print(f"            parsed = {parsed}")
            break

        tested_lower = {a.expression.strip().lower() for a in result.history}

        new_attempts: List[Attempt] = []
        for c in candidates:
            expr = (c.get("expression") if isinstance(c, dict) else c) or ""
            expr = expr.strip()
            if not expr or expr.lower() in tested_lower:
                continue
            rationale = c.get("rationale") if isinstance(c, dict) else None
            if logger:
                logger.log("candidate_proposed", seq=clip.seq_name, obj=clip.obj_id,
                           iter=it, expression=expr, rationale=rationale,
                           analysis=analysis if not new_attempts else None)

            m = evaluate(expr)
            attempt = Attempt(
                expression=expr,
                iteration=it,
                source=f"llm_iter_{it}",
                metrics=m,
                rationale=rationale,
            )
            new_attempts.append(attempt)
            result.history.append(attempt)
            tested_lower.add(expr.lower())

            iou_str = f"{m.mean_iou:.3f}" if m.mean_iou is not None else "N/A"
            print(f"    iter {it}  IoU={iou_str}  \"{expr}\"")

            if logger:
                # `expression` is already in m.to_dict(), don't pass it twice.
                logger.log("candidate_evaluated", seq=clip.seq_name, obj=clip.obj_id,
                           iter=it, **m.to_dict())

        # Early stop
        iter_best = max(
            (a.metrics.mean_iou for a in new_attempts if a.metrics.mean_iou is not None),
            default=-1.0,
        )
        if iter_best > best_iou + 1e-4:
            best_iou = iter_best
            last_improve_iter = it
        elif it - last_improve_iter >= config.early_stop_no_improve:
            print(f"  no improvement for {config.early_stop_no_improve} iters — stopping")
            break

    # ── pick best ───────────────────────────────────────────────────────────
    scored = [a for a in result.history if a.metrics.mean_iou is not None]
    if scored:
        result.best = max(scored, key=lambda a: a.metrics.mean_iou)
        print(f"  best IoU={result.best.metrics.mean_iou:.3f}  "
              f"\"{result.best.expression}\"")

    if logger:
        logger.log("clip_done", seq=clip.seq_name, obj=clip.obj_id,
                   best_expression=result.best.expression if result.best else None,
                   best_iou=result.best.metrics.mean_iou if result.best else None,
                   history_len=len(result.history))

    return result
