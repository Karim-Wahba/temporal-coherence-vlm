"""
optimizer/outer_loop.py
-----------------------
Meta-prompt distillation.

The meta-LLM is asked for STRUCTURED RULES (do / avoid lists keyed to the
dependency structure), not for a free-form prompt. We then assemble those
rules into a deterministic, directive inner-loop system prompt via
`render_inner_system_from_meta()`. The benefit: the rendered prompt is always
in the right format for the inner loop (role, rules, JSON schema), regardless
of how the meta-LLM phrases its reasoning.

Flow:
  inner trajectories ──► distill_meta_prompt() ──► {do_rules, avoid_rules,
                                                    patterns, key_insight}
                                                  │
                                                  ▼
                                render_inner_system_from_meta()
                                                  │
                                                  ▼
                            full inner-loop `system` text used by
                            evaluate_meta_prompt() on held-out clips
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from linguistic.syntactic import extract_structure
from optimizer.inner_loop import ClipResult, InnerLoopConfig, run_clip


# ── Meta-LLM system prompt: structured rules only, no free-form prompt ───────

_META_SYSTEM = """\
You are analysing OPRO trajectories of natural-language referring expressions
optimised for video object grounding (Qwen3-VL + IoU).

For each clip you receive: the seed expression, every candidate the inner
loop tried, per-candidate IoU + attention-mass metrics, and a parsed
dependency structure (head noun, adjective modifiers, verb relations,
prepositional phrases) for every expression.

Your job: distill the trajectories into STRUCTURED RULES that a downstream
LLM can follow. DO NOT write a prompt. DO NOT include narrative reasoning.
Just produce the rules. Reason at the dependency-structure level (head /
modifiers / verbs / prep_phrases), not surface word choice.

Output ONLY valid JSON (no markdown fences):
{
  "transferable_patterns": [
    {
      "pattern": "short structural description, e.g. 'replace VP head with NP head matching the visual target'",
      "evidence_clips": ["seq__objN", ...],
      "typical_delta_iou": 0.12
    }
  ],
  "do_rules": [
    "Imperative rule the inner LLM should follow, e.g. 'Use the visual target as the head noun of every candidate'",
    "..."
  ],
  "avoid_rules": [
    "Imperative rule the inner LLM should NOT do, e.g. 'Do not use prep_phrases that anchor to frame position (in the middle, on the left)'",
    "..."
  ],
  "head_noun_policy": "one short sentence on when to change the head noun vs keep it",
  "modifier_policy":  "one short sentence on adjective / participle modifiers",
  "verb_policy":      "one short sentence on verb phrases",
  "key_insight":      "one sentence — the single most important takeaway"
}

Rules must be:
  - imperative ('Use ...', 'Avoid ...', 'Replace ...'); NOT first-person ('I noticed ...')
  - concrete and structural; reference head / amod / verb / prep_phrase
  - actionable on a single seed expression without seeing more data
"""


# ── Trajectory formatting (unchanged) ─────────────────────────────────────────

def _format_trajectory(c: ClipResult) -> str:
    lines = [f'### {c.seq_name}__obj{c.obj_id}  seed: "{c.seed_expression}"']
    for a in sorted(c.history, key=lambda a: a.iteration):
        m = a.metrics
        iou = f"{m.mean_iou:.3f}" if m.mean_iou is not None else " N/A"
        mg  = f"{m.mean_mass_in_gt:.3f}" if m.mean_mass_in_gt is not None else " N/A"
        src = a.source
        struct = extract_structure(a.expression)
        line = (
            f'  [{src}] IoU={iou} MassGT={mg}  "{a.expression}"\n'
            f'      structure: {struct["summary"]}'
        )
        if a.rationale:
            line += f"\n      rationale: {a.rationale[:100]}"
        lines.append(line)
    return "\n".join(lines)


# ── Distillation call ─────────────────────────────────────────────────────────

def distill_meta_prompt(
    clip_results: List[ClipResult],
    llm_client,
    out_dir: str | Path,
    version: Optional[str] = None,
    max_new_tokens: int = 4096,
    logger=None,
) -> dict:
    """Ask the meta-LLM for structured rules; assemble an inner-loop prompt.

    Returns a dict with at minimum:
      {rules:..., inner_system_prompt: "<full directive prompt>", version,
       n_source_clips, source_clips}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if version is None:
        version = f"v{time.strftime('%Y%m%d_%H%M%S')}"

    corpus = "\n\n".join(_format_trajectory(c) for c in clip_results)
    prompt = (
        f"Analyse the following {len(clip_results)} OPRO trajectory/ies "
        f"and return the structured rules as JSON.\n\n"
        f"{corpus}\n"
    )

    parsed = llm_client.generate_json(
        prompt, system=_META_SYSTEM, max_new_tokens=max_new_tokens
    )

    if not parsed or not isinstance(parsed, dict):
        raw = llm_client.generate(
            prompt, system=_META_SYSTEM, max_new_tokens=max_new_tokens
        )
        parsed = {"raw_response": raw, "parse_error": True}

    # Always render the directive inner-loop system prompt deterministically
    inner_system = render_inner_system_from_meta(parsed)

    out = {
        **parsed,
        "version":        version,
        "n_source_clips": len(clip_results),
        "source_clips":   [f"{c.seq_name}__obj{c.obj_id}" for c in clip_results],
        "inner_system_prompt": inner_system,
    }

    path = out_dir / f"meta_prompt_{version}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  meta-prompt written to {path}")
    print(f"  --- assembled inner-loop system prompt ---")
    print(inner_system)
    print(f"  --- end ---")

    if logger:
        logger.log("meta_prompt",
                   version=version,
                   path=str(path),
                   source_clips=out["source_clips"],
                   inner_system_prompt=inner_system[:1000])

    return out


# ── Deterministic assembler: rules ──► directive inner-loop system prompt ────

_INNER_TEMPLATE = """\
You are an expert in video object grounding with vision-language models.
Your task: given a video clip's seed referring expression and its grounding
scores, propose better referring expressions that achieve higher IoU.

The grounding model is Qwen3-VL. You will see, for every attempt so far:
  IoU             box-overlap with ground truth (0–1)
  MassGT          attention mass inside the GT box
  MassPred        attention mass inside the predicted box
  noun/adj/verb   mass-in-GT broken down by token POS category
  Structure       parsed dependency summary (head noun, adjective modifiers,
                  verb relations, prepositional phrases)

DISTILLED RULES (from prior optimisation runs, evidence-based):

DO:
{do_block}

DO NOT:
{avoid_block}

STRUCTURAL POLICIES:
  - Head noun  : {head_policy}
  - Modifiers  : {modifier_policy}
  - Verbs      : {verb_policy}

KEY INSIGHT: {key_insight}

When proposing each candidate, name the structural change you are making
(head / modifier / verb / prep_phrase) so the next round can learn from it.

Output ONLY valid JSON (no markdown fences, no <think> blocks) of the form:
{{
  "analysis": "1-2 sentences naming the dominant failure pattern in the
               low-IoU entries and what structural move you'll try next",
  "candidates": [
    {{"expression": "<new candidate>",
      "kind":       "syntactic|semantic|mixed",
      "rationale":  "what structural change this tests"}},
    ...
  ]
}}"""


def render_inner_system_from_meta(meta: dict) -> str:
    """Build the inner-loop system prompt from the meta-LLM's structured rules.

    Falls back to a minimal default if any field is missing — the inner loop
    always gets a usable prompt.
    """
    def _bullets(items, fallback):
        items = [str(s).strip() for s in (items or []) if str(s).strip()]
        if not items:
            items = [fallback]
        return "\n".join(f"  ✓ {s}" if s == fallback else f"  • {s}" for s in items)

    do_items    = meta.get("do_rules")    or []
    avoid_items = meta.get("avoid_rules") or []

    do_block = "\n".join(f"  • {str(s).strip()}" for s in do_items if str(s).strip()) \
               or "  • Use the visual target as the head noun"
    avoid_block = "\n".join(f"  • {str(s).strip()}" for s in avoid_items if str(s).strip()) \
                  or "  • Avoid action verbs and position-anchored prep phrases"

    return _INNER_TEMPLATE.format(
        do_block=do_block,
        avoid_block=avoid_block,
        head_policy=meta.get("head_noun_policy")
                    or "Keep the head noun that the best entry uses unless it is wrong.",
        modifier_policy=meta.get("modifier_policy")
                        or "Prefer 0–2 static amod-type modifiers (color, texture).",
        verb_policy=meta.get("verb_policy")
                    or "Avoid verb phrases unless they consistently helped IoU.",
        key_insight=meta.get("key_insight") or "Short, head-noun-led NPs ground best.",
    )


# ── Held-out evaluation ───────────────────────────────────────────────────────

def evaluate_meta_prompt(
    holdout_clips,                      # iterable[Clip]
    grounder,
    llm_client,
    meta: dict,                         # full distilled meta dict (or just a string for back-compat)
    n_iters_for_meta: int = 1,
    n_candidates: int = 5,
    logger=None,
) -> List[ClipResult]:
    """Run the inner loop on held-out clips using the assembled directive
    system prompt. Accepts either the full meta dict (preferred — uses the
    cached inner_system_prompt) or a raw string for legacy callers."""
    if isinstance(meta, str):
        inner_system = meta
    else:
        inner_system = meta.get("inner_system_prompt") or render_inner_system_from_meta(meta)

    results: List[ClipResult] = []
    cfg = InnerLoopConfig(
        n_candidates=n_candidates,
        n_iterations=n_iters_for_meta,
        early_stop_no_improve=99,       # disable early stop for clean iter-1 comparison
        system_prompt=inner_system,
    )
    for clip in holdout_clips:
        print(f"\n[holdout] {clip.seq_name} obj{clip.obj_id}")
        res = run_clip(clip, grounder, llm_client, cfg, logger=logger)
        results.append(res)
    return results
