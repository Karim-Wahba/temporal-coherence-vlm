"""
failure_classifier.py
---------------------
Two-stage failure mode classifier for grounding expressions.

Stage 1 (always runs): rule-based regex patterns derived from the 8 documented
  failure examples in failure_examples_table.csv.
Stage 2 (optional, requires Qwen): LLM re-classification for ambiguous cases
  or to produce a richer natural-language explanation.

Failure modes
-------------
  ACTION_MOTION        : expression relies on dynamic action/motion verbs
  ORDINAL_SPATIAL      : uses frame-relative spatial positions ("at the end",
                          "top half", "bottom half")
  ORDINAL_SUPERLATIVE  : uses superlatives or ordinal ranks ("smallest", "middle")
  OVER_SPECIFIC        : ≥4 visual attributes; too many descriptors
  WRONG_HEAD_NOUN      : head noun does not match the object category
  VAGUE_NOUN           : head noun is too generic ("bike" vs "stunt bike")
  CONTEXT_DISAM        : relies on relational/contextual reference to another object
  GOOD                 : no detected failure mode
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen_text import QwenTextLLM

# ── Failure mode labels ────────────────────────────────────────────────────────

FAILURE_MODES = [
    "ACTION_MOTION",
    "ORDINAL_SPATIAL",
    "ORDINAL_SUPERLATIVE",
    "OVER_SPECIFIC",
    "WRONG_HEAD_NOUN",
    "VAGUE_NOUN",
    "CONTEXT_DISAM",
    "GOOD",
]

# ── Rule-based patterns ────────────────────────────────────────────────────────

_ACTION_VERBS = (
    # Explicit forms only — avoids false matches on "goldfish" (go), "surfing rope" (surf)
    r"\b(jump|jumps|jumping|jumped"
    r"|run|runs|running|ran"
    r"|swim|swims|swimming|swam"
    r"|move|moves|moving|moved"
    r"|walk|walks|walking|walked"
    r"|hanging(?! on)|hangs|hung"   # "hanging on" handled by CONTEXT_DISAM
    r"|fly|flies|flying|flew"
    r"|going|goes|went"             # bare "go" excluded (matches "goldfish")
    r"|gallop|gallops|galloping|galloped"
    r"|leap|leaps|leaping|leapt"
    r"|trot|trots|trotting|trotted"
    r"|rolling|rolls|rolled"        # bare "roll" excluded (matches "roller")
    r"|ride|rides|riding|rode"
    r"|spin|spins|spinning|spun"
    r"|fall|falls|falling|fell"
    r"|climb|climbs|climbing|climbed"
    r"|skating|skates|skated"
    r"|kick|kicks|kicking|kicked"
    r"|throw|throws|throwing|threw"
    r"|catch|catches|catching|caught"
    r"|dribbling|dribbles|dribbled"
    r"|shooting(?!__)|shoots|shot"  # avoid matching sequence name "shooting"
    r"|racing|races|raced"
    r"|drifting|drifts|drifted)\b"
)

_SPATIAL_ORDINALS = (
    r"\b(at the end|at the top|at the bottom|on the top|on the bottom|"
    r"in the middle|top half|bottom half|left half|right half|"
    r"at the left|at the right|on the left|on the right|"
    r"top (of|corner)|bottom (of|corner)|end of|back of|front of)\b"
)

_SUPERLATIVES = (
    r"\b(smallest|largest|biggest|tallest|shortest|closest|farthest|"
    r"widest|narrowest|fastest|slowest|most \w+|least \w+)\b"
    r"|\bthe (first|second|third|last|middle)\b"
)

_RELATIONAL = (
    r"\b(next to|beside|adjacent to|in front of|behind|to the (left|right) of|"
    r"near|hanging on|attached to|connected to|leaning on|on top of|"
    r"which the \w+ is|that the \w+ is)\b"
)


@dataclass
class ClassificationResult:
    expression: str
    rule_modes: list[str]
    primary_mode: str
    llm_mode: str | None = None
    llm_explanation: str | None = None

    @property
    def final_mode(self) -> str:
        return self.llm_mode if self.llm_mode else self.primary_mode


# ── Rule-based classifier ──────────────────────────────────────────────────────

_COLOR_TEXTURE = re.compile(
    r"\b(black|white|grey|gray|red|blue|green|yellow|orange|brown|purple|"
    r"pink|dark|light|golden|silver|striped|spotted|patterned|"
    r"bright|coloured|colored)\b",
    re.IGNORECASE,
)


def _count_visual_attributes(expr: str) -> int:
    """Count color/texture adjectives. ≥4 suggests over-specification."""
    return len(_COLOR_TEXTURE.findall(expr))


def classify_rule_based(expression: str) -> ClassificationResult:
    expr_lower = expression.lower()
    modes = []

    # Check in priority order — most-specific / highest-impact first
    if re.search(_RELATIONAL, expr_lower):
        modes.append("CONTEXT_DISAM")
    if re.search(_SUPERLATIVES, expr_lower):
        modes.append("ORDINAL_SUPERLATIVE")
    if re.search(_SPATIAL_ORDINALS, expr_lower):
        modes.append("ORDINAL_SPATIAL")
    if re.search(_ACTION_VERBS, expr_lower):
        modes.append("ACTION_MOTION")
    if _count_visual_attributes(expression) >= 4:
        modes.append("OVER_SPECIFIC")

    primary = modes[0] if modes else "GOOD"
    return ClassificationResult(
        expression=expression,
        rule_modes=modes,
        primary_mode=primary,
    )


# ── LLM-based classifier (optional refinement) ───────────────────────────────

_SYSTEM_CLASSIFY = """\
You are an expert analyzing natural language expressions for video object grounding.
Given an expression, classify its primary failure mode from this fixed list:

  ACTION_MOTION       – relies on motion/action verbs (jumping, moving, swimming)
  ORDINAL_SPATIAL     – uses frame-relative position (at the end, top half, bottom half)
  ORDINAL_SUPERLATIVE – uses superlatives or ordinal ranks (smallest, middle, first)
  OVER_SPECIFIC       – packs 4+ visual attributes, confusing the model
  WRONG_HEAD_NOUN     – head noun doesn't match the actual object category
  VAGUE_NOUN          – head noun is too generic (bike vs stunt bike, car vs go-cart)
  CONTEXT_DISAM       – relies on relational reference to another object
  GOOD                – no failure mode detected

Key facts from our grounding experiments:
- Nouns and static adjectives (color, texture, shape) drive attention most (40% of frames)
- Action verbs contribute only 3% of oracle attention — they actively mislead the model
- Ordinal/spatial references break completely in long videos (IoU=0 observed)
- Superlatives fail because "smallest" depends on which objects are visible per frame

Output ONLY valid JSON with keys: "mode" (one of the above labels) and "explanation" (≤2 sentences)."""


def classify_with_llm(
    expression: str,
    llm: "QwenTextLLM",
    rule_result: ClassificationResult | None = None,
) -> ClassificationResult:
    if rule_result is None:
        rule_result = classify_rule_based(expression)

    hint = ""
    if rule_result.rule_modes:
        hint = f'\nRule-based pre-classification flagged: {", ".join(rule_result.rule_modes)}. Confirm or override.'

    prompt = (
        f'Expression: "{expression}"{hint}\n\n'
        f"Classify the failure mode and explain briefly."
    )

    parsed = llm.generate_json(prompt, system=_SYSTEM_CLASSIFY)

    if parsed and isinstance(parsed, dict):
        llm_mode = parsed.get("mode", "").strip()
        llm_expl = parsed.get("explanation", "").strip()
        if llm_mode not in FAILURE_MODES:
            llm_mode = rule_result.primary_mode
    else:
        llm_mode = rule_result.primary_mode
        llm_expl = None

    return ClassificationResult(
        expression=expression,
        rule_modes=rule_result.rule_modes,
        primary_mode=rule_result.primary_mode,
        llm_mode=llm_mode,
        llm_explanation=llm_expl,
    )


# ── Batch helper ───────────────────────────────────────────────────────────────

def classify_group(
    expressions: list[str],
    llm: "QwenTextLLM | None" = None,
) -> list[ClassificationResult]:
    results = [classify_rule_based(e) for e in expressions]
    if llm is not None:
        results = [classify_with_llm(r.expression, llm, r) for r in results]
    return results
