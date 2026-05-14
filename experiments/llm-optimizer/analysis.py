"""
analysis.py
-----------
Two data-driven analyses of all (expression, IoU) pairs from results.json.
Both feed into a dynamically-built system prompt for the optimizer.

Phase 1 — SyntacticAnalyzer
    POS-tags every expression using the pos_tagger already in the repo.
    Computes expression-level features (n_verb, n_adj, pos_pattern, …) and
    correlates each feature with IoU across all 244+ pairs.
    Produces a ranked table: which structural properties predict IoU.

Phase 2 — SemanticAnalyzer
    Shows Qwen the full ranked list of (expression, IoU) pairs from the dataset.
    Asks it to discover its own failure taxonomy — no categories are given in advance.
    Asks it to characterise both failure modes AND success patterns.

build_system_prompt(syntactic, semantic)
    Combines both outputs into a coherent system prompt string for the optimizer.
    This replaces the hardcoded _SYSTEM constant in optimizer.py.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qwen_text import QwenTextLLM

# Reuse the repo's POS tagger (spaCy → NLTK → fallback)
_HERE  = Path(__file__).resolve().parent
_POSDOM = _HERE.parent / "pos-dominance"
if str(_POSDOM) not in sys.path:
    sys.path.insert(0, str(_POSDOM))

try:
    from pos_tagger import build_pos_map, tagger_info  # type: ignore
    _HAS_POS_TAGGER = True
except ImportError:
    _HAS_POS_TAGGER = False


# ── helpers ────────────────────────────────────────────────────────────────────

def _words(expr: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", expr.lower())


def _pos_map_safe(expr: str) -> dict[str, str]:
    if _HAS_POS_TAGGER:
        try:
            return build_pos_map(expr)
        except Exception:
            pass
    # minimal fallback: nouns heuristic (last word), rest other
    ws = _words(expr)
    return {w: ("label_noun" if i == len(ws) - 1 else "label_other_pos")
            for i, w in enumerate(ws)}


_POS_ABBR = {
    "label_noun":      "NOUN",
    "label_adj":       "ADJ",
    "label_verb":      "VERB",
    "label_adv":       "ADV",
    "label_other_pos": "OTHER",
}

_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
              "and", "or", "with", "its", "by", "is", "are"}


def _extract_features(expr: str, pos_map: dict[str, str]) -> dict:
    ws = _words(expr)
    content = [w for w in ws if w not in _STOPWORDS]

    counts = Counter(pos_map.get(w, "label_other_pos") for w in content)
    n_noun = counts["label_noun"]
    n_adj  = counts["label_adj"]
    n_verb = counts["label_verb"]
    n_adv  = counts["label_adv"]

    # Simplified POS sequence (deduplicated consecutive tags)
    seq: list[str] = []
    for w in ws:
        if w in _STOPWORDS:
            tag = "DET" if w in {"a", "an", "the"} else "PREP"
        else:
            tag = _POS_ABBR.get(pos_map.get(w, "label_other_pos"), "OTHER")
        if not seq or seq[-1] != tag:
            seq.append(tag)
    pos_pattern = " > ".join(seq)

    # Head noun: last NOUN in expression
    nouns = [w for w in ws if pos_map.get(w) == "label_noun"]
    head_noun = nouns[-1] if nouns else ""

    return {
        "n_words":          len(ws),
        "n_noun":           n_noun,
        "n_adj":            n_adj,
        "n_verb":           n_verb,
        "n_adv":            n_adv,
        "has_verb":         n_verb > 0,
        "has_adj":          n_adj > 0,
        "multi_adj":        n_adj >= 2,
        "adj_heavy":        n_adj >= 3,
        "has_adv":          n_adv > 0,
        "short_expr":       len(ws) <= 4,
        "medium_expr":      5 <= len(ws) <= 7,
        "long_expr":        len(ws) >= 8,
        "noun_only":        n_noun >= 1 and n_verb == 0 and n_adj == 0,
        "adj_noun":         n_adj >= 1 and n_verb == 0,
        "pos_pattern":      pos_pattern,
        "head_noun":        head_noun,
    }


def _feature_correlation(
    triples: list[tuple[str, float, dict]],
    feature: str,
) -> dict:
    present = [iou for _, iou, f in triples if f.get(feature) is True]
    absent  = [iou for _, iou, f in triples if f.get(feature) is False]
    return {
        "n_present":        len(present),
        "n_absent":         len(absent),
        "mean_iou_present": round(float(np.mean(present)), 4) if present else None,
        "mean_iou_absent":  round(float(np.mean(absent)),  4) if absent  else None,
        "delta":            round(float(np.mean(present) - np.mean(absent)), 4)
                            if present and absent else None,
    }


# ── Phase 1: SyntacticAnalyzer ────────────────────────────────────────────────

class SyntacticAnalyzer:
    """
    Correlates POS-level expression features with mean IoU across the full dataset.

    Input : list of (expression, mean_iou) pairs
    Output: {
        "tagger":               str,
        "n_expressions":        int,
        "feature_correlations": {feature: {mean_iou_present, mean_iou_absent, delta, …}},
        "pos_pattern_stats":    [{pattern, mean_iou, count}, …],   # top-N patterns
        "per_expression":       [{expression, iou, features, pos_pattern}, …]
    }
    """

    BOOLEAN_FEATURES = [
        "has_verb", "has_adj", "multi_adj", "adj_heavy", "has_adv",
        "short_expr", "medium_expr", "long_expr",
        "noun_only", "adj_noun",
    ]

    def analyze(self, pairs: list[tuple[str, float]]) -> dict:
        triples: list[tuple[str, float, dict]] = []
        for expr, iou in pairs:
            pm = _pos_map_safe(expr)
            feats = _extract_features(expr, pm)
            triples.append((expr, iou, feats))

        # Boolean feature × IoU correlations
        correlations = {
            f: _feature_correlation(triples, f)
            for f in self.BOOLEAN_FEATURES
        }
        # Sort by absolute delta (most predictive first)
        correlations = dict(sorted(
            correlations.items(),
            key=lambda kv: abs(kv[1]["delta"] or 0),
            reverse=True,
        ))

        # POS pattern stats
        pattern_groups: dict[str, list[float]] = defaultdict(list)
        for _, iou, f in triples:
            pattern_groups[f["pos_pattern"]].append(iou)
        pattern_stats = sorted(
            [
                {
                    "pattern":  p,
                    "mean_iou": round(float(np.mean(vs)), 4),
                    "count":    len(vs),
                }
                for p, vs in pattern_groups.items()
                if len(vs) >= 3  # only patterns with enough data
            ],
            key=lambda x: x["mean_iou"],
            reverse=True,
        )

        per_expr = [
            {
                "expression":  expr,
                "iou":         round(iou, 4),
                "pos_pattern": f["pos_pattern"],
                "features":    {k: v for k, v in f.items() if k != "pos_pattern"},
            }
            for expr, iou, f in triples
        ]

        return {
            "tagger":               tagger_info() if _HAS_POS_TAGGER else "fallback",
            "n_expressions":        len(triples),
            "feature_correlations": correlations,
            "pos_pattern_stats":    pattern_stats[:20],  # top 20
            "per_expression":       per_expr,
        }


# ── Phase 2: SemanticAnalyzer ─────────────────────────────────────────────────

_SEMANTIC_SYSTEM = """\
You are a research assistant analysing natural language expression quality \
for video object grounding.

You will be given a list of expressions used to describe specific objects in \
video sequences, each paired with its mean IoU score (0 = model completely \
missed the object, 1 = perfect localisation).

Your task:
1. Discover a taxonomy of FAILURE modes — what properties make expressions \
   score poorly? Derive the categories purely from the data; do NOT use \
   pre-existing labels.
2. Discover SUCCESS patterns — what properties make expressions score well?
3. Pay attention to sentence structure (noun phrases, verb phrases, \
   prepositional phrases, superlatives, relative clauses) as well as \
   semantic content (static vs dynamic attributes, object category specificity, \
   relational references).

Output ONLY valid JSON (no markdown fences):
{
  "failure_categories": [
    {
      "name": "short label",
      "description": "what this failure is and why it hurts grounding",
      "linguistic_pattern": "the grammatical/syntactic signature (e.g. VP with action verb, PP with spatial anchor)",
      "examples": ["expression 1", "expression 2"],
      "typical_iou_range": "e.g. 0.00–0.15"
    }
  ],
  "success_patterns": [
    {
      "name": "short label",
      "description": "what makes this work",
      "linguistic_pattern": "e.g. DET ADJ NOUN",
      "examples": ["expression 1", "expression 2"],
      "typical_iou_range": "e.g. 0.45–0.75"
    }
  ],
  "key_insight": "one sentence — the single most important finding"
}"""


class SemanticAnalyzer:
    """
    Feeds the full (expression, IoU) dataset to Qwen and asks it to discover
    its own failure taxonomy — no categories are given in advance.
    """

    def __init__(self, top_n_show: int = 120):
        # Limit how many pairs we show (sorted by IoU, evenly sampled if > top_n_show)
        self.top_n_show = top_n_show

    def analyze(self, pairs: list[tuple[str, float]], llm: "QwenTextLLM") -> dict:
        # Sort best → worst; sub-sample evenly if dataset is large
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        if len(sorted_pairs) > self.top_n_show:
            step = len(sorted_pairs) / self.top_n_show
            sorted_pairs = [sorted_pairs[int(i * step)] for i in range(self.top_n_show)]

        lines = [
            f'  IoU={iou:.3f}  "{expr}"'
            for expr, iou in sorted_pairs
        ]
        data_block = "\n".join(lines)

        prompt = (
            f"Here are {len(sorted_pairs)} expressions (from {len(pairs)} total) "
            f"used to describe objects in videos, sorted best → worst by IoU:\n\n"
            f"{data_block}\n\n"
            f"Analyse these expressions and return your discovered taxonomy as JSON."
        )

        parsed = llm.generate_json(prompt, system=_SEMANTIC_SYSTEM)

        if parsed and isinstance(parsed, dict):
            return parsed

        # Fallback if JSON parse failed: return raw text in a wrapper
        raw = llm.generate(prompt, system=_SEMANTIC_SYSTEM)
        return {"raw_response": raw, "parse_error": True}


# ── build_system_prompt ────────────────────────────────────────────────────────

def build_system_prompt(syntactic: dict, semantic: dict) -> str:
    """
    Combines syntactic correlations + Qwen's semantic taxonomy into a
    system prompt for the ExpressionOptimizer.
    """
    lines: list[str] = []
    lines.append(
        "You are an expert in video object grounding with vision-language models.\n"
        "Your task: improve natural language expressions so that a model can better "
        "locate a specific object across all frames of a video.\n"
    )

    # ── Syntactic section ──────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("SYNTACTIC ANALYSIS  "
                 f"(from {syntactic.get('n_expressions', '?')} expressions, "
                 f"tagger: {syntactic.get('tagger', '?')})")
    lines.append("")
    lines.append("Feature impact on mean IoU (Δ = present − absent):")

    corr = syntactic.get("feature_correlations", {})
    for feat, stats in list(corr.items())[:8]:  # top 8 most predictive
        delta = stats.get("delta")
        if delta is None:
            continue
        sign  = "+" if delta >= 0 else ""
        p_iou = stats.get("mean_iou_present", 0) or 0
        a_iou = stats.get("mean_iou_absent",  0) or 0
        np_   = stats.get("n_present", 0)
        lines.append(
            f"  {feat:<20}  present={p_iou:.3f} (n={np_:3d})  "
            f"absent={a_iou:.3f}  Δ={sign}{delta:.3f}"
        )

    lines.append("")
    lines.append("Top POS patterns by mean IoU:")
    for row in syntactic.get("pos_pattern_stats", [])[:8]:
        lines.append(
            f"  mean={row['mean_iou']:.3f}  n={row['count']:3d}  {row['pattern']}"
        )

    # ── Semantic section ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 60)
    if semantic.get("parse_error"):
        lines.append("SEMANTIC ANALYSIS  (raw Qwen response — JSON parse failed)")
        lines.append(semantic.get("raw_response", "")[:800])
    else:
        lines.append("SEMANTIC FAILURE TAXONOMY  (discovered by Qwen from data)")
        for cat in semantic.get("failure_categories", []):
            lines.append(f"\n  [{cat.get('name', '?')}]")
            lines.append(f"    {cat.get('description', '')}")
            lines.append(f"    Pattern : {cat.get('linguistic_pattern', '')}")
            lines.append(f"    IoU range: {cat.get('typical_iou_range', '')}")
            exs = cat.get("examples", [])[:2]
            if exs:
                lines.append(f"    Examples : {' | '.join(exs)}")

        lines.append("")
        lines.append("SUCCESS PATTERNS  (discovered by Qwen from data)")
        for pat in semantic.get("success_patterns", []):
            lines.append(f"\n  [{pat.get('name', '?')}]")
            lines.append(f"    {pat.get('description', '')}")
            lines.append(f"    Pattern : {pat.get('linguistic_pattern', '')}")
            lines.append(f"    IoU range: {pat.get('typical_iou_range', '')}")
            exs = pat.get("examples", [])[:2]
            if exs:
                lines.append(f"    Examples : {' | '.join(exs)}")

        if semantic.get("key_insight"):
            lines.append(f"\nKEY INSIGHT: {semantic['key_insight']}")

    # ── Rules derived from both analyses ──────────────────────────────────────
    lines.append("")
    lines.append("=" * 60)
    lines.append("OPTIMIZATION RULES (derived from both analyses above):")
    lines.append("  ✓ Target the highest-IoU POS patterns (see table above)")
    lines.append("  ✓ Use the correct specific object category as the head noun")
    lines.append("  ✓ Add at most 1-2 static discriminative adjectives")
    lines.append("  ✓ Keep expression short (≤4-6 words where possible)")
    lines.append("  ✗ Avoid verbs — the syntactic analysis shows they strongly hurt IoU")
    lines.append("  ✗ Avoid superlatives, spatial/ordinal anchors, relational clauses")
    lines.append("  ✗ Do not paraphrase failure patterns found in the taxonomy above")
    lines.append("")
    lines.append("Output ONLY valid JSON (no markdown fences).")

    return "\n".join(lines)
