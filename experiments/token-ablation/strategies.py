"""
strategies.py
-------------
Token selection strategies for the GT-mass ablation study.

Each strategy is a callable:

    strategy(ctx: StrategyContext) -> Dict[int, np.ndarray]

that maps sampled_frame_idx → (H_tam, W_tam) averaged attention heatmap.
Frames that cannot be covered by a strategy are absent from the dict.

Context holds a single forward pass worth of TAM data plus GT metadata so
strategies can be compared fairly on identical model outputs.
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Box = Optional[Tuple[int, int, int, int]]

# ── common English stopwords (no external dependency needed) ──────────────────
_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "is", "it",
    "this", "that", "with", "for", "by", "as", "be", "was", "were", "are",
    "he", "she", "his", "her", "its", "their", "there", "has", "have",
    "had", "do", "does", "did", "but", "not", "from", "into", "over",
    "up", "down", "out", "about", "who", "which", "what", "when", "where",
    "while", "if", "then", "so", "no", "can", "will", "just", "also",
    "both", "all", "more", "some", "one", "two", "three", "four",
    "performing", "performs", "wearing",   # high-frequency verbs in DAVIS labels
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_tok(tok: str) -> str:
    """Strip BPE-prefix markers (▁, Ġ) and surrounding whitespace."""
    return tok.replace("▁", "").replace("Ġ", "").strip()


def _is_content_word(tok: str) -> bool:
    """True for alphabetic tokens (≥ 2 chars) that are not stopwords."""
    w = _clean_tok(tok).lower()
    return len(w) >= 2 and w.isalpha() and w not in _STOPWORDS


def _avg_heatmap_3d(
    tam_maps: List[Optional[np.ndarray]],
    token_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Element-wise mean of (T, H, W) tam_maps for the given token indices.
    Returns None if no valid map exists.
    """
    valid = [
        tam_maps[i].astype(np.float32)
        for i in token_indices
        if i < len(tam_maps) and tam_maps[i] is not None
    ]
    if not valid:
        return None
    return np.stack(valid, axis=0).mean(axis=0)   # (T, H, W)


def _find_field_token_indices(
    gen_tokens: List[str],
    field_name: str,
    search_from: int = 0,
) -> Tuple[List[int], int]:
    """
    In the concatenated token stream, locate the JSON value of `field_name`
    after `search_from` and return (token_indices_of_value, position_of_key).
    Returns ([], search_from) if not found.
    """
    raw = "".join(gen_tokens)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in gen_tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)

    key = f'"{field_name}"'
    key_pos = raw.find(key, search_from)
    if key_pos == -1:
        return [], search_from

    colon_pos = raw.find(":", key_pos + len(key))
    if colon_pos == -1:
        return [], search_from

    open_q = raw.find('"', colon_pos + 1)
    close_q = raw.find('"', open_q + 1) if open_q != -1 else -1

    if open_q != -1 and close_q != -1:
        # String value
        val_start, val_end = open_q + 1, close_q
    else:
        # Numeric / array value: take everything up to the next comma or }
        val_start = colon_pos + 1
        while val_start < len(raw) and raw[val_start] in " \t\n":
            val_start += 1
        # Find end: next comma or closing bracket/brace not inside nested []
        depth = 0
        val_end = val_start
        while val_end < len(raw):
            c = raw[val_end]
            if c in "[({":
                depth += 1
            elif c in "])}":
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            val_end += 1

    idxs = [
        i for i, (s, e) in enumerate(spans)
        if s < val_end and e > val_start
    ]
    return idxs, key_pos


# ── context ───────────────────────────────────────────────────────────────────

@dataclass
class StrategyContext:
    gen_tokens:      List[str]                     # all generated tokens
    tam_maps:        List[Optional[np.ndarray]]    # (T, H, W) per token
    vision_T:        int                           # number of sampled frames
    label_token_map: List[Tuple[int, List[int]]]  # (sampled_t, [tok_idxs])
    gen_text:        str                           # raw decoded output
    gt_boxes:        List[Box]                     # per original frame
    frame_H:         int
    frame_W:         int
    sample_rate:     int
    expression:      str = ""                      # query expression for this item
    rng_seed:        int = 42


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, "StrategyFn"] = {}


def register(name: str):
    def decorator(fn):
        STRATEGIES[name] = fn
        return fn
    return decorator


# ── 1. whole_label ────────────────────────────────────────────────────────────

@register("whole_label")
def strategy_whole_label(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Average all tokens between the label quotes for each detected frame."""
    result: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T or not tok_idxs:
            continue
        hm3d = _avg_heatmap_3d(ctx.tam_maps, tok_idxs)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 2. first_word ─────────────────────────────────────────────────────────────

@register("first_word")
def strategy_first_word(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Tokens of the first whitespace-delimited word of each frame's label."""
    result: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T or not tok_idxs:
            continue
        # collect tokens until we hit a word boundary after the first word
        first_word_idxs: List[int] = []
        seen_content = False
        for i in tok_idxs:
            w = _clean_tok(ctx.gen_tokens[i])
            if w == "":
                if seen_content:
                    break
                continue
            if w[0] == " " and seen_content:
                break
            if w.replace(" ", ""):
                seen_content = True
                first_word_idxs.append(i)
        hm3d = _avg_heatmap_3d(ctx.tam_maps, first_word_idxs or tok_idxs[:1])
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 3. last_word ──────────────────────────────────────────────────────────────

@register("last_word")
def strategy_last_word(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Tokens of the last whitespace-delimited word of each frame's label."""
    result: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T or not tok_idxs:
            continue
        last_word_idxs: List[int] = []
        for i in reversed(tok_idxs):
            w = _clean_tok(ctx.gen_tokens[i])
            if w == "":
                if last_word_idxs:
                    break
                continue
            if w[0] == " " and last_word_idxs:
                break
            if w.replace(" ", ""):
                last_word_idxs.insert(0, i)
        hm3d = _avg_heatmap_3d(ctx.tam_maps, last_word_idxs or tok_idxs[-1:])
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 4. content_words_in_label ─────────────────────────────────────────────────

@register("content_words_in_label")
def strategy_content_words_in_label(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Non-stopword alphabetic tokens within each frame's label value."""
    result: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T or not tok_idxs:
            continue
        content = [i for i in tok_idxs if _is_content_word(ctx.gen_tokens[i])]
        hm3d = _avg_heatmap_3d(ctx.tam_maps, content or tok_idxs)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 5. all_content_tokens ─────────────────────────────────────────────────────

@register("all_content_tokens")
def strategy_all_content_tokens(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Non-stopword alphabetic tokens from anywhere in generated output (same heatmap all frames)."""
    idxs = [
        i for i, tok in enumerate(ctx.gen_tokens)
        if _is_content_word(tok) and i < len(ctx.tam_maps) and ctx.tam_maps[i] is not None
    ]
    hm3d = _avg_heatmap_3d(ctx.tam_maps, idxs)
    if hm3d is None:
        return {}
    return {t: hm3d[t] for t in range(ctx.vision_T)}


# ── 6. bbox_tokens ────────────────────────────────────────────────────────────

@register("bbox_tokens")
def strategy_bbox_tokens(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Tokens that form the bbox_2d coordinate values for each detected frame."""
    result: Dict[int, np.ndarray] = {}
    search_from = 0
    for sampled_t, _ in ctx.label_token_map:
        if sampled_t >= ctx.vision_T:
            continue
        bbox_idxs, key_pos = _find_field_token_indices(
            ctx.gen_tokens, "bbox_2d", search_from
        )
        if not bbox_idxs:
            continue
        search_from = key_pos + 8
        hm3d = _avg_heatmap_3d(ctx.tam_maps, bbox_idxs)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 7. frame_tokens ───────────────────────────────────────────────────────────

@register("frame_tokens")
def strategy_frame_tokens(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Tokens that encode the integer frame index value (video mode: 'frame' key)."""
    result: Dict[int, np.ndarray] = {}
    search_from = 0
    for sampled_t, _ in ctx.label_token_map:
        if sampled_t >= ctx.vision_T:
            continue
        frame_idxs, key_pos = _find_field_token_indices(
            ctx.gen_tokens, "frame", search_from
        )
        if not frame_idxs:
            continue
        search_from = key_pos + 7
        hm3d = _avg_heatmap_3d(ctx.tam_maps, frame_idxs)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 7b. time_tokens ───────────────────────────────────────────────────────────

@register("time_tokens")
def strategy_time_tokens(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """
    Tokens that encode the timestamp value (image mode: 'time' key, e.g. 0.33).
    Distinct from frame_tokens: only fires when the model outputs 'time' rather
    than 'frame', so at most one of {frame_tokens, time_tokens} will be non-empty
    for a given forward pass.
    """
    result: Dict[int, np.ndarray] = {}
    search_from = 0
    for sampled_t, _ in ctx.label_token_map:
        if sampled_t >= ctx.vision_T:
            continue
        time_idxs, key_pos = _find_field_token_indices(
            ctx.gen_tokens, "time", search_from
        )
        if not time_idxs:
            continue
        search_from = key_pos + 6
        hm3d = _avg_heatmap_3d(ctx.tam_maps, time_idxs)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 8. label_and_bbox ─────────────────────────────────────────────────────────

@register("label_and_bbox")
def strategy_label_and_bbox(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Union of label tokens and bbox_2d tokens for each detected frame."""
    label_hms  = strategy_whole_label(ctx)
    bbox_hms   = strategy_bbox_tokens(ctx)
    result: Dict[int, np.ndarray] = {}
    for sampled_t, label_tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T:
            continue
        search_from = 0
        bbox_idxs, key_pos = _find_field_token_indices(
            ctx.gen_tokens, "bbox_2d", search_from
        )
        combined = list(set(label_tok_idxs + bbox_idxs))
        hm3d = _avg_heatmap_3d(ctx.tam_maps, combined)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── 9. label_and_frame ────────────────────────────────────────────────────────

@register("label_and_frame")
def strategy_label_and_frame(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Union of label tokens and temporal-index tokens ('frame' or 'time') per detection."""
    result: Dict[int, np.ndarray] = {}
    search_from = 0
    for sampled_t, label_tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T:
            continue
        frame_idxs, key_pos = _find_field_token_indices(
            ctx.gen_tokens, "frame", search_from
        )
        if not frame_idxs:
            frame_idxs, key_pos = _find_field_token_indices(
                ctx.gen_tokens, "time", search_from
            )
        search_from = max(search_from, key_pos + 7) if key_pos > search_from else search_from + 1
        combined = list(set(label_tok_idxs + frame_idxs))
        hm3d = _avg_heatmap_3d(ctx.tam_maps, combined)
        if hm3d is not None:
            result[sampled_t] = hm3d[sampled_t]
    return result


# ── label-noun aggregation helpers ───────────────────────────────────────────

def _expression_words(expression: str) -> frozenset:
    import re
    return frozenset(re.findall(r"[a-zA-Z]+", expression.lower()))


def _label_noun_indices(tok_idxs: List[int], gen_tokens: List[str],
                        expr_words: frozenset) -> List[int]:
    """Filter tok_idxs to those that are content words appearing in the expression."""
    result = []
    for i in tok_idxs:
        w = _clean_tok(gen_tokens[i]).lower()
        if _is_content_word(w) and w in expr_words:
            result.append(i)
    return result or tok_idxs          # fall back to all label tokens if none qualify


def _valid_maps(tam_maps, tok_idxs):
    """Return list of (H, W) float32 slices for a given frame dimension, skipping None."""
    return [
        tam_maps[i].astype(np.float32)
        for i in tok_idxs
        if i < len(tam_maps) and tam_maps[i] is not None
    ]


def _agg_mean(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    if not maps: return None
    return np.stack(maps).mean(axis=0)


def _agg_max(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    """Pixel-wise maximum — any token attending to a region pulls it up."""
    if not maps: return None
    return np.stack(maps).max(axis=0)


def _agg_geomean(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    """Geometric mean — only regions attended by *all* tokens score high."""
    if not maps: return None
    log_sum = np.zeros_like(maps[0])
    for m in maps:
        log_sum += np.log(m + 1e-8)
    return np.exp(log_sum / len(maps))


def _agg_weighted_entropy(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Weight each token map by the inverse of its spatial entropy — more focused
    (peakier) maps contribute more to the aggregate.
    """
    if not maps: return None
    weights = []
    for m in maps:
        flat = m.ravel().astype(np.float64)
        total = flat.sum()
        if total > 0:
            p = flat / total
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p)))
        else:
            entropy = 1.0
        weights.append(1.0 / (entropy + 1e-8))
    w = np.array(weights)
    w /= w.sum()
    return sum(wi * m for wi, m in zip(w, maps))


def _agg_weighted_peak(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    """Weight each token map by its peak activation — brighter maps lead."""
    if not maps: return None
    weights = np.array([float(m.max()) for m in maps])
    total = weights.sum()
    if total == 0:
        return _agg_mean(maps)
    weights /= total
    return sum(wi * m for wi, m in zip(weights, maps))


def _agg_top1(maps: List[np.ndarray]) -> Optional[np.ndarray]:
    """Single map with the highest total activation (no averaging)."""
    if not maps: return None
    return max(maps, key=lambda m: float(m.sum()))


def _apply_agg(agg_fn, ctx, sampled_t, tok_idxs):
    """Extract per-frame slices for tok_idxs, run agg_fn, return (H_tam, W_tam) or None."""
    frame_maps = []
    for i in tok_idxs:
        if i >= len(ctx.tam_maps) or ctx.tam_maps[i] is None:
            continue
        tm = ctx.tam_maps[i]
        if tm.ndim == 3 and sampled_t < tm.shape[0]:
            frame_maps.append(tm[sampled_t].astype(np.float32))
    return agg_fn(frame_maps)


def _label_noun_strategy(agg_fn, ctx: StrategyContext) -> Dict[int, np.ndarray]:
    expr_words = _expression_words(ctx.expression)
    result: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in ctx.label_token_map:
        if sampled_t >= ctx.vision_T or not tok_idxs:
            continue
        noun_idxs = _label_noun_indices(tok_idxs, ctx.gen_tokens, expr_words)
        hm = _apply_agg(agg_fn, ctx, sampled_t, noun_idxs)
        if hm is not None:
            result[sampled_t] = hm
    return result


# ── label-noun aggregation strategies ────────────────────────────────────────

@register("label_nouns_mean")
def strategy_label_nouns_mean(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Label-noun tokens, averaged (add-then-divide)."""
    return _label_noun_strategy(_agg_mean, ctx)


@register("label_nouns_max")
def strategy_label_nouns_max(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Label-noun tokens, pixel-wise maximum across token maps."""
    return _label_noun_strategy(_agg_max, ctx)


@register("label_nouns_geomean")
def strategy_label_nouns_geomean(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Label-noun tokens, geometric mean — rewards spatial consensus."""
    return _label_noun_strategy(_agg_geomean, ctx)


@register("label_nouns_weighted_entropy")
def strategy_label_nouns_weighted_entropy(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Label-noun tokens, weighted by inverse spatial entropy (focused maps dominate)."""
    return _label_noun_strategy(_agg_weighted_entropy, ctx)


@register("label_nouns_weighted_peak")
def strategy_label_nouns_weighted_peak(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Label-noun tokens, weighted by peak activation value."""
    return _label_noun_strategy(_agg_weighted_peak, ctx)


@register("label_nouns_top1")
def strategy_label_nouns_top1(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Single label-noun token with highest total activation (no averaging)."""
    return _label_noun_strategy(_agg_top1, ctx)


# ── 10. per_frame_oracle ──────────────────────────────────────────────────────

@register("per_frame_oracle")
def strategy_per_frame_oracle(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """
    Upper bound: for each sampled frame, use the single token whose heatmap
    maximises GT mass in that frame.
    """
    from strategies import _compute_mass_in_gt_fast

    result: Dict[int, np.ndarray] = {}
    for sampled_t in range(ctx.vision_T):
        orig_t = sampled_t * ctx.sample_rate
        if orig_t >= len(ctx.gt_boxes) or ctx.gt_boxes[orig_t] is None:
            continue
        gt_box = ctx.gt_boxes[orig_t]
        best_mass = -1.0
        best_hm: Optional[np.ndarray] = None
        for i, tm in enumerate(ctx.tam_maps):
            if tm is None or tm.ndim != 3 or sampled_t >= tm.shape[0]:
                continue
            hm = tm[sampled_t].astype(np.float32)
            mass = _compute_mass_in_gt_fast(hm, gt_box, ctx.frame_H, ctx.frame_W)
            if mass > best_mass:
                best_mass = mass
                best_hm = hm
        if best_hm is not None:
            result[sampled_t] = best_hm
    return result


# ── 11. global_best_token ─────────────────────────────────────────────────────

@register("global_best_token")
def strategy_global_best_token(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """
    Single token with highest mean GT mass across all sampled frames
    (oracle over token choice, fixed across frames).
    """
    from strategies import _compute_mass_in_gt_fast

    token_scores: List[Tuple[float, int]] = []
    for i, tm in enumerate(ctx.tam_maps):
        if tm is None or tm.ndim != 3:
            continue
        masses = []
        for sampled_t in range(min(ctx.vision_T, tm.shape[0])):
            orig_t = sampled_t * ctx.sample_rate
            if orig_t >= len(ctx.gt_boxes) or ctx.gt_boxes[orig_t] is None:
                continue
            hm = tm[sampled_t].astype(np.float32)
            masses.append(
                _compute_mass_in_gt_fast(hm, ctx.gt_boxes[orig_t], ctx.frame_H, ctx.frame_W)
            )
        if masses:
            token_scores.append((float(np.mean(masses)), i))

    if not token_scores:
        return {}

    _, best_idx = max(token_scores)
    tm = ctx.tam_maps[best_idx]
    return {t: tm[t].astype(np.float32) for t in range(min(ctx.vision_T, tm.shape[0]))}


# ── 12. all_tokens_mean ───────────────────────────────────────────────────────

@register("all_tokens_mean")
def strategy_all_tokens_mean(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Mean over all non-None generated token maps (global average baseline)."""
    valid_idxs = [
        i for i, tm in enumerate(ctx.tam_maps)
        if tm is not None and tm.ndim == 3
    ]
    hm3d = _avg_heatmap_3d(ctx.tam_maps, valid_idxs)
    if hm3d is None:
        return {}
    return {t: hm3d[t] for t in range(ctx.vision_T)}


# ── 13. random_tokens ─────────────────────────────────────────────────────────

@register("random_tokens")
def strategy_random_tokens(ctx: StrategyContext) -> Dict[int, np.ndarray]:
    """Random sample of 5 non-None tokens (averaged, fixed seed)."""
    rng = random.Random(ctx.rng_seed)
    valid_idxs = [
        i for i, tm in enumerate(ctx.tam_maps)
        if tm is not None and tm.ndim == 3
    ]
    k = min(5, len(valid_idxs))
    chosen = rng.sample(valid_idxs, k)
    hm3d = _avg_heatmap_3d(ctx.tam_maps, chosen)
    if hm3d is None:
        return {}
    return {t: hm3d[t] for t in range(ctx.vision_T)}


# ── oracle token analysis ─────────────────────────────────────────────────────

def analyze_oracle_tokens(ctx: StrategyContext) -> List[dict]:
    """
    For every sampled frame that has a GT box, rank all tokens by GT mass and
    return the full ranked list per frame.

    Returns
    -------
    List of dicts, one per (frame, token) pair that has a valid TAM map:
        sampled_t    : int   — sampled frame index
        orig_t       : int   — original frame index
        tok_idx      : int   — token position in generated output
        token        : str   — raw token string (includes BPE markers)
        token_clean  : str   — token string with BPE markers stripped
        gt_mass      : float — fraction of heatmap activation inside GT box
        rank         : int   — 1 = oracle winner for this frame
    """
    rows: List[dict] = []
    for sampled_t in range(ctx.vision_T):
        orig_t = sampled_t * ctx.sample_rate
        if orig_t >= len(ctx.gt_boxes) or ctx.gt_boxes[orig_t] is None:
            continue
        gt_box = ctx.gt_boxes[orig_t]

        frame_scores: List[Tuple[float, int]] = []
        for i, tm in enumerate(ctx.tam_maps):
            if tm is None or tm.ndim != 3 or sampled_t >= tm.shape[0]:
                continue
            hm = tm[sampled_t].astype(np.float32)
            mass = _compute_mass_in_gt_fast(hm, gt_box, ctx.frame_H, ctx.frame_W)
            frame_scores.append((mass, i))

        frame_scores.sort(reverse=True)
        for rank, (mass, tok_idx) in enumerate(frame_scores, start=1):
            rows.append({
                "sampled_t":   sampled_t,
                "orig_t":      orig_t,
                "tok_idx":     tok_idx,
                "token":       ctx.gen_tokens[tok_idx],
                "token_clean": _clean_tok(ctx.gen_tokens[tok_idx]),
                "gt_mass":     float(mass),
                "rank":        rank,
            })
    return rows


# ── helper exposed to strategy functions ─────────────────────────────────────

def _compute_mass_in_gt_fast(
    heatmap: np.ndarray,
    gt_box: Tuple[int, int, int, int],
    frame_H: int,
    frame_W: int,
) -> float:
    """Fraction of heatmap activation inside gt_box (scaled to heatmap resolution)."""
    total = float(heatmap.sum())
    if total == 0:
        return 0.0
    H_tam, W_tam = heatmap.shape
    x1, y1, x2, y2 = gt_box
    hx1 = max(0, int(x1 * W_tam / frame_W))
    hx2 = min(W_tam, int(x2 * W_tam / frame_W))
    hy1 = max(0, int(y1 * H_tam / frame_H))
    hy2 = min(H_tam, int(y2 * H_tam / frame_H))
    inside = float(heatmap[hy1:hy2, hx1:hx2].sum())
    return inside / total
