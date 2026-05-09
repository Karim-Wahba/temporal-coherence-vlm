"""
experiment.py
-------------
Core runner for the POS dominance experiment.

For each (sequence, expression) item:
1. Runs QwenVOTRunner.run_with_tam() — one forward pass.
2. Parses JSON output to locate label tokens.
3. Assigns a POS-aware category to every generated token using:
     - JSON context (bbox_coord, frame_index, json_structure …)
     - POS tag from the expression (label_noun, label_adj, label_verb, label_adv …)
4. For each sampled frame with a valid GT box, records which token category
   is the oracle winner (highest GT mass in box) and its normalised frame
   position within the sequence [0, 1].
5. Aggregates per-sequence dominance statistics.
"""

import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
_REF_DAVIS  = _EXPERIMENTS / "Ref-DAVIS"
_GS         = _EXPERIMENTS / "grounding-stability"
_ABLATION   = _EXPERIMENTS / "token-ablation"

# Insert dependency paths; _HERE is always placed at position 0 last so
# our local modules (visualizer, pos_tagger) shadow same-named siblings.
for p in [str(_REF_DAVIS), str(_REF_DAVIS / "benchmark"),
          str(_GS), str(_ABLATION)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Re-insert _HERE unconditionally so it always wins name conflicts.
if str(_HERE) in sys.path:
    sys.path.remove(str(_HERE))
sys.path.insert(0, str(_HERE))

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem  # noqa: E402
from token_parser import parse_frame_labels, find_label_token_indices  # noqa: E402
from strategies import StrategyContext, analyze_oracle_tokens            # noqa: E402
from pos_tagger import build_pos_map, token_to_pos_category             # noqa: E402


# ── token categorisation helpers (mirrors categorize.py, kept local) ──────────

_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "and", "or", "to", "is", "it",
    "this", "that", "with", "for", "by", "as", "be", "was", "were", "are",
    "he", "she", "his", "her", "its", "their", "there", "has", "have",
    "had", "do", "does", "did", "but", "not", "from", "into", "over",
    "up", "down", "out", "about", "who", "which", "what", "when", "where",
    "while", "if", "then", "so", "no", "can", "will", "just", "also",
    "both", "all", "more", "some", "one", "two", "three", "four",
})

_JSON_STRUCTURAL = frozenset({"{", "}", "[", "]", ":", ",", '"', "'", "``", "''"})
_JSON_KEYS       = frozenset({"frame", "time", "bbox_2d", "bbox", "label", "object"})


def _clean(tok: str) -> str:
    return tok.replace("▁", "").replace("Ġ", "").strip()


def _is_content_word(w: str) -> bool:
    return len(w) >= 2 and w.isalpha() and w not in _STOPWORDS


def _looks_numeric(w: str) -> bool:
    try:
        float(w.replace(",", ""))
        return True
    except ValueError:
        return False


def _expression_words(expression: str) -> frozenset:
    return frozenset(re.findall(r"[a-zA-Z]+", expression.lower()))


def _build_context_labels(gen_tokens: List[str]) -> List[str]:
    """
    Tag each token position with its JSON structural role:
    'bbox_coord' | 'frame_value' | 'time_value' | 'label_value' |
    'json_key' | 'json_structure' | 'other'
    """
    raw   = "".join(gen_tokens)
    spans: List[Tuple[int, int]] = []
    pos   = 0
    for tok in gen_tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)

    labels = ["other"] * len(gen_tokens)

    def _tag_range(start_char: int, end_char: int, tag: str):
        for i, (s, e) in enumerate(spans):
            if s < end_char and e > start_char:
                labels[i] = tag

    for m in re.finditer(r'"(bbox_2d|frame|time|label)"(\s*):(\s*)', raw):
        key_name = m.group(1)
        after    = m.end()

        if key_name == "bbox_2d":
            ob = raw.find("[", after)
            if ob != -1:
                cb = raw.find("]", ob)
                if cb != -1:
                    _tag_range(ob, cb + 1, "bbox_coord")

        elif key_name in ("frame", "time"):
            depth, end = 0, after
            while end < len(raw):
                c = raw[end]
                if c in "[({":   depth += 1
                elif c in "])}":
                    if depth == 0: break
                    depth -= 1
                elif c == "," and depth == 0: break
                end += 1
            _tag_range(after, end,
                       "frame_value" if key_name == "frame" else "time_value")

        elif key_name == "label":
            oq = raw.find('"', after)
            if oq != -1:
                cq = raw.find('"', oq + 1)
                if cq != -1:
                    _tag_range(oq + 1, cq, "label_value")

    for i, tok in enumerate(gen_tokens):
        w = _clean(tok)
        if (w in _JSON_STRUCTURAL or w in _JSON_KEYS) and labels[i] == "other":
            labels[i] = "json_structure"

    return labels


# ── POS-aware categorisation ──────────────────────────────────────────────────

#: Full ordered list of categories — order defines stacking / color order in plots.
CATEGORY_ORDER: List[str] = [
    "label_noun",
    "label_adj",
    "label_verb",
    "label_adv",
    "label_other_pos",
    "label_function",
    "bbox_coord",
    "frame_index",
    "other_content",
    "stopword",
    "json_structure",
    "special",
]

CATEGORY_COLORS: Dict[str, str] = {
    "label_noun":      "#27ae60",   # dark green
    "label_adj":       "#16a085",   # teal
    "label_verb":      "#2980b9",   # blue
    "label_adv":       "#8e44ad",   # purple
    "label_other_pos": "#a8e6cf",   # pale teal
    "label_function":  "#c8e6c9",   # pale green
    "bbox_coord":      "#e74c3c",   # red
    "frame_index":     "#e67e22",   # orange
    "other_content":   "#5dade2",   # light blue
    "stopword":        "#95a5a6",   # grey
    "json_structure":  "#bdc3c7",   # light grey
    "special":         "#ecf0f1",   # near-white
}

N_BINS = 10


def categorize_token_with_pos(
    tok_idx:       int,
    token:         str,
    token_clean:   str,
    expression:    str,
    context_labels: List[str],
    pos_map:       Dict[str, str],
) -> str:
    """
    Assign a POS-aware category to one token.

    The decision tree mirrors categorize.categorize_token but splits the
    'label_noun' bucket into label_noun / label_adj / label_verb / label_adv /
    label_other_pos using the POS map built from the expression.
    """
    w   = token_clean.lower()
    ctx = context_labels[tok_idx] if tok_idx < len(context_labels) else "other"

    # special / empty
    if not w or "<|" in token or token.strip() == "":
        return "special"

    # JSON structure
    if ctx == "json_structure" or w in _JSON_STRUCTURAL or w in _JSON_KEYS:
        return "json_structure"

    # bounding-box coordinate
    if ctx == "bbox_coord":
        return "bbox_coord"

    # frame / time index
    if ctx in ("frame_value", "time_value"):
        return "frame_index"

    # remaining numeric → treat as bbox coord
    if _looks_numeric(w):
        return "bbox_coord"

    # word tokens — check expression membership
    expr_words = _expression_words(expression)
    if w in expr_words:
        if _is_content_word(w):
            return token_to_pos_category(w, pos_map)   # POS-aware split
        return "label_function"

    # not in expression
    if _is_content_word(w):
        return "other_content"
    if w in _STOPWORDS:
        return "stopword"

    return "json_structure"   # catch-all for stray punctuation


# ── experiment runner ─────────────────────────────────────────────────────────

class POSDominanceExperiment:
    """
    Parameters
    ----------
    runner     : QwenVOTRunner (or _DryRunRunner for layout testing)
    davis_root : path to the DAVIS dataset root
    save_dir   : directory for results.json and plots
    split      : "valid" or "train"
    """

    def __init__(self, runner, davis_root: str, save_dir: str, split: str = "valid"):
        self.runner     = runner
        self.davis_root = davis_root
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.split      = split

    # ── single item ───────────────────────────────────────────────────────────

    def run_item(self, item: DAVISVOTItem) -> dict:
        H, W = item.frame_size()
        prefix = f"{item.seq_name} exp{item.exp_id}"

        try:
            boxes, gen_text, tam_result = self.runner.run_with_tam(
                item.frames_pil, item.expression
            )
        except Exception as e:
            traceback.print_exc()
            return {
                "seq_name":  item.seq_name,
                "exp_id":    item.exp_id,
                "expression": item.expression,
                "error":     str(e),
            }

        gen_tokens = tam_result["gen_tokens"]
        tam_maps   = tam_result["tam_maps"]
        vision_T   = tam_result["vision_shape"][0]

        parsed          = parse_frame_labels(gen_text, fps=self.runner.fps,
                                             sample_rate=self.runner.sample_rate)
        label_token_map = find_label_token_indices(gen_tokens, parsed)

        ctx = StrategyContext(
            gen_tokens=gen_tokens, tam_maps=tam_maps, vision_T=vision_T,
            label_token_map=label_token_map, gen_text=gen_text,
            gt_boxes=item.gt_boxes, frame_H=H, frame_W=W,
            sample_rate=self.runner.sample_rate, expression=item.expression,
        )

        oracle_rows = analyze_oracle_tokens(ctx)

        ctx_labels = _build_context_labels(gen_tokens)
        pos_map    = build_pos_map(item.expression)

        for row in oracle_rows:
            row["category"] = categorize_token_with_pos(
                tok_idx=row["tok_idx"], token=row["token"],
                token_clean=row["token_clean"], expression=item.expression,
                context_labels=ctx_labels, pos_map=pos_map,
            )

        # ── per-frame category data (pairwise / position analyses) ────────────
        # Group oracle rows by frame (already rank-sorted within each frame).
        by_t: Dict[int, List[dict]] = defaultdict(list)
        for row in oracle_rows:
            by_t[row["sampled_t"]].append(row)

        # Per-sequence normalisation is needed by both per_frame_cat_best
        # (for unconditional position profiles) and per_frame_tokens.
        sampled_ts_with_data = sorted(by_t.keys())
        max_t_seq = sampled_ts_with_data[-1] if sampled_ts_with_data else 0

        top2_per_frame: List[dict] = []
        per_frame_cat_best: List[dict] = []
        per_frame_tokens:   List[dict] = []
        for t in sampled_ts_with_data:
            rows_t = by_t[t]   # rank-sorted
            norm_pos = t / max(max_t_seq, 1)
            bin_idx  = min(N_BINS - 1, int(norm_pos * N_BINS))

            # top-2 categories
            entry: dict = {"sampled_t": t}
            if rows_t:
                entry["rank1_cat"]  = rows_t[0]["category"]
                entry["rank1_mass"] = rows_t[0]["gt_mass"]
            if len(rows_t) >= 2:
                entry["rank2_cat"]  = rows_t[1]["category"]
                entry["rank2_mass"] = rows_t[1]["gt_mass"]
            top2_per_frame.append(entry)

            # best mass per category (first occurrence in rank order = highest mass)
            cat_best: Dict[str, float] = {}
            for row in rows_t:
                if row["category"] not in cat_best:
                    cat_best[row["category"]] = row["gt_mass"]
            per_frame_cat_best.append({
                "sampled_t": t,
                "norm_pos":  norm_pos,
                "bin_idx":   bin_idx,
                "cat_best":  cat_best,
            })

            # full per-frame token list, sorted by tok_idx for adjacency analysis
            rows_by_idx = sorted(rows_t, key=lambda r: r["tok_idx"])
            per_frame_tokens.append({
                "sampled_t": t,
                "norm_pos":  norm_pos,
                "bin_idx":   bin_idx,
                "tokens": [
                    {"tok_idx":  r["tok_idx"],
                     "category": r["category"],
                     "gt_mass":  float(r["gt_mass"])}
                    for r in rows_by_idx
                ],
            })

        # winners = oracle token (rank 1) per frame
        winners = [r for r in oracle_rows if r["rank"] == 1]
        max_t   = max((r["sampled_t"] for r in winners), default=0)
        for r in winners:
            r["norm_pos"] = r["sampled_t"] / max(max_t, 1)
            r["bin_idx"]  = min(N_BINS - 1, int(r["norm_pos"] * N_BINS))

        # per-sequence dominance summary
        dom: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "masses": []})
        for r in winners:
            dom[r["category"]]["count"] += 1
            dom[r["category"]]["masses"].append(r["gt_mass"])
        total = len(winners)

        dominance_by_category = {
            cat: {
                "count":        v["count"],
                "fraction":     v["count"] / total if total else 0.0,
                "mean_gt_mass": float(np.mean(v["masses"])),
            }
            for cat, v in dom.items()
        }
        primary_category = max(dom, key=lambda c: dom[c]["count"]) if dom else "none"

        print(f"  {prefix}: {len(winners)} frames  primary={primary_category}"
              f"  pos_map={pos_map}")

        return {
            "seq_name":               item.seq_name,
            "exp_id":                 item.exp_id,
            "expression":             item.expression,
            "num_frames":             item.num_frames,
            "vision_T":               vision_T,
            "pos_map":                pos_map,
            "oracle_winners":         winners,
            "top2_per_frame":         top2_per_frame,
            "per_frame_cat_best":     per_frame_cat_best,
            "per_frame_tokens":       per_frame_tokens,
            "dominance_by_category":  dominance_by_category,
            "primary_category":       primary_category,
        }

    # ── full dataset ──────────────────────────────────────────────────────────

    def run_all(
        self,
        sequences:           Optional[List[str]] = None,
        max_sequences:       Optional[int] = None,
        expressions_per_seq: int = 1,
    ) -> List[dict]:
        loader = DAVISVOTLoader(
            davis_root=self.davis_root,
            split=self.split,
            sequences=sequences,
            expressions_per_seq=expressions_per_seq,
        )
        items = list(loader)

        if max_sequences is not None:
            seen:     set = set()
            filtered: List = []
            for it in items:
                if it.seq_name not in seen:
                    if len(seen) >= max_sequences:
                        break
                    seen.add(it.seq_name)
                filtered.append(it)
            items = filtered

        results: List[dict] = []
        for idx, item in enumerate(items):
            print(f"\n[{idx+1}/{len(items)}] {item.seq_name}  "
                  f"\"{item.expression[:60]}\"")
            res = self.run_item(item)
            results.append(res)
            with open(self.save_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

        return results

    # ── aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: List[dict]) -> dict:
        valid    = [r for r in results if "error" not in r]
        n_errors = len(results) - len(valid)

        all_winners: List[dict] = []
        for r in valid:
            all_winners.extend(r.get("oracle_winners", []))

        total = len(all_winners)

        # overall per-category statistics
        dom: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "masses": []})
        for w in all_winners:
            dom[w["category"]]["count"] += 1
            dom[w["category"]]["masses"].append(w["gt_mass"])

        overall_dominance: Dict[str, dict] = {}
        for cat in CATEGORY_ORDER:
            if cat not in dom:
                continue
            v = dom[cat]
            overall_dominance[cat] = {
                "count":        v["count"],
                "fraction":     v["count"] / total if total else 0.0,
                "mean_gt_mass": float(np.mean(v["masses"])),
            }

        # temporal bins: which category wins at each normalised position
        bin_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(N_BINS)]
        bin_n:      List[int]            = [0] * N_BINS
        for w in all_winners:
            b = w.get("bin_idx", 0)
            if 0 <= b < N_BINS:
                bin_counts[b][w["category"]] += 1
                bin_n[b] += 1

        temporal_bins = []
        for b in range(N_BINS):
            n = bin_n[b]
            temporal_bins.append({
                "bin_idx":  b,
                "range":    f"{b*10}-{b*10+9}%",
                "n_frames": n,
                "dominance": {
                    cat: bin_counts[b].get(cat, 0) / n if n else 0.0
                    for cat in CATEGORY_ORDER
                },
            })

        return {
            "n_sequences":       len(valid),
            "n_errors":          n_errors,
            "n_frames":          total,
            "overall_dominance": overall_dominance,
            "temporal_bins":     temporal_bins,
        }

    @staticmethod
    def compute_pairwise(results: List[dict]) -> dict:
        """
        For every pair of categories (A, B) — including A==B (diagonal) —
        compute the mean combined GT-mass when both are present in the same frame.

        Combined mass ≈ (best_mass_A + best_mass_B) / 2.  This is exact under
        the assumption that mass_in_gt is linear in the heatmap (which holds
        when both heatmaps are normalised to the same total before averaging).

        Diagonal cells (A==A) give the single-category mean best mass.

        Returns a dict keyed by "cat_A|cat_B" (upper triangle + diagonal only;
        use symmetry to fill the full matrix in the visualiser).
        """
        pair_sums:   Dict[str, float] = defaultdict(float)
        pair_counts: Dict[str, int]   = defaultdict(int)

        for r in results:
            if "error" in r:
                continue
            for frame_info in r.get("per_frame_cat_best", []):
                cb = frame_info["cat_best"]
                for i, cat_A in enumerate(CATEGORY_ORDER):
                    if cat_A not in cb:
                        continue
                    for cat_B in CATEGORY_ORDER[i:]:   # upper triangle + diagonal
                        if cat_B not in cb:
                            continue
                        key = f"{cat_A}|{cat_B}"
                        pair_sums[key]   += (cb[cat_A] + cb[cat_B]) / 2
                        pair_counts[key] += 1

        return {
            key: {
                "cat_A":         key.split("|")[0],
                "cat_B":         key.split("|")[1],
                "mean_combined": pair_sums[key] / pair_counts[key],
                "n_frames":      pair_counts[key],
            }
            for key in pair_sums
        }

    @staticmethod
    def compute_top2_pairs(results: List[dict]) -> dict:
        """
        For every frame, record which (rank-1 category, rank-2 category) pair
        appeared.  Returns counts and fractions for every ordered pair.

        This answers: "when category A is the single most activated token,
        which category tends to come second?"
        """
        pair_counts: Dict[str, int] = defaultdict(int)
        total = 0

        for r in results:
            if "error" in r:
                continue
            for entry in r.get("top2_per_frame", []):
                r1 = entry.get("rank1_cat")
                r2 = entry.get("rank2_cat")
                if r1 and r2:
                    pair_counts[f"{r1}|{r2}"] += 1
                    total += 1

        return {
            "counts":       dict(pair_counts),
            "fractions":    {k: v / total for k, v in pair_counts.items()} if total else {},
            "total_frames": total,
        }

    @staticmethod
    def compute_adjacent_pairwise(results: List[dict]) -> dict:
        """
        Adjacency-based pairwise scores.

        For every ordered pair of textually adjacent generated tokens
        (positions tok_idx i and i+1) and every sampled frame with valid
        TAM data, average the combined GT-mass = (mass_i + mass_{i+1}) / 2.
        Bucket by the **ordered** category pair (cat_i, cat_{i+1}).

        Differs from compute_pairwise (which uses per-frame best-per-category):
          • Uses sequential adjacency in the generated text, not co-occurrence.
          • The matrix is asymmetric — NOUN→ADJ may differ from ADJ→NOUN.
          • The diagonal cell (A→A) measures runs of same-category tokens.

        Returns
        -------
        dict keyed by "cat_A|cat_B" with:
            cat_A, cat_B, mean_combined, n_pairs
        """
        pair_sums:   Dict[str, float] = defaultdict(float)
        pair_counts: Dict[str, int]   = defaultdict(int)

        for r in results:
            if "error" in r:
                continue
            for frame_info in r.get("per_frame_tokens", []):
                tokens = frame_info.get("tokens", [])  # sorted by tok_idx
                for k in range(len(tokens) - 1):
                    a, b = tokens[k], tokens[k + 1]
                    if b["tok_idx"] != a["tok_idx"] + 1:
                        continue   # not strictly adjacent
                    key = f"{a['category']}|{b['category']}"
                    pair_sums[key]   += (float(a["gt_mass"]) + float(b["gt_mass"])) / 2
                    pair_counts[key] += 1

        return {
            key: {
                "cat_A":         key.split("|")[0],
                "cat_B":         key.split("|")[1],
                "mean_combined": pair_sums[key] / pair_counts[key],
                "n_pairs":       pair_counts[key],
            }
            for key in pair_sums
        }

    @staticmethod
    def compute_category_position_profile(results: List[dict]) -> dict:
        """
        Per-category mean GT-mass at each normalised frame-position bin,
        WITHOUT conditioning on the category being the per-frame oracle winner.

        For each (sequence, frame), takes the best gt_mass for each category
        present in that frame (= per_frame_cat_best). Bins by the frame's
        normalised position within the sequence and averages across all
        sequences.

        Returns
        -------
        dict[category] -> {
            "bin_means":  [mean_mass_b0, ..., mean_mass_b{N_BINS-1}]   (None if empty)
            "bin_counts": [n_frames_b0, ..., n_frames_b{N_BINS-1}]
            "overall_mean": mean over all bins (weighted by counts)
        }
        """
        bin_data: Dict[str, List[List[float]]] = {
            cat: [[] for _ in range(N_BINS)] for cat in CATEGORY_ORDER
        }

        for r in results:
            if "error" in r:
                continue
            for frame_info in r.get("per_frame_cat_best", []):
                bin_idx = frame_info.get("bin_idx", 0)
                if not (0 <= bin_idx < N_BINS):
                    continue
                cat_best = frame_info.get("cat_best", {})
                for cat, m in cat_best.items():
                    if cat in bin_data:
                        bin_data[cat][bin_idx].append(float(m))

        profile: Dict[str, dict] = {}
        for cat in CATEGORY_ORDER:
            per_bin = bin_data[cat]
            all_vals = [v for b in per_bin for v in b]
            profile[cat] = {
                "bin_means":  [
                    float(np.mean(per_bin[b])) if per_bin[b] else None
                    for b in range(N_BINS)
                ],
                "bin_counts": [len(per_bin[b]) for b in range(N_BINS)],
                "overall_mean": float(np.mean(all_vals)) if all_vals else None,
            }
        return profile


# ── dry-run stub ──────────────────────────────────────────────────────────────

class _DryRunRunner:
    """Synthetic runner for layout and categorisation testing (no GPU needed)."""

    fps        = 24.0
    video_mode = True

    def __init__(self, sample_rate: int = 8):
        self.sample_rate = sample_rate

    def run_with_tam(self, frames_pil, expression):
        T      = max(1, len(frames_pil) // self.sample_rate)
        H_tam, W_tam = 14, 14

        entries   = []
        gen_tokens: List[str] = ["["]
        for t in range(T):
            orig_t = t * self.sample_rate
            frame_toks = (
                ["{", '"frame"', ":", f" {orig_t}", ",",
                 '"bbox_2d"', ":", "[", "100", ",", "100", ",", "400", ",", "400", "]",
                 ",", '"label"', ":", '"', "red", " swan", '"', "}"]
            )
            if t > 0:
                gen_tokens.append(",")
            gen_tokens.extend(frame_toks)
            entries.append({"frame": orig_t,
                            "bbox_2d": [100, 100, 400, 400],
                            "label":   "red swan"})
        gen_tokens.append("]")

        tam_maps = [
            np.random.rand(T, H_tam, W_tam).astype(np.float32)
            for _ in gen_tokens
        ]
        gen_text = json.dumps(entries)

        return (
            [None] * len(frames_pil),
            gen_text,
            {"gen_tokens": gen_tokens, "tam_maps": tam_maps,
             "vision_shape": (T, H_tam, W_tam), "gen_text": gen_text},
        )
