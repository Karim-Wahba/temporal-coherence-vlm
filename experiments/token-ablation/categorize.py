"""
categorize.py
-------------
Post-hoc categorisation of oracle-winner tokens from a completed ablation run.

Reads results.json, assigns every oracle_row a category, writes
results_categorized.json, and produces a breakdown plot.

Can also be imported and called programmatically.

Usage
-----
    python categorize.py \\
        --results  results/token_ablation/results.json \\
        --save_dir results/token_ablation

Categories
----------
    label_noun        content word (noun / adj / verb) that appears in the expression
    label_function    function / stopword that appears in the expression
    bbox_coord        numeric token whose position in the output is inside a bbox_2d array
    frame_index       numeric token whose position in the output is the frame/time value
    json_structure    JSON punctuation  { } [ ] : , "
    other_content     content word NOT in the expression
    stopword          stopword NOT in the expression
    special           empty, whitespace, or model special tokens <|...|>
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── stopword list (same as strategies.py) ────────────────────────────────────

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

# JSON field key names that appear as tokens
_JSON_KEYS = frozenset({"frame", "time", "bbox_2d", "bbox", "label", "object"})


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(tok: str) -> str:
    return tok.replace("▁", "").replace("Ġ", "").strip()


def _expression_words(expression: str) -> frozenset:
    """Lowercase words in the expression."""
    return frozenset(re.findall(r"[a-zA-Z]+", expression.lower()))


def _is_content_word(w: str) -> bool:
    return len(w) >= 2 and w.isalpha() and w not in _STOPWORDS


def _looks_numeric(w: str) -> bool:
    """True for integer or float tokens."""
    try:
        float(w.replace(",", ""))
        return True
    except ValueError:
        return False


# ── position-context helpers ──────────────────────────────────────────────────

def _build_context_labels(gen_tokens: List[str]) -> List[str]:
    """
    Walk the token stream and tag each position with one of:
        'bbox_coord', 'frame_value', 'time_value', 'label_value',
        'json_key', 'json_structure', 'other'

    This lets us distinguish numeric tokens by their JSON context.
    """
    raw = "".join(gen_tokens)
    spans = []
    pos = 0
    for tok in gen_tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)

    labels = ["other"] * len(gen_tokens)

    def _tag_range(start_char: int, end_char: int, tag: str):
        for i, (s, e) in enumerate(spans):
            if s < end_char and e > start_char:
                labels[i] = tag

    # Find all "key": value spans
    for m in re.finditer(r'"(bbox_2d|frame|time|label)"(\s*):(\s*)', raw):
        key_name = m.group(1)
        after = m.end()

        if key_name == "bbox_2d":
            # value is a [ ... ] array
            ob = raw.find("[", after)
            if ob != -1:
                cb = raw.find("]", ob)
                if cb != -1:
                    _tag_range(ob, cb + 1, "bbox_coord")

        elif key_name in ("frame", "time"):
            # value is a number until next comma or }
            depth = 0
            end = after
            while end < len(raw):
                c = raw[end]
                if c in "[({":
                    depth += 1
                elif c in "])}":
                    if depth == 0:
                        break
                    depth -= 1
                elif c == "," and depth == 0:
                    break
                end += 1
            tag = "frame_value" if key_name == "frame" else "time_value"
            _tag_range(after, end, tag)

        elif key_name == "label":
            # value is a "..." string
            oq = raw.find('"', after)
            if oq != -1:
                cq = raw.find('"', oq + 1)
                if cq != -1:
                    _tag_range(oq + 1, cq, "label_value")

    # Tag JSON structural characters
    for i, tok in enumerate(gen_tokens):
        w = _clean(tok)
        if w in _JSON_STRUCTURAL or w in _JSON_KEYS:
            if labels[i] == "other":
                labels[i] = "json_structure"

    return labels


# ── main categorisation ───────────────────────────────────────────────────────

CATEGORY_ORDER = [
    "label_noun",
    "label_function",
    "bbox_coord",
    "frame_index",
    "other_content",
    "stopword",
    "json_structure",
    "special",
]

CATEGORY_COLORS = {
    "label_noun":     "#2ecc71",   # green  — best candidate
    "label_function": "#a8e6a3",   # light green
    "bbox_coord":     "#e74c3c",   # red
    "frame_index":    "#e67e22",   # orange
    "other_content":  "#3498db",   # blue
    "stopword":       "#95a5a6",   # grey
    "json_structure": "#bdc3c7",   # light grey
    "special":        "#ecf0f1",   # near-white
}


def categorize_token(
    tok_idx: int,
    token: str,
    token_clean: str,
    expression: str,
    context_labels: List[str],   # from _build_context_labels
) -> str:
    """
    Assign a single token to one category.
    Context labels break ties for numeric tokens.
    """
    w = token_clean.lower()

    # Special / empty
    if not w or "<|" in token or token.strip() == "":
        return "special"

    # JSON structural
    ctx = context_labels[tok_idx] if tok_idx < len(context_labels) else "other"
    if ctx == "json_structure" or w in _JSON_STRUCTURAL or w in _JSON_KEYS:
        return "json_structure"

    # Bbox coordinate (numeric in bbox context)
    if ctx == "bbox_coord" or (_looks_numeric(w) and ctx == "bbox_coord"):
        return "bbox_coord"

    # Frame / time index
    if ctx in ("frame_value", "time_value"):
        return "frame_index"

    # Remaining numerics not in labelled context → treat as bbox by default
    if _looks_numeric(w):
        return "bbox_coord"

    # Word tokens: check expression membership
    expr_words = _expression_words(expression)
    if w in expr_words:
        if _is_content_word(w):
            return "label_noun"
        else:
            return "label_function"

    # Not in expression
    if _is_content_word(w):
        return "other_content"
    if w in _STOPWORDS:
        return "stopword"

    return "json_structure"   # catch-all for punctuation not in structural set


def apply_categories(
    oracle_rows: List[dict],
    expression: str,
    gen_tokens: List[str],
) -> List[dict]:
    """
    Add a "category" field to every row in oracle_rows.
    gen_tokens must be the token list from the same forward pass.
    """
    ctx_labels = _build_context_labels(gen_tokens)
    for row in oracle_rows:
        row["category"] = categorize_token(
            tok_idx       = row["tok_idx"],
            token         = row["token"],
            token_clean   = row["token_clean"],
            expression    = expression,
            context_labels= ctx_labels,
        )
    return oracle_rows


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_category_breakdown(
    oracle_rows: List[dict],
    save_path: str,
    title: str = "Oracle token categories — breakdance",
):
    """
    Three-panel figure:
      Left   — stacked bar: win-count per category (all frames pooled)
      Centre — mean GT mass per category (box plot across winning frames)
      Right  — per-frame category strip: which category wins each frame
                (rows = expressions, columns = sampled frames)
    """
    winners = [r for r in oracle_rows if r["rank"] == 1]
    if not winners:
        print("  [WARN] no oracle winners to plot")
        return

    # ── aggregate ─────────────────────────────────────────────────────────
    cats_present = [c for c in CATEGORY_ORDER if any(r.get("category") == c for r in winners)]
    count_by_cat: Counter = Counter(r.get("category", "special") for r in winners)
    mass_by_cat:  defaultdict = defaultdict(list)
    for r in winners:
        mass_by_cat[r.get("category", "special")].append(r["gt_mass"])

    # ── layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(cats_present) * 0.5 + 2)))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Left: win count bar chart
    ax = axes[0]
    counts = [count_by_cat[c] for c in cats_present]
    colors = [CATEGORY_COLORS[c] for c in cats_present]
    y = np.arange(len(cats_present))
    bars = ax.barh(y, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(cats_present, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Win count")
    ax.set_title("Wins per category")

    # Centre: GT mass distribution per category
    ax = axes[1]
    data = [mass_by_cat[c] for c in cats_present]
    bp = ax.boxplot(
        data, vert=False, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, c in zip(bp["boxes"], cats_present):
        patch.set_facecolor(CATEGORY_COLORS[c])
        patch.set_alpha(0.8)
    ax.set_yticks(range(1, len(cats_present) + 1))
    ax.set_yticklabels(cats_present, fontsize=9)
    ax.set_xlabel("GT mass when winning")
    ax.set_title("GT mass distribution per category")
    ax.set_xlim(0, 1)

    # Right: per-frame strip coloured by category
    ax = axes[2]
    # group by expression
    exp_ids = sorted({r["exp_id"] for r in winners})
    all_ts  = sorted({r["orig_t"] for r in winners})
    cat_to_int = {c: i for i, c in enumerate(CATEGORY_ORDER)}

    # Build grid: rows = exp_id, cols = orig_t
    grid = np.full((len(exp_ids), len(all_ts)), np.nan)
    for r in winners:
        row_i = exp_ids.index(r["exp_id"])
        col_i = all_ts.index(r["orig_t"])
        grid[row_i, col_i] = cat_to_int.get(r.get("category", "special"), len(CATEGORY_ORDER) - 1)

    cmap_colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, len(CATEGORY_ORDER) + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(all_ts)))
    ax.set_xticklabels(all_ts, rotation=90, fontsize=6)
    ax.set_yticks(range(len(exp_ids)))
    ax.set_yticklabels([f"exp{e}" for e in exp_ids], fontsize=8)
    ax.set_xlabel("Original frame index")
    ax.set_title("Oracle category per frame")

    legend_patches = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[c], edgecolor="black",
                       linewidth=0.5, label=c)
        for c in cats_present
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="upper right",
              bbox_to_anchor=(1.45, 1.0))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  category breakdown → {save_path}")


def print_summary(oracle_rows: List[dict]):
    """Print a compact summary table to stdout."""
    winners = [r for r in oracle_rows if r["rank"] == 1]
    count_by_cat: Counter = Counter(r.get("category", "?") for r in winners)
    mass_by_cat:  defaultdict = defaultdict(list)
    for r in winners:
        mass_by_cat[r.get("category", "?")].append(r["gt_mass"])

    print(f"\n{'Category':<20}  {'Wins':>5}  {'%':>5}  {'Mean GT mass':>12}  Example tokens")
    print("-" * 80)
    total = len(winners)
    for cat in CATEGORY_ORDER:
        if cat not in count_by_cat:
            continue
        cnt  = count_by_cat[cat]
        pct  = 100 * cnt / total if total else 0
        mean = np.mean(mass_by_cat[cat])
        # top 5 most common token_clean values in this category
        toks = Counter(
            r["token_clean"] for r in winners if r.get("category") == cat
        ).most_common(5)
        tok_str = ", ".join(f"{t!r}×{c}" for t, c in toks)
        print(f"  {cat:<18}  {cnt:>5}  {pct:>4.1f}%  {mean:>12.4f}  {tok_str}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("Token category analysis")
    p.add_argument("--results",  required=True, help="Path to results.json")
    p.add_argument("--save_dir", required=True, help="Directory for output files")
    args = p.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    save_dir = Path(args.save_dir)
    all_oracle_rows: List[dict] = []

    for r in results:
        if "error" in r:
            continue
        expression  = r["expression"]
        gen_tokens  = [entry["token"] for entry in r.get("token_scores", [])]
        oracle_rows = r.get("oracle_rows", [])

        if not oracle_rows:
            print(f"  [WARN] {r['seq_name']} exp{r['exp_id']}: no oracle_rows — "
                  "re-run ablation to generate them")
            continue

        # Reconstruct gen_tokens list in index order from oracle_rows
        # (token_scores is sorted by score, not by index — so we rebuild
        # a sparse index→token map and fill gaps with empty string)
        idx_to_tok: Dict[int, str] = {}
        for entry in r.get("token_scores", []):
            idx_to_tok[entry["tok_idx"]] = entry["token"]
        max_idx = max((row["tok_idx"] for row in oracle_rows), default=0)
        full_gen_tokens = [idx_to_tok.get(i, "") for i in range(max_idx + 1)]

        # Add exp_id to each row so the strip plot can separate expressions
        for row in oracle_rows:
            row["exp_id"] = r["exp_id"]

        categorized = apply_categories(oracle_rows, expression, full_gen_tokens)
        all_oracle_rows.extend(categorized)

    if not all_oracle_rows:
        print("No oracle rows found. Make sure the ablation was run with the latest code.")
        return

    print_summary(all_oracle_rows)

    # Save enriched results
    # (mutates in-place — re-write results.json with category field added)
    out_path = save_dir / "results_categorized.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  categorized results → {out_path}")

    plot_category_breakdown(
        all_oracle_rows,
        save_path=str(save_dir / "plots" / "category_breakdown.png"),
    )


if __name__ == "__main__":
    main()
