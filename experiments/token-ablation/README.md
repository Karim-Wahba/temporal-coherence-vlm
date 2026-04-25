# Token Selection Ablation

Ablation study comparing token selection strategies for localising objects in video using TAM (Token Activation Maps) from Qwen3-VL.

## Motivation

TAM produces one attention heatmap per generated token. To localise the target object in each video frame, you need to pick which token(s) to read the heatmap from. This experiment measures how much ground-truth bounding box attention mass each selection strategy achieves — higher GT mass means the heatmap points more accurately at the target.

## Research Question

**Which token selection strategy best focuses TAM attention on the ground-truth object?**

## Method

1. Run a single Qwen3-VL forward pass per (sequence, expression) pair on the DAVIS 2017 dataset.
2. Apply each strategy to produce a per-frame heatmap from the generated token stream.
3. Compute **GT mass**: fraction of heatmap activation inside the GT bounding box.
4. Rank strategies by mean GT mass across frames and expressions.
5. Categorise oracle-winner tokens to understand what kinds of tokens carry the most spatial information.

## Strategies

| Strategy | Description |
|---|---|
| `whole_label` | Average all tokens between the label quotes for the detected frame |
| `first_word` | Tokens of the first word of the label |
| `last_word` | Tokens of the last word of the label |
| `content_words_in_label` | Non-stopword alphabetic tokens within the label |
| `all_content_tokens` | Content words from anywhere in the generated output (frame-agnostic) |
| `bbox_tokens` | Tokens that form the `bbox_2d` coordinate values |
| `frame_tokens` | Tokens encoding the `frame` integer index (video mode) |
| `time_tokens` | Tokens encoding the `time` float value (image mode) |
| `label_and_bbox` | Union of label tokens and bbox tokens |
| `label_and_frame` | Union of label tokens and temporal-index tokens |
| `label_nouns_mean` | Label tokens that also appear in the expression query — mean aggregation |
| `label_nouns_max` | Same as above — pixel-wise max aggregation |
| `label_nouns_geomean` | Same — geometric mean (rewards spatial consensus) |
| `label_nouns_weighted_entropy` | Same — weighted by inverse spatial entropy |
| `label_nouns_weighted_peak` | Same — weighted by peak activation |
| `label_nouns_top1` | Single label-noun token with highest total activation |
| `per_frame_oracle` | Upper bound: best possible single token per frame |
| `global_best_token` | Best single token globally (fixed across all frames) |
| `all_tokens_mean` | Mean over all tokens (global average baseline) |
| `random_tokens` | Average of 5 randomly sampled tokens (random baseline, seed=42) |

## Files

| File | Role |
|---|---|
| `run_ablation.py` | Main entry point — loads data, runs forward passes, scores all strategies, writes results |
| `strategies.py` | Strategy implementations + `StrategyContext` dataclass + oracle analysis |
| `categorize.py` | Post-hoc categorisation of oracle-winner tokens (label noun, bbox coord, frame index, etc.) |
| `visualize.py` | Plotting utilities (bar chart, heatmap grids, per-frame curves, token spotlight) |

## Usage

```bash
# Full run on the breakdance sequence
python run_ablation.py \
    --davis_root /path/to/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/token_ablation \
    --sequence   breakdance \
    --expressions_per_seq 4 \
    --sample_rate 8

# Dry run (no model, synthetic TAM maps — useful for layout testing)
python run_ablation.py --dry_run --save_dir results/dry_run

# Run only a subset of strategies
python run_ablation.py --strategies whole_label label_nouns_mean per_frame_oracle all_tokens_mean

# Post-hoc token categorisation on existing results
python categorize.py \
    --results  results/token_ablation/results.json \
    --save_dir results/token_ablation
```

## Outputs

```
results/token_ablation/
    results.json                    per-expression, per-strategy scores + oracle rows
    summary.json                    aggregated mean ± std per strategy, ranked
    results_categorized.json        results enriched with oracle token categories
    plots/
        bar_chart.png               strategy comparison bar chart with error bars
        per_frame_curves.png        GT mass vs frame index per strategy
        token_spotlight.png         top-25 tokens by mean GT mass
        oracle_token_analysis.png   oracle winner frequency + per-frame mass grid
        category_breakdown.png      win count + GT mass distribution per token category
        heatmaps/<seq>_<exp>_t<N>.png  per-strategy heatmap grids for sampled frames
```

## Dependencies

Relies on shared infrastructure from sibling experiment folders:

- `experiments/Ref-DAVIS/benchmark/davis_vot_loader.py` — DAVIS dataset loader
- `experiments/Ref-DAVIS/benchmark/qwen_vot_runner.py` — Qwen3-VL inference + TAM extraction
- `experiments/grounding-stability/token_parser.py` — JSON output parser (shared)
