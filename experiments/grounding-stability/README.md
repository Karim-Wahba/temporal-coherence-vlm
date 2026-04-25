# Grounding Stability

Studies whether TAM (Token Activation Map) attention heatmaps from Qwen3-VL are temporally stable across video frames, and whether that stability correlates with grounding quality.

## Research Questions

| # | Question | Metric |
|---|---|---|
| Q1 | Does grounding break over time? | IoU per frame, IoU variance |
| Q2 | Are TAM heatmaps temporally stable? | TAM instability (mean absolute pixel diff), mass-in-GT, attention entropy |
| Q3 | Does instability predict grounding failure? | Pearson / Spearman r between instability and (1 − IoU) |
| Q4 | Does attention localisation predict grounding quality? | Pearson / Spearman r between mass-in-GT and IoU |
| Q5 | Do heatmaps track scene motion? | Pearson r between optical flow magnitudes of image vs heatmap |

## Method

1. For each (sequence, expression) pair from Ref-DAVIS, run a single Qwen3-VL forward pass with TAM enabled.
2. Parse the JSON output to identify which generated tokens correspond to each frame's label.
3. Average TAM heatmaps across the label tokens to get one heatmap per detected frame.
4. Compute all five sets of metrics above.
5. Save per-sequence results and a two-row visualisation figure.

## Files

| File | Role |
|---|---|
| `run.py` | CLI entry point — loads model, runs experiment, saves results and summary |
| `experiment.py` | `GroundingStabilityExperiment` class — full pipeline per sequence, batch runner, summariser |
| `metrics.py` | All metric computations: IoU, instability, mass-in-GT, entropy, correlations, optical flow |
| `token_parser.py` | Parses model JSON output to find label token indices per frame (shared with token-ablation) |
| `visualizer.py` | Per-sequence figures, flow figures, dataset-level correlation plots |
| `sbatch_run.sh` | SLURM job script for the cluster (A100, 4 h, 100 GB RAM) |

## Usage

```bash
# Local run (small subset)
python run.py \
    --davis_root /path/to/DAVIS2017/unsupervised \
    --model_id   Qwen/Qwen3-VL-8B-Instruct \
    --save_dir   results/grounding_stability \
    --split      valid \
    --sample_rate 8 \
    --max_sequences 5 \
    --expressions_per_seq 2

# Full dataset run (image mode, 4 expressions per sequence)
python run.py \
    --davis_root /path/to/DAVIS2017/unsupervised \
    --save_dir   results/grounding_stability_full \
    --image_mode \
    --expressions_per_seq 4

# Cluster (SLURM)
sbatch sbatch_run.sh
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--davis_root` | required | Path to DAVIS dataset root (must contain `Annotations_bbox/`) |
| `--model_id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `--save_dir` | `results/grounding_stability` | Output directory |
| `--split` | `valid` | DAVIS split (`valid` or `train`) |
| `--sample_rate` | `8` | Send every Nth frame to the model |
| `--image_mode` | off | Use interleaved image mode instead of video mode (3D RoPE) |
| `--max_sequences` | none | Cap on unique sequences to process |
| `--expressions_per_seq` | `1` | Expressions per sequence |

## Outputs

```
{save_dir}/
    results.json                per-sequence metrics (IoU, instability, mass, correlations, flow)
    summary.json                dataset-level aggregates for all five questions
    visualizations/
        <seq>_exp<N>.png        two-row figure: frames + heatmaps with GT/pred boxes
        <seq>_exp<N>_flow.png   optical flow comparison (image vs heatmap)
        correlation_plots.png   scatter plots for Q3, Q4, Q5 across all sequences
```

### `results.json` fields (per sequence)

```
mean_iou, iou_variance, iou_per_frame          # Q1
mass_in_gt, mean_mass_in_gt                    # Q2 – GT attention accuracy
mass_in_pred, mean_mass_in_pred                # Q2 – predicted-box attention accuracy
mean_instability, entropy_per_frame            # Q2 – temporal stability
correlations                                   # Q3 – instability vs IoU failure
mass_accuracy_correlation                      # Q4 – mass-in-GT vs IoU
mass_pred_accuracy_correlation                 # Q4 – mass-in-pred vs IoU
flow_correlation                               # Q5 – image flow vs heatmap flow
```

## Dependencies

- `experiments/Ref-DAVIS/benchmark/davis_vot_loader.py` — DAVIS dataset loader
- `experiments/Ref-DAVIS/benchmark/qwen_vot_runner.py` — Qwen3-VL inference + TAM extraction
- `scipy` — Pearson / Spearman correlations
- `opencv-python` — Farneback optical flow
