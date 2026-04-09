# Temporal Coherence Benchmark for Qwen3-VL

A dual-arm evaluation framework for diagnosing and quantifying temporal coherence
failures in Qwen3-VL video-language models on DAVIS.

```
Ref-DAVIS/
├── benchmark.py                  ← Main orchestrator
├── run_tam_experiments.py        ← Standalone TAM diagnostic experiments
├── compare_models.py             ← Multi-model comparison reports
├── benchmark/
│   ├── ref_davis_loader.py       ← Ref-DAVIS dataset loader (VOS)
│   ├── davis_vot_loader.py       ← DAVIS VOT dataset loader
│   ├── qwen_vos_runner.py        ← VOS inference (image / video mode)
│   ├── qwen_vot_runner.py        ← VOT inference (image / video mode)
│   └── metrics.py                ← J, F, J&F, IoU, decay, variance
├── diagnostics/
│   ├── tam_runner.py             ← TAM inference wrapper
│   ├── tam_analyzer.py           ← Diagnostic experiments
│   └── failure_classifier.py    ← Rule-based failure mode classifier (VOS + VOT)
└── visualization/
    └── visualizer.py             ← All plots and galleries
```

---

## Step 0: Extend DAVIS → Ref-DAVIS

Download the text annotations (required for VOS task only):

```bash
wget https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research/OneVOS/davis_text_annotations.zip
cd /path/to/your/davis
unzip /path/to/davis_text_annotations.zip
```

Expected layout:
```
/your/davis/
├── JPEGImages/480p/<seq>/<frame>.jpg
├── Annotations/480p/<seq>/<frame>.png
└── davis_text_annotations/
    ├── train/meta_expressions.json
    └── valid/meta_expressions.json
```

---

## Step 1: Run Benchmark

### VOS (mask J&F)

```bash
# Quick run: 5 sequences, image mode (default)
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/vos_image \
    --task vos \
    --split valid \
    --strategy joint \
    --sample_rate 8 \
    --expressions_per_seq 1 \
    --max_sequences 5

# Native video mode (3D RoPE, official Qwen3-VL evaluation method)
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/vos_video \
    --task vos \
    --video_mode \
    --sample_rate 8 \
    --expressions_per_seq 1 \
    --max_sequences 5

# Resume interrupted run
python benchmark.py \
    ... \
    --checkpoint_json results/vos_video/checkpoint.json
```

### VOT (bbox IoU)

```bash
# Image mode
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/vot_image \
    --task vot \
    --sample_rate 8 \
    --max_sequences 30

# Video mode
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/vot_video \
    --task vot \
    --video_mode \
    --sample_rate 8 \
    --max_sequences 30
```

### Input Modes

| Flag | Description |
|------|-------------|
| *(default)* | **Image mode** — frames interleaved with text timestamps, 2D RoPE per frame. Model outputs `{"time": t, "bbox_2d": [...]}`. |
| `--video_mode` | **Video mode** — all frames as a single `{"type":"video"}` block, 3D RoPE across the sequence. Model outputs `{"frame": i, "bbox_2d": [...]}`. This is the official Qwen3-VL evaluation method. |

In both modes, only sampled frames (every `sample_rate`-th frame) are sent to the model. Evaluation and visualisation are performed on sampled frames only — no nearest-neighbour interpolation is applied. Frames the model skips score IoU/J = 0.

**Outputs:**
```
results/<run>/
├── metrics.csv                          ← per-sequence metrics + failure mode
├── summary.json                         ← aggregate stats + config
├── checkpoint.json                      ← resume support
├── plots/
│   ├── j_curves.png / iou_curves.png
│   └── aggregate_summary.png
├── result_cases/                        ← VOT: per-sequence grids
│   ├── <seq>__exp<id>__iou<x>__<MODE>.png
│   └── <seq>__exp<id>__iou<x>__<MODE>__raw.txt
└── failure_cases/                       ← VOS: per-sequence grids
    ├── <seq>__exp<id>__<MODE>.png
    └── <seq>__exp<id>__<MODE>__raw.txt
```

The `__raw.txt` file contains the model's verbatim output for that sequence, useful for debugging parse failures and inspecting temporal consistency.

---

## Step 2: Run TAM Diagnostics (Arm 2 — Diagnostic)

```bash
# Temporal Collapse (fastest, most revealing)
python run_tam_experiments.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --experiment collapse \
    --max_sequences 10 \
    --save_dir results/tam_collapse \
    --tam_submodule_path /path/to/submodules/TAM

# Prompt Temporal Binding
python run_tam_experiments.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --experiment binding \
    --max_sequences 10 \
    --save_dir results/tam_binding

# All experiments
python run_tam_experiments.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --experiment all \
    --sequences blackswan camel car-roundabout \
    --save_dir results/tam_all
```

**Outputs per experiment:**
- `*_frame_mass.png` — (tokens × frames) attention heatmap — reveals collapse
- `*_centroid.png` — attention centroid trajectory vs GT — reveals drift
- `*_binding_scores.png` — binding score per prompt — reveals steerability
- `exp_<name>_results.json` — all scalar metrics

---

## Step 3: Compare Two Models

```bash
python compare_models.py \
    --runs results/vot_image results/vot_video \
    --names "Image Mode" "Video Mode" \
    --save_dir results/comparison
```

**Prints:**
```
Metric                 Image Mode    Video Mode      Delta(A-B)
────────────────────────────────────────────────────────────────
  Mean IoU ↑               0.2341          0.3012    +0.0671 ✓
  Success@0.5 ↑            0.1823          0.2441    +0.0618 ✓
  IoU-Decay ↑             -0.0821         -0.0412    +0.0409 ✓
  ...
```

---

## Metrics Reference

### VOS Metrics (Ref-DAVIS)
| Metric | Description |
|--------|-------------|
| **J** | Region similarity (IoU) between predicted and GT mask |
| **F** | Boundary F-measure |
| **J&F** | Primary VOS metric |
| **J-Decay** | Linear slope of J over time. Negative = losing track. |
| **J-Variance** | Std of per-frame J. High = unstable tracking. |

### VOT Metrics
| Metric | Description |
|--------|-------------|
| **Mean IoU** | Mean bbox IoU over sampled frames |
| **Success@0.5 / @0.75** | % sampled frames with IoU above threshold |
| **Precision@20** | % sampled frames with center error < 20px |
| **IoU-Decay** | Linear slope of IoU over time |
| **IoU-Variance** | Std of per-frame IoU |

### TAM Diagnostic Metrics
| Metric | Description |
|--------|-------------|
| **Collapse Rate** | % tokens where >80% attention on a single frame |
| **Mean Drift Error** | Mean L2 distance between TAM centroid and GT centroid |
| **Binding Score** | Fraction of attention mass on prompt-targeted frames |

---

## Failure Modes

Applied to both VOS (J-based) and VOT (IoU-based):

| Mode | Detection | Meaning |
|------|-----------|---------|
| `SUCCESS` | mean ≥ 0.5 | Tracked correctly |
| `NEVER_FOUND` | first-third mean ≈ 0 | Never localized the object |
| `LOST_TRACK` | score[0] > 0.3, then 3+ consecutive frames near 0 | Initially tracked, then lost |
| `PARTIAL_TRACK` | mean in 0.05–0.4 | Partially correct but imprecise |
| `TEMPORAL_COLLAPSE` | collapse_rate > 0.5 (TAM, VOS only) | Attention ignores most frames |
| `IDENTITY_SWAP` | sudden centroid jump (TAM, VOS only) | Switches to wrong object mid-sequence |
| `ATTENTION_DRIFT` | mean_drift_error > 5px (TAM, VOS only) | Attention wanders from object |
| `UNSTABLE` | variance > 0.15 | Erratic tracking, no clear cause |

---

## Notes

- **`--video_mode`** is the official Qwen3-VL evaluation method (native `{"type":"video"}` block, 3D RoPE). Default image-interleaved mode uses 2D RoPE per frame with injected text timestamps.
- **`--sample_rate N`** sends every Nth frame to the model. Lower = more temporal coverage, less spatial resolution per frame within the token budget. Evaluation is on sampled frames only.
- **Checkpoint**: Auto-saves after each sequence. Resume with `--checkpoint_json`.
- **TAM path**: Pass `--tam_submodule_path` pointing to the directory containing `tam.py`.
