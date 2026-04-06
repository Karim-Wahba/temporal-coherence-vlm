# Temporal Coherence Benchmark for Qwen3-VL

A dual-arm evaluation framework for diagnosing and quantifying temporal coherence
failures in Qwen3-VL video-language models.

```
temporal_bench/
├── benchmark.py                  ← Main orchestrator (Ref-DAVIS benchmark)
├── run_tam_experiments.py        ← Standalone TAM diagnostic experiments
├── compare_models.py             ← Multi-model comparison reports
├── benchmark/
│   ├── ref_davis_loader.py       ← Ref-DAVIS dataset loader
│   ├── qwen_vos_runner.py        ← Qwen inference + bbox parsing
│   └── metrics.py                ← J, F, J&F, J-decay, J-variance
├── diagnostics/
│   ├── tam_runner.py             ← TAM inference wrapper
│   ├── tam_analyzer.py           ← 5 diagnostic experiments
│   └── failure_classifier.py    ← Rule-based failure mode classifier
└── visualization/
    └── visualizer.py             ← All plots and galleries
```

---

## Step 0: Extend DAVIS → Ref-DAVIS

You only need to download one file — the text annotations:

```bash
# Download from MPI-INF
wget https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research/OneVOS/davis_text_annotations.zip

# Unzip into your DAVIS root
cd /path/to/your/davis
unzip /path/to/davis_text_annotations.zip
```

After this, your DAVIS root should contain:
```
/your/davis/
├── JPEGImages/480p/<seq>/<frame>.jpg    ← already have
├── Annotations/480p/<seq>/<frame>.png   ← already have
└── davis_text_annotations/              ← NEW
    ├── train/meta_expressions.json
    └── valid/meta_expressions.json
```

---

## Step 1: Run Benchmark (Arm 1 — Quantitative)

```bash
# Quick run: 5 sequences, expression 0 only, no TAM
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/qwen3vl_8b \
    --split valid \
    --strategy joint \
    --expressions_per_seq 1 \
    --max_sequences 5

# Full validation set (30 seqs × 4 expressions = 120 items)
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/qwen3vl_8b \
    --split valid \
    --strategy joint \
    --expressions_per_seq 4

# With TAM diagnostics (adds ~3x time per sequence)
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/qwen3vl_8b_tam \
    --run_tam \
    --tam_submodule_path /path/to/submodules/TAM \
    --expressions_per_seq 1

# Resume interrupted run
python benchmark.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --save_dir results/qwen3vl_8b \
    --checkpoint_json results/qwen3vl_8b/checkpoint.json
```

**Outputs:**
```
results/qwen3vl_8b/
├── metrics.csv              ← per-sequence J, F, J&F, J-decay, J-variance, failure mode
├── failure_analysis.csv
├── summary.json             ← aggregate stats + model config
├── checkpoint.json          ← resume support
├── plots/
│   ├── j_curves.png         ← J over time per sequence
│   ├── failure_gallery.png  ← GT vs pred overlays
│   └── aggregate_summary.png← 4-panel summary
└── failure_cases/           ← per-sequence failure figures
```

---

## Step 2: Run TAM Diagnostics (Arm 2 — Diagnostic)

```bash
# Experiment 2: Temporal Collapse (fastest, most revealing)
python run_tam_experiments.py \
    --davis_root /path/to/davis \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --experiment collapse \
    --max_sequences 10 \
    --save_dir results/tam_collapse \
    --tam_submodule_path /path/to/submodules/TAM

# Experiment 5: Prompt Temporal Binding (key for steerability analysis)
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
- `exp_<name>_results.json` — all scalar metrics for statistical analysis

---

## Step 3: Compare Two Models

```bash
# After running benchmark.py for two models:
python compare_models.py \
    --runs results/qwen3vl_8b results/qwen25vl_7b \
    --names "Qwen3-VL-8B" "Qwen2.5-VL-7B" \
    --save_dir results/comparison
```

**Prints:**
```
Metric                 Qwen3-VL-8B   Qwen2.5-VL-7B      Delta(A-B)
────────────────────────────────────────────────────────────────────
  J&F ↑                    0.3821          0.3540    +0.0281 ✓
  Mean J ↑                 0.3612          0.3301    +0.0311 ✓
  J-Decay ↑               -0.1823         -0.2441    +0.0618 ✓
  J-Variance ↓             0.1203          0.1589    -0.0386 ✓
  ...
```

**Saves:**
- `metric_comparison.png` — grouped bar chart
- `j_decay_per_sequence.png` — per-sequence J-decay comparison
- `failure_mode_comparison.png` — side-by-side failure mode pies

---

## Metrics Reference

### Standard VOS Metrics (Ref-DAVIS)
| Metric | Description |
|--------|-------------|
| **J** | Region similarity (IoU) between predicted and GT mask |
| **F** | Boundary F-measure (contour accuracy) |
| **J&F** | Primary benchmark metric, mean of J and F |
| **Success@0.5** | % frames with J > 0.5 |

### Temporal Coherence Metrics (New)
| Metric | Description |
|--------|-------------|
| **J-Decay** | Linear slope of J over time. Negative = losing track. Comparable across sequences (normalised). |
| **J-Variance** | Std of per-frame J. High = unstable/oscillating tracking. |
| **J-First / J-Last** | IoU on first vs last frame. Gap reveals long-term degradation. |

### TAM Diagnostic Metrics (Explanatory)
| Metric | Experiment | Description |
|--------|-----------|-------------|
| **Collapse Rate** | Exp 2 | % tokens where >80% attention on single frame |
| **Temporal Entropy** | Exp 2 | Normalised entropy of frame attention distribution |
| **Mean Drift Error** | Exp 1 | Mean L2 distance between TAM centroid and GT centroid |
| **Binding Score** | Exp 5 | Fraction of attention mass on prompt-targeted frames |
| **Binding Std** | Exp 5 | Variability of binding across prompts (high = steerable) |

---

## Failure Modes

| Mode | Detection | Meaning |
|------|-----------|---------|
| `SUCCESS` | mean J ≥ 0.5 | Model tracked correctly |
| `NEVER_FOUND` | first-third J ≈ 0 | Never localized the object |
| `LOST_TRACK` | J[0]>0.3, then 3+ consecutive frames near 0 | Initially tracked, then lost |
| `PARTIAL_TRACK` | mean J in 0.05–0.4 | Partially correct but imprecise |
| `TEMPORAL_COLLAPSE` | collapse_rate > 0.5 (TAM) | Attention ignores most frames |
| `IDENTITY_SWAP` | sudden centroid jump | Switches to wrong object mid-sequence |
| `ATTENTION_DRIFT` | mean_drift_error > 5px (TAM) | Attention wanders from object |
| `UNSTABLE` | J-variance > 0.15 | Erratic tracking, no clear cause |

---

## Notes

- **Strategy `joint`**: All frames in one Qwen call. Fastest, most natural.
- **Strategy `per_frame`**: One Qwen call per frame. Slowest, most controlled baseline.
- **SAM2**: Not included by default. To add mask quality, wrap `box_to_mask` in
  `qwen_vos_runner.py` with a SAM2 predictor using the box as prompt.
- **Checkpoint**: The benchmark auto-saves `checkpoint.json` after each sequence
  and can resume with `--checkpoint_json`.
- **TAM path**: TAM must be in your Python path. Pass `--tam_submodule_path`
  pointing to the directory containing `tam.py` and `qwen_utils.py`.
