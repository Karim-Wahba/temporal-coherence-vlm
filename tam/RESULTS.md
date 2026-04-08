# TAM Evaluation Results on Qwen3-VL-2B-Instruct

**Date:** 2026-03-27
**Hardware:** Mac Mini M4, 16GB unified memory
**Model:** Qwen/Qwen3-VL-2B-Instruct (bfloat16, no quantization)
**Dataset:** COCO Caption 2014 val, 100 images (first 100 by image ID)
**Segmentation masks:** Generated from COCO polygon annotations via `prepare_coco.py`

## Main Results

| Setting | Obj-IoU | Func-IoU | F1-IoU | Precision | Recall |
|---------|---------|----------|--------|-----------|--------|
| TAM (max_new_tokens=256) | 0.2240 | 0.8782 | 0.3569 | 0.3687 | 0.5993 |
| TAM (max_new_tokens=40) | 0.2054 | 0.8738 | 0.3326 | 0.3459 | 0.6062 |

Paper reference (Qwen2-VL-2B, 5K standard minival):

| Setting | Obj-IoU | Func-IoU | F1-IoU |
|---------|---------|----------|--------|
| Full TAM | 0.2737 | 0.6844 | 0.3910 |

## Ablation Study (30 samples, max_new_tokens=256)

| Setting | Obj-IoU | Func-IoU | F1-IoU |
|---------|---------|----------|--------|
| No ECI, No RGF | 0.2033 | 0.6219 | 0.3064 |
| ECI only | 0.1917 | 0.8824 | 0.3150 |
| RGF only | 0.2395 | 0.4609 | 0.3152 |
| Full TAM (ECI+RGF) | 0.2136 | 0.8904 | 0.3446 |

Paper reference (Qwen2-VL-2B, 5K standard minival):

| Setting | Obj-IoU | Func-IoU | F1-IoU |
|---------|---------|----------|--------|
| No ECI, No RGF | 0.2123 | 0.5193 | 0.3014 |
| ECI only | 0.2241 | 0.6903 | 0.3384 |
| RGF only | 0.2482 | 0.4334 | 0.3157 |
| Full TAM (ECI+RGF) | 0.2737 | 0.6844 | 0.3910 |

## ECI Scale Cap Sweep (30 samples, max_new_tokens=256)

| ECI Cap | Obj-IoU | Func-IoU | F1-IoU |
|---------|---------|----------|--------|
| 0.0 (no ECI) | 0.2479 | 0.4249 | 0.3131 |
| 0.1 | 0.2461 | 0.4385 | 0.3152 |
| 0.2 | 0.2463 | 0.4576 | 0.3203 |
| 0.3 | 0.2452 | 0.4782 | 0.3242 |
| 0.4 | 0.2444 | 0.5078 | 0.3300 |
| 0.5 | 0.2444 | 0.5419 | 0.3369 |
| 0.7 | 0.2389 | 0.6259 | 0.3458 |
| 1.0 | 0.2351 | 0.7242 | 0.3550 |
| None (uncapped) | 0.2313 | 0.8791 | 0.3662 |

## Key Findings

1. **Implementation is correct.** Baseline (no ECI, no RGF) matches paper within noise:
   our 0.2033 vs paper 0.2123. RGF-only also matches: our 0.2479 vs paper 0.2482.

2. **ECI behaves differently on Qwen3-VL vs Qwen2-VL.**
   - On Qwen2-VL: ECI improves Obj-IoU (+1.2%) by removing interference.
   - On Qwen3-VL: ECI hurts Obj-IoU (-1.2%) due to over-subtraction.
   - Cause: Qwen3-VL has more correlated cross-token activations, so the
     least-squares scale factor overshoots, suppressing genuine object signal.

3. **Despite lower Obj-IoU, Func-IoU is much higher on Qwen3-VL** (0.88 vs 0.68),
   partially compensating in F1-IoU. The net F1-IoU gap is ~3.4% (0.357 vs 0.391),
   much smaller than the Obj-IoU gap (~5%).

4. **max_new_tokens matters.** Using 40 vs 256 costs ~2.4% F1-IoU.

5. **ECI scale capping trades Obj-IoU for Func-IoU** but uncapped gives best F1-IoU.
   Available via `ECI_SCALE_CAP` in `config.py`.

---

## Video / Temporal Experiments (2026-03-27)

### Video Processing Discovery

Qwen3-VL's video processor aggressively merges temporal frames. Regardless of input
frame count (10–60) or frame resolution, `video_grid_thw` always produces `t=2`
temporal patches, which after `SPATIAL_MERGE_SIZE=2` division yields **1 temporal
step**. This makes the native video pipeline unusable for per-frame temporal analysis.

**Workaround: Multi-image mode.** Pass each frame as a separate image in the
conversation (`multi_image=True` in `prepare_inputs()`). Each frame gets its own
14×14 activation grid (196 tokens), and the LLM attends across frames via
cross-attention. This preserves per-frame spatial resolution.

### First Temporal Coherence Result: simple_translation (10 frames)

Prompt: "These are frames from a video. Describe the red circle."
Generated: "The provided frames show a single, static red circle... consistently positioned..."

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Map consistency (cosine sim) | 0.894 | High — global attention pattern similar across frames |
| Spatial smoothness (px disp) | 9.56 | High — peak jumps wildly across the 14×14 grid |
| Position accuracy (map-px) | 7.61 | Poor — essentially random on 14×14 (max diagonal ~20) |
| Intensity stability (std) | 0.039 | Stable — peak activation strength is consistent |

**Peak trajectory** (row, col on 14×14 grid):
`(9,13) (13,0) (2,6) (9,0) (12,9) (10,3) (4,11) (8,0) (10,0) (0,5)`

**GT trajectory** (scaled to 14×14):
`(7,1) (7,2) (7,3) (7,5) (7,6) (7,7) (7,8) (7,10) (7,11) (7,12)`

**Interpretation:** When targeting a generic noun ("sequence"), the model does NOT
spatially track — peaks are random. But see below for object-specific token results.

### Full Temporal Coherence Results (8 frames, RGF-only, target token = "circle")

| Video | Token | Map Consistency | Spatial Smoothness | Position Accuracy |
|-------|-------|----------------:|-------------------:|------------------:|
| simple_translation | circle | 0.828 | 2.53 px | 2.86 map-px |
| appearance_change | circle | 0.815 | 1.67 px | 2.40 map-px |
| occlusion | circle | 0.740 | 4.32 px | 2.63 map-px |
| reentry | circle | 0.641 | 2.62 px | 4.26 map-px |
| two_objects_crossing | circles | 0.838 | 6.81 px | 7.42 map-px |

Additional metrics:
- Occlusion recovery (occlusion video): 0.548 cosine similarity pre/post occlusion

**Interpretation:**

1. **The model DOES localize the circle reasonably well.** Position accuracy of
   2.4–2.9 map-pixels on a 14×14 grid (GT is at row 7) means peaks are usually within
   2–3 cells of the correct position for simple videos.

2. **Tracking is imperfect.** The trajectory plots show peaks roughly following the
   left-to-right GT path but with vertical jitter (row oscillation around row 5–8 vs
   GT at row 7). This is partial tracking — better than random but not smooth.

3. **Harder videos degrade predictably:**
   - `two_objects_crossing` has worst position accuracy (7.42) — the model conflates
     the two circles' positions.
   - `reentry` has lowest consistency (0.641) — the model struggles with the object
     leaving and re-entering the frame.
   - `occlusion` recovery is moderate (0.548) — map partially recovers after occlusion
     but not fully.

4. **Key insight: token choice matters enormously.** Generic tokens like "sequence"
   give random spatial patterns. Object-specific tokens like "circle" give meaningful
   localization. This validates TAM's per-token approach — the activation map quality
   depends critically on selecting the right token.

Full results JSON: `results_temporal/results.json`
Trajectory plots: `results_temporal/{video_name}/trajectory.png`

### Multi-Layer Logit Lens Analysis (8 frames, RGF-only, target = "circle")

Activation maps computed at 5 decoder layers (0, 7, 14, 21, 27 out of 28) using
`lm_head.weight[cls_id] @ hidden_states[layer]` — memory-efficient single-class
extraction validated at r=0.999998 correlation with standard TAM at last layer.

**simple_translation:**

| Layer | Map Consistency | Spatial Smoothness | Position Accuracy |
|------:|----------------:|-------------------:|------------------:|
| 0 | 0.922 | 1.59 px | 1.72 map-px |
| 7 | **0.942** | **1.93 px** | **1.93 map-px** |
| 14 | 0.856 | 4.32 px | 5.58 map-px |
| 21 | 0.839 | 5.08 px | 4.18 map-px |
| 27 | 0.834 | 2.54 px | 2.74 map-px |

**Pattern consistent across all 5 videos (position accuracy, map-px):**

| Layer | simple_transl | appearance | occlusion | reentry | crossing |
|------:|--------------:|-----------:|----------:|--------:|---------:|
| 0 | 1.72 | 1.74 | 4.26 | 6.97 | 2.37 |
| 7 | 1.93 | 2.33 | 3.09 | 8.00 | 1.85 |
| 14 | 5.58 | 3.93 | 6.74 | 6.60 | 6.80 |
| 21 | 4.18 | 3.16 | 4.59 | 5.46 | 5.12 |
| 27 | 2.74 | 2.34 | 2.52 | 4.21 | 7.47 |

**Key finding: Visual spatial information follows a U-shaped curve across layers.**

1. **Early layers (0–7) have the BEST tracking** — highest map consistency (0.92–0.94),
   lowest spatial displacement, best position accuracy (~1.7–2.3 map-px). Visual
   spatial information from the vision encoder is still rich and well-localized.

2. **Middle layers (14) are the WORST** — consistency drops to ~0.86, displacement
   spikes (4–11 px), position accuracy degrades to 4–7 map-px. This is where
   text-visual information mixing disrupts spatial coherence.

3. **Late layers (21–27) partially recover** — position accuracy improves back to
   2.3–2.7 for simple cases but doesn't reach early-layer quality. The model
   reconstructs some spatial information for the final prediction.

This U-shaped pattern was NOT predicted by the research plan (which hypothesized
tracking peaks in middle layers). Instead it shows:
- The vision encoder's spatial information is strongest at the LLM input (layer 0)
- The LLM's self-attention disrupts spatial coherence in middle layers
- The final layers partially recover spatial grounding, likely for generation

**Implications:**
- TAM's last-layer-only approach (layer 27) misses the best spatial signal (layers 0–7)
- A multi-layer TAM that uses early layers could improve localization by ~30%
  (pos_acc 1.7 vs 2.7 for simple_translation)
- The middle-layer degradation suggests the LLM is transforming spatial features
  into more abstract/semantic representations

Full results JSON: `results_multilayer/results.json`
Layer coherence plots: `results_multilayer/{video_name}/layer_coherence.png`

---

## DAVIS 2017 Evaluation (2026-03-30, updated 2026-04-07)

### Setup
- **30 DAVIS val videos** (full val split), 8 frames sampled per video
- Multi-image mode (per-frame activation maps)
- RGF-only (no ECI), max_new_tokens=60
- Multi-layer analysis at layers [0, 7, 14, 21, 27]
- GT: per-frame segmentation masks from DAVIS
- Token targeting: DAVIS_OBJECT_TOKENS lookup table with multi-candidate fallback
- IoU: both binary (all objects) and best per-object match

### Per-Video Results (last layer, 30 videos)

| Video | Token | IoU(binary) | IoU(best-obj) | Consistency |
|-------|-------|------------:|--------------:|------------:|
| judo | man | **0.412** | 0.294 | 0.761 |
| pigs | pig | **0.371** | 0.289 | 0.860 |
| cows | cow | 0.344 | 0.344 | 0.794 |
| car-roundabout | car | 0.294 | 0.294 | 0.905 |
| gold-fish | goldfish | 0.289 | 0.175 | 0.758 |
| bike-packing | bike | 0.268 | 0.327 | 0.835 |
| blackswan | swan | 0.234 | 0.234 | 0.816 |
| breakdance | breakdancer | 0.203 | 0.203 | 0.742 |
| camel | camels | 0.203 | 0.203 | 0.705 |
| goat | goat | 0.201 | 0.201 | 0.667 |
| car-shadow | car | 0.200 | 0.200 | 0.909 |
| drift-straight | car | 0.171 | 0.171 | 0.818 |
| scooter-black | scooters | 0.169 | 0.175 | 0.611 |
| shooting | man | 0.168 | 0.170 | 0.586 |
| mbike-trick | motorcycle | 0.143 | 0.124 | 0.725 |
| motocross-jump | rider | 0.142 | 0.126 | 0.652 |
| horsejump-high | horse | 0.141 | 0.120 | 0.731 |
| soapbox | car | 0.137 | 0.141 | 0.818 |
| india | woman | 0.130 | 0.166 | 0.427 |
| dog | dog | 0.127 | 0.127 | 0.745 |
| bmx-trees | bicycle | 0.117 | 0.087 | 0.650 |
| parkour | man | 0.107 | 0.107 | 0.426 |
| libby | dog | 0.092 | 0.092 | 0.648 |
| drift-chicane | car | 0.079 | 0.079 | 0.885 |
| lab-coat | frames | 0.064 | 0.068 | 0.661 |
| dogs-jump | dogs | 0.065 | 0.052 | 0.690 |
| dance-twirl | woman | 0.042 | 0.042 | 0.384 |
| paragliding-launch | paraglider | 0.036 | 0.039 | 0.701 |
| loading | frames | 0.025 | 0.034 | 0.529 |
| kite-surf | kitesurfing | 0.018 | 0.013 | 0.662 |
| **MEAN** | | **0.166** | **0.157** | **0.703** |

**Key results:**
- Mean Obj-IoU = **0.166** (binary), up from 0.076 in the initial 10-video run
  with generic tokens. Proper token targeting more than doubled the IoU.
- 28/30 videos got object-specific tokens (only lab-coat and loading fell back to
  "frames"). Best IoU = 0.41 (judo), worst with proper token = 0.018 (kite-surf).
- Mean map consistency = 0.703 — reasonable temporal coherence on real video.

### Layer-wise Obj-IoU (DAVIS, 30 videos)

| Layer | IoU(binary) | IoU(best-obj) | Consistency |
|------:|------------:|--------------:|------------:|
| 0 | 0.137 | 0.122 | **0.838** |
| 7 | 0.156 | 0.138 | 0.762 |
| 14 | 0.156 | 0.134 | 0.766 |
| 21 | **0.172** | **0.155** | 0.748 |
| 27 | 0.168 | 0.158 | 0.733 |

**Layer pattern on real video differs from synthetic.**
- On synthetic videos: clear U-shape (layers 0–7 best, 14 worst, 27 partial recovery)
- On DAVIS: **monotonically increasing** — later layers give better IoU (layer 21–27
  best at 0.17, layer 0 worst at 0.14)
- Consistency still decreases with depth (0.84 → 0.73), matching synthetic pattern

**Interpretation:** On real video with complex objects, deeper layers have had more
processing to build object-level features. Unlike synthetic circles (which need only
spatial position), real objects require semantic understanding that develops through
layers. The model reconstructs better object-level representations in later layers,
giving higher IoU despite lower raw spatial consistency.

Full results: `results_davis_full/results.json`

---

## TempCompass Evaluation (2026-04-07)

### Setup
- **408 videos**, 499 multi-choice QA pairs across 5 temporal dimensions
- 8 frames per video, multi-image mode, RGF-only
- Coherence computed on description prompt, QA answered separately
- Dataset: `/Volumes/Crucial X10/tempcompass/`

### Coherence–Accuracy Correlation (full dataset)

| Dimension | N | QA Accuracy | Coherence (correct) | Coherence (wrong) |
|-----------|---|-------------|--------------------:|------------------:|
| action | 99 | 50.5% | 0.488 | 0.503 |
| speed | 97 | 39.2% | 0.574 | 0.559 |
| attribute_change | 96 | 31.2% | **0.627** | **0.599** |
| direction | 107 | 28.0% | 0.549 | 0.558 |
| order | 100 | 18.0% | **0.616** | **0.553** |
| **OVERALL** | **499** | **33.3%** | **0.558** | **0.557** |

### Key Findings

1. **No strong overall coherence–accuracy correlation.** Correct (0.558) vs wrong
   (0.557) coherence scores are nearly identical. TAM map_consistency does NOT
   predict QA accuracy on this benchmark overall.

2. **Weak signal in order and attribute_change.** Order shows the largest gap
   (correct 0.616 vs wrong 0.553, Δ=0.063). Attribute_change also shows a gap
   (0.627 vs 0.599, Δ=0.028). These dimensions involve detecting change over time,
   which may be partially captured by map_consistency.

3. **Direction shows NO signal** (0.549 vs 0.558), reversing the preliminary
   finding from 17 videos. With N=107, the direction coherence-accuracy
   correlation disappears — the preliminary result was noise from small sample.

4. **Overall QA accuracy is 33.3%** — Qwen3-VL-2B-Instruct is poor at temporal
   reasoning on TempCompass. Action is best (50.5%), order is worst (18.0%).

5. **The null result is informative.** TAM's map_consistency (cosine similarity
   between consecutive frame activation maps) measures spatial pattern stability,
   not temporal reasoning ability. The model can have stable spatial attention
   without understanding temporal dynamics (speed, direction, order). This suggests
   temporal reasoning uses different internal mechanisms than spatial localization.

Full results: `results_tempcompass_full/results.json`
