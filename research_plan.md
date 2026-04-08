# Research Plan: Temporal Activation Maps for Video Object Tracking Interpretability in MLLMs

## Context

**Problem**: Video reasoning in MLLMs requires tracking objects across frames, but we have no way to verify whether MLLMs genuinely track or just re-detect per frame. Asking MLLMs to predict tracking is unreliable (possible hallucination). Activation maps provide a more direct window into the model's actual behavior.

**Gap**: No published work uses interpretability tools to assess cross-frame object tracking in MLLMs. TAM handles video but treats each frame independently. DEX-AR is image-only.

**Scope**: Standalone interpretability paper. Build a TAM-based (forward-pass, lightweight) method that extends activation maps to reveal temporal object tracking behavior in video MLLMs. Improving reasoning is a follow-up.

---

## Current Status (2026-04-08)

### Experiments Completed

**1. TAM on Qwen3-VL-2B (COCO):** Obj-IoU 0.224, F1-IoU 0.357. ECI over-subtracts; RGF-only gives 0.248.

**2. Synthetic Video Temporal Coherence:** Object localization works (2.4-2.9 map-px accuracy on 14x14 grid). U-shaped layer curve observed on synthetic data (early layers best).

**3. DAVIS 2017 (30 videos):** Obj-IoU 0.166 with proper token targeting (doubled from 0.077 with generic tokens). **Layer pattern on real video differs from synthetic: monotonically increasing IoU with depth** (layer 0: 0.137 → layer 21: 0.172). Complex objects benefit from deeper semantic processing.

**4. TempCompass (408 videos, 499 QA):** **NULL RESULT.** No coherence-accuracy correlation overall (correct 0.558 vs wrong 0.557). Weak signal only in "order" (Δ=0.063) and "attribute_change" (Δ=0.028). Model QA accuracy only 33.3% — too weak for temporal reasoning.

### What's Dead
- U-shaped curve as universal finding (synthetic-specific, not confirmed on real video)
- Temporal coherence predicting hallucination/accuracy (null result on TempCompass)
- "Early layers always better for localization" (real video shows opposite)

### What Still Holds
- TAM works on Qwen3-VL for spatial localization (0.166 IoU on DAVIS)
- Token selection is a critical bottleneck (2x improvement with proper tokens)
- Layer patterns differ by content complexity (interesting finding in itself)
- Weak signal in "order" and "attribute_change" temporal dimensions
- Spatial coherence ≠ temporal reasoning (informative null result)

### Open Hypothesis
The null result may be an artifact of **model weakness** (2B model, 33.3% accuracy). A stronger model (8B non-quantized, targeting 50%+ accuracy) might show clearer patterns. See `NEXT_STEPS.md` for detailed experiment plan.

### Infrastructure Ready
- TAM on Qwen3-VL: working (COCO, DAVIS, TempCompass evaluation pipelines)
- Multi-layer logit lens: working at 5 layer depths [0, 7, 14, 21, 27]
- Temporal analysis module + synthetic test videos
- eval_davis.py, eval_tempcompass.py: ready for 8B model

---

## Decisions Made

- **Paper scope**: Interpretability method first (standalone), reasoning improvement later
- **Base method**: TAM with multi-layer logit lens extension (DEX-AR doesn't work on Qwen, confirmed by authors)
- **Starting model**: Qwen3-VL-2B-Instruct (ported, working)
- **Key insight**: TAM and DEX-AR solve two orthogonal problems:

| | TAM | DEX-AR |
|---|---|---|
| **What it solves** | Inter-token interference | Inter-layer visual decay |
| **Dimension** | Token (causal inference across tokens) | Layer (dynamic head filtering across layers) |
| **Mechanism** | ECI subtracts scaled context-token interference | Weights heads by `(S_img - S_text)⁺` across all layers |
| **Limitation** | Only uses last layer | No inter-token causal inference |

A hybrid method that combines TAM's ECI (clean per-token maps) with DEX-AR's multi-layer head filtering (visual info from all layers) would address both limitations — **this combination is novel and neither paper does it**.

---

## Literature Gap

| Area | Image MLLMs | Video MLLMs |
|---|---|---|
| Per-token activation maps | TAM, DEX-AR | TAM (per-frame only) |
| Attention/mechanistic analysis | Attention sinks, Cross-modal flow, NOTICE, EmbedLens | Attention knockouts (empirical) |
| Object tracking via interpretability | N/A | **No published work** |

Closest related work:
- "Causality Matters" (2025) — temporal info pathways, event ordering not object tracking
- "How Video-LLMs Answer Video Questions" (2025) — attention knockouts, ablation-based
- "Temporal-Aware Activation Engineering" (NeurIPS 2025) — activation analysis for hallucination
- NExT-GQA (CVPR 2024) — attention at frame level, not spatial tracking
- VTCD (CVPR 2024) — concept discovery in video transformers (not MLLMs)

---

## Proposed Method: Temporal Activation Map (TAM-V)

### Step 1: Per-Frame Activation Maps — TAM as Foundation + Multi-Layer Extension

**Practical constraint**: DEX-AR does not work well with Qwen series models (confirmed by DEX-AR authors, March 2026). Their gradient-based approach has architectural compatibility issues with Qwen's attention mechanism (likely GQA + SDPA/Flash Attention not exposing attention weights easily). So we build on TAM and implement the multi-layer ideas ourselves.

**Base: TAM on Qwen3-VL (in progress)**
- Project last-layer hidden states at visual token positions through LM head
- Apply Estimated Causal Inference (ECI) to remove inter-token interference
- Apply Rank Gaussian Filter for denoising
- Result: clean per-token activation maps (last layer)

**Extension: Logit Lens at Multiple Layers (our implementation, inspired by DEX-AR)**
- TAM already captures hidden states from all layers via `output_hidden_states=True` (see `demo.py` line 34)
- Currently only uses the last layer: `model.lm_head(feats[-1])`
- Simple extension: apply `model.lm_head(feats[l])` for each layer l → intermediate activation maps
- Apply TAM's ECI at each layer independently → clean per-token maps at every layer
- No gradient computation needed — this is still forward-pass only, keeping it lightweight
- Compare temporal coherence across layers to find WHERE tracking information lives

**Why this works without DEX-AR's gradient approach:**
- The logit lens (projecting intermediate hidden states through LM head) is the core idea
- DEX-AR's dynamic head filtering requires attention map gradients, which is what breaks on Qwen
- But we can achieve a similar goal: instead of filtering heads, we filter LAYERS by their temporal coherence
- Layers where activation maps are temporally consistent are the ones carrying tracking info
- Layers where maps are noisy/inconsistent are not useful for tracking
- This is actually a cleaner approach: let the temporal coherence metric itself tell us which layers matter

**Code: TAM multi-layer extension is minimal:**
```python
# Current TAM (last layer only):
logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

# Multi-layer extension:
n_layers = len(outputs.hidden_states[0])  # number of layers
logits_per_layer = {}
for layer_idx in range(n_layers):
    logits_per_layer[layer_idx] = [model.lm_head(feats[layer_idx]) for feats in outputs.hidden_states]
```

**Codebase**: TAM at `/Users/owl/Documents/Karim_project/TAM_code/` — already supports Qwen2-VL, being ported to Qwen3-VL

### Step 2: Cross-Frame Temporal Analysis (NEW)

**Key insight**: TAM's video code already splits visual token scores by frame (`np.array_split(img_scores, b)` in `tam.py` line 372). So for a generated token like "dog", we already get separate activation maps per frame: `[map_frame_0, map_frame_1, ..., map_frame_N]`. The token identity provides the cross-frame correspondence — no explicit linking needed for the basic case.

**Approach A — Direct map comparison (primary, no linking needed):**
- For token "dog", we have N per-frame activation maps from the same forward pass
- Compare consecutive maps directly:
  - Cosine similarity between flattened maps
  - IoU of thresholded activation regions
  - Peak displacement between consecutive frames
- If the model tracks: maps smoothly follow the object across frames
- If the model re-detects per frame: maps may be inconsistent, noisy, or snap to different regions

**Approach B — Peak trajectory extraction (secondary, for evaluation):**
- Find the argmax peak in each frame's map → (x, y) per frame → trajectory
- Compare against ground truth object trajectory using tracking metrics
- Handles single-object case cleanly
- For multi-peak maps (e.g., token "dogs" with two objects): apply peak detection + Hungarian matching across frames

**When do we need explicit linking?**
- Single object, specific token (e.g., "dog", "car") → **no linking needed**, Approach A suffices
- Multiple objects of the same category (e.g., "dogs") → need multi-peak detection + matching
- Sentence-level aggregation → use DEX-AR's token weighting (δ^t) to combine per-token maps

### Step 3: Temporal Coherence Score (NEW)
Quantitative metric for how well the model "tracks", computed per token across frames:
- **Map consistency**: cosine similarity between consecutive per-frame activation maps (high = consistent attention)
- **Spatial smoothness**: peak displacement between frames (smooth = tracking, jumpy = re-detection)
- **Intensity stability**: variance of peak activation strength across frames
- **Occlusion robustness** (for occlusion test videos): does the map recover the correct location post-occlusion?
- Combine into a single scalar "temporal coherence score" per object-token

### Step 4: Layer-wise Tracking Analysis (NEW — enabled by multi-layer TAM extension)
- Apply logit lens at each layer: `model.lm_head(feats[l])` for l = 0, ..., L-1
- Compute per-frame activation maps at each layer, apply ECI at each layer
- Compute temporal coherence at each layer independently
- Plot: temporal coherence score vs. layer depth → **find where tracking lives**
- Hypothesis: tracking coherence peaks in middle layers and drops at the final layer
- This would explain why last-layer-only TAM might underestimate tracking capability
- No gradient computation or attention map access needed — purely forward-pass based, compatible with Qwen3-VL

### Step 5: Causal Temporal Probing (NEW — key experiment)
- Ablate visual tokens from frame t-1, measure activation map change at frame t
- "Temporal dependency score" reveals: does the model genuinely use previous frames (tracking) or independently process each frame (re-detection)?
- Run with both TAM and DEX-AR maps to see if the answer differs by layer
- This directly answers: **is the MLLM tracking or re-detecting?**

---

## Experimental Design

### Experiment 1: Controlled Synthetic Videos
Create videos with ground truth to isolate specific tracking behaviors:
- Single object moving on plain background (baseline)
- Two identical objects crossing paths (requires identity tracking)
- Object undergoing appearance change (identity vs. appearance)
- Object occluded then re-emerging (object permanence)
- Object leaving and re-entering frame (long-range identity)

### Experiment 2: Benchmark Evaluation
- Convert activation map peaks to point tracks
- Evaluate on TAP-Vid or similar with standard tracking metrics
- Compare against dedicated trackers (SAM2, CoTracker) — not to "beat" them but to quantify what MLLMs capture internally

### Experiment 3: Causal Ablation ("Tracking vs. Re-Detection")
- For each frame, ablate previous frames' visual tokens
- Measure activation map change → temporal dependency score
- Vary the gap (ablate t-1, t-2, ..., t-k) to map temporal receptive field
- Compare across models and layers

### Experiment 4: Correlation with Video QA Performance
- Run on video QA benchmarks (VideoHallucer, STAR, etc.)
- Compute temporal coherence scores
- Test whether low coherence predicts incorrect answers / hallucination

### Experiment 5: Cross-Architecture Comparison
- Qwen3-VL, LLaVA-Video, InternVL3
- Compare temporal coherence profiles
- Identify architectural features that promote genuine tracking

---

## Key Files and Code to Build On

- **TAM codebase**: `/Users/owl/Documents/Karim_project/TAM_code/`
  - `tam.py` — core TAM function (line 440), activation map computation (line 560-572), ECI (line 578-593)
  - `demo.py` — Qwen2-VL integration (line 9-87), hidden states → logits pipeline (line 43)
  - Video support already exists: `vision_shape` handles `(batch, t_h, t_w)` for video in `multimodal_process` (line 365-398)
- **DEX-AR codebase**: `github.com/WalBouss/DEX-AR` / `pip install dexar_torch`
  - Reference implementation for logit lens + dynamic head filtering concepts
  - **Does NOT work well with Qwen series** (confirmed by authors, March 2026)
  - We implement the logit lens idea directly on TAM's Qwen3-VL pipeline instead (forward-pass only, no gradients needed)
- **Papers**: `/Users/owl/Documents/Karim_project/2506.23270.pdf` (TAM), `/Users/owl/Documents/Karim_project/2603.06302.pdf` (DEX-AR)
- **Analysis doc**: `/Users/owl/Documents/Karim_project/TAM_vs_DEXAR_analysis.md`

---

## What Needs to Be Built (New Code)

1. **TAM multi-layer extension** — modify TAM's pipeline to compute activation maps at every layer via logit lens (`model.lm_head(feats[l])` for each layer), not just the last. Apply ECI at each layer. This is a small code change on top of the existing TAM Qwen3-VL pipeline.

2. **Temporal coherence module** — DONE (`temporal_analysis.py`). Computes map consistency, peak trajectory, spatial smoothness, intensity stability. Also evaluates against ground truth trajectories.

3. **Synthetic test videos** — in progress (background agent). 5 controlled video types with ground truth trajectories.

4. **Causal ablation framework** — zero out / mean-replace visual tokens from specific frames, re-run forward pass, compare activation maps. Requires identifying which token positions correspond to which video frame in Qwen3-VL.

5. **Evaluation pipeline** — convert activation trajectories to point tracks, compute tracking metrics (position accuracy, occlusion recovery). Partially done in `temporal_analysis.py`.

6. **Layer-wise tracking analysis** — compute temporal coherence at each layer independently using the multi-layer TAM extension, plot coherence vs layer depth to find where tracking lives.

---

## Anticipated Findings and Paper Framing

The paper framing depends on what we find. All outcomes are publishable:

| Finding | Narrative | Impact |
|---|---|---|
| MLLMs don't track (re-detect per frame) | "We reveal that video MLLMs lack genuine temporal tracking — explaining systematic video reasoning failures" | High — motivates architectural changes |
| MLLMs partially track (some layers/heads do) | "Temporal tracking emerges in middle layers but is lost by the final layer" | High — connects to visual info decay |
| MLLMs do track | "Internal tracking in MLLMs can be read out as zero-shot video trackers" | Moderate-high — novel capability discovery |
| Coherence predicts hallucination | "Training-free video hallucination detection via temporal activation coherence" | High — practical application |

---

## Research Roadmap

**Phase 1 (2-3 weeks)**: Get TAM running on Qwen3-VL for video (in progress). Run on synthetic test videos (ready). Compute first temporal coherence scores. Visualize and sanity-check.

**Phase 2 (2-3 weeks)**: Extend TAM pipeline with multi-layer logit lens. Compute activation maps at every layer. Run temporal coherence at each layer → plot coherence vs depth → find where tracking lives.

**Phase 3 (2-3 weeks)**: Implement causal ablation experiments (frame-level visual token ablation). Answer "tracking vs. re-detection?" Test at multiple layers.

**Phase 4 (2 weeks)**: Cross-architecture comparison (Qwen3-VL, InternVL3, LLaVA-Video). Hallucination correlation. Full quantitative evaluation on real benchmarks.

**Phase 5 (2 weeks)**: Paper writing. Identify strongest narrative based on findings.

**Note**: DEX-AR exploration is still useful as reference for understanding the logit lens and head filtering concepts, but we won't depend on their codebase for Qwen3-VL. Our multi-layer extension is simpler (forward-pass only, no gradients).
