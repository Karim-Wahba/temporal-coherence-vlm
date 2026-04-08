# Prototyping Plan: Temporal Activation Maps for Video Tracking in MLLMs

## Prerequisites

- TAM ported to Qwen3-VL (in progress via separate agent)
- DEX-AR installed (`pip install dexar_torch`) — currently LLaVA-only, needs Qwen3-VL extension
- GPU with enough VRAM for Qwen3-VL (2B for prototyping, 7B for full experiments)

---

## Phase 0: Setup and Verify TAM on Qwen3-VL — COMPLETE ✓

TAM ported and working on Qwen3-VL-2B-Instruct. COCO evaluation done (100 images).

**Key result**: Obj-IoU 0.224 (vs paper 0.274 on Qwen2-VL). Gap is due to ECI over-subtraction on Qwen3-VL. RGF-only gives 0.248 (matches paper). See `Qwen3-VL/tam/RESULTS.md`.

**Decision for temporal experiments**: Use **RGF-only** mode (skip ECI or cap at 0.3) for cleaner per-frame maps. ECI adaptation is future work.

**Next**: Run TAM on synthetic test videos and begin temporal analysis.

---

## Phase 1: Temporal Coherence — Does the Model Track?

**Goal**: Compute temporal coherence scores on TAM's per-frame activation maps. First signal of whether Qwen3-VL tracks objects.

### Experiment 1a: Simple moving object
```
Input: Video of a single object (e.g., car driving across the scene)
Prompt: "Describe this video."
Expected output token: "car"
```

**What to measure**:
- Extract per-frame activation map for "car": already available as `img_scores` split by frame in TAM
- Compute pairwise cosine similarity between consecutive frame maps
- Plot peak location (argmax x, y) across frames — does it follow the object?
- Compute spatial smoothness: mean displacement between consecutive peaks

**What to look for**:
- Smooth peak trajectory following the car → model tracks
- Random/static peaks → model re-detects or doesn't localize temporally
- Peaks following the car but noisy → partial tracking, needs temporal smoothing

### Experiment 1b: Two objects diverging
```
Input: Video of two objects (e.g., two people walking in opposite directions)
Prompt: "Describe this video."
Expected tokens: "person" (appears twice or as "people")
```

**What to measure**:
- If model generates separate tokens ("man", "woman") → compare their activation maps per frame
- If model generates "people" → look for multiple peaks, check if they diverge across frames
- Measure whether the model distinguishes the two objects spatially across time

### Experiment 1c: Occlusion
```
Input: Video where an object goes behind another object and re-emerges
Prompt: "Describe this video."
```

**What to measure**:
- Does the activation map "survive" occlusion (maintains rough location)?
- Does it snap to a wrong location during occlusion?
- Does it recover the correct location post-occlusion?
- Compare pre-occlusion and post-occlusion map similarity

### Code to write for Phase 1

```python
# temporal_analysis.py — core analysis module

import numpy as np
from scipy.spatial.distance import cosine

def compute_temporal_coherence(per_frame_maps):
    """
    Given a list of per-frame activation maps for one token,
    compute temporal coherence metrics.

    Args:
        per_frame_maps: list of np.ndarray, each shape (H, W)

    Returns:
        dict with coherence metrics
    """
    n_frames = len(per_frame_maps)

    # 1. Map consistency: cosine similarity between consecutive maps
    cos_sims = []
    for i in range(n_frames - 1):
        flat_a = per_frame_maps[i].flatten()
        flat_b = per_frame_maps[i+1].flatten()
        sim = 1 - cosine(flat_a, flat_b)
        cos_sims.append(sim)

    # 2. Peak trajectory: argmax location per frame
    peaks = []
    for m in per_frame_maps:
        idx = np.unravel_index(np.argmax(m), m.shape)
        peaks.append(idx)

    # 3. Spatial smoothness: displacement between consecutive peaks
    displacements = []
    for i in range(len(peaks) - 1):
        dy = peaks[i+1][0] - peaks[i][0]
        dx = peaks[i+1][1] - peaks[i][1]
        displacements.append(np.sqrt(dy**2 + dx**2))

    # 4. Intensity stability: variance of peak values
    peak_values = [m[p] for m, p in zip(per_frame_maps, peaks)]

    return {
        'map_consistency': np.mean(cos_sims),
        'map_consistency_std': np.std(cos_sims),
        'peak_trajectory': peaks,
        'spatial_smoothness': np.mean(displacements),
        'spatial_smoothness_std': np.std(displacements),
        'intensity_stability': np.std(peak_values),
        'peak_values': peak_values,
    }
```

**Integration with TAM**: Modify the video loop in TAM to collect per-frame `img_scores` before they get merged into the visualization. The raw scores are already available — we just need to save them.

Specifically, in `tam.py` `multimodal_process()` (video branch, line 365+):
- `img_scores` is split into per-frame maps via `np.array_split(img_scores, b)`
- These per-frame maps are what we feed into `compute_temporal_coherence()`

---

## Phase 2: Causal Temporal Probing — Tracking vs. Re-Detection

**Goal**: Directly test whether the model uses previous frames' visual information when generating activation maps for the current frame.

### Experiment design

For a video with N frames:
1. **Full run**: Generate normally, extract per-frame activation maps for token "X" → `maps_full`
2. **Ablated run**: Zero out (or mean-replace) the visual tokens for frame t-1, re-run forward pass, extract activation maps → `maps_ablated`
3. **Compare**: How much does frame t's map change between full and ablated runs?

```python
def temporal_dependency_score(maps_full, maps_ablated, frame_idx):
    """
    Measure how much frame t's activation map depends on frame t-1.

    High score → model uses previous frame (tracking)
    Low score → model processes frames independently (re-detection)
    """
    full_map = maps_full[frame_idx].flatten()
    ablated_map = maps_ablated[frame_idx].flatten()

    # Change in map when previous frame is ablated
    change = 1 - cosine(full_map, ablated_map)  # 0 = no change, 1 = totally different
    return change
```

### Implementation considerations
- **How to ablate**: Replace visual tokens for frame t-1 with zeros or the mean visual token value across all frames
- **Need to re-run model inference**: This requires modifying the model's input embeddings AFTER the vision encoder but BEFORE the LLM decoder
- **For Qwen3-VL**: The visual tokens are interleaved with text tokens. We need to identify which token positions correspond to which frame (using `video_grid_thw` from the processor)
- **Cost**: Each ablation requires a separate forward pass. For N frames, that's N+1 forward passes total. Start with short videos (10 frames).

### What to vary
- Ablate frame t-1 only → immediate temporal dependency
- Ablate frames t-1 through t-k → temporal receptive field (how far back does the model look?)
- Ablate ALL other frames → does the model use temporal context at all?

---

## Phase 3: Multi-Layer TAM Extension (Logit Lens)

**Goal**: Extend TAM's pipeline to compute activation maps at every layer, not just the last.

**Update**: DEX-AR does not work well with Qwen series (confirmed by authors, March 2026). We implement the multi-layer idea directly on TAM instead. This is simpler (forward-pass only, no gradients) and compatible with Qwen3-VL.

### Implementation

The change is minimal. TAM already captures all layer hidden states:
```python
# demo.py line 30-34:
outputs = model.generate(..., output_hidden_states=True, return_dict_in_generate=True)
```

Current code uses only the last layer:
```python
# demo.py line 43:
logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
```

Multi-layer extension:
```python
n_layers = len(outputs.hidden_states[0])
logits_per_layer = {}
for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:  # sample 5 layers
    logits_per_layer[layer_idx] = [model.lm_head(feats[layer_idx]) for feats in outputs.hidden_states]
```

Then run the full TAM pipeline (ECI + Rank Gaussian Filter) at each layer independently, generating per-frame activation maps at each layer depth.

### Validation
- Run on a simple image first — do middle-layer maps look different from last-layer maps?
- For video: compute temporal coherence at each layer
- Plot coherence vs layer depth → find where tracking info lives

### Note on DEX-AR
DEX-AR codebase is still worth reading as a reference for understanding the logit lens and head filtering concepts (exploration agent is running). But we won't try to port their code to Qwen3-VL.

---

## Phase 4: Layer-wise Tracking Analysis

**Goal**: Using multi-layer TAM extension on Qwen3-VL video, compute temporal coherence at each layer independently.

### Experiment
- For the same test videos from Phase 1, run multi-layer TAM (Phase 3)
- Extract per-frame activation maps at sampled layers (e.g., layers 0, L/4, L/2, 3L/4, L-1)
- Compute temporal coherence at each layer using `temporal_analysis.py`
- Plot: temporal coherence vs. layer depth

### Expected findings
- **If tracking peaks in middle layers**: Visual information decay confirmed — the model computes tracking features but loses them by the final layer. This is a core finding.
- **If tracking is uniform across layers**: The model either consistently tracks or consistently doesn't — layer depth doesn't matter.
- **If tracking is only in final layers**: Against our hypothesis, but still interesting — means last-layer methods (TAM) are sufficient.

---

## Phase 6: Create Synthetic Test Videos

**Goal**: Controlled test videos with ground truth for rigorous evaluation.

### Video types to create

1. **Simple translation**: Solid-color circle/square moving linearly across a plain background. 10 frames. Ground truth: linear trajectory. (Easiest sanity check.)

2. **Two identical objects crossing**: Two identical circles moving toward each other, crossing paths, continuing. Ground truth: two trajectories that cross. (Tests identity tracking vs. detection.)

3. **Appearance change**: Object gradually changes color while moving. Ground truth: same trajectory. (Tests identity maintenance.)

4. **Occlusion**: Object moves behind a barrier and re-emerges on the other side. Ground truth: trajectory with occluded segment. (Tests object permanence.)

5. **Re-entry**: Object exits the frame and re-enters from a different side. Ground truth: trajectory with gap. (Tests long-range identity.)

### Implementation
- Use OpenCV or PIL to generate simple synthetic videos
- Each video: 10-20 frames, 336x336 or 448x448 pixels
- Store as individual frame images (compatible with TAM's video input format)
- Include ground truth trajectories as JSON (frame_idx → (x, y))

```python
# synthetic_videos.py — minimal example

import numpy as np
from PIL import Image, ImageDraw

def make_moving_circle(n_frames=10, size=336, radius=20):
    """Single circle moving left to right."""
    frames = []
    trajectory = []
    for i in range(n_frames):
        img = Image.new('RGB', (size, size), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        x = int(radius + (size - 2*radius) * i / (n_frames - 1))
        y = size // 2
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='red')
        frames.append(img)
        trajectory.append((x, y))
    return frames, trajectory
```

---

## Summary: What to Do When

| Phase | Depends on | Status |
|---|---|---|
| Phase 0: Verify TAM on Qwen3-VL | TAM-Qwen3-VL port | In progress (separate agent) |
| Phase 1: Temporal coherence experiments | Phase 0 | After Phase 0. Analysis module ready (`temporal_analysis.py`) |
| Phase 2: Causal temporal probing | Phase 0 | After Phase 0 |
| Phase 3: Multi-layer TAM extension | Phase 0 | Small code change on TAM pipeline |
| Phase 4: Layer-wise tracking analysis | Phase 3 | After Phase 3 |
| Phase 5: Synthetic test videos | Nothing | In progress (background agent) |

**Recommended order**:
- Phase 0 (wait for TAM port) → Phase 1 (first coherence results) → Phase 3 (multi-layer extension) → Phase 4 (layer-wise analysis) → Phase 2 (causal ablation)
- Phase 5 runs in parallel (already started)

**Note**: DEX-AR exploration agent is running in background as reference material, but we no longer depend on their codebase for Qwen3-VL.
