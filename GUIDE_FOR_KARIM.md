# Guide: What's on this branch and how to use it

Hi Karim — this branch (`experiment/tam-eval-pipelines`) contains evaluation
pipelines and analysis tools we built on a Mac Mini M4 with Qwen3-VL-2B. Your
codebase is different (GPU + 8B + original TAM submodule), so **don't try to
merge this branch directly**. Instead, cherry-pick the pieces you need.

---

## What's here and what's useful to you

### 1. NEXT_STEPS.md — READ THIS FIRST
The prioritized experiment plan. Tells you exactly what to run next, why, and
what results to look for. The most important thing on this branch.

### 2. Evaluation scripts (most useful to adapt)

**`eval_tempcompass.py`** — TempCompass temporal reasoning evaluation.
Computes QA accuracy + temporal coherence per dimension, tests whether
coherence predicts accuracy.

**`eval_davis.py`** — DAVIS 2017 evaluation. Computes per-frame Obj-IoU
against ground truth segmentation masks, with multi-layer analysis.

These scripts use our `tam/` module, which is different from yours. To use
them in your setup, you'll need to adapt the model loading and TAM calls to
match your existing code. The key parts to extract:

- **Obj-IoU computation** (in `tam/evaluation.py`) — OTSU thresholding +
  IoU against GT masks. This is model-independent.
- **Temporal coherence metrics** (in `temporal_analysis.py`) — cosine
  similarity between consecutive per-frame activation maps. Also
  model-independent.

### 3. Results (for reference)

**`tam/RESULTS.md`** — All our results on 2B. The key findings:

- **COCO**: Obj-IoU 0.224, F1-IoU 0.357
- **DAVIS 30 videos**: Obj-IoU 0.166. Token targeting matters hugely.
- **TempCompass 499 QA**: 33.3% accuracy, **no coherence-accuracy
  correlation** overall. Weak signal only in "order" (Δ=0.063) and
  "attribute_change" (Δ=0.028).
- **Multi-layer**: On synthetic video, early layers (0-7) have best spatial
  info. On real video (DAVIS), later layers (21-27) give best IoU.
  Pattern depends on content complexity.
- **ECI hurts on Qwen3-VL** — use `--no-eci` (RGF-only mode).

### 4. Multi-layer logit lens (small but important)

In `run_temporal_experiment.py` we compute activation maps at multiple
decoder layers, not just the last one. The core change is tiny. In your
code where you do:

```python
logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
```

To get multi-layer maps, do:

```python
layer_indices = [0, 7, 14, 21, 27]  # sample 5 of 28 layers
logits_per_layer = {}
for l in layer_indices:
    logits_per_layer[l] = [model.lm_head(feats[l]) for feats in outputs.hidden_states]
```

Then run TAM separately with each layer's logits. Compare Obj-IoU and
temporal coherence across layers to see where spatial information lives.

**Note:** This uses more memory (5x the logits). On your GPU with 8B you
may want to do one layer at a time and free between runs.

### 5. Temporal coherence metrics

`temporal_analysis.py` is standalone — you can copy it directly into your
repo. Usage:

```python
from temporal_analysis import compute_temporal_coherence, evaluate_tracking

# per_frame_maps: list of numpy arrays, each shape (H, W)
# These are the activation maps for one token across video frames
metrics = compute_temporal_coherence(per_frame_maps)
print(metrics['map_consistency'])    # cosine sim between consecutive frames
print(metrics['spatial_smoothness']) # mean peak displacement
print(metrics['peak_trajectory'])    # list of (row, col) per frame
```

In your video script, the per-frame maps are already available — TAM's
`multimodal_process` splits `img_scores` by frame via
`np.array_split(img_scores, b)`. You just need to save those before they
get turned into visualizations.

---

## What to run next (Priority 1)

**TempCompass on 8B** — this is the most important experiment. Our 2B model
got 33.3% accuracy (too weak for temporal reasoning), and the coherence-
accuracy correlation was null. The hypothesis is that 8B will be strong
enough (50%+) to show real signal.

You'll need:
1. TempCompass dataset (videos + QA JSON)
2. For each video: generate a description (to get TAM maps + coherence),
   then answer the multi-choice question separately
3. Compare coherence scores for correct vs wrong answers per dimension

The decision criterion from NEXT_STEPS.md:
- If 8B accuracy >50% AND coherence gap >0.05 → we have a research direction
- If 8B accuracy >50% BUT gap <0.03 → method doesn't capture temporal reasoning
- If 8B accuracy <40% → model still too weak, try larger or different family

---

## Files you can ignore

- `tam/` — our rewritten TAM module. You already have TAM via submodules.
- `tam/demo.py`, `tam/model_utils.py` — our model loading (Mac M4 specific).
- `synthetic_videos.py`, `test_videos/` — synthetic test video generator.
  Useful if you want controlled experiments but not urgent.
- `prototyping_plan.md`, `research_plan.md`, `TAM_vs_DEXAR_analysis.md` —
  background docs. Read if you want context on our thinking.
