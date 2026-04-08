# Next Steps: Model Scaling & Deeper Investigation

**Date:** 2026-04-08
**Status:** Decision point after initial experiments on Qwen3-VL-2B

---

## Why We Need to Scale Up

Our experiments on Qwen3-VL-2B revealed:
1. **The model is too weak for temporal reasoning** — 33.3% on TempCompass (near random for multi-choice). A model that can't do the task can't show interpretable patterns.
2. **Spatial localization works** — 0.166 IoU on DAVIS, 0.703 temporal coherence. TAM captures WHERE objects are, just not temporal dynamics.
3. **Two TempCompass dimensions showed weak signal** — "order" (gap=0.063) and "attribute_change" (gap=0.028) both involve detecting change over time. Worth investigating with a stronger model.
4. **Layer patterns differ between synthetic and real video** — synthetic shows U-shape, DAVIS shows monotonic increase. Need to check if this holds at larger scale.

**Hypothesis:** A stronger model (8B, non-quantized) that actually succeeds at temporal reasoning (50%+ accuracy) should show clearer interpretable patterns. The null result may be an artifact of model weakness, not method failure.

---

## Experiments to Run

### Priority 1: TempCompass on Qwen3-VL-8B (Most Important)

**Why:** This directly tests whether the null result was due to model weakness.

```bash
python3 eval_tempcompass.py \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --dataset-path /path/to/tempcompass/ \
    --output-dir results_tempcompass_8B/ \
    --no-quantize --no-eci --max-frames 8 --max-new-tokens 60
```

**What to look for:**
- Overall QA accuracy: if it's 50%+ (vs 33% for 2B), the model is actually doing temporal reasoning
- Coherence-accuracy gap: does the gap widen for "order" and "attribute_change"?
- Does a new gap appear for "direction" or "speed"?
- Compare the coherence-accuracy table side-by-side with the 2B results

**Expected outcome:** If the 8B model scores 55%+ and shows coherence-accuracy gap > 0.05, we have a viable research direction. If still null, the method genuinely doesn't capture temporal reasoning.

### Priority 2: DAVIS on Qwen3-VL-8B

**Why:** Check if layer patterns change with a stronger model.

```bash
python3 eval_davis.py \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --dataset-path /path/to/DAVIS/ \
    --output-dir results_davis_8B/ \
    --no-quantize --no-eci --max-frames 8 --max-new-tokens 60 \
    --multilayer --layer-indices 0 7 14 21 27
```

**What to look for:**
- Does overall Obj-IoU improve? (expect yes — bigger model, better features)
- Does the layer pattern change? (U-shape vs monotonic increase)
- Is temporal coherence higher? (better model might maintain more consistent attention)

### Priority 3: Deep Dive into "Order" and "Attribute Change"

**Why:** These two dimensions showed the only positive signal. Understanding WHY could reveal what TAM actually captures about temporal processing.

**What to do:**
1. For TempCompass "order" questions (N=100 on 2B, likely similar on 8B):
   - Sort by coherence-accuracy gap (largest to smallest)
   - Examine the top-5 and bottom-5 cases manually
   - What makes "order" videos where high coherence → correct answer?
   - Hypothesis: these might be videos where object positions change in a way that TAM maps can capture (e.g., "A happens on the left first, then B on the right")

2. For "attribute_change" questions (N=96):
   - Same manual analysis
   - Hypothesis: attribute changes (color, size, shape) might cause activation map shifts that correlate with understanding

3. On 8B model: re-run just the "order" subset and compare

### Priority 4: Multi-Layer Fusion on DAVIS (Path 3)

**Why:** DAVIS showed IoU increases with depth (semantics) but consistency decreases (spatial precision). Combining layers might get best of both.

**What to try:**
- Weighted average of layer maps: `fused_map = 0.3 * layer_0 + 0.3 * layer_7 + 0.4 * layer_27`
- Compare fused IoU vs single-layer IoU
- Try learned weights (optimize on a held-out set)

This can be done on the existing 2B results first (no re-running needed — just re-combine the per-layer maps from `results_davis_full/results.json`).

---

## What We're Looking For

The experiments above test two hypotheses:

**Hypothesis A (Path 2):** TAM's temporal coherence captures *some* aspects of temporal reasoning (specifically change detection), but this signal is masked by model weakness on 2B. A stronger model will show clearer signal.
- **Test:** TempCompass on 8B
- **Kill criterion:** If 8B model gets >50% accuracy AND coherence gap is still < 0.03 overall, the method doesn't capture temporal reasoning. Period.

**Hypothesis B (Path 3):** Multi-layer fusion can improve spatial localization quality by combining early-layer spatial precision with late-layer semantic understanding.
- **Test:** Layer fusion on DAVIS
- **Kill criterion:** If fused maps don't beat last-layer maps on IoU, the insight isn't practically useful.

---

## Hardware Requirements

- **Qwen3-VL-8B non-quantized:** ~16GB VRAM minimum (bfloat16). Needs a GPU server — won't fit on Mac Mini M4 with 16GB unified memory.
- **Inference time:** Expect ~2-3x slower than 2B per sample. TempCompass (499 samples) should take a few hours on a single A100/H100.
- **DAVIS (30 videos, 8 frames each):** Should take ~1-2 hours on a single A100.

---

## Decision Points

After running Priority 1 (TempCompass 8B):

| Outcome | Next Action |
|---|---|
| 8B accuracy >50% AND coherence gap >0.05 | **PURSUE Path 2** — dig into temporal coherence as a real signal |
| 8B accuracy >50% BUT coherence gap <0.03 | **KILL Path 2** — method doesn't capture temporal reasoning. Focus on Path 3 only |
| 8B accuracy still <40% | Model is still too weak — try 72B or consider different model family (InternVL, LLaVA-Video) |

After running Priority 4 (Multi-layer fusion):

| Outcome | Next Action |
|---|---|
| Fused IoU > best single-layer by >0.01 | **PURSUE Path 3** — multi-layer fusion is a real contribution |
| Fused IoU ≈ best single-layer | **KILL Path 3** — insight isn't practically useful |
