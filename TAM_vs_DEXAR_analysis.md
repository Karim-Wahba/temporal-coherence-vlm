# Analysis: TAM vs DEX-AR for MLLM Explainability

## Papers

- **TAM**: Token Activation Map to Visually Explain Multimodal LLMs (Li et al., 2025) — arXiv:2506.23270
- **DEX-AR**: A Dynamic Explainability Method for Autoregressive Vision-Language Models (Bousselham et al., 2026) — arXiv:2603.06302

---

## 1. How TAM Works

### Core Mechanism

TAM generates activation maps for individual generated tokens in MLLMs by adapting the Class Activation Map (CAM) concept from CNNs.

The activation map for a given token is computed as:

```
A_i = ⌊F^v · w_{t_i}⌋₊
```

where F^v are visual features and w_t is the LM head (token classifier) weight vector for a specific token.

### What "Visual Features" Actually Are (Confirmed from Code)

From the TAM codebase (`demo.py`), the visual features are the **last-layer hidden states at visual token positions** — not the raw vision encoder outputs before entering the LLM decoder.

```python
# Generate with hidden states
outputs = model.generate(..., output_hidden_states=True, return_dict_in_generate=True)

# Apply LM head to LAST layer hidden states
logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]
```

The full pipeline:

```
Image → Vision Encoder → Projector → visual tokens enter LLM decoder
                                          ↓
                          All decoder layers (self-attention mixes
                          visual + text tokens across every layer)
                                          ↓
                          Last layer hidden states [batch, seq_len, c]
                                          ↓
                          model.lm_head() → logits [batch, seq_len, vocab]
                                          ↓
                          logits[:, visual_positions, target_token_id]
                                          ↓
                          = activation map over visual positions
```

By this point, visual and textual information have been fully mixed through self-attention. Each visual token position still retains spatial identity (corresponding to a specific image patch), so the activation map can be reshaped to a 2D spatial heatmap.

### CAM vs TAM Analogy

| | CAM (CNNs) | TAM (MLLMs) |
|---|---|---|
| Feature map | Last conv layer spatial features | Last-layer hidden states at visual positions (F^v) |
| Classifier weights | FC layer weights for one class | LM head weights for each generated token |
| Output | 1 activation map per class | 1 activation map per generated token |

### TAM's Two Key Innovations

1. **Estimated Causal Inference (ECI)**: Because MLLMs generate tokens autoregressively, earlier context tokens introduce redundant activations that "bleed into" later tokens' maps. TAM subtracts a scaled interference map (Eq. 2) estimated from all earlier context tokens, using least-squares optimization to find the right scale factor.

2. **Rank Gaussian Filter**: Transformer activations exhibit salt-and-pepper noise. TAM applies a novel denoising filter that ranks values within a sliding window and applies a Gaussian kernel weighted by the coefficient of variation.

---

## 2. The Layer Problem: Visual Information Decay in MLLMs

### The Issue

TAM is locked to the **last layer only** because the LM head is trained to work with last-layer representations. However, research has shown that visual information gradually decreases as representations pass through deeper LLM decoder layers — the representations become increasingly text-dominant.

This creates a fundamental tension:

| | Early/Mid Layers | Last Layer |
|---|---|---|
| Visual info at visual positions | Rich | Diminished |
| Compatible with LM head | No | Yes |
| TAM can use it | No | Yes |

TAM explains what visual information the last layer still retains, which may not fully reflect what the model actually "saw" and used during reasoning in earlier layers. Visual information consumed in middle layers may leave no trace at the final visual positions.

Despite this limitation, TAM still achieves good empirical localization (IoU scores), suggesting the last layer retains enough spatial signal for explanation — even if it's not the complete picture.

### Alternative Approaches for Middle Layers

- **Probing classifiers** trained on intermediate layer features
- **Attention-based methods** (e.g., attention rollout) that track information flow across layers
- **Logit lens** — projecting intermediate hidden states through the LM head (imperfect but informative)

---

## 3. How DEX-AR Addresses These Issues

### Multi-Layer Analysis

DEX-AR computes intermediate logits at **every layer** using the logit lens approach:

```
o^{l,t} = LM_Head(Z^{l,t}_{-1})    for each layer l ∈ {1,...,L}
```

It then computes gradients of these logits w.r.t. attention maps at each layer. The final heatmap aggregates across all layers and all heads:

```
E̅^(t) = Σ_l Σ_i  w^{l,t,i} · ∇A^{l,t}_{-1,v}
```

This means early/middle layers where visual information is still rich contribute to the final explanation.

### Dynamic Head Filtering (Handling Visual Decay)

For each attention head at each layer, DEX-AR computes:

- S_img = max gradient magnitude for visual tokens
- S_text = max gradient magnitude for text tokens
- Weight: **w = (S_img − S_text)⁺**

Effect:

- **Later layers** where heads become text-dominant → S_text > S_img → weight ≈ 0 (automatically filtered out)
- **Early/middle layers** where heads still attend to visual info → S_img > S_text → weight is high

The visual decay problem is naturally handled — the method automatically discovers which layers and heads still carry visual information and upweights them.

### Token-Level Filtering (Filler Word Suppression)

DEX-AR also computes a per-token weight δ^t that suppresses tokens predicted from linguistic context rather than visual content:

```
δ^t = (max_{l,i} S^{l,t,i}_img − max_{l,i} S^{l,t,i}_text)⁺
```

This filters out filler tokens ("the", "is", "and") when aggregating to a sequence-level map.

### DEX-AR's Explicit Positioning Against TAM

From their related work: *"TAM relies on static visual features and post-hoc statistical estimation. In contrast, DEX-AR utilizes layer-wise gradients to capture the dynamic attention mechanism at each specific generation step."*

---

## 4. Remaining Limitations of DEX-AR

1. **Logit lens imperfection**: Applying `LM_Head()` to intermediate hidden states is an approximation — the LM head is only trained with last-layer features. DEX-AR uses these for gradient signals rather than direct feature maps, which partially mitigates this, but it is still a known concern.

2. **Attention ≠ contribution**: DEX-AR relies on gradients of attention maps, not hidden state features directly. Attention weights don't always reflect true information flow (e.g., attention sinks, residual stream contributions that bypass attention).

3. **No explicit causal inference**: Unlike TAM's estimated causal inference for inter-token interference, DEX-AR uses head/token filtering as a proxy. It filters filler tokens but doesn't explicitly model the causal contamination from context tokens.

4. **Computational cost**: Backpropagating through all layers × all heads × all generation steps is significantly more expensive than TAM's forward-pass-only approach.

---

## 5. Comparative Summary

| Aspect | TAM | DEX-AR |
|---|---|---|
| Layers used | Last only | All layers (dynamically weighted) |
| Visual info decay | Not addressed | Handled via dynamic head filtering |
| Signal source | Hidden state features (forward pass) | Attention map gradients (backprop) |
| Inter-token interference | Explicit causal inference (ECI) | Proxy via filler-word filtering |
| Denoising | Rank Gaussian filter | Head filtering acts as implicit denoising |
| Computational cost | Light (forward only) | Heavy (gradients at all layers) |
| LM head at intermediate layers | N/A (only uses last) | Present but mitigated by using gradients |
| Token-level maps | Yes | Yes |
| Sequence-level maps | No (per-token only) | Yes (weighted aggregation of per-token maps) |
| Models evaluated | Qwen2-VL, InternVL2.5, LLaVA1.5 | LLaVA-1.5, BakLLaVA, PaliGemma, Florence-2 |

### Complementarity

The two methods address different weaknesses and are somewhat complementary:

- **TAM excels at**: handling inter-token interference through explicit causal modeling; lightweight computation
- **DEX-AR excels at**: capturing multi-layer visual information; distinguishing visually-grounded vs. linguistic tokens; working across diverse architectures (decoder-only, encoder-decoder, prefix-decoder)
