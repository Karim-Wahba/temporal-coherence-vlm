"""
layer_tam.py
------------
Multi-layer / multi-aggregation TAM extractor.

Builds, for a single completed forward pass, one TAM result per requested
"variant" — either a single intermediate layer's logit-lens, or the average
of the last K layers' logit-lens.

The hidden states from `model.generate(..., output_hidden_states=True)` are
reused across all variants — only the LM head + (optional) final RMSNorm and
the per-token TAM loop re-run per variant.

Variant keys
------------
  "layer_-1", "layer_-2", ..., "layer_-10"  → single-layer logit-lens
  "cumavg_1", "cumavg_2",  ..., "cumavg_10" → mean of last K layers' logits

Each variant is independent — its own raw_map_records list (the ECI cache
TAM mutates internally), its own tam_maps and frame_mass.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'submodules' / 'TAM'))
from tam import TAM  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_final_norm(model) -> Optional[torch.nn.Module]:
    """Locate the language model's final RMSNorm. Returns None if not found."""
    candidates = [
        "model.norm",
        "language_model.model.norm",
        "model.model.norm",
        "model.language_model.model.norm",
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.Module):
            return obj
    return None


def _resolve_special_ids(inputs, processor):
    """Replicate the special_ids construction from extract_tam_from_generation."""
    token_ids = inputs.input_ids[0].cpu().tolist()
    token_strs = [
        processor.tokenizer.decode([t], skip_special_tokens=False)
        for t in token_ids
    ]

    vision_end_id = video_pad_id = None
    for tid, ts in zip(token_ids, token_strs):
        if "<|vision_end|>" in ts:
            vision_end_id = tid
        elif "<|video_pad|>" in ts or "<|image_pad|>" in ts:
            if video_pad_id is None:
                video_pad_id = tid

    im_start_id = None
    for tid, ts in zip(token_ids, token_strs):
        if "<|im_start|>" in ts:
            im_start_id = tid

    answer_start_positions = [
        i for i in range(len(token_ids) - 1) if token_ids[i] == im_start_id
    ]
    assistant_header_pos = (
        answer_start_positions[-1] if answer_start_positions
        else len(token_ids) - 1
    )
    input_len = inputs.input_ids.shape[1]
    answer_boundary = token_ids[assistant_header_pos:input_len]

    return {
        "img_id": [video_pad_id],
        "prompt_id": [[vision_end_id], answer_boundary],
        "answer_id": [answer_boundary, -1],
    }


def _resolve_vision_shape(inputs):
    if "video_grid_thw" in inputs:
        return (
            inputs["video_grid_thw"][0, 0].item(),
            inputs["video_grid_thw"][0, 1].item() // 2,
            inputs["video_grid_thw"][0, 2].item() // 2,
        )
    if "image_grid_thw" in inputs:
        H_tam = inputs["image_grid_thw"][0, 1].item() // 2
        W_tam = inputs["image_grid_thw"][0, 2].item() // 2
        T_img = inputs["image_grid_thw"].shape[0]
        return (T_img, H_tam, W_tam)
    raise ValueError("inputs missing video_grid_thw / image_grid_thw")


# ── per-variant logit construction ────────────────────────────────────────────

def _build_logits_for_layers(
    outputs,
    model,
    layer_idxs: List[int],
    apply_norm: bool,
):
    """
    Build a per-generation-step logit list = mean over layer_idxs of
    lm_head(norm(feats[L])) for each L.

    Streaming-averaged so we never hold more than one layer's worth of logits
    on the GPU at a time.

    Returns
    -------
    List[Tensor]  — one per generation step, each [1, seq_len_t, vocab]
    """
    norm = _find_final_norm(model) if apply_norm else None
    logits_per_step = []

    with torch.no_grad():
        for step_feats in outputs.hidden_states:
            avg = None
            for L in layer_idxs:
                hs = step_feats[L]
                if norm is not None:
                    hs = norm(hs)
                lg = model.lm_head(hs)
                avg = lg if avg is None else (avg + lg)
            avg = avg / len(layer_idxs)
            logits_per_step.append(avg)
    return logits_per_step


def _free_logits(logits):
    for i in range(len(logits)):
        logits[i] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── single-variant TAM run ────────────────────────────────────────────────────

def _run_tam_for_variant(
    all_gen_ids: List[int],
    vision_shape,
    logits_per_step,
    special_ids,
    vis_inputs,
    processor,
    n_gen_tokens: int,
) -> Dict:
    """
    Run TAM for every generated token using the supplied per-step logits.
    Returns dict with tam_maps + frame_mass.
    """
    raw_map_records: List[np.ndarray] = []
    tam_maps: List[Optional[np.ndarray]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in tqdm(range(n_gen_tokens), desc="    TAM", leave=False):
            out_path = os.path.join(tmpdir, f"{i}.jpg")
            try:
                img_map = TAM(
                    all_gen_ids,
                    vision_shape,
                    logits_per_step,
                    special_ids,
                    vis_inputs,
                    processor,
                    out_path,
                    i,
                    raw_map_records,
                    False,
                    skip_latex=True,
                )
            except Exception as e:
                print(f"      [TAM err tok {i}]: {e}")
                img_map = None
            tam_maps.append(img_map if isinstance(img_map, np.ndarray) else None)

    T = vision_shape[0]
    frame_mass = np.zeros((len(tam_maps), T), dtype=np.float32)
    for i, m in enumerate(tam_maps):
        if m is not None and m.ndim == 3 and m.shape[0] == T:
            total = m.sum()
            if total > 0:
                for t in range(T):
                    frame_mass[i, t] = m[t].sum() / total

    return {"tam_maps": tam_maps, "frame_mass": frame_mass}


# ── public entry-point ────────────────────────────────────────────────────────

def extract_layer_tam(
    inputs,
    outputs,
    sampled_frames,
    model,
    processor,
    layer_indices: List[int],
    cumavg_Ks: List[int],
    apply_norm: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Run TAM once per requested variant, sharing the underlying generation.

    Parameters
    ----------
    inputs / outputs / sampled_frames / model / processor
        Same as extract_tam_from_generation.
    layer_indices : list of negative ints
        e.g. [-1, -2, ..., -10] — each is run as an independent variant.
    cumavg_Ks : list of positive ints
        e.g. [1, 2, ..., 10]. K=k means average of layers [-1, -2, ..., -k].
    apply_norm : bool
        Apply final RMSNorm before the LM head when reading intermediate
        layers (the standard logit-lens recipe).
    verbose : bool
        Print per-variant timing.

    Returns
    -------
    dict with:
        gen_text        : str
        gen_tokens      : List[str]
        gen_ids         : List[int]
        vision_shape    : (T, H_tam, W_tam)
        frames_pil      : List[PIL.Image]   (the T sampled frames TAM saw)
        variants        : Dict[str, dict]   per-variant {tam_maps, frame_mass}
        norm_applied    : bool              whether the final norm was applied
    """
    generated_ids = outputs.sequences
    input_len = inputs.input_ids.shape[1]
    gen_ids = generated_ids[0][input_len:].cpu().tolist()
    gen_tokens = [
        processor.tokenizer.decode([t], skip_special_tokens=False)
        for t in gen_ids
    ]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    special_ids = _resolve_special_ids(inputs, processor)
    vision_shape = _resolve_vision_shape(inputs)
    T = vision_shape[0]
    vis_inputs = [list(sampled_frames[:T])]

    all_gen_ids = generated_ids[0].cpu().tolist()
    n_gen_tokens = len(outputs.hidden_states)

    norm_module = _find_final_norm(model) if apply_norm else None
    actually_normed = (norm_module is not None) and apply_norm
    if verbose:
        print(f"    [layer_tam] vision_shape={vision_shape}, "
              f"n_gen_tokens={n_gen_tokens}, apply_norm={actually_normed}")

    variants: Dict[str, Dict] = {}

    # Single-layer variants
    for L in layer_indices:
        key = f"layer_{L}"
        t0 = time.time()
        logits = _build_logits_for_layers(outputs, model, [L], apply_norm)
        if verbose:
            print(f"    [variant {key}] logits built in {time.time()-t0:.1f}s")
        t1 = time.time()
        variants[key] = _run_tam_for_variant(
            all_gen_ids, vision_shape, logits, special_ids,
            vis_inputs, processor, n_gen_tokens,
        )
        if verbose:
            print(f"    [variant {key}] TAM in {time.time()-t1:.1f}s")
        _free_logits(logits)

    # Cumulative-average variants
    for K in cumavg_Ks:
        key = f"cumavg_{K}"
        layer_set = list(range(-1, -K - 1, -1))
        t0 = time.time()
        logits = _build_logits_for_layers(outputs, model, layer_set, apply_norm)
        if verbose:
            print(f"    [variant {key}] logits built in {time.time()-t0:.1f}s")
        t1 = time.time()
        variants[key] = _run_tam_for_variant(
            all_gen_ids, vision_shape, logits, special_ids,
            vis_inputs, processor, n_gen_tokens,
        )
        if verbose:
            print(f"    [variant {key}] TAM in {time.time()-t1:.1f}s")
        _free_logits(logits)

    return {
        "gen_text": gen_text,
        "gen_tokens": gen_tokens,
        "gen_ids": gen_ids,
        "vision_shape": vision_shape,
        "frames_pil": list(sampled_frames[:T]),
        "variants": variants,
        "norm_applied": actually_normed,
    }
