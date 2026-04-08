"""Model loading and memory-optimized inference for TAM on Qwen3-VL.

Designed for 16GB Apple M4 Mac Mini:
- 4-bit quantization when available, float16 fallback
- Standard generation with per-step logit extraction (primary)
- Optional memory-efficient custom loop (experimental)
"""

import sys
import os
import gc
import torch
import warnings
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .config import SPECIAL_IDS, SPATIAL_MERGE_SIZE

# Add qwen-vl-utils to path for vision processing
_QWEN_VL_UTILS = os.path.join(os.path.dirname(__file__), '..', 'qwen-vl-utils', 'src')
if _QWEN_VL_UTILS not in sys.path:
    sys.path.insert(0, _QWEN_VL_UTILS)
from qwen_vl_utils import process_vision_info


def load_model(model_path, device_map="auto", use_quantization=True):
    """Load Qwen3-VL model with optional 4-bit quantization.

    Args:
        model_path: HuggingFace model ID or local path.
        device_map: Device placement strategy.
        use_quantization: Try 4-bit quantization to save memory.

    Returns:
        (model, processor) tuple.
    """
    processor = AutoProcessor.from_pretrained(model_path)

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }

    # Try 4-bit quantization for memory savings
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization.")
        except (ImportError, Exception) as e:
            warnings.warn(f"4-bit quantization unavailable ({e}), using float16.")
            load_kwargs["dtype"] = torch.float16
    else:
        load_kwargs["dtype"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    model.eval()

    return model, processor


def prepare_inputs(processor, image_or_video, prompt, device=None, multi_image=False):
    """Prepare model inputs from image/video path and prompt text.

    Args:
        processor: The model processor.
        image_or_video: str path (image), list of str paths (video frames), or PIL.Image.
        prompt: Text prompt string.
        device: Target device (inferred from model if None).
        multi_image: If True and input is a list of frames, treat each frame as a
            separate image instead of a video. This preserves per-frame spatial
            resolution and gives separate activation maps per frame — needed for
            temporal coherence analysis since Qwen3-VL's video processor merges
            all temporal frames into very few grid steps.

    Returns:
        (inputs_dict, vis_inputs, is_video) tuple.
        When multi_image=True with frame list: is_video=False, vision_shape will be
        a list of (h, w) tuples (one per frame) via get_vision_shape().
    """
    # Build conversation message
    if isinstance(image_or_video, list) and not multi_image:
        content = [{"type": "video", "video": image_or_video}, {"type": "text", "text": prompt}]
        is_video = True
    elif isinstance(image_or_video, list) and multi_image:
        # Multi-image mode: each frame as a separate image element
        content = [{"type": "image", "image": f} for f in image_or_video]
        content.append({"type": "text", "text": prompt})
        is_video = False
    elif isinstance(image_or_video, str) and image_or_video.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        content = [{"type": "video", "video": image_or_video}, {"type": "text", "text": prompt}]
        is_video = True
    else:
        content = [{"type": "image", "image": image_or_video}, {"type": "text", "text": prompt}]
        is_video = False

    messages = [{"role": "user", "content": content}]

    # Process text and vision
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device is not None:
        inputs = inputs.to(device)

    # Prepare vision inputs for visualization
    if is_video:
        if video_inputs is not None:
            vis_inputs = [[video_inputs[0][i] for i in range(len(video_inputs[0]))]]
        else:
            vis_inputs = image_or_video
    else:
        vis_inputs = image_inputs

    return inputs, vis_inputs, is_video


def get_vision_shape(inputs, is_video=False):
    """Extract vision token spatial shape from model inputs.

    Args:
        inputs: Processor output dict with image_grid_thw or video_grid_thw.
        is_video: Whether input is video.

    Returns:
        - Single image: tuple (h, w)
        - Multiple images (multi_image mode): list of (h, w) tuples
        - Video: tuple (num_frames, h, w)
    """
    if is_video:
        grid = inputs['video_grid_thw']
        return (
            int(grid[0, 0]) // SPATIAL_MERGE_SIZE,
            int(grid[0, 1]) // SPATIAL_MERGE_SIZE,
            int(grid[0, 2]) // SPATIAL_MERGE_SIZE,
        )
    else:
        grid = inputs['image_grid_thw']
        if grid.shape[0] == 1:
            return (
                int(grid[0, 1]) // SPATIAL_MERGE_SIZE,
                int(grid[0, 2]) // SPATIAL_MERGE_SIZE,
            )
        else:
            # Multiple images — return list of (h, w) tuples
            return [
                (int(grid[i, 1]) // SPATIAL_MERGE_SIZE,
                 int(grid[i, 2]) // SPATIAL_MERGE_SIZE)
                for i in range(grid.shape[0])
            ]


def generate_with_logits(model, inputs, max_new_tokens=256, memory_efficient=False):
    """Generate tokens and collect per-step logits for TAM.

    Args:
        model: The Qwen3-VL model.
        inputs: Preprocessed inputs dict.
        max_new_tokens: Maximum generation length.
        memory_efficient: If True, use experimental custom loop (may have issues on MPS).
            Default False uses model.generate() which is more reliable.

    Returns:
        (generated_ids_tensor, logits_list) tuple.
        - generated_ids_tensor: shape [1, total_seq_len]
        - logits_list: list of tensors, one per generation step.
          Step 0 has shape [1, prompt_len, vocab_size].
          Steps 1+ have shape [1, 1, vocab_size].
    """
    if memory_efficient:
        return _generate_memory_efficient(model, inputs, max_new_tokens)
    else:
        return _generate_standard(model, inputs, max_new_tokens)


def _generate_standard(model, inputs, max_new_tokens):
    """Standard generation using model.generate() with output_hidden_states.

    Uses model.generate() which properly handles vision token embedding,
    KV caching, and position IDs through prepare_inputs_for_generation().
    Stores all hidden states — works for moderate generation lengths on 16GB.
    """
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences

    # Compute logits from last hidden states via lm_head
    logits = []
    for feats in outputs.hidden_states:
        last_layer_hs = feats[-1]
        step_logits = model.lm_head(last_layer_hs).cpu()
        logits.append(step_logits)

    # Free the large hidden_states
    del outputs.hidden_states
    del outputs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return generated_ids, logits


@torch.no_grad()
def _generate_memory_efficient(model, inputs, max_new_tokens):
    """Memory-efficient generation using model's own prepare_inputs_for_generation.

    Uses the model's prepare_inputs_for_generation() to handle vision embedding
    and KV cache management correctly, then manually drives the generation loop.
    """
    input_ids = inputs["input_ids"]
    device = input_ids.device

    logits_list = []
    past_key_values = None
    eos_token_id = None

    if hasattr(model, 'generation_config') and model.generation_config is not None:
        eos_token_id = model.generation_config.eos_token_id

    # Build kwargs that prepare_inputs_for_generation expects
    model_kwargs = {}
    for k, v in inputs.items():
        if k != "input_ids":
            model_kwargs[k] = v

    for step in range(max_new_tokens):
        # Use the model's own preparation to handle pixel_values, position_ids, etc.
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=model_kwargs.get("attention_mask"),
            use_cache=True,
            **{k: v for k, v in model_kwargs.items() if k != "attention_mask"},
        )
        prepared["output_hidden_states"] = True
        prepared["return_dict"] = True

        outputs = model(**prepared)

        # Extract last hidden state -> logits
        last_hidden = outputs.hidden_states[-1]
        step_logits = model.lm_head(last_hidden).cpu()
        logits_list.append(step_logits)

        # Update KV cache and state
        past_key_values = outputs.past_key_values

        # Greedy decode
        next_token_id = step_logits[0, -1, :].argmax(dim=-1)
        input_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(0).unsqueeze(0).to(device)], dim=1
        )

        # Update attention mask
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            model_kwargs["attention_mask"] = torch.cat([
                model_kwargs["attention_mask"],
                torch.ones((1, 1), dtype=model_kwargs["attention_mask"].dtype, device=device),
            ], dim=1)

        # After first step, pixel_values should not be passed again
        for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            model_kwargs.pop(key, None)

        # Check EOS
        if eos_token_id is not None:
            token_val = next_token_id.item()
            if isinstance(eos_token_id, list):
                if token_val in eos_token_id:
                    break
            elif token_val == eos_token_id:
                break

        # Free memory
        del outputs.hidden_states, last_hidden
        gc.collect()

    return input_ids, logits_list


@torch.no_grad()
def extract_multilayer_scores(model, inputs, generated_ids, cls_id, layer_indices):
    """Extract per-layer activation scores for a single token class.

    Memory-efficient: computes lm_head.weight[cls_id] @ hidden_states[layer]
    instead of full lm_head(hidden_states), avoiding the [seq_len, vocab_size]
    intermediate tensor (~1.1 GB). Result is [seq_len] floats per layer (~8 KB).

    Uses a single forward pass with the full prompt+generated sequence and
    output_hidden_states=True. The model re-embeds vision tokens from the
    pixel_values in the inputs dict.

    Args:
        model: The Qwen3-VL model.
        inputs: Original preprocessed inputs dict (must contain pixel_values etc.).
        generated_ids: Full token sequence [1, total_seq_len] from model.generate().
        cls_id: The vocabulary class ID to extract scores for.
        layer_indices: List of layer indices (0-based, 0=first decoder layer,
            N-1=last). Note: hidden_states[0] is the embedding output, so
            decoder layer i is at hidden_states[i+1].

    Returns:
        dict mapping layer_idx -> numpy array of shape [seq_len] with
        activation scores (logit for cls_id at every position).
    """
    # Build forward pass inputs: use generated_ids as input_ids,
    # keep vision inputs (pixel_values etc.) for re-embedding
    fwd_kwargs = {}
    for k, v in inputs.items():
        if k not in ('input_ids', 'attention_mask'):
            fwd_kwargs[k] = v

    attention_mask = torch.ones_like(generated_ids)

    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        **fwd_kwargs,
    )

    # lm_head weight vector for the target class
    lm_weight = model.lm_head.weight[cls_id].float()  # shape [hidden_dim]

    scores_per_layer = {}
    for layer_idx in layer_indices:
        # hidden_states[0] = embedding, hidden_states[i+1] = decoder layer i
        hs_idx = layer_idx + 1
        if hs_idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[hs_idx][0].float()  # shape [seq_len, hidden_dim]
        scores = (hs @ lm_weight).cpu().numpy()  # shape [seq_len]
        scores_per_layer[layer_idx] = scores

    del outputs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return scores_per_layer
