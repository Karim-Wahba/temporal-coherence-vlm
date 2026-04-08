"""Core TAM algorithm: Rank Gaussian Filter, Estimated Causal Inference, and TAM orchestrator.

Adapted from the reference implementation at github.com/xmed-lab/TAM.
"""

import os
import torch
import numpy as np
from scipy.optimize import minimize_scalar

from .config import ECI_SCALE_CAP, USE_ECI


def rank_gaussian_filter(img, kernel_size=3):
    """Apply rank-based Gaussian-weighted filter for activation map denoising (Eq. 6-7).

    Within each sliding window, values are sorted by rank, then weighted by a
    custom 1D Gaussian kernel using the coefficient of variation (sigma/mu)
    for robust noise reduction.

    Args:
        img: 2D numpy array (activation map).
        kernel_size: Size of the square kernel (must be odd).

    Returns:
        Denoised 2D numpy array.
    """
    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width:i + pad_width + 1,
                                j - pad_width:j + pad_width + 1]
            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean  # coefficient of variation
                kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value

    return filtered_img


def least_squares(map1, map2):
    """Find optimal scale factor s minimizing ||map1 - s*map2||^2 (Eq. 5).

    Args:
        map1: Target activation map (numpy array).
        map2: Interference map (numpy array).

    Returns:
        Optimal scalar multiplier.
    """
    def diff(x, m1, m2):
        return np.sum((m1 - m2 * x) ** 2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x


def id2idx(inp_id, target_id, return_last=False):
    """Find the index of target_id (int or sequence) in inp_id list.

    Args:
        inp_id: List of token IDs to search.
        target_id: Single int or list of ints to find.
        return_last: If True and target_id is a list, return index of the last
            token in the matched sequence.

    Returns:
        Index position, or -1 if not found.
    """
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1) if inp_id[i:i + n] == target_id]
        if len(indexes) > 0:
            idx = indexes[-1]
            if return_last:
                idx += len(target_id) - 1
        else:
            idx = -1
    else:
        try:
            idx = inp_id.index(target_id)
        except ValueError:
            idx = -1
    return idx


def TAM_multilayer(tokens, vision_shape, scores_per_layer, special_ids, layer_indices):
    """Compute per-frame activation maps at multiple layers for a target token.

    Uses pre-computed per-layer scores from extract_multilayer_scores() to produce
    RGF-denoised per-frame activation maps at each layer. Skips ECI (RGF-only).

    Args:
        tokens: Full token ID sequence (input + generated).
        vision_shape: List of (h, w) tuples (multi-image mode, one per frame).
        scores_per_layer: dict mapping layer_idx -> numpy array of shape [seq_len]
            (raw activation scores for target class at every position).
        special_ids: Token boundary IDs dict.
        layer_indices: List of layer indices to process.

    Returns:
        dict mapping layer_idx -> list of per-frame float maps, each shape (h, w).
    """
    # Find all image token positions
    img_id = special_ids['img_id']
    start_id, end_id = img_id[0], img_id[1]
    tok_arr = np.array(tokens)
    start_positions = np.where(tok_arr == start_id)[0]
    end_positions = np.where(tok_arr == end_id)[0]

    if len(start_positions) > 1 and len(end_positions) > 1:
        # Multi-image: collect all image token indices from all pairs
        all_img_positions = []
        for s, e in zip(start_positions, end_positions):
            all_img_positions.extend(range(s + 1, e))
        img_positions = np.array(all_img_positions)
    else:
        s = id2idx(tokens, start_id, True)
        e = id2idx(tokens, end_id)
        img_positions = np.arange(s + 1, e)

    n_images = len(vision_shape)
    tokens_per_image = vision_shape[0][0] * vision_shape[0][1]

    result = {}
    for layer_idx in layer_indices:
        if layer_idx not in scores_per_layer:
            continue

        raw_scores = scores_per_layer[layer_idx]
        img_scores = np.clip(raw_scores[img_positions], 0, None)

        # Normalize
        score_range = img_scores.max() - img_scores.min()
        if score_range > 0:
            img_scores = (img_scores - img_scores.min()) / score_range

        # Split by frame and apply RGF
        per_frame = []
        for i in range(n_images):
            start = i * tokens_per_image
            end = start + tokens_per_image
            frame_scores = img_scores[start:end]
            h, w = vision_shape[i]
            if len(frame_scores) == h * w:
                filtered = rank_gaussian_filter(frame_scores.reshape(h, w), 3)
            else:
                filtered = np.zeros((h, w))
            per_frame.append(filtered)

        result[layer_idx] = per_frame

    return result


def TAM(tokens, vision_shape, logit_list, special_ids, vision_input,
        processor, save_fn, target_token, img_scores_list, eval_only=False):
    """Generate a Token Activation Map with ECI and Rank Gaussian Filter.

    Args:
        tokens: Token ID sequence (input + generated).
        vision_shape: Shape of vision tokens — (h, w) for image, (b, h, w) for video,
            or list of (h, w) tuples for multiple images.
        logit_list: List of logits tensors, one per generation step.
        special_ids: Dict with 'img_id', 'prompt_id', 'answer_id' boundaries.
        vision_input: Raw image(s) or video frames for visualization.
        processor: Model processor for token decoding.
        save_fn: Path to save visualization (empty string to skip).
        target_token: Generation round index (int), or (round, prompt_token_idx) tuple.
        img_scores_list: Accumulator list for ECI across rounds (pass empty list initially).
        eval_only: If True, skip visualization and return only the map.

    Returns:
        img_map: The activation map for evaluation.
    """
    import cv2
    from .visualization import multimodal_process

    # Parse segment boundaries
    img_id = special_ids['img_id']
    prompt_id = special_ids['prompt_id']
    answer_id = special_ids['answer_id']

    # Locate image tokens
    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        # Find all VISION_START/VISION_END pairs for multi-image support
        start_id, end_id = img_id[0], img_id[1]
        tok_arr = np.array(tokens)
        start_positions = np.where(tok_arr == start_id)[0]
        end_positions = np.where(tok_arr == end_id)[0]

        if len(start_positions) > 1 and len(end_positions) > 1:
            # Multi-image: collect all image token indices from all pairs
            all_img_positions = []
            for s, e in zip(start_positions, end_positions):
                all_img_positions.extend(range(s + 1, e))
            img_idx = np.array(all_img_positions)
        else:
            # Single image: use boundary pair
            img_idx = [id2idx(tokens, start_id, True), id2idx(tokens, end_id)]

    # Locate prompt and answer tokens
    prompt_idx = [id2idx(tokens, prompt_id[0], True), id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]

    # Decode prompt and answer text tokens
    prompt = processor.tokenizer.tokenize(
        processor.batch_decode([tokens[prompt_idx[0] + 1: prompt_idx[1]]],
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    )
    answer = processor.tokenizer.tokenize(
        processor.batch_decode([tokens[answer_idx[0] + 1:]],
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    )
    txt_all = prompt + answer

    # Determine round and target token index
    round_idx = -1
    this_token_idx = 0

    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1
        vis_token_idx = len(prompt) + target_token
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    # Round 0: recursively process prompt tokens first
    if round_idx == 0 and isinstance(target_token, int):
        for t in range(len(prompt) + 1):
            img_map = TAM(tokens, vision_shape, logit_list, special_ids, vision_input,
                          processor, save_fn if t == len(prompt) else '',
                          [0, t], img_scores_list, eval_only)
            if t == 0:
                first_ori = img_map
        return first_ori

    # Determine the class ID for activation extraction
    if round_idx == 0:
        if prompt_token_idx == len(prompt):
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]
        elif prompt_token_idx == 0:
            cls_id = logit_list[0][0, prompt_idx[0] + 1].argmax(0)
        else:
            cls_id = tokens[this_token_idx]
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    # Compute class activation scores across all rounds (Eq. 1)
    scores = torch.cat(
        [logit_list[_][0, :, cls_id] for _ in range(round_idx + 1)], -1
    ).clip(min=0)
    scores = scores.detach().cpu().float().numpy()

    # Extract segment scores
    prompt_scores = scores[prompt_idx[0] + 1: prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1: logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)

    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1: img_idx[1]]
    else:
        img_scores = scores[img_idx]

    # Save for ECI accumulation
    img_scores_list.append(img_scores)

    # Estimated Causal Inference (Eq. 2, 4, 5)
    if USE_ECI and len(img_scores_list) > 1 and vis_token_idx < len(txt_all):
        non_repeat_idx = []
        for i in range(vis_token_idx):
            if i < len(txt_all) and txt_all[i] != txt_all[vis_token_idx]:
                non_repeat_idx.append(i)
        txt_scores_ = txt_scores[non_repeat_idx]
        img_scores_list_ = [img_scores_list[_] for _ in non_repeat_idx]

        # Weighted interference map
        w = txt_scores_
        w = w / (w.sum() + 1e-8)
        interf_img_scores = (np.stack(img_scores_list_, 0) * w.reshape(-1, 1)).sum(0)

        # Subtract scaled interference with least-squares scale factor
        scaled_map = least_squares(img_scores, interf_img_scores)
        if ECI_SCALE_CAP is not None:
            scaled_map = min(scaled_map, ECI_SCALE_CAP)
        img_scores = (img_scores - interf_img_scores * scaled_map).clip(min=0)

    # Prepare vision input for visualization
    def _to_bgr(img):
        """Convert PIL image or tensor (C,H,W) to BGR numpy array (H,W,3)."""
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = arr.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

    if isinstance(vision_shape[0], tuple):
        cv_img = [_to_bgr(_) for _ in vision_input]
    elif len(vision_shape) == 2:
        cv_img = np.array(vision_input)
        if len(cv_img.shape) == 4 and cv_img.shape[0] == 1:
            cv_img = cv_img[0]
        if cv_img.ndim == 3 and cv_img.shape[0] in (3, 4):
            cv_img = cv_img.transpose(1, 2, 0)
        if cv_img.dtype != np.uint8:
            cv_img = (cv_img * 255).clip(0, 255).astype(np.uint8) if cv_img.max() <= 1.0 else cv_img.clip(0, 255).astype(np.uint8)
        cv_img = cv2.cvtColor(cv_img[:, :, :3], cv2.COLOR_RGB2BGR)
    else:  # video
        cv_img = [_to_bgr(_) for _ in vision_input[0]]

    # Top candidates for text visualization
    candi_scores, candi_ids = logit_list[round_idx][0, this_token_idx].topk(3)
    candi_scores = candi_scores.softmax(0)
    candidates = processor.batch_decode([[_] for _ in candi_ids])

    # Generate multimodal activation map (includes Rank Gaussian Filter)
    vis_img, img_map = multimodal_process(
        cv_img, vision_shape, img_scores, txt_scores, txt_all,
        candidates, candi_scores, vis_token_idx, save_fn,
        eval_only=eval_only, vis_width=-1 if eval_only else 500,
    )

    # Save visualization
    if save_fn != '' and vis_token_idx < (len(txt_all) - 1) and isinstance(vis_img, np.ndarray):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        cv2.imwrite(save_fn, vis_img)

    return img_map
