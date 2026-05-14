"""
attribution/tam.py
------------------
TAM-based attention heat map extraction. Wraps the existing token_parser +
label-token-index machinery from grounding-stability-max.

build_frame_heatmaps(tam_result, fps, sample_rate)
    -> dict[sampled_frame_index -> 2D heatmap (H_tam, W_tam)]

extract_token_category_indices(tam_result, fps, sample_rate, pos_map)
    -> dict[category -> list[(sampled_t, token_indices)]]
       categories: 'noun' | 'adj' | 'verb' | 'other'
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import _paths  # noqa: F401

import numpy as np

from token_parser import parse_frame_labels, find_label_token_indices


def build_frame_heatmaps(
    tam_result: dict,
    fps: float,
    sample_rate: int,
) -> Dict[int, np.ndarray]:
    """Average all label-noun tokens per frame into a single heat map."""
    gen_tokens = tam_result["gen_tokens"]
    tam_maps   = tam_result["tam_maps"]
    vision_T   = tam_result["vision_shape"][0]
    gen_text   = tam_result.get("gen_text", "")

    parsed_entries  = parse_frame_labels(gen_text, fps=fps, sample_rate=sample_rate)
    label_token_map = find_label_token_indices(gen_tokens, parsed_entries)

    frame_heatmaps: Dict[int, np.ndarray] = {}
    for sampled_t, tok_idxs in label_token_map:
        if sampled_t >= vision_T:
            continue
        valid = [
            i for i in tok_idxs
            if i < len(tam_maps)
            and tam_maps[i] is not None
            and tam_maps[i].ndim == 3
            and sampled_t < tam_maps[i].shape[0]
        ]
        if not valid:
            continue
        slices = [tam_maps[i][sampled_t].astype(np.float32) for i in valid]
        avg = np.mean(slices, axis=0)
        mx = avg.max()
        if mx > 0:
            avg /= mx
        frame_heatmaps[sampled_t] = avg
    return frame_heatmaps


def extract_label_token_map(
    tam_result: dict,
    fps: float,
    sample_rate: int,
) -> List[Tuple[int, List[int]]]:
    """Raw (sampled_t, token_indices) pairs without averaging."""
    gen_tokens = tam_result["gen_tokens"]
    gen_text   = tam_result.get("gen_text", "")
    parsed = parse_frame_labels(gen_text, fps=fps, sample_rate=sample_rate)
    return find_label_token_indices(gen_tokens, parsed)
