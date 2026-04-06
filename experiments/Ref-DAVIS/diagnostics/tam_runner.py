"""
tam_runner.py
-------------
Wraps the TAM pipeline for temporal coherence diagnostics.

Given a sequence + expression, runs Qwen with TAM hooks and returns:
  - per-token TAM maps: shape (num_gen_tokens, T, H_tam, W_tam)
  - generated token strings
  - frame activation mass: for each token, how much attention mass
    is on each temporal frame → shape (num_gen_tokens, T)

This is the diagnostic engine feeding into tam_analyzer.py.

Requires:
    sys.path includes ../../submodules/TAM  (same as original script)
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / 'submodules' / 'TAM'))


from tam import TAM
from qwen_utils import process_vision_info

def _add_tam_to_path(tam_submodule_path: Optional[str] = None):
    """Add TAM submodule to sys.path."""
    if tam_submodule_path:
        sys.path.insert(0, tam_submodule_path)
        return
    # Try to find it relative to this file
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../submodules/TAM"),
        os.path.join(os.path.dirname(__file__), "../submodules/TAM"),
        os.path.expanduser("~/submodules/TAM"),
    ]
    for c in candidates:
        if os.path.exists(c):
            sys.path.insert(0, os.path.abspath(c))
            return
    raise ImportError(
        "TAM submodule not found. Pass tam_submodule_path= explicitly."
    )


class TAMRunner:
    """
    Runs TAM on a video sequence and returns structured attention maps.

    Parameters
    ----------
    model, processor : loaded Qwen model and processor
    tam_submodule_path : path to the TAM submodule directory
    repeat_frames : int, Qwen FRAME_FACTOR (default 2)
    """

    def __init__(self, model, processor,
                 tam_submodule_path: Optional[str] = None,
                 repeat_frames: int = 2):
        self.model = model
        self.processor = processor
        self.repeat_frames = repeat_frames
        # _add_tam_to_path(tam_submodule_path)

    def run(
        self,
        frames_pil: List[Image.Image],
        expression: str,
        max_new_tokens: int = 128,
    ) -> dict:
        """
        Run TAM on frames with the given expression as prompt.

        Returns
        -------
        dict with:
            "gen_text"         : str
            "gen_tokens"       : List[str]   (one per generated token)
            "gen_ids"          : List[int]
            "tam_maps"         : List[np.ndarray | None]
                                 each entry is (T, H_tam, W_tam) float32 or None
            "frame_mass"       : np.ndarray shape (num_gen_tokens, T)
                                 fractional attention mass per frame per token
            "vision_shape"     : (T, H_tam, W_tam)
        """
        from transformers.video_utils import VideoMetadata

        frames_for_model = [f for f in frames_pil for _ in range(self.repeat_frames)]
        N_unique = len(frames_pil)

        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames_for_model},
            {"type": "text",  "text": expression},
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        n_model_frames = len(frames_for_model)
        video_meta = VideoMetadata(
            total_num_frames=n_model_frames,
            fps=2.0 * self.repeat_frames,
            frames_indices=list(range(n_model_frames)),
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            do_sample_frames=False,
            video_metadata=[video_meta],
        ).to(self.model.device)

        # Generate with hidden states
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        import torch
        logits = [self.model.lm_head(feats[-1]) for feats in outputs.hidden_states]
        generated_ids = outputs.sequences
        input_len = inputs.input_ids.shape[1]
        gen_ids = generated_ids[0][input_len:].cpu().tolist()
        gen_tokens = [
            self.processor.tokenizer.decode([t], skip_special_tokens=False)
            for t in gen_ids
        ]
        gen_text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Build special_ids (same logic as original script)
        token_ids = inputs.input_ids[0].cpu().tolist()
        token_strs = [
            self.processor.tokenizer.decode([t], skip_special_tokens=False)
            for t in token_ids
        ]

        vision_start_id = vision_end_id = video_pad_id = None
        for tid, ts in zip(token_ids, token_strs):
            if "<|vision_start|>" in ts:
                vision_start_id = tid
            elif "<|vision_end|>" in ts:
                vision_end_id = tid
            elif "<|video_pad|>" in ts or "<|image_pad|>" in ts:
                if video_pad_id is None:
                    video_pad_id = tid

        im_end_id = im_start_id = None
        for tid, ts in zip(token_ids, token_strs):
            if "<|im_end|>" in ts:
                im_end_id = tid
            if "<|im_start|>" in ts:
                im_start_id = tid

        answer_start_positions = [
            i for i in range(len(token_ids) - 1) if token_ids[i] == im_start_id
        ]
        assistant_header_pos = (
            answer_start_positions[-1] if answer_start_positions
            else len(token_ids) - 1
        )
        answer_boundary = token_ids[assistant_header_pos:input_len]

        special_ids = {
            "img_id": [video_pad_id],
            "prompt_id": [[vision_end_id], answer_boundary],
            "answer_id": [answer_boundary, -1],
        }

        vision_shape = (
            inputs["video_grid_thw"][0, 0].item(),
            inputs["video_grid_thw"][0, 1].item() // 2,
            inputs["video_grid_thw"][0, 2].item() // 2,
        )
        T = vision_shape[0]

        # vis_inputs for TAM: list of one list of T PIL frames
        vis_inputs = [frames_pil[:T]]

        # Run TAM for each token, collecting 3D maps
        raw_map_records = []
        tam_maps = []

        with tempfile.TemporaryDirectory() as tmpdir:
            all_gen_ids = generated_ids[0].cpu().tolist()
            for i in range(len(logits)):
                out_path = os.path.join(tmpdir, f"{i}.jpg")
                img_map = TAM(
                    all_gen_ids,
                    vision_shape,
                    logits,
                    special_ids,
                    vis_inputs,
                    self.processor,
                    out_path,
                    i,
                    raw_map_records,
                    False,
                )
                tam_maps.append(img_map if isinstance(img_map, np.ndarray) else None)

        # Compute frame activation mass: for each token, normalised attention per frame
        frame_mass = np.zeros((len(tam_maps), T), dtype=np.float32)
        for i, m in enumerate(tam_maps):
            if m is not None and m.ndim == 3 and m.shape[0] == T:
                total = m.sum()
                if total > 0:
                    for t in range(T):
                        frame_mass[i, t] = m[t].sum() / total

        return {
            "gen_text": gen_text,
            "gen_tokens": gen_tokens,
            "gen_ids": gen_ids,
            "tam_maps": tam_maps,         # List[(T,H,W) float32 or None]
            "frame_mass": frame_mass,     # (num_tokens, T)
            "vision_shape": vision_shape, # (T, H_tam, W_tam)
            "frames_pil": frames_pil[:T],
        }
