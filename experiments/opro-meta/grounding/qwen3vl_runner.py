"""
grounding/qwen3vl_runner.py
---------------------------
Thin wrapper around the existing QwenVOTRunner.run_with_tam.

Loads Qwen3-VL once, then exposes a single ground(clip, expression) call that
returns the box sequence, generated text, and TAM result needed for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import _paths  # noqa: F401

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from benchmark.qwen_vot_runner import QwenVOTRunner


Box = Optional[Tuple[int, int, int, int]]


@dataclass
class GroundingOutput:
    boxes: List[Box]
    gen_text: str
    tam_result: Optional[dict]   # None when skip_tam=True
    expression: str


class Qwen3VLGrounder:
    """Holds the model + processor + runner. ground() runs one expression.

    skip_tam=True takes the fast path (boxes + text only, no attention map
    extraction). This roughly halves per-iteration time on long clips, at
    the cost of losing MassGT / MassPred / token-category breakdown.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        sample_rate: int = 8,
        max_new_tokens: int = 4096,
        video_mode: bool = True,
        seed: int = 0,
        skip_tam: bool = False,
    ):
        print(f"[grounder] loading {model_id} on {device}…  skip_tam={skip_tam}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.runner = QwenVOTRunner(
            self.model, self.processor,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
            video_mode=video_mode,
            seed=seed,
        )
        self.sample_rate = sample_rate
        self.fps = self.runner.fps
        self.skip_tam = skip_tam

    def ground(self, frames_pil, expression: str,
               skip_tam: Optional[bool] = None) -> GroundingOutput:
        """If skip_tam (or self.skip_tam) is True, run the fast no-TAM path."""
        use_skip = self.skip_tam if skip_tam is None else skip_tam
        if use_skip:
            boxes, raw_text = self.runner.run(frames_pil, expression)
            return GroundingOutput(
                boxes=boxes,
                gen_text=raw_text,
                tam_result=None,
                expression=expression,
            )
        boxes, _, tam_result = self.runner.run_with_tam(frames_pil, expression)
        return GroundingOutput(
            boxes=boxes,
            gen_text=tam_result.get("gen_text", ""),
            tam_result=tam_result,
            expression=expression,
        )
