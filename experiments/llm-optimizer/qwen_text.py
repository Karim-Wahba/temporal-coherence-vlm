"""
qwen_text.py
------------
Thin wrapper around Qwen3-VL-8B-Instruct for text-only generation.
Reuses the same model/processor loaded for the grounding experiments
so we don't pay the GPU memory cost twice.
"""

import json
import re
from typing import Any

import torch


class QwenTextLLM:
    """Text-only generation using the Qwen3-VL model (no vision inputs)."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 512):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def generate_json(self, prompt: str, system: str | None = None) -> dict | list | None:
        """Generate and parse JSON from the model output. Returns None on parse failure."""
        raw = self.generate(prompt, system=system)
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract first JSON object/array from response
            match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        return None
