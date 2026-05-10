"""
refcoco_runner.py
-----------------
Runs Qwen3-VL on a single RefCOCO image for referring expression grounding,
and optionally extracts TAM attention maps from the same forward pass.

The model receives one image and a referring expression, and is asked to
output a single bounding box in JSON format with normalized [0, 1000] coords.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_TAM     = _HERE.parent.parent / "submodules" / "TAM"
_DIAG    = _HERE.parent / "Ref-DAVIS" / "diagnostics"
sys.path.insert(0, str(_TAM))
sys.path.insert(0, str(_DIAG))

Box = Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None

GROUNDING_PROMPT = (
    'Referring expression: "{expression}". '
    'Localize the described object in the image. '
    'Output only JSON: {{"bbox_2d": [x_min, y_min, x_max, y_max], "label": ""}}'
)


# ── Output parsing ────────────────────────────────────────────────────────────

def _parse_grounding(text: str, W: int, H: int) -> Tuple[Box, str]:
    """
    Parse model output into a pixel-space bounding box and label string.

    Model outputs normalized [0, 1000] coordinates. Returns (box, label)
    where box is (x1, y1, x2, y2) in pixel coords, or (None, "") on failure.
    """
    text = text.strip()
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`")

    # Try to parse as JSON object or as a one-element list
    for attempt in (text, f"[{text}]"):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict) and "bbox_2d" in parsed:
                bbox = parsed["bbox_2d"]
                label = str(parsed.get("label", ""))
                if len(bbox) == 4:
                    x1 = int(bbox[0] * W / 1000)
                    y1 = int(bbox[1] * H / 1000)
                    x2 = int(bbox[2] * W / 1000)
                    y2 = int(bbox[3] * H / 1000)
                    return (x1, y1, x2, y2), label
        except Exception:
            pass

    # Fallback: regex extraction
    m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if m:
        try:
            vals = [float(v) for v in m.group(1).split(",")]
            if len(vals) == 4:
                x1 = int(vals[0] * W / 1000)
                y1 = int(vals[1] * H / 1000)
                x2 = int(vals[2] * W / 1000)
                y2 = int(vals[3] * H / 1000)
                lm = re.search(r'"label"\s*:\s*"([^"]*)"', text)
                label = lm.group(1) if lm else ""
                return (x1, y1, x2, y2), label
        except Exception:
            pass

    return None, ""


def _find_label_token_indices(gen_tokens: List[str]) -> List[int]:
    """
    Find generated token indices whose text falls between the open and close
    quote of the "label" value in the concatenated token stream.
    Returns empty list if the "label" key is not found.
    """
    raw = "".join(gen_tokens)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in gen_tokens:
        spans.append((pos, pos + len(tok)))
        pos += len(tok)

    key_pos = raw.find('"label"')
    if key_pos == -1:
        return []
    colon_pos = raw.find(":", key_pos + 7)
    if colon_pos == -1:
        return []
    open_q = raw.find('"', colon_pos + 1)
    if open_q == -1:
        return []
    close_q = raw.find('"', open_q + 1)
    if close_q == -1:
        return []

    val_start, val_end = open_q + 1, close_q
    return [i for i, (s, e) in enumerate(spans) if s < val_end and e > val_start]


# ── Runner ────────────────────────────────────────────────────────────────────

class RefCOCORunner:
    """
    Wraps Qwen3-VL for single-image referring expression grounding.

    Parameters
    ----------
    model, processor : loaded HuggingFace model and processor
    max_new_tokens   : token budget
    """

    def __init__(self, model, processor, max_new_tokens: int = 256):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def _build_messages(self, image: Image.Image, expression: str) -> list:
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": GROUNDING_PROMPT.format(expression=expression)},
        ]}]

    def _process(self, messages: list):
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True,
            image_patch_size=16,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs  = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def run(self, image: Image.Image, expression: str) -> Tuple[Box, str]:
        """Grounding inference only. Returns (pred_box, raw_text)."""
        W, H = image.size
        messages = self._build_messages(image, expression)
        inputs = self._process(messages)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        box, _ = _parse_grounding(raw, W, H)
        return box, raw

    def run_with_tam(self, image: Image.Image, expression: str) -> Tuple[Box, str, dict]:
        """
        Grounding inference + TAM extraction in one forward pass.

        Returns
        -------
        pred_box   : (x1, y1, x2, y2) or None
        raw_text   : str
        tam_result : dict  – same schema as extract_tam_from_generation()
            keys: gen_text, gen_tokens, gen_ids, tam_maps, frame_mass,
                  vision_shape, frames_pil
            tam_maps[i] has shape (1, H_tam, W_tam) for a single image;
            use tam_maps[i][0] for the 2-D spatial heatmap.
        """
        from tam_runner import extract_tam_from_generation

        W, H = image.size
        messages = self._build_messages(image, expression)
        inputs = self._process(messages)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        seq = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, seq)
        ]
        raw = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        box, label = _parse_grounding(raw, W, H)

        tam_result = extract_tam_from_generation(
            inputs, outputs, [image], self.model, self.processor
        )
        return box, raw, tam_result
