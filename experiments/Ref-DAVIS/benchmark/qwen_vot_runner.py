"""
qwen_vot_runner.py
------------------
Runs Qwen3-VL on a DAVIS sequence for Visual Object Tracking (VOT).

The model receives a text expression and all frames as a video sequence
(interleaved timestamps + images), then predicts bounding boxes for every
frame — no ground-truth initialisation is provided.

Processing mirrors qwen_vos_runner.py / qwen3vl_video_grounding.py exactly:
  - Interleaved image + timestamp content list
  - sample_rate subsampling
  - process_vision_info with return_video_kwargs + return_video_metadata
  - processor called with do_resize=False and unpacked video_kwargs
  - Time-indexed, 0-1000-normalised bbox output parsed and mapped back to
    all frames via nearest-neighbour interpolation

Evaluation is against GT bboxes from Annotations_bbox (bbox IoU instead of
mask J&F).
"""

import re
import json
import bisect
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from qwen_vl_utils import process_vision_info

Box = Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None

DAVIS_FPS = 24.0

# ─── Prompt Template ─────────────────────────────────────────────────────────

JOINT_PROMPT = (
    'Given the query "{expression}", for each frame, detect and localize '
    'the visual content described in JSON format. If not present, skip. '
    'Output Format: [{{"time": 0.0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": ""}}, ...]'
)


# ─── Box Parsing ─────────────────────────────────────────────────────────────

def _parse_time_detections(text: str) -> list:
    """
    Parse model output:
      [{"time": t, "bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]
    where coordinates are normalized to [0, 1000].
    Returns list of dicts (empty on failure).
    """
    text = text.strip()
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`")
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            val = json.loads(text[start:end])
            if isinstance(val, list):
                return val
    except Exception:
        pass
    return []


def _map_detections_to_frames(
    detections: list, N: int, W: int, H: int, fps: float = DAVIS_FPS
) -> List[Box]:
    """
    Convert time-indexed, 0-1000-normalised detections to per-frame pixel boxes.
    Undetected frames filled via nearest-neighbour from detected frames.
    """
    time_map: dict = {}
    for det in detections:
        t = det.get("time", 0.0)
        bbox = det.get("bbox_2d", None)
        if not bbox or len(bbox) != 4:
            continue
        frame_idx = max(0, min(int(round(t * fps)), N - 1))
        x1 = int(bbox[0] * W / 1000)
        y1 = int(bbox[1] * H / 1000)
        x2 = int(bbox[2] * W / 1000)
        y2 = int(bbox[3] * H / 1000)
        time_map[frame_idx] = (x1, y1, x2, y2)

    if not time_map:
        return [None] * N

    sorted_keys = sorted(time_map.keys())
    boxes: List[Box] = []
    for i in range(N):
        pos = bisect.bisect_left(sorted_keys, i)
        if pos == 0:
            nearest = sorted_keys[0]
        elif pos == len(sorted_keys):
            nearest = sorted_keys[-1]
        else:
            lo, hi = sorted_keys[pos - 1], sorted_keys[pos]
            nearest = lo if abs(i - lo) <= abs(i - hi) else hi
        boxes.append(time_map[nearest])
    return boxes


# ─── Main Runner ─────────────────────────────────────────────────────────────

class QwenVOTRunner:
    """
    Wraps Qwen model + processor for VOT inference.

    All frames are sent as a video sequence (interleaved timestamps + images)
    in a single forward pass. The model predicts bounding boxes for the
    described object in every frame from the expression alone.

    Parameters
    ----------
    model, processor : loaded HuggingFace model and processor
    max_new_tokens   : token budget for generation
    fps              : dataset frame rate (DAVIS = 24)
    sample_rate      : send every Nth frame to the model; results for all
                       frames filled via nearest-neighbour interpolation
    """

    def __init__(self, model, processor, max_new_tokens: int = 8192,
                 fps: float = DAVIS_FPS, sample_rate: int = 2):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self.sample_rate = sample_rate

    def _generate(self, messages: list) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
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

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def run(self, frames: List[Image.Image], expression: str) -> List[Box]:
        """
        Run VOT inference on a full video sequence.

        All sampled frames are sent in one pass with interleaved timestamp +
        image content, mirroring qwen3vl_video_grounding.py. Results for
        non-sampled frames are filled via nearest-neighbour interpolation.
        """
        N = len(frames)
        W, H = frames[0].size

        sampled_frames = frames[::self.sample_rate]

        content_list = []
        for i, frame in enumerate(sampled_frames):
            actual_idx = i * self.sample_rate
            timestamp = actual_idx / self.fps
            content_list.append({"type": "text", "text": f"<{timestamp:.2f} seconds>"})
            content_list.append({"type": "image", "image": frame})

        content_list.append({"sample_fps": DAVIS_FPS})
        content_list.append({"type": "text", "text": JOINT_PROMPT.format(expression=expression)})

        messages = [{"role": "user", "content": content_list}]

        raw = self._generate(messages)
        detections = _parse_time_detections(raw)
        return _map_detections_to_frames(detections, N, W, H, fps=self.fps)
