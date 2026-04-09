"""
qwen_vot_runner.py
------------------
Runs Qwen3-VL on a DAVIS sequence for Visual Object Tracking (VOT).

Two input modes (set video_mode in constructor):
  image_mode (default) — interleaved timestamp + image content list, 2D RoPE;
                         model outputs time-indexed detections
  video_mode           — single {"type":"video"} block, 3D RoPE; model outputs
                         0-indexed sampled frame numbers instead of timestamps

Evaluation is against GT bboxes from Annotations_bbox (bbox IoU instead of
mask J&F).
"""

import re
import json
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

JOINT_PROMPT_VIDEO = (
    'Given the query "{expression}", for each frame, detect and localize '
    'the visual content described in JSON format. If not present, skip. '
    'Each entry must have a unique frame index. '
    'Output Format: [{{"frame": 0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": "..."}}]'
)


# ─── Box Parsing ─────────────────────────────────────────────────────────────

def _parse_frame_detections(text: str) -> list:
    """
    Parse video-mode model output:
      [{"frame": i, "bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]
    where i is a 0-indexed sampled frame number and coordinates are
    normalized to [0, 1000]. Returns list of dicts (empty on failure).
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


def _map_frame_detections_to_frames(
    detections: list, N: int, W: int, H: int, sample_rate: int
) -> List[Box]:
    """
    Convert frame-index, 0-1000-normalised detections to a per-frame box list
    of length N. Only sampled frame positions (multiples of sample_rate) are
    filled from model predictions; all other positions are None.
    Skipped sampled frames (model said object absent) remain None — IoU=0.
    No nearest-neighbour interpolation is applied.
    """
    boxes: List[Box] = [None] * N
    for det in detections:
        sampled_idx = det.get("frame", -1)
        if sampled_idx < 0:
            continue
        original_idx = sampled_idx * sample_rate
        if original_idx >= N:
            continue
        bbox = det.get("bbox_2d", None)
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(bbox[0] * W / 1000)
        y1 = int(bbox[1] * H / 1000)
        x2 = int(bbox[2] * W / 1000)
        y2 = int(bbox[3] * H / 1000)
        boxes[original_idx] = (x1, y1, x2, y2)
    return boxes


def _map_detections_to_frames(
    detections: list, N: int, W: int, H: int, fps: float = DAVIS_FPS,
    sample_rate: int = 1,
) -> List[Box]:
    """
    Convert time-indexed, 0-1000-normalised detections to a per-frame box list
    of length N. Each detection's timestamp is converted to the nearest sampled
    frame index (multiple of sample_rate). Non-detected frames remain None.
    No nearest-neighbour interpolation is applied.
    """
    boxes: List[Box] = [None] * N
    for det in detections:
        t = det.get("time", 0.0)
        bbox = det.get("bbox_2d", None)
        if not bbox or len(bbox) != 4:
            continue
        raw_idx = int(round(t * fps))
        # Snap to nearest sampled frame index
        snapped_idx = round(raw_idx / sample_rate) * sample_rate
        original_idx = max(0, min(snapped_idx, N - 1))
        x1 = int(bbox[0] * W / 1000)
        y1 = int(bbox[1] * H / 1000)
        x2 = int(bbox[2] * W / 1000)
        y2 = int(bbox[3] * H / 1000)
        boxes[original_idx] = (x1, y1, x2, y2)
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
                 fps: float = DAVIS_FPS, sample_rate: int = 8,
                 video_mode: bool = True):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self.sample_rate = sample_rate
        self.video_mode = video_mode

    def _generate(self, messages: list, is_video: bool = False) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pvi_kwargs = dict(return_video_kwargs=True, return_video_metadata=True)
        if not is_video:
            pvi_kwargs["image_patch_size"] = 16
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, **pvi_kwargs
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        proc_kwargs = dict(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            return_tensors="pt",
        )
        if not is_video:
            proc_kwargs["do_resize"] = False

        inputs = self.processor(**proc_kwargs).to(self.model.device)

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

    def run(self, frames: List[Image.Image], expression: str) -> Tuple[List[Box], str]:
        """Run VOT inference on a full video sequence. Returns (boxes, raw_text)."""
        N = len(frames)
        W, H = frames[0].size
        sampled_frames = frames[::self.sample_rate]

        if self.video_mode:
            effective_fps = self.fps / self.sample_rate
            messages = [{"role": "user", "content": [
                {
                    "type": "video",
                    "video": sampled_frames,
                    "fps": effective_fps,
                },
                {"type": "text", "text": JOINT_PROMPT_VIDEO.format(expression=expression)},
            ]}]
            raw = self._generate(messages, is_video=True)
            detections = _parse_frame_detections(raw)
            return _map_frame_detections_to_frames(detections, N, W, H, self.sample_rate), raw
        else:
            content_list = []
            for i, frame in enumerate(sampled_frames):
                actual_idx = i * self.sample_rate
                timestamp = actual_idx / self.fps
                content_list.append({"type": "text", "text": f"<{timestamp:.2f} seconds>"})
                content_list.append({"type": "image", "image": frame})
            content_list.append({"sample_fps": DAVIS_FPS})
            content_list.append({"type": "text", "text": JOINT_PROMPT.format(expression=expression)})
            messages = [{"role": "user", "content": content_list}]
            raw = self._generate(messages, is_video=False)
            detections = _parse_time_detections(raw)
            return _map_detections_to_frames(detections, N, W, H, fps=self.fps, sample_rate=self.sample_rate), raw
