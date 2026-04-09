"""
qwen_vos_runner.py
------------------
Runs Qwen3-VL on a Ref-DAVIS sequence, asking it to produce bounding boxes
for the referred object in each frame.

Two input modes (set video_mode in constructor):
  image_mode (default) — interleaved timestamp + image content list, 2D RoPE
  video_mode           — single {"type":"video"} block, 3D RoPE; model outputs
                         0-indexed sampled frame numbers instead of timestamps

Two strategies (joint mode only; per_frame always uses image mode):
  "joint"     — all sampled frames in one pass (recommended)
  "per_frame" — each frame independently (slow, no subsampling)

The caller is responsible for loading the model/processor.
"""

import re
import json
import textwrap
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from qwen_vl_utils import process_vision_info

Box = Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None

DAVIS_FPS = 24.0

# ─── Prompt Templates ────────────────────────────────────────────────────────

JOINT_PROMPT = (
    'Given the query "{expression}", for each frame, detect and localize '
    'the visual content described in JSON format. If not present, skip. '
    'Output Format: [{{"time": 0.0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": ""}}, ...]'
)

JOINT_PROMPT_VIDEO = (
    'Given the query "{expression}", for each frame where the target is visible, '
    'output one JSON entry with the 0-indexed frame number, '
    'a bounding box [x_min, y_min, x_max, y_max] (normalized 0-1000), '
    'and a short label. Omit frames where the target is absent. '
    'Each entry must have a unique frame index. '
    'Output only the JSON array. '
    'Format: [{{"frame": 0, "bbox_2d": [x_min, y_min, x_max, y_max], "label": "..."}}]'
)

PER_FRAME_PROMPT = textwrap.dedent("""
The object of interest is: "{expression}"

Output the bounding box of this object in the image as [x1, y1, x2, y2]
in pixel coordinates (integers).
If the object is not visible, output [0, 0, 0, 0].

Respond ONLY with a JSON array of 4 integers: [x1, y1, x2, y2]
No explanation. No markdown.
""").strip()


# ─── Box Parsing ─────────────────────────────────────────────────────────────

def _parse_box(text: str) -> Box:
    """Extract first [x1,y1,x2,y2] from text. Returns None on failure."""
    text = text.strip()
    try:
        val = json.loads(text)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return tuple(int(v) for v in val)
    except Exception:
        pass
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if m:
        return tuple(int(m.group(i)) for i in range(1, 5))
    return None


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


def _map_frame_detections_to_frames(
    detections: list, N: int, W: int, H: int, sample_rate: int
) -> List[Box]:
    """
    Convert frame-index, 0-1000-normalised detections to a per-frame box list
    of length N. Only sampled frame positions (multiples of sample_rate) are
    filled from model predictions; all other positions are None.
    Skipped sampled frames (model said object absent) remain None.
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


# ─── Box → Mask ──────────────────────────────────────────────────────────────

def box_to_mask(box: Box, H: int, W: int) -> np.ndarray:
    """Convert a bounding box to a filled binary mask (H,W) uint8."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


# ─── Main Runner ─────────────────────────────────────────────────────────────

class QwenVOSRunner:
    """
    Wraps Qwen model + processor for VOS inference.

    Parameters
    ----------
    model, processor : loaded HuggingFace model and processor
    strategy         : "joint" or "per_frame"
    max_new_tokens   : token budget for generation
    fps              : dataset frame rate (DAVIS = 24)
    sample_rate      : send every Nth frame to the model in joint mode.
                       sample_rate=1 sends all frames and tends to cause the
                       model to collapse to a single repeated box.
    """

    def __init__(self, model, processor, strategy: str = "joint",
                 max_new_tokens: int = 8192, fps: float = DAVIS_FPS,
                 sample_rate: int = 2, video_mode: bool = False):
        self.model = model
        self.processor = processor
        self.strategy = strategy
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self.sample_rate = sample_rate
        self.video_mode = video_mode

    def _generate(self, messages: list, is_video: bool = False) -> str:
        """
        One forward pass.
        image mode: process_vision_info with image_patch_size=16, do_resize=False
        video mode: process_vision_info without image_patch_size, no do_resize
        """
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

    def run_joint(self, frames: List[Image.Image], expression: str) -> Tuple[List[Box], str]:
        """
        Image mode: all sampled frames in one call with interleaved timestamp + image content.
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

        raw = self._generate(messages, is_video=False)
        detections = _parse_time_detections(raw)
        return _map_detections_to_frames(detections, N, W, H, fps=self.fps, sample_rate=self.sample_rate), raw

    def run_joint_video(self, frames: List[Image.Image], expression: str) -> Tuple[List[Box], str]:
        """
        Video mode: all sampled frames as a single video block with 3D RoPE.
        Model outputs 0-indexed sampled frame numbers instead of timestamps.
        """
        N = len(frames)
        W, H = frames[0].size

        sampled_frames = frames[::self.sample_rate]
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

    def run_per_frame(self, frames: List[Image.Image], expression: str) -> Tuple[List[Box], str]:
        """One call per frame. Slow but maximally fair."""
        boxes = []
        raws = []
        prompt = PER_FRAME_PROMPT.format(expression=expression)
        for frame in frames:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": frame},
                {"type": "text",  "text": prompt},
            ]}]
            raw = self._generate(messages)
            boxes.append(_parse_box(raw))
            raws.append(raw)
        return boxes, "\n---\n".join(raws)

    def run(self, frames: List[Image.Image], expression: str) -> Tuple[List[Box], str]:
        """Run inference using configured strategy and mode. Returns (boxes, raw_text)."""
        if self.strategy == "per_frame":
            return self.run_per_frame(frames, expression)
        if self.video_mode:
            return self.run_joint_video(frames, expression)
        return self.run_joint(frames, expression)

    def run_and_get_masks(
        self, frames: List[Image.Image], expression: str, H: int, W: int
    ) -> Tuple[List[Box], List[np.ndarray]]:
        """
        Run inference and convert predicted boxes to binary masks.

        Returns
        -------
        boxes : list of Box (may contain None)
        masks : list of (H,W) uint8 binary masks
        """
        boxes = self.run(frames, expression)
        masks = [box_to_mask(b, H, W) for b in boxes]
        return boxes, masks
