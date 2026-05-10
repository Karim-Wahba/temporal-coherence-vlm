"""
loader.py
---------
Loads Ref-YouTube-VOS (valid split) for the grounding-stability experiment.

Actual directory layout after unzipping valid.zip + meta_expressions.zip:

    data_root/
    └── valid/
        ├── JPEGImages/
        │   └── {video_id}/
        │       └── {frame_id}.jpg          # annotated frames only (~6 fps)
        ├── Annotations/
        │   └── {video_id}/
        │       └── {exp_id}/               # one dir per expression (0-indexed)
        │           └── {frame_id}.png      # binary mask (0 / 255)
        ├── meta_expressions_challenge.json # use this one (has obj_id)
        └── meta_expressions.json           # plain version (no obj_id, not used)

Key facts discovered from the data:
  • Each expression gets its own annotation directory (index = expression key).
    Expressions for the same object share identical masks.
  • JPEGImages contains ONLY the annotated frames; there are no intermediate frames.
  • Masks are binary PNGs (pixel 255 = foreground, 0 = background).
  • Annotated frame rate is ~6 fps (every 5th frame of original 30 fps video).

Interface is duck-type compatible with DAVISVOTItem so
RefYouTubeVOSGroundingExperiment can drive the same run_sequence() logic.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Box = Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None

ANNOTATED_FPS = 6.0   # every 5th frame of 30 fps → ~6 fps effective

META_JSON = "meta_expressions_challenge.json"


def _mask_to_box(png_path: str) -> Box:
    """
    Load a binary mask PNG (0/255) and return the tight bounding box
    of the foreground region as (x1, y1, x2, y2). Returns None if empty.
    """
    arr = np.array(Image.open(png_path))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


@dataclass
class RefYouTubeVOSItem:
    """
    Duck-type compatible with DAVISVOTItem.
    frame_paths and gt_boxes are parallel lists over annotated frames only.
    """
    seq_name:    str         # video_id (hex string, e.g. "0062f687f1")
    exp_id:      str         # expression index as string ("0", "1", ...)
    expression:  str
    obj_id:      int
    frame_paths: List[str]   # JPEGImages paths, annotated frames only
    gt_boxes:    List[Box]   # (x1,y1,x2,y2) from binary mask, or None
    _frames_pil: Optional[List[Image.Image]] = field(default=None, repr=False)

    @property
    def frames_pil(self) -> List[Image.Image]:
        if self._frames_pil is None:
            self._frames_pil = [Image.open(p).convert("RGB") for p in self.frame_paths]
        return self._frames_pil

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    def frame_size(self) -> Tuple[int, int]:
        """Returns (H, W)."""
        img = Image.open(self.frame_paths[0])
        return img.height, img.width

    @property
    def masks(self) -> List[np.ndarray]:
        """Binary masks (H,W) uint8 derived from GT boxes (for compatibility)."""
        H, W = self.frame_size()
        result = []
        for box in self.gt_boxes:
            m = np.zeros((H, W), dtype=np.uint8)
            if box is not None:
                x1, y1, x2, y2 = box
                m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
            result.append(m)
        return result


class RefYouTubeVOSLoader:
    """
    Iterates over (video, expression) pairs with per-frame GT bounding boxes
    derived from the binary segmentation masks.

    Parameters
    ----------
    data_root : str
        Root containing the valid/ directory (output of unzipping valid.zip).
    split : str
        Only "valid" is supported (train annotations are not downloaded yet).
    expressions_per_seq : int or None
        Cap on expressions per video.
    sequences : list[str] or None
        Whitelist of video IDs; None = all.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "valid",
        expressions_per_seq: Optional[int] = None,
        sequences: Optional[List[str]] = None,
    ):
        if split != "valid":
            raise ValueError(
                "Only split='valid' is supported — train annotations not downloaded."
            )

        self.data_root = Path(data_root)
        self.split = split
        self.expressions_per_seq = expressions_per_seq
        self.sequences = sequences

        split_dir       = self.data_root / split
        self.jpeg_root  = split_dir / "JPEGImages"
        self.ann_root   = split_dir / "Annotations"
        self.expr_path  = split_dir / META_JSON

        if not self.expr_path.exists():
            raise FileNotFoundError(
                f"{META_JSON} not found at {self.expr_path}.\n"
                f"Expected layout: {data_root}/valid/{META_JSON}"
            )

        with open(self.expr_path) as f:
            self._meta = json.load(f)

        self._items: List[RefYouTubeVOSItem] = self._build_items()

    def _build_items(self) -> List[RefYouTubeVOSItem]:
        items: List[RefYouTubeVOSItem] = []

        for video_id, video_data in self._meta["videos"].items():
            if self.sequences and video_id not in self.sequences:
                continue

            frame_ids   = video_data["frames"]   # annotated frames only
            expressions = video_data["expressions"]  # {"0": {"exp":..,"obj_id":..}, ...}

            # Validate JPEG presence
            first_jpeg = self.jpeg_root / video_id / f"{frame_ids[0]}.jpg"
            if not first_jpeg.exists():
                print(f"  [WARN] {video_id}: JPEGImages not found at {first_jpeg}, skipping")
                continue

            frame_paths = [
                str(self.jpeg_root / video_id / f"{fid}.jpg") for fid in frame_ids
            ]

            exp_items = list(expressions.items())
            if self.expressions_per_seq is not None:
                exp_items = exp_items[:self.expressions_per_seq]

            for exp_id, exp_data in exp_items:
                gt_boxes = self._load_gt_boxes(video_id, exp_id, frame_ids)

                # Skip if the object is never visible
                if all(b is None for b in gt_boxes):
                    continue

                items.append(RefYouTubeVOSItem(
                    seq_name=video_id,
                    exp_id=exp_id,
                    expression=exp_data["exp"],
                    obj_id=int(exp_data.get("obj_id", 0)),
                    frame_paths=frame_paths,
                    gt_boxes=gt_boxes,
                ))

        return items

    def _load_gt_boxes(
        self, video_id: str, exp_id: str, frame_ids: List[str]
    ) -> List[Box]:
        """
        For each annotated frame, load the binary mask from
        Annotations/{video_id}/{exp_id}/{frame_id}.png and convert to bbox.
        """
        boxes: List[Box] = []
        for fid in frame_ids:
            png_path = self.ann_root / video_id / exp_id / f"{fid}.png"
            if not png_path.exists():
                boxes.append(None)
            else:
                try:
                    boxes.append(_mask_to_box(str(png_path)))
                except Exception:
                    boxes.append(None)
        return boxes

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx) -> RefYouTubeVOSItem:
        return self._items[idx]

    def sequence_names(self) -> List[str]:
        seen: List[str] = []
        for it in self._items:
            if it.seq_name not in seen:
                seen.append(it.seq_name)
        return seen

    def summary(self) -> dict:
        seqs = self.sequence_names()
        return {
            "split": self.split,
            "num_sequences": len(seqs),
            "num_items": len(self._items),
        }
