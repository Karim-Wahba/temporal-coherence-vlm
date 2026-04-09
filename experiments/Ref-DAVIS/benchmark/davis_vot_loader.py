"""
davis_vot_loader.py
-------------------
Loads DAVIS sequences for VOT evaluation using Annotations_bbox and
Ref-DAVIS text expressions.

Expected directory layout:

    davis_root/
    ├── JPEGImages/480p/<sequence>/<frame>.jpg
    ├── Annotations_bbox/480p/<sequence>.json   (per-frame bbox, {x,y,w,h})
    ├── ImageSets/2017/val.txt                  (split sequence lists)
    └── davis_text_annotations/
        ├── train/meta_expressions.json
        └── valid/meta_expressions.json

Bbox JSON format per sequence:
    {
      "00000": {"1": {"x": 127, "y": 72, "w": 384, "h": 313}},
      ...
    }
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Box = Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None


@dataclass
class DAVISVOTItem:
    seq_name: str
    exp_id: str
    expression: str
    obj_id: int
    frame_paths: List[str]
    gt_boxes: List[Box]          # (x1, y1, x2, y2) per frame, None if missing
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


class DAVISVOTLoader:
    """
    Iterates over (sequence, expression) pairs with GT bounding boxes.

    Parameters
    ----------
    davis_root : str
        Root of the DAVIS dataset.
    split : str
        "train" or "valid"
    expressions_per_seq : int or None
        If set, only load the first N expressions per sequence.
    sequences : list[str] or None
        If set, only load these sequences.
    """

    def __init__(
        self,
        davis_root: str,
        split: str = "valid",
        expressions_per_seq: Optional[int] = None,
        sequences: Optional[List[str]] = None,
    ):
        self.davis_root = Path(davis_root)
        self.split = split
        self.expressions_per_seq = expressions_per_seq
        self.sequences = sequences

        self.jpeg_root = self.davis_root / "JPEGImages" / "480p"
        self.bbox_root = self.davis_root / "Annotations_bbox" / "480p"
        self.expr_path = (
            self.davis_root
            / "davis_text_annotations"
            / split
            / "meta_expressions.json"
        )

        if not self.bbox_root.exists():
            raise FileNotFoundError(
                f"Bbox annotations not found at {self.bbox_root}."
            )
        if not self.expr_path.exists():
            raise FileNotFoundError(
                f"Text annotations not found at {self.expr_path}."
            )

        with open(self.expr_path) as f:
            self._meta = json.load(f)

        self._items: List[DAVISVOTItem] = self._build_items()

    def _load_gt_boxes(self, seq_name: str, frame_ids: List[str], obj_id: int) -> List[Box]:
        """Load GT bboxes for a sequence, converting {x,y,w,h} to (x1,y1,x2,y2)."""
        bbox_file = self.bbox_root / f"{seq_name}.json"
        if not bbox_file.exists():
            return [None] * len(frame_ids)

        with open(bbox_file) as f:
            raw = json.load(f)

        obj_key = str(obj_id)
        boxes = []
        for fid in frame_ids:
            frame_data = raw.get(fid, {})
            obj_data = frame_data.get(obj_key)
            if obj_data:
                x, y, w, h = obj_data["x"], obj_data["y"], obj_data["w"], obj_data["h"]
                boxes.append((x, y, x + w, y + h))
            else:
                boxes.append(None)
        return boxes

    def _build_items(self) -> List[DAVISVOTItem]:
        items = []
        for seq_name, seq_data in self._meta["videos"].items():
            if self.sequences and seq_name not in self.sequences:
                continue

            # Skip sequences without bbox annotations
            if not (self.bbox_root / f"{seq_name}.json").exists():
                continue

            frame_ids = seq_data["frames"]
            frame_paths = [
                str(self.jpeg_root / seq_name / f"{fid}.jpg") for fid in frame_ids
            ]

            missing = [p for p in frame_paths if not os.path.exists(p)]
            if missing:
                print(
                    f"  [WARN] {seq_name}: {len(missing)} frames missing "
                    f"(first: {missing[0]})"
                )
                continue

            expressions = seq_data["expressions"]
            exp_items = list(expressions.items())
            if self.expressions_per_seq is not None:
                exp_items = exp_items[: self.expressions_per_seq]

            for exp_id, exp_data in exp_items:
                obj_id = int(exp_data["obj_id"])
                gt_boxes = self._load_gt_boxes(seq_name, frame_ids, obj_id)
                items.append(
                    DAVISVOTItem(
                        seq_name=seq_name,
                        exp_id=exp_id,
                        expression=exp_data["exp"],
                        obj_id=obj_id,
                        frame_paths=frame_paths,
                        gt_boxes=gt_boxes,
                    )
                )

        return items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def sequence_names(self) -> List[str]:
        seen = []
        for it in self._items:
            if it.seq_name not in seen:
                seen.append(it.seq_name)
        return seen

    def summary(self) -> Dict:
        seqs = self.sequence_names()
        return {
            "split": self.split,
            "num_sequences": len(seqs),
            "num_items": len(self._items),
            "sequences": seqs,
        }
