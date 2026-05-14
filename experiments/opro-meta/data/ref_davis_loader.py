"""
data/ref_davis_loader.py
------------------------
Thin wrapper around the existing DAVISVOTLoader. Exposes a Clip dataclass
that the inner loop iterates over.

The underlying loader yields one DAVISVOTItem per (sequence, expression, obj_id).
For OPRO we want unique (sequence, obj_id) targets — each target keeps its
original Ref-DAVIS expression as the seed. We pick the first available
expression for each (seq, obj_id) by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

import _paths  # noqa: F401  -- side-effect: extends sys.path

from benchmark.davis_vot_loader import DAVISVOTLoader, DAVISVOTItem


Box = tuple  # (x1, y1, x2, y2) or None


@dataclass
class Clip:
    """A single (seq, obj) target with its seed expression and GT boxes."""
    seq_name: str
    obj_id: int
    exp_id: str
    seed_expression: str
    item: DAVISVOTItem = field(repr=False)

    @property
    def frames_pil(self) -> List[Image.Image]:
        return self.item.frames_pil

    @property
    def gt_boxes(self) -> List[Optional[Box]]:
        return self.item.gt_boxes

    @property
    def num_frames(self) -> int:
        return self.item.num_frames

    def frame_size(self) -> tuple[int, int]:
        return self.item.frame_size()


class RefDavisClipLoader:
    """Iterates over unique (seq, obj_id) clips with one seed expression each."""

    def __init__(
        self,
        davis_root: str,
        split: str = "valid",
        sequences: Optional[List[str]] = None,
        max_clips: Optional[int] = None,
    ):
        self.loader = DAVISVOTLoader(davis_root, split=split)
        self.sequences = set(sequences) if sequences else None
        self.max_clips = max_clips
        self._clips: Optional[List[Clip]] = None

    def clips(self) -> List[Clip]:
        if self._clips is not None:
            return self._clips
        seen: set[tuple[str, int]] = set()
        clips: List[Clip] = []
        for item in self.loader:
            if self.sequences and item.seq_name not in self.sequences:
                continue
            key = (item.seq_name, int(item.obj_id))
            if key in seen:
                continue
            seen.add(key)
            clips.append(Clip(
                seq_name=item.seq_name,
                obj_id=int(item.obj_id),
                exp_id=item.exp_id,
                seed_expression=item.expression,
                item=item,
            ))
            if self.max_clips and len(clips) >= self.max_clips:
                break
        self._clips = clips
        return clips

    def __iter__(self):
        return iter(self.clips())

    def __len__(self):
        return len(self.clips())
