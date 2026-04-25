"""
ref_davis_loader.py
-------------------
Loads Ref-DAVIS 2017: DAVIS 2017 frames + masks + davis_text_annotations.

Expected directory layout (after downloading davis_text_annotations.zip):

    davis_root/
    ├── JPEGImages/480p/<sequence>/<frame>.jpg
    ├── Annotations/480p/<sequence>/<frame>.png   (palette PNG, object IDs)
    └── davis_text_annotations/
        ├── train/meta_expressions.json
        └── valid/meta_expressions.json

Usage
-----
    loader = RefDAVISLoader("/path/to/davis_root", split="valid")
    for item in loader:
        # item.seq_name, item.expression, item.obj_id,
        # item.frame_paths, item.mask_paths, item.frames_pil, item.masks
        ...
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class RefDAVISItem:
    seq_name: str
    exp_id: str
    expression: str
    obj_id: int
    frame_paths: List[str]
    mask_paths: List[str]
    # Lazy-loaded
    _frames_pil: Optional[List[Image.Image]] = field(default=None, repr=False)
    _masks: Optional[List[np.ndarray]] = field(default=None, repr=False)

    @property
    def frames_pil(self) -> List[Image.Image]:
        if self._frames_pil is None:
            self._frames_pil = [Image.open(p).convert("RGB") for p in self.frame_paths]
        return self._frames_pil

    @property
    def masks(self) -> List[np.ndarray]:
        """Binary masks (H,W) uint8 for obj_id, one per frame."""
        if self._masks is None:
            self._masks = []
            for p in self.mask_paths:
                ann = np.array(Image.open(p))
                self._masks.append((ann == self.obj_id).astype(np.uint8))
        return self._masks

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    def frame_size(self) -> Tuple[int, int]:
        """Returns (H, W)."""
        img = Image.open(self.frame_paths[0])
        return img.height, img.width


class RefDAVISLoader:
    """
    Iterates over all (sequence, expression) pairs in the Ref-DAVIS split.

    Parameters
    ----------
    davis_root : str
        Root of the DAVIS dataset containing JPEGImages/, Annotations/, and
        davis_text_annotations/.
    split : str
        "train" or "valid"
    expressions_per_seq : int or None
        If set, only load the first N expressions per sequence (useful for
        quick runs; set to 1 for expression-0 only).
    sequences : list[str] or None
        If set, only load these sequences (useful for debugging).
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
        self.ann_root = self.davis_root / "Annotations" / "480p"
        self.expr_path = (
            self.davis_root
            / "davis_text_annotations"
            / split
            / "meta_expressions.json"
        )

        if not self.expr_path.exists():
            raise FileNotFoundError(
                f"Text annotations not found at {self.expr_path}.\n"
                "Download davis_text_annotations.zip from:\n"
                "  https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning"
                "/research/video-segmentation/video-object-segmentation-with-language-referring-expressions\n"
                "and unzip it inside your DAVIS root."
            )

        with open(self.expr_path) as f:
            self._meta = json.load(f)

        self._items: List[RefDAVISItem] = self._build_items()

    def _build_items(self) -> List[RefDAVISItem]:
        items = []
        for seq_name, seq_data in self._meta["videos"].items():
            if self.sequences and seq_name not in self.sequences:
                continue

            frame_ids = seq_data["frames"]
            frame_paths = [
                str(self.jpeg_root / seq_name / f"{fid}.jpg") for fid in frame_ids
            ]
            mask_paths = [
                str(self.ann_root / seq_name / f"{fid}.png") for fid in frame_ids
            ]

            # Validate paths exist
            missing_frames = [p for p in frame_paths if not os.path.exists(p)]
            if missing_frames:
                print(
                    f"  [WARN] {seq_name}: {len(missing_frames)} frames missing "
                    f"(first: {missing_frames[0]})"
                )
                continue

            expressions = seq_data["expressions"]
            exp_items = list(expressions.items())
            if self.expressions_per_seq is not None:
                exp_items = exp_items[: self.expressions_per_seq]

            for exp_id, exp_data in exp_items:
                items.append(
                    RefDAVISItem(
                        seq_name=seq_name,
                        exp_id=exp_id,
                        expression=exp_data["exp"],
                        obj_id=int(exp_data["obj_id"]),
                        frame_paths=frame_paths,
                        mask_paths=mask_paths,
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
