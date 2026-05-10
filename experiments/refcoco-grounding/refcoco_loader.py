"""
refcoco_loader.py
-----------------
Loads RefCOCO / RefCOCO+ / RefCOCOg items (image, expression, GT bbox).

Expected directory layout:

    refcoco_root/
    ├── images/
    │   └── train2014/
    │       └── COCO_train2014_*.jpg
    └── annotations/
        ├── refcoco/
        │   ├── refs(unc).p
        │   └── instances.json
        ├── refcoco+/
        │   ├── refs(unc).p
        │   └── instances.json
        └── refcocog/
            ├── refs(google).p
            └── instances.json

Download commands:
    cd ~/git/data/refcoco/annotations
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip  && unzip refcoco.zip
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip && unzip refcoco+.zip
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip && unzip refcocog.zip
    cd ~/git/data/refcoco/images
    wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip

Dataset and split choices:
    dataset : "refcoco" | "refcoco+" | "refcocog"
    split   : "train" | "val" | "testA" | "testB"   (refcocog has "val" and "test" only)
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

# Pickle filename per dataset
_REFS_PICKLE = {
    "refcoco":  "refs(unc).p",
    "refcoco+": "refs(unc).p",
    "refcocog": "refs(google).p",
}


@dataclass
class RefCOCOItem:
    ref_id:      int
    sent_id:     int
    image_id:    int
    ann_id:      int
    category_id: int
    file_name:   str    # e.g. COCO_train2014_000000123456.jpg
    image_path:  str
    expression:  str
    gt_box:      Box    # (x1, y1, x2, y2) pixel coords
    split:       str

    _image_pil: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def image_pil(self) -> Image.Image:
        if self._image_pil is None:
            self._image_pil = Image.open(self.image_path).convert("RGB")
        return self._image_pil

    def image_size(self) -> Tuple[int, int]:
        """Returns (H, W)."""
        img = Image.open(self.image_path)
        return img.height, img.width


class RefCOCOLoader:
    """
    Iterates over (image, expression, gt_bbox) triples for the given split.

    Parameters
    ----------
    refcoco_root : str
        Root directory containing images/ and annotations/ subdirs.
    dataset : str
        One of "refcoco", "refcoco+", "refcocog".
    split : str
        One of "train", "val", "testA", "testB" (refcocog: "val", "test").
    max_items : int or None
        Cap total number of items loaded (useful for quick runs).
    sents_per_ref : int or None
        Cap number of sentences (expressions) per ref object.
    """

    def __init__(
        self,
        refcoco_root: str,
        dataset: str = "refcoco",
        split: str = "val",
        max_items: Optional[int] = None,
        sents_per_ref: Optional[int] = 1,
    ):
        if dataset not in _REFS_PICKLE:
            raise ValueError(f"dataset must be one of {list(_REFS_PICKLE)}; got {dataset!r}")

        self.root = Path(refcoco_root)
        self.dataset = dataset
        self.split = split

        ann_dir = self.root / "annotations" / dataset
        refs_path = ann_dir / _REFS_PICKLE[dataset]
        inst_path = ann_dir / "instances.json"

        if not refs_path.exists():
            raise FileNotFoundError(
                f"Refs pickle not found at {refs_path}.\n"
                "Run the download commands at the top of refcoco_loader.py."
            )
        if not inst_path.exists():
            raise FileNotFoundError(f"instances.json not found at {inst_path}.")

        with open(refs_path, "rb") as f:
            refs = pickle.load(f, encoding="latin1")

        with open(inst_path) as f:
            instances = json.load(f)

        # Build ann_id → {bbox, category_id} map
        self._ann_map: Dict[int, dict] = {
            a["id"]: a for a in instances["annotations"]
        }
        # Build image_id → file_name map
        self._img_map: Dict[int, str] = {
            img["id"]: img["file_name"] for img in instances["images"]
        }

        self._items: List[RefCOCOItem] = self._build_items(
            refs, max_items, sents_per_ref
        )

    def _build_items(
        self,
        refs: list,
        max_items: Optional[int],
        sents_per_ref: Optional[int],
    ) -> List[RefCOCOItem]:
        items: List[RefCOCOItem] = []
        for ref in refs:
            if ref["split"] != self.split:
                continue

            ann_id = ref["ann_id"]
            ann = self._ann_map.get(ann_id)
            if ann is None:
                continue

            # COCO bbox: [x, y, w, h] → (x1, y1, x2, y2)
            x, y, w, h = ann["bbox"]
            gt_box: Box = (int(x), int(y), int(x + w), int(y + h))

            image_id = ref["image_id"]
            file_name = self._img_map.get(image_id, ref.get("file_name", ""))

            # images may live under train2014/ or val2014/
            image_path = self._find_image(file_name)
            if image_path is None:
                continue

            sentences = ref["sentences"]
            if sents_per_ref is not None:
                sentences = sentences[:sents_per_ref]

            for sent in sentences:
                items.append(RefCOCOItem(
                    ref_id=ref["ref_id"],
                    sent_id=sent["sent_id"],
                    image_id=image_id,
                    ann_id=ann_id,
                    category_id=ref["category_id"],
                    file_name=file_name,
                    image_path=str(image_path),
                    expression=sent["sent"],
                    gt_box=gt_box,
                    split=ref["split"],
                ))
                if max_items and len(items) >= max_items:
                    return items

        return items

    def _find_image(self, file_name: str) -> Optional[Path]:
        """Search train2014/ then val2014/ for the image."""
        for subdir in ("train2014", "val2014"):
            p = self.root / "images" / subdir / file_name
            if p.exists():
                return p
        # Some datasets store images directly under images/
        p = self.root / "images" / file_name
        if p.exists():
            return p
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx) -> RefCOCOItem:
        return self._items[idx]

    def summary(self) -> dict:
        unique_refs = len({it.ref_id for it in self._items})
        unique_imgs = len({it.image_id for it in self._items})
        return {
            "dataset": self.dataset,
            "split": self.split,
            "num_items": len(self._items),
            "unique_refs": unique_refs,
            "unique_images": unique_imgs,
        }
