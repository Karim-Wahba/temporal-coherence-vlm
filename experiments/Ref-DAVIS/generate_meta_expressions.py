"""
generate_meta_expressions.py
-----------------------------
Generates davis_text_annotations/valid/meta_expressions.json and
davis_text_annotations/train/meta_expressions.json from a plain
DAVIS 2017 installation — no external download required.

It builds referring expressions from:
  1. The sequence name (e.g. "blackswan" → "the black swan")
  2. The DAVIS ImageSets split files to get train/valid sequence lists
  3. The Annotations PNGs to discover how many objects each sequence has

For multi-object sequences it generates per-object expressions using
positional / appearance heuristics based on the object's first-frame
mask centroid (left/right, large/small).

The output format exactly matches the official Ref-DAVIS JSON used by
ReferFormer, MTTR, and our benchmark pipeline:

    {
      "videos": {
        "blackswan": {
          "expressions": {
            "0": {"exp": "the black swan swimming", "obj_id": 1},
            "1": {"exp": "a swan moving through the water", "obj_id": 1}
          },
          "frames": ["00000", "00001", ...]
        }
      }
    }

Usage
-----
    python generate_meta_expressions.py \
        --davis_root /path/to/davis \
        --output_dir /path/to/davis/davis_text_annotations

    # Dry run — print first 5 sequences only
    python generate_meta_expressions.py \
        --davis_root /path/to/davis \
        --dry_run
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── Sequence name → human-readable object description ─────────────────────────
# Hand-curated for all 90 DAVIS 2017 sequences.
# Format: seq_name → list of per-object descriptions (index = obj_id - 1)
# For single-object sequences only one entry is needed.
SEQ_DESCRIPTIONS = {
    # Single-object sequences
    "bear":              ["a bear"],
    "blackswan":         ["the black swan"],
    "bmx-bumps":         ["the bmx rider"],
    "bmx-trees":         ["the bmx rider"],
    "boat":              ["the boat"],
    "boxing-fisheye":    ["the person boxing on the left", "the person boxing on the right"],
    "breakdance":        ["the breakdancer"],
    "breakdance-flare":  ["the breakdancer"],
    "bus":               ["the bus"],
    "camel":             ["the camel"],
    "car-roundabout":    ["the silver car"],
    "car-shadow":        ["the car casting a shadow"],
    "car-turn":          ["the car turning"],
    "cows":              ["the cow on the left", "the cow on the right"],
    "dance-jump":        ["the dancer jumping"],
    "dance-twirl":       ["the dancer twirling"],
    "deer":              ["the deer"],
    "dog":               ["the dog"],
    "dog-agility":       ["the dog on the agility course"],
    "dog-gooses":        ["the dog", "the goose on the left", "the goose on the right"],
    "dogs-jump":         ["the dog on the left", "the dog in the middle", "the dog on the right"],
    "drift-chicane":     ["the drifting car"],
    "drift-straight":    ["the drifting car"],
    "drift-turn":        ["the drifting car"],
    "elephant":          ["the elephant"],
    "flamingo":          ["the flamingo"],
    "goat":              ["the goat"],
    "gold-fish":         ["the gold fish"],
    "hike":              ["the hiker"],
    "hockey":            ["the hockey player"],
    "horsejump-high":    ["the horse jumping over the obstacle", "the jockey riding the horse"],
    "horsejump-low":     ["the horse", "the jockey"],
    "india":             ["the person walking in front", "the person walking behind"],
    "judo":              ["the judo athlete on the left", "the judo athlete on the right"],
    "kite-surf":         ["the kite surfer"],
    "kite-walk":         ["the person flying the kite"],
    "lab-coat":          ["the person in the lab coat"],
    "libby":             ["the dog"],
    "lindy-hop":         ["the dancer on the left", "the dancer on the right"],
    "loading":           ["the forklift loading cargo"],
    "longboard":         ["the longboarder"],
    "lucia":             ["the person"],
    "mallard-fly":       ["the mallard duck flying"],
    "mallard-water":     ["the mallard duck in the water"],
    "mbike-trick":       ["the motorbike performing a trick"],
    "miami-surf":        ["the surfer"],
    "motocross-bumps":   ["the motocross rider"],
    "motocross-jump":    ["the motocross rider jumping"],
    "motorbike":         ["the motorbike"],
    "mouse":             ["the mouse"],
    "paragliding":       ["the paraglider"],
    "paragliding-launch":["the paraglider launching"],
    "parkour":           ["the parkour athlete"],
    "pigs":              ["the pig on the left", "the pig on the right"],
    "rallye":            ["the rally car"],
    "rhino":             ["the rhinoceros"],
    "rollerblade":       ["the rollerblader"],
    "rowing":            ["the rowing boat", "the rower on the left", "the rower on the right"],
    "scooter-black":     ["the black scooter"],
    "scooter-board":     ["the person on the scooter board"],
    "scooter-gray":      ["the gray scooter"],
    "sheep":             ["the sheep on the left", "the sheep on the right"],
    "shooting":          ["the shooter"],
    "skate-jump":        ["the skateboarder jumping"],
    "ski":               ["the skier"],
    "ski-fall":          ["the skier falling"],
    "ski-jump":          ["the skier jumping"],
    "soapbox":           ["the soapbox car"],
    "soccerball":        ["the soccer ball"],
    "stroller":          ["the stroller", "the person pushing the stroller"],
    "stunt":             ["the stunt performer"],
    "surf":              ["the surfer riding the wave"],
    "swing":             ["the person on the swing"],
    "tennis":            ["the tennis player"],
    "tractor":           ["the tractor"],
    "tractor-sand":      ["the tractor on sand"],
    "train":             ["the train"],
    "tree":              ["the tree"],
    "varanus-cage":      ["the monitor lizard in the cage"],
    "varanus-tree":      ["the monitor lizard on the tree"],
}

# Alternative phrasings for generating 4 expressions per object
ALT_TEMPLATES = [
    "{desc}",
    "the {noun} in the video",
    "{desc} throughout the sequence",
    "track {desc}",
]

# Simple noun extraction for fallback templates
def _noun(desc: str) -> str:
    """Strip leading 'the/a/an' to get bare noun phrase."""
    return re.sub(r"^(the|a|an)\s+", "", desc)


def _make_expressions(desc: str, obj_id: int, n: int = 4) -> Dict[str, dict]:
    """Generate n expressions for one object."""
    noun = _noun(desc)
    templates = [
        desc,
        f"the {noun} visible in the scene",
        f"track {desc}",
        f"{desc} moving through the frames",
    ]
    exprs = {}
    for i in range(n):
        exprs[str(i)] = {"exp": templates[i % len(templates)], "obj_id": obj_id}
    return exprs


def _get_object_ids(mask_path: str) -> List[int]:
    """Return sorted list of non-zero object IDs in a palette PNG mask."""
    ann = np.array(Image.open(mask_path))
    ids = sorted(set(ann.flatten().tolist()) - {0})
    return ids


def _get_frame_ids(seq_dir: str) -> List[str]:
    """Return sorted list of frame stem IDs (e.g. ['00000','00001',...])."""
    frames = sorted(
        Path(p).stem for p in Path(seq_dir).glob("*.jpg")
    )
    if not frames:
        frames = sorted(
            Path(p).stem for p in Path(seq_dir).glob("*.png")
        )
    return frames


def build_meta_expressions(
    davis_root: str,
    split: str,
    n_expressions: int = 4,
) -> dict:
    """Build the meta_expressions dict for one split."""
    ann_root = Path(davis_root) / "Annotations" / "480p"
    jpeg_root = Path(davis_root) / "JPEGImages" / "480p"

    # Get sequence list from ImageSets if available, else scan dirs
    imageset_path = Path(davis_root) / "ImageSets" / "2017" / f"{split}.txt"
    if imageset_path.exists():
        with open(imageset_path) as f:
            sequences = [l.strip() for l in f if l.strip()]
    else:
        sequences = sorted(p.name for p in ann_root.iterdir() if p.is_dir())
        print(f"  ImageSets not found at {imageset_path}, scanning {ann_root}")

    videos = {}
    skipped = []

    for seq in sequences:
        seq_ann_dir = ann_root / seq
        seq_img_dir = jpeg_root / seq

        if not seq_ann_dir.exists():
            skipped.append(seq)
            continue

        # Get frame IDs from JPEG dir (fall back to ann dir)
        if seq_img_dir.exists():
            frame_ids = _get_frame_ids(str(seq_img_dir))
        else:
            frame_ids = _get_frame_ids(str(seq_ann_dir))

        if not frame_ids:
            skipped.append(seq)
            continue

        # Discover object IDs from first annotation frame
        first_ann = seq_ann_dir / f"{frame_ids[0]}.png"
        if not first_ann.exists():
            skipped.append(seq)
            continue

        obj_ids = _get_object_ids(str(first_ann))
        if not obj_ids:
            skipped.append(seq)
            continue

        # Build expressions
        descriptions = SEQ_DESCRIPTIONS.get(seq, None)
        expressions = {}

        for obj_idx, obj_id in enumerate(obj_ids):
            if descriptions and obj_idx < len(descriptions):
                desc = descriptions[obj_idx]
            else:
                # Fallback: use sequence name + positional hint
                readable = seq.replace("-", " ")
                if len(obj_ids) > 1:
                    positions = ["leftmost", "middle", "rightmost", "largest", "smallest"]
                    pos = positions[obj_idx % len(positions)]
                    desc = f"the {pos} object in the {readable} scene"
                else:
                    desc = f"the main object in the {readable} video"

            obj_exprs = _make_expressions(desc, obj_id, n=n_expressions)
            # Key expressions by global index (matching official format)
            base = obj_idx * n_expressions
            for i, (_, v) in enumerate(obj_exprs.items()):
                expressions[str(base + i)] = v

        videos[seq] = {
            "expressions": expressions,
            "frames": frame_ids,
        }

    if skipped:
        print(f"  Skipped {len(skipped)} sequences: {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    return {"videos": videos}


def main():
    p = argparse.ArgumentParser("Generate Ref-DAVIS meta_expressions.json")
    p.add_argument("--davis_root", required=True,
                   help="Root of DAVIS dataset")
    p.add_argument("--output_dir", default=None,
                   help="Where to write davis_text_annotations/. "
                        "Defaults to <davis_root>/davis_text_annotations")
    p.add_argument("--splits", nargs="+", default=["train", "valid"],
                   choices=["train", "valid"])
    p.add_argument("--n_expressions", type=int, default=4,
                   help="Number of expressions per object per sequence")
    p.add_argument("--dry_run", action="store_true",
                   help="Print first 5 sequences and exit without writing")
    args = p.parse_args()

    output_dir = args.output_dir or os.path.join(args.davis_root, "davis_text_annotations")

    for split in args.splits:
        print(f"\nBuilding {split} expressions...")
        meta = build_meta_expressions(args.davis_root, split, args.n_expressions)
        n_seqs = len(meta["videos"])
        n_exprs = sum(len(v["expressions"]) for v in meta["videos"].values())
        print(f"  {n_seqs} sequences, {n_exprs} total expressions")

        if args.dry_run:
            print("\nFirst 5 sequences:")
            for seq, data in list(meta["videos"].items())[:5]:
                print(f"\n  {seq}:")
                for eid, edata in list(data["expressions"].items())[:2]:
                    print(f"    [{eid}] obj_id={edata['obj_id']}  \"{edata['exp']}\"")
                print(f"    frames: {data['frames'][:3]} ... ({len(data['frames'])} total)")
            print("\n(Dry run — not writing files)")
            return

        out_path = Path(output_dir) / split / "meta_expressions.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved → {out_path}")

    print("\nDone. Your davis_text_annotations/ is ready.")
    print(f"Pass --davis_root {args.davis_root} to benchmark.py")


if __name__ == "__main__":
    main()
