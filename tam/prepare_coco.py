"""Download and prepare COCO 2014 minival dataset for TAM evaluation.

Creates the directory structure expected by eval_coco.py:
    {output_dir}/
        annotations/
            instances_minival2014.json
            captions_val2014.json
        image/
            000000XXXXXX.jpg
        seg_label/
            000000XXXXXX.png

Usage:
    python -m tam.prepare_coco --output-dir ./coco_data [--max-images 100]
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_file(url, dest, desc=None):
    """Download a file with progress."""
    desc = desc or os.path.basename(dest)
    if os.path.exists(dest):
        print(f"  [skip] {desc} already exists")
        return

    print(f"  Downloading {desc}...")
    tmp = dest + ".tmp"
    try:
        def _reporthook(count, block_size, total_size):
            pct = count * block_size * 100 // max(total_size, 1)
            print(f"\r  {desc}: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()
        os.rename(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Failed to download {url}: {e}")


def download_annotations(output_dir):
    """Download COCO 2014 val annotations."""
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # Full val annotations zip
    ann_zip = os.path.join(output_dir, "annotations_trainval2014.zip")
    captions_path = os.path.join(ann_dir, "captions_val2014.json")
    instances_path = os.path.join(ann_dir, "instances_val2014.json")

    if not os.path.exists(captions_path) or not os.path.exists(instances_path):
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            ann_zip,
            "COCO 2014 annotations (241 MB)",
        )
        print("  Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as zf:
            for name in zf.namelist():
                if name.endswith("captions_val2014.json") or name.endswith("instances_val2014.json"):
                    data = zf.read(name)
                    out = os.path.join(ann_dir, os.path.basename(name))
                    with open(out, "wb") as f:
                        f.write(data)
        # Clean up zip
        os.remove(ann_zip)

    return captions_path, instances_path


def create_minival_split(instances_path, ann_dir, max_images=-1):
    """Create minival split from full val annotations.

    The standard COCO minival2014 is a 5K subset. We create our own split
    by taking the first N images from the val set (sorted by id).
    """
    minival_path = os.path.join(ann_dir, "instances_minival2014.json")
    if os.path.exists(minival_path) and max_images <= 0:
        print(f"  [skip] minival split already exists")
        return minival_path

    print("  Creating minival split...")
    with open(instances_path) as f:
        data = json.load(f)

    # Sort images by ID and take subset
    images = sorted(data["images"], key=lambda x: x["id"])
    n = max_images if max_images > 0 else 5000
    images = images[:n]
    image_ids = {img["id"] for img in images}

    # Filter annotations to only include minival images
    annotations = [a for a in data["annotations"] if a["image_id"] in image_ids]

    minival = {
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }

    with open(minival_path, "w") as f:
        json.dump(minival, f)

    print(f"  Created minival split with {len(images)} images, {len(annotations)} annotations")
    return minival_path


def download_images(minival_path, output_dir):
    """Download only the images in the minival split."""
    img_dir = os.path.join(output_dir, "image")
    os.makedirs(img_dir, exist_ok=True)

    with open(minival_path) as f:
        data = json.load(f)

    images = data["images"]
    existing = set(os.listdir(img_dir))

    to_download = []
    for img in images:
        # Output filename: zero-padded ID
        fn = str(img["id"]).zfill(12) + ".jpg"
        if fn not in existing:
            to_download.append((img["coco_url"], fn))

    if not to_download:
        print(f"  [skip] All {len(images)} images already downloaded")
        return

    print(f"  Downloading {len(to_download)} images ({len(images) - len(to_download)} already exist)...")
    for i, (url, fn) in enumerate(to_download):
        dest = os.path.join(img_dir, fn)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"\n  [warn] Failed to download {fn}: {e}")
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(to_download):
            print(f"\r  Images: {i + 1}/{len(to_download)}", end="", flush=True)

    print()


def generate_seg_labels(minival_path, output_dir):
    """Generate per-pixel category label PNGs from COCO polygon annotations.

    Each pixel in the output PNG contains the category ID (0 = background).
    """
    import cv2
    import numpy as np

    seg_dir = os.path.join(output_dir, "seg_label")
    os.makedirs(seg_dir, exist_ok=True)

    with open(minival_path) as f:
        data = json.load(f)

    # Build category ID mapping (COCO category_id -> our label)
    # The COCO category IDs are not contiguous, but our config.py COCO_CATEGORIES
    # maps names to the original COCO category IDs, so we use those directly.
    cat_id_map = {c["id"]: c["id"] for c in data["categories"]}

    # Group annotations by image
    img_anns = {}
    for ann in data["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    existing = set(os.listdir(seg_dir))
    images = data["images"]
    generated = 0

    for i, img_info in enumerate(images):
        fn = str(img_info["id"]).zfill(12) + ".png"
        if fn in existing:
            continue

        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in img_anns.get(img_info["id"], []):
            cat_id = cat_id_map.get(ann["category_id"], 0)
            if cat_id == 0:
                continue

            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                # Polygon format
                for poly in seg:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], int(cat_id))
            # Skip RLE format (uncommon in val set)

        out_path = os.path.join(seg_dir, fn)
        cv2.imwrite(out_path, mask)
        generated += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"\r  Seg labels: {i + 1}/{len(images)}", end="", flush=True)

    print(f"\n  Generated {generated} new segmentation masks ({len(existing)} already existed)")


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO dataset for TAM evaluation")
    parser.add_argument("--output-dir", type=str, default="./coco_data",
                        help="Output directory for dataset")
    parser.add_argument("--max-images", type=int, default=100,
                        help="Max images in minival split (-1 for 5000)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Preparing COCO dataset in: {output_dir}")
    print(f"Max images: {args.max_images if args.max_images > 0 else 5000}")
    print()

    # Step 1: Download annotations
    print("[1/4] Downloading annotations...")
    captions_path, instances_path = download_annotations(output_dir)

    # Step 2: Create minival split
    print("[2/4] Creating minival split...")
    minival_path = create_minival_split(instances_path, os.path.join(output_dir, "annotations"),
                                         max_images=args.max_images)

    # Step 3: Download images
    print("[3/4] Downloading images...")
    download_images(minival_path, output_dir)

    # Step 4: Generate segmentation labels
    print("[4/4] Generating segmentation labels...")
    generate_seg_labels(minival_path, output_dir)

    print(f"\nDone! Dataset ready at: {output_dir}")
    print(f"\nRun evaluation with:")
    print(f"  python -m tam.eval_coco --model-path Qwen/Qwen3-VL-2B-Instruct \\")
    print(f"    --dataset-path {output_dir} --no-quantize --max-new-tokens 40")


if __name__ == "__main__":
    main()
