"""
test_runner.py
--------------
Standalone test for QwenVOSRunner — loads a few DAVIS frames, runs the
runner, and prints raw model output + parsed boxes so you can verify the
pipeline without running the full benchmark.

Usage
-----
    python test_runner.py \
        --davis_root /home/geiger/gwb913/git/davis/DAVIS2017/unsupervised \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --sequence blackswan \
        --expression "the black swan" \
        --strategy joint \
        --max_frames 5
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmark"))

from PIL import Image
from benchmark.qwen_vos_runner import QwenVOSRunner, _parse_time_detections, _parse_box


def load_frames(davis_root: str, sequence: str, max_frames: int):
    frames_dir = Path(davis_root) / "JPEGImages" / "480p" / sequence
    paths = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frames found at {frames_dir}")
    frames = [Image.open(p).convert("RGB") for p in paths]
    print(f"Loaded {len(frames)} frames from {frames_dir}")
    return frames


def load_model(model_id: str):
    print(f"Loading model: {model_id} ...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.")
    return model, processor


def main():
    p = argparse.ArgumentParser("Test QwenVOSRunner standalone")
    p.add_argument("--davis_root", default="/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--sequence", default="blackswan")
    p.add_argument("--expression", default="the black swan")
    p.add_argument("--strategy", default="joint", choices=["joint", "per_frame"])
    p.add_argument("--max_frames", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--sample_rate", type=int, default=2,
                   help="Send every Nth frame to the model (1=all frames, tends to fail)")
    args = p.parse_args()

    frames = load_frames(args.davis_root, args.sequence, args.max_frames)
    W, H = frames[0].size

    model, processor = load_model(args.model_id)
    runner = QwenVOSRunner(
        model, processor,
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        sample_rate=args.sample_rate,
    )

    # ── Patch _generate to also print raw output ──────────────────────────────
    _orig_generate = runner._generate
    def _generate_verbose(messages):
        raw = _orig_generate(messages)
        print("\n" + "=" * 60)
        print("RAW MODEL OUTPUT:")
        print("=" * 60)
        print(raw)
        print("=" * 60 + "\n")
        return raw
    runner._generate = _generate_verbose

    # ── Run ───────────────────────────────────────────────────────────────────
    n_sent = len(frames[::args.sample_rate])
    print(f"\nRunning strategy='{args.strategy}' on '{args.sequence}' | expr: \"{args.expression}\"")
    print(f"  {len(frames)} total frames, sample_rate={args.sample_rate} → {n_sent} sent to model")
    boxes = runner.run(frames, args.expression)

    print("\nPARSED BOXES (pixel coords):")
    for i, box in enumerate(boxes):
        print(f"  frame {i:03d}: {box}")

    n_found = sum(1 for b in boxes if b is not None)
    print(f"\n{n_found}/{len(boxes)} frames have a box.")
    if n_found == 0:
        print("WARNING: no boxes found — check raw output above for the actual model response format.")


if __name__ == "__main__":
    main()
