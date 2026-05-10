"""
test_determinism.py
-------------------
Quick smoke test: run QwenVOTRunner twice on the same (short) DAVIS sequence
and check that gen_text, predicted boxes, and TAM maps all match bit-for-bit.

Usage
-----
    python test_determinism.py                       # picks the shortest seq
    python test_determinism.py --sequence bear       # force a specific seq
    python test_determinism.py --davis_root /path    # override DAVIS root

Exit code: 0 = deterministic, 1 = mismatch.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # for `benchmark.*` imports
sys.path.insert(0, str(_HERE))

from davis_vot_loader import DAVISVOTLoader
from qwen_vot_runner import QwenVOTRunner


def _shortest_item(loader):
    items = list(loader)
    items.sort(key=lambda it: it.num_frames)
    return items[0]


def _pick_item(loader, sequence: str | None):
    if sequence:
        for it in loader:
            if it.seq_name == sequence:
                return it
        raise SystemExit(f"sequence {sequence!r} not found")
    return _shortest_item(loader)


def _compare_boxes(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return False
        if tuple(x) != tuple(y):
            return False
    return True


def _compare_tam(a: dict, b: dict) -> tuple[bool, str]:
    if a.keys() != b.keys():
        return False, f"tam_result keys differ: {set(a) ^ set(b)}"
    if a["gen_tokens"] != b["gen_tokens"]:
        return False, "gen_tokens differ"
    if a["gen_text"] != b["gen_text"]:
        return False, "gen_text differ"
    if a["vision_shape"] != b["vision_shape"]:
        return False, f"vision_shape differ: {a['vision_shape']} vs {b['vision_shape']}"
    maps_a, maps_b = a["tam_maps"], b["tam_maps"]
    if len(maps_a) != len(maps_b):
        return False, f"tam_maps length differ: {len(maps_a)} vs {len(maps_b)}"
    for i, (ma, mb) in enumerate(zip(maps_a, maps_b)):
        if (ma is None) != (mb is None):
            return False, f"tam_maps[{i}] presence differs"
        if ma is None:
            continue
        if ma.shape != mb.shape:
            return False, f"tam_maps[{i}] shape: {ma.shape} vs {mb.shape}"
        if not np.array_equal(ma, mb):
            diff = float(np.max(np.abs(ma.astype(np.float64) - mb.astype(np.float64))))
            return False, f"tam_maps[{i}] values differ (max abs diff = {diff:.3e})"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root", default="/home/wahba/git/data/davis/davis/DAVIS2017/unsupervised")
    ap.add_argument("--model_id",   default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--split",      default="valid")
    ap.add_argument("--sequence",   default=None,
                    help="Specific DAVIS sequence to test; default = shortest")
    ap.add_argument("--sample_rate", type=int, default=16,
                    help="High default keeps the test cheap")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_tam", action="store_true",
                    help="Only run .run(), skip .run_with_tam()")
    args = ap.parse_args()

    print(f"Loading DAVIS from {args.davis_root}")
    loader = DAVISVOTLoader(args.davis_root, split=args.split, expressions_per_seq=1)
    item = _pick_item(loader, args.sequence)
    print(f"  test item: seq={item.seq_name!r} exp_id={item.exp_id} "
          f"num_frames={item.num_frames} expression={item.expression!r}")

    print(f"Loading model: {args.model_id}")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    runner = QwenVOTRunner(
        model, processor,
        max_new_tokens=args.max_new_tokens,
        sample_rate=args.sample_rate,
        video_mode=True,
        seed=args.seed,
    )

    failures = []

    # ── pass 1: plain run() twice ────────────────────────────────────────────
    print("\n[1/2] runner.run() x2")
    t0 = time.time()
    boxes_a, raw_a = runner.run(item.frames_pil, item.expression)
    print(f"  run #1 done in {time.time()-t0:.1f}s ({len(raw_a)} chars)")
    t0 = time.time()
    boxes_b, raw_b = runner.run(item.frames_pil, item.expression)
    print(f"  run #2 done in {time.time()-t0:.1f}s ({len(raw_b)} chars)")

    if raw_a == raw_b:
        print("  raw_text:    MATCH")
    else:
        failures.append("raw_text differs")
        print("  raw_text:    MISMATCH")
        print(f"    A (first 200): {raw_a[:200]!r}")
        print(f"    B (first 200): {raw_b[:200]!r}")

    if _compare_boxes(boxes_a, boxes_b):
        print("  pred_boxes:  MATCH")
    else:
        failures.append("pred_boxes differ")
        print("  pred_boxes:  MISMATCH")
        for i, (x, y) in enumerate(zip(boxes_a, boxes_b)):
            if x != y:
                print(f"    frame {i}: {x} vs {y}")
                if i > 5: break

    # ── pass 2: run_with_tam() twice ─────────────────────────────────────────
    if not args.skip_tam:
        print("\n[2/2] runner.run_with_tam() x2")
        t0 = time.time()
        boxes_c, raw_c, tam_c = runner.run_with_tam(item.frames_pil, item.expression)
        print(f"  run #1 done in {time.time()-t0:.1f}s")
        t0 = time.time()
        boxes_d, raw_d, tam_d = runner.run_with_tam(item.frames_pil, item.expression)
        print(f"  run #2 done in {time.time()-t0:.1f}s")

        if raw_c == raw_d:
            print("  raw_text:    MATCH")
        else:
            failures.append("run_with_tam raw_text differs")
            print("  raw_text:    MISMATCH")

        if _compare_boxes(boxes_c, boxes_d):
            print("  pred_boxes:  MATCH")
        else:
            failures.append("run_with_tam pred_boxes differ")
            print("  pred_boxes:  MISMATCH")

        ok, why = _compare_tam(tam_c, tam_d)
        if ok:
            print("  tam_maps:    MATCH")
        else:
            failures.append(f"tam_maps differ: {why}")
            print(f"  tam_maps:    MISMATCH ({why})")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n=== Result ===")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS — runner is deterministic across repeated calls.")


if __name__ == "__main__":
    main()
