"""
sanity_check.py
---------------
Does noun-only TAM beat standard (all-token) TAM at grounding?

Compares two heatmap aggregations on N DAVIS valid sequences:

  H_std  = mean over ALL generated-token TAM maps         (current default)
  H_noun = mean over ONLY label_noun TAM maps             (w(noun)=1, others=0)

Per frame (with a GT box) records two metrics:

  * mass-in-GT       — fraction of heatmap activation inside the GT box
  * thresholded-IoU  — bbox of pixels >= 0.5 × peak vs GT box

Reports per-frame win rate, mean delta, and a paired Wilcoxon p-value.

Usage
-----
    python sanity_check.py \\
        --davis_root /path/to/DAVIS2017/unsupervised \\
        --n_sequences 50

    # layout test, no GPU:
    python sanity_check.py --davis_root <...> --dry_run --n_sequences 5
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Insert local dir at front so our modules win name conflicts with siblings.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiment import (                                       # noqa: E402
    _DryRunRunner,
    _build_context_labels,
    categorize_token_with_pos,
)
from pos_tagger import build_pos_map, tagger_info              # noqa: E402


Box = Optional[Tuple[int, int, int, int]]


# ── metric helpers ────────────────────────────────────────────────────────────

def _mass_in_gt(hm: np.ndarray, gt_box: Tuple[int, int, int, int],
                frame_H: int, frame_W: int) -> float:
    total = float(hm.sum())
    if total == 0:
        return 0.0
    H_tam, W_tam = hm.shape
    x1, y1, x2, y2 = gt_box
    hx1 = max(0, int(x1 * W_tam / frame_W))
    hx2 = min(W_tam, int(x2 * W_tam / frame_W))
    hy1 = max(0, int(y1 * H_tam / frame_H))
    hy2 = min(H_tam, int(y2 * H_tam / frame_H))
    return float(hm[hy1:hy2, hx1:hx2].sum()) / total


def _heatmap_to_box(hm: np.ndarray, frame_H: int, frame_W: int,
                    threshold_frac: float = 0.5) -> Box:
    """Threshold heatmap at `threshold_frac × peak`; return enclosing box in
    original-frame pixel coordinates, or None if the heatmap is empty."""
    if hm is None or hm.size == 0:
        return None
    peak = float(hm.max())
    if peak <= 0:
        return None
    mask = hm >= peak * threshold_frac
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H_tam, W_tam = hm.shape
    x1 = int(xs.min()       * frame_W / W_tam)
    x2 = int((xs.max() + 1) * frame_W / W_tam)
    y1 = int(ys.min()       * frame_H / H_tam)
    y2 = int((ys.max() + 1) * frame_H / H_tam)
    return (x1, y1, x2, y2)


def _box_iou(b1: Box, b2: Box) -> float:
    if b1 is None or b2 is None:
        return 0.0
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    iw = max(0, min(x2, X2) - max(x1, X1))
    ih = max(0, min(y2, Y2) - max(y1, Y1))
    inter = iw * ih
    a1    = max(0, x2 - x1) * max(0, y2 - y1)
    a2    = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _aggregate_heatmap(tam_maps: List[Optional[np.ndarray]],
                       indices: List[int], t: int) -> Optional[np.ndarray]:
    """Mean of the per-frame slice for the given token indices."""
    valid = [
        tam_maps[i][t].astype(np.float32)
        for i in indices
        if (i < len(tam_maps)
            and tam_maps[i] is not None
            and tam_maps[i].ndim == 3
            and t < tam_maps[i].shape[0])
    ]
    if not valid:
        return None
    return np.mean(np.stack(valid), axis=0)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    if args.dry_run:
        print("Dry-run mode: synthetic TAM maps.")
        runner = _DryRunRunner(sample_rate=args.sample_rate)
    else:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from benchmark.qwen_vot_runner import QwenVOTRunner

        print(f"Loading model: {args.model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        model.eval()
        runner = QwenVOTRunner(
            model, processor,
            max_new_tokens=args.max_new_tokens,
            sample_rate=args.sample_rate,
            video_mode=not args.image_mode,
        )

    from benchmark.davis_vot_loader import DAVISVOTLoader
    loader = DAVISVOTLoader(
        davis_root=args.davis_root, split=args.split,
        expressions_per_seq=1,
    )

    seen, items = set(), []
    for it in loader:
        if it.seq_name not in seen:
            if len(seen) >= args.n_sequences:
                break
            seen.add(it.seq_name)
        items.append(it)
    print(f"Sanity check on {len(items)} sequences (POS tagger queued).")

    rows: List[dict] = []
    skipped_no_noun: List[str] = []

    for idx, item in enumerate(items):
        H, W = item.frame_size()
        print(f"\n[{idx+1}/{len(items)}] {item.seq_name}  "
              f"\"{item.expression[:60]}\"")
        try:
            _, _, tam_result = runner.run_with_tam(item.frames_pil, item.expression)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            continue

        gen_tokens = tam_result["gen_tokens"]
        tam_maps   = tam_result["tam_maps"]
        vision_T   = tam_result["vision_shape"][0]

        ctx_labels = _build_context_labels(gen_tokens)
        pos_map    = build_pos_map(item.expression)

        token_cats: List[str] = []
        for i, tok in enumerate(gen_tokens):
            tc  = tok.replace("▁", "").replace("Ġ", "").strip()
            cat = categorize_token_with_pos(
                i, tok, tc, item.expression, ctx_labels, pos_map
            )
            token_cats.append(cat)

        all_idx  = [i for i, tm in enumerate(tam_maps)
                    if tm is not None and tm.ndim == 3]
        noun_idx = [i for i in all_idx if token_cats[i] == "label_noun"]

        if not noun_idx:
            print(f"  [SKIP] no label_noun tokens (pos_map={pos_map})")
            skipped_no_noun.append(item.seq_name)
            continue

        n_seq_rows = 0
        for sampled_t in range(vision_T):
            orig_t = sampled_t * runner.sample_rate
            if orig_t >= len(item.gt_boxes) or item.gt_boxes[orig_t] is None:
                continue
            gt_box = item.gt_boxes[orig_t]

            H_std  = _aggregate_heatmap(tam_maps, all_idx,  sampled_t)
            H_noun = _aggregate_heatmap(tam_maps, noun_idx, sampled_t)
            if H_std is None or H_noun is None:
                continue

            mass_std  = _mass_in_gt(H_std,  gt_box, H, W)
            mass_noun = _mass_in_gt(H_noun, gt_box, H, W)

            box_std  = _heatmap_to_box(H_std,  H, W)
            box_noun = _heatmap_to_box(H_noun, H, W)
            iou_std  = _box_iou(box_std,  gt_box)
            iou_noun = _box_iou(box_noun, gt_box)

            rows.append({
                "seq":           item.seq_name,
                "expression":    item.expression,
                "frame":         orig_t,
                "n_noun_tokens": len(noun_idx),
                "n_all_tokens":  len(all_idx),
                "mass_std":      mass_std,
                "mass_noun":     mass_noun,
                "iou_std":       iou_std,
                "iou_noun":      iou_noun,
            })
            n_seq_rows += 1
        print(f"  recorded {n_seq_rows} frames "
              f"(noun_tokens={len(noun_idx)}, all_tokens={len(all_idx)})")

    if not rows:
        print("\nNo per-frame rows collected — nothing to report.")
        return

    # ── aggregate ────────────────────────────────────────────────────────────
    mass_std  = np.array([r["mass_std"]  for r in rows])
    mass_noun = np.array([r["mass_noun"] for r in rows])
    iou_std   = np.array([r["iou_std"]   for r in rows])
    iou_noun  = np.array([r["iou_noun"]  for r in rows])

    print("\n" + "=" * 64)
    print(f"Sanity check: noun-only TAM vs standard TAM")
    print(f"  {len(rows)} frames over {len({r['seq'] for r in rows})} sequences")
    if skipped_no_noun:
        print(f"  skipped {len(skipped_no_noun)} sequences with no noun tokens: "
              f"{skipped_no_noun[:6]}{'...' if len(skipped_no_noun) > 6 else ''}")
    print(f"  POS tagger: {tagger_info()}")
    print("=" * 64)

    print("\n— Mass-in-GT —")
    print(f"  standard:    mean = {mass_std.mean():.4f}   median = {np.median(mass_std):.4f}")
    print(f"  noun-only:   mean = {mass_noun.mean():.4f}   median = {np.median(mass_noun):.4f}")
    print(f"  delta:       mean = {(mass_noun - mass_std).mean():+.4f}   "
          f"median = {np.median(mass_noun - mass_std):+.4f}")
    print(f"  win rate:    noun > std in {(mass_noun > mass_std).mean():.1%} of frames")
    print(f"  tie rate:    noun = std in {(mass_noun == mass_std).mean():.1%} of frames")

    print("\n— Thresholded IoU (heatmap >= 0.5 × peak  →  bbox) —")
    print(f"  standard:    mean = {iou_std.mean():.4f}   median = {np.median(iou_std):.4f}")
    print(f"  noun-only:   mean = {iou_noun.mean():.4f}   median = {np.median(iou_noun):.4f}")
    print(f"  delta:       mean = {(iou_noun - iou_std).mean():+.4f}   "
          f"median = {np.median(iou_noun - iou_std):+.4f}")
    print(f"  win rate:    noun > std in {(iou_noun > iou_std).mean():.1%} of frames")
    print(f"  tie rate:    noun = std in {(iou_noun == iou_std).mean():.1%} of frames")

    try:
        from scipy.stats import wilcoxon
        w_mass = wilcoxon(mass_noun, mass_std, alternative="greater",
                          zero_method="wilcox")
        w_iou  = wilcoxon(iou_noun,  iou_std,  alternative="greater",
                          zero_method="wilcox")
        print("\n— Paired Wilcoxon (H1: noun > std) —")
        print(f"  mass-in-GT:  W = {w_mass.statistic:>10.0f}   p = {w_mass.pvalue:.3e}")
        print(f"  iou:         W = {w_iou.statistic:>10.0f}   p = {w_iou.pvalue:.3e}")
    except ImportError:
        print("\n[scipy not available — skipping Wilcoxon test]")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sanity_check_results.json"
    with open(out, "w") as f:
        json.dump({
            "n_rows":           len(rows),
            "n_sequences":      len({r["seq"] for r in rows}),
            "skipped_no_noun":  skipped_no_noun,
            "rows":             rows,
        }, f, indent=2, default=str)
    print(f"\nRaw → {out}")


def parse_args():
    p = argparse.ArgumentParser("Noun-only TAM sanity check")
    p.add_argument("--davis_root",     required=True)
    p.add_argument("--model_id",       default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--save_dir",       default="results/sanity_check")
    p.add_argument("--split",          default="valid", choices=["valid", "train"])
    p.add_argument("--n_sequences",    type=int, default=50)
    p.add_argument("--sample_rate",    type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--image_mode",     action="store_true")
    p.add_argument("--dry_run",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
