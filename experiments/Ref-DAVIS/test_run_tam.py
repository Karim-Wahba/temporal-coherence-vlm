"""
test_run_tam.py
---------------
Validates the TAM diagnostic pipeline — both the analysis/visualisation logic
(no model required) and the full end-to-end ``run_tam_diagnostics`` path.

Two modes
---------
1. Dry-run (default, no GPU needed):
   Builds a synthetic ``tam_result`` dict with random numpy attention maps and
   runs every downstream function: ``pearson_temporal_coherence``,
   ``plot_tam_summary_grid``, ``plot_tam_selected_words``,
   ``save_tam_word_strips``, ``plot_tam_coherence_summary``.
   Asserts all expected output files are created and the coherence score is in
   [0, 1].

2. End-to-end (requires GPU + DAVIS dataset):
   Loads the real Qwen3-VL model, creates a ``RefDAVISItem`` stub from real
   DAVIS frames, calls ``run_tam_diagnostics``, and asserts that the complete
   set of TAM output files is written and the return dict is well-formed.

Usage
-----
    # Dry-run only (fast, no model):
    python test_run_tam.py

    # Full end-to-end (slow, needs GPU + DAVIS):
    python test_run_tam.py \\
        --full \\
        --davis_root /home/geiger/gwb913/git/davis/DAVIS2017/unsupervised \\
        --model_id Qwen/Qwen3-VL-8B-Instruct \\
        --sequence blackswan \\
        --expression "the black swan" \\
        --max_frames 6
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "benchmark"))
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
sys.path.insert(0, os.path.join(HERE, "visualization"))
sys.path.insert(0, str(Path(HERE).parents[1] / "submodules" / "TAM"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures: List[str] = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"  [{FAIL}] {msg}")
        _failures.append(msg)


def check_file(path: str):
    exists = os.path.isfile(path)
    check(os.path.basename(path), exists,
          detail=f"not found at {path}" if not exists else "")


def check_dir_nonempty(path: str, ext: str = ".jpg"):
    exists = os.path.isdir(path)
    if not exists:
        check(f"dir {os.path.basename(path)}/", False, detail=f"directory not found: {path}")
        return
    files = [f for f in os.listdir(path) if f.endswith(ext)]
    check(f"dir {os.path.basename(path)}/ non-empty ({ext})",
          len(files) > 0,
          detail=f"found {len(files)} {ext} files" if not files else "")


# ─────────────────────────────────────────────────────────────────────────────
# Build a synthetic tam_result (no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_tam_result(T: int = 5, H: int = 8, W: int = 14,
                               n_tokens: int = 20) -> dict:
    """
    Build a fake ``tam_result`` that mimics the structure returned by
    ``TAMRunner.run()``.

    Each tam_map entry is a (T, H, W) float32 array with values in [0, 255].
    gen_tokens are simple word-like strings with realistic spacing/subword
    patterns so the word-grouping logic has something to work with.
    """
    rng = np.random.default_rng(42)

    # Realistic token stream: mix of leading-space words and sub-word pieces
    raw_tokens = [
        " The", " black", " swan", " swim", "ming", " slow", "ly",
        " across", " the", " lake", ".", " It", " turns", " its",
        " head", " right", " then", " left", ".", "<|endoftext|>",
    ]
    # Pad / trim to requested length
    while len(raw_tokens) < n_tokens:
        raw_tokens.append(" object")
    raw_tokens = raw_tokens[:n_tokens]

    gen_ids = list(range(n_tokens))

    tam_maps = []
    for _ in range(n_tokens):
        # Most tokens get a (T, H, W) map; a few are None
        if rng.random() < 0.85:
            m = rng.uniform(0, 255, (T, H, W)).astype(np.float32)
            tam_maps.append(m)
        else:
            tam_maps.append(None)

    # frame_mass: fraction of attention mass per frame per token
    frame_mass = np.zeros((n_tokens, T), dtype=np.float32)
    for i, m in enumerate(tam_maps):
        if m is not None:
            total = m.sum()
            if total > 0:
                for t in range(T):
                    frame_mass[i, t] = m[t].sum() / total

    # Build PIL frames (tiny solid-colour images)
    colours = [(100 + t * 20, 80, 120) for t in range(T)]
    frames_pil = [Image.new("RGB", (W * 4, H * 4), colour) for colour in colours]

    return {
        "gen_text": "The black swan swimming slowly across the lake.",
        "gen_tokens": raw_tokens,
        "gen_ids": gen_ids,
        "tam_maps": tam_maps,
        "frame_mass": frame_mass,
        "vision_shape": (T, H, W),
        "frames_pil": frames_pil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run: test analysis + visualisation without a model
# ─────────────────────────────────────────────────────────────────────────────

def run_dry_run(save_dir: str):
    print("\n" + "=" * 60)
    print("DRY-RUN: synthetic tam_result — no model needed")
    print("=" * 60)

    from diagnostics.tam_analyzer import (
        pearson_temporal_coherence,
        pearson_coherence_in_gt_region,
        attention_mass_in_gt,
        temporal_collapse,
        attention_drift,
    )
    from visualization.visualizer import (
        plot_tam_summary_grid,
        plot_tam_selected_words,
        save_tam_word_strips,
        plot_tam_coherence_summary,
        plot_frame_mass_heatmap,
        plot_gam_over_time,
    )

    os.makedirs(save_dir, exist_ok=True)
    tam_result = _make_synthetic_tam_result()
    frames_pil = tam_result["frames_pil"]
    T = tam_result["vision_shape"][0]

    # ── 1. pearson_temporal_coherence ─────────────────────────────────────────
    print("\n[1] pearson_temporal_coherence")
    coherence_result = pearson_temporal_coherence(tam_result)

    check("returns dict", isinstance(coherence_result, dict))
    check("word_groups is list", isinstance(coherence_result.get("word_groups"), list))
    check("word_groups non-empty", len(coherence_result.get("word_groups", [])) > 0)
    check("word_maps same length as word_groups",
          len(coherence_result.get("word_maps", [])) ==
          len(coherence_result.get("word_groups", [])))
    check("target_word is str", isinstance(coherence_result.get("target_word"), str))

    cs = coherence_result.get("coherence_score")
    check("coherence_score not None", cs is not None)
    if cs is not None:
        check("coherence_score in [0, 1]", 0.0 <= cs <= 1.0,
              detail=f"got {cs:.4f}")

    pp = coherence_result.get("per_pair_corr", [])
    check(f"per_pair_corr has T-1={T-1} entries", len(pp) == T - 1,
          detail=f"got {len(pp)}")
    if pp:
        check("all per_pair values in [0, 1]",
              all(0.0 <= v <= 1.0 for v in pp),
              detail=str(pp))

    # ── 2. temporal_collapse (existing, verify still works) ───────────────────
    print("\n[2] temporal_collapse")
    collapse = temporal_collapse(tam_result)
    check("collapse_rate in [0, 1]",
          0.0 <= collapse.get("collapse_rate", -1) <= 1.0)
    check("mean_entropy in [0, 1]",
          0.0 <= collapse.get("mean_entropy", -1) <= 1.0)

    # ── 3. attention_drift (no GT masks) ──────────────────────────────────────
    print("\n[3] attention_drift (no GT)")
    drift = attention_drift(tam_result)
    check("per_frame_centroid shape (T, 2)",
          drift.get("per_frame_centroid", np.array([])).shape == (T, 2))
    check("gt_centroids is None (no GT)", drift.get("gt_centroids") is None)

    # ── 4. plot_frame_mass_heatmap ────────────────────────────────────────────
    print("\n[4] plot_frame_mass_heatmap")
    p = os.path.join(save_dir, "dry_frame_mass.png")
    plot_frame_mass_heatmap(tam_result, p, seq_name="dry", expression="test")
    check_file(p)

    # ── 5. pearson_coherence_in_gt_region ────────────────────────────────────
    print("\n[5] pearson_coherence_in_gt_region")
    # Synthetic GT masks: foreground square in the centre of the TAM map
    H_tam, W_tam = tam_result["vision_shape"][1], tam_result["vision_shape"][2]
    gt_mask_tam = np.zeros((H_tam, W_tam), dtype=np.uint8)
    gt_mask_tam[H_tam // 4: 3 * H_tam // 4, W_tam // 4: 3 * W_tam // 4] = 1
    # Scale up to a fake "frame" size so the resize path is exercised
    frame_h, frame_w = H_tam * 4, W_tam * 4
    gt_mask_frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    gt_mask_frame[frame_h // 4: 3 * frame_h // 4,
                  frame_w // 4: 3 * frame_w // 4] = 1
    gt_masks_frames = [gt_mask_frame.copy() for _ in range(T)]

    gt_coherence_result = pearson_coherence_in_gt_region(
        coherence_result=coherence_result,
        gt_masks=gt_masks_frames,
        frame_size=(frame_h, frame_w),
        vision_shape=tam_result["vision_shape"],
    )
    check("returns dict", isinstance(gt_coherence_result, dict))
    check("word_scores_in_gt same length as word_groups",
          len(gt_coherence_result.get("word_scores_in_gt", [])) ==
          len(coherence_result.get("word_groups", [])))
    check("gt_masks_tam length == T",
          len(gt_coherence_result.get("gt_masks_tam", [])) == T)

    cs_gt = gt_coherence_result.get("target_coherence_in_gt")
    check("target_coherence_in_gt not None", cs_gt is not None)
    if cs_gt is not None:
        check("target_coherence_in_gt in [0, 1]", 0.0 <= cs_gt <= 1.0,
              detail=f"got {cs_gt:.4f}")

    pp_gt = gt_coherence_result.get("target_per_pair_in_gt", [])
    check(f"target_per_pair_in_gt has T-1={T-1} entries", len(pp_gt) == T - 1,
          detail=f"got {len(pp_gt)}")
    if pp_gt:
        check("all target_per_pair_in_gt values in [0, 1]",
              all(0.0 <= v <= 1.0 for v in pp_gt),
              detail=str(pp_gt))

    # ── 5d. attention_mass_in_gt (GAM) ───────────────────────────────────────
    print("\n[5d] attention_mass_in_gt (GAM)")
    gam_result = attention_mass_in_gt(
        coherence_result=coherence_result,
        gt_masks=gt_masks_frames,
        frame_size=(frame_h, frame_w),
        vision_shape=tam_result["vision_shape"],
    )
    check("returns dict", isinstance(gam_result, dict))
    check("per_frame_gam length == T",
          len(gam_result.get("per_frame_gam", [])) == T)
    check("target_per_frame_gam length == T",
          len(gam_result.get("target_per_frame_gam", [])) == T)
    check("mean_gam in [0, 1]",
          0.0 <= gam_result.get("mean_gam", -1) <= 1.0,
          detail=f"got {gam_result.get('mean_gam')}")
    check("lost_track_rate in [0, 1]",
          0.0 <= gam_result.get("lost_track_rate", -1) <= 1.0)
    check("gam_decay is float", isinstance(gam_result.get("gam_decay"), float))
    check("word_per_frame_gam is dict",
          isinstance(gam_result.get("word_per_frame_gam"), dict))

    pfg = gam_result.get("per_frame_gam", np.array([]))
    if len(pfg) > 0:
        check("all per_frame_gam in [0, 1]",
              bool(np.all((pfg >= 0) & (pfg <= 1))),
              detail=f"range=[{pfg.min():.3f}, {pfg.max():.3f}]")

    # With a centred GT mask, GAM should be > 0 (attention overlaps the region)
    check("mean_gam > 0 with centred GT mask",
          gam_result.get("mean_gam", 0.0) > 0.0)

    print("\n[5e] plot_gam_over_time")
    p = os.path.join(save_dir, "dry_gam.png")
    plot_gam_over_time(
        gam_result, p,
        seq_name="dry", expression="test",
        target_word=coherence_result.get("target_word", ""),
    )
    check_file(p)

    # ── 5b. plot_tam_coherence_summary (without GT) ───────────────────────────
    print("\n[5b] plot_tam_coherence_summary (without GT — 2-panel)")
    p = os.path.join(save_dir, "dry_coherence_no_gt.png")
    plot_tam_coherence_summary(coherence_result, p, seq_name="dry", expression="test")
    check_file(p)

    # ── 5c. plot_tam_coherence_summary (with GT) ──────────────────────────────
    print("\n[5c] plot_tam_coherence_summary (with GT — 3-panel)")
    p = os.path.join(save_dir, "dry_coherence_with_gt.png")
    plot_tam_coherence_summary(coherence_result, p, seq_name="dry", expression="test",
                               gt_coherence_result=gt_coherence_result)
    check_file(p)

    # ── 6. plot_tam_summary_grid ──────────────────────────────────────────────
    print("\n[6] plot_tam_summary_grid")
    p = os.path.join(save_dir, "dry_summary_grid.png")
    plot_tam_summary_grid(coherence_result, frames_pil, p,
                          seq_name="dry", expression="test")
    check_file(p)

    # ── 7. plot_tam_selected_words ────────────────────────────────────────────
    print("\n[7] plot_tam_selected_words")
    p = os.path.join(save_dir, "dry_selected_words.png")
    plot_tam_selected_words(coherence_result, frames_pil, p, seq_name="dry")
    check_file(p)

    # ── 8. save_tam_word_strips ───────────────────────────────────────────────
    print("\n[8] save_tam_word_strips")
    words_dir = os.path.join(save_dir, "dry_words")
    save_tam_word_strips(coherence_result, frames_pil, words_dir)
    check_dir_nonempty(words_dir, ".jpg")

    print(f"\nDry-run outputs written to: {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: test run_tam_diagnostics with real model + DAVIS
# ─────────────────────────────────────────────────────────────────────────────

class _FakeItem:
    """Minimal stub matching the interface of RefDAVISItem."""

    def __init__(self, seq_name, exp_id, expression, frames_pil, masks):
        self.seq_name = seq_name
        self.exp_id = exp_id
        self.expression = expression
        self._frames_pil = frames_pil
        self._masks = masks

    @property
    def frames_pil(self):
        return self._frames_pil

    @property
    def masks(self):
        return self._masks

    @property
    def num_frames(self):
        return len(self._frames_pil)

    def frame_size(self):
        h, w = np.array(self._frames_pil[0]).shape[:2]
        return h, w


def run_full(args, save_dir: str):
    print("\n" + "=" * 60)
    print("END-TO-END: real model + DAVIS frames")
    print("=" * 60)

    # ── Load DAVIS frames ─────────────────────────────────────────────────────
    frames_dir = Path(args.davis_root) / "JPEGImages" / "480p" / args.sequence
    frame_paths = sorted(frames_dir.glob("*.jpg"))[: args.max_frames]
    check("DAVIS frames found", len(frame_paths) > 0,
          detail=f"looked in {frames_dir}")
    if not frame_paths:
        return

    frames_pil = [Image.open(p).convert("RGB") for p in frame_paths]
    H, W = np.array(frames_pil[0]).shape[:2]
    print(f"  Loaded {len(frames_pil)} frames ({H}×{W}) from {frames_dir}")

    # Build simple square masks (centre 20% of the frame)
    cy, cx = H // 2, W // 2
    r = min(H, W) // 5
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[cy - r: cy + r, cx - r: cx + r] = 1
    masks = [mask.copy() for _ in frames_pil]

    item = _FakeItem(
        seq_name=args.sequence,
        exp_id="0",
        expression=args.expression,
        frames_pil=frames_pil,
        masks=masks,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    print("Model loaded.")

    # ── Call run_tam_diagnostics ──────────────────────────────────────────────
    # Import benchmark.py directly by file path so we don't accidentally
    # pick up the benchmark/ package directory instead.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark_script", os.path.join(HERE, "benchmark.py")
    )
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nRunning run_tam_diagnostics → {save_dir}/")
    result = bm.run_tam_diagnostics(
        model, processor, item, pred_masks=masks, save_dir_tam=save_dir
    )

    # ── [E1] Basic return dict ────────────────────────────────────────────────
    print("\n[E1] return dict keys")
    check("returned a dict", isinstance(result, dict))
    for key in ("tam_result", "drift", "collapse",
                "coherence", "coherence_score",
                "gt_coherence", "coherence_score_in_gt",
                "gam", "mean_gam", "gam_decay", "lost_track_rate_gam"):
        check(f"key '{key}' present", key in result)

    # ── [E2] Full-frame Pearson coherence ─────────────────────────────────────
    print("\n[E2] full-frame Pearson coherence")
    coherence_result = result.get("coherence", {})
    check("coherence dict non-empty", bool(coherence_result),
          detail="run_tam_diagnostics returned empty coherence — check for exception in benchmark output above")
    if coherence_result:
        wg = coherence_result.get("word_groups", [])
        wm = coherence_result.get("word_maps", [])
        check("word_groups non-empty", len(wg) > 0)
        check("word_maps same length as word_groups", len(wm) == len(wg))
        print(f"      word count : {len(wg)}")
        print(f"      gen text   : {coherence_result.get('gen_text', '')[:80]}")
        print(f"      target noun: \"{coherence_result.get('target_word')}\"")

    cs = result.get("coherence_score")
    check("coherence_score not None", cs is not None,
          detail="returned None — NLTK noun-finding or TAM map may have failed")
    if cs is not None:
        check("coherence_score in [0, 1]", 0.0 <= cs <= 1.0, detail=f"got {cs:.4f}")
        print(f"      coherence_score = {cs:.4f}")

    # ── [E3] GT-region Pearson coherence ──────────────────────────────────────
    print("\n[E3] GT-region Pearson coherence")
    gt_coh = result.get("gt_coherence", {})
    check("gt_coherence dict non-empty", bool(gt_coh),
          detail="empty — pearson_coherence_in_gt_region may have failed or item.masks was empty")
    if gt_coh and coherence_result:
        wg = coherence_result.get("word_groups", [])
        check("word_scores_in_gt same length as word_groups",
              len(gt_coh.get("word_scores_in_gt", [])) == len(wg),
              detail=f"got {len(gt_coh.get('word_scores_in_gt', []))} vs {len(wg)}")
        check("gt_masks_tam length == T",
              len(gt_coh.get("gt_masks_tam", [])) == result["tam_result"]["vision_shape"][0])

    cs_gt = result.get("coherence_score_in_gt")
    check("coherence_score_in_gt not None", cs_gt is not None,
          detail="returned None — GT masks may be empty or computation failed")
    if cs_gt is not None:
        check("coherence_score_in_gt in [0, 1]", 0.0 <= cs_gt <= 1.0,
              detail=f"got {cs_gt:.4f}")
        print(f"      coherence_score_in_gt = {cs_gt:.4f}")
    if cs is not None and cs_gt is not None:
        print(f"      full={cs:.4f}  in-GT={cs_gt:.4f}  "
              f"delta={cs_gt - cs:+.4f}")

    # ── [E4] GT Attention Mass (GAM) ──────────────────────────────────────────
    print("\n[E4] GT Attention Mass (GAM)")
    gam = result.get("gam", {})
    check("gam dict non-empty", bool(gam),
          detail="empty — attention_mass_in_gt may have failed or item.masks was empty")
    if gam:
        T_vision = result["tam_result"]["vision_shape"][0]
        pfg = np.asarray(gam.get("per_frame_gam", []))
        tpfg = np.asarray(gam.get("target_per_frame_gam", []))
        check("per_frame_gam length == T", len(pfg) == T_vision,
              detail=f"got {len(pfg)} vs T={T_vision}")
        check("target_per_frame_gam length == T", len(tpfg) == T_vision)
        check("all per_frame_gam in [0, 1]",
              bool(np.all((pfg >= 0) & (pfg <= 1))) if len(pfg) else True,
              detail=f"range=[{pfg.min():.3f}, {pfg.max():.3f}]" if len(pfg) else "")
        check("word_per_frame_gam is non-empty dict",
              isinstance(gam.get("word_per_frame_gam"), dict)
              and len(gam["word_per_frame_gam"]) > 0)

    mean_gam = result.get("mean_gam")
    check("mean_gam not None", mean_gam is not None,
          detail="returned None — GAM computation failed")
    if mean_gam is not None:
        check("mean_gam in [0, 1]", 0.0 <= mean_gam <= 1.0, detail=f"got {mean_gam:.4f}")
        print(f"      mean_gam           = {mean_gam:.4f}")

    gam_decay = result.get("gam_decay")
    check("gam_decay is float", isinstance(gam_decay, float),
          detail=f"got {type(gam_decay)}")
    if gam_decay is not None:
        print(f"      gam_decay          = {gam_decay:+.5f}/frame"
              f"  ({'losing track' if gam_decay < 0 else 'stable/improving'})")

    ltr = result.get("lost_track_rate_gam")
    check("lost_track_rate_gam in [0, 1]",
          ltr is not None and 0.0 <= ltr <= 1.0,
          detail=f"got {ltr}")
    if ltr is not None:
        print(f"      lost_track_rate    = {ltr:.1%}")
        # With a centred square mask and real attention maps, GAM > 0 is a
        # basic sanity check — but lost_track_rate may legitimately be high
        # if the model attends to background. Just print, don't hard-fail.

    if gam:
        per_word = gam.get("word_per_frame_gam", {})
        print(f"      words tracked      = {len(per_word)}")
        if per_word:
            worst = min(per_word, key=lambda w: np.mean(per_word[w]))
            best  = max(per_word, key=lambda w: np.mean(per_word[w]))
            print(f"      most on-target     : \"{best}\"  "
                  f"avg={np.mean(per_word[best]):.3f}")
            print(f"      least on-target    : \"{worst}\"  "
                  f"avg={np.mean(per_word[worst]):.3f}")

    # ── [E5] Output files ─────────────────────────────────────────────────────
    prefix = f"{item.seq_name}_exp{item.exp_id}"
    print("\n[E5] output files")
    for fname in (
        f"{prefix}_frame_mass.png",
        f"{prefix}_centroid.png",
        f"{prefix}_coherence.png",
        f"{prefix}_gam.png",
        f"{prefix}_tam_summary_grid.png",
        f"{prefix}_tam_selected_words.png",
    ):
        check_file(os.path.join(save_dir, fname))

    words_dir = os.path.join(save_dir, f"{prefix}_words")
    check_dir_nonempty(words_dir, ".jpg")

    print(f"\nEnd-to-end outputs written to: {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("Test run_tam_diagnostics")
    p.add_argument("--full", action="store_true",
                   help="Also run the end-to-end test with the real model")
    p.add_argument("--save_dir", default="results/test_run_tam",
                   help="Where to write test outputs")
    # End-to-end options (only used with --full)
    p.add_argument("--davis_root",
                   default="/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--sequence", default="blackswan")
    p.add_argument("--expression", default="the black swan")
    p.add_argument("--max_frames", type=int, default=6,
                   help="Number of DAVIS frames to load (keep small for speed)")
    args = p.parse_args()

    dry_dir = os.path.join(args.save_dir, "dry_run")
    e2e_dir = os.path.join(args.save_dir, "e2e")

    # Always run the dry-run
    try:
        run_dry_run(dry_dir)
    except Exception:
        traceback.print_exc()
        _failures.append("dry_run raised an exception")

    # Optionally run end-to-end
    if args.full:
        try:
            run_full(args, e2e_dir)
        except Exception:
            traceback.print_exc()
            _failures.append("end-to-end run raised an exception")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if _failures:
        print(f"\033[31mFAILED — {len(_failures)} check(s):\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[32mAll checks passed.\033[0m")
        if not args.full:
            print("  (Run with --full to also test the end-to-end model path)")


if __name__ == "__main__":
    main()
