"""
Usage:
  python tam_qwen3vl_video.py \
      --frames_dir /path/to/frames/ \
      --prompt "Describe what is happening in this video." \
      --save_dir results/tam_video_vis

  # Control FPS subsampling (default: sample at 2 fps from a 25 fps video)
  python tam_qwen3vl_video.py \
      --frames_dir /path/to/frames/ \
      --prompt "Describe what is happening in this video." \
      --fps 2.0 \
      --video_fps 24.0 \
      --save_dir results/tam_video_vis

Notes:
  - frames_dir should contain sequentially named .jpg or .png files.
  - Frames are subsampled via FPS to avoid degraded results from too many frames.
  - Each sampled frame is passed twice to account for Qwen's temporal merge factor of 2.
  - Outputs include per-token TAM maps, per-word multi-frame overlays, and a
    summary grid (words x frames) showing activation on every frame.
"""

import argparse, glob, os, sys, shutil
import torch
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

# --- MONKEY PATCH MATPLOTLIB ---
# Intercept every time any module tries to draw text and force LaTeX off
original_text = plt.text
original_set_title = plt.Axes.set_title

def safe_text(*args, **kwargs):
    kwargs['usetex'] = False
    return original_text(*args, **kwargs)

def safe_set_title(self, *args, **kwargs):
    kwargs['usetex'] = False
    return original_set_title(self, *args, **kwargs)

plt.text = safe_text
plt.Axes.set_title = safe_set_title
# -------------------------------

import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.video_utils import VideoMetadata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../submodules/TAM"))

from tam import TAM
from qwen_utils import process_vision_info


def main():
    p = argparse.ArgumentParser("TAM for Qwen3-VL-8B — Video (frame sequence)")
    p.add_argument("--frames_dir", required=True,
                   help="Directory containing sequentially named frame images (.jpg/.png)")
    p.add_argument("--prompt", default="Describe what is happening in this video.")
    p.add_argument("--save_dir", default="results/tam_video_vis")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Target sampling FPS — Qwen default is 2.0")
    p.add_argument("--video_fps", type=float, default=24.0,
                   help="Original video FPS used to compute subsampling stride")
    p.add_argument("--repeat_frames", type=int, default=2,
                   help="Times each frame is repeated for model input; "
                        "matches Qwen FRAME_FACTOR=2 so each frame gets its own temporal token")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Step 1: Load and subsample frames ────────────────────────────────────
    all_paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.jpg")) +
        glob.glob(os.path.join(args.frames_dir, "*.png"))
    )
    if not all_paths:
        raise FileNotFoundError(f"No .jpg/.png files found in {args.frames_dir}")

    # FPS-based stride subsampling (Qwen's default FPS=2.0, FPS_MIN_FRAMES=4, FPS_MAX_FRAMES=768)
    stride = max(1, round(args.video_fps / args.fps))
    frame_paths = all_paths[::stride]
    frame_paths = frame_paths[:768]
    if len(frame_paths) < 4:
        frame_paths = all_paths[:max(4, len(all_paths))]

    frames_pil = [Image.open(fp).convert("RGB") for fp in frame_paths]
    N = len(frames_pil)
    print(f"Frames in dir: {len(all_paths)}, stride: {stride}, sampled: {N}")
    print(f"Frame paths: {[os.path.basename(fp) for fp in frame_paths]}")

    # Duplicate each frame to produce one temporal token per frame
    frames_for_model = [f for f in frames_pil for _ in range(args.repeat_frames)]
    print(f"Frames passed to model: {len(frames_for_model)} "
          f"({N} unique × {args.repeat_frames})")

    # ── Step 2: Load model ────────────────────────────────────────────────────
    print(f"\nLoading {args.model_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Prepare inputs ────────────────────────────────────────────────────────
    # Pass PIL images as a list; process_vision_info (from qwen_utils bundled
    # with TAM) keeps ALL list-of-images frames without fps-based resampling.
    # Using the two-step approach (tokenize text separately from vision) avoids
    # the processor's internal fps resampling that collapsed frames to T=2.
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames_for_model},
        {"type": "text", "text": args.prompt},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Provide video metadata so the processor knows the source fps and doesn't
    # default to 24fps (which collapses 14 frames down to min_frames=4 → T=2).
    # do_sample_frames=False prevents the processor's internal resampling since
    # we already did our own fps-based subsampling above.
    n_model_frames = len(frames_for_model)
    video_meta = VideoMetadata(
        total_num_frames=n_model_frames,
        fps=args.fps * args.repeat_frames,  # effective fps of the repeated stream
        frames_indices=list(range(n_model_frames)),
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
        video_metadata=[video_meta],
    ).to(model.device)

    # ── Step 3: Generate with hidden states ───────────────────────────────────
    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences

    # Compute logits from last hidden states via lm_head
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # Print generated text
    input_len = inputs.input_ids.shape[1]
    gen_text = processor.tokenizer.decode(
        generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Output: {gen_text[:200]}")

    # ── Step 4: Prepare special_ids and vision_shape ──────────────────────────
    token_ids = inputs.input_ids[0].cpu().tolist()
    token_strs = [processor.tokenizer.decode([t], skip_special_tokens=False)
                  for t in token_ids]

    print("\nToken sequence (first 20 + last 10):")
    for i in range(min(20, len(token_ids))):
        print(f"  [{i}] id={token_ids[i]}  '{token_strs[i]}'")
    print("  ...")
    for i in range(max(0, len(token_ids) - 10), len(token_ids)):
        print(f"  [{i}] id={token_ids[i]}  '{token_strs[i]}'")

    vision_start_id = None
    vision_end_id = None
    video_pad_id = None
    video_pad_count = 0

    for tid, ts in zip(token_ids, token_strs):
        if "<|vision_start|>" in ts:
            vision_start_id = tid
        elif "<|vision_end|>" in ts:
            vision_end_id = tid
        elif "<|video_pad|>" in ts or "<|image_pad|>" in ts:
            if video_pad_id is None:
                video_pad_id = tid
            video_pad_count += 1

    # Count how many separate vision blocks there are (Qwen3-VL uses one per temporal group)
    vision_start_count = sum(1 for t in token_ids if t == vision_start_id)
    print(f"\nvision_start_id: {vision_start_id}  (×{vision_start_count})")
    print(f"vision_end_id:   {vision_end_id}")
    print(f"video_pad_id:    {video_pad_id}")
    print(f"video pad tokens: {video_pad_count}")

    im_end_id = None
    im_start_id = None
    for tid, ts in zip(token_ids, token_strs):
        if "<|im_end|>" in ts:
            im_end_id = tid
        if "<|im_start|>" in ts:
            im_start_id = tid

    answer_start_positions = []
    for i in range(len(token_ids) - 1):
        if token_ids[i] == im_start_id:
            answer_start_positions.append(i)

    if answer_start_positions:
        assistant_header_pos = answer_start_positions[-1]
    else:
        assistant_header_pos = len(token_ids) - 1

    answer_boundary = token_ids[assistant_header_pos:input_len]
    print(f"Answer boundary tokens: {answer_boundary}")
    print(f"  = {[processor.tokenizer.decode([t]) for t in answer_boundary]}")

    # Qwen3-VL creates SEPARATE <|vision_start|>...<|vision_end|> blocks per
    # temporal group.  TAM's id2idx for INT ids always returns the *first*
    # occurrence, so the old [vision_start, vision_end] pair only captured the
    # first block.
    #
    # Fix:
    #   img_id = [video_pad_id]   → single-ID mode gathers ALL pad tokens
    #   prompt_id[0] = [vision_end_id]  → LIST mode finds the *last* vision_end
    special_ids = {
        'img_id': [video_pad_id],
        'prompt_id': [[vision_end_id], answer_boundary],
        'answer_id': [answer_boundary, -1],
    }
    print(f"\nspecial_ids: {special_ids}")

    # Video shape: (T, H//2, W//2) from video_grid_thw
    vision_shape = (
        inputs['video_grid_thw'][0, 0].item(),
        inputs['video_grid_thw'][0, 1].item() // 2,
        inputs['video_grid_thw'][0, 2].item() // 2,
    )
    T = vision_shape[0]
    print(f"vision_shape (T, H, W): {vision_shape}")

    # vis_inputs: list containing one list of T PIL frames (no duplication)
    # tam.py video branch: cv_img = [cv2.cvtColor(np.array(_), ...) for _ in vision_input[0]]
    vis_inputs = [frames_pil[:T]]

    # ── Step 5: Call TAM for each generation round ────────────────────────────
    gen_ids_list = generated_ids[0][input_len:].cpu().tolist()
    gen_words = [processor.tokenizer.decode([t], skip_special_tokens=False)
                 for t in gen_ids_list]

    print(f"\nRunning TAM for {len(logits)} rounds...")
    print(f"\n  Token-to-file mapping:")
    print(f"  {'File':>8}  {'Token ID':>8}  {'Word'}")
    print(f"  {'─' * 40}")
    for i, (tid, word) in enumerate(zip(gen_ids_list, gen_words)):
        clean = word.replace('\n', '\\n')
        print(f"  {i:>8}  {tid:>8}  '{clean}'")

    with open(os.path.join(args.save_dir, "token_map.txt"), "w") as f:
        f.write(f"Generated text: {gen_text}\n\n")
        f.write(f"{'Index':>6}  {'Token ID':>8}  {'Word'}\n")
        f.write(f"{'─' * 40}\n")
        for i, (tid, word) in enumerate(zip(gen_ids_list, gen_words)):
            f.write(f"{i:>6}  {tid:>8}  {repr(word)}\n")

    raw_map_records = []
    all_img_maps = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(args.save_dir, f"{i}.jpg"),
            i,
            raw_map_records,
            False,
        )
        all_img_maps.append(img_map)

    # ── Rename files to include the word ──────────────────────────────────────
    named_dir = os.path.join(args.save_dir, "named")
    os.makedirs(named_dir, exist_ok=True)

    for i in range(len(gen_words)):
        src = os.path.join(args.save_dir, f"{i}.jpg")
        if os.path.exists(src):
            word = gen_words[i].strip().replace("/", "_").replace(" ", "_")
            word = word.replace("|", "").replace("<", "").replace(">", "")
            word = word.replace("\n", "NL")
            if not word:
                word = "empty"
            dst = os.path.join(named_dir, f"{i:03d}_{word[:20]}.jpg")
            shutil.copy2(src, dst)

    # ══════════════════════════════════════════════════════════════════════════
    # POST-PROCESSING: Group subword tokens into words, per-frame overlays
    # ══════════════════════════════════════════════════════════════════════════

    # ── Helper functions ──────────────────────────────────────────────────────

    def enhance_contrast(heatmap_uint8):
        """Enhance heatmap contrast with percentile stretching."""
        m = heatmap_uint8.astype(np.float32)
        p5 = np.percentile(m, 5)
        p95 = np.percentile(m, 95)
        if p95 > p5:
            m = (m - p5) / (p95 - p5)
            m = np.clip(m, 0, 1) * 255
        return m.astype(np.uint8)

    def overlay_heatmap(frame_arr, heatmap_2d_uint8, alpha=0.5):
        """Overlay JET heatmap on a single frame (H,W,C) array.

        Uses the raw TAM scores directly (no extra contrast stretching)
        so the overlays match the raw per-token .jpg heatmaps produced
        by TAM's multimodal_process.
        """
        h, w = frame_arr.shape[:2]
        # Resize the smooth grayscale FIRST, then apply colormap.
        # This produces smooth gradients instead of blocky JET patches.
        smooth = cv2.resize(heatmap_2d_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(smooth, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return (frame_arr * (1 - alpha) + colored * alpha).astype(np.uint8)

    def load_map_3d(token_idx):
        """Return (T,H,W) float32 map for token_idx, or None."""
        if token_idx < len(all_img_maps) and all_img_maps[token_idx] is not None:
            m = all_img_maps[token_idx]
            if isinstance(m, np.ndarray) and m.ndim == 3:
                return m.astype(np.float32)
        return None

    # ── Group subword tokens into whole words ─────────────────────────────────
    word_groups = []
    current_word = ""
    current_indices = []

    for i, raw_word in enumerate(gen_words):
        if "<|" in raw_word:
            continue

        stripped = raw_word
        is_new_word = (
            len(current_indices) == 0 or
            stripped.startswith(" ") or
            stripped.startswith("\n") or
            (len(stripped.strip()) == 1 and not stripped.strip().isalnum())
        )

        if is_new_word and current_indices:
            word_groups.append({"word": current_word.strip(), "token_indices": current_indices})
            current_word = stripped
            current_indices = [i]
        else:
            current_word += stripped
            current_indices.append(i)

    if current_indices:
        word_groups.append({"word": current_word.strip(), "token_indices": current_indices})

    word_groups = [wg for wg in word_groups
                   if wg["word"] and not all(c in ' \n\t' for c in wg["word"])]

    print(f"\n  Grouped {len(gen_words)} tokens → {len(word_groups)} words:")
    for wg in word_groups:
        print(f"    \"{wg['word']}\" ← tokens {wg['token_indices']}")

    # ── Merge TAM maps per word (element-wise max across subword tokens) ───────
    word_maps = []  # each entry is (T, H, W) float32 or None
    for wg in word_groups:
        maps = [load_map_3d(ti) for ti in wg["token_indices"]]
        maps = [m for m in maps if m is not None]
        if maps:
            merged = maps[0].copy()
            for m in maps[1:]:
                if m.shape == merged.shape:
                    merged = np.maximum(merged, m)
                else:
                    # Resize each temporal slice to match
                    resized = np.stack([
                        cv2.resize(m[t], (merged.shape[2], merged.shape[1]))
                        for t in range(m.shape[0])
                    ])
                    merged = np.maximum(merged, resized)
            word_maps.append(merged)
        else:
            word_maps.append(None)

    # ── Summary grid: words (rows) × frames (cols) ───────────────────────────
    valid_words = [(wg, wm) for wg, wm in zip(word_groups, word_maps) if wm is not None]
    n_words = min(20, len(valid_words))
    n_frames = T

    if n_words > 0 and n_frames > 0:
        # Extra left margin for word labels
        label_col_w = 1.2  # inches reserved for the word label column
        fig_w = label_col_w + 3 * n_frames
        fig_h = 3.5 * n_words
        fig, axes = plt.subplots(n_words, n_frames,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)
        fig.suptitle(
            f"TAM: Words (rows) × Frames (cols)\n\"{gen_text[:80]}\"",
            fontsize=12, fontweight="bold"
        )

        for row, (wg, wm) in enumerate(valid_words[:n_words]):
            word_label = wg["word"]
            if len(word_label) > 14:
                word_label = word_label[:12] + "…"
            for col in range(n_frames):
                frame_arr = np.array(frames_pil[col] if col < len(frames_pil) else frames_pil[-1])
                per_frame_map = wm[col].astype(np.uint8)  # (H,W) slice
                overlaid = overlay_heatmap(frame_arr, per_frame_map, alpha=0.55)
                axes[row, col].imshow(overlaid)
                axes[row, col].axis("off")
            # Draw the word label directly on the first cell so it never gets clipped
            axes[row, 0].text(-0.05, 0.5, f'"{word_label}"',
                             transform=axes[row, 0].transAxes,
                             fontsize=8, fontweight="bold",
                             ha="right", va="center")

        # Frame index labels along top
        for col in range(n_frames):
            axes[0, col].set_title(f"t={col}", fontsize=7)

        fig.subplots_adjust(left=label_col_w / fig_w)
        plt.savefig(os.path.join(args.save_dir, "tam_summary_grid.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] tam_summary_grid.png ({n_words} words × {n_frames} frames)")

    # ── Selected meaningful words grid (same words × frames layout) ───────────
    meaningful = []
    seen = set()
    for wg, wm in zip(word_groups, word_maps):
        w = wg["word"]
        wl = w.lower()
        if (wm is not None and
            len(w) > 2 and
            w.replace("-", "").replace("'", "").isalpha() and
            wl not in seen and
            wl not in {"the", "and", "for", "are", "was", "has", "its",
                       "this", "that", "with", "from", "they", "their",
                       "which", "have", "been", "also", "into", "can",
                       "not", "but", "all", "some", "more", "very"}):
            meaningful.append((wg, wm))
            seen.add(wl)

    if meaningful:
        n_m = min(12, len(meaningful))
        label_col_w = 1.4  # inches for word labels
        fig_w = label_col_w + 3 * n_frames
        fig_h = 4 * n_m
        fig, axes = plt.subplots(n_m, n_frames,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)
        fig.suptitle("TAM: Selected Meaningful Words × Frames",
                     fontsize=14, fontweight="bold")

        for row, (wg, wm) in enumerate(meaningful[:n_m]):
            word_label = wg["word"]
            if len(word_label) > 14:
                word_label = word_label[:12] + "…"
            for col in range(n_frames):
                frame_arr = np.array(frames_pil[col] if col < len(frames_pil) else frames_pil[-1])
                per_frame_map = wm[col].astype(np.uint8)
                overlaid = overlay_heatmap(frame_arr, per_frame_map, alpha=0.55)
                axes[row, col].imshow(overlaid)
                axes[row, col].axis("off")
            axes[row, 0].text(-0.05, 0.5, f'"{word_label}"',
                             transform=axes[row, 0].transAxes,
                             fontsize=10, fontweight="bold",
                             ha="right", va="center")

        for col in range(n_frames):
            axes[0, col].set_title(f"t={col}", fontsize=8)

        fig.subplots_adjust(left=label_col_w / fig_w)
        plt.savefig(os.path.join(args.save_dir, "tam_selected_words.png"),
                    dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] tam_selected_words.png ({n_m} words × {n_frames} frames)")

    # ── Save per-word multi-frame strips to words/ dir ────────────────────────
    words_dir = os.path.join(args.save_dir, "words")
    os.makedirs(words_dir, exist_ok=True)

    for i, (wg, wm) in enumerate(zip(word_groups, word_maps)):
        if wm is None:
            continue
        # Build horizontal strip: all T frames with overlay
        strips = []
        for col in range(n_frames):
            frame_arr = np.array(frames_pil[col] if col < len(frames_pil) else frames_pil[-1])
            per_frame_map = wm[col].astype(np.uint8)
            overlaid = overlay_heatmap(frame_arr, per_frame_map, alpha=0.55)
            strips.append(cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        strip = np.concatenate(strips, axis=1)

        safe = wg["word"].replace("/", "_").replace(" ", "_").replace("'", "")
        safe = safe.replace("|", "").replace("<", "").replace(">", "")[:20]
        if not safe:
            safe = "empty"
        cv2.imwrite(os.path.join(words_dir, f"{i:03d}_{safe}.jpg"), strip)

    print(f"  [SAVED] {len(word_groups)} word strips → {words_dir}/")

    # ── Save word mapping ─────────────────────────────────────────────────────
    with open(os.path.join(args.save_dir, "word_map.txt"), "w") as f:
        f.write(f"Generated text: {gen_text}\n\n")
        f.write(f"{'Idx':>4}  {'Word':<25}  {'Token Indices'}\n")
        f.write(f"{'─' * 55}\n")
        for i, wg in enumerate(word_groups):
            f.write(f"{i:>4}  {wg['word']:<25}  {wg['token_indices']}\n")

    print(f"\nDone — outputs in: {args.save_dir}/")
    print(f"  {len(logits)} raw TAM heatmaps (0.jpg, 1.jpg, ...)")
    print(f"  named/                — raw maps labeled with token text")
    print(f"  words/                — per-word horizontal frame strips ({n_frames} frames each)")
    print(f"  tam_summary_grid.png  — all words × all frames grid")
    print(f"  tam_selected_words.png — meaningful words × all frames grid")
    print(f"  token_map.txt         — token index → subword mapping")
    print(f"  word_map.txt          — word index → grouped word mapping")


if __name__ == "__main__":
    main()
