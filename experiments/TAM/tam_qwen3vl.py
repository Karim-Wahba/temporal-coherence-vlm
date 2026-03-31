"""
Usage:
  python tam_qwen3vl.py \
      --image images/shark.jpg \
      --prompt "Describe this image." \
      --save_dir tam_vis

  python tam_qwen3vl.py \
      --image images/fatherkiddog.jpg \
      --prompt "Describe this image in detail." \
      --save_dir tam_vis
"""

import argparse, os, sys, shutil
import torch
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../submodules/TAM"))

from tam import TAM


def main():
    p = argparse.ArgumentParser("TAM for Qwen3-VL-8B")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--save_dir", default="results/tam_vis")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print(f"Loading {args.model_id}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Prepare inputs ────────────────────────────────────────────────────────
    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.prompt},
    ]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # ── Step 2: Generate with hidden states and compute logits ────────────────
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

    # ── Step 3: Prepare special_ids and vision_shape ──────────────────────────

    # Find the special token IDs for Qwen3-VL by inspecting the input tokens
    token_ids = inputs.input_ids[0].cpu().tolist()
    token_strs = [processor.tokenizer.decode([t], skip_special_tokens=False)
                  for t in token_ids]

    # Print token sequence for debugging special IDs
    print("\nToken sequence (first 20 + last 10):")
    for i in range(min(20, len(token_ids))):
        print(f"  [{i}] id={token_ids[i]}  '{token_strs[i]}'")
    print("  ...")
    for i in range(max(0, len(token_ids) - 10), len(token_ids)):
        print(f"  [{i}] id={token_ids[i]}  '{token_strs[i]}'")

    # Find vision_start and vision_end token IDs
    # In Qwen3-VL: <|vision_start|> ... <|image_pad|> ... <|vision_end|>
    vision_start_id = None
    vision_end_id = None
    image_pad_positions = []

    for i, (tid, ts) in enumerate(zip(token_ids, token_strs)):
        if "<|vision_start|>" in ts:
            vision_start_id = tid
        elif "<|vision_end|>" in ts:
            vision_end_id = tid
        elif "<|image_pad|>" in ts:
            image_pad_positions.append(i)

    print(f"\nvision_start_id: {vision_start_id}")
    print(f"vision_end_id:   {vision_end_id}")
    print(f"image_pad range: {image_pad_positions[0]}..{image_pad_positions[-1]} "
          f"({len(image_pad_positions)} tokens)")

    # Find the prompt start/end and answer start token IDs
    # Prompt is between vision_end and the assistant header
    # In Qwen3-VL the chat template typically ends with:
    #   ... <|vision_end|> {prompt_text} <|im_end|>\n<|im_start|>assistant\n
    # We need to find the boundary token sequences

    # Find where the answer starts (after the assistant header)
    # Look for the sequence that marks end-of-prompt / start-of-generation
    # This is model-specific — inspect the actual tokens:
    im_end_id = None
    im_start_id = None
    for tid, ts in zip(token_ids, token_strs):
        if "<|im_end|>" in ts:
            im_end_id = tid
        if "<|im_start|>" in ts:
            im_start_id = tid

    # Build special_ids for TAM
    # Format: [start_marker, end_marker] where selected tokens are [start+1 : end]
    # img_id: markers around the image tokens
    # prompt_id: markers around prompt text tokens
    # answer_id: markers around answer tokens

    # For Qwen3-VL:
    # Image tokens are between vision_start and vision_end
    # Prompt text is between vision_end and the last im_end before assistant
    # Answer starts after the assistant\n header

    # Find the answer start sequence (im_start + "assistant" + newline)
    # Look for the pattern in token_ids
    answer_start_tokens = []
    for i in range(len(token_ids) - 1):
        if token_ids[i] == im_start_id:
            # Check if this is the assistant header (second im_start)
            # The first im_start is for the user, second is for assistant
            answer_start_tokens.append(i)

    # The assistant header is the LAST im_start in the input
    if answer_start_tokens:
        assistant_header_pos = answer_start_tokens[-1]
    else:
        assistant_header_pos = len(token_ids) - 1

    # Collect the token IDs that form the answer boundary
    # The answer starts right after the assistant header tokens
    # Find the exact boundary: look for "\n" after "assistant"
    answer_boundary = token_ids[assistant_header_pos:input_len]
    print(f"Answer boundary tokens: {answer_boundary}")
    print(f"  = {[processor.tokenizer.decode([t]) for t in answer_boundary]}")

    # Build special_ids
    # TAM format: [start, end] where tokens[start+1 : end] are selected
    special_ids = {
        'img_id': [vision_start_id, vision_end_id],
        'prompt_id': [vision_end_id, answer_boundary],
        'answer_id': [answer_boundary, -1],
    }

    print(f"\nspecial_ids: {special_ids}")

    # Vision shape (post-merge grid)
    vision_shape = (
        inputs['image_grid_thw'][0, 1].item() // 2,
        inputs['image_grid_thw'][0, 2].item() // 2,
    )
    print(f"vision_shape: {vision_shape}")

    # ── Step 4: Call TAM for each generation round ────────────────────────────
    print(f"\nRunning TAM for {len(logits)} rounds...")
    vis_inputs = [image]

    # Decode each generated token for labeling
    gen_ids_list = generated_ids[0][input_len:].cpu().tolist()
    gen_words = [processor.tokenizer.decode([t], skip_special_tokens=False)
                 for t in gen_ids_list]

    # Print the word mapping
    print(f"\n  Token-to-file mapping:")
    print(f"  {'File':>8}  {'Token ID':>8}  {'Word'}")
    print(f"  {'─' * 40}")
    for i, (tid, word) in enumerate(zip(gen_ids_list, gen_words)):
        clean = word.replace('\n', '\\n')
        print(f"  {i:>8}  {tid:>8}  '{clean}'")
    print()

    # Save the mapping to a text file
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
    # POST-PROCESSING: Group subword tokens into words, enhance contrast
    # ══════════════════════════════════════════════════════════════════════════

    img_arr = np.array(image)

    # ── Group subword tokens into whole words ─────────────────────────────────
    # Qwen3 tokenizer uses space-prefixed tokens for word starts.
    # A new word starts when the decoded token starts with a space, or is
    # punctuation, or is a special token.
    word_groups = []  # list of {"word": str, "token_indices": [int], "maps": [ndarray]}
    current_word = ""
    current_indices = []

    for i, raw_word in enumerate(gen_words):
        # Skip special tokens entirely
        if "<|" in raw_word:
            continue

        stripped = raw_word

        # Detect word boundary: space prefix, punctuation, or first token
        is_new_word = (
            len(current_indices) == 0 or     # first token
            stripped.startswith(" ") or       # space-prefixed = new word
            stripped.startswith("\n") or      # newline
            (len(stripped.strip()) == 1 and not stripped.strip().isalnum())  # punctuation
        )

        if is_new_word and current_indices:
            # Save previous word
            word_groups.append({
                "word": current_word.strip(),
                "token_indices": current_indices,
            })
            current_word = stripped
            current_indices = [i]
        else:
            current_word += stripped
            current_indices.append(i)

    # Don't forget the last word
    if current_indices:
        word_groups.append({
            "word": current_word.strip(),
            "token_indices": current_indices,
        })

    # Filter out empty/whitespace-only/punctuation-only words
    word_groups = [wg for wg in word_groups
                   if wg["word"] and not all(c in ' \n\t' for c in wg["word"])]

    print(f"\n  Grouped {len(gen_words)} tokens → {len(word_groups)} words:")
    for wg in word_groups:
        print(f"    \"{wg['word']}\" ← tokens {wg['token_indices']}")

    # ── Merge TAM maps for grouped words ──────────────────────────────────────
    # For multi-token words, take the element-wise MAX across their maps
    # (the most activated region across all subword tokens)
    def load_and_enhance_map(token_idx):
        """Load the TAM map, return as float array with enhanced contrast."""
        if token_idx < len(all_img_maps) and all_img_maps[token_idx] is not None:
            m = all_img_maps[token_idx]
            if isinstance(m, np.ndarray):
                return m.astype(np.float32)
        return None

    def enhance_contrast(heatmap_uint8):
        """Enhance heatmap contrast with CLAHE + percentile stretching."""
        m = heatmap_uint8.astype(np.float32)
        # Percentile stretch: map [p5, p95] → [0, 255]
        p5 = np.percentile(m, 5)
        p95 = np.percentile(m, 95)
        if p95 > p5:
            m = (m - p5) / (p95 - p5)
            m = np.clip(m, 0, 1) * 255
        return m.astype(np.uint8)

    def overlay_heatmap(img, heatmap_uint8, alpha=0.5):
        """Overlay JET heatmap on image with enhanced visibility."""
        enhanced = enhance_contrast(heatmap_uint8)
        colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
        return (img * (1 - alpha) + colored * alpha).astype(np.uint8)

    # Build merged word maps
    word_maps = []
    for wg in word_groups:
        maps = [load_and_enhance_map(ti) for ti in wg["token_indices"]]
        maps = [m for m in maps if m is not None]
        if maps:
            # Element-wise max across subword maps
            merged = maps[0].copy()
            for m in maps[1:]:
                # Handle different shapes
                if m.shape == merged.shape:
                    merged = np.maximum(merged, m)
                else:
                    # Resize to match
                    m_resized = cv2.resize(m, (merged.shape[1], merged.shape[0]))
                    merged = np.maximum(merged, m_resized)
            word_maps.append(merged)
        else:
            word_maps.append(None)

    # ── Summary grid: ALL grouped words ───────────────────────────────────────
    valid_words = [(wg, wm) for wg, wm in zip(word_groups, word_maps) if wm is not None]
    n_show = min(40, len(valid_words))
    if n_show > 0:
        cols = min(8, n_show)
        rows = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
        fig.suptitle(f"TAM: Per-Word Visual Explanation (grouped subwords)\n"
                     f"\"{gen_text[:80]}\"",
                     fontsize=12, fontweight="bold")
        axes_flat = np.array(axes).flatten() if rows > 1 else (
            np.array([axes]) if cols == 1 else axes)

        for i, (wg, wm) in enumerate(valid_words[:n_show]):
            overlaid = overlay_heatmap(img_arr, wm, alpha=0.55)
            ax = axes_flat[i]
            ax.imshow(overlaid)
            word = wg["word"]
            if len(word) > 14: word = word[:12] + "…"
            ax.set_title(f"\"{word}\"", fontsize=9, fontweight="bold")
            ax.axis("off")

        for i in range(n_show, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "tam_summary_grid.png"),
                    dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [SAVED] tam_summary_grid.png")

    # ── Selected meaningful words (nouns, long words, deduplicated) ───────────
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
        n_m = min(16, len(meaningful))
        cols_m = min(4, n_m)
        rows_m = (n_m + cols_m - 1) // cols_m

        fig, axes = plt.subplots(rows_m, cols_m, figsize=(6 * cols_m, 6 * rows_m))
        fig.suptitle("TAM: Selected Meaningful Words (enhanced contrast)",
                     fontsize=14, fontweight="bold")
        axes_flat = np.array(axes).flatten() if rows_m > 1 else (
            np.array([axes]) if cols_m == 1 else axes)

        for idx, (wg, wm) in enumerate(meaningful[:n_m]):
            overlaid = overlay_heatmap(img_arr, wm, alpha=0.55)
            ax = axes_flat[idx]
            ax.imshow(overlaid)
            ax.set_title(f"\"{wg['word']}\"", fontsize=13, fontweight="bold")
            ax.axis("off")

        for idx in range(n_m, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "tam_selected_words.png"),
                    dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [SAVED] tam_selected_words.png")

    # ── Save enhanced per-word maps ───────────────────────────────────────────
    words_dir = os.path.join(args.save_dir, "words")
    os.makedirs(words_dir, exist_ok=True)

    for i, (wg, wm) in enumerate(zip(word_groups, word_maps)):
        if wm is None:
            continue
        overlaid = overlay_heatmap(img_arr, wm, alpha=0.55)
        safe = wg["word"].replace("/", "_").replace(" ", "_").replace("'", "")
        safe = safe.replace("|", "").replace("<", "").replace(">", "")[:20]
        if not safe: safe = "empty"
        path = os.path.join(words_dir, f"{i:03d}_{safe}.jpg")
        cv2.imwrite(path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    print(f"  [SAVED] {len(word_groups)} word maps → {words_dir}/")

    # ── Save word mapping ─────────────────────────────────────────────────────
    with open(os.path.join(args.save_dir, "word_map.txt"), "w") as f:
        f.write(f"Generated text: {gen_text}\n\n")
        f.write(f"{'Idx':>4}  {'Word':<25}  {'Token Indices'}\n")
        f.write(f"{'─' * 55}\n")
        for i, wg in enumerate(word_groups):
            f.write(f"{i:>4}  {wg['word']:<25}  {wg['token_indices']}\n")

    print(f"\nDone — outputs in: {args.save_dir}/")
    print(f"  {len(logits)} raw TAM heatmaps (0.jpg, 1.jpg, ...)")
    print(f"  named/              — raw maps labeled with token text")
    print(f"  words/              — enhanced maps grouped by whole word")
    print(f"  tam_summary_grid.png   — all words in one grid")
    print(f"  tam_selected_words.png — meaningful words only")
    print(f"  token_map.txt       — token index → subword mapping")
    print(f"  word_map.txt        — word index → grouped word mapping")


if __name__ == "__main__":
    main()