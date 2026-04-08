"""DAVIS 2017 evaluation: per-frame Obj-IoU + layer-wise analysis + temporal coherence.

Evaluates TAM activation maps against DAVIS per-frame segmentation masks.
For each val video: extracts frames, runs TAM in multi-image mode, computes
per-frame Obj-IoU (activation map vs GT mask), and optionally runs multi-layer
analysis to validate the U-shaped layer curve on real video.

Usage:
    python3 eval_davis.py \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --dataset-path "/Volumes/Crucial X10/davis2017/DAVIS" \
        --output-dir results_davis/ \
        --no-quantize --no-eci --max-frames 8 --max-new-tokens 60 \
        --multilayer --layer-indices 0 7 14 21 27
"""

import os
import sys
import json
import argparse
import gc

import cv2
import numpy as np
import torch
import nltk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen-vl-utils', 'src'))

from tam.model_utils import (load_model, prepare_inputs, generate_with_logits,
                             get_vision_shape, extract_multilayer_scores)
from tam.tam_core import TAM, TAM_multilayer, id2idx
from tam.config import SPECIAL_IDS
from temporal_analysis import compute_temporal_coherence


# DAVIS video name -> likely object tokens (priority order).
# Used to find the right generated noun for activation map evaluation.
DAVIS_OBJECT_TOKENS = {
    "bike-packing": ["bicycle", "bike", "cyclist", "person"],
    "blackswan": ["swan", "bird"],
    "bmx-trees": ["rider", "cyclist", "bicycle", "boy"],
    "breakdance": ["dancer", "breakdancer", "person", "man"],
    "camel": ["camel", "camels"],
    "car-roundabout": ["car", "vehicle", "mini"],
    "car-shadow": ["car", "vehicle", "fiat"],
    "cows": ["cow", "cows", "cattle"],
    "dance-twirl": ["dancer", "woman", "girl"],
    "dog": ["dog", "retriever", "puppy"],
    "dogs-jump": ["dog", "dogs"],
    "drift-chicane": ["car", "vehicle"],
    "drift-straight": ["car", "vehicle"],
    "goat": ["goat"],
    "gold-fish": ["fish", "goldfish"],
    "horsejump-high": ["horse", "rider"],
    "india": ["person", "man", "woman", "rickshaw"],
    "judo": ["judoka", "fighter", "person", "man"],
    "kite-surf": ["surfer", "kite", "person"],
    "lab-coat": ["person", "man", "woman"],
    "libby": ["dog"],
    "loading": ["truck", "vehicle"],
    "mbike-trick": ["motorcycle", "rider", "bike"],
    "motocross-jump": ["motorcycle", "rider", "motocross"],
    "paragliding-launch": ["paraglider", "person"],
    "parkour": ["person", "man"],
    "pigs": ["pig", "pigs"],
    "scooter-black": ["scooter", "rider", "person"],
    "shooting": ["person", "shooter", "man"],
    "soapbox": ["car", "vehicle", "racer"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="DAVIS 2017 TAM evaluation")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to DAVIS root (contains JPEGImages/, Annotations/)")
    parser.add_argument("--output-dir", type=str, default="results_davis/")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-eci", action="store_true")
    parser.add_argument("--max-videos", type=int, default=-1)
    parser.add_argument("--multilayer", action="store_true")
    parser.add_argument("--layer-indices", type=int, nargs='+', default=[0, 7, 14, 21, 27])
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    return parser.parse_args()


def load_davis_val_list(dataset_path, split="val"):
    """Load list of video names from DAVIS ImageSets."""
    split_file = os.path.join(dataset_path, "ImageSets", "2017", f"{split}.txt")
    if not os.path.exists(split_file):
        # Fallback: list directories
        img_dir = os.path.join(dataset_path, "JPEGImages", "480p")
        return sorted(os.listdir(img_dir))
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def sample_frames(frame_dir, n_frames):
    """Sample N evenly-spaced frames from a directory."""
    all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg') and not f.startswith('.')])
    if len(all_frames) <= n_frames:
        indices = list(range(len(all_frames)))
    else:
        indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int).tolist()
    return [all_frames[i] for i in indices], indices


def compute_per_frame_iou(activation_maps, mask_dir, frame_names, vision_shapes):
    """Compute Obj-IoU between TAM activation maps and DAVIS GT masks.

    For multi-object masks, tries each object separately and reports the best
    per-object IoU (since we can't match token to object ID automatically).

    Args:
        activation_maps: list of uint8 activation maps (one per frame), shape (h, w)
        mask_dir: path to DAVIS Annotations for this video
        frame_names: list of frame filenames (e.g., '00000.jpg')
        vision_shapes: list of (h, w) tuples

    Returns:
        (binary_ious, best_obj_ious, mean_binary, mean_best_obj)
    """
    binary_ious = []
    best_obj_ious = []

    for i, (amap, fname) in enumerate(zip(activation_maps, frame_names)):
        mask_name = fname.replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt_mask.shape
        resized_map = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        _, pred = cv2.threshold(resized_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Binary IoU (all objects vs background)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        if gt_binary.sum() > 0:
            tp = float((gt_binary * pred > 0).sum())
            union = ((gt_binary + pred / 255) > 0).sum()
            binary_ious.append(tp / max(union, 1))

        # Per-object IoU (best match among individual objects)
        unique_ids = np.unique(gt_mask)
        unique_ids = unique_ids[unique_ids > 0]
        obj_ious_frame = []
        for obj_id in unique_ids:
            gt_obj = (gt_mask == obj_id).astype(np.uint8)
            if gt_obj.sum() == 0:
                continue
            tp = float((gt_obj * pred > 0).sum())
            union = ((gt_obj + pred / 255) > 0).sum()
            obj_ious_frame.append(tp / max(union, 1))
        if obj_ious_frame:
            best_obj_ious.append(max(obj_ious_frame))

    mean_binary = sum(binary_ious) / max(len(binary_ious), 1)
    mean_best_obj = sum(best_obj_ious) / max(len(best_obj_ious), 1)
    return binary_ious, best_obj_ious, mean_binary, mean_best_obj


def find_noun_token_round(generated_ids, prompt_len, processor, target_word=None):
    """Find generation round for first noun token (reused from run_temporal_experiment.py)."""
    gen_ids = generated_ids[0, prompt_len:].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(gen_ids)

    words, word_rounds = [], []
    current_word, current_round = "", 0

    for i, tok in enumerate(tokens):
        decoded = processor.tokenizer.decode(
            processor.tokenizer.convert_tokens_to_ids(tok))
        if i == 0 or decoded.startswith(' ') or tok.startswith('▁'):
            if current_word:
                words.append(current_word.strip())
                word_rounds.append(current_round)
            current_word = decoded
            current_round = i
        else:
            current_word += decoded

    if current_word:
        words.append(current_word.strip())
        word_rounds.append(current_round)

    for w, r in zip(words, word_rounds):
        clean = w.strip('.,!?;:').lower()
        if not clean:
            continue
        if target_word and target_word.lower() in clean:
            return r, w
        if target_word is None:
            try:
                tagged = nltk.pos_tag([clean])
                if tagged[0][1] in ('NN', 'NNS', 'NNP', 'NNPS'):
                    # Skip generic words
                    if clean not in ('video', 'sequence', 'frame', 'image', 'scene',
                                     'clip', 'footage', 'shot', 'picture'):
                        return r, w
            except Exception:
                continue

    return None, None


def run_single_video(model, processor, device, dataset_path, video_name, args):
    """Run TAM evaluation on a single DAVIS video."""
    frame_dir = os.path.join(dataset_path, "JPEGImages", "480p", video_name)
    mask_dir = os.path.join(dataset_path, "Annotations", "480p", video_name)

    if not os.path.isdir(frame_dir):
        print(f"  [skip] No frames: {frame_dir}")
        return None

    # Sample frames
    frame_names, frame_indices = sample_frames(frame_dir, args.max_frames)
    frame_paths = [os.path.join(frame_dir, f) for f in frame_names]
    n_frames = len(frame_paths)
    print(f"  {n_frames} frames sampled from {len(os.listdir(frame_dir))} total")

    # Prepare inputs (multi-image mode)
    prompt = "Describe what you see in these video frames."
    inputs, vis_inputs, is_video = prepare_inputs(
        processor, frame_paths, prompt, device=device, multi_image=True
    )
    vision_shape = get_vision_shape(inputs, is_video)

    # Generate
    generated_ids, logits = generate_with_logits(
        model, inputs, max_new_tokens=args.max_new_tokens
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = processor.batch_decode(
        [generated_ids[0, prompt_len:]], skip_special_tokens=True
    )[0]
    print(f"  Generated: {gen_text[:80]}...")

    # Try DAVIS object token candidates in priority order
    candidates = DAVIS_OBJECT_TOKENS.get(video_name, [video_name.split('-')[-1]])
    round_idx, target_word = None, None
    for candidate in candidates:
        round_idx, target_word = find_noun_token_round(
            generated_ids, prompt_len, processor, target_word=candidate
        )
        if round_idx is not None:
            break
    # Fallback: any object noun
    if round_idx is None:
        round_idx, target_word = find_noun_token_round(
            generated_ids, prompt_len, processor, target_word=None
        )
    if round_idx is None:
        print(f"  [warn] No object noun found, using round 0")
        round_idx, target_word = 0, "(first)"

    print(f"  Target: '{target_word}' at round {round_idx}")

    # Run TAM
    raw_records = []
    all_maps = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape, logits, SPECIAL_IDS, vis_inputs,
            processor, '', i, raw_records, eval_only=True,
        )
        all_maps.append(img_map)

    # Get per-frame maps for target token
    target_maps = all_maps[round_idx]
    if not isinstance(target_maps, list):
        print(f"  [warn] Expected list of per-frame maps, got single map")
        return None

    # Compute per-frame Obj-IoU
    binary_ious, best_obj_ious, mean_binary, mean_best_obj = compute_per_frame_iou(
        target_maps, mask_dir, frame_names, vision_shape)
    print(f"  Obj-IoU (binary): {mean_binary:.4f}  (best-obj): {mean_best_obj:.4f}")

    # Temporal coherence
    float_maps = [m.astype(np.float64) / 255.0 if m.dtype == np.uint8 else m.astype(np.float64)
                  for m in target_maps]
    coherence = compute_temporal_coherence(float_maps)
    print(f"  Map consistency: {coherence['map_consistency']:.4f}")

    result = {
        'video_name': video_name,
        'n_frames': n_frames,
        'generated_text': gen_text,
        'target_token': target_word,
        'target_round': round_idx,
        'per_frame_iou_binary': binary_ious,
        'per_frame_iou_best_obj': best_obj_ious,
        'mean_iou_binary': mean_binary,
        'mean_iou_best_obj': mean_best_obj,
        'temporal_coherence': coherence,
    }

    # Multi-layer analysis
    if args.multilayer:
        tokens = generated_ids[0].cpu().tolist()
        answer_id = SPECIAL_IDS['answer_id']
        answer_idx_pair = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]
        cls_id = tokens[answer_idx_pair[0] + round_idx + 1]

        print(f"  Multi-layer analysis at layers {args.layer_indices}...")
        ml_scores = extract_multilayer_scores(
            model, inputs, generated_ids, cls_id, args.layer_indices
        )
        ml_maps = TAM_multilayer(tokens, vision_shape, ml_scores, SPECIAL_IDS, args.layer_indices)

        layer_ious = {}
        layer_coherence = {}
        for lidx in args.layer_indices:
            if lidx not in ml_maps:
                continue
            fmaps = ml_maps[lidx]
            # Convert to uint8 for IoU
            uint8_maps = [(m * 255).astype(np.uint8) for m in fmaps]
            _, _, l_binary, l_best = compute_per_frame_iou(uint8_maps, mask_dir, frame_names, vision_shape)
            layer_ious[lidx] = {'mean_binary': l_binary, 'mean_best_obj': l_best}
            layer_coherence[lidx] = compute_temporal_coherence(fmaps)
            print(f"    Layer {lidx:>2}: IoU(bin)={l_binary:.4f} IoU(best)={l_best:.4f}  consistency={layer_coherence[lidx]['map_consistency']:.4f}")

        result['multilayer'] = {
            'layer_indices': args.layer_indices,
            'iou_per_layer': {str(k): v for k, v in layer_ious.items()},
            'coherence_per_layer': {str(k): v for k, v in layer_coherence.items()},
        }

    # Cleanup
    del logits, generated_ids, inputs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def main():
    args = parse_args()

    if args.no_eci:
        import tam.config as cfg
        cfg.USE_ECI = False
        print("ECI disabled (RGF-only mode)")

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path, use_quantization=not args.no_quantize)
    device = next(model.parameters()).device

    # Load video list
    video_names = load_davis_val_list(args.dataset_path, args.split)
    if args.max_videos > 0:
        video_names = video_names[:args.max_videos]
    print(f"Evaluating {len(video_names)} DAVIS {args.split} videos")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for video_name in video_names:
        print(f"\n{'='*60}")
        print(f"  {video_name}")
        print(f"{'='*60}")

        result = run_single_video(model, processor, device, args.dataset_path, video_name, args)
        if result:
            all_results.append(result)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  DAVIS {args.split} Summary ({len(all_results)} videos)")
    print(f"{'='*70}")

    if all_results:
        # Per-video table
        print(f"\n  {'Video':<22} {'Token':<15} {'IoU(bin)':>9} {'IoU(best)':>10} {'Consistency':>12}")
        print(f"  {'-'*68}")
        for r in all_results:
            print(f"  {r['video_name']:<22} {r['target_token']:<15} "
                  f"{r['mean_iou_binary']:>9.4f} {r['mean_iou_best_obj']:>10.4f} "
                  f"{r['temporal_coherence']['map_consistency']:>12.4f}")

        all_binary = [r['mean_iou_binary'] for r in all_results]
        all_best = [r['mean_iou_best_obj'] for r in all_results]
        all_cons = [r['temporal_coherence']['map_consistency'] for r in all_results]
        print(f"  {'-'*68}")
        print(f"  {'MEAN':<22} {'':15} {sum(all_binary)/len(all_binary):>9.4f} "
              f"{sum(all_best)/len(all_best):>10.4f} {sum(all_cons)/len(all_cons):>12.4f}")

        # Layer-wise summary
        if args.multilayer and any('multilayer' in r for r in all_results):
            print(f"\n  Layer-wise Obj-IoU:")
            print(f"  {'Layer':>6} {'IoU(bin)':>10} {'IoU(best)':>11} {'Consistency':>12}")
            for lidx in args.layer_indices:
                lb = [r['multilayer']['iou_per_layer'].get(str(lidx), {}).get('mean_binary', 0)
                      for r in all_results if 'multilayer' in r]
                lo = [r['multilayer']['iou_per_layer'].get(str(lidx), {}).get('mean_best_obj', 0)
                      for r in all_results if 'multilayer' in r]
                lc = [r['multilayer']['coherence_per_layer'].get(str(lidx), {}).get('map_consistency', 0)
                      for r in all_results if 'multilayer' in r]
                if lb:
                    print(f"  {lidx:>6} {sum(lb)/len(lb):>10.4f} {sum(lo)/len(lo):>11.4f} {sum(lc)/len(lc):>12.4f}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
