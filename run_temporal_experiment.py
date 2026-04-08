"""Run TAM temporal coherence experiments on synthetic test videos.

For each video:
1. Load frames as multi-image input (separate activation map per frame)
2. Run TAM to generate per-frame activation maps
3. Identify noun tokens referring to tracked objects
4. Compute temporal coherence metrics
5. Evaluate against ground truth trajectory
6. Save results JSON + trajectory plot

Usage:
    python3 run_temporal_experiment.py \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --video-dir test_videos/ --output-dir results_temporal/ \
        --no-quantize --no-eci --max-new-tokens 40
"""

import os
import sys
import json
import glob
import argparse
import gc

import numpy as np
import torch
import nltk

from tam.model_utils import (load_model, prepare_inputs, generate_with_logits,
                             get_vision_shape, extract_multilayer_scores)
from tam.tam_core import TAM, TAM_multilayer
from tam.config import SPECIAL_IDS
from temporal_analysis import compute_temporal_coherence, evaluate_tracking, load_trajectory, print_report


def parse_args():
    parser = argparse.ArgumentParser(description="TAM temporal coherence experiments")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--video-dir", type=str, default="test_videos/",
                        help="Directory containing video subdirectories with frames")
    parser.add_argument("--output-dir", type=str, default="results_temporal/")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Max frames per video (memory constraint)")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-eci", action="store_true",
                        help="Disable ECI (RGF-only mode, recommended for temporal)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override prompt (default: auto-generated per video)")
    parser.add_argument("--target-token", type=str, default=None,
                        help="Override target token (default: first noun)")
    parser.add_argument("--videos", type=str, nargs='+', default=None,
                        help="Specific video names to run (default: all)")
    parser.add_argument("--multilayer", action="store_true",
                        help="Run multi-layer logit lens analysis")
    parser.add_argument("--layer-indices", type=int, nargs='+', default=None,
                        help="Layer indices for multi-layer (default: 5 evenly spaced)")
    return parser.parse_args()


def find_noun_token_round(generated_ids, prompt_len, processor, target_word=None):
    """Find the generation round index for the first noun token (or target_word).

    Returns (round_idx, word) or (None, None) if not found.
    """
    gen_ids = generated_ids[0, prompt_len:].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(gen_ids)

    # Build word groups
    words = []
    word_rounds = []  # which generation round each word starts at
    current_word = ""
    current_round = 0

    for i, tok in enumerate(tokens):
        decoded = processor.tokenizer.decode(processor.tokenizer.convert_tokens_to_ids(tok))
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

    # Search for target
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
                    return r, w
            except Exception:
                continue

    return None, None


def get_default_prompt(video_name):
    """Generate an appropriate prompt for each video type."""
    prompts = {
        "simple_translation": "These are frames from a video. What is the red circle doing?",
        "two_objects_crossing": "These are frames from a video with two colored circles. Describe their movement.",
        "appearance_change": "These are frames from a video. Describe what happens to the circle.",
        "occlusion": "These are frames from a video. Describe the red circle and the green barrier.",
        "reentry": "These are frames from a video. Describe the red circle's movement.",
    }
    return prompts.get(video_name, "These are frames from a video. Describe what you see.")


def run_single_video(model, processor, device, video_dir, video_name, args):
    """Run TAM temporal experiment on a single video."""
    # Load frames
    frame_paths = sorted(glob.glob(os.path.join(video_dir, video_name, "*.jpg")))
    if not frame_paths:
        print(f"  [skip] No frames found in {video_name}/")
        return None

    frame_paths = frame_paths[:args.max_frames]
    n_frames = len(frame_paths)

    # Load ground truth
    gt_path = os.path.join(video_dir, video_name, "trajectory.json")
    gt = load_trajectory(gt_path) if os.path.exists(gt_path) else None
    if gt:
        gt = [e for e in gt if e['frame_idx'] < n_frames]

    # Prepare prompt
    prompt = args.prompt or get_default_prompt(video_name)

    # Prepare inputs (multi-image mode)
    inputs, vis_inputs, is_video = prepare_inputs(
        processor, frame_paths, prompt, device=device, multi_image=True
    )
    vision_shape = get_vision_shape(inputs, is_video)
    print(f"  {n_frames} frames, vision_shape: {vision_shape[0]} per frame")

    # Generate
    generated_ids, logits = generate_with_logits(
        model, inputs, max_new_tokens=args.max_new_tokens
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = processor.batch_decode(
        [generated_ids[0, prompt_len:]], skip_special_tokens=True
    )[0]
    print(f"  Generated: {gen_text[:100]}...")

    # Find target noun token
    round_idx, target_word = find_noun_token_round(
        generated_ids, prompt_len, processor, args.target_token
    )
    if round_idx is None:
        print(f"  [warn] No noun token found, using round 0")
        round_idx, target_word = 0, "(first token)"

    print(f"  Target: '{target_word}' at round {round_idx}")

    # Run TAM for all rounds, collect per-frame maps
    raw_records = []
    all_maps = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape, logits, SPECIAL_IDS, vis_inputs,
            processor, '', i, raw_records, eval_only=True,
        )
        all_maps.append(img_map)

    # Extract per-frame maps for target token
    target_maps = all_maps[round_idx]
    if not isinstance(target_maps, list):
        print(f"  [warn] Expected list of per-frame maps, got {type(target_maps)}")
        return None

    # Convert to float for analysis
    float_maps = [m.astype(np.float64) / 255.0 if m.dtype == np.uint8 else m.astype(np.float64)
                  for m in target_maps]

    # Compute temporal coherence
    coherence = compute_temporal_coherence(float_maps)

    # Evaluate against GT if available
    tracking = None
    if gt:
        tracking = evaluate_tracking(float_maps, gt, map_shape=(448, 448))

    # Build result
    result = {
        'video_name': video_name,
        'n_frames': n_frames,
        'prompt': prompt,
        'generated_text': gen_text,
        'target_token': target_word,
        'target_round': round_idx,
        'eci_enabled': not args.no_eci,
        'temporal_coherence': coherence,
    }
    if tracking:
        result['tracking'] = {
            'position_accuracy': tracking['position_accuracy'],
            'position_accuracy_visible': tracking['position_accuracy_visible'],
            'occlusion_recovery': tracking['occlusion_recovery'],
        }

    # Print summary
    print(f"  Map consistency: {coherence['map_consistency']:.4f}")
    print(f"  Spatial smoothness: {coherence['spatial_smoothness']:.2f} px")
    if tracking and tracking['position_accuracy'] is not None:
        print(f"  Position accuracy: {tracking['position_accuracy']:.2f} map-px")
    if tracking and tracking['occlusion_recovery'] is not None:
        print(f"  Occlusion recovery: {tracking['occlusion_recovery']:.4f}")

    # Save visualization of trajectory
    save_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)

    _plot_trajectory(coherence, gt, float_maps[0].shape,
                     os.path.join(save_dir, "trajectory.png"), video_name, target_word)

    # Save TAM visualizations for target token
    for i in [round_idx]:
        vis_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape, logits, SPECIAL_IDS, vis_inputs,
            processor, os.path.join(save_dir, f"tam_round{i}.jpg"),
            i, [], eval_only=False,
        )

    # Multi-layer analysis
    if args.multilayer:
        tokens = generated_ids[0].cpu().tolist()
        answer_id = SPECIAL_IDS['answer_id']
        from tam.tam_core import id2idx
        answer_idx_pair = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]
        cls_id = tokens[answer_idx_pair[0] + round_idx + 1]

        n_layers = 28  # Qwen3-VL-2B
        layer_idx_list = args.layer_indices or [0, 7, 14, 21, 27]
        layer_idx_list = [l for l in layer_idx_list if l < n_layers]

        print(f"  Multi-layer analysis at layers {layer_idx_list}...")
        ml_scores = extract_multilayer_scores(
            model, inputs, generated_ids, cls_id, layer_idx_list
        )
        ml_maps = TAM_multilayer(tokens, vision_shape, ml_scores, SPECIAL_IDS, layer_idx_list)

        layer_coherence = {}
        layer_tracking = {}
        for lidx in layer_idx_list:
            if lidx not in ml_maps:
                continue
            fmaps = ml_maps[lidx]
            lc = compute_temporal_coherence(fmaps)
            layer_coherence[lidx] = lc
            if gt:
                lt = evaluate_tracking(fmaps, gt, map_shape=(448, 448))
                layer_tracking[lidx] = {
                    'position_accuracy': lt['position_accuracy'],
                    'occlusion_recovery': lt['occlusion_recovery'],
                }
            print(f"    Layer {lidx:>2}: consistency={lc['map_consistency']:.4f}  "
                  f"smoothness={lc['spatial_smoothness']:.2f}  "
                  f"pos_acc={layer_tracking.get(lidx, {}).get('position_accuracy', 'N/A')}")

        result['multilayer'] = {
            'layer_indices': layer_idx_list,
            'cls_id': cls_id,
            'coherence_per_layer': {str(k): v for k, v in layer_coherence.items()},
            'tracking_per_layer': {str(k): v for k, v in layer_tracking.items()},
        }

        # Plot coherence vs layer
        _plot_layer_coherence(layer_coherence, layer_tracking, layer_idx_list,
                              os.path.join(save_dir, "layer_coherence.png"),
                              video_name, target_word)

    # Cleanup
    del logits, generated_ids, inputs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def _plot_layer_coherence(layer_coherence, layer_tracking, layer_indices, save_path, title, token):
    """Plot temporal coherence metrics vs layer depth."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = sorted(layer_coherence.keys())
    consistency = [layer_coherence[l]['map_consistency'] for l in layers]
    smoothness = [layer_coherence[l]['spatial_smoothness'] for l in layers]
    pos_acc = [layer_tracking.get(l, {}).get('position_accuracy') for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{title} — token '{token}' — Layer Analysis", fontsize=11)

    axes[0].plot(layers, consistency, 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Map Consistency")
    axes[0].set_title("Cosine Similarity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, smoothness, 'rs-', linewidth=2, markersize=6)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Spatial Smoothness (px)")
    axes[1].set_title("Peak Displacement")
    axes[1].grid(True, alpha=0.3)

    if any(p is not None for p in pos_acc):
        valid = [(l, p) for l, p in zip(layers, pos_acc) if p is not None]
        axes[2].plot([v[0] for v in valid], [v[1] for v in valid], 'g^-', linewidth=2, markersize=6)
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Position Accuracy (map-px)")
        axes[2].set_title("Distance to GT")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No GT", ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("Position Accuracy")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_trajectory(coherence, gt, map_shape, save_path, title, token):
    """Plot predicted vs GT trajectory."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    peaks = coherence['peak_trajectory']
    h, w = map_shape

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f"{title} — token '{token}'")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # Predicted trajectory
    pred_rows = [p[0] for p in peaks]
    pred_cols = [p[1] for p in peaks]
    ax.plot(pred_cols, pred_rows, 'ro-', label='Predicted peaks', markersize=6, linewidth=1.5)
    for i, (r, c) in enumerate(peaks):
        ax.annotate(str(i), (c, r), fontsize=7, ha='center', va='bottom')

    # GT trajectory
    if gt:
        gt_rows = [int(e['y'] * h / 448) for e in gt]
        gt_cols = [int(e['x'] * w / 448) for e in gt]
        ax.plot(gt_cols, gt_rows, 'bs--', label='Ground truth', markersize=5, linewidth=1)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    # Toggle ECI
    if args.no_eci:
        import tam.config as cfg
        cfg.USE_ECI = False
        print("ECI disabled (RGF-only mode)")

    # Ensure NLTK data
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # Load model
    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path, use_quantization=not args.no_quantize)
    device = next(model.parameters()).device

    # Discover videos
    if args.videos:
        video_names = args.videos
    else:
        video_names = sorted([
            d for d in os.listdir(args.video_dir)
            if os.path.isdir(os.path.join(args.video_dir, d))
        ])

    print(f"Found {len(video_names)} videos: {video_names}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiments
    all_results = []
    for video_name in video_names:
        print(f"\n{'='*60}")
        print(f"  {video_name}")
        print(f"{'='*60}")

        result = run_single_video(model, processor, device, args.video_dir, video_name, args)
        if result:
            all_results.append(result)

    # Save all results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Summary: Temporal Coherence Across Video Types")
    print(f"{'='*80}")
    print(f"{'Video':<25} {'Token':<12} {'Consistency':>12} {'Smoothness':>12} {'Pos. Acc.':>10}")
    print(f"{'-'*80}")
    for r in all_results:
        tc = r['temporal_coherence']
        pa = r.get('tracking', {}).get('position_accuracy')
        pa_str = f"{pa:.2f}" if pa is not None else "N/A"
        print(f"{r['video_name']:<25} {r['target_token']:<12} "
              f"{tc['map_consistency']:>12.4f} {tc['spatial_smoothness']:>12.2f} {pa_str:>10}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
