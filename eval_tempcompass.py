"""TempCompass evaluation: QA accuracy + TAM temporal coherence correlation.

Tests whether TAM temporal coherence scores predict correct temporal reasoning.
For each video: (1) answer multi-choice QA, (2) compute TAM coherence,
(3) test correlation between coherence and accuracy per temporal dimension.

Usage:
    python3 eval_tempcompass.py \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --dataset-path "/Volumes/Crucial X10/tempcompass" \
        --output-dir results_tempcompass/ \
        --no-quantize --no-eci --max-new-tokens 60 --max-frames 8
"""

import os
import sys
import json
import argparse
import gc
import tempfile
import re

import cv2
import numpy as np
import torch
import nltk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen-vl-utils', 'src'))

from tam.model_utils import (load_model, prepare_inputs, generate_with_logits,
                             get_vision_shape)
from tam.tam_core import TAM
from tam.config import SPECIAL_IDS
from temporal_analysis import compute_temporal_coherence


def parse_args():
    parser = argparse.ArgumentParser(description="TempCompass TAM evaluation")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results_tempcompass/")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-eci", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--task", type=str, default="multi-choice",
                        choices=["multi-choice", "yes_no", "caption_matching"])
    return parser.parse_args()


def extract_frames_from_video(video_path, n_frames=8):
    """Extract N evenly-spaced frames from a video file. Returns list of temp file paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    tmpdir = tempfile.mkdtemp()

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(tmpdir, f"{i:04d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)

    cap.release()
    return frames


def answer_qa(model, processor, device, frame_paths, question, max_new_tokens=60):
    """Run model on video frames + question, return generated answer text."""
    inputs, _, _ = prepare_inputs(
        processor, frame_paths, question, device=device, multi_image=True
    )
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(
        [generated_ids[0, prompt_len:]], skip_special_tokens=True
    )[0].strip()

    del generated_ids, inputs
    gc.collect()
    return answer


def parse_mc_answer(response):
    """Extract answer letter (A/B/C/D) from model response."""
    response = response.strip()
    # Try direct match first
    match = re.match(r'^([A-D])\b', response)
    if match:
        return match.group(1)
    # Try "The answer is X" pattern
    match = re.search(r'(?:answer|option)\s*(?:is|:)\s*([A-D])\b', response, re.I)
    if match:
        return match.group(1)
    # Try finding any standalone letter
    match = re.search(r'\b([A-D])\.\s', response)
    if match:
        return match.group(1)
    return response[:1] if response else ""


def compute_coherence_for_video(model, processor, device, frame_paths, max_new_tokens=60):
    """Run TAM on video frames and compute temporal coherence for first noun token."""
    prompt = "Describe what happens in these video frames."
    inputs, vis_inputs, is_video = prepare_inputs(
        processor, frame_paths, prompt, device=device, multi_image=True
    )
    vision_shape = get_vision_shape(inputs, is_video)

    generated_ids, logits = generate_with_logits(
        model, inputs, max_new_tokens=max_new_tokens
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = processor.batch_decode(
        [generated_ids[0, prompt_len:]], skip_special_tokens=True
    )[0]

    # Find first object noun
    round_idx, target_word = _find_noun(generated_ids, prompt_len, processor)
    if round_idx is None:
        round_idx, target_word = 0, "(first)"

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

    target_maps = all_maps[round_idx]
    if not isinstance(target_maps, list):
        del logits, generated_ids, inputs
        gc.collect()
        return None, target_word, gen_text

    float_maps = [m.astype(np.float64) / 255.0 if m.dtype == np.uint8 else m.astype(np.float64)
                  for m in target_maps]
    coherence = compute_temporal_coherence(float_maps)

    del logits, generated_ids, inputs
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return coherence, target_word, gen_text


def _find_noun(generated_ids, prompt_len, processor):
    """Find first object noun in generated text."""
    gen_ids = generated_ids[0, prompt_len:].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(gen_ids)
    skip_words = {'video', 'sequence', 'frame', 'image', 'scene', 'clip',
                  'footage', 'shot', 'picture', 'series', 'set'}

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
        if not clean or clean in skip_words:
            continue
        try:
            tagged = nltk.pos_tag([clean])
            if tagged[0][1] in ('NN', 'NNS', 'NNP', 'NNPS'):
                return r, w
        except Exception:
            continue
    return None, None


def main():
    args = parse_args()

    if args.no_eci:
        import tam.config as cfg
        cfg.USE_ECI = False

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path, use_quantization=not args.no_quantize)
    device = next(model.parameters()).device

    # Load questions
    q_path = os.path.join(args.dataset_path, "questions", f"{args.task}.json")
    with open(q_path) as f:
        questions = json.load(f)

    video_dir = os.path.join(args.dataset_path, "videos")
    video_ids = list(questions.keys())
    if args.max_samples > 0:
        video_ids = video_ids[:args.max_samples]

    print(f"Evaluating {len(video_ids)} videos, task={args.task}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    dim_correct = {}  # dim -> list of (correct_bool, coherence)

    for vid_idx, video_id in enumerate(video_ids):
        # Find video file — strip _reverse/_concat suffix to find base video
        base_id = video_id.split('_')[0]
        video_path = os.path.join(video_dir, f"{base_id}.mp4")
        if not os.path.exists(video_path):
            continue

        print(f"\n[{vid_idx+1}/{len(video_ids)}] Video {video_id}")

        # Extract frames
        frame_paths = extract_frames_from_video(video_path, args.max_frames)
        if len(frame_paths) < 2:
            print(f"  [skip] Could not extract frames")
            continue

        # Compute TAM coherence
        coherence, target_word, desc = compute_coherence_for_video(
            model, processor, device, frame_paths, args.max_new_tokens
        )
        coh_val = coherence['map_consistency'] if coherence else None
        print(f"  Coherence: {coh_val:.4f}" if coh_val else "  Coherence: N/A")
        print(f"  Target noun: '{target_word}'")

        # Answer QA for each temporal dimension
        video_qs = questions[video_id]
        video_result = {
            'video_id': video_id,
            'coherence': coherence,
            'target_token': target_word,
            'description': desc,
            'qa_results': {},
        }

        for dim, dim_qs in video_qs.items():
            if not dim_qs:
                continue
            # Take first question per dimension
            q = dim_qs[0]
            question_text = q['question']
            gt_answer = q['answer']
            gt_letter = gt_answer[0] if gt_answer else ""

            # Get model answer
            answer = answer_qa(model, processor, device, frame_paths,
                               question_text, args.max_new_tokens)
            pred_letter = parse_mc_answer(answer)
            correct = pred_letter.upper() == gt_letter.upper()

            video_result['qa_results'][dim] = {
                'question': question_text[:100],
                'gt_answer': gt_answer,
                'model_answer': answer[:100],
                'pred_letter': pred_letter,
                'correct': correct,
            }

            # Track for correlation analysis
            if dim not in dim_correct:
                dim_correct[dim] = []
            dim_correct[dim].append((correct, coh_val))

            print(f"  {dim}: {'✓' if correct else '✗'} (pred={pred_letter}, gt={gt_letter})")

        all_results.append(video_result)

        # Cleanup temp frames
        if frame_paths:
            import shutil
            shutil.rmtree(os.path.dirname(frame_paths[0]), ignore_errors=True)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*75}")
    print(f"  TempCompass {args.task} Summary ({len(all_results)} videos)")
    print(f"{'='*75}")
    print(f"{'Dimension':<18} {'N':>4} {'Accuracy':>9} {'Coh(correct)':>13} {'Coh(wrong)':>11}")
    print(f"{'-'*75}")

    for dim in sorted(dim_correct.keys()):
        pairs = [(c, v) for c, v in dim_correct[dim] if v is not None]
        if not pairs:
            continue
        n = len(pairs)
        acc = sum(1 for c, _ in pairs if c) / n
        correct_cohs = [v for c, v in pairs if c]
        wrong_cohs = [v for c, v in pairs if not c]
        cc = sum(correct_cohs) / max(len(correct_cohs), 1) if correct_cohs else float('nan')
        wc = sum(wrong_cohs) / max(len(wrong_cohs), 1) if wrong_cohs else float('nan')
        print(f"{dim:<18} {n:>4} {acc:>9.1%} {cc:>13.4f} {wc:>11.4f}")

    # Overall
    all_pairs = [(c, v) for pairs in dim_correct.values() for c, v in pairs if v is not None]
    if all_pairs:
        overall_acc = sum(1 for c, _ in all_pairs if c) / len(all_pairs)
        correct_all = [v for c, v in all_pairs if c]
        wrong_all = [v for c, v in all_pairs if not c]
        print(f"{'-'*75}")
        cc = sum(correct_all) / len(correct_all) if correct_all else float('nan')
        wc = sum(wrong_all) / len(wrong_all) if wrong_all else float('nan')
        print(f"{'OVERALL':<18} {len(all_pairs):>4} {overall_acc:>9.1%} {cc:>13.4f} {wc:>11.4f}")

    print(f"{'='*75}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
