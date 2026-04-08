"""COCO Caption evaluation entry point for TAM on Qwen3-VL.

Usage:
    python -m tam.eval_coco --model-path Qwen/Qwen3-VL-2B-Instruct --dataset-path /path/to/coco [--vis-path /path/to/save]
"""

import os
import sys
import argparse
from tqdm import tqdm

from .model_utils import load_model, prepare_inputs, generate_with_logits, get_vision_shape
from .tam_core import TAM
from .evaluation import evaluate, prepare_coco_input
from .config import SPECIAL_IDS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TAM on COCO Caption dataset")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--vis-path", type=str, default="", help="Path to save visualizations (optional)")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit number of samples (-1 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--memory-efficient", action="store_true", help="Use experimental memory-efficient loop")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path, use_quantization=not args.no_quantize)
    device = next(model.parameters()).device

    # Load dataset
    print(f"Loading COCO dataset: {args.dataset_path}")
    input_data = prepare_coco_input(args.dataset_path)
    if args.max_samples > 0:
        input_data = input_data[:args.max_samples]
    print(f"Evaluating {len(input_data)} samples...")

    results = []
    for sample_id, (img_path, prompt, caption, mask_path, category) in enumerate(
            tqdm(input_data, unit='sample')):

        # Prepare inputs
        inputs, vis_inputs, is_video = prepare_inputs(processor, img_path, prompt, device=device)
        vision_shape = get_vision_shape(inputs, is_video)

        # Generate with logits
        generated_ids, logits = generate_with_logits(
            model, inputs, max_new_tokens=args.max_new_tokens,
            memory_efficient=args.memory_efficient,
        )

        # Trim generated IDs for evaluation
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = [generated_ids[0, prompt_len:].cpu().tolist()]

        # Set up visualization path
        if args.vis_path:
            save_dir = os.path.join(args.vis_path,
                                    f"{sample_id}_{os.path.basename(img_path).split('.')[0]}")
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = ""

        # Run TAM for each generation round
        img_maps, raw_vis_records = [], []
        for i in range(len(logits)):
            save_fn = os.path.join(save_dir, f"{i}.jpg") if save_dir else ""
            img_map = TAM(
                generated_ids[0].cpu().tolist(),
                vision_shape, logits, SPECIAL_IDS, vis_inputs,
                processor, save_fn, i, raw_vis_records, eval_only=(save_dir == ""),
            )
            img_maps.append(img_map)

        # Evaluate
        metrics = evaluate(img_maps, generated_ids_trimmed, processor, caption, mask_path, category)
        results.append(metrics)

    # Aggregate results
    res = []
    for i in range(len(results[0])):
        values = []
        for r in results:
            values.extend(r[i])
        res.append(sum(values) / max(len(values), 1))

    # Print summary
    obj_iou, func_iou = res[0], res[1]
    f1_iou = 2 * obj_iou * func_iou / max(obj_iou + func_iou, 1e-8)
    print(f"\n{'=' * 60}")
    print(f"TAM Evaluation Results on COCO Caption ({len(input_data)} samples)")
    print(f"{'=' * 60}")
    print(f"  Obj-IoU:   {obj_iou:.4f}")
    print(f"  Func-IoU:  {func_iou:.4f}")
    print(f"  F1-IoU:    {f1_iou:.4f}")
    if len(res) > 2 and res[2] > 0:
        print(f"  ROUGE-L:   {res[2]:.4f}")
    if len(res) > 3 and res[3] > 0:
        print(f"  METEOR:    {res[3]:.4f}")
    if len(res) > 4 and res[4] > 0:
        print(f"  Precision: {res[4]:.4f}")
    if len(res) > 5 and res[5] > 0:
        print(f"  Recall:    {res[5]:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
