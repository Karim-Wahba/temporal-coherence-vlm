"""TAM demo for Qwen3-VL: generate token activation maps for a single image or video.

Usage:
    python -m tam.demo \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --image path/to/image.jpg \
        --prompt "Describe this image." \
        --output-dir ./tam_outputs/

    # Video demo (list of frame paths):
    python -m tam.demo \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --video frame1.jpg frame2.jpg frame3.jpg \
        --prompt "Describe this video." \
        --output-dir ./tam_outputs/
"""

import os
import argparse
import torch

from .model_utils import load_model, prepare_inputs, generate_with_logits, get_vision_shape
from .tam_core import TAM
from .config import SPECIAL_IDS, DEFAULT_CAPTION_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="TAM Demo for Qwen3-VL")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, nargs='+', default=None,
                        help="Paths to video frames (list of image paths)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_CAPTION_PROMPT,
                        help="Text prompt for the model")
    parser.add_argument("--output-dir", type=str, default="tam_outputs",
                        help="Directory to save visualization outputs")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use experimental memory-efficient generation loop")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.video is None:
        print("Error: Provide either --image or --video.")
        return

    # Load model
    print(f"Loading model: {args.model_path}")
    model, processor = load_model(args.model_path, use_quantization=not args.no_quantize)
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Prepare input
    image_or_video = args.video if args.video else args.image
    inputs, vis_inputs, is_video = prepare_inputs(processor, image_or_video, args.prompt, device=device)
    vision_shape = get_vision_shape(inputs, is_video)
    print(f"Vision shape: {vision_shape}")

    # Generate
    print("Generating tokens...")
    generated_ids, logits = generate_with_logits(
        model, inputs, max_new_tokens=args.max_new_tokens,
        memory_efficient=args.memory_efficient,
    )

    # Decode generated text
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = processor.batch_decode(
        [generated_ids[0, prompt_len:]],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    print(f"Generated: {gen_text}")

    # Run TAM for each generation round
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating TAM visualizations ({len(logits)} rounds)...")

    raw_map_records = []
    for i in range(len(logits)):
        save_fn = os.path.join(args.output_dir, f"{i}.jpg")
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape, logits, SPECIAL_IDS, vis_inputs,
            processor, save_fn, i, raw_map_records, eval_only=False,
        )

    print(f"Done! Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
