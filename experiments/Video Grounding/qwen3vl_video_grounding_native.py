import os
import json
import math
import torch
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
DAVIS_ROOT = "/home/geiger/gwb913/git/davis/DAVIS2017/unsupervised"
OUTPUT_DIR = "./davis_local_results"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DAVIS_FPS = 24.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)


def get_davis_frames(davis_root, sequence_name):
    img_dir = os.path.join(davis_root, "JPEGImages", "480p", sequence_name)
    return sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])


def process_davis_video(sequence_name, prompt_text, sample_rate=2):
    frames = get_davis_frames(DAVIS_ROOT, sequence_name)
    sampled_frames = frames[::sample_rate]

    # Effective FPS after sampling
    effective_fps = DAVIS_FPS / sample_rate

    # Pass all frames as a single video content block.
    # Qwen3-VL uses 3D RoPE (spatial + temporal) across the sequence,
    # unlike the image-interleaved approach which uses per-image 2D RoPE.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{p}" for p in sampled_frames],
                    "fps": effective_fps,
                },
                {
                    "type": "text",
                    "text": (
                        f"For each frame where the target is visible, output one JSON entry with the "
                        f"0-indexed frame number, a bounding box in [x_min, y_min, x_max, y_max] format "
                        f"(normalized 0-1000), and a short label. Omit frames where the target is absent. "
                        f"Each entry must have a unique frame index.\n"
                        f"Output only the JSON array, nothing else.\n"
                        f"Format: [{{\"frame\": 0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"...\"}}]"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        return_tensors="pt",
    ).to(model.device)

    print(f"Running video-mode inference on {len(sampled_frames)} frames at {effective_fps:.1f} FPS...")
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response = output_text[0]

    with open(os.path.join(OUTPUT_DIR, f"{sequence_name}_raw_video_sr{sample_rate}.txt"), "w") as f:
        f.write(response)

    return response, sampled_frames


def save_results(response_text, frame_paths, sequence_name, sample_rate=2):
    try:
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        model_detections = json.loads(response_text[start:end])
    except Exception as e:
        print(f"Failed to parse model output: {e}")
        print(f"Raw response: {response_text}")
        return

    with Image.open(frame_paths[0]) as img:
        width, height = img.size

    final_json_results = {"sequence": sequence_name, "frames": {}}
    vis_images = []

    # model outputs 0-indexed sampled frame numbers; map back to original DAVIS frame indices
    detection_map = {}
    for det in model_detections:
        sampled_idx = det.get("frame", -1)
        if sampled_idx < 0:
            continue
        original_idx = sampled_idx * sample_rate
        print(f"Detection -> sampled frame {sampled_idx} -> original frame_idx: {original_idx}")
        detection_map[original_idx] = det

    for i, frame_path in enumerate(frame_paths):
        frame_idx = i * sample_rate
        davis_frame_id = f"{frame_idx:05d}"

        img = Image.open(frame_path).convert("RGB")

        if frame_idx in detection_map:
            det = detection_map[frame_idx]
            coords = det.get("bbox_2d", [0, 0, 0, 0])
            x1 = int(coords[0] * width / 1000)
            y1 = int(coords[1] * height / 1000)
            x2 = int(coords[2] * width / 1000)
            y2 = int(coords[3] * height / 1000)

            final_json_results["frames"][davis_frame_id] = {"1": [x1, y1, x2, y2]}

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((10, 10), f"F:{frame_idx} ({frame_idx / DAVIS_FPS:.2f}s)", fill="red")
        else:
            print(f"  -> No detection for frame {frame_idx}")

        vis_images.append(img)

    json_path = os.path.join(OUTPUT_DIR, f"{sequence_name}_video_sr{sample_rate}.json")
    with open(json_path, "w") as f:
        json.dump(final_json_results, f, indent=2)
    print(f"JSON saved to: {json_path}")

    if vis_images:
        num_columns = 5
        num_rows = math.ceil(len(vis_images) / num_columns)
        display_w = 400
        display_h = int(height * (display_w / width))
        grid = Image.new('RGB', (num_columns * display_w, num_rows * display_h))
        for idx, image in enumerate(vis_images):
            image = image.resize((display_w, display_h))
            grid.paste(image, ((idx % num_columns) * display_w, (idx // num_columns) * display_h))
        grid_path = os.path.join(OUTPUT_DIR, f"{sequence_name}_grid_video_sr{sample_rate}.jpg")
        grid.save(grid_path)
        print(f"Grid saved to: {grid_path}")

    print(f"Mapped {len(final_json_results['frames'])} frames.")


if __name__ == "__main__":
    seq_name = "breakdance"
    query = "A man in a red sweatshirt performing breakdance."
    sample_rate = 1

    resp, processed_paths = process_davis_video(seq_name, query, sample_rate=sample_rate)
    save_results(resp, processed_paths, seq_name, sample_rate)
