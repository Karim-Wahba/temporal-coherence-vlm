import os
import json
import math
import torch
import ast
from PIL import Image, ImageDraw
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
DAVIS_ROOT = "/home/wahba/git/davis/DAVIS2017/semisupervised"
OUTPUT_DIR = "./davis_local_results"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # Or "3B-Instruct" for lower VRAM
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Model and Processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def get_davis_frames(davis_root, sequence_name):
    img_dir = os.path.join(davis_root, "JPEGImages", "480p", sequence_name)
    return sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])

def process_davis_locally(sequence_name, prompt_text, sample_rate=2):
    frames = get_davis_frames(DAVIS_ROOT, sequence_name)
    
    # Process every Nth frame
    sampled_frames = frames[::sample_rate] 
    
    # 2. Build the Interleaved Message List
    content_list = []
    for i, frame_path in enumerate(sampled_frames):
        # KEY FIX: Use actual frame index, not sampled index
        actual_frame_idx = i * sample_rate
        timestamp = actual_frame_idx / 24.0  # DAVIS is 24 FPS
        
        content_list.append({"type": "text", "text": f"<{timestamp:.2f} seconds>"})
        content_list.append({"type": "image", "image": f"file://{frame_path}"})
    
    content_list.append({"sample_fps": 24})

    full_prompt = (
        f"Given the query \"{prompt_text}\", for each frame, detect and localize "
        f"the visual content described in JSON format. If not present, skip. "
        f"Output Format: [{{'time': 0.0, 'bbox_2d': [x_min, y_min, x_max, y_max], 'label': ''}}, ...]"
    )
    content_list.append({"type": "text", "text": full_prompt})
    messages = [{"role": "user", "content": content_list}]

    # 3. Preparation for local inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        return_video_kwargs=True, 
        image_patch_size=16,
        return_video_metadata=True
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
        do_resize=False, 
        return_tensors="pt"
    ).to(model.device)

    # 4. Generate Output
    print(f"Generating local inference for {len(sampled_frames)} frames...")
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    response = output_text[0]
    
    # Save Raw Text
    with open(os.path.join(OUTPUT_DIR, f"{sequence_name}_raw_local_sr{sample_rate}.txt"), "w") as f:
        f.write(response)
        
    return response, sampled_frames

def save_local_results(response_text, frame_paths, sequence_name, sample_rate=2):
    # 1. Extract JSON
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

    # 2. CREATE THE LOOKUP MAP
    detection_map = {}
    for det in model_detections:
        t = det.get("time", 0.0)
        # Convert time back to frame index (24 FPS)
        idx = int(round(t * 24))
        print(f"Model output -> time: {t:.2f}s, mapped to frame_idx: {idx}")
        detection_map[idx] = det

    # 3. Process Frames
    for i, frame_path in enumerate(frame_paths):
        # Calculate actual frame index in the original sequence
        frame_idx = i * sample_rate 
        davis_frame_id = f"{frame_idx:05d}"
        print(f"Processing sampled frame {i} -> actual frame_idx: {frame_idx}, davis_id: {davis_frame_id}")

        img = Image.open(frame_path).convert("RGB")
        
        # 4. LOOKUP - now frame_idx should match detection_map keys
        if frame_idx in detection_map:
            det = detection_map[frame_idx]
            coords = det.get("bbox_2d", [0, 0, 0, 0])
            
            # De-normalize (assuming model outputs normalized [0-1000] coordinates)
            x1 = int(coords[0] * width / 1000)
            y1 = int(coords[1] * height / 1000)
            x2 = int(coords[2] * width / 1000)
            y2 = int(coords[3] * height / 1000)

            final_json_results["frames"][davis_frame_id] = {"1": [x1, y1, x2, y2]}
            
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((10, 10), f"F:{frame_idx} ({det['time']:.2f}s)", fill="red")
        else:
            print(f"  -> No detection found for frame {frame_idx}")
        
        vis_images.append(img)

    # 4. Save JSON File
    json_path = os.path.join(OUTPUT_DIR, f"{sequence_name}_mapped_sr{sample_rate}.json")
    with open(json_path, "w") as f:
        json.dump(final_json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # 5. Create and Save Image Grid
    if vis_images:
        num_columns = 5
        num_rows = math.ceil(len(vis_images) / num_columns)
        
        # Scale down grid images
        display_w = 400
        display_h = int(height * (display_w / width))
        
        grid_image = Image.new('RGB', (num_columns * display_w, num_rows * display_h))
        
        for idx, image in enumerate(vis_images):
            image = image.resize((display_w, display_h))
            row_idx = idx // num_columns
            col_idx = idx % num_columns
            grid_image.paste(image, (col_idx * display_w, row_idx * display_h))
        
        grid_path = os.path.join(OUTPUT_DIR, f"{sequence_name}_grid_vis_sr{sample_rate}.jpg")
        grid_image.save(grid_path)
        print(f"Grid saved to: {grid_path}")

    print(f"Successfully mapped {len(final_json_results['frames'])} frames.")

if __name__ == "__main__":
    seq_name = "night-race"
    query = "car"
    sample_rate = 4
    
    resp, processed_paths = process_davis_locally(seq_name, query, sample_rate=sample_rate)
    save_local_results(resp, processed_paths, seq_name, sample_rate)