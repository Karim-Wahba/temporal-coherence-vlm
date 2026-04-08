"""Constants and hyperparameters for TAM on Qwen3-VL."""

# Special token IDs (shared between Qwen2-VL and Qwen3-VL)
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
IM_START_ID = 151644
ENDOFTEXT_ID = 151645
ASSISTANT_TOKEN_ID = 77091
NEWLINE_ID = 198

# Token boundary identifiers for TAM segment parsing.
# Format: [start_id (int/list), end_id (int/list)]
# Selected tokens are [start + 1 : end].
# Start list uses the idx of the last token; end uses the first.
SPECIAL_IDS = {
    'img_id': [VISION_START_TOKEN_ID, VISION_END_TOKEN_ID],
    'prompt_id': [VISION_END_TOKEN_ID, [ENDOFTEXT_ID, NEWLINE_ID, IM_START_ID, ASSISTANT_TOKEN_ID]],
    'answer_id': [[NEWLINE_ID, IM_START_ID, ASSISTANT_TOKEN_ID, NEWLINE_ID], -1],
}

# Vision processing
SPATIAL_MERGE_SIZE = 2

# Rank Gaussian Filter
RGF_KERNEL_SIZE = 3

# ECI scale cap for Qwen3-VL.
# Qwen3-VL has more correlated activation patterns between context and target
# tokens than Qwen2-VL, so the least-squares scale factor can overshoot.
# This caps the scale factor s in Eq. 5. Values < 1.0 reduce ECI subtraction,
# trading Func-IoU for Obj-IoU. Set to None to disable capping (default).
ECI_SCALE_CAP = None

# Toggle ECI on/off. Set False to use RGF-only mode, which gives best Obj-IoU
# on Qwen3-VL (0.248 vs 0.213 with ECI). Recommended False for temporal experiments.
USE_ECI = True

# Default prompt
DEFAULT_CAPTION_PROMPT = "Write a one-sentence caption for this image:"

# POS tags for evaluation
FUNCTION_WORD_TAGS = ['CC', 'DT', 'EX', 'MD', 'POS', 'PRP', 'PRP$', 'UH', 'WDT', 'WP', 'WP$', 'WRB']
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']

# COCO category name -> label id mapping
COCO_CATEGORIES = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
    'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
    'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21,
    'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27,
    'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34,
    'skis': 35, 'snowboard': 36, 'ball': 37, 'kite': 38, 'baseball bat': 39,
    'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43,
    'bottle': 44, 'glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50,
    'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55,
    'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60,
    'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74,
    'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78,
    'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84,
    'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88,
    'hair drier': 89, 'toothbrush': 90,
}
