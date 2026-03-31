# TAM × Qwen3-VL

Token Activation Map (TAM) visual grounding applied to Qwen3-VL-8B-Instruct.
Visualises which image regions the model attends to for each generated word.

## Setup

From the repo root:

```bash
conda env create -f environment.yml
conda activate vlm_probe
pip install -r requirements.txt
```

## Running

Run from the **repo root**. The model (`Qwen/Qwen3-VL-8B-Instruct`, ~17 GB) downloads automatically from HuggingFace on first run.

```bash
python experiments/TAM/tam_qwen3vl.py \
    --image images/shark.jpg \
    --prompt "Describe this image." \
    --save_dir experiments/TAM/results
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to input image |
| `--prompt` | `"Describe this image."` | Text prompt |
| `--save_dir` | `results/tam_vis` | Output directory |
| `--max_new_tokens` | `256` | Max tokens to generate |
| `--model_id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID |

**Requirements:** A CUDA GPU with sufficient VRAM (~24 GB recommended for the 8B model at full precision; `device_map="auto"` will use what's available).

## Outputs

All written to `--save_dir`:

| File | Description |
|------|-------------|
| `{i}.jpg` | Raw TAM heatmap for the i-th generated token |
| `named/` | Same heatmaps labeled with token text |
| `words/` | Enhanced heatmaps grouped by whole word |
| `tam_summary_grid.png` | All words in one grid |
| `tam_selected_words.png` | Meaningful content words only |
| `token_map.txt` | Token index → subword mapping |
| `word_map.txt` | Word index → grouped word mapping |
