# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-VL vision-language model repository with a custom **TAM (Token Activation Map)** module for visual explainability. TAM generates per-token activation maps showing which image regions each generated token attends to, adapted from the reference implementation (arXiv:2506.23270, ICCV 2025) to work with Qwen3-VL.

**Hardware target:** Mac Mini M4 with 16GB unified memory — TAM uses 4-bit quantization and a custom generation loop to fit within memory constraints.

## Common Commands

```bash
# TAM demo (single image, MPS/Mac)
python3 -m tam.demo --model-path Qwen/Qwen3-VL-2B-Instruct --image path/to/image.jpg --output-dir ./tam_outputs/ --no-quantize

# TAM COCO evaluation
python -m tam.eval_coco --model-path Qwen/Qwen3-VL-2B-Instruct --dataset-path /path/to/coco --max-samples 100

# Web demo (Gradio)
python web_demo_mm.py --checkpoint-path Qwen/Qwen3-VL-2B-Instruct --backend hf --server-port 7860

# Fine-tuning (GPU cluster)
cd qwen-vl-finetune && torchrun --nproc_per_node=8 qwenvl/train/train_qwen.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --dataset_use your_dataset%100 --output_dir ./checkpoints

# Syntax check TAM module
python3 -c "import ast; [ast.parse(open(f'tam/{f}').read()) for f in ['config.py','tam_core.py','model_utils.py','visualization.py','evaluation.py','demo.py','eval_coco.py']]"
```

## Architecture

### TAM Module (`tam/`)

The core explainability pipeline. Data flows through:

1. **`model_utils.py`** — Loads Qwen3-VL (4-bit quantized), prepares inputs via `qwen-vl-utils`, runs a memory-efficient generation loop that computes `model.lm_head(hidden_states[-1])` per step and immediately discards intermediate states.

2. **`tam_core.py`** — For each generated token: extracts class activation scores from logits across all positions (Eq. 1), applies Estimated Causal Inference to subtract interference from context tokens (Eq. 2,4,5), then denoises with Rank Gaussian Filter (Eq. 6-7).

3. **`visualization.py`** — Reshapes activation scores to spatial grid, applies RGF, generates JET colormap heatmap overlay + text token rendering (LaTeX with matplotlib fallback).

4. **`evaluation.py`** — Computes Obj-IoU (object localization), Func-IoU (function word suppression), F1-IoU against COCO Caption ground-truth masks using OTSU thresholding and NLTK POS tagging.

### Circular Import Prevention

`tam_core.py` and `visualization.py` have a mutual dependency (tam_core calls `multimodal_process`, visualization calls `rank_gaussian_filter`). Both use **late imports** inside functions to break the cycle. `__init__.py` uses `__getattr__` for lazy loading.

### Token Boundary Parsing

TAM segments the token stream using special IDs defined in `config.py`:
- **Image tokens:** between `VISION_START (151652)` and `VISION_END (151653)`
- **Prompt tokens:** between `VISION_END` and the assistant header sequence
- **Answer tokens:** after the assistant header to end of sequence

These IDs are shared between Qwen2-VL and Qwen3-VL. The `id2idx()` function handles both single-int and subsequence matching.

### Generation Modes

**Standard (default, `memory_efficient=False`):** Uses `model.generate(output_hidden_states=True)`. Reliable on all backends including MPS. Stores all hidden states — works for moderate generation lengths (~40 tokens) on 16GB.

**Memory-efficient (experimental, `--memory-efficient` flag):** Custom loop that discards hidden states per step. **Does NOT work on Apple MPS** due to `masked_scatter` bug in Qwen3VL's vision embedding that flattens 3D tensors to 2D. Only use on CUDA.

### Other Key Directories

- **`qwen-vl-utils/`** — Installable package for image/video preprocessing (`process_vision_info`, `fetch_image`, `smart_resize`). TAM adds it to sys.path directly.
- **`qwen-vl-finetune/`** — DeepSpeed-based training with differential LR for vision/merger/LLM components. Supports LoRA.
- **`evaluation/`** — Benchmark suites (MMMU, MathVision, ODinW-13, RealWorldQA, VideoMME) with vLLM inference scripts.

## Key Constants

- `SPATIAL_MERGE_SIZE = 2` — Qwen3-VL merges 2x2 vision patches, so vision shape = `image_grid_thw // 2`
- `RGF_KERNEL_SIZE = 3` — Rank Gaussian Filter window (9 elements)
- Model classes: `Qwen3VLForConditionalGeneration` (dense), `Qwen3VLMoeForConditionalGeneration` (MoE)
- All model definitions live in the `transformers` library, not in this repo

## Dependencies

TAM requires: `torch`, `transformers>=4.57.0`, `accelerate`, `numpy`, `scipy`, `opencv-python`, `Pillow`, `matplotlib`, `nltk`. Optional: `bitsandbytes` (4-bit), `pymupdf`+`xelatex` (LaTeX text rendering), `rouge` (NLG metrics).
