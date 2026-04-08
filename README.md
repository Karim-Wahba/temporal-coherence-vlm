# TAM for Qwen3-VL: Video MLLM Interpretability

Token Activation Maps (TAM) ported to Qwen3-VL for studying visual interpretability in video MLLMs.

## Setup

```bash
pip install torch transformers>=4.57.0 accelerate numpy scipy opencv-python Pillow matplotlib nltk
# Optional: bitsandbytes (4-bit quantization), pymupdf (LaTeX text rendering)
```

This repo requires the Qwen3-VL model from HuggingFace (`Qwen/Qwen3-VL-2B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct`). The model is downloaded automatically on first use.

For video preprocessing, you also need `qwen-vl-utils`:
```bash
pip install qwen-vl-utils
```
Or clone the [Qwen2.5-VL repo](https://github.com/QwenLM/Qwen2.5-VL) and add `qwen-vl-utils/src` to your Python path.

## Quick Start

### Single Image TAM
```bash
python -m tam.demo --model-path Qwen/Qwen3-VL-2B-Instruct --image path/to/image.jpg --output-dir ./outputs/ --no-quantize
```

### Video TAM
```bash
python -m tam.demo --model-path Qwen/Qwen3-VL-2B-Instruct --video frame1.jpg frame2.jpg frame3.jpg --output-dir ./outputs/ --no-quantize
```

### DAVIS Evaluation
```bash
python eval_davis.py --model-path Qwen/Qwen3-VL-8B-Instruct --dataset-path /path/to/DAVIS --output-dir results_davis/ --no-quantize --no-eci --max-frames 8 --multilayer --layer-indices 0 7 14 21 27
```

### TempCompass Evaluation
```bash
python eval_tempcompass.py --model-path Qwen/Qwen3-VL-8B-Instruct --dataset-path /path/to/tempcompass/ --output-dir results_tempcompass/ --no-quantize --no-eci --max-frames 8
```

## Project Structure

```
tam/                    # Core TAM module
  config.py             # Special token IDs, hyperparameters
  tam_core.py           # TAM algorithm (ECI, RGF, activation extraction)
  model_utils.py        # Model loading, generation with hidden states
  visualization.py      # Heatmap generation, text rendering
  evaluation.py         # Obj-IoU, Func-IoU, F1-IoU metrics
  demo.py               # Single image/video demo
  eval_coco.py          # COCO Caption evaluation
  RESULTS.md            # All experiment results

eval_davis.py           # DAVIS 2017 video evaluation
eval_tempcompass.py     # TempCompass temporal reasoning evaluation
run_temporal_experiment.py  # Synthetic video experiments
temporal_analysis.py    # Temporal coherence metrics
synthetic_videos.py     # Generate synthetic test videos

NEXT_STEPS.md           # Current experiment plan (READ THIS FIRST)
research_plan.md        # Research direction and status
```

## Current Results (Qwen3-VL-2B)

See `tam/RESULTS.md` for detailed results. Summary:

- **COCO Caption**: Obj-IoU 0.224, F1-IoU 0.357
- **DAVIS 2017** (30 videos): Obj-IoU 0.166
- **TempCompass** (499 QA pairs): 33.3% accuracy, no coherence-accuracy correlation
- **Multi-layer analysis**: Layer patterns differ between synthetic and real video

## Next Steps

**See `NEXT_STEPS.md`** for the prioritized experiment plan. The key next experiment is running TempCompass on Qwen3-VL-8B to test whether model scaling reveals temporal coherence signals.
