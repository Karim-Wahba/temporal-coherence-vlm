# Temporal Coherence in Vision-Language Models

Research repository exploring temporal coherence, visual grounding, and interpretability in Vision-Language Models (VLMs).

## Structure

```
.
├── experiments/          # Experiment scripts per method
│   └── TAM/              # TAM × Qwen3-VL experiments
├── submodules/           # Modified upstream libraries
│   └── TAM/              # Fork: Karim-Wahba/TAM (modified tam.py)
├── images/               # Shared test images
├── logs/                 # Experiment logs (not tracked)
└── README.md
```

## Experiments

### TAM — Token Activation Map + Qwen3-VL

Adapter to run [TAM (ICCV 2025 Oral)](https://github.com/xmed-lab/TAM) on **Qwen3-VL-8B**, with support for single images and video frame sequences.

The upstream `tam.py` was modified to:
- Replace `fitz` with `pymupdf` (API-compatible, better maintained)
- Comment out the `xelatex`-based `compile_latex_to_jpg` / `vis_text` pipeline in favour of a pure-Python renderer

#### Usage

```bash
# Single image
python experiments/TAM/tam_qwen3vl_v4.2.py \
    --image images/dog.jpg \
    --prompt "Describe this image." \
    --save_dir logs/tam_dog

# Video frames directory
python experiments/TAM/tam_qwen3vl_v4.2.py \
    --video_dir path/to/frames \
    --prompt "Describe this video." \
    --save_dir logs/tam_video
```

#### Setup

```bash
pip install -r submodules/TAM/requirements.txt
# latex renderer (optional, replaced by pymupdf in this fork)
# sudo apt-get install texlive-xetex
```

## Submodules

```bash
git submodule update --init --recursive
```

## Citation

If you use TAM, please cite the original paper:

```bibtex
@InProceedings{Li_2025_ICCV,
    author    = {Li, Yi and Wang, Hualiang and Ding, Xinpeng and Wang, Haonan and Li, Xiaomeng},
    title     = {Token Activation Map to Visually Explain Multimodal LLMs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {48-58}
}
```
