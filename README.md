# Temporal Coherence in Vision-Language Models

Research on temporal coherence, visual grounding, and interpretability in Vision-Language Models (VLMs).

## Structure

```
.
├── experiments/          # One folder per method/experiment
├── submodules/           # Forked upstream libraries with local modifications
├── images/               # Shared test images
├── logs/                 # Experiment logs (gitignored)
└── README.md
```

## Experiments

| Folder | Method | Model | Description |
|--------|--------|-------|-------------|
| `experiments/TAM` | [TAM](https://github.com/xmed-lab/TAM) | Qwen3-VL-8B | Token Activation Map visual grounding |
| `experiments/Ref-DAVIS` | TAM + Ref-DAVIS | Qwen3-VL-8B | VOT benchmarking on Ref-DAVIS with natural language expressions |
| `experiments/grounding-stability` | TAM | Qwen3-VL-8B | Temporal stability of TAM heatmaps and correlation with grounding quality |
| `experiments/token-ablation` | TAM | Qwen3-VL-8B | Ablation of token selection strategies for object localisation via TAM |

## Submodules

| Folder | Upstream | Fork | Notes |
|--------|----------|------|-------|
| `submodules/TAM` | [xmed-lab/TAM](https://github.com/xmed-lab/TAM) | [Karim-Wahba/TAM](https://github.com/Karim-Wahba/TAM) | Replaced `fitz` with `pymupdf` |

```bash
git submodule update --init --recursive
```

## Setup

```bash
conda env create -f environment.yml
conda activate vlm_probe
pip install -r requirements.txt
```

## Reproducing Results

See the README in each experiment folder for run instructions and expected outputs.
