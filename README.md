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

## Submodules

| Folder | Upstream | Fork | Notes |
|--------|----------|------|-------|
| `submodules/TAM` | [xmed-lab/TAM](https://github.com/xmed-lab/TAM) | [Karim-Wahba/TAM](https://github.com/Karim-Wahba/TAM) | Replaced `fitz` with `pymupdf` |

```bash
git submodule update --init --recursive
```

## Setup

Each experiment folder may have its own dependencies. Check the relevant submodule's `requirements.txt`:

```bash
pip install -r submodules/<method>/requirements.txt
```
