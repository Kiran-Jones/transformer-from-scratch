# Transformer from Scratch

A complete implementation of the Transformer architecture (Vaswani et al., 2017) built using only Python/NumPy. As a proof of concept, the project also contains a pretrained English-to-Spanish translation model, which is deployed on [Hugging Face](https://huggingface.co/spaces/KiranJones/transformer-from-scratch) with attention visualization.

The trained model performs adequately, achieving a ~30 sacreBLEU score. While not optimized for maximum performance, the project is intended as a learning and reference implementation, with an emphasis on clarity and inspectability over performance.

## Project Structure

```
src/
├── model/          # Transformer architecture
├── inference/      # Tokenization and decoding
├── training/       # Training and evaluation utilities
└── visualize.py    # Gradio-based attention demo
```

## Overview

This project implements a full encoder–decoder Transformer, including:
- Multi-head scaled dot-product attention
- Sinusoidal positional encodings
- Byte Pair Encoding (BPE) tokenization
- Beam search decoding
- Interactive attention visualization via Gradio

All core components are implemented manually, including forward and backward passes.

## Installation

Requires Python 3.12+ and `uv`

```bash
git clone https://github.com/kiran-jones/transformer-from-scratch.git
cd transformer-from-scratch
uv sync
```

## Usage

### Training

```bash
uv run python -m src.training.train_seq2seq -i path/to/training/data.tsv
```
For available options: 
```bash
uv run python -m src.training.train_seq2seq --help
```

### Evaluation
```bash
# Example evaluation on a 1k sentence subset 
uv run python -m src.training.bleu \
  --data src/data/en2es_test.tsv \
  --test-size 1000 \
  -m seq2seq_model.pkl
```

### Attention Visualization Demo

```bash
uv run python src/visualize.py -m seq2seq_model.pkl
```
The demo can be run locally or accessed via the hosted [Hugging Face Space](https://huggingface.co/spaces/KiranJones/transformer-from-scratch).

#### Attention Visualization Modes
- **Single View**: One layer and head with summary statistics
- **All Heads**: All heads for a selected layer
- **All Layers**: A single head across all layers

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." [Paper](https://arxiv.org/abs/1706.03762)
- Radford A., et al. (2019). "Language Models are Unsupervised Multitask Learners." [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT
