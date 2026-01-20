# Tiny Infini-Gram

A minimal implementation of the [Infini-gram](https://arxiv.org/abs/2401.17377) language model using Go's built-in suffix array, plus a character-level GPT for comparison.

![Infini-gram vs GPT](https://nathan.rs/images/unbounded-n-gram.gif)

## How it works

Instead of using a fixed n-gram size, infini-gram finds multiple suffix matches of varying lengths in the training data and combines their next-token distributions using exponential decay weighting. Longer matches (higher n) are weighted more heavily. The `k` parameter controls how many n-gram levels to use (`k=2` by default, `k=-1` uses all levels).

## Setup

```bash
# Download the dataset (Shakespeare text)
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt

# Download the trained GPT weights (optional - can train from scratch)
mkdir -p weights && wget -P weights https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/gpt.pt
```

## Usage

```bash
# Run infini-gram
go run infini-gram.go

# Run GPT (uses pre-trained weights if available)
uv run gpt.py

# Train GPT from scratch
uv run gpt.py --train

# Run side-by-side visualization comparing both models
uv run visualization.py
```

Both models generate 1000 characters with temperature `0.8` by default. The visualization shows an animated comparison with generation speed proportional to actual inference time.
