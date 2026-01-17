# Tiny Infini-Gram

A minimal implementation of the [Infini-gram](https://arxiv.org/abs/2401.17377) language model using Go's built-in suffix array.

## How it works

Instead of using a fixed n-gram size, infini-gram finds the longest suffix of your context that exists in the training data, then samples from the distribution of next tokens. This allows it to use arbitrarily long context when available.

## Setup

```bash
# Download the dataset
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt

# Download the trained GPT weights (if you don't want to train it from scratch)
mkdir -p weights && wget -P weights https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/gpt.pt

# Run infini-gram
go run main.go

# Run GPT
uv run gpt.py
```

The file generates 1000 characters and uses a sampling temperature of `0.8` by default. These (and the initial prompt) can be changed in the main function.
