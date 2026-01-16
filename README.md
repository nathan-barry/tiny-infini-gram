# Tiny Infini-Gram

A minimal implementation of the [Infini-gram](https://arxiv.org/abs/2401.17377) language model using Go's built-in suffix array.

## How it works

Instead of using a fixed n-gram size, infini-gram finds the longest suffix of your context that exists in the training data, then samples from the distribution of next tokens. This allows it to use arbitrarily long context when available.

## Setup

```bash
# Download the dataset
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt

# Run
go run main.go
```
