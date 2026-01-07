# py-infinity ⚡

**Minimal, high-performance CPU text embedding server (Optimum-first).**

A slimmed-down fork of [infinity-emb](https://github.com/michaelfeil/infinity) focused on CPU-only text embeddings with ONNX Runtime.

## Features

- **Optimum-First** - ONNX Runtime is the primary engine (fast cpu inference)
- **Lazy Torch** - PyTorch is optional! Run totally torch-free for ~300MB image size
- **Fast CPU Inference** - Optimized for modern CPUs
- **Dynamic Batching** - High throughput with automatic batching
- **OpenAI-compatible API** - Drop-in replacement

## Quick Start

### Docker (Recommended)

**Option 1: Slim (default)**  
ONNX-only, no torch, ~364MB uncompressed. Best for pre-exported ONNX models.
```bash
# Build
docker build -t infinity:slim libs/infinity_emb

# Run
docker run -it -p 7997:7997 infinity:slim \
  v2 --model-id BAAI/bge-small-en-v1.5 --engine optimum
```

**Option 2: Full**  
Includes PyTorch + Optimum (~1.1GB). Supports model export and torch fallback.
```bash
# Build
docker build --build-arg ENGINE=full -t infinity:full libs/infinity_emb

# Run (can use --engine torch or --engine optimum)
docker run -it -p 7997:7997 infinity:full \
  v2 --model-id BAAI/bge-small-en-v1.5 --engine torch
```

### Installation (Python 3.13+)

We recommend **uv** for fast installation:

```bash
# Install slim (ONNX only)
uv pip install "libs/infinity_emb[slim]"

# Install full (Torch + Optimum)
uv pip install "libs/infinity_emb[full]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## Supported Models

### Text Embeddings
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)

### Reranking
- [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)

## Inference Engines

| Engine | Description | Included in |
|--------|-------------|-------------|
| `--engine optimum` | **Default**. ONNX Runtime. Fastest CPU inference. | Slim & Full |
| `--engine torch` | PyTorch fallback. Slower, widely compatible. | Full Only |

## Configuration

```bash
# Environment variables
export INFINITY_MODEL_ID="BAAI/bge-small-en-v1.5"
export INFINITY_ENGINE="optimum" # or "torch"
export INFINITY_BATCH_SIZE="32"
export INFINITY_PORT="7997"

infinity_emb v2
```

## What's Removed (vs upstream)

This fork removes heavy dependencies for a minimal footprint:

- ❌ CTranslate2 engine (use Optimum instead)
- ❌ GPU support (CUDA, ROCm, TensorRT)
- ❌ Vision models (CLIP, ColPali)
- ❌ Audio models (CLAP)
- ❌ Telemetry (PostHog)
- ❌ Client library

## License

MIT License - Based on [michaelfeil/infinity](https://github.com/michaelfeil/infinity)
