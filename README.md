# py-infinity ⚡

**Minimal, high-performance CPU text embedding server.**

A slimmed-down fork of [infinity-emb](https://github.com/michaelfeil/infinity) focused on CPU-only text embeddings and reranking.

## Features

- **Text Embeddings** - Deploy any sentence-transformer model from HuggingFace
- **Reranking** - Cross-encoder models for document reranking
- **Fast CPU Inference** - ONNX Runtime, CTranslate2, BetterTransformer
- **Dynamic Batching** - Optimized throughput with automatic batching
- **OpenAI-compatible API** - Drop-in replacement for OpenAI embeddings

## Quick Start

### Docker (Recommended)

```bash
docker build -f libs/infinity_emb/Dockerfile.cpu_auto -t infinity-cpu .

docker run -it -p 7997:7997 \
  -v $PWD/cache:/app/.cache \
  infinity-cpu \
  v2 --model-id BAAI/bge-small-en-v1.5 --engine optimum
```

### pip install

```bash
pip install -e "libs/infinity_emb[server,torch,optimum,ct2,cache]"
infinity_emb v2 --model-id BAAI/bge-small-en-v1.5
```

## API Usage

### Embeddings

```bash
curl -X POST http://localhost:7997/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-small-en-v1.5", "input": ["Hello world", "Test sentence"]}'
```

### Reranking

```bash
curl -X POST http://localhost:7997/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "query": "What is Python?",
    "documents": ["Python is a programming language", "Paris is in France"]
  }'
```

## Supported Models

### Text Embeddings
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- Most [sentence-transformers](https://huggingface.co/models?library=sentence-transformers) models

### Reranking
- [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

## Inference Engines

| Engine | Best For |
|--------|----------|
| `--engine optimum` | ONNX models, fastest CPU inference |
| `--engine torch` | PyTorch models, widest compatibility |
| `--engine ctranslate2` | BERT models, int8 quantization |

## Configuration

```bash
# Environment variables
export INFINITY_MODEL_ID="BAAI/bge-small-en-v1.5"
export INFINITY_ENGINE="optimum"
export INFINITY_BATCH_SIZE="32"
export INFINITY_PORT="7997"

infinity_emb v2
```

## Python API

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs

sentences = ["Hello world", "Test sentence"]
engine = AsyncEngineArray.from_args([
    EngineArgs(model_name_or_path="BAAI/bge-small-en-v1.5", engine="torch")
])[0]

async def embed():
    async with engine:
        embeddings, usage = await engine.embed(sentences=sentences)
        print(f"Embeddings shape: {len(embeddings)}x{len(embeddings[0])}")

asyncio.run(embed())
```

## What's Removed (vs upstream)

This fork removes GPU/vision/audio support for a minimal footprint:

- ❌ GPU support (CUDA, ROCm, TensorRT)
- ❌ Vision models (CLIP, ColPali)
- ❌ Audio models (CLAP)
- ❌ Telemetry (PostHog)
- ❌ Client library

## License

MIT License - Based on [michaelfeil/infinity](https://github.com/michaelfeil/infinity)
