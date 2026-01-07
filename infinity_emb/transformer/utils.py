# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""
CPU-only transformer utilities - optimum-first architecture.
"""

from enum import Enum
from typing import Callable

from infinity_emb.primitives import InferenceEngine
from infinity_emb.transformer.classifier.torch import SentenceClassifier
from infinity_emb.transformer.classifier.optimum import OptimumClassifier
from infinity_emb.transformer.crossencoder.optimum import OptimumCrossEncoder
from infinity_emb.transformer.crossencoder.torch import (
    CrossEncoderPatched as CrossEncoderTorch,
)
from infinity_emb.transformer.embedder.dummytransformer import DummyTransformer
from infinity_emb.transformer.embedder.optimum import OptimumEmbedder
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)

__all__ = [
    "length_tokenizer",
    "get_lengths_with_tokenize",
]


class EmbedderEngine(Enum):
    """
    Available embedding engines.
    optimum (ONNX) is the primary engine for CPU performance.
    torch is available as a fallback for maximum compatibility.
    """
    optimum = OptimumEmbedder
    torch = SentenceTransformerPatched
    debugengine = DummyTransformer

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.optimum:
            return EmbedderEngine.optimum
        elif engine == InferenceEngine.torch:
            return EmbedderEngine.torch
        elif engine == InferenceEngine.debugengine:
            return EmbedderEngine.debugengine
        else:
            raise NotImplementedError(f"EmbedderEngine for {engine} not implemented")


class RerankEngine(Enum):
    optimum = OptimumCrossEncoder
    torch = CrossEncoderTorch

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.optimum:
            return RerankEngine.optimum
        elif engine == InferenceEngine.torch:
            return RerankEngine.torch
        else:
            raise NotImplementedError(f"RerankEngine for {engine} not implemented")


class PredictEngine(Enum):
    optimum = OptimumClassifier
    torch = SentenceClassifier

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.optimum:
            return PredictEngine.optimum
        elif engine == InferenceEngine.torch:
            return PredictEngine.torch
        else:
            raise NotImplementedError(f"PredictEngine for {engine} not implemented")


def length_tokenizer(
    _sentences: list[str],
) -> list[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: list[str], tokenize: Callable = length_tokenizer
) -> tuple[list[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)
