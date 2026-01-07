# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""
CPU acceleration utilities.
Note: BetterTransformer is deprecated in optimum 2.x
"""

import os
from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_TORCH, CHECK_TRANSFORMERS
from infinity_emb.primitives import Device

if CHECK_TORCH.is_available:
    import torch

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
        # allow TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoConfig  # type: ignore[import-untyped]


if TYPE_CHECKING:
    from logging import Logger

    from transformers import PreTrainedModel  # type: ignore[import-untyped]

    from infinity_emb.args import EngineArgs


def check_if_bettertransformer_possible(engine_args: "EngineArgs") -> bool:
    """
    BetterTransformer is deprecated in optimum 2.x.
    Modern transformers use SDPA (Scaled Dot Product Attention) by default.
    """
    # BetterTransformer deprecated - return False
    return False


def to_bettertransformer(model: "PreTrainedModel", engine_args: "EngineArgs", logger: "Logger"):
    """
    BetterTransformer is deprecated in optimum 2.x.
    Modern transformers use SDPA by default which provides similar benefits.
    This function now just returns the model unchanged.
    """
    if engine_args.bettertransformer:
        logger.info(
            "BetterTransformer is deprecated in optimum 2.x. "
            "Modern transformers use SDPA (Scaled Dot Product Attention) by default, "
            "which provides similar performance benefits. Continuing without BetterTransformer."
        )
    
    return model
