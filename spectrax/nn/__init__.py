# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neural-network layers.

Every class here is a :class:`~spectrax.Module` subclass. Containers
(:class:`~spectrax.nn.Sequential`, :class:`~spectrax.nn.ModuleList`,
:class:`~spectrax.nn.StackedModuleList`, :class:`~spectrax.nn.ModuleDict`,
:class:`~spectrax.nn.ParameterList`)
re-export from :mod:`spectrax.core.containers` for convenience.
"""

from ..core.containers import ModuleDict, ModuleList, ParameterList, Sequential, StackedModuleList
from .activation import GELU, ReLU, Sigmoid, SiLU, Tanh
from .attention import CausalSelfAttention, MultiheadAttention
from .conv import (
    Conv,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dense import DenseGeneral, Einsum
from .dropout import Dropout
from .embed import Embed
from .fp8 import Fp8DotGeneral, Fp8Einsum, Fp8Linear, Fp8Meta
from .identity import Identity
from .linear import Bilinear, Linear
from .lora import LoRA, LoRALinear, LoraParameter, wrap_lora
from .mlp import MLPBlock
from .norm import BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm, LayerNorm, RMSNorm
from .pipeline_sequential import PipelineSequential
from .pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from .recurrent import (
    RNN,
    Bidirectional,
    ConvLSTMCell,
    GRUCell,
    LSTMCell,
    OptimizedLSTMCell,
    RNNCellBase,
    SimpleRNNCell,
)

__all__ = [
    "GELU",
    "RNN",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Bidirectional",
    "Bilinear",
    "CausalSelfAttention",
    "Conv",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvLSTMCell",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DenseGeneral",
    "Dropout",
    "Einsum",
    "Embed",
    "Fp8DotGeneral",
    "Fp8Einsum",
    "Fp8Linear",
    "Fp8Meta",
    "GRUCell",
    "GroupNorm",
    "Identity",
    "InstanceNorm",
    "LSTMCell",
    "LayerNorm",
    "Linear",
    "LoRA",
    "LoRALinear",
    "LoraParameter",
    "MLPBlock",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "ModuleDict",
    "ModuleList",
    "MultiheadAttention",
    "OptimizedLSTMCell",
    "ParameterList",
    "PipelineSequential",
    "RMSNorm",
    "RNNCellBase",
    "ReLU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SimpleRNNCell",
    "StackedModuleList",
    "Tanh",
    "wrap_lora",
]
