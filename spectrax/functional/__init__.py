# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stateless tensor operations.

Every function here is pure: it takes JAX-compatible inputs and returns
an :class:`~spectrax.typing.Array`. The :mod:`spectrax.nn` layers build
on these implementations, so users can drop down to the functional form
whenever a module wrapper feels heavy.
"""

from .activation import (
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    mish,
    prelu,
    relu,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    tanh,
)
from .attention import scaled_dot_product_attention
from .conv import conv, conv_transpose
from .dropout import dropout
from .linear import linear
from .norm import layer_norm, rms_norm
from .pool import avg_pool, max_pool, pool
from .util import promote_dtype

__all__ = [
    "avg_pool",
    "celu",
    "conv",
    "conv_transpose",
    "dropout",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_sigmoid",
    "log_softmax",
    "max_pool",
    "mish",
    "pool",
    "prelu",
    "promote_dtype",
    "relu",
    "rms_norm",
    "scaled_dot_product_attention",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "tanh",
]
