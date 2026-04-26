# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kaiming / He initializers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape
from .xavier import _fan_in_fan_out


def _gain(nonlinearity: str) -> float:
    """Return the standard Kaiming gain for a named nonlinearity.

    Supported names: ``"linear"``, ``"sigmoid"``, ``"tanh"``, ``"relu"``,
    ``"gelu"``, ``"silu"``. Unknown names fall back to ``1.0``.
    """
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity in ("relu", "gelu", "silu"):
        return math.sqrt(2.0)
    return 1.0


def kaiming_uniform(nonlinearity: str = "relu", mode: str = "fan_in") -> Initializer:
    """He-uniform initializer.

    Draws from ``U(-bound, +bound)`` with
    ``bound = gain * sqrt(3 / fan)`` where ``fan`` is ``fan_in`` or
    ``fan_out`` depending on ``mode``.
    """
    gain = _gain(nonlinearity)

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Uniformly sample scaled by the Kaiming bound."""
        fan_in, fan_out = _fan_in_fan_out(shape)
        fan = fan_in if mode == "fan_in" else fan_out
        bound = gain * math.sqrt(3.0 / max(fan, 1))
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

    return init


def kaiming_normal(nonlinearity: str = "relu", mode: str = "fan_in") -> Initializer:
    """He-normal initializer.

    Draws from ``N(0, std**2)`` with ``std = gain / sqrt(fan)``.
    """
    gain = _gain(nonlinearity)

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Normal sample scaled by the Kaiming std."""
        fan_in, fan_out = _fan_in_fan_out(shape)
        fan = fan_in if mode == "fan_in" else fan_out
        std = gain / math.sqrt(max(fan, 1))
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init
