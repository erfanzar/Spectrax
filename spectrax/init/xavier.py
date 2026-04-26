# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Xavier / Glorot initializers.

Both variants scale the random draw by a gain derived from the layer's
fan-in and fan-out. The companion helper :func:`_fan_in_fan_out` is the
shared fan-computation used by the Kaiming initializers too.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def _fan_in_fan_out(shape: Shape) -> tuple[int, int]:
    """Compute ``(fan_in, fan_out)`` for a weight of the given ``shape``.

    Handles three cases:

    * 1-D or scalar shape — both fans equal ``shape[0]``.
    * 2-D shape ``(in, out)`` — dense-layer fans.
    * >=3-D shape ``(k1, k2, …, in, out)`` — convolution fans multiplied
      by the receptive-field volume (product of the leading spatial dims).
    """
    if len(shape) < 2:
        return shape[0], shape[0]
    if len(shape) > 2:
        receptive = 1
        for s in shape[:-2]:
            receptive *= s
        fan_in = shape[-2] * receptive
        fan_out = shape[-1] * receptive
    else:
        fan_in = shape[-2]
        fan_out = shape[-1]
    return fan_in, fan_out


def xavier_uniform(gain: float = 1.0) -> Initializer:
    """Glorot-uniform initializer.

    Draws from ``U(-a, +a)`` with ``a = gain * sqrt(6 / (fan_in + fan_out))``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Uniformly sample scaled by the Glorot gain."""
        fan_in, fan_out = _fan_in_fan_out(shape)
        a = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, dtype=dtype, minval=-a, maxval=a)

    return init


def xavier_normal(gain: float = 1.0) -> Initializer:
    """Glorot-normal initializer.

    Draws from ``N(0, std**2)`` with ``std = gain * sqrt(2 / (fan_in + fan_out))``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Normal sample scaled by the Glorot gain."""
        fan_in, fan_out = _fan_in_fan_out(shape)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init
