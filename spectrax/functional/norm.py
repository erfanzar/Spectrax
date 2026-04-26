# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-level normalization primitives: LayerNorm and RMSNorm."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike


def layer_norm(
    x: ArrayLike,
    scale: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    *,
    axis: int = -1,
    eps: float = 1e-5,
) -> Array:
    """Layer normalization.

    Subtracts the mean and divides by the standard deviation along
    ``axis``, then optionally applies a per-feature affine transform.

    Args:
        x: Input tensor.
        scale: Optional per-feature scale applied after normalization.
        bias: Optional per-feature bias applied after scaling.
        axis: Axis to normalize over.
        eps: Variance floor for numerical stability.
    """
    xa = jnp.asarray(x)
    mean = jnp.mean(xa, axis=axis, keepdims=True)
    var = jnp.var(xa, axis=axis, keepdims=True)
    y = (xa - mean) * _rsqrt(var + eps)
    if scale is not None:
        y = y * jnp.asarray(scale)
    if bias is not None:
        y = y + jnp.asarray(bias)
    return y


def rms_norm(
    x: ArrayLike,
    scale: ArrayLike | None = None,
    *,
    axis: int = -1,
    eps: float = 1e-6,
) -> Array:
    """Root-mean-square normalization.

    Divides by ``sqrt(mean(x**2) + eps)`` along ``axis`` and optionally
    applies a per-feature scale. Cheaper than LayerNorm and popular in
    modern language models.

    Args:
        x: Input tensor.
        scale: Optional per-feature scale.
        axis: Axis to normalize over.
        eps: Mean-of-squares floor for numerical stability.
    """
    xa = jnp.asarray(x)
    sq_mean = jnp.mean(xa * xa, axis=axis, keepdims=True)
    y = xa * _rsqrt(sq_mean + eps)
    if scale is not None:
        y = y * jnp.asarray(scale)
    return y


def _rsqrt(x: Array) -> Array:
    """Reciprocal square root: ``1 / sqrt(x)``."""
    return 1.0 / jnp.sqrt(x)
