# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pointwise activation functions, thin wrappers over :mod:`jax.nn`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike


def relu(x: ArrayLike) -> Array:
    """Rectified linear unit: ``max(0, x)``."""
    return jax.nn.relu(x)


def gelu(x: ArrayLike, *, approximate: bool = False) -> Array:
    """Gaussian error linear unit.

    Args:
        x: Input.
        approximate: When ``True`` uses the tanh-based approximation;
            otherwise computes the exact erf form.
    """
    return jax.nn.gelu(x, approximate=approximate)


def silu(x: ArrayLike) -> Array:
    """Sigmoid-weighted linear unit (a.k.a. swish): ``x * sigmoid(x)``."""
    return jax.nn.silu(x)


def tanh(x: ArrayLike) -> Array:
    """Hyperbolic tangent."""
    return jnp.tanh(x)


def sigmoid(x: ArrayLike) -> Array:
    """Logistic sigmoid: ``1 / (1 + exp(-x))``."""
    return jax.nn.sigmoid(x)


def softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Numerically-stable softmax along ``axis``."""
    return jax.nn.softmax(x, axis=axis)


def leaky_relu(x: ArrayLike, negative_slope: float = 0.01) -> Array:
    """Leaky ReLU: ``max(0, x) + negative_slope * min(0, x)``."""
    return jax.nn.leaky_relu(x, negative_slope=negative_slope)


def elu(x: ArrayLike, alpha: float = 1.0) -> Array:
    """Exponential linear unit."""
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


def selu(x: ArrayLike) -> Array:
    """Scaled exponential linear unit."""
    return jax.nn.selu(x)


def celu(x: ArrayLike, alpha: float = 1.0) -> Array:
    """Continuously differentiable ELU."""
    return jax.nn.celu(x, alpha=alpha)


def glu(x: ArrayLike, axis: int = -1) -> Array:
    """Gated linear unit: ``a * sigmoid(b)`` splitting along ``axis``."""
    return jax.nn.glu(x, axis=axis)


def hard_sigmoid(x: ArrayLike) -> Array:
    """Piecewise-linear approximation of the sigmoid."""
    return jax.nn.hard_sigmoid(x)


def hard_tanh(x: ArrayLike) -> Array:
    """Clipped linear in ``[-1, 1]``."""
    return jax.nn.hard_tanh(x)


def hard_silu(x: ArrayLike) -> Array:
    """Piecewise-linear approximation of the silu (swish)."""
    return jax.nn.hard_silu(x)


def hard_swish(x: ArrayLike) -> Array:
    """Alias for :func:`hard_silu`."""
    return jax.nn.hard_silu(x)


def mish(x: ArrayLike) -> Array:
    """Mish activation: ``x * tanh(softplus(x))``."""
    return jnp.asarray(x) * jnp.tanh(jax.nn.softplus(x))


def soft_sign(x: ArrayLike) -> Array:
    """``x / (1 + |x|)``."""
    return jax.nn.soft_sign(x)


def log_softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Numerically-stable log-softmax."""
    return jax.nn.log_softmax(x, axis=axis)


def log_sigmoid(x: ArrayLike) -> Array:
    """Log of the logistic sigmoid."""
    return jax.nn.log_sigmoid(x)


def prelu(x: ArrayLike, alpha: ArrayLike) -> Array:
    """Parametric ReLU: ``max(0, x) + alpha * min(0, x)``.

    ``alpha`` broadcasts against ``x``.
    """
    xa = jnp.asarray(x)
    return jnp.where(xa > 0, xa, jnp.asarray(alpha) * xa)
