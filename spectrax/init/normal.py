# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian (normal / truncated normal) initializers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def normal(stddev: float = 1.0, mean: float = 0.0) -> Initializer:
    """Return an initializer drawing samples from ``N(mean, stddev**2)``."""

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw ``jax.random.normal(key, shape) * stddev + mean``."""
        return jax.random.normal(key, shape, dtype=dtype) * stddev + mean

    return init


def truncated_normal(
    stddev: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> Initializer:
    """Truncated-normal initializer.

    Samples from ``N(0, 1)`` truncated to ``[lower, upper]`` and scales
    by ``stddev``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw a truncated-normal sample and scale by ``stddev``."""
        return jax.random.truncated_normal(key, lower, upper, shape, dtype=dtype) * stddev

    return init
