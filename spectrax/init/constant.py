# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic constant initializers."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def constant(value: float | int) -> Initializer:
    """Return an initializer that fills with ``value``.

    Args:
        value: The fill value; broadcast to ``shape`` at call time.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Return ``jnp.full(shape, value, dtype)`` — the PRNG key is ignored."""
        del key
        return jnp.full(shape, value, dtype=dtype)

    return init


def zeros(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
    """All-zeros initializer (PRNG key ignored)."""
    del key
    return jnp.zeros(shape, dtype=dtype)


def ones(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
    """All-ones initializer (PRNG key ignored)."""
    del key
    return jnp.ones(shape, dtype=dtype)
