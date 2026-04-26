# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symmetric uniform initializer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def uniform(scale: float = 1.0) -> Initializer:
    """Return an initializer drawing from ``U(-scale, +scale)``."""

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw ``jax.random.uniform(key, shape, minval=-scale, maxval=scale)``."""
        return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)

    return init
