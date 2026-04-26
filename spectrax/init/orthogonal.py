# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Orthogonal initializer obtained via a QR decomposition of random noise."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def orthogonal(gain: float = 1.0) -> Initializer:
    """Orthogonal initializer.

    For 2-D-or-higher shapes this computes a QR decomposition of a
    Gaussian matrix, corrects the signs of the diagonal of ``R``, and
    reshapes the orthonormal matrix back to ``shape``. For ranks below
    2 it falls back to scaled Gaussian noise.

    Args:
        gain: Scalar multiplier applied to the orthonormal matrix.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Materialize an orthogonal weight of ``shape``."""
        if len(shape) < 2:
            return jax.random.normal(key, shape, dtype=dtype) * gain
        flat: tuple[int, int] = (shape[0], math.prod(shape[1:]))
        a = np.asarray(jax.random.normal(key, flat, dtype=jnp.float32))
        q, r = np.linalg.qr(a if flat[0] >= flat[1] else a.T)
        d = np.sign(np.diag(r))
        q = q * d
        if flat[0] < flat[1]:
            q = q.T
        return np.asarray(gain * q.reshape(shape), dtype=np.dtype(dtype))

    return init
