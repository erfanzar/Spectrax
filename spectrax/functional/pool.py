# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pooling (reduce-window) primitives on channels-last tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.lax as lax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike

__all__ = ["avg_pool", "max_pool", "pool"]

_PadSpec = str | Sequence[tuple[int, int]]


def _pool_window(
    window_shape: Sequence[int],
    strides: Sequence[int] | None,
    x_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pad window/stride specs with leading/trailing 1s for batch and channel axes."""
    ws = (1, *tuple(window_shape), 1)
    if strides is None:
        strides = window_shape
    st = (1, *tuple(strides), 1)
    if len(ws) != x_ndim or len(st) != x_ndim:
        raise ValueError(
            f"window_shape / strides of length {len(window_shape)} do not match input with {x_ndim - 2} spatial axes"
        )
    return ws, st


def pool(
    x: ArrayLike,
    init_value: ArrayLike,
    reduce_fn: Callable[[Array, Array], Array],
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
) -> Array:
    """General reduce-window pooling on ``(N, *spatial, C)`` inputs.

    ``reduce_fn`` must be a commutative, associative binary combiner
    (e.g. :func:`jax.lax.max`, :func:`jax.lax.add`).
    """
    xa = jnp.asarray(x)
    ws, st = _pool_window(window_shape, strides, xa.ndim)
    if isinstance(padding, str):
        pad = padding
    else:
        pad = [(0, 0), *[tuple(p) for p in padding], (0, 0)]
    return lax.reduce_window(xa, jnp.asarray(init_value), reduce_fn, ws, st, pad)


def max_pool(
    x: ArrayLike,
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
) -> Array:
    """Max-pool ``x`` over ``window_shape``."""
    xa = jnp.asarray(x)
    init = jnp.array(-jnp.inf, dtype=xa.dtype)
    return pool(xa, init, lax.max, window_shape, strides=strides, padding=padding)


def avg_pool(
    x: ArrayLike,
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
    count_include_pad: bool = True,
) -> Array:
    """Average-pool ``x`` over ``window_shape``.

    When ``count_include_pad=False`` the denominator is the number of
    non-padded positions in each window; otherwise it's the constant
    window volume.
    """
    xa = jnp.asarray(x)
    summed = pool(xa, jnp.array(0.0, dtype=xa.dtype), lax.add, window_shape, strides=strides, padding=padding)
    window_size = 1
    for w in window_shape:
        window_size *= w
    if count_include_pad or padding == "VALID":
        return summed / window_size
    ones = jnp.ones_like(xa)
    counts = pool(ones, jnp.array(0.0, dtype=xa.dtype), lax.add, window_shape, strides=strides, padding=padding)
    return summed / counts
