# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-dimensional convolution over channels-last inputs."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike

PaddingSpec = str | Sequence[tuple[int, int]]
"""Either a padding mode (``"SAME"`` / ``"VALID"``) or a per-axis ``(lo, hi)``
pair sequence.
"""


def conv(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = "VALID",
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> Array:
    """Apply an N-D convolution.

    Layout conventions:

    * Input ``x`` has shape ``(N, *spatial, C_in)``.
    * Kernel ``w`` has shape ``(*kernel_spatial, C_in // groups, C_out)``.
    * Output has shape ``(N, *spatial_out, C_out)``.

    Args:
        x: Input tensor.
        w: Convolution kernel.
        b: Optional bias of shape ``(C_out,)`` added to the output.
        stride: Per-axis stride (int broadcasts to all axes).
        padding: See :data:`PaddingSpec`.
        dilation: Per-axis kernel dilation (atrous convolution).
        groups: Depthwise-style grouping. Must divide ``C_in``.

    Returns:
        The convolved tensor with optional bias.
    """
    xa = jnp.asarray(x)
    wa = jnp.asarray(w)
    n_spatial = xa.ndim - 2
    if isinstance(stride, int):
        stride = (stride,) * n_spatial
    if isinstance(dilation, int):
        dilation = (dilation,) * n_spatial
    spatial_dims = "".join("HWDTUVXY"[:n_spatial])
    lhs_spec = "N" + spatial_dims + "C"
    rhs_spec = spatial_dims + "IO"
    out_spec = lhs_spec
    dim_numbers = jax.lax.conv_dimension_numbers(xa.shape, wa.shape, (lhs_spec, rhs_spec, out_spec))
    y = jax.lax.conv_general_dilated(
        lhs=xa,
        rhs=wa,
        window_strides=tuple(stride),
        padding=padding,
        rhs_dilation=tuple(dilation),
        dimension_numbers=dim_numbers,
        feature_group_count=groups,
    )
    if b is not None:
        y = y + jnp.asarray(b)
    return y


def conv_transpose(
    x: ArrayLike,
    w: ArrayLike,
    b: ArrayLike | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: PaddingSpec = "VALID",
    dilation: int | Sequence[int] = 1,
) -> Array:
    """Apply an N-D transposed convolution (fractionally-strided conv).

    Layout mirrors :func:`conv`:

    * Input ``x``: ``(N, *spatial, C_in)``.
    * Kernel ``w``: ``(*kernel_spatial, C_in, C_out)``.
    * Output: ``(N, *spatial_out, C_out)``.

    Uses :func:`jax.lax.conv_transpose` under the hood.
    """
    xa = jnp.asarray(x)
    wa = jnp.asarray(w)
    n_spatial = xa.ndim - 2
    if isinstance(stride, int):
        stride = (stride,) * n_spatial
    if isinstance(dilation, int):
        dilation = (dilation,) * n_spatial
    spatial_dims = "".join("HWDTUVXY"[:n_spatial])
    lhs_spec = "N" + spatial_dims + "C"
    rhs_spec = spatial_dims + "IO"
    out_spec = lhs_spec
    dim_numbers = (lhs_spec, rhs_spec, out_spec)
    y = jax.lax.conv_transpose(
        lhs=xa,
        rhs=wa,
        strides=tuple(stride),
        padding=padding,
        rhs_dilation=tuple(dilation),
        dimension_numbers=dim_numbers,
        transpose_kernel=False,
    )
    if b is not None:
        y = y + jnp.asarray(b)
    return y
