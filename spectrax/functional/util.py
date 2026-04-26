# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small shared utilities for functional ops."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType

__all__ = ["promote_dtype"]


def promote_dtype(
    *arrays: ArrayLike,
    dtype: DType | None = None,
) -> tuple[Array, ...]:
    """Cast every array to a common dtype.

    If ``dtype`` is ``None``, the result uses
    :func:`jax.numpy.promote_types` across the inputs' dtypes.
    """
    arrs = [jnp.asarray(a) for a in arrays]
    if dtype is None:
        if not arrs:
            return ()
        out_dtype = arrs[0].dtype
        for a in arrs[1:]:
            out_dtype = jnp.promote_types(out_dtype, a.dtype)
    else:
        out_dtype = dtype
    return tuple(a.astype(out_dtype) for a in arrs)
