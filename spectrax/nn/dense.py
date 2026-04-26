# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`DenseGeneral` and :class:`Einsum`.

Generalizations of :class:`~spectrax.nn.Linear` that allow contractions
over arbitrary axes (``DenseGeneral``) or fully-custom einsum equations
(``Einsum``).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Parameter
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs

__all__ = ["DenseGeneral", "Einsum"]


def _to_tuple(x: int | Sequence[int]) -> tuple[int, ...]:
    """Return ``x`` as a tuple (wrapping ints)."""
    return (x,) if isinstance(x, int) else tuple(x)


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    """Resolve negative axis indices against ``ndim``."""
    return tuple(a % ndim for a in axes)


class DenseGeneral(Module):
    """Dense layer with contraction over arbitrary axes.

    ``axis`` selects which axes of the input are contracted; ``features``
    is the shape of the trailing new axes. The weight has shape
    ``(*contracted_shape, *features)``; the bias has shape ``features``.

    Example::

        >>> layer = DenseGeneral(features=(4, 8), axis=(-2, -1), rngs=Rngs(0))
        >>> y = layer(jnp.zeros((3, 5, 2, 6)))
        >>> y.shape
        (3, 5, 4, 8)
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        features: int | Sequence[int],
        *,
        axis: int | Sequence[int] = -1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        in_shape: Sequence[int] | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Output feature shape (one int or a tuple).
            axis: Contraction axis / axes on the input (negatives ok).
            use_bias: When ``True``, allocate a bias.
            rngs: PRNG source.
            w_init: Weight initializer (default: kaiming_uniform).
            b_init: Bias initializer (default: zeros).
            dtype: Parameter dtype.
            param_dtype: Alias for *dtype*.
            in_shape: Shape of the contracted axes on the input. Required
                because the weight is allocated eagerly; pass the
                contracted-axis sizes in the same order as ``axis``.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        self.features = _to_tuple(features)
        self.axis = _to_tuple(axis)
        self.use_bias = use_bias
        if in_shape is None:
            raise ValueError("DenseGeneral requires in_shape=(..) for the contracted axes")
        self.in_shape = tuple(in_shape)
        if len(self.in_shape) != len(self.axis):
            raise ValueError("in_shape length must equal axis length")
        resolved = resolve_rngs(rngs)
        dt = param_dtype or dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        wshape = (*self.in_shape, *self.features)
        self.weight = Parameter(w_init(resolved.parameters, wshape, dt), sharding=sharding)
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, tuple(self.features), dt),
                sharding=bias_sharding,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Contract ``x`` with ``weight`` along ``axis`` and add ``bias``."""
        xa = jnp.asarray(x)
        axes = _normalize_axes(self.axis, xa.ndim)
        contracting = list(axes)
        n_contract = len(contracting)
        n_features = len(self.features)
        y = jnp.tensordot(xa, self.weight.value, axes=(contracting, list(range(n_contract))))
        if self.use_bias:
            expand = (None,) * (y.ndim - n_features) + (slice(None),) * n_features
            y = y + self.bias.value[expand]
        return y


class Einsum(Module):
    """Learnable einsum.

    The equation describes how the input ``x`` and the parameter
    ``weight`` combine. ``shape`` is the parameter shape. For example::

        >>> e = Einsum("...ij,jk->...ik", shape=(4, 8), rngs=Rngs(0))
        >>> e(jnp.zeros((3, 2, 4))).shape
        (3, 2, 8)
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        equation: str,
        shape: Sequence[int],
        *,
        use_bias: bool = False,
        bias_shape: Sequence[int] | None = None,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            equation: An :func:`jnp.einsum` equation where the first
                operand is the input and the second is :attr:`weight`.
            shape: Shape of :attr:`weight`.
            use_bias: When ``True``, add a broadcasting bias.
            bias_shape: Shape of the bias (defaults to the output
                shape inferred from the equation; supply if ambiguous).
            rngs: PRNG source.
            w_init: Weight initializer (default: kaiming_uniform).
            b_init: Bias initializer (default: zeros).
            dtype: Parameter dtype.
            param_dtype: Alias for *dtype*.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        if "->" not in equation:
            raise ValueError("Einsum equation must contain '->'")
        self.equation = equation
        self.shape = tuple(shape)
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = param_dtype or dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        self.weight = Parameter(w_init(resolved.parameters, self.shape, dt), sharding=sharding)
        if use_bias:
            if bias_shape is None:
                raise ValueError("Einsum(use_bias=True) requires bias_shape=(..)")
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, tuple(bias_shape), dt),
                sharding=bias_sharding,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply the einsum then optionally broadcast-add ``bias``."""
        y = jnp.einsum(self.equation, jnp.asarray(x), self.weight.value)
        if self.use_bias:
            y = y + self.bias.value
        return y
