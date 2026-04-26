# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dense (fully-connected) layers: :class:`Linear` and :class:`Bilinear`."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.policy import current_policy
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..functional import linear as F_linear
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs


class Linear(Module):
    """Standard dense layer: ``y = x @ W + b``.

    The weight has shape ``(in_features, out_features)``; the bias is
    optional. Logical axis names ``("in", "out")`` are attached to the
    weight and ``("out",)`` to the bias for sharding resolution.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the dense layer.

        Args:
            in_features: Trailing input feature count.  ``None`` defers
                shape inference to the first forward pass.
            out_features: Output feature count.
            use_bias: When ``True`` (default), create and add a bias.
            rngs: Source of PRNG keys for parameter initialization.
            w_init: Weight initializer (default:
                :func:`~spectrax.init.kaiming_uniform` with ``"linear"``
                gain).
            b_init: Bias initializer (default: :func:`~spectrax.init.zeros`).
            dtype: Parameter dtype; defaults to ``float32``.
            param_dtype: Alias for *dtype*.  Provided for compatibility with
                frameworks that separate computation dtype from parameter dtype.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        w_init = w_init or kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_features is None:
            self.weight = DeferredParameter(
                (None, out_features),
                w_init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=("in", "out"),
            )
        else:
            self.weight = Parameter(
                w_init(resolved.parameters, (in_features, out_features), weight_dtype),
                sharding=sharding,
                axis_names=("in", "out"),
            )
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, (out_features,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Compute ``y = x @ W + b`` (bias optional)."""
        xa = jnp.asarray(x)
        if self.in_features is None:
            in_features = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (in_features, self.out_features))
            self.in_features = in_features
        pol = current_policy()
        W = self.weight.value
        b = self.bias.value if self.use_bias else None
        if pol is not None and pol.compute_dtype is not None:
            xa = xa.astype(pol.compute_dtype)
            W = W.astype(pol.compute_dtype)
            if b is not None:
                b = b.astype(pol.compute_dtype)
        if self.use_bias:
            return F_linear(xa, W, b)
        return F_linear(xa, W)


class Bilinear(Module):
    """Bilinear layer: ``y[..., o] = sum(x1[..., i] * W[i, j, o] * x2[..., j]) + b[o]``.

    Useful for interactions between two distinct input streams
    (feature-feature products, encoder-decoder mixing, …).
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the bilinear layer.

        Args:
            in1_features: Feature count of the first input.
            in2_features: Feature count of the second input.
            out_features: Output feature count.
            use_bias: When ``True`` (default), create and add a bias.
            rngs: Source of PRNG keys.
            dtype: Parameter dtype; defaults to ``float32``.
            param_dtype: Alias for *dtype*.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        self.weight = Parameter(
            init(
                resolved.parameters,
                (in1_features, in2_features, out_features),
                weight_dtype,
            ),
            sharding=sharding,
            axis_names=("in1", "in2", "out"),
        )
        if use_bias:
            self.bias = Parameter(
                jnp.zeros((out_features,), dtype=weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x1: ArrayLike, x2: ArrayLike, **_: object) -> Array:
        """Compute the bilinear form and optionally add the bias."""
        y = jnp.einsum("...i,ijo,...j->...o", x1, self.weight.value, x2)
        if self.use_bias:
            y = y + self.bias.value
        return y
