# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalization layers: :class:`LayerNorm`, :class:`RMSNorm`,
:class:`BatchNorm1d`, :class:`BatchNorm2d`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Buffer, Parameter
from ..functional import layer_norm as F_layer_norm
from ..functional import rms_norm as F_rms_norm

_FEATURE_AXIS: AxisNames = ("features",)
_CHANNEL_AXIS: AxisNames = ("channels",)


class LayerNorm(Module):
    """Per-feature layer normalization.

    Normalizes along the trailing ``features`` axis and optionally
    applies a learned per-feature affine transform.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        features: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Size of the trailing (normalization) axis.
            eps: Variance floor for numerical stability.
            elementwise_affine: When ``True``, allocate a learned scale
                (and optional bias).
            use_bias: When ``True`` and ``elementwise_affine`` is set,
                also allocate a bias.
            dtype: Parameter dtype.
            sharding: Optional sharding for the scale.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Parameter(
                jnp.ones((features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_FEATURE_AXIS,
            )
            if use_bias:
                self.bias = Parameter(
                    jnp.zeros((features,), dtype=dtype or jnp.float32),
                    sharding=bias_sharding,
                    axis_names=_FEATURE_AXIS,
                )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize ``x`` along the last axis and apply any affine transform."""
        scale = self.weight.value if self.elementwise_affine else None
        bias = self.bias.value if (self.elementwise_affine and self.use_bias) else None
        return F_layer_norm(x, scale=scale, bias=bias, axis=-1, eps=self.eps)


class RMSNorm(Module):
    """Root-mean-square normalization with an optional learned scale."""

    weight: Parameter

    def __init__(
        self,
        features: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Size of the trailing (normalization) axis.
            eps: Mean-of-squares floor for numerical stability.
            elementwise_affine: When ``True``, allocate a learned
                per-feature scale.
            dtype: Parameter dtype.
            sharding: Optional sharding for the scale.
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(
                jnp.ones((features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_FEATURE_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply RMSNorm along the last axis with the optional scale."""
        scale = self.weight.value if self.elementwise_affine else None
        return F_rms_norm(x, scale=scale, axis=-1, eps=self.eps)


class _BatchNormND(Module):
    """Shared N-D batch-normalization implementation (channels-last).

    Running statistics are kept in ``batch_stats``-kind
    :class:`~spectrax.Buffer` s; training-mode calls mutate them in
    place, which under transforms requires
    ``mutable="batch_stats"``.
    """

    _SPATIAL: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter
    running_mean: Buffer
    running_var: Buffer

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
        stats_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            num_features: Trailing channel count.
            eps: Variance floor.
            momentum: EMA factor for running statistics
                (``new = (1 - momentum) * old + momentum * batch``).
            affine: When ``True``, allocate learned per-channel scale
                and bias.
            dtype: Parameter / buffer dtype.
            sharding: Optional sharding for the scale.
            bias_sharding: Optional sharding for the bias.
            stats_sharding: Optional sharding for the running statistics buffers.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_features,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )
        self.running_mean = Buffer(
            jnp.zeros((num_features,), dtype=dtype or jnp.float32),
            kind="batch_stats",
            sharding=stats_sharding,
            axis_names=_CHANNEL_AXIS,
        )
        self.running_var = Buffer(
            jnp.ones((num_features,), dtype=dtype or jnp.float32),
            kind="batch_stats",
            sharding=stats_sharding,
            axis_names=_CHANNEL_AXIS,
        )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize ``x`` and update running stats when in training mode.

        Training mode computes batch statistics and blends them into
        :attr:`running_mean` / :attr:`running_var` via an exponential
        moving average. Evaluation mode uses the stored running stats
        verbatim.
        """
        xa = jnp.asarray(x)
        reduce_axes = tuple(range(xa.ndim - 1))
        if self.training:
            mean = jnp.mean(xa, axis=reduce_axes)
            var = jnp.var(xa, axis=reduce_axes)
            self.running_mean.value = (1 - self.momentum) * self.running_mean.value + self.momentum * mean
            self.running_var.value = (1 - self.momentum) * self.running_var.value + self.momentum * var
        else:
            mean = self.running_mean.value
            var = self.running_var.value
        inv = 1.0 / jnp.sqrt(var + self.eps)
        y = (xa - mean) * inv
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y


class BatchNorm1d(_BatchNormND):
    """BatchNorm for ``(N, L, C)`` channels-last inputs (1-D sequences)."""

    _SPATIAL: ClassVar[int] = 1


class BatchNorm2d(_BatchNormND):
    """BatchNorm for ``(N, H, W, C)`` channels-last inputs (2-D images)."""

    _SPATIAL: ClassVar[int] = 2


class GroupNorm(Module):
    """Group normalization (Wu & He, 2018).

    Normalizes ``(..., C)`` inputs by splitting the channel axis into
    ``num_groups`` groups and computing per-group mean/variance over
    ``(spatial, C/G)``.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            num_groups: Number of groups; must divide ``num_channels``.
            num_channels: Size of the trailing channel axis.
            eps: Variance floor.
            affine: When ``True``, allocate learned per-channel scale and bias.
            dtype: Parameter dtype.
            sharding: Optional sharding for the scale.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_channels,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_channels,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize within each channel group and optionally affine-scale."""
        xa = jnp.asarray(x)
        c = xa.shape[-1]
        if c != self.num_channels:
            raise ValueError(f"GroupNorm expected {self.num_channels} channels, got {c}")
        g = self.num_groups
        shape = (*xa.shape[:-1], g, c // g)
        xg = xa.reshape(shape)
        reduce_axes = (*range(1, xg.ndim - 2), xg.ndim - 1)
        mean = jnp.mean(xg, axis=reduce_axes, keepdims=True)
        var = jnp.var(xg, axis=reduce_axes, keepdims=True)
        xn = (xg - mean) / jnp.sqrt(var + self.eps)
        y = xn.reshape(xa.shape)
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y


class InstanceNorm(Module):
    """Instance normalization: per-sample, per-channel mean/variance over spatial axes."""

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_channels,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_channels,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize across spatial axes per sample and per channel."""
        xa = jnp.asarray(x)
        if xa.shape[-1] != self.num_channels:
            raise ValueError(f"InstanceNorm expected {self.num_channels} channels, got {xa.shape[-1]}")
        reduce_axes = tuple(range(1, xa.ndim - 1))
        mean = jnp.mean(xa, axis=reduce_axes, keepdims=True)
        var = jnp.var(xa, axis=reduce_axes, keepdims=True)
        y = (xa - mean) / jnp.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y
