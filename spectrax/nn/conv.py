# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolution layers — :class:`Conv1d`, :class:`Conv2d`, :class:`Conv3d`.

All three subclass a shared N-D base that stores the kernel and bias and
delegates to :func:`spectrax.functional.conv`. Inputs are channels-last:
``(N, *spatial, C_in)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..functional.conv import PaddingSpec
from ..functional.conv import conv as F_conv
from ..functional.conv import conv_transpose as F_conv_transpose
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs


def _tup(x: int | Sequence[int], n: int) -> tuple[int, ...]:
    """Broadcast an int to an ``n``-tuple or validate a length-``n`` sequence."""
    if isinstance(x, int):
        return (x,) * n
    t = tuple(x)
    if len(t) != n:
        raise ValueError(f"Expected length-{n} tuple, got {t}")
    return t


class _ConvND(Module):
    """Shared N-D convolution implementation.

    Subclasses set :attr:`_N` to the spatial rank and inherit the weight
    shape ``(*kernel_size, in_channels // groups, out_channels)``.
    """

    _N: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the N-D convolution.

        Args:
            in_channels: Input channel count.  ``None`` defers shape
                inference to the first forward pass.
            out_channels: Output channel count.
            kernel_size: Per-axis kernel size (int broadcasts).
            stride: Per-axis stride.
            padding: Padding mode or per-axis ``(lo, hi)`` pairs.
            dilation: Per-axis kernel dilation.
            groups: Group count for grouped / depthwise convolutions.
            use_bias: When ``True``, add a bias.
            rngs: PRNG source.
            dtype: Parameter dtype.
            param_dtype: Alias for *dtype*.
            sharding: Optional sharding for the kernel.
            bias_sharding: Optional sharding for the bias.
        """
        super().__init__()
        n = self._N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tup(kernel_size, n)
        self.stride = _tup(stride, n)
        self.padding = padding if isinstance(padding, str) else tuple(tuple(p) for p in padding)
        self.dilation = _tup(dilation, n)
        self.groups = groups
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_channels is None:
            self.weight = DeferredParameter(
                (*self.kernel_size, None, out_channels),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        else:
            kshape = (*self.kernel_size, in_channels // groups, out_channels)
            self.weight = Parameter(
                init(resolved.parameters, kshape, weight_dtype),
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        if use_bias:
            self.bias = Parameter(
                zeros(resolved.parameters, (out_channels,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.conv` with the stored parameters."""
        xa = jnp.asarray(x)
        if self.in_channels is None:
            in_channels = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (*self.kernel_size, in_channels // self.groups, self.out_channels))
            self.in_channels = in_channels
        return F_conv(
            xa,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d(_ConvND):
    """1-D convolution over ``(N, L, C)`` channels-last inputs."""

    _N: ClassVar[int] = 1


class Conv2d(_ConvND):
    """2-D convolution over ``(N, H, W, C)`` channels-last inputs."""

    _N: ClassVar[int] = 2


class Conv3d(_ConvND):
    """3-D convolution over ``(N, D, H, W, C)`` channels-last inputs."""

    _N: ClassVar[int] = 3


class Conv(_ConvND):
    """N-D convolution with rank inferred from *kernel_size*.

    This mirrors ``flax.nnx.Conv``: a single int ``kernel_size`` produces a
    1-D convolution, while an ``n``-tuple produces an ``n``-D convolution.
    All other arguments are forwarded unchanged.
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Construct an N-D conv layer; ``N`` is inferred from ``kernel_size``.

        An ``int`` ``kernel_size`` selects a 1-D convolution and stamps
        ``self._N = 1``; a tuple of ``n`` ints selects an ``n``-D
        convolution. All other arguments are forwarded to the base
        :class:`_ConvND`: see its constructor for ``stride``, ``padding``,
        ``dilation``, ``groups``, ``use_bias``, RNG, dtype, and sharding
        semantics.
        """
        if isinstance(kernel_size, int):
            object.__setattr__(self, "_N", 1)
        else:
            object.__setattr__(self, "_N", len(tuple(kernel_size)))
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
            bias_sharding=bias_sharding,
        )


class _ConvTransposeND(Module):
    """Shared N-D transposed-convolution implementation."""

    _N: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        n = self._N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tup(kernel_size, n)
        self.stride = _tup(stride, n)
        self.padding = padding if isinstance(padding, str) else tuple(tuple(p) for p in padding)
        self.dilation = _tup(dilation, n)
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_channels is None:
            self.weight = DeferredParameter(
                (*self.kernel_size, None, out_channels),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        else:
            kshape = (*self.kernel_size, in_channels, out_channels)
            self.weight = Parameter(
                init(resolved.parameters, kshape, weight_dtype),
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        if use_bias:
            self.bias = Parameter(
                zeros(resolved.parameters, (out_channels,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`spectrax.functional.conv_transpose`."""
        xa = jnp.asarray(x)
        if self.in_channels is None:
            in_channels = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (*self.kernel_size, in_channels, self.out_channels))
            self.in_channels = in_channels
        return F_conv_transpose(
            xa,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class ConvTranspose1d(_ConvTransposeND):
    """Transposed 1-D convolution (channels-last)."""

    _N: ClassVar[int] = 1


class ConvTranspose2d(_ConvTransposeND):
    """Transposed 2-D convolution (channels-last)."""

    _N: ClassVar[int] = 2


class ConvTranspose3d(_ConvTransposeND):
    """Transposed 3-D convolution (channels-last)."""

    _N: ClassVar[int] = 3
