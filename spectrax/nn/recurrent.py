# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recurrent cells and sequence wrappers.

Each cell implements a shape-stateless step ``(carry, x) -> (new_carry, y)``
and an :meth:`initial_carry` helper. :class:`RNN` scans a cell across a
time axis; :class:`Bidirectional` runs two cells forwards/backwards and
merges their outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.variable import Parameter
from ..functional.conv import conv as F_conv
from ..init import kaiming_uniform, orthogonal, zeros
from ..rng.rngs import Rngs, resolve_rngs

__all__ = [
    "RNN",
    "Bidirectional",
    "ConvLSTMCell",
    "GRUCell",
    "LSTMCell",
    "OptimizedLSTMCell",
    "RNNCellBase",
    "SimpleRNNCell",
]


class RNNCellBase(Module):
    """Abstract recurrent cell.

    Subclasses implement :meth:`forward` as a pure step
    ``(carry, x) -> (new_carry, y)`` and :meth:`initial_carry` returning
    the zero carry for a given batch shape.
    """

    num_feats: ClassVar[int] = 0

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> Any:
        """Return the zero carry for the given batch shape."""
        raise NotImplementedError

    def forward(self, carry: Any, x: ArrayLike) -> tuple[Any, Array]:
        """Run one step; return ``(new_carry, y)``."""
        raise NotImplementedError


class SimpleRNNCell(RNNCellBase):
    """Elman cell: ``h' = tanh(x @ W_xh + h @ W_hh + b)``, output ``y = h'``."""

    W_xh: Parameter
    W_hh: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        activation: Callable[[Array], Array] = jnp.tanh,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the simple RNN cell."""
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        self._activation = activation if callable(activation) else jnp.tanh
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_xh = Parameter(w_init(resolved.parameters, (in_features, hidden_features), dt), axis_names=("in", "out"))
        self.W_hh = Parameter(
            h_init(resolved.parameters, (hidden_features, hidden_features), dt), axis_names=("in", "out")
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (hidden_features,), dt), axis_names=("out",))

    def initial_carry(self, batch_shape: Sequence[int], *, dtype: DType | None = None) -> Array:
        """Zero hidden state of shape ``(*batch_shape, hidden_features)``."""
        dt = dtype or self.W_xh.dtype
        return jnp.zeros((*tuple(batch_shape), self.hidden_features), dtype=dt)

    def forward(self, carry: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
        """Advance the cell by one step."""
        h = jnp.asarray(carry)
        z = jnp.asarray(x) @ self.W_xh.value + h @ self.W_hh.value
        if self.use_bias:
            z = z + self.b.value
        h_new = self._activation(z)
        return h_new, h_new


class LSTMCell(RNNCellBase):
    """Standard LSTM cell with separate input/hidden matrices.

    Computes the four gates ``{i, f, g, o}`` as
    ``x @ W_x + h @ W_h + b`` with ``i, f, o`` under sigmoid and ``g``
    under tanh; carries ``(h, c)``.
    """

    W_x: Parameter
    W_h: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the LSTM cell."""
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_x = Parameter(
            w_init(resolved.parameters, (in_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        self.W_h = Parameter(
            h_init(resolved.parameters, (hidden_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (4 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Zero ``(h, c)`` tuple."""
        dt = dtype or self.W_x.dtype
        shape = (*tuple(batch_shape), self.hidden_features)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the LSTM by one step."""
        h, c = carry
        z = jnp.asarray(x) @ self.W_x.value + h @ self.W_h.value
        if self.use_bias:
            z = z + self.b.value
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class OptimizedLSTMCell(RNNCellBase):
    """LSTM cell with a single fused matmul over the concatenated input.

    Mathematically identical to :class:`LSTMCell`, but computes the gates
    as ``[x, h] @ W + b``, which on most accelerators is faster than
    two separate matmuls.
    """

    W: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the fused LSTM cell."""
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        self.W = Parameter(
            w_init(resolved.parameters, (in_features + hidden_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (4 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Zero ``(h, c)`` tuple."""
        dt = dtype or self.W.dtype
        shape = (*tuple(batch_shape), self.hidden_features)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the fused LSTM by one step."""
        h, c = carry
        xh = jnp.concatenate([jnp.asarray(x), h], axis=-1)
        z = xh @ self.W.value
        if self.use_bias:
            z = z + self.b.value
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class GRUCell(RNNCellBase):
    """Gated recurrent unit.

    Gates ``r`` and ``z`` are computed as ``sigmoid(x @ W_x + h @ W_h + b)``
    on their respective slices; the candidate ``n`` is
    ``tanh(x @ W_x_n + r * (h @ W_h_n))`` — the reset gate ``r`` modulates
    the hidden contribution to the candidate. The update rule is
    ``h' = (1 - z) * n + z * h``, where ``z = 1`` retains the previous
    hidden state.
    """

    W_x: Parameter
    W_h: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the GRU cell."""
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_x = Parameter(
            w_init(resolved.parameters, (in_features, 3 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        self.W_h = Parameter(
            h_init(resolved.parameters, (hidden_features, 3 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (3 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(self, batch_shape: Sequence[int], *, dtype: DType | None = None) -> Array:
        """Zero hidden state."""
        dt = dtype or self.W_x.dtype
        return jnp.zeros((*tuple(batch_shape), self.hidden_features), dtype=dt)

    def forward(self, carry: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
        """Advance the GRU by one step."""
        h = jnp.asarray(carry)
        x_part = jnp.asarray(x) @ self.W_x.value
        h_part = h @ self.W_h.value
        if self.use_bias:
            x_part = x_part + self.b.value
        rz_x, n_x = x_part[..., : 2 * self.hidden_features], x_part[..., 2 * self.hidden_features :]
        rz_h, n_h = h_part[..., : 2 * self.hidden_features], h_part[..., 2 * self.hidden_features :]
        rz = jax.nn.sigmoid(rz_x + rz_h)
        r, z = jnp.split(rz, 2, axis=-1)
        n = jnp.tanh(n_x + r * n_h)
        h_new = (1.0 - z) * n + z * h
        return h_new, h_new


class ConvLSTMCell(RNNCellBase):
    """2-D convolutional LSTM cell.

    Inputs are channels-last ``(N, H, W, C_in)``; the carry is a pair
    of feature maps ``(h, c)`` of shape ``(N, H, W, C_out)``. Gates are
    computed via a single convolution over the channel-concatenation
    ``[x, h]``.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the ConvLSTM cell."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if len(ks) != 2:
            raise ValueError("ConvLSTMCell kernel_size must be int or length-2 sequence")
        self.kernel_size = ks
        self.padding = padding
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        kshape = (*ks, in_channels + out_channels, 4 * out_channels)
        self.weight = Parameter(w_init(resolved.parameters, kshape, dt), axis_names=("kh", "kw", "in", "out"))
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(b_init(resolved.parameters, (4 * out_channels,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Zero ``(h, c)`` feature maps.

        ``batch_shape`` must be ``(N, H, W)``; the channel dim is added.
        """
        if len(batch_shape) != 3:
            raise ValueError("ConvLSTMCell batch_shape must be (N, H, W)")
        dt = dtype or self.weight.dtype
        shape = (*tuple(batch_shape), self.out_channels)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the ConvLSTM by one step."""
        h, c = carry
        xh = jnp.concatenate([jnp.asarray(x), h], axis=-1)
        z = F_conv(
            xh,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            padding=self.padding,
        )
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class RNN(Module):
    """Scan an :class:`RNNCellBase` across a time axis.

    By default the sequence axis is axis ``1`` (``(N, T, ...)``); set
    ``time_major=True`` to use axis ``0``. ``reverse=True`` processes the
    sequence right-to-left. ``return_carry=True`` returns a
    ``(ys, final_carry)`` tuple.
    """

    cell: RNNCellBase

    def __init__(
        self,
        cell: RNNCellBase,
        *,
        time_major: bool = False,
        reverse: bool = False,
        return_carry: bool = False,
    ) -> None:
        """Wrap a cell.

        Args:
            cell: The underlying cell whose step drives the scan.
            time_major: When ``True`` the sequence axis is axis 0,
                otherwise axis 1.
            reverse: Process the sequence right-to-left.
            return_carry: Return ``(ys, final_carry)`` rather than
                just ``ys``.
        """
        super().__init__()
        if not isinstance(cell, RNNCellBase):
            raise TypeError("RNN requires an RNNCellBase instance as cell")
        self.cell = cell
        self.time_major = time_major
        self.reverse = reverse
        self.return_carry = return_carry

    def _batch_shape(self, xs: Array) -> tuple[int, ...]:
        """Derive the batch shape (everything except time and trailing features/channels)."""
        if isinstance(self.cell, ConvLSTMCell):
            time_axis = 0 if self.time_major else 1
            shape = list(xs.shape)
            shape.pop(time_axis)
            return tuple(shape[:-1])
        time_axis = 0 if self.time_major else 1
        shape = list(xs.shape)
        shape.pop(time_axis)
        return tuple(shape[:-1])

    def forward(
        self,
        xs: ArrayLike,
        *,
        initial_carry: Any = None,
    ) -> Array | tuple[Array, Any]:
        """Run the cell over every step of the time axis."""
        xs = jnp.asarray(xs)
        time_axis = 0 if self.time_major else 1
        if not self.time_major:
            xs_t = jnp.swapaxes(xs, 0, 1)
        else:
            xs_t = xs
        if self.reverse:
            xs_t = xs_t[::-1]

        if initial_carry is None:
            batch_shape = self._batch_shape(xs)
            carry = self.cell.initial_carry(batch_shape, dtype=xs.dtype)
        else:
            carry = initial_carry

        cell = self.cell

        def step(c: Any, x: Array) -> tuple[Any, Array]:
            """Single ``lax.scan`` step: advance the recurrent carry through one time slice."""
            return cell.forward(c, x)

        final_carry, ys = jax.lax.scan(step, carry, xs_t)
        if self.reverse:
            ys = ys[::-1]
        if not self.time_major:
            ys = jnp.swapaxes(ys, 0, 1)
        _ = time_axis
        if self.return_carry:
            return ys, final_carry
        return ys


class Bidirectional(Module):
    """Run a forward and a reverse RNN and combine their outputs.

    ``merge_mode`` controls the combination:

    * ``"concat"`` (default) — concatenate along the trailing feature axis.
    * ``"sum"`` — element-wise sum.
    * ``"mul"`` — element-wise product.
    * ``"ave"`` — element-wise mean.
    """

    forward_rnn: RNN
    backward_rnn: RNN

    def __init__(
        self,
        forward_rnn: RNN,
        backward_rnn: RNN,
        *,
        merge_mode: str = "concat",
    ) -> None:
        """Wrap a forward and a (reverse-configured) backward :class:`RNN`."""
        super().__init__()
        if not isinstance(forward_rnn, RNN) or not isinstance(backward_rnn, RNN):
            raise TypeError("Bidirectional requires two RNN instances")
        if merge_mode not in {"concat", "sum", "mul", "ave"}:
            raise ValueError(f"Unknown merge_mode {merge_mode!r}")
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.merge_mode = merge_mode

    def forward(
        self,
        xs: ArrayLike,
        *,
        initial_carry: tuple[Any, Any] | None = None,
    ) -> Array:
        """Run both directions and merge the outputs."""
        fwd_c = None if initial_carry is None else initial_carry[0]
        bwd_c = None if initial_carry is None else initial_carry[1]
        saved_fwd = self.forward_rnn.return_carry
        saved_bwd = self.backward_rnn.return_carry
        saved_rev = self.backward_rnn.reverse
        try:
            object.__setattr__(self.forward_rnn, "return_carry", False)
            object.__setattr__(self.backward_rnn, "return_carry", False)
            object.__setattr__(self.backward_rnn, "reverse", True)
            ys_f = self.forward_rnn.forward(xs, initial_carry=fwd_c)
            ys_b = self.backward_rnn.forward(xs, initial_carry=bwd_c)
        finally:
            object.__setattr__(self.forward_rnn, "return_carry", saved_fwd)
            object.__setattr__(self.backward_rnn, "return_carry", saved_bwd)
            object.__setattr__(self.backward_rnn, "reverse", saved_rev)

        if self.merge_mode == "concat":
            return jnp.concatenate([ys_f, ys_b], axis=-1)
        if self.merge_mode == "sum":
            return ys_f + ys_b
        if self.merge_mode == "mul":
            return ys_f * ys_b
        return 0.5 * (ys_f + ys_b)
