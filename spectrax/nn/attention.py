# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head attention layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Buffer
from ..functional.attention import scaled_dot_product_attention
from ..rng.rngs import Rngs
from .linear import Linear


class MultiheadAttention(Module):
    """Standard multi-head attention with independent Q/K/V projections
    and an output projection.

    Inputs are ``(..., seq, embed_dim)``; heads are split internally
    into ``(..., num_heads, seq, head_dim)`` and re-merged after
    attention.
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    cache_k: Buffer
    cache_v: Buffer
    cache_index: Buffer

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        use_bias: bool = True,
        dropout: float = 0.0,
        decode: bool = False,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        qkv_sharding: Sharding | AxisNames | None = None,
        out_sharding: Sharding | AxisNames | None = None,
        qkv_bias_sharding: Sharding | AxisNames | None = None,
        out_bias_sharding: Sharding | AxisNames | None = None,
        cache_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            embed_dim: Model dimension. Must be divisible by
                ``num_heads``.
            num_heads: Number of attention heads.
            use_bias: Whether Q / K / V / out projections carry biases.
            dropout: Attention-weight dropout probability.
            rngs: PRNG source for parameter initialization.
            dtype: Parameter dtype.
            param_dtype: Optional parameter storage dtype, forwarded to
                the projection linears.
            qkv_sharding: Optional sharding for the Q / K / V weights.
            out_sharding: Optional sharding for the output projection weight.
            qkv_bias_sharding: Optional sharding for the Q / K / V biases.
            out_bias_sharding: Optional sharding for the output projection bias.
            cache_sharding: Optional sharding for the decode K/V caches.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.decode = decode
        self.cache_sharding = cache_sharding
        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=qkv_sharding,
            bias_sharding=qkv_bias_sharding,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=out_sharding,
            bias_sharding=out_bias_sharding,
        )

    def init_cache(
        self,
        batch_shape: tuple[int, ...],
        max_length: int,
        *,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the ``cache_k`` / ``cache_v`` / ``cache_index`` buffers.

        Shapes: ``(*batch_shape, num_heads, max_length, head_dim)`` for
        the K/V caches; a scalar ``int32`` for the write index.
        """
        dt = dtype or jnp.float32
        shape = (*batch_shape, self.num_heads, max_length, self.head_dim)
        self.cache_k = Buffer(jnp.zeros(shape, dtype=dt), kind="cache", sharding=self.cache_sharding)
        self.cache_v = Buffer(jnp.zeros(shape, dtype=dt), kind="cache", sharding=self.cache_sharding)
        self.cache_index = Buffer(jnp.zeros((), dtype=jnp.int32), kind="cache")
        self.max_length = max_length

    def _split_heads(self, x: Array) -> Array:
        """Reshape ``(..., seq, embed_dim)`` into
        ``(..., num_heads, seq, head_dim)``.
        """
        *batch, seq, _ = x.shape
        return x.reshape(*batch, seq, self.num_heads, self.head_dim).swapaxes(-2, -3)

    def _merge_heads(self, x: Array) -> Array:
        """Inverse of :meth:`_split_heads`."""
        *batch, _, seq, _ = x.shape
        x = x.swapaxes(-2, -3)
        return x.reshape(*batch, seq, self.embed_dim)

    def forward(
        self,
        q: ArrayLike,
        k: ArrayLike | None = None,
        v: ArrayLike | None = None,
        *,
        mask: ArrayLike | None = None,
        is_causal: bool = False,
        rngs: Rngs | None = None,
        **_: object,
    ) -> Array:
        """Project, attend, merge, and project.

        Args:
            q: Query tensor.
            k: Key tensor; defaults to ``q`` (self-attention).
            v: Value tensor; defaults to ``k``.
            mask: Optional attention mask broadcastable to
                ``(..., seq_q, seq_k)``.
            is_causal: When ``True`` adds a lower-triangular mask on
                top of ``mask``.
            rngs: Required when :attr:`dropout` is positive and
                :attr:`training` is ``True``.
        """
        if k is None:
            k = q
        if v is None:
            v = k
        qp = self._split_heads(self.q_proj(q))
        kp = self._split_heads(self.k_proj(k))
        vp = self._split_heads(self.v_proj(v))
        drop = self.dropout if self.training else 0.0
        key = rngs.key("dropout") if (drop > 0.0 and rngs is not None) else None

        if self.decode:
            if not hasattr(self, "cache_k"):
                raise RuntimeError("decode=True requires init_cache(batch_shape, max_length) first.")
            idx = self.cache_index.value
            step = kp.shape[-2]
            ck = jax.lax.dynamic_update_slice_in_dim(self.cache_k.value, kp, idx, axis=-2)
            cv = jax.lax.dynamic_update_slice_in_dim(self.cache_v.value, vp, idx, axis=-2)
            self.cache_k.value = ck
            self.cache_v.value = cv
            self.cache_index.value = idx + step
            max_len = self.cache_k.value.shape[-2]
            positions = jnp.arange(max_len)
            q_pos = idx + jnp.arange(step)
            decode_mask = positions[None, :] <= q_pos[:, None]
            if mask is None:
                mask_in = decode_mask
            else:
                mask_in = jnp.logical_and(mask, decode_mask)
            out = scaled_dot_product_attention(
                qp,
                ck,
                cv,
                mask=mask_in,
                dropout_rate=drop,
                key=key,
                is_causal=False,
            )
        else:
            out = scaled_dot_product_attention(
                qp,
                kp,
                vp,
                mask=mask,
                dropout_rate=drop,
                key=key,
                is_causal=is_causal,
            )
        out = self._merge_heads(out)
        return self.out_proj(out)


class CausalSelfAttention(Module):
    """Self-attention with a lower-triangular causal mask.

    Convenience wrapper around :class:`MultiheadAttention` with
    ``is_causal=True``.
    """

    attn: MultiheadAttention

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        qkv_sharding: Sharding | AxisNames | None = None,
        out_sharding: Sharding | AxisNames | None = None,
        qkv_bias_sharding: Sharding | AxisNames | None = None,
        out_bias_sharding: Sharding | AxisNames | None = None,
        cache_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the wrapped :class:`MultiheadAttention` instance."""
        super().__init__()
        self.attn = MultiheadAttention(
            embed_dim,
            num_heads,
            use_bias=use_bias,
            dropout=dropout,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            qkv_sharding=qkv_sharding,
            out_sharding=out_sharding,
            qkv_bias_sharding=qkv_bias_sharding,
            out_bias_sharding=out_bias_sharding,
            cache_sharding=cache_sharding,
        )

    def forward(self, x: ArrayLike, *, rngs: Rngs | None = None, **_: object) -> Array:
        """Call the wrapped attention with ``is_causal=True``."""
        return self.attn(x, is_causal=True, rngs=rngs)
