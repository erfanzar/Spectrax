# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embedding lookup table with an output-head (:meth:`Embed.attend`) mode."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..init import normal
from ..rng.rngs import Rngs, resolve_rngs


class Embed(Module):
    """Lookup table mapping integer ids to dense vectors.

    :meth:`lookup` returns per-id embeddings; :meth:`attend` computes
    logits against the transpose of the table, which is the standard
    setup for weight-tying the input embedding and the output
    classification head.
    """

    weight: Parameter

    def __init__(
        self,
        num_embeddings: int | None,
        features: int,
        *,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        w_init: Initializer | None = None,
        sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the table.

        Args:
            num_embeddings: Vocabulary size.  ``None`` defers inference to
                the first forward pass (uses ``ids.max() + 1``).
            features: Embedding dimension.
            rngs: PRNG source for initialization.
            dtype: Parameter dtype.
            param_dtype: Alias for *dtype*.
            w_init: Weight initializer (default:
                :func:`~spectrax.init.normal` with ``stddev=1``).
            sharding: Optional sharding for the embedding table.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.features = features
        resolved = resolve_rngs(rngs)
        init = w_init or normal(stddev=1.0)
        weight_dtype = param_dtype or dtype or jnp.float32
        if num_embeddings is None:
            self.weight = DeferredParameter(
                (None, features),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=("vocab", "embed"),
            )
        else:
            self.weight = Parameter(
                init(resolved.parameters, (num_embeddings, features), weight_dtype),
                sharding=sharding,
                axis_names=("vocab", "embed"),
            )

    def lookup(self, ids: ArrayLike) -> Array:
        """Return the embedding vectors for integer ``ids``."""
        ids_arr = jnp.asarray(ids)
        if self.num_embeddings is None:
            self._spx_guard_not_in_transform("DeferredParameter materialization")
            num_embeddings = int(ids_arr.max()) + 1
            self._resolve_deferred(self.weight, (num_embeddings, self.features))
            self.num_embeddings = num_embeddings
        return self.weight.value[ids_arr]

    def attend(self, q: ArrayLike) -> Array:
        """Compute logits ``q @ W.T`` against the embedding table."""
        return jnp.asarray(q) @ self.weight.value.T

    def forward(self, ids: ArrayLike, **_: object) -> Array:
        """Shortcut for :meth:`lookup`."""
        return self.lookup(ids)
