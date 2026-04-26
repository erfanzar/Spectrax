# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simple feed-forward MLP block."""

from __future__ import annotations

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..functional.activation import gelu, relu, silu
from ..rng.rngs import Rngs
from .dropout import Dropout
from .linear import Linear


class MLPBlock(Module):
    """Two-layer feed-forward block ``Linear -> activation -> dropout -> Linear``.

    A compact drop-in for the feed-forward half of a transformer block.
    Defaults to ``4 * features`` hidden and GELU activation.
    """

    fc1: Linear
    fc2: Linear
    drop: Dropout

    def __init__(
        self,
        features: int,
        hidden_features: int | None = None,
        *,
        out_features: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        fc1_sharding: Sharding | AxisNames | None = None,
        fc2_sharding: Sharding | AxisNames | None = None,
        fc1_bias_sharding: Sharding | AxisNames | None = None,
        fc2_bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Input feature count (and default for
                ``out_features``).
            hidden_features: Hidden size; defaults to ``4 * features``.
            out_features: Output feature count; defaults to
                ``features``.
            dropout: Dropout applied between the two linear layers.
            activation: Name of the activation: ``"gelu"`` / ``"relu"``
                / ``"silu"``.
            rngs: PRNG source for parameter initialization.
            dtype: Parameter dtype.
            fc1_sharding: Optional sharding for the first linear weight.
            fc2_sharding: Optional sharding for the second linear weight.
            fc1_bias_sharding: Optional sharding for the first linear bias.
            fc2_bias_sharding: Optional sharding for the second linear bias.
        """
        super().__init__()
        hidden = hidden_features if hidden_features is not None else 4 * features
        out = out_features if out_features is not None else features
        self.features = features
        self.hidden_features = hidden
        self.out_features = out
        self.activation = activation
        self.fc1 = Linear(
            features,
            hidden,
            rngs=rngs,
            dtype=dtype,
            sharding=fc1_sharding,
            bias_sharding=fc1_bias_sharding,
        )
        self.fc2 = Linear(
            hidden,
            out,
            rngs=rngs,
            dtype=dtype,
            sharding=fc2_sharding,
            bias_sharding=fc2_bias_sharding,
        )
        self.drop = Dropout(dropout)

    def forward(self, x: ArrayLike, *, rngs: Rngs | None = None, **_: object) -> Array:
        """Thread ``x`` through ``fc1 -> activation -> dropout -> fc2``.

        Raises:
            ValueError: If :attr:`activation` is not a recognized name.
        """
        y = self.fc1(x)
        if self.activation == "gelu":
            y = gelu(y)
        elif self.activation == "relu":
            y = relu(y)
        elif self.activation == "silu":
            y = silu(y)
        else:
            raise ValueError(f"Unknown activation: {self.activation!r}")
        y = self.drop(y, rngs=rngs)
        return self.fc2(y)
