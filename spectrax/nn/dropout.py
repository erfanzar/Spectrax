# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dropout layer."""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike
from ..core.module import Module
from ..functional.dropout import dropout as F_dropout
from ..rng.rngs import Rngs


class Dropout(Module):
    """Element-wise inverted dropout.

    Requires a :class:`~spectrax.Rngs` in training mode so each call
    draws a fresh mask from the ``"dropout"`` stream. Eval-mode calls
    (``deterministic=True`` or ``module.eval()``) pass the input
    through unchanged.
    """

    def __init__(self, rate: float = 0.5, *, rngs: Rngs | None = None) -> None:
        """Record the drop rate.

        Args:
            rate: Drop probability, must be in ``[0, 1)``.
            rngs: Optional :class:`~spectrax.Rngs` to use as a default in
                ``forward``.  Kept for API compatibility with callers that
                pass ``rngs`` to ``__init__``.

        Raises:
            ValueError: On out-of-range ``rate``.
        """
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        if rngs is not None:
            self.rngs = rngs

    def forward(
        self,
        x: ArrayLike | None = None,
        *,
        inputs: ArrayLike | None = None,
        rngs: Rngs | None = None,
        deterministic: bool | None = None,
        **_: object,
    ) -> Array:
        """Apply dropout to ``x``.

        Args:
            x: Input tensor.
            inputs: Alias for ``x`` (backward compatibility).
            rngs: :class:`~spectrax.Rngs` whose ``"dropout"`` stream
                feeds the mask.
            deterministic: Explicit override. When ``None`` the default
                is ``not self.training``.

        Returns:
            The input if deterministic / zero-rate, else a sampled mask
            applied with the inverted-dropout scale.

        Raises:
            RuntimeError: If dropout is active but no ``rngs`` is given.
            TypeError: If ``rngs`` is present but not an :class:`Rngs`.
        """
        if x is not None and inputs is not None:
            raise TypeError("Dropout.forward() got both 'x' and 'inputs'; pass only one.")
        if x is None and inputs is not None:
            x = inputs
        if x is None:
            raise TypeError("Dropout.forward() missing required argument: 'x'")
        if deterministic is None:
            deterministic = not self.training
        if deterministic or self.rate == 0.0:
            return jnp.asarray(x)
        if rngs is None:
            rngs = getattr(self, "rngs", None)
        if rngs is None:
            raise RuntimeError(
                "Dropout in training mode requires `rngs=...`. Pass rngs through "
                "forward(), or set deterministic=True / call eval()."
            )
        if not isinstance(rngs, Rngs):
            raise TypeError(f"rngs must be an Rngs, got {type(rngs).__name__}")
        return F_dropout(x, self.rate, key=rngs.key("dropout"), deterministic=False)
