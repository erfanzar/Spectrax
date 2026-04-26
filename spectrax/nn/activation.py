# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Activation layers.

Each class is a thin :class:`~spectrax.Module` wrapper over the
corresponding function in :mod:`spectrax.functional.activation`. Use
these when you want an activation as a drop-in layer (e.g. inside a
:class:`~spectrax.nn.Sequential`); otherwise call the functional form
directly from your :meth:`forward`.
"""

from __future__ import annotations

from ..core._typing import Array, ArrayLike
from ..core.module import Module
from ..functional import activation as F


class ReLU(Module):
    """:math:`\\mathrm{ReLU}(x) = \\max(0, x)`."""

    def __init__(self) -> None:
        """No parameters."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.relu` to ``x``."""
        return F.relu(x)


class GELU(Module):
    """Gaussian error linear unit.

    Exact by default; set ``approximate=True`` for the tanh approximation.
    """

    def __init__(self, approximate: bool = False) -> None:
        """Record the ``approximate`` flag as a static field.

        Args:
            approximate: When ``True``, use the tanh-based approximation.
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.gelu` honoring :attr:`approximate`."""
        return F.gelu(x, approximate=self.approximate)


class SiLU(Module):
    """Sigmoid-weighted linear unit: ``x * sigmoid(x)``."""

    def __init__(self) -> None:
        """No parameters."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.silu` to ``x``."""
        return F.silu(x)


class Tanh(Module):
    """Hyperbolic tangent."""

    def __init__(self) -> None:
        """No parameters."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.tanh` to ``x``."""
        return F.tanh(x)


class Sigmoid(Module):
    """Logistic sigmoid."""

    def __init__(self) -> None:
        """No parameters."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.sigmoid` to ``x``."""
        return F.sigmoid(x)
