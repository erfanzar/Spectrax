# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Identity layer."""

from __future__ import annotations

from typing import Any

from ..core.module import Module


class Identity(Module):
    """Pass-through layer: returns its first argument unchanged.

    Useful as a placeholder in :class:`~spectrax.nn.Sequential` chains or
    a no-op branch in ``if``-gated architectures.
    """

    def __init__(self) -> None:
        """No parameters; initialization is trivial."""
        super().__init__()

    def forward(self, x: Any, **_: Any) -> Any:
        """Return ``x`` unmodified."""
        return x
