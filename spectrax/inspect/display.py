# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`display` — treescope-based rich rendering of a module."""

from __future__ import annotations

from ..core.module import Module
from .repr import repr_module

__all__ = ["display"]


def display(module: Module, *, roundtrip: bool = False) -> None:
    """Render ``module`` via treescope when available, else the text repr.

    In a notebook this produces interactive HTML with collapsible nodes;
    on a plain TTY it prints the :func:`~spectrax.inspect.repr.repr_module`
    text. Returns ``None``.
    """
    try:
        import treescope

        treescope.display(module)
    except Exception:
        print(repr_module(module))
    _ = roundtrip
