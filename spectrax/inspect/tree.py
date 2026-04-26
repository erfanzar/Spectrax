# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structural inspection helpers."""

from __future__ import annotations

from typing import Any

from ..core.graph import export
from ..core.graph import tree_state as _tree_state
from ..core.module import Module
from ..core.state import State

__all__ = ["paths_and_shapes", "tree_state"]


def tree_state(module: Module) -> State:
    """Return the :class:`~spectrax.State` half of :func:`spectrax.export`.

    Convenience wrapper for users who only want the state tree.
    """
    return _tree_state(module)


def paths_and_shapes(module: Module) -> list[tuple[str, str, tuple[int, ...], Any]]:
    """List every leaf in ``module`` as ``(collection, path, shape, dtype)``.

    Sorted by ``(collection, path)`` for stable diagnostic output.
    """
    _gdef, state = export(module)
    out: list[tuple[str, str, tuple[int, ...], Any]] = []
    for c, p, v in state.items():
        shape = tuple(getattr(v, "shape", ()))
        dtype = getattr(v, "dtype", None)
        out.append((c, p, shape, dtype))
    return sorted(out, key=lambda r: (r[0], r[1]))
