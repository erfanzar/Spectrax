# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch-style pretty-print for :class:`~spectrax.Module` instances.

The format for each module is::

    ClassName(static_field=value, ...)(
      (child_name): <child repr>
      ...
    )

Leaf modules (no children) collapse to a single line. Shared submodules
are printed once; subsequent appearances render as ``<shared>`` to keep
the output finite.
"""

from __future__ import annotations

from typing import Any

from ..core.module import Module
from ..core.variable import Variable

__all__ = ["repr_module"]


_INDENT = "  "


def repr_module(module: Module) -> str:
    """Render ``module`` as a PyTorch-style multi-line tree."""
    seen: set[int] = set()
    return _render(module, seen, indent=0)


def _render(module: Module, seen: set[int], indent: int) -> str:
    """Recursive renderer. ``seen`` tracks already-rendered module ids."""
    cls = type(module).__name__
    head_args = _head_args(module)
    head = f"{cls}({head_args})"

    mid = id(module)
    if mid in seen:
        return f"{cls}(<shared>)"
    seen.add(mid)

    module_children = [(k, c) for k, c in _iter_children(module) if isinstance(c, Module)]
    if not module_children:
        return head

    pad = _INDENT * (indent + 1)
    lines: list[str] = [f"{cls}("] if not head_args else [f"{cls}({head_args})("]
    for key, child in module_children:
        sub = _render(child, seen, indent + 1)
        lines.append(f"{pad}({key}): {sub}")
    lines.append(f"{_INDENT * indent})")
    return "\n".join(lines)


def _head_args(module: Module) -> str:
    """Render static hyperparameters as ``key=value, ...`` — PyTorch style."""
    try:
        static = module._spx_static_fields()
    except Exception:
        return ""
    if not static:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in static.items())


def _iter_children(module: Module):
    """Yield ``(key, child)`` for every Module / Variable child in order."""
    try:
        yield from module._spx_graph_children()
    except Exception:
        return


def _format_key(key: Any) -> str:
    """Format a graph-child key PyTorch-style: ``(name)`` or ``(0)``."""
    return f"({key})"


def _render_variable(var: Variable) -> str:
    """One-line repr for a :class:`~spectrax.Variable` leaf."""
    try:
        shape = tuple(getattr(var, "shape", ()))
        dtype = getattr(var, "dtype", "")
        return f"{type(var).__name__}(kind={var.kind!r}, shape={shape}, dtype={dtype})"
    except Exception:
        return f"{type(var).__name__}(kind={var.kind!r})"


def _ascii_tree(module: Module, *, indent: int = 0) -> str:
    """Compatibility alias kept for legacy callers and tests."""
    return _render(module, set(), indent)
