# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-local default-:class:`Rngs` context manager.

Layers optionally fall back to a default :class:`~spectrax.Rngs` when
none is passed explicitly. ``spectrax.seed(n)`` pushes such a default
onto a thread-local stack so user code can opt into implicit RNG for
the duration of a block without losing explicit control elsewhere.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

from .rngs import Rngs

__all__ = ["default_rngs", "has_default_rngs", "seed"]


_state = threading.local()


def _stack() -> list[Rngs]:
    """Return the thread-local :class:`Rngs` stack, initializing it lazily."""
    s = getattr(_state, "stack", None)
    if s is None:
        s = []
        _state.stack = s
    return s


@contextlib.contextmanager
def seed(n: int | Rngs) -> Iterator[Rngs]:
    """Push an :class:`Rngs` as the thread-local default for this block.

    Args:
        n: Either an integer seed (wrapped as ``Rngs(n)``) or an
            existing :class:`Rngs` pushed verbatim.

    Yields:
        The :class:`Rngs` that is active for the duration of the block.
    """
    rngs = n if isinstance(n, Rngs) else Rngs(n)
    stack = _stack()
    stack.append(rngs)
    try:
        yield rngs
    finally:
        stack.pop()


def default_rngs() -> Rngs:
    """Return the current thread-local default :class:`Rngs`.

    Raises:
        RuntimeError: If no :func:`seed` context is active.
    """
    stack = _stack()
    if not stack:
        raise RuntimeError(
            "No default Rngs active. Pass rngs=... explicitly or wrap the block with `with spectrax.seed(n): ...`."
        )
    return stack[-1]


def has_default_rngs() -> bool:
    """Return ``True`` iff a :func:`seed` context is currently active."""
    return bool(_stack())
