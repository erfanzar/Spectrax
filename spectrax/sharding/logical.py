# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logical axis-name -> mesh axis rules, managed as a thread-local stack."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping, Sequence

__all__ = ["current_axis_rules", "logical_axis_rules"]


_STACK: threading.local = threading.local()


def _get_stack() -> list[dict[str, str | None]]:
    """Return the thread-local stack of axis-rule mappings."""
    s = getattr(_STACK, "stack", None)
    if s is None:
        s = []
        _STACK.stack = s
    return s


@contextlib.contextmanager
def logical_axis_rules(rules: Sequence[tuple[str, str | None]]) -> Iterator[None]:
    """Push a logical -> mesh axis mapping onto the stack for the body.

    Inside the ``with`` block, :func:`current_axis_rules` returns the
    merged mapping (inner rules override outer). On exit the pushed
    frame is popped.
    """
    mapping = dict(rules)
    stack = _get_stack()
    stack.append(mapping)
    try:
        yield
    finally:
        stack.pop()


def current_axis_rules() -> Mapping[str, str | None]:
    """Return the merged logical -> mesh axis mapping currently in effect."""
    merged: dict[str, str | None] = {}
    for frame in _get_stack():
        merged.update(frame)
    return merged
