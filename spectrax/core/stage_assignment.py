# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Construction-time pipeline stage ownership hints.

``assign_stage(total=..., current=...)`` lets model construction stamp a
coarse-grained "where in the logical layer order was this variable
created?" hint onto every :class:`~spectrax.Variable` allocated inside
the scope. MPMD runtimes can then resolve that hint against the active
pipeline width and place the variable on the owning stage sub-mesh.

Example::

    for i in range(32):
        with assign_stage(total=32, current=i):
            blocks.append(MyBlock(...))

On a 4-stage mesh, layers 0-7 map to stage 0, 8-15 to stage 1, and so
on. The hint is stored in ``Variable.metadata["pipeline_stage"]`` as the
hashable tuple ``(current, total)``.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from .context import get as _scope_get
from .context import scope

__all__ = [
    "PIPELINE_STAGE_METADATA_KEY",
    "assign_stage",
    "current_stage_assignment",
    "metadata_stage_assignment",
    "resolve_stage_rank",
]


_PIPELINE_STAGE_SCOPE_KEY = "_spx_pipeline_stage_assignment"
PIPELINE_STAGE_METADATA_KEY = "pipeline_stage"


def _normalize_assignment(current: Any, total: Any) -> tuple[int, int]:
    """Validate and normalize a ``(current, total)`` stage hint."""
    cur = operator.index(current)
    tot = operator.index(total)
    if tot <= 0:
        raise ValueError(f"assign_stage(...): total must be >= 1; got {tot}.")
    if not 0 <= cur < tot:
        raise ValueError(
            f"assign_stage(...): current must satisfy 0 <= current < total; got current={cur}, total={tot}."
        )
    return cur, tot


@contextmanager
def assign_stage(*, total: int, current: int) -> Iterator[None]:
    """Stamp subsequently-created variables with a pipeline stage hint.

    ``current`` is a zero-based logical position inside ``total`` slots,
    not a literal pipeline rank. The eventual physical stage is derived
    later from the active MPMD mesh width via :func:`resolve_stage_rank`.
    """
    assignment = _normalize_assignment(current=current, total=total)
    with scope(**{_PIPELINE_STAGE_SCOPE_KEY: assignment}):
        yield


def current_stage_assignment() -> tuple[int, int] | None:
    """Return the innermost active ``assign_stage`` hint, if any."""
    assignment = _scope_get(_PIPELINE_STAGE_SCOPE_KEY, None)
    if assignment is None:
        return None
    return _normalize_assignment(*assignment)


def metadata_stage_assignment(metadata: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """Extract a normalized stage hint from variable metadata."""
    if not metadata or PIPELINE_STAGE_METADATA_KEY not in metadata:
        return None
    raw = metadata[PIPELINE_STAGE_METADATA_KEY]
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ValueError(
            f"Variable metadata[{PIPELINE_STAGE_METADATA_KEY!r}] must be a 2-tuple ``(current, total)``; got {raw!r}."
        )
    return _normalize_assignment(*raw)


def resolve_stage_rank(assignment: tuple[int, int] | None, mpmd_dim: int) -> int | None:
    """Resolve a logical ``(current, total)`` hint to a physical MPMD rank."""
    if assignment is None:
        return None
    if mpmd_dim < 1:
        raise ValueError(f"mpmd_dim must be >= 1; got {mpmd_dim}.")
    current, total = _normalize_assignment(*assignment)
    return min(mpmd_dim - 1, (current * mpmd_dim) // total)
