# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`PipelineStage` dataclass and helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["PipelineStage", "_is_empty_state"]


def _is_empty_state(s: Any) -> bool:
    """Detect the "no state" sentinel.

    Treats ``()`` and ``None`` identically so users can pass either
    without tripping the runtime.
    """
    return s is None or s == ()


@dataclass
class PipelineStage:
    """A single rank's worth of pipeline work.

    ``fn(parameters, state, x) -> (y, new_state)``. Use ``()`` (or ``None``)
    for ``init_state`` on stateless stages.
    """

    fn: Callable[[Any, Any, Any], tuple[Any, Any]]
    parameters: Any
    init_state: Any = ()
