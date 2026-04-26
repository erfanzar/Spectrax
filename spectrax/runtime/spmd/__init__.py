# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPMD pipeline runtime.

Single HLO program with ``shard_map`` / manual axis. Faster on TPU
but requires homogeneous stages.
"""

from __future__ import annotations

from .api import pipeline_call, pipeline_step
from .hybrid import hybrid_linear_run
from .runtime import spmd_run
from .shard_map import make_scheduled_body

__all__ = [
    "hybrid_linear_run",
    "make_scheduled_body",
    "pipeline_call",
    "pipeline_step",
    "spmd_run",
]
