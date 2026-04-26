# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPMD pipeline runtime.

Compiles each pipeline step into a single HLO program. The pipeline
axis of the mesh is made *manual* under :func:`jax.shard_map` so XLA
keeps each rank's branch of the per-rank ``lax.switch`` distinct
after partitioning; cross-rank transport uses :func:`jax.lax.ppermute`
inside the body. This collapses what would otherwise be N per-rank
jits into one shared compile, which is much faster on TPU and avoids
the dispatch-per-rank overhead the MPMD path has on small steps.

Public surface
--------------
* :func:`pipeline_call` — generic SPMD pipeline entry. Supports
  ``mode="forward"`` and ``mode="train"``, both flat and virtual-stage
  schedules. Falls back to a heterogeneous multi-jit MPMD path when
  the stages are not parameter-homogeneous.
* :func:`pipeline_step` — thin convenience wrapper over
  :func:`spmd_run` for use with :class:`PipelineSequential` modules.
* :func:`spmd_run` — scan-free runtime for :class:`PipelineSequential`:
  shards stacked params along the pipeline axis and lets XLA route
  forward/backward across stages based on the placement.
* :func:`make_scheduled_body` — turn any
  :class:`~spectrax.runtime.schedules.Schedule` into a ``shard_map``
  body suitable for use under :func:`pipeline_call(schedule=...)`.
* :func:`hybrid_linear_run` — special-case path for models with a
  homogeneous middle and heterogeneous edges (embed + blocks +
  head): runs the middle through SPMD and the edges through MPMD
  per-stage jits.
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
