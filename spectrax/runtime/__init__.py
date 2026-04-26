# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline-parallel runtimes and schedules.

Subpackages
-----------
* :mod:`spectrax.runtime.mpmd` — True MPMD runtime (per-rank separate HLOs).
* :mod:`spectrax.runtime.spmd` — SPMD runtime (single HLO via ``shard_map``).
* :mod:`spectrax.runtime.schedules` — Pipeline schedules (GPipe, 1F1B, …).
* :mod:`spectrax.runtime.types` — Shared types (``MpMdMesh``, ``PipelineStage``, …).
* :mod:`spectrax.runtime.primitives` — Shared primitives (``boundary``, ``auto_split``).
"""

from __future__ import annotations

from .mpmd import (
    sxcall,
    sxenter_loop,
    sxexit_loop,
    sxgrad,
    sxjit,
    sxloop,
    sxstage_iter,
    sxvalue_and_grad,
)
from .schedules import (
    Action,
    DualPipeV,
    Eager1F1B,
    FusedTask,
    GPipe,
    Interleaved1F1BPlusOne,
    InterleavedGPipe,
    InterleavedH1,
    KimiK2,
    Phase,
    Schedule,
    Std1F1B,
    ZeroBubbleH1,
    dualpipev_tasks,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)

__all__ = (
    "Action",
    "DualPipeV",
    "Eager1F1B",
    "FusedTask",
    "GPipe",
    "Interleaved1F1BPlusOne",
    "InterleavedGPipe",
    "InterleavedH1",
    "KimiK2",
    "Phase",
    "Schedule",
    "Std1F1B",
    "ZeroBubbleH1",
    "dualpipev_tasks",
    "fuse_1f1b_steady_state",
    "fuse_zerobubble_bwd_pair",
    "sxcall",
    "sxenter_loop",
    "sxexit_loop",
    "sxgrad",
    "sxjit",
    "sxloop",
    "sxstage_iter",
    "sxvalue_and_grad",
)
