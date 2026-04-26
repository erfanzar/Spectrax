# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DualPipe-V schedule and per-rank task builder (DeepSeek-V3).

DualPipe-V is the V-shaped bidirectional schedule used by DeepSeek-V3.
Every physical rank hosts two virtual stages: one that runs in the
forward direction (rank ``r`` -> ``r+1``) and one that runs in the
reverse direction (rank ``r+1`` -> ``r``). The two virtual stages
mirror each other so each microbatch visits every rank twice before
hitting the loss.

This module provides:

* :class:`DualPipeV` — the :class:`Schedule` subclass that emits the
  V-shape grid for any spectrax pipeline runtime.
* :func:`dualpipev_tasks` — a per-rank, eight-section task list that
  matches DeepSeek's reference ``dualpipev.py``. Useful for building
  custom executors or experimental runners outside spectrax's grid
  representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Action, FusedTask, Phase, Schedule
from .one_f_one_b import Std1F1B


@dataclass
class DualPipeV(Schedule):
    """DualPipe-V: V-shaped bidirectional pipeline (DeepSeek).

    Every physical rank hosts **two** virtual stages in a V topology:
    rank ``r`` owns logical stage ``r`` (forward direction) and
    logical stage ``2n - 1 - r`` (reverse direction). Activations flow
    through physical ranks ``0 -> n-1``, then bounce back ``n-1 -> 0``,
    so each microbatch visits every rank twice before the loss.

    Pros:

    * Halves the pipeline bubble vs :class:`Std1F1B` at the same peak
      activation memory (the mirrored stage fills what would be idle).
    * End-to-end latency comparable to :class:`InterleavedH1` but
      without the cross-rank ppermute-per-virtual-stage cost — the
      second virtual stage is adjacent to the first on the same rank.

    Cons:

    * Requires ``n_stages`` pipeline ranks for ``2 * n_stages``
      logical stages — callers must structure the model as
      ``2 * n_stages`` :class:`PipelineSequential` entries.
    * Currently emits a naive V-shape (no zero-bubble W-grad
      filling); for finer overlap use :class:`ZeroBubbleH1`.

    Reference: DeepSeek-V3 technical report; DeepSeek
    ``dualpipev.py``.

    Attributes:
        microbatches: see :class:`Schedule`.
    """

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the V-shape grid.

        Implementation: build a dependency-correct 1F1B schedule over
        ``2 * n_stages`` *logical* stages, then fold each logical
        stage onto its physical rank using the V mapping::

            logical l  ->  (phys, virt) = (l, 0)              if l < n
                       ->  (2n - 1 - l, 1)                    if l >= n

        When two virtual stages on the same physical rank want to run
        at the same time step, we extend the grid with an extra row
        to serialize them (mirrors :class:`InterleavedH1`).
        """
        n = n_stages
        m = self.microbatches
        n_logical = 2 * n

        logical = Std1F1B(m).build(n_logical)

        grid: list[list[Action | None]] = []
        for row in logical:
            new_row: list[Action | None] = [None] * n
            for l_stage, action in enumerate(row):
                if action is None:
                    continue
                if l_stage < n:
                    phys, virt = l_stage, 0
                else:
                    phys, virt = 2 * n - 1 - l_stage, 1
                if new_row[phys] is not None:
                    grid.append(new_row)
                    new_row = [None] * n
                new_row[phys] = Action(action.phase, action.microbatch, virt)
            grid.append(new_row)

        while grid and all(c is None for c in grid[-1]):
            grid.pop()
        return grid

    def virtual_stages_per_rank(self) -> int:
        """Always ``2`` for the V-shape topology (forward and reverse virtuals)."""
        return 2

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Map ``(rank, virt)`` to its logical stage under the V shape.

        ``virt == 0`` traces ranks ``0..n-1`` as logical stages
        ``0..n-1`` (forward direction); ``virt == 1`` traces ranks
        ``n-1..0`` as logical stages ``n..2n-1`` (reverse direction).

        Args:
            rank: Physical rank index in ``[0, n_stages)``.
            virt: Virtual-stage index, ``0`` (forward leg) or ``1``
                (reverse leg).
            n_stages: Number of physical pipeline ranks.

        Returns:
            Logical-stage index in ``[0, 2 * n_stages)``.
        """
        return rank if virt == 0 else 2 * n_stages - 1 - rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int):
        """Return the ``(rank, virt)`` of the next logical stage, or ``None``.

        Walks the V-shape: in the forward leg the next stage is the
        next physical rank; at the apex (logical ``n - 1``) the path
        bounces and starts walking ranks back down on virtual stage 1.

        Args:
            rank: Current physical rank.
            virt: Current virtual-stage index.
            n_stages: Number of physical pipeline ranks.

        Returns:
            Downstream ``(rank, virt)`` or ``None`` if this is the
            terminal logical stage (``logical == 2 * n_stages - 1``).
        """
        current = self.logical_at(rank, virt, n_stages)
        nxt = current + 1
        if nxt >= 2 * n_stages:
            return None
        if nxt < n_stages:
            return (nxt, 0)
        return (2 * n_stages - 1 - nxt, 1)

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Return ``(0, 1)`` — terminal logical stage ``2n-1`` lives on rank 0's reverse leg.

        Because the V bounces back, the final logical stage runs on
        physical rank ``0``'s reverse virtual stage. The runtime fires
        ``loss_fn`` here.

        Args:
            n_stages: Number of physical pipeline ranks (unused).

        Returns:
            Always ``(0, 1)``.
        """
        return (0, 1)

    def peak_activations(self, n_stages: int) -> int:
        """Peak ≈ ``2 * n_stages`` saved activations per rank (one per virtual leg).

        Each rank holds ``n_stages`` saved activations from the
        forward leg and another ``n_stages`` from the reverse leg.

        Args:
            n_stages: Number of physical pipeline ranks.

        Returns:
            ``2 * n_stages``.
        """
        return 2 * n_stages


def dualpipev_tasks(mpmd_dim: int, mpmd_idx: int, n_mubatches: int) -> list[Action | FusedTask]:
    """Per-rank task list for DualPipe-V (DeepSeek-V3).

    Returns the ordered task sequence a single physical rank
    (``mpmd_idx``) executes under DualPipe-V for ``n_mubatches``.
    Each task is either a plain :class:`Action` (pure FWD or pure
    BWD_I/BWD_W on one virtual stage) or a :class:`FusedTask` pairing
    a forward on one virtual stage with a (split) backward on the
    other — the steady-state workhorse that DeepSeek's kernel uses
    to overlap compute across the V.

    The construction is the 8-section rank-centric formulation from
    Based on the DeepSeek DualPipe-V reference implementation.
    Sections::

        1. nF0                     — warmup fwd on stage0 only
        2. nF0F1                   — warmup alternating fwd stage0 / stage1
        3. nB1W1F1                 — zero-bubble (bwd_i, bwd_w, fwd) triplet
        4. nF0B1F1B0 (main step)   — steady state, fused fwd/bwd pairs
        5. nB1F1B0                 — cooldown start
        6. nB1B0                   — mid cooldown; enable ZB for some ranks
        7. nWB0                    — ZB-only cooldown
        8. nW                      — final weight-grad flushes

    Each rank owns two virtual stages: ``stage0 = mpmd_idx`` and
    ``stage1 = 2 * mpmd_dim - 1 - mpmd_idx``. The backward direction
    traverses ``stage1 -> stage0`` per microbatch.

    Args:
        mpmd_dim: Number of physical pipeline ranks.
        mpmd_idx: Index of the rank whose task list to build.
        n_mubatches: Number of microbatches per step.

    Returns:
        A list of :class:`Action` / :class:`FusedTask` entries in
        execution order for ``mpmd_idx``. Callers that want a
        time-step grid across all ranks (spectrax's standard
        representation) should use :class:`DualPipeV`.\\ :meth:`build`
        instead — this function is for building custom per-rank
        executors or experimental schedule runners.
    """
    stage_counts: dict[tuple[int, Phase], int] = {}

    def _next_mb(stage_id: int, phase: Phase) -> int:
        """Return the next microbatch index for ``(stage_id, phase)`` and bump the counter.

        Each ``(logical_stage, phase)`` pair gets its own
        monotonically increasing microbatch counter so the same
        logical stage produces microbatches 0, 1, 2, ... in the order
        the schedule emits them.

        Args:
            stage_id: Logical-stage index (``0 .. 2 * mpmd_dim - 1``).
            phase: Which phase counter to advance.

        Returns:
            The microbatch index for this action; counter is
            incremented for the next call.
        """
        key = (stage_id, phase)
        mb = stage_counts.get(key, 0)
        stage_counts[key] = mb + 1
        return mb

    def fwd(stage_id: int) -> Action:
        """Build a forward :class:`Action` for ``stage_id`` at its next microbatch.

        Virtual stage 0 for stages ``< mpmd_dim`` (the forward leg of
        the V), virtual stage 1 for the rest (the reverse leg) —
        matching the DualPipe-V layout where each rank owns two
        virtual stages.

        Args:
            stage_id: Logical-stage index.

        Returns:
            A FWD :class:`Action` with the right virtual-stage tag.
        """
        return Action(Phase.FWD, _next_mb(stage_id, Phase.FWD), 0 if stage_id < mpmd_dim else 1)

    def bwd_a(stage_id: int) -> Action:
        """Build a BWD_I (input-grad) :class:`Action` for ``stage_id``.

        Args:
            stage_id: Logical-stage index.

        Returns:
            A :attr:`Phase.BWD_I` :class:`Action` for the next
            microbatch on this stage.
        """
        return Action(Phase.BWD_I, _next_mb(stage_id, Phase.BWD_I), 0 if stage_id < mpmd_dim else 1)

    def bwd_w(stage_id: int) -> Action:
        """Build a BWD_W (weight-grad) :class:`Action` for ``stage_id``.

        Args:
            stage_id: Logical-stage index.

        Returns:
            A :attr:`Phase.BWD_W` :class:`Action` for the next
            microbatch on this stage.
        """
        return Action(Phase.BWD_W, _next_mb(stage_id, Phase.BWD_W), 0 if stage_id < mpmd_dim else 1)

    def bwd(stage_id: int) -> FusedTask:
        """Build a fused (BWD_I + BWD_W) task on the same microbatch.

        Returned as a :class:`FusedTask` whose ``fwd`` slot holds the
        BWD_I action and ``bwd`` slot holds the BWD_W action; this is
        a deliberate repurposing of the field names for the DualPipe
        pairing convention (a single backward dispatched as one
        kernel that produces both input- and weight-gradients).

        Args:
            stage_id: Logical-stage index.

        Returns:
            A :class:`FusedTask` packing the two backward halves.
        """
        a = bwd_a(stage_id)
        w = bwd_w(stage_id)
        return FusedTask(fwd=a, bwd=w, virtual_stage=a.virtual_stage)

    def fwd_bwd(fwd_stage: int, bwd_stage: int) -> FusedTask:
        """Build a fused (FWD on ``fwd_stage``, BWD_I on ``bwd_stage``) task.

        The BWD_W half is allocated by side-effect (advancing its
        microbatch counter) but not packed into the returned task —
        DualPipe-V emits the weight-grad later in the schedule, which
        keeps it free to slot into bubble time and is what makes the
        schedule "near zero-bubble" in steady state.

        Args:
            fwd_stage: Logical stage to forward on.
            bwd_stage: Logical stage to BWD_I on (typically the
                mirror partner of ``fwd_stage``).

        Returns:
            A :class:`FusedTask` whose ``fwd`` is the forward action
            and ``bwd`` is the BWD_I action; the deferred BWD_W is
            counted but not emitted here.
        """
        f = fwd(fwd_stage)
        a = bwd_a(bwd_stage)
        _ = bwd_w(bwd_stage)
        return FusedTask(fwd=f, bwd=a, virtual_stage=f.virtual_stage)

    stage0 = mpmd_idx
    stage1 = mpmd_dim * 2 - mpmd_idx - 1
    tasks: list[Action | FusedTask] = []

    section_1 = (mpmd_dim - mpmd_idx - 1) * 2
    tasks.extend(fwd(stage0) for _ in range(section_1))

    section_2 = mpmd_idx + 1
    for _ in range(section_2):
        tasks.append(fwd(stage0))
        tasks.append(fwd(stage1))

    section_3 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_3):
        tasks.append(bwd_a(stage1))
        tasks.append(bwd_w(stage1))
        tasks.append(fwd(stage1))

    section_4 = n_mubatches - mpmd_dim * 2 + mpmd_idx + 1
    for idx in range(section_4):
        if idx == 0:
            if mpmd_idx == mpmd_dim - 1:
                tasks.append(fwd(stage0))
                tasks.append(bwd(stage1))
            else:
                tasks.append(fwd_bwd(stage0, stage1))
        else:
            tasks.append(fwd_bwd(stage0, stage1))
        tasks.append(fwd_bwd(stage1, stage0))

    section_5 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_5):
        tasks.append(bwd(stage1))
        tasks.append(fwd_bwd(stage1, stage0))

    section_6 = mpmd_idx + 1
    enable_zb_at = section_6 // 2
    for idx in range(section_6):
        if idx >= enable_zb_at and mpmd_idx % 2 == 1:
            tasks.append(bwd_a(stage1))
        else:
            tasks.append(bwd(stage1))
        if stage0 != 0 and idx >= enable_zb_at and mpmd_idx % 2 == 0:
            tasks.append(bwd_a(stage0))
        else:
            tasks.append(bwd(stage0))

    section_7 = mpmd_dim - mpmd_idx - 1
    for _ in range(section_7):
        if stage0 == 0:
            tasks.append(bwd(stage0))
        else:
            tasks.append(bwd_a(stage0))

    remaining_w = [(stage_id, stage_counts.get((stage_id, Phase.BWD_W), 0)) for stage_id in (stage0, stage1)]
    remaining_w_tasks: list[Action] = []
    for stage_id, done_mb in remaining_w:
        for _mb in range(done_mb, n_mubatches):
            remaining_w_tasks.append(bwd_w(stage_id))
    remaining_w_tasks.sort(key=lambda a: (a.microbatch, a.virtual_stage))
    tasks.extend(remaining_w_tasks)

    return tasks
