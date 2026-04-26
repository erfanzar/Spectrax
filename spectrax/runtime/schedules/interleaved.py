# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interleaved (virtual-stage) schedules: InterleavedH1, InterleavedGPipe, Interleaved1F1BPlusOne, KimiK2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import Action, Phase, Schedule
from .gpipe import GPipe


@dataclass
class InterleavedH1(Schedule):
    """Interleaved 1F1B with virtual pipeline parallelism (Narayanan 2021).

    Each physical device owns ``virtual_stages`` non-contiguous logical
    stages. A device with virtual count ``v`` and physical rank ``r``
    in an ``n``-device pipeline owns logical stages
    ``r, r + n, r + 2n, …, r + (v - 1) n``.

    Pros:

    * Pipeline bubble ratio shrinks by a factor of ``virtual_stages``.
      With ``v = 4`` stages per device on a 4-device pipeline, the
      bubble is ~25% of 1F1B's.

    Cons:

    * ``(virtual_stages - 1) x n`` extra ``ppermute`` hops per
      microbatch. Makes sense on high-bandwidth interconnect (NVLink,
      TPU v5+ ICI); less beneficial on PCIe.

    Attributes:
        virtual_stages: Number of virtual stages per physical device.
    """

    virtual_stages: int = 2
    stage_layout: Literal["contiguous", "interleaved", "loop"] = "loop"

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if self.virtual_stages < 1:
            raise ValueError(f"InterleavedH1.virtual_stages must be >= 1, got {self.virtual_stages}.")
        if self.stage_layout not in ("contiguous", "interleaved", "loop"):
            raise ValueError(
                f"InterleavedH1.stage_layout must be 'contiguous', 'interleaved', or 'loop', got {self.stage_layout!r}."
            )

    def virtual_stages_per_rank(self) -> int:
        """Number of virtual stages per physical device."""
        return self.virtual_stages

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Map ``(rank, virt)`` to its logical-stage index.

        ``stage_layout="contiguous"`` hosts consecutive logical chunks per
        physical rank, reducing cross-rank activation transfers. The classic
        virtual-pipeline layout is available with
        ``stage_layout="interleaved"``.
        """
        if self.stage_layout == "contiguous":
            return rank * self.virtual_stages + virt
        if self.stage_layout == "loop":
            offset = rank if virt % 2 == 0 else n_stages - 1 - rank
            return virt * n_stages + offset
        return virt * n_stages + rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int):
        """Return the ``(rank, virt)`` of the next logical stage (or ``None``)."""
        nxt = self.logical_at(rank, virt, n_stages) + 1
        if nxt >= self.virtual_stages * n_stages:
            return None
        if self.stage_layout == "contiguous":
            return (nxt // self.virtual_stages, nxt % self.virtual_stages)
        if self.stage_layout == "loop":
            virt = nxt // n_stages
            offset = nxt % n_stages
            rank = offset if virt % 2 == 0 else n_stages - 1 - offset
            return (rank, virt)
        return (nxt % n_stages, nxt // n_stages)

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Return the ``(rank, virt)`` hosting the terminal logical stage."""
        terminal = self.virtual_stages * n_stages - 1
        if self.stage_layout == "contiguous":
            return (terminal // self.virtual_stages, terminal % self.virtual_stages)
        if self.stage_layout == "loop":
            virt = terminal // n_stages
            offset = terminal % n_stages
            rank = offset if virt % 2 == 0 else n_stages - 1 - offset
            return (rank, virt)
        return (terminal % n_stages, terminal // n_stages)

    def _skip_auto_fuse_1f1b(self) -> bool:
        """Physical list scheduling already optimizes rank occupancy."""
        return True

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the interleaved-1F1B grid.

        The total number of logical stages is ``n_stages x
        virtual_stages``. Each logical stage handles ``M`` microbatches
        exactly like 1F1B; the physical device that owns a logical
        stage does its work.

        Implementation: build a 1F1B grid for ``n_logical = n * v``
        logical stages, then remap each logical stage ``l`` back to
        physical device ``l % n``, virtual stage ``l // n``. If two
        virtual stages on the same physical device want the same time
        step (shouldn't happen given 1F1B's construction on
        ``n_logical``, but handled defensively), we emit the current
        row and serialize the loser into a fresh row.
        """
        return _build_physical_virtual_1f1b(
            n_stages=n_stages,
            virtual_stages=self.virtual_stages,
            microbatches=self.microbatches,
            extra_warmup=0,
            logical_at=self.logical_at,
        )

    def peak_activations(self, n_stages: int) -> int:
        """Peak ≈ ``n_stages x virtual_stages``."""
        return n_stages * self.virtual_stages


@dataclass
class InterleavedGPipe(Schedule):
    """Interleaved GPipe: all-fwd then all-bwd with virtual stages.

    The GPipe analog of :class:`InterleavedH1`: each physical device
    owns ``virtual_stages`` non-contiguous logical stages, but unlike
    1F1B the schedule runs every forward before any backward.

    Pros over :class:`InterleavedH1`:

    * Simpler activation lifetime — every stage's M activations are
      saved until the backward pass starts (O(M x v) per device).
    * Natural pairing with ``lax.remat`` for activation checkpointing.

    Cons:

    * Peak memory scales as ``virtual_stages x microbatches`` —
      large on long pipelines with many microbatches.

    Attributes:
        virtual_stages: Virtual stages per physical device.
    """

    virtual_stages: int = 2
    stage_layout: Literal["contiguous", "interleaved", "loop"] = "loop"

    def __post_init__(self) -> None:
        """Validate ``virtual_stages >= 1``."""
        super().__post_init__()
        if self.virtual_stages < 1:
            raise ValueError(f"InterleavedGPipe.virtual_stages must be >= 1, got {self.virtual_stages}.")
        if self.stage_layout not in ("contiguous", "interleaved", "loop"):
            raise ValueError(
                "InterleavedGPipe.stage_layout must be 'contiguous', 'interleaved', or 'loop', "
                f"got {self.stage_layout!r}."
            )

    def virtual_stages_per_rank(self) -> int:
        """Number of virtual stages per physical device."""
        return self.virtual_stages

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Map ``(rank, virt)`` to its logical-stage index."""
        if self.stage_layout == "contiguous":
            return rank * self.virtual_stages + virt
        if self.stage_layout == "loop":
            offset = rank if virt % 2 == 0 else n_stages - 1 - rank
            return virt * n_stages + offset
        return virt * n_stages + rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int):
        """Return the ``(rank, virt)`` of the next logical stage (or ``None``)."""
        nxt = self.logical_at(rank, virt, n_stages) + 1
        if nxt >= self.virtual_stages * n_stages:
            return None
        if self.stage_layout == "contiguous":
            return (nxt // self.virtual_stages, nxt % self.virtual_stages)
        if self.stage_layout == "loop":
            virt = nxt // n_stages
            offset = nxt % n_stages
            rank = offset if virt % 2 == 0 else n_stages - 1 - offset
            return (rank, virt)
        return (nxt % n_stages, nxt // n_stages)

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Return the ``(rank, virt)`` hosting the terminal logical stage."""
        terminal = self.virtual_stages * n_stages - 1
        if self.stage_layout == "contiguous":
            return (terminal // self.virtual_stages, terminal % self.virtual_stages)
        if self.stage_layout == "loop":
            virt = terminal // n_stages
            offset = terminal % n_stages
            rank = offset if virt % 2 == 0 else n_stages - 1 - offset
            return (rank, virt)
        return (terminal % n_stages, terminal // n_stages)

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the interleaved-GPipe grid.

        Same logical->physical mapping as :class:`InterleavedH1`, but
        the underlying per-logical-stage schedule is :class:`GPipe`
        (all fwds, then all bwds) instead of 1F1B.
        """
        n = n_stages
        v = self.virtual_stages
        m = self.microbatches
        n_logical = n * v

        logical = GPipe(m).build(n_logical)
        grid: list[list[Action | None]] = []
        for row in logical:
            new_row: list[Action | None] = [None] * n
            for l_stage, action in enumerate(row):
                if action is None:
                    continue
                if self.stage_layout == "contiguous":
                    phys = l_stage // v
                    virt = l_stage % v
                elif self.stage_layout == "loop":
                    virt = l_stage // n
                    offset = l_stage % n
                    phys = offset if virt % 2 == 0 else n - 1 - offset
                else:
                    phys = l_stage % n
                    virt = l_stage // n
                if new_row[phys] is not None:
                    grid.append(new_row)
                    new_row = [None] * n
                new_row[phys] = Action(action.phase, action.microbatch, virt)
            grid.append(new_row)

        while grid and all(c is None for c in grid[-1]):
            grid.pop()
        return grid

    def peak_activations(self, n_stages: int) -> int:
        """Peak ≈ ``virtual_stages x microbatches`` (all saved before BWD)."""
        return self.virtual_stages * self.microbatches


@dataclass
class Interleaved1F1BPlusOne(InterleavedH1):
    """:class:`InterleavedH1` with one extra warmup forward prepended.

    Variant that pulls a single forward op (stage 0, virtual 0,
    microbatch 0) into its own leading time step, then runs the rest
    of the standard interleaved-1F1B grid. Total op count is
    identical to :class:`InterleavedH1`; only the leading ordering
    differs.

    Honest framing: in a Python-driven runtime this strictly *adds*
    one empty-ish dispatch slot vs :class:`InterleavedH1`, so it's
    typically slower. The form is useful as a building block / sanity
    check for schedules that re-shuffle further (e.g. true Kimi-K2,
    which we ship as :class:`KimiK2`).

    Attributes:
        virtual_stages: Virtual stages per physical device.
    """

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the grid: ``InterleavedH1`` with +1 warmup row prepended.

        Prepends a single-action row — ``(stage 0, virt 0, mb 0)`` —
        and removes the duplicate FWD of that same action from the
        base :class:`InterleavedH1` schedule to preserve total action
        counts. If ``microbatches < 1`` or the base grid is empty we
        fall through to the base unchanged.
        """
        base = super().build(n_stages)
        if self.microbatches < 1 or not base:
            return base
        first_row: list[Action | None] = [None] * n_stages
        first_row[0] = Action(Phase.FWD, 0, 0)
        found = False
        new_base: list[list[Action | None]] = []
        for row in base:
            new_row: list[Action | None] = list(row)
            if not found:
                for s in range(n_stages):
                    a = new_row[s]
                    if a is not None and a.phase == Phase.FWD and a.microbatch == 0 and a.virtual_stage == 0 and s == 0:
                        new_row[s] = None
                        found = True
                        break
            new_base.append(new_row)
        return [first_row, *new_base]


def _warmup_bumped_1f1b_queue(logical: int, n_logical: int, microbatches: int, *, extra_warmup: int) -> list[Action]:
    """Build one logical stage's 1F1B action queue with Kimi-style warmup."""
    warmup = min(min(n_logical - logical, microbatches) + extra_warmup, microbatches)
    queue: list[Action] = [Action(Phase.FWD, mb) for mb in range(warmup)]
    fwd_head = warmup
    bwd_head = 0
    while fwd_head < microbatches or bwd_head < microbatches:
        if fwd_head < microbatches:
            queue.append(Action(Phase.FWD, fwd_head))
            fwd_head += 1
        if bwd_head < microbatches:
            queue.append(Action(Phase.BWD, bwd_head))
            bwd_head += 1
    return queue


def _build_physical_virtual_1f1b(
    *,
    n_stages: int,
    virtual_stages: int,
    microbatches: int,
    extra_warmup: int,
    logical_at,
) -> list[list[Action | None]]:
    """List-schedule virtual 1F1B directly on physical ranks.

    A logical-grid remap can leave physical-rank bubbles when two virtual
    stages collide on one rank. This keeps each logical stage's 1F1B queue and
    all data dependencies, then chooses ready work per physical rank by
    remaining critical path.
    """
    n = n_stages
    v = virtual_stages
    m = microbatches
    n_logical = n * v

    loc_for_logical: list[tuple[int, int] | None] = [None] * n_logical
    for rank in range(n):
        for virt in range(v):
            logical = logical_at(rank, virt, n)
            if logical < 0 or logical >= n_logical:
                raise ValueError(f"logical_at({rank}, {virt}, {n}) returned out-of-range stage {logical}.")
            if loc_for_logical[logical] is not None:
                raise ValueError(f"logical stage {logical} is assigned to multiple virtual locations.")
            loc_for_logical[logical] = (rank, virt)
    if any(loc is None for loc in loc_for_logical):
        missing = [idx for idx, loc in enumerate(loc_for_logical) if loc is None]
        raise ValueError(f"virtual schedule did not assign logical stages {missing}.")

    queues = [
        _warmup_bumped_1f1b_queue(logical, n_logical, m, extra_warmup=extra_warmup) for logical in range(n_logical)
    ]
    task_keys = [(logical, action.phase, action.microbatch) for logical, queue in enumerate(queues) for action in queue]
    predecessors: dict[tuple[int, Phase, int], set[tuple[int, Phase, int]]] = {key: set() for key in task_keys}
    successors: dict[tuple[int, Phase, int], set[tuple[int, Phase, int]]] = {key: set() for key in task_keys}

    def add_dep(task: tuple[int, Phase, int], dep: tuple[int, Phase, int]) -> None:
        """Record ``task`` -> ``dep`` in both adjacency lists (predecessors and successors)."""
        predecessors.setdefault(task, set()).add(dep)
        successors.setdefault(dep, set()).add(task)

    for logical, queue in enumerate(queues):
        previous: tuple[int, Phase, int] | None = None
        for action in queue:
            task = (logical, action.phase, action.microbatch)
            if previous is not None:
                add_dep(task, previous)
            previous = task

    for logical in range(n_logical):
        for mb in range(m):
            fwd = (logical, Phase.FWD, mb)
            bwd = (logical, Phase.BWD, mb)
            add_dep(bwd, fwd)
            if logical > 0:
                add_dep(fwd, (logical - 1, Phase.FWD, mb))
            if logical + 1 < n_logical:
                add_dep(bwd, (logical + 1, Phase.BWD, mb))

    critical_cache: dict[tuple[int, Phase, int], int] = {}

    def critical_path(task: tuple[int, Phase, int]) -> int:
        """Memoized longest-finish-time of ``task`` over its successor DAG.

        Uses simple per-action latencies (BWD costs ~2x FWD on most
        hardware) summed along the longest downstream path. Used to
        rank ready tasks so the most schedule-critical work fires first.
        """
        cached = critical_cache.get(task)
        if cached is not None:
            return cached
        action_latency = 4 if task[1] is Phase.BWD else 2
        value = action_latency + max((critical_path(dep) for dep in successors.get(task, ())), default=0)
        critical_cache[task] = value
        return value

    for task in task_keys:
        critical_path(task)

    positions = [0] * n_logical
    done: set[tuple[int, Phase, int]] = set()
    rows: list[list[Action | None]] = []
    total_actions = sum(len(queue) for queue in queues)
    max_rows = 4 * (total_actions + n_logical) + 10

    while len(done) < total_actions:
        row: list[Action | None] = [None] * n
        selected: list[tuple[int, Phase, int]] = []
        for rank in range(n):
            candidates: list[tuple[int, Phase, int]] = []
            for virt in range(v):
                logical = logical_at(rank, virt, n)
                pos = positions[logical]
                if pos >= len(queues[logical]):
                    continue
                action = queues[logical][pos]
                task = (logical, action.phase, action.microbatch)
                if predecessors.get(task, set()).issubset(done):
                    candidates.append(task)
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda task: (
                    critical_path(task),
                    1 if task[1] is Phase.BWD else 0,
                    -task[2],
                    -task[0],
                ),
            )
            logical, phase, mb = chosen
            loc = loc_for_logical[logical]
            assert loc is not None
            _rank, virt = loc
            row[rank] = Action(phase, mb, virt)
            selected.append(chosen)

        if not selected:
            blocked = {
                logical: queues[logical][positions[logical]]
                for logical in range(n_logical)
                if positions[logical] < len(queues[logical])
            }
            raise RuntimeError(f"virtual 1F1B scheduler made no progress; blocked={blocked!r}")

        for logical, phase, mb in selected:
            positions[logical] += 1
            done.add((logical, phase, mb))
        rows.append(row)
        if len(rows) > max_rows:
            raise RuntimeError(f"virtual 1F1B scheduler exceeded {max_rows} rows (n={n}, v={v}, microbatches={m}).")

    return rows


@dataclass
class KimiK2(InterleavedH1):
    """Kimi-K2 schedule (Moonshot AI training infra, 2024).

    Variant of :class:`InterleavedH1` that bumps the per-logical-stage
    warmup forward count by 1, folded into the steady-state 1F1B walk
    so no extra time-step row is wasted. The +1 warmup gives every
    physical rank one more forward in flight before the first
    backward fires, which trims the cooldown tail at large
    microbatch counts.

    The +1 warmup matches the Kimi-K2 paper's design intent. In a
    Python-driven runtime the gain is small (the schedule does the
    same total ops) and may be invisible against dispatch noise; the
    win shows up most clearly in compiler-aware runtimes that can use
    the longer warmup to overlap stages more aggressively.

    Compare to:

    * :class:`InterleavedH1` — the base interleaved 1F1B
    * :class:`Interleaved1F1BPlusOne` — naïve "+1 by prepended row"
      variant; structurally adds an idle slot (slower in dispatch-
      bound runtimes).

    Attributes:
        virtual_stages: Virtual stages per physical rank (vp).
        extra_warmup: Additional per-logical-stage forward warmup.
    """

    extra_warmup: int = 1

    def __post_init__(self) -> None:
        """Validate the warmup bump."""
        super().__post_init__()
        if self.extra_warmup < 0:
            raise ValueError(f"KimiK2.extra_warmup must be >= 0, got {self.extra_warmup}.")

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Same logical->physical remap as :class:`InterleavedH1` but
        the underlying per-logical-stage 1F1B uses +1 warmup."""
        return _build_physical_virtual_1f1b(
            n_stages=n_stages,
            virtual_stages=self.virtual_stages,
            microbatches=self.microbatches,
            extra_warmup=self.extra_warmup,
            logical_at=self.logical_at,
        )
