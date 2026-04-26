# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Base schedule primitives: Phase, Action, FusedTask, and the Schedule ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """Action phase: what the stage is computing.

    * :attr:`FWD` — forward pass on a microbatch's activations.
    * :attr:`BWD` — full backward pass (both input-grad and
      weight-grad fused).
    * :attr:`BWD_I` — input-grad only (zero-bubble split).
    * :attr:`BWD_W` — weight-grad only (zero-bubble split).
    """

    FWD = "fwd"
    BWD = "bwd"
    BWD_I = "bwd_i"
    BWD_W = "bwd_w"


_DEFAULT_LATENCIES: dict[Phase, int] = {
    Phase.FWD: 2,
    Phase.BWD: 4,
    Phase.BWD_I: 2,
    Phase.BWD_W: 2,
}


@dataclass(frozen=True)
class Action:
    """Single scheduled action for one stage at one time step.

    Attributes:
        phase: What the stage does (see :class:`Phase`).
        microbatch: Which microbatch the action operates on.
        virtual_stage: For :class:`InterleavedH1`, which of the
            device's multiple virtual stages runs. Zero for other
            schedules.
        latency: Relative cost weight for latency-aware scheduling.
            Defaults to the phase's default latency (FWD=2, BWD=4,
            BWD_I=BWD_W=2). Used by schedule optimizers that want to
            minimize critical-path length; the absolute value is
            arbitrary (only the ratio between phases matters).
    """

    phase: Phase
    microbatch: int
    virtual_stage: int = 0
    latency: int | None = None

    def __post_init__(self) -> None:
        """Populate latency from phase defaults if not provided."""
        if self.latency is None:
            object.__setattr__(self, "latency", _DEFAULT_LATENCIES.get(self.phase, 1))


@dataclass(frozen=True)
class FusedTask:
    """A steady-state action that combines a forward and a backward.

    Used by 1F1B-family schedules to let a runtime fuse the
    forward of microbatch ``a`` and the backward of microbatch ``b``
    into a single compiled XLA kernel, reducing dispatch overhead
    and improving register reuse. Runtimes that don't support fusion
    can treat a :class:`FusedTask` as two sequential :class:`Action`
    objects via :meth:`split`.

    Attributes:
        fwd: The forward half.
        bwd: The backward half.
        virtual_stage: Shared virtual stage index.
    """

    fwd: Action
    bwd: Action
    virtual_stage: int = 0

    def split(self) -> tuple[Action, Action]:
        """Return the underlying ``(fwd, bwd)`` :class:`Action` pair."""
        return (self.fwd, self.bwd)


@dataclass
class Schedule(ABC):
    """Abstract base for pipeline schedules.

    A schedule maps time step x physical stage to a :class:`Action`
    (or ``None`` for idle). The :meth:`build` method returns the full
    2D plan; :meth:`peak_activations` reports the worst-case number of
    activation tensors any single stage holds at once (a proxy for
    peak memory).

    Virtual-stage schedules (Interleaved*, KimiK2, DualPipeV) assign
    multiple *logical* stages to each physical rank. The runtime needs
    three schedule-specific pieces of information to execute them:

    * :meth:`virtual_stages_per_rank` — how many logical stages each
      rank hosts. Flat schedules return 1.
    * :meth:`logical_at` — given ``(rank, virt)``, which logical
      stage's parameters live there.
    * :meth:`next_logical_loc` — after this ``(rank, virt)`` produces
      an activation, which ``(rank, virt)`` consumes it next (or
      ``None`` if this is the terminal stage).

    Defaults on the base class assume a flat (one-logical-per-rank)
    pipeline; virtual schedules override.

    Attributes:
        microbatches: Number of microbatches per global batch.
        lazy_bwd_batching: When ``True``, the MPMD runtime collects all
            backward actions for each logical stage during the forward
            grid walk, then dispatches them in a single vmapped backward
            per stage at the end. This reduces dispatch count (fewer
            Python->XLA round-trips) at the cost of higher peak
            activation memory because all saved inputs/outputs must be
            retained until the final batched backward. Default ``False``.
    """

    microbatches: int
    lazy_bwd_batching: bool = False

    def __post_init__(self) -> None:
        """Validate ``microbatches >= 1``."""
        if self.microbatches < 1:
            raise ValueError(f"Schedule.microbatches must be >= 1, got {self.microbatches}.")

    @abstractmethod
    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Return the ``(T, n_stages)`` action grid for ``n_stages`` stages.

        ``result[t][s]`` is the action performed by stage ``s`` at
        time step ``t``, or ``None`` if the stage is idle at ``t``.

        Args:
            n_stages: Number of physical pipeline stages (matches the
                mesh's pipeline-axis size).

        Returns:
            A list of ``T`` rows, each a list of ``n_stages``
            :class:`Action` (or ``None``) entries.
        """

    def virtual_stages_per_rank(self) -> int:
        """Number of logical stages per physical rank (1 for flat schedules)."""
        return 1

    def logical_at(self, rank: int, virt: int, n_stages: int) -> int:
        """Return the logical-stage index hosted at ``(rank, virt)``."""
        return rank

    def next_logical_loc(self, rank: int, virt: int, n_stages: int) -> tuple[int, int] | None:
        """Return the ``(rank, virt)`` of the downstream logical stage.

        ``None`` means the current position is terminal — its output
        feeds ``loss_fn`` directly.
        """
        if rank + 1 < n_stages:
            return (rank + 1, 0)
        return None

    def terminal_loc(self, n_stages: int) -> tuple[int, int]:
        """Where the model's final output is produced (loss goes here)."""
        return (n_stages - 1, 0)

    @abstractmethod
    def peak_activations(self, n_stages: int) -> int:
        """Return the worst-case number of live activations per stage.

        A diagnostic used to reason about peak memory. The worst-case
        stage (typically stage 0 for GPipe, the middle stage for
        1F1B) holds this many saved activations at its memory peak.

        Args:
            n_stages: Number of physical pipeline stages.

        Returns:
            Integer upper bound on simultaneous live activation
            tensors per stage.
        """

    def total_steps(self, n_stages: int) -> int:
        """Total number of time steps in the schedule."""
        return len(self.build(n_stages))

    def bubble_ratio(self, n_stages: int) -> float:
        """Fraction of (stage x time) slots that are idle.

        ``0.0`` means perfectly packed; ``1.0`` means nothing runs.
        Useful for comparing schedules at the same ``(n_stages,
        microbatches)``.
        """
        grid = self.build(n_stages)
        total = len(grid) * n_stages
        idle = sum(1 for row in grid for cell in row if cell is None)
        return idle / total
