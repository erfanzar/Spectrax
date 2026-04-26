# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""1F1B-family schedules: Std1F1B, Eager1F1B, _Std1F1BWarmupBumped."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Action, Phase, Schedule


@dataclass
class Std1F1B(Schedule):
    """Standard 1-forward-1-backward schedule (Narayanan et al., 2021).

    After a warmup that fills the pipeline, each stage alternates
    between one forward and one backward until all microbatches are
    done. Peak activation memory is bounded by ``n_stages`` (each
    stage has at most ``n_stages - stage_idx`` live activations at
    the steady-state peak, far less than GPipe's ``M``).

    Structure::

        Warmup  (duration n_stages - 1): forwards fill the pipe
        Steady  (duration M - (n_stages - 1)): 1F1B alternation
        Cooldown (duration n_stages - 1): backwards drain

    Pros:

    * Peak memory independent of ``microbatches``.
    * Same bubble time as GPipe.
    * Widely used in production (Megatron-LM, DeepSpeed).

    Cons:

    * Still has the ``2(n - 1)`` bubble at the ends.
    * Doesn't parallelize backward I-grad / W-grad — use
      :class:`ZeroBubbleH1` for that.
    """

    def _warmup(self, s: int, n: int, m: int) -> int:
        """Per-stage warmup length; subclasses override for eager variants."""
        return min(n - s, m)

    def build(self, n_stages: int) -> list[list[Action | None]]:
        """Emit the warmup/steady/cooldown grid respecting data dependencies.

        Per-stage action queue is built up front: a warmup run of
        :meth:`_warmup` forwards, then an alternating 1F1B steady
        state until all forwards are done, then remaining backwards.

        The grid is then filled via a greedy time-step loop: at each
        ``t``, every stage fires its next queued action if its
        dependencies are met (forward waits for the upstream stage's
        forward on the same microbatch; backward waits for the
        downstream stage's backward). This guarantees the schedule is
        **data-dependency correct** — a property the naive
        construction violates when stage 0's BWD fires before stage 1
        has produced the corresponding cotangent.
        """
        n, m = n_stages, self.microbatches
        if m < n:
            raise ValueError(
                "Std1F1B requires microbatches >= n_stages so the 1F1B steady state is real; "
                f"got microbatches={m}, n_stages={n}. Use GPipe explicitly for this shape."
            )

        per_stage_queue: list[list[tuple[Phase, int]]] = []
        for s in range(n):
            warmup = self._warmup(s, n, m)
            queue: list[tuple[Phase, int]] = []
            for mb in range(warmup):
                queue.append((Phase.FWD, mb))
            fwd_head = warmup
            bwd_head = 0
            while fwd_head < m or bwd_head < m:
                if fwd_head < m:
                    queue.append((Phase.FWD, fwd_head))
                    fwd_head += 1
                if bwd_head < m:
                    queue.append((Phase.BWD, bwd_head))
                    bwd_head += 1
            per_stage_queue.append(queue)

        pos = [0] * n
        done_time: dict[tuple[Phase, int, int], int] = {}
        grid: list[list[Action | None]] = []
        max_t = 4 * (m + n) + 10
        t = 0
        while any(pos[s] < len(per_stage_queue[s]) for s in range(n)):
            row: list[Action | None] = [None] * n
            for s in range(n):
                if pos[s] >= len(per_stage_queue[s]):
                    continue
                phase, mb = per_stage_queue[s][pos[s]]
                if phase == Phase.FWD and s > 0:
                    prev_key = (Phase.FWD, s - 1, mb)
                    if prev_key not in done_time or done_time[prev_key] >= t:
                        continue
                if phase == Phase.BWD and s < n - 1:
                    next_key = (Phase.BWD, s + 1, mb)
                    if next_key not in done_time or done_time[next_key] >= t:
                        continue
                row[s] = Action(phase, mb)
                pos[s] += 1
                done_time[(phase, s, mb)] = t
            grid.append(row)
            t += 1
            if t > max_t:
                raise RuntimeError(
                    f"Std1F1B schedule failed to converge after {max_t} "
                    f"steps (n={n}, m={m}); dependency cycle — file a bug."
                )

        while grid and all(c is None for c in grid[-1]):
            grid.pop()
        return grid

    def peak_activations(self, n_stages: int) -> int:
        """Peak is the warmup length for the first stage: ``n_stages``."""
        return n_stages


@dataclass
class Eager1F1B(Std1F1B):
    """Eager 1F1B: longer warmup so first backward fires sooner.

    A variant of :class:`Std1F1B` that extends each stage's warmup
    from ``n - s`` to ``2(n - s) - 1`` forwards (capped at
    ``microbatches``). This fills the pipeline with enough forwards
    that the first backward on every stage can fire earlier —
    reducing the cooldown-tail portion of the schedule on balanced
    workloads. Per-stage FWD and BWD counts still equal ``M``; only
    the time at which the first backward starts shifts earlier.

    On typical LLM training shapes (``n_stages >= 4``, ``M >= 2n``)
    this saves one or two time steps of cooldown vs :class:`Std1F1B`
    at negligible extra peak activation.

    Reference: Narayanan et al., *Efficient Large-Scale Language
    Model Training on GPU Clusters Using Megatron-LM* (SC '21).

    """

    def _warmup(self, s: int, n: int, m: int) -> int:
        """Extended warmup: ``2(n - s) - 1`` forwards, capped at ``m``."""
        return max(0, min(2 * (n - s) - 1, m))

    def peak_activations(self, n_stages: int) -> int:
        """Peak ≈ ``2 * n_stages - 1`` (at stage 0 after full warmup)."""
        return min(2 * n_stages - 1, self.microbatches)


@dataclass
class _Std1F1BWarmupBumped(Std1F1B):
    """Internal: :class:`Std1F1B` with each stage's warmup increased by 1.

    Used as the underlying logical-stage schedule for :class:`KimiK2`.
    Bumping the warmup count folds an extra forward into each logical
    stage's pre-steady-state run; in the interleaved remap this lets
    every physical rank squeeze one more forward in before the first
    backward fires, mimicking the Moonshot K2 paper's per-rank +1
    warmup adjustment without needing per-physical-rank bookkeeping
    inside the dependency-aware 1F1B builder.
    """

    extra_warmup: int = 1

    def _warmup(self, s: int, n: int, m: int) -> int:
        """Return the base :class:`Std1F1B` warmup plus ``extra_warmup`` (capped at ``m``)."""
        base = super()._warmup(s, n, m)
        return min(base + self.extra_warmup, m)
