# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`sxcall`: multiple-program pipeline runtime for **heterogeneous stages**.

Single-HLO pipeline runtimes compile one jaxpr shared by every
pipeline device and rely on the same-shape constraint to keep that
jaxpr consistent. For models whose stages
have **different shapes** — e.g. a transformer with a bulky
embedding stage, many uniform middle stages, and an lm-head stage —
this runtime compiles a **separate jitted function per stage** and
orchestrates them in Python. Activation and cotangent transfers use
:func:`jax.device_put` between per-stage sub-meshes.

Trade-offs:

* **Flexibility**: every stage can have a different class, different
  parameter shapes, different input/output shapes. The
  :class:`~spectrax.pipeline.PipelineSequential` container still
  represents the model; the MPMD runtime just doesn't require
  matching ``GraphDef`` s.
* **Mesh composition**: stages live on the
  :class:`~spectrax.pipeline.MpMdMesh`'s MPMD axis; other mesh axes
  are free for intra-stage FSDP / TP / DP.
* **Lower throughput**: no :func:`shard_map` fusion, so cross-device
  communication is coarser-grained. The upside is that each rank keeps
  a separate compiled program instead of seeing a union graph.
* **Python orchestration overhead**: the schedule loop runs in
  Python at dispatch time. Each step's stage work is jitted, so the
  compute cost inside each step is compiled to XLA, but the outer
  loop doesn't fuse across time steps.

**Architecture.** :func:`sxcall` is the public entry point. It
builds per-(rank, virt) jitted forward/backward callables, places
each stage's params+rest state on its rank's sub-mesh, microbatches
the batch, and then runs the schedule. GPipe + flat (V=1) schedules
hit a vmap fast-path (:func:`_gpipe_run`) that collapses M microbatch
dispatches into 1 per stage/phase; other schedules fall through to
the per-action Python loop that honours :class:`Schedule` ordering.

The schedule parameter actually controls execution order here: GPipe
queues all forwards then all backwards, Std1F1B interleaves forward
and backward microbatches in the steady state, ZeroBubbleH1 splits
each backward into input-grad and weight-grad to slot weight-grads
into bubble time, and so on. This gives MPMD its real value over
SPMD's auto-pipelined single-jit step: different schedules produce
measurably different step times and activation-memory profiles.

**Virtual-stage support.** ``V`` logical stages live on each physical
rank (V=1 for flat schedules like GPipe / Std1F1B). The schedule
tells us where each logical stage goes and where activations flow
next, so the runtime stays schedule-agnostic. Models must supply
``V * mpmd_dim`` logical stages in logical order; the runtime routes
each to its ``(rank, virt)`` slot via ``schedule.logical_at``.

**Caches.** Several module-level caches keep steady-state dispatch
free of retracing and repeated ``jax.device_put`` costs:

* ``_STAGE_CALLABLE_CACHE``: per-stage jitted fwd/bwd keyed on
  ``(id(stage), donate_fwd, donate_bwd)`` so two Module instances
  with identical donation patterns reuse the same jits.
* ``_MPMD_SETUP_CACHE``: full ``sxcall`` setup (placed parameters/rest
  + fwd/bwd jits per ``(rank, virt)``), keyed on
  ``(id(model), id(mpmd_mesh), V, schedule_class_name, donate_fwd,
  donate_bwd)`` — avoids ~40 ``jax.device_put`` calls per step
  (each ~0.2 ms) for the same model/mesh.
* ``_GPIPE_VMAP_CACHE`` / ``_GPIPE_TERM_CACHE``: vmapped fwd/bwd
  pairs and fused (fwd + loss + bwd) terminal jits for the GPipe
  fast-path.
* ``_LOSS_JIT_CACHE`` / ``_VMAP_LOSS_CACHE``: jitted
  ``loss_and_g_y`` wrappers (scalar and vmapped) keyed by
  ``(id(loss_fn), has_aux, donate_argnums)`` and
  ``(id(loss_fn), donate_argnums)`` respectively.

**Observability.** :func:`collect_task_times_ms` is a context manager
that attributes wall-clock milliseconds to named sub-tasks (each
stage fwd/bwd/loss plus cross-stage transfers). Timing uses a
thread-local profiler so concurrent ``sxcall`` calls from different
threads stay independent.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
from jax.extend.core import Var

from ...core._weakcache import weak_invalidate
from ...core.graph import bind, export, live_variables
from ...core.module import Module
from ...core.stage_assignment import metadata_stage_assignment, resolve_stage_rank
from ...core.state import State
from ...core.variable import Variable
from ...nn.pipeline_sequential import PipelineSequential
from ...sharding.partition import get_named_sharding, sanitize_partition_spec_for_mesh_and_shape
from ..primitives.split import auto_split
from ..schedules import (
    FusedTask,
    GPipe,
    Phase,
    Schedule,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)
from ..types.array import StagesArray
from ..types.mesh import MpMdMesh, resolve_mpmd_mesh
from .markers import cluster_jaxpr_by_markers, marker_edge_shardings, sxstage_iter_p
from .pscan_compiler import (
    _build_invar_sources,
    _build_logical_locs,
    _build_schedule_grid,
    _collect_used_constvars,
    _filtered_cluster,
    _iter_actions,
    _make_bwd_i_jit,
    _make_bwd_jit,
    _make_bwd_w_jit,
    _make_fwd_jit,
    _make_terminal_jit,
    _materialize_cotangents,
    build_pscan_plan,
    dispatch_pscan,
    has_pscan,
)

if TYPE_CHECKING:
    from ...sharding.mesh import SpxMesh


__all__ = ["collect_task_times_ms", "sxcall", "sxgrad", "sxjit", "sxvalue_and_grad"]

_PROFILER_STATE = threading.local()
_STAGE_CALLABLE_CACHE: dict[int, tuple[Callable[..., Any], Callable[..., Any]]] = {}
_MPMD_SETUP_CACHE: dict[
    tuple[int, int, int],
    tuple[
        dict[tuple[int, int], Callable[..., Any]],
        dict[tuple[int, int], Callable[..., Any]],
        dict[tuple[int, int], State],
        dict[tuple[int, int], State],
        list[Any],
        list[Any],
    ],
] = {}
_INV_M_CACHE: dict[tuple[int, int], jax.Array] = {}
_GPIPE_VMAP_CACHE: dict[int, tuple[Callable[..., Any], Callable[..., Any]]] = {}
_GPIPE_TERM_CACHE: dict[tuple[int, int, str], Callable[..., Any]] = {}
_VMAP_LOSS_CACHE: dict[int, Callable[..., Any]] = {}
_FUSED_FWDBWD_CACHE: dict[tuple[int, int], Callable[..., Any]] = {}
_SCHEDULE_FUSED_FWDBWD_CACHE: dict[tuple[int, ...], Callable[..., Any]] = {}
_SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE: dict[tuple[int, int], Callable[..., Any]] = {}
_TRANSFER_SHARDING_DECISION_CACHE: dict[tuple[int | None, int | None, tuple[int, ...] | None], bool] = {}
_LOSS_JIT_CACHE: dict[int, Callable[..., Any]] = {}
_MPMD_CALL_NORMALIZED_CACHE: dict[tuple[int, int], PipelineSequential] = {}
_FWD_ONLY_VMAP_CACHE: dict[int, Callable[..., Any]] = {}


@dataclass(frozen=True)
class _ScheduleUnit:
    """One executable unit in a schedule-driven MPMD dispatch.

    A unit is the smallest granularity the schedule dispatcher fires:
    either a single :class:`Action` (``kind == "action"``) or a fused
    forward+backward pair (``kind == "fused"``) on the same physical
    rank. Units are produced by walking ``schedule.build(n)`` and form
    the nodes of the dependency DAG used by the async dispatcher.

    Attributes:
        index: Stable global ordering key (insertion order in the unit
            list). Used as the dependency-graph node id.
        row: Source row in the schedule grid; mostly diagnostic.
        kind: Either ``"action"`` (a single :class:`Action`) or
            ``"fused"`` (a :class:`FusedTask`).
        rank: Physical pipeline rank that owns this unit.
        virt: Virtual sub-stage on ``rank`` (always 0 for flat
            schedules with ``V == 1``).
        payload: The underlying :class:`Action` or :class:`FusedTask`.
        fwd_logical: Logical stage index of the forward half (None for
            pure-backward units).
        fwd_mb: Microbatch index of the forward half (None for
            pure-backward units).
        bwd_logical: Logical stage index of the backward half (None
            for pure-forward units).
        bwd_mb: Microbatch index of the backward half (None for
            pure-forward units).
        bwd_phase: Specific backward phase (``Phase.BWD``,
            ``Phase.BWD_I``, or ``Phase.BWD_W``) or ``None`` for
            forward-only units.
    """

    index: int
    row: int
    kind: Literal["action", "fused"]
    rank: int
    virt: int
    payload: Any
    fwd_logical: int | None
    fwd_mb: int | None
    bwd_logical: int | None
    bwd_mb: int | None
    bwd_phase: Phase | None


class _ScheduleStatsCollector:
    """Non-blocking schedule runtime counters.

    Timings are host enqueue durations, not device completion times. They are
    useful for launch/dispatch critical-path analysis without adding per-task
    ``block_until_ready`` calls.
    """

    def __init__(
        self,
        *,
        dispatcher: str,
        unit_count: int | None = None,
        action_count: int | None = None,
        fused_count: int | None = None,
        window_count: int | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        """Initialize an empty stats collector for one ``sxcall`` invocation.

        Args:
            dispatcher: Tag identifying which scheduler path produced
                the units (e.g. ``"async"``, ``"sequential"``,
                ``"gpipe-vmap"``). Reported back in :meth:`as_dict` so
                downstream tooling can attribute timings.
            unit_count: Optional total number of schedule units
                planned. Reported as-is for sanity checking.
            action_count: Optional count of plain (non-fused) actions
                in the schedule.
            fused_count: Optional count of fwd+bwd fused units.
            window_count: Optional count of dependency windows the
                async dispatcher planned.
            fallback_reason: Optional human-readable string explaining
                why a faster path (e.g. GPipe vmap) was *not* taken.
                ``None`` when the fast path was used.
        """
        self.dispatcher = dispatcher
        self.unit_count = unit_count
        self.action_count = action_count
        self.fused_count = fused_count
        self.window_count = window_count
        self.fallback_reason = fallback_reason
        self.transfer_count = 0
        self.transfer_skipped_count = 0
        self.transfer_cache_hit_count = 0
        self.transfer_bytes = 0
        self.transfer_edges: dict[str, dict[str, int]] = {}
        self.per_rank_launch_count: dict[int, int] = {}
        self.per_rank_launch_enqueue_ms: dict[int, float] = {}
        self.per_rank_enqueue_ms: dict[int, float] = {}
        self.unit_enqueue_ms: dict[int, float] = {}
        self.lock = threading.Lock()

    def record_launch(self, rank: int, elapsed_ms: float) -> None:
        """Increment the launch (top-level dispatch) counter for one rank.

        Distinct from :meth:`record_unit`: a launch is the wall time
        spent submitting *all* the rank's units to JAX during this
        sxcall, while a unit timing is per-individual-unit.
        """
        with self.lock:
            self.per_rank_launch_count[rank] = self.per_rank_launch_count.get(rank, 0) + 1
            self.per_rank_launch_enqueue_ms[rank] = self.per_rank_launch_enqueue_ms.get(rank, 0.0) + elapsed_ms

    def record_unit(self, unit_index: int, rank: int, elapsed_ms: float) -> None:
        """Record the host enqueue time for one schedule unit.

        ``elapsed_ms`` is the wall time spent in Python+XLA dispatch
        for this unit; it does *not* include device execution. Used
        to find dispatch hot-spots.
        """
        with self.lock:
            self.unit_enqueue_ms[unit_index] = elapsed_ms
            self.per_rank_enqueue_ms[rank] = self.per_rank_enqueue_ms.get(rank, 0.0) + elapsed_ms

    def record_transfer(
        self,
        *,
        nbytes: int,
        skipped: bool,
        cache_hit: bool,
        src_rank: int | None,
        dst_rank: int | None,
    ) -> None:
        """Record one cross-rank transfer (or attempted transfer) plus its byte size.

        Tallies counts and bytes both globally and per
        ``"src->dst"`` edge. ``skipped=True`` means the transfer was
        elided (e.g. source and target sharding already matched);
        ``cache_hit=True`` means a sharding-decision cache lookup
        was reused. Both flags are also tracked separately so the
        downstream dashboard can show "% skipped".
        """
        edge = f"{src_rank if src_rank is not None else '?'}->{dst_rank if dst_rank is not None else '?'}"
        with self.lock:
            self.transfer_count += 1
            self.transfer_bytes += int(nbytes)
            if skipped:
                self.transfer_skipped_count += 1
            if cache_hit:
                self.transfer_cache_hit_count += 1
            bucket = self.transfer_edges.setdefault(edge, {"count": 0, "bytes": 0, "skipped": 0})
            bucket["count"] += 1
            bucket["bytes"] += int(nbytes)
            if skipped:
                bucket["skipped"] += 1

    def as_dict(
        self, deps: dict[int, set[int]] | None = None, units: list[_ScheduleUnit] | None = None
    ) -> dict[str, Any]:
        """Render the recorded counters as a JSON-friendly dict.

        Optional ``deps``/``units`` compute a critical-path timing
        and per-phase / per-rank-phase breakdowns by walking the
        dependency DAG with memoization: each unit's longest-path
        finish time is the max over its predecessors' finish times
        plus its own enqueue time. Without ``deps``/``units`` the
        result still includes raw counters but no critical path.

        Args:
            deps: Optional mapping ``unit_index -> {predecessor indices}``.
            units: Optional list of all units (in any order); needed
                to look up phase metadata.

        Returns:
            A nested dict with the schedule's totals, per-rank
            timings, per-phase timings, and the top-N highest-cost
            units (capped at 16) sorted by enqueue time.
        """
        per_rank_critical_path_ms: dict[int, float] = {}
        critical_path_ms = 0.0
        per_phase_enqueue_ms: dict[str, float] = {}
        per_rank_phase_enqueue_ms: dict[str, float] = {}
        top_unit_enqueue_ms: list[dict[str, Any]] = []
        total_unit_enqueue_ms = 0.0
        total_launch_enqueue_ms = 0.0
        if deps is not None and units is not None:
            {unit.index: unit for unit in units}
            memo: dict[int, float] = {}

            def cp(idx: int) -> float:
                """Memoised critical-path (longest-finish) time for unit ``idx``.

                The critical path of any unit is the maximum critical
                path of its predecessors plus the unit's own enqueue
                time. Walking the DAG with memoisation collapses the
                recursion to ``O(|edges|)`` even when the graph has
                wide fan-in.
                """
                if idx in memo:
                    return memo[idx]
                dep_best = max((cp(dep) for dep in deps.get(idx, ())), default=0.0)
                total = dep_best + self.unit_enqueue_ms.get(idx, 0.0)
                memo[idx] = total
                return total

            for unit in units:
                value = cp(unit.index)
                critical_path_ms = max(critical_path_ms, value)
                rank = unit.rank
                per_rank_critical_path_ms[rank] = max(per_rank_critical_path_ms.get(rank, 0.0), value)

            def unit_phase(unit: _ScheduleUnit) -> str:
                """Classify a schedule unit by execution phase.

                Returns one of ``"fused"``, ``"fwd"``, the lowercased
                ``Phase.name`` for backward variants
                (e.g. ``"bwd"``, ``"bwd_w"``), or ``"unknown"`` if
                the unit doesn't carry phase metadata.
                """
                if unit.kind == "fused":
                    return unit.kind
                if unit.fwd_logical is not None:
                    return "fwd"
                if unit.bwd_phase is not None:
                    return unit.bwd_phase.name.lower()
                return "unknown"

            def unit_label(unit: _ScheduleUnit) -> str:
                """Render a short human-readable label for one schedule unit.

                Format examples:

                * ``r1/v0 fwd L2:mb3`` — forward of microbatch 3 on
                  logical stage 2 at rank 1, virtual 0.
                * ``r0/v0 bwd_w L0:mb1`` — weight-grad backward.
                * ``r2/v0 fused L4:fwd_mb0+bwd_mb2`` — paired fused
                  task.
                """
                phase = unit_phase(unit)
                loc = f"r{unit.rank}/v{unit.virt}"
                if unit.kind == "fused":
                    return f"{loc} fused L{unit.fwd_logical}:fwd_mb{unit.fwd_mb}+bwd_mb{unit.bwd_mb}"
                if unit.fwd_logical is not None:
                    return f"{loc} fwd L{unit.fwd_logical}:mb{unit.fwd_mb}"
                return f"{loc} {phase} L{unit.bwd_logical}:mb{unit.bwd_mb}"

            for unit in units:
                elapsed = float(self.unit_enqueue_ms.get(unit.index, 0.0))
                total_unit_enqueue_ms += elapsed
                phase = unit_phase(unit)
                per_phase_enqueue_ms[phase] = per_phase_enqueue_ms.get(phase, 0.0) + elapsed
                rank_phase = f"r{unit.rank}:{phase}"
                per_rank_phase_enqueue_ms[rank_phase] = per_rank_phase_enqueue_ms.get(rank_phase, 0.0) + elapsed
                top_unit_enqueue_ms.append(
                    {
                        "index": unit.index,
                        "row": unit.row,
                        "rank": unit.rank,
                        "virt": unit.virt,
                        "phase": phase,
                        "fwd_logical": unit.fwd_logical,
                        "fwd_mb": unit.fwd_mb,
                        "bwd_logical": unit.bwd_logical,
                        "bwd_mb": unit.bwd_mb,
                        "elapsed_ms": round(elapsed, 3),
                        "label": unit_label(unit),
                    }
                )
            top_unit_enqueue_ms.sort(key=lambda item: item["elapsed_ms"], reverse=True)
            top_unit_enqueue_ms = top_unit_enqueue_ms[:16]
        total_launch_enqueue_ms = sum(float(value) for value in self.per_rank_launch_enqueue_ms.values())

        return {
            "dispatcher": self.dispatcher,
            "unit_count": self.unit_count,
            "action_count": self.action_count,
            "fused_count": self.fused_count,
            "window_count": self.window_count,
            "fallback_reason": self.fallback_reason,
            "transfer_count": self.transfer_count,
            "transfer_skipped_count": self.transfer_skipped_count,
            "transfer_cache_hit_count": self.transfer_cache_hit_count,
            "transfer_bytes": self.transfer_bytes,
            "transfer_edges": dict(sorted(self.transfer_edges.items())),
            "total_launch_enqueue_ms": round(total_launch_enqueue_ms, 3),
            "total_unit_enqueue_ms": round(total_unit_enqueue_ms, 3),
            "per_rank_launch_count": dict(sorted(self.per_rank_launch_count.items())),
            "per_rank_launch_enqueue_ms": {k: round(v, 3) for k, v in sorted(self.per_rank_launch_enqueue_ms.items())},
            "per_rank_enqueue_ms": {k: round(v, 3) for k, v in sorted(self.per_rank_enqueue_ms.items())},
            "per_phase_enqueue_ms": {k: round(v, 3) for k, v in sorted(per_phase_enqueue_ms.items())},
            "per_rank_phase_enqueue_ms": {k: round(v, 3) for k, v in sorted(per_rank_phase_enqueue_ms.items())},
            "top_unit_enqueue_ms": top_unit_enqueue_ms,
            "per_rank_critical_path_ms": {k: round(v, 3) for k, v in sorted(per_rank_critical_path_ms.items())},
            "critical_path_ms": round(critical_path_ms, 3),
        }


def _arg_leaf_ranges(args: tuple[Any, ...]) -> list[tuple[int, int]]:
    """Return ``[start, end)`` flat-leaf index ranges for each positional argument.

    Used when a downstream pass needs to map an outer-jaxpr flat-arg
    index back to which user-supplied positional argument it came from
    (e.g. to decide which leaves belong to the captured Module).

    Args:
        args: The user-facing positional arguments before flattening.

    Returns:
        A list parallel to ``args``; entry ``i`` is the half-open
        leaf-index range that ``args[i]`` occupies in the flat-leaf
        tuple.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    for arg in args:
        n_leaves = len(jax.tree.leaves(arg))
        ranges.append((start, start + n_leaves))
        start += n_leaves
    return ranges


def _normalize_argnums(argnums: int | tuple[int, ...] | None, total: int) -> tuple[int, ...]:
    """Coerce ``argnums`` to a validated tuple of non-negative indices.

    Accepts ``None`` (returns empty tuple), a single int, or any
    iterable of ints. Negative values are interpreted Python-style
    relative to ``total``. Raises if any resolved index falls outside
    ``[0, total)``.

    Args:
        argnums: User-supplied gradient argnum spec.
        total: Total number of positional arguments to ``fn``.

    Returns:
        Normalised tuple of indices in ``[0, total)``.

    Raises:
        ValueError: If a normalised index is out of range.
    """
    if argnums is None:
        return ()
    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)
    normalized = []
    for i in argnums:
        if i < 0:
            i += total
        if i < 0 or i >= total:
            raise ValueError(f"argnum {i} out of range for function with {total} positional arguments")
        normalized.append(i)
    return tuple(normalized)


def _normalize_argnames(argnames: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Coerce ``argnames`` to a tuple, accepting ``None``/``str``/iterable.

    Mirrors :func:`_normalize_argnums` for keyword-style arguments
    that ``sxjit`` treats as static. Empty for ``None``, single-element
    for a bare string, otherwise the iterable as a tuple.
    """
    if argnames is None:
        return ()
    if isinstance(argnames, str):
        return (argnames,)
    return tuple(argnames)


def _result_treedef_for_call(
    fn: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    static_argnums: int | tuple[int, ...] | None,
    static_argnames: str | tuple[str, ...] | None,
) -> Any | None:
    """Trace ``fn`` symbolically to capture its output pytree structure.

    The MPMD runtime returns flat tuples of arrays from per-rank
    dispatch; this helper runs :func:`jax.eval_shape` once with the
    same static/dynamic argument split sxjit will use, so the wrapper
    can later re-pack flat tuples into the user's nested return
    pytree. Failures (e.g. ``fn`` cannot be eval-shaped because it
    requires concrete data) return ``None`` and the wrapper passes
    flat tuples through unchanged.

    Args:
        fn: The user-decorated function.
        args: Positional arguments at the current call.
        kwargs: Keyword arguments at the current call.
        static_argnums: Argnums treated as static.
        static_argnames: Argnames treated as static.

    Returns:
        A :func:`jax.tree_util.PyTreeDef` describing ``fn``'s output
        structure, or ``None`` when tracing failed.
    """
    static_nums = set(_normalize_argnums(static_argnums, len(args)))
    static_names = set(_normalize_argnames(static_argnames))
    dynamic_nums = tuple(i for i in range(len(args)) if i not in static_nums)
    dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_names}
    static_kwargs = {k: kwargs[k] for k in static_names if k in kwargs}

    def _shape_fn(*dynamic_call_args, **dynamic_call_kwargs):
        """Stitch dynamic+static args back into ``fn(*args, **kwargs)`` for ``eval_shape``."""
        call_args = list(args)
        for idx, value in zip(dynamic_nums, dynamic_call_args, strict=False):
            call_args[idx] = value
        call_kwargs = dict(static_kwargs)
        call_kwargs.update(dynamic_call_kwargs)
        return fn(*call_args, **call_kwargs)

    try:
        template = jax.eval_shape(_shape_fn, *(args[i] for i in dynamic_nums), **dynamic_kwargs)
    except Exception:
        return None
    return jax.tree_util.tree_structure(template)


def _restore_result_treedef(result: Any, treedef: Any | None) -> Any:
    """Re-pack a flat result tuple into the user's original output pytree.

    ``sxjit`` returns a flat tuple of arrays from the runtime, but the
    user's function may have returned a dict, namedtuple, or other
    nested pytree. The captured ``treedef`` (from
    :func:`_result_treedef_for_call`) is used to reconstruct the
    nesting. A ``None`` treedef (eval_shape failed) or a length
    mismatch falls back to returning the flat tuple unchanged.
    """
    if treedef is None:
        return result
    if isinstance(result, tuple) and len(result) == treedef.num_leaves:
        return jax.tree_util.tree_unflatten(treedef, list(result))
    return result


def _has_array_leaf(x: Any) -> bool:
    """Return ``True`` iff ``x`` flattens to at least one array-like leaf.

    Used to distinguish "real" runtime data (which should stay dynamic)
    from pure metadata such as ints, dataclasses, or schedule objects
    (which sxjit can safely treat as static argnum candidates).

    Args:
        x: A pytree to inspect.

    Returns:
        ``True`` when at least one leaf is a :class:`jax.Array`,
        carries a ``__jax_array__`` protocol, or has both ``shape``
        and ``dtype`` attributes.
    """
    leaves = jax.tree.leaves(x, is_leaf=_is_leaf)
    return any(
        isinstance(leaf, jax.Array)
        or hasattr(leaf, "__jax_array__")
        or (hasattr(leaf, "shape") and hasattr(leaf, "dtype"))
        for leaf in leaves
    )


def _infer_schedule_static_argnums(args: tuple[Any, ...]) -> tuple[int, ...]:
    """Infer schedule static args without freezing array batches.

    Module arguments are staged as constants so schedule gradients can flow
    through their parameter leaves. Plain array pytrees, such as input batches,
    stay dynamic by default. Non-array metadata remains static.
    """
    return tuple(i for i, arg in enumerate(args) if isinstance(arg, Module) or not _has_array_leaf(arg))


def _compute_donation(
    clusters: list,
    original_id_to_idx: dict[int, int],
    orig_flat_to_dynamic_flat: dict[int, int],
    args: tuple[Any, ...],
    donate_nums: set[int],
    static_nums: set[int],
    n: int,
) -> list[tuple[int, ...]]:
    """Translate user-facing ``donate_argnums`` into per-stage jit donate-argnum tuples.

    Each top-level argument may flatten to many leaves, and each leaf
    may flow into one or more cluster sub-jaxprs. Donation is only
    safe when a leaf is consumed by exactly one cluster (otherwise XLA
    would invalidate the buffer mid-pipeline). For each donated
    argument we walk its flat leaves, find which ``(rank, invar_pos)``
    pairs they end up at, and only record the donation when the leaf
    is single-use.

    Args:
        clusters: Per-stage marker-clustered sub-jaxprs.
        original_id_to_idx: ``id(jaxpr_invar) -> dynamic-flat index``
            mapping from the outer jaxpr.
        orig_flat_to_dynamic_flat: Mapping from the user-flat index
            (no static args removed) to the dynamic-flat index used
            inside the outer jaxpr.
        args: The original positional arguments.
        donate_nums: User's donate-argnum spec.
        static_nums: Argnums that are treated as static.
        n: Number of physical pipeline ranks.

    Returns:
        ``donate_per_stage[rank]`` is the sorted tuple of cluster
        invar positions safe to donate at that rank.

    Raises:
        ValueError: If a donated arg is also marked static.
    """
    donate_per_stage: list[set[int]] = [set() for _ in range(n)]
    for donate_num in donate_nums:
        if donate_num in static_nums:
            raise ValueError(
                f"sxjit: cannot donate static argument at index {donate_num}. "
                "Static arguments are compile-time constants and cannot be donated."
            )
        flat_start = sum(len(jax.tree.leaves(args[i])) for i in range(donate_num))
        n_leaves = len(jax.tree.leaves(args[donate_num]))
        for leaf_offset in range(n_leaves):
            orig_flat = flat_start + leaf_offset
            dyn_flat = orig_flat_to_dynamic_flat.get(orig_flat)
            if dyn_flat is None:
                continue
            used_by: list[tuple[int, int]] = []
            for rank, cluster in enumerate(clusters):
                for pos, v in enumerate(cluster.invars):
                    if original_id_to_idx.get(id(v)) == dyn_flat:
                        used_by.append((rank, pos))
            if len(used_by) == 1:
                rank, pos = used_by[0]
                donate_per_stage[rank].add(pos)
    return [tuple(sorted(s)) for s in donate_per_stage]


def _array_device_set(value: Any) -> set[Any] | None:
    """Return the device set holding ``value``'s shards, or ``None`` when unknown.

    Tries the ``.devices`` attribute on the array (callable on newer
    JAX, plain attribute on older). Returns ``None`` whenever the
    object is not a JAX array (Python scalars, ``None``, etc.) or
    when the attribute access raises.

    Args:
        value: Any value (array or otherwise).

    Returns:
        A set of :class:`jax.Device` objects or ``None``.
    """
    devices = getattr(value, "devices", None)
    if devices is None:
        return None
    try:
        return set(devices() if callable(devices) else devices)
    except Exception:
        return None


def _device_id_tuple(devices: set[Any] | None) -> tuple[int, ...] | None:
    """Return a sorted tuple of integer device IDs (or ``None``).

    Used as a stable, hashable key for sharding-decision caches —
    ``set`` itself is not hashable, and device objects don't always
    sort by their natural ``__lt__``.
    """
    if devices is None:
        return None
    return tuple(sorted(int(getattr(device, "id", idx)) for idx, device in enumerate(devices)))


def _sharding_device_set(sharding: Any) -> set[Any] | None:
    """Return the device set backing a sharding spec, or ``None``.

    Tries the spec's ``device_set`` attribute first (callable or
    direct), then falls back to flattening the spec's ``mesh.devices``.
    Used to compare two shardings for "same physical placement"
    independent of axis layout.
    """
    devices = getattr(sharding, "device_set", None)
    if devices is not None:
        try:
            return set(devices() if callable(devices) else devices)
        except Exception:
            pass
    mesh = getattr(sharding, "mesh", None)
    if mesh is not None:
        try:
            return set(mesh.devices.flat)
        except Exception:
            return None
    return None


def _tree_nbytes(x: Any) -> int:
    """Sum the byte sizes of every array leaf in ``x`` without touching devices.

    Computes ``size * dtype.itemsize`` per leaf, ignoring leaves that
    are not arrays. The result is reported to the schedule stats
    collector as a transfer-size estimate. No :func:`block_until_ready`
    is issued so the cost is purely metadata access.

    Args:
        x: A pytree whose array leaves should be measured.

    Returns:
        Total bytes across all array leaves.
    """
    total = 0
    for leaf in jax.tree.leaves(x, is_leaf=_is_leaf):
        size = getattr(leaf, "size", None)
        dtype = getattr(leaf, "dtype", None)
        if size is None or dtype is None:
            continue
        try:
            total += int(size) * int(jnp.dtype(dtype).itemsize)
        except Exception:
            continue
    return total


def _same_sharding(a: Any, b: Any) -> bool:
    """Return ``True`` iff ``a`` and ``b`` represent the same physical sharding.

    Reflexive equality is checked first, then type, partition spec,
    and finally physical device set equivalence. Two shardings can
    have the same spec but different meshes (different device sets);
    both must match. ``None`` operands always compare unequal so that
    "no sharding" is never confused with "any sharding".
    """
    if a is None or b is None:
        return False
    if a == b:
        return True
    if type(a) is not type(b):
        return False
    if getattr(a, "spec", None) != getattr(b, "spec", None):
        return False
    return _sharding_device_set(a) == _sharding_device_set(b)


def _partition_spec_axes(spec: Any) -> set[str]:
    """Return the flat set of mesh-axis names referenced anywhere in ``spec``.

    Handles three encodings: ``None`` per dim (skipped), a bare string
    (added), or a sub-tuple of strings (flattened). Used to test
    whether a spec is "compatible" with a given mesh's axis-name set.
    """
    axes: set[str] = set()
    if spec is None:
        return axes
    try:
        parts = tuple(spec)
    except Exception:
        return axes
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            axes.add(part)
        else:
            try:
                axes.update(axis for axis in part if isinstance(axis, str))
            except TypeError:
                continue
    return axes


def _retarget_transfer_sharding(value: Any, fallback_sharding: Any) -> Any:
    """Re-bind each leaf's intra-stage partition spec to the destination mesh.

    When transporting an activation across pipeline ranks, the leaf
    may already have a non-trivial partition spec (e.g. for tensor
    parallelism inside the source rank). If the destination mesh has
    matching axis names, we re-wrap the same spec on the destination
    mesh so the TP layout survives the transport — otherwise we fall
    back to ``fallback_sharding`` (the destination rank's default).

    Args:
        value: The array (or pytree of arrays) about to be moved.
        fallback_sharding: The default destination sharding, typically
            the destination rank's replicated sharding.

    Returns:
        Either ``fallback_sharding`` directly (when no leaf benefits
        from re-binding) or a pytree of per-leaf shardings matching
        ``value``'s structure.
    """
    fallback_mesh = getattr(fallback_sharding, "mesh", None)
    if fallback_mesh is None:
        return fallback_sharding
    mesh_axes = set(getattr(fallback_mesh, "axis_names", ()))

    def leaf_target(leaf: Any) -> Any:
        """Pick a per-leaf placement for the cross-rank transfer.

        If the leaf's existing partition spec uses only axes present
        on the destination mesh and is non-trivial (not all-replicated),
        re-bind that spec to the destination mesh; otherwise fall
        through to the caller's ``fallback_sharding`` (typically the
        destination stage's default sharding).
        """
        current = getattr(leaf, "sharding", None)
        spec = getattr(current, "spec", None)
        if spec is None:
            return fallback_sharding
        if not _partition_spec_axes(spec).issubset(mesh_axes):
            return fallback_sharding
        try:
            if all(part is None for part in tuple(spec)):
                return fallback_sharding
        except Exception:
            return fallback_sharding
        return jax.sharding.NamedSharding(fallback_mesh, spec)

    leaves, treedef = jax.tree.flatten(value, is_leaf=_is_leaf)
    if not leaves:
        return fallback_sharding
    targets = [leaf_target(leaf) for leaf in leaves]
    if all(target is fallback_sharding for target in targets):
        return fallback_sharding
    return jax.tree.unflatten(treedef, targets)


def _can_skip_device_put(value: Any, dest_sharding: Any) -> tuple[bool, bool]:
    """Decide whether ``jax.device_put(value, dest_sharding)`` would be a no-op.

    Memoised in :data:`_TRANSFER_SHARDING_DECISION_CACHE` keyed by the
    Python ids of the source sharding, destination sharding, and the
    value's device set, so repeated transfers of similarly-placed
    arrays skip the comparison after the first time.

    Args:
        value: The candidate array to transport.
        dest_sharding: Target sharding (or ``None`` to skip the check).

    Returns:
        ``(skip, cache_hit)`` — ``skip`` is ``True`` when the
        transport can be elided (matching shardings or matching device
        sets); ``cache_hit`` is ``True`` when the answer came from the
        memo rather than a fresh comparison.
    """
    current = getattr(value, "sharding", None)
    value_devices = _array_device_set(value)
    dest_devices = _sharding_device_set(dest_sharding)
    key = (
        id(current) if current is not None else None,
        id(dest_sharding) if dest_sharding is not None else None,
        _device_id_tuple(value_devices),
    )
    cached = _TRANSFER_SHARDING_DECISION_CACHE.get(key)
    if cached is not None:
        return cached, True
    skip = False
    if dest_sharding is not None:
        skip = _same_sharding(current, dest_sharding)
        if not skip and value_devices is not None and dest_devices is not None:
            skip = value_devices == dest_devices
    _TRANSFER_SHARDING_DECISION_CACHE[key] = skip
    return skip, False


def _is_replicated_sharding(sharding: Any) -> bool:
    """Return ``True`` when ``sharding``'s spec replicates over every dimension.

    Detects shardings where every entry of the underlying
    :class:`PartitionSpec` is ``None`` (no axis sharded). Falls back to
    ``False`` if the object exposes no ``.spec`` attribute or fails
    iteration — it is a best-effort fast path for the placement
    decision in :func:`_place_schedule_const_value`.

    Args:
        sharding: Any sharding-like object (typically
            :class:`NamedSharding`).

    Returns:
        ``True`` only when every dimension is replicated.
    """
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return False
    try:
        return all(part is None for part in tuple(spec))
    except Exception:
        return False


def _place_schedule_const_value(
    value: Any,
    *,
    loc: tuple[int, int],
    flat_idx: int | None,
    leaf_shardings: list[dict[int, Any]],
    leaf_stage_owners: dict[int, int],
    stage_shardings: list[Any],
    rank_submeshes: list[Any],
) -> Any:
    """Place one schedule const while preserving per-leaf intra-stage sharding.

    Schedule splitting traces module arguments as closed-over consts. Those
    consts are still runtime arguments to the per-stage JITs, so their placement
    must follow the original variable sharding metadata instead of blindly
    replicating across the whole stage sub-mesh.
    """
    rank = loc[0]
    if flat_idx is not None:
        owner = leaf_stage_owners.get(flat_idx)
        if owner is not None and owner != rank:
            raise ValueError(
                f"sxjit: flat argument leaf {flat_idx} is assigned to pipeline "
                f"stage {owner}, but traced stage {rank} uses it. Move the "
                "corresponding layer into the matching pipeline segment or "
                "update its assign_stage(...) hint."
            )
        else:
            target = leaf_shardings[rank].get(flat_idx)
    else:
        target = None

    rank_devices = set(rank_submeshes[rank].devices.flat)
    value_devices = _array_device_set(value)
    current_sharding = getattr(value, "sharding", None)
    if value_devices == rank_devices:
        if target is None or current_sharding == target:
            return value
        current_is_replicated = _is_replicated_sharding(current_sharding)
        target_is_replicated = _is_replicated_sharding(target)
        if not (current_is_replicated and not target_is_replicated):
            return value
    if target is not None:
        return jax.device_put(value, target)
    if value_devices is not None and value_devices.issubset(rank_devices):
        return value
    return jax.device_put(value, stage_shardings[rank])


def _build_schedule_plan(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    schedule: Schedule,
    mpmd_mesh: MpMdMesh,
    stage_shardings: list,
    rank_submeshes: list,
    static_argnums: tuple[int, ...] | None,
    donate_argnums: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build the per-call dispatch plan for schedule-driven :func:`sxjit`.

    Trace ``fn`` once with the schedule context in scope to produce the
    body jaxpr, cluster it by :func:`sxstage_iter` markers, map every
    logical stage onto its ``(rank, virt)`` location via
    ``schedule.logical_at``/``next_logical_loc``, build per-location
    forward / backward / terminal jits, place all const tensors on
    their owning rank's sub-mesh, and pre-compute the schedule grid +
    invar-source table consumed at dispatch time.

    Args:
        fn: User function being compiled by :func:`sxjit`.
        args: Positional arguments captured at trace time.
        kwargs: Keyword arguments captured at trace time.
        schedule: Active :class:`Schedule`.
        mpmd_mesh: The MPMD mesh whose ``mpmd_dim`` equals the number
            of physical pipeline ranks.
        stage_shardings: Per-rank replicated shardings (one per
            physical rank).
        rank_submeshes: Per-rank sub-meshes (one per physical rank).
        static_argnums: Optional explicit static-argnum spec; ``None``
            triggers heuristic inference via
            :func:`_infer_schedule_static_argnums`.
        donate_argnums: Optional argnums whose buffers may be donated
            into the compiled stage jits.

    Returns:
        A plan dict consumed by the schedule dispatchers
        (:func:`_dispatch_schedule_faithful`,
        :func:`_dispatch_schedule_fused_async`, etc.) and the
        forward/backward/terminal jits keyed by ``(rank, virt)``.
    """
    n = mpmd_mesh.mpmd_dim
    v = schedule.virtual_stages_per_rank()
    n_logical = n * v
    m = schedule.microbatches

    if static_argnums is None:
        static_nums = set(_infer_schedule_static_argnums(args))
    else:
        static_nums = set(_normalize_argnums(static_argnums, len(args)))
    dynamic_argnums = tuple(i for i in range(len(args)) if i not in static_nums)

    placeholder_args = list(args)
    for i in dynamic_argnums:
        placeholder_args[i] = None
    placeholder_args = tuple(placeholder_args)

    def _wrapper(*dyn_args: Any) -> Any:
        """Re-pack dynamic args back into the original ``fn(*args, **kwargs)`` call.

        Static args are baked in via ``placeholder_args`` (the closure
        list, with dynamic slots set to ``None`` until filled here).
        Used by :func:`jax.make_jaxpr` so the produced jaxpr's invars
        contain only the dynamic positional arguments, matching the
        per-stage runtime invocation.
        """
        full_args = list(placeholder_args)
        for idx, darg in zip(dynamic_argnums, dyn_args, strict=False):
            full_args[idx] = darg
        return fn(*full_args, **kwargs)

    mb_dynamic_args = tuple(jax.tree.map(lambda a: _microbatch(a, m)[0], args[i]) for i in dynamic_argnums)
    closed_jaxpr = jax.make_jaxpr(_wrapper)(*mb_dynamic_args)

    edge_shardings = marker_edge_shardings(closed_jaxpr.jaxpr)
    clusters = cluster_jaxpr_by_markers(closed_jaxpr.jaxpr)
    if len(clusters) != n_logical:
        raise ValueError(
            f"sxjit schedule path: function has {len(clusters)} stages "
            f"({len(clusters) - 1} sxstage_iter markers) but mesh has "
            f"{n} ranks with V={v} virtual stages. Need exactly "
            f"{n_logical} stages ({n_logical - 1} markers)."
        )

    loc_for_logical, logical_for_loc, terminal_loc = _build_logical_locs(schedule, n, v)
    invar_sources = _build_invar_sources(closed_jaxpr.jaxpr, clusters)

    all_constvars = list(closed_jaxpr.jaxpr.constvars)
    concrete_consts = tuple(closed_jaxpr.consts)
    all_const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    flat_args = jax.tree.leaves(args)

    def _schedule_stage_owner(assignment: tuple[int, int] | None) -> int | None:
        """Map a stage assignment (current, total) to its physical rank.

        Handles two cases: assignments expressed in physical-rank space
        (``total <= n``) resolve directly; assignments in logical-stage
        space (``total > n``) are first resolved to a logical index and
        then translated through ``loc_for_logical`` to get the rank.
        Returns ``None`` for unassigned values or out-of-range logicals.
        """
        if assignment is None:
            return None
        _current, total = assignment
        if total <= n:
            return resolve_stage_rank(assignment, n)
        logical = resolve_stage_rank(assignment, n_logical)
        if logical is None:
            return None
        return loc_for_logical[logical][0]

    leaf_shardings, leaf_stage_owners = _infer_leaf_shardings(
        args,
        flat_args,
        n,
        rank_submeshes,
        stage_rank_resolver=_schedule_stage_owner,
    )
    const_idx_to_flat_idx: dict[int, int] = {}
    for ci, cval in enumerate(concrete_consts):
        for fi, fval in enumerate(flat_args):
            if fval is cval:
                const_idx_to_flat_idx[ci] = fi
                break

    dynamic_flat_to_global_flat: dict[int, int] = {}
    global_ranges = _arg_leaf_ranges(args)
    dyn_local_idx = 0
    for arg_idx in dynamic_argnums:
        g_start, _g_end = global_ranges[arg_idx]
        local_leaves = jax.tree.leaves(args[arg_idx])
        for li, _leaf in enumerate(local_leaves):
            dynamic_flat_to_global_flat[dyn_local_idx] = g_start + li
            dyn_local_idx += 1

    donate_nums = set(_normalize_argnums(donate_argnums, len(args))) if donate_argnums is not None else set()
    donate_invars_per_logical: dict[int, set[int]] = {i: set() for i in range(n_logical)}
    if donate_nums:
        for donate_num in donate_nums:
            if donate_num in static_nums:
                raise ValueError(
                    f"sxjit: cannot donate static argument at index {donate_num}. "
                    "Static arguments are compile-time constants and cannot be donated."
                )
        start_end = global_ranges
        for donate_num in donate_nums:
            dstart, dend = start_end[donate_num]
            for dyn_idx, global_idx in dynamic_flat_to_global_flat.items():
                if dstart <= global_idx < dend:
                    used_by: list[tuple[int, int]] = []
                    for logical, sources in enumerate(invar_sources):
                        for invar_pos, (kind, src_a, _src_b) in enumerate(sources):
                            if kind == "body_invar" and src_a == dyn_idx:
                                used_by.append((logical, invar_pos))
                    if len(used_by) == 1:
                        logical, invar_pos = used_by[0]
                        donate_invars_per_logical[logical].add(invar_pos)

    per_loc_consts: dict[tuple[int, int], tuple[Any, ...]] = {}
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]] = {}
    n_invars_per_loc: dict[tuple[int, int], int] = {}
    cluster_jaxprs_per_loc: dict[tuple[int, int], Any] = {}
    fwd_jits: dict[tuple[int, int], Callable[..., Any]] = {}
    bwd_jits: dict[tuple[int, int], Callable[..., Any] | None] = {}
    bwd_i_jits: dict[tuple[int, int], Callable[..., Any] | None] = {}
    bwd_w_jits: dict[tuple[int, int], Callable[..., Any] | None] = {}
    terminal_jit: Callable[..., Any] | None = None

    for logical, cluster in enumerate(clusters):
        loc = loc_for_logical[logical]
        _rank, _virt = loc
        used_constvars = _collect_used_constvars(cluster)
        filtered_cluster = _filtered_cluster(cluster, used_constvars)
        const_indices = tuple(all_const_idx_by_id[id(v)] for v in used_constvars)
        n_invars = len(cluster.invars)

        placed_consts = tuple(
            _place_schedule_const_value(
                concrete_consts[idx],
                loc=loc,
                flat_idx=const_idx_to_flat_idx.get(idx),
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )
            for idx in const_indices
        )

        per_loc_consts[loc] = placed_consts
        const_indices_per_loc[loc] = const_indices
        n_invars_per_loc[loc] = n_invars
        cluster_jaxprs_per_loc[loc] = filtered_cluster
        donate_positions = tuple(1 + pos for pos in sorted(donate_invars_per_logical[logical]))
        fwd_jits[loc] = _make_fwd_jit(filtered_cluster, donate_argnums=donate_positions)

        if loc != terminal_loc:
            bwd_jits[loc] = _make_bwd_jit(filtered_cluster, n_invars, donate_argnums=donate_positions)
            bwd_i_jits[loc] = _make_bwd_i_jit(filtered_cluster, n_invars, donate_argnums=donate_positions)
            bwd_w_jits[loc] = _make_bwd_w_jit(filtered_cluster, n_invars, donate_argnums=donate_positions)
        else:
            bwd_jits[loc] = None
            bwd_i_jits[loc] = None
            bwd_w_jits[loc] = None
            terminal_jit = _make_terminal_jit(filtered_cluster, n_invars, donate_argnums=donate_positions)

    assert terminal_jit is not None

    vbwd_jits: dict[tuple[int, int], Callable[..., Any]] = {}
    if schedule.lazy_bwd_batching:
        for logical, loc in enumerate(loc_for_logical):
            if loc == terminal_loc:
                continue
            n_invars = n_invars_per_loc[loc]
            n_outs = len(clusters[logical].outvars)
            bwd = bwd_jits[loc]
            vbwd = jax.jit(jax.vmap(bwd, in_axes=(None,) + (0,) * n_invars + (0,) * n_outs))
            vbwd_jits[loc] = vbwd

    grid = _build_schedule_grid(schedule, n)

    n_flat = len(flat_args)
    dynamic_mask = [False] * n_flat
    for argnum in dynamic_argnums:
        start, end = global_ranges[argnum]
        for i in range(start, end):
            dynamic_mask[i] = True

    return {
        "n": n,
        "v": v,
        "n_logical": n_logical,
        "m": m,
        "schedule": schedule,
        "loc_for_logical": loc_for_logical,
        "logical_for_loc": logical_for_loc,
        "terminal_loc": terminal_loc,
        "invar_sources": invar_sources,
        "per_loc_consts": per_loc_consts,
        "const_indices_per_loc": const_indices_per_loc,
        "n_invars_per_loc": n_invars_per_loc,
        "cluster_jaxprs_per_loc": cluster_jaxprs_per_loc,
        "fwd_jits": fwd_jits,
        "bwd_jits": bwd_jits,
        "bwd_i_jits": bwd_i_jits,
        "bwd_w_jits": bwd_w_jits,
        "terminal_jit": terminal_jit,
        "vbwd_jits": vbwd_jits,
        "grid": grid,
        "stage_shardings": stage_shardings,
        "rank_submeshes": rank_submeshes,
        "edge_shardings": edge_shardings,
        "mpmd_mesh": mpmd_mesh,
        "dynamic_mask": dynamic_mask,
        "n_flat": n_flat,
        "flat_args": flat_args,
        "const_idx_to_flat_idx": const_idx_to_flat_idx,
        "dynamic_flat_to_global_flat": dynamic_flat_to_global_flat,
        "leaf_shardings": leaf_shardings,
        "leaf_stage_owners": leaf_stage_owners,
        "clusters": clusters,
    }


def _schedule_per_call_consts(plan: dict[str, Any], args: tuple[Any, ...]) -> dict[tuple[int, int], tuple[Any, ...]]:
    """Return stage const tuples rebound to the live call's argument leaves.

    Schedule planning traces module/metadata arguments as closed-over consts so
    the graph can be split into stage jaxprs. The compiled stage functions still
    take those consts as explicit runtime arguments, so trainable arrays must be
    refreshed from the current call instead of frozen from the first trace.
    """
    flat_args_live = jax.tree.leaves(args)
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    const_indices_per_loc = plan["const_indices_per_loc"]
    stage_shardings = plan["stage_shardings"]
    rank_submeshes = plan["rank_submeshes"]
    leaf_shardings = plan["leaf_shardings"]
    leaf_stage_owners = plan["leaf_stage_owners"]
    rebound: dict[tuple[int, int], tuple[Any, ...]] = {}

    for loc, planned_consts in plan["per_loc_consts"].items():
        consts = list(planned_consts)
        changed = False
        for local_idx, const_idx in enumerate(const_indices_per_loc[loc]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            consts[local_idx] = _place_schedule_const_value(
                flat_args_live[flat_idx],
                loc=loc,
                flat_idx=flat_idx,
                leaf_shardings=leaf_shardings,
                leaf_stage_owners=leaf_stage_owners,
                stage_shardings=stage_shardings,
                rank_submeshes=rank_submeshes,
            )
            changed = True
        rebound[loc] = tuple(consts) if changed else planned_consts
    return rebound


def _dispatch_gpipe_fwd(
    plan: dict[str, Any],
    args: tuple,
) -> tuple[jax.Array, dict[str, Any]]:
    """All-forward path for schedule-driven ``sxjit``.

    Splits dynamic args into microbatches, walks the logical pipeline
    forward one microbatch at a time, and returns the scalar loss plus
    saved activations.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    loc_for_logical = plan["loc_for_logical"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    terminal_jit = plan["terminal_jit"]
    terminal_loc = plan["terminal_loc"]
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    flat_args = jax.tree.leaves(args)

    mb_args: list[Any] = []
    for i, arg in enumerate(flat_args):
        if dynamic_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    saved_outputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    loss_acc = jnp.asarray(0.0)

    for mb in range(m):
        for logical in range(n_logical):
            loc = loc_for_logical[logical]
            rank, virt = loc
            consts = per_loc_consts[loc]
            submesh = rank_submeshes[rank]

            invars: list[Any] = []
            for source_kind, source_a, source_b in invar_sources[logical]:
                if source_kind == "body_invar":
                    flat_idx = dynamic_flat_to_global_flat[source_a]
                    val = mb_args[flat_idx]
                    if dynamic_mask[flat_idx]:
                        val = val[mb]
                    invars.append(val)
                elif source_kind == "cluster_out":
                    producer_loc = loc_for_logical[source_a]
                    val = saved_outputs[(producer_loc[0], producer_loc[1], mb)][source_b]
                    if producer_loc[0] != rank:
                        val = jax.device_put(
                            val,
                            _transfer_target_for_edge(
                                val,
                                producer_logical=source_a,
                                dst_rank=rank,
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                        )
                    invars.append(val)

            with submesh:
                if loc == terminal_loc:
                    loss, _ = terminal_jit(consts, *invars)
                    loss_acc = loss_acc + loss
                else:
                    out = fwd_jits[loc](consts, *invars)

            saved_inputs[(rank, virt, mb)] = tuple(invars)
            if loc != terminal_loc:
                saved_outputs[(rank, virt, mb)] = out

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    return mean_loss, {"saved_inputs": saved_inputs, "saved_outputs": saved_outputs, "per_loc_consts": per_loc_consts}


def _dispatch_gpipe_bwd(
    plan: dict[str, Any],
    saved: dict[str, Any],
    g: Any,
) -> tuple[Any, ...]:
    """All-backward path for custom_vjp.

    Uses saved activations from ``_dispatch_gpipe_fwd``, walks backward
    stages, and computes gradients for all argnums.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    loc_for_logical = plan["loc_for_logical"]
    invar_sources = plan["invar_sources"]
    bwd_jits = plan["bwd_jits"]
    terminal_jit = plan["terminal_jit"]
    terminal_loc = plan["terminal_loc"]
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = saved.get("per_loc_consts", plan["per_loc_consts"])
    dynamic_mask = plan["dynamic_mask"]
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    flat_args = plan["flat_args"]

    saved_inputs = saved["saved_inputs"]
    saved_outputs = saved["saved_outputs"]

    grad_accums: dict[int, Any] = {}
    recv_cots: dict[tuple[int, int, int], list[Any | None]] = {}

    for mb in range(m):
        for logical in reversed(range(n_logical)):
            loc = loc_for_logical[logical]
            rank, virt = loc
            consts = per_loc_consts[loc]
            submesh = rank_submeshes[rank]
            key = (rank, virt, mb)
            invars = saved_inputs[key]

            with submesh:
                if loc == terminal_loc:
                    _, (g_consts, g_invars) = terminal_jit(consts, *invars)
                    cotangent = jnp.asarray(1.0, dtype=jnp.float32) if g is None else g
                    scale = cotangent / jnp.asarray(m, dtype=jnp.float32)
                    g_consts = jax.tree.map(lambda x, s=scale: _scale_grad(x, s), g_consts, is_leaf=_is_leaf)
                    g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                else:
                    cotangents = _materialize_cotangents(
                        recv_cots.get(key),
                        saved_outputs[key],
                    )
                    g_consts, g_invars = bwd_jits[loc](consts, *invars, *cotangents)

            for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][loc]):
                flat_idx = const_idx_to_flat_idx.get(const_idx)
                if flat_idx is None:
                    continue
                grad = g_consts[local_idx]
                if flat_idx not in grad_accums:
                    grad_accums[flat_idx] = grad
                else:
                    grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)

            for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                if source_kind != "body_invar":
                    continue
                flat_idx = dynamic_flat_to_global_flat.get(source_a)
                if flat_idx is None:
                    continue
                grad = g_invars[invar_idx]
                if dynamic_mask[flat_idx]:
                    if flat_idx not in grad_accums:
                        grad_accums[flat_idx] = [None] * m
                    grad_accums[flat_idx][mb] = grad
                else:
                    if flat_idx not in grad_accums:
                        grad_accums[flat_idx] = grad
                    else:
                        grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)

            if logical > 0:
                for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "cluster_out":
                        continue
                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = loc_for_logical[producer_logical]
                    p_key = (producer_loc[0], producer_loc[1], mb)
                    cot = g_invars[invar_idx]
                    cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                    if producer_loc[0] != rank:
                        cot = jax.device_put(
                            cot,
                            _transfer_target_for_edge(
                                cot,
                                producer_logical=producer_logical,
                                dst_rank=producer_loc[0],
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                        )
                    slots = recv_cots.setdefault(
                        p_key,
                        [None] * len(saved_outputs[p_key]),
                    )
                    if slots[producer_out_idx] is None:
                        slots[producer_out_idx] = cot
                    else:
                        slots[producer_out_idx] = _add_grad(slots[producer_out_idx], cot)

    final_grads: list[Any] = []
    for i in range(n_flat):
        if i in grad_accums:
            grad = grad_accums[i]
            if dynamic_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(jnp.zeros_like(flat_args[i]))

    return tuple(final_grads)


def _dispatch_schedule_faithful_serial(
    plan: dict[str, Any],
    args: tuple,
    return_loss: bool = False,
) -> tuple[jax.Array | None, tuple[Any, ...]]:
    """Schedule-faithful forward+backward grid walker.

    Walks ``plan.grid`` step by step, executes FWD and BWD actions in
    schedule order, accumulates gradients, and returns ``(loss, grads_flat)``.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    grid = plan["grid"]
    loc_for_logical = plan["loc_for_logical"]
    logical_for_loc = plan["logical_for_loc"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    bwd_jits = plan["bwd_jits"]
    terminal_jit = plan["terminal_jit"]
    terminal_loc = plan["terminal_loc"]
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    flat_args = plan["flat_args"]
    lazy_bwd_batching = plan["schedule"].lazy_bwd_batching

    flat_args_live = jax.tree.leaves(args)
    mb_args: list[Any] = []
    for i, arg in enumerate(flat_args_live):
        if dynamic_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    saved_outputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    terminal_grads: dict[tuple[int, int, int], tuple[Any, tuple[Any, ...]]] = {}
    recv_cots: dict[tuple[int, int, int], list[Any | None]] = {}
    grad_accums: dict[int, Any] = {}
    terminal_const_grad_accums: dict[int, Any] = {}
    loss_acc = jnp.asarray(0.0)
    lazy_bwd_actions: dict[tuple[int, int], list[tuple[Any, int]]] = {}

    for row in grid:
        for rank, virt, action in _iter_actions(row):
            mb = action.microbatch
            phase = action.phase
            loc = (rank, virt)
            logical = logical_for_loc[loc]
            submesh = rank_submeshes[rank]
            key = (rank, virt, mb)
            consts = per_loc_consts[loc]

            if phase is Phase.FWD:
                invars: list[Any] = []
                for source_kind, source_a, source_b in invar_sources[logical]:
                    if source_kind == "body_invar":
                        flat_idx = dynamic_flat_to_global_flat[source_a]
                        val = mb_args[flat_idx]
                        if dynamic_mask[flat_idx]:
                            val = val[mb]
                        invars.append(val)
                    elif source_kind == "cluster_out":
                        producer_loc = loc_for_logical[source_a]
                        val = saved_outputs[(producer_loc[0], producer_loc[1], mb)][source_b]
                        if producer_loc[0] != rank:
                            val = _transport(
                                "device_put",
                                val,
                                _transfer_target_for_edge(
                                    val,
                                    producer_logical=source_a,
                                    dst_rank=rank,
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=f"transfer_fwd_stage{source_a}_to_stage{logical}_mb{mb}",
                            )
                        invars.append(val)

                with submesh:
                    if loc == terminal_loc:
                        loss, (g_consts, g_invars) = _time_call(
                            f"stage{logical}_terminal_fwd_mb{mb}",
                            terminal_jit,
                            consts,
                            *invars,
                        )
                        loss_acc = loss_acc + loss
                        terminal_grads[key] = (g_consts, g_invars)
                    else:
                        out = _time_call(f"stage{logical}_fwd_mb{mb}", fwd_jits[loc], consts, *invars)

                saved_inputs[key] = tuple(invars)
                if loc != terminal_loc:
                    saved_outputs[key] = out

            elif phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                if lazy_bwd_batching:
                    lazy_bwd_actions.setdefault(loc, []).append((action, mb))
                    continue

                invars = saved_inputs[key]
                phase_label = phase.name.lower()

                with submesh:
                    if loc == terminal_loc:
                        cached_terminal_grads = terminal_grads.pop(key, None)
                        if cached_terminal_grads is None:
                            _, cached_terminal_grads = _time_call(
                                f"stage{logical}_terminal_{phase_label}_mb{mb}",
                                terminal_jit,
                                consts,
                                *invars,
                            )
                        g_consts, g_invars = cached_terminal_grads
                        scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                        g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                    else:
                        cotangents = _materialize_cotangents(
                            recv_cots.get(key),
                            saved_outputs[key],
                        )
                        g_consts, g_invars = _time_call(
                            f"stage{logical}_{phase_label}_mb{mb}",
                            bwd_jits[loc],
                            consts,
                            *invars,
                            *cotangents,
                        )

                if phase is not Phase.BWD_I:
                    const_accums = terminal_const_grad_accums if loc == terminal_loc else grad_accums
                    for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][loc]):
                        flat_idx = const_idx_to_flat_idx.get(const_idx)
                        if flat_idx is None:
                            continue
                        grad = g_consts[local_idx]
                        if flat_idx not in const_accums:
                            const_accums[flat_idx] = grad
                        else:
                            const_accums[flat_idx] = _add_grad(const_accums[flat_idx], grad)

                if phase is not Phase.BWD_W:
                    for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "body_invar":
                            continue
                        flat_idx = dynamic_flat_to_global_flat.get(source_a)
                        if flat_idx is None:
                            continue
                        grad = g_invars[invar_idx]
                        if dynamic_mask[flat_idx]:
                            if flat_idx not in grad_accums:
                                grad_accums[flat_idx] = [None] * m
                            grad_accums[flat_idx][mb] = grad
                        else:
                            if flat_idx not in grad_accums:
                                grad_accums[flat_idx] = grad
                            else:
                                grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)

                if phase is not Phase.BWD_W:
                    for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "cluster_out":
                            continue
                        producer_logical = source_a
                        producer_out_idx = source_b
                        producer_loc = loc_for_logical[producer_logical]
                        p_key = (producer_loc[0], producer_loc[1], mb)
                        cot = g_invars[invar_idx]
                        cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = _transport(
                                "device_put",
                                cot,
                                _transfer_target_for_edge(
                                    cot,
                                    producer_logical=producer_logical,
                                    dst_rank=producer_loc[0],
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=(f"transfer_{phase_label}_stage{logical}_to_stage{producer_logical}_mb{mb}"),
                            )
                        slots = recv_cots.setdefault(
                            p_key,
                            [None] * len(saved_outputs[p_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = _add_grad(slots[producer_out_idx], cot)

    if lazy_bwd_batching:
        vbwd_jits = plan.get("vbwd_jits", {})
        for logical in reversed(range(n_logical)):
            loc = loc_for_logical[logical]
            rank = loc[0]
            actions = lazy_bwd_actions.get(loc, [])
            if not actions:
                continue
            actions.sort(key=lambda x: x[1])
            mbs = [mb for _, mb in actions]
            submesh = rank_submeshes[rank]
            consts = per_loc_consts[loc]

            if loc == terminal_loc:
                scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                for mb in mbs:
                    key = (rank, loc[1], mb)
                    invars = saved_inputs[key]
                    with submesh:
                        cached_terminal_grads = terminal_grads.pop(key, None)
                        if cached_terminal_grads is None:
                            _, cached_terminal_grads = _time_call(
                                f"stage{logical}_terminal_lazy_bwd_mb{mb}",
                                terminal_jit,
                                consts,
                                *invars,
                            )
                        g_consts, g_invars = cached_terminal_grads
                        g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
                    for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][loc]):
                        flat_idx = const_idx_to_flat_idx.get(const_idx)
                        if flat_idx is None:
                            continue
                        grad = g_consts[local_idx]
                        if flat_idx not in terminal_const_grad_accums:
                            terminal_const_grad_accums[flat_idx] = grad
                        else:
                            terminal_const_grad_accums[flat_idx] = _add_grad(terminal_const_grad_accums[flat_idx], grad)
                    for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "body_invar":
                            continue
                        flat_idx = dynamic_flat_to_global_flat.get(source_a)
                        if flat_idx is None:
                            continue
                        grad = g_invars[invar_idx]
                        if dynamic_mask[flat_idx]:
                            if flat_idx not in grad_accums:
                                grad_accums[flat_idx] = [None] * m
                            grad_accums[flat_idx][mb] = grad
                        else:
                            if flat_idx not in grad_accums:
                                grad_accums[flat_idx] = grad
                            else:
                                grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)
                    for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                        if source_kind != "cluster_out":
                            continue
                        producer_logical = source_a
                        producer_out_idx = source_b
                        producer_loc = loc_for_logical[producer_logical]
                        p_key = (producer_loc[0], producer_loc[1], mb)
                        cot = g_invars[invar_idx]
                        cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = _transport(
                                "device_put",
                                cot,
                                _transfer_target_for_edge(
                                    cot,
                                    producer_logical=producer_logical,
                                    dst_rank=producer_loc[0],
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=f"transfer_lazy_bwd_stage{logical}_to_stage{producer_logical}_mb{mb}",
                            )
                        slots = recv_cots.setdefault(
                            p_key,
                            [None] * len(saved_outputs[p_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = _add_grad(slots[producer_out_idx], cot)
            else:
                invars_stack = []
                for invar_idx in range(len(invar_sources[logical])):
                    stacked = jnp.stack(
                        [saved_inputs[(rank, loc[1], mb)][invar_idx] for mb in mbs],
                        axis=0,
                    )
                    invars_stack.append(stacked)

                n_outs = len(saved_outputs[(rank, loc[1], mbs[0])])
                cots_stack = []
                for out_idx in range(n_outs):
                    cots_mb = []
                    for mb in mbs:
                        key = (rank, loc[1], mb)
                        slots = recv_cots.get(key, [None] * n_outs)
                        out = saved_outputs[key][out_idx]
                        cot = slots[out_idx]
                        if cot is None:
                            cot = jnp.zeros_like(out)
                        elif getattr(cot, "dtype", None) == jax.dtypes.float0:
                            pass
                        elif (
                            hasattr(cot, "astype") and hasattr(out, "dtype") and getattr(cot, "dtype", None) != out.dtype
                        ):
                            cot = cot.astype(out.dtype)
                        cots_mb.append(cot)
                    cots_stack.append(jnp.stack(cots_mb, axis=0))

                with submesh:
                    g_consts, g_invars = _time_call(
                        f"stage{logical}_vbwd_mbs{mbs[0]}_{mbs[-1]}",
                        vbwd_jits[loc],
                        consts,
                        *invars_stack,
                        *cots_stack,
                    )

                for local_idx, const_idx in enumerate(plan["const_indices_per_loc"][loc]):
                    flat_idx = const_idx_to_flat_idx.get(const_idx)
                    if flat_idx is None:
                        continue
                    grad = g_consts[local_idx].sum(axis=0)
                    if flat_idx not in grad_accums:
                        grad_accums[flat_idx] = grad
                    else:
                        grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)

                for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "body_invar":
                        continue
                    flat_idx = dynamic_flat_to_global_flat.get(source_a)
                    if flat_idx is None:
                        continue
                    grad = g_invars[invar_idx]
                    if dynamic_mask[flat_idx]:
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = [None] * m
                        for idx, mb in enumerate(mbs):
                            grad_accums[flat_idx][mb] = grad[idx]
                    else:
                        summed_grad = grad.sum(axis=0)
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = summed_grad
                        else:
                            grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], summed_grad)

                for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "cluster_out":
                        continue
                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = loc_for_logical[producer_logical]
                    for idx, mb in enumerate(mbs):
                        p_key = (producer_loc[0], producer_loc[1], mb)
                        cot = g_invars[invar_idx][idx]
                        cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = _transport(
                                "device_put",
                                cot,
                                _transfer_target_for_edge(
                                    cot,
                                    producer_logical=producer_logical,
                                    dst_rank=producer_loc[0],
                                    edge_shardings=edge_shardings,
                                    stage_shardings=stage_shardings,
                                    rank_submeshes=rank_submeshes,
                                    mpmd_mesh=mpmd_mesh,
                                ),
                                task_name=f"transfer_vbwd_stage{logical}_to_stage{producer_logical}_mb{mb}",
                            )
                        slots = recv_cots.setdefault(
                            p_key,
                            [None] * len(saved_outputs[p_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = _add_grad(slots[producer_out_idx], cot)

    final_grads: list[Any] = []
    terminal_const_scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
    for i in range(n_flat):
        if i in grad_accums or i in terminal_const_grad_accums:
            grad = grad_accums.get(i)
            terminal_grad = terminal_const_grad_accums.get(i)
            if terminal_grad is not None:
                terminal_grad = jax.tree.map(
                    lambda x, s=terminal_const_scale: _scale_grad(x, s),
                    terminal_grad,
                    is_leaf=_is_leaf,
                )
                grad = terminal_grad if grad is None else _add_grad(grad, terminal_grad)
            if dynamic_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(jnp.zeros_like(flat_args[i]))

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    return (mean_loss if return_loss else None), tuple(final_grads)


def _get_schedule_fused_fwd_bwd_jit(
    fwd_jit: Callable[..., Any],
    bwd_jit: Callable[..., Any],
    n_invars: int,
) -> Callable[..., Any]:
    """Return a cached jit that fuses FWD on microbatch A with BWD on microbatch B.

    The 1F1B steady state alternates a forward on a fresh microbatch
    with a backward on an in-flight one. Folding both into a single
    :func:`jax.jit` cuts the per-step Python dispatch count in half and
    lets XLA overlap the two computations on the same rank. The result
    is memoised in :data:`_SCHEDULE_FUSED_FWDBWD_CACHE` so subsequent
    fused units reuse the same compiled program.

    Args:
        fwd_jit: Per-cluster forward jit (from :func:`_make_fwd_jit`).
        bwd_jit: Per-cluster backward jit (from :func:`_make_bwd_jit`).
        n_invars: Number of cluster input variables (used to slice the
            packed ``(fwd_invars, bwd_invars, cotangents)`` tuple).

    Returns:
        Jitted ``(consts, *args) -> (fwd_outs, g_consts, g_bwd_invars)``
        callable.
    """
    key = (id(fwd_jit), id(bwd_jit), n_invars)
    cached = _SCHEDULE_FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fused(consts: tuple[Any, ...], *args: Any) -> tuple[Any, Any, Any]:
        """One-launch fwd(A) + bwd(B) for paired schedule microbatches.

        ``args`` is the concatenation of ``(fwd_invars, bwd_invars,
        cotangents)``, each of length ``n_invars`` (cotangents may be
        a different length depending on the cluster's outputs). The
        function returns ``(fwd_outs, g_consts, g_bwd_invars)`` so the
        caller can route the forward outputs to the next stage and
        accumulate the backward gradients.
        """
        fwd_invars = args[:n_invars]
        bwd_invars = args[n_invars : 2 * n_invars]
        cotangents = args[2 * n_invars :]
        fwd_outs = fwd_jit(consts, *fwd_invars)
        g_consts, g_bwd_invars = bwd_jit(consts, *bwd_invars, *cotangents)
        return fwd_outs, g_consts, g_bwd_invars

    _SCHEDULE_FUSED_FWDBWD_CACHE[key] = fused
    weak_invalidate(fwd_jit, _SCHEDULE_FUSED_FWDBWD_CACHE, key)
    weak_invalidate(bwd_jit, _SCHEDULE_FUSED_FWDBWD_CACHE, key)
    return fused


def _eval_schedule_cluster_fwd(cluster_jaxpr: Any, consts: tuple[Any, ...], *invars: Any) -> tuple[Any, ...]:
    """Evaluate a cluster sub-jaxpr without nesting a pre-compiled stage jit.

    Used by direct-fused dispatch paths that compose multiple cluster
    operations inside a single :func:`jax.jit`; calling the precompiled
    forward jit instead would force a re-trace on every invocation.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        consts: Placed constants for that cluster.
        *invars: Stage input activations.

    Returns:
        Tuple of stage outputs (matching the cluster's outvars).
    """
    return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(consts), *invars))


def _eval_schedule_cluster_bwd(
    cluster_jaxpr: Any,
    n_invars: int,
    consts: tuple[Any, ...],
    *invars_and_cotangents: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """Compute ``(g_consts, g_invars)`` for a cluster inside a fused jit.

    Mirrors :func:`_make_bwd_jit` but calls the cluster jaxpr inline so
    a fused fwd+bwd jit can carry both halves in one HLO graph.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        n_invars: Number of cluster invars (used to slice
            ``invars_and_cotangents``).
        consts: Placed constants for the cluster.
        *invars_and_cotangents: ``(*invars, *cotangents)`` packed as a
            single positional sequence.

    Returns:
        ``(g_consts, g_invars)`` aligned with ``consts`` and the
        cluster's invars respectively.
    """
    invars = invars_and_cotangents[:n_invars]
    cotangents = invars_and_cotangents[n_invars:]

    def pure(c: tuple[Any, ...], *xs: Any) -> tuple[Any, ...]:
        """Pure (consts, invars) -> outs interpreter for ``jax.vjp`` linearization."""
        return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

    _, vjp_fn = jax.vjp(pure, consts, *invars)
    grads = vjp_fn(tuple(cotangents))
    return grads[0], tuple(grads[1:])


def _eval_schedule_cluster_terminal(
    cluster_jaxpr: Any,
    n_invars: int,
    consts: tuple[Any, ...],
    *invars: Any,
) -> tuple[Any, tuple[Any, tuple[Any, ...]]]:
    """Run the terminal cluster's loss + gradient computation in-place.

    Used as a building block for direct-fused jits that compose the
    cluster jaxpr inline rather than calling a pre-compiled stage jit.
    Wraps the cluster's scalar-loss evaluator in
    :func:`jax.value_and_grad` over both ``consts`` and ``invars`` so a
    single trace produces the loss value and the seed cotangents the
    upstream backward sweep needs.

    Args:
        cluster_jaxpr: Terminal cluster sub-jaxpr.
        n_invars: Cluster's invar count.
        consts: Placed cluster constants.
        *invars: Cluster's positional inputs.

    Returns:
        ``(loss, (g_consts, g_invars))`` mirroring
        :func:`_make_terminal_jit`'s output.
    """

    def pure(c: tuple[Any, ...], *xs: Any) -> Any:
        """Pure (consts, invars) -> scalar interpreter for the loss cluster.

        The terminal cluster is required to produce a single scalar
        (the per-microbatch loss). Anything else is a tracing error.
        """
        outs = jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs)
        if len(outs) != 1:
            raise ValueError(
                f"Terminal cluster must produce exactly one scalar output (the per-microbatch loss); got {len(outs)}."
            )
        return outs[0]

    argnums = tuple(range(1 + n_invars))
    loss, grads = jax.value_and_grad(pure, argnums=argnums, allow_int=True)(consts, *invars)
    return loss, (grads[0], tuple(grads[1:]))


def _get_schedule_direct_fused_fwd_bwd_jit(
    cluster_jaxpr: Any,
    n_invars: int,
) -> Callable[..., Any]:
    """Return a cached fused FWD(A)+BWD(B) jit built directly from the cluster jaxpr.

    Variant of :func:`_get_schedule_fused_fwd_bwd_jit` that bypasses
    the per-cluster pre-compiled forward/backward jits and re-evaluates
    the cluster's jaxpr inside one outer :func:`jax.jit`. This lets
    XLA see the entire fwd+bwd as a single graph (better fusion) at
    the cost of recompiling the cluster body inside this jit.

    Cached on ``(id(cluster_jaxpr), n_invars)`` in
    :data:`_SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE`.

    Args:
        cluster_jaxpr: The stage's sub-jaxpr.
        n_invars: Number of cluster input variables.

    Returns:
        Jitted ``(consts, *args) -> (fwd_outs, g_consts, g_bwd_invars)``
        callable.
    """
    key = (id(cluster_jaxpr), n_invars)
    cached = _SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fused(consts: tuple[Any, ...], *args: Any) -> tuple[Any, Any, Any]:
        """Fused fwd(A)+bwd(B) that evaluates the cluster jaxpr directly.

        Bypasses the per-stage compiled fwd/bwd JITs (used elsewhere)
        and instead lets XLA compile the entire fwd+bwd as a single
        function. Profile-driven cache: the result is keyed on
        ``(id(cluster_jaxpr), n_invars)``.
        """
        fwd_invars = args[:n_invars]
        bwd_invars = args[n_invars : 2 * n_invars]
        cotangents = args[2 * n_invars :]
        fwd_outs = _eval_schedule_cluster_fwd(cluster_jaxpr, consts, *fwd_invars)
        g_consts, g_bwd_invars = _eval_schedule_cluster_bwd(
            cluster_jaxpr,
            n_invars,
            consts,
            *bwd_invars,
            *cotangents,
        )
        return fwd_outs, g_consts, g_bwd_invars

    _SCHEDULE_DIRECT_FUSED_FWDBWD_CACHE[key] = fused
    return fused


def _schedule_action_unit(
    *,
    index: int,
    row: int,
    rank: int,
    action: Any,
    logical_for_loc: dict[tuple[int, int], int],
) -> _ScheduleUnit:
    """Wrap a single :class:`Action` cell as a :class:`_ScheduleUnit`.

    Forward actions populate ``fwd_logical``/``fwd_mb``; backward
    actions populate ``bwd_logical``/``bwd_mb``/``bwd_phase``. Used by
    :func:`_build_schedule_units_from_plan` when expanding the
    schedule grid into the dependency DAG.

    Args:
        index: Stable global ordering key.
        row: Source row in the schedule grid.
        rank: Physical pipeline rank.
        action: The :class:`Action` cell.
        logical_for_loc: Mapping from ``(rank, virt)`` to logical
            stage index.

    Returns:
        A :class:`_ScheduleUnit` reflecting the action's phase.
    """
    virt = action.virtual_stage
    logical = logical_for_loc[(rank, virt)]
    if action.phase is Phase.FWD:
        return _ScheduleUnit(
            index=index,
            row=row,
            kind="action",
            rank=rank,
            virt=virt,
            payload=action,
            fwd_logical=logical,
            fwd_mb=action.microbatch,
            bwd_logical=None,
            bwd_mb=None,
            bwd_phase=None,
        )
    return _ScheduleUnit(
        index=index,
        row=row,
        kind="action",
        rank=rank,
        virt=virt,
        payload=action,
        fwd_logical=None,
        fwd_mb=None,
        bwd_logical=logical,
        bwd_mb=action.microbatch,
        bwd_phase=action.phase,
    )


def _schedule_fused_unit(
    *,
    index: int,
    row: int,
    rank: int,
    fused: FusedTask,
    logical_for_loc: dict[tuple[int, int], int],
) -> _ScheduleUnit:
    """Wrap a paired FWD+BWD :class:`FusedTask` as a single :class:`_ScheduleUnit`.

    Both halves share the same physical ``(rank, virt)`` location so
    the unit carries one logical-stage index but two microbatch
    indices (one for the forward half, one for the backward half).

    Args:
        index: Stable global ordering key.
        row: Source row in the schedule grid.
        rank: Physical pipeline rank.
        fused: The :class:`FusedTask` cell.
        logical_for_loc: Mapping from ``(rank, virt)`` to logical
            stage index.

    Returns:
        A :class:`_ScheduleUnit` with ``kind="fused"``.
    """
    virt = fused.virtual_stage
    logical = logical_for_loc[(rank, virt)]
    return _ScheduleUnit(
        index=index,
        row=row,
        kind="fused",
        rank=rank,
        virt=virt,
        payload=fused,
        fwd_logical=logical,
        fwd_mb=fused.fwd.microbatch,
        bwd_logical=logical,
        bwd_mb=fused.bwd.microbatch,
        bwd_phase=fused.bwd.phase,
    )


def _fuse_cross_virtual_schedule_units(plan: dict[str, Any], units: list[_ScheduleUnit]) -> list[_ScheduleUnit]:
    """Return schedule units without hidden env-gated fusion.

    Real schedule cells such as ``FusedTask`` are still preserved by
    ``_build_schedule_units_from_plan``. This helper intentionally does not
    rewrite adjacent rows through a private OS flag; that keeps the MPMD
    runtime's behavior explicit and benchmarkable.
    """
    del plan
    return units


def _build_schedule_units_from_plan(plan: dict[str, Any]) -> list[_ScheduleUnit]:
    """Walk the schedule grid and emit executable units in row-major order.

    Genuine forward+backward :class:`FusedTask` cells stay folded into
    a single fused unit; mixed-phase ``FusedTask`` (e.g. fwd+bwd_i)
    is split into its component actions because the dispatcher
    handles the halves separately. The terminal-rank backward action
    is omitted when ``eager_terminal_bwd`` is set (the terminal
    backward is fired eagerly inside the forward stub).

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.

    Returns:
        Ordered list of :class:`_ScheduleUnit` objects ready for
        dependency analysis.
    """
    units: list[_ScheduleUnit] = []
    next_index = 0
    logical_for_loc = plan["logical_for_loc"]
    terminal_loc = plan["terminal_loc"]
    eager_terminal_bwd = True
    for row_idx, row in enumerate(plan["grid"]):
        for rank, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, FusedTask):
                if cell.fwd.phase is Phase.FWD and cell.bwd.phase is Phase.BWD:
                    units.append(
                        _schedule_fused_unit(
                            index=next_index,
                            row=row_idx,
                            rank=rank,
                            fused=cell,
                            logical_for_loc=logical_for_loc,
                        )
                    )
                    next_index += 1
                else:
                    units.append(
                        _schedule_action_unit(
                            index=next_index,
                            row=row_idx,
                            rank=rank,
                            action=cell.fwd,
                            logical_for_loc=logical_for_loc,
                        )
                    )
                    next_index += 1
                    units.append(
                        _schedule_action_unit(
                            index=next_index,
                            row=row_idx,
                            rank=rank,
                            action=cell.bwd,
                            logical_for_loc=logical_for_loc,
                        )
                    )
                    next_index += 1
            else:
                if eager_terminal_bwd and cell.phase is Phase.BWD and (rank, cell.virtual_stage) == terminal_loc:
                    continue
                units.append(
                    _schedule_action_unit(
                        index=next_index,
                        row=row_idx,
                        rank=rank,
                        action=cell,
                        logical_for_loc=logical_for_loc,
                    )
                )
                next_index += 1
    return _fuse_cross_virtual_schedule_units(plan, units)


def _build_schedule_unit_dependencies(plan: dict[str, Any], units: list[_ScheduleUnit]) -> dict[int, set[int]]:
    """Compute the predecessor-set for each schedule unit.

    Three classes of edge are added:

    * **Same-rank order**: each unit depends on the previous unit fired
      on the same rank (preserves the schedule's intended sequencing).
    * **Forward-data dependencies**: a forward unit depends on the
      forward units that produced each of its cluster inputs (looked
      up via ``invar_sources``).
    * **Backward-cotangent dependencies**: a backward unit depends on
      its own paired forward (so saved activations are available) and
      on the backward of every downstream consumer that supplies a
      cotangent. ``BWD_W`` units are excluded from
      cotangent-supplier tracking because they only produce weight
      grads.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        units: Units returned by :func:`_build_schedule_units_from_plan`.

    Returns:
        A mapping ``unit_index -> {predecessor unit indices}``.
    """
    n_logical = plan["n_logical"]
    invar_sources = plan["invar_sources"]
    fwd_units: dict[tuple[int, int], int] = {}
    bwd_cot_units: dict[tuple[int, int], int] = {}
    consumers_by_producer: dict[int, set[int]] = {logical: set() for logical in range(n_logical)}
    for consumer_logical, sources in enumerate(invar_sources):
        for source_kind, source_a, _source_b in sources:
            if source_kind == "cluster_out":
                consumers_by_producer.setdefault(source_a, set()).add(consumer_logical)

    for unit in units:
        if unit.fwd_logical is not None and unit.fwd_mb is not None:
            fwd_units[(unit.fwd_logical, unit.fwd_mb)] = unit.index
        if unit.bwd_logical is not None and unit.bwd_mb is not None and unit.bwd_phase is not Phase.BWD_W:
            bwd_cot_units[(unit.bwd_logical, unit.bwd_mb)] = unit.index

    terminal_logical = plan["logical_for_loc"][plan["terminal_loc"]]
    for mb in range(plan["m"]):
        fwd_idx = fwd_units.get((terminal_logical, mb))
        if fwd_idx is not None:
            bwd_cot_units[(terminal_logical, mb)] = fwd_idx

    deps: dict[int, set[int]] = {unit.index: set() for unit in units}
    previous_by_rank: dict[int, int] = {}

    def add_dep(unit: _ScheduleUnit, dep: int | None) -> None:
        """Insert ``dep`` into ``unit``'s predecessor set if it is real and distinct.

        Skips ``None`` (no dependency) and self-references (a unit
        cannot depend on itself). The caller may pass a missing
        index from a dictionary lookup directly without an extra
        ``if`` check.
        """
        if dep is not None and dep != unit.index:
            deps[unit.index].add(dep)

    for unit in units:
        add_dep(unit, previous_by_rank.get(unit.rank))
        previous_by_rank[unit.rank] = unit.index

        if unit.fwd_logical is not None and unit.fwd_mb is not None:
            for source_kind, source_a, _source_b in invar_sources[unit.fwd_logical]:
                if source_kind == "cluster_out":
                    add_dep(unit, fwd_units.get((source_a, unit.fwd_mb)))

        if unit.bwd_logical is not None and unit.bwd_mb is not None:
            add_dep(unit, fwd_units.get((unit.bwd_logical, unit.bwd_mb)))
            for consumer_logical in consumers_by_producer.get(unit.bwd_logical, ()):
                add_dep(unit, bwd_cot_units.get((consumer_logical, unit.bwd_mb)))

    return deps


def _dispatch_schedule_faithful(
    plan: dict[str, Any],
    args: tuple,
    return_loss: bool = False,
) -> tuple[jax.Array | None, tuple[Any, ...]]:
    """Run the schedule-driven training dispatch and return ``(loss, grads)``.

    The default path lowers the schedule into dependency-tracked units
    and fires them through :func:`_dispatch_schedule_fused_async`.
    Schedules that opt into ``lazy_bwd_batching`` (currently the
    research-style serial path) instead delegate to
    :func:`_dispatch_schedule_faithful_serial` and bypass the async
    DAG entirely. The choice is recorded in
    ``plan["last_schedule_runtime_stats"]`` for diagnostics.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        args: Flat positional call arguments.
        return_loss: When ``True``, also return the scalar loss
            (``False`` is used by ``sxgrad`` which wants only grads).

    Returns:
        ``(loss_or_None, flat_grads_tuple)``.
    """
    if getattr(plan["schedule"], "lazy_bwd_batching", False):
        plan["last_schedule_runtime_stats"] = {
            "dispatcher": "serial",
            "unit_count": None,
            "window_count": None,
            "fallback_reason": "lazy_bwd_batching",
        }
        return _dispatch_schedule_faithful_serial(plan, args, return_loss=return_loss)
    units = _build_schedule_units_from_plan(plan)
    deps = _build_schedule_unit_dependencies(plan, units)
    return _dispatch_schedule_fused_async(plan, args, return_loss=return_loss, units=units, deps=deps)


def _dispatch_schedule_fused_async(
    plan: dict[str, Any],
    args: tuple,
    return_loss: bool = False,
    *,
    units: list[_ScheduleUnit] | None = None,
    deps: dict[int, set[int]] | None = None,
) -> tuple[jax.Array | None, tuple[Any, ...]]:
    """Run a schedule grid using real fused FWD+BWD units where possible.

    The schedule grid is lowered into dependency-tracked units. Same-rank
    order is preserved, cross-rank units dispatch as soon as their saved
    activations/cotangents are ready, and fusable non-terminal FWD+BWD cells
    run as one compiled stage function.
    """
    m = plan["m"]
    n_logical = plan["n_logical"]
    grid = plan["grid"]
    loc_for_logical = plan["loc_for_logical"]
    logical_for_loc = plan["logical_for_loc"]
    invar_sources = plan["invar_sources"]
    fwd_jits = plan["fwd_jits"]
    bwd_jits = plan["bwd_jits"]
    bwd_i_jits = plan.get("bwd_i_jits", {})
    bwd_w_jits = plan.get("bwd_w_jits", {})
    terminal_jit = plan["terminal_jit"]
    terminal_loc = plan["terminal_loc"]
    rank_submeshes = plan["rank_submeshes"]
    stage_shardings = plan["stage_shardings"]
    edge_shardings = plan.get("edge_shardings", ())
    mpmd_mesh = plan["mpmd_mesh"]
    per_loc_consts = _schedule_per_call_consts(plan, args)
    dynamic_mask = plan["dynamic_mask"]
    const_idx_to_flat_idx = plan["const_idx_to_flat_idx"]
    dynamic_flat_to_global_flat = plan["dynamic_flat_to_global_flat"]
    n_flat = plan["n_flat"]
    flat_args = plan["flat_args"]
    const_indices_per_loc = plan["const_indices_per_loc"]
    n_invars_per_loc = plan["n_invars_per_loc"]
    cluster_jaxprs_per_loc = plan.get("cluster_jaxprs_per_loc", {})
    cache_terminal_grads = True
    eager_terminal_bwd = True

    flat_args_live = jax.tree.leaves(args)
    mb_args: list[Any] = []
    for i, arg in enumerate(flat_args_live):
        if dynamic_mask[i]:
            mb_args.append(_microbatch(arg, m))
        else:
            mb_args.append(arg)

    saved_inputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    saved_outputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    pretransferred_outputs: dict[tuple[int, tuple[int, int, int]], tuple[Any, ...] | concurrent.futures.Future[Any]] = {}
    terminal_grads: dict[tuple[int, int, int], tuple[Any, tuple[Any, ...]]] = {}
    recv_cots: dict[tuple[int, int, int], list[Any | None]] = {}
    grad_accums: dict[int, Any] = {}
    const_tuple_accums: dict[tuple[int, int], Any] = {}
    terminal_const_tuple_accums: dict[tuple[int, int], Any] = {}
    loss_acc = jnp.asarray(0.0)
    state_lock = threading.Lock()
    stats_collector: _ScheduleStatsCollector | None = None
    transfer_executor: concurrent.futures.ThreadPoolExecutor | None = None
    consumers_by_producer: dict[int, set[int]] = {logical: set() for logical in range(n_logical)}
    for consumer_logical, sources in enumerate(invar_sources):
        for source_kind, source_a, _source_b in sources:
            if source_kind == "cluster_out":
                consumers_by_producer.setdefault(source_a, set()).add(consumer_logical)

    def _stage_call(rank: int, task_name: str, fn: Callable[..., Any], *call_args: Any) -> Any:
        """Time-instrumented per-stage launch helper.

        Wraps the per-stage function call with profiler timing and
        per-rank launch accounting on ``stats_collector``. The timing
        captures host enqueue duration, not device execution.
        """
        t0 = time.perf_counter_ns()
        out = _time_call(task_name, fn, *call_args)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        if stats_collector is not None:
            stats_collector.record_launch(rank, elapsed_ms)
        return out

    def _collect_fwd_invars(logical: int, rank: int, mb: int) -> list[Any]:
        """Gather forward-pass input arrays for one (logical stage, rank, microbatch).

        Walks ``invar_sources[logical]`` to either pull a microbatch
        slice from ``mb_args`` (``body_invar``) or fetch a saved
        producer output from another stage (``cluster_out``). Cross-
        rank activations are pulled via ``device_put`` (or a
        previously-prefetched future if one is available), with
        bytes-moved accounting reported into ``stats_collector``.
        """
        invars: list[Any] = []
        for source_kind, source_a, source_b in invar_sources[logical]:
            if source_kind == "body_invar":
                flat_idx = dynamic_flat_to_global_flat[source_a]
                val = mb_args[flat_idx]
                if dynamic_mask[flat_idx]:
                    val = val[mb]
                invars.append(val)
            elif source_kind == "cluster_out":
                producer_loc = loc_for_logical[source_a]
                producer_key = (producer_loc[0], producer_loc[1], mb)
                with state_lock:
                    pretransferred = pretransferred_outputs.get((rank, producer_key))
                    producer_outputs = saved_outputs[producer_key]
                if isinstance(pretransferred, concurrent.futures.Future):
                    pretransferred = pretransferred.result()
                val = (pretransferred if pretransferred is not None else producer_outputs)[source_b]
                if producer_loc[0] != rank:
                    if pretransferred is None:
                        val = _transport(
                            "device_put",
                            val,
                            _transfer_target_for_edge(
                                val,
                                producer_logical=source_a,
                                dst_rank=rank,
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                            task_name=f"transfer_fwd_stage{source_a}_to_stage{logical}_mb{mb}",
                            stats=stats_collector,
                            src_rank=producer_loc[0],
                            dst_rank=rank,
                        )
                invars.append(val)
        return invars

    def _pretransfer_fwd_outputs(logical: int, rank: int, virt: int, mb: int, outputs: tuple[Any, ...]) -> None:
        """Start cross-rank activation movement as soon as producer FWD returns."""
        key = (rank, virt, mb)
        for consumer_logical in consumers_by_producer.get(logical, ()):
            consumer_loc = loc_for_logical[consumer_logical]
            dst_rank = consumer_loc[0]
            if dst_rank == rank:
                continue

            def do_transfer(dst: int = dst_rank, consumer: int = consumer_logical) -> tuple[Any, ...]:
                """Run one cross-rank ``device_put`` for the producer's forward outputs."""
                return _transport(
                    "device_put",
                    outputs,
                    _transfer_target_for_edge(
                        outputs,
                        producer_logical=logical,
                        dst_rank=dst,
                        edge_shardings=edge_shardings,
                        stage_shardings=stage_shardings,
                        rank_submeshes=rank_submeshes,
                        mpmd_mesh=mpmd_mesh,
                    ),
                    task_name=f"transfer_fwd_stage{logical}_to_stage{consumer}_mb{mb}",
                    stats=stats_collector,
                    src_rank=rank,
                    dst_rank=dst,
                )

            if transfer_executor is not None:
                moved = transfer_executor.submit(do_transfer)
            else:
                moved = do_transfer()
            with state_lock:
                pretransferred_outputs[(dst_rank, key)] = moved

    def _accumulate_bwd_result(
        *,
        loc: tuple[int, int],
        logical: int,
        rank: int,
        mb: int,
        phase: Phase,
        g_consts: Any,
        g_invars: tuple[Any, ...],
        const_grad_accums: dict[tuple[int, int], Any] | None = None,
        consts_already_accumulated: bool = False,
    ) -> None:
        """Fold one backward unit's gradients into accumulators / cotangent buffers.

        Routes the per-invar cotangents to their producer stage's
        cotangent buffer (with cross-rank ``device_put`` if needed)
        and updates the per-(rank, virt) gradient accumulators for
        consts. Honors :class:`Phase.BWD_W` (weight-grad-only, no
        invar cotangents) and the ``consts_already_accumulated``
        flag (used when fused units already added const grads
        themselves).
        """
        phase_label = phase.name.lower()
        const_accums = const_tuple_accums if const_grad_accums is None else const_grad_accums
        cot_updates: list[tuple[tuple[int, int, int], int, Any]] = []
        if phase is not Phase.BWD_W:
            for invar_idx, (source_kind, source_a, source_b) in enumerate(invar_sources[logical]):
                if source_kind != "cluster_out":
                    continue
                producer_logical = source_a
                producer_out_idx = source_b
                producer_loc = loc_for_logical[producer_logical]
                p_key = (producer_loc[0], producer_loc[1], mb)
                cot = g_invars[invar_idx]
                cot = _cast_cotangent_like(cot, saved_outputs[p_key][producer_out_idx])
                if producer_loc[0] != rank:

                    def do_transfer(
                        value: Any = cot,
                        dst_rank: int = producer_loc[0],
                        src_rank: int = rank,
                        src_logical: int = logical,
                        dst_logical: int = producer_logical,
                    ) -> Any:
                        """Push one backward cotangent across ranks for the producer to consume."""
                        return _transport(
                            "device_put",
                            value,
                            _transfer_target_for_edge(
                                value,
                                producer_logical=dst_logical,
                                dst_rank=dst_rank,
                                edge_shardings=edge_shardings,
                                stage_shardings=stage_shardings,
                                rank_submeshes=rank_submeshes,
                                mpmd_mesh=mpmd_mesh,
                            ),
                            task_name=f"transfer_{phase_label}_stage{src_logical}_to_stage{dst_logical}_mb{mb}",
                            stats=stats_collector,
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                        )

                    if transfer_executor is not None:
                        cot = transfer_executor.submit(do_transfer)
                    else:
                        cot = do_transfer()
                cot_updates.append((p_key, producer_out_idx, cot))

        with state_lock:
            if phase is not Phase.BWD_I and g_consts is not None:
                if consts_already_accumulated:
                    const_accums[loc] = g_consts
                elif loc not in const_accums:
                    const_accums[loc] = g_consts
                else:
                    const_accums[loc] = _accumulate_grad_tree(const_accums[loc], g_consts)

            if phase is not Phase.BWD_W:
                for invar_idx, (source_kind, source_a, _source_b) in enumerate(invar_sources[logical]):
                    if source_kind != "body_invar":
                        continue
                    flat_idx = dynamic_flat_to_global_flat.get(source_a)
                    if flat_idx is None:
                        continue
                    grad = g_invars[invar_idx]
                    if dynamic_mask[flat_idx]:
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = [None] * m
                        grad_accums[flat_idx][mb] = grad
                    else:
                        if flat_idx not in grad_accums:
                            grad_accums[flat_idx] = grad
                        else:
                            grad_accums[flat_idx] = _add_grad(grad_accums[flat_idx], grad)

            for p_key, producer_out_idx, cot in cot_updates:
                slots = recv_cots.setdefault(
                    p_key,
                    [None] * len(saved_outputs[p_key]),
                )
                if slots[producer_out_idx] is None:
                    slots[producer_out_idx] = cot
                else:
                    existing = slots[producer_out_idx]
                    existing_result = getattr(existing, "result", None)
                    cot_result = getattr(cot, "result", None)
                    if callable(existing_result):
                        existing = existing_result()
                    if callable(cot_result):
                        cot = cot_result()
                    slots[producer_out_idx] = _add_grad(existing, cot)

    def _run_fwd(rank: int, virt: int, action: Any) -> None:
        """Execute one forward action for the given (rank, virt) location.

        Two paths: terminal (loss) stages compute loss-and-grads in
        one call (when ``cache_terminal_grads`` is on) and optionally
        accumulate the backward gradients eagerly; non-terminal
        stages execute the forward jit, save inputs+outputs for the
        backward pass, and prefetch outputs to downstream consumers.
        """
        nonlocal loss_acc
        mb = action.microbatch
        loc = (rank, virt)
        logical = logical_for_loc[loc]
        key = (rank, virt, mb)
        consts = per_loc_consts[loc]
        invars = _collect_fwd_invars(logical, rank, mb)
        with rank_submeshes[rank]:
            if loc == terminal_loc:
                if cache_terminal_grads:
                    loss, (g_consts, g_invars) = _stage_call(
                        rank,
                        f"stage{logical}_terminal_fwd_mb{mb}",
                        terminal_jit,
                        consts,
                        *invars,
                    )
                else:
                    loss_out = _stage_call(
                        rank,
                        f"stage{logical}_terminal_loss_mb{mb}",
                        fwd_jits[loc],
                        consts,
                        *invars,
                    )
                    loss = loss_out[0]
                with state_lock:
                    loss_acc = loss_acc + loss
                    if cache_terminal_grads and not eager_terminal_bwd:
                        terminal_grads[key] = (g_consts, g_invars)
                    saved_inputs[key] = tuple(invars)
                if eager_terminal_bwd:
                    scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                    _accumulate_bwd_result(
                        loc=loc,
                        logical=logical,
                        rank=rank,
                        mb=mb,
                        phase=Phase.BWD,
                        g_consts=g_consts,
                        g_invars=tuple(_scale_grad(x, scale) for x in g_invars),
                        const_grad_accums=terminal_const_tuple_accums,
                    )
            else:
                out = _stage_call(rank, f"stage{logical}_fwd_mb{mb}", fwd_jits[loc], consts, *invars)
                with state_lock:
                    saved_inputs[key] = tuple(invars)
                    saved_outputs[key] = out
                _pretransfer_fwd_outputs(logical, rank, virt, mb, out)

    def _run_bwd(rank: int, virt: int, action: Any) -> None:
        """Execute one backward action (full BWD, BWD_I, or BWD_W).

        Picks the appropriate compiled jit based on the action's
        :class:`Phase` and the availability of split bwd_i/bwd_w
        jits, then routes results through :func:`_accumulate_bwd_result`.
        Terminal-stage backward is a no-op when ``eager_terminal_bwd``
        is set (its grads were already accumulated in :func:`_run_fwd`).
        """
        mb = action.microbatch
        phase = action.phase
        loc = (rank, virt)
        logical = logical_for_loc[loc]
        key = (rank, virt, mb)
        consts = per_loc_consts[loc]
        invars = saved_inputs[key]
        phase_label = phase.name.lower()
        consts_already_accumulated = False

        with rank_submeshes[rank]:
            if loc == terminal_loc:
                if eager_terminal_bwd:
                    return
                cached_terminal_grads = terminal_grads.pop(key, None)
                if cached_terminal_grads is None:
                    _, cached_terminal_grads = _stage_call(
                        rank,
                        f"stage{logical}_terminal_{phase_label}_mb{mb}",
                        terminal_jit,
                        consts,
                        *invars,
                    )
                g_consts, g_invars = cached_terminal_grads
                scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
                g_invars = tuple(_scale_grad(x, scale) for x in g_invars)
            else:
                cotangents = _materialize_cotangents(
                    recv_cots.get(key),
                    saved_outputs[key],
                )
                if phase is Phase.BWD_I and bwd_i_jits.get(loc) is not None:
                    g_consts = None
                    g_invars = _stage_call(
                        rank,
                        f"stage{logical}_{phase_label}_mb{mb}",
                        bwd_i_jits[loc],
                        consts,
                        *invars,
                        *cotangents,
                    )
                elif phase is Phase.BWD_W and bwd_w_jits.get(loc) is not None:
                    g_consts = _stage_call(
                        rank,
                        f"stage{logical}_{phase_label}_mb{mb}",
                        bwd_w_jits[loc],
                        consts,
                        *invars,
                        *cotangents,
                    )
                    g_invars = ()
                else:
                    g_consts, g_invars = _stage_call(
                        rank,
                        f"stage{logical}_{phase_label}_mb{mb}",
                        bwd_jits[loc],
                        consts,
                        *invars,
                        *cotangents,
                    )

        _accumulate_bwd_result(
            loc=loc,
            logical=logical,
            rank=rank,
            mb=mb,
            phase=phase,
            g_consts=g_consts,
            g_invars=g_invars,
            const_grad_accums=terminal_const_tuple_accums if loc == terminal_loc else None,
            consts_already_accumulated=consts_already_accumulated,
        )

    def _run_fused(rank: int, virt: int, fused: FusedTask) -> None:
        """Execute a paired (fwd_A + bwd_B) action as a single compiled stage.

        Used in 1F1B-style schedules where a forward microbatch and
        a backward microbatch fire in the same slot. Falls back to
        sequential :func:`_run_fwd` + :func:`_run_bwd` when fusion
        isn't applicable (e.g. terminal stage, weight-grad-only
        backward, non-FWD forward).
        """
        fwd_action = fused.fwd
        bwd_action = fused.bwd
        fwd_mb = fwd_action.microbatch
        bwd_mb = bwd_action.microbatch
        loc = (rank, virt)
        logical = logical_for_loc[loc]
        if loc == terminal_loc or bwd_action.phase is not Phase.BWD or fwd_action.phase is not Phase.FWD:
            _run_fwd(rank, virt, fwd_action)
            _run_bwd(rank, virt, bwd_action)
            return

        consts = per_loc_consts[loc]
        fwd_key = (rank, virt, fwd_mb)
        bwd_key = (rank, virt, bwd_mb)
        fwd_invars = _collect_fwd_invars(logical, rank, fwd_mb)
        bwd_invars = saved_inputs[bwd_key]
        cotangents = _materialize_cotangents(
            recv_cots.get(bwd_key),
            saved_outputs[bwd_key],
        )
        cluster_jaxpr = cluster_jaxprs_per_loc.get(loc)
        if cluster_jaxpr is None:
            fused_jit = _get_schedule_fused_fwd_bwd_jit(
                fwd_jits[loc],
                bwd_jits[loc],
                n_invars_per_loc[loc],
            )
        else:
            fused_jit = _get_schedule_direct_fused_fwd_bwd_jit(cluster_jaxpr, n_invars_per_loc[loc])
        with rank_submeshes[rank]:
            fwd_outs, g_consts, g_bwd_invars = _stage_call(
                rank,
                f"stage{logical}_fused_fwd{fwd_mb}_bwd{bwd_mb}",
                fused_jit,
                consts,
                *fwd_invars,
                *bwd_invars,
                *cotangents,
            )

        with state_lock:
            saved_inputs[fwd_key] = tuple(fwd_invars)
            saved_outputs[fwd_key] = fwd_outs
        _pretransfer_fwd_outputs(logical, rank, virt, fwd_mb, fwd_outs)
        _accumulate_bwd_result(
            loc=loc,
            logical=logical,
            rank=rank,
            mb=bwd_mb,
            phase=Phase.BWD,
            g_consts=g_consts,
            g_invars=g_bwd_invars,
        )

    def _action_unit(index: int, row: int, rank: int, action: Any) -> _ScheduleUnit:
        """Wrap a plain (non-fused) schedule action in a :class:`_ScheduleUnit`.

        Splits FWD vs BWD/BWD_I/BWD_W into the right unit fields so
        the dependency builder and stats collector can tell them
        apart without re-inspecting the raw action.
        """
        virt = action.virtual_stage
        logical = logical_for_loc[(rank, virt)]
        if action.phase is Phase.FWD:
            return _ScheduleUnit(
                index=index,
                row=row,
                kind="action",
                rank=rank,
                virt=virt,
                payload=action,
                fwd_logical=logical,
                fwd_mb=action.microbatch,
                bwd_logical=None,
                bwd_mb=None,
                bwd_phase=None,
            )
        return _ScheduleUnit(
            index=index,
            row=row,
            kind="action",
            rank=rank,
            virt=virt,
            payload=action,
            fwd_logical=None,
            fwd_mb=None,
            bwd_logical=logical,
            bwd_mb=action.microbatch,
            bwd_phase=action.phase,
        )

    def _fused_unit(index: int, row: int, rank: int, fused: FusedTask) -> _ScheduleUnit:
        """Wrap a :class:`FusedTask` (paired fwd+bwd) as a :class:`_ScheduleUnit`."""
        virt = fused.virtual_stage
        logical = logical_for_loc[(rank, virt)]
        return _ScheduleUnit(
            index=index,
            row=row,
            kind="fused",
            rank=rank,
            virt=virt,
            payload=fused,
            fwd_logical=logical,
            fwd_mb=fused.fwd.microbatch,
            bwd_logical=logical,
            bwd_mb=fused.bwd.microbatch,
            bwd_phase=fused.bwd.phase,
        )

    def _build_schedule_units() -> list[_ScheduleUnit]:
        """Convert the schedule grid into a flat list of dispatchable units.

        Walks the per-row, per-rank cells, splits ``FusedTask``\\s
        that the runtime can't actually fuse (e.g. terminal stage,
        non-FWD/BWD phase combinations) into separate units, and
        keeps track of monotonically increasing unit indices.
        """
        units: list[_ScheduleUnit] = []
        next_index = 0
        for row_idx, row in enumerate(grid):
            for rank, cell in enumerate(row):
                if cell is None:
                    continue
                if isinstance(cell, FusedTask):
                    if cell.fwd.phase is Phase.FWD and cell.bwd.phase is Phase.BWD:
                        units.append(_fused_unit(next_index, row_idx, rank, cell))
                        next_index += 1
                    else:
                        units.append(_action_unit(next_index, row_idx, rank, cell.fwd))
                        next_index += 1
                        units.append(_action_unit(next_index, row_idx, rank, cell.bwd))
                        next_index += 1
                else:
                    units.append(_action_unit(next_index, row_idx, rank, cell))
                    next_index += 1
        return units

    def _run_unit(unit: _ScheduleUnit) -> None:
        """Dispatch one unit through the right runner and record its enqueue time."""
        t0 = time.perf_counter_ns()
        try:
            if unit.kind == "fused":
                _run_fused(unit.rank, unit.virt, unit.payload)
            elif unit.payload.phase is Phase.FWD:
                _run_fwd(unit.rank, unit.virt, unit.payload)
            else:
                _run_bwd(unit.rank, unit.virt, unit.payload)
        finally:
            if stats_collector is not None:
                stats_collector.record_unit(unit.index, unit.rank, (time.perf_counter_ns() - t0) / 1e6)

    def _build_unit_dependencies(units: list[_ScheduleUnit]) -> dict[int, set[int]]:
        """Compute the predecessor set for each unit on the dependency DAG.

        Three classes of edge are added: (1) within-rank FIFO ordering
        (each unit must run after the previous unit on the same rank);
        (2) producer/consumer dependencies between forward outputs
        and forward inputs in another stage; and (3) backward-cotangent
        dependencies between a unit's backward and any *downstream*
        consumer's backward (for the same microbatch).
        """
        fwd_units: dict[tuple[int, int], int] = {}
        bwd_cot_units: dict[tuple[int, int], int] = {}
        consumers_by_producer: dict[int, set[int]] = {logical: set() for logical in range(n_logical)}
        for consumer_logical, sources in enumerate(invar_sources):
            for source_kind, source_a, _source_b in sources:
                if source_kind == "cluster_out":
                    consumers_by_producer.setdefault(source_a, set()).add(consumer_logical)

        for unit in units:
            if unit.fwd_logical is not None and unit.fwd_mb is not None:
                fwd_units[(unit.fwd_logical, unit.fwd_mb)] = unit.index
            if unit.bwd_logical is not None and unit.bwd_mb is not None and unit.bwd_phase is not Phase.BWD_W:
                bwd_cot_units[(unit.bwd_logical, unit.bwd_mb)] = unit.index

        deps: dict[int, set[int]] = {unit.index: set() for unit in units}
        previous_by_rank: dict[int, int] = {}

        def add_dep(unit: _ScheduleUnit, dep: int | None) -> None:
            """Add ``dep`` as a predecessor of ``unit``, ignoring null/self-deps."""
            if dep is not None and dep != unit.index:
                deps[unit.index].add(dep)

        for unit in units:
            add_dep(unit, previous_by_rank.get(unit.rank))
            previous_by_rank[unit.rank] = unit.index

            if unit.fwd_logical is not None and unit.fwd_mb is not None:
                for source_kind, source_a, _source_b in invar_sources[unit.fwd_logical]:
                    if source_kind == "cluster_out":
                        add_dep(unit, fwd_units.get((source_a, unit.fwd_mb)))

            if unit.bwd_logical is not None and unit.bwd_mb is not None:
                add_dep(unit, fwd_units.get((unit.bwd_logical, unit.bwd_mb)))
                for consumer_logical in consumers_by_producer.get(unit.bwd_logical, ()):
                    add_dep(unit, bwd_cot_units.get((consumer_logical, unit.bwd_mb)))

        return deps

    def _run_units_dependency_async(units: list[_ScheduleUnit], deps: dict[int, set[int]]) -> None:
        """Drive the unit DAG asynchronously across two thread-pool executors.

        One executor runs the actual stage compute (one in-flight unit
        per rank at a time, to avoid serializing stage-local kernels);
        the other is reserved for cross-rank ``device_put`` transfers.
        Ready units are dispatched in row-major order; on each
        completion, dependents whose predecessors are now satisfied
        become ready. Detects dependency cycles by checking that
        ``ready`` is non-empty whenever there are no outstanding
        futures.
        """
        nonlocal transfer_executor
        by_index = {unit.index: unit for unit in units}
        dependents: dict[int, set[int]] = {unit.index: set() for unit in units}
        remaining = {idx: set(unit_deps) for idx, unit_deps in deps.items()}
        for idx, unit_deps in deps.items():
            for dep in unit_deps:
                dependents.setdefault(dep, set()).add(idx)

        ready = sorted(
            (idx for idx, unit_deps in remaining.items() if not unit_deps), key=lambda i: (by_index[i].row, i)
        )
        active_by_rank: dict[int, concurrent.futures.Future[Any]] = {}
        future_to_index: dict[concurrent.futures.Future[Any], int] = {}

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rank_submeshes))) as executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rank_submeshes))) as tx_executor,
        ):
            transfer_executor = tx_executor
            try:
                while ready or future_to_index:
                    launched = False
                    pos = 0
                    while pos < len(ready):
                        idx = ready[pos]
                        unit = by_index[idx]
                        if unit.rank in active_by_rank:
                            pos += 1
                            continue
                        ready.pop(pos)
                        future = executor.submit(_run_unit, unit)
                        active_by_rank[unit.rank] = future
                        future_to_index[future] = idx
                        launched = True

                    if not future_to_index:
                        blocked = {idx: sorted(unit_deps) for idx, unit_deps in remaining.items() if unit_deps}
                        raise RuntimeError(f"schedule executor dependency cycle or missing dependency: {blocked}")

                    done, _pending = concurrent.futures.wait(
                        tuple(future_to_index),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        idx = future_to_index.pop(future)
                        unit = by_index[idx]
                        active_by_rank.pop(unit.rank, None)
                        future.result()
                        for dependent in dependents.get(idx, ()):
                            remaining[dependent].discard(idx)
                            if not remaining[dependent]:
                                ready.append(dependent)
                        if launched or done:
                            ready.sort(key=lambda i: (by_index[i].row, i))
            finally:
                transfer_executor = None

    use_async = _active_profiler() is None
    if units is None:
        units = _build_schedule_units()
    if deps is None:
        deps = _build_unit_dependencies(units)
    action_count = sum(2 if unit.kind == "fused" else 1 for unit in units)
    fused_count = sum(1 for unit in units if unit.kind == "fused")
    stats_collector = _ScheduleStatsCollector(
        dispatcher="fused_async" if use_async else "fused_serial_units",
        unit_count=len(units),
        action_count=action_count,
        fused_count=fused_count,
        window_count=None,
        fallback_reason=plan.get("last_schedule_runtime_stats", {}).get("fallback_reason"),
    )
    if use_async:
        _run_units_dependency_async(units, deps)
    else:
        for unit in units:
            _run_unit(unit)

    final_grads: list[Any] = []
    terminal_const_scale = 1.0 / jnp.asarray(m, dtype=jnp.float32)
    for loc, g_consts in const_tuple_accums.items():
        for local_idx, const_idx in enumerate(const_indices_per_loc[loc]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            grad = g_consts[local_idx]
            if flat_idx not in grad_accums:
                grad_accums[flat_idx] = grad
            else:
                grad_accums[flat_idx] = _accumulate_grad_tree(grad_accums[flat_idx], grad)
    for loc, g_consts in terminal_const_tuple_accums.items():
        scaled_consts = _scale_grad_tree(g_consts, terminal_const_scale)
        for local_idx, const_idx in enumerate(const_indices_per_loc[loc]):
            flat_idx = const_idx_to_flat_idx.get(const_idx)
            if flat_idx is None:
                continue
            grad = scaled_consts[local_idx]
            if flat_idx not in grad_accums:
                grad_accums[flat_idx] = grad
            else:
                grad_accums[flat_idx] = _accumulate_grad_tree(grad_accums[flat_idx], grad)
    for i in range(n_flat):
        if i in grad_accums:
            grad = grad_accums.get(i)
            if dynamic_mask[i]:
                if isinstance(grad, list):
                    template = next(g for g in grad if g is not None)
                    for mb in range(m):
                        if grad[mb] is None:
                            grad[mb] = jnp.zeros_like(template)
                    final_grads.append(jnp.concatenate(grad, axis=0))
                else:
                    final_grads.append(grad)
            else:
                final_grads.append(grad)
        else:
            final_grads.append(jnp.zeros_like(flat_args[i]))

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    plan["last_schedule_runtime_stats"] = stats_collector.as_dict(deps, units)
    return (mean_loss if return_loss else None), tuple(final_grads)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _schedule_forward(plan: dict[str, Any], *args: Any) -> jax.Array:
    """Forward-only entry point used to wire ``sxjit`` into JAX autodiff.

    Wrapped with :func:`jax.custom_vjp` so that gradients of
    schedule-driven functions are taken by replaying the same MPMD
    pipeline rather than by retracing the cluster jaxprs through JAX's
    standard transpose. The plan is non-differentiable
    (``nondiff_argnums=(0,)``) so changes to the plan do not propagate
    cotangents.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        *args: Flattened user arguments.

    Returns:
        The forward-pass scalar loss as a :class:`jax.Array`.
    """
    loss, _ = _dispatch_gpipe_fwd(plan, args)
    return loss


def _schedule_forward_fwd(plan: dict[str, Any], *args: Any) -> tuple[jax.Array, dict[str, Any]]:
    """Custom-VJP forward rule: returns ``(loss, residuals)``.

    The residuals dict captures every per-microbatch activation needed
    by :func:`_schedule_forward_bwd` so the backward pass can run the
    pipeline cotangent sweep without re-tracing.

    Args:
        plan: Dispatch plan from :func:`_build_schedule_plan`.
        *args: Flattened user arguments.

    Returns:
        ``(loss, saved)`` — the forward output plus the residuals
        consumed by the backward rule.
    """
    loss, saved = _dispatch_gpipe_fwd(plan, args)
    return loss, saved


def _schedule_forward_bwd(plan: dict[str, Any], saved: dict[str, Any], g: Any) -> tuple[Any, ...]:
    """Custom-VJP backward rule: replay the schedule and return arg cotangents.

    Args:
        plan: Dispatch plan (non-differentiable).
        saved: Residuals from :func:`_schedule_forward_fwd`.
        g: Cotangent of the scalar loss output (typically ``1.0``).

    Returns:
        Tuple of cotangents aligned with the differentiable
        positional arguments of :func:`_schedule_forward`.
    """
    grads = _dispatch_gpipe_bwd(plan, saved, g)
    return grads


_schedule_forward.defvjp(_schedule_forward_fwd, _schedule_forward_bwd)


def sxgrad(fn: Callable, argnums: int | tuple[int, ...] = 0) -> Callable:
    """Schedule-faithful gradient of a schedule-driven ``sxjit`` function.

    Args:
        fn: A function decorated with ``@sxjit(..., schedule=...)``.
        argnums: Positional argument indices to differentiate w.r.t.

    Returns:
        A callable with the same signature as ``fn`` that returns a tuple
        of gradients for the requested ``argnums``.
    """
    if not hasattr(fn, "_mpmd_state") or not fn._mpmd_state.get("schedule_requested", False):
        raise TypeError("sxgrad requires an sxjit-decorated function with a schedule.")

    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    def grad_fn(*args: Any) -> tuple[Any, ...]:
        """Run the schedule for grads only, then re-pack into the user's pytree shape."""
        validated_argnums = _normalize_argnums(argnums, len(args))
        plan = _ensure_schedule_plan(fn, args)
        _loss, grads_flat = _dispatch_schedule_faithful(plan, args, return_loss=False)
        leaf_ranges = _arg_leaf_ranges(args)
        result = []
        for argnum in validated_argnums:
            start, end = leaf_ranges[argnum]
            arg_leaves = grads_flat[start:end]
            if len(arg_leaves) == 1:
                result.append(arg_leaves[0])
            else:
                arg_grad = jax.tree.unflatten(jax.tree.structure(args[argnum]), list(arg_leaves))
                result.append(arg_grad)
        return tuple(result)

    return grad_fn


def sxvalue_and_grad(fn: Callable, argnums: int | tuple[int, ...] = 0) -> Callable:
    """Schedule-faithful ``value_and_grad`` of a schedule-driven ``sxjit`` function.

    Args:
        fn: A function decorated with ``@sxjit(..., schedule=...)``.
        argnums: Positional argument indices to differentiate w.r.t.

    Returns:
        A callable with the same signature as ``fn`` that returns
        ``(loss, grads_tuple)``.
    """
    if not hasattr(fn, "_mpmd_state") or not fn._mpmd_state.get("schedule_requested", False):
        raise TypeError("sxvalue_and_grad requires an sxjit-decorated function with a schedule.")

    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    def vg_fn(*args: Any) -> tuple[jax.Array, tuple[Any, ...]]:
        """Run the schedule for both loss and grads, returning ``(loss, grad_tuple)``."""
        validated_argnums = _normalize_argnums(argnums, len(args))
        plan = _ensure_schedule_plan(fn, args)
        loss, grads_flat = _dispatch_schedule_faithful(plan, args, return_loss=True)
        leaf_ranges = _arg_leaf_ranges(args)
        result = []
        for argnum in validated_argnums:
            start, end = leaf_ranges[argnum]
            arg_leaves = grads_flat[start:end]
            if len(arg_leaves) == 1:
                result.append(arg_leaves[0])
            else:
                arg_grad = jax.tree.unflatten(jax.tree.structure(args[argnum]), list(arg_leaves))
                result.append(arg_grad)
        return loss, tuple(result)

    return vg_fn


def _ensure_schedule_plan(fn: Callable, args: tuple[Any, ...]) -> dict[str, Any]:
    """Return ``fn``'s cached schedule plan, building it on the first call.

    :func:`sxgrad` and :func:`sxvalue_and_grad` may be invoked before
    the wrapped function ever ran (so its on-demand build never
    fired). This helper triggers ``fn._mpmd_build`` once with the
    user's arguments and returns the resulting plan from
    ``fn._mpmd_state``.

    Args:
        fn: A function decorated by ``@sxjit(..., schedule=...)``.
        args: User-provided positional arguments (used to seed the
            initial trace).

    Returns:
        The cached schedule plan dict.

    Raises:
        TypeError: If ``fn`` does not expose ``_mpmd_build`` or the
            build never produced a schedule plan (i.e. ``schedule=...``
            was not supplied).
    """
    plan = fn._mpmd_state.get("schedule_plan")
    if plan is not None:
        return plan
    build = getattr(fn, "_mpmd_build", None)
    if build is None:
        raise TypeError("schedule gradients require a function decorated with sxjit(..., schedule=...).")
    build(args, {})
    plan = fn._mpmd_state.get("schedule_plan")
    if plan is None:
        raise TypeError("sxjit did not produce a schedule plan. Did you pass schedule=... to sxjit?")
    return plan


def sxjit(
    fn: Callable | None = None,
    *,
    mesh: "SpxMesh | MpMdMesh",
    schedule: Schedule | None = None,
    static_argnums: int | tuple[int, ...] | None = None,
    static_argnames: str | tuple[str, ...] | None = None,
    donate_argnums: int | tuple[int, ...] | None = None,
    in_shardings: Any | None = None,
    out_shardings: Any | None = None,
) -> Callable:
    """Decorator that traces a function, splits it at :func:`sxstage_iter`
    markers, and compiles each stage into a separate XLA executable per rank.

    True MPMD: rank 0 compiles only stage 0's ops, rank N-1 compiles only
    stage N-1. No ``lax.cond``, no ``shard_map``, no shared HLO.

    The decorated function must call :func:`sxstage_iter` to mark
    stage boundaries. For an N-rank mesh, use exactly N-1 markers::

        @sxjit(mesh=mesh)
        def forward(model, x):
            x = model.embed(x)
            for blk in model.blocks[:16]:
                x = blk(x)
            x = sxstage_iter(x)
            for blk in model.blocks[16:]:
                x = blk(x)
            return model.head(x)

        logits = forward(model, token_ids)

    On the first call the decorator traces ``fn`` via :func:`jax.make_jaxpr`,
    splits the jaxpr at the markers via :func:`cluster_jaxpr_by_markers`,
    builds a ``@jax.jit`` per cluster on its rank's sub-mesh, and places
    model parameters. Subsequent calls reuse the compiled executables and
    placed parameters, dispatching only the per-rank jits with
    :func:`jax.device_put` for cross-rank activation transfer.

    Return values may originate from any stage. The outvar map tracks
    which cluster produced each return value so that per-rank carry state
    (e.g. KV cache pages) is returned from the correct rank with its
    device placement preserved.

    Args:
        fn: The function to pipeline.
        mesh: An MPMD-capable mesh (:class:`SpxMesh` or :class:`MpMdMesh`).
        schedule: Optional :class:`Schedule` for schedule-driven training
            with ``jax.grad`` support. When provided, the function must
            return a scalar loss and ``sxgrad`` / ``sxvalue_and_grad``
            can be used for faithful schedule-aware backprop.
        static_argnums: Which positional arguments are static (compile-time
            constants). Static args are traced as constants and their values
            are embedded in the compiled XLA. This is useful for configuration
            objects, boolean flags, or small non-array data. When not provided,
            the legacy forward-only path uses its historical inference. The
            schedule path keeps :class:`Module` and non-array metadata static
            while leaving array pytrees such as batches dynamic.
        static_argnames: Which keyword arguments are static (compile-time
            constants). Behaves like ``static_argnums`` but for kwargs.
        donate_argnums: Which positional arguments should have their device
            buffers donated to the computation. This can reduce memory usage
            for large inputs that are only used by a single pipeline stage.
            An argument used by multiple stages cannot be donated safely and
            will be silently skipped.
        in_shardings: Per-leaf input shardings as a pytree matching ``fn``'s
            args. ``None`` entries fall through to auto-inference from
            :class:`Module` logical axis annotations. If the entire argument
            is ``None``, all shardings are inferred automatically. Arrays
            already on the correct rank's devices are never moved.
        out_shardings: Sharding applied to all outputs after dispatch.
            Can be a single :class:`~jax.sharding.Sharding` (applied to
            every output), a list/tuple of shardings (one per output, with
            ``None`` meaning "preserve"), or ``None`` (preserve whatever
            sharding each output has from its producing rank).

    Returns:
        A wrapped callable with the same signature as ``fn``.
    """
    mpmd_mesh = resolve_mpmd_mesh(mesh)

    def decorator(fn: Callable) -> Callable:
        """Build the per-rank dispatch plan on first call, replay on subsequent calls."""
        n = mpmd_mesh.mpmd_dim
        stage_shardings = [mpmd_mesh.sub_sharding(i) for i in range(n)]
        rank_submeshes = [mpmd_mesh.submesh(i) for i in range(n)]
        _state: dict[str, Any] = {"schedule_requested": schedule is not None}

        def _build(args, kwargs):
            """Trace ``fn``, cluster by markers, compile per-rank jits, place params.

            Called exactly once on the first invocation. Populates
            ``_state`` with the compiled dispatch plan, placed static
            parameters, dynamic-index set, explicit sharding overrides,
            and the output variable map.

            Three code paths branch off the traced jaxpr:
            * If a ``pscan_p`` equation is present (user called
              :func:`treduce`), route through :mod:`pscan_compiler`.
            * If ``schedule`` is provided, build a schedule-driven plan.
            * Otherwise fall through to the forward-only marker-cluster path.
            """
            static_nums = set(_normalize_argnums(static_argnums, len(args)))
            donate_nums = set(_normalize_argnums(donate_argnums, len(args)))
            static_names = _normalize_argnames(static_argnames)
            _state["result_treedef"] = _result_treedef_for_call(
                fn,
                args,
                kwargs,
                static_argnums,
                static_argnames,
            )

            use_legacy_path = static_argnums is None and static_argnames is None and donate_argnums is None

            if use_legacy_path:
                closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
                dynamic_flat_to_orig_flat = None
                orig_flat_to_dynamic_flat = None
                constvar_id_to_idx = None
            else:
                if static_nums and donate_nums and static_nums & donate_nums:
                    overlap = sorted(static_nums & donate_nums)
                    raise ValueError(f"sxjit: arguments at indices {overlap} cannot be both static and donated.")

                static_kwargs = {k: kwargs[k] for k in static_names if k in kwargs}
                dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_names}
                dynamic_nums = tuple(i for i in range(len(args)) if i not in static_nums)

                placeholder_args = list(args)
                for i in dynamic_nums:
                    placeholder_args[i] = None

                def _wrapper(*dyn_args, **dyn_kwargs):
                    """Re-pack dynamic args+kwargs back into the original ``fn(...)`` call.

                    Mirror of the inner ``_wrapper`` in
                    :func:`_build_schedule_plan` but for the
                    forward-only ``sxjit`` path: static positional
                    args are baked in via ``placeholder_args`` and
                    static kwargs are spread back in here.
                    """
                    full_args = list(placeholder_args)
                    for idx, darg in zip(dynamic_nums, dyn_args, strict=False):
                        full_args[idx] = darg
                    return fn(*full_args, **static_kwargs, **dyn_kwargs)

                dynamic_args = tuple(args[i] for i in dynamic_nums)
                closed_jaxpr = jax.make_jaxpr(_wrapper)(*dynamic_args, **dynamic_kwargs)

                dynamic_flat_to_orig_flat: dict[int, int] = {}
                orig_flat_to_dynamic_flat: dict[int, int] = {}
                dyn_flat_idx = 0
                orig_flat_idx = 0
                for i, arg in enumerate(args):
                    n_leaves = len(jax.tree.leaves(arg))
                    if i not in static_nums:
                        for j in range(n_leaves):
                            dynamic_flat_to_orig_flat[dyn_flat_idx + j] = orig_flat_idx + j
                            orig_flat_to_dynamic_flat[orig_flat_idx + j] = dyn_flat_idx + j
                        dyn_flat_idx += n_leaves
                    orig_flat_idx += n_leaves
                for _k, v in dynamic_kwargs.items():
                    n_leaves = len(jax.tree.leaves(v))
                    for j in range(n_leaves):
                        dynamic_flat_to_orig_flat[dyn_flat_idx + j] = orig_flat_idx + j
                        orig_flat_to_dynamic_flat[orig_flat_idx + j] = dyn_flat_idx + j
                    dyn_flat_idx += n_leaves
                    orig_flat_idx += n_leaves

                constvar_id_to_idx = {id(v): i for i, v in enumerate(closed_jaxpr.jaxpr.constvars)}

            pscan_eqns = has_pscan(closed_jaxpr.jaxpr)
            if pscan_eqns:
                if not use_legacy_path:
                    raise NotImplementedError(
                        "sxjit: static_argnums / donate_argnums are not yet supported with pscan paths."
                    )
                if len(pscan_eqns) > 1:
                    raise NotImplementedError(
                        "sxjit supports at most one pscan_p equation (treduce call) per decorated function in the MVP."
                    )
                outer_flat_args = tuple(jax.tree.leaves(args))
                plan = build_pscan_plan(
                    closed_jaxpr,
                    args,
                    outer_flat_args,
                    pscan_eqns[0],
                    mpmd_mesh,
                    stage_shardings,
                    rank_submeshes,
                )
                _state["pscan_plan"] = plan
                return

            if schedule is not None:
                plan = _build_schedule_plan(
                    fn,
                    args,
                    kwargs,
                    schedule,
                    mpmd_mesh,
                    stage_shardings,
                    rank_submeshes,
                    static_argnums,
                    donate_argnums,
                )
                _state["schedule_plan"] = plan
                return

            edge_shardings = marker_edge_shardings(closed_jaxpr.jaxpr)
            clusters = cluster_jaxpr_by_markers(closed_jaxpr.jaxpr)
            consts = closed_jaxpr.consts

            if len(clusters) != n:
                raise ValueError(
                    f"sxjit: function has {len(clusters)} stages "
                    f"({len(clusters) - 1} sxstage_iter markers) "
                    f"but mesh has {n} MPMD ranks. Need exactly "
                    f"{n - 1} markers."
                )

            original_id_to_idx = {id(v): i for i, v in enumerate(closed_jaxpr.jaxpr.invars)}

            fn_outvar_map = _build_outvar_map(
                closed_jaxpr,
                clusters,
                original_id_to_idx,
                constvar_id_to_idx=constvar_id_to_idx,
                consts=consts,
            )

            donate_per_stage = None
            if not use_legacy_path and donate_nums:
                donate_per_stage = _compute_donation(
                    clusters,
                    original_id_to_idx,
                    orig_flat_to_dynamic_flat,
                    args,
                    donate_nums,
                    static_nums,
                    n,
                )

            cluster_plans = _build_cluster_plans(
                clusters,
                consts,
                stage_shardings,
                rank_submeshes,
                original_id_to_idx,
                n,
                body_jaxpr=closed_jaxpr.jaxpr,
                edge_shardings=edge_shardings,
                donate_argnums_per_stage=donate_per_stage,
                all_constvars=list(closed_jaxpr.jaxpr.constvars) if not use_legacy_path else None,
            )

            flat_init = jax.tree.leaves(args)
            if use_legacy_path:
                n_model_leaves = len(jax.tree.leaves(args[:-1]))
                dynamic = set(range(n_model_leaves, len(flat_init)))
            else:
                static_flat = set()
                for i in static_nums:
                    start = sum(len(jax.tree.leaves(args[j])) for j in range(i))
                    n_leaves = len(jax.tree.leaves(args[i]))
                    static_flat.update(range(start, start + n_leaves))
                dynamic = set(range(len(flat_init))) - static_flat

            leaf_shardings, leaf_stage_owners = _infer_leaf_shardings(
                args,
                flat_init,
                n,
                rank_submeshes,
            )

            explicit_in_sh = _resolve_explicit_shardings(in_shardings, flat_init)

            placed = _place_static_args(
                cluster_plans,
                flat_init,
                dynamic,
                explicit_in_sh,
                leaf_shardings,
                leaf_stage_owners,
                rank_submeshes,
            )

            _state["compiled"] = cluster_plans
            _state["placed"] = placed
            _state["dynamic"] = dynamic
            _state["explicit_in_sh"] = explicit_in_sh
            _state["fn_outvar_map"] = fn_outvar_map
            if not use_legacy_path:
                _state["dynamic_flat_to_orig_flat"] = dynamic_flat_to_orig_flat

        def _dispatch(args):
            """Fire per-rank executables with pre-placed params and fresh dynamic inputs.

            Static args (model params) use the cached placement from
            ``_build``. Dynamic args (user inputs) and inter-stage
            activations are placed per-call. Arrays already on the
            correct rank's devices pass through without ``device_put``.

            If a ``pscan_plan`` is present (schedule-driven training
            path), delegate to the schedule-aware dispatcher in
            :mod:`pscan_compiler`. Otherwise run the forward-only
            marker-cluster path.
            """
            if "pscan_plan" in _state:
                results = dispatch_pscan(_state["pscan_plan"])
                if len(results) == 1:
                    return _restore_result_treedef(results[0], _state.get("result_treedef"))
                return _restore_result_treedef(tuple(results), _state.get("result_treedef"))

            compiled = _state["compiled"]
            placed = _state["placed"]
            dynamic = _state["dynamic"]
            explicit_in_sh = _state["explicit_in_sh"]
            flat_args = jax.tree.leaves(args)
            all_cluster_outputs: list[tuple] = []
            prev_outputs: tuple = ()

            for ri, (stage_jit, submesh, my_sh, _, invar_map) in enumerate(compiled):
                rank_devices = set(rank_submeshes[ri].devices.flat)
                invars = _assemble_invars(
                    invar_map,
                    flat_args,
                    placed,
                    dynamic,
                    explicit_in_sh,
                    prev_outputs,
                    all_cluster_outputs,
                    ri,
                    my_sh,
                    rank_devices,
                    rank_submeshes,
                    mpmd_mesh,
                    dynamic_flat_to_orig_flat=_state.get("dynamic_flat_to_orig_flat"),
                )
                with submesh:
                    prev_outputs = stage_jit(*invars)
                all_cluster_outputs.append(prev_outputs)

            return _assemble_outputs(
                _state["fn_outvar_map"],
                all_cluster_outputs,
                flat_args,
            )

        # @erfanzar NOTE:
        # Bug this fixes: the original ``wrapped(...)`` below traced once
        # on the first call (``_build(args, kwargs)``) and then dispatched
        # against any future inputs without re-checking shapes. Plain
        # ``jax.jit`` re-traces automatically when leaf shapes change;
        # this MPMD wrapper did not. So calling the same compiled
        # function with a different input shape — e.g. eSurge's compile
        # loop hitting ``[1/11]: 4 tokens`` then ``[2/11]: 8 tokens`` —
        # silently reused the old jaxpr (which baked in shape ``(3, 4)``
        # for ``expand_dims``) against the new ``(3, 8)`` input and blew
        # up inside ``broadcast_in_dim``.
        #
        # Fix: key a per-shape cache of compiled-state snapshots by a
        # signature derived from the input pytree treedef + leaf
        # shape/dtype. ``_state`` itself remains the currently-active
        # build (so closures captured by ``_build`` / ``_dispatch``
        # continue to see the dict they always did) — we just swap its
        # contents in/out from ``_state_cache`` as the input signature
        # changes. Same shape repeated → cache hit, no rebuild.
        # Alternating shapes → both cached, no rebuild on either side.

        def _shape_signature(args: tuple, kwargs: dict) -> tuple:
            """Build a hashable signature capturing pytree structure and leaf shape/dtype.

            Two calls share a signature exactly when their flattened
            inputs share both treedef and per-leaf ``(shape, dtype)``.
            Used as the cache key for swappable plan snapshots so that
            the wrapper retraces only when the input layout actually
            changes.

            Args:
                args: Positional call arguments.
                kwargs: Keyword call arguments.

            Returns:
                ``(treedef, ((shape0, dtype0), (shape1, dtype1), ...))``.
            """
            leaves, treedef = jax.tree.flatten((args, kwargs))
            leaf_sig = tuple((getattr(leaf, "shape", None), getattr(leaf, "dtype", None)) for leaf in leaves)
            return (treedef, leaf_sig)

        _SIG_KEY = "__shape_signature__"
        _BUILT_KEYS = ("compiled", "schedule_plan", "pscan_plan")
        _state_cache: dict = {}

        def _swap_in(snapshot: dict | None) -> None:
            """Replace ``_state``'s entries with those of ``snapshot`` in place.

            Mutating ``_state`` keeps every closure that already captured
            it (the ``_build`` and ``_dispatch`` inner functions) seeing
            the new contents. Passing ``None`` clears ``_state`` so a
            fresh ``_build`` can populate it.

            Args:
                snapshot: A previously-captured plan snapshot, or
                    ``None`` to wipe ``_state``.
            """
            for k in list(_state.keys()):
                del _state[k]
            if snapshot is not None:
                _state.update(snapshot)

        def wrapped(*args, **kwargs):
            """The user-visible callable returned by :func:`sxjit`.

            On every call we compute the shape signature and look up
            a matching cached plan; if none exists we run ``_build``
            once. Once a plan is in scope, the call routes to the
            schedule dispatcher, the pscan dispatcher, or the legacy
            per-stage path depending on which key ``_build`` populated.
            Output pytree structure (lost when the runtime returns
            flat tuples) is restored via the captured ``result_treedef``.
            """
            sig = _shape_signature(args, kwargs)
            cur_sig = _state.get(_SIG_KEY)
            if cur_sig != sig:
                # Save the current build under its signature so it is
                # reusable if the shape comes back later.
                if cur_sig is not None and any(k in _state for k in _BUILT_KEYS):
                    _state_cache[cur_sig] = {k: v for k, v in _state.items()}
                cached = _state_cache.get(sig)
                if cached is not None:
                    _swap_in(cached)
                else:
                    _swap_in(None)
                    _build(args, kwargs)
                    _state[_SIG_KEY] = sig
                    _state_cache[sig] = {k: v for k, v in _state.items()}

            if "schedule_plan" in _state:
                plan = _state["schedule_plan"]
                return _schedule_forward(plan, *args)

            if "pscan_plan" in _state:
                results = dispatch_pscan(_state["pscan_plan"])
                if len(results) == 1:
                    return _restore_result_treedef(results[0], _state.get("result_treedef"))
                return _restore_result_treedef(tuple(results), _state.get("result_treedef"))

            result = _dispatch(args)
            result = _apply_out_shardings(result, out_shardings)
            return _restore_result_treedef(result, _state.get("result_treedef"))

        wrapped.__name__ = getattr(fn, "__name__", "mpmd_jit_fn")
        wrapped.__qualname__ = getattr(fn, "__qualname__", "mpmd_jit_fn")
        wrapped._mpmd_state = _state
        wrapped._mpmd_build = _build
        return wrapped

    if fn is not None:
        return decorator(fn)
    return decorator


def _build_outvar_map(
    closed_jaxpr: Any,
    clusters: list,
    original_id_to_idx: dict[int, int],
    constvar_id_to_idx: dict[int, int] | None = None,
    consts: tuple[Any, ...] | None = None,
) -> list[tuple]:
    """Map each of the original function's output vars to the cluster that defines it.

    Returns a list parallel to ``closed_jaxpr.jaxpr.outvars``. Each entry
    is ``(cluster_rank, position_in_cluster_outvars)`` for values produced
    by a cluster, ``("orig_passthrough", flat_arg_index)`` for values
    that are original function inputs passed through unchanged, or
    ``("const_passthrough", concrete_value)`` for static constants.
    """
    resolve_alias = _marker_alias_resolver(closed_jaxpr.jaxpr)
    original_idx_by_id = {id(resolve_alias(v)): i for i, v in enumerate(closed_jaxpr.jaxpr.invars) if isinstance(v, Var)}
    if not original_idx_by_id:
        original_idx_by_id = original_id_to_idx
    cluster_outvar_ids: list[dict[int, int]] = [
        {id(resolve_alias(v)): pos for pos, v in enumerate(c.outvars) if isinstance(v, Var)} for c in clusters
    ]
    fn_outvar_map: list[tuple] = []
    for v in closed_jaxpr.jaxpr.outvars:
        vid = id(resolve_alias(v)) if isinstance(v, Var) else id(v)
        found = None
        for ri in range(len(clusters) - 1, -1, -1):
            if vid in cluster_outvar_ids[ri]:
                found = (ri, cluster_outvar_ids[ri][vid])
                break
        if found is not None:
            fn_outvar_map.append(found)
        else:
            orig_idx = original_idx_by_id.get(vid, original_id_to_idx.get(id(v)))
            if orig_idx is not None:
                fn_outvar_map.append(("orig_passthrough", orig_idx))
            elif constvar_id_to_idx is not None:
                const_idx = constvar_id_to_idx.get(vid)
                if const_idx is not None and consts is not None:
                    fn_outvar_map.append(("const_passthrough", consts[const_idx]))
                else:
                    fn_outvar_map.append(("missing", -1))
            else:
                fn_outvar_map.append(("missing", -1))
    return fn_outvar_map


def _marker_alias_resolver(body_jaxpr: Any) -> Callable[[Any], Any]:
    """Return a closure that follows ``sxstage_iter`` outvar -> invar identity edges.

    Marker primitives are identities, so two clusters that read the
    "same" value really see distinct :class:`Var` objects: the first
    sees the marker's input, the second its output. To match producers
    and consumers we walk the chain back to the originating var
    whenever we look up by id. The returned resolver is loop-safe via
    a per-call ``seen`` set in case a malformed jaxpr cycles.

    Args:
        body_jaxpr: A jaxpr that may contain :data:`sxstage_iter_p`
            equations.

    Returns:
        ``resolve_alias(var) -> Var`` walking through marker edges.
    """
    alias_by_id = {
        id(outvar): invar
        for eqn in body_jaxpr.eqns
        if eqn.primitive is sxstage_iter_p
        for invar, outvar in zip(eqn.invars, eqn.outvars, strict=True)
        if isinstance(invar, Var) and isinstance(outvar, Var)
    }

    def resolve_alias(var: Any) -> Any:
        """Walk through ``sxstage_iter`` output->input chains to the originating var.

        The marker primitive forwards values through identity equations
        (``out = sxstage_iter(in)``); for cluster planning we want to
        treat ``out`` as if it were ``in`` so cross-stage sharing maps
        to the right producer. ``seen`` guards against pathological
        cycles in malformed jaxprs.
        """
        cur = var
        seen: set[int] = set()
        while isinstance(cur, Var) and id(cur) in alias_by_id and id(cur) not in seen:
            seen.add(id(cur))
            cur = alias_by_id[id(cur)]
        return cur

    return resolve_alias


def _build_cluster_plans(
    clusters: list,
    consts: tuple,
    stage_shardings: list,
    rank_submeshes: list,
    original_id_to_idx: dict[int, int],
    n: int,
    body_jaxpr: Any | None = None,
    edge_shardings: list[Any] | tuple[Any, ...] | None = None,
    donate_argnums_per_stage: list[tuple[int, ...]] | None = None,
    all_constvars: list | None = None,
) -> list[tuple]:
    """Build per-rank ``(stage_jit, submesh, sharding, next_sharding, invar_map)`` tuples.

    Each ``stage_jit`` is a ``@jax.jit``-wrapped evaluator for that
    cluster's sub-jaxpr with constants pre-placed on the rank's sub-mesh.
    ``invar_map`` classifies each cluster invar as either ``("orig", idx)``
    (from the original function args at flat index ``idx``), ``("stage",
    rank, pos)`` (from an earlier cluster output), or legacy ``("prev",
    pos)`` entries.
    """
    if donate_argnums_per_stage is None:
        donate_argnums_per_stage = [()] * n
    if edge_shardings is None:
        edge_shardings = ()

    const_idx_by_id: dict[int, int] | None = None
    if all_constvars is not None:
        const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    resolve_alias = _marker_alias_resolver(body_jaxpr) if body_jaxpr is not None else (lambda v: v)
    original_idx_by_id = original_id_to_idx
    if body_jaxpr is not None:
        original_idx_by_id = {id(resolve_alias(v)): i for i, v in enumerate(body_jaxpr.invars) if isinstance(v, Var)}
    producer_by_var_id: dict[int, tuple[int, int]] = {}

    plans = []
    for rank, cluster in enumerate(clusters):
        sub_sharding = stage_shardings[rank]
        invar_map: list[tuple] = []
        for v in cluster.invars:
            canonical = resolve_alias(v)
            producer = producer_by_var_id.get(id(canonical))
            if producer is not None:
                src_rank, src_pos = producer
                if src_rank >= rank:
                    raise ValueError(
                        "sxjit: cluster input was mapped to a non-earlier stage "
                        f"(stage {rank} input from stage {src_rank}, output {src_pos})."
                    )
                invar_map.append(("stage", src_rank, src_pos, _edge_sharding_for_logical(edge_shardings, src_rank)))
                continue

            orig_idx = original_idx_by_id.get(id(canonical), original_id_to_idx.get(id(v)))
            if orig_idx is not None:
                invar_map.append(("orig", orig_idx))
                continue

            raise ValueError(
                "sxjit: could not map a stage input to an original argument "
                f"or an earlier stage output. Stage={rank}, input={v}."
            )

        for out_idx, outvar in enumerate(cluster.outvars):
            canonical = resolve_alias(outvar)
            if isinstance(canonical, Var):
                producer_by_var_id[id(canonical)] = (rank, out_idx)

        if const_idx_by_id is not None:
            used_constvars = _collect_used_constvars(cluster)
            filtered_cluster = _filtered_cluster(cluster, used_constvars)
            const_indices = tuple(const_idx_by_id[id(v)] for v in used_constvars)
            pc = tuple(jax.device_put(consts[idx], sub_sharding) for idx in const_indices)
            eval_jaxpr = filtered_cluster
        else:
            pc = tuple(jax.device_put(c, sub_sharding) for c in consts)
            eval_jaxpr = cluster

        donate = donate_argnums_per_stage[rank]
        if donate:

            @functools.partial(jax.jit, donate_argnums=donate)
            def stage_jit(*invars, _c=pc, _j=eval_jaxpr):
                """Run the cluster sub-jaxpr with constants pre-placed on the rank.

                ``_c`` and ``_j`` are bound at definition so each stage's
                jit closes over its own placed constants and jaxpr (no
                cross-stage closure leaks).

                Args:
                    *invars: Stage input activations.

                Returns:
                    Tuple of stage outputs aligned with the cluster's
                    outvars.
                """
                return tuple(jax.core.eval_jaxpr(_j, list(_c), *invars))

        else:

            @jax.jit
            def stage_jit(*invars, _c=pc, _j=eval_jaxpr):
                """Run the cluster sub-jaxpr with constants pre-placed on the rank.

                ``_c`` and ``_j`` are bound at definition so each stage's
                jit closes over its own placed constants and jaxpr (no
                cross-stage closure leaks).

                Args:
                    *invars: Stage input activations.

                Returns:
                    Tuple of stage outputs aligned with the cluster's
                    outvars.
                """
                return tuple(jax.core.eval_jaxpr(_j, list(_c), *invars))

        plans.append(
            (
                stage_jit,
                rank_submeshes[rank],
                sub_sharding,
                stage_shardings[rank + 1] if rank < n - 1 else None,
                invar_map,
            )
        )
    return plans


def _infer_leaf_shardings(
    args: tuple,
    flat_init: list,
    n: int,
    rank_submeshes: list,
    *,
    stage_rank_resolver: Callable[[tuple[int, int] | None], int | None] | None = None,
) -> tuple[list[dict[int, Any]], dict[int, int]]:
    """Auto-infer per-leaf shardings from :class:`Module` logical axis annotations.

    Scans ``args`` for :class:`Module` instances, calls
    :func:`get_named_sharding` for each rank's sub-mesh, and returns a
    list of dicts mapping flat-arg indices to :class:`NamedSharding`
    objects plus a flat-index -> owning-rank map derived from any
    explicit ``assign_stage(...)`` metadata. Non-Module args get no
    entry (fall through to replicated).
    """
    leaf_shardings: list[dict[int, Any]] = [{} for _ in range(n)]
    leaf_stage_owners: dict[int, int] = {}
    for arg in args:
        if not isinstance(arg, Module):
            continue
        _, state = export(arg)
        cache = arg._spx_export_cache
        leaf_spec = (
            cache[6] if cache is not None and len(cache) >= 7 else tuple((col, path) for col, path in state.paths())
        )
        vars_by_key = {(var.kind, path): var for path, var in live_variables(arg)}
        arg_leaves = jax.tree.leaves(arg)
        first_leaf_id = id(arg_leaves[0]) if arg_leaves else None
        offset = None
        for fi, fl in enumerate(flat_init):
            if id(fl) == first_leaf_id:
                offset = fi
                break
        if offset is None:
            continue
        leaf_entries: list[tuple[int, str, str, int | None]] = []
        for li, (col, path) in enumerate(leaf_spec):
            flat_idx = offset + li
            var = vars_by_key.get((col, path))
            assignment = metadata_stage_assignment(var.metadata) if var is not None else None
            owner = (
                stage_rank_resolver(assignment) if stage_rank_resolver is not None else resolve_stage_rank(assignment, n)
            )
            if owner is not None:
                leaf_stage_owners[flat_idx] = owner
            leaf_entries.append((flat_idx, col, path, owner))
        for rank in range(n):
            per_leaf = get_named_sharding(arg, rank_submeshes[rank])
            for flat_idx, col, path, owner in leaf_entries:
                if owner is not None and owner != rank:
                    continue
                sh = per_leaf.get(col, {}).get(path)
                if sh is not None:
                    leaf_shardings[rank][flat_idx] = sh
    return leaf_shardings, leaf_stage_owners


def _resolve_explicit_shardings(
    in_shardings: Any | None,
    flat_init: list,
) -> dict[int, Any]:
    """Flatten explicit ``in_shardings`` into a flat-index to sharding map.

    Returns an empty dict when ``in_shardings`` is ``None``.
    """
    if in_shardings is None:
        return {}
    explicit: dict[int, Any] = {}
    leaves = jax.tree_util.tree_leaves(in_shardings, is_leaf=lambda x: x is None)
    for i, sh in enumerate(leaves):
        if sh is not None and i < len(flat_init):
            explicit[i] = sh
    return explicit


def _place_static_args(
    cluster_plans: list[tuple],
    flat_init: list,
    dynamic: set[int],
    explicit_in_sh: dict[int, Any],
    leaf_shardings: list[dict[int, Any]],
    leaf_stage_owners: dict[int, int],
    rank_submeshes: list,
) -> dict[tuple[int, int], Any]:
    """Place static (non-dynamic) args on each rank's sub-mesh, cached for reuse.

    Placement priority per leaf:

    1. Explicit ``assign_stage(...)`` ownership, when present.
    2. Explicit ``in_shardings`` override.
    3. Already on the correct rank's devices — skip ``device_put``
       (preserves carry state like KV cache pages).
    4. Inferred from :class:`Module` logical axis annotations.
    5. Fallback: replicated on the rank's sub-mesh.
    """
    placed: dict[tuple[int, int], Any] = {}
    for ri, (_, _, fallback_sh, _, imap) in enumerate(cluster_plans):
        rank_devices = set(rank_submeshes[ri].devices.flat)
        for source in imap:
            kind = source[0]
            idx = source[1]
            if kind != "orig" or idx in dynamic:
                continue
            owner = leaf_stage_owners.get(idx)
            if owner is not None and owner != ri:
                raise ValueError(
                    f"sxjit: flat argument leaf {idx} is assigned to pipeline "
                    f"stage {owner}, but traced stage {ri} uses it. Move the "
                    f"corresponding layer into the matching pipeline segment or "
                    f"update its assign_stage(...) hint."
                )
            leaf = flat_init[idx]
            if idx in explicit_in_sh:
                placed[(ri, idx)] = jax.device_put(leaf, explicit_in_sh[idx])
            elif hasattr(leaf, "devices") and set(leaf.devices()).issubset(rank_devices):
                placed[(ri, idx)] = leaf
            elif idx in leaf_shardings[ri]:
                placed[(ri, idx)] = jax.device_put(leaf, leaf_shardings[ri][idx])
            else:
                placed[(ri, idx)] = jax.device_put(leaf, fallback_sh)
    return placed


def _assemble_invars(
    invar_map: list[tuple],
    flat_args: list,
    placed: dict[tuple[int, int], Any],
    dynamic: set[int],
    explicit_in_sh: dict[int, Any],
    prev_outputs: tuple,
    all_cluster_outputs: list[tuple],
    ri: int,
    my_sh: Any,
    rank_devices: set,
    rank_submeshes: list[Any],
    mpmd_mesh: MpMdMesh,
    dynamic_flat_to_orig_flat: dict[int, int] | None = None,
) -> list:
    """Assemble invars for one stage dispatch from placed params, dynamic args,
    and previous stage outputs.

    Dynamic args use the same placement priority as static args: explicit
    sharding > already on correct devices > fallback replicated.
    """
    invars = []
    for source in invar_map:
        kind = source[0]
        idx = source[1]
        if kind == "orig":
            if dynamic_flat_to_orig_flat is not None:
                orig_idx = dynamic_flat_to_orig_flat.get(idx, idx)
            else:
                orig_idx = idx
            if orig_idx in dynamic:
                leaf = flat_args[orig_idx]
                if orig_idx in explicit_in_sh:
                    invars.append(jax.device_put(leaf, explicit_in_sh[orig_idx]))
                elif hasattr(leaf, "devices") and set(leaf.devices()).issubset(rank_devices):
                    invars.append(leaf)
                else:
                    invars.append(jax.device_put(leaf, my_sh))
            else:
                invars.append(placed[(ri, orig_idx)])
        elif kind == "stage":
            src_rank, src_pos = source[1], source[2]
            value = all_cluster_outputs[src_rank][src_pos]
            invars.append(
                jax.device_put(
                    value,
                    _edge_transfer_sharding(
                        value,
                        edge_sharding=source[3] if len(source) > 3 else None,
                        fallback_sharding=my_sh,
                        dst_rank=ri,
                        rank_submeshes=rank_submeshes,
                        mpmd_mesh=mpmd_mesh,
                    ),
                )
            )
        else:
            invars.append(jax.device_put(prev_outputs[idx], my_sh))
    return invars


def _assemble_outputs(
    fn_outvar_map: list[tuple],
    all_cluster_outputs: list[tuple],
    flat_args: list,
) -> Any:
    """Collect return values from all clusters using the outvar map.

    Each function outvar is sourced from the cluster that defined it,
    preserving per-rank device placement for carry state.
    """
    final = []
    for mapping in fn_outvar_map:
        src_rank, src_pos = mapping
        if isinstance(src_rank, int):
            final.append(all_cluster_outputs[src_rank][src_pos])
        elif src_rank == "orig_passthrough":
            final.append(flat_args[src_pos])
        elif src_rank == "const_passthrough":
            final.append(src_pos)
        else:
            final.append(None)
    if len(final) == 1:
        return final[0]
    return tuple(final)


def _apply_out_shardings(result: Any, out_shardings: Any | None) -> Any:
    """Apply ``out_shardings`` to the dispatch result.

    ``None`` preserves whatever sharding each output has from its
    producing rank. A single :class:`~jax.sharding.Sharding` is
    broadcast to all outputs. A list/tuple applies per-output (``None``
    entries mean "preserve").
    """
    if out_shardings is None:
        return result
    if isinstance(result, jax.Array):
        return jax.device_put(result, out_shardings)
    if isinstance(result, tuple):
        if isinstance(out_shardings, (list, tuple)):
            return tuple(
                jax.device_put(r, s) if s is not None and isinstance(r, jax.Array) else r
                for r, s in zip(result, out_shardings, strict=False)
            )
        return tuple(jax.device_put(r, out_shardings) if isinstance(r, jax.Array) else r for r in result)
    return result


def _place_state_on_rank(
    state: State,
    rank: int,
    stage: Module,
    rank_submeshes: list[Any],
    stage_shardings: list[Any],
) -> State:
    """Place ``state`` on ``rank``'s sub-mesh with per-leaf shardings.

    Uses :func:`get_named_sharding` to derive logical-axis-aware
    shardings for each leaf; leaves without a registered sharding fall
    back to the rank's replicated sharding.
    """
    per_leaf = get_named_sharding(stage, rank_submeshes[rank])
    replicated = stage_shardings[rank]
    out: dict[str, dict[str, Any]] = {}
    for col, path, leaf in state.items():
        sh = per_leaf.get(col, {}).get(path, replicated)
        out.setdefault(col, {})[path] = jax.device_put(leaf, sh)
    return type(state)(out)


def _active_profiler() -> "_Profiler | None":
    """Return the innermost active profiler on this thread, or ``None``.

    Uses thread-local storage so concurrent ``sxcall`` calls from
    different threads stay independent — useful for nested tests and
    any future multi-run orchestration.
    """
    return getattr(_PROFILER_STATE, "active", None)


class _Profiler:
    """Per-task millisecond accumulator used by :func:`collect_task_times_ms`.

    Holds a flat ``task_name -> list[float_ms]`` dict that the
    :func:`_time_call` helper appends to whenever a labelled MPMD
    sub-task completes. One :class:`_Profiler` is active per thread at
    a time; nested ``collect_task_times_ms`` contexts share the outer
    profiler's dict so timings remain comparable.
    """

    def __init__(self) -> None:
        """Create a profiler with an empty ``task_name -> list[ms]`` map.

        The map is populated by :meth:`record` as :func:`sxcall`'s
        wrapped callables fire. A fresh profiler is constructed each
        time :func:`collect_task_times_ms` enters a new (non-nested)
        context.
        """
        self.times_ms: dict[str, list[float]] = {}

    def record(self, task_name: str, dt_ms: float) -> None:
        """Append a millisecond duration to the bucket for ``task_name``.

        The bucket is created on first use so callers do not need to
        register names ahead of time.

        Args:
            task_name: Profiler label (e.g. ``"stage0_fwd_mb3"``).
            dt_ms: Wall-clock duration of the task, including
                :func:`jax.block_until_ready`.
        """
        self.times_ms.setdefault(task_name, []).append(dt_ms)


@contextlib.contextmanager
def collect_task_times_ms() -> Iterator[dict[str, list[float]]]:
    """Record wall-clock milliseconds for each MPMD task in the body.

    Yields a ``dict[str, list[float]]`` that's filled as
    :func:`sxcall` executes: keys are task names like
    ``"stage0_fwd_mb3"`` / ``"stage2_bwd_i_mb0"``, values are a list
    of per-call durations in milliseconds (one entry per schedule
    action). :func:`jax.block_until_ready` is called before recording,
    so timings include actual device-side work, not just dispatch.

    Only one profiler may be active per thread at a time — nested
    calls share the outer profiler's dict.

    Example::

        with collect_task_times_ms() as times:
            loss, grads = sxcall(model, (x, y), mpmd_mesh=mm, ...)

        for name, ms in sorted(times.items()):
            print(f"{name}: {ms}")
    """
    outer = _active_profiler()
    if outer is not None:
        yield outer.times_ms
        return
    prof = _Profiler()
    _PROFILER_STATE.active = prof
    try:
        yield prof.times_ms
    finally:
        _PROFILER_STATE.active = None


def _time_call(
    task_name: str,
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """Invoke ``fn(*args)`` and record its wall time when a profiler is active.

    When no profiler is on the current thread the call is dispatched
    directly with no overhead. When one is active we wrap the call in
    :func:`time.perf_counter_ns` and :func:`jax.block_until_ready` so
    the recorded time reflects device-side work rather than just
    Python dispatch latency.

    Args:
        task_name: Profiler bucket label.
        fn: Callable to invoke.
        *args: Positional arguments for ``fn``.

    Returns:
        Whatever ``fn(*args)`` returned.
    """
    prof = _active_profiler()
    if prof is None:
        return fn(*args)
    t0 = time.perf_counter_ns()
    out = fn(*args)
    jax.block_until_ready(out)
    prof.record(task_name, (time.perf_counter_ns() - t0) / 1e6)
    return out


TransportKind = Literal["device_put"]


def _transport(
    kind: TransportKind,
    x: Any,
    dest_sharding: Any,
    *,
    task_name: str | None = None,
    stats: _ScheduleStatsCollector | None = None,
    src_rank: int | None = None,
    dst_rank: int | None = None,
) -> Any:
    """Move ``x`` to ``dest_sharding`` via :func:`jax.device_put`.

    Args:
        kind: Currently only ``"device_put"`` is supported. Portable
            across CPU / TPU / single-process GPU backends.
        x: The source :class:`jax.Array`.
        dest_sharding: A :class:`jax.sharding.NamedSharding` whose mesh
            is the destination sub-mesh.
        task_name: Optional profiler label so :func:`collect_task_times_ms`
            can attribute the wall time to a specific transfer.

    Returns:
        A :class:`jax.Array` placed according to ``dest_sharding``.
    """
    if kind != "device_put":
        raise ValueError(f"Unknown transport kind: {kind!r}.")
    target_sharding = _retarget_transfer_sharding(x, dest_sharding)
    nbytes = _tree_nbytes(x)
    skip, cache_hit = _can_skip_device_put(x, target_sharding)
    if stats is not None:
        stats.record_transfer(
            nbytes=nbytes,
            skipped=skip,
            cache_hit=cache_hit,
            src_rank=src_rank,
            dst_rank=dst_rank,
        )
    if skip:
        return x

    def put_with_target() -> Any:
        """``device_put`` to the resharded target, falling back to the bare destination on failure.

        :func:`_retarget_transfer_sharding` may produce a per-leaf
        target that XLA rejects (e.g. shape/spec mismatch); in that
        case we retry with the caller's plain ``dest_sharding`` so
        the transfer at least succeeds on the right device set.
        """
        try:
            return jax.device_put(x, target_sharding)
        except (TypeError, ValueError):
            if target_sharding is dest_sharding:
                raise
            return jax.device_put(x, dest_sharding)

    if task_name is not None:
        return _time_call(task_name, put_with_target)
    return put_with_target()


def _edge_transfer_sharding(
    value: Any,
    *,
    edge_sharding: Any,
    fallback_sharding: Any,
    dst_rank: int,
    rank_submeshes: list[Any],
    mpmd_mesh: MpMdMesh,
) -> Any:
    """Resolve a marker edge ``PartitionSpec`` against the destination rank's sub-mesh.

    Mirrors :func:`_edge_transfer_target` from :mod:`pscan_compiler` but
    accepts the meshes as explicit parameters so the schedule
    dispatcher can call it without a full :class:`PscanPlan`. When
    ``edge_sharding`` is ``None`` the caller's ``fallback_sharding``
    (the destination rank's replicated sharding) is returned.

    Args:
        value: The array (or pytree of arrays) being transported —
            shape information is used to sanitise the spec.
        edge_sharding: ``PartitionSpec`` declared on the producing
            :func:`sxstage_iter` marker (or ``None``).
        fallback_sharding: Sharding to use when ``edge_sharding`` does
            not apply or the leaf is not array-like.
        dst_rank: Destination physical rank index.
        rank_submeshes: Per-rank sub-meshes.
        mpmd_mesh: The full MPMD mesh (used as the first sanitisation
            target).

    Returns:
        Either ``fallback_sharding`` or a (pytree of)
        :class:`NamedSharding` derived from ``edge_sharding``.
    """
    if edge_sharding is None:
        return fallback_sharding
    dst_mesh = rank_submeshes[dst_rank]

    def leaf_target(leaf: Any) -> Any:
        """Per-leaf NamedSharding derived from ``edge_sharding`` on ``dst_mesh``.

        Non-array leaves fall back to ``fallback_sharding``. The
        :class:`PartitionSpec` is sanitised twice — once against the
        global MPMD mesh (so axes outside the mesh are dropped) and
        once against the rank-local sub-mesh.
        """
        if not hasattr(leaf, "shape"):
            return fallback_sharding
        spec = sanitize_partition_spec_for_mesh_and_shape(
            edge_sharding,
            mesh=mpmd_mesh,
            shape=tuple(getattr(leaf, "shape", ())),
        )
        spec = sanitize_partition_spec_for_mesh_and_shape(
            spec,
            mesh=dst_mesh,
            shape=tuple(getattr(leaf, "shape", ())),
        )
        return jax.sharding.NamedSharding(dst_mesh, spec)

    if hasattr(value, "shape"):
        return leaf_target(value)
    leaves = jax.tree.leaves(value, is_leaf=_is_leaf)
    if not any(hasattr(leaf, "shape") for leaf in leaves):
        return fallback_sharding
    return jax.tree.map(leaf_target, value, is_leaf=_is_leaf)


def _edge_sharding_for_logical(
    edge_shardings: list[Any] | tuple[Any, ...],
    producer_logical: int,
) -> Any:
    """Look up the marker edge sharding declared by ``producer_logical``.

    ``edge_shardings[i]`` is the ``PartitionSpec`` (or ``None``) carried
    by the :func:`sxstage_iter` marker that ends logical stage ``i``.
    Out-of-range indices return ``None`` so the caller can fall through
    to the destination rank's default sharding.

    Args:
        edge_shardings: Per-logical-stage edge specs (ordered to match
            logical stage indices).
        producer_logical: Index of the producing logical stage.

    Returns:
        The marker's edge spec or ``None``.
    """
    if 0 <= producer_logical < len(edge_shardings):
        return edge_shardings[producer_logical]
    return None


def _transfer_target_for_edge(
    value: Any,
    *,
    producer_logical: int,
    dst_rank: int,
    edge_shardings: list[Any] | tuple[Any, ...],
    stage_shardings: list[Any],
    rank_submeshes: list[Any],
    mpmd_mesh: MpMdMesh,
) -> Any:
    """Compute the destination sharding for a cross-stage activation or cotangent.

    Combines :func:`_edge_sharding_for_logical` with
    :func:`_edge_transfer_sharding`: looks up the producing stage's
    marker edge sharding and applies it on the destination rank's
    sub-mesh, falling back to the replicated rank sharding when the
    edge is unannotated.

    Args:
        value: The transported array / pytree (used to drive
            per-leaf spec sanitisation).
        producer_logical: Logical stage index that produced ``value``.
        dst_rank: Destination physical rank.
        edge_shardings: Per-logical-stage marker edge specs.
        stage_shardings: Per-rank replicated shardings.
        rank_submeshes: Per-rank sub-meshes.
        mpmd_mesh: Global MPMD mesh.

    Returns:
        The sharding (or pytree of shardings) usable with
        :func:`jax.device_put`.
    """
    edge_sharding = _edge_sharding_for_logical(edge_shardings, producer_logical)
    return _edge_transfer_sharding(
        value,
        edge_sharding=edge_sharding,
        fallback_sharding=stage_shardings[dst_rank],
        dst_rank=dst_rank,
        rank_submeshes=rank_submeshes,
        mpmd_mesh=mpmd_mesh,
    )


def _last_use_table(grid: list[list[Any]]) -> dict[tuple[int, int], int]:
    """Compute the last-use time step for each ``(stage, microbatch)``.

    ``last_use[(s, mb)] = t`` means the stage-``s`` activation for
    microbatch ``mb`` is read (as saved_inputs / saved_outputs /
    g_y_cache) for the last time at time step ``t``. Runtimes can
    use this table to free the corresponding buffer afterwards.
    """
    last: dict[tuple[int, int], int] = {}
    for t, row in enumerate(grid):
        for s, action in enumerate(row):
            if action is None:
                continue
            key = (s, action.microbatch)
            last[key] = t
    return last


def _delete_if_possible(x: Any) -> None:
    """Free an array's device buffer if JAX allows it.

    :class:`jax.Array` exposes a ``.delete()`` method on newer JAX
    versions; older versions / non-committed arrays silently ignore.
    Exceptions are swallowed so the schedule loop never crashes on a
    donation miss.
    """
    try:
        delete = getattr(x, "delete", None)
        if callable(delete):
            delete()
    except Exception:
        pass


def _is_leaf(x: Any) -> bool:
    """Stop pytree traversal at JAX arrays and Spectrax :class:`Variable` nodes.

    Used as the ``is_leaf`` argument throughout the MPMD runtime so
    that :class:`Variable` containers (which are themselves pytrees of
    metadata + array) are kept whole — otherwise their internal
    metadata leaks out as separate flat-leaf entries and breaks the
    flat-arg <-> outer-jaxpr-invar correspondence.

    Args:
        x: Any pytree node.

    Returns:
        ``True`` when ``x`` is a :class:`jax.Array` or
        :class:`Variable`.
    """
    return isinstance(x, jax.Array | Variable)


def _is_float0(x: Any) -> bool:
    """Return ``True`` when ``x`` carries the JAX ``float0`` zero-sized sentinel.

    JAX uses ``float0`` to mark cotangents of integer-valued primals
    (produced when ``allow_int=True`` is passed to autodiff). These
    leaves cannot participate in arithmetic; the runtime must short
    them out before scaling or addition.

    Args:
        x: Any pytree leaf.

    Returns:
        ``True`` iff ``x.dtype == jax.dtypes.float0``.
    """
    return getattr(x, "dtype", None) == jax.dtypes.float0


def _scale_grad(x: Any, scale: Any) -> Any:
    """Multiply ``x`` by ``scale`` unless ``x`` is a ``float0`` sentinel.

    ``float0`` leaves are returned unchanged so the resulting pytree
    can still be passed back through JAX's autodiff plumbing.

    Args:
        x: Cotangent leaf.
        scale: Scalar multiplier.

    Returns:
        ``x * scale`` for normal arrays, ``x`` for ``float0``.
    """
    if _is_float0(x):
        return x
    return x * scale


def _add_grad(a: Any, b: Any) -> Any:
    """Add two cotangent leaves treating ``float0`` as the additive identity.

    When either operand is ``float0`` the other is returned untouched.
    Mirrors JAX's autodiff convention so accumulating grads from
    integer-input branches does not raise.

    Args:
        a: First cotangent leaf.
        b: Second cotangent leaf.

    Returns:
        ``a + b`` (or whichever operand is non-``float0``).
    """
    if _is_float0(a):
        return b
    if _is_float0(b):
        return a
    return a + b


def _cast_cotangent_like(cotangent: Any, primal: Any) -> Any:
    """Cast ``cotangent`` to its matching ``primal`` dtype before transport.

    Some XLA backends complain when a cotangent's dtype differs from
    the producer's output dtype; casting here keeps the transport
    well-typed without forcing the upstream backward jit to widen its
    grads. ``float0`` cotangents are returned untouched.

    Args:
        cotangent: The incoming cotangent array.
        primal: The forward output whose dtype defines the target.

    Returns:
        ``cotangent`` cast to ``primal.dtype`` (or unchanged when
        already matching, when one side has no dtype, or when the
        cotangent is ``float0``).
    """
    if _is_float0(cotangent):
        return cotangent
    cot_dtype = getattr(cotangent, "dtype", None)
    primal_dtype = getattr(primal, "dtype", None)
    if cot_dtype is not None and primal_dtype is not None and cot_dtype != primal_dtype and hasattr(cotangent, "astype"):
        return cotangent.astype(primal_dtype)
    return cotangent


def _microbatch(x: jax.Array, m: int) -> jax.Array:
    """Reshape a ``(B, ...)`` array into ``(m, B // m, ...)`` microbatches.

    The leading batch axis is split into ``m`` microbatches of equal
    size; ``B`` must be evenly divisible by ``m``.

    Args:
        x: Input array with leading batch dimension.
        m: Number of microbatches.

    Returns:
        A new array shaped ``(m, B // m, *x.shape[1:])``.

    Raises:
        ValueError: If ``B`` is not a multiple of ``m``.
    """
    b = x.shape[0]
    if b % m:
        raise ValueError(f"Batch size {b} not divisible by number of microbatches {m}.")
    return x.reshape(m, b // m, *x.shape[1:])


def _split_params_rest(state: State) -> tuple[State, State]:
    """Partition a :class:`State` into differentiable params and the remainder.

    The MPMD runtime treats the ``"parameters"`` collection as the
    grad-bearing portion and everything else (e.g. RNG state, batch
    norm running stats) as ``rest``. Splitting up front lets each
    stage's forward / backward jits accept ``params`` as the gradient
    target without touching the rest.

    Args:
        state: The full module state.

    Returns:
        ``(params_state, rest_state)`` — each a :class:`State` with
        the corresponding subset of collections.
    """
    raw = state.raw()
    params_raw: dict[str, dict[str, Any]] = {}
    rest_raw: dict[str, dict[str, Any]] = {}
    for c, d in raw.items():
        (params_raw if c == "parameters" else rest_raw)[c] = dict(d)
    return State(params_raw), State(rest_raw)


def _get_fused_fwd_bwd_jit(
    fwd_jit: Callable[..., Any],
    bwd_jit: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a cached jit that performs a ``(fwd_A, bwd_B)`` pair in one dispatch.

    For 1F1B-family schedules at steady state each rank alternates one
    forward (microbatch ``A``) and one backward (microbatch ``B``).
    Dispatching each as its own jit pays two trace/dispatch costs per
    microbatch; fusing them into one compiled kernel halves that cost
    and lets XLA interleave their HLO for better register reuse.

    The fused jit's signature is::

        (params, rest, x_fwd, x_bwd, g_y_bwd)
            -> (y_fwd, g_params_bwd, g_x_bwd)

    ``x_bwd`` is the saved activation for mb ``B`` (captured during
    its earlier forward); the runtime still manages saved_inputs /
    recv_cots / grad_accum the same way — only the dispatch count
    changes.
    """
    key = (id(fwd_jit), id(bwd_jit))
    cached = _FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fused(params, rest, x_fwd, x_bwd, g_y_bwd):
        """Run forward on ``x_fwd`` and backward on ``(x_bwd, g_y_bwd)`` in one HLO."""
        y_fwd = fwd_jit(params, rest, x_fwd)
        g_params, g_x = bwd_jit(params, rest, x_bwd, g_y_bwd)
        return y_fwd, g_params, g_x

    _FUSED_FWDBWD_CACHE[key] = fused
    weak_invalidate(fwd_jit, _FUSED_FWDBWD_CACHE, key)
    weak_invalidate(bwd_jit, _FUSED_FWDBWD_CACHE, key)
    return fused


def _get_vmap_loss_and_g_y(
    loss_fn: Callable[..., jax.Array],
    donate_argnums: tuple[int, ...] = (),
) -> Callable[..., Any]:
    """Return a cached jit that vmaps ``loss_fn`` + ``d_loss/d_y`` over microbatches.

    The wrapper takes ``y_stack`` of shape ``(M, ...)`` plus matching
    target stacks and returns ``(loss_stack, g_y_stack)``. Used by the
    GPipe vmap fast-path to compute every microbatch's loss/cotangent
    in a single device-side launch. Cached on
    ``(id(loss_fn), donate_argnums)`` in :data:`_VMAP_LOSS_CACHE`.

    Args:
        loss_fn: User loss callable ``(y, *targets) -> scalar``.
        donate_argnums: Argnums whose buffers may be donated.

    Returns:
        Jitted ``(y_stack, *t_stack) -> (loss_stack, g_y_stack)``.
    """
    key = (id(loss_fn), donate_argnums)
    cached = _VMAP_LOSS_CACHE.get(key)
    if cached is not None:
        return cached

    if donate_argnums:

        @functools.partial(jax.jit, donate_argnums=donate_argnums)
        def vmap_loss(y_stack, *t_stack):
            """Vmap ``per_mb`` over the leading microbatch axis under one jit.

            Returns ``(loss_stack, g_y_stack)`` with leading axis ``M``;
            both per-mb losses and per-mb cotangents are produced in
            one compiled program so the GPipe fast-path can fuse the
            terminal forward, loss, and backward.
            """

            def per_mb(y_, *t_):
                """Compute ``(loss, d_loss/d_y)`` for a single microbatch slice.

                Wrapped in :func:`jax.vmap` upstream so this body sees
                one microbatch at a time even though the input tensors
                are the full ``(M, ...)`` stacks.
                """
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            return jax.vmap(per_mb)(y_stack, *t_stack)

    else:

        @jax.jit
        def vmap_loss(y_stack, *t_stack):
            """Vmap ``per_mb`` over the leading microbatch axis under one jit.

            Returns ``(loss_stack, g_y_stack)`` with leading axis ``M``;
            both per-mb losses and per-mb cotangents are produced in
            one compiled program so the GPipe fast-path can fuse the
            terminal forward, loss, and backward.
            """

            def per_mb(y_, *t_):
                """Compute ``(loss, d_loss/d_y)`` for a single microbatch slice.

                Wrapped in :func:`jax.vmap` upstream so this body sees
                one microbatch at a time even though the input tensors
                are the full ``(M, ...)`` stacks.
                """
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            return jax.vmap(per_mb)(y_stack, *t_stack)

    _VMAP_LOSS_CACHE[key] = vmap_loss
    weak_invalidate(loss_fn, _VMAP_LOSS_CACHE, key)
    return vmap_loss


@jax.jit
def _vmap_sum_grads(g_stack):
    """Sum a per-microbatch gradient stack along its leading axis.

    The GPipe vmap fast-path produces gradients shaped
    ``(M, *param_shape)`` because each microbatch contributes
    independently. Summing along axis 0 collapses the stack into the
    same parameter shape as a serial accumulation would yield.

    Args:
        g_stack: Pytree of arrays whose leading axis indexes
            microbatches.

    Returns:
        Pytree of arrays with the leading axis summed away.
    """
    return jax.tree.map(lambda x: x.sum(axis=0), g_stack, is_leaf=_is_leaf)


@jax.jit
def _accumulate_state(acc, add):
    """Module-level cached grad accumulator: ``acc + add`` leaf-wise.

    Defined at module scope so JAX's trace cache hits across every
    ``sxcall`` call and across every stage with matching pytree
    shape — eliminates the per-call re-trace cost that previously
    dominated step time at small batch sizes.
    """
    return jax.tree.map(lambda a, b: a + b, acc, add, is_leaf=_is_leaf)


@jax.jit
def _accumulate_grad_tree(acc, add):
    """Add two grad pytrees leaf-wise under a cached jit, preserving ``float0``.

    Module-scope so JAX's trace cache reuses the compiled HLO across
    every :func:`sxcall` invocation that handles the same param tree
    shape.

    Args:
        acc: Running gradient accumulator pytree.
        add: New gradient contribution to fold in.

    Returns:
        ``acc + add`` leaf-wise, with ``float0`` leaves treated as
        additive zero.
    """
    return jax.tree.map(_add_grad, acc, add, is_leaf=_is_leaf)


@jax.jit
def _scale_grad_tree(state, scalar):
    """Scale every leaf of a grad pytree by ``scalar`` under a cached jit.

    Companion to :func:`_accumulate_grad_tree`; ``float0`` leaves
    pass through unchanged so integer-input branches stay valid.

    Args:
        state: Pytree of grad leaves.
        scalar: Multiplier (typically ``1/M``).

    Returns:
        Grad pytree with each leaf scaled.
    """
    return jax.tree.map(lambda x: _scale_grad(x, scalar), state, is_leaf=_is_leaf)


@jax.jit
def _zeros_like_state(state):
    """Module-level cached ``zeros_like`` over a State pytree.

    Replaces the per-call ``jax.tree.map(jnp.zeros_like, sp)`` which
    issued one eager dispatch per parameter leaf — ~0.3 ms each on
    TPU, easily 60+ ms per step on medium models.
    """
    return jax.tree.map(jnp.zeros_like, state, is_leaf=_is_leaf)


@jax.jit
def _scale_state(state, scalar):
    """Multiply every array leaf of ``state`` by ``scalar`` under one jit.

    Used to apply the ``1/M`` mean-loss / mean-grad scaling. Defined at
    module scope so JAX's trace cache hits across every :func:`sxcall`
    invocation with the same state pytree shape.

    Args:
        state: Pytree of arrays (or :class:`State` /
            :class:`Variable`-leaved tree).
        scalar: Multiplicative factor (typically ``1.0 / M``).

    Returns:
        Same pytree structure with every array leaf scaled.
    """
    return jax.tree.map(lambda g: g * scalar, state, is_leaf=_is_leaf)


def _get_loss_and_g_y(
    loss_fn: Callable[..., jax.Array],
    has_aux: bool = False,
    donate_argnums: tuple[int, ...] = (),
) -> Callable[..., Any]:
    """Return a jitted ``(y, *targets) -> (loss, grad_wrt_y, [aux])`` for ``loss_fn``.

    When ``has_aux=True``, ``loss_fn`` must return ``(scalar, aux_pytree)``.
    The returned wrapper yields ``(loss, g_y, aux)`` so the caller can
    accumulate aux across microbatches.

    Cached on ``(id(loss_fn), has_aux, donate_argnums)``.
    """
    key = (id(loss_fn), has_aux, donate_argnums)
    cached = _LOSS_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    if has_aux:
        if donate_argnums:

            @functools.partial(jax.jit, donate_argnums=donate_argnums)
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y, aux)`` for an aux-returning loss.

                The auxiliary pytree is passed through unchanged so the
                caller can accumulate it across microbatches without
                running a second pass through the loss.
                """

                def local_loss(y_):
                    """Loss closure used by :func:`jax.value_and_grad`; returns ``(scalar, aux)``."""
                    return loss_fn(y_, *targets)

                (loss_val, aux), g_y = jax.value_and_grad(local_loss, has_aux=True)(y)
                return loss_val, g_y, aux

        else:

            @jax.jit
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y, aux)`` for an aux-returning loss.

                The auxiliary pytree is passed through unchanged so the
                caller can accumulate it across microbatches without
                running a second pass through the loss.
                """

                def local_loss(y_):
                    """Loss closure used by :func:`jax.value_and_grad`; returns ``(scalar, aux)``."""
                    return loss_fn(y_, *targets)

                (loss_val, aux), g_y = jax.value_and_grad(local_loss, has_aux=True)(y)
                return loss_val, g_y, aux

    else:
        if donate_argnums:

            @functools.partial(jax.jit, donate_argnums=donate_argnums)
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y)`` for a plain scalar-loss callable.

                ``targets`` are bound at call time; the returned grad
                is taken with respect to ``y`` only.
                """

                def local_loss(y_):
                    """Scalar loss closure passed to :func:`jax.value_and_grad`."""
                    return loss_fn(y_, *targets)

                return jax.value_and_grad(local_loss)(y)

        else:

            @jax.jit
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y)`` for a plain scalar-loss callable.

                ``targets`` are bound at call time; the returned grad
                is taken with respect to ``y`` only.
                """

                def local_loss(y_):
                    """Scalar loss closure passed to :func:`jax.value_and_grad`."""
                    return loss_fn(y_, *targets)

                return jax.value_and_grad(local_loss)(y)

    _LOSS_JIT_CACHE[key] = loss_and_g_y
    weak_invalidate(loss_fn, _LOSS_JIT_CACHE, key)
    return loss_and_g_y


def _build_stage_callables(
    stage: Module,
    donate_fwd: tuple[int, ...] = (),
    donate_bwd: tuple[int, ...] = (),
) -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    State,
    State,
    Any,
]:
    """Compile forward and unified-backward functions for a stage.

    Two jits per stage:

    * ``fwd_only(params, rest, x) -> y`` — the forward pass.
    * ``bwd_only(params, rest, x, g_y) -> (g_params, g_x)`` — the full
      VJP via :func:`jax.vjp` (``linearize`` is avoided because it
      fails on integer-valued inputs such as token-id embeddings).
      For :class:`~spectrax.runtime.schedules.ZeroBubbleH1`, both
      :attr:`Phase.BWD_I` and :attr:`Phase.BWD_W` call this same jit
      but discard one of the two outputs; XLA's dead-code elimination
      collapses each half-call to roughly half the work of the full
      backward.

    Compared to an earlier implementation with three separate VJP
    jits (``bwd``, ``bwd_i``, ``bwd_w``), this trims tracing cost by
    eliminating the redundant re-tracings of ``stage_fn``.

    ``rest`` and ``gdef`` are passed as explicit arguments to the jit
    (not closure-captured) so JAX's trace cache keys on ``(arg avals)``
    alone — every subsequent call with the same shape signature hits
    the cache, even across distinct ``sxcall`` invocations.

    Returns:
        ``(fwd_only, bwd_only, params, rest, gdef)`` — ``parameters`` /
        ``rest`` are the initial state split; ``gdef`` is the stage's
        :class:`GraphDef` (retained for potential reuse).
    """
    gdef, state = export(stage)
    params, rest = _split_params_rest(state)

    n_leaves = len(jax.tree.leaves(params))
    cache_key = (id(stage), donate_fwd, donate_bwd)
    cached = _STAGE_CALLABLE_CACHE.get(cache_key)
    if cached is not None:
        cached_fwd, cached_bwd, cached_n_leaves = cached
        if cached_n_leaves == n_leaves:
            return cached_fwd, cached_bwd, params, rest, gdef
        del _STAGE_CALLABLE_CACHE[cache_key]

    if donate_fwd:

        @functools.partial(jax.jit, donate_argnums=donate_fwd)
        def fwd_only(params, rest, x):
            """Run a single stage forward by re-binding ``(params, rest)`` into ``gdef``.

            Args:
                params: The differentiable parameter :class:`State`
                    placed on this rank.
                rest: Non-parameter state (overlaid on ``params``).
                x: Stage input activation.

            Returns:
                The stage's output activation.
            """
            module = bind(gdef, params.overlay(rest))
            return module(x)

    else:

        @jax.jit
        def fwd_only(params, rest, x):
            """Run a single stage forward by re-binding ``(params, rest)`` into ``gdef``.

            Args:
                params: The differentiable parameter :class:`State`
                    placed on this rank.
                rest: Non-parameter state (overlaid on ``params``).
                x: Stage input activation.

            Returns:
                The stage's output activation.
            """
            module = bind(gdef, params.overlay(rest))
            return module(x)

    if donate_bwd:

        @functools.partial(jax.jit, donate_argnums=donate_bwd)
        def bwd_only(params, rest, x, g_y):
            """(g_params, g_x) via :func:`jax.vjp`.

            Uses ``vjp`` instead of ``linearize + linear_transpose`` because
            ``linearize`` fails on stages whose inputs contain integers
            (e.g. an embedding layer taking token IDs). ``vjp`` handles the
            int-to-float boundary correctly.
            """

            def stage_fn(p, r, xi):
                """Pure forward closure used as the :func:`jax.vjp` target.

                Re-binds the stage from ``gdef`` on every call so the
                VJP can differentiate through fresh leaves rather than
                the captured originals (necessary because :func:`vjp`
                tracks the identity of its inputs).
                """
                return bind(gdef, p.overlay(r))(xi)

            _y, vjp_fn = jax.vjp(stage_fn, params, rest, x)
            g_params, _g_rest, g_x = vjp_fn(g_y)
            return g_params, g_x

    else:

        @jax.jit
        def bwd_only(params, rest, x, g_y):
            """(g_params, g_x) via :func:`jax.vjp`.

            Uses ``vjp`` instead of ``linearize + linear_transpose`` because
            ``linearize`` fails on stages whose inputs contain integers
            (e.g. an embedding layer taking token IDs). ``vjp`` handles the
            int-to-float boundary correctly.
            """

            def stage_fn(p, r, xi):
                """Pure forward closure used as the :func:`jax.vjp` target.

                Re-binds the stage from ``gdef`` on every call so the
                VJP can differentiate through fresh leaves rather than
                the captured originals (necessary because :func:`vjp`
                tracks the identity of its inputs).
                """
                return bind(gdef, p.overlay(r))(xi)

            _y, vjp_fn = jax.vjp(stage_fn, params, rest, x)
            g_params, _g_rest, g_x = vjp_fn(g_y)
            return g_params, g_x

    _STAGE_CALLABLE_CACHE[cache_key] = (fwd_only, bwd_only, n_leaves)
    weak_invalidate(stage, _STAGE_CALLABLE_CACHE, cache_key)
    return fwd_only, bwd_only, params, rest, gdef


def _normalize_target(
    target: "PipelineSequential | Module",
    n_logical: int,
) -> PipelineSequential:
    """Coerce ``target`` into a :class:`PipelineSequential` of ``n_logical`` stages.

    Accepts either a pre-built :class:`PipelineSequential` (returned
    unchanged) or a bare :class:`~spectrax.Module` (auto-split via
    :func:`auto_split` and wrapped in :class:`PipelineSequential`).

    The auto-split result is cached on ``(id(target), n_logical)`` so
    repeat calls with the same model object reuse the same stage objects
    — stable ``id`` for those stages keeps the downstream jit and
    placement caches hot.
    """
    if isinstance(target, PipelineSequential):
        return target
    if isinstance(target, Module):
        cache_key = (id(target), n_logical)
        cached = _MPMD_CALL_NORMALIZED_CACHE.get(cache_key)
        if cached is not None:
            return cached
        stage_modules = auto_split(target, n_logical)
        seq = PipelineSequential(*stage_modules)
        _MPMD_CALL_NORMALIZED_CACHE[cache_key] = seq
        weak_invalidate(target, _MPMD_CALL_NORMALIZED_CACHE, cache_key)
        return seq
    raise TypeError(f"sxcall target must be a PipelineSequential or a Module, got {type(target).__name__}.")


def sxcall(
    target: "PipelineSequential | Module",
    batch: tuple[Any, ...],
    *,
    mesh: SpxMesh | MpMdMesh,
    schedule: Schedule,
    loss_fn: Callable[..., jax.Array] | None = None,
    transport: TransportKind = "device_put",
    donate_activations: bool = False,
    static_argnums: int | tuple[int, ...] | None = None,
    donate_argnums: int | tuple[int, ...] | None = None,
    chunks: int | None = None,
    fuse_1f1b: bool = False,
    fuse_zb: bool = False,
    mode: Literal["train", "forward"] = "train",
    has_aux: bool = False,
) -> tuple[jax.Array, StagesArray] | tuple[jax.Array, StagesArray, Any] | jax.Array:
    """Execute one pipeline-parallel step with heterogeneous stages.

    Each stage runs on its own sub-mesh of ``mpmd_mesh``; the schedule
    loop is driven in Python with :func:`jax.device_put` handling
    cross-stage transfers of activations (forward) and cotangents
    (backward).

    Args:
        target: :class:`PipelineSequential` whose stages can have
            **different shapes, classes, or parameter structures** —
            no same-GraphDef constraint. A bare :class:`Module` is also
            accepted and is auto-split via :func:`auto_split` into
            ``V * mpmd_dim`` logical stages.
        batch: Tuple of inputs. First element is the pipeline input;
            remaining elements are targets / aux args forwarded to
            ``loss_fn`` on the final stage.
        mesh: A :class:`SpxMesh` or :class:`MpMdMesh` whose
            ``mpmd_dim`` equals the number of physical pipeline ranks.
            Non-MPMD axes (if any) are available for intra-stage SPMD
            sharding; activations land replicated across them.
        schedule: A :class:`Schedule` (``GPipe``, ``Std1F1B``,
            ``ZeroBubbleH1``, ``InterleavedH1``, ``Eager1F1B``, ...).
        loss_fn: ``(final_stage_output, *batch[1:]) -> scalar``. Required
            when ``mode='train'``.
        transport: Cross-stage copy mechanism. Only ``"device_put"``
            (default) is supported today; it uses :func:`jax.device_put`
            and is portable across CPU / TPU / GPU backends.
        donate_activations: When ``True``, free the device buffer
            for each saved activation / cotangent / ``g_y`` cache
            entry as soon as the schedule's last-use time step passes.
            Reduces peak memory at the cost of losing the arrays for
            any post-hoc inspection. Defaults to ``False``.
        static_argnums: Which elements of ``batch`` are static (compile-time
            constants). Static batch elements are not microbatched and are
            passed directly to ``loss_fn``. Index 0 (pipeline input) cannot
            be static.
        donate_argnums: Which elements of ``batch`` should have their device
            buffers donated. Target elements (indices >= 1) are donated to
            ``loss_fn``. The pipeline input (index 0) can only be donated in
            ``mode='forward'``.

    Returns:
        For ``mode='train'``: ``(loss, per_stage_param_grads)`` (or
        ``(loss, per_stage_param_grads, aux)`` when ``has_aux=True``):
        mean loss over all microbatches and a :class:`StagesArray`
        whose shards are the per-logical-stage :class:`State` s
        carrying the ``parameters`` gradients. Gradients are resident
        on the stage's sub-mesh.

        For ``mode='forward'``: a single :class:`jax.Array` with the
        terminal-stage activations stitched back into batch order.

    The full setup (placed parameters/rest + fwd/bwd jits per
    ``(rank, virt)``) is cached in ``_MPMD_SETUP_CACHE`` by
    ``(id(model), id(mpmd_mesh), V)`` so repeat calls skip the ~40
    ``jax.device_put`` calls that placement re-runs would do. The
    two-pass execution within each time step (FWDs in ascending
    logical order, then BWDs in descending logical order) respects
    data dependencies while letting unrelated stages dispatch
    concurrently. Grads are returned in LOGICAL order so indices line
    up with the input ``PipelineSequential`` — flat schedules produce
    ``n`` grads, virtual schedules produce ``V*n``.

    Per-rank sub-mesh context is entered around each stage's jit so
    any ``with_sharding_constraint`` inside the stage resolves named
    axes against THIS rank's device set (not the full PP x TP mesh).
    Per-leaf shardings come from the model's logical axis annotations
    via ``get_named_sharding``; leaves without a registered sharding
    fall back to the rank's replicated sharding (legacy single-axis
    behavior). Resolution of logical -> physical axis names uses
    whatever :func:`logical_axis_rules` context is active at the call
    site.

    Example::

        from jax.sharding import Mesh
        from spectrax.runtime.schedules import Std1F1B
        from spectrax.runtime.types import MpMdMesh
        from spectrax.nn import PipelineSequential
        from spectrax.runtime.mpmd import sxcall

        devices = np.array(jax.devices()[:8]).reshape(4, 2)
        mm = MpMdMesh(Mesh(devices, ("pp", "fsdp")), "pp")
        model = PipelineSequential(
            EmbedStage(vocab=50_000, d=512, rngs=rngs),
            BlockStage(d=512, rngs=rngs),
            BlockStage(d=512, rngs=rngs),
            HeadStage(d=512, vocab=50_000, rngs=rngs),
        )
        loss, grads = sxcall(
            model, (ids, targets),
            mesh=mm,
            schedule=Std1F1B(microbatches=8),
            loss_fn=softmax_xent,
        )
    """
    mpmd_mesh = resolve_mpmd_mesh(mesh)
    if mode not in {"forward", "train"}:
        raise ValueError(f"sxcall mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "train" and loss_fn is None:
        raise ValueError("sxcall with mode='train' requires loss_fn.")
    n = mpmd_mesh.mpmd_dim
    m = schedule.microbatches

    V = schedule.virtual_stages_per_rank()
    n_logical = V * n

    model = _normalize_target(target, n_logical)
    stages = model.stages

    if len(stages) != n_logical:
        raise ValueError(
            f"{type(schedule).__name__} with virtual_stages={V} needs a "
            f"PipelineSequential of {n_logical} logical stages "
            f"({V} per rank x {n} ranks); got {len(stages)}. "
            f"Build the model with stages in logical order — the runtime "
            f"routes each to its (rank, virt) slot via "
            f"``schedule.logical_at``."
        )

    static_nums = set(_normalize_argnums(static_argnums, len(batch)))
    donate_nums = set(_normalize_argnums(donate_argnums, len(batch)))

    if 0 in static_nums:
        raise ValueError(
            "sxcall: batch[0] (pipeline input) cannot be static. "
            "static_argnums must refer to target/aux arguments (indices >= 1)."
        )
    if 0 in donate_nums and mode == "train":
        raise ValueError(
            "sxcall: cannot donate batch[0] (pipeline input) in train mode. "
            "Use mode='forward' or remove 0 from donate_argnums."
        )

    donate_fwd = (2,) if (0 in donate_nums and mode == "forward") else ()
    donate_bwd = ()
    loss_donate = tuple(i for i in donate_nums if i > 0)

    setup_key = (id(model), id(mpmd_mesh), V, type(schedule).__name__, donate_fwd, donate_bwd)
    cached_setup = _MPMD_SETUP_CACHE.get(setup_key)
    if cached_setup is not None:
        (fwd_jits, bwd_jits, stage_params, stage_rest, stage_shardings, rank_submeshes) = cached_setup
        _setup_done = True
    else:
        _setup_done = False

    stage_shardings = [mpmd_mesh.sub_sharding(i) for i in range(n)] if not _setup_done else stage_shardings
    rank_submeshes = [mpmd_mesh.submesh(i) for i in range(n)] if not _setup_done else rank_submeshes

    def _place_state(state: State, rank: int, stage: Module) -> State:
        """Apply per-leaf shardings derived from the stage's logical-axis metadata.

        Leaves without a registered sharding fall back to the rank's
        replicated sharding (legacy single-axis behavior). Resolution
        of logical -> physical axis names uses whatever
        :func:`logical_axis_rules` context is active at the call site.
        """
        per_leaf = get_named_sharding(stage, rank_submeshes[rank])
        replicated = stage_shardings[rank]
        out: dict[str, dict[str, Any]] = {}
        for col, path, leaf in state.items():
            sh = per_leaf.get(col, {}).get(path, replicated)
            out.setdefault(col, {})[path] = jax.device_put(leaf, sh)
        return type(state)(out)

    if not _setup_done:
        fwd_jits = {}
        bwd_jits = {}
        stage_params = {}
        stage_rest = {}
        for rank in range(n):
            for virt in range(V):
                logical = schedule.logical_at(rank, virt, n)
                stage = stages[logical]
                fwd, bwd, params, rest, _ = _build_stage_callables(stage, donate_fwd=donate_fwd, donate_bwd=donate_bwd)
                fwd_jits[(rank, virt)] = fwd
                bwd_jits[(rank, virt)] = bwd
                stage_params[(rank, virt)] = _place_state(params, rank, stage)
                stage_rest[(rank, virt)] = _place_state(rest, rank, stage)
        _MPMD_SETUP_CACHE[setup_key] = (
            fwd_jits,
            bwd_jits,
            stage_params,
            stage_rest,
            stage_shardings,
            rank_submeshes,
        )
        weak_invalidate(model, _MPMD_SETUP_CACHE, setup_key)
        weak_invalidate(mpmd_mesh, _MPMD_SETUP_CACHE, setup_key)

    mb_batch = []
    for i, x in enumerate(batch):
        if i in static_nums:
            mb_batch.append(x)
        else:
            mb_batch.append(_microbatch(x, m))
    xs = mb_batch[0]
    target_args = mb_batch[1:]
    static_target_mask = [i in static_nums for i in range(1, len(batch))]

    is_forward_only = mode == "forward"

    if not is_forward_only and loss_fn is None:
        raise ValueError("sxcall with mode='train' requires loss_fn.")

    if is_forward_only:
        return _forward_only_run(
            n=n,
            V=V,
            m=m,
            schedule=schedule,
            fwd_jits=fwd_jits,
            stage_params=stage_params,
            stage_rest=stage_rest,
            stage_shardings=stage_shardings,
            rank_submeshes=rank_submeshes,
            xs=xs,
        )

    loss_and_g_y = (
        _get_loss_and_g_y(loss_fn, has_aux=has_aux, donate_argnums=loss_donate) if not is_forward_only else None
    )
    aux_accum: list[Any] = []

    saved_inputs: dict[tuple[int, int], dict[int, Any]] = {k: {} for k in fwd_jits}
    saved_outputs: dict[tuple[int, int], dict[int, Any]] = {k: {} for k in fwd_jits}
    recv_cots: dict[tuple[int, int], dict[int, Any]] = {k: {} for k in fwd_jits}
    g_y_cache: dict[tuple[int, int], dict[int, Any]] = {k: {} for k in fwd_jits}
    grad_accum: dict[tuple[int, int], State] = {k: _zeros_like_state(v) for k, v in stage_params.items()}
    loss_acc: jax.Array = jnp.asarray(0.0)

    terminal_rank, terminal_virt = schedule.terminal_loc(n)

    if isinstance(schedule, GPipe) and V == 1 and not static_nums and not donate_nums:
        return _gpipe_run(
            n=n,
            m=m,
            fwd_jits=fwd_jits,
            bwd_jits=bwd_jits,
            stage_params=stage_params,
            stage_rest=stage_rest,
            stage_shardings=stage_shardings,
            rank_submeshes=rank_submeshes,
            xs=xs,
            target_args=target_args,
            loss_fn=loss_fn,
            transport_kind=transport,
            chunks=chunks,
        )

    def _transport_to(src_loc, dst_loc, arr, task_name=None):
        """Move ``arr`` to the device hosting ``dst_loc``.

        Within the same rank (virtual-stage shift on the same device)
        no transfer is needed — just return the array.
        """
        if src_loc[0] == dst_loc[0]:
            return arr
        return _transport(transport, arr, stage_shardings[dst_loc[0]], task_name=task_name)

    grid: list[list[Any]] = [list(row) for row in schedule.build(n)]
    if fuse_1f1b:
        grid = fuse_1f1b_steady_state(grid)
    if fuse_zb:
        grid = fuse_zerobubble_bwd_pair(grid)

    def _expand_cell(cell: Any) -> list[Any]:
        """Expand a grid cell into one or more dispatch units.

        A plain :class:`Action` or :class:`FusedTask` with an
        unsupported phase combo returns as its component
        :class:`Action` s (runtime falls back to per-action dispatch).
        A :class:`FusedTask(FWD, BWD)` returns as a single element
        so the loop fires one fused jit.
        """
        if cell is None:
            return []
        if isinstance(cell, FusedTask):
            if cell.fwd.phase == Phase.FWD and cell.bwd.phase == Phase.BWD:
                return [cell]
            return [cell.fwd, cell.bwd]
        return [cell]

    for t, row in enumerate(grid):
        fwd_acts: list[tuple[int, int, Any]] = []
        bwd_acts: list[tuple[int, int, Any]] = []
        fused_acts: list[tuple[int, int, FusedTask]] = []
        for rank, cell in enumerate(row):
            for unit in _expand_cell(cell):
                if isinstance(unit, FusedTask):
                    fused_acts.append((rank, unit.virtual_stage, unit))
                    continue
                if unit.phase == Phase.FWD:
                    fwd_acts.append((rank, unit.virtual_stage, unit))
                else:
                    bwd_acts.append((rank, unit.virtual_stage, unit))
        fwd_acts.sort(key=lambda t: schedule.logical_at(t[0], t[1], n))
        if not is_forward_only:
            bwd_acts.sort(key=lambda t: -schedule.logical_at(t[0], t[1], n))

        for rank, virt, fused in [] if is_forward_only else fused_acts:
            loc = (rank, virt)
            fwd_act = fused.fwd
            bwd_act = fused.bwd
            fwd_mb = fwd_act.microbatch
            bwd_mb = bwd_act.microbatch
            logical = schedule.logical_at(rank, virt, n)
            next_loc = schedule.next_logical_loc(rank, virt, n)
            with rank_submeshes[rank]:
                if logical == 0:
                    x_fwd = _transport(transport, xs[fwd_mb], stage_shardings[rank])
                else:
                    x_fwd = saved_inputs[loc][fwd_mb]
                x_bwd = saved_inputs[loc][bwd_mb]
                if loc == (terminal_rank, terminal_virt):
                    x_out_bwd = saved_outputs[loc][bwd_mb]
                    targets_mb = tuple(
                        (
                            _transport(transport, t, stage_shardings[rank])
                            if static_target_mask[i]
                            else _transport(transport, t[bwd_mb], stage_shardings[rank])
                        )
                        for i, t in enumerate(target_args)
                    )
                    loss_mb, g_y_bwd = _time_call(
                        f"L{logical}_loss_mb{bwd_mb}",
                        loss_and_g_y,
                        x_out_bwd,
                        *targets_mb,
                    )
                    loss_acc = loss_acc + loss_mb
                else:
                    g_y_bwd = recv_cots[loc][bwd_mb]
                fused_jit = _get_fused_fwd_bwd_jit(fwd_jits[loc], bwd_jits[loc])
                y_fwd, g_params, g_x_bwd = _time_call(
                    f"L{logical}_fused_fwd{fwd_mb}_bwd{bwd_mb}",
                    fused_jit,
                    stage_params[loc],
                    stage_rest[loc],
                    x_fwd,
                    x_bwd,
                    g_y_bwd,
                )
                saved_inputs[loc][fwd_mb] = x_fwd
                saved_outputs[loc][fwd_mb] = y_fwd
                if next_loc is not None:
                    saved_inputs[next_loc][fwd_mb] = _transport_to(
                        loc,
                        next_loc,
                        y_fwd,
                        task_name=f"transfer_fwd_L{logical}_to_L{logical + 1}_mb{fwd_mb}",
                    )
                grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                if logical > 0:
                    prev_loc = _prev_loc(schedule, rank, virt, n)
                    recv_cots[prev_loc][bwd_mb] = _transport_to(
                        loc,
                        prev_loc,
                        g_x_bwd,
                        task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{bwd_mb}",
                    )
                if donate_activations:
                    _delete_if_possible(saved_inputs[loc].pop(bwd_mb, None))
                    _delete_if_possible(saved_outputs[loc].pop(bwd_mb, None))
                    recv_cots[loc].pop(bwd_mb, None)

        actions_to_run = fwd_acts if is_forward_only else (*fwd_acts, *bwd_acts)
        for rank, virt, action in actions_to_run:
            loc = (rank, virt)
            mb = action.microbatch
            logical = schedule.logical_at(rank, virt, n)
            next_loc = schedule.next_logical_loc(rank, virt, n)
            with rank_submeshes[rank]:
                if action.phase == Phase.FWD:
                    if logical == 0:
                        x_in = _transport(transport, xs[mb], stage_shardings[rank])
                    else:
                        x_in = saved_inputs[loc][mb]
                    x_out = _time_call(
                        f"L{logical}_fwd_mb{mb}",
                        fwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                    )
                    saved_inputs[loc][mb] = x_in
                    saved_outputs[loc][mb] = x_out
                    if next_loc is not None:
                        saved_inputs[next_loc][mb] = _transport_to(
                            loc,
                            next_loc,
                            x_out,
                            task_name=f"transfer_fwd_L{logical}_to_L{logical + 1}_mb{mb}",
                        )
                elif action.phase == Phase.BWD:
                    x_in = saved_inputs[loc][mb]
                    if loc == (terminal_rank, terminal_virt):
                        x_out = saved_outputs[loc][mb]
                        targets_mb = tuple(
                            (
                                _transport(transport, t, stage_shardings[rank])
                                if static_target_mask[i]
                                else _transport(transport, t[mb], stage_shardings[rank])
                            )
                            for i, t in enumerate(target_args)
                        )
                        _loss_result = _time_call(
                            f"L{logical}_loss_mb{mb}",
                            loss_and_g_y,
                            x_out,
                            *targets_mb,
                        )
                        if has_aux:
                            loss_mb, g_y, aux_mb = _loss_result
                            aux_accum.append(aux_mb)
                        else:
                            loss_mb, g_y = _loss_result
                        loss_acc = loss_acc + loss_mb
                    else:
                        g_y = recv_cots[loc][mb]
                    g_params, g_x = _time_call(
                        f"L{logical}_bwd_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                    if logical > 0:
                        prev_loc = _prev_loc(schedule, rank, virt, n)
                        recv_cots[prev_loc][mb] = _transport_to(
                            loc,
                            prev_loc,
                            g_x,
                            task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{mb}",
                        )
                    if donate_activations:
                        _delete_if_possible(saved_inputs[loc].pop(mb, None))
                        _delete_if_possible(saved_outputs[loc].pop(mb, None))
                        recv_cots[loc].pop(mb, None)
                elif action.phase == Phase.BWD_I:
                    x_in = saved_inputs[loc][mb]
                    if loc == (terminal_rank, terminal_virt):
                        x_out = saved_outputs[loc][mb]
                        targets_mb = tuple(
                            (
                                _transport(transport, t, stage_shardings[rank])
                                if static_target_mask[i]
                                else _transport(transport, t[mb], stage_shardings[rank])
                            )
                            for i, t in enumerate(target_args)
                        )
                        _loss_result = _time_call(
                            f"L{logical}_loss_mb{mb}",
                            loss_and_g_y,
                            x_out,
                            *targets_mb,
                        )
                        if has_aux:
                            loss_mb, g_y, aux_mb = _loss_result
                            aux_accum.append(aux_mb)
                        else:
                            loss_mb, g_y = _loss_result
                        loss_acc = loss_acc + loss_mb
                    else:
                        g_y = recv_cots[loc][mb]
                    g_y_cache[loc][mb] = g_y
                    _, g_x = _time_call(
                        f"L{logical}_bwd_i_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    if logical > 0:
                        prev_loc = _prev_loc(schedule, rank, virt, n)
                        recv_cots[prev_loc][mb] = _transport_to(
                            loc,
                            prev_loc,
                            g_x,
                            task_name=f"transfer_bwd_L{logical}_to_L{logical - 1}_mb{mb}",
                        )
                    if donate_activations:
                        _delete_if_possible(saved_outputs[loc].pop(mb, None))
                        recv_cots[loc].pop(mb, None)
                elif action.phase == Phase.BWD_W:
                    x_in = saved_inputs[loc][mb]
                    g_y = g_y_cache[loc][mb]
                    g_params, _ = _time_call(
                        f"L{logical}_bwd_w_mb{mb}",
                        bwd_jits[loc],
                        stage_params[loc],
                        stage_rest[loc],
                        x_in,
                        g_y,
                    )
                    grad_accum[loc] = _accumulate_state(grad_accum[loc], g_params)
                    del g_y_cache[loc][mb]
                    if donate_activations:
                        _delete_if_possible(saved_inputs[loc].pop(mb, None))

    if is_forward_only:
        terminal_loc = (terminal_rank, terminal_virt)
        outputs = saved_outputs.get(terminal_loc, {})
        if outputs:
            output_stack = jnp.stack([outputs[mb_i] for mb_i in sorted(outputs.keys())], axis=0)
            return output_stack.reshape(-1, *output_stack.shape[2:])
        return jnp.zeros(())

    mean_loss = loss_acc / jnp.asarray(m, dtype=loss_acc.dtype)
    inv_m = jnp.asarray(1.0 / m, dtype=jnp.float32)
    logical_grads: list[State] = []
    for logical in range(n_logical):
        loc = _loc_for_logical(schedule, logical, n, V)
        logical_grads.append(_scale_state(grad_accum[loc], inv_m))
    grads_out = StagesArray(shards=dict(enumerate(logical_grads)))

    if has_aux and aux_accum:
        mean_aux = jax.tree.map(
            lambda *vals: sum(vals) / len(vals),
            *aux_accum,
        )
        return mean_loss, grads_out, mean_aux
    return mean_loss, grads_out


def _forward_only_run(
    *,
    n: int,
    V: int,
    m: int,
    schedule: Schedule,
    fwd_jits: dict[tuple[int, int], Callable[..., Any]],
    stage_params: dict[tuple[int, int], State],
    stage_rest: dict[tuple[int, int], State],
    stage_shardings: list[Any],
    rank_submeshes: list[Any],
    xs: jax.Array,
) -> jax.Array:
    """Forward-only fast-path for all schedules (flat and virtual-stage).

    Skips all backward machinery — no ``bwd_jits``, no ``loss_fn``, no
    ``grad_accum``, no ``_zeros_like_state``. Follows the schedule's
    logical stage routing via ``logical_at`` / ``next_logical_loc`` to
    handle virtual-stage schedules (KimiK2, DualPipeV) where data
    bounces between physical ranks.
    """

    def _get_vfwd(loc: tuple[int, int]) -> Callable[..., Any]:
        """Return (and cache) a vmapped forward jit for stage ``loc``.

        Wraps the location's ``fwd_jit`` in :func:`jax.vmap` over the
        microbatch axis (axis 0 of the input activation; ``params`` and
        ``rest`` are broadcast). The result is memoised in
        :data:`_FWD_ONLY_VMAP_CACHE` keyed by the underlying jit's
        ``id``.

        Args:
            loc: ``(rank, virt)`` location for the stage.

        Returns:
            A jitted ``(params, rest, x_stack) -> y_stack`` callable.
        """
        key = id(fwd_jits[loc])
        cached = _FWD_ONLY_VMAP_CACHE.get(key)
        if cached is not None:
            return cached
        vfwd = jax.jit(jax.vmap(fwd_jits[loc], in_axes=(None, None, 0)))
        _FWD_ONLY_VMAP_CACHE[key] = vfwd
        weak_invalidate(fwd_jits[loc], _FWD_ONLY_VMAP_CACHE, key)
        return vfwd

    n_logical = V * n

    logical_chain: list[tuple[int, int]] = []
    for logical in range(n_logical):
        for r in range(n):
            for v in range(V):
                if schedule.logical_at(r, v, n) == logical:
                    logical_chain.append((r, v))

    first_rank = logical_chain[0][0]
    x_curr = jax.device_put(xs, stage_shardings[first_rank])
    for i, (rank, virt) in enumerate(logical_chain):
        loc = (rank, virt)
        vfwd = _get_vfwd(loc)
        with rank_submeshes[rank]:
            x_curr = vfwd(stage_params[loc], stage_rest[loc], x_curr)
        if i < n_logical - 1:
            next_rank, _ = logical_chain[i + 1]
            if next_rank != rank:
                x_curr = jax.device_put(x_curr, stage_shardings[next_rank])

    return x_curr.reshape(-1, *x_curr.shape[2:])


def _gpipe_run(
    *,
    n: int,
    m: int,
    fwd_jits: dict[tuple[int, int], Callable[..., Any]],
    bwd_jits: dict[tuple[int, int], Callable[..., Any]],
    stage_params: dict[tuple[int, int], State],
    stage_rest: dict[tuple[int, int], State],
    stage_shardings: list[Any],
    rank_submeshes: list[Any],
    xs: jax.Array,
    target_args: tuple[jax.Array, ...],
    loss_fn: Callable[..., jax.Array],
    transport_kind: TransportKind,
    chunks: int | None = None,
) -> tuple[jax.Array, tuple[State, ...]]:
    """GPipe fast-path: chunked-vmap execution over M microbatches.

    Pipelining semantics under GPipe (all-fwds then all-bwds, no
    interleaving) are preserved exactly: each stage still runs all
    microbatches before the next stage starts. The Python loop just
    issues 1 vmapped dispatch per stage instead of M per-microbatch
    dispatches, slashing dispatch overhead at small/medium configs.

    Compute on TPU is identical (vmap fuses cleanly through dense
    matmuls); only Python+dispatch overhead drops.

    K-chunked execution (real stage overlap): splits M microbatches
    into K chunks (K in {2, m}). Each chunk issues its own vmap per
    stage, and JAX's async dispatch lets rank ``r+1``'s ``chunk_k``
    start as soon as rank ``r``'s ``chunk_k`` output is enqueued —
    while rank ``r`` moves to ``chunk_{k+1}``. vmap-collapse is broken
    at the K-boundary; true cross-stage overlap at 2x (not Mx)
    dispatch cost. Picks K=2 when M >= 2 for the 2-stage sweet spot;
    falls back to no chunking when M=1 (no microbatching).

    Adaptive K: picks no-vmap full-unroll (``K=m``) when per-microbatch
    compute is large enough to hide per-dispatch Python cost; else
    K=2 (minimal chunked vmap). Heuristic uses per-mb element count
    (batch x seq-len-equivalent) as a compute proxy — threshold 2M
    elements is empirical. At bs=4 seq=128 M=4 (128 elements/mb) K=2
    wins; at bs=16 seq=1024 M=4 (4096 elements/mb) K=m wins within
    1.20x of SPMD.

    K (chunk count) controls the overlap/dispatch trade-off:

    * K=1: one big vmap per stage — max vmap-collapse, no overlap
      (baseline).
    * K=2: two chunks — one sync point, partial overlap (safe
      default).
    * K=m: full unroll — no vmap, Mx dispatches, full cross-stage
      overlap.

    User override: pass ``chunks=m`` via sxcall for large-compute
    configs (bs x seq-per-mb >> dispatch cost) where K=m unlocks
    <=1.20x of SPMD. Default K=2 is the safe choice that improves
    every tested config.
    """

    def _vmap_pair(loc: tuple[int, int]) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Return the cached (vfwd, vbwd) pair for stage location ``loc``.

        ``inv_m`` is static (Python float) — value is fixed for a
        given M so JAX bakes it into the HLO and skips the per-call
        re-cast. ``static_argnums`` keeps the jit cache hot and avoids
        re-tracing when the caller passes Python primitives (e.g.
        inv_m as float).
        """
        key = id(fwd_jits[loc])
        cached = _GPIPE_VMAP_CACHE.get(key)
        if cached is not None:
            return cached
        vfwd = jax.jit(jax.vmap(fwd_jits[loc], in_axes=(None, None, 0)))
        base_vbwd = jax.vmap(bwd_jits[loc], in_axes=(None, None, 0, 0))

        @functools.partial(jax.jit, static_argnames=("inv_m_const",))
        def vbwd(p, r, x_stack, gy_stack, inv_m_const):
            """Vmapped backward that also folds the ``1/M`` mean-grad scaling in.

            The per-microbatch param grads are summed along the
            leading axis and scaled by ``inv_m_const`` (a Python float
            so XLA can bake it into the HLO via ``static_argnames``).
            Activation cotangents (``g_x_stack``) are returned per
            microbatch so the next upstream stage can vmap straight on
            them.
            """
            g_params_stack, g_x_stack = base_vbwd(p, r, x_stack, gy_stack)
            g_params = jax.tree.map(
                lambda a: (a.sum(axis=0) * inv_m_const).astype(a.dtype),
                g_params_stack,
                is_leaf=_is_leaf,
            )
            return g_params, g_x_stack

        _GPIPE_VMAP_CACHE[key] = (vfwd, vbwd)
        weak_invalidate(fwd_jits[loc], _GPIPE_VMAP_CACHE, key)
        return vfwd, vbwd

    def _terminal_full(loc: tuple[int, int]) -> Callable[..., Any]:
        """Fused (vfwd + loss + d_loss/dy + vbwd) for the terminal stage.

        All three live on the same sub-mesh, so combining them into
        one jit cuts dispatch count + lets XLA fuse fwd -> loss -> bwd
        with no materialization gap.
        """
        key = (id(fwd_jits[loc]), id(loss_fn), "term_full")
        cached = _GPIPE_TERM_CACHE.get(key)
        if cached is not None:
            return cached

        base_fwd = fwd_jits[loc]
        base_bwd = bwd_jits[loc]
        base_fwd_vmapped = jax.vmap(base_fwd, in_axes=(None, None, 0))
        base_bwd_vmapped = jax.vmap(base_bwd, in_axes=(None, None, 0, 0))

        @functools.partial(jax.jit, static_argnames=("inv_m_const",))
        def fwd_loss_bwd(p, r, x_in_stack, *t_stack, inv_m_const):
            """Run forward, loss, and backward for the terminal stage in one HLO.

            Operates on full ``(M, ...)`` microbatch stacks: the
            forward and backward halves are vmapped, the loss runs per
            microbatch via :func:`jax.value_and_grad`, and the per-mb
            grads are summed + scaled by ``inv_m_const`` (baked-in
            static value) before returning.

            Returns:
                ``(mean_loss, g_params, g_x_stack)``.
            """
            y_stack = base_fwd_vmapped(p, r, x_in_stack)

            def per_mb_loss(y_, *t_):
                """Compute ``(loss, d_loss/d_y)`` for one microbatch under vmap."""
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            loss_stack, gy_stack = jax.vmap(per_mb_loss)(y_stack, *t_stack)

            g_params_stack, g_x_stack = base_bwd_vmapped(p, r, x_in_stack, gy_stack)
            g_params = jax.tree.map(
                lambda a: (a.sum(axis=0) * inv_m_const).astype(a.dtype),
                g_params_stack,
                is_leaf=_is_leaf,
            )
            return loss_stack.mean(), g_params, g_x_stack

        _GPIPE_TERM_CACHE[key] = fwd_loss_bwd
        weak_invalidate(fwd_jits[loc], _GPIPE_TERM_CACHE, key)
        weak_invalidate(loss_fn, _GPIPE_TERM_CACHE, key)
        return fwd_loss_bwd

    if chunks is not None:
        K = max(1, min(int(chunks), m))
    elif m >= 2:
        K = m if (int(xs.size // max(m, 1))) >= 2_000_000 else 2
    else:
        K = 1
    if m % K:
        K = 2 if m >= 2 and m % 2 == 0 else 1
    chunk_size = m // K
    assert m % K == 0, f"microbatches={m} not divisible by K={K}"

    def _chunk(x):
        """Reshape leading-M axis to ``(K, chunk_size, ...)`` for chunked vmap."""
        return x.reshape(K, chunk_size, *x.shape[1:])

    xs_chunks = [xs.reshape(K, chunk_size, *xs.shape[1:])[k] for k in range(K)] if K > 1 else [xs]
    target_chunks = tuple(
        [t.reshape(K, chunk_size, *t.shape[1:])[k] for k in range(K)] if K > 1 else [t] for t in target_args
    )

    inv_m = float(1.0 / m)
    terminal_loc = (n - 1, 0)
    saved_inputs: dict[tuple[int, int], list[jax.Array]] = {}

    x_curr_chunks = list(xs_chunks)
    for rank in range(n - 1):
        loc = (rank, 0)
        vfwd, _ = _vmap_pair(loc)
        out_chunks = []
        with rank_submeshes[rank]:
            for k in range(K):
                y_k = _time_call(
                    f"L{rank}_fwd_chunk{k}",
                    vfwd,
                    stage_params[loc],
                    stage_rest[loc],
                    x_curr_chunks[k],
                )
                out_chunks.append(y_k)
        saved_inputs[loc] = list(x_curr_chunks)
        x_curr_chunks = [jax.device_put(y_k, stage_shardings[rank + 1]) for y_k in out_chunks]

    fused_term = _terminal_full(terminal_loc)
    saved_inputs[terminal_loc] = list(x_curr_chunks)
    loss_chunks = []
    g_params_term_chunks = []
    g_x_chunks = []
    with rank_submeshes[n - 1]:
        for k in range(K):
            tgt_k = tuple(tc[k] for tc in target_chunks)
            loss_k, g_params_k, g_x_k = _time_call(
                f"L{n - 1}_fwd_loss_bwd_chunk{k}",
                functools.partial(fused_term, inv_m_const=inv_m),
                stage_params[terminal_loc],
                stage_rest[terminal_loc],
                x_curr_chunks[k],
                *tgt_k,
            )
            loss_chunks.append(loss_k)
            g_params_term_chunks.append(g_params_k)
            g_x_chunks.append(g_x_k)
        loss = sum(loss_chunks) * (1.0 / K)
        g_params_term = g_params_term_chunks[0]
        for g in g_params_term_chunks[1:]:
            g_params_term = _accumulate_state(g_params_term, g)

    grads: dict[tuple[int, int], State] = {terminal_loc: g_params_term}
    if n > 1:
        g_y_chunks = [jax.device_put(g, stage_shardings[n - 2]) for g in g_x_chunks]
    else:
        g_y_chunks = g_x_chunks
    for rank in range(n - 2, -1, -1):
        loc = (rank, 0)
        _, vbwd = _vmap_pair(loc)
        g_params_accum = None
        next_g_x_chunks = []
        with rank_submeshes[rank]:
            for k in range(K):
                g_params_k, g_x_k = _time_call(
                    f"L{rank}_bwd_chunk{k}",
                    functools.partial(vbwd, inv_m_const=inv_m),
                    stage_params[loc],
                    stage_rest[loc],
                    saved_inputs[loc][k],
                    g_y_chunks[k],
                )
                next_g_x_chunks.append(g_x_k)
                g_params_accum = g_params_k if g_params_accum is None else _accumulate_state(g_params_accum, g_params_k)
            grads[loc] = g_params_accum
        if rank > 0:
            g_y_chunks = [jax.device_put(g, stage_shardings[rank - 1]) for g in next_g_x_chunks]

    return loss, StagesArray(shards={r: grads[(r, 0)] for r in range(n)})


def _prev_loc(schedule: Schedule, rank: int, virt: int, n: int) -> tuple[int, int]:
    """Find the ``(rank, virt)`` location whose logical stage is ``logical - 1``.

    Used when a backward sweep needs to send cotangents upstream: the
    schedule provides ``next_logical_loc`` for forward routing, but
    backward routing needs the inverse. Falls back to a linear scan
    over ``(rank, virt)`` slots.

    Args:
        schedule: The active :class:`Schedule`.
        rank: Current physical rank.
        virt: Current virtual sub-stage.
        n: Number of physical pipeline ranks.

    Returns:
        ``(prev_rank, prev_virt)`` hosting the previous logical stage.

    Raises:
        ValueError: If no slot maps to ``logical - 1`` under
            ``schedule``.
    """
    logical = schedule.logical_at(rank, virt, n)
    prev_logical = logical - 1
    V = schedule.virtual_stages_per_rank()
    for r in range(n):
        for v in range(V):
            if schedule.logical_at(r, v, n) == prev_logical:
                return (r, v)
    raise ValueError(f"Schedule {type(schedule).__name__} has no rank/virt producing logical={prev_logical}.")


def _loc_for_logical(schedule: Schedule, logical: int, n: int, V: int) -> tuple[int, int]:
    """Return the ``(rank, virt)`` location that hosts logical stage ``logical``.

    Inverts ``schedule.logical_at`` by linear scan over the
    ``(rank, virt)`` grid. Used by callers that have a logical stage
    index in hand and need to dispatch to the corresponding physical
    location.

    Args:
        schedule: The active :class:`Schedule`.
        logical: Logical stage index in ``[0, n * V)``.
        n: Number of physical pipeline ranks.
        V: Virtual stages per rank
            (``schedule.virtual_stages_per_rank()``).

    Returns:
        ``(rank, virt)`` for the requested logical stage.

    Raises:
        ValueError: If no ``(rank, virt)`` maps to ``logical`` under
            ``schedule``.
    """
    for r in range(n):
        for v in range(V):
            if schedule.logical_at(r, v, n) == logical:
                return (r, v)
    raise ValueError(f"Schedule {type(schedule).__name__} has no rank/virt for logical={logical}.")
