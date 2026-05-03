# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host wavefront executor for forward-only ``sxjit`` MPMD pipelines.

The regular ``sxjit`` forward path is intentionally direct: prepare an MPMD
plan, execute stage 0, pass its outputs to stage 1, and continue until the final
stage returns the user-facing pytree. That path is correct and simple, but an
inference server has a stronger requirement: while later stages are finishing
microbatch ``N``, earlier stages should be able to enqueue microbatch ``N + 1``.

This module provides that server-oriented execution layer. It consumes the same
private plan exposed by ``sxjit._mpmd_prepare`` and replays it as a host-side
pipeline wavefront. The executor does not invent model-specific semantics:
request packing, KV-cache layout, sampling, and final token routing remain the
caller/runtime's job. Spectrax owns the generic pieces that are independent of a
model family: prepared-stage reuse, activation transport, stage-local carries,
submesh placement, optional stage worker threads, and result reconstruction.

The executor is deliberately forward-only. Training schedules with true MPMD
forward/backward execution use ``sxcall``/``sxgrad`` and the schedule library;
this class targets low-latency decode/prefill pipelines where each microbatch is
a normal forward ``sxjit`` call.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
import typing as tp
from concurrent.futures import Future

import jax

from .runtime import (
    _apply_out_shardings,
    _assemble_invars,
    _assemble_invars_from_plan,
    _assemble_outputs,
    _prepare_invar_assembly_plan,
    _restore_result_treedef,
)

__all__ = [
    "MpmdPipelineDispatchStats",
    "MpmdPipelineExecutor",
]

_MpmdState = dict[str, tp.Any]
_CompiledStage = tuple[tp.Callable[..., tp.Any], tp.Any, tp.Any, tp.Any, list[tuple]]


class _MpmdPreparedCallable(tp.Protocol):
    """Forward callable produced by ``sxjit`` with an exposed MPMD plan.

    ``MpmdPipelineExecutor`` intentionally depends on a tiny structural
    protocol instead of a concrete wrapper class. Tests can provide lightweight
    fakes, while production callers pass the object returned by ``sxjit``. The
    callable behavior is the semantic fallback; ``_mpmd_prepare`` is the fast
    path that exposes compiled per-stage executables and routing metadata.
    """

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Execute the wrapped function with normal ``sxjit`` semantics."""
        ...

    def _mpmd_prepare(self, *args: tp.Any) -> _MpmdState:
        """Return the prepared forward-only MPMD state for ``args``.

        The returned mapping is produced by ``sxjit`` and contains the compiled
        per-rank stage functions, input/output routing maps, explicit input and
        output shardings, pre-placed static leaves, and the original result
        treedef. The executor treats this state as immutable for a stable shape
        bucket so repeated decode calls can reuse it.
        """
        ...


@dataclasses.dataclass(frozen=True)
class MpmdPipelineDispatchStats:
    """Telemetry captured for the most recent pipeline dispatch.

    The fields are intentionally coarse. They are cheap enough to update during
    every decode step and are meant for runtime logging, regression checks, and
    quick performance triage. They do not replace XProf/XLA traces; they simply
    reveal whether time is going into prepare, host submission, final assembly,
    or waiting on stage worker futures.

    Attributes:
        stage_launches: Total number of compiled stage executable invocations.
            For a successful wavefront this is ``microbatches * num_stages``.
        microbatches: Number of same-shaped argument batches executed in this
            dispatch.
        stage_dispatch_time: Cumulative wall-clock seconds spent between
            enqueueing a worker-backed stage task and receiving its result. This
            is primarily meaningful when ``use_workers=True``.
        queue_wait_time: Currently the same lower-bound measurement as
            ``stage_dispatch_time``; kept separate for callers that already
            display queue-specific telemetry.
        prepare_time: Wall-clock seconds spent preparing ``sxjit`` state or
            flattening runtime arguments from a cached prepare entry.
        assemble_time: Wall-clock seconds spent rebuilding user-facing output
            pytrees from per-stage flat outputs.
        submit_time: Wall-clock seconds spent assembling stage inputs and
            enqueueing/executing stage work on the host.
        stage_submit_times_ms: Per-stage wall-clock milliseconds spent from
            stage input assembly through the ``stage_jit`` enqueue/worker
            submission call. JAX execution is asynchronous, so this is a
            boundary/launch timing signal, not a device-compute measurement.
        stage_assemble_times_ms: Per-stage wall-clock milliseconds spent
            materializing the positional input list for the stage call.
        stage_execute_times_ms: Per-stage wall-clock milliseconds spent inside
            the actual ``stage_jit`` enqueue call or resident-worker submit.
            This is still host dispatch time, not device execution time.
    """

    stage_launches: int
    microbatches: int
    stage_dispatch_time: float
    queue_wait_time: float
    prepare_time: float = 0.0
    assemble_time: float = 0.0
    submit_time: float = 0.0
    stage_submit_times_ms: tuple[float, ...] = ()
    stage_assemble_times_ms: tuple[float, ...] = ()
    stage_execute_times_ms: tuple[float, ...] = ()


@dataclasses.dataclass(frozen=True)
class _PreparedCall:
    """Prepared ``sxjit`` state plus flattened runtime argument leaves.

    Attributes:
        state: Private state returned by ``sxjit._mpmd_prepare``. It includes
            compiled stage functions, input/output maps, placed static values,
            sharding metadata, and result reconstruction metadata.
        flat_args: Positional call arguments flattened with ``jax.tree.leaves``.
            Stage input assembly indexes this list using Spectrax's flat-leaf
            routing maps.
    """

    state: _MpmdState
    flat_args: list[tp.Any]


@dataclasses.dataclass
class _PrepareCacheEntry:
    """Shape-stable prepare result reused across same-bucket microbatches.

    Decode servers repeatedly call the same compiled bucket with different token
    and cache leaves but identical graph/weight/static arguments. Re-running
    ``_mpmd_prepare`` and re-flattening very large static pytrees for every
    token adds visible host overhead. This cache stores the prepared state plus
    enough flattening metadata to rebuild only the dynamic leaves on later calls.

    Attributes:
        state: Cached private ``sxjit`` MPMD state for the bucket.
        flat_args_template: Flattened argument list from the first call. Leaves
            belonging to runtime-static arguments are reused as-is; leaves for
            dynamic arguments are overwritten on each dispatch. Keeping this
            template avoids appending hundreds of stable graph/state leaves on
            every decode token.
        runtime_static_cache: Per-stage device-placed dynamic leaves that the
            caller declares stable for the bucket. Keys are ``(stage, flat_idx)``
            because each stage may require a different destination sharding.
        runtime_static_flat_args: Original flattened leaves for runtime-static
            positional arguments.
        arg_offsets: Starting flat-leaf index for each positional argument.
        arg_leaf_counts: Number of flat leaves contributed by each positional
            argument.
        invar_plans: Pre-classified stage input assembly plans derived from the
            cached state.
        runtime_static_invar_plan_cache: Invar plans with runtime-static dynamic
            slots already folded into the template. The key is the set of flat
            argument leaves treated as runtime-static for this bucket.
    """

    state: _MpmdState
    flat_args_template: list[tp.Any]
    runtime_static_cache: dict[tuple[int, int], tp.Any]
    runtime_static_flat_args: dict[int, tp.Any]
    arg_offsets: tuple[int, ...]
    arg_leaf_counts: tuple[int, ...]
    invar_plans: tuple[tp.Any, ...]
    runtime_static_invar_plan_cache: dict[frozenset[int], tuple[tp.Any, ...]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _StageTask:
    """One queued stage invocation for a worker-backed pipeline dispatch.

    Attributes:
        stage_jit: Compiled JAX callable for one physical pipeline rank.
        submesh: Rank-local mesh context used while invoking ``stage_jit``.
        invars: Fully materialized positional inputs for the stage.
        future: Future completed by the worker with the stage's flat outputs.
    """

    stage_jit: tp.Callable[..., tp.Any]
    submesh: tp.Any
    invars: list[tp.Any]
    future: Future[tuple]


class _StageWorker:
    """Daemon worker pinned logically to one physical pipeline rank.

    Worker mode is optional. The default inline path relies on JAX returning
    asynchronous device futures, which is usually enough to enqueue a wavefront
    quickly. Worker mode is useful for experiments where each physical stage
    should have an independent host thread, for example when caller-side Python
    work or mesh context management becomes visible in traces.
    """

    def __init__(self, *, rank: int) -> None:
        """Start a resident worker for ``rank``.

        Args:
            rank: Physical pipeline rank this worker represents. The value is
                used for naming/debugging only; device placement is still driven
                by the ``submesh`` carried on each submitted task.
        """
        self.rank = int(rank)
        self._queue: queue.Queue[_StageTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"spx-mpmd-stage-{rank}", daemon=True)
        self._thread.start()

    def submit(self, *, stage_jit: tp.Callable[..., tp.Any], submesh: tp.Any, invars: list[tp.Any]) -> Future[tuple]:
        """Queue one stage invocation and return its completion future.

        Args:
            stage_jit: Compiled per-rank executable.
            submesh: Mesh context to enter before calling ``stage_jit``.
            invars: Positional inputs already placed for the destination stage.

        Returns:
            A future that resolves to the stage's flat output tuple, or carries
            the raised exception from the worker thread.
        """
        future: Future[tuple] = Future()
        self._queue.put(_StageTask(stage_jit=stage_jit, submesh=submesh, invars=invars, future=future))
        return future

    def shutdown(self) -> None:
        """Ask the worker to exit and wait briefly for its thread to finish."""
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Worker event loop that executes queued stage tasks under their submesh."""
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            if task.future.set_running_or_notify_cancel():
                try:
                    with jax.named_scope(f"spectrax/mpmd/pipeline/worker/stage_{self.rank}"):
                        out = task.stage_jit(*task.invars)
                    task.future.set_result(out)
                except BaseException as exc:
                    task.future.set_exception(exc)
            self._queue.task_done()


class MpmdPipelineExecutor:
    """Host wavefront executor for forward-only ``sxjit`` MPMD plans.

    ``dispatch`` executes one call with the same semantics as invoking the
    wrapped ``sxjit`` function directly. ``dispatch_many`` accepts same-shaped
    microbatches and runs the physical pipeline as a host-side wavefront:
    stage 0 can enqueue microbatch 1 before the final stage has finished
    microbatch 0. JAX/XLA still owns device-side dependency ordering; the host
    only sequences launches whose inputs are available.

    The executor is intentionally small and reusable. It has no opinion about
    request queues, paged attention, logits processors, or samplers. Callers pass
    already-packed argument tuples, optional stage-local carry wiring for KV
    state, and an optional prepare-cache key for a stable decode bucket.
    """

    def __init__(self, *, stage_meshes: tp.Sequence[tp.Any] | None = None, use_workers: bool = False) -> None:
        """Create an executor.

        Args:
            stage_meshes: Optional physical stage meshes used as a mesh lookup
                fallback for tests or older prepared states. Modern ``sxjit``
                plans carry the owning MPMD mesh directly.
            use_workers: When ``False`` stage calls are enqueued inline on the
                caller thread. When ``True`` the executor creates one resident
                daemon worker per physical stage and submits stage calls through
                futures.
        """
        self.stage_meshes = tuple(stage_meshes or ())
        self.use_workers = bool(use_workers)
        self._workers: list[_StageWorker] = []
        self._worker_count = 0
        self._last_stats = MpmdPipelineDispatchStats(0, 0, 0.0, 0.0)
        self._prepare_cache: dict[tp.Hashable, _PrepareCacheEntry] = {}

    @property
    def last_stats(self) -> MpmdPipelineDispatchStats:
        """Return telemetry from the most recent ``dispatch``/``dispatch_many`` call."""
        return self._last_stats

    def shutdown(self) -> None:
        """Stop resident workers and clear bucket-local prepare caches."""
        for worker in self._workers:
            worker.shutdown()
        self._workers = []
        self._worker_count = 0
        self._prepare_cache.clear()

    def clear_prepare_cache(self) -> None:
        """Drop cached ``sxjit`` prepare state while keeping resident workers alive.

        Use this when a caller invalidates a decode bucket, swaps weights, or
        changes a static call argument but wants to keep the executor object and
        any worker threads around.
        """
        self._prepare_cache.clear()

    def dispatch(self, sxjit_fn: _MpmdPreparedCallable, *args: tp.Any) -> tp.Any:
        """Execute one ``sxjit`` call through the pipeline executor.

        This is a convenience wrapper around ``dispatch_many`` for callers that
        do not need wavefront overlap. It still uses the same stage input
        assembly and result reconstruction path, which makes it useful for
        validating pipeline executor correctness against direct ``sxjit`` calls.
        """
        outputs = self.dispatch_many(sxjit_fn, (args,))
        return outputs[0]

    def dispatch_many(
        self,
        sxjit_fn: _MpmdPreparedCallable,
        arg_batches: tp.Iterable[tuple],
        *,
        carry_input_output_map: tp.Mapping[int, tp.Mapping[int, int]] | None = None,
        prepare_cache_key: tp.Hashable | None = None,
        runtime_static_argnums: tp.Iterable[int] | None = None,
    ) -> list[tp.Any]:
        """Execute same-shaped microbatches as a pipeline wavefront.

        The executor consumes SpectraX's private forward-only MPMD plan. For a
        single call it is semantically equivalent to invoking the ``sxjit``
        function normally; for multiple calls it walks the physical pipeline in
        wave order:

        ``(mb0, stage0) -> (mb1, stage0), (mb0, stage1) -> ...``

        When a stage-local carry map is supplied, recurrent leaves such as KV
        cache pages are sourced from the previous output of the same stage. That
        keeps cache ownership local to each pipeline rank while allowing the
        host to overlap independent stage launches.

        Args:
            sxjit_fn: A forward-only ``spectrax.sxjit`` callable exposing
                ``_mpmd_prepare``.
            arg_batches: Iterable of positional-argument tuples, one per
                microbatch. All microbatches must resolve to compatible
                forward-only stage plans.
            carry_input_output_map: Optional stage-local recurrent-state
                mapping. Keys are stage indices. Values map original flat input
                leaf indices to output positions produced by that same stage in
                the previous microbatch. This lets callers pipeline stateful
                decode: stage 0 consumes stage 0's prior KV output while stage 1
                independently consumes stage 1's prior KV output.

            prepare_cache_key: Optional hashable bucket key. When supplied, the
                executor caches the prepared stage plan, static flattening
                metadata, and stage input assembly plans for reuse on later
                calls with the same bucket shape.
            runtime_static_argnums: Positional argument indices whose leaves are
                dynamic from Spectrax's tracing perspective but stable for the
                runtime bucket, such as graph definitions or weights in an
                inference engine. Those leaves are flattened and placed once per
                cache entry.

        Returns:
            One output pytree per microbatch, in input order.
        """
        t_prepare = time.time()
        arg_batches = list(arg_batches)
        cached_entry = self._prepare_cache.get(prepare_cache_key) if prepare_cache_key is not None else None
        if cached_entry is None:
            prepared = [self._prepare_call(sxjit_fn, args) for args in arg_batches]
            if prepare_cache_key is not None and prepared:
                self._prepare_cache[prepare_cache_key] = self._make_cache_entry(
                    prepared[0].state,
                    arg_batches[0],
                    prepared[0].flat_args,
                    runtime_static_argnums=runtime_static_argnums,
                )
                cached_entry = self._prepare_cache[prepare_cache_key]
        else:
            prepared = [
                _PreparedCall(
                    state=cached_entry.state,
                    flat_args=self._flatten_args_with_runtime_static_cache(
                        cached_entry,
                        args,
                        runtime_static_argnums=runtime_static_argnums,
                    ),
                )
                for args in arg_batches
            ]
        prepare_time = time.time() - t_prepare
        if not prepared:
            self._last_stats = MpmdPipelineDispatchStats(0, 0, 0.0, 0.0)
            return []

        compiled: list[_CompiledStage] = prepared[0].state["compiled"]
        if self.use_workers:
            self._ensure_workers(len(compiled))
        for call in prepared[1:]:
            other = call.state["compiled"]
            if other is not compiled:
                raise ValueError(
                    "MpmdPipelineExecutor.dispatch_many requires every microbatch to use the same compiled plan. "
                    "Bucket or pad microbatches to the same shape before calling it."
                )

        carry_map = self._normalize_carry_map(carry_input_output_map, len(compiled), len(prepared[0].flat_args))
        runtime_static_flat_indices = self._runtime_static_flat_indices(
            cached_entry,
            runtime_static_argnums,
            len(arg_batches[0]),
        )
        runtime_static_cache = cached_entry.runtime_static_cache if cached_entry is not None else None
        invar_plans = cached_entry.invar_plans if cached_entry is not None else None
        runtime_static_plan_key: frozenset[int] | None = None
        runtime_static_plan_updates: list[tp.Any | None] | None = None
        if cached_entry is not None and runtime_static_flat_indices:
            runtime_static_plan_key = frozenset(runtime_static_flat_indices)
            resolved_plans = cached_entry.runtime_static_invar_plan_cache.get(runtime_static_plan_key)
            if resolved_plans is not None:
                invar_plans = resolved_plans
            elif invar_plans is not None:
                runtime_static_plan_updates = [None] * len(invar_plans)
        rank_submeshes = [stage[1] for stage in compiled]
        rank_device_sets = [set(submesh.devices.flat) for submesh in rank_submeshes]
        mpmd_mesh = self._resolve_mpmd_mesh(prepared[0].state, rank_submeshes, sxjit_fn=sxjit_fn)
        futures: list[list[Future[tuple] | None]] = [[None] * len(compiled) for _ in prepared]
        cluster_outputs: list[list[tuple | None]] = [[None] * len(compiled) for _ in prepared]
        stage_dispatch_time = 0.0
        queue_wait_time = 0.0
        submit_time = 0.0
        stage_submit_times_ms = [0.0] * len(compiled)
        stage_assemble_times_ms = [0.0] * len(compiled)
        stage_execute_times_ms = [0.0] * len(compiled)

        if not self.use_workers and len(prepared) == 1:
            call = prepared[0]
            cluster_outputs: list[tuple | None] = [None] * len(compiled)
            for stage_idx, (stage_jit, _submesh, my_sh, _, invar_map) in enumerate(compiled):
                t_submit = time.time()
                prev_outputs = cluster_outputs[stage_idx - 1] if stage_idx > 0 else ()
                if prev_outputs is None:
                    raise RuntimeError(f"internal error: missing previous output for stage {stage_idx}")

                t_assemble_stage = time.time()
                with jax.named_scope(f"spectrax/mpmd/pipeline/single/stage_{stage_idx}/assemble"):
                    invar_plan = invar_plans[stage_idx] if invar_plans is not None else None
                    invars = self._assemble_stage_invars(
                        call=call,
                        stage_idx=stage_idx,
                        invar_map=invar_map,
                        invar_plan=invar_plan,
                        my_sh=my_sh,
                        rank_devices=rank_device_sets[stage_idx],
                        rank_submeshes=rank_submeshes,
                        mpmd_mesh=mpmd_mesh,
                        prev_outputs=prev_outputs,
                        all_cluster_outputs=cluster_outputs,
                        runtime_static_flat_indices=runtime_static_flat_indices,
                        runtime_static_cache=runtime_static_cache,
                    )
                    if (
                        runtime_static_plan_updates is not None
                        and invar_plan is not None
                        and runtime_static_flat_indices is not None
                    ):
                        runtime_static_plan_updates[stage_idx] = self._fold_runtime_static_slots_into_plan(
                            invar_plan,
                            invars,
                            runtime_static_flat_indices,
                        )
                stage_assemble_elapsed = time.time() - t_assemble_stage
                stage_assemble_times_ms[stage_idx] += stage_assemble_elapsed * 1000.0

                t_execute_stage = time.time()
                with jax.named_scope(f"spectrax/mpmd/pipeline/single/stage_{stage_idx}/execute"):
                    cluster_outputs[stage_idx] = stage_jit(*invars)
                stage_execute_elapsed = time.time() - t_execute_stage
                stage_execute_times_ms[stage_idx] += stage_execute_elapsed * 1000.0

                stage_submit_elapsed = time.time() - t_submit
                submit_time += stage_submit_elapsed
                stage_submit_times_ms[stage_idx] += stage_submit_elapsed * 1000.0

            t_assemble = time.time()
            result = self._assemble_result(call, cluster_outputs)
            assemble_time = time.time() - t_assemble
            if (
                cached_entry is not None
                and runtime_static_plan_key is not None
                and runtime_static_plan_updates is not None
                and all(plan is not None for plan in runtime_static_plan_updates)
            ):
                cached_entry.runtime_static_invar_plan_cache[runtime_static_plan_key] = tuple(
                    runtime_static_plan_updates
                )
            self._last_stats = MpmdPipelineDispatchStats(
                stage_launches=len(compiled),
                microbatches=1,
                stage_dispatch_time=0.0,
                queue_wait_time=0.0,
                prepare_time=prepare_time,
                assemble_time=assemble_time,
                submit_time=submit_time,
                stage_submit_times_ms=tuple(stage_submit_times_ms),
                stage_assemble_times_ms=tuple(stage_assemble_times_ms),
                stage_execute_times_ms=tuple(stage_execute_times_ms),
            )
            return [result]

        def wait_stage(mb_idx: int, stage_idx: int) -> tuple:
            nonlocal stage_dispatch_time, queue_wait_time
            ready = cluster_outputs[mb_idx][stage_idx]
            if ready is not None:
                return ready
            if not self.use_workers:
                raise RuntimeError(f"internal error: stage {stage_idx} microbatch {mb_idx} was not dispatched")
            future = futures[mb_idx][stage_idx]
            if future is None:
                raise RuntimeError(f"internal error: stage {stage_idx} microbatch {mb_idx} was not submitted")
            t_wait = time.time()
            ready = future.result()
            elapsed = time.time() - t_wait
            stage_dispatch_time += elapsed
            queue_wait_time += max(0.0, elapsed)
            cluster_outputs[mb_idx][stage_idx] = ready
            return ready

        num_microbatches = len(prepared)
        num_stages = len(compiled)
        for wave_idx in range(num_microbatches + num_stages - 1):
            hi_stage = min(wave_idx, num_stages - 1)
            for stage_idx in range(0, hi_stage + 1):
                mb_idx = wave_idx - stage_idx
                if mb_idx < 0 or mb_idx >= num_microbatches:
                    continue
                if futures[mb_idx][stage_idx] is not None:
                    continue

                t_submit = time.time()
                stage_jit, submesh, my_sh, _, invar_map = compiled[stage_idx]
                rank_devices = rank_device_sets[stage_idx]
                prev_outputs: tuple = ()
                if stage_idx > 0:
                    prev_outputs = wait_stage(mb_idx, stage_idx - 1)

                call = prepared[mb_idx]
                stage_carries = carry_map.get(stage_idx, {})
                if stage_carries and mb_idx > 0:
                    previous_stage_outputs = wait_stage(mb_idx - 1, stage_idx)
                    flat_args = list(call.flat_args)
                    for orig_flat_idx, stage_out_pos in stage_carries.items():
                        flat_args[orig_flat_idx] = previous_stage_outputs[stage_out_pos]
                    call = _PreparedCall(state=call.state, flat_args=flat_args)

                t_assemble_stage = time.time()
                with jax.named_scope(f"spectrax/mpmd/pipeline/microbatch_{mb_idx}/stage_{stage_idx}/assemble"):
                    invar_plan = invar_plans[stage_idx] if invar_plans is not None else None
                    invars = self._assemble_stage_invars(
                        call=call,
                        stage_idx=stage_idx,
                        invar_map=invar_map,
                        invar_plan=invar_plan,
                        my_sh=my_sh,
                        rank_devices=rank_devices,
                        rank_submeshes=rank_submeshes,
                        mpmd_mesh=mpmd_mesh,
                        prev_outputs=prev_outputs,
                        all_cluster_outputs=cluster_outputs[mb_idx],
                        runtime_static_flat_indices=runtime_static_flat_indices,
                        runtime_static_cache=runtime_static_cache,
                    )
                    if (
                        runtime_static_plan_updates is not None
                        and mb_idx == 0
                        and invar_plan is not None
                        and runtime_static_flat_indices is not None
                    ):
                        runtime_static_plan_updates[stage_idx] = self._fold_runtime_static_slots_into_plan(
                            invar_plan,
                            invars,
                            runtime_static_flat_indices,
                        )
                stage_assemble_elapsed = time.time() - t_assemble_stage
                stage_assemble_times_ms[stage_idx] += stage_assemble_elapsed * 1000.0
                t_execute_stage = time.time()
                if self.use_workers:
                    futures[mb_idx][stage_idx] = self._workers[stage_idx].submit(
                        stage_jit=stage_jit,
                        submesh=submesh,
                        invars=invars,
                    )
                else:
                    with jax.named_scope(f"spectrax/mpmd/pipeline/microbatch_{mb_idx}/stage_{stage_idx}/execute"):
                        cluster_outputs[mb_idx][stage_idx] = stage_jit(*invars)
                stage_execute_elapsed = time.time() - t_execute_stage
                stage_execute_times_ms[stage_idx] += stage_execute_elapsed * 1000.0
                stage_submit_elapsed = time.time() - t_submit
                submit_time += stage_submit_elapsed
                stage_submit_times_ms[stage_idx] += stage_submit_elapsed * 1000.0

        final_stage = num_stages - 1
        for mb_idx in range(num_microbatches):
            wait_stage(mb_idx, final_stage)

        t_assemble = time.time()
        results = [self._assemble_result(call, outputs) for call, outputs in zip(prepared, cluster_outputs, strict=True)]
        assemble_time = time.time() - t_assemble
        if (
            cached_entry is not None
            and runtime_static_plan_key is not None
            and runtime_static_plan_updates is not None
            and all(plan is not None for plan in runtime_static_plan_updates)
        ):
            cached_entry.runtime_static_invar_plan_cache[runtime_static_plan_key] = tuple(runtime_static_plan_updates)
        self._last_stats = MpmdPipelineDispatchStats(
            stage_launches=len(compiled) * len(prepared),
            microbatches=len(prepared),
            stage_dispatch_time=stage_dispatch_time,
            queue_wait_time=queue_wait_time,
            prepare_time=prepare_time,
            assemble_time=assemble_time,
            submit_time=submit_time,
            stage_submit_times_ms=tuple(stage_submit_times_ms),
            stage_assemble_times_ms=tuple(stage_assemble_times_ms),
            stage_execute_times_ms=tuple(stage_execute_times_ms),
        )
        return results

    def _normalize_carry_map(
        self,
        carry_input_output_map: tp.Mapping[int, tp.Mapping[int, int]] | None,
        num_stages: int,
        num_flat_args: int,
    ) -> dict[int, dict[int, int]]:
        """Validate and normalize caller-provided stage-local carry wiring.

        The carry map is intentionally expressed in flat-leaf positions because
        it sits below model/runtime abstractions. Each stage map says: before
        dispatching stage ``S`` for microbatch ``N > 0``, replace original flat
        input ``I`` with output position ``O`` from stage ``S`` of microbatch
        ``N - 1``. This preserves stage-local cache ownership and avoids routing
        one stage's KV state through another stage.
        """
        if not carry_input_output_map:
            return {}
        normalized: dict[int, dict[int, int]] = {}
        for stage_idx_raw, stage_map_raw in carry_input_output_map.items():
            stage_idx = int(stage_idx_raw)
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(f"carry_input_output_map contains invalid stage index {stage_idx}.")
            stage_map: dict[int, int] = {}
            for orig_flat_idx_raw, stage_out_pos_raw in stage_map_raw.items():
                orig_flat_idx = int(orig_flat_idx_raw)
                stage_out_pos = int(stage_out_pos_raw)
                if orig_flat_idx < 0 or orig_flat_idx >= num_flat_args:
                    raise ValueError(
                        f"carry_input_output_map stage {stage_idx} references invalid flat input {orig_flat_idx}."
                    )
                if stage_out_pos < 0:
                    raise ValueError(
                        f"carry_input_output_map stage {stage_idx} references invalid output {stage_out_pos}."
                    )
                stage_map[orig_flat_idx] = stage_out_pos
            if stage_map:
                normalized[stage_idx] = stage_map
        return normalized

    def _prepare_call(self, sxjit_fn: _MpmdPreparedCallable, args: tuple) -> _PreparedCall:
        """Ask ``sxjit`` for its MPMD stage plan and flatten runtime leaves.

        Args:
            sxjit_fn: A callable implementing ``_mpmd_prepare``.
            args: Positional arguments for one microbatch.

        Returns:
            A prepared call containing the immutable stage plan plus the flat
            dynamic leaves for this specific microbatch.

        Raises:
            TypeError: If the callable is not an ``sxjit`` MPMD wrapper or the
                prepared state is not a forward-only compiled plan.
        """
        prepare = getattr(sxjit_fn, "_mpmd_prepare", None)
        if prepare is None:
            raise TypeError("MpmdPipelineExecutor requires a SpectraX sxjit function with _mpmd_prepare.")
        state = dict(prepare(*args))
        if "compiled" not in state:
            raise TypeError("MpmdPipelineExecutor only supports forward-only sxjit plans.")
        return _PreparedCall(state=state, flat_args=jax.tree.leaves(args))

    def _make_cache_entry(
        self,
        state: _MpmdState,
        args: tuple,
        flat_args: list[tp.Any],
        *,
        runtime_static_argnums: tp.Iterable[int] | None,
    ) -> _PrepareCacheEntry:
        """Build a reusable prepare-cache entry for a stable bucket shape.

        The entry records positional-argument leaf offsets so later calls can
        flatten only non-static arguments while replaying cached leaves for
        runtime-static ones. It also precomputes stage input assembly plans from
        the prepared ``sxjit`` state so decode steps avoid repeatedly scanning
        full invar maps.
        """
        offsets: list[int] = []
        counts: list[int] = []
        offset = 0
        for arg in args:
            count = len(jax.tree.leaves(arg))
            offsets.append(offset)
            counts.append(count)
            offset += count
        static_argnums = self._normalize_runtime_static_argnums(runtime_static_argnums, len(args))
        static_flat_args: dict[int, tp.Any] = {}
        for argnum in static_argnums:
            start = offsets[argnum]
            count = counts[argnum]
            for flat_idx in range(start, start + count):
                static_flat_args[flat_idx] = flat_args[flat_idx]
        return _PrepareCacheEntry(
            state=state,
            flat_args_template=list(flat_args),
            runtime_static_cache={},
            runtime_static_flat_args=static_flat_args,
            arg_offsets=tuple(offsets),
            arg_leaf_counts=tuple(counts),
            invar_plans=self._make_invar_plans(state),
        )

    def _fold_runtime_static_slots_into_plan(
        self,
        plan: tp.Any,
        invars: list[tp.Any],
        runtime_static_flat_indices: set[int],
    ) -> tp.Any:
        """Return an invar plan with stable dynamic slots pre-filled.

        Runtime integrations such as eSurge pass graph definitions and weights
        as normal dynamic arguments so they are not JAX compile-time constants.
        They are nevertheless stable for a decode bucket. The generic hot path
        used to visit every such slot on every token, doing set membership and
        cache lookups for roughly a hundred weight leaves per stage. After the
        first call has materialized those leaves with the correct stage
        placement, this helper folds them into the plan template. Later decode
        steps only iterate over truly changing slots: KV/cache leaves, metadata,
        and inter-stage activations.
        """
        template = list(plan.template)
        dynamic_slots: list[tuple[int, int]] = []
        for out_pos, orig_idx in plan.dynamic_slots:
            if int(orig_idx) in runtime_static_flat_indices:
                template[int(out_pos)] = invars[int(out_pos)]
            else:
                dynamic_slots.append((int(out_pos), int(orig_idx)))
        return type(plan)(
            template=tuple(template),
            dynamic_slots=tuple(dynamic_slots),
            stage_slots=plan.stage_slots,
            prev_slots=plan.prev_slots,
        )

    def _normalize_runtime_static_argnums(
        self,
        runtime_static_argnums: tp.Iterable[int] | None,
        num_args: int,
    ) -> set[int]:
        """Convert Python-style argnums, including negatives, into a checked set.

        Args:
            runtime_static_argnums: User-provided argument indices. Negative
                values follow normal Python indexing from the end.
            num_args: Number of positional arguments in the call signature.

        Returns:
            A deduplicated set of valid non-negative argument indices.

        Raises:
            ValueError: If an index falls outside the positional argument range.
        """
        if runtime_static_argnums is None:
            return set()
        normalized: set[int] = set()
        for argnum_raw in runtime_static_argnums:
            argnum = int(argnum_raw)
            if argnum < 0:
                argnum += num_args
            if argnum < 0 or argnum >= num_args:
                raise ValueError(f"runtime_static_argnums contains invalid arg index {argnum_raw}.")
            normalized.add(argnum)
        return normalized

    def _flatten_args_with_runtime_static_cache(
        self,
        entry: _PrepareCacheEntry,
        args: tuple,
        *,
        runtime_static_argnums: tp.Iterable[int] | None,
    ) -> list[tp.Any]:
        """Flatten args while reusing selected static leaves from the cache.

        EasyDeL passes graph definitions and weight pytrees through the same
        Python call signature for every decode step, but those values are static
        for a compiled bucket. Reusing their flattened leaves avoids repeatedly
        walking very large graph/state pytrees on the host.

        The method still verifies leaf counts for non-static arguments. A count
        mismatch means the caller changed the bucket shape or argument treedef
        while reusing the same prepare-cache key, which would make Spectrax's
        flat-leaf routing maps invalid.
        """
        static_argnums = self._normalize_runtime_static_argnums(runtime_static_argnums, len(args))
        flat_args = list(entry.flat_args_template)
        for argnum, arg in enumerate(args):
            start = entry.arg_offsets[argnum]
            expected_count = entry.arg_leaf_counts[argnum]
            if argnum in static_argnums:
                continue
            leaves = jax.tree.leaves(arg)
            if len(leaves) != expected_count:
                raise ValueError(
                    "MpmdPipelineExecutor cached prepare shape changed: "
                    f"arg {argnum} had {expected_count} leaves, now has {len(leaves)}."
                )
            flat_args[start : start + expected_count] = leaves
        return flat_args

    def _runtime_static_flat_indices(
        self,
        entry: _PrepareCacheEntry | None,
        runtime_static_argnums: tp.Iterable[int] | None,
        num_args: int,
    ) -> set[int] | None:
        """Return flat-leaf indices covered by runtime-static positional args.

        Args:
            entry: Active prepare cache entry, or ``None`` when no cache is in
                use.
            runtime_static_argnums: Positional arguments declared static for the
                current runtime bucket.
            num_args: Number of positional arguments in the call.

        Returns:
            ``None`` when there is no cache/static declaration, otherwise the
            flat-leaf indices that should use stage-local placement caching.
        """
        if entry is None or runtime_static_argnums is None:
            return None
        indices: set[int] = set()
        for argnum in self._normalize_runtime_static_argnums(runtime_static_argnums, num_args):
            start = entry.arg_offsets[argnum]
            count = entry.arg_leaf_counts[argnum]
            indices.update(range(start, start + count))
        return indices

    def _assemble_stage_invars(
        self,
        *,
        call: _PreparedCall,
        stage_idx: int,
        invar_map: list[tuple],
        invar_plan: tp.Any | None,
        my_sh: tp.Any,
        rank_devices: set,
        rank_submeshes: list[tp.Any],
        mpmd_mesh: tp.Any,
        prev_outputs: tuple,
        all_cluster_outputs: list[tuple | None],
        runtime_static_flat_indices: set[int] | None,
        runtime_static_cache: dict[tuple[int, int], tp.Any] | None,
    ) -> list:
        """Materialize one stage's positional inputs for one microbatch.

        SpectraX's compiled stage plan describes each input as one of three
        sources: an original function argument, the immediately previous stage,
        or an earlier cluster output. This helper delegates to the fast prepared
        invar plan when available and otherwise falls back to the generic
        assembler from ``runtime.py``.

        Runtime-static cache information is passed through to the runtime helper
        so graph/weight-like dynamic leaves are placed once per stage and reused
        across decode steps in the same bucket.
        """
        state = call.state
        if invar_plan is not None:
            return _assemble_invars_from_plan(
                invar_plan,
                call.flat_args,
                state["explicit_in_sh"],
                prev_outputs,
                all_cluster_outputs,
                stage_idx,
                my_sh,
                rank_devices,
                rank_submeshes,
                mpmd_mesh,
                runtime_static_flat_indices=runtime_static_flat_indices,
                runtime_static_cache=runtime_static_cache,
            )
        return _assemble_invars(
            invar_map,
            call.flat_args,
            state["placed"],
            state["dynamic"],
            state["explicit_in_sh"],
            prev_outputs,
            all_cluster_outputs,
            stage_idx,
            my_sh,
            rank_devices,
            rank_submeshes,
            mpmd_mesh,
            dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
            runtime_static_flat_indices=runtime_static_flat_indices,
            runtime_static_cache=runtime_static_cache,
        )

    def _make_invar_plans(self, state: _MpmdState) -> tuple[tp.Any, ...]:
        """Precompute stage-input routing plans for the cached prepare state.

        Each compiled stage carries an ``invar_map`` describing where every
        positional input should come from. This method turns those maps into
        compact templates with holes for only dynamic arguments and inter-stage
        transfers, shaving Python branching out of hot decode dispatch.
        """
        compiled: list[_CompiledStage] = state["compiled"]
        return tuple(
            _prepare_invar_assembly_plan(
                invar_map,
                state["placed"],
                state["dynamic"],
                stage_idx,
                dynamic_flat_to_orig_flat=state.get("dynamic_flat_to_orig_flat"),
            )
            for stage_idx, (*_, invar_map) in enumerate(compiled)
        )

    def _assemble_result(self, call: _PreparedCall, outputs: list[tuple | None]) -> tp.Any:
        """Rebuild the user-facing pytree from per-stage flat outputs.

        Args:
            call: Prepared call whose state contains the output routing map and
                result treedef.
            outputs: Per-stage flat output tuples collected by the wavefront.

        Returns:
            The exact pytree shape that a direct ``sxjit`` call would return,
            with optional ``out_shardings`` applied by Spectrax's normal rules.
        """
        ready_outputs: list[tuple] = []
        for idx, value in enumerate(outputs):
            if value is None:
                raise RuntimeError(f"internal error: missing output for stage {idx}")
            ready_outputs.append(value)
        result = _assemble_outputs(
            call.state["fn_outvar_map"],
            ready_outputs,
            call.flat_args,
            dynamic_flat_to_orig_flat=call.state.get("dynamic_flat_to_orig_flat"),
        )
        result = _apply_out_shardings(result, call.state.get("out_shardings"))
        return _restore_result_treedef(result, call.state.get("result_treedef"))

    def _resolve_mpmd_mesh(
        self,
        state: _MpmdState,
        rank_submeshes: list[tp.Any],
        sxjit_fn: _MpmdPreparedCallable | None,
    ) -> tp.Any:
        """Find the owning MPMD mesh for device placement decisions.

        The forward plan normally carries this directly in ``state``. The
        fallbacks exist because older prepared states and tests may only expose
        the mesh through submeshes or the wrapped sxjit function.

        Returning ``None`` is allowed for host-only/unit-test fakes where the
        transfer helper never needs to resolve a real marker edge sharding.
        """
        mpmd_mesh = state.get("mpmd_mesh")
        if mpmd_mesh is not None:
            return mpmd_mesh
        if rank_submeshes:
            mpmd_mesh = getattr(getattr(rank_submeshes[0], "spmd_mesh", None), "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        if self.stage_meshes:
            mpmd_mesh = getattr(getattr(self.stage_meshes[0], "spmd_mesh", None), "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
            mpmd_mesh = getattr(self.stage_meshes[0], "mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        if sxjit_fn is not None:
            mpmd_mesh = getattr(sxjit_fn, "_mpmd_mesh", None)
            if mpmd_mesh is not None:
                return mpmd_mesh
        return None

    def _ensure_workers(self, worker_count: int) -> None:
        """Create exactly one resident worker per physical pipeline stage.

        The worker pool is rebuilt when the stage count changes because each
        worker is named and logically associated with one rank. Rebuilding also
        clears the prepare cache through ``shutdown`` so a caller cannot
        accidentally reuse state prepared for a different physical pipeline.
        """
        if self._worker_count == worker_count and len(self._workers) == worker_count:
            return
        self.shutdown()
        self._workers = [_StageWorker(rank=rank) for rank in range(worker_count)]
        self._worker_count = worker_count
