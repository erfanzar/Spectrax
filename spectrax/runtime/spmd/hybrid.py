# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid SPMD/MPMD runtime for mostly-homogeneous pipelines.

When a model has heterogeneous edges (embed, head) and a homogeneous
middle (repeated transformer blocks), this runtime routes the middle
through a fast ``shard_map``-based SPMD core while keeping the edges
as separate per-stage MPMD jits.

**Scope** (honest):

* Supports **linear pipelines only** — no schedules (GPipe / 1F1B / etc.).
  For scheduled pipelines, virtual stages (InterleavedH1, KimiK2) are
  the recommended way to pack homogeneous blocks onto the same physical
  rank.
* The core must be **consecutive** stages with the same param structure
  and the same stage ``fn``.
* Cross-rank transport between prefix / core / suffix uses
  ``jax.device_put`` just like the pure MPMD path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..types.stage import PipelineStage, _is_empty_state


def _detect_homogeneous_core(
    stages: tuple[PipelineStage, ...],
) -> tuple[int, int] | None:
    """Return ``(start, end)`` indices of the longest homogeneous run.

    Stages are homogeneous when their ``parameters`` have identical pytree
    structure, leaf shapes, and leaf dtypes, and their ``fn`` is the
    same callable object.
    """
    n = len(stages)
    if n < 3:
        return None

    def _signature(stage: PipelineStage) -> tuple[Any, int]:
        """Pack a stage's parameter shape/dtype tuple plus ``id(stage.fn)`` for run-detection."""
        leaves = jax.tree.leaves(stage.parameters)
        shapes = tuple(tuple(a.shape) for a in leaves)
        dtypes = tuple(str(a.dtype) for a in leaves)
        return (shapes, dtypes, id(stage.fn))

    sigs = [_signature(s) for s in stages]
    best_start, best_len = None, 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and sigs[j] == sigs[i]:
            j += 1
        run_len = j - i
        if run_len > best_len and run_len >= 2:
            best_start, best_len = i, run_len
        i = j

    if best_start is None:
        return None
    return best_start, best_start + best_len


def _make_hybrid_core_fwd_body(
    stage_fn: Callable[..., Any],
    n_core: int,
    pp_axis: str,
):
    """Build a ``shard_map`` body that runs a linear chain through ``n_core`` stages.

    Each device in the core sub-mesh executes one stage.  The last
    stage's output is broadcast to all devices via ``psum`` so the
    caller (outside ``shard_map``) can retrieve it without indexing.
    """

    def _drop0(t: Any) -> Any:
        """Index axis 0 of every leaf — collapses the stacked-stages axis to per-rank vars."""
        return jax.tree.map(lambda a: a[0], t)

    def body(stacked_params: Any, stacked_state: Any, x: jax.Array) -> jax.Array:
        """``shard_map`` body: walk ``n_core`` stages with ``ppermute`` between them.

        Each device on the pipeline axis acts as one stage. At step
        ``k`` the device with ``axis_index(pp_axis) == k`` runs the
        stage; all others ``cond`` into a no-op. After the step, the
        per-stage output is ``ppermute``-d to the next rank, completing
        the pipeline. The final output is broadcast to all ranks via
        ``psum`` so callers outside ``shard_map`` can read it
        regardless of which rank produced it.
        """
        rank = jax.lax.axis_index(pp_axis)
        p = _drop0(stacked_params)
        st = _drop0(stacked_state) if not _is_empty_state(stacked_state) else ()
        cur = x
        for step in range(n_core):
            is_active = rank == step

            def _run(args: tuple[Any, Any]) -> tuple[Any, Any]:
                """Run the stage on (carry, state); used for the active rank."""
                c, s = args
                return stage_fn(p, s, c)

            def _skip(args: tuple[Any, Any]) -> tuple[Any, Any]:
                """No-op branch: pass (carry, state) through; used for inactive ranks."""
                return args

            cur, st = jax.lax.cond(is_active, _run, _skip, (cur, st))
            if step < n_core - 1:
                cur = jax.lax.ppermute(
                    cur,
                    pp_axis,
                    perm=[(step, step + 1)],
                )

        out = jax.lax.psum(
            jnp.where(rank == n_core - 1, cur, jnp.zeros_like(cur)),
            pp_axis,
        )
        return out

    return body


def _build_core_mesh(
    full_mesh: Mesh,
    pp_axis: str,
    core_start: int,
    core_end: int,
) -> Mesh:
    """Slice a contiguous chunk of the pipeline axis into a sub-mesh."""
    pp_idx = full_mesh.axis_names.index(pp_axis)
    devices = np.take(
        full_mesh.devices,
        indices=np.arange(core_start, core_end),
        axis=pp_idx,
    )
    return Mesh(devices, full_mesh.axis_names)


def _stack_pytree_list(items: tuple[Any, ...]) -> Any:
    """Stack a tuple of identically-structured pytrees along a new leading axis."""
    if not items:
        return items
    first = items[0]
    leaves0, treedef = jax.tree.flatten(first)

    def stack_one(idx: int) -> jax.Array:
        """Gather leaf ``idx`` from every item and stack into a single array."""
        leaves = [jax.tree.flatten(item)[0][idx] for item in items]
        return jnp.stack(leaves, axis=0)

    stacked_leaves = [stack_one(i) for i in range(len(leaves0))]
    return jax.tree.unflatten(treedef, stacked_leaves)


def _unstack_pytree(stacked: Any, n: int) -> tuple[Any, ...]:
    """Inverse of :func:`_stack_pytree_list`."""
    leaves, treedef = jax.tree.flatten(stacked)
    sliced = tuple(tuple(leaf[i] for leaf in leaves) for i in range(n))
    return tuple(jax.tree.unflatten(treedef, list(sl)) for sl in sliced)


def _vmap_stage_fwd(stage_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Jitted vmap of the per-stage forward over the microbatch axis."""
    return jax.jit(jax.vmap(lambda p, s, x: stage_fn(p, s, x), in_axes=(None, None, 0)))


def _vmap_stage_bwd(stage_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Jitted vmap of the per-stage backward over the microbatch axis."""

    @jax.jit
    def bwd(params, state, x, g_y):
        """VJP of ``stage_fn`` returning ``(g_params, g_x)``.

        The state cotangent ``_g_s`` is intentionally discarded —
        SPMD pipeline stages here are stateless w.r.t. their carry,
        so the state's gradient is not propagated.
        """

        def fwd_only(p, s, xi):
            """Stage forward returning only the primary output ``y`` for ``jax.vjp``."""
            y, _ = stage_fn(p, s, xi)
            return y

        _, vjp_fn = jax.vjp(fwd_only, params, state, x)
        g_p, _g_s, g_x = vjp_fn(g_y)
        return g_p, g_x

    return jax.vmap(bwd, in_axes=(None, None, 0, 0))


def hybrid_linear_run(
    stages: tuple[PipelineStage, ...],
    inputs: tuple[Any, ...],
    *,
    mesh: Mesh,
    pp_axis: str,
    m: int = 1,
    mode: Literal["forward", "train"] = "forward",
    loss_fn: Callable[..., jax.Array] | None = None,
) -> Any:
    """Run a linear pipeline with a homogeneous SPMD core + MPMD edges.

    Args:
        stages: ``PipelineStage`` tuple.  A contiguous homogeneous
            subsequence is auto-detected; if none is found, falls back
            to the standard MPMD loop.
        inputs: ``(x, *targets)`` where ``x`` already has a leading
            microbatch axis of size ``m``.
        mesh: Full pipeline mesh (pp axis size == ``len(stages)``).
        pp_axis: Name of the manual pipeline axis.
        m: Number of microbatches along the leading axis of ``inputs[0]``.
        mode: ``"forward"`` or ``"train"``.
        loss_fn: Required for ``mode="train"``.

    Returns:
        Same shapes as the standard heterogeneous path:
        * ``mode="train"``   -> ``(loss_scalar, tuple[per_stage_grads])``
        * ``mode="forward"`` -> ``(final_output, tuple[per_rank_new_state])``
    """
    core_range = _detect_homogeneous_core(stages)
    if core_range is None:
        return None

    core_start, core_end = core_range
    n = len(stages)
    prefix_stages = stages[:core_start]
    core_stages = stages[core_start:core_end]
    suffix_stages = stages[core_end:]
    n_core = len(core_stages)

    xs = inputs[0]
    targets = inputs[1:]
    inv_m = float(1.0 / m)

    submeshes = [
        Mesh(
            np.take(mesh.devices, indices=[r], axis=mesh.axis_names.index(pp_axis)),
            mesh.axis_names,
        )
        for r in range(n)
    ]
    replicated = [NamedSharding(sm, PartitionSpec()) for sm in submeshes]

    cur = jax.device_put(xs, replicated[0] if prefix_stages else replicated[core_start])
    prefix_states: list[Any] = []
    prefix_inputs: list[Any] = []
    for r, stage in enumerate(prefix_stages):
        prefix_inputs.append(cur)
        y, st = _vmap_stage_fwd(stage.fn)(stage.parameters, stage.init_state, cur)
        prefix_states.append(st)
        cur = jax.device_put(y, replicated[r + 1])

    core_mesh = _build_core_mesh(mesh, pp_axis, core_start, core_end)
    core_params = _stack_pytree_list(tuple(s.parameters for s in core_stages))
    core_states = (
        _stack_pytree_list(tuple(s.init_state for s in core_stages))
        if not _is_empty_state(core_stages[0].init_state)
        else ()
    )

    params_spec = jax.tree.map(lambda _: PartitionSpec(pp_axis), core_params)
    states_spec = (
        jax.tree.map(lambda _: PartitionSpec(pp_axis), core_states)
        if not _is_empty_state(core_states)
        else PartitionSpec()
    )
    xs_spec = PartitionSpec()

    fwd_body = _make_hybrid_core_fwd_body(core_stages[0].fn, n_core, pp_axis)
    core_smap = shard_map(
        fwd_body,
        mesh=core_mesh,
        in_specs=(params_spec, states_spec, xs_spec),
        out_specs=PartitionSpec(),
        axis_names=frozenset({pp_axis}),
        check_vma=False,
    )
    core_fwd_jit = jax.jit(jax.vmap(core_smap, in_axes=(None, None, 0)))

    core_input = cur
    with core_mesh:
        core_output = core_fwd_jit(core_params, core_states, core_input)

    cur = jax.device_put(core_output, replicated[core_end] if suffix_stages else replicated[n - 1])
    suffix_states: list[Any] = []
    suffix_inputs: list[Any] = []
    for r, stage in enumerate(suffix_stages):
        idx = core_end + r
        suffix_inputs.append(cur)
        y, st = _vmap_stage_fwd(stage.fn)(stage.parameters, stage.init_state, cur)
        suffix_states.append(st)
        cur = jax.device_put(y, replicated[idx + 1] if idx + 1 < n else replicated[idx])

    final_output = cur

    if mode == "forward":
        all_states = prefix_states + list(_unstack_pytree(core_states, n_core)) + suffix_states
        return final_output, tuple(all_states)

    assert loss_fn is not None

    targets_on_last = tuple(jax.device_put(t, replicated[n - 1]) for t in targets)

    @jax.jit
    def loss_and_g_y(y_stack: jax.Array, *targs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Vmapped per-microbatch ``(loss, dloss/dy)`` over the stacked output."""

        def per_mb(yy: jax.Array, *tt: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Single-microbatch ``value_and_grad`` of ``loss_fn`` w.r.t. ``yy``."""
            return jax.value_and_grad(lambda y: loss_fn(y, *tt))(yy)

        return jax.vmap(per_mb)(y_stack, *targs)

    loss_stack, g_y = loss_and_g_y(final_output, *targets_on_last)
    loss_val = loss_stack.mean()

    suffix_grads: list[Any] = []
    for r in range(len(suffix_stages) - 1, -1, -1):
        idx = core_end + r
        stage = suffix_stages[r]
        g_params_stack, g_x_stack = _vmap_stage_bwd(stage.fn)(
            stage.parameters,
            stage.init_state,
            suffix_inputs[r],
            g_y,
        )
        g_params = jax.tree.map(lambda a: (a.sum(axis=0) * inv_m).astype(a.dtype), g_params_stack)
        suffix_grads.insert(0, g_params)
        if r > 0 or idx > 0:
            g_y = jax.device_put(g_x_stack, replicated[idx - 1])

    g_y_core = g_y

    @jax.jit
    def core_bwd(stacked_params: Any, stacked_state: Any, x: jax.Array, g_y_target: jax.Array) -> tuple[Any, jax.Array]:
        """VJP through the homogeneous core, returning ``(g_params, g_x)``."""

        def _core_forward(p: Any, x_in: jax.Array) -> jax.Array:
            """Core forward with stacked state closed over as nondiff data."""
            return fwd_body(p, stacked_state, x_in)

        _y, vjp_fn = jax.vjp(_core_forward, stacked_params, x)
        g_params, g_x = vjp_fn(g_y_target)
        return g_params, g_x

    core_bwd_vmap = jax.vmap(core_bwd, in_axes=(None, None, 0, 0))
    with core_mesh:
        core_grads_stacked_stack, g_y_stack = core_bwd_vmap(core_params, core_states, core_input, g_y_core)
    core_grads_stacked = jax.tree.map(lambda a: (a.sum(axis=0) * inv_m).astype(a.dtype), core_grads_stacked_stack)
    g_y = g_y_stack
    core_grads = list(_unstack_pytree(core_grads_stacked, n_core))

    prefix_grads: list[Any] = []
    for r in range(len(prefix_stages) - 1, -1, -1):
        stage = prefix_stages[r]
        g_params_stack, g_x_stack = _vmap_stage_bwd(stage.fn)(
            stage.parameters,
            stage.init_state,
            prefix_inputs[r],
            g_y,
        )
        g_params = jax.tree.map(lambda a: (a.sum(axis=0) * inv_m).astype(a.dtype), g_params_stack)
        prefix_grads.insert(0, g_params)
        if r > 0:
            g_y = jax.device_put(g_x_stack, replicated[r - 1])

    all_grads = prefix_grads + core_grads + suffix_grads
    return loss_val, tuple(all_grads)
