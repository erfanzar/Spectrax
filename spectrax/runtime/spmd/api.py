# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real MPMD pipeline primitive.

One ``pipeline_call`` covers training and forward-only inference
(``mode='forward'`` is the decode path too — "generate" is just
"forward with state threaded in and out"). Works for both
homogeneous (identical stage shapes) and heterogeneous (different
stage shapes) pipelines.

**Mechanism.** This is *not* a registered ``jax.extend.core.Primitive``
with a custom MLIR lowering. It's real MPMD built out of standard
JAX pieces:

* **Homogeneous path**: ``jax.shard_map`` with the MPMD axis **manual**
  (``axis_names={pp_axis}``) and every other mesh axis auto; inside the
  body, ``jax.lax.switch(axis_index(pp_axis), stage_branches)`` — each
  device's partitioned HLO retains only its rank's branch; cross-stage
  transport via ``jax.lax.ppermute``. One compiled HLO, real per-rank
  divergent code after XLA partitioning.

* **Heterogeneous path** (different param shapes per rank): each rank's
  params placed on its own sub-mesh; per-stage jits cached;
  cross-stage transport via ``jax.device_put``. Multi-program MPMD.

``pipeline_call`` detects which path applies and dispatches
automatically; the user-facing API is the same.

**Public API** ::

    loss, grads  = pipeline_call(stages, (x, y), mesh=mm,
                                 microbatches=4, mode="train",
                                 loss_fn=mse)

    out, state   = pipeline_call(stages, (x,), mesh=mm, mode="forward")

decode is just repeated forward with the returned state fed back in.

**Stage signature** ::

    fn(params, state, x) -> (y, new_state)

``state`` is any pytree the stage wants to carry across calls (KV
cache, hidden state, beam records). Use ``PipelineStage.init_state=()``
for stateless stages. The state comes *out* of every ``mode='forward'``
call and you can feed it back in on the next one — that's decode.

**Return shapes (both paths identical)**:

* ``mode="train"``   -> ``(loss_scalar, tuple[per_rank_grads])``
* ``mode="forward"`` -> ``(final_output, tuple[per_rank_new_state])``

TP / FSDP composition works via ``params_intra_spec`` on both paths:
pass a pytree of ``PartitionSpec`` s matching your stage's params
(homogeneous) or a tuple of such pytrees (heterogeneous, one per rank).

**Caches.** Two module-level caches (``_STEP_CACHE`` for the jitted
shard_map step, ``_PLACED_CACHE`` for stacked+placed params) keep
steady-state dispatch free of both retracing and repeated
``jax.device_put`` calls. A separate ``_HET_STAGE_JIT_CACHE`` does the
same job for the heterogeneous path's per-stage fwd/bwd jits.
"""

from __future__ import annotations

import functools as _functools
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as _np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax.nn import PipelineSequential

from ...core.state import State
from ...sharding.mesh import SpxMesh
from ..types.mesh import MpMdMesh, resolve_mpmd_mesh
from ..types.stage import PipelineStage, _is_empty_state
from .hybrid import hybrid_linear_run
from .runtime import spmd_run
from .shard_map import make_scheduled_body

__all__ = ["PipelineStage", "pipeline_call", "pipeline_step"]

_HET_STAGE_JIT_CACHE: dict[tuple[int, str], Callable[..., Any]] = {}
_STEP_CACHE: dict[Any, Callable[..., Any]] = {}
_PLACED_CACHE: dict[Any, tuple[Any, Any]] = {}
_SCHEDULED_STEP_CACHE: dict[Any, Callable[..., Any]] = {}


def _stack_pytree_list(items: tuple[Any, ...]) -> Any:
    """Stack a tuple of identically-structured pytrees along a new leading axis.

    Each leaf becomes a stacked array with leading dimension
    ``len(items)``; the pytree structure is preserved. Used to build
    the ``(pp, ...)``-leading pipeline parameter pytree from
    per-stage parameters before sharding the leading axis along the
    pipeline mesh axis.

    Args:
        items: Tuple of identically-structured pytrees.

    Returns:
        A single pytree with the same structure as ``items[0]`` but
        with each leaf an :func:`jnp.stack` of the corresponding
        leaves across ``items``.

    Raises:
        Any error :func:`jnp.stack` raises when leaves disagree on
        shape or dtype.
    """
    if not items:
        return items
    first = items[0]
    leaves0, treedef = jax.tree.flatten(first)

    def stack_one(idx: int) -> jax.Array:
        """Stack leaf ``idx`` across every item into a single array.

        Args:
            idx: Index into the flattened leaf tuple.

        Returns:
            A new array with shape ``(len(items),) + leaf.shape``.
        """
        leaves = [jax.tree.flatten(item)[0][idx] for item in items]
        return jnp.stack(leaves, axis=0)

    stacked_leaves = [stack_one(i) for i in range(len(leaves0))]
    return jax.tree.unflatten(treedef, stacked_leaves)


@_functools.partial(jax.jit, static_argnums=(1,))
def _unstack_leaves_jit(flat_leaves: tuple[jax.Array, ...], n: int) -> tuple[tuple[jax.Array, ...], ...]:
    """Jit-fused inverse of leaf stacking — slice every leaf along axis 0.

    One compiled kernel slices every leaf by ``n`` along axis 0.
    Without this, iterating ``leaf[i]`` for each of N stages x ~200
    leaves costs ~400 eager dispatches and several ms of Python
    overhead per training step.

    Args:
        flat_leaves: Flattened leaf tuple from
            :func:`jax.tree.flatten`.
        n: Number of stages to slice into. Marked static so XLA can
            fully unroll.

    Returns:
        A tuple of length ``n``, each element a tuple of ``n``-th
        slices of every leaf along axis 0.
    """
    return tuple(tuple(leaf[i] for leaf in flat_leaves) for i in range(n))


def _unstack_pytree(stacked: Any, n: int) -> tuple[Any, ...]:
    """Split a stacked pytree back into ``n`` independent pytrees.

    The inverse of :func:`_stack_pytree_list`. Slicing is delegated
    to :func:`_unstack_leaves_jit` so the per-leaf ``[i]`` ops fuse
    into a single compiled kernel.

    Args:
        stacked: A pytree whose leaves all have a leading dimension
            of size ``n``.
        n: Number of pytrees to recover.

    Returns:
        A tuple of ``n`` pytrees with the same structure as
        ``stacked`` but leaves shorter by one axis.
    """
    leaves, treedef = jax.tree.flatten(stacked)
    sliced_tuples = _unstack_leaves_jit(tuple(leaves), n)
    return tuple(jax.tree.unflatten(treedef, list(rank_leaves)) for rank_leaves in sliced_tuples)


def _homogeneous(items: tuple[Any, ...]) -> bool:
    """Return ``True`` iff every pytree in ``items`` matches structure and leaf avals.

    Cheap pre-check before attempting to stack: if any pytree differs
    in pytree structure, leaf count, or per-leaf shape/dtype, the
    fast SPMD ``shard_map`` path can't apply and the runtime falls
    back to per-rank MPMD. Comparison is structural, not by value.

    Args:
        items: Tuple of pytrees (typically per-stage parameters).

    Returns:
        ``True`` if all items can be stacked (same structure, shapes
        and dtypes); ``False`` otherwise.
    """
    if len(items) < 2:
        return True
    t0 = jax.tree.structure(items[0])
    l0 = jax.tree.leaves(items[0])
    for it in items[1:]:
        if jax.tree.structure(it) != t0:
            return False
        li = jax.tree.leaves(it)
        if len(li) != len(l0):
            return False
        for a, b in zip(l0, li, strict=False):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
    return True


def _make_body(
    stage_fns: tuple[Callable[..., Any], ...],
    n: int,
    m: int,
    pp_axis: str,
    mode: Literal["forward", "train"],
    loss_fn: Callable[..., jax.Array] | None,
):
    """Return the callable passed to ``shard_map`` for the homogeneous fast path.

    Inside the body, ``pp_axis`` is manual. ``stacked_params`` /
    ``stacked_state`` arrive pre-partitioned (leading axis size 1 per
    rank). The body drops that axis once up-front, runs the pipeline,
    and re-adds it on outputs that leave through a ``P(pp)`` spec.

    This runs one step of GPipe inside ``shard_map(pp-manual)``.

    Args:
        stage_fns: Per-stage forward callables. Length ``n``.
        n: Number of pipeline stages (== mesh's ``pp_axis`` size).
        m: Number of microbatches per step.
        pp_axis: Manual pipeline-parallel mesh axis name.
        mode: ``"forward"`` or ``"train"``.
        loss_fn: Required for ``"train"`` mode; ``(y, *targets) ->
            scalar``.

    Returns:
        A function suitable to pass to :func:`jax.shard_map` that
        accepts ``(stacked_params, stacked_state, xs, *extras)`` and
        returns ``(loss, grads)`` for train mode or ``(out, state)``
        for forward mode.
    """

    def _drop0(t: Any) -> Any:
        """Drop the size-1 leading pp axis from every leaf in ``t``."""
        return jax.tree.map(lambda a: a[0], t)

    def _add0(t: Any) -> Any:
        """Re-add a size-1 leading pp axis to every leaf in ``t``."""
        return jax.tree.map(lambda a: a[None, ...], t)

    def _dispatch_my_stage(local_params, local_state, x, rank, stage_extras=()):
        """Run this rank's own stage via ``lax.switch`` on the rank index.

        After XLA partitions on the manual ``pp`` axis, each device's
        compiled HLO retains only its matching branch — genuine
        per-rank different code despite the single shared HLO.

        Args:
            local_params: Per-rank parameters (post-:func:`_drop0`).
            local_state: Per-rank state (post-:func:`_drop0`) or
                ``()``.
            x: Activation flowing in.
            rank: ``jax.lax.axis_index(pp_axis)``.
            stage_extras: Optional extra positional args passed to
                every stage_fn.

        Returns:
            ``stage_fn(local_params, local_state, x, *stage_extras)``
            for the active rank.
        """
        branches = []
        for fn in stage_fns:

            def _branch(p_, s_, x_, _fn=fn, _extras=stage_extras):
                """Invoke the captured stage fn with this rank's inputs."""
                return _fn(p_, s_, x_, *_extras)

            branches.append(_branch)
        return jax.lax.switch(rank, branches, local_params, local_state, x)

    def _forward_chain(stacked_params, stacked_state, x, rank, stage_extras=()):
        """Run the linear pipeline: step ``i`` activates rank ``i`` then ppermutes.

        Every rank iterates the same Python loop; :func:`jax.lax.cond`
        gates the actual compute to the rank whose ``axis_index``
        matches the step, so non-active ranks contribute no work.
        After each step :func:`jax.lax.ppermute` shifts the
        activation to rank ``i + 1`` so it's in place for the next
        step's compute.

        Args:
            stacked_params: Stacked params with leading pp axis.
            stacked_state: Stacked state with leading pp axis or
                ``()``.
            x: Initial activation (flows in on rank 0).
            rank: ``axis_index(pp_axis)``.
            stage_extras: Extras forwarded to
                :func:`_dispatch_my_stage`.

        Returns:
            ``(cur, st)`` — the final activation and the per-rank
            updated state. After the loop ``cur`` is the activation
            that rank ``n - 1`` produced; the caller broadcasts it
            to every rank.
        """
        p = _drop0(stacked_params)
        st = _drop0(stacked_state) if not _is_empty_state(stacked_state) else ()
        cur = x
        for step in range(n):
            is_active = rank == step

            def _run(args):
                """Dispatch the active rank's stage for this step."""
                c, s = args
                return _dispatch_my_stage(p, s, c, rank, stage_extras)

            def _skip(args):
                """Pass-through for ranks that aren't active on this step."""
                return args

            cur, st = jax.lax.cond(is_active, _run, _skip, (cur, st))
            if step < n - 1:
                cur = jax.lax.ppermute(
                    cur,
                    pp_axis,
                    perm=[(step, step + 1)],
                )
        return cur, st

    def body(stacked_params, stacked_state, xs, *extras):
        """Per-rank shard_map body: forward chain then optional loss/VJP.

        In ``forward`` mode, rank ``n - 1``'s final activation is
        broadcast to all ranks via ``all_gather + index`` so
        ``out_specs=P()`` (replicated) is satisfied; XLA DCEs the
        unused ranks.

        In ``train`` mode, only rank ``n - 1`` actually runs
        ``loss_fn``; non-last ranks short-circuit to zero via
        :func:`jax.lax.cond`. The :func:`jax.lax.psum` then
        broadcasts rank ``n - 1``'s loss value to every rank so the
        VJP flows correctly back through the ppermute chain. The loss
        value is forced to ``float32`` on both branches so cond's
        shape/dtype match regardless of what dtype ``loss_fn``
        returns on bf16 inputs (common source of ``lax.cond``
        mismatches).

        Args:
            stacked_params: Stacked parameters (leading ``pp`` axis,
                size 1 inside the body).
            stacked_state: Stacked state, or ``()``.
            xs: Stacked microbatched inputs (leading mb axis).
            *extras: Stacked extras forwarded to either each stage
                (forward mode) or ``loss_fn`` (train mode).

        Returns:
            * forward: ``(final_stack, state_stack)``.
            * train: ``(loss_scalar, stacked_params_grads)``.
        """
        rank = jax.lax.axis_index(pp_axis)
        last = n - 1

        if mode == "forward":

            def per_mb(x, *t):
                """Forward-only microbatch body with broadcast-to-replicated output.

                Args:
                    x: One microbatch's input.
                    *t: One microbatch's extras (forwarded to stages).

                Returns:
                    ``(out, st)`` — output (broadcast across ranks)
                    and per-rank updated state.
                """
                final, st = _forward_chain(stacked_params, stacked_state, x, rank, t)
                gathered = jax.lax.all_gather(final, pp_axis)
                out = gathered[last]
                if not _is_empty_state(st):
                    st = _add0(st)
                return out, st

            final_stack, state_stack = jax.vmap(per_mb)(xs, *extras)
            return final_stack, state_stack

        assert loss_fn is not None

        def per_mb_loss(params, x, *t):
            """Per-microbatch scalar loss, broadcast to all ranks via psum.

            Only rank ``n - 1`` runs ``loss_fn``; other ranks
            contribute zero. The :func:`jax.lax.psum` makes the
            terminal-rank loss visible on every rank so backward
            VJP through ppermute is correct.

            Args:
                params: Stacked parameters.
                x: One microbatch's input.
                *t: One microbatch's targets.

            Returns:
                The microbatch's loss as a ``float32`` scalar.
            """
            final, _ = _forward_chain(params, stacked_state, x, rank)
            is_last = rank == last
            per_rank = jax.lax.cond(
                is_last,
                lambda: loss_fn(final, *t).astype(jnp.float32),
                lambda: jnp.zeros((), dtype=jnp.float32),
            )
            return jax.lax.psum(per_rank, pp_axis)

        def mean_loss(params):
            """Mean of per-microbatch losses (vmapped over the leading mb axis).

            Args:
                params: Stacked parameters (the variable being
                    differentiated).

            Returns:
                Scalar mean loss across all microbatches.
            """
            per_mb_losses = jax.vmap(per_mb_loss, in_axes=(None, 0, *([0] * len(extras))))(params, xs, *extras)
            return per_mb_losses.mean()

        loss_val, g_params = jax.value_and_grad(mean_loss)(stacked_params)
        return loss_val, g_params

    return body


def _rank_submesh(mesh: Mesh, pp_axis: str, rank: int) -> Mesh:
    """Return a sub-mesh containing only ``rank``'s slice along ``pp_axis``.

    Used by the heterogeneous path to place each rank's parameters
    onto its own per-stage device subset. The returned mesh keeps
    every original axis (the ``pp_axis`` is just size 1 in the
    slice).

    Args:
        mesh: Full pipeline mesh.
        pp_axis: Name of the pipeline axis.
        rank: Pipeline rank to slice out.

    Returns:
        A :class:`~jax.sharding.Mesh` of the same rank as ``mesh``
        but with size 1 on ``pp_axis``.
    """
    pp_idx = mesh.axis_names.index(pp_axis)
    devs = _np.take(mesh.devices, indices=[rank], axis=pp_idx)
    return Mesh(devs, mesh.axis_names)


def _het_get_fwd(stage_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return the cached jitted forward for ``stage_fn``.

    Caches by the Python identity of ``stage_fn`` so repeated calls
    for the same stage hit the cache; ``jax.jit`` itself keys on
    function identity so this layering is consistent.

    Args:
        stage_fn: Per-stage forward callable
            ``(params, state, x) -> (y, new_state)``.

    Returns:
        A jitted callable with the same signature as ``stage_fn``.
    """
    key = (id(stage_fn), "fwd")
    cached = _HET_STAGE_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def fwd(params, state, x):
        """Jitted forward: call ``stage_fn`` directly."""
        return stage_fn(params, state, x)

    _HET_STAGE_JIT_CACHE[key] = fwd
    return fwd


def _het_get_bwd(stage_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return the cached jitted backward for ``stage_fn``.

    The backward returns ``(g_params, g_x)`` via :func:`jax.vjp`
    against a forward that only emits ``y`` (the ``new_state``
    output is silently dropped because the heterogeneous training
    path doesn't differentiate through the per-stage state).

    Args:
        stage_fn: Per-stage forward callable.

    Returns:
        A jitted callable
        ``(params, state, x, g_y) -> (g_params, g_x)``.
    """
    key = (id(stage_fn), "bwd")
    cached = _HET_STAGE_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def bwd(params, state, x, g_y):
        """Jitted VJP: returns ``(g_params, g_x)``, drops ``g_state``."""

        def fwd_only(p, s, xi):
            """Stage forward returning only ``y`` (drops ``new_state``)."""
            y, _ = stage_fn(p, s, xi)
            return y

        _, vjp_fn = jax.vjp(fwd_only, params, state, x)
        g_p, _g_s, g_x = vjp_fn(g_y)
        return g_p, g_x

    _HET_STAGE_JIT_CACHE[key] = bwd
    return bwd


def _build_per_rank_placements(
    mesh: Mesh,
    pp_axis: str,
    n: int,
    params_list: tuple[Any, ...],
    params_intra_spec: Any,
) -> list[Any]:
    """Build per-rank :class:`NamedSharding` pytrees for the heterogeneous path.

    Each rank gets a sub-mesh (size 1 on ``pp_axis``) plus an
    intra-stage spec describing how to shard the rank's parameters
    across the remaining axes.

    ``params_intra_spec`` interpretation:

    * ``None`` -> replicate every leaf on its rank's sub-mesh.
    * A single pytree matching ``params_list[0]``'s structure ->
      apply uniformly across ranks (homogeneous intra-stage
      sharding).
    * A tuple of per-rank pytrees of length ``n`` -> apply the
      rank's entry to that rank's params (heterogeneous intra-stage
      sharding).

    Args:
        mesh: Full pipeline mesh.
        pp_axis: Name of the pipeline axis.
        n: Number of ranks (and entries in ``params_list``).
        params_list: Per-rank parameter pytrees.
        params_intra_spec: Optional intra-stage sharding spec(s).

    Returns:
        A list of length ``n`` whose entries are pytrees of
        :class:`NamedSharding` matching ``params_list[r]``'s
        structure.
    """
    submeshes = [_rank_submesh(mesh, pp_axis, r) for r in range(n)]

    per_rank_intra: list[Any]
    if params_intra_spec is None:
        per_rank_intra = [None] * n
    elif isinstance(params_intra_spec, tuple) and len(params_intra_spec) == n:
        per_rank_intra = list(params_intra_spec)
    else:
        per_rank_intra = [params_intra_spec] * n

    placements = []
    for r in range(n):
        sm = submeshes[r]
        intra = per_rank_intra[r]
        if intra is None:
            placements.append(
                jax.tree.map(
                    lambda _, _sm=sm: NamedSharding(_sm, PartitionSpec()),
                    params_list[r],
                )
            )
        else:

            def _to_ns(spec, _sm=sm):
                """Convert a PartitionSpec (or tuple/None) to a NamedSharding on ``_sm``.

                Args:
                    spec: ``None`` for replicated, a
                        :class:`PartitionSpec`, or a sequence of axis
                        names to wrap in one.
                    _sm: Captured per-rank sub-mesh.

                Returns:
                    A :class:`NamedSharding` on ``_sm``.
                """
                if spec is None:
                    return NamedSharding(_sm, PartitionSpec())
                if isinstance(spec, PartitionSpec):
                    return NamedSharding(_sm, spec)
                return NamedSharding(_sm, PartitionSpec(*spec))

            placements.append(jax.tree.map(_to_ns, intra))
    return placements


def _heterogeneous_call(
    *,
    stages,
    stage_fns: tuple[Callable[..., Any], ...],
    params_list: tuple[Any, ...],
    states_list: tuple[Any, ...],
    xs: jax.Array,
    extras: tuple[jax.Array, ...],
    mesh: Mesh,
    pp_axis: str,
    m: int,
    mode: Literal["forward", "train"],
    loss_fn: Callable[..., jax.Array] | None,
    params_intra_spec: Any,
) -> Any:
    """Per-rank multi-jit MPMD for heterogeneous stages.

    Each rank's stage compiles to its own program. Cross-stage
    transport is ``jax.device_put`` between sub-meshes. Per-stage
    jits and placements are cached so steady-state dispatch hits
    those caches.

    Different param shapes per rank (embed + blocks + lm_head) mean we
    can't stack on a leading PP axis, so each rank's params live on its
    own sub-mesh, the runtime issues per-stage jits orchestrated in
    Python, and cross-stage transport is ``jax.device_put`` between
    sub-meshes.
    """
    n = len(stages)
    submeshes = [_rank_submesh(mesh, pp_axis, r) for r in range(n)]
    replicated = [NamedSharding(sm, PartitionSpec()) for sm in submeshes]
    param_shardings = _build_per_rank_placements(
        mesh,
        pp_axis,
        n,
        params_list,
        params_intra_spec,
    )

    placed_params = [
        jax.tree.map(lambda x, ns: jax.device_put(x, ns), p, s)
        for p, s in zip(params_list, param_shardings, strict=True)
    ]
    placed_states = [
        jax.tree.map(lambda x, _r=r: jax.device_put(x, replicated[_r]), s) if not _is_empty_state(s) else ()
        for r, s in enumerate(states_list)
    ]

    def _vmap_fwd(stage_fn):
        """Jitted vmap of the per-stage forward over the microbatch axis.

        Uses ``in_axes=(None, None, 0)`` so params and state are
        broadcast across the microbatch axis while the input ``x`` is
        sliced.

        Args:
            stage_fn: Per-stage forward callable.

        Returns:
            A jitted callable that processes a stack of microbatches
            in one HLO call.
        """
        return jax.jit(jax.vmap(_het_get_fwd(stage_fn), in_axes=(None, None, 0)))

    def _vmap_bwd(stage_fn):
        """Jitted vmap of the per-stage backward over the microbatch axis.

        ``in_axes=(None, None, 0, 0)`` keeps params and state shared
        and slices both the saved input and the per-microbatch
        cotangent.

        Args:
            stage_fn: Per-stage forward callable.

        Returns:
            A jitted vmap'd backward returning ``(g_params_stack,
            g_x_stack)``; ``g_params_stack`` is summed over the
            microbatch axis by the caller.
        """
        return jax.jit(jax.vmap(_het_get_bwd(stage_fn), in_axes=(None, None, 0, 0)))

    vfwds = [_vmap_fwd(fn) for fn in stage_fns]
    vbwds = [_vmap_bwd(fn) for fn in stage_fns]

    saved_inputs: list[Any] = [None] * n
    cur = jax.device_put(xs, replicated[0])
    new_states: list[Any] = list(placed_states)
    with mesh:
        for r in range(n):
            saved_inputs[r] = cur
            y, new_st = vfwds[r](placed_params[r], placed_states[r], cur)
            new_states[r] = new_st
            cur = jax.device_put(y, replicated[r + 1]) if r < n - 1 else y

        if mode == "forward":
            return cur, tuple(new_states)

        assert loss_fn is not None
        targets_on_dev = tuple(jax.device_put(t, replicated[n - 1]) for t in extras)

        @jax.jit
        def loss_and_g_y(y_stack, *t_stack):
            """Vmap of ``(loss, d_loss/d_y)`` over the microbatch axis."""

            def per_mb(y_, *t_):
                """Per-microbatch ``(loss, g_y)`` via value_and_grad."""
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            return jax.vmap(per_mb)(y_stack, *t_stack)

        loss_stack, g_y = loss_and_g_y(cur, *targets_on_dev)
        loss = loss_stack.mean()

        inv_m = float(1.0 / m)
        grads_per_rank: list[Any] = [None] * n
        for r in range(n - 1, -1, -1):
            g_params_stack, g_x_stack = vbwds[r](
                placed_params[r],
                placed_states[r],
                saved_inputs[r],
                g_y,
            )
            grads_per_rank[r] = jax.tree.map(
                lambda a: (a.sum(axis=0) * inv_m).astype(a.dtype),
                g_params_stack,
            )
            if r > 0:
                g_y = jax.device_put(g_x_stack, replicated[r - 1])

        return loss, tuple(grads_per_rank)


def _maybe_squeeze_mb(x: Any, m: int) -> Any:
    """Strip the leading microbatch axis from every leaf when ``m == 1``.

    The runtime always reshapes the user's input into ``(M, ...)``
    even when ``M == 1`` so the per-microbatch vmap works uniformly.
    On the way out we strip the meaningless size-1 axis so
    ``m == 1`` callers don't have to ``squeeze`` themselves.

    Args:
        x: Pytree of arrays (typically the final pipeline output).
        m: Number of microbatches the runtime used.

    Returns:
        ``x`` with leaf leading axis dropped iff ``m == 1`` and the
        leaf has a leading size-1 axis; otherwise ``x`` unchanged.
    """
    if m == 1:
        return jax.tree.map(lambda a: a[0] if hasattr(a, "shape") and a.shape and a.shape[0] == 1 else a, x)
    return x


def pipeline_call(
    stages: list[PipelineStage] | tuple[PipelineStage, ...],
    inputs: tuple[Any, ...],
    *,
    mesh: SpxMesh | MpMdMesh,
    microbatches: int = 1,
    mode: Literal["forward", "train"] = "forward",
    loss_fn: Callable[..., jax.Array] | None = None,
    params_intra_spec: Any = None,
    schedule: Any = None,
) -> Any:
    """Execute one pipeline step — train or forward.

    ``mode="forward"`` is both inference and the decode primitive: the
    returned ``tuple[per_rank_state]`` is just whatever the stages put
    in ``state`` on this call. To decode, feed those states back in
    next call (see :doc:`examples/04_decode_loop`).

    Args:
        stages: ``PipelineStage`` per rank. Homogeneous (identical
            param structure/shape) hits the single-HLO fast path;
            heterogeneous drops to multi-jit MPMD. Both paths return
            the same shape.
        inputs: ``(x, *targets)``. ``x`` is the pipeline input,
            microbatched along its leading axis. For ``mode="train"``
            the trailing elements are passed to ``loss_fn`` per
            microbatch.
        mesh: An :class:`SpxMesh` (preferred) or :class:`MpMdMesh`
            whose ``mpmd_dim`` equals ``len(stages)``. The MPMD axis
            is manual inside the step; every other axis stays auto
            so intra-stage TP/FSDP/DP composes.
        microbatches: Number of microbatches along the leading axis of
            ``inputs[0]``. ``1`` (default) = no microbatching; the
            leading size-1 axis is squeezed off the return.
        mode: ``"forward"`` or ``"train"``.
        loss_fn: ``(final_out, *targets) -> scalar``. Required for
            ``mode="train"``.
        params_intra_spec: Optional intra-stage sharding. Either a
            pytree of :class:`PartitionSpec` s matching one stage's
            params (applies uniformly — homogeneous intra-stage spec)
            or a tuple of such pytrees matching ``stages`` (per-rank
            intra-stage spec — heterogeneous).

    Returns:
        * ``mode="train"``   -> ``(loss_scalar, tuple[per_rank_grads])``
        * ``mode="forward"`` -> ``(final_output, tuple[per_rank_new_state])``

        Per-rank pytrees are returned as a **tuple**, homogeneous or
        heterogeneous. Apply optimizer updates uniformly by iterating
        the tuple.

    The ``mesh`` is unwrapped once to the underlying jax.sharding.Mesh
    so the rest of the body (shard_map, NamedSharding, sub-mesh
    helpers) works against the raw mesh. Homogeneous params hit the
    single-HLO fast path via ``shard_map`` with the manual pp axis;
    heterogeneous params drop to multi-jit MPMD. For the homogeneous
    path, in_specs / out_specs only mention the manual axis — intra-
    stage TP/FSDP comes from the concrete input shardings. Return
    shape is unified with the heterogeneous path: a tuple of per-rank
    pytrees, not a single stacked pytree.
    """
    mpmd_mesh = resolve_mpmd_mesh(mesh)
    mesh = mpmd_mesh.jax_mesh
    pp_axis = mpmd_mesh.mpmd_axis_name
    n = len(stages)
    if schedule is not None:
        V_expected = schedule.virtual_stages_per_rank()
        expected_stages = mpmd_mesh.mpmd_dim * V_expected
        if n != expected_stages:
            raise ValueError(
                f"schedule {type(schedule).__name__} with "
                f"virtual_stages_per_rank={V_expected} on "
                f"mpmd_dim={mpmd_mesh.mpmd_dim} expects "
                f"{expected_stages} stages; got {n}."
            )
    elif mpmd_mesh.mpmd_dim != n:
        raise ValueError(f"mesh.mpmd_dim={mpmd_mesh.mpmd_dim} but got {n} stages.")
    if mode == "train" and loss_fn is None:
        raise ValueError("loss_fn required for mode='train'.")

    m = microbatches

    def _mb(x: jax.Array) -> jax.Array:
        """Reshape ``x`` (leading axis = batch) into ``(m, batch // m, ...)``.

        Args:
            x: Array whose leading axis is the global batch.

        Returns:
            ``x`` reshaped to expose a microbatch axis as the new
            leading dimension.

        Raises:
            ValueError: If the batch size isn't divisible by ``m``.
        """
        b = x.shape[0]
        if b % m:
            raise ValueError(f"leading dim {b} not divisible by microbatches {m}.")
        return x.reshape(m, b // m, *x.shape[1:])

    xs = _mb(inputs[0])
    extras = tuple(_mb(t) for t in inputs[1:])

    stage_fns = tuple(s.fn for s in stages)
    params_list = tuple(s.parameters for s in stages)
    states_list = tuple(s.init_state for s in stages)

    if not _homogeneous(params_list):
        if schedule is not None:
            raise ValueError(
                "schedule= requires homogeneous stages (same params "
                "structure on every rank). Heterogeneous-stage "
                "pipelines still use the non-scheduled fallback."
            )
        hybrid_result = hybrid_linear_run(
            tuple(stages),
            (xs, *extras),
            mesh=mesh,
            pp_axis=pp_axis,
            m=m,
            mode=mode,
            loss_fn=loss_fn,
        )
        if hybrid_result is not None:
            if mode != "train":
                final, state = hybrid_result
                return _maybe_squeeze_mb(final, m), state
            return hybrid_result
        out = _heterogeneous_call(
            stages=stages,
            stage_fns=stage_fns,
            params_list=params_list,
            states_list=states_list,
            xs=xs,
            extras=extras,
            mesh=mesh,
            pp_axis=pp_axis,
            m=m,
            mode=mode,
            loss_fn=loss_fn,
            params_intra_spec=params_intra_spec,
        )
        if mode != "train":
            final, state = out
            return _maybe_squeeze_mb(final, m), state
        return out

    if schedule is not None:
        if mode != "train":
            raise ValueError(
                f"schedule= is only wired for mode='train'; got mode={mode!r}. "
                f"Use mode='train' with a loss_fn, or drop schedule= for "
                f"forward-only inference (GPipe default)."
            )
        if loss_fn is None:
            raise ValueError("schedule= with mode='train' requires loss_fn.")
        return _scheduled_call(
            stages=stages,
            stage_fns=stage_fns,
            params_list=params_list,
            xs=xs,
            extras=extras,
            mesh=mesh,
            pp_axis=pp_axis,
            mpmd_dim=mpmd_mesh.mpmd_dim,
            m=m,
            loss_fn=loss_fn,
            schedule=schedule,
            params_intra_spec=params_intra_spec,
        )

    cache_key = (
        tuple(id(fn) for fn in stage_fns),
        id(mesh),
        pp_axis,
        m,
        mode,
        id(loss_fn) if loss_fn else 0,
    )
    step_fn = _STEP_CACHE.get(cache_key)
    if step_fn is None:
        body = _make_body(stage_fns, n, m, pp_axis, mode, loss_fn)

        params_spec = jax.tree.map(
            lambda _: PartitionSpec(pp_axis),
            params_list[0],
        )
        states_spec = (
            jax.tree.map(lambda _: PartitionSpec(pp_axis), states_list[0])
            if not _is_empty_state(states_list[0])
            else PartitionSpec()
        )
        xs_spec = PartitionSpec()
        extras_specs = tuple(PartitionSpec() for _ in extras)

        if mode == "train":
            out_specs = (
                PartitionSpec(),
                jax.tree.map(lambda _: PartitionSpec(pp_axis), params_list[0]),
            )
        else:
            out_specs = (PartitionSpec(), states_spec)

        smap = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(params_spec, states_spec, xs_spec, *extras_specs),
            out_specs=out_specs,
            axis_names=frozenset({pp_axis}),
            check_vma=False,
        )
        step_fn = jax.jit(smap)
        _STEP_CACHE[cache_key] = step_fn

    placed_key = (
        tuple(id(p) for p in params_list),
        id(mesh),
        pp_axis,
    )
    placed = _PLACED_CACHE.get(placed_key)
    if placed is None:
        stacked_params = _stack_pytree_list(params_list)
        stacked_states = _stack_pytree_list(states_list) if not _is_empty_state(states_list[0]) else ()
        if params_intra_spec is None:
            leaf_specs = jax.tree.map(
                lambda _: PartitionSpec(pp_axis),
                stacked_params,
            )
        else:

            def _prepend(intra):
                """Prepend the pp axis to a leaf's intra-stage spec.

                Stacked params have a leading ``pp`` axis (one entry
                per stage); intra-stage sharding axes follow.

                Args:
                    intra: Intra-stage :class:`PartitionSpec` (or
                        ``None`` / sequence) for this leaf.

                Returns:
                    A :class:`PartitionSpec` whose first axis is
                    ``pp_axis`` followed by the intra-stage spec.
                """
                if intra is None or intra == PartitionSpec():
                    return PartitionSpec(pp_axis)
                if isinstance(intra, PartitionSpec):
                    return PartitionSpec(pp_axis, *intra)
                return PartitionSpec(pp_axis, *intra)

            leaf_specs = jax.tree.map(_prepend, params_intra_spec)
        stacked_params = jax.tree.map(
            lambda x, sp: jax.device_put(x, NamedSharding(mesh, sp)),
            stacked_params,
            leaf_specs,
        )
        if not _is_empty_state(stacked_states):
            pp_sharding = NamedSharding(mesh, PartitionSpec(pp_axis))
            stacked_states = jax.tree.map(
                lambda x: jax.device_put(x, pp_sharding),
                stacked_states,
            )
        _PLACED_CACHE[placed_key] = (stacked_params, stacked_states)
    else:
        stacked_params, stacked_states = placed

    with mesh:
        result = step_fn(stacked_params, stacked_states, xs, *extras)

    if mode == "train":
        loss, g_stacked = result
        return loss, _unstack_pytree(g_stacked, n)
    final, state = result
    if not _is_empty_state(state):
        state = _unstack_pytree(state, n)
    else:
        state = tuple(() for _ in range(n))
    return _maybe_squeeze_mb(final, m), state


def _scheduled_call(
    *,
    stages,
    stage_fns: tuple[Callable[..., Any], ...],
    params_list: tuple[Any, ...],
    xs: jax.Array,
    extras: tuple[jax.Array, ...],
    mesh,
    pp_axis: str,
    mpmd_dim: int,
    m: int,
    loss_fn: Callable[..., jax.Array],
    schedule: Any,
    params_intra_spec: Any,
):
    """Run one scheduled training step via :func:`make_scheduled_body`.

    Homogeneous pipeline only. For flat schedules (V=1) ``stages`` must
    have ``mpmd_dim`` entries aligned with physical ranks. For
    virtual-stage schedules (V>1) ``stages`` must have ``mpmd_dim * V``
    entries — the runtime re-stacks them into a ``(pp, V, ...)`` param
    pytree respecting :meth:`Schedule.logical_at`.

    Args:
        stages: Homogeneous :class:`PipelineStage` list.
        stage_fns: Per-stage forward callables (assumed equivalent).
        params_list: Per-stage params pytrees (already homogeneity-checked).
        xs: Microbatched model input, shape ``(M, per_mb_batch, ...)``.
        extras: Microbatched targets.
        mesh: Underlying :class:`jax.sharding.Mesh`.
        pp_axis: Name of the manual pipeline-parallel mesh axis.
        mpmd_dim: Number of physical pipeline ranks.
        m: Microbatches per step.
        loss_fn: ``(y, *targets) -> scalar``.
        schedule: A :class:`~spectrax.runtime.schedules.Schedule`.
        params_intra_spec: Optional intra-stage sharding spec.

    Returns:
        ``(loss_scalar, per_rank_grads)``.
    """
    V = schedule.virtual_stages_per_rank()
    n_logical = mpmd_dim * V
    if len(stages) != n_logical:
        raise ValueError(
            f"schedule {type(schedule).__name__} has "
            f"virtual_stages_per_rank={V} on mpmd_dim={mpmd_dim}, "
            f"expecting {n_logical} stages; got {len(stages)}."
        )

    stage_fn = stage_fns[0]

    def fwd_fn(params, x):
        """Apply one stage's forward pass (stateless); drops any stage state output.

        Used by :func:`make_scheduled_body` as the per-microbatch
        forward primitive. The ``init_state`` is hard-coded to ``()``
        because scheduled SPMD pipelines treat stages as stateless
        (state-carrying scheduled pipelines are not yet supported).

        Args:
            params: One stage's parameters.
            x: Input activation.

        Returns:
            ``y`` only (the ``new_state`` half of ``stage_fn`` is
            discarded).
        """
        y, _ = stage_fn(params, (), x)
        return y

    def bwd_fn(params, x, g_y):
        """VJP of :func:`fwd_fn` — returns ``(g_params, g_x)``.

        Args:
            params: One stage's parameters.
            x: Saved forward input.
            g_y: Cotangent at the stage's output.

        Returns:
            ``(g_params, g_x)`` — gradients flowing back to params
            and to the upstream activation.
        """

        def _only_y(p, xi):
            """Stage forward returning only the primary output ``y`` (drops stage state)."""
            y, _ = stage_fn(p, (), xi)
            return y

        _, vjp = jax.vjp(_only_y, params, x)
        g_p, g_x = vjp(g_y)
        return g_p, g_x

    def loss_and_g_y(y, *t):
        """Return ``(loss_scalar, g_y)`` via :func:`jax.value_and_grad` on ``loss_fn``.

        Used at the terminal stage to seed the backward chain with
        the loss's cotangent ``dloss/dy``.

        Args:
            y: Final-stage output for one microbatch.
            *t: Targets / auxiliary args forwarded to ``loss_fn``.

        Returns:
            ``(loss_scalar, g_y)``.
        """

        def _loss(yy):
            """Closure: ``loss_fn(yy, *targets)`` for ``jax.value_and_grad``."""
            return loss_fn(yy, *t)

        return jax.value_and_grad(_loss)(y)

    if V == 1:
        rank_stacked_params = params_list
    else:
        rank_buckets: list[list[Any]] = [[None] * V for _ in range(mpmd_dim)]
        for logical in range(n_logical):
            r_ = 0
            v_ = 0
            found = False
            for r in range(mpmd_dim):
                for v in range(V):
                    if schedule.logical_at(r, v, mpmd_dim) == logical:
                        r_, v_ = r, v
                        found = True
                        break
                if found:
                    break
            rank_buckets[r_][v_] = params_list[logical]
        for r in range(mpmd_dim):
            for v in range(V):
                if rank_buckets[r][v] is None:
                    raise ValueError(
                        f"schedule {type(schedule).__name__}.logical_at "
                        f"does not map any logical stage to rank={r}, "
                        f"virt={v}; stages and schedule disagree on "
                        f"layout."
                    )
        rank_stacked_params = tuple(_stack_pytree_list(tuple(rank_buckets[r])) for r in range(mpmd_dim))

    stacked_params = _stack_pytree_list(rank_stacked_params)

    cache_key = (
        id(stage_fn),
        id(loss_fn),
        type(schedule),
        schedule.microbatches,
        mpmd_dim,
        m,
        pp_axis,
        id(mesh),
        V,
    )
    step_fn = _SCHEDULED_STEP_CACHE.get(cache_key)
    if step_fn is None:
        body = make_scheduled_body(
            schedule=schedule,
            n_stages=mpmd_dim,
            microbatches=m,
            pp_axis=pp_axis,
            fwd_fn=fwd_fn,
            bwd_fn=bwd_fn,
            loss_and_g_y=loss_and_g_y,
            mode="train",
            checkpoint_policy=True,
        )
        params_spec = jax.tree.map(
            lambda _: PartitionSpec(pp_axis),
            stacked_params,
        )
        xs_spec = PartitionSpec()
        extras_specs = tuple(PartitionSpec() for _ in extras)
        out_specs = (PartitionSpec(), params_spec)
        smap = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(params_spec, xs_spec, *extras_specs),
            out_specs=out_specs,
            axis_names=frozenset({pp_axis}),
            check_vma=False,
        )
        step_fn = jax.jit(smap)
        _SCHEDULED_STEP_CACHE[cache_key] = step_fn

    pp_sharding_tree = jax.tree.map(
        lambda _: NamedSharding(mesh, PartitionSpec(pp_axis)),
        stacked_params,
    )
    stacked_params = jax.tree.map(jax.device_put, stacked_params, pp_sharding_tree)

    with mesh:
        loss, g_stacked = step_fn(stacked_params, xs, *extras)
    if V == 1:
        per_rank_grads = _unstack_pytree(g_stacked, mpmd_dim)
    else:
        per_rank_stacks = _unstack_pytree(g_stacked, mpmd_dim)
        per_rank_grads_list: list[Any] = [None] * n_logical
        for r in range(mpmd_dim):
            virt_stack = _unstack_pytree(per_rank_stacks[r], V)
            for v in range(V):
                logical = schedule.logical_at(r, v, mpmd_dim)
                per_rank_grads_list[logical] = virt_stack[v]
        per_rank_grads = tuple(per_rank_grads_list)
    return loss, per_rank_grads


def pipeline_step(
    model: PipelineSequential,
    batch: tuple[Any, ...],
    *,
    mesh: SpxMesh | MpMdMesh | Mesh,
    axis: str = "pp",
    schedule: Any,
    loss_fn: Callable[..., Any],
) -> tuple[Any, tuple[State, ...]]:
    """Execute one pipeline-parallel forward + backward step.

    Thin wrapper over :func:`~spectrax.runtime.spmd.spmd_run`
    providing the primary user-facing API. Runs ``schedule`` over
    ``mesh[axis]`` devices (one stage per device), microbatches the
    ``batch`` along the leading axis, and returns the mean loss plus
    per-stage gradient :class:`State` s.

    Args:
        model: :class:`PipelineSequential` whose ``num_stages`` must
            equal the mesh's ``axis`` dimension. All stages must
            share a structurally identical ``GraphDef``.
        batch: Tuple of positional tensors. The first element is the
            pipeline input (flows through stages); remaining elements
            are targets / auxiliary arguments passed to ``loss_fn``
            on the final stage.
        mesh: :class:`jax.sharding.Mesh`.
        axis: Named axis of ``mesh`` reserved for pipeline stages.
        schedule: One of :class:`GPipe`, :class:`Std1F1B`,
            :class:`ZeroBubbleH1`, :class:`InterleavedH1`.
        loss_fn: Scalar loss. Called as ``loss_fn(final_stage_output,
            *batch[1:])`` on each microbatch. Returned values are
            mean-reduced over microbatches.

    Returns:
        ``(loss, per_stage_grads)`` where ``per_stage_grads`` is a
        tuple of :class:`State` s (one per stage) suitable for
        feeding into an optimizer update.
    """
    if isinstance(mesh, SpxMesh):
        if mesh.is_mpmd and axis != mesh.mpmd_axis:
            raise ValueError(f"axis={axis!r} doesn't match SpxMesh.mpmd_axis={mesh.mpmd_axis!r}.")
        jax_mesh = mesh.jax_mesh
    elif isinstance(mesh, MpMdMesh):
        jax_mesh = mesh.jax_mesh
    else:
        jax_mesh = mesh
    return spmd_run(
        model,
        batch,
        mesh=jax_mesh,
        axis=axis,
        schedule=schedule,
        loss_fn=loss_fn,
    )
