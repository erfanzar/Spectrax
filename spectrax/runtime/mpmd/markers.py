# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`sxstage_iter` — marker primitive for stage boundaries.

A user writes one model body, inserting :func:`sxstage_iter`
between layers to declare pipeline-stage cut points::

    def model(x):
        h = embed(x)
        h = sxstage_iter(h, stage=0)
        h = blocks_a(h)
        h = sxstage_iter(h, stage=1)
        h = blocks_b(h)
        return head(h)

At trace time the marker is an identity — the model runs unchanged on
a single device. When the function is passed to
:func:`~spectrax.runtime.mpmd.sxjit` (or executed via
:func:`~spectrax.runtime.mpmd.sxcall`), the tracer detects the markers
and splits the jaxpr at those points, producing one clustered sub-
:class:`~jax.extend.core.Jaxpr` per stage.

This module hosts both the primitive itself (registered with eager,
abstract, MLIR-lowering and linear-transpose rules) and the clustering
helpers that slice a traced :class:`Jaxpr` into per-stage sub-jaxprs.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
from jax import core
from jax.extend.core import Jaxpr, JaxprEqn, Primitive, Var
from jax.interpreters import ad, batching, mlir
from jax.sharding import NamedSharding, PartitionSpec

__all__ = [
    "cluster_jaxpr_by_markers",
    "marker_edge_shardings",
    "split_by_markers",
    "sxenter_loop",
    "sxexit_loop",
    "sxloop",
    "sxstage_iter",
]


sxstage_iter_p = Primitive("sxstage_iter")
sxstage_iter_p.multiple_results = True

sxenter_loop_p = Primitive("sxenter_loop")
sxenter_loop_p.multiple_results = True

sxexit_loop_p = Primitive("sxexit_loop")
sxexit_loop_p.multiple_results = True


def _normalize_sharding_axis(axis: Any) -> Any:
    """Return a hashable axis-spec component for primitive metadata."""
    if isinstance(axis, list | tuple):
        return tuple(_normalize_sharding_axis(part) for part in axis)
    return axis


def _normalize_edge_sharding(sharding: Any) -> PartitionSpec | None:
    """Normalize user-facing edge sharding metadata to a ``PartitionSpec``."""
    if sharding is None:
        return None
    if isinstance(sharding, NamedSharding):
        sharding = sharding.spec
    if isinstance(sharding, PartitionSpec):
        parts = tuple(sharding)
    elif isinstance(sharding, str):
        parts = (sharding,)
    else:
        try:
            parts = tuple(sharding)
        except TypeError as exc:
            raise TypeError(
                "sxstage_iter(..., sharding=...) expects a PartitionSpec, "
                "NamedSharding, axis name, or iterable of axis specs."
            ) from exc
    return PartitionSpec(*(_normalize_sharding_axis(part) for part in parts))


def sxstage_iter(x: Any, *, stage: int | None = None, sharding: Any = None) -> Any:
    """Declare a pipeline-stage boundary in the traced function.

    Functionally the identity — the marker survives in the jaxpr but
    lowers to a pass-through at MLIR time so single-device execution is
    unaffected. When the enclosing function is processed by
    :func:`~spectrax.runtime.mpmd.sxjit` /
    :func:`~spectrax.runtime.mpmd.sxcall`, the cluster splitter uses the
    marker positions to slice the jaxpr into per-stage sub-jaxprs.
    Gradient flows through the marker as an identity as well, so
    autograd on a marked model produces equivalent gradients to the
    unmarked version.

    ``x`` may be any JAX pytree (dict, tuple, dataclass, or plain array).

    Args:
        x: The activation(s) to flag as the end-of-stage boundary.
        stage: Optional integer hint for the stage index (for
            validation / debugging). Pure annotation — not read by the
            clustering algorithm, which partitions purely by the
            sequence in which markers appear.
        sharding: Optional edge ``PartitionSpec`` for the activation
            transfer leaving this stage. The MPMD runtime binds this
            spec to the destination stage-local mesh when moving values
            across pipeline ranks. A ``NamedSharding`` is accepted for
            convenience, but only its ``.spec`` is stored; concrete
            meshes are always resolved by the runtime.

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxstage_iter_p.bind(*flat, stage=stage, sharding=_normalize_edge_sharding(sharding), treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


@sxstage_iter_p.def_impl
def _mpmd_stage_iter_impl(*args, stage, sharding, treedef):
    """Concrete-evaluation rule for :data:`sxstage_iter_p`.

    Returns ``args`` verbatim — at trace-less / eager dispatch the
    marker has no observable effect, so single-device runs match the
    unmarked function exactly. The ``stage``, ``sharding`` and
    ``treedef`` parameters are pure metadata consumed only by the MPMD
    compiler pass.
    """
    del stage, sharding, treedef
    return args


@sxstage_iter_p.def_abstract_eval
def _mpmd_stage_iter_abs(*args, stage, sharding, treedef):
    """Abstract-eval rule for :data:`sxstage_iter_p`.

    The marker is the identity, so its output avals equal its input
    avals. Used by JAX during tracing to propagate shape/dtype
    information through marked code paths.
    """
    del stage, sharding, treedef
    return args


def _mpmd_stage_iter_transpose(cotangents, *args, stage, sharding, treedef):
    """Linear-transpose rule for :data:`sxstage_iter_p`.

    Because the marker is the identity, its transpose is also the
    identity: incoming cotangents flow back through unchanged. This
    keeps :func:`jax.grad` / :func:`jax.vjp` numerically equivalent
    between marked and unmarked models.
    """
    del args, stage, sharding, treedef
    return cotangents


ad.deflinear2(sxstage_iter_p, _mpmd_stage_iter_transpose)


def _mpmd_stage_iter_lowering(ctx, *args, stage, sharding, treedef):
    """MLIR lowering rule for :data:`sxstage_iter_p`.

    Emits no HLO of its own — the marker's operands are returned as
    its results, so XLA sees only a pass-through and the compiled
    program matches the unmarked function. The MPMD compiler removes
    these primitives before lowering when it splits the jaxpr by
    marker boundaries.
    """
    del ctx, stage, sharding, treedef
    return list(args)


mlir.register_lowering(sxstage_iter_p, _mpmd_stage_iter_lowering)


def _mpmd_stage_iter_batch(vector_arg_values, batch_axes, *, stage, sharding, treedef):
    """``vmap`` rule for ``sxstage_iter_p``: the marker is identity, so axes pass through unchanged."""
    del stage, sharding, treedef
    return vector_arg_values, batch_axes


batching.primitive_batchers[sxstage_iter_p] = _mpmd_stage_iter_batch


def sxenter_loop(x: Any, *, name: str | None = None) -> Any:
    """Mark the start of a repeated computation block.

    Functionally the identity.  When the enclosing function is traced by
    :func:`~spectrax.runtime.mpmd.sxjit`, the marker is preserved and
    the region between ``sxenter_loop`` and its matching
    :func:`sxexit_loop` can be identified for loop-aware optimisations.

    ``x`` may be any JAX pytree.

    Args:
        x: The activation(s) at the loop entry point.
        name: Optional human-readable name (for debugging).

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxenter_loop_p.bind(*flat, name=name, treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


def sxexit_loop(x: Any, *, name: str | None = None) -> Any:
    """Mark the end of a repeated computation block.

    See :func:`sxenter_loop` for details.

    Args:
        x: The activation(s) at the loop exit point.
        name: Optional human-readable name (for debugging).

    Returns:
        ``x`` unchanged.
    """
    flat, treedef = jax.tree_util.tree_flatten(x)
    out_flat = sxexit_loop_p.bind(*flat, name=name, treedef=treedef)
    return jax.tree_util.tree_unflatten(treedef, out_flat)


@sxenter_loop_p.def_impl
def _mpmd_enter_loop_impl(*args, name, treedef):
    """Eager impl rule for :data:`sxenter_loop_p`: return inputs unchanged (identity)."""
    del name, treedef
    return args


@sxenter_loop_p.def_abstract_eval
def _mpmd_enter_loop_abs(*args, name, treedef):
    """Abstract-eval rule for :data:`sxenter_loop_p`: outputs have the same avals as inputs."""
    del name, treedef
    return args


def _mpmd_enter_loop_transpose(cotangents, *args, name, treedef):
    """Linear-transpose rule for :data:`sxenter_loop_p`: cotangents flow through unchanged."""
    del args, name, treedef
    return cotangents


ad.deflinear2(sxenter_loop_p, _mpmd_enter_loop_transpose)


@sxexit_loop_p.def_impl
def _mpmd_exit_loop_impl(*args, name, treedef):
    """Eager impl rule for :data:`sxexit_loop_p`: return inputs unchanged (identity)."""
    del name, treedef
    return args


@sxexit_loop_p.def_abstract_eval
def _mpmd_exit_loop_abs(*args, name, treedef):
    """Abstract-eval rule for :data:`sxexit_loop_p`: outputs have the same avals as inputs."""
    del name, treedef
    return args


def _mpmd_exit_loop_transpose(cotangents, *args, name, treedef):
    """Linear-transpose rule for :data:`sxexit_loop_p`: cotangents flow through unchanged."""
    del args, name, treedef
    return cotangents


ad.deflinear2(sxexit_loop_p, _mpmd_exit_loop_transpose)


def _mpmd_enter_loop_batch(vector_arg_values, batch_axes, *, name, treedef):
    """``vmap`` rule for :data:`sxenter_loop_p`: identity primitive, axes pass through."""
    del name, treedef
    return vector_arg_values, batch_axes


def _mpmd_exit_loop_batch(vector_arg_values, batch_axes, *, name, treedef):
    """``vmap`` rule for :data:`sxexit_loop_p`: identity primitive, axes pass through."""
    del name, treedef
    return vector_arg_values, batch_axes


batching.primitive_batchers[sxenter_loop_p] = _mpmd_enter_loop_batch
batching.primitive_batchers[sxexit_loop_p] = _mpmd_exit_loop_batch


def _mpmd_enter_loop_lowering(ctx, *args, name, treedef):
    """MLIR lowering for :data:`sxenter_loop_p`: pass operands through (identity)."""
    del ctx, name, treedef
    return list(args)


def _mpmd_exit_loop_lowering(ctx, *args, name, treedef):
    """MLIR lowering for :data:`sxexit_loop_p`: pass operands through (identity)."""
    del ctx, name, treedef
    return list(args)


mlir.register_lowering(sxenter_loop_p, _mpmd_enter_loop_lowering)
mlir.register_lowering(sxexit_loop_p, _mpmd_exit_loop_lowering)


def sxloop(
    body_fn: Any,
    init: Any,
    xs: Any = None,
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> Any:
    """Repeatedly apply ``body_fn`` using :func:`jax.lax.scan`.

    This is a thin wrapper with a friendlier name.  The main benefit over
    a plain Python ``for`` loop is that the loop body stays as a single
    ``scan`` primitive inside the traced jaxpr, which is dramatically
    cheaper for ``eval_jaxpr`` / XLA compilation than thousands of
    unrolled primitive equations.

    Args:
        body_fn: ``(carry, x) -> new_carry``  or  ``(carry, x) -> (new_carry, y)``
        init: Initial carry value.
        xs: Sequence of inputs scanned over.  May be ``None`` if *length*
            is provided (the body still receives ``None`` on each step).
        length: Number of loop iterations.  Required when ``xs`` is ``None``.
        reverse: If ``True``, scan in reverse order.
        unroll: Unroll factor passed to :func:`jax.lax.scan`.

    Returns:
        ``new_carry`` if ``body_fn`` returns a single value, otherwise
        ``(new_carry, ys)`` where ``ys`` is the stacked sequence of
        second return values.
    """
    return jax.lax.scan(body_fn, init, xs, length=length, reverse=reverse, unroll=unroll)


def _collect_used_vars(eqns: list[JaxprEqn]) -> set[Var]:
    """Return the set of :class:`Var` s read by any eqn in ``eqns``."""
    used: set[Var] = set()
    for eqn in eqns:
        for invar in eqn.invars:
            if isinstance(invar, Var):
                used.add(invar)
    return used


def _collect_defined_vars(eqns: list[JaxprEqn]) -> set[Var]:
    """Return the set of :class:`Var` s written by any eqn in ``eqns``."""
    defined: set[Var] = set()
    for eqn in eqns:
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                defined.add(outvar)
    return defined


def marker_edge_shardings(jaxpr: Jaxpr) -> list[PartitionSpec | None]:
    """Return ``sxstage_iter`` edge shardings in marker order.

    Entry ``i`` describes the transfer edge leaving logical stage
    ``i``. ``None`` means the runtime should keep its default transfer
    target for that boundary.
    """
    return [eqn.params.get("sharding") for eqn in jaxpr.eqns if eqn.primitive is sxstage_iter_p]


def cluster_jaxpr_by_markers(jaxpr: Jaxpr) -> list[Jaxpr]:
    """Split ``jaxpr`` into sub-jaxprs at every ``sxstage_iter`` eqn.

    The marker eqns themselves are dropped (they're identity). Each
    returned sub-jaxpr represents one pipeline stage:

    * **Invars**: the subset of vars read by eqns in the cluster but
      defined *before* it (i.e. inputs from the previous stage or
      from the enclosing ``jaxpr.invars``).
    * **Outvars**: the subset of vars defined in the cluster that are
      read *after* it. When the cluster ends at a marker eqn, the
      marker's output vars are routed back to their corresponding
      input vars (the marker is identity), and any otherwise-unlisted
      marker invars are appended so the next stage receives them.
    * **Eqns**: the subsequence of ``jaxpr.eqns`` lying between two
      consecutive markers, with marker eqns excluded.
    * **Constvars**: inherited from the parent ``jaxpr`` — each sub-
      jaxpr may refer to any of the parent's constants.

    The final sub-jaxpr's outvars include the parent ``jaxpr.outvars``.

    Args:
        jaxpr: The traced :class:`Jaxpr`, typically produced by
            :func:`jax.make_jaxpr`.

    Returns:
        A list of ``n_markers + 1`` sub-jaxprs, in execution order.

    Notes:
        ``read_after[i]`` caches the set of vars that are read at or
        after eqn ``i`` — used to pick each cluster's outvars from its
        downstream-live set. ``defined_up_to[i]`` mirrors that from the
        other direction: the set of vars available *before* executing
        eqn ``i`` (jaxpr invars are defined from the start).

        When a cluster ends at a marker eqn, the eqn itself is stripped
        but its outvar still appears as a consumer in subsequent
        clusters. Because the marker is an identity, we route its
        invars directly through in place of its outvars — numerically
        identical and avoids needing a pass-through eqn.
    """
    marker_positions: list[int] = [i for i, eqn in enumerate(jaxpr.eqns) if eqn.primitive is sxstage_iter_p]
    boundaries = [0, *[p + 1 for p in marker_positions], len(jaxpr.eqns)]

    n_eqns = len(jaxpr.eqns)
    eqn_index_by_id = {id(eqn): i for i, eqn in enumerate(jaxpr.eqns)}
    producer_by_var_id: dict[int, JaxprEqn] = {}
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                producer_by_var_id[id(outvar)] = eqn
    jaxpr_invar_ids = {id(v) for v in jaxpr.invars if isinstance(v, Var)}
    jaxpr_constvar_ids = {id(v) for v in jaxpr.constvars if isinstance(v, Var)}
    marker_input_ids = {
        id(invar)
        for eqn in jaxpr.eqns
        if eqn.primitive is sxstage_iter_p
        for invar in eqn.invars
        if isinstance(invar, Var)
    }

    remat_cache: dict[int, bool] = {}

    def can_rematerialize(var: Var) -> bool:
        """Whether ``var`` can be cheaply/safely rebuilt inside later stages.

        Values derived only from dynamic body inputs and literals, such as
        masks or position ids built before the first stage cut, should not be
        shipped through the pipeline as activations. Values that touch closed
        consts are left as real stage outputs because those consts may be
        stage-owned trainable weights.
        """
        var_id = id(var)
        cached = remat_cache.get(var_id)
        if cached is not None:
            return cached
        if var_id in jaxpr_invar_ids:
            remat_cache[var_id] = True
            return True
        if var_id in jaxpr_constvar_ids or var_id in marker_input_ids:
            remat_cache[var_id] = False
            return False
        eqn = producer_by_var_id.get(var_id)
        if eqn is None or eqn.primitive is sxstage_iter_p or getattr(eqn, "effects", core.no_effects):
            remat_cache[var_id] = False
            return False
        remat_cache[var_id] = False
        ok = all(not isinstance(invar, Var) or can_rematerialize(invar) for invar in eqn.invars)
        remat_cache[var_id] = ok
        return ok

    def collect_remat_eqns(
        var: Var,
        *,
        local_defined_ids: set[int],
        local_eqn_ids: set[int],
        out: dict[int, JaxprEqn],
    ) -> None:
        """Collect equations transitively needed to rematerialize ``var``.

        Walks backwards from ``var`` through the jaxpr's def-use chain,
        skipping equations whose outputs are already locally defined,
        whose input is the original jaxpr's invars/constvars, or which
        cross an :data:`sxstage_iter_p` boundary (which is the
        cluster's seam — recomputing across it would re-run the
        upstream stage). Output is accumulated into the ``out`` dict
        keyed by equation ``id`` so the same equation isn't added
        twice.
        """
        if id(var) in jaxpr_invar_ids or id(var) in jaxpr_constvar_ids or id(var) in local_defined_ids:
            return
        if not can_rematerialize(var):
            return
        eqn = producer_by_var_id.get(id(var))
        if eqn is None or eqn.primitive is sxstage_iter_p:
            return
        for invar in eqn.invars:
            if isinstance(invar, Var):
                collect_remat_eqns(
                    invar,
                    local_defined_ids=local_defined_ids,
                    local_eqn_ids=local_eqn_ids,
                    out=out,
                )
        if id(eqn) not in local_eqn_ids:
            out[id(eqn)] = eqn

    read_after: list[set[Var]] = [set() for _ in range(n_eqns + 1)]
    post: set[Var] = set()
    for v in jaxpr.outvars:
        if isinstance(v, Var):
            post.add(v)
    read_after[n_eqns] = set(post)
    for i in range(n_eqns - 1, -1, -1):
        post = set(post)
        for invar in jaxpr.eqns[i].invars:
            if isinstance(invar, Var):
                post.add(invar)
        read_after[i] = post

    defined_up_to: list[set[Var]] = [set(jaxpr.invars)]
    pre: set[Var] = set(jaxpr.invars)
    for eqn in jaxpr.eqns:
        pre = set(pre)
        for outvar in eqn.outvars:
            if isinstance(outvar, Var):
                pre.add(outvar)
        defined_up_to.append(pre)

    clusters: list[Jaxpr] = []
    for idx, (start, end) in enumerate(itertools.pairwise(boundaries)):
        base_eqns = [e for e in jaxpr.eqns[start:end] if e.primitive is not sxstage_iter_p]
        base_eqn_ids = {id(eqn) for eqn in base_eqns}
        base_defined = _collect_defined_vars(base_eqns)
        base_defined_ids = {id(v) for v in base_defined}
        remat_eqns_by_id: dict[int, JaxprEqn] = {}
        for used_var in _collect_used_vars(base_eqns):
            collect_remat_eqns(
                used_var,
                local_defined_ids=base_defined_ids,
                local_eqn_ids=base_eqn_ids,
                out=remat_eqns_by_id,
            )
        remat_eqns = sorted(remat_eqns_by_id.values(), key=lambda eqn: eqn_index_by_id[id(eqn)])
        eqns = [*remat_eqns, *base_eqns]
        used = _collect_used_vars(eqns)
        defined_before = defined_up_to[start]
        defined_here = _collect_defined_vars(eqns)
        invars = [v for v in defined_before if v in used and v not in defined_here]
        if end < n_eqns:
            needed_downstream = read_after[end]
            outvars: list[Var] = [v for v in defined_here if v in needed_downstream and not can_rematerialize(v)]
        else:
            needed_downstream = set(v for v in jaxpr.outvars if isinstance(v, Var))
            outvars = [v for v in jaxpr.outvars if isinstance(v, Var) and v in defined_here]
        if end - 1 >= start and end - 1 < n_eqns:
            last = jaxpr.eqns[end - 1]
            if last.primitive is sxstage_iter_p:
                for mv in last.outvars:
                    if isinstance(mv, Var) and mv not in outvars:
                        outvars.append(mv)
                marker_invars = [v for v in last.invars if isinstance(v, Var)]
                marker_outvars = [v for v in last.outvars if isinstance(v, Var)]
                outvars = [marker_invars[marker_outvars.index(v)] if v in marker_outvars else v for v in outvars]
                for mv in marker_invars:
                    if mv not in outvars:
                        outvars.append(mv)

        seen: set[Var] = set()
        dedup_outvars = []
        for v in outvars:
            if v not in seen:
                dedup_outvars.append(v)
                seen.add(v)

        sub = Jaxpr(
            constvars=list(jaxpr.constvars),
            invars=invars,
            outvars=dedup_outvars,
            eqns=eqns,
            effects=core.no_effects,
        )
        clusters.append(sub)
        del idx
    return clusters


def split_by_markers(
    fn: Any,
    *abstract_args: Any,
    return_clusters: bool = False,
) -> Any:
    """Trace ``fn`` and split it at every :func:`sxstage_iter`.

    Returns a list of per-stage Python callables. Each callable takes
    the stage's input activations (as positional arrays) and returns
    the stage's output activations. Constants captured by tracing
    are baked in — the caller does not need to pass them.

    Args:
        fn: The user's model function. Must use
            :func:`sxstage_iter` between logical stages.
        *abstract_args: Example arguments used to trace ``fn``. Can
            be concrete :class:`jax.Array` s or :class:`jax.ShapeDtypeStruct` s.
        return_clusters: If ``True``, also return the raw cluster
            :class:`Jaxpr` list (for debugging / advanced pipeline
            construction).

    Returns:
        ``list[Callable]`` — one per stage, in execution order. If
        ``return_clusters=True``, returns
        ``(list[Callable], list[Jaxpr], consts)``.
    """
    closed = jax.make_jaxpr(fn)(*abstract_args)
    clusters = cluster_jaxpr_by_markers(closed.jaxpr)
    consts = closed.consts

    def make_stage(cluster_jaxpr: Jaxpr):
        """Build a Python callable that evaluates ``cluster_jaxpr`` with bound consts."""

        def stage_fn(*args):
            """Evaluate ``cluster_jaxpr`` against ``args`` with the captured consts.

            Args:
                *args: Stage input activations as positional arrays,
                    matching ``cluster_jaxpr.invars`` in order.

            Returns:
                Tuple of stage output activations matching
                ``cluster_jaxpr.outvars``.
            """
            return tuple(core.eval_jaxpr(cluster_jaxpr, consts, *args))

        stage_fn.__name__ = f"stage_fn_{id(cluster_jaxpr) & 0xFFFF:04x}"
        return stage_fn

    stage_fns = [make_stage(c) for c in clusters]
    if return_clusters:
        return stage_fns, clusters, consts
    return stage_fns
