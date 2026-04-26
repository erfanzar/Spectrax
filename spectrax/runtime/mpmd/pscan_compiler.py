# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule-driven compiler pass for :func:`sxjit`.

When ``sxjit`` traces a function that calls :func:`treduce` internally,
the outer jaxpr contains a single ``pscan_p`` equation whose parameters
carry the user body's jaxpr, the pipeline schedule, and accumulator ops.
This module turns that equation into a per-stage compilation plan and a
schedule-aware Python dispatch loop.

Algorithm:

1. Cluster the body's scalar-loss jaxpr by :func:`sxstage_iter`
   markers — one cluster per logical pipeline stage.
2. Map each logical cluster to the physical ``(rank, virt)`` location
   specified by the schedule's ``logical_at`` / ``next_logical_loc``.
3. Build jitted forward and backward (VJP) callables per logical stage.
   The terminal cluster uses :func:`jax.value_and_grad` so cotangents
   start at ``1.0`` on the loss.
4. At dispatch time, walk ``schedule.build(n)`` step by step. FWD phases
   chain activations along the logical pipeline, BWD phases route
   cotangents back along the same logical chain, and per-rank gradient
   accumulators sum all virtual-stage contributions.

Supported bodies:

* ``fun(i) -> scalar_loss``
* ``fun(i) -> (scalar_loss, grads_pytree)``

The compiled path always pipelines the scalar-loss jaxpr and reconstructs
the final model-shaped gradient pytree from the captured module consts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.extend.core import Jaxpr, JaxprEqn, Var

from ...core.graph import export, live_variables
from ...core.module import Module
from ...core.selector import as_selector
from ...core.stage_assignment import metadata_stage_assignment, resolve_stage_rank
from ...sharding.partition import get_named_sharding, sanitize_partition_spec_for_mesh_and_shape
from ..schedules import (
    Eager1F1B,
    FusedTask,
    InterleavedH1,
    Phase,
    Std1F1B,
    ZeroBubbleH1,
    fuse_1f1b_steady_state,
    fuse_zerobubble_bwd_pair,
)
from ..types.mesh import MpMdMesh
from .markers import cluster_jaxpr_by_markers, marker_edge_shardings, sxstage_iter_p
from .treduce import Op, _unwrap_ops, _unwrap_schedule, pscan_p

__all__ = [
    "PscanPlan",
    "build_pscan_plan",
    "dispatch_pscan",
    "has_pscan",
]


def has_pscan(jaxpr: Jaxpr) -> list[JaxprEqn]:
    """Return ``pscan_p`` equations found at the top level of ``jaxpr``.

    Shallow scan only — does not recurse into nested jaxprs.
    """
    return [e for e in jaxpr.eqns if e.primitive is pscan_p]


def _eqn_params(eqn: JaxprEqn) -> dict[str, Any]:
    """Return a JAX equation's primitive parameter dict.

    Older JAX exposed the parameters dict as ``eqn.params``; some newer
    snapshots renamed it to ``eqn.parameters``. This helper falls back
    so the rest of the compiler does not have to branch.

    Args:
        eqn: Any :class:`JaxprEqn`.

    Returns:
        The equation's parameter mapping (empty if neither attribute
        exists, which should not happen with supported JAX versions).
    """
    return getattr(eqn, "params", getattr(eqn, "parameters", {}))


@dataclass
class PscanPlan:
    """Pre-compiled dispatch plan for one ``pscan_p`` equation.

    Built once by :func:`build_pscan_plan`, reused across calls. Holds
    per-logical-stage placed constants, per-``(rank, virt)`` jitted
    forward/backward callables, the schedule's action grid, and
    accumulator metadata.
    """

    n: int
    v: int
    n_logical: int
    m: int
    schedule: Any
    ops: tuple[Op, ...]
    n_outs: int
    n_outer_consts: int
    body_mode: str

    stage_shardings: list[Any]
    rank_submeshes: list[Any]
    mpmd_mesh: MpMdMesh

    loc_for_logical: tuple[tuple[int, int], ...]
    logical_for_loc: dict[tuple[int, int], int]
    terminal_loc: tuple[int, int]

    per_loc_consts: dict[tuple[int, int], tuple[Any, ...]]
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]]
    n_invars_per_loc: dict[tuple[int, int], int]

    fwd_jits: dict[tuple[int, int], Callable[..., Any]]
    bwd_jits: dict[tuple[int, int], Callable[..., Any] | None]
    terminal_jit: Callable[..., Any]

    init_state_template: list[Any]

    grad_tree: Any | None = None
    grad_const_indices: tuple[int, ...] = ()
    grad_template_leaves: tuple[Any, ...] = ()
    grad_output_sharding: Any | None = None

    invar_sources: list[list[tuple[str, int, int]]] = field(default_factory=list)
    edge_shardings: list[Any] = field(default_factory=list)

    grid: list[list[Any]] = field(default_factory=list)


def _collect_used_constvars(cluster: Jaxpr) -> list[Var]:
    """Return constvars of ``cluster`` referenced by any of its equations.

    Order preserved by first use so downstream filtering matches the
    variable ordering inside the cluster's eqns.
    """
    cv_set = {id(v): v for v in cluster.constvars}
    seen: set[int] = set()
    order: list[Var] = []
    for eqn in cluster.eqns:
        for iv in eqn.invars:
            if isinstance(iv, Var) and id(iv) in cv_set and id(iv) not in seen:
                seen.add(id(iv))
                order.append(cv_set[id(iv)])
    return order


def _filtered_cluster(cluster: Jaxpr, used_constvars: list[Var]) -> Jaxpr:
    """Return a copy of ``cluster`` whose ``constvars`` are restricted to ``used_constvars``."""
    return Jaxpr(
        constvars=used_constvars,
        invars=list(cluster.invars),
        outvars=list(cluster.outvars),
        eqns=list(cluster.eqns),
        effects=cluster.effects,
    )


def _place_cluster_consts(
    used_vars: list[Var],
    all_constvars: list[Var],
    concrete_consts: tuple[Any, ...],
    const_flat_arg_indices: tuple[int | None, ...],
    leaf_shardings: dict[int, Any],
    leaf_stage_owners: dict[int, int],
    fallback_sharding: Any,
    expected_rank: int,
) -> tuple[Any, ...]:
    """Pick concrete values for ``used_vars`` and place them on the owning stage mesh.

    ``concrete_consts`` is aligned with ``all_constvars``: entry ``i``
    is the runtime value for variable ``all_constvars[i]``. Returns
    only the subset for ``used_vars``, each placed on the rank's
    sub-mesh. When a const originated from a captured Module leaf with
    sharding metadata, that leaf's stage-local NamedSharding wins;
    otherwise we fall back to a replicated sharding on the stage's
    sub-mesh.
    """
    const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}
    placed: list[Any] = []
    for var in used_vars:
        const_idx = const_idx_by_id[id(var)]
        flat_arg_idx = const_flat_arg_indices[const_idx]
        if flat_arg_idx is not None:
            owner = leaf_stage_owners.get(flat_arg_idx)
            if owner is not None and owner != expected_rank:
                raise ValueError(
                    f"sxjit+treduce: flat argument leaf {flat_arg_idx} is "
                    f"assigned to pipeline stage {owner}, but traced stage "
                    f"{expected_rank} uses it. Move the corresponding layer into "
                    f"the matching pipeline segment or update its "
                    f"assign_stage(...) hint."
                )
        sharding = leaf_shardings.get(flat_arg_idx) if flat_arg_idx is not None else None
        placed.append(jax.device_put(concrete_consts[const_idx], sharding or fallback_sharding))
    return tuple(placed)


def _make_fwd_jit(cluster_jaxpr: Jaxpr, donate_argnums: tuple[int, ...] = ()) -> Callable[..., tuple[Any, ...]]:
    """Return ``@jax.jit`` callable ``(consts, *invars) -> outvars`` for a non-terminal cluster.

    Consts are passed as an explicit first argument (not closure-captured)
    so the dispatcher can route placed constants uniformly and so the
    backward VJP can differentiate w.r.t. them.
    """

    def fwd(consts: tuple[Any, ...], *invars: Any) -> tuple[Any, ...]:
        """Evaluate the cluster sub-jaxpr with an explicit const tuple.

        Args:
            consts: Concrete values aligned with the cluster's
                ``constvars`` (placed beforehand on the rank's
                sub-mesh).
            *invars: Stage input activations as positional arrays.

        Returns:
            Tuple of cluster outputs matching the sub-jaxpr's outvars.
        """
        return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(consts), *invars))

    if donate_argnums:
        return jax.jit(fwd, donate_argnums=donate_argnums)
    return jax.jit(fwd)


def _make_bwd_jit(
    cluster_jaxpr: Jaxpr, n_invars: int, donate_argnums: tuple[int, ...] = ()
) -> Callable[..., tuple[Any, tuple[Any, ...]]]:
    """Return ``@jax.jit`` VJP callable for a non-terminal cluster.

    Signature: ``(consts, *invars, *cotangents) -> (g_consts, g_invars)``.
    Used by BWD / BWD_I / BWD_W schedule phases. XLA DCEs unused
    outputs so BWD_I (discards ``g_consts``) and BWD_W (discards
    ``g_invars``) each take ~half the cost of a full BWD.
    """

    def bwd(consts: tuple[Any, ...], *invars_and_cotangents: Any) -> tuple[Any, tuple[Any, ...]]:
        """Run ``jax.vjp`` on the cluster and return ``(g_consts, g_invars)``.

        ``invars_and_cotangents`` is the concatenation of the cluster's
        positional inputs (first ``n_invars`` entries) and the per-output
        cotangents seeded by the downstream stage.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``(*invars, *cotangents)`` packed as
                a single positional list — flattened so :func:`jax.jit`
                can infer donate positions cleanly.

        Returns:
            ``(g_consts, g_invars)`` where ``g_consts`` mirrors
            ``consts`` and ``g_invars`` is a tuple aligned with the
            cluster's invars.
        """
        invars = invars_and_cotangents[:n_invars]
        cotangents = invars_and_cotangents[n_invars:]

        def pure(c: tuple[Any, ...], *xs: Any) -> tuple[Any, ...]:
            """Closed-over jaxpr evaluator with ``consts`` as the first VJP argument."""
            return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

        _, vjp_fn = jax.vjp(pure, consts, *invars)
        grads = vjp_fn(tuple(cotangents))
        g_consts = grads[0]
        g_invars = tuple(grads[1:])
        return g_consts, g_invars

    if donate_argnums:
        return jax.jit(bwd, donate_argnums=donate_argnums)
    return jax.jit(bwd)


def _make_bwd_i_jit(
    cluster_jaxpr: Jaxpr, n_invars: int, donate_argnums: tuple[int, ...] = ()
) -> Callable[..., tuple[Any, ...]]:
    """Return a ``@jax.jit`` VJP callable that yields only input cotangents.

    Companion of :func:`_make_bwd_w_jit`. Together the pair lets
    ZeroBubble-style schedules send activation cotangents upstream as
    soon as input grads are ready while deferring the costlier
    weight-grad computation into pipeline bubble slots.
    """

    def bwd_i(consts: tuple[Any, ...], *invars_and_cotangents: Any) -> tuple[Any, ...]:
        """Compute ``grad(invars)`` only, dropping the const grads.

        ``invars_and_cotangents`` packs ``(*invars, *cotangents)`` as a
        single positional sequence for clean :func:`jax.jit` donation
        bookkeeping.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``invars`` followed by per-output
                cotangents from the downstream stage.

        Returns:
            ``g_invars`` aligned with the cluster's invars.
        """
        invars = invars_and_cotangents[:n_invars]
        cotangents = invars_and_cotangents[n_invars:]

        def pure(c: tuple[Any, ...], *xs: Any) -> tuple[Any, ...]:
            """Closed-over consts+invars -> outs interpreter used by :func:`jax.vjp`."""
            return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

        _, vjp_fn = jax.vjp(pure, consts, *invars)
        grads = vjp_fn(tuple(cotangents))
        return tuple(grads[1:])

    if donate_argnums:
        return jax.jit(bwd_i, donate_argnums=donate_argnums)
    return jax.jit(bwd_i)


def _make_bwd_w_jit(cluster_jaxpr: Jaxpr, n_invars: int, donate_argnums: tuple[int, ...] = ()) -> Callable[..., Any]:
    """Return ``@jax.jit`` VJP callable for weight/const gradients only."""

    def bwd_w(consts: tuple[Any, ...], *invars_and_cotangents: Any) -> Any:
        """Compute ``grad(consts)`` only, dropping invar cotangents.

        Companion to :func:`_make_bwd_i_jit`. Splitting the backward
        into ``BWD_I`` then ``BWD_W`` is the core trick of ZeroBubble:
        the ``BWD_W`` half can run later, in the bubble that would
        otherwise idle the rank waiting for downstream cotangents.

        Args:
            consts: Cluster constants placed on this rank.
            *invars_and_cotangents: ``invars`` followed by per-output
                cotangents from the downstream stage.

        Returns:
            ``g_consts`` matching the structure of ``consts``.
        """
        invars = invars_and_cotangents[:n_invars]
        cotangents = invars_and_cotangents[n_invars:]

        def pure(c: tuple[Any, ...], *xs: Any) -> tuple[Any, ...]:
            """Closed-over consts+invars -> outs interpreter used by :func:`jax.vjp`."""
            return tuple(jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs))

        _, vjp_fn = jax.vjp(pure, consts, *invars)
        grads = vjp_fn(tuple(cotangents))
        return grads[0]

    if donate_argnums:
        return jax.jit(bwd_w, donate_argnums=donate_argnums)
    return jax.jit(bwd_w)


def _make_terminal_jit(cluster_jaxpr: Jaxpr, n_invars: int, donate_argnums: tuple[int, ...] = ()) -> Callable[..., Any]:
    """Return ``@jax.jit`` ``value_and_grad`` callable for the terminal cluster.

    Signature: ``(consts, *invars) -> (loss, (g_consts, g_invars))``.

    The terminal cluster produces exactly one scalar output (the loss).
    :func:`jax.value_and_grad` supplies the initial cotangent of
    ``1.0`` automatically so we don't have to thread it from outside.
    """

    def term(consts: tuple[Any, ...], *invars: Any) -> tuple[Any, tuple[Any, tuple[Any, ...]]]:
        """Compute the loss and its gradients w.r.t. ``(consts, *invars)`` in one jit.

        Wraps the cluster's scalar-loss evaluator in
        :func:`jax.value_and_grad` over every positional argument so a
        single compiled program produces both the per-microbatch loss
        value and the seed cotangents the upstream backward sweep
        needs.

        Args:
            consts: Terminal cluster's placed constants.
            *invars: Positional inputs to the cluster (typically the
                activations entering the loss layer).

        Returns:
            ``(loss, (g_consts, g_invars))`` — the scalar loss plus
            its gradients.
        """

        def pure(c: tuple[Any, ...], *xs: Any) -> Any:
            """Scalar-loss evaluator, asserts a single cluster output."""
            outs = jax.core.eval_jaxpr(cluster_jaxpr, list(c), *xs)
            if len(outs) != 1:
                raise ValueError(
                    f"Terminal cluster must produce exactly one scalar output "
                    f"(the per-microbatch loss); got {len(outs)}."
                )
            return outs[0]

        argnums = tuple(range(1 + n_invars))
        loss, grads = jax.value_and_grad(pure, argnums=argnums, allow_int=True)(consts, *invars)
        g_consts = grads[0]
        g_invars = tuple(grads[1:])
        return loss, (g_consts, g_invars)

    if donate_argnums:
        return jax.jit(term, donate_argnums=donate_argnums)
    return jax.jit(term)


def _build_logical_locs(
    schedule: Any,
    n: int,
    v: int,
) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], int], tuple[int, int]]:
    """Build ``logical <-> (rank, virt)`` maps and validate the schedule chain."""
    n_logical = n * v
    locs: list[tuple[int, int] | None] = [None] * n_logical
    logical_for_loc: dict[tuple[int, int], int] = {}

    for rank in range(n):
        for virt in range(v):
            loc = (rank, virt)
            logical = schedule.logical_at(rank, virt, n)
            if logical < 0 or logical >= n_logical:
                raise ValueError(
                    f"Schedule {type(schedule).__name__} mapped {(rank, virt)} to "
                    f"logical stage {logical}, outside [0, {n_logical})."
                )
            if locs[logical] is not None:
                raise ValueError(
                    f"Schedule {type(schedule).__name__} maps multiple locations to "
                    f"logical stage {logical}: {locs[logical]} and {loc}."
                )
            locs[logical] = loc
            logical_for_loc[loc] = logical

    if any(loc is None for loc in locs):
        missing = [i for i, loc in enumerate(locs) if loc is None]
        raise ValueError(f"Schedule {type(schedule).__name__} did not assign locations for logical stages {missing}.")

    loc_for_logical = tuple(loc for loc in locs if loc is not None)
    terminal_loc = schedule.terminal_loc(n)
    terminal_logical = logical_for_loc.get(terminal_loc)
    if terminal_logical != n_logical - 1:
        raise NotImplementedError(
            "sxjit+treduce requires the schedule terminal location to host the "
            f"last logical stage; got terminal {terminal_loc} -> logical "
            f"{terminal_logical} under {type(schedule).__name__}."
        )

    for logical, loc in enumerate(loc_for_logical):
        expected_next = None if logical == n_logical - 1 else loc_for_logical[logical + 1]
        actual_next = schedule.next_logical_loc(loc[0], loc[1], n)
        if actual_next != expected_next:
            raise NotImplementedError(
                "sxjit+treduce currently requires `next_logical_loc` to follow "
                f"the logical stage chain. For logical {logical} at {loc}, expected "
                f"{expected_next}, got {actual_next} under {type(schedule).__name__}."
            )

    return loc_for_logical, logical_for_loc, terminal_loc


def _resolve_concrete_consts(
    outer_jaxpr: jax.core.ClosedJaxpr,
    outer_flat_args: tuple[Any, ...],
    pscan_eqn: JaxprEqn,
    n_body_consts: int,
) -> tuple[tuple[Any, ...], tuple[int | None, ...]]:
    """Map the first ``n_body_consts`` operands of ``pscan_eqn`` to concrete values.

    At trace time, the inner body jaxpr's "consts" (closure captures
    from the enclosing ``sxjit`` trace) became outer-jaxpr tracers
    and were passed as the first ``n_body_consts`` operands of the
    ``pscan_p`` equation. Here we resolve each operand back to its
    concrete runtime value using the outer jaxpr's constvars + invars
    mapping.

    Args:
        outer_jaxpr: The outer ``ClosedJaxpr`` from ``jax.make_jaxpr(fn)``.
        outer_flat_args: Flattened concrete runtime args (one per
            ``outer_jaxpr.jaxpr.invars`` entry, in order).
        pscan_eqn: The ``pscan_p`` equation in the outer jaxpr.
        n_body_consts: Number of operands at the head of
            ``pscan_eqn.invars`` that represent body consts.

    Returns:
        A tuple ``(values, flat_arg_indices)`` where ``values`` are the
        concrete body consts in ``fn_jaxpr.constvars`` order and
        ``flat_arg_indices[i]`` is the originating outer flat-arg index
        when const ``i`` came from a traced input leaf, else ``None``.
    """
    outer_constvars = list(outer_jaxpr.jaxpr.constvars)
    outer_consts = tuple(outer_jaxpr.consts)
    const_by_id: dict[int, Any] = {id(v): c for v, c in zip(outer_constvars, outer_consts, strict=True)}

    outer_invars = list(outer_jaxpr.jaxpr.invars)
    invar_idx_by_id: dict[int, int] = {id(v): i for i, v in enumerate(outer_invars)}

    resolved: list[Any] = []
    flat_arg_indices: list[int | None] = []
    for operand in pscan_eqn.invars[:n_body_consts]:
        if isinstance(operand, Var):
            if id(operand) in invar_idx_by_id:
                flat_idx = invar_idx_by_id[id(operand)]
                resolved.append(outer_flat_args[flat_idx])
                flat_arg_indices.append(flat_idx)
            elif id(operand) in const_by_id:
                resolved.append(const_by_id[id(operand)])
                flat_arg_indices.append(None)
            else:
                raise RuntimeError(
                    f"pscan operand Var {operand} is not in outer jaxpr's invars or constvars. Shape: {operand.aval}."
                )
        else:
            resolved.append(operand.val if hasattr(operand, "val") else operand)
            flat_arg_indices.append(None)
    return tuple(resolved), tuple(flat_arg_indices)


def _arg_leaf_ranges(args: tuple[Any, ...]) -> list[tuple[int, int]]:
    """Return flat-leaf ``[start, end)`` ranges for each positional argument."""
    ranges: list[tuple[int, int]] = []
    start = 0
    for arg in args:
        n_leaves = len(jax.tree.leaves(arg))
        ranges.append((start, start + n_leaves))
        start += n_leaves
    return ranges


def _infer_outer_leaf_shardings(
    outer_args: tuple[Any, ...],
    outer_flat_args: tuple[Any, ...],
    n: int,
    rank_submeshes: list[Any],
) -> tuple[list[dict[int, Any]], dict[int, int]]:
    """Infer per-rank NamedShardings for captured Module leaves.

    Mirrors ``sxjit``'s forward-only path: each rank resolves the
    captured Module's logical-axis annotations against *its own*
    stage-local sub-mesh, so a TP annotation lands on that rank's TP
    devices rather than on the global mesh.
    """
    leaf_shardings: list[dict[int, Any]] = [{} for _ in range(n)]
    leaf_stage_owners: dict[int, int] = {}
    for arg in outer_args:
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
        for fi, fl in enumerate(outer_flat_args):
            if id(fl) == first_leaf_id:
                offset = fi
                break
        if offset is None:
            continue
        leaf_entries: list[tuple[int, str, str, int | None]] = []
        for li, (col, path) in enumerate(leaf_spec):
            flat_idx = offset + li
            var = vars_by_key.get((col, path))
            owner = resolve_stage_rank(
                metadata_stage_assignment(var.metadata) if var is not None else None,
                n,
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


def _build_grad_metadata(
    outer_args: tuple[Any, ...],
    outer_jaxpr: jax.core.ClosedJaxpr,
    pscan_eqn: JaxprEqn,
    n_body_consts: int,
    probed_grad_tree: Any | None,
) -> tuple[Any, tuple[int, ...], tuple[Any, ...]]:
    """Identify the captured module arg and map its grad leaves to body const indices."""
    module_arg_indices = [i for i, arg in enumerate(outer_args) if isinstance(arg, Module)]
    if len(module_arg_indices) != 1:
        raise ValueError(
            "sxjit+treduce scalar-loss autodiff currently requires exactly one "
            f"captured Module positional argument; found {len(module_arg_indices)}."
        )

    module_arg_index = module_arg_indices[0]
    model = outer_args[module_arg_index]
    _, state = export(model)
    grad_state, _ = as_selector("parameters").partition_state(model, state)
    grad_tree = jax.tree.structure(grad_state)
    grad_template_leaves = tuple(jax.tree.leaves(grad_state))
    if not grad_template_leaves:
        raise ValueError(
            "sxjit+treduce scalar-loss autodiff requires the captured Module "
            "to expose at least one trainable `parameters` leaf."
        )
    if probed_grad_tree is not None and probed_grad_tree != grad_tree:
        raise ValueError(
            "treduce: pre-differentiated body gradient tree does not match the captured Module's `parameters` tree."
        )

    model_leaves = tuple(jax.tree.leaves(model))
    model_leaf_idx_by_id = {id(leaf): idx for idx, leaf in enumerate(model_leaves)}
    grad_leaf_to_model_leaf: list[int] = []
    for leaf in grad_template_leaves:
        model_leaf_idx = model_leaf_idx_by_id.get(id(leaf))
        if model_leaf_idx is None:
            raise RuntimeError(
                "Captured Module params could not be matched back to the Module's "
                "flattened leaves while building the pscan plan."
            )
        grad_leaf_to_model_leaf.append(model_leaf_idx)

    leaf_ranges = _arg_leaf_ranges(outer_args)
    model_start, _ = leaf_ranges[module_arg_index]
    outer_invar_idx_by_id = {id(v): i for i, v in enumerate(outer_jaxpr.jaxpr.invars)}
    const_outer_invar_indices: list[int | None] = []
    for operand in pscan_eqn.invars[:n_body_consts]:
        if isinstance(operand, Var):
            const_outer_invar_indices.append(outer_invar_idx_by_id.get(id(operand)))
        else:
            const_outer_invar_indices.append(None)

    grad_const_indices: list[int] = []
    for model_leaf_idx in grad_leaf_to_model_leaf:
        outer_flat_idx = model_start + model_leaf_idx
        body_const_idx = -1
        for const_idx, outer_idx in enumerate(const_outer_invar_indices):
            if outer_idx == outer_flat_idx:
                body_const_idx = const_idx
                break
        grad_const_indices.append(body_const_idx)

    return grad_tree, tuple(grad_const_indices), grad_template_leaves


def _build_invar_sources(
    body_jaxpr: Jaxpr,
    clusters: list[Jaxpr],
) -> list[list[tuple[str, int, int]]]:
    """Map each cluster input to either a body invar or a prior cluster output."""
    alias_by_id = {
        id(outvar): invar
        for eqn in body_jaxpr.eqns
        if eqn.primitive is sxstage_iter_p
        for invar, outvar in zip(eqn.invars, eqn.outvars, strict=True)
        if isinstance(invar, Var) and isinstance(outvar, Var)
    }

    def _resolve_alias(var: Var) -> Var:
        """Walk through ``sxstage_iter`` outvar -> invar chains to the original ``Var``.

        Mirrors :func:`spectrax.runtime.mpmd.runtime._marker_alias_resolver`'s
        inner helper: stage markers are identity passes, so for the
        purpose of mapping cluster invars back to the body's invars
        we want to skip past them. The ``seen`` set guards against
        cycles.
        """
        cur = var
        seen: set[int] = set()
        while id(cur) in alias_by_id and id(cur) not in seen:
            seen.add(id(cur))
            nxt = alias_by_id[id(cur)]
            if not isinstance(nxt, Var):
                break
            cur = nxt
        return cur

    body_invar_idx_by_id = {id(_resolve_alias(v)): i for i, v in enumerate(body_jaxpr.invars) if isinstance(v, Var)}
    producer_by_var_id: dict[int, tuple[int, int]] = {}
    for logical, cluster in enumerate(clusters):
        for out_idx, outvar in enumerate(cluster.outvars):
            if not isinstance(outvar, Var):
                continue
            producer_by_var_id[id(_resolve_alias(outvar))] = (logical, out_idx)

    invar_sources: list[list[tuple[str, int, int]]] = []
    for logical, cluster in enumerate(clusters):
        cluster_sources: list[tuple[str, int, int]] = []
        for invar in cluster.invars:
            if not isinstance(invar, Var):
                raise TypeError(f"sxjit+treduce expected cluster invars to be JAX Vars; got {type(invar).__name__}.")
            canonical = _resolve_alias(invar)
            producer = producer_by_var_id.get(id(canonical))
            if producer is not None:
                producer_logical, out_idx = producer
                if producer_logical >= logical:
                    raise ValueError(
                        "sxjit+treduce requires cluster inputs to be produced by an "
                        f"earlier stage. Logical stage {logical} read output {out_idx} "
                        f"from logical stage {producer_logical}."
                    )
                cluster_sources.append(("cluster_out", producer_logical, out_idx))
                continue

            body_invar_idx = body_invar_idx_by_id.get(id(canonical))
            if body_invar_idx is not None:
                cluster_sources.append(("body_invar", body_invar_idx, -1))
                continue

            raise ValueError(
                "sxjit+treduce could not map a cluster input to either a prior "
                f"cluster output or a body input. Logical stage {logical} input "
                f"{invar} has no known producer."
            )
        invar_sources.append(cluster_sources)

    return invar_sources


def _build_schedule_grid(schedule: Any, n: int) -> list[list[Any]]:
    """Build a schedule grid with the same fusion passes used by :func:`sxcall`.

    Calls :meth:`Schedule.build` then applies :func:`fuse_1f1b_steady_state`
    for 1F1B-family schedules and :func:`fuse_zerobubble_bwd_pair` for
    :class:`ZeroBubbleH1`. Schedules can opt out by exposing a
    ``_skip_auto_fuse_1f1b`` attribute that returns truthy.

    Args:
        schedule: The active :class:`Schedule`.
        n: Number of physical pipeline ranks.

    Returns:
        The post-processed grid as a list of mutable rows.
    """
    grid = [list(row) for row in schedule.build(n)]
    skip_1f1b_fusion = getattr(schedule, "_skip_auto_fuse_1f1b", False)
    if callable(skip_1f1b_fusion):
        skip_1f1b_fusion = bool(skip_1f1b_fusion())
    if not skip_1f1b_fusion and isinstance(schedule, (Std1F1B, Eager1F1B, InterleavedH1)):
        grid = fuse_1f1b_steady_state(grid)
    if isinstance(schedule, ZeroBubbleH1):
        grid = fuse_zerobubble_bwd_pair(grid)
    return [list(row) for row in grid]


def _put_tree(tree: Any, sharding: Any) -> Any:
    """Apply :func:`jax.device_put` to every array leaf of ``tree``.

    Non-array leaves (e.g. Python scalars or ``None``) are passed through
    unchanged so the function is safe to call on heterogeneous pytrees.

    Args:
        tree: A pytree of arrays / scalars.
        sharding: Sharding (or device list) accepted by
            :func:`jax.device_put`.

    Returns:
        The same pytree structure with array leaves moved to
        ``sharding``.
    """
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if hasattr(x, "shape") else x,
        tree,
    )


def _transport_tuple(vals: tuple[Any, ...], src_rank: int, dst_rank: int, stage_shardings: list[Any]) -> tuple[Any, ...]:
    """Move a tuple of arrays from ``src_rank`` onto ``dst_rank`` if they differ.

    Skips the transport when source and destination physical ranks
    coincide so ``(rank, virt0) -> (rank, virt1)`` cluster edges stay
    in-rank without a redundant ``device_put``.

    Args:
        vals: Arrays to relocate.
        src_rank: Source physical rank.
        dst_rank: Destination physical rank.
        stage_shardings: Per-rank replicated shardings.

    Returns:
        ``vals`` unchanged when ranks match, otherwise a new tuple of
        arrays placed on ``stage_shardings[dst_rank]``.
    """
    if src_rank == dst_rank:
        return vals
    return tuple(jax.device_put(v, stage_shardings[dst_rank]) for v in vals)


def _edge_transfer_target(value: Any, plan: PscanPlan, producer_logical: int, dst_rank: int) -> Any:
    """Resolve a destination sharding for a marker-edge cross-rank transport.

    When the producing logical stage's :func:`sxstage_iter` carried an
    edge ``PartitionSpec``, that spec is sanitised against both the MPMD
    mesh and the destination rank's sub-mesh (axes incompatible with
    either are dropped) and wrapped in a :class:`NamedSharding`.
    Otherwise the destination rank's default replicated sharding is
    used.

    Args:
        value: The array (or pytree of arrays) being transported —
            shape information is needed to sanitise the spec.
        plan: The :class:`PscanPlan` holding edge metadata and meshes.
        producer_logical: Logical stage index that produced ``value``.
        dst_rank: Destination physical rank.

    Returns:
        A sharding (or pytree of shardings) usable with
        :func:`jax.device_put`.
    """
    if not (0 <= producer_logical < len(plan.edge_shardings)):
        return plan.stage_shardings[dst_rank]
    edge_sharding = plan.edge_shardings[producer_logical]
    if edge_sharding is None:
        return plan.stage_shardings[dst_rank]
    dst_mesh = plan.rank_submeshes[dst_rank]

    def leaf_target(leaf: Any) -> Any:
        """Per-leaf NamedSharding on ``dst_mesh`` derived from the edge spec."""
        if not hasattr(leaf, "shape"):
            return plan.stage_shardings[dst_rank]
        spec = sanitize_partition_spec_for_mesh_and_shape(
            edge_sharding,
            mesh=plan.mpmd_mesh,
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
    return jax.tree.map(leaf_target, value)


def _materialize_cotangents(
    partial: list[Any | None] | None,
    outputs: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Replace missing cotangent slots with zero arrays shaped like ``outputs``.

    The dispatcher accumulates downstream cotangents lazily into a
    ``[None, None, ...]`` slot list so an early backward call does not
    have to allocate zero arrays it may never use. By the time the
    upstream backward jit fires, any still-missing slot represents a
    cluster output that no consumer needed; we substitute zero of the
    correct shape so :func:`jax.vjp` sees a complete cotangent tuple.
    Float0-typed slots and dtype-mismatched slots are passed through /
    cast as needed to match XLA's expectations.

    Args:
        partial: Per-output slot list (``None`` means "not yet supplied").
            ``None`` for the whole list means no consumer ever filled
            anything, in which case all-zeros is returned.
        outputs: Original forward outputs used as shape/dtype templates.

    Returns:
        A complete cotangent tuple aligned with ``outputs``.
    """
    if partial is None:
        return tuple(jnp.zeros_like(out) for out in outputs)
    full: list[Any] = []
    for slot, out in zip(partial, outputs, strict=True):
        result = getattr(slot, "result", None)
        if callable(result):
            slot = result()
        if slot is None:
            full.append(jnp.zeros_like(out))
        elif getattr(slot, "dtype", None) == jax.dtypes.float0:
            full.append(slot)
        elif hasattr(slot, "astype") and hasattr(out, "dtype") and getattr(slot, "dtype", None) != out.dtype:
            full.append(slot.astype(out.dtype))
        else:
            full.append(slot)
    return tuple(full)


def _cast_cotangent_like(cotangent: Any, primal: Any) -> Any:
    """Cast a cotangent to its primal output dtype before moving devices."""
    if getattr(cotangent, "dtype", None) == jax.dtypes.float0:
        return cotangent
    cot_dtype = getattr(cotangent, "dtype", None)
    primal_dtype = getattr(primal, "dtype", None)
    if cot_dtype is not None and primal_dtype is not None and cot_dtype != primal_dtype and hasattr(cotangent, "astype"):
        return cotangent.astype(primal_dtype)
    return cotangent


def _add_grad(a: Any, b: Any) -> Any:
    """Add gradient leaves while preserving JAX ``float0`` sentinels."""
    if getattr(a, "dtype", None) == jax.dtypes.float0:
        return b
    if getattr(b, "dtype", None) == jax.dtypes.float0:
        return a
    return a + b


def _project_upstream_cotangents(
    g_invars: tuple[Any, ...],
    src_outs: tuple[Any, ...],
    src_to_dst: list[tuple[int, ...]],
) -> tuple[Any, ...]:
    """Expand downstream input cotangents to the full upstream output tuple."""
    full: list[Any] = []
    for out_idx, dst_indices in enumerate(src_to_dst):
        if not dst_indices:
            full.append(jnp.zeros_like(src_outs[out_idx]))
            continue
        cot = g_invars[dst_indices[0]]
        for dst_idx in dst_indices[1:]:
            cot = jax.tree.map(_add_grad, cot, g_invars[dst_idx])
        full.append(cot)
    return tuple(full)


def _accumulate_const_grads(
    accum: list[Any | None] | None,
    const_indices: tuple[int, ...],
    g_consts: tuple[Any, ...],
    n_total_consts: int,
) -> list[Any | None]:
    """Scatter a cluster's const-grad tuple back into the full body-const slot list.

    A cluster only references a subset of the full body's constvars
    (``const_indices`` are the body-const indices in order). When
    backward fires, we add each per-cluster gradient into the matching
    full-body slot, leaving untouched slots as ``None`` so callers can
    distinguish "no contribution" from "contributed zero".

    Args:
        accum: Per-rank running accumulator (or ``None`` to allocate).
        const_indices: Body-const indices owned by this cluster, in
            cluster-local order.
        g_consts: Per-cluster const grads from a backward jit (parallel
            to ``const_indices``).
        n_total_consts: Total number of body constvars — sets the
            allocated accumulator length.

    Returns:
        Updated ``accum`` (newly allocated when ``accum`` was ``None``).
    """
    if accum is None:
        accum = [None] * n_total_consts
    for local_idx, const_idx in enumerate(const_indices):
        grad = g_consts[local_idx]
        if accum[const_idx] is None:
            accum[const_idx] = grad
        else:
            accum[const_idx] = jax.tree.map(_add_grad, accum[const_idx], grad)
    return accum


def _sum_rank_grads(grad_accums: list[list[Any | None] | None], output_sharding: Any) -> Any | None:
    """Move per-rank const grads to one sharding and sum them leafwise."""
    total = None
    for grads in grad_accums:
        if grads is None:
            continue
        moved = tuple(jax.device_put(g, output_sharding) if g is not None else None for g in grads)
        if total is None:
            total = moved
        else:
            total = tuple(
                b if a is None else a if b is None else _add_grad(a, b) for a, b in zip(total, moved, strict=True)
            )
    return total


def _pack_grad_tree(plan: PscanPlan, total_const_grads: Any | None) -> Any:
    """Unflatten selected const grads into the final model-shaped grad pytree."""
    total_const_grads = () if total_const_grads is None else tuple(total_const_grads)
    grad_leaves: list[Any] = []
    for leaf_idx, const_idx in enumerate(plan.grad_const_indices):
        if const_idx < 0 or const_idx >= len(total_const_grads) or total_const_grads[const_idx] is None:
            leaf = jnp.zeros_like(plan.grad_template_leaves[leaf_idx])
            grad_leaves.append(jax.device_put(leaf, plan.grad_output_sharding))
        else:
            grad_leaves.append(total_const_grads[const_idx])
    return jax.tree_util.tree_unflatten(plan.grad_tree, grad_leaves)


def build_pscan_plan(
    outer_jaxpr: jax.core.ClosedJaxpr,
    outer_args: tuple[Any, ...],
    outer_flat_args: tuple[Any, ...],
    pscan_eqn: JaxprEqn,
    mpmd_mesh: MpMdMesh,
    stage_shardings: list[Any],
    rank_submeshes: list[Any],
) -> PscanPlan:
    """Build the :class:`PscanPlan` from a single ``pscan_p`` equation.

    Resolves the body's closure consts to concrete runtime values via
    the outer jaxpr's invar/constvar mapping, clusters the body by
    markers, maps logical stages onto schedule-defined ``(rank, virt)``
    locations, and compiles forward / backward / terminal jits.

    Args:
        outer_jaxpr: The full outer ``ClosedJaxpr``.
        outer_flat_args: Concrete values for ``outer_jaxpr.jaxpr.invars``.
        pscan_eqn: The target ``pscan_p`` equation.
        mpmd_mesh: MPMD mesh; ``mpmd_dim`` must equal the number of
            physical pipeline ranks.
        stage_shardings: Per-rank replicated shardings.
        rank_submeshes: Per-rank sub-meshes.
    """
    eqn_params = _eqn_params(pscan_eqn)
    loss_closed: jax.core.ClosedJaxpr = eqn_params["loss_jaxpr"]
    body_mode: str = eqn_params["body_mode"]
    probed_grad_tree = eqn_params["grad_tree"]
    schedule = _unwrap_schedule(eqn_params["schedule"])
    ops = _unwrap_ops(eqn_params["ops"])
    m = eqn_params["n_mubatches"]
    n_outs = eqn_params["n_outs"]
    n_outer_consts = eqn_params["n_consts"]

    n = mpmd_mesh.mpmd_dim
    v = schedule.virtual_stages_per_rank()
    n_logical = n * v
    loc_for_logical, logical_for_loc, terminal_loc = _build_logical_locs(schedule, n, v)

    edge_shardings = marker_edge_shardings(loss_closed.jaxpr)
    clusters = cluster_jaxpr_by_markers(loss_closed.jaxpr)
    if len(clusters) != n_logical:
        raise ValueError(
            f"pscan body has {len(clusters)} stages "
            f"({len(clusters) - 1} sxstage_iter markers) but mesh "
            f"has {n} MPMD ranks with V={v} virtual stages. Need exactly "
            f"{n_logical} logical stages ({n_logical - 1} markers) in the "
            f"function passed to treduce for {type(schedule).__name__}."
        )

    all_constvars = list(loss_closed.jaxpr.constvars)
    concrete_consts, const_flat_arg_indices = _resolve_concrete_consts(
        outer_jaxpr,
        outer_flat_args,
        pscan_eqn,
        n_outer_consts,
    )
    if len(concrete_consts) != len(all_constvars):
        raise RuntimeError(
            f"pscan body has {len(all_constvars)} constvars but only "
            f"{len(concrete_consts)} concrete operands resolved. "
            f"Structural mismatch in pscan_p bind."
        )

    grad_tree, grad_const_indices, grad_template_leaves = _build_grad_metadata(
        outer_args,
        outer_jaxpr,
        pscan_eqn,
        n_outer_consts,
        probed_grad_tree,
    )

    leaf_shardings, leaf_stage_owners = _infer_outer_leaf_shardings(
        outer_args,
        outer_flat_args,
        n,
        rank_submeshes,
    )

    per_loc_consts: dict[tuple[int, int], tuple[Any, ...]] = {}
    const_indices_per_loc: dict[tuple[int, int], tuple[int, ...]] = {}
    n_invars_per_loc: dict[tuple[int, int], int] = {}
    fwd_jits: dict[tuple[int, int], Callable[..., Any]] = {}
    bwd_jits: dict[tuple[int, int], Callable[..., Any] | None] = {}

    terminal_jit: Callable[..., Any] | None = None
    all_const_idx_by_id = {id(v): i for i, v in enumerate(all_constvars)}

    for logical, cluster in enumerate(clusters):
        loc = loc_for_logical[logical]
        rank, _virt = loc
        used_constvars = _collect_used_constvars(cluster)
        filtered_cluster = _filtered_cluster(cluster, used_constvars)
        const_indices = tuple(all_const_idx_by_id[id(v)] for v in used_constvars)
        n_invars = len(cluster.invars)

        per_loc_consts[loc] = _place_cluster_consts(
            used_constvars,
            all_constvars,
            concrete_consts,
            const_flat_arg_indices,
            leaf_shardings[rank],
            leaf_stage_owners,
            stage_shardings[rank],
            rank,
        )
        const_indices_per_loc[loc] = const_indices
        n_invars_per_loc[loc] = n_invars
        fwd_jits[loc] = _make_fwd_jit(filtered_cluster)

        if loc != terminal_loc:
            bwd_jits[loc] = _make_bwd_jit(filtered_cluster, n_invars)
        else:
            bwd_jits[loc] = None
            terminal_jit = _make_terminal_jit(filtered_cluster, n_invars)

    assert terminal_jit is not None

    invar_sources = _build_invar_sources(loss_closed.jaxpr, clusters)

    grid = _build_schedule_grid(schedule, n)

    init_state_template = [ops[0].state(loss_closed.jaxpr.outvars[0].aval)]

    return PscanPlan(
        n=n,
        v=v,
        n_logical=n_logical,
        m=m,
        schedule=schedule,
        ops=ops,
        n_outs=n_outs,
        n_outer_consts=n_outer_consts,
        body_mode=body_mode,
        stage_shardings=stage_shardings,
        rank_submeshes=rank_submeshes,
        mpmd_mesh=mpmd_mesh,
        loc_for_logical=loc_for_logical,
        logical_for_loc=logical_for_loc,
        terminal_loc=terminal_loc,
        per_loc_consts=per_loc_consts,
        const_indices_per_loc=const_indices_per_loc,
        n_invars_per_loc=n_invars_per_loc,
        fwd_jits=fwd_jits,
        bwd_jits=bwd_jits,
        terminal_jit=terminal_jit,
        init_state_template=init_state_template,
        grad_tree=grad_tree,
        grad_const_indices=grad_const_indices,
        grad_template_leaves=grad_template_leaves,
        grad_output_sharding=stage_shardings[0],
        invar_sources=invar_sources,
        edge_shardings=edge_shardings,
        grid=grid,
    )


def _iter_actions(row: list[Any]):
    """Yield ``(rank, virt, action)`` triples, expanding :class:`FusedTask` cells.

    A grid cell may be a plain ``Action``, ``None``, or a fused fwd+bwd
    pair. We do not perform downstream fusion here — each component
    action dispatches separately.
    """
    for rank, cell in enumerate(row):
        if cell is None:
            continue
        if isinstance(cell, FusedTask):
            yield rank, cell.fwd.virtual_stage, cell.fwd
            yield rank, cell.bwd.virtual_stage, cell.bwd
        else:
            yield rank, cell.virtual_stage, cell


def dispatch_pscan(plan: PscanPlan) -> list[Any]:
    """Run the schedule-driven dispatch loop and return accumulator values.

    Walks ``plan.grid`` step by step, firing the per-rank cluster jit
    appropriate to each action's phase. Cross-rank arrays move via
    :func:`jax.device_put` onto the destination rank's sub-mesh.

    Returns ``[losses, grads]`` where ``losses`` is the concatenated
    per-microbatch loss buffer and ``grads`` matches the captured
    model's param pytree.
    """
    n = plan.n
    ops = plan.ops
    grid = plan.grid

    fwd_inputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    fwd_outputs: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    cotangents_into: dict[tuple[int, int, int], list[Any | None]] = {}

    grad_accums: list[list[Any | None] | None] = [None] * n
    loss_acc = plan.init_state_template[0] if plan.n_outs >= 1 else None

    for row in grid:
        for rank, virt, action in _iter_actions(row):
            mb = action.microbatch
            phase = action.phase
            loc = (rank, virt)
            logical = plan.logical_for_loc[loc]
            submesh = plan.rank_submeshes[rank]
            mb_idx = jnp.asarray(mb, dtype=jnp.int32)
            consts = plan.per_loc_consts[loc]
            key = (rank, virt, mb)

            if phase is Phase.FWD:
                invars_list: list[Any] = []
                for source_kind, source_a, source_b in plan.invar_sources[logical]:
                    if source_kind == "body_invar":
                        if source_a != 0:
                            raise NotImplementedError(
                                "sxjit+treduce currently supports only the microbatch index as a direct body input."
                            )
                        invars_list.append(mb_idx)
                        continue

                    producer_logical = source_a
                    producer_out_idx = source_b
                    producer_loc = plan.loc_for_logical[producer_logical]
                    producer_key = (producer_loc[0], producer_loc[1], mb)
                    val = fwd_outputs[producer_key][producer_out_idx]
                    if producer_loc[0] != rank:
                        val = jax.device_put(val, _edge_transfer_target(val, plan, producer_logical, rank))
                    invars_list.append(val)

                invars = tuple(invars_list)
                with submesh:
                    out = plan.fwd_jits[loc](consts, *invars)
                fwd_inputs[key] = invars
                fwd_outputs[key] = out

                if loc == plan.terminal_loc:
                    loss_val = out[0]
                    loss_acc = ops[0].update(loss_acc, loss_val, mb_idx)

            elif phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                invars = fwd_inputs[key]

                if loc == plan.terminal_loc:
                    with submesh:
                        _, (g_consts, g_invars) = plan.terminal_jit(consts, *invars)
                else:
                    cot = _materialize_cotangents(cotangents_into.get(key), fwd_outputs[key])
                    with submesh:
                        g_consts, g_invars = plan.bwd_jits[loc](
                            consts,
                            *invars,
                            *cot,
                        )

                if phase is not Phase.BWD_I:
                    grad_accums[rank] = _accumulate_const_grads(
                        grad_accums[rank],
                        plan.const_indices_per_loc[loc],
                        g_consts,
                        plan.n_outer_consts,
                    )

                if phase is not Phase.BWD_W:
                    for invar_idx, (source_kind, source_a, source_b) in enumerate(plan.invar_sources[logical]):
                        if source_kind != "cluster_out":
                            continue
                        producer_logical = source_a
                        producer_out_idx = source_b
                        producer_loc = plan.loc_for_logical[producer_logical]
                        producer_key = (producer_loc[0], producer_loc[1], mb)
                        cot = g_invars[invar_idx]
                        cot = _cast_cotangent_like(cot, fwd_outputs[producer_key][producer_out_idx])
                        if producer_loc[0] != rank:
                            cot = jax.device_put(
                                cot,
                                _edge_transfer_target(cot, plan, producer_logical, producer_loc[0]),
                            )
                        slots = cotangents_into.setdefault(
                            producer_key,
                            [None] * len(fwd_outputs[producer_key]),
                        )
                        if slots[producer_out_idx] is None:
                            slots[producer_out_idx] = cot
                        else:
                            slots[producer_out_idx] = slots[producer_out_idx] + cot

    total_const_grads = _sum_rank_grads(grad_accums, plan.grad_output_sharding)
    grads = _pack_grad_tree(plan, total_const_grads)
    return [loss_acc, grads]
