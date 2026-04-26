# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPMD pipeline-parallel runtime — scan-free, vmap-based.

:func:`spmd_run` is the public entry point. It extracts+stacks per-stage
state (cached by model id), shards params along the pipeline axis so
stage ``i`` lives on device ``i``, microbatches the batch, and dispatches
a jitted step built by :func:`_build_spmd_step`.

The jitted step traces a straight-line forward over every stage via
static indexing into the stacked params (``stacked_params[s]``), then
wraps the whole thing in :func:`jax.value_and_grad` so XLA produces the
reverse pass and schedules forward and backward ops across devices based
on the initial param sharding — no manual schedule table, no ppermutes,
no nested conds: just regular SPMD.

All stages must share the same :class:`GraphDef`; heterogeneous stages
should use :func:`spectrax.runtime.sxcall` instead.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ...core.graph import bind, export
from ...core.state import State
from ...core.variable import Variable
from ...nn.pipeline_sequential import PipelineSequential
from ..schedules import Schedule

__all__ = ["spmd_run"]

_COMPILE_CACHE: dict[Any, Callable[..., Any]] = {}
_StageSignature = tuple[tuple[str, str, int, tuple[int, ...], str], ...]

_STAGE_EXTRACT_CACHE: dict[int, tuple[_StageSignature, Any, State, State]] = {}
_PLACED_CACHE: dict[Any, tuple[State, State]] = {}


def _microbatch(x: jax.Array, m: int) -> jax.Array:
    """Reshape ``x`` (leading axis = batch) into ``(m, batch//m, ...)``."""
    b = x.shape[0]
    if b % m:
        raise ValueError(f"Batch size {b} not divisible by number of microbatches {m}.")
    return x.reshape(m, b // m, *x.shape[1:])


def _is_leaf(x: Any) -> bool:
    """Treat JAX arrays / Variables as pytree leaves."""
    return isinstance(x, jax.Array | Variable)


def _extract_stages(container: PipelineSequential) -> tuple[Any, tuple[State, ...]]:
    """Export each stage; verify all share the same :class:`GraphDef`."""
    if not isinstance(container, PipelineSequential):
        raise TypeError(f"spx.pipeline expects a PipelineSequential, got {type(container).__name__}.")
    stages = container.stages
    gdef0, state0 = export(stages[0])
    states = [state0]
    for i, stage in enumerate(stages[1:], start=1):
        gdef_i, state_i = export(stage)
        if gdef_i != gdef0:
            raise ValueError(
                f"Stage {i} has a different GraphDef than stage 0. SPMD "
                f"pipeline requires all stages to be structurally "
                f"identical. For heterogeneous stages, use sxcall."
            )
        states.append(state_i)
    return gdef0, tuple(states)


def _stack_states(states: tuple[State, ...]) -> State:
    """Stack per-stage states along a leading ``n_stages`` axis."""
    out: dict[str, dict[str, Any]] = {}
    first_items = tuple(states[0].items())
    for c, p, _ in first_items:
        out.setdefault(c, {})[p] = jnp.stack([s.get(c, p) for s in states], axis=0)
    return State(out)


@functools.partial(jax.jit, static_argnums=(1,))
def _unstack_leaves_jit(stacked_leaves, n: int):
    """Split a flat tuple of stacked arrays into ``n`` tuples of slices.

    ``n`` is static at trace time; each output slice is a static
    offset index. Wrapping in ``@jax.jit`` collapses what was a
    ~0.9 ms per-leaf eager dispatch (hundreds of ms total on TPU)
    into one compiled kernel that XLA can fuse with downstream work.
    """
    return tuple(tuple(leaf[i] for leaf in stacked_leaves) for i in range(n))


def _unstack_state(stacked: State, n: int) -> tuple[State, ...]:
    """Inverse of :func:`_stack_states`.

    Runs the per-leaf slicing inside one jitted kernel instead of N
    eager dispatches per leaf. For ~200-leaf State on TPU v5p this
    cuts ~170 ms off every call.
    """
    raw = stacked.raw()
    paths: list[tuple[str, str]] = []
    leaves: list[Any] = []
    for c, d in raw.items():
        for p, v in d.items():
            paths.append((c, p))
            leaves.append(v)
    if not leaves:
        return tuple(State({c: {} for c in raw}) for _ in range(n))
    sliced = _unstack_leaves_jit(tuple(leaves), n)
    out: list[dict[str, dict[str, Any]]] = [{c: {} for c in raw} for _ in range(n)]
    for i in range(n):
        for (c, p), v in zip(paths, sliced[i], strict=True):
            out[i][c][p] = v
    return tuple(State(o) for o in out)


def _split_params_rest(state: State) -> tuple[State, State]:
    """Split ``state`` into ``(params, rest)``. Only params differentiate."""
    raw = state.raw()
    params_raw: dict[str, dict[str, Any]] = {}
    rest_raw: dict[str, dict[str, Any]] = {}
    for c, d in raw.items():
        (params_raw if c == "parameters" else rest_raw)[c] = dict(d)
    return State(params_raw), State(rest_raw)


def spmd_run(
    model: PipelineSequential,
    batch: tuple[Any, ...],
    *,
    mesh: Mesh,
    axis: str,
    schedule: Schedule,
    loss_fn: Callable[..., jax.Array],
) -> tuple[jax.Array, tuple[State, ...]]:
    """Execute one pipeline-parallel forward + backward step.

    See module docstring for the runtime's design. Returns
    ``(mean_loss, per_stage_param_grads)``.

    Args:
        model: :class:`PipelineSequential` (all stages same GraphDef).
        batch: Tuple of positional inputs; first element is the
            pipeline input, rest are targets for ``loss_fn`` at the
            final stage. Each leading axis is microbatched by
            ``schedule.microbatches``.
        mesh: JAX mesh. ``mesh.shape[axis]`` must equal
            ``model.num_stages``.
        axis: Named pipeline axis of ``mesh``.
        schedule: A :class:`Schedule` (GPipe, Std1F1B, ZB-H1,
            Interleaved).
        loss_fn: ``(final_stage_output, *batch[1:]) -> scalar``.

    Returns:
        ``(loss, (per_stage_grads, ...))``. ``loss`` is the mean loss
        across microbatches (broadcast identical on every pipeline
        device via an all-reduce); ``per_stage_grads`` is a tuple of
        ``parameters``-only :class:`State` s.

    The stacked state is sharded along the pipeline axis so stage
    ``i``'s parameters live on device ``i``. XLA uses this placement to
    automatically route activations / cotangents through the pipeline
    in the :func:`jax.jit`'d step function below. Placement is cached
    by ``(id(model), id(mesh), axis)`` so repeat calls skip the
    ``device_put`` entirely — even ``jax.device_put`` on already-placed
    arrays is ~10 ms on TPU v5p for a ~200-leaf pytree. The step
    function returns grads already per-stage-unstacked inside the jit
    (one compiled layout transform rather than a separate outer
    dispatch).
    """
    if model.num_stages != mesh.shape[axis]:
        raise ValueError(f"Model has {model.num_stages} stages but mesh axis {axis!r} has size {mesh.shape[axis]}.")

    gdef, state_signature, stacked_params, stacked_rest = _extract_and_stack(model)

    n = model.num_stages
    m = schedule.microbatches

    pp_sharding = NamedSharding(mesh, PartitionSpec(axis))
    placed_key = (id(model), id(mesh), axis, state_signature)
    placed = _PLACED_CACHE.get(placed_key)
    if placed is None:
        stacked_params = jax.device_put(stacked_params, pp_sharding)
        stacked_rest = jax.device_put(stacked_rest, pp_sharding)
        _PLACED_CACHE[placed_key] = (stacked_params, stacked_rest)
    else:
        stacked_params, stacked_rest = placed

    mb_batch = tuple(_microbatch(x, m) for x in batch)

    step_fn = _get_cached_spmd_step(
        gdef=gdef,
        n_stages=n,
        microbatches=m,
        loss_fn=loss_fn,
    )

    loss, grads_per_stage = step_fn(stacked_params, stacked_rest, *mb_batch)
    return loss, tuple(grads_per_stage)


def _extract_and_stack(
    container: PipelineSequential,
) -> tuple[Any, _StageSignature, State, State]:
    """Return the cached ``(gdef, stacked_params, stacked_rest)`` for ``container``.

    The first call walks every stage, exports their state, and stacks
    along a leading stage axis. Subsequent calls with the same
    container instance return the cached tensors directly.

    Keyed on ``id(container)`` since users typically reuse the same
    model across training steps. Without this, every ``spmd_run`` call
    re-exports every stage and re-runs ``jnp.stack`` on every parameter
    leaf — eager dispatches that cost ~0.3 ms each on TPU and dominate
    step time at small batch sizes.
    """
    gdef, per_stage_states = _extract_stages(container)
    signature = _stage_state_signature(per_stage_states)
    cache_key = id(container)
    cached = _STAGE_EXTRACT_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[0], cached[2], cached[3]
    per_stage_pr = tuple(_split_params_rest(s) for s in per_stage_states)
    stacked_params = _stack_states(tuple(p for p, _ in per_stage_pr))
    stacked_rest = _stack_states(tuple(r for _, r in per_stage_pr))
    _STAGE_EXTRACT_CACHE[cache_key] = (signature, gdef, stacked_params, stacked_rest)
    return gdef, signature, stacked_params, stacked_rest


def _stage_state_signature(states: tuple[State, ...]) -> _StageSignature:
    """Signature that changes when live stage leaves are replaced."""
    out: list[tuple[str, str, int, tuple[int, ...], str]] = []
    for stage_idx, state in enumerate(states):
        for collection, path, leaf in state.items():
            out.append(
                (
                    f"{stage_idx}:{collection}",
                    path,
                    id(leaf),
                    tuple(int(dim) for dim in getattr(leaf, "shape", ())),
                    str(getattr(leaf, "dtype", type(leaf).__name__)),
                )
            )
    return tuple(out)


def _get_cached_spmd_step(
    *,
    gdef: Any,
    n_stages: int,
    microbatches: int,
    loss_fn: Callable[..., jax.Array],
) -> Callable[..., Any]:
    """Return the jitted step fn for this config, building + caching if needed.

    The cache is keyed on ``(gdef, n_stages, microbatches, id(loss_fn))``.
    ``loss_fn`` is keyed by identity (not structural equality) so two
    distinct closures computing the same loss would still miss the
    cache — that matches how :func:`jax.jit` itself keys on function
    identity.
    """
    key = (gdef, n_stages, microbatches, id(loss_fn))
    fn = _COMPILE_CACHE.get(key)
    if fn is None:
        fn = _build_spmd_step(
            gdef=gdef,
            n_stages=n_stages,
            microbatches=microbatches,
            loss_fn=loss_fn,
        )
        _COMPILE_CACHE[key] = fn
    return fn


def _build_spmd_step(
    *,
    gdef: Any,
    n_stages: int,
    microbatches: int,
    loss_fn: Callable[..., jax.Array],
) -> Callable[..., Any]:
    """Build a jitted step function: ``(stacked_params, stacked_rest, *mb_batch) -> (loss, stacked_grads)``.

    The step traces a straight-line computation:

    1. For each microbatch ``mb``, thread the input through every
       stage sequentially by indexing ``stacked_params[s]`` at the
       static offset ``s``. Because ``stacked_params`` is sharded
       along the pipeline axis before the call, XLA places stage
       ``s``'s forward on device ``s`` and automatically inserts the
       cross-device transfer between stages.
    2. Compute the microbatch loss and add it to an accumulator.
    3. After all microbatches, divide by ``M`` to recover the
       single-device mean.
    4. Wrap the whole thing in :func:`jax.value_and_grad` so JAX
       generates the reverse pass. XLA schedules forward and
       backward ops across devices based on the initial param
       sharding — no manual schedule table, no ppermutes, no nested
       conds: just regular SPMD.

    The per-microbatch forward is vmapped over the leading
    microbatches axis rather than unrolled into ``M`` copies of the
    HLO. Vmap produces a single compiled forward that's reused across
    microbatches — smaller HLO, faster compile, and on modern TPU
    XLA pipelines the vmapped axis across stages efficiently. The
    grad unstack runs inside the jit: XLA turns this into a layout
    transform (free at runtime). The alternative — an outer jitted
    ``_unstack_state`` call — costs an extra dispatch (~1-3 ms on TPU)
    that we can save here.
    """

    def stage_apply(params, rest, x):
        """Apply a stage's forward to ``x`` using its ``(params, rest)``."""
        return bind(gdef, params.overlay(rest))(x)

    def forward_through_stages(stacked_params, stacked_rest, x):
        """Run ``x`` forward through all ``n_stages`` stages in order."""
        for s in range(n_stages):
            idx = s
            p_s = jax.tree.map(lambda t, i=idx: t[i], stacked_params, is_leaf=_is_leaf)
            r_s = jax.tree.map(lambda t, i=idx: t[i], stacked_rest, is_leaf=_is_leaf)
            x = stage_apply(p_s, r_s, x)
        return x

    def total_loss(stacked_params, stacked_rest, mb_batch):
        """Mean per-microbatch loss, vmapped over the mb axis."""
        xs = mb_batch[0]
        targets = mb_batch[1:]

        def per_mb(x_mb, *tgt_mb):
            """Scalar loss for a single microbatch."""
            out = forward_through_stages(stacked_params, stacked_rest, x_mb)
            return loss_fn(out, *tgt_mb)

        per_mb_losses = jax.vmap(per_mb)(xs, *targets)
        return per_mb_losses.mean().astype(jnp.float32)

    @jax.jit
    def step(stacked_params, stacked_rest, *mb_batch):
        """Jitted step: ``(stacked_params, stacked_rest, *mb_batch) -> (loss, per_stage_grads)``."""
        loss, grads_stacked = jax.value_and_grad(total_loss)(stacked_params, stacked_rest, mb_batch)
        grads_per_stage = tuple(
            jax.tree.map(lambda t, i=s: t[i], grads_stacked, is_leaf=_is_leaf) for s in range(n_stages)
        )
        return loss, grads_per_stage

    return step


def _clear_cache() -> None:
    """Wipe the :func:`spmd_run` compile cache. Mainly for tests."""
    _COMPILE_CACHE.clear()
