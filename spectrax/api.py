# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`spectrax.run` — one entry point for SPMD or MPMD execution.

Write the model once as a normal :class:`spectrax.Module`. Pass it
to ``spx.run(model, inputs=..., mesh=mesh, mode=...)`` along with a
mesh; the mesh decides whether to run pure SPMD (``pjit``) or MPMD
(``sxcall`` with auto-split into per-rank stages).

Examples::

    # SPMD only — pjit, no pipeline parallelism
    mesh = spx.create_mesh(axis_dims=(1, 1, -1, 1, 1, 1))
    out  = spx.run(model, inputs=ids, mesh=mesh, mode='forward')

    # Add pipeline parallelism — same call site, mesh changes
    mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis='pp')
    loss, grads = spx.run(model, inputs=ids, targets=labels,
                          mesh=mesh, mode='train', loss_fn=ce,
                          microbatches=4)

``inputs`` and ``targets`` accept three forms (Option C):

* a single array  -> forwarded as-is
* a tuple/list    -> unpacked as positional args
* a dict          -> unpacked as kwargs

So ``inputs=ids`` calls ``model.forward(ids)``;
``inputs=(ids, mask)`` calls ``model.forward(ids, mask)``;
``inputs=dict(ids=ids, mask=mask)`` calls ``model.forward(ids=ids, mask=mask)``.
Same shape rules for ``targets`` against ``loss_fn``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax

from .core._weakcache import weak_invalidate
from .core.graph import bind, export
from .core.module import Module
from .core.paths import str_to_path
from .core.state import _nested_set
from .runtime.mpmd.runtime import sxcall
from .runtime.schedules import GPipe as _DefaultGPipe
from .sharding.mesh import SpxMesh
from .sharding.partition import get_named_sharding

__all__ = ["run"]

_SPMD_FWD_CACHE: dict[int, Callable] = {}
_SPMD_TRAIN_CACHE: dict[tuple, Callable] = {}
_SPMD_STATE_CACHE: dict[tuple, Any] = {}


def _as_call(payload: Any) -> tuple[tuple, dict]:
    """Normalize an inputs/targets payload into ``(args, kwargs)`` for call.

    * ``None``                -> ``((), {})``
    * single array            -> ``((array,), {})``
    * tuple / list of values  -> ``(tuple(...), {})``
    * dict                    -> ``((), dict(...))``
    """
    if payload is None:
        return (), {}
    if isinstance(payload, Mapping):
        return (), dict(payload)
    if isinstance(payload, (tuple, list)):
        return tuple(payload), {}
    return (payload,), {}


def _place_state(state, model: Module, mesh):
    """Apply per-leaf shardings derived from the model's logical
    axis-name annotations (resolved against the active
    :func:`logical_axis_rules` context) to a State pytree."""
    shards = get_named_sharding(model, mesh)
    out: dict[str, dict[str, Any]] = {}
    for col, path, leaf in state.items():
        sh = shards.get(col, {}).get(path)
        placed = jax.device_put(leaf, sh) if sh is not None else leaf
        _nested_set(out.setdefault(col, {}), str_to_path(path), placed)
    return type(state)._from_raw(out, writers=state._writers)


def _mpmd_dummy_loss(y, *_a):
    """Stable dummy loss for forward-only MPMD (module-level so ``id()`` is constant)."""
    return jax.numpy.zeros((), dtype=jax.numpy.float32)


def _run_spmd(
    model: Module,
    args: tuple,
    kwargs: dict,
    *,
    mesh: SpxMesh,
    mode: str,
    loss_args: tuple,
    loss_kwargs: dict,
    loss_fn: Callable | None,
):
    """Run ``model`` under pjit: forward-only or train (value_and_grad).

    Jit-wrapped functions and placed state are cached so repeated
    calls (e.g. decode loop) reuse the compiled program and skip
    redundant ``_place_state`` walks.
    """
    gdef, state = export(model)
    jax_mesh = mesh.jax_mesh

    state_key = (id(gdef), id(jax_mesh))
    cached_state = _SPMD_STATE_CACHE.get(state_key)
    if cached_state is not None:
        state = cached_state
    else:
        state = _place_state(state, model, jax_mesh)
        _SPMD_STATE_CACHE[state_key] = state
        weak_invalidate(gdef, _SPMD_STATE_CACHE, state_key)
        weak_invalidate(jax_mesh, _SPMD_STATE_CACHE, state_key)

    if mode == "forward":
        cache_key = id(gdef)
        _fwd = _SPMD_FWD_CACHE.get(cache_key)
        if _fwd is None:

            @jax.jit
            def _fwd(state, *a, **kw):
                """Bind ``gdef`` to ``state`` and call the resulting module."""
                return bind(gdef, state)(*a, **kw)

            _SPMD_FWD_CACHE[cache_key] = _fwd
            weak_invalidate(gdef, _SPMD_FWD_CACHE, cache_key)

        with jax_mesh:
            return _fwd(state, *args, **kwargs)

    if loss_fn is None:
        raise ValueError("loss_fn required for mode='train'.")

    train_key = (id(gdef), id(loss_fn))
    _step = _SPMD_TRAIN_CACHE.get(train_key)
    if _step is None:

        @jax.jit
        def _step(state, args, kwargs, l_args, l_kwargs):
            """Compute ``(loss, grads)`` via :func:`jax.value_and_grad` on ``state``."""

            def loss(state):
                """Forward pass + loss closure over captured args/kwargs."""
                out = bind(gdef, state)(*args, **kwargs)
                return loss_fn(out, *l_args, **l_kwargs)

            return jax.value_and_grad(loss)(state)

        _SPMD_TRAIN_CACHE[train_key] = _step
        weak_invalidate(gdef, _SPMD_TRAIN_CACHE, train_key)
        weak_invalidate(loss_fn, _SPMD_TRAIN_CACHE, train_key)

    with jax_mesh:
        loss_val, grads = _step(state, args, kwargs, loss_args, loss_kwargs)
    return loss_val, grads


def _run_mpmd(
    model: Module,
    args: tuple,
    kwargs: dict,
    *,
    mesh: SpxMesh,
    mode: str,
    loss_args: tuple,
    loss_kwargs: dict,
    loss_fn: Callable | None,
    microbatches: int,
    schedule: Any = None,
    fuse_1f1b: bool | None = None,
    fuse_zb: bool | None = None,
    has_aux: bool = False,
):
    """Auto-split ``model`` into per-rank stages and dispatch.

    When ``schedule`` is provided, routes through :func:`sxcall`
    which handles all 9 schedules (flat + virtual-stage) at any model
    scale. When ``schedule`` is ``None``, uses the default GPipe path
    via :func:`sxcall`.

    ``fuse_1f1b`` and ``fuse_zb`` enable :class:`FusedTask` steady-state
    fusion on :func:`sxcall` (dispatch-count reduction for 1F1B and
    ZeroBubble families). When ``None`` (default), fusion is enabled
    automatically for compatible schedules.
    """
    if mode not in {"forward", "train"}:
        raise ValueError(f"mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "train" and loss_fn is None:
        raise ValueError("loss_fn required for mode='train'.")
    if not args:
        raise ValueError("MPMD execution requires at least one positional input batch.")

    if loss_fn is not None and loss_kwargs:
        target_keys = tuple(loss_kwargs.keys())
        target_vals = tuple(loss_kwargs.values())
        original_loss = loss_fn

        def _wrapped_loss(out, *vals):
            """Call ``original_loss`` with positional ``vals`` re-keyed to ``target_keys``."""
            return original_loss(out, **dict(zip(target_keys, vals, strict=True)))

        loss_fn = _wrapped_loss
        loss_args = loss_args + target_vals

    if schedule is not None:
        batch = (args[0], *tuple(loss_args)) if mode == "train" else (args[0],)
        return sxcall(
            model,
            batch,
            mesh=mesh.mpmd_mesh,
            schedule=schedule,
            loss_fn=loss_fn,
            fuse_1f1b=fuse_1f1b,
            fuse_zb=fuse_zb,
            mode=mode,
            has_aux=has_aux,
        )

    batch = (args[0], *tuple(loss_args)) if mode == "train" else (args[0],)
    return sxcall(
        model,
        batch,
        mesh=mesh.mpmd_mesh,
        schedule=_DefaultGPipe(microbatches=max(microbatches, 1)),
        loss_fn=loss_fn if mode == "train" else _mpmd_dummy_loss,
        fuse_1f1b=fuse_1f1b,
        fuse_zb=fuse_zb,
        mode=mode,
        has_aux=has_aux,
    )


def run(
    model: Module,
    *,
    inputs: Any,
    targets: Any = None,
    mesh: SpxMesh,
    mode: Literal["train", "forward"] = "forward",
    loss_fn: Callable | None = None,
    microbatches: int = 1,
    schedule: Any = None,
    fuse_1f1b: bool | None = None,
    fuse_zb: bool | None = None,
    has_aux: bool = False,
) -> Any:
    """Run a model under SPMD or MPMD — the mesh decides which.

    Two modes, because that's all there is:

    * ``"forward"`` — run the model, no autograd. This is both
      inference and the decode primitive: to generate, call
      ``"forward"`` in a loop and feed the returned per-stage state
      back in next step (KV cache, beam records, whatever lives in
      stage state).
    * ``"train"`` — forward + ``loss_fn`` + backward. Returns
      ``(loss, grads)``.

    Args:
        model: An :class:`spectrax.Module`. For MPMD, the model is
            auto-split into per-rank stages — either via
            ``model.pipeline_split(n_pp)`` if defined or by detecting
            a ``blocks: ModuleList`` attribute.
        inputs: Forward arguments for ``model.forward``. Accepts:

            * a single array -> ``forward(array)``
            * tuple/list      -> ``forward(*payload)``
            * dict            -> ``forward(**payload)``

        targets: Loss targets passed to ``loss_fn`` after the output.
            Same shape rules as ``inputs``. Required for ``mode="train"``.
        mesh: An :class:`SpxMesh` (built via :func:`spectrax.create_mesh`).
            ``mesh.is_mpmd`` decides the path:

            * ``False`` -> pjit (pure SPMD, FSDP/TP via the model's
              :func:`logical_axis_rules` annotations).
            * ``True``  -> split the model and call
              :func:`sxcall` (PP x FSDP x TP).

        mode: ``"forward"`` or ``"train"``.
        loss_fn: Required for ``mode="train"``. Called as
            ``loss_fn(output, *target_args, **target_kwargs)``.
        microbatches: Pipeline microbatch count. Ignored for SPMD.

    Returns:
        * ``mode="forward"``  -> ``output`` (same shape under SPMD and MPMD).
        * ``mode="train"``    -> ``(loss_scalar, grads)``. Under SPMD ``grads``
          is a single State; under MPMD it's a ``tuple[per_rank_State]``.

    Note:
        For MPMD decode that needs the per-rank stage state out (KV cache
        threading), drop to :func:`sxcall` directly — its
        ``mode='forward'`` returns ``(output, tuple[per_rank_state])``.
        ``spx.run`` hides that to keep its return shape uniform across
        SPMD and MPMD.
    """
    if not isinstance(mesh, SpxMesh):
        raise TypeError(f"mesh must be an SpxMesh (build via spx.create_mesh); got {type(mesh).__name__}.")
    if mode not in {"forward", "train"}:
        raise ValueError(f"mode must be 'forward' or 'train', got {mode!r}.")
    if mode == "forward" and targets is not None:
        raise ValueError("targets are only valid for mode='train'; forward mode does not consume loss targets.")

    args, kwargs = _as_call(inputs)
    loss_args, loss_kwargs = _as_call(targets)

    if mesh.is_mpmd:
        if len(args) != 1 or kwargs:
            raise ValueError(
                "MPMD (pipeline) execution requires exactly one positional "
                "model input, microbatched along its leading axis; got "
                f"{len(args)} positional and {len(kwargs)} keyword inputs. "
                "Pass `inputs=<single_array>` (a bare array or a 1-tuple). "
                "Multi-input / dict-shaped inputs are not supported under "
                "MPMD yet — use an SPMD mesh or fold the inputs into one "
                "batched array."
            )
        return _run_mpmd(
            model,
            args,
            kwargs,
            mesh=mesh,
            mode=mode,
            loss_args=loss_args,
            loss_kwargs=loss_kwargs,
            loss_fn=loss_fn,
            microbatches=microbatches,
            schedule=schedule,
            fuse_1f1b=fuse_1f1b,
            fuse_zb=fuse_zb,
            has_aux=has_aux,
        )
    if schedule is not None:
        raise ValueError(
            "schedule= is only meaningful with an MPMD mesh "
            "(create_mesh(mpmd_axis=...)); got a pure SPMD mesh. "
            "SPMD has no pipeline schedule — drop schedule= or use "
            "an MPMD mesh."
        )
    return _run_spmd(
        model,
        args,
        kwargs,
        mesh=mesh,
        mode=mode,
        loss_args=loss_args,
        loss_kwargs=loss_kwargs,
        loss_fn=loss_fn,
    )
