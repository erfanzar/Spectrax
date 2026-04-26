# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware ``jax.jit`` wrapper.

Every keyword supported by :func:`jax.jit` is available here with the
same default as upstream. SpecTrax adds ``mutable`` (selects which
collections, e.g. ``"batch_stats"``, ``"cache"``, may be written back
to live modules after the transformed call — any other write raises
:class:`~spectrax.IllegalMutationError`) plus ``mesh`` and ``schedule``
for routing to the MPMD ``sxjit`` runtime. For SPMD meshes (or when
``mesh`` is not given) the call dispatches to :func:`jax.jit`; if the
mesh is MPMD-shaped it dispatches to
:func:`spectrax.runtime.mpmd.sxjit`.

When ``mutable`` is set, the compiled function's argument layout is
``(states, stripped_args, stripped_kwargs)`` rather than the user's
original signature, so ``static_argnums`` / ``donate_argnums`` /
``in_shardings`` / ``out_shardings`` index into that 3-tuple. When in
doubt use the ``*_argnames`` variants.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import jax

from ..core.context import _STACK as _CTX_STACK
from ..core.context import partition as _ctx_partition
from ..core.graph import export
from ..core.module import Module, _graph_epoch
from ..core.selector import SelectorSugar
from .split_merge import (
    apply_mutations,
    locate_and_strip,
    locate_and_strip_fast,
    make_direct_readonly,
    make_pure,
    make_pure_ctx,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["jit"]

F = TypeVar("F", bound=Callable[..., Any])

_UNSET: Any = object()
"""Sentinel indicating that a keyword was not supplied, so JAX's own
``UnspecifiedValue`` default is used.
"""


def _live_module_refs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    """Return ``(layout_key, gdef_key, id_key)`` for top-level Module args."""
    layout: list[Any] = []
    gdefs: list[Any] = []
    ids: list[int] = []
    epoch = _graph_epoch()
    for i, value in enumerate(args):
        if isinstance(value, Module):
            cache = value._spx_export_cache
            if cache is None or cache[0] != epoch:
                export(value)
                cache = value._spx_export_cache
            assert cache is not None
            layout.append(("arg", i))
            gdefs.append(cache[1])
            ids.append(id(value))
    for key, value in kwargs.items():
        if isinstance(value, Module):
            cache = value._spx_export_cache
            if cache is None or cache[0] != epoch:
                export(value)
                cache = value._spx_export_cache
            assert cache is not None
            layout.append(("kwarg", key))
            gdefs.append(cache[1])
            ids.append(id(value))
    return tuple(layout), tuple(gdefs), tuple(ids)


def jit(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    mesh: Any | None = None,
    schedule: Any | None = None,
    in_shardings: Any = _UNSET,
    out_shardings: Any = _UNSET,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: Any = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, Any] | None = None,
) -> F:
    """Module-aware ``jax.jit``.

    Args:
        fn: The function to compile. When called without ``fn`` the
            decorator returns a factory: ``@spx.jit(mutable=...)``.
        mutable: Selector (or collection-name sugar) controlling which
            collections may be written back after the call.
        mesh: Optional SpectraX mesh. If this is an MPMD mesh, dispatches
            directly to :func:`spectrax.runtime.mpmd.sxjit`.
        schedule: Optional MPMD schedule, forwarded only when ``mesh`` is
            MPMD.
        in_shardings, out_shardings, static_argnums, static_argnames,
            donate_argnums, donate_argnames, keep_unused, device,
            backend, inline, compiler_options: Forwarded verbatim to
            :func:`jax.jit`.

    Returns:
        A wrapped function. The first call per distinct
        :class:`~spectrax.GraphDef` tuple triggers a JAX trace; later
        calls with matching graph-defs re-use the cached compile.
    """
    if fn is None:
        return lambda f: jit(
            f,
            mutable=mutable,
            mesh=mesh,
            schedule=schedule,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            compiler_options=compiler_options,
        )

    if mesh is not None and _is_mpmd_mesh(mesh):
        _raise_if_unsupported_mpmd_jit_options(
            mutable=mutable,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            compiler_options=compiler_options,
        )
        from ..runtime.mpmd import sxjit

        return sxjit(
            fn,
            mesh=mesh,
            schedule=schedule,
            static_argnums=_normalize_argnums_for_sxjit(static_argnums),
            static_argnames=_normalize_argnames_for_sxjit(static_argnames),
            donate_argnums=_normalize_argnums_for_sxjit(donate_argnums),
            in_shardings=None if in_shardings is _UNSET else in_shardings,
            out_shardings=None if out_shardings is _UNSET else out_shardings,
        )
    if schedule is not None:
        raise ValueError("spx.jit(..., schedule=...) requires an MPMD mesh.")

    mutable_sel = resolve_mutable(mutable)

    jit_kwargs: dict[str, Any] = {
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "donate_argnums": donate_argnums,
        "donate_argnames": donate_argnames,
        "keep_unused": keep_unused,
        "device": device,
        "backend": backend,
        "inline": inline,
        "compiler_options": compiler_options,
    }
    if in_shardings is not _UNSET:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not _UNSET:
        jit_kwargs["out_shardings"] = out_shardings

    _compile_cache: dict[tuple[Any, ...], Any] = {}
    _id_cache: dict[tuple[int, ...], tuple[int, tuple[Any, ...], Any]] = {}
    _id_cache_one: dict[int, tuple[int, tuple[Any, ...], Any]] = {}
    _ctx_compile_cache: dict[tuple[Any, ...], Any] = {}

    _locate = locate_and_strip
    _locate_fast = locate_and_strip_fast
    _epoch_fn = _graph_epoch
    _apply = apply_mutations
    _make_pure = make_pure
    _make_pure_readonly = make_pure_readonly
    _make_pure_single = make_pure_single_positional
    _make_pure_readonly_single = make_pure_readonly_single_positional
    _make_pure_ctx = make_pure_ctx
    _jax_jit = jax.jit
    _ctx_stack_get = _CTX_STACK.get
    _empty_kwargs: dict[str, Any] = {}
    _direct_guarded = make_direct_readonly(fn)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Dispatch through the graph-def-keyed compile cache.

        Two-level cache:

        1. Identity cache (``_id_cache``) keyed by the Python ``id``
           tuple of the input modules plus the current global graph
           epoch. Hot path: same model instance + no structural change
           returns the cached jitted callable in O(1) without ever
           touching the graph-def hash.
        2. Structural cache (``_compile_cache``) keyed by the full
           graph-def tuple. Handles model swaps, reloads, and distinct
           instances with identical structure.

        A kwargs-empty fast path uses :func:`locate_and_strip_fast`
        which does a single pass over ``args`` and skips the kwargs
        iteration entirely.

        Scope-aware slow path activates when the caller has active
        :func:`~spectrax.scope` bindings: static context values are
        folded into the compile cache key (so different static
        snapshots specialize cleanly); array-typed context values are
        lifted into the jit input tuple and reinstated as a scope
        frame inside the traced body (see :func:`make_pure_ctx`). The
        no-scope hot path above is completely unaffected — a single
        :class:`~contextvars.ContextVar` read (~50 ns) decides which
        path to take.
        """
        ctx_stack = _ctx_stack_get()
        if ctx_stack:
            return _wrapped_with_ctx(ctx_stack, args, kwargs)

        if mutable_sel is None:
            layout_key, gdef_key, id_key = _live_module_refs(args, kwargs)
            if len(id_key) == 1:
                id_hit = _id_cache_one.get(id_key[0])
                if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == layout_key:
                    jitted = id_hit[2]
                else:
                    epoch = _epoch_fn()
                    key = ("direct", layout_key, gdef_key)
                    jitted = _compile_cache.get(key)
                    if jitted is None:
                        jitted = _jax_jit(_direct_guarded, **jit_kwargs)
                        _compile_cache[key] = jitted
                    _id_cache_one[id_key[0]] = (epoch, layout_key, jitted)
                return jitted(*args, **kwargs)

            id_hit = _id_cache.get(id_key)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = ("direct", layout_key, gdef_key)
                jitted = _compile_cache.get(key)
                if jitted is None:
                    jitted = _jax_jit(_direct_guarded, **jit_kwargs)
                    _compile_cache[key] = jitted
                _id_cache[id_key] = (epoch, layout_key, jitted)
            return jitted(*args, **kwargs)

        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        n = len(refs)
        layout_key = tuple((r.kind, r.locator) for r in refs)
        if n == 1:
            r0 = refs[0]
            mid = id(r0.module)
            states_in: tuple = (r0.state,)
            single_positional = (not kwargs) and r0.kind == "arg"
            call_layout_key = (layout_key, single_positional)
            id_hit = _id_cache_one.get(mid)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == call_layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = (layout_key, r0.gdef, single_positional)
                jitted = _compile_cache.get(key)
                if jitted is None:
                    if single_positional:
                        pure = _make_pure_readonly_single(fn, r0) if mutable_sel is None else _make_pure_single(fn, r0)
                    else:
                        pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                    jitted = _jax_jit(pure, **jit_kwargs)
                    _compile_cache[key] = jitted
                _id_cache_one[mid] = (epoch, call_layout_key, jitted)
            if single_positional:
                other_args = stripped_args[: int(r0.locator)] + stripped_args[int(r0.locator) + 1 :]
                if mutable_sel is None:
                    return jitted(r0.state, *other_args)
                out, new_state = jitted(r0.state, *other_args)
                _apply([r0], [new_state], mutable_sel)
                return out
        else:
            id_key_list = []
            states_list = []
            for r in refs:
                id_key_list.append(id(r.module))
                states_list.append(r.state)
            id_key = tuple(id_key_list)
            states_in = tuple(states_list)
            id_hit = _id_cache.get(id_key)
            if id_hit is not None and id_hit[0] == _epoch_fn() and id_hit[1] == layout_key:
                jitted = id_hit[2]
            else:
                epoch = _epoch_fn()
                key = (layout_key, tuple([r.gdef for r in refs]))
                jitted = _compile_cache.get(key)
                if jitted is None:
                    pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                    jitted = _jax_jit(pure, **jit_kwargs)
                    _compile_cache[key] = jitted
                _id_cache[id_key] = (epoch, layout_key, jitted)
        if mutable_sel is None:
            return jitted(states_in, stripped_args, stripped_kwargs)
        out, new_states = jitted(states_in, stripped_args, stripped_kwargs)
        _apply(refs, new_states, mutable_sel)
        return out

    def _wrapped_with_ctx(
        ctx_stack: tuple[dict[str, Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Scope-active dispatch path.

        Flattens the scope stack, splits it into
        :data:`~spectrax.core.context.partition`'s traced/static halves,
        augments the compile-cache key with the static snapshot, and
        routes the call through a separate ``pure`` built by
        :func:`make_pure_ctx` so deep ``spx.scope.get(...)`` calls
        inside the traced body resolve to tracers rather than the
        constants captured at the trace-time snapshot.
        """
        snap: dict[str, Any] = {}
        for frame in ctx_stack:
            snap.update(frame)
        traced_ctx, static_ctx = _ctx_partition(snap)
        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        key = (tuple((r.kind, r.locator) for r in refs), tuple([r.gdef for r in refs]), static_ctx)
        jitted = _ctx_compile_cache.get(key)
        if jitted is None:
            pure = _make_pure_ctx(fn, refs)
            jitted = _jax_jit(pure, **jit_kwargs)
            _ctx_compile_cache[key] = jitted
        states_in = tuple([r.state for r in refs])
        out, new_states = jitted(states_in, traced_ctx, stripped_args, stripped_kwargs)
        _apply(refs, new_states, mutable_sel)
        return out

    def lower(*args: Any, **kwargs: Any) -> Any:
        """Lower the same module-aware call shape as ``wrapped``.

        This mirrors ``jax.jit(...).lower(...)`` for AOT users while still
        preserving SpecTrax's graph-def keyed cache.  Mutation write-back is a
        runtime side effect, so lowering only prepares the transformed call.
        """
        ctx_stack = _ctx_stack_get()
        if ctx_stack:
            snap: dict[str, Any] = {}
            for frame in ctx_stack:
                snap.update(frame)
            traced_ctx, static_ctx = _ctx_partition(snap)
            if kwargs:
                refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
            else:
                refs, stripped_args = _locate_fast(args)
                stripped_kwargs = _empty_kwargs
            key = (tuple((r.kind, r.locator) for r in refs), tuple([r.gdef for r in refs]), static_ctx)
            jitted = _ctx_compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_ctx(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _ctx_compile_cache[key] = jitted
            states_in = tuple([r.state for r in refs])
            return jitted.lower(states_in, traced_ctx, stripped_args, stripped_kwargs)

        if mutable_sel is None:
            layout_key, gdef_key, id_key = _live_module_refs(args, kwargs)
            key = ("direct", layout_key, gdef_key)
            jitted = _compile_cache.get(key)
            if jitted is None:
                jitted = _jax_jit(_direct_guarded, **jit_kwargs)
                _compile_cache[key] = jitted
            epoch = _epoch_fn()
            if len(id_key) == 1:
                _id_cache_one[id_key[0]] = (epoch, layout_key, jitted)
            elif id_key:
                _id_cache[id_key] = (epoch, layout_key, jitted)
            return jitted.lower(*args, **kwargs)

        if kwargs:
            refs, stripped_args, stripped_kwargs = _locate(args, kwargs)
        else:
            refs, stripped_args = _locate_fast(args)
            stripped_kwargs = _empty_kwargs
        layout_key = tuple((r.kind, r.locator) for r in refs)
        if len(refs) == 1:
            r0 = refs[0]
            single_positional = (not kwargs) and r0.kind == "arg"
            key = (layout_key, r0.gdef, single_positional)
            jitted = _compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_readonly_single(fn, r0) if mutable_sel is None else _make_pure_single(fn, r0)
                if not single_positional:
                    pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _compile_cache[key] = jitted
            _id_cache_one[id(r0.module)] = (_epoch_fn(), (layout_key, single_positional), jitted)
            if single_positional:
                other_args = stripped_args[: int(r0.locator)] + stripped_args[int(r0.locator) + 1 :]
                return jitted.lower(r0.state, *other_args)
        else:
            key = (layout_key, tuple([r.gdef for r in refs]))
            jitted = _compile_cache.get(key)
            if jitted is None:
                pure = _make_pure_readonly(fn, refs) if mutable_sel is None else _make_pure(fn, refs)
                jitted = _jax_jit(pure, **jit_kwargs)
                _compile_cache[key] = jitted
            _id_cache[tuple(id(r.module) for r in refs)] = (_epoch_fn(), layout_key, jitted)
        states_in = tuple([r.state for r in refs])
        return jitted.lower(states_in, stripped_args, stripped_kwargs)

    wrapped._spx_compile_cache = _compile_cache
    wrapped._spx_id_cache = _id_cache
    wrapped._spx_ctx_compile_cache = _ctx_compile_cache
    wrapped.lower = lower
    return wrapped


def _is_mpmd_mesh(mesh: Any) -> bool:
    """Return whether ``mesh`` should use the MPMD ``sxjit`` path."""
    return bool(getattr(mesh, "is_mpmd", False)) or (
        hasattr(mesh, "mpmd_dim") and hasattr(mesh, "submesh") and hasattr(mesh, "sub_sharding")
    )


def _normalize_argnums_for_sxjit(argnums: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
    """Coerce ``argnums`` into the form ``sxjit`` expects: ``None``/``int``/tuple."""
    if argnums is None or isinstance(argnums, int):
        return argnums
    return tuple(argnums)


def _normalize_argnames_for_sxjit(argnames: str | Iterable[str] | None) -> str | tuple[str, ...] | None:
    """Coerce ``argnames`` into the form ``sxjit`` expects: ``None``/``str``/tuple."""
    if argnames is None or isinstance(argnames, str):
        return argnames
    return tuple(argnames)


def _raise_if_unsupported_mpmd_jit_options(
    *,
    mutable: SelectorSugar,
    donate_argnames: str | Iterable[str] | None,
    keep_unused: bool,
    device: Any,
    backend: str | None,
    inline: bool,
    compiler_options: dict[str, Any] | None,
) -> None:
    """Raise if the user passed a ``jax.jit`` option that ``sxjit`` cannot honor.

    ``spx.jit`` routes to :func:`sxjit` whenever the mesh is MPMD-shaped,
    but several ``jax.jit`` knobs (``mutable``, ``donate_argnames``,
    ``keep_unused``, ``device``, ``backend``, ``inline``,
    ``compiler_options``) have no MPMD analog. Bundling all rejected
    options into a single error makes the message actionable.
    """
    unsupported: list[str] = []
    if resolve_mutable(mutable) is not None:
        unsupported.append("mutable")
    if donate_argnames is not None:
        unsupported.append("donate_argnames")
    if keep_unused:
        unsupported.append("keep_unused")
    if device is not None:
        unsupported.append("device")
    if backend is not None:
        unsupported.append("backend")
    if inline:
        unsupported.append("inline")
    if compiler_options is not None:
        unsupported.append("compiler_options")
    if unsupported:
        opts = ", ".join(unsupported)
        raise ValueError(f"spx.jit(..., mesh=<MPMD>) routes to sxjit, which does not support: {opts}.")
