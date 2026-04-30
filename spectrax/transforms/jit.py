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
    """Build cache-key tuples for the top-level :class:`~spectrax.Module` arguments.

    Used by the readonly fast path in :func:`jit`: instead of going
    through the full split/merge shim, the wrapped call dispatches a
    direct :func:`jax.jit` over the live module pytree, keyed by the
    triple ``(layout, gdef, id)``. This avoids an export per call when
    the same module instance is repeatedly passed in.

    For each module the export cache is consulted; if it is stale (the
    global graph epoch advanced since the last export) the cache is
    refreshed via :func:`~spectrax.export`.

    Args:
        args: Positional arguments passed to the wrapped function.
        kwargs: Keyword arguments passed to the wrapped function.

    Returns:
        A triple ``(layout_key, gdef_key, id_key)``:

        * ``layout_key`` records the ``("arg"|"kwarg", index|name)``
          position of every module.
        * ``gdef_key`` collects the corresponding :class:`~spectrax.GraphDef`
          values for the structural compile cache.
        * ``id_key`` collects the Python ``id`` of each module instance
          for the identity-based hot-path cache.
    """
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
    batch_argnums: int | Sequence[int] | None = None,
    keep_unused: bool = False,
    device: Any = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, Any] | None = None,
) -> F:
    """Module-aware ``jax.jit``.

    Compiles ``fn`` via :func:`jax.jit` while transparently handling
    :class:`~spectrax.Module` arguments. On the first call with a given
    module structure, each :class:`~spectrax.Module` argument is
    exported to ``(GraphDef, State)``, the state is threaded through the
    compiled function, and after the call any mutations to
    declared-mutable collections are written back to the live module.

    **Compiled signature when ``mutable`` is set**

    When ``mutable`` resolves to a non-empty selector, the compiled
    function's argument layout becomes ``(states, stripped_args,
    stripped_kwargs)`` rather than the user's original signature. This
    means ``static_argnums``, ``donate_argnums``, ``in_shardings``, and
    ``out_shardings`` index into that 3-tuple. When in doubt, prefer
    the ``*_argnames`` variants which resolve by name before stripping.

    **In-Out kwargs**

    ``in_shardings`` and ``out_shardings`` (plus ``static_argnums``,
    ``static_argnames``, ``donate_argnums``, ``donate_argnames``,
    ``batch_argnums``, ``keep_unused``, ``device``, ``backend``,
    ``inline``, ``compiler_options``) are forwarded verbatim to
    :func:`jax.jit`.

    **Compile caching**

    The returned wrapper maintains two internal caches:

    * ``_spx_id_cache`` — identity-based hot-path cache keyed by Python
      ``id()`` of input modules plus the current graph epoch. Same
      instance + no structural change reuses the jitted callable in O(1).
    * ``_spx_compile_cache`` — structural cache keyed by the full
      ``GraphDef`` tuple. Handles model swaps and distinct instances
      with identical structure.

    **MPMD dispatch**

    If ``mesh`` is an MPMD mesh, the call routes to
    :func:`spectrax.runtime.mpmd.sxjit` instead of :func:`jax.jit`. In
    that mode ``mutable``, ``donate_argnames``, ``keep_unused``,
    ``device``, ``backend``, ``inline``, and ``compiler_options`` are
    unsupported and raise :class:`ValueError`.

    **Lowered representation**

    The returned callable has a ``.lower(*args, **kwargs)`` method that
    mirrors :meth:`jax.jit.lower` and returns a
    :class:`jax.stages.Lowered` object without dispatching the compiled
    function.

    Args:
        fn: The function to compile. When called without ``fn`` the
            decorator returns a factory: ``@spx.jit(mutable=...)``.
        mutable: Selector (or collection-name sugar) controlling which
            collections may be written back after the call.
        mesh: Optional SpectraX mesh. If this is an MPMD mesh, dispatches
            directly to :func:`spectrax.runtime.mpmd.sxjit`.
        schedule: Optional MPMD schedule, forwarded only when ``mesh`` is
            MPMD.
        in_shardings: Optional sharding constraint for inputs; forwarded
            to :func:`jax.jit`.
        out_shardings: Optional sharding constraint for outputs;
            forwarded to :func:`jax.jit`.
        static_argnums: Indices of positional arguments that should be
            treated as compile-time constants; forwarded to
            :func:`jax.jit`.
        static_argnames: Names of keyword arguments that should be
            treated as compile-time constants; forwarded to
            :func:`jax.jit`.
        donate_argnums: Indices of positional arguments whose buffers
            may be donated to the output; forwarded to :func:`jax.jit`.
        donate_argnames: Names of keyword arguments whose buffers may
            be donated; forwarded to :func:`jax.jit`.
        batch_argnums: MPMD-only; indices of positional arguments that
            represent batch dimensions. Requires an MPMD mesh.
        keep_unused: Forwarded to :func:`jax.jit`.
        device: Forwarded to :func:`jax.jit`.
        backend: Forwarded to :func:`jax.jit`.
        inline: Forwarded to :func:`jax.jit`.
        compiler_options: Forwarded to :func:`jax.jit`.

    Returns:
        A wrapped function. The first call per distinct
        :class:`~spectrax.GraphDef` tuple triggers a JAX trace; later
        calls with matching graph-defs re-use the cached compile.

    Raises:
        ValueError: If ``schedule`` is provided without an MPMD mesh, or
            if ``batch_argnums`` is provided without an MPMD mesh with
            a schedule, or if an unsupported option is passed when
            routing to ``sxjit``.
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
            batch_argnums=batch_argnums,
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
            batch_argnums=_normalize_argnums_for_sxjit(batch_argnums),
            in_shardings=None if in_shardings is _UNSET else in_shardings,
            out_shardings=None if out_shardings is _UNSET else out_shardings,
        )
    if schedule is not None:
        raise ValueError("spx.jit(..., schedule=...) requires an MPMD mesh.")
    if batch_argnums is not None:
        raise ValueError("spx.jit(..., batch_argnums=...) requires an MPMD mesh with schedule=.")

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
        """Lower the call to a :class:`jax.stages.Lowered` without dispatching it.

        Mirrors :meth:`jax.jit.lower` for ahead-of-time users while still
        preserving the graph-def-keyed compile cache. Picks the same
        scope-aware / direct / pure dispatch path as ``wrapped`` would,
        builds the right pure body for that path, calls ``.lower(...)``
        on the resulting jitted function, and returns the
        :class:`~jax.stages.Lowered`. Side effects of the runtime
        wrapper (specifically
        :func:`~spectrax.transforms.split_merge.apply_mutations`) are
        skipped — lowering is purely about preparing the compiled
        artifact.
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
    """Return whether ``mesh`` is an MPMD mesh requiring the ``sxjit`` runtime.

    Treats either an explicit ``is_mpmd`` boolean attribute or the
    structural duck-typing trio (``mpmd_dim``, ``submesh``,
    ``sub_sharding``) as evidence of an MPMD mesh.
    """
    return bool(getattr(mesh, "is_mpmd", False)) or (
        hasattr(mesh, "mpmd_dim") and hasattr(mesh, "submesh") and hasattr(mesh, "sub_sharding")
    )


def _normalize_argnums_for_sxjit(argnums: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
    """Coerce ``argnums`` into the ``None``/``int``/``tuple`` shape ``sxjit`` accepts."""
    if argnums is None or isinstance(argnums, int):
        return argnums
    return tuple(argnums)


def _normalize_argnames_for_sxjit(argnames: str | Iterable[str] | None) -> str | tuple[str, ...] | None:
    """Coerce ``argnames`` into the ``None``/``str``/``tuple`` shape ``sxjit`` accepts."""
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
