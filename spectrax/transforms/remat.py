# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :func:`jax.checkpoint` (gradient checkpointing) wrapper."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from typing import Any, TypeVar

import jax

from ..core.module import Module
from ..core.selector import SelectorSugar
from .split_merge import (
    _run_pure_body,
    _run_readonly_body,
    apply_mutations,
    locate_and_strip,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    resolve_mutable,
)

__all__ = ["remat"]

F = TypeVar("F", bound=Callable[..., Any])


_REMAT_CLASS_CACHE: dict[tuple, type] = {}
"""Cache of remat'd module subclasses keyed by ``(parent_class, config)``.

Ensures ``spx.remat(Llama3Block)`` returns the *same* class on every
call: avoids re-registering the pytree and lets
:func:`resolve_class` find the remat'd subclass during bind.
"""


def _hashable_cache_key(value: Any) -> Any:
    """Convert selector-like containers into deterministic cache-key values."""
    if isinstance(value, dict):
        return tuple(sorted((_hashable_cache_key(k), _hashable_cache_key(v)) for k, v in value.items()))
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(_hashable_cache_key(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def remat(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    prevent_cse: bool = True,
    policy: Callable[..., bool] | None = None,
    static_argnums: int | tuple[int, ...] = (),
) -> F:
    """Module-aware gradient checkpointing.

    Two usage modes:

    * **Function**: ``spx.remat(fn)`` returns a wrapped function that
      runs ``fn`` under :func:`jax.checkpoint`. Per-call wrapping.

    * **Module class**: ``spx.remat(MyBlock)`` where ``MyBlock`` is a
      subclass of :class:`spectrax.Module` returns a new subclass whose
      ``forward`` is checkpointed once. Build instances normally and
      use them in a loop — every block call recomputes during backward
      without per-iteration ``spx.remat(...)`` calls in the model body::

          RematBlock = spx.remat(MyBlock)
          blocks = [RematBlock(cfg, rngs=rngs) for _ in range(N)]
          # inside model.forward:
          for blk in blocks:
              x = blk(x)   # already remat-wrapped

      The returned class is registered in the parent class's module
      namespace so ``spectrax.export`` / ``bind`` can round-trip it.

    Args:
        fn: Function or :class:`Module` subclass to checkpoint; when
            omitted, returns a decorator factory.
        mutable: Selector controlling writable collections.
        prevent_cse, policy, static_argnums: Forwarded verbatim to
            :func:`jax.checkpoint`.
    """
    if not isinstance(prevent_cse, bool):
        raise TypeError(f"prevent_cse must be a bool, got {type(prevent_cse).__name__}.")
    if fn is None:
        return lambda f: remat(
            f,
            mutable=mutable,
            prevent_cse=prevent_cse,
            policy=policy,
            static_argnums=static_argnums,
        )

    if isinstance(fn, type):
        if issubclass(fn, Module):
            return _remat_module_class(
                fn,
                mutable=mutable,
                prevent_cse=prevent_cse,
                policy=policy,
                static_argnums=static_argnums,
            )

    mutable_sel = resolve_mutable(mutable)

    def _should_be_static_kwarg(x: Any) -> bool:
        """Return True if x should not flow through jax.checkpoint as a traced arg.

        Strings and booleans cannot be traced by jax.checkpoint (strings raise
        TypeError, booleans raise TracerBoolConversionError when used in Python
        control flow).  We close over them instead so they become Python constants
        inside the checkpointed function.
        """
        if isinstance(x, (str, bool)):
            return True
        if isinstance(x, (tuple, list)):
            return any(_should_be_static_kwarg(elem) for elem in x)
        if isinstance(x, dict):
            return any(_should_be_static_kwarg(v) for v in x.values())
        return False

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Dispatch through :func:`jax.checkpoint` wrapped around the pure fn."""
        refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)

        static_kwargs = {k: v for k, v in stripped_kwargs.items() if _should_be_static_kwarg(v)}
        dynamic_kwargs = {k: v for k, v in stripped_kwargs.items() if k not in static_kwargs}

        readonly = mutable_sel is None

        if readonly and not kwargs and len(refs) == 1 and refs[0].kind == "arg":
            ref = refs[0]
            pure = make_pure_readonly_single_positional(fn, ref)
            checkpointed = jax.checkpoint(
                pure,
                prevent_cse=prevent_cse,
                policy=policy,
                static_argnums=static_argnums,
            )
            locator = int(ref.locator)
            other_args = stripped_args[:locator] + stripped_args[locator + 1 :]
            return checkpointed(ref.state, *other_args)

        if static_kwargs:
            _fn = fn
            _refs = refs
            _static = static_kwargs

            def pure_with_static(
                states: tuple[Any, ...],
                stripped_args_: tuple[Any, ...],
                stripped_kwargs_: dict[str, Any],
            ) -> Any:
                """Pure ``(states, stripped_args, stripped_kwargs) -> output`` adapter.

                Re-injects the captured ``static_kwargs`` (those filtered
                out before ``jax.checkpoint`` saw them) at call time, then
                routes through either the readonly or read/write body
                helper depending on the wrapper's mode.
                """
                merged_kwargs = {**stripped_kwargs_, **_static}
                if readonly:
                    return _run_readonly_body(
                        _fn,
                        tuple(r.gdef for r in _refs),
                        tuple(r.module for r in _refs),
                        _refs,
                        states,
                        stripped_args_,
                        merged_kwargs,
                        None,
                    )
                return _run_pure_body(
                    _fn,
                    tuple(r.gdef for r in _refs),
                    tuple(r.module for r in _refs),
                    _refs,
                    states,
                    stripped_args_,
                    merged_kwargs,
                    None,
                )

            pure = pure_with_static
        else:
            pure = make_pure_readonly(fn, refs) if readonly else make_pure(fn, refs)

        checkpointed = jax.checkpoint(
            pure,
            prevent_cse=prevent_cse,
            policy=policy,
            static_argnums=static_argnums,
        )
        states_in = tuple(r.state for r in refs)
        if readonly:
            return checkpointed(states_in, stripped_args, dynamic_kwargs)
        out, new_states = checkpointed(states_in, stripped_args, dynamic_kwargs)
        apply_mutations(refs, list(new_states), mutable_sel)
        return out

    return wrapped


def _remat_module_class(
    cls: type,
    *,
    mutable: SelectorSugar,
    prevent_cse: bool,
    policy: Callable[..., bool] | None,
    static_argnums: int | tuple[int, ...],
) -> type:
    """Build (and cache) a ``Remat<Cls>`` subclass whose ``forward`` is
    permanently wrapped in :func:`jax.checkpoint`. The new class is
    injected into ``cls``'s defining module under the name
    ``Remat<Qualname>`` so ``importlib`` / ``resolve_class`` finds it.

    On ``ImportError`` during parent-module registration, resolved
    class identity falls back to in-process use.
    """
    cache_key = (cls, _hashable_cache_key(mutable), prevent_cse, policy, static_argnums)
    cached = _REMAT_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parent_forward = cls.forward
    new_qualname = f"Remat{cls.__qualname__.replace('.', '_')}"

    class RematSubclass(cls):
        """Subclass of ``cls`` whose ``forward`` is wrapped in :func:`jax.checkpoint`."""

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Run the parent ``forward`` under :func:`remat`."""
            cached = getattr(self, "_spx_remat_forward", None)
            if cached is None:
                cached = remat(
                    parent_forward.__get__(self),
                    mutable=mutable,
                    prevent_cse=prevent_cse,
                    policy=policy,
                    static_argnums=static_argnums,
                )
                object.__setattr__(self, "_spx_remat_forward", cached)
            return cached(*args, **kwargs)

    RematSubclass.__name__ = new_qualname
    RematSubclass.__qualname__ = new_qualname
    RematSubclass.__module__ = cls.__module__

    try:
        parent_module = importlib.import_module(cls.__module__)
        if not hasattr(parent_module, new_qualname):
            setattr(parent_module, new_qualname, RematSubclass)
    except ImportError:
        pass

    _REMAT_CLASS_CACHE[cache_key] = RematSubclass
    return RematSubclass
