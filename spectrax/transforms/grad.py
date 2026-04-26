# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware autodiff: :func:`grad`, :func:`value_and_grad`,
:func:`jvp`, and :func:`vjp`.

The differentiation target is selected by ``wrt`` — a
:class:`~spectrax.Selector` or one of its sugar forms. By default
spectrax differentiates the ``"parameters"`` collection of the first
:class:`~spectrax.Module` argument. Every non-spectrax keyword is
forwarded to :func:`jax.value_and_grad` verbatim.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import jax

from ..core.graph import bind, export
from ..core.module import Module, _set_inside_transform
from ..core.selector import SelectorSugar, as_selector
from ..core.state import State
from .split_merge import (
    apply_mutations,
    locate_and_strip_fast,
    make_direct_readonly,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["grad", "jvp", "value_and_grad", "vjp"]

F = TypeVar("F", bound=Callable[..., Any])

AxisName = Any
"""Type alias for a JAX axis-name sentinel (no canonical type exists)."""

_MISSING = object()
"""Sentinel for deferred ``jvp`` arguments."""


def value_and_grad(
    fn: F | None = None,
    *,
    wrt: SelectorSugar = "parameters",
    argnum: int | None = None,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> F:
    """Module-aware :func:`jax.value_and_grad`.

    Args:
        fn: Function whose first :class:`Module` argument is the target
            of differentiation. When called without ``fn`` returns a
            decorator factory.
        wrt: Selector for the state to differentiate (default:
            ``"parameters"``).
        argnum: Which positional argument is the differentiated
            :class:`Module`. Defaults to the first Module argument.
        has_aux: Forwarded to :func:`jax.value_and_grad`. When ``True``,
            ``fn`` is expected to return ``(value, aux)``.
        holomorphic, allow_int, reduce_axes: Forwarded to
            :func:`jax.value_and_grad`.

    Returns:
        A wrapped callable returning ``(value, grads)`` — or
        ``((value, aux), grads)`` if ``has_aux``. ``grads`` is a
        :class:`~spectrax.State` with the same shape as the selected
        subset.
    """
    if fn is None:
        return lambda f: value_and_grad(
            f,
            wrt=wrt,
            argnum=argnum,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )

    wrt_sel = as_selector(wrt)
    direct_guarded = make_direct_readonly(fn)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Partition the module's state, differentiate, and re-merge."""
        idx = argnum if argnum is not None else _find_first_module(args)
        model = args[idx]
        if not isinstance(model, Module):
            raise TypeError(f"Argument {idx} must be a Module, got {type(model).__name__}")
        gdef, state = export(model)
        target, rest = wrt_sel.partition_state(model, state)
        if _state_is_empty(rest):
            vg_kwargs: dict[str, Any] = {
                "argnums": idx,
                "has_aux": has_aux,
                "holomorphic": holomorphic,
                "allow_int": allow_int,
            }
            if reduce_axes:
                vg_kwargs["reduce_axes"] = reduce_axes
            vg = jax.value_and_grad(direct_guarded, **vg_kwargs)
            out, grads_module = vg(*args, **kwargs)
            return out, _module_like_to_state(grads_module)
        other_args = tuple(args[:idx]) + tuple(args[idx + 1 :])

        def pure(
            target_state: State,
            rest_state: State,
            other: tuple[Any, ...],
            kw: dict[str, Any],
        ) -> tuple[Any, Any]:
            """Pure closure fed to :func:`jax.value_and_grad`."""
            merged = target_state.overlay(rest_state)
            m = bind(gdef, merged)
            spliced = list(other)
            spliced.insert(idx, m)
            _set_inside_transform(True)
            try:
                out = fn(*spliced, **kw)
            finally:
                _set_inside_transform(False)
            if has_aux:
                val, aux = out
                return val, aux
            return out, None

        vg_kwargs: dict[str, Any] = {
            "has_aux": True,
            "holomorphic": holomorphic,
            "allow_int": allow_int,
        }
        if reduce_axes:
            vg_kwargs["reduce_axes"] = reduce_axes
        vg = jax.value_and_grad(pure, **vg_kwargs)
        (value, aux), grads_target = vg(target, rest, other_args, kwargs)
        if has_aux:
            return (value, aux), grads_target
        return value, grads_target

    return wrapped


def grad(
    fn: F | None = None,
    *,
    wrt: SelectorSugar = "parameters",
    argnum: int | None = None,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> F:
    """Module-aware :func:`jax.grad`.

    Thin wrapper over :func:`value_and_grad` that drops the value
    component. See :func:`value_and_grad` for argument semantics.
    """
    if fn is None:
        return lambda f: grad(
            f,
            wrt=wrt,
            argnum=argnum,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )

    vg = value_and_grad(
        fn,
        wrt=wrt,
        argnum=argnum,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Return only the gradient half of the :func:`value_and_grad` output."""
        out = vg(*args, **kwargs)
        if has_aux:
            (_, aux), grads = out
            return grads, aux
        _, grads = out
        return grads

    return wrapped


def vjp(
    fn: F | None = None,
    *primals: Any,
    has_aux: bool = False,
    reduce_axes: Sequence[AxisName] = (),
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.vjp`.

    When called directly, behaves like ``jax.vjp`` but returns
    :class:`State` cotangents for any :class:`Module` primal arguments.
    When called without primals it returns a wrapped function.

    ``mutable=`` controls which module collections may be written back
    during the primal forward pass. The returned pullback is pure and
    only computes cotangents.
    """
    if fn is None:
        return lambda f: vjp(f, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    if primals:
        return _vjp_call(fn, primals, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Decorator-mode wrapper: defer the actual ``_vjp_call`` until args are supplied."""
        _ensure_no_kwargs("vjp", kwargs)
        return _vjp_call(fn, args, has_aux=has_aux, reduce_axes=reduce_axes, mutable=mutable)

    return wrapped


def jvp(
    fn: F | None = None,
    primals: Sequence[Any] | None | object = _MISSING,
    tangents: Sequence[Any] | None | object = _MISSING,
    *,
    has_aux: bool = False,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.jvp`.

    Direct-call form:

    ``spx.jvp(fn, primals, tangents, ...)``

    Decorator/wrapper form:

    ``spx.jvp(fn, ...)(primals, tangents)``

    Module primals accept tangents either as a matching :class:`Module`,
    a :class:`State`, or an arbitrary pytree matching the module's
    exported state.
    """
    if fn is None:
        return lambda f: jvp(f, has_aux=has_aux, mutable=mutable)

    if primals is not _MISSING and tangents is not _MISSING:
        return _jvp_call(fn, primals, tangents, has_aux=has_aux, mutable=mutable)

    @functools.wraps(fn)
    def wrapped(
        primals_: Sequence[Any],
        tangents_: Sequence[Any],
    ) -> Any:
        """Decorator-mode wrapper: defer ``_jvp_call`` until ``(primals, tangents)`` are supplied."""
        return _jvp_call(fn, primals_, tangents_, has_aux=has_aux, mutable=mutable)

    return wrapped


def _find_first_module(args: tuple[Any, ...]) -> int:
    """Return the positional index of the first :class:`Module` in ``args``.

    Raises:
        TypeError: If no positional argument is a :class:`Module`.
    """
    for i, a in enumerate(args):
        if isinstance(a, Module):
            return i
    raise TypeError("spectrax.grad requires at least one Module argument")


def _ensure_no_kwargs(name: str, kwargs: dict[str, Any]) -> None:
    """Reject kwargs for transforms whose public API mirrors raw JAX."""
    if kwargs:
        raise TypeError(
            f"spectrax.{name}() does not support keyword arguments in wrapped-call form. "
            "Wrap the function with lambda/partial if you need kwargs."
        )


def _split_module_tangents(refs: list[Any], tangents: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Split positional tangents into module-state tangents and stripped tangents."""
    stripped_tangents = list(tangents)
    state_tangents: list[Any] = []
    for ref in refs:
        tangent = tangents[ref.locator]
        if isinstance(tangent, Module):
            _, tangent = export(tangent)
        state_tangents.append(tangent)
        stripped_tangents[ref.locator] = None
    return tuple(state_tangents), tuple(stripped_tangents)


def _splice_module_cotangents(
    refs: list[Any], stripped_cotangents: tuple[Any, ...], state_cotangents: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Rebuild a cotangent tuple parallel to the original primal args."""
    out = list(stripped_cotangents)
    for ref, ct in zip(refs, state_cotangents, strict=False):
        out[ref.locator] = ct
    return tuple(out)


def _splice_one_cotangent(locator: int, other_cotangents: tuple[Any, ...], state_cotangent: Any) -> tuple[Any, ...]:
    """Rebuild a cotangent tuple for one positional Module primal."""
    out = list(other_cotangents)
    out.insert(locator, state_cotangent)
    return tuple(out)


def _zeros_like_tree(tree: Any) -> Any:
    """Build a zero cotangent pytree matching ``tree``."""
    return jax.tree.map(jax.numpy.zeros_like, tree)


def _state_is_empty(state: State) -> bool:
    """Return ``True`` when ``state`` contains no leaves."""
    return not state.collections()


def _module_like_to_state(value: Any) -> State:
    """Convert a Module-shaped cotangent into the public State form."""
    if isinstance(value, State):
        return value
    if isinstance(value, Module):
        _, state = export(value)
        return state
    raise TypeError(f"Expected Module or State value, got {type(value).__name__}")


def _convert_direct_tangents(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Convert State tangents into Module tangents for direct JAX autodiff."""
    converted = list(tangents)
    for i, primal in enumerate(primals):
        if not isinstance(primal, Module):
            continue
        primal_treedef = jax.tree.structure(primal)
        tangent = tangents[i]
        if isinstance(tangent, Module):
            converted[i] = jax.tree_util.tree_unflatten(primal_treedef, jax.tree.leaves(tangent))
        elif isinstance(tangent, State):
            gdef, _ = export(primal)
            tangent_module = bind(gdef, tangent)
            converted[i] = jax.tree_util.tree_unflatten(primal_treedef, jax.tree.leaves(tangent_module))
        else:
            return None
    return tuple(converted)


def _convert_direct_cotangents(primals: tuple[Any, ...], cotangents: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert direct-JAX Module cotangents back to SpecTrax public types."""
    out: list[Any] = []
    for primal, cotangent in zip(primals, cotangents, strict=False):
        out.append(_module_like_to_state(cotangent) if isinstance(primal, Module) else cotangent)
    return tuple(out)


def _vjp_call(
    fn: Callable[..., Any],
    primals: Sequence[Any],
    *,
    has_aux: bool,
    reduce_axes: Sequence[AxisName],
    mutable: SelectorSugar,
) -> Any:
    """Shared implementation for direct and wrapped :func:`vjp`."""
    args = tuple(primals)
    mutable_sel = resolve_mutable(mutable)
    direct_guarded = make_direct_readonly(fn)
    if mutable_sel is None:
        vjp_kwargs: dict[str, Any] = {"has_aux": has_aux} if has_aux else {}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if has_aux:
            out, pullback, aux = jax.vjp(direct_guarded, *args, **vjp_kwargs)

            def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.
                """
                cotangents = pullback(cotangent)
                return _convert_direct_cotangents(args, cotangents)

            return out, wrapped_pullback, aux

        out, pullback = jax.vjp(direct_guarded, *args, **vjp_kwargs)

        def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.
            """
            cotangents = pullback(cotangent)
            return _convert_direct_cotangents(args, cotangents)

        return out, wrapped_pullback

    refs, stripped_args = locate_and_strip_fast(args)
    if len(refs) == 1 and refs[0].kind == "arg":
        ref = refs[0]
        locator = int(ref.locator)
        other_args = args[:locator] + args[locator + 1 :]
        state_in = ref.state
        pure_one = (
            make_pure_readonly_single_positional(fn, ref)
            if mutable_sel is None
            else make_pure_single_positional(fn, ref)
        )

        if has_aux:
            vjp_kwargs: dict[str, Any] = {"has_aux": True}
            if reduce_axes:
                vjp_kwargs["reduce_axes"] = reduce_axes
            if mutable_sel is None:
                out, pullback, aux = jax.vjp(pure_one, state_in, *other_args, **vjp_kwargs)

                def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
                    """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                    Translates the cotangent into the right shape for the wrapped
                    function, calls the underlying ``vjp_fn``, and re-packages the
                    result so the user sees module-shaped grads (where applicable)
                    rather than raw state pytrees.
                    """
                    cotangents = pullback(cotangent)
                    return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

                return out, wrapped_pullback, aux

            def pure_with_updates(state: State, *other: Any) -> tuple[tuple[Any, State], Any]:
                """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

                Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
                pure ``(state, *args) -> (output, new_state)`` interface even though
                the user wrote a stateful module-method. The caller separately
                applies ``new_state`` back to the live module via :func:`apply_mutations`.
                """
                (out, aux), new_state = pure_one(state, *other)
                return (out, new_state), aux

            (out, new_state), pullback, aux = jax.vjp(pure_with_updates, state_in, *other_args, **vjp_kwargs)
            apply_mutations([ref], [new_state], mutable_sel)
            zero_state = _zeros_like_tree(new_state)

            def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.
                """
                cotangents = pullback((cotangent, zero_state))
                return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

            return out, wrapped_pullback, aux

        vjp_kwargs = {}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if mutable_sel is None:
            out, pullback = jax.vjp(pure_one, state_in, *other_args, **vjp_kwargs)

            def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.
                """
                cotangents = pullback(cotangent)
                return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

            return out, wrapped_pullback

        def pure_with_updates(state: State, *other: Any) -> tuple[Any, State]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.
            """
            return pure_one(state, *other)

        (out, new_state), pullback = jax.vjp(pure_with_updates, state_in, *other_args, **vjp_kwargs)
        apply_mutations([ref], [new_state], mutable_sel)
        zero_state = _zeros_like_tree(new_state)

        def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.
            """
            cotangents = pullback((cotangent, zero_state))
            return _splice_one_cotangent(locator, tuple(cotangents[1:]), cotangents[0])

        return out, wrapped_pullback

    states_in = tuple(r.state for r in refs)
    pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)
    empty_kwargs: dict[str, Any] = {}

    if has_aux:
        vjp_kwargs: dict[str, Any] = {"has_aux": True}
        if reduce_axes:
            vjp_kwargs["reduce_axes"] = reduce_axes
        if mutable_sel is None:
            out, pullback, aux = jax.vjp(pure, states_in, stripped_args, empty_kwargs, **vjp_kwargs)

            def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
                """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

                Translates the cotangent into the right shape for the wrapped
                function, calls the underlying ``vjp_fn``, and re-packages the
                result so the user sees module-shaped grads (where applicable)
                rather than raw state pytrees.
                """
                state_cts, arg_cts, _ = pullback(cotangent)
                return _splice_module_cotangents(refs, arg_cts, state_cts)

            return out, wrapped_pullback, aux

        def pure_with_updates(
            states: tuple[State, ...], stripped: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[Any, Any]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.
            """
            (out, aux), new_states = pure(states, stripped, kwargs)
            return (out, new_states), aux

        (out, new_states), pullback, aux = jax.vjp(
            pure_with_updates, states_in, stripped_args, empty_kwargs, **vjp_kwargs
        )
        apply_mutations(refs, list(new_states), mutable_sel)
        zero_states = _zeros_like_tree(new_states)

        def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.
            """
            state_cts, arg_cts, _ = pullback((cotangent, zero_states))
            return _splice_module_cotangents(refs, arg_cts, state_cts)

        return out, wrapped_pullback, aux

    vjp_kwargs = {}
    if reduce_axes:
        vjp_kwargs["reduce_axes"] = reduce_axes
    if mutable_sel is None:
        out, pullback = jax.vjp(pure, states_in, stripped_args, empty_kwargs, **vjp_kwargs)

        def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
            """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

            Translates the cotangent into the right shape for the wrapped
            function, calls the underlying ``vjp_fn``, and re-packages the
            result so the user sees module-shaped grads (where applicable)
            rather than raw state pytrees.
            """
            state_cts, arg_cts, _ = pullback(cotangent)
            return _splice_module_cotangents(refs, arg_cts, state_cts)

        return out, wrapped_pullback

    def pure_with_updates(
        states: tuple[State, ...], stripped: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, tuple[State, ...]]:
        """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

        Used by ``jax.vjp`` / ``jax.grad`` so the autodiff transform sees a
        pure ``(state, *args) -> (output, new_state)`` interface even though
        the user wrote a stateful module-method. The caller separately
        applies ``new_state`` back to the live module via :func:`apply_mutations`.
        """
        out, new_states = pure(states, stripped, kwargs)
        return out, new_states

    (out, new_states), pullback = jax.vjp(pure_with_updates, states_in, stripped_args, empty_kwargs, **vjp_kwargs)
    apply_mutations(refs, list(new_states), mutable_sel)
    zero_states = _zeros_like_tree(new_states)

    def wrapped_pullback(cotangent: Any) -> tuple[Any, ...]:
        """User-facing pullback: lifts a JAX VJP into module/state-aware tangents.

        Translates the cotangent into the right shape for the wrapped
        function, calls the underlying ``vjp_fn``, and re-packages the
        result so the user sees module-shaped grads (where applicable)
        rather than raw state pytrees.
        """
        state_cts, arg_cts, _ = pullback((cotangent, zero_states))
        return _splice_module_cotangents(refs, arg_cts, state_cts)

    return out, wrapped_pullback


def _jvp_call(
    fn: Callable[..., Any],
    primals: Sequence[Any] | None | object,
    tangents: Sequence[Any] | None | object,
    *,
    has_aux: bool,
    mutable: SelectorSugar,
) -> Any:
    """Shared implementation for direct and wrapped :func:`jvp`."""
    args = tuple(primals)
    tangent_args = tuple(tangents)
    if len(args) != len(tangent_args):
        raise TypeError(
            f"spectrax.jvp() requires primals and tangents with the same arity; "
            f"got {len(args)} primals and {len(tangent_args)} tangents."
        )

    mutable_sel = resolve_mutable(mutable)
    refs, stripped_args = locate_and_strip_fast(args)
    if len(refs) == 1 and refs[0].kind == "arg":
        ref = refs[0]
        locator = int(ref.locator)
        other_args = args[:locator] + args[locator + 1 :]
        module_tangent = tangent_args[locator]
        if isinstance(module_tangent, Module):
            _, module_tangent = export(module_tangent)
        other_tangents = tangent_args[:locator] + tangent_args[locator + 1 :]
        pure_one = (
            make_pure_readonly_single_positional(fn, ref)
            if mutable_sel is None
            else make_pure_single_positional(fn, ref)
        )

        if has_aux:
            if mutable_sel is None:
                out, aux = pure_one(ref.state, *other_args)
            else:
                (out, aux), new_state = pure_one(ref.state, *other_args)
                apply_mutations([ref], [new_state], mutable_sel)

            def fn_noaux(*a: Any) -> Any:
                """``has_aux=False`` adapter: drop the aux output so ``jax.grad`` sees a scalar."""
                value, _aux = fn(*a)
                return value

            pure_noaux = (
                make_pure_readonly_single_positional(fn_noaux, ref)
                if mutable_sel is None
                else make_pure_single_positional(fn_noaux, ref)
            )
            if mutable_sel is None:
                _, tangent_out = jax.jvp(
                    pure_noaux,
                    (ref.state, *other_args),
                    (module_tangent, *other_tangents),
                )
            else:

                def out_only(state: State, *other: Any) -> Any:
                    """Pure-output adapter: return only the primary output (no state) for ``jax.value_and_grad``."""
                    out_only_val, _ignored_state = pure_noaux(state, *other)
                    return out_only_val

                _, tangent_out = jax.jvp(
                    out_only,
                    (ref.state, *other_args),
                    (module_tangent, *other_tangents),
                )
            return out, tangent_out, aux

        if mutable_sel is None:
            out, tangent_out = jax.jvp(
                pure_one,
                (ref.state, *other_args),
                (module_tangent, *other_tangents),
            )
            return out, tangent_out

        def pure_with_updates(state: State, *other: Any) -> tuple[Any, State]:
            """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

            Used by ``jax.jvp`` so the autodiff transform sees a
            pure ``(state, *args) -> (output, new_state)`` interface even though
            the user wrote a stateful module-method. The caller separately
            applies ``new_state`` back to the live module via :func:`apply_mutations`.
            """
            return pure_one(state, *other)

        (out, new_state), (tangent_out, _tangent_state) = jax.jvp(
            pure_with_updates,
            (ref.state, *other_args),
            (module_tangent, *other_tangents),
        )
        apply_mutations([ref], [new_state], mutable_sel)
        return out, tangent_out

    states_in = tuple(r.state for r in refs)
    state_tangents, stripped_tangents = _split_module_tangents(refs, tangent_args)
    empty_kwargs: dict[str, Any] = {}
    pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)

    if has_aux:
        if mutable_sel is None:
            out, aux = pure(states_in, stripped_args, empty_kwargs)
        else:
            (out, aux), new_states = pure(states_in, stripped_args, empty_kwargs)
            apply_mutations(refs, list(new_states), mutable_sel)

        def fn_noaux(*a: Any) -> Any:
            """``has_aux=False`` adapter: drop the aux output so ``jax.grad`` sees a scalar."""
            value, _aux = fn(*a)
            return value

        pure_noaux = make_pure_readonly(fn_noaux, refs) if mutable_sel is None else make_pure(fn_noaux, refs)
        if mutable_sel is None:
            _, tangent_out = jax.jvp(
                pure_noaux,
                (states_in, stripped_args, empty_kwargs),
                (state_tangents, stripped_tangents, empty_kwargs),
            )
        else:

            def out_only(states: tuple[State, ...], stripped: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
                """Pure-output adapter: return only the primary output (no state) for ``jax.value_and_grad``."""
                out_only_val, _ignored_states = pure_noaux(states, stripped, kwargs)
                return out_only_val

            _, tangent_out = jax.jvp(
                out_only,
                (states_in, stripped_args, empty_kwargs),
                (state_tangents, stripped_tangents, empty_kwargs),
            )
        return out, tangent_out, aux

    if mutable_sel is None:
        out, tangent_out = jax.jvp(
            pure,
            (states_in, stripped_args, empty_kwargs),
            (state_tangents, stripped_tangents, empty_kwargs),
        )
        return out, tangent_out

    def pure_with_updates(
        states: tuple[State, ...], stripped: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, tuple[State, ...]]:
        """Closure that runs the (re-bound) function and returns ``(out, new_state)``.

        Used by ``jax.jvp`` so the autodiff transform sees a
        pure ``(state, *args) -> (output, new_state)`` interface even though
        the user wrote a stateful module-method. The caller separately
        applies ``new_state`` back to the live module via :func:`apply_mutations`.
        """
        out, new_states = pure(states, stripped, kwargs)
        return out, new_states

    (out, new_states), (tangent_out, _tangent_states) = jax.jvp(
        pure_with_updates,
        (states_in, stripped_args, empty_kwargs),
        (state_tangents, stripped_tangents, empty_kwargs),
    )
    apply_mutations(refs, list(new_states), mutable_sel)
    return out, tangent_out
