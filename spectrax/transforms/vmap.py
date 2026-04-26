# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :func:`jax.vmap` wrapper.

Module states are not vmapped (passed through with ``in_axes=None``);
only the remaining pytree arguments follow the user-provided
``in_axes``. Mutations to declared-mutable collections are returned
with ``out_axes=None`` so a single post-vmap state update is applied
back to the live module.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import jax

from ..core.selector import SelectorSugar
from .split_merge import (
    apply_mutations,
    locate_and_strip,
    locate_and_strip_fast,
    make_pure,
    make_pure_readonly,
    make_pure_readonly_single_positional,
    make_pure_single_positional,
    resolve_mutable,
)

__all__ = ["vmap"]

F = TypeVar("F", bound=Callable[..., Any])

AxisName = Any
"""Placeholder for JAX axis-name sentinels (hashable values)."""


def _specialized_in_axes(in_axes: Any, nargs: int, locator: int) -> tuple[Any, ...]:
    """Drop the module position from positional ``in_axes``."""
    if isinstance(in_axes, tuple):
        axes = in_axes
    elif isinstance(in_axes, list):
        axes = tuple(in_axes)
    else:
        return tuple(in_axes for _ in range(nargs - 1))
    return axes[:locator] + axes[locator + 1 :]


def vmap(
    fn: F | None = None,
    *,
    mutable: SelectorSugar = (),
    in_axes: Any = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    sum_match: bool = False,
) -> F:
    """Module-aware :func:`jax.vmap`.

    Args:
        fn: Function to vmap. When omitted, returns a decorator
            factory.
        mutable: Selector controlling which collections may be written
            back after the transform.
        in_axes, out_axes, axis_name, axis_size, spmd_axis_name,
            sum_match: Forwarded to :func:`jax.vmap`.
    """
    if fn is None:
        return lambda f: vmap(
            f,
            mutable=mutable,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            sum_match=sum_match,
        )

    mutable_sel = resolve_mutable(mutable)
    empty_kwargs: dict[str, Any] = {}

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Locate modules, build the pure callable, and dispatch through :func:`jax.vmap`."""
        if kwargs:
            refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)
        else:
            refs, stripped_args = locate_and_strip_fast(args)
            stripped_kwargs = empty_kwargs
        if not kwargs and len(refs) == 1 and refs[0].kind == "arg":
            ref = refs[0]
            locator = int(ref.locator)
            other_args = stripped_args[:locator] + stripped_args[locator + 1 :]
            other_in_axes = _specialized_in_axes(in_axes, len(args), locator)
            pure = (
                make_pure_readonly_single_positional(fn, ref)
                if mutable_sel is None
                else make_pure_single_positional(fn, ref)
            )
            vmapped = jax.vmap(
                pure,
                in_axes=(None, *other_in_axes),
                out_axes=out_axes if mutable_sel is None else (out_axes, None),
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
                sum_match=sum_match,
            )
            if mutable_sel is None:
                return vmapped(ref.state, *other_args)
            out, new_state = vmapped(ref.state, *other_args)
            apply_mutations([ref], [new_state], mutable_sel)
            return out
        pure = make_pure_readonly(fn, refs) if mutable_sel is None else make_pure(fn, refs)
        pure_in_axes = (None, in_axes, None)
        states_in = tuple(r.state for r in refs)
        if mutable_sel is None:
            vmapped = jax.vmap(
                pure,
                in_axes=pure_in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
                spmd_axis_name=spmd_axis_name,
                sum_match=sum_match,
            )
            return vmapped(states_in, stripped_args, stripped_kwargs)
        vmapped = jax.vmap(
            pure,
            in_axes=pure_in_axes,
            out_axes=(out_axes, None),
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            sum_match=sum_match,
        )
        out, new_states = vmapped(states_in, stripped_args, stripped_kwargs)
        apply_mutations(refs, list(new_states), mutable_sel)
        return out

    return wrapped
