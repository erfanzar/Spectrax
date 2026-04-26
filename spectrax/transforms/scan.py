# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware :mod:`jax.lax` scan wrappers.

Signature::

    scan(fn, init_module, xs, *, length=None, mutable=(), unroll=1)

``scan`` uses ``fn(module, x) -> y``. The module's state is partitioned
into a *carry* (the declared-mutable collections) and an *invariant*
(everything else). Only the carry is threaded through the scan; the
invariant is required to stay structurally identical from step to step.

This module also provides :func:`associative_scan`, a module-aware
wrapper around :func:`jax.lax.associative_scan` for *pure* associative
binary functions. Unlike :func:`scan`, there is no module-state carry,
so mutations are rejected explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.lax as lax

from ..core.errors import IllegalMutationError
from ..core.graph import bind, export, live_variables
from ..core.module import Module, _graph_epoch, _set_inside_transform
from ..core.selector import SelectorSugar
from ..core.state import State
from .split_merge import _ModuleRef, apply_mutations, assert_state_unchanged, make_direct_readonly, resolve_mutable

__all__ = ["associative_scan", "scan"]


def associative_scan(
    fn: Callable[[Module, Any, Any], Any],
    module: Module,
    elems: Any,
    *,
    reverse: bool = False,
    axis: int = 0,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.lax.associative_scan`.

    ``fn`` has the shape ``(module, a, b) -> c`` and must be associative
    over ``a`` and ``b`` in the same way required by upstream JAX.

    Unlike :func:`scan`, :func:`jax.lax.associative_scan` performs a
    tree-structured parallel prefix with no state carry. That means
    module mutations have no well-defined semantics here, so the module
    is rebound from its original state for each combine and any write
    to that rebound module raises :class:`IllegalMutationError`.

    Args:
        fn: Associative binary combine function ``(module, a, b) -> c``.
        module: Read-only :class:`Module` captured by the combine.
        elems: Input pytree whose ``axis`` dimension is scanned.
        reverse: Forwarded to :func:`jax.lax.associative_scan`.
        axis: Forwarded to :func:`jax.lax.associative_scan`.
        mutable: Unsupported for associative scans; anything other than
            the empty value raises :class:`ValueError`.

    Returns:
        The prefix-combined values with the same structure as ``elems``.
    """
    mutable_sel = resolve_mutable(mutable)
    if mutable_sel is not None:
        raise ValueError(
            "associative_scan() does not support mutable= because "
            "jax.lax.associative_scan has no module-state carry. "
            "Keep the combine function pure."
        )
    if not isinstance(module, Module):
        raise TypeError("associative_scan() requires a Module as module")

    gdef, state = export(module)

    def combine(a: Any, b: Any) -> Any:
        """Rebind a fresh module for this pairwise combine and forbid writes."""
        m = bind(gdef, state)
        epoch_before = _graph_epoch()
        snapshots = [(path, var.kind, var, var._value) for path, var in live_variables(m)]
        _set_inside_transform(True)
        try:
            out = fn(m, a, b)
        finally:
            _set_inside_transform(False)

        if _graph_epoch() != epoch_before:
            raise IllegalMutationError(
                "associative_scan() does not support structural module "
                "mutations because jax.lax.associative_scan has no "
                "module-state carry."
            )

        for path, kind, var, initial in snapshots:
            if var._value is not initial:
                raise IllegalMutationError(
                    "associative_scan() does not support module mutations "
                    "because jax.lax.associative_scan has no module-state "
                    f"carry. The combine function mutated {kind!r} at {path!r}; "
                    "keep it pure."
                )
        return out

    return lax.associative_scan(combine, elems, reverse=reverse, axis=axis)


def scan(
    fn: Callable[[Module, Any], Any],
    init_module: Module,
    xs: Any,
    *,
    length: int | None = None,
    mutable: SelectorSugar = (),
    unroll: int = 1,
) -> Any:
    """Scan ``fn`` over ``xs`` threading ``init_module`` as state.

    Args:
        fn: Per-step function ``(module, x) -> y``.
        init_module: :class:`Module` providing the initial state.
        xs: Scanned sequence (pytree with a leading axis).
        length: Optional explicit sequence length.
        mutable: Selector for collections that may be carried through
            the scan. Collections outside this selector must be
            structurally invariant from step to step.
        unroll: Forwarded to :func:`jax.lax.scan`.

    Returns:
        The stacked ``ys`` from every step. When ``mutable=`` declares
        any collections, the final carry for those collections is
        written back to ``init_module`` in place; otherwise the module
        is left untouched (and any mutation raises).
    """
    mutable_sel = resolve_mutable(mutable)
    if not isinstance(init_module, Module):
        raise TypeError("scan() requires a Module as init_module")

    if mutable_sel is None:
        guarded_step = make_direct_readonly(
            lambda _carry, x: (None, fn(init_module, x)),
            explicit_modules=(init_module,),
            structural_error_message=(
                "scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
            ),
        )

        _carry, ys = lax.scan(guarded_step, None, xs, length=length, unroll=unroll)
        return ys
    gdef, state = export(init_module)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def step(carry: State, x: Any) -> tuple[State, Any]:
        """Single scan step: merge state, run ``fn``, re-partition."""
        full = carry.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            y = fn(m, x)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_carry, new_invariant = (
            (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        )
        _check_invariant_equal(invariant, new_invariant)
        assert_state_unchanged(m, invariant, new_invariant)
        return new_carry, y

    final_carry, ys = lax.scan(step, carry_state, xs, length=length, unroll=unroll)
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_carry.overlay(invariant)],
        mutable_sel,
    )
    return ys


def _check_invariant_equal(a: State, b: State) -> None:
    """Assert ``a`` and ``b`` have the same ``(collection, path)`` keys.

    Structural (key-set) check only; value equality is too strong under
    tracing. Mismatches raise :class:`ValueError` prompting the user to
    declare the differing collection as mutable.
    """
    a_keys = {(c, p) for c, p in a.paths()}
    b_keys = {(c, p) for c, p in b.paths()}
    if a_keys != b_keys:
        raise ValueError(
            "scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
        )
