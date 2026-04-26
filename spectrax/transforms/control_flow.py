# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware lifts of :mod:`jax.lax` control-flow primitives.

All wrappers in this module use the same split/merge pattern: the
module is exported, partitioned by ``mutable`` into a carry-state and
an invariant, and the user function is wrapped into a pure ``(state,
...) -> (state, ...)`` form before being handed to the underlying
``jax.lax.*`` primitive.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.lax as lax

from ..core.graph import bind, export
from ..core.module import Module, _set_inside_transform
from ..core.selector import SelectorSugar
from ..core.state import State
from .scan import _check_invariant_equal
from .split_merge import _ModuleRef, apply_mutations, assert_state_unchanged, make_direct_readonly, resolve_mutable

__all__ = ["cond", "fori_loop", "remat_scan", "switch", "while_loop"]


def _find_modulelists(module: Module, prefix: tuple[str, ...] = ()):
    """Yield ``(path, ModuleList)`` for every ``ModuleList`` in *module*."""
    from ..core.containers import ModuleList

    for name in module._spx_attr_order:
        value = getattr(module, name)
        if isinstance(value, ModuleList):
            yield (*prefix, name), value
        elif isinstance(value, Module):
            yield from _find_modulelists(value, (*prefix, name))


def _inject_traced_caches(module: Module, caches: dict[tuple[str, ...], tuple[Any, Any]]) -> None:
    """Inject pre-computed ``(gdef, stacked)`` caches into *module*'s ModuleLists."""
    for path, cache in caches.items():
        target = module
        for attr in path:
            target = getattr(target, attr)
        object.__setattr__(target, "_spx_traced_cache", cache)


def _run_branch(
    gdef: Any,
    state: State,
    fn: Callable[..., Any],
    operands: tuple[Any, ...],
) -> tuple[Any, State]:
    """Rebind a fresh module, run ``fn``, and re-export the state."""
    m = bind(gdef, state)
    _set_inside_transform(True)
    try:
        y = fn(m, *operands)
    finally:
        _set_inside_transform(False)
    _, new_state = export(m)
    return y, new_state


def cond(
    pred: Any,
    on_true: Callable[..., Any],
    on_false: Callable[..., Any],
    module: Module,
    *operands: Any,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.lax.cond`.

    ``on_true`` and ``on_false`` must take ``(module, *operands)`` and
    produce the same structure of outputs **and** the same (collection,
    path) set of mutations. Everything outside ``mutable`` must remain
    structurally identical between the two branches.
    """
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(module)
    if mutable_sel is None:
        invariant_msg = (
            "cond() invariant state changed across branches. Declare the changing collection via `mutable=...`."
        )
        t_fn = make_direct_readonly(
            lambda ops: on_true(module, *ops),
            explicit_modules=(module,),
            structural_error_message=invariant_msg,
        )
        f_fn = make_direct_readonly(
            lambda ops: on_false(module, *ops),
            explicit_modules=(module,),
            structural_error_message=invariant_msg,
        )
        return lax.cond(pred, t_fn, f_fn, operands)
    carry, invariant = mutable_sel.partition_state(module, state)

    def _wrap(branch: Callable[..., Any]) -> Callable[[State, tuple[Any, ...]], tuple[Any, State]]:
        """Wrap a user branch into a ``(carry, ops) -> (y, new_carry)`` ``lax.cond`` body.

        The wrapper merges the invariant state back in before running
        the branch, then re-partitions the resulting state into
        carry+invariant and verifies that the invariant didn't change
        (if it had, the user's ``mutable=`` declaration was
        incomplete, which would corrupt cross-branch state).
        """

        def wrapped(c: State, ops: tuple[Any, ...]) -> tuple[Any, State]:
            """Merge invariant+carry, run branch, re-partition."""
            full = c.overlay(invariant)
            y, new_state = _run_branch(gdef, full, branch, ops)
            new_c, new_inv = (
                (State({}), new_state)
                if mutable_sel is None
                else mutable_sel.partition_state(
                    bind(gdef, new_state),
                    new_state,
                )
            )
            _check_invariant_equal(invariant, new_inv)
            assert_state_unchanged(module, invariant, new_inv)
            return y, new_c

        return wrapped

    t_fn = _wrap(on_true)
    f_fn = _wrap(on_false)
    (y, new_carry) = lax.cond(pred, t_fn, f_fn, carry, operands)
    apply_mutations(
        [_ModuleRef("arg", 0, module, gdef, state)],
        [new_carry.overlay(invariant)],
        mutable_sel,
    )
    return y


def switch(
    index: Any,
    branches: Sequence[Callable[..., Any]],
    module: Module,
    *operands: Any,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.lax.switch`."""
    if not branches:
        raise ValueError("switch() requires at least one branch")
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(module)
    if mutable_sel is None:
        invariant_msg = (
            "switch() invariant state changed across branches. Declare the changing collection via `mutable=...`."
        )
        wrapped_branches = [
            make_direct_readonly(
                lambda ops, branch=branch: branch(module, *ops),
                explicit_modules=(module,),
                structural_error_message=invariant_msg,
            )
            for branch in branches
        ]
        return lax.switch(index, wrapped_branches, operands)
    carry, invariant = mutable_sel.partition_state(module, state)

    def _wrap(branch: Callable[..., Any]) -> Callable[[State, tuple[Any, ...]], tuple[Any, State]]:
        """Wrap a user branch into a ``(carry, ops) -> (y, new_carry)`` ``lax.switch`` body.

        Same semantics as the :func:`cond` wrapper: merge invariant
        state, run the branch, re-partition, verify that the invariant
        survived. Per-branch closures capture the branch by default
        argument so the loop variable doesn't leak between branches.
        """

        def wrapped(c: State, ops: tuple[Any, ...]) -> tuple[Any, State]:
            """Merge invariant+carry, run the branch, re-partition new state."""
            full = c.overlay(invariant)
            y, new_state = _run_branch(gdef, full, branch, ops)
            new_c, new_inv = (
                (State({}), new_state)
                if mutable_sel is None
                else mutable_sel.partition_state(
                    bind(gdef, new_state),
                    new_state,
                )
            )
            _check_invariant_equal(invariant, new_inv)
            assert_state_unchanged(module, invariant, new_inv)
            return y, new_c

        return wrapped

    wrapped_branches = [_wrap(b) for b in branches]
    (y, new_carry) = lax.switch(index, wrapped_branches, carry, operands)
    apply_mutations(
        [_ModuleRef("arg", 0, module, gdef, state)],
        [new_carry.overlay(invariant)],
        mutable_sel,
    )
    return y


def while_loop(
    cond_fn: Callable[[Module, Any], Any],
    body_fn: Callable[[Module, Any], Any],
    init_module: Module,
    init_carry: Any,
    *,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.lax.while_loop`.

    The loop carry is ``(state_carry, user_carry)``. ``cond_fn`` and
    ``body_fn`` are both called with a fresh module rebound from the
    current state and the current ``user_carry``.
    """
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(init_module)
    if mutable_sel is None:
        invariant_msg = (
            "while_loop() invariant state changed across iterations. Declare the changing collection via `mutable=...`."
        )
        cond_wrap = make_direct_readonly(
            lambda user_carry: cond_fn(init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message=invariant_msg,
        )
        body_wrap = make_direct_readonly(
            lambda user_carry: body_fn(init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message=invariant_msg,
        )
        return lax.while_loop(cond_wrap, body_wrap, init_carry)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def cond_wrap(loop_carry: tuple[State, Any]) -> Any:
        """``lax.while_loop`` predicate: bind the latest carry+invariant and call user ``cond_fn``."""
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            return cond_fn(m, uc)
        finally:
            _set_inside_transform(False)

    def body_wrap(loop_carry: tuple[State, Any]) -> tuple[State, Any]:
        """``lax.while_loop`` body: run user ``body_fn``, re-export, verify invariants."""
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            new_uc = body_fn(m, uc)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, new_uc

    final_c, final_uc = lax.while_loop(cond_wrap, body_wrap, (carry_state, init_carry))
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return final_uc


def fori_loop(
    lower: int,
    upper: int,
    body_fn: Callable[[int, Module, Any], Any],
    init_module: Module,
    init_carry: Any,
    *,
    mutable: SelectorSugar = (),
) -> Any:
    """Module-aware :func:`jax.lax.fori_loop`."""
    mutable_sel = resolve_mutable(mutable)
    gdef, state = export(init_module)

    from ..core.containers import _stack_module_states

    modulelist_caches: dict[tuple[str, ...], tuple[Any, Any]] = {}
    for path, ml in _find_modulelists(init_module):
        if not ml._spx_items:
            continue
        gdef_ml, stacked = _stack_module_states(ml._spx_items, context="fori_loop ModuleList cache")
        modulelist_caches[path] = (gdef_ml, stacked)

    if mutable_sel is None and not modulelist_caches:
        body_wrap = make_direct_readonly(
            lambda i, user_carry: body_fn(i, init_module, user_carry),
            explicit_modules=(init_module,),
            structural_error_message="fori_loop() invariant state changed across iterations. Declare the changing collection via `mutable=...`.",
        )
        return lax.fori_loop(lower, upper, body_wrap, init_carry)
    if mutable_sel is None:

        def body_wrap(i: Any, user_carry: Any) -> Any:
            """``fori_loop`` body (no mutable state): bind module, run user body, assert invariants."""
            m = bind(gdef, state)
            if modulelist_caches:
                _inject_traced_caches(m, modulelist_caches)
            _set_inside_transform(True)
            try:
                new_user_carry = body_fn(i, m, user_carry)
            finally:
                _set_inside_transform(False)
            _, new_state = export(m)
            _check_invariant_equal(state, new_state)
            assert_state_unchanged(init_module, state, new_state)
            return new_user_carry

        return lax.fori_loop(lower, upper, body_wrap, init_carry)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def body_wrap(i: Any, loop_carry: tuple[State, Any]) -> tuple[State, Any]:
        """``fori_loop`` body (with mutable state): run user body, re-partition state, verify invariants."""
        c_state, uc = loop_carry
        full = c_state.overlay(invariant)
        m = bind(gdef, full)
        if modulelist_caches:
            _inject_traced_caches(m, modulelist_caches)
        _set_inside_transform(True)
        try:
            new_uc = body_fn(i, m, uc)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, new_uc

    final_c, final_uc = lax.fori_loop(lower, upper, body_wrap, (carry_state, init_carry))
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return final_uc


def remat_scan(
    fn: Callable[[Module, Any], Any],
    init_module: Module,
    xs: Any,
    *,
    length: int | None = None,
    mutable: SelectorSugar = (),
    policy: Callable[..., Any] | None = None,
    prevent_cse: bool = True,
    unroll: int = 1,
) -> Any:
    """``scan(fn, init_module, xs)`` with each step wrapped in :func:`jax.checkpoint`.

    This is the standard pattern for large transformer stacks: the
    forward pass stores only activations at step boundaries, halving
    peak memory at the cost of recomputing intermediate activations
    during the backward pass.
    """
    mutable_sel = resolve_mutable(mutable)
    if mutable_sel is None:
        step = make_direct_readonly(
            lambda _carry, x: (None, fn(init_module, x)),
            explicit_modules=(init_module,),
            structural_error_message="scan() invariant state changed across iterations. Declare the changing collection via `mutable=...`.",
        )
        rematted = jax.checkpoint(step, policy=policy, prevent_cse=prevent_cse)
        _carry, ys = lax.scan(rematted, None, xs, length=length, unroll=unroll)
        return ys
    gdef, state = export(init_module)
    carry_state, invariant = mutable_sel.partition_state(init_module, state)

    def step(c: State, x: Any) -> tuple[State, Any]:
        """Single ``remat_scan`` step: bind module, run ``fn``, re-partition + invariant-check."""
        full = c.overlay(invariant)
        m = bind(gdef, full)
        _set_inside_transform(True)
        try:
            y = fn(m, x)
        finally:
            _set_inside_transform(False)
        _, new_state = export(m)
        new_c, new_inv = (State({}), new_state) if mutable_sel is None else mutable_sel.partition_state(m, new_state)
        _check_invariant_equal(invariant, new_inv)
        assert_state_unchanged(init_module, invariant, new_inv)
        return new_c, y

    rematted = jax.checkpoint(step, policy=policy, prevent_cse=prevent_cse)
    final_c, ys = lax.scan(rematted, carry_state, xs, length=length, unroll=unroll)
    apply_mutations(
        [_ModuleRef("arg", 0, init_module, gdef, state)],
        [final_c.overlay(invariant)],
        mutable_sel,
    )
    return ys
