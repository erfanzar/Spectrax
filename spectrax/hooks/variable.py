# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variable write observer hooks (eager only).

:class:`_VarHookHandle` returned by :func:`register_variable_hook`
carries a :meth:`~_VarHookHandle.remove` method so hooks can be
cleanly detached. Observers do not fire under spectrax transforms —
writes are intercepted by the transform machinery before observers
would otherwise run.
"""

from __future__ import annotations

from ..core._typing import VariableObserver
from ..core.variable import Variable

__all__ = ["register_variable_hook"]


class _VarHookHandle:
    """Handle returned by :func:`register_variable_hook`.

    Holds a reference to the target :class:`Variable` and the observer
    so that :meth:`remove` can detach the observer on demand.
    """

    __slots__ = ("_fn", "_var")

    _var: Variable
    _fn: VariableObserver

    def __init__(self, var: Variable, fn: VariableObserver) -> None:
        """Record the variable and the observer function."""
        self._var = var
        self._fn = fn

    def remove(self) -> None:
        """Detach the observer from the variable's observer list."""
        self._var.remove_observer(self._fn)


def register_variable_hook(var: Variable, fn: VariableObserver) -> _VarHookHandle:
    """Register ``fn`` to be called on every eager write to ``var``.

    The observer receives ``(var, old, new)``; exceptions it raises are
    swallowed by :class:`~spectrax.Variable`'s write path.
    """
    var.add_observer(fn)
    return _VarHookHandle(var, fn)
