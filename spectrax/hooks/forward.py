# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward pre-hook and post-hook helpers.

These are thin function wrappers around the
:meth:`~spectrax.Module.register_forward_pre_hook` and
:meth:`~spectrax.Module.register_forward_hook` methods, for users who
prefer a functional call site.
"""

from __future__ import annotations

from ..core._typing import ForwardHook, ForwardPreHook
from ..core.module import Module, _HookHandle

Handle = _HookHandle
"""Alias for the hook handle type returned by the register functions."""

__all__ = ["Handle", "register_forward_hook", "register_forward_pre_hook"]


def register_forward_pre_hook(module: Module, fn: ForwardPreHook) -> Handle:
    """Attach ``fn`` to ``module`` as a forward pre-hook.

    See :class:`spectrax.typing.ForwardPreHook` for the callable shape.
    The returned :class:`Handle` can be used to remove the hook.
    """
    return module.register_forward_pre_hook(fn)


def register_forward_hook(module: Module, fn: ForwardHook) -> Handle:
    """Attach ``fn`` to ``module`` as a forward post-hook.

    See :class:`spectrax.typing.ForwardHook` for the callable shape.
    The returned :class:`Handle` can be used to remove the hook.
    """
    return module.register_forward_hook(fn)
