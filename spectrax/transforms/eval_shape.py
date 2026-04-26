# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware abstract evaluation via :func:`jax.eval_shape`."""

from __future__ import annotations

from typing import Any

import jax

from .split_merge import locate_and_strip, make_pure

__all__ = ["eval_shape"]


def eval_shape(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Module-aware :func:`jax.eval_shape`.

    Input :class:`~spectrax.Module` arguments are snapshotted and rebound
    inside a pure abstract-evaluation wrapper so shape inference works
    with the same module-aware calling style as the other spectrax
    transforms.

    Unlike mutating transforms such as :func:`~spectrax.jit` or
    :func:`~spectrax.vmap`, any variable writes that happen while
    abstract-evaluating are kept local to the abstract trace and are
    never written back to the live input modules. This keeps
    :func:`eval_shape` safe for read-mostly shape inference while still
    allowing functions to return abstract Modules or pytrees containing
    Modules.
    """
    refs, stripped_args, stripped_kwargs = locate_and_strip(args, kwargs)
    pure = make_pure(fn, refs)
    states_in = tuple(ref.state for ref in refs)
    out, _new_states = jax.eval_shape(pure, states_in, stripped_args, stripped_kwargs)
    return out
