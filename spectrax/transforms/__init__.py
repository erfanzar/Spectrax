# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-aware JAX transforms.

Every function here is a spectrax counterpart to a JAX transform:
:func:`eval_shape`, :func:`jit`, :func:`grad`, :func:`value_and_grad`,
:func:`vmap`, :func:`scan`, :func:`remat`. They share a split/merge
shim defined in :mod:`spectrax.transforms.split_merge` that locates
:class:`~spectrax.Module` arguments, converts them into
``(GraphDef, State)`` pairs on the way in, and writes any declared
mutations back on the way out. :func:`eval_shape` is the read-only
exception: it snapshots module inputs but never writes abstract updates
back to the live modules.
"""

from .control_flow import cond, fori_loop, remat_scan, switch, while_loop
from .eval_shape import eval_shape
from .grad import grad, jvp, value_and_grad, vjp
from .jit import jit
from .remat import remat
from .rng_axes import StateAxes, split_rngs, split_stream_keys
from .scan import associative_scan, scan
from .vmap import vmap

__all__ = [
    "StateAxes",
    "associative_scan",
    "cond",
    "eval_shape",
    "fori_loop",
    "grad",
    "jit",
    "jvp",
    "remat",
    "remat_scan",
    "scan",
    "split_rngs",
    "split_stream_keys",
    "switch",
    "value_and_grad",
    "vjp",
    "vmap",
    "while_loop",
]
