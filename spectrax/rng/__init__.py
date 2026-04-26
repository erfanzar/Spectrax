# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit named-stream RNG.

This subpackage implements the :class:`~spectrax.Rngs` object and a
thread-local ``seed`` context manager. Stream state is stored in JAX
arrays inside :class:`~spectrax.RngStream` variables, so it travels
with the model's :class:`~spectrax.State` and survives
``jit`` / ``grad`` / ``vmap`` / ``scan`` / ``remat``.
"""

from .rngs import Rngs, RngStream, resolve_rngs
from .seed import default_rngs, seed

__all__ = ["RngStream", "Rngs", "default_rngs", "resolve_rngs", "seed"]
