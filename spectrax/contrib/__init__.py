# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in helpers that depend on third-party libraries.

Anything here lives behind an optional install (e.g.
``pip install spectrax-lib[contrib]``) and must import gracefully when its
dependency is missing.
"""

from .optimizer import MultiOptimizer, Optimizer

__all__ = ["MultiOptimizer", "Optimizer"]
