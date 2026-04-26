# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure initializer factories.

Each public function here returns a callable implementing the
:class:`~spectrax.typing.Initializer` protocol —
``(key, shape, dtype) -> Array`` — suitable for direct use when
constructing a :class:`~spectrax.Parameter`.
"""

from .constant import constant, ones, zeros
from .kaiming import kaiming_normal, kaiming_uniform
from .normal import normal, truncated_normal
from .orthogonal import orthogonal
from .uniform import uniform
from .xavier import xavier_normal, xavier_uniform

__all__ = [
    "constant",
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "ones",
    "orthogonal",
    "truncated_normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
