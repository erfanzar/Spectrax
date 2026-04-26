# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward hooks and variable-write observers.

Forward hooks fire in eager mode around :meth:`Module.__call__`; under
a spectrax transform they are suppressed with a single warning per
module (use ``self.sow("intermediates", ...)`` for transform-safe
capture). Variable observers fire on successful eager writes.
"""

from .forward import Handle, register_forward_hook, register_forward_pre_hook
from .variable import register_variable_hook

__all__ = [
    "Handle",
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_variable_hook",
]
