# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parameter and byte counting helpers."""

from __future__ import annotations

from .tabulate import count_bytes, count_parameters

__all__ = ["count_bytes", "count_parameters", "format_parameters"]


def format_parameters(n: int) -> str:
    """Render a parameter count compactly: ``12_345 -> '12.3K'``."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}B"
