# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tiny compatibility layer formerly imported from ``spectrax._internal.pytree``."""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping

PyTree = tp.Any


def flatten_dict(
    xs: dict | Mapping,
    keep_empty_nodes: bool = False,
    is_leaf: tp.Callable[[tuple, tp.Any], bool] | None = None,
    sep: str | None = None,
    fumap: bool = False,
) -> dict[tuple | str, tp.Any]:
    """Flatten a nested dictionary into a single-level mapping.

    Recursively walks *xs* and produces a flat dictionary whose keys are
    either tuples of nested keys (when *sep* is ``None``) or strings
    joined by *sep*.

    Args:
        xs: Dictionary or mapping to flatten.
        keep_empty_nodes: Whether to emit entries for empty dictionary
            nodes. Defaults to ``False``.
        is_leaf: Optional predicate ``fn(path, obj) -> bool``. When it
            returns ``True``, *obj* is treated as a leaf even if it is a
            dict.
        sep: Separator used to join nested keys into a single string.
            If ``None``, keys remain as tuples. Defaults to ``None``.
        fumap: If ``True``, accept any :class:`~collections.abc.Mapping`
            without requiring the top-level object to be a ``dict``.
            Defaults to ``False``.

    Returns:
        A flat dictionary mapping string or tuple keys to the original
        leaf values.

    Raises:
        TypeError: If *xs* is not a dict or Mapping and *fumap* is
            ``False``.
    """
    if not fumap and not isinstance(xs, dict):
        if not isinstance(xs, Mapping):
            raise TypeError(f"expected dict or Mapping; got {type(xs)}")

    def _key(path: tuple) -> tuple | str:
        if sep is None:
            return path
        return sep.join(str(p) for p in path)

    def _flatten(obj: tp.Any, prefix: tuple) -> dict:
        if not isinstance(obj, dict) or (is_leaf and is_leaf(prefix, obj)):
            return {_key(prefix): obj}
        result: dict = {}
        is_empty = True
        for key, value in obj.items():
            is_empty = False
            result.update(_flatten(value, (*prefix, key)))
        if keep_empty_nodes and is_empty:
            if prefix == ():
                return {}
            return {_key(prefix): {}}
        return result

    return _flatten(xs, ())
