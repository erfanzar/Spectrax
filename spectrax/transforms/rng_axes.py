# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`split_rngs` and :class:`StateAxes` for transform-aware RNG handling.

Without these helpers, ``spx.vmap`` / ``spx.scan`` broadcast the
:class:`~spectrax.Rngs` state across the mapped axis — so every batch
element draws *the same* dropout mask. :func:`split_rngs` re-seeds the
relevant streams so each index along the axis gets an independent key.

:class:`StateAxes` is declarative metadata attached to a transform call
describing how each variable collection should be mapped (``None``
broadcasts, an ``int`` maps along that axis, ``"split"`` auto-splits
rngs).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..core._typing import Array
from ..core.module import Module as _Module
from ..core.variable import Variable
from ..rng.rngs import Rngs, RngStream

__all__ = ["StateAxes", "split_rngs", "split_stream_keys"]


@dataclass(frozen=True)
class StateAxes:
    """Per-collection axis spec for :func:`spectrax.vmap` / :func:`spectrax.scan`.

    Values:

    * ``None`` — the collection is broadcast (replicated) along the axis.
    * ``int`` — the collection is mapped along that leaf axis.
    * ``"split"`` — auto-split: each index along the mapped axis receives
      an independent PRNG-derived state.
    """

    axes: Mapping[str, int | str | None]

    def get(self, collection: str, default: int | str | None = None) -> int | str | None:
        """Return the axis spec for ``collection``, or ``default`` if absent."""
        return self.axes.get(collection, default)

    def __iter__(self) -> Iterator[tuple[str, int | str | None]]:
        """Iterate ``(collection, axis)`` pairs."""
        return iter(self.axes.items())


def split_stream_keys(stream: RngStream, axis_size: int) -> Array:
    """Return ``axis_size`` independent raw keys derived from ``stream``.

    The returned array has shape ``(axis_size, key_size)`` and dtype
    ``uint32``. Splitting advances the stream counter by one.
    """
    if axis_size <= 0:
        raise ValueError(f"axis_size must be > 0, got {axis_size}.")
    raw, hi, lo = stream._unpack()
    typed = jax.random.fold_in(jax.random.fold_in(jax.random.wrap_key_data(raw), hi.astype(jnp.int32)), lo.astype(jnp.int32))
    sub = jax.random.split(typed, axis_size)
    new_lo = lo + jnp.uint32(1)
    carry = jnp.where(new_lo == jnp.uint32(0), jnp.uint32(1), jnp.uint32(0))
    stream.value = jnp.concatenate([raw, jnp.array([hi + carry, new_lo], dtype=jnp.uint32)])
    return jax.vmap(jax.random.key_data)(sub)


def _clone_stream(stream: RngStream) -> RngStream:
    """Clone a stream without aliasing mutable counter state between forks."""
    clone = object.__new__(RngStream)
    Variable.__init__(
        clone,
        stream._raw_get(),
        kind="rng",
        metadata=dict(stream.metadata),
    )
    return clone


@contextlib.contextmanager
def split_rngs(rngs: Rngs, *, axis_size: int, only: tuple[str, ...] | None = None) -> Iterator[list[Rngs]]:
    """Context that yields a list of ``axis_size`` independent :class:`Rngs`.

    Inside the block, each ``Rngs`` in the list draws fresh keys from an
    independent sub-seed derived from the parent. On exit, any state
    remains on the yielded list (no copy-back to ``rngs`` — callers who
    need per-element mutations should write them back explicitly).

    Args:
        rngs: The parent :class:`~spectrax.Rngs`.
        axis_size: Number of independent forks to produce.
        only: Optional tuple of stream names to split. When ``None``
            (default), every stream in ``rngs`` is split.
    """
    if axis_size <= 0:
        raise ValueError(f"axis_size must be > 0, got {axis_size}.")
    names = tuple(rngs._spx_items.keys()) if only is None else tuple(only)
    split_map: dict[str, Any] = {}
    for nm in names:
        stream = rngs.stream(nm)
        split_map[nm] = split_stream_keys(stream, axis_size)
    forks: list[Rngs] = []
    for i in range(axis_size):
        fork = Rngs.__new__(Rngs)
        _Module.__init__(fork)
        object.__setattr__(fork, "_spx_items", {})
        for nm in names:
            key_i = split_map[nm][i]
            fork._spx_items[nm] = RngStream(key_i)
        for nm, s in rngs._spx_items.items():
            if nm not in names:
                fork._spx_items[nm] = _clone_stream(s)
        forks.append(fork)
    try:
        yield forks
    finally:
        pass
