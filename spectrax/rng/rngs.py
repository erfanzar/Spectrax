# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`Rngs` — explicit, named-stream RNG whose state lives in :class:`~spectrax.State`.

:class:`Rngs` is a :class:`~spectrax.Module` that owns a bag of
:class:`RngStream` variables of kind ``'rng'``. Each stream packs a
PRNG key and a 64-bit counter (split into two uint32 words) into a
single JAX array leaf so it round-trips cleanly through ``jit`` and
friends.

Every attribute access on a stream (``rngs.parameters``, ``rngs.dropout``)
derives a fresh PRNG key from the stream's ``(key, counter)`` pair via
:func:`jax.random.fold_in` and advances the counter by one. The
equivalent method form is :meth:`Rngs.key`.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, PRNGKey
from ..core.module import Module, _bump_graph_epoch, _inside_transform
from ..core.variable import Variable

__all__ = ["RngStream", "Rngs", "resolve_rngs"]


class RngStream(Variable):
    """A single named RNG stream.

    The leaf value is a ``uint32`` 1-D array laid out as
    ``[*key_words, counter_hi, counter_lo]`` so the entire stream
    (including its counter) is a single JAX-compatible array.
    """

    default_kind: ClassVar[str] = "rng"
    inherit_stage_assignment: ClassVar[bool] = False

    def __init__(self, key: ArrayLike, *, counter: int = 0, ref_id: int | None = None) -> None:
        """Construct a stream from a PRNG key and a starting counter."""
        raw = _to_raw_key(key)
        if raw.ndim != 1:
            raise ValueError(f"RngStream key must be 1-D uint32, got shape {raw.shape}")
        hi = jnp.uint32((int(counter) >> 32) & 0xFFFFFFFF)
        lo = jnp.uint32(int(counter) & 0xFFFFFFFF)
        packed = jnp.concatenate([raw, jnp.array([hi, lo], dtype=jnp.uint32)])
        super().__init__(
            packed,
            kind="rng",
            ref_id=ref_id,
            metadata={"key_size": int(raw.shape[0])},
        )

    def _unpack(self) -> tuple[Array, Array, Array]:
        """Return ``(raw_key_u32, counter_hi, counter_lo)`` from the packed leaf."""
        n = int(self.metadata["key_size"])
        raw = self._raw_get()
        return raw[:n], raw[n], raw[n + 1]

    def _repack(self, key: Array, hi: Array, lo: Array) -> Array:
        """Pack ``(key, counter_hi, counter_lo)`` back into a single leaf."""
        return jnp.concatenate([jnp.asarray(key, dtype=jnp.uint32), jnp.asarray([hi, lo], dtype=jnp.uint32)])

    def next_key(self) -> PRNGKey:
        """Return a fresh typed PRNG key and advance the counter by one."""
        raw, hi, lo = self._unpack()
        typed = jax.random.wrap_key_data(raw)
        out_key = jax.random.fold_in(jax.random.fold_in(typed, hi.astype(jnp.int32)), lo.astype(jnp.int32))
        new_lo = lo + jnp.uint32(1)
        carry = jnp.where(new_lo == jnp.uint32(0), jnp.uint32(1), jnp.uint32(0))
        new_hi = hi + carry
        self.value = self._repack(raw, new_hi, new_lo)
        return out_key

    def fold_in(self, tag: int | str) -> RngStream:
        """Return a derived stream whose key is ``fold_in(self.key, tag)``."""
        raw, *_ = self._unpack()
        h = tag if isinstance(tag, int) else _str_hash(tag)
        typed = jax.random.wrap_key_data(raw)
        return RngStream(jax.random.fold_in(typed, jnp.int32(h)))

    def __call__(self) -> PRNGKey:
        """Alias for :meth:`next_key`."""
        return self.next_key()


class Rngs(Module):
    """Named collection of RNG streams.

    Typical usage::

        rngs = Rngs(0)                 # root seed 0
        rngs = Rngs(0, dropout=1)      # explicit per-stream seed
        key = rngs.parameters              # fresh key from 'parameters' stream
        key = rngs.dropout             # fresh key from 'dropout' stream
        key = rngs.key("custom")       # equivalent method form
        stream = rngs.stream("parameters") # underlying :class:`RngStream`
        rngs.fork(B)                   # B independent Rngs (for vmap)

    Accessing an undeclared stream lazily derives it from ``default`` by
    folding in the attribute name. Derived streams are cached so repeated
    accesses advance a single counter rather than spawning independent
    streams.
    """

    _spx_container_kind: ClassVar[str] = "dict"
    _spx_items: dict[str, RngStream]

    def __init__(self, default: int | ArrayLike = 0, **streams: int | ArrayLike) -> None:
        """Construct an :class:`Rngs` with an explicit default stream."""
        super().__init__()
        object.__setattr__(self, "_spx_items", {"default": RngStream(_coerce_seed(default))})
        for name, seed in streams.items():
            if name == "default":
                continue
            self._spx_items[name] = RngStream(_coerce_seed(seed))

    def _spx_graph_children(self):
        """Yield ``(name, stream)`` for every declared / derived stream."""
        yield from self._spx_items.items()

    def stream(self, name: str) -> RngStream:
        """Return (creating if needed) the underlying :class:`RngStream`.

        The default stream is always present; any other name is derived
        lazily from ``default`` via :meth:`RngStream.fold_in` and then
        cached.
        """
        items = self._spx_items
        if name in items:
            return items[name]
        items[name] = items["default"].fold_in(name)
        if not _inside_transform():
            _bump_graph_epoch()
        return items[name]

    def key(self, name: str = "default") -> PRNGKey:
        """Return the next PRNG key from the named stream (method form)."""
        if name != "default" and name not in self._spx_items and _inside_transform():
            key = self._spx_items["default"].next_key()
            tag = _str_hash(name)
            return jax.random.fold_in(key, jnp.int32(tag))
        return self.stream(name).next_key()

    def __getattr__(self, name: str) -> PRNGKey:
        """Attribute access returns a fresh PRNG key from the named stream.

        Private names (starting with ``_``) and class methods are not
        routed here because normal attribute lookup handles them first.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        items = self.__dict__.get("_spx_items")
        if items is None:
            raise AttributeError(name)
        return self.key(name)

    def fold_in(self, tag: int | str) -> Rngs:
        """Return a new :class:`Rngs` with every stream folded by ``tag``."""
        new = Rngs.__new__(Rngs)
        Module.__init__(new)
        object.__setattr__(new, "_spx_items", {})
        for name, stream in self._spx_items.items():
            new._spx_items[name] = stream.fold_in(tag)
        return new

    def fork(self, n: int) -> _ForkedRngs:
        """Produce ``n`` independent :class:`Rngs` along a leading axis."""
        if n <= 0:
            raise ValueError(f"fork count must be > 0, got {n}.")
        typed = self._spx_items["default"].next_key()
        keys = jax.random.split(typed, n)
        return _ForkedRngs(keys)


class _ForkedRngs:
    """A stack of :class:`Rngs` instances sharing a single keys array."""

    __slots__ = ("_keys",)

    _keys: Array

    def __init__(self, keys: Array) -> None:
        """Wrap a stacked-keys array of shape ``(n, key_size)``."""
        self._keys = keys

    def __len__(self) -> int:
        """Return the leading axis length."""
        return int(self._keys.shape[0])

    def __getitem__(self, i: int) -> Rngs:
        """Return an :class:`Rngs` seeded with the ``i``-th slice."""
        return Rngs(self._keys[i])

    def as_stack(self) -> Array:
        """Return the backing stacked-keys array."""
        return self._keys


def resolve_rngs(rngs: Rngs | int | None = None) -> Rngs:
    """Coerce an ``rngs`` argument into a concrete :class:`Rngs`.

    Accepted inputs:

    * :class:`Rngs` — returned as-is.
    * ``int`` — wrapped as ``Rngs(seed)``.
    * ``None`` — falls back to the thread-local
      :func:`spectrax.seed` context, or raises :class:`RuntimeError`
      if none is active.

    Used by every layer constructor that needs PRNG keys at
    initialization time.
    """
    from .seed import default_rngs, has_default_rngs

    if rngs is not None:
        return rngs if isinstance(rngs, Rngs) else Rngs(rngs)
    if has_default_rngs():
        return default_rngs()
    raise RuntimeError("Layer construction requires rngs. Pass rngs=... or wrap with `spectrax.seed(n)`.")


def _coerce_seed(s: int | ArrayLike) -> Array:
    """Coerce an int seed or existing key into a typed PRNG key."""
    if isinstance(s, int):
        return jax.random.PRNGKey(s)
    return _to_typed_key(s)


def _to_typed_key(x: ArrayLike) -> Array:
    """Return a typed PRNG key, wrapping raw uint32 data when necessary."""
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        return x
    return jax.random.wrap_key_data(jnp.asarray(x, dtype=jnp.uint32))


def _to_raw_key(x: ArrayLike) -> Array:
    """Return raw ``uint32`` key data, unwrapping a typed key when necessary."""
    if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(x)
    return jnp.asarray(x, dtype=jnp.uint32)


def _str_hash(s: str) -> int:
    """Deterministic FNV-1a 32-bit hash of a UTF-8 string, mapped to int32."""
    h = 0x811C9DC5
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h
