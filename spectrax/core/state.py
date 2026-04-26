# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`State`: the collection-partitioned, path-keyed pytree of arrays.

Every :class:`~spectrax.Variable` in a module graph stores its array in a
:class:`State`. The layout is two-level: the outer dict is keyed by
collection (``"parameters"``, ``"batch_stats"``, ...), the inner dict is a
*nested* dictionary keyed by the path components of the variable's
canonical location. :class:`State` is registered as a JAX pytree so it
passes transparently through ``jax.jit`` / ``grad`` / ``vmap`` / ``scan``
/ ``remat``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Any

import jax

from ._typing import Array, Path
from .paths import path_to_str, str_to_path

__all__ = ["State"]

Leaf = Array | Any
"""A stored leaf. Typically an :class:`Array` but left wide for traced values."""

Writer = Callable[[Leaf], None]
"""Setter callback for live-backed leaves exported from a module."""


def _is_leaf(v: Any) -> bool:
    """Return ``True`` if *v* is a leaf (not a nested dict)."""
    return not isinstance(v, dict)


def _nested_items(d: dict[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path_tuple, value)`` for every leaf in nested dict *d*."""
    for k in sorted(d.keys(), key=_sort_key):
        v = d[k]
        if isinstance(v, dict) and v:
            yield from _nested_items(v, (*prefix, k))
        else:
            yield (*prefix, k), v


def _nested_paths(d: dict[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Yield ``path_tuple`` for every leaf in nested dict *d*."""
    for k in sorted(d.keys(), key=_sort_key):
        v = d[k]
        if isinstance(v, dict) and v:
            yield from _nested_paths(v, (*prefix, k))
        else:
            yield (*prefix, k)


def _sorted_nested_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively key-sorted copy of nested dict *d*."""
    return {
        k: _sorted_nested_dict(v) if isinstance(v, dict) and v else v
        for k, v in sorted(d.items(), key=lambda item: _sort_key(item[0]))
    }


def _nested_get(d: dict[str, Any], path: tuple[str, ...], default: Any = ...) -> Any:
    """Traverse nested dict *d* along *path* and return the leaf.

    Raises ``KeyError`` when a segment is missing and no *default* was
    supplied.
    """
    try:
        for key in path:
            d = d[key]
        return d
    except (KeyError, TypeError):
        if default is not ...:
            return default
        raise


def _nested_set(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set *value* at *path* inside nested dict *d*, mutating in place."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _deep_copy_nested(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively copy nested dict structure while sharing leaf objects."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = _deep_copy_nested(v) if isinstance(v, dict) else v
    return out


def _merge_nested(into: dict[str, Any], other: dict[str, Any]) -> None:
    """Merge *other* into *into* in place, sharing leaf objects."""
    for k, v in other.items():
        if k in into and isinstance(into[k], dict) and isinstance(v, dict):
            _merge_nested(into[k], v)
        elif isinstance(v, dict):
            into[k] = _deep_copy_nested(v)
        else:
            into[k] = v


def _normalize_inner_mapping(value: Mapping[str, Leaf]) -> dict[str, Leaf]:
    """Normalize an inner collection mapping to SpectraX's nested layout."""
    inner = _mapping_to_nested_dict(value)
    is_nested = any(isinstance(v, Mapping) and v for v in inner.values())
    return inner if is_nested else _flat_to_nested(inner)


def _mapping_to_nested_dict(value: Mapping[Any, Any]) -> dict[Any, Any]:
    """Return a plain nested dict from any mapping/proxy tree."""
    out: dict[Any, Any] = {}
    for key, leaf in value.items():
        out[key] = _mapping_to_nested_dict(leaf) if isinstance(leaf, Mapping) else leaf
    return out


def _sync_nested(state: State, collection: str, subtree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> None:
    """Sync every leaf in ``subtree`` through ``state``'s live writers."""
    for key, value in subtree.items():
        path = (*prefix, key)
        if isinstance(value, Mapping) and value:
            _sync_nested(state, collection, value, path)
        else:
            state._sync_leaf(collection, path, value)


def _map_fn_arity(fn: Callable[..., Any]) -> int:
    """Return the supported positional arity for :meth:`State.map`.

    ``1`` means ``fn(value)``, ``2`` means ``fn(path, value)``, and ``3``
    means ``fn(path, value, collection)``. When the callable cannot be
    introspected we conservatively fall back to ``1``.
    """
    code = getattr(fn, "__code__", None)
    if code is not None:
        flags = code.co_flags
        if flags & 0x04:
            return 3
        positional_count = code.co_argcount
        if inspect.ismethod(fn) and getattr(fn, "__self__", None) is not None:
            positional_count = max(0, positional_count - 1)
        defaults = getattr(fn, "__defaults__", None) or ()
        required_positional = max(0, positional_count - len(defaults))
        names = list(code.co_varnames[:positional_count])
        if positional_count >= 3:
            if required_positional >= 3 or names[0] in {"path", "key"} or names[2] in {"collection", "kind"}:
                return 3
            return 1
        if positional_count == 2 and (
            required_positional >= 2 or names[0] in {"path", "key"} or names[1] in {"value", "leaf"}
        ):
            return 2
        return 1

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1

    positional: list[inspect.Parameter] = []
    has_varargs = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue
        positional.append(parameter)

    if has_varargs:
        return 3

    required_positional = sum(1 for p in positional if p.default is inspect.Parameter.empty)
    if len(positional) >= 3:
        names = [p.name for p in positional[:3]]
        if required_positional >= 3 or names[0] in {"path", "key"} or names[2] in {"collection", "kind"}:
            return 3
        return 1
    if len(positional) == 2 and (
        required_positional >= 2 or positional[0].name in {"path", "key"} or positional[1].name in {"value", "leaf"}
    ):
        return 2
    return 1


def _call_map_fn(
    fn: Callable[..., Any],
    *,
    arity: int,
    collection: str,
    path: str,
    value: Any,
) -> Any:
    """Call a :meth:`State.map` callback using its supported signature."""
    if arity == 1:
        return fn(value)
    if arity == 2:
        return fn(path, value)
    return fn(path, value, collection)


def _map_nested_values(d: dict[str, Any], fn: Callable[..., Any]) -> dict[str, Any]:
    """Fast value-only mapper for ``fn(value)`` callbacks."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[k] = _map_nested_values(v, fn) if isinstance(v, dict) else fn(v)
    return out


def _map_nested(
    d: dict[str, Any],
    fn: Callable[..., Any],
    *,
    collection: str,
    prefix: tuple[str, ...] = (),
    arity: int,
) -> dict[str, Any]:
    """Return a new nested dict with *fn* applied to every leaf."""
    if arity == 1:
        return _map_nested_values(d, fn)

    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _map_nested(v, fn, collection=collection, prefix=(*prefix, k), arity=arity)
        else:
            out[k] = _call_map_fn(
                fn,
                arity=arity,
                collection=collection,
                path=path_to_str((*prefix, k)),
                value=v,
            )
    return out


class _StateDictProxy(MutableMapping[Any, Any]):
    """Mutable nested mapping view that routes writes through :class:`State`."""

    __slots__ = ("_collection", "_prefix", "_state")

    def __init__(self, state: State, collection: str, prefix: tuple[Any, ...] = ()) -> None:
        """Initialize a proxy view rooted at ``state[collection]`` with optional ``prefix`` path."""
        self._state = state
        self._collection = collection
        self._prefix = prefix

    def _target(self) -> dict[Any, Any]:
        """Resolve the nested dict at ``self._prefix`` inside ``state[collection]``.

        Auto-creates intermediate dicts. Raises :class:`TypeError` if
        any segment of ``prefix`` already names a leaf rather than a
        nested mapping.
        """
        target = self._state._data.setdefault(self._collection, {})
        traversed: list[Any] = []
        for key in self._prefix:
            traversed.append(key)
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"State path {path_to_str(tuple(traversed))!r} is a leaf, not a nested mapping")
            target = child
        return target

    def __getitem__(self, key: Any) -> Any:
        """Return the value at ``prefix + (key,)``; nested dicts return a new proxy view."""
        value = self._target()[key]
        if isinstance(value, dict):
            return type(self)(self._state, self._collection, (*self._prefix, key))
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        """Write ``value`` at ``prefix + (key,)``, routing through the parent ``State``.

        Nested mappings are deep-copied and synced (so any registered
        writers on the leaves run); plain leaves go through
        :meth:`State._sync_leaf` to update both the data dict and any
        writer callbacks.
        """
        path = (*self._prefix, key)
        if isinstance(value, Mapping) and value:
            subtree = _mapping_to_nested_dict(value)
            self._target()[key] = subtree
            _sync_nested(self._state, self._collection, subtree, path)
            self._state._restrict_writers()
            return
        self._state._sync_leaf(self._collection, path, value)

    def __delitem__(self, key: Any) -> None:
        """Remove ``prefix + (key,)`` from the proxied state and prune dead writers."""
        del self._target()[key]
        self._state._restrict_writers()

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the keys at the proxy's current path."""
        return iter(self._target())

    def __len__(self) -> int:
        """Return the number of keys at the proxy's current path."""
        return len(self._target())

    def __repr__(self) -> str:
        """Render as the underlying detached dict — easier to read than internal pointers."""
        return repr(self.as_dict())

    def __eq__(self, other: object) -> bool:
        """Compare by detached snapshot value with another mapping (deep)."""
        if isinstance(other, Mapping):
            return self.as_dict() == _mapping_to_nested_dict(other)
        return False

    def as_dict(self) -> dict[Any, Any]:
        """Return a detached plain-dict snapshot of this view."""
        return _deep_copy_nested(self._target())

    def copy(self) -> dict[Any, Any]:
        """Return a detached plain-dict snapshot, matching ``dict.copy`` ergonomics."""
        return self.as_dict()


def _flat_to_nested(d: dict[str, Any]) -> dict[str, Any]:
    """Convert a dotted-path flat dict into a nested dict."""
    out: dict[str, Any] = {}
    for path_str, value in d.items():
        _nested_set(out, str_to_path(path_str), value)
    return out


def _nested_to_flat(d: dict[str, Any]) -> dict[str, Any]:
    """Convert a nested dict into a dotted-path flat dict."""
    out: dict[str, Any] = {}
    for path_tuple, value in _nested_items(d):
        out[path_to_str(path_tuple)] = value
    return out


def _sort_key(k: str) -> tuple[bool, str]:
    """Sort numeric-looking keys before others (so ``0`` < ``10`` < ``a``)."""
    try:
        return (True, int(k))
    except ValueError:
        return (False, k)


class State:
    """Collection-partitioned, nested-dict state container.

    Use ``state[collection]`` to get the nested dict for a collection.
    :class:`State` is intentionally mutable. Mutation-capable methods
    such as :meth:`filter`, :meth:`exclude`, :meth:`merge`,
    :meth:`map`, and :meth:`set` operate in-place by default and accept
    ``copy=True`` to return a detached snapshot instead.
    """

    __slots__ = ("_data", "_writers")

    _data: dict[str, dict[str, Leaf]]
    _writers: dict[tuple[str, str], Writer] | None

    def __init__(self, data: Mapping[str, Mapping[str, Leaf]] | None = None) -> None:
        """Construct from an optional ``{collection: {path: leaf}}`` mapping.

        The inner mapping may be either a *nested* dict
        (``{"layer": {"weight": arr}}``) or a *flat* dotted-path dict
        (``{"layer.weight": arr}``).  Nested form is preferred and is
        what :meth:`raw` returns.
        """
        if data is None:
            d: dict[str, dict[str, Leaf]] = {}
        else:
            d = {}
            for c, inner in data.items():
                inner_dict = dict(inner)
                is_nested = any(isinstance(v, dict) and v for v in inner_dict.values())
                d[c] = inner_dict if is_nested else _flat_to_nested(inner_dict)
        object.__setattr__(self, "_data", d)
        object.__setattr__(self, "_writers", None)

    @classmethod
    def _from_raw(
        cls,
        data: dict[str, dict[str, Leaf]],
        *,
        writers: dict[tuple[str, str], Writer] | None = None,
    ) -> State:
        """Fast-path constructor that adopts ``data`` without copying.

        Used by the pytree unflattener and other internal hot paths where
        ``data`` is already a freshly allocated nested dict — skipping
        the defensive dict copy that ``__init__`` otherwise performs.
        """
        obj = cls.__new__(cls)
        object.__setattr__(obj, "_data", data)
        object.__setattr__(obj, "_writers", dict(writers) if writers else None)
        return obj

    def copy(self) -> State:
        """Return a detached nested structure with shared immutable leaves."""
        return State._from_raw(_deep_copy_nested(self._data))

    def overlay(self, other: State) -> State:
        """Return ``self`` overlaid with ``other`` without mutating either input.

        This rebuilds only the nested dictionary structure; leaves are shared.
        It is the internal fast path for temporary bind states where we need a
        merged view but do not want mutable ``merge`` write-through semantics.
        """
        data = _deep_copy_nested(self._data)
        _merge_nested(data, other._data)
        return State._from_raw(data)

    def _copy_writers(self) -> dict[tuple[str, str], Writer] | None:
        """Return a shallow copy of the writer map (or ``None`` if none registered)."""
        if self._writers is None:
            return None
        return dict(self._writers)

    def _set_writers(self, writers: dict[tuple[str, str], Writer] | None) -> None:
        """Replace the writer map; an empty/``None`` argument clears it."""
        object.__setattr__(self, "_writers", writers if writers else None)

    def _sync_leaf(self, collection: str, path: tuple[str, ...], value: Leaf) -> None:
        """Write ``value`` at ``data[collection][path]`` and fire any registered writer.

        Used by both the proxy and direct API to keep the nested
        dict and any module-side variable references in sync.
        """
        dotted = path_to_str(path)
        _nested_set(self._data.setdefault(collection, {}), path, value)
        if self._writers is None:
            return
        writer = self._writers.get((collection, dotted))
        if writer is not None:
            writer(value)

    def _restrict_writers(self) -> None:
        """Drop writer entries whose ``(collection, path)`` is no longer in ``self``.

        Called after deletions / replacements so that future writes
        don't dispatch through stale writers.
        """
        if self._writers is None:
            return
        live_keys = {(collection, path) for collection, path in self.paths()}
        self._set_writers({key: writer for key, writer in self._writers.items() if key in live_keys})

    def __getitem__(self, collection: str) -> MutableMapping[Any, Leaf]:
        """Return a mutable nested view for ``collection``, creating it
        on demand if absent so downstream code may index without guards.
        """
        self._data.setdefault(collection, {})
        return _StateDictProxy(self, collection)

    def __setitem__(self, collection: str, value: Mapping[str, Leaf]) -> None:
        """Replace the inner nested dict for ``collection``."""
        replacement = _normalize_inner_mapping(value)
        self._data[collection] = {}
        _sync_nested(self, collection, replacement)
        self._restrict_writers()

    def __contains__(self, collection: object) -> bool:
        """Return ``True`` iff ``collection`` is a non-empty collection name."""
        if not isinstance(collection, str):
            return False
        return collection in self._data and bool(self._data[collection])

    def __iter__(self) -> Iterator[str]:
        """Iterate over collection names."""
        return iter(self._data)

    def __len__(self) -> int:
        """Total number of leaves across all collections."""
        return sum(sum(1 for _ in _nested_paths(d)) for d in self._data.values())

    def collections(self) -> set[str]:
        """Return the set of non-empty collection names."""
        return {c for c, v in self._data.items() if v}

    def raw(self) -> dict[str, dict[str, Leaf]]:
        """Return the backing nested-dict.

        Direct nested-dict mutation bypasses live write-through hooks.
        Prefer :meth:`set`, :meth:`merge`, or :meth:`map` when you want
        live-backed updates to propagate.
        """
        return self._data

    def items(self) -> Iterator[tuple[str, str, Leaf]]:
        """Yield ``(collection, dotted_path, leaf)`` tuples over every leaf."""
        for c, d in self._data.items():
            for path_tuple, v in _nested_items(d):
                yield c, path_to_str(path_tuple), v

    def paths(self, collection: str | None = None) -> list[tuple[str, str]]:
        """Return every ``(collection, dotted_path)`` pair, optionally filtered."""
        if collection is not None:
            return [(collection, path_to_str(p)) for p in _nested_paths(self._data.get(collection, {}))]
        return [(c, path_to_str(p)) for c, d in self._data.items() for p in _nested_paths(d)]

    def filter(self, *collections: str, copy: bool = False) -> State:
        """Keep only the named collections.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.
        """
        if copy:
            filtered = {c: _deep_copy_nested(self._data[c]) for c in collections if c in self._data}
            return State._from_raw(filtered)
        filtered = {c: self._data[c] for c in collections if c in self._data}
        self._data.clear()
        self._data.update(filtered)
        self._restrict_writers()
        return self

    def exclude(self, *collections: str, copy: bool = False) -> State:
        """Drop the named collections.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.
        """
        if copy:
            remaining = {c: _deep_copy_nested(d) for c, d in self._data.items() if c not in collections}
            return State._from_raw(remaining)
        remaining = {c: d for c, d in self._data.items() if c not in collections}
        self._data.clear()
        self._data.update(remaining)
        self._restrict_writers()
        return self

    def merge(self, other: State, *, copy: bool = False) -> State:
        """Merge ``other`` into ``self``.

        Entries in ``other`` win on collision. Mutates ``self`` by
        default; pass ``copy=True`` for a detached merged result.
        """
        target = self.copy() if copy else self
        if target._writers is None:
            _merge_nested(target._data, other._data)
            return target
        for c, path, value in other.items():
            target._sync_leaf(c, str_to_path(path), value)
        return target

    def map(self, fn: Callable[..., Leaf], *collections: str, copy: bool = False) -> State:
        """Apply ``fn`` to every leaf, optionally restricted to some collections.

        ``fn`` may use one of these signatures:

        - ``fn(value)``
        - ``fn(path, value)``
        - ``fn(path, value, collection)``

        where ``path`` is the dotted path within the collection.
        Mutates ``self`` by default; pass ``copy=True`` for a detached result.
        """
        target_collections: set[str] | None = set(collections) if collections else None
        arity = _map_fn_arity(fn)
        target = self.copy() if copy else self
        for c, d in list(target._data.items()):
            if target_collections is not None and c not in target_collections:
                continue
            mapped = _map_nested(d, fn, collection=c, arity=arity)
            target._data[c] = mapped
            if target._writers is not None:
                for path_tuple, value in _nested_items(mapped):
                    writer = target._writers.get((c, path_to_str(path_tuple)))
                    if writer is not None:
                        writer(value)
        return target

    def set(self, collection: str, path: str | Path, value: Leaf, *, copy: bool = False) -> State:
        """Set ``value`` at ``(collection, path)``.

        Mutates ``self`` by default; pass ``copy=True`` for a detached result.
        """
        path_tuple = str_to_path(path) if isinstance(path, str) else path
        target = self.copy() if copy else self
        target._sync_leaf(collection, path_tuple, value)
        return target

    def get(self, collection: str, path: str | Path, default: Any = None) -> Any:
        """Return the leaf at ``(collection, path)`` or ``default`` if missing."""
        path_tuple = str_to_path(path) if isinstance(path, str) else path
        return _nested_get(self._data.get(collection, {}), path_tuple, default)

    def flatten(self) -> dict[str, Leaf]:
        """Return a flat ``{'collection/path': leaf}`` dict."""
        out: dict[str, Leaf] = {}
        for c, d in self._data.items():
            for path_tuple, v in _nested_items(d):
                out[f"{c}/{path_to_str(path_tuple)}"] = v
        return out

    @classmethod
    def from_flat(cls, flat: Mapping[str, Leaf]) -> State:
        """Construct a :class:`State` from the dict produced by :meth:`flatten`.

        Keys are split on the first ``/``.
        """
        out: dict[str, dict[str, Leaf]] = {}
        for key, v in flat.items():
            if "/" not in key:
                raise ValueError(f"from_flat key must be 'collection/path', got {key!r}")
            c, p = key.split("/", 1)
            _nested_set(out.setdefault(c, {}), str_to_path(p), v)
        return cls(out)

    def __repr__(self) -> str:
        """Compact summary: total leaf count and per-collection counts."""
        total = len(self)
        cols = ", ".join(f"{c}={sum(1 for _ in _nested_paths(d))}" for c, d in self._data.items())
        return f"State({total} leaves | {cols})"


_StateAux = tuple[tuple[str, str], ...]


def _state_flatten(s: State) -> tuple[tuple[Any, ...], _StateAux]:
    """Direct leaf flattener for :class:`State`.

    Avoids materializing sorted nested dict copies on the hot path by
    emitting the leaves directly alongside a deterministic
    ``(collection, dotted_path)`` leaf specification.
    """
    leaves: list[Any] = []
    spec: list[tuple[str, str]] = []
    for c, inner in sorted(s._data.items(), key=lambda x: _sort_key(x[0])):
        for path_tuple, v in _nested_items(inner):
            leaves.append(v)
            spec.append((c, path_to_str(path_tuple)))
    return tuple(leaves), tuple(spec)


def _state_flatten_with_keys(
    s: State,
) -> tuple[tuple[tuple[tuple[jax.tree_util.DictKey, ...], Any], ...], _StateAux]:
    """JAX pytree flattener for :class:`State` with per-leaf keypaths.

    Emits explicit :class:`DictKey` tuples for every leaf while sharing
    the same leaf spec as :func:`_state_flatten`.
    """
    key_leaves: list[tuple[tuple[jax.tree_util.DictKey, ...], Any]] = []
    spec: list[tuple[str, str]] = []
    for c, inner in sorted(s._data.items(), key=lambda x: _sort_key(x[0])):
        for path_tuple, v in _nested_items(inner):
            key = (jax.tree_util.DictKey(c), *[jax.tree_util.DictKey(seg) for seg in path_tuple])
            key_leaves.append((key, v))
            spec.append((c, path_to_str(path_tuple)))
    return tuple(key_leaves), tuple(spec)


def _state_unflatten(aux: _StateAux, children: tuple[Any, ...]) -> State:
    """JAX pytree unflattener for the direct-leaf format."""
    if len(aux) != len(children):
        raise ValueError(
            "State pytree leaf count mismatch during unflatten: "
            f"expected {len(aux)} leaves from the auxiliary spec, got {len(children)}."
        )
    data: dict[str, dict[str, Any]] = {}
    for (collection, path), leaf in zip(aux, children, strict=True):
        _nested_set(data.setdefault(collection, {}), str_to_path(path), leaf)
    return State._from_raw(data)


jax.tree_util.register_pytree_with_keys(
    State,
    _state_flatten_with_keys,
    _state_unflatten,
    flatten_func=_state_flatten,
)
