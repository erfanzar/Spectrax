# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module containers: :class:`Sequential`, :class:`ModuleList`,
:class:`StackedModuleList`, :class:`ModuleDict`, :class:`ParameterList`.

Containers are :class:`~spectrax.Module` subclasses that expose their
elements under integer or string keys instead of attribute names. They
override :meth:`_spx_graph_children` so traversal emits those keys
directly, producing paths like ``"blocks.0.fc.weight"``.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, TypeVar, overload

import jax
import jax.numpy as jnp

from .graph import GraphDef, ModuleNode, VarNode, iter_variables
from .module import Module, Opaque, _bump_graph_epoch, _graph_epoch
from .paths import str_to_path
from .registry import resolve_class
from .sharding import Sharding, normalize_sharding
from .stage_assignment import PIPELINE_STAGE_METADATA_KEY
from .state import State, _nested_set
from .static import Static
from .variable import Parameter, Variable, _initialize_value

__all__ = [
    "ModuleDict",
    "ModuleList",
    "ParameterList",
    "Sequential",
    "StackedModuleList",
]


M = TypeVar("M", bound=Module)
P = TypeVar("P", bound=Parameter)


def _stack_module_states(items: list[Module], *, context: str) -> tuple[GraphDef, State]:
    """Export homogeneous modules and stack their states on a leading axis."""
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    exports = [export(m) for m in items]
    gdef = exports[0][0]
    signature = _scan_graph_signature(gdef)
    for index, (other_gdef, _state) in enumerate(exports[1:], start=1):
        if _scan_graph_signature(other_gdef) != signature:
            raise ValueError(
                f"{context} requires every item to have the same graph structure; "
                f"item 0 and item {index} differ. Use a Python loop for heterogeneous layers."
            )
    states = [s for _, s in exports]
    return gdef, jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)


def _stack_module_scan_states(items: list[Module], *, context: str) -> tuple[tuple[GraphDef, ...], State]:
    """Export modules and stack compatible states for scanned execution.

    Repeated neural network blocks often share state topology but carry
    per-layer static fields (for example layer indices, helper closures, or
    placement metadata). ``ModuleList.scan`` can preserve those per-layer
    graph definitions by dispatching the scan body through ``lax.switch``.
    """
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    exports = [export(m) for m in items]
    graph_defs = tuple(g for g, _s in exports)
    topology = _scan_graph_topology_signature(graph_defs[0])
    for index, other_gdef in enumerate(graph_defs[1:], start=1):
        if _scan_graph_topology_signature(other_gdef) != topology:
            raise ValueError(
                f"{context} requires every item to have compatible graph topology; "
                f"item 0 and item {index} differ. Use a Python loop for heterogeneous layers."
            )

    states = [s for _g, s in exports]
    try:
        stacked = jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)
    except Exception as exc:
        raise ValueError(
            f"{context} requires every item to have compatible state structure; "
            "use a Python loop for heterogeneous layers."
        ) from exc
    return graph_defs, stacked


def _scan_graph_signature(gdef: GraphDef) -> GraphDef:
    """Return a graph signature suitable for repeated-layer scans.

    Pipeline stage metadata is intentionally per layer, so it must not make
    homogeneous transformer blocks fail scan/stack checks. Normal export still
    preserves the metadata for sharding and placement.
    """
    nodes = []
    changed = False
    for node in gdef.nodes:
        if isinstance(node, VarNode):
            metadata = tuple((k, v) for k, v in node.metadata if k != PIPELINE_STAGE_METADATA_KEY)
            if metadata != node.metadata:
                changed = True
                node = VarNode(class_name=node.class_name, collection=node.collection, metadata=metadata)
        nodes.append(node)
    if not changed:
        return gdef
    return GraphDef(
        nodes=tuple(nodes),
        root=gdef.root,
        var_refs=gdef.var_refs,
        var_canonical=gdef.var_canonical,
        shared_paths=gdef.shared_paths,
    )


def _scan_graph_topology_signature(gdef: GraphDef) -> tuple[Any, ...]:
    """Return the state/child topology used to decide scan compatibility.

    Static values and opaque object identities are intentionally excluded.
    ``ModuleList.scan`` preserves those values by binding each layer with its
    own graph definition inside a ``lax.switch`` branch.
    """
    nodes: list[Any] = []
    for node in gdef.nodes:
        if isinstance(node, ModuleNode):
            nodes.append(
                (
                    "module",
                    node.class_name,
                    tuple(name for name, _value in node.static_fields),
                    node.children,
                    node.container_kind,
                    tuple(name for name, _value in node.opaque),
                )
            )
        elif isinstance(node, VarNode):
            metadata_keys = tuple(name for name, _value in node.metadata if name != PIPELINE_STAGE_METADATA_KEY)
            nodes.append(("var", node.class_name, node.collection, metadata_keys))
        else:
            nodes.append(node)
    return (
        tuple(nodes),
        gdef.root,
        gdef.var_refs,
        gdef.var_canonical,
        gdef.shared_paths,
    )


def _scan_normalize_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """Normalize static/opaque values for scan template compatibility checks."""
    if _seen is None:
        _seen = set()
    if _depth > 4:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, Opaque):
        return ("opaque", _scan_normalize_value(value.value, _depth=_depth + 1, _seen=_seen))
    if isinstance(value, Static):
        return ("static", _scan_normalize_value(value.value, _depth=_depth + 1, _seen=_seen))
    if isinstance(value, tuple):
        return tuple(_scan_normalize_value(v, _depth=_depth + 1, _seen=_seen) for v in value)
    if isinstance(value, list):
        return ("list", tuple(_scan_normalize_value(v, _depth=_depth + 1, _seen=_seen) for v in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        _scan_normalize_value(k, _depth=_depth + 1, _seen=_seen),
                        _scan_normalize_value(v, _depth=_depth + 1, _seen=_seen),
                    )
                    for k, v in value.items()
                )
            ),
        )
    if isinstance(value, types.FunctionType):
        code = value.__code__
        closure = ()
        if value.__closure__:
            closure = tuple(
                _scan_normalize_value(cell.cell_contents, _depth=_depth + 1, _seen=_seen) for cell in value.__closure__
            )
        return (
            "function",
            value.__module__,
            value.__qualname__,
            code.co_code,
            code.co_consts,
            _scan_normalize_value(value.__defaults__, _depth=_depth + 1, _seen=_seen),
            closure,
        )
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return ("callable", value.__module__, value.__qualname__, repr(value))

    ident = id(value)
    if ident in _seen:
        return ("cycle", type(value).__module__, type(value).__qualname__)
    _seen.add(ident)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value).__module__,
                type(value).__qualname__,
                tuple(
                    (field.name, _scan_normalize_value(getattr(value, field.name), _depth=_depth + 1, _seen=_seen))
                    for field in fields(value)
                ),
            )
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            public_attrs = tuple(
                sorted(
                    (k, _scan_normalize_value(v, _depth=_depth + 1, _seen=_seen))
                    for k, v in attrs.items()
                    if not k.startswith("_")
                )
            )
            return ("object", type(value).__module__, type(value).__qualname__, public_attrs)
        return ("repr", type(value).__module__, type(value).__qualname__, repr(value))
    finally:
        _seen.discard(ident)


_SCAN_SAFE_VALUE = ("spectrax", "scan-safe-field")


@dataclass(frozen=True)
class _ScanSegment:
    """One consecutive run that can be lowered with one graph template."""

    start: int
    stop: int
    gdef: GraphDef
    stacked: State
    family_id: int

    @property
    def length(self) -> int:
        """Number of layers covered by this scan segment (``stop - start``)."""
        return self.stop - self.start


@dataclass(frozen=True)
class _ScanPlan:
    """Internal lowering plan for repeated-layer scans."""

    segments: tuple[_ScanSegment, ...]
    graph_family_ids: tuple[int, ...]
    lowering: str
    fallback_reason: str | None = None


def _scan_class_field_names(class_name: str, attr_name: str) -> frozenset[str]:
    """Return scan metadata field names declared by a module class."""
    try:
        cls = resolve_class(class_name)
    except Exception:
        return frozenset()
    names = getattr(cls, attr_name, ())
    if callable(names):
        names = names()
    return frozenset(names or ())


def _scan_static_field_key(node: ModuleNode, name: str, value: Any) -> Any:
    """Return the graph-family key payload for a static field."""
    safe_fields = _scan_class_field_names(node.class_name, "_spx_scan_safe_static_fields")
    if name in safe_fields:
        return _SCAN_SAFE_VALUE
    return repr(_scan_normalize_value(value))


def _scan_opaque_field_key(node: ModuleNode, name: str, value: Any) -> Any:
    """Return the graph-family key payload for an opaque field."""
    safe_fields = _scan_class_field_names(node.class_name, "_spx_scan_safe_opaque_fields")
    if name in safe_fields:
        return _SCAN_SAFE_VALUE
    return repr(_scan_normalize_value(value))


def _scan_graph_family_key(gdef: GraphDef) -> tuple[Any, ...]:
    """Return a scan graph-family key.

    Equal keys mean the graph definitions can share one scan body template.
    Differing behavior-changing statics intentionally produce different keys;
    statics explicitly marked safe by the module class are ignored.
    """
    nodes: list[Any] = []
    for node in gdef.nodes:
        if isinstance(node, ModuleNode):
            nodes.append(
                (
                    "module",
                    node.class_name,
                    tuple((name, _scan_static_field_key(node, name, value)) for name, value in node.static_fields),
                    node.children,
                    node.container_kind,
                    tuple((name, _scan_opaque_field_key(node, name, value)) for name, value in node.opaque),
                )
            )
        elif isinstance(node, VarNode):
            metadata = tuple((k, v) for k, v in node.metadata if k != PIPELINE_STAGE_METADATA_KEY)
            nodes.append(("var", node.class_name, node.collection, metadata))
        else:
            nodes.append(node)
    return (
        tuple(nodes),
        gdef.root,
        gdef.var_refs,
        gdef.var_canonical,
        gdef.shared_paths,
    )


def _stack_states(states: list[State], *, context: str) -> State:
    """Stack same-structure states on a leading layer axis."""
    try:
        return jax.tree.map(lambda *vs: jnp.stack(vs, axis=0), *states)
    except Exception as exc:
        raise ValueError(
            f"{context} requires every item in a scan segment to have compatible state structure; "
            "use trace=True for heterogeneous Python-side layers."
        ) from exc


def _stage_place_trace_carry(layer: Module, carry: Any) -> Any:
    """Move the leading activation carry onto a layer's stage-local mesh."""
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(carry)):
        return carry

    stage_mesh = None
    for _path, var in iter_variables(layer):
        value = getattr(var, "value", None)
        sharding = getattr(value, "sharding", None)
        if (
            getattr(var, "stage_assignment", None) is not None
            and isinstance(sharding, jax.sharding.NamedSharding)
            and sharding.mesh.devices.size < jax.device_count()
        ):
            stage_mesh = sharding.mesh
            break
    if stage_mesh is not None:
        return _device_put_first_carry_leaf(carry, stage_mesh)

    try:
        from ..sharding.mesh import current_mesh

        mesh = current_mesh()
    except Exception:
        return carry
    if mesh is None or not getattr(mesh, "is_mpmd", False) or mesh.mpmd_mesh.mpmd_dim <= 1:
        return carry

    for _path, var in iter_variables(layer):
        try:
            owner = var.resolved_stage_index(mesh)
        except Exception:
            owner = None
        if owner is not None:
            stage_mesh = mesh.mpmd_mesh.submesh(owner)
            break
    if stage_mesh is None:
        return carry
    return _device_put_first_carry_leaf(carry, stage_mesh)


def _device_put_first_carry_leaf(carry: Any, stage_mesh: Any) -> Any:
    """Place the leading array carry on ``stage_mesh`` while preserving carry shape."""

    def place(value: Any) -> Any:
        """``device_put`` ``value`` onto ``stage_mesh`` (replicated) if it's a JAX array; else passthrough."""
        if isinstance(value, jax.Array):
            return jax.device_put(value, jax.sharding.NamedSharding(stage_mesh, jax.sharding.PartitionSpec()))
        return value

    if isinstance(carry, tuple) and carry:
        return (place(carry[0]), *carry[1:])
    if isinstance(carry, list) and carry:
        return [place(carry[0]), *carry[1:]]
    return place(carry)


def _slice_stacked_state(stacked: State, start: int, stop: int) -> State:
    """Slice a stacked state along its leading layer axis."""
    return jax.tree.map(lambda leaf: leaf[start:stop], stacked)


def _scan_static_template_signature(
    graph_defs: tuple[GraphDef, ...],
    family_keys: tuple[Any, ...] | None = None,
) -> GraphDef | None:
    """Return a reusable graph template when per-layer differences are safe.

    The template path is the fast path: it binds every scanned state slice with
    one graph definition, avoiding per-layer dispatch. Differing static values
    are collapsed only when the module class explicitly marks those fields as
    scan-safe metadata.
    """
    if not graph_defs:
        return None
    if len(graph_defs) == 1:
        return graph_defs[0]
    if family_keys is not None and all(key == family_keys[0] for key in family_keys[1:]):
        return graph_defs[0]
    key = _scan_graph_family_key(graph_defs[0])
    if any(_scan_graph_family_key(g) != key for g in graph_defs[1:]):
        return None
    return graph_defs[0]


def _build_scan_plan_from_exports(exports: list[tuple[GraphDef, State]], *, context: str) -> _ScanPlan:
    """Build a segmented scan plan from exported module states."""
    if not exports:
        raise ValueError(f"{context} requires at least one module")

    family_key_to_id: dict[tuple[Any, ...], int] = {}
    graph_family_ids: list[int] = []
    graph_defs = [gdef for gdef, _state in exports]
    for gdef in graph_defs:
        key = _scan_graph_family_key(gdef)
        family_id = family_key_to_id.setdefault(key, len(family_key_to_id))
        graph_family_ids.append(family_id)

    segments: list[_ScanSegment] = []
    start = 0
    while start < len(exports):
        family_id = graph_family_ids[start]
        stop = start + 1
        while stop < len(exports) and graph_family_ids[stop] == family_id:
            stop += 1
        run_defs = tuple(gdef for gdef, _state in exports[start:stop])
        template = _scan_static_template_signature(run_defs)
        if template is None:
            template = run_defs[0]
        stacked = _stack_states([state for _gdef, state in exports[start:stop]], context=context)
        segments.append(_ScanSegment(start=start, stop=stop, gdef=template, stacked=stacked, family_id=family_id))
        start = stop

    lowering = "single_template" if len(segments) == 1 else "segmented_templates"
    return _ScanPlan(segments=tuple(segments), graph_family_ids=tuple(graph_family_ids), lowering=lowering)


def _scan_plan_cache_key(
    graph_defs: tuple[GraphDef, ...],
    family_keys: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Return a stable key for cached scan segmentation metadata."""
    if family_keys is not None:
        return (_graph_epoch(), len(graph_defs), tuple(hash(key) for key in family_keys))
    return (_graph_epoch(), len(graph_defs), tuple(hash(gdef) for gdef in graph_defs))


def _cache_plan(owner: Module, key: tuple[Any, ...], plan: _ScanPlan) -> None:
    """Store state-free scan segmentation metadata on a container."""
    segment_specs = tuple((segment.start, segment.stop, segment.gdef, segment.family_id) for segment in plan.segments)
    object.__setattr__(
        owner,
        "_spx_scan_plan_cache",
        (key, segment_specs, plan.graph_family_ids, plan.lowering, plan.fallback_reason),
    )


def _cached_plan_metadata(owner: Module, key: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Return cached scan segmentation metadata when it matches ``key``."""
    cache = getattr(owner, "_spx_scan_plan_cache", None)
    if cache is None or cache[0] != key:
        return None
    return cache


def _build_cached_scan_plan_from_exports(
    owner: Module,
    exports: list[tuple[GraphDef, State]],
    *,
    context: str,
) -> _ScanPlan:
    """Build or reuse a segmented scan plan for live ModuleList items."""
    graph_defs = tuple(gdef for gdef, _state in exports)
    key = _scan_plan_cache_key(graph_defs)
    cached = _cached_plan_metadata(owner, key)
    if cached is None:
        plan = _build_scan_plan_from_exports(exports, context=context)
        _cache_plan(owner, key, plan)
        return plan

    _key, segment_specs, graph_family_ids, lowering, fallback_reason = cached
    segments = tuple(
        _ScanSegment(
            start=start,
            stop=stop,
            gdef=gdef,
            stacked=_stack_states([state for _gdef, state in exports[start:stop]], context=context),
            family_id=family_id,
        )
        for start, stop, gdef, family_id in segment_specs
    )
    return _ScanPlan(
        segments=segments,
        graph_family_ids=graph_family_ids,
        lowering=lowering,
        fallback_reason=fallback_reason,
    )


def _build_scan_plan_from_modules(items: list[Module], *, context: str) -> _ScanPlan:
    """Export live modules and build a segmented scan plan."""
    from .graph import export

    if not items:
        raise ValueError(f"{context} requires at least one module")
    return _build_scan_plan_from_exports([export(m) for m in items], context=context)


def _build_scan_plan_from_stacked(
    graph_defs: tuple[GraphDef, ...],
    stacked: State,
    *,
    context: str,
    family_keys: tuple[Any, ...] | None = None,
) -> _ScanPlan:
    """Build a segmented scan plan for pre-stacked leaves."""
    if not graph_defs:
        raise ValueError(f"{context} requires at least one module")
    if family_keys is None:
        family_keys = tuple(_scan_graph_family_key(gdef) for gdef in graph_defs)
    if len(family_keys) != len(graph_defs):
        raise ValueError(f"{context} received mismatched graph/family key counts")
    family_key_to_id: dict[tuple[Any, ...], int] = {}
    graph_family_ids: list[int] = []
    for key in family_keys:
        family_id = family_key_to_id.setdefault(key, len(family_key_to_id))
        graph_family_ids.append(family_id)

    segments: list[_ScanSegment] = []
    start = 0
    while start < len(graph_defs):
        family_id = graph_family_ids[start]
        stop = start + 1
        while stop < len(graph_defs) and graph_family_ids[stop] == family_id:
            stop += 1
        run_defs = graph_defs[start:stop]
        run_family_keys = family_keys[start:stop]
        template = _scan_static_template_signature(run_defs, run_family_keys)
        if template is None:
            template = run_defs[0]
        segments.append(
            _ScanSegment(
                start=start,
                stop=stop,
                gdef=template,
                stacked=_slice_stacked_state(stacked, start, stop),
                family_id=family_id,
            )
        )
        start = stop

    lowering = "single_template" if len(segments) == 1 else "segmented_templates"
    return _ScanPlan(segments=tuple(segments), graph_family_ids=tuple(graph_family_ids), lowering=lowering)


def _build_cached_scan_plan_from_stacked(
    owner: Module,
    graph_defs: tuple[GraphDef, ...],
    stacked: State,
    *,
    context: str,
) -> _ScanPlan:
    """Build or reuse a segmented scan plan for pre-stacked leaves."""
    family_keys = getattr(owner, "_spx_item_family_keys", None)
    key = _scan_plan_cache_key(graph_defs, family_keys)
    cached = _cached_plan_metadata(owner, key)
    if cached is None:
        plan = _build_scan_plan_from_stacked(graph_defs, stacked, context=context, family_keys=family_keys)
        _cache_plan(owner, key, plan)
        return plan

    _key, segment_specs, graph_family_ids, lowering, fallback_reason = cached
    segments = tuple(
        _ScanSegment(
            start=start,
            stop=stop,
            gdef=gdef,
            stacked=_slice_stacked_state(stacked, start, stop),
            family_id=family_id,
        )
        for start, stop, gdef, family_id in segment_specs
    )
    return _ScanPlan(
        segments=segments,
        graph_family_ids=graph_family_ids,
        lowering=lowering,
        fallback_reason=fallback_reason,
    )


def _scan_effective_unroll(unroll: int | None, length: int) -> int:
    """Resolve explicit ``jax.lax.scan`` unroll values."""
    if length <= 0:
        return 1
    if unroll is None:
        return _scan_default_unroll(length)
    unroll_value = int(unroll)
    if unroll_value < 0:
        raise ValueError(f"scan unroll must be >= 0, got {unroll}.")
    return unroll_value


def _scan_default_unroll(_length: int) -> int:
    """Choose the compile-oriented default unroll for real scans."""
    return 1


def _scan_constraint_for_metadata(metadata: dict[str, Any]) -> Any:
    """Resolve per-layer variable metadata to a scan-body sharding constraint."""
    from ..sharding.mesh import current_mesh
    from ..sharding.partition import named_sharding_for_metadata

    mesh = current_mesh()
    if mesh is not None:
        return named_sharding_for_metadata(metadata, mesh)
    return None


def _scan_state_constraint_specs(gdef: GraphDef) -> State | None:
    """Return a State-shaped tree of per-layer sharding constraints."""
    data: dict[str, dict[str, Any]] = {}
    canonical: dict[int, str] = dict(gdef.var_canonical)
    seen_refs: set[int] = set()
    for node_idx, local_ref_id in gdef.var_refs:
        if local_ref_id in seen_refs:
            continue
        seen_refs.add(local_ref_id)
        node = gdef.nodes[node_idx]
        if not isinstance(node, VarNode):
            continue
        constraint = _scan_constraint_for_metadata(dict(node.metadata))
        if constraint is None:
            continue
        _nested_set(data.setdefault(node.collection, {}), str_to_path(canonical[local_ref_id]), constraint)
    if not data:
        return None
    return State._from_raw(data)


def _apply_nested_constraints(values: dict[str, Any], specs: dict[str, Any]) -> dict[str, Any]:
    """Apply sharding constraints to a nested state collection."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        spec = specs.get(key) if isinstance(specs, dict) else None
        if isinstance(value, dict):
            out[key] = _apply_nested_constraints(value, spec if isinstance(spec, dict) else {})
        elif spec is not None:
            out[key] = jax.lax.with_sharding_constraint(value, spec)
        else:
            out[key] = value
    return out


def _apply_scan_state_constraints(state: State, specs: State | None) -> State:
    """Apply State-shaped scan-body sharding constraints."""
    if specs is None:
        return state
    constrained: dict[str, dict[str, Any]] = {}
    spec_raw = specs.raw()
    for collection, values in state.raw().items():
        constrained[collection] = _apply_nested_constraints(values, spec_raw.get(collection, {}))
    return State._from_raw(constrained)


def _scan_segment_with_explicit_unroll(segment: _ScanSegment, fn, carry, bind, unroll: int | None):
    """Run one segment with the direct layer-wise ``lax.scan`` lowering."""
    gdef = segment.gdef
    effective_unroll = _scan_effective_unroll(unroll, segment.length)
    constraint_specs = _scan_state_constraint_specs(gdef)

    def body(carry, layer_state, *, gdef=gdef):
        """``lax.scan`` body: bind a layer's state to the segment graphdef and apply ``fn``."""
        layer_state = _apply_scan_state_constraints(layer_state, constraint_specs)
        live = bind(gdef, layer_state)
        return fn(live, carry), None

    return jax.lax.scan(
        body,
        carry,
        segment.stacked,
        unroll=effective_unroll,
    )[0]


class _ListContainer(Module):
    """Shared implementation for list-shaped containers.

    Stores elements in :attr:`_spx_items` and exposes ``__len__`` /
    ``__getitem__`` / ``__iter__`` / ``append`` / ``extend``. Subclasses
    override :meth:`_validate_item` to enforce an element type, and
    inherit :meth:`_spx_graph_children` which yields integer-keyed
    children.
    """

    _spx_items: list[Any]

    def __init__(self, items: Iterable[Any] = ()) -> None:
        """Construct the container from an iterable of items.

        Items are validated one by one via :meth:`_validate_item` before
        being stored.
        """
        super().__init__()
        materialized = list(items)
        for item in materialized:
            self._validate_item(item)
        object.__setattr__(self, "_spx_items", materialized)

    def __len__(self) -> int:
        """Number of stored elements."""
        return len(self._spx_items)

    @overload
    def __getitem__(self, idx: int) -> Any:
        """Overload: integer index returns one stored element."""
        ...

    @overload
    def __getitem__(self, idx: slice) -> _ListContainer:
        """Overload: slice returns a new container of the same concrete type."""
        ...

    def __getitem__(self, idx: int | slice) -> Any:
        """Index or slice into the container.

        Integer indices return the single element. Slices return a new
        container of the same concrete type.
        """
        if isinstance(idx, slice):
            return type(self)(self._spx_items[idx])
        return self._spx_items[idx]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over stored elements."""
        return iter(self._spx_items)

    def append(self, value: Any) -> None:
        """Append ``value``, validating its type first."""
        self._validate_item(value)
        self._spx_items.append(value)
        object.__setattr__(self, "_spx_scan_plan_cache", None)
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def extend(self, values: Iterable[Any]) -> None:
        """Append every item from ``values``."""
        for v in values:
            self.append(v)

    def _validate_item(self, value: Any) -> None:
        """Raise :class:`TypeError` if ``value`` is of an unacceptable type."""
        raise NotImplementedError

    def _spx_graph_children(self) -> Iterator[tuple[int, Module | Variable]]:
        """Yield ``(index, child)`` for every Module/Variable in the list."""
        for i, it in enumerate(self._spx_items):
            if isinstance(it, Module | Variable):
                yield i, it

    def _spx_static_fields(self) -> dict[str, Any]:
        """Containers have no static fields."""
        return {}


class ModuleList(_ListContainer):
    """Ordered list of :class:`~spectrax.Module` s, indexable by integer.

    Not callable; use it as a plain Python container iterated or indexed
    by the owning module.

    When indexed with a JAX tracer (e.g. inside ``spx.fori_loop`` or
    ``jax.lax.scan``), the container transparently exports all modules,
    stacks their states, slices the requested index, and returns a live
    bound module. This makes patterns like::

        def body(i, m, x):
            return m.blocks[i](x)

    work inside compiled transforms without materialising constants.
    """

    _spx_container_kind: ClassVar[str] = "list"

    def __init__(self, items: Iterable[Module] = ()) -> None:
        """Construct from an iterable of modules."""
        super().__init__(items)

    def _validate_item(self, value: Any) -> None:
        """Require every item to be a :class:`~spectrax.Module`."""
        if not isinstance(value, Module):
            raise TypeError(f"ModuleList accepts Modules only, got {type(value).__name__}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Always raises — :class:`ModuleList` is not a callable layer."""
        raise RuntimeError("ModuleList is not callable; iterate or index it.")

    def __getitem__(self, idx: int | slice) -> Any:
        """Index/slice with extra support for tracer indices (used inside ``jit``/``scan``).

        Concrete integers and slices behave like a normal Python list.
        A non-concrete tracer index dispatches to :meth:`_get_traced`,
        which exports all child modules, stacks their states, and
        slices into the stack — letting layer code write
        ``self.blocks[i]`` inside a :func:`spectrax.fori_loop`.
        """
        if isinstance(idx, slice):
            return type(self)(self._spx_items[idx])
        if not jax.core.is_concrete(idx):
            return self._get_traced(idx)
        return self._spx_items[idx]

    def _get_traced(self, idx: Any) -> Module:
        """Return the module at tracer index ``idx`` via export/stack/bind."""
        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            gdef, stacked = cache
            layer_state = jax.tree.map(lambda leaf: leaf[idx], stacked)
            return bind(gdef, layer_state)

        gdef, stacked = _stack_module_states(self._spx_items, context="ModuleList traced indexing")
        layer_state = jax.tree.map(lambda leaf: leaf[idx], stacked)
        return bind(gdef, layer_state)

    def scan(self, fn, init_carry, *, trace: bool = False, unroll: int | None = None):
        """Scan over modules: ``fn(module, carry) -> new_carry``.

        When ``trace`` is true, executes a normal Python loop over live
        modules. When ``trace`` is false, runs a real ``jax.lax.scan`` over the
        repeated layer state. ``unroll=None`` selects SpectraX's default
        unroll (``1``); pass an explicit value to override it.
        """
        if trace:
            carry = init_carry
            for layer in self:
                carry = _stage_place_trace_carry(layer, carry)
                carry = fn(layer, carry)
            return carry

        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            plan = _build_cached_scan_plan_from_stacked(self, (cache[0],), cache[1], context="ModuleList.scan")
        else:
            from .graph import export

            exports = [export(m) for m in self._spx_items]
            plan = _build_cached_scan_plan_from_exports(self, exports, context="ModuleList.scan")

        carry = init_carry
        for segment in plan.segments:
            carry = _scan_segment_with_explicit_unroll(segment, fn, carry, bind, unroll)
        return carry

    def stack(self) -> StackedModuleList:
        """Return a stacked view optimized for repeated-layer scans.

        The returned :class:`StackedModuleList` stores each variable with
        a leading layer axis, so ``.scan(...)`` does not need to build
        ``jnp.stack`` operations inside the compiled forward pass. Items
        must share compatible state topology; safe per-layer static
        differences such as ``layer_idx`` are collapsed into one template
        graph.
        """
        return StackedModuleList(self._spx_items)

    def as_stacked(self) -> StackedModuleList:
        """Alias for :meth:`stack` for call sites that prefer adjective naming."""
        return self.stack()

    def fori_loop(self, fn, init_carry):
        """fori_loop over modules: ``fn(i, module, carry) -> new_carry``.

        Exports all modules, stacks their states, and runs
        ``jax.lax.fori_loop`` with pytree-level slicing.
        """
        from .graph import bind

        cache = getattr(self, "_spx_traced_cache", None)
        if cache is not None:
            gdef, stacked = cache
        else:
            gdef, stacked = _stack_module_states(self._spx_items, context="ModuleList.fori_loop")

        def body(i, carry):
            """``fori_loop`` body: bind the i-th layer's state and apply ``fn(i, layer, carry)``."""
            layer_state = jax.tree.map(lambda leaf: leaf[i], stacked)
            live = bind(gdef, layer_state)
            return fn(i, live, carry)

        return jax.lax.fori_loop(0, len(self), body, init_carry)


def _prepend_stacked_axis_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Adjust variable metadata after adding a leading layer axis."""
    out = dict(metadata)
    if "axis_names" in out:
        out["axis_names"] = (None, *tuple(out["axis_names"]))
    sharding = normalize_sharding(out.get("sharding"))
    if sharding is not None:
        if sharding.axis_names is not None:
            out["sharding"] = Sharding(axis_names=(None, *tuple(sharding.axis_names)))
        elif sharding.mesh_axes is not None:
            out["sharding"] = Sharding(mesh_axes=(None, *tuple(sharding.mesh_axes)))
    return out


def _stacked_variable_like(template: Variable, value: Any) -> Variable:
    """Create a variable of ``template``'s class for a stacked leaf value."""
    cls = type(template)
    var = cls.__new__(cls)
    metadata = _prepend_stacked_axis_metadata(template.metadata)
    value = _initialize_value(value, None, metadata=metadata, explicit_sharding="sharding" in metadata)
    Variable.__init__(
        var,
        value,
        kind=template.kind,
        metadata=metadata,
    )
    return var


class StackedModuleList(Module):
    """Homogeneous repeated-layer container with stacked variable leaves.

    ``ModuleList.scan`` is ergonomic but must stack per-layer leaves inside
    the traced function when the owning model is passed as a normal pytree.
    This container pays that stacking cost once at construction time and
    exposes the stacked leaves as the model state. It is intended for
    transformer-style blocks that share an identical graph definition.
    """

    _spx_container_kind: ClassVar[str] = "module"

    def __init__(self, items: Iterable[Module] = ()) -> None:
        """Eagerly stack the variable leaves of each item along a new layer axis.

        On construction the container exports each child module, asserts
        every export shares a compatible graph definition, and stacks
        the per-layer leaves into a single state pytree. After this
        the container holds *one* graphdef plus a state shaped
        ``[L, ...]`` per leaf, eliminating the per-traced-call stack
        cost paid by :class:`ModuleList`.

        Empty constructions defer the metadata until the first
        ``append``, so ``StackedModuleList()`` is cheap.
        """
        super().__init__()
        materialized = list(items)
        if not materialized:
            object.__setattr__(self, "_spx_length", 0)
            object.__setattr__(self, "_spx_item_gdef", None)
            object.__setattr__(self, "_spx_item_gdefs", ())
            object.__setattr__(self, "_spx_item_family_keys", ())
            object.__setattr__(self, "_spx_leaf_specs", ())
            return
        for item in materialized:
            if not isinstance(item, Module):
                raise TypeError(f"StackedModuleList accepts Modules only, got {type(item).__name__}")

        from .graph import export

        exports = [export(m) for m in materialized]
        graph_defs = tuple(gdef for gdef, _state in exports)
        topology = _scan_graph_topology_signature(graph_defs[0])
        if any(_scan_graph_topology_signature(gdef) != topology for gdef in graph_defs[1:]):
            raise ValueError(
                "StackedModuleList requires every item to have compatible graph topology. "
                "Use ModuleList for heterogeneous layers or remove behavior-changing per-layer static differences."
            )
        item_gdef = _scan_static_template_signature(graph_defs)

        states = [state for _gdef, state in exports]
        stacked = _stack_states(states, context="StackedModuleList")
        first_cache = materialized[0]._spx_export_cache
        if first_cache is None:
            export(materialized[0])
            first_cache = materialized[0]._spx_export_cache
        assert first_cache is not None
        templates = first_cache[7] if len(first_cache) >= 8 else first_cache[2]
        leaf_specs = tuple((collection, path) for collection, path, _var in templates)

        object.__setattr__(self, "_spx_length", len(materialized))
        object.__setattr__(self, "_spx_item_gdef", item_gdef)
        object.__setattr__(self, "_spx_item_gdefs", graph_defs)
        object.__setattr__(self, "_spx_item_family_keys", tuple(_scan_graph_family_key(gdef) for gdef in graph_defs))
        object.__setattr__(self, "_spx_leaf_specs", leaf_specs)

        for i, (collection, path, template_var) in enumerate(templates):
            value = stacked.get(collection, path)
            object.__setattr__(self, f"v{i}", _stacked_variable_like(template_var, value))

    def __len__(self) -> int:
        """Number of stacked modules."""
        return int(self._spx_length)

    def __iter__(self) -> Iterator[Module]:
        """Iterate by materializing read-only layer views."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int | slice) -> Any:
        """Return one layer view or a sliced stacked container."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return ModuleList([self[i] for i in indices]).stack()
        return self._bind_index(idx)

    def _spx_static_fields(self) -> dict[str, Any]:
        """Persist the item graph and original leaf paths through bind."""
        return {
            "_spx_item_gdef": self._spx_item_gdef,
            "_spx_item_gdefs": self._spx_item_gdefs,
            "_spx_item_family_keys": self._spx_item_family_keys,
            "_spx_leaf_specs": self._spx_leaf_specs,
            "_spx_length": self._spx_length,
        }

    def _spx_graph_children(self) -> Iterator[tuple[str, Module | Variable]]:
        """Yield stacked leaf variables in deterministic order."""
        for i in range(len(self._spx_leaf_specs)):
            name = f"v{i}"
            if hasattr(self, name):
                yield name, getattr(self, name)

    def _spx_delete_graph_children(self, names: Iterable[str | int]) -> None:
        """Remove stacked leaf variables while keeping the leaf table dense."""
        remove: set[int] = set()
        for name in names:
            if isinstance(name, str) and name.startswith("v"):
                try:
                    remove.add(int(name[1:]))
                except ValueError:
                    continue
            elif isinstance(name, int):
                remove.add(name)
        if not remove:
            return

        old_specs = tuple(self._spx_leaf_specs)
        keep_specs = []
        keep_vars = []
        for i, spec in enumerate(old_specs):
            attr = f"v{i}"
            if i in remove:
                if hasattr(self, attr):
                    object.__delattr__(self, attr)
                continue
            if hasattr(self, attr):
                keep_specs.append(spec)
                keep_vars.append(getattr(self, attr))

        for i in range(len(old_specs)):
            attr = f"v{i}"
            if hasattr(self, attr):
                object.__delattr__(self, attr)
        for i, var in enumerate(keep_vars):
            object.__setattr__(self, f"v{i}", var)
        object.__setattr__(self, "_spx_leaf_specs", tuple(keep_specs))
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def _stacked_state(self) -> State:
        """Rebuild the per-item stacked state expected by ``bind``."""
        data: dict[str, dict[str, Any]] = {}
        for i, (collection, path) in enumerate(self._spx_leaf_specs):
            var = getattr(self, f"v{i}")
            _nested_set(data.setdefault(collection, {}), str_to_path(path), var.value)
        return State._from_raw(data)

    def _bind_index(self, idx: Any) -> Module:
        """Bind the module at ``idx`` from the stacked state."""
        from .graph import bind

        graph_defs = getattr(self, "_spx_item_gdefs", ())
        if not graph_defs:
            raise IndexError("Cannot index an empty StackedModuleList")
        if jax.core.is_concrete(idx):
            gdef = graph_defs[int(idx)]
        elif self._spx_item_gdef is not None:
            gdef = self._spx_item_gdef
        else:
            raise TypeError(
                "Cannot tracer-index a multi-graph StackedModuleList; use .scan(..., trace=False) "
                "or trace=True for Python debugging."
            )
        state = jax.tree.map(lambda leaf: leaf[idx], self._stacked_state())
        return bind(gdef, state)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Always raises — :class:`StackedModuleList` is not callable."""
        raise RuntimeError("StackedModuleList is not callable; iterate, index, or scan it.")

    def scan(self, fn, init_carry, *, trace: bool = False, unroll: int | None = None):
        """Scan over pre-stacked modules: ``fn(module, carry) -> new_carry``."""
        if trace:
            carry = init_carry
            for layer in self:
                carry = _stage_place_trace_carry(layer, carry)
                carry = fn(layer, carry)
            return carry

        from .graph import bind

        graph_defs = getattr(self, "_spx_item_gdefs", ())
        if not graph_defs:
            return init_carry
        stacked = self._stacked_state()
        plan = _build_cached_scan_plan_from_stacked(self, graph_defs, stacked, context="StackedModuleList.scan")

        carry = init_carry
        for segment in plan.segments:
            carry = _scan_segment_with_explicit_unroll(segment, fn, carry, bind, unroll)
        return carry

    def fori_loop(self, fn, init_carry):
        """fori_loop over pre-stacked modules."""
        from .graph import bind

        if not getattr(self, "_spx_item_gdefs", ()):
            return init_carry
        if self._spx_item_gdef is None:
            raise TypeError("StackedModuleList.fori_loop requires a single scan-compatible graph template.")
        stacked = self._stacked_state()
        gdef: GraphDef = self._spx_item_gdef

        def body(i, carry):
            """``fori_loop`` body: bind the i-th layer's state and apply ``fn(i, layer, carry)``."""
            layer_state = jax.tree.map(lambda leaf: leaf[i], stacked)
            live = bind(gdef, layer_state)
            return fn(i, live, carry)

        return jax.lax.fori_loop(0, len(self), body, init_carry)


class Sequential(_ListContainer):
    """Callable chain of modules: output of one is input of the next.

    Forwards ``**kwargs`` through the chain; if a child's ``forward``
    does not accept them the call falls back to a positional-only
    invocation.
    """

    _spx_container_kind: ClassVar[str] = "sequential"

    def __init__(self, *modules: Module) -> None:
        """Construct from positional modules."""
        super().__init__(modules)

    def _validate_item(self, value: Any) -> None:
        """Require every item to be a :class:`~spectrax.Module`."""
        if not isinstance(value, Module):
            raise TypeError(f"Sequential accepts Modules only, got {type(value).__name__}")

    def forward(self, x: Any, **kwargs: Any) -> Any:
        """Thread ``x`` through the chain, passing ``**kwargs`` where accepted."""
        for m in self._spx_items:
            try:
                x = m(x, **kwargs)
            except TypeError:
                x = m(x)
        return x


class ParameterList(_ListContainer):
    """Ordered list of :class:`~spectrax.Parameter` s."""

    _spx_container_kind: ClassVar[str] = "list"

    def __init__(self, items: Iterable[Parameter] = ()) -> None:
        """Construct from an iterable of parameters."""
        super().__init__(items)

    def _validate_item(self, value: Any) -> None:
        """Require every item to be a :class:`~spectrax.Parameter`."""
        if not isinstance(value, Parameter):
            raise TypeError(f"ParameterList accepts Parameters only, got {type(value).__name__}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Always raises — :class:`ParameterList` is not a callable layer."""
        raise RuntimeError("ParameterList is not callable.")


class ModuleDict(Module):
    """String-keyed dict of :class:`~spectrax.Module` s.

    Keys are plain Python strings; iteration preserves insertion order.
    Not callable; access children by key.
    """

    _spx_container_kind: ClassVar[str] = "dict"

    _spx_items: dict[str, Module]

    def __init__(self, items: Mapping[str, Module] | None = None) -> None:
        """Construct from an optional ``{name: module}`` mapping."""
        super().__init__()
        object.__setattr__(self, "_spx_items", {})
        if items:
            for k, v in items.items():
                self[k] = v

    def __setitem__(self, key: str, value: Module) -> None:
        """Assign ``value`` under ``key``, validating both types."""
        if not isinstance(value, Module):
            raise TypeError(f"ModuleDict accepts Modules only, got {type(value).__name__}")
        if not isinstance(key, str):
            raise TypeError(f"ModuleDict keys must be str, got {type(key).__name__}")
        self._spx_items[key] = value
        object.__setattr__(self, "_spx_export_cache", None)
        _bump_graph_epoch()

    def __getitem__(self, key: str) -> Module:
        """Return the module stored under ``key``."""
        return self._spx_items[key]

    def __contains__(self, key: object) -> bool:
        """Membership test by key."""
        return key in self._spx_items

    def __len__(self) -> int:
        """Number of stored entries."""
        return len(self._spx_items)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._spx_items)

    def keys(self) -> Iterable[str]:
        """Return the dict's keys."""
        return self._spx_items.keys()

    def values(self) -> Iterable[Module]:
        """Return the dict's values."""
        return self._spx_items.values()

    def items(self) -> Iterable[tuple[str, Module]]:
        """Return the dict's ``(key, value)`` pairs."""
        return self._spx_items.items()

    def _spx_graph_children(self) -> Iterator[tuple[str, Module | Variable]]:
        """Yield ``(key, child)`` for every entry in insertion order."""
        yield from self._spx_items.items()

    def _spx_static_fields(self) -> dict[str, Any]:
        """Containers have no static fields."""
        return {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Always raises — :class:`ModuleDict` is not a callable layer."""
        raise RuntimeError("ModuleDict is not callable; index by key.")
