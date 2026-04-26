# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Derive JAX :class:`PartitionSpec` / :class:`NamedSharding` trees from modules.

This module is the bridge between SpectraX's internal sharding
metadata (carried on :class:`~spectrax.core.variable.Variable` and
:class:`~spectrax.core.module.Module`) and the JAX sharding APIs
(:class:`jax.sharding.PartitionSpec`, :class:`jax.sharding.NamedSharding`,
:func:`jax.lax.with_sharding_constraint`).

It handles four distinct responsibilities:

1. **Spec extraction** — :func:`get_partition_spec`,
   :func:`get_named_sharding`, :func:`extract_shardings`,
   :func:`extract_sharding_structure` walk a module's variables,
   read their ``axis_names``/``sharding`` metadata, and produce
   a tree of ``PartitionSpec`` or ``NamedSharding`` objects.

2. **Constraint application** — :func:`with_sharding_constraint`
   and :func:`lax_reshard` wrap :func:`jax.lax.with_sharding_constraint`
   with MPMD awareness: on an :class:`~spectrax.runtime.types.mesh.MpMdMesh`
   the constraint is targeted at the *stage-local* sub-mesh, never
   the full pipeline mesh, so a constraint on rank-2 doesn't try to
   shard onto ranks 0/1's devices.

3. **Spec sanitization** — :func:`sanitize_partition_spec_for_mesh_and_shape`
   and :func:`get_corrected_named_sharding` drop mesh axes that
   don't exist on the active mesh or that don't divide the tensor's
   shape evenly, so the same spec can be reused across meshes of
   different sizes without per-call validation.

4. **Pattern matching / utility** — :func:`match_partition_rules`
   resolves regex-keyed partition rules against tree paths;
   :func:`make_shard_and_gather_fns` builds parallel shard/gather
   pytrees for use in checkpointing.

The MPMD-awareness is the most subtle piece: every constraint path
threads ``stage`` / ``stage_mesh`` / ``metadata`` arguments through
:func:`_resolve_constraint_target`, which picks the right sub-mesh
based on (in priority order) the explicit args, the variable's
``stage`` metadata, the active ``assign_stage(...)`` context, the
array's existing sharding, and finally the current process's MPMD
rank. When no stage can be resolved, the constraint is a no-op
rather than a silent miscompile.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from ..common_types import NOT_GIVEN
from ..core._typing import Array, ArrayLike, Initializer
from ..core.module import Module
from ..core.selector import select
from ..core.sharding import AxisNames, Sharding, _resolve_axis_names, normalize_sharding
from ..core.stage_assignment import current_stage_assignment, metadata_stage_assignment, resolve_stage_rank
from ..core.variable import Variable
from .logical import current_axis_rules
from .mesh import SpxMesh

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from ..runtime.types.mesh import MpMdMesh

__all__ = [
    "apply_logical_sharding",
    "extract_sharding_structure",
    "extract_shardings",
    "get_axes_size_in_mesh",
    "get_corrected_named_sharding",
    "get_incontext_mesh",
    "get_named_sharding",
    "get_partition_spec",
    "lax_reshard",
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "named_sharding_for_metadata",
    "named_sharding_for_variable",
    "names_in_current_mesh",
    "sanitize_partition_spec_for_mesh_and_shape",
    "to_jax_mesh",
    "with_partitioning",
    "with_sharding_constraint",
    "with_sharding_constraint_by_name",
]


def with_partitioning(
    init: Initializer,
    axis_names: AxisNames | Sharding,
) -> Callable[..., Array]:
    """Wrap an initializer so its output carries ``axis_names`` metadata.

    The returned callable has the same ``(key, shape, dtype)`` signature
    as the wrapped initializer but additionally returns arrays whose
    sharding metadata has been pre-stamped via the companion
    ``_spx_sharding`` attribute (consumed by constructors that don't
    take a ``sharding=`` keyword but still want to record it). The
    same attribute is also set on the wrapper itself so callers can
    introspect the configured sharding without invoking the initializer.

    In practice the cleanest use is::

        w_init = with_partitioning(normal(0.02), ("embed", "features"))
    """
    sharding = normalize_sharding(axis_names)

    def wrapped(key: Any, shape: tuple[int, ...], dtype: Any) -> Array:
        """Run the wrapped initializer and stamp its output with sharding metadata."""
        arr = init(key, shape, dtype)
        try:
            object.__setattr__(arr, "_spx_sharding", sharding)
        except (AttributeError, TypeError):
            pass
        return arr

    wrapped._spx_sharding = sharding
    return wrapped


def _iter_variables(module: Module):
    """Yield ``(path, var)`` for every variable in ``module`` in graph order."""
    yield from select().apply(module)


def get_partition_spec(module: Module) -> dict[str, dict[str, PartitionSpec | None]]:
    """Return ``{collection: {path: PartitionSpec}}`` derived from axis_names.

    Any variable without ``axis_names`` metadata maps to a fully
    replicated :class:`PartitionSpec`. Resolution honors the
    currently-active :func:`~spectrax.sharding.logical_axis_rules`.
    """
    rules = current_axis_rules()
    out: dict[str, dict[str, PartitionSpec | None]] = {}
    for p, v in _iter_variables(module):
        col = v.kind
        sharding: Sharding | None = v.metadata.get("sharding")
        if sharding is None:
            names = v.metadata.get("axis_names")
            sharding = Sharding(axis_names=tuple(names)) if names is not None else Sharding()
        spec = sharding.to_partition_spec(rules or None)
        out.setdefault(col, {})[p] = spec
    return out


def _resolve_named_sharding_mesh(mesh: "Mesh | SpxMesh | MpMdMesh") -> tuple["Mesh", "MpMdMesh | None"]:
    """Return ``(base_mesh, mpmd_mesh_or_none)`` for sharding resolution."""
    from ..runtime.types.mesh import MpMdMesh

    if isinstance(mesh, SpxMesh):
        return mesh.jax_mesh, mesh.mpmd_mesh
    if isinstance(mesh, MpMdMesh):
        return mesh.jax_mesh, mesh
    return mesh, None


def to_jax_mesh(mesh: "Mesh | SpxMesh | MpMdMesh | None") -> "Mesh | None":
    """Return the underlying JAX mesh for SpectraX mesh wrappers.

    Public JAX APIs such as ``shard_map`` and ``NamedSharding`` require
    ``jax.sharding.Mesh``. SpectraX APIs prefer ``SpxMesh`` because it
    carries MPMD metadata, so boundary code should call this helper
    instead of touching ``.jax_mesh`` manually.
    """
    from ..runtime.types.mesh import MpMdMesh

    if mesh is None:
        return None
    if isinstance(mesh, SpxMesh):
        return mesh.jax_mesh
    if isinstance(mesh, MpMdMesh):
        return mesh.jax_mesh
    return mesh


def _physical_mesh_or_none() -> "Mesh | None":
    """Return the JAX physical mesh from thread-local pxla state, or ``None``.

    Reads ``jax.interpreters.pxla.thread_resources.env.physical_mesh``
    and converts the empty-mesh sentinel into a plain ``None`` so
    callers can do a simple null check.
    """
    mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
    if getattr(mesh, "empty", False):
        return None
    return mesh


def get_incontext_mesh(raise_error: bool = True) -> "SpxMesh | None":
    """Return the active mesh as :class:`SpxMesh`, if one is available.

    This is the SpectraX-facing helper and intentionally preserves
    SpectraX mesh metadata such as ``mpmd_axis``. Use
    :func:`to_jax_mesh` at explicit JAX API boundaries that require a
    raw :class:`jax.sharding.Mesh`.
    """
    from .mesh import _wrap_spx, current_mesh

    spx_mesh = current_mesh()
    if spx_mesh is not None:
        return spx_mesh
    mesh = _physical_mesh_or_none()
    if mesh is None:
        if raise_error:
            raise ValueError("No active JAX/SpectraX mesh context is available.")
        return None
    return _wrap_spx(mesh, None)


def _mpmd_axis_name(mesh: "Mesh | SpxMesh | MpMdMesh | None") -> str | None:
    """Return the pipeline-axis name on an MPMD-aware mesh, else ``None``.

    For :class:`SpxMesh` and :class:`MpMdMesh` the name comes from the
    mesh's ``mpmd_axis`` / ``mpmd_axis_name`` attribute. Plain JAX
    meshes have no MPMD axis and return ``None``.
    """
    from ..runtime.types.mesh import MpMdMesh

    if isinstance(mesh, SpxMesh):
        return mesh.mpmd_axis
    if isinstance(mesh, MpMdMesh):
        return mesh.mpmd_axis_name
    return None


def _drop_axis_name(axis: Any, name: str) -> Any:
    """Drop ``name`` from a single axis spec, collapsing trivial tuples.

    Used to strip the pipeline axis out of a per-dimension spec so the
    sharding does not constrain the MPMD axis. Handles three cases:
    a bare string equal to ``name`` (returns ``None``), a tuple of
    strings (filters out ``name``; returns the lone survivor or
    ``None`` for an empty result), or anything else (returned unchanged).
    """
    if axis == name:
        return None
    if isinstance(axis, tuple):
        kept = tuple(part for part in axis if part != name)
        if not kept:
            return None
        if len(kept) == 1:
            return kept[0]
        return kept
    return axis


def _mesh_axis_names(mesh: "Mesh | SpxMesh | MpMdMesh | None") -> set[str] | None:
    """Return the set of axis names on the underlying JAX mesh.

    Unwraps :class:`SpxMesh` / :class:`MpMdMesh` via their ``jax_mesh``
    attribute. Returns ``None`` when ``mesh`` is ``None`` or has no
    ``axis_names`` attribute.
    """
    if mesh is None:
        return None
    raw_mesh = getattr(mesh, "jax_mesh", mesh)
    axis_names = getattr(raw_mesh, "axis_names", None)
    if axis_names is None:
        return None
    return {str(axis) for axis in axis_names}


def _mesh_axis_size(mesh: "Mesh | SpxMesh | MpMdMesh | None", axis: str) -> int:
    """Return the device count along ``axis`` on the underlying JAX mesh.

    Falls back to ``1`` for any axis the mesh does not know about
    (treating "missing axis" the same as "size-1 axis"), which lets
    callers compute partition products without an explicit existence
    check.
    """
    raw_mesh = getattr(mesh, "jax_mesh", mesh)
    try:
        return int(raw_mesh.shape[axis])
    except Exception:
        return 1


def _query_mesh(mesh: "Mesh | SpxMesh | MpMdMesh | None" = None, *, raise_error: bool = True) -> Any:
    """Return ``mesh`` if given, else fall back to the active mesh context.

    The fallback chain is: explicit ``mesh`` argument → SpectraX
    :func:`current_mesh` → :func:`get_incontext_mesh` (which itself
    falls back to the JAX physical mesh).
    """
    if mesh is not None:
        return mesh
    from .mesh import current_mesh

    spx_mesh = current_mesh()
    if spx_mesh is not None:
        return spx_mesh
    return get_incontext_mesh(raise_error=raise_error)


def _same_mesh_devices(a: Any, b: Any) -> bool:
    """Return whether two raw JAX meshes cover the same physical devices."""
    try:
        if tuple(a.axis_names) != tuple(b.axis_names):
            return False
        return tuple(a.devices.reshape(-1)) == tuple(b.devices.reshape(-1))
    except Exception:
        return False


def _promote_active_spx_mesh(mesh: "Mesh | SpxMesh | MpMdMesh | None") -> "Mesh | SpxMesh | MpMdMesh | None":
    """Recover active ``SpxMesh`` metadata when a raw JAX mesh leaks in.

    Some downstream code builds ``NamedSharding`` objects from
    ``spx_mesh.jax_mesh`` and later passes only the raw JAX mesh back
    to ``with_sharding_constraint``. If the active SpectraX context is
    MPMD and points at that same physical mesh, keep the MPMD metadata
    so constraints resolve to stage-local submeshes instead of the full
    PP mesh.
    """
    if mesh is None or _mpmd_axis_name(mesh) is not None:
        return mesh
    from .mesh import current_mesh

    active = current_mesh()
    if active is not None and active.is_mpmd and _same_mesh_devices(mesh, active.jax_mesh):
        return active
    return mesh


def _available_axis_names(mesh: "Mesh | SpxMesh | MpMdMesh | None") -> tuple[str, ...]:
    """Return the SPMD-only axis names visible on ``mesh``.

    The pipeline (MPMD) axis is intentionally excluded: callers that
    inspect "what axes can I shard on?" should not see the rank axis,
    because constraining a tensor along that axis would attempt to
    shard across stages.
    """
    if mesh is None:
        return ()
    from ..runtime.types.mesh import MpMdMesh

    if isinstance(mesh, SpxMesh):
        if mesh.mpmd_mesh is not None:
            return mesh.mpmd_mesh.spmd_axis_names
        return tuple(mesh.jax_mesh.axis_names)
    if isinstance(mesh, MpMdMesh):
        return mesh.spmd_axis_names
    return tuple(getattr(mesh, "axis_names", ()))


def names_in_current_mesh(*names: str, mesh: "Mesh | SpxMesh | MpMdMesh | None" = None) -> bool:
    """Return ``True`` iff all names are usable SPMD axes in the current mesh.

    On MPMD meshes the pipeline axis is intentionally excluded because
    it represents rank/program placement, not a stage-local sharding
    axis. Pass an explicit pure JAX mesh if full raw mesh membership is
    needed.
    """
    resolved_mesh = _query_mesh(mesh, raise_error=False)
    if resolved_mesh is None:
        return False
    return set(names) <= set(_available_axis_names(resolved_mesh))


def get_axes_size_in_mesh(
    axis_names: str | tuple[str | None, ...] | list[str | None] | None,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
) -> int:
    """Return the product of stage-local mesh sizes for ``axis_names``.

    MPMD-aware behavior: the MPMD/pipeline axis contributes ``1`` so
    callers do not accidentally treat pipeline rank count as an
    intra-stage tensor partition factor.
    """
    resolved_mesh = _query_mesh(mesh)
    if axis_names is None:
        return 1
    if isinstance(axis_names, str):
        axis_iter = (axis_names,)
    elif isinstance(axis_names, (tuple, list)):
        axis_iter = tuple(axis for axis in axis_names if axis is not None)
    else:
        raise TypeError(f"axis_names must be str, sequence[str], or None; got {type(axis_names)}")

    mpmd_axis = _mpmd_axis_name(resolved_mesh)
    product = 1
    for axis in axis_iter:
        if axis == mpmd_axis:
            continue
        product *= _mesh_axis_size(resolved_mesh, axis)
    return product


def _sanitize_axis_for_mesh(axis: Any, mesh: "Mesh | SpxMesh | MpMdMesh | None") -> Any:
    """Drop axis-name parts that are not present on ``mesh``.

    Symmetric to :func:`_drop_axis_name` but instead of removing one
    specific name, this keeps only names that *exist* on the mesh.
    Useful when reusing a spec built for a richer mesh on a simpler
    one (e.g. dropping ``"ep"`` when running with EP=1).
    """
    names = _mesh_axis_names(mesh)
    if names is None:
        return axis
    if axis is None:
        return None
    if isinstance(axis, tuple):
        kept = tuple(part for part in axis if part in names)
        if not kept:
            return None
        if len(kept) == 1:
            return kept[0]
        return kept
    return axis if axis in names else None


def _axis_partition_product(axis: Any, mesh: "Mesh | SpxMesh | MpMdMesh | None") -> int:
    """Return the total device count implied by an axis spec.

    A bare name returns its mesh size; a tuple of names returns the
    product of all parts; ``None`` returns ``1``. Used to test
    "does this axis spec divide the tensor's dim evenly?".
    """
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        product = 1
        for part in axis:
            product *= _mesh_axis_size(mesh, part)
        return product
    return _mesh_axis_size(mesh, axis)


def sanitize_partition_spec_for_mesh_and_shape(
    spec: PartitionSpec | None,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
    shape: tuple[int, ...] | None = None,
) -> PartitionSpec:
    """Make a ``PartitionSpec`` safe for the target mesh and optional value shape."""
    if spec is None:
        spec = PartitionSpec()
    elif not isinstance(spec, PartitionSpec):
        spec = PartitionSpec(*tuple(spec))

    mpmd_axis = _mpmd_axis_name(mesh)
    if mpmd_axis is not None:
        spec = PartitionSpec(*(_drop_axis_name(axis, mpmd_axis) for axis in spec))

    spec = PartitionSpec(*(_sanitize_axis_for_mesh(axis, mesh) for axis in spec))

    if shape is not None and len(spec) > len(shape):
        return PartitionSpec()
    if shape is not None:
        axes = list(tuple(spec))
        changed = False
        for dim, axis in enumerate(axes):
            product = _axis_partition_product(axis, mesh)
            if product > 1 and int(shape[dim]) % product != 0:
                axes[dim] = None
                changed = True
        if changed:
            return PartitionSpec(*axes)
    return spec


def _sharding_spec(sharding: Any) -> PartitionSpec:
    """Coerce any sharding-like input to a plain :class:`PartitionSpec`.

    Accepts a :class:`NamedSharding` (extracts ``.spec``), a
    :class:`PartitionSpec` (returned as-is), ``None`` (becomes the
    fully-replicated spec), or anything iterable (its contents are
    spread into a new ``PartitionSpec``).
    """
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    if isinstance(sharding, PartitionSpec):
        return sharding
    if sharding is None:
        return PartitionSpec()
    return PartitionSpec(*tuple(sharding))


def _stage_mesh_from_existing_sharding(
    arr: ArrayLike,
    mpmd_mesh: "MpMdMesh",
) -> "Mesh | None":
    """Recover the stage-local mesh from ``arr.sharding`` when possible.

    If the array is already living on a sub-mesh of the MPMD mesh
    (same axis names, MPMD axis size 1), return that sub-mesh so a
    follow-up constraint can target the same stage. Otherwise return
    ``None``.
    """
    existing = getattr(arr, "sharding", None)
    if not isinstance(existing, NamedSharding):
        return None
    mesh = existing.mesh
    if mesh.axis_names != mpmd_mesh.jax_mesh.axis_names:
        return None
    try:
        if int(mesh.shape[mpmd_mesh.mpmd_axis_name]) == 1:
            return mesh
    except Exception:
        return None
    return None


def _stage_mesh_from_current_context(mpmd_mesh: "MpMdMesh") -> "Mesh | None":
    """Recover the stage-local mesh from the active JAX mesh context.

    Mirrors :func:`_stage_mesh_from_existing_sharding` but reads from
    ``jax.interpreters.pxla.thread_resources`` instead of the array's
    own sharding. Used as a secondary fallback in
    :func:`_resolve_constraint_target`.
    """
    mesh = _physical_mesh_or_none()
    if mesh is None or mesh.axis_names != mpmd_mesh.jax_mesh.axis_names:
        return None
    try:
        if int(mesh.shape[mpmd_mesh.mpmd_axis_name]) == 1:
            return mesh
    except Exception:
        return None
    return None


def _resolve_constraint_target(
    arr: ArrayLike,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None",
    stage: int | tuple[int, int] | None,
    stage_mesh: "Mesh | None",
    metadata: dict[str, Any] | None,
) -> tuple["Mesh | None", "MpMdMesh | None", bool]:
    """Return ``(target_mesh, mpmd_mesh, unresolved_mpmd)`` for a constraint."""
    base_mesh, mpmd_mesh = (None, None)
    if mesh is not None:
        base_mesh, mpmd_mesh = _resolve_named_sharding_mesh(mesh)

    if mpmd_mesh is None:
        target = to_jax_mesh(mesh)
        if target is None:
            existing = getattr(arr, "sharding", None)
            if isinstance(existing, NamedSharding):
                target = existing.mesh
        return target, None, False

    if stage_mesh is not None:
        return stage_mesh, mpmd_mesh, False

    if stage is not None:
        if isinstance(stage, tuple):
            owner = resolve_stage_rank(stage, mpmd_mesh.mpmd_dim)
        else:
            owner = int(stage)
        return mpmd_mesh.submesh(owner), mpmd_mesh, False

    assignment = metadata_stage_assignment(metadata)
    if assignment is None:
        assignment = current_stage_assignment()
    owner = resolve_stage_rank(assignment, mpmd_mesh.mpmd_dim)
    if owner is not None:
        return mpmd_mesh.submesh(owner), mpmd_mesh, False

    inferred = _stage_mesh_from_existing_sharding(arr, mpmd_mesh)
    if inferred is not None:
        return inferred, mpmd_mesh, False

    inferred = _stage_mesh_from_current_context(mpmd_mesh)
    if inferred is not None:
        return inferred, mpmd_mesh, False

    owner = mpmd_mesh.my_mpmd_axis_index()
    if owner is not None:
        return mpmd_mesh.submesh(owner), mpmd_mesh, False

    if mpmd_mesh.mpmd_dim == 1:
        return mpmd_mesh.submesh(0), mpmd_mesh, False

    return base_mesh, mpmd_mesh, True


def with_sharding_constraint(
    arr: ArrayLike,
    sharding: PartitionSpec | NamedSharding | Any,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
    stage: int | tuple[int, int] | None = None,
    stage_mesh: "Mesh | None" = None,
    metadata: dict[str, Any] | None = None,
    ignore_mpmd: bool = False,
) -> ArrayLike:
    """MPMD-aware variant of ``jax.lax.with_sharding_constraint``.

    Pure SPMD meshes use the same semantics as a normal sharding
    constraint. MPMD meshes resolve a stage-local target mesh from, in
    order: ``stage_mesh``, ``stage``, variable ``metadata``, active
    ``assign_stage(...)`` context, the array's existing sharding, the
    current JAX mesh context, or the process-local MPMD rank. If no stage
    can be resolved on a multi-stage single-host mesh, this is a safe
    no-op rather than accidentally constraining on the full PP mesh.
    Passing ``ignore_mpmd=True`` makes MPMD meshes an explicit no-op,
    while still constraining normally on SPMD meshes.
    """
    resolved_mesh = mesh
    if resolved_mesh is None:
        resolved_mesh = _query_mesh(raise_error=False)
    if resolved_mesh is None and isinstance(sharding, NamedSharding):
        resolved_mesh = sharding.mesh
    resolved_mesh = _promote_active_spx_mesh(resolved_mesh)
    if ignore_mpmd and _mpmd_axis_name(resolved_mesh) is not None:
        return arr
    if not hasattr(arr, "shape"):
        leaves = jax.tree_util.tree_leaves(arr)
        if any(hasattr(leaf, "shape") for leaf in leaves):
            return lax_reshard(
                arr,
                sharding,
                mesh=resolved_mesh,
                stage=stage,
                stage_mesh=stage_mesh,
                metadata=metadata,
                ignore_mpmd=ignore_mpmd,
            )
        return arr
    target_mesh, mpmd_mesh, unresolved_mpmd = _resolve_constraint_target(
        arr,
        mesh=resolved_mesh,
        stage=stage,
        stage_mesh=stage_mesh,
        metadata=metadata,
    )
    if unresolved_mpmd:
        return arr

    sanitize_mesh = mpmd_mesh if mpmd_mesh is not None else target_mesh
    spec = sanitize_partition_spec_for_mesh_and_shape(
        _sharding_spec(sharding),
        mesh=sanitize_mesh,
        shape=tuple(getattr(arr, "shape", ())),
    )
    constraint: PartitionSpec | NamedSharding
    if target_mesh is None:
        constraint = spec
    else:
        spec = sanitize_partition_spec_for_mesh_and_shape(
            spec,
            mesh=target_mesh,
            shape=tuple(getattr(arr, "shape", ())),
        )
        constraint = NamedSharding(target_mesh, spec)
    try:
        return jax.lax.with_sharding_constraint(arr, constraint)
    except RuntimeError as exc:
        if "requires a non-empty mesh" in str(exc):
            return arr
        raise


def _is_axis_name_like(value: Any) -> bool:
    """Return ``True`` iff ``value`` is a valid axis-spec component.

    Axis names are strings or ``None``; nested tuples are also valid
    (each element is checked recursively). Anything else (lists,
    ints, ``PartitionSpec``s, …) is rejected.
    """
    if value is None or isinstance(value, str):
        return True
    if isinstance(value, tuple):
        return all(_is_axis_name_like(part) for part in value)
    return False


def _is_single_sharding_spec(value: Any) -> bool:
    """Return ``True`` iff ``value`` is a *single* sharding spec, not a pytree of specs.

    Used as the ``is_leaf`` predicate in :func:`lax_reshard` so the
    function can disambiguate "one spec to broadcast" from "pytree of
    per-leaf specs". A bare ``PartitionSpec``, ``NamedSharding``,
    ``None``, or a list/tuple of axis-name-likes counts as a single
    spec.
    """
    if isinstance(value, (NamedSharding, PartitionSpec)):
        return True
    if value is None:
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_axis_name_like(axis) for axis in value)
    return False


def lax_reshard(
    arr: Any,
    sharding: PartitionSpec | NamedSharding | Any,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
    stage: int | tuple[int, int] | None = None,
    stage_mesh: "Mesh | None" = None,
    metadata: dict[str, Any] | None = None,
    ignore_mpmd: bool = False,
) -> Any:
    """Apply an MPMD-aware sharding constraint to an array or pytree.

    ``PartitionSpec`` / ``NamedSharding`` values apply to every array
    leaf. A pytree of specs can also be passed, matching the input tree.
    Non-array leaves are returned unchanged.
    """
    if _is_single_sharding_spec(sharding):
        return jax.tree_util.tree_map(
            lambda leaf: with_sharding_constraint(
                leaf,
                sharding,
                mesh=mesh,
                stage=stage,
                stage_mesh=stage_mesh,
                metadata=metadata,
                ignore_mpmd=ignore_mpmd,
            ),
            arr,
        )
    return jax.tree_util.tree_map(
        lambda leaf, leaf_sharding: with_sharding_constraint(
            leaf,
            leaf_sharding,
            mesh=mesh,
            stage=stage,
            stage_mesh=stage_mesh,
            metadata=metadata,
            ignore_mpmd=ignore_mpmd,
        ),
        arr,
        sharding,
        is_leaf=_is_single_sharding_spec,
    )


def _named_sharding_for_leaf(leaf: Any, mesh: "Mesh | SpxMesh | MpMdMesh | None") -> Any:
    """Return a ``NamedSharding`` for ``leaf`` aligned to ``mesh``.

    If the leaf already has a :class:`NamedSharding`, its spec is
    sanitized (mesh-axis-filtered, shape-divisibility-checked) and
    re-bound to the mesh argument. Leaves without a ``shape``
    attribute (non-arrays) return ``None``.
    """
    if not hasattr(leaf, "shape"):
        return None

    existing = getattr(leaf, "sharding", None)
    if isinstance(existing, NamedSharding):
        spec = existing.spec
    else:
        spec = PartitionSpec()

    spec = sanitize_partition_spec_for_mesh_and_shape(
        spec,
        mesh=mesh,
        shape=tuple(getattr(leaf, "shape", ())),
    )
    target_mesh = to_jax_mesh(mesh) or getattr(existing, "mesh", None)
    if target_mesh is None:
        return existing
    return NamedSharding(target_mesh, spec)


def extract_shardings(tree: Any, mesh: "Mesh | SpxMesh | MpMdMesh | None" = None) -> Any:
    """Extract a pytree of compatible ``NamedSharding`` objects from ``tree``."""
    return jax.tree.map(lambda leaf: _named_sharding_for_leaf(leaf, mesh), tree)


def _extracted_sharding_for_leaf(
    leaf: Any,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None",
    stage: int | tuple[int, int] | None,
    stage_mesh: "Mesh | None",
    metadata: dict[str, Any] | None,
) -> NamedSharding | None:
    """Return a sanitized, MPMD-aware ``NamedSharding`` from a leaf's existing sharding.

    Mirrors :func:`_named_sharding_for_leaf` but additionally resolves
    the stage-local mesh through :func:`_resolve_constraint_target` so
    that pipeline-axis sharding is dropped and the spec lands on the
    correct sub-mesh. Returns ``None`` if the leaf has no
    ``NamedSharding``.
    """
    existing = getattr(leaf, "sharding", None)
    if not isinstance(existing, NamedSharding):
        return None
    if not hasattr(leaf, "shape"):
        return existing

    target_mesh, mpmd_mesh, unresolved_mpmd = _resolve_constraint_target(
        leaf,
        mesh=mesh,
        stage=stage,
        stage_mesh=stage_mesh,
        metadata=metadata,
    )
    if mesh is None or unresolved_mpmd or target_mesh is None:
        return existing

    sanitize_mesh = mpmd_mesh if mpmd_mesh is not None else target_mesh
    spec = sanitize_partition_spec_for_mesh_and_shape(
        existing.spec,
        mesh=sanitize_mesh,
        shape=tuple(getattr(leaf, "shape", ())),
    )
    spec = sanitize_partition_spec_for_mesh_and_shape(
        spec,
        mesh=target_mesh,
        shape=tuple(getattr(leaf, "shape", ())),
    )
    return NamedSharding(target_mesh, spec)


def extract_sharding_structure(
    pytree: Any,
    *,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
    stage: int | tuple[int, int] | None = None,
    stage_mesh: "Mesh | None" = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Mirror ``pytree`` with extracted ``NamedSharding`` leaves.

    If an MPMD mesh is active or explicitly provided, returned
    shardings are sanitized to the resolved stage-local mesh and the
    pipeline axis is removed from the spec. Leaves without a
    ``NamedSharding`` become ``None``.
    """
    resolved_mesh = mesh if mesh is not None else _query_mesh(raise_error=False)
    return jax.tree_util.tree_map(
        lambda leaf: _extracted_sharding_for_leaf(
            leaf,
            mesh=resolved_mesh,
            stage=stage,
            stage_mesh=stage_mesh,
            metadata=metadata,
        ),
        pytree,
    )


def match_partition_rules(
    rules: list[tuple[str, PartitionSpec]] | tuple[tuple[str, PartitionSpec], ...],
    tree: Any,
    min_size: int | None = None,
    strict: bool = True,
) -> Any:
    """Match regex partition rules against tree paths and preserve tree structure."""
    import re

    def _path_to_string(path: tuple[Any, ...], sep: str = "/") -> str:
        """Render a JAX tree-key path as a forward-slash-separated string."""
        parts: list[str] = []
        for key in path:
            if isinstance(key, jax.tree_util.SequenceKey):
                parts.append(str(key.idx))
            elif isinstance(key, jax.tree_util.DictKey):
                parts.append(str(key.key))
            elif isinstance(key, jax.tree_util.GetAttrKey):
                parts.append(str(key.name))
            elif isinstance(key, jax.tree_util.FlattenedIndexKey):
                parts.append(str(key.key))
            else:
                parts.append(str(key))
        return sep.join(parts)

    def _spec_for(path: str, leaf: Any) -> PartitionSpec:
        """Pick the first matching rule's ``PartitionSpec`` for one leaf.

        Honors the ``strict`` and ``min_size`` outer arguments:
        non-arrays / scalars / tiny tensors collapse to fully
        replicated, and over-long specs are truncated to the leaf's
        ``ndim``.
        """
        if strict:
            if not hasattr(leaf, "shape"):
                return PartitionSpec()
            if len(getattr(leaf, "shape", ())) == 0:
                return PartitionSpec()
        if min_size is not None and hasattr(leaf, "shape") and int(np.prod(leaf.shape)) < min_size:
            return PartitionSpec()
        for pattern, spec in rules:
            if re.search(pattern, path):
                out = spec if isinstance(spec, PartitionSpec) else PartitionSpec(*tuple(spec))
                if strict and hasattr(leaf, "ndim") and len(out) > leaf.ndim:
                    out = PartitionSpec(*tuple(out)[: leaf.ndim])
                return out
        raise ValueError(f"Partition rule not found for param: {path}")

    return jax.tree_util.tree_map_with_path(lambda path, leaf: _spec_for(_path_to_string(path), leaf), tree)


def make_shard_and_gather_fns(
    partition_specs: Any,
    mesh: "Mesh | SpxMesh | MpMdMesh | None" = None,
) -> tuple[Any, Any]:
    """Build shard/gather helper pytrees matching ``partition_specs``."""
    target_mesh = to_jax_mesh(mesh)
    if target_mesh is None:
        raise ValueError("mesh must be provided to make_shard_and_gather_fns")

    def _named(raw_spec: Any) -> NamedSharding:
        """Wrap a raw spec in a ``NamedSharding`` bound to the closure's mesh."""
        if not isinstance(raw_spec, PartitionSpec):
            raw_spec = PartitionSpec()
        spec = sanitize_partition_spec_for_mesh_and_shape(raw_spec, mesh=mesh)
        return NamedSharding(target_mesh, spec)

    named_shardings = jax.tree_util.tree_map(
        _named,
        partition_specs,
        is_leaf=lambda x: isinstance(x, PartitionSpec),
    )

    def _make_shard_fn(sharding: NamedSharding) -> Callable:
        """Build a per-leaf "device_put with this sharding" closure."""

        def _shard(x, _sharding=sharding):
            """Place ``x`` on devices according to the captured ``NamedSharding``."""
            if hasattr(x, "shape"):
                spec = sanitize_partition_spec_for_mesh_and_shape(_sharding.spec, mesh=mesh, shape=tuple(x.shape))
                return jax.device_put(x, NamedSharding(_sharding.mesh, spec))
            return x

        return _shard

    def _make_gather_fn(sharding: NamedSharding) -> Callable:
        """Build a per-leaf "device_put as fully replicated" closure."""
        replicated = NamedSharding(sharding.mesh, PartitionSpec())

        def _gather(x, _replicated=replicated):
            """Replicate ``x`` across every device on the captured mesh."""
            if hasattr(x, "shape"):
                return jax.device_put(x, _replicated)
            return x

        return _gather

    return (
        jax.tree_util.tree_map(_make_shard_fn, named_shardings),
        jax.tree_util.tree_map(_make_gather_fn, named_shardings),
    )


def get_corrected_named_sharding(
    shape: tuple[int, ...],
    partition_spec: PartitionSpec,
    raise_mesh_error: bool = True,
) -> NamedSharding:
    """Return an active-mesh ``NamedSharding`` whose spec is valid for ``shape``."""
    mesh = get_incontext_mesh(raise_error=raise_mesh_error)
    if mesh is None:
        target_mesh = jax.sharding.Mesh(jax.devices()[:1], ("spx",))
        sanitize_mesh = target_mesh
    else:
        target_mesh = to_jax_mesh(mesh)
        sanitize_mesh = mesh
    spec = sanitize_partition_spec_for_mesh_and_shape(partition_spec, mesh=sanitize_mesh, shape=tuple(shape))
    return NamedSharding(target_mesh, spec)


def apply_logical_sharding(
    x: ArrayLike,
    partition_manager: Any | None = NOT_GIVEN,
    axes: tuple[Any, ...] | list[Any] | None = NOT_GIVEN,
    mode: Any = NOT_GIVEN,
    dynamic_axes: Any | None = NOT_GIVEN,
    auto_correct: bool = True,
) -> ArrayLike:
    """Apply a logical sharding constraint without depending on eformer.

    If ``partition_manager`` provides a ``resolve`` method, it is used
    for compatibility with existing EasyDeL config objects. Otherwise
    ``axes`` is treated as an already-physical ``PartitionSpec``. MPMD
    placement is inherited from the manager's mesh, or from the active
    :class:`SpxMesh` plus ``assign_stage(...)`` context.
    """
    if not hasattr(x, "shape"):
        return x
    resolved_manager = partition_manager
    if resolved_manager is NOT_GIVEN or resolved_manager is None:
        try:
            from .manager import get_current_partition_manager, get_partition_manager

            resolved_manager = get_current_partition_manager() or get_partition_manager()
        except Exception:
            resolved_manager = None

    if resolved_manager is not None and hasattr(resolved_manager, "shard"):
        return resolved_manager.shard(
            x,
            axes=axes,
            mode=mode,
            dynamic_axes=dynamic_axes,
            auto_correct=auto_correct,
        )
    if resolved_manager is not None and hasattr(resolved_manager, "resolve"):
        spec = resolved_manager.resolve(axes=axes, mode=mode, dynamic_axes=dynamic_axes, shape=x.shape)
    elif axes is NOT_GIVEN or axes is None:
        spec = PartitionSpec()
    else:
        spec = PartitionSpec(*tuple(axes))
    if auto_correct:
        spec = get_corrected_named_sharding(tuple(x.shape), spec, raise_mesh_error=False).spec
    return with_sharding_constraint(x, spec)


def _identity_mesh_axis_rules(base_mesh: "Mesh", mpmd_mesh: "MpMdMesh | None") -> dict[str, str]:
    """Build a no-op logical-to-physical map (every axis maps to itself).

    Used as the baseline ``current_axis_rules()`` overrides, so that
    ``Sharding.to_partition_spec`` can resolve raw mesh-axis names
    even when no logical-rule context is active.
    """
    axis_names = mpmd_mesh.spmd_axis_names if mpmd_mesh is not None else base_mesh.axis_names
    return {name: name for name in axis_names}


def named_sharding_for_metadata(metadata: dict[str, Any], mesh: "Mesh | SpxMesh | MpMdMesh") -> NamedSharding | None:
    """Resolve raw variable-style metadata to a ``NamedSharding``."""
    sharding = normalize_sharding(metadata.get("sharding"))
    if sharding is None:
        names = metadata.get("axis_names")
        sharding = Sharding(axis_names=tuple(names)) if names is not None else None
    if sharding is None:
        return None

    base_mesh, mpmd_mesh = _resolve_named_sharding_mesh(mesh)
    mesh_map = _identity_mesh_axis_rules(base_mesh, mpmd_mesh)
    mesh_map.update(current_axis_rules())
    spec = sharding.to_partition_spec(mesh_map)
    if mpmd_mesh is not None:
        owner = resolve_stage_rank(metadata_stage_assignment(metadata), mpmd_mesh.mpmd_dim)
        if owner is not None:
            return mpmd_mesh.sub_sharding(owner, spec)
    return NamedSharding(base_mesh, spec)


def named_sharding_for_variable(var: Variable, mesh: "Mesh | SpxMesh | MpMdMesh") -> NamedSharding:
    """Resolve one variable's metadata to a ``NamedSharding``.

    Stage-tagged variables on an MPMD mesh resolve against their owning
    stage sub-mesh. Untagged variables resolve against the full mesh,
    which means they replicate across the pipeline axis unless their
    spec explicitly names it.
    """
    resolved = named_sharding_for_metadata(var.metadata, mesh)
    if resolved is not None:
        return resolved

    base_mesh, _ = _resolve_named_sharding_mesh(mesh)
    return NamedSharding(base_mesh, Sharding().to_partition_spec(current_axis_rules() or None))


def get_named_sharding(module: Module, mesh: "Mesh | SpxMesh | MpMdMesh") -> dict[str, dict[str, NamedSharding]]:
    """Wrap each variable's resolved spec in a ``NamedSharding``.

    Accepts a plain JAX mesh, an :class:`~spectrax.sharding.SpxMesh`,
    or an :class:`~spectrax.runtime.types.MpMdMesh`. When the mesh is
    MPMD and a variable carries a stage hint, the returned sharding is
    stage-local automatically.
    """
    out: dict[str, dict[str, NamedSharding]] = {}
    for path, var in _iter_variables(module):
        out.setdefault(var.kind, {})[path] = named_sharding_for_variable(var, mesh)
    return out


def with_sharding_constraint_by_name(x: ArrayLike, axis_names: AxisNames) -> Array:
    """Apply a sharding constraint using the active logical -> mesh rules.

    Inside an active :func:`logical_axis_rules` context the logical names
    are resolved to physical mesh axes; outside one the constraint is a
    no-op replicate.
    """

    rules = current_axis_rules()
    resolved = _resolve_axis_names(tuple(axis_names), rules)
    spec = PartitionSpec(*resolved)
    return jax.lax.with_sharding_constraint(x, spec)


_ = Variable
