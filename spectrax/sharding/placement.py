# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup-time device placement helpers for JAX shardings.

These helpers are intentionally for initialization/loading paths. They avoid
multi-controller JAX's illegal direct reshard from one physical device set to a
different physical device set by staging through host memory. They never invent
array values; callers that need optimizer-zero synthesis should do that in the
optimizer initialization layer where the zero invariant is known.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import device_get

from spectrax._internal.logging import get_logger

logger = get_logger(__name__)


def _device_set_from_sharding(sharding: object) -> set[object] | None:
    devices = getattr(sharding, "device_set", None)
    if devices is not None:
        try:
            return set(devices() if callable(devices) else devices)
        except Exception:
            pass
    mesh = getattr(sharding, "mesh", None)
    if mesh is not None:
        try:
            return set(mesh.devices.flat)
        except Exception:
            return None
    return None


def _device_set_from_value(value: object) -> set[object] | None:
    sharding = getattr(value, "sharding", None)
    devices = _device_set_from_sharding(sharding)
    if devices is not None:
        return devices
    value_devices = getattr(value, "devices", None)
    if value_devices is None:
        return None
    try:
        return set(value_devices() if callable(value_devices) else value_devices)
    except Exception:
        return None


def _device_ids(devices: set[object] | None) -> tuple[int, ...] | None:
    if devices is None:
        return None
    return tuple(sorted(int(getattr(device, "id", idx)) for idx, device in enumerate(devices)))


def _device_id_preview(device_ids: tuple[int, ...] | None) -> str:
    if device_ids is None:
        return "unknown"
    if len(device_ids) <= 12:
        return repr(device_ids)
    head = ", ".join(str(device_id) for device_id in device_ids[:6])
    tail = ", ".join(str(device_id) for device_id in device_ids[-3:])
    return f"({head}, ..., {tail})"


def _mesh_axis_names(sharding: object) -> tuple[object, ...] | None:
    mesh = getattr(sharding, "mesh", None)
    if mesh is None:
        return None
    axis_names = getattr(mesh, "axis_names", None)
    return tuple(axis_names) if axis_names is not None else None


def _same_setup_sharding(value: object, sharding: object) -> bool:
    current = getattr(value, "sharding", None)
    if current is None or sharding is None:
        return False
    if current is sharding or current == sharding:
        return True
    if type(current) is not type(sharding):
        return False
    if getattr(current, "spec", None) != getattr(sharding, "spec", None):
        return False
    if getattr(current, "memory_kind", None) != getattr(sharding, "memory_kind", None):
        return False
    return _device_set_from_sharding(current) == _device_set_from_sharding(sharding)


def _path_to_string(path: tuple[object, ...]) -> str:
    parts: list[str] = []
    for entry in path:
        key = getattr(entry, "key", None)
        if key is not None:
            parts.append(str(key))
            continue
        idx = getattr(entry, "idx", None)
        if idx is not None:
            parts.append(str(idx))
            continue
        name = getattr(entry, "name", None)
        if name is not None:
            parts.append(str(name))
            continue
        parts.append(str(entry))
    return "/".join(parts)


def place_setup_leaf_with_sharding(
    leaf: tp.Any,
    sharding: jax.sharding.Sharding,
    *,
    path: str = "",
    label: str = "SpectraX setup placement",
    diagnostics: dict[str, int] | None = None,
    donate: bool = False,
) -> tp.Any:
    """Place one setup leaf without illegal cross-device-set resharding.

    The helper is intentionally conservative: when device sets differ, it
    stages the existing value through host memory and places that value on the
    target sharding. It never synthesizes replacement data.
    """
    if sharding is None or not hasattr(leaf, "shape"):
        return leaf
    if _same_setup_sharding(leaf, sharding):
        return leaf

    source_devices = _device_set_from_value(leaf)
    target_devices = _device_set_from_sharding(sharding)
    if source_devices is None or target_devices is None or source_devices == target_devices:
        return jax.device_put(leaf, sharding, donate=donate)

    if diagnostics is None:
        diagnostics = {}
    diagnostics["cross_device_set"] = diagnostics.get("cross_device_set", 0) + 1
    process_index = jax.process_index()
    if process_index == 0 and diagnostics.get("logged", 0) < 5:
        source_ids = _device_ids(source_devices)
        target_ids = _device_ids(target_devices)
        global_device_count = jax.device_count()
        source_is_full_global = len(source_devices) == global_device_count
        target_is_subset = len(target_devices) < len(source_devices) and target_devices <= source_devices
        source_sharding = getattr(leaf, "sharding", None)
        logger.warning(
            "%s detected cross-device-set setup placement at %s on process %d; "
            "shape=%s dtype=%s source_sharding=%s source_axes=%s source_device_count=%d "
            "source_device_ids=%s target_axes=%s target_spec=%s target_device_count=%d "
            "target_device_ids=%s source_is_full_global=%s target_is_subset=%s. "
            "Staging this setup leaf through host before placing on the target sharding.",
            label,
            path or "<leaf>",
            process_index,
            tuple(getattr(leaf, "shape", ())),
            getattr(leaf, "dtype", None),
            type(source_sharding).__name__ if source_sharding is not None else None,
            _mesh_axis_names(source_sharding),
            len(source_devices),
            _device_id_preview(source_ids),
            _mesh_axis_names(sharding),
            getattr(sharding, "spec", None),
            len(target_devices),
            _device_id_preview(target_ids),
            source_is_full_global,
            target_is_subset,
        )
        diagnostics["logged"] = diagnostics.get("logged", 0) + 1

    host_leaf = device_get(leaf)
    return jax.device_put(host_leaf, sharding, donate=donate)


def place_setup_tree_with_shardings(
    tree: tp.Any,
    shardings: tp.Any,
    *,
    label: str = "SpectraX setup placement",
    donate: bool = False,
) -> tp.Any:
    """Place a setup-time pytree with per-leaf sharding diagnostics."""
    diagnostics: dict[str, int] = {"cross_device_set": 0, "logged": 0}

    def _place(path: tuple[object, ...], leaf: tp.Any, sharding: tp.Any) -> tp.Any:
        if not isinstance(sharding, jax.sharding.Sharding) or not hasattr(leaf, "shape"):
            return leaf
        return place_setup_leaf_with_sharding(
            leaf,
            sharding,
            path=_path_to_string(path),
            label=label,
            diagnostics=diagnostics,
            donate=donate,
        )

    placed = jax.tree_util.tree_map_with_path(
        _place,
        tree,
        shardings,
        is_leaf=lambda x: isinstance(x, jax.sharding.Sharding) or x is None,
    )
    if diagnostics["cross_device_set"] and jax.process_index() == 0:
        logger.warning(
            "%s staged %d setup leaves through host because source and target device sets differed. "
            "This helper is for setup/loading, not compiled training steps.",
            label,
            diagnostics["cross_device_set"],
        )
    return placed
