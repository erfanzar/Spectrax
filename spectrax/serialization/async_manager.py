# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import base64
import json
import os
import pickle
import typing as tp
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from jax.distributed import is_initialized
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax._internal.logging import get_logger
from spectrax._version import __version__
from spectrax.serialization.serialization import leaf_key_paths

from . import _fs, fsspec_utils
from ._compat import PyTree
from .serialization import tree_serialize_leaves

logger = get_logger("AsyncCheckpointManager")
GLOBAL_CHECKPOINT_TIMEOUT = int(os.getenv("GLOBAL_CHECKPOINT_TIMEOUT", "400"))


def _is_array_like(x):
    """Check if an object is array-like (has shape and dtype attributes)."""
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _treedef_to_b64(treedef) -> str:
    """Serialize a JAX tree definition to base64 string."""
    return base64.b64encode(pickle.dumps(treedef)).decode("utf-8")


def _treedef_from_b64(s: str):
    """Deserialize a JAX tree definition from base64 string."""
    return pickle.loads(base64.b64decode(s.encode("utf-8")))


def _structure_path(path: str, prefix: str | None) -> str:
    """Return the JSON structure-file path for a given checkpoint dir and prefix.

    Args:
        path: Checkpoint directory (local or remote).
        prefix: Logical namespace/prefix (e.g. ``"model"``). If ``None``,
            defaults to ``"pytree"``.

    Returns:
        The joined path string ending in ``"{prefix}_structure.json"``.
    """
    name = f"{prefix or 'pytree'}_structure.json"
    return _fs.joinpath(path, name)


def _is_none(x):
    """Check if a value is None."""
    return x is None


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for checkpoints with versioning and validation."""

    version: str = __version__
    timestamp: str = None
    custom_metadata: dict = None

    def to_dict(self) -> dict:
        """Serialize the metadata dataclass to a plain dictionary.

        Returns:
            Dictionary with keys ``"version"``, ``"timestamp"``,
            and ``"custom_metadata"``.
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "custom_metadata": self.custom_metadata or {},
        }


class AsyncCheckpointManager:
    """Checkpoint manager built on top of JAX GlobalAsyncCheckpointManager (TensorStore).

    Provides checkpoint saving and loading with support for parallel operations
    and tensorstore backend. Preserves existing array shardings (TP/FSDP)
    without performing all-gather operations.
    """

    def __init__(self, float_dtype: jnp.dtype = jnp.bfloat16):
        """Initialize the async checkpoint manager.

        Args:
            float_dtype: Default dtype used when ``dtype`` is not explicitly
                passed to :meth:`save_pytree`. Defaults to ``jnp.bfloat16``.

        Raises:
            RuntimeError: If running on multiple processes and JAX distributed
                has not been initialized.
        """
        if jax.process_count() > 1:
            if not is_initialized():
                raise RuntimeError("you should call jax distribution init before running process.")

        self.float_dtype = float_dtype
        self._global_manager = None

    @property
    def global_manager(self) -> GlobalAsyncCheckpointManager:
        """Get or create the global async checkpoint manager."""
        if self._global_manager is None:
            self._global_manager = GlobalAsyncCheckpointManager(timeout_secs=GLOBAL_CHECKPOINT_TIMEOUT)
        return self._global_manager

    def save_pytree(
        self,
        pytree: PyTree,
        path: str | os.PathLike,
        *,
        prefix: str,
        dtype: jnp.dtype | None = None,
        extras: dict | None = None,
        write_index: bool = True,
    ) -> str:
        """Save a PyTree with exact structure and prefix via TensorStore.

        This method preserves the original JAX PyTree definition (treedef) so
        that :meth:`load_pytree` can reconstruct the exact same structure
        without requiring a template. Array shardings (TP/FSDP) are preserved
        without performing any all-gather.

        Args:
            pytree: Arbitrary nested structure containing JAX arrays, NumPy
                arrays, and other serializable Python objects.
            path: Destination directory (local path or remote URL such as
                ``"gs://bucket/path"``).
            mesh: Optional JAX mesh used for sharding context. If provided,
                shared metadata writes are coordinated across hosts.
            prefix: Logical namespace for the saved tree (e.g. ``"model"``,
                ``"tx"``). Must be a non-empty string.
            dtype: Optional dtype to cast floating-point arrays to before
                saving. If ``None``, the original dtypes are preserved.
            extras: Optional dictionary of extra metadata stored inside the
                structure JSON file.
            write_index: Whether to write/update ``tensorstore_index.json``.
                Defaults to ``True``.

        Returns:
            The checkpoint directory path (same as *path* but normalized to a
            string).

        Raises:
            ValueError: If *prefix* is empty or not a string, or if the
                TensorStore index keys do not match the PyTree leaf keys.
            FileNotFoundError: If ``tensorstore_index.json`` is missing after
                the save completes.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        root = str(path)
        write_shared_files = fsspec_utils.should_write_shared_checkpoint_files(root)
        if write_shared_files:
            _fs.mkdir(root, exist_ok=True)

        if dtype is not None:
            pytree = jax.tree_util.tree_map(
                lambda x: x.astype(dtype) if _is_array_like(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
                pytree,
            )

        leaves, treedef = jax.tree_util.tree_flatten(pytree, is_leaf=_is_none)

        leaf_keys_tree = leaf_key_paths(pytree, prefix=prefix, is_leaf=_is_none)
        leaf_keys_full: list[str] = jax.tree_util.tree_leaves(leaf_keys_tree, is_leaf=_is_none)
        if len(leaf_keys_full) != len(leaves):
            raise ValueError(
                f"Mismatch between leaf_keys ({len(leaf_keys_full)}) and leaves ({len(leaves)}). "
                "Ensure treedef and leaves use the same is_leaf and no leaves are dropped."
            )

        arr_mask = [_is_array_like(x) for x in leaves]
        array_keys = [k for k, m in zip(leaf_keys_full, arr_mask, strict=False) if m]
        nonarray_indices = [i for i, m in enumerate(arr_mask) if not m]
        nonarray_payload = {str(i): base64.b64encode(pickle.dumps(leaves[i])).decode("utf-8") for i in nonarray_indices}

        backend = "tensorstore"
        array_relpaths: list[str] = []

        tree_serialize_leaves(
            checkpoint_dir=root,
            pytree=pytree,
            manager=self.global_manager,
            prefix=prefix,
            write_index=write_index and write_shared_files,
        )

        self.global_manager.wait_until_finished()
        if not write_shared_files:
            return root

        index_path = _fs.joinpath(root, "tensorstore_index.json")
        if not _fs.exists(index_path):
            raise FileNotFoundError(f"Missing tensorstore_index.json in {root}")
        idx = json.loads(_fs.read_text(index_path))
        arrays_info = idx.get("prefixes", {}).get(prefix, [])
        if not arrays_info:
            raise ValueError(f"No arrays recorded in index for prefix={prefix!r}")

        relpaths_from_index = [info["path"] for info in arrays_info]
        keys_from_index = [".".join(p.split("/")) for p in relpaths_from_index]
        if set(keys_from_index) != set(array_keys):
            missing = set(array_keys) - set(keys_from_index)
            extra = set(keys_from_index) - set(array_keys)
            raise ValueError(
                f"TensorStore index keys mismatch for prefix={prefix!r}. "
                f"Missing: {sorted(missing)}; Extra: {sorted(extra)}"
            )

        key_to_rel = dict(zip(keys_from_index, relpaths_from_index, strict=False))
        array_relpaths = [key_to_rel[k] for k in array_keys]

        if sum(arr_mask) != len(array_relpaths):
            raise ValueError(
                f"Structure mismatch: arr_mask expects {sum(arr_mask)} arrays, but index provided {len(array_relpaths)}."
            )

        structure = {
            "format": "pytree-structure",
            "version": __version__,
            "backend": backend,
            "prefix": prefix,
            "treedef_b64": _treedef_to_b64(treedef),
            "leaf_keys_full": leaf_keys_full,
            "arr_mask": arr_mask,
            "array_keys": array_keys,
            "array_relpaths": array_relpaths,
            "nonarray_payload": nonarray_payload,
            "safetensors_file": None,
            "extras": extras or {},
        }
        _fs.write_text(_structure_path(root, prefix), json.dumps(structure, indent=2))

        meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=extras)
        _fs.write_text(_fs.joinpath(root, "checkpoint_metadata.json"), json.dumps(meta.to_dict(), indent=2))

        return root

    def load_pytree(
        self,
        path: str | os.PathLike,
        mesh: Mesh,
        *,
        prefix: str,
        shardings: dict[str, tp.Callable] | None = None,
        sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None = None,
        partition_rules: tp.Sequence[tuple[str, PartitionSpec]] | None = None,
        dtype: jnp.dtype | None = None,
        template: PyTree | None = None,
        strict_shapes: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[PyTree, dict]:
        """Load a PyTree previously saved by :meth:`save_pytree`.

        Reads the ``pytree_structure.json`` file written during save to
        reconstruct the exact treedef, then deserializes arrays via TensorStore.

        Args:
            path: Checkpoint directory (local or remote URL).
            mesh: JAX mesh used to create shardings for loaded arrays.
            prefix: Logical namespace that matches the one used during save.
            shardings: Optional mapping from leaf key strings to explicit
                :class:`~jax.sharding.NamedSharding` objects.
            sharding_rules: Optional sequence of ``(regex_pattern,
                NamedSharding)`` pairs. For each array key the first matching
                pattern determines its sharding (fallback is fully replicated).
            dtype: Optional dtype to cast loaded arrays to.
            template: Optional PyTree template for shape coercion. When given,
                loaded arrays are matched against the template leaves by key;
                shape mismatches trigger ``ValueError`` if *strict_shapes* is
                ``True``.
            strict_shapes: Whether to raise on shape mismatches when a
                *template* is provided. Defaults to ``True``.
            callback: Optional per-array callback ``fn(array, key) -> array``
                invoked after each array is loaded.
            chunk_size: If set, arrays are loaded in batches of this size to
                reduce peak memory. Defaults to ``None`` (load all at once).

        Returns:
            A 2-tuple ``(pytree, metadata)`` where *pytree* has the exact same
            structure as the saved tree (or is coerced to *template* when
            provided), and *metadata* is the ``extras`` dict written at save
            time.

        Raises:
            ValueError: If *prefix* is empty, or if the saved prefix does not
                match the requested prefix, or if any arrays are missing.
            FileNotFoundError: If ``pytree_structure.json`` or any array
                subdirectories are missing.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        if partition_rules is not None and sharding_rules is None:
            sharding_rules = [(pat, NamedSharding(mesh=mesh, spec=spec)) for pat, spec in partition_rules]

        root = str(path)
        struct_path = _structure_path(root, prefix)
        if not _fs.exists(struct_path):
            raise FileNotFoundError(f"Missing pytree_structure.json in {root}")

        struct = json.loads(_fs.read_text(struct_path))
        if struct.get("prefix") != prefix:
            raise ValueError(
                f"Structure recorded for prefix={struct.get('prefix')!r}, "
                f"but you requested prefix={prefix!r}. Use the same prefix you saved with."
            )

        treedef = _treedef_from_b64(struct["treedef_b64"])
        leaf_keys_full: list[str] = struct["leaf_keys_full"]
        arr_mask: list[bool] = struct["arr_mask"]
        if len(arr_mask) != treedef.num_leaves:
            raise ValueError(
                f"Structure/treedef mismatch: arr_mask has {len(arr_mask)} leaves, "
                f"treedef expects {treedef.num_leaves}. The structure file may be stale "
                "or saved with a different JAX PyTree definition."
            )
        array_keys: list[str] = struct["array_keys"]
        metadata = struct.get("extras", {})

        def default_sharding():
            return NamedSharding(mesh=mesh, spec=PartitionSpec())

        relpaths: list[str] = struct["array_relpaths"]
        if len(relpaths) != len(array_keys):
            raise ValueError("array_relpaths and array_keys length mismatch")

        abs_paths = [_fs.joinpath(root, rp) for rp in relpaths]

        missing = [p for p in abs_paths if not _fs.exists(_fs.joinpath(p, ".zarray"))]
        if missing:
            idx = _fs.joinpath(root, "tensorstore_index.json")
            prefixes = []
            if _fs.exists(idx):
                idx_data = json.loads(_fs.read_text(idx))
                prefixes = sorted(list(idx_data.get("prefixes", {}).keys()))
            raise FileNotFoundError(
                f"{len(missing)} arrays missing (example: {missing[0]}). "
                f"Check that the prefix you pass matches the one saved. "
                f"Available prefixes in this directory: {prefixes}"
            )

        if sharding_rules is not None:
            import re as _re

            apply_shardings = []
            for k in array_keys:
                found = None
                for pattern, sharding in sharding_rules:
                    if _re.search(pattern, k.replace(".", "/")):
                        found = sharding
                        break
                apply_shardings.append(found if found is not None else default_sharding())
        else:
            apply_shardings = [
                shardings.get(k, default_sharding()) if shardings else default_sharding() for k in array_keys
            ]

        if chunk_size is None or chunk_size <= 0:
            array_leaves = self.global_manager.deserialize_with_paths(shardings=apply_shardings, paths=abs_paths)
            self.global_manager.wait_until_finished()
            expected_arrays = sum(arr_mask)
            if len(array_leaves) != expected_arrays:
                raise ValueError(
                    f"Loaded {len(array_leaves)} arrays but structure expects {expected_arrays}. "
                    "Index or structure may be stale."
                )
            if dtype is not None:
                array_leaves = [jnp.asarray(x, dtype=dtype) for x in array_leaves]
            if callback is not None:
                array_leaves = [callback(arr, key) for arr, key in zip(array_leaves, array_keys, strict=False)]
        else:
            array_leaves = []
            expected_arrays = sum(arr_mask)
            for start in range(0, len(abs_paths), chunk_size):
                end = min(start + chunk_size, len(abs_paths))
                chunk_paths = abs_paths[start:end]
                chunk_shardings = apply_shardings[start:end]
                chunk_keys = array_keys[start:end]
                chunk_arrays = self.global_manager.deserialize_with_paths(shardings=chunk_shardings, paths=chunk_paths)
                self.global_manager.wait_until_finished()
                if dtype is not None:
                    chunk_arrays = [jnp.asarray(x, dtype=dtype) for x in chunk_arrays]
                if callback is not None:
                    chunk_arrays = [callback(arr, key) for arr, key in zip(chunk_arrays, chunk_keys, strict=False)]
                array_leaves.extend(chunk_arrays)
            if len(array_leaves) != expected_arrays:
                raise ValueError(
                    f"Loaded {len(array_leaves)} arrays but structure expects {expected_arrays}. "
                    "Index or structure may be stale."
                )

        if template is None:
            leaves_full = [None] * len(leaf_keys_full)
            it = iter(array_leaves)
            nonarray_payload: dict[str, str] = struct.get("nonarray_payload", {})
            for i, is_arr in enumerate(arr_mask):
                if is_arr:
                    leaves_full[i] = next(it)
                else:
                    payload_b64 = nonarray_payload.get(str(i))
                    if payload_b64 is None:
                        raise ValueError(f"Missing non-array payload for leaf index {i}")
                    leaves_full[i] = pickle.loads(base64.b64decode(payload_b64))
            pytree = jax.tree_util.tree_unflatten(treedef, leaves_full)
            return pytree, metadata

        saved_arrays_by_key = {k: v for k, v in zip(array_keys, array_leaves, strict=False)}

        tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template, is_leaf=_is_none)
        tpl_leaf_keys_tree = leaf_key_paths(template, prefix=prefix, is_leaf=_is_none)
        tpl_leaf_keys_full: list[str] = jax.tree_util.tree_leaves(tpl_leaf_keys_tree, is_leaf=_is_none)
        tpl_arr_mask = [_is_array_like(x) for x in tpl_leaves]

        def _coerce_or_fallback(loaded, expected, key):
            if not (_is_array_like(loaded) and _is_array_like(expected)):
                return loaded
            if loaded.shape == expected.shape:
                return loaded
            if not strict_shapes and (loaded.ndim == expected.ndim + 1) and (loaded.shape[1:] == expected.shape):
                return loaded[0]
            if not strict_shapes and np.prod(loaded.shape) == np.prod(expected.shape):
                return jnp.reshape(loaded, expected.shape)
            if strict_shapes:
                raise ValueError(f"Array shape mismatch for key '{key}': got {loaded.shape}, expected {expected.shape}.")
            return expected

        tpl_leaves_full = [None] * len(tpl_leaf_keys_full)
        for i, key in enumerate(tpl_leaf_keys_full):
            if tpl_arr_mask[i]:
                expected = tpl_leaves[i]
                loaded = saved_arrays_by_key.get(key)
                if loaded is None:
                    if strict_shapes:
                        raise KeyError(f"Missing array for key '{key}' in checkpoint.")
                    tpl_leaves_full[i] = expected
                else:
                    tpl_leaves_full[i] = _coerce_or_fallback(loaded, expected, key)
            else:
                tpl_leaves_full[i] = tpl_leaves[i]

        pytree = jax.tree_util.tree_unflatten(tpl_treedef, tpl_leaves_full)
        return pytree, metadata
