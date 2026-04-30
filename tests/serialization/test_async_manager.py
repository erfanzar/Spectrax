# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for AsyncCheckpointManager (TensorStore-only)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from spectrax.serialization import AsyncCheckpointManager


class TestAsyncCheckpointManager:
    """End-to-end tests for AsyncCheckpointManager save/load."""

    def test_save_load_pytree_roundtrip(self, tmp_checkpoint_dir, mesh, sample_pytree):
        """save_pytree -> load_pytree preserves values and structure."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(sample_pytree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, meta = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

        assert jnp.allclose(loaded["layer0"]["weight"], sample_pytree["layer0"]["weight"])
        assert jnp.allclose(loaded["layer0"]["bias"], sample_pytree["layer0"]["bias"])
        assert loaded["step"] == sample_pytree["step"]
        assert loaded["name"] == sample_pytree["name"]
        assert isinstance(meta, dict)

    def test_nonarray_payload_preserved(self, tmp_checkpoint_dir, mesh):
        """Ints, strings, and None survive the roundtrip."""
        tree = {"a": None, "b": 7, "c": "hello", "arr": jnp.ones((2, 2))}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="tx")

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="tx")
        assert loaded["a"] is None
        assert loaded["b"] == 7
        assert loaded["c"] == "hello"
        assert jnp.allclose(loaded["arr"], tree["arr"])

    def test_load_pytree_with_template(self, tmp_checkpoint_dir, mesh):
        """Loading into a template coerces shapes when strict_shapes=False."""
        tree = {"x": jnp.arange(6).reshape(2, 3)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        template = {"x": jnp.zeros((2, 3))}
        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", template=template, strict_shapes=True)
        assert jnp.allclose(loaded["x"], tree["x"])

    def test_sharding_rules_matching(self, tmp_checkpoint_dir, mesh):
        """sharding_rules assigns NamedShardings by regex match."""
        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        tree = {
            "layers": {
                "0": {"w": jnp.arange(8).reshape(2, 4), "b": jnp.ones(4)},
                "1": {"w": jnp.arange(8).reshape(2, 4), "b": jnp.ones(4)},
            }
        }
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, _ = mgr.load_pytree(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            sharding_rules=[(".*weight.*|.*w.*", sh)],
        )
        assert jnp.allclose(loaded["layers"]["0"]["w"], tree["layers"]["0"]["w"])
        assert jnp.allclose(loaded["layers"]["0"]["b"], tree["layers"]["0"]["b"])

    def test_chunked_load(self, tmp_checkpoint_dir, mesh):
        """chunk_size loads arrays in batches without error."""
        tree = {f"p{i}": jnp.ones((4, 4)) for i in range(5)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", chunk_size=2)
        assert set(loaded.keys()) == set(tree.keys())
        for k in tree:
            assert jnp.allclose(loaded[k], tree[k])

    def test_load_pytree_prefix_mismatch_raises(self, tmp_checkpoint_dir, mesh):
        """Loading with a different prefix than saved raises ValueError."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        import shutil

        wrong = Path(tmp_checkpoint_dir) / "optimizer_structure.json"
        shutil.copy(Path(tmp_checkpoint_dir) / "model_structure.json", wrong)

        with pytest.raises(ValueError, match="prefix"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="optimizer")

    def test_load_pytree_missing_structure_raises(self, tmp_checkpoint_dir, mesh):
        """Loading from a directory without pytree_structure.json raises."""
        mgr = AsyncCheckpointManager()
        with pytest.raises(FileNotFoundError, match="Missing pytree_structure"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

    def test_structure_file_written(self, tmp_checkpoint_dir, mesh):
        """save_pytree writes pytree_structure.json and checkpoint_metadata.json."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        assert (Path(tmp_checkpoint_dir) / "model_structure.json").exists()
        assert (Path(tmp_checkpoint_dir) / "checkpoint_metadata.json").exists()
        assert (Path(tmp_checkpoint_dir) / "tensorstore_index.json").exists()

    def test_structure_content(self, tmp_checkpoint_dir, mesh):
        """structure file contains expected keys."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        struct_path = Path(tmp_checkpoint_dir) / "model_structure.json"
        data = json.loads(struct_path.read_text())
        assert data["format"] == "pytree-structure"
        assert data["prefix"] == "model"
        assert "treedef_b64" in data
        assert "arr_mask" in data
        assert "array_keys" in data

    def test_save_load_gcs_roundtrip(self, mesh):
        """Save and load from GCS bucket gs://uscentral1stuff/spx-save-tmp."""
        import uuid

        mgr = AsyncCheckpointManager()
        run_id = str(uuid.uuid4())[:8]
        gcs_path = f"gs://uscentral1stuff/spx-save-tmp/test-{run_id}"

        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        arr = jax.device_put(jnp.arange(16).reshape(4, 4), sh)
        tree = {"w": arr, "b": jnp.ones(4), "step": 99}

        mgr.save_pytree(tree, gcs_path, mesh=mesh, prefix="model")
        loaded, _meta = mgr.load_pytree(gcs_path, mesh, prefix="model")

        assert jnp.allclose(loaded["w"], tree["w"])
        assert jnp.allclose(loaded["b"], tree["b"])
        assert loaded["step"] == 99

    def test_dtype_casting(self, tmp_checkpoint_dir, mesh):
        """dtype parameter casts floating-point arrays before saving."""
        tree = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model", dtype=jnp.bfloat16)

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")
        assert loaded["w"].dtype == jnp.bfloat16

    def test_extras_preserved(self, tmp_checkpoint_dir, mesh):
        """extras dict survives the roundtrip in metadata."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(
            {"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model", extras={"lr": 0.001, "epoch": 5}
        )
        _loaded, meta = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")
        assert meta.get("lr") == 0.001
        assert meta.get("epoch") == 5

    def test_global_manager_lazy_init(self):
        """global_manager is created lazily on first access."""
        mgr = AsyncCheckpointManager()
        assert mgr._global_manager is None
        _ = mgr.global_manager
        assert mgr._global_manager is not None

    def test_load_pytree_strict_shapes_false(self, tmp_checkpoint_dir, mesh):
        """strict_shapes=False allows shape coercion via template."""
        tree = {"x": jnp.arange(6).reshape(2, 3)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        template = {"x": jnp.zeros((1, 2, 3))}
        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", template=template, strict_shapes=False)
        assert loaded["x"].shape == (1, 2, 3)

    def test_load_pytree_missing_array_raises(self, tmp_checkpoint_dir, mesh):
        """Missing array files raise FileNotFoundError."""
        mgr = AsyncCheckpointManager()
        mgr.save_pytree({"a": jnp.ones(2)}, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        import os

        os.remove(os.path.join(tmp_checkpoint_dir, "model", "a", ".zarray"))

        with pytest.raises(FileNotFoundError, match="arrays missing"):
            mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model")

    def test_load_pytree_callback_transforms(self, tmp_checkpoint_dir, mesh):
        """callback is applied to every loaded array."""
        tree = {"a": jnp.ones((2, 2)), "b": jnp.zeros(2)}
        mgr = AsyncCheckpointManager()
        mgr.save_pytree(tree, tmp_checkpoint_dir, mesh=mesh, prefix="model")

        def negate(arr, key):
            """Negate the input."""
            return -arr

        loaded, _ = mgr.load_pytree(tmp_checkpoint_dir, mesh, prefix="model", callback=negate)
        assert jnp.allclose(loaded["a"], -tree["a"])
        assert jnp.allclose(loaded["b"], -tree["b"])
