# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Checkpointer high-level checkpoint manager."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from spectrax.serialization import Checkpointer, CheckpointInterval


class TestCheckpointer:
    """End-to-end tests for Checkpointer save/load/discovery."""

    def test_save_load_pytree(self, tmp_checkpoint_dir, mesh, sample_pytree):
        """Checkpointer save_pytree -> load_pytree roundtrip."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        cp.save_pytree(sample_pytree, prefix="model", step=10, mesh=mesh)

        loaded, meta = cp.load_pytree(mesh, prefix="model")
        assert jnp.allclose(loaded["layer0"]["weight"], sample_pytree["layer0"]["weight"])
        assert meta.get("step") == 10

    def test_discover_latest(self, tmp_checkpoint_dir, mesh):
        """discover_latest finds the most recent checkpoint by metadata."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"w": jnp.ones((2, 2))}
        cp.save_pytree(tree, prefix="model", step=10, mesh=mesh)
        cp.save_pytree(tree, prefix="model", step=20, mesh=mesh)

        _loaded, meta = cp.load_pytree(mesh, prefix="model", discover_latest=True)
        assert meta.get("step") == 20

    def test_discover_latest_with_explicit_path(self, tmp_checkpoint_dir, mesh):
        """Loading from an explicit path bypasses discovery."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"w": jnp.ones((2, 2))}
        cp.save_pytree(tree, prefix="model", step=10, mesh=mesh)
        cp.save_pytree(tree, prefix="model", step=20, mesh=mesh)

        explicit = str(Path(tmp_checkpoint_dir) / "run-10")
        _loaded, meta = cp.load_pytree(mesh, prefix="model", path=explicit, discover_latest=False)
        assert meta.get("step") == 10

    def test_load_treedef_false_returns_flat_dict(self, tmp_checkpoint_dir, mesh):
        """load_treedef=False flattens the result to dotted keys."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"layer0": {"weight": jnp.ones((2, 2)), "bias": jnp.zeros(2)}}
        cp.save_pytree(tree, prefix="model", step=10, mesh=mesh)

        loaded, _ = cp.load_pytree(mesh, prefix="model", load_treedef=False, discover_latest=True)
        assert isinstance(loaded, dict)
        assert "layer0.weight" in loaded
        assert "layer0.bias" in loaded
        assert jnp.allclose(loaded["layer0.weight"], tree["layer0"]["weight"])

    def test_temporary_checkpoint_metadata(self, tmp_checkpoint_dir, mesh):
        """Temporary checkpoints write is_temporary=true in metadata."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        cp.save_pytree({"w": jnp.ones(2)}, prefix="model", step=10, mesh=mesh, temporary=True)

        meta_path = Path(tmp_checkpoint_dir) / "run-10" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["is_temporary"] is True
        assert meta["step"] == 10

    def test_on_step_force_save(self, tmp_checkpoint_dir, mesh):
        """on_step with force=True always saves."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=100)],
        )
        tree = {"w": jnp.ones((2, 2))}
        cp.on_step(mesh, tree, force=True, step=5, prefix="model")
        cp.wait_until_finished()

        meta_path = Path(tmp_checkpoint_dir) / "run-5" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["step"] == 5

    def test_on_step_policy_triggers(self, tmp_checkpoint_dir, mesh):
        """on_step saves when step matches the policy interval."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"w": jnp.ones((2, 2))}
        cp.on_step(mesh, tree, step=10, prefix="model")
        cp.wait_until_finished()

        meta_path = Path(tmp_checkpoint_dir) / "run-10" / "metadata.json"
        assert meta_path.exists()

    def test_on_step_skips_duplicate_step(self, tmp_checkpoint_dir, mesh):
        """Duplicate step saves are skipped unless force=True."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"w": jnp.ones((2, 2))}
        cp.on_step(mesh, tree, step=10, prefix="model")
        cp.on_step(mesh, tree, step=10, prefix="model")
        cp.wait_until_finished()

        dirs = [d for d in Path(tmp_checkpoint_dir).iterdir() if d.is_dir()]
        assert len(dirs) == 1

    def test_on_step_callbacks_executed(self, tmp_checkpoint_dir, mesh):
        """true_callbacks are called when a save is triggered."""
        called_with = {}

        def cb(dest, m, meta):
            """Callback function."""
            called_with["dest"] = dest
            called_with["step"] = meta["step"]

        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        cp.on_step(mesh, None, step=10, true_callbacks=[cb], prefix="model")
        cp.wait_until_finished()

        assert called_with["step"] == 10
        assert "run-10" in called_with["dest"]

    def test_wait_until_finished(self, tmp_checkpoint_dir, mesh):
        """wait_until_finished does not raise."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        cp.save_pytree({"w": jnp.ones(2)}, prefix="model", step=10, mesh=mesh)
        cp.wait_until_finished()

    def test_load_pytree_discover_raise_false(self, tmp_checkpoint_dir, mesh):
        """discover_raise=False returns None when no checkpoint exists."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        loaded, _meta = cp.load_pytree(mesh, prefix="model", discover_latest=True, discover_raise=False)
        assert loaded is None

    def test_gcs_save_load(self, mesh):
        """Checkpointer save/load roundtrip on GCS."""
        import uuid

        run_id = str(uuid.uuid4())[:8]
        gcs_base = f"gs://uscentral1stuff/spx-save-tmp/test-checkpointer-{run_id}"

        cp = Checkpointer(
            base_path=gcs_base,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        tree = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
        cp.save_pytree(tree, prefix="model", step=10, mesh=mesh)

        loaded, meta = cp.load_pytree(mesh, prefix="model", discover_latest=False, path=f"{gcs_base}/run-10")
        assert jnp.allclose(loaded["w"], tree["w"])
        assert meta.get("step") == 10
        cp.wait_until_finished()

    def test_invalid_step_policies_unsorted(self):
        """Unsorted step_policies raise ValueError."""
        with pytest.raises(ValueError, match="sorted"):
            Checkpointer(
                base_path="/tmp/fake",
                save_interval=None,
                step_policies=[
                    CheckpointInterval(every=100, until=500),
                    CheckpointInterval(every=50, until=200),
                ],
            )

    def test_invalid_step_policies_none_not_last(self):
        """until=None on a non-final policy raises ValueError."""
        with pytest.raises(ValueError, match="last"):
            Checkpointer(
                base_path="/tmp/fake",
                save_interval=None,
                step_policies=[
                    CheckpointInterval(every=100),
                    CheckpointInterval(every=50, until=500),
                ],
            )

    def test_load_pytree_with_sharding_rules(self, tmp_checkpoint_dir, mesh):
        """sharding_rules passed through to AsyncCheckpointManager."""
        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=None,
            step_policies=[CheckpointInterval(every=10)],
        )
        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        arr = jax.device_put(jnp.arange(16).reshape(4, 4), sh)
        cp.save_pytree({"weight": arr, "bias": jnp.ones(4)}, prefix="model", step=10, mesh=mesh)

        loaded, _ = cp.load_pytree(mesh, prefix="model", sharding_rules=[(".*weight.*", sh)])
        assert jnp.allclose(loaded["weight"], arr)
        assert jnp.allclose(loaded["bias"], jnp.ones(4))

    def test_on_step_time_based_save(self, tmp_checkpoint_dir, mesh):
        """on_step triggers time-based saves when elapsed >= save_interval."""
        import time
        from datetime import timedelta

        cp = Checkpointer(
            base_path=tmp_checkpoint_dir,
            save_interval=timedelta(milliseconds=1),
            step_policies=[CheckpointInterval(every=100000)],
        )
        tree = {"w": jnp.ones(2)}
        time.sleep(0.01)
        cp.on_step(mesh, tree, step=1, prefix="model")
        cp.wait_until_finished()

        meta_path = Path(tmp_checkpoint_dir) / "run-1" / "metadata.json"
        assert meta_path.exists()
