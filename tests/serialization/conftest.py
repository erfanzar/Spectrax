# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serialization-test fixtures for TPU."""

from __future__ import annotations

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@pytest.fixture
def mesh():
    """A 2x2 mesh over all local TPU devices."""
    devices = np.array(jax.devices()).reshape(2, 2)
    return Mesh(devices, ("x", "y"))


@pytest.fixture
def tmp_checkpoint_dir():
    """Temporary directory for checkpoint I/O."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_pytree(mesh):
    """Nested pytree with arrays and non-array leaves."""
    sh = NamedSharding(mesh, PartitionSpec("x", "y"))
    arr = jax.device_put(jnp.arange(16).reshape(4, 4), sh)
    return {
        "layer0": {
            "weight": arr,
            "bias": jnp.ones(4),
        },
        "step": 42,
        "name": "test",
    }
