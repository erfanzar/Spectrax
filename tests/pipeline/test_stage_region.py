# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`sxstage_region` markers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import spectrax as spx
from spectrax.runtime.mpmd import sxjit
from spectrax.runtime.mpmd.markers import stage_region_specs
from spectrax.runtime.types import MpMdMesh


def test_stage_region_is_identity_for_eager_jit_and_grad():
    """Stage region is identity for eager jit and grad."""
    region = spx.sxstage_region("encoder", schedule=spx.GPipe(microbatches=2))

    def body(x):
        """Loop body function."""
        return region(lambda y: y * y + 1.0)(x).sum()

    x = jnp.arange(4, dtype=jnp.float32)
    assert np.allclose(np.asarray(body(x)), np.asarray((x * x + 1.0).sum()))
    assert np.allclose(np.asarray(jax.jit(body)(x)), np.asarray(body(x)))
    assert np.allclose(np.asarray(jax.grad(body)(x)), np.asarray(2.0 * x))


def test_stage_region_markers_and_metadata_are_in_jaxpr():
    """Stage region markers and metadata are in jaxpr."""
    region = spx.sxstage_region(
        "decoder",
        schedule=spx.DualPipeV(microbatches=4),
        batch_argnums=(1,),
        static_argnums=(2,),
        donate_argnums=(0,),
    )

    def body(x):
        """Loop body function."""
        return region(lambda y: y + 1)(x)

    jaxpr = jax.make_jaxpr(body)(jnp.ones((2,), dtype=jnp.float32)).jaxpr
    primitive_names = [eqn.primitive.name for eqn in jaxpr.eqns]
    assert "sxstage_region_enter" in primitive_names
    assert "sxstage_region_exit" in primitive_names

    specs = stage_region_specs(jaxpr)
    assert specs
    assert {spec.name for spec in specs} == {"decoder"}
    assert {spec.schedule_name for spec in specs} == {"DualPipeV"}
    assert {spec.microbatches for spec in specs} == {4}
    assert {spec.batch_argnums for spec in specs} == {(1,)}
    assert {spec.static_argnums for spec in specs} == {(2,)}
    assert {spec.donate_argnums for spec in specs} == {(0,)}


def test_stage_region_in_scheduled_sxjit_fails_loudly_until_region_dispatch_exists():
    """Stage region in scheduled sxjit fails loudly until region dispatch exists."""
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1)
    mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    region = spx.sxstage_region("encoder", schedule=spx.GPipe(microbatches=1))

    @sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=1))
    def body(x):
        """Loop body function."""
        return region(lambda y: y + 1)(x).sum()

    with pytest.raises(NotImplementedError, match="sxstage_region markers were found"):
        body(jnp.ones((1,), dtype=jnp.float32))
