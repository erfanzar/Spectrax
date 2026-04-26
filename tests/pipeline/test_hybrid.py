# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid SPMD/MPMD runtime tests.

Models with a homogeneous core (repeated blocks) and heterogeneous edges
(embed / head) should hit the hybrid path and produce numerically identical
results to the pure MPMD fallback.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from spectrax.runtime.spmd.api import pipeline_call
from spectrax.runtime.types import MpMdMesh
from spectrax.runtime.types.stage import PipelineStage

_D = 4
_M = 2
_BATCH = 4


def _make_embed_stage(d: int, rng: jax.Array) -> PipelineStage:
    """Prefix stage: (B, D) -> (B, 2D)."""
    key_w, key_b = jax.random.split(rng)
    w = jax.random.normal(key_w, (d, 2 * d))
    b = jax.random.normal(key_b, (2 * d,))
    params = {"w": w, "b": b}

    def fn(p, s, x):
        return jax.nn.relu(x @ p["w"] + p["b"]), ()

    return PipelineStage(fn=fn, parameters=params, init_state=())


def _make_block_stage(d: int, rng: jax.Array) -> PipelineStage:
    """Core stage: (B, D) -> (B, D)."""
    key_w, key_b = jax.random.split(rng)
    w = jax.random.normal(key_w, (d, d))
    b = jax.random.normal(key_b, (d,))
    params = {"w": w, "b": b}

    def fn(p, s, x):
        return jax.nn.relu(x @ p["w"] + p["b"]), ()

    return PipelineStage(fn=fn, parameters=params, init_state=())


def _make_head_stage(d: int, rng: jax.Array) -> PipelineStage:
    """Suffix stage: (B, 2D) -> (B, D)."""
    key_w, key_b = jax.random.split(rng)
    w = jax.random.normal(key_w, (2 * d, d))
    b = jax.random.normal(key_b, (d,))
    params = {"w": w, "b": b}

    def fn(p, s, x):
        return x @ p["w"] + p["b"], ()

    return PipelineStage(fn=fn, parameters=params, init_state=())


def _loss_fn(out, y):
    return ((out - y) ** 2).mean()


@pytest.fixture(scope="module")
def hybrid_mesh():
    """4-rank mesh for embed + 2 blocks + head."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip(f"need 4 devices; have {len(devs)}")
    return MpMdMesh(Mesh(devs, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def hybrid_stages():
    """embed -> block -> block -> head."""
    return (
        _make_embed_stage(_D, jax.random.PRNGKey(1)),
        _make_block_stage(2 * _D, jax.random.PRNGKey(2)),
        _make_block_stage(2 * _D, jax.random.PRNGKey(3)),
        _make_head_stage(_D, jax.random.PRNGKey(4)),
    )


@pytest.fixture(scope="module")
def xy():
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    return x, y


def _reference_forward(stages, x):
    """Sequential forward through all stages."""
    for stage in stages:
        x, _ = stage.fn(stage.parameters, stage.init_state, x)
    return x


def _reference_loss_and_grads(stages, x, y):
    """Single-device loss + per-stage grads."""

    def loss_fn(params_list):
        cur = x
        for i, stage in enumerate(stages):
            p = params_list[i]
            cur, _ = stage.fn(p, stage.init_state, cur)
        return ((cur - y) ** 2).mean()

    params_list = tuple(s.parameters for s in stages)
    loss, grads = jax.value_and_grad(loss_fn)(params_list)
    return loss, grads


def test_hybrid_loss_matches_reference(hybrid_stages, xy, hybrid_mesh):
    """Hybrid path loss matches single-device reference."""
    x, y = xy
    ref_loss, _ = _reference_loss_and_grads(hybrid_stages, x, y)

    loss, _grads = pipeline_call(
        hybrid_stages,
        (x, y),
        mesh=hybrid_mesh,
        mode="train",
        loss_fn=_loss_fn,
    )
    assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)


def test_hybrid_grads_match_reference(hybrid_stages, xy, hybrid_mesh):
    """Hybrid path grads match single-device reference."""
    x, y = xy
    _, ref_grads = _reference_loss_and_grads(hybrid_stages, x, y)

    _loss, grads = pipeline_call(
        hybrid_stages,
        (x, y),
        mesh=hybrid_mesh,
        mode="train",
        loss_fn=_loss_fn,
    )

    assert len(grads) == 4
    for i, pg in enumerate(grads):
        for path in pg:
            assert jnp.allclose(pg[path], ref_grads[i][path], atol=1e-3, rtol=1e-3), (
                f"Grad mismatch at stage {i}, path {path!r}"
            )


def test_homogeneous_forward_passes_extra_inputs_to_stages():
    """Forward-mode homogeneous pipeline should not silently drop aux inputs."""
    devs = jax.devices()[:2]
    if len(devs) < 2:
        pytest.skip(f"need 2 devices; have {len(devs)}")
    mesh = MpMdMesh(Mesh(devs, axis_names=("pp",)), "pp")

    def fn(p, s, x, mask):
        del s
        return x + p["scale"] * mask, ()

    stages = tuple(PipelineStage(fn=fn, parameters={"scale": jnp.asarray(1.0)}, init_state=()) for _ in range(2))
    x = jnp.ones((_BATCH, _D), dtype=jnp.float32)
    mask = jnp.arange(_BATCH * _D, dtype=jnp.float32).reshape(_BATCH, _D)

    out, _state = pipeline_call(stages, (x, mask), mesh=mesh, microbatches=_M, mode="forward")

    expected = (x + 2 * mask).reshape((_M, _BATCH // _M, _D))
    assert jnp.allclose(out, expected)
