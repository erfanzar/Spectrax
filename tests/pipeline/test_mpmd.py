# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MPMD runtime tests: heterogeneous stages, loss + grad parity.

Module-scoped fixtures reuse the homogeneous model + reference grads
across schedules so we don't recompile for every parametrize entry.

Homogeneous runs use 2 stages x 2 microbatches to keep the per-call MPMD
compile small. The heterogeneous test uses 4 stages because it needs
distinct stage shapes (Block -> Expand -> Wide -> Contract).

There is intentionally no end-to-end ``sxcall`` test for DualPipeV: the
V-shape schedule places the final logical stage on physical rank 0 (not
rank n-1), which conflicts with the current MPMD runtime's "loss is
computed at rank n-1" assumption. DualPipeV correctness is covered
structurally in ``test_schedules.py``; end-to-end integration will need
the Tier-2 runtime (single jitted step) to route loss to the right rank.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential
from spectrax.runtime.mpmd import collect_task_times_ms, sxcall, sxgrad, sxjit, sxstage_iter, sxvalue_and_grad
from spectrax.runtime.schedules import Eager1F1B, GPipe, InterleavedH1, KimiK2, Std1F1B, ZeroBubbleH1
from spectrax.runtime.types import MpMdMesh

_N = 2
_HET_N = 4
_D = 4
_M = 2
_BATCH = 4


class _Block(spx.Module):
    """Homogeneous stage: single Linear followed by ReLU, same input/output width."""

    def __init__(self, d, *, rngs):
        """Build a ``(d, d)`` Linear layer."""
        super().__init__()
        self.fc = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Apply the linear layer followed by a ReLU."""
        return jax.nn.relu(self.fc(x))


class _Expand(spx.Module):
    """Shape-changing heterogeneous stage: ``(B, D) -> (B, 2D)``."""

    def __init__(self, d, *, rngs):
        """Build a ``(d, 2d)`` Linear layer."""
        super().__init__()
        self.fc = nn.Linear(d, 2 * d, rngs=rngs)

    def forward(self, x):
        """Apply the linear layer followed by a GELU."""
        return jax.nn.gelu(self.fc(x))


class _Wide(spx.Module):
    """Different-class heterogeneous stage: ``(B, 2D) -> (B, 2D)`` through two Linears."""

    def __init__(self, d, *, rngs):
        """Build two ``(d, d)`` Linear layers separated by a SiLU."""
        super().__init__()
        self.fc1 = nn.Linear(d, d, rngs=rngs)
        self.fc2 = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        """Apply ``fc2(silu(fc1(x)))``."""
        return self.fc2(jax.nn.silu(self.fc1(x)))


class _Contract(spx.Module):
    """Shape-changing heterogeneous stage: ``(B, 2D) -> (B, D)``."""

    def __init__(self, d, *, rngs):
        """Build a ``(2d, d)`` Linear layer."""
        super().__init__()
        self.fc = nn.Linear(2 * d, d, rngs=rngs)

    def forward(self, x):
        """Apply the linear projection back to the smaller width."""
        return self.fc(x)


class _TpBlock(spx.Module):
    """Tiny block with TP-sharded parameter metadata for composite MPMD tests."""

    def __init__(self, scale: float):
        super().__init__()
        base = jnp.arange(_D * _D, dtype=jnp.float32).reshape(_D, _D)
        self.weight = spx.Parameter((base + 1.0) * scale, axis_names=(None, "tp"))
        self.bias = spx.Parameter(jnp.zeros((_D,), dtype=jnp.float32), axis_names=("tp",))

    def forward(self, x):
        return x @ self.weight + self.bias


class _UnannotatedBlock(spx.Module):
    """Tiny block whose runtime value sharding is stronger than metadata."""

    def __init__(self, scale: float):
        super().__init__()
        base = jnp.arange(_D * _D, dtype=jnp.float32).reshape(_D, _D)
        self.weight = spx.Parameter((base + 1.0) * scale)

    def forward(self, x):
        return x @ self.weight


def _loss_fn(out, y):
    """Mean-squared-error loss; module-level so ``id(loss_fn)`` stays stable for cache hits."""
    return ((out - y) ** 2).mean()


def _full_loss(m, x, y):
    """Single-device baseline loss computed on the un-pipelined model."""
    return ((m(x) - y) ** 2).mean()


@pytest.fixture(scope="module")
def mpmd_mesh():
    """Module-scoped ``MpMdMesh`` with ``_N`` pipeline ranks."""
    devs = jax.devices()[:_N]
    if len(devs) < _N:
        pytest.skip(f"need {_N} devices; have {len(devs)}")
    return MpMdMesh(Mesh(devs, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def het_mpmd_mesh():
    """Module-scoped ``MpMdMesh`` sized for the heterogeneous test (``_HET_N`` ranks)."""
    devs = jax.devices()[:_HET_N]
    if len(devs) < _HET_N:
        pytest.skip(f"need {_HET_N} devices; have {len(devs)}")
    return MpMdMesh(Mesh(devs, axis_names=("pp",)), "pp")


@pytest.fixture(scope="module")
def hom_model():
    """Module-scoped homogeneous ``PipelineSequential`` of ``_N`` ``_Block`` stages."""
    return PipelineSequential(*[_Block(_D, rngs=spx.Rngs(i + 1)) for i in range(_N)])


@pytest.fixture(scope="module")
def xy():
    """Module-scoped ``(x, y)`` tensors sampled once per module."""
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    return x, y


@pytest.fixture(scope="module")
def hom_reference(hom_model, xy):
    """Single-device loss + grads for the homogeneous model."""
    x, y = xy
    loss_val, grads_full = spx.value_and_grad(_full_loss)(hom_model, x, y)
    return loss_val, grads_full


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B, ZeroBubbleH1, Eager1F1B])
def test_mpmd_homogeneous_loss_matches_single_device(schedule_cls, hom_model, xy, mpmd_mesh, hom_reference):
    """MPMD loss on homogeneous stages matches single-device run."""
    x, y = xy
    ref_loss, _ = hom_reference
    loss, _grads = sxcall(
        hom_model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=schedule_cls(microbatches=_M),
        loss_fn=_loss_fn,
    )
    assert jnp.allclose(loss, ref_loss, atol=1e-4, rtol=1e-4)


def test_mpmd_homogeneous_grads_match_single_device(hom_model, xy, mpmd_mesh, hom_reference):
    """MPMD gradients on homogeneous stages match single-device grads."""
    x, y = xy
    _, ref_full = hom_reference
    _loss, pipeline_grads = sxcall(
        hom_model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )
    for i, pg in enumerate(pipeline_grads):
        prefix = f"{i}."
        for c, path, pipe_leaf in pg.items():
            if c != "parameters":
                continue
            ref_leaf = ref_full.get("parameters", prefix + path)
            assert jnp.allclose(pipe_leaf, ref_leaf, atol=1e-3, rtol=1e-3), f"Grad mismatch at stage {i}, path {path!r}"


@pytest.fixture(scope="module")
def het_model():
    """Module-scoped heterogeneous ``PipelineSequential`` (Block, Expand, Wide, Contract)."""
    return PipelineSequential(
        _Block(_D, rngs=spx.Rngs(1)),
        _Expand(_D, rngs=spx.Rngs(2)),
        _Wide(2 * _D, rngs=spx.Rngs(3)),
        _Contract(_D, rngs=spx.Rngs(4)),
    )


def test_mpmd_heterogeneous_stages_run_and_grads_match(het_model, xy, het_mpmd_mesh):
    """MPMD runs heterogeneous stages and matches single-device grads.

    Combines the prior shape-check + grad-parity tests so we only pay
    the heterogeneous compile once.
    """
    x, y = xy
    _, ref_full = spx.value_and_grad(_full_loss)(het_model, x, y)
    ref_loss = _full_loss(het_model, x, y)

    loss, grads = sxcall(
        het_model,
        (x, y),
        mesh=het_mpmd_mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )

    assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)

    assert len(grads) == _HET_N
    assert grads[0].get("parameters", "fc.weight").shape == (_D, _D)
    assert grads[1].get("parameters", "fc.weight").shape == (_D, 2 * _D)
    assert "fc1" in grads[2]["parameters"]
    assert "fc2" in grads[2]["parameters"]
    assert grads[3].get("parameters", "fc.weight").shape == (2 * _D, _D)

    for i, pg in enumerate(grads):
        prefix = f"{i}."
        for c, path, pipe_leaf in pg.items():
            if c != "parameters":
                continue
            ref_leaf = ref_full.get("parameters", prefix + path)
            assert jnp.allclose(pipe_leaf, ref_leaf, atol=1e-3, rtol=1e-3), f"Grad mismatch at stage {i}, path {path!r}"


def test_mpmd_with_composite_mesh(hom_model, xy):
    """2-stage MpMdMesh on a (pp=2, dp=2) mesh exercises the sub_sharding path."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip("need 4 devices for composite mesh test")
    composite = MpMdMesh(
        Mesh(np.array(devs).reshape(2, 2), ("pp", "dp")),
        "pp",
    )

    ref_mesh = MpMdMesh(Mesh(devs[:2], ("pp",)), "pp")
    x, y = xy
    ref_loss, _ = sxcall(
        hom_model,
        (x, y),
        mesh=ref_mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )

    loss, _grads = sxcall(
        hom_model,
        (x, y),
        mesh=composite,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )
    assert np.allclose(jax.device_get(loss), jax.device_get(ref_loss), atol=1e-5, rtol=1e-5)


def test_mpmd_interleaved_virtual_stages_match_single_device(xy, mpmd_mesh):
    """MPMD with InterleavedH1 (v=2) on a 4-logical-stage model matches ref.

    Builds a PipelineSequential of ``V*N = 4`` logical stages (N=2, V=2),
    runs it through MPMD's virtual-stage path, and compares loss + grads
    against a single-device reference.
    """
    x, y = xy
    model = PipelineSequential(*[_Block(_D, rngs=spx.Rngs(i + 1)) for i in range(4)])

    ref_loss, ref_full = spx.value_and_grad(_full_loss)(model, x, y)
    loss, grads = sxcall(
        model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=InterleavedH1(microbatches=_M, virtual_stages=2),
        loss_fn=_loss_fn,
    )
    assert jnp.allclose(loss, ref_loss, atol=1e-3, rtol=1e-3)
    assert len(grads) == 4
    for i, pg in enumerate(grads):
        prefix = f"{i}."
        for c, path, pipe_leaf in pg.items():
            if c != "parameters":
                continue
            ref_leaf = ref_full.get("parameters", prefix + path)
            assert jnp.allclose(
                pipe_leaf, ref_leaf, atol=1e-3, rtol=1e-3
            ), f"Grad mismatch at logical stage {i}, path {path!r}"


def test_mpmd_profiler_records_per_task_times(hom_model, xy, mpmd_mesh):
    """``collect_task_times_ms`` captures at least one entry per schedule action."""
    x, y = xy
    with collect_task_times_ms() as times:
        sxcall(
            hom_model,
            (x, y),
            mesh=mpmd_mesh,
            schedule=GPipe(microbatches=_M),
            loss_fn=_loss_fn,
        )
    assert any(name.startswith("L0_fwd_") for name in times)
    assert any("bwd_" in name for name in times)
    assert any("loss" in name for name in times)
    for ms_list in times.values():
        for t in ms_list:
            assert t >= 0.0


def test_mpmd_donate_activations_preserves_loss(hom_model, xy, mpmd_mesh, hom_reference):
    """``donate_activations=True`` does not change the numerical result."""
    x, y = xy
    ref_loss, _ = hom_reference
    loss, _ = sxcall(
        hom_model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
        donate_activations=True,
    )
    assert jnp.allclose(loss, ref_loss, atol=1e-4, rtol=1e-4)


def test_mpmd_mesh_dim_mismatch(hom_model, mpmd_mesh):
    """``mpmd_mesh.mpmd_dim`` must equal ``model.num_stages``."""
    wrong = MpMdMesh(Mesh(jax.devices()[:1], ("pp",)), "pp")
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))
    with pytest.raises(ValueError, match="stages"):
        sxcall(
            hom_model,
            (x, y),
            mesh=wrong,
            schedule=GPipe(microbatches=_M),
            loss_fn=_loss_fn,
        )


def test_mpmd_call_accepts_module_directly(mpmd_mesh):
    """``sxcall`` auto-splits a bare :class:`Module` via ``_normalize_target``.

    Confirms the model-agnostic entry point: passing a single Module
    (not a pre-built :class:`PipelineSequential`) triggers ``auto_split``
    inside ``sxcall``.
    """

    class MultiBlock(spx.Module):
        """Tiny model with a ``blocks`` list so ``auto_split`` can slice it."""

        def __init__(self, d, n_blocks, *, rngs):
            """Build ``n_blocks`` identical ``_Block`` stages."""
            super().__init__()
            self.blocks = nn.ModuleList([_Block(d, rngs=rngs) for _ in range(n_blocks)])

        def forward(self, x):
            """Apply each block sequentially."""
            for blk in self.blocks:
                x = blk(x)
            return x

    model = MultiBlock(_D, _N, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _D))

    loss, _grads = sxcall(
        model,
        (x, y),
        mesh=mpmd_mesh,
        schedule=GPipe(microbatches=_M),
        loss_fn=_loss_fn,
    )
    assert jnp.isfinite(loss)


def test_mpmd_jit_marker_fn_forward(mpmd_mesh):
    """``sxjit`` decorator on a pure-Python function with markers.

    Verifies the model-agnostic marker-function forward path. The
    decorated function can be any callable — Module, raw pytree, or
    pure JAX — as long as it inserts :func:`sxstage_iter`
    markers between logical stages.
    """
    model = _Block(_D, rngs=spx.Rngs(0)), _Block(_D, rngs=spx.Rngs(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _D))

    @sxjit(mesh=mpmd_mesh)
    def forward(stage0, stage1, x):
        """Two-stage forward with an explicit pipeline marker."""
        x = stage0(x)
        x = sxstage_iter(x)
        x = stage1(x)
        return x

    out = forward(model[0], model[1], x)
    ref = model[1](model[0](x))
    assert jnp.allclose(out, ref, atol=1e-5)


def _ref_forward(w0, b0, w1, b1, x, y):
    """Single-device reference forward (same ops as the pipelined version)."""
    h = jnp.maximum(x @ w0 + b0, 0)
    h = jnp.maximum(h @ w1 + b1, 0)
    return ((h - y) ** 2).mean()


def _make_pipe_forward(mpmd_mesh, schedule_cls, static_argnums=(0, 1, 2, 3), microbatches=_M, **schedule_kwargs):
    """Build a decorated ``pipe_forward`` for schedule-driven sxjit tests."""

    @sxjit(
        mesh=mpmd_mesh,
        schedule=schedule_cls(microbatches=microbatches, **schedule_kwargs),
        static_argnums=static_argnums,
    )
    def pipe_forward(w0, b0, w1, b1, x, y):
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        h = jnp.maximum(h @ w1 + b1, 0)
        return ((h - y) ** 2).mean()

    return pipe_forward


def _ref_forward_4stage(w0, b0, w1, b1, w2, b2, w3, b3, x, y):
    h = jnp.maximum(x @ w0 + b0, 0)
    h = jnp.maximum(h @ w1 + b1, 0)
    h = jnp.maximum(h @ w2 + b2, 0)
    h = jnp.maximum(h @ w3 + b3, 0)
    return ((h - y) ** 2).mean()


def _make_pipe_forward_4stage(mpmd_mesh, schedule_cls, *, microbatches=_M, **schedule_kwargs):
    @sxjit(
        mesh=mpmd_mesh,
        schedule=schedule_cls(microbatches=microbatches, **schedule_kwargs),
        static_argnums=tuple(range(8)),
    )
    def pipe_forward(w0, b0, w1, b1, w2, b2, w3, b3, x, y):
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        h = jnp.maximum(h @ w1 + b1, 0)
        h = sxstage_iter(h)
        h = jnp.maximum(h @ w2 + b2, 0)
        h = sxstage_iter(h)
        h = jnp.maximum(h @ w3 + b3, 0)
        return ((h - y) ** 2).mean()

    return pipe_forward


@pytest.fixture
def pipe_args():
    """Concrete args for a 2-stage linear pipeline."""
    w0 = jnp.ones((_D, _D))
    b0 = jnp.zeros((_D,))
    w1 = jnp.ones((_D, _D))
    b1 = jnp.zeros((_D,))
    x = jax.random.normal(jax.random.PRNGKey(10), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(11), (_BATCH, _D))
    return (w0, b0, w1, b1, x, y)


@pytest.fixture
def pipe_args_4stage():
    weights = []
    for i in range(4):
        weights.extend((jnp.ones((_D, _D)) * (i + 1), jnp.zeros((_D,))))
    x = jax.random.normal(jax.random.PRNGKey(12), (_BATCH, _D))
    y = jax.random.normal(jax.random.PRNGKey(13), (_BATCH, _D))
    return (*weights, x, y)


def test_mpmd_jit_schedule_forward(mpmd_mesh, pipe_args):
    """Schedule-driven ``sxjit`` forward pass matches single-device result."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, GPipe)
    loss = pipe_forward(*pipe_args)
    ref_loss = _ref_forward(*pipe_args)
    assert jnp.allclose(loss, ref_loss, atol=1e-5)


def test_mpmd_jit_schedule_default_keeps_array_args_dynamic(mpmd_mesh, pipe_args):
    """Default schedule static inference must not freeze batch arrays."""
    stage0 = _Block(_D, rngs=spx.Rngs(20))
    stage1 = _Block(_D, rngs=spx.Rngs(21))

    @sxjit(mesh=mpmd_mesh, schedule=GPipe(microbatches=_M))
    def pipe_forward(stage0, stage1, x, y):
        h = stage0(x)
        h = sxstage_iter(h)
        h = stage1(h)
        return ((h - y) ** 2).mean()

    x, y = pipe_args[4], pipe_args[5]
    first_loss = pipe_forward(stage0, stage1, x, y)
    second_loss = pipe_forward(stage0, stage1, x + 0.25, y - 0.5)

    assert jnp.allclose(first_loss, ((stage1(stage0(x)) - y) ** 2).mean(), atol=1e-5)
    assert jnp.allclose(second_loss, ((stage1(stage0(x + 0.25)) - (y - 0.5)) ** 2).mean(), atol=1e-5)
    assert not jnp.allclose(first_loss, second_loss)


def test_mpmd_jit_schedule_rebinds_live_weight_values(mpmd_mesh, pipe_args):
    """Schedule plans cache graph structure, not stale parameter values."""
    stage0 = _Block(_D, rngs=spx.Rngs(22))
    stage1 = _Block(_D, rngs=spx.Rngs(23))
    x, y = pipe_args[4], pipe_args[5]

    @sxjit(mesh=mpmd_mesh, schedule=GPipe(microbatches=_M))
    def pipe_forward(stage0, stage1, x, y):
        h = stage0(x)
        h = sxstage_iter(h)
        h = stage1(h)
        return ((h - y) ** 2).mean()

    first_loss = pipe_forward(stage0, stage1, x, y)
    stage0.fc.weight.value = stage0.fc.weight.value * 1.5
    second_loss = pipe_forward(stage0, stage1, x, y)

    assert jnp.allclose(second_loss, ((stage1(stage0(x)) - y) ** 2).mean(), atol=1e-5)
    assert not jnp.allclose(first_loss, second_loss)


def test_mpmd_jit_schedule_preserves_tp_sharded_module_consts(pipe_args):
    """Schedule const rebinding must not collapse intra-stage TP sharding."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip("need 4 devices for pp x tp schedule sharding test")

    mesh = MpMdMesh(Mesh(np.array(devs).reshape(2, 2), ("pp", "tp")), "pp")
    stage0 = _TpBlock(0.25)
    stage1 = _TpBlock(0.5)
    x, y = pipe_args[4], pipe_args[5]

    @sxjit(mesh=mesh, schedule=GPipe(microbatches=_M))
    def pipe_forward(stage0, stage1, x, y):
        h = stage0(x)
        h = sxstage_iter(h)
        h = stage1(h)
        return ((h - y) ** 2).mean()

    loss = pipe_forward(stage0, stage1, x, y)
    ref_h = jax.device_get(x) @ jax.device_get(stage0.weight.value)
    ref_h = ref_h @ jax.device_get(stage1.weight.value)
    ref = ((ref_h - jax.device_get(y)) ** 2).mean()
    assert jnp.allclose(loss, ref, atol=1e-2, rtol=3e-3)

    plan = pipe_forward._mpmd_state["schedule_plan"]
    for loc in ((0, 0), (1, 0)):
        stage_devices = set(mesh.submesh(loc[0]).devices.flat)
        weight_consts = [
            const
            for const in plan["per_loc_consts"][loc]
            if getattr(const, "shape", None) == (_D, _D) and getattr(const, "sharding", None) is not None
        ]
        assert weight_consts
        assert any(tuple(const.sharding.spec) == (None, "tp") for const in weight_consts)
        assert all(set(const.devices()).issubset(stage_devices) for const in weight_consts)


def test_mpmd_jit_schedule_preserves_existing_nonreplicated_const_sharding(pipe_args):
    """Already-sharded leaves should not be weakened by replicated metadata fallback."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip("need 4 devices for existing sharding preservation test")

    mesh = MpMdMesh(Mesh(np.array(devs).reshape(2, 2), ("pp", "tp")), "pp")
    stage0 = _UnannotatedBlock(0.25)
    stage1 = _UnannotatedBlock(0.5)
    stage0.weight.value = jax.device_put(
        stage0.weight.value,
        jax.sharding.NamedSharding(mesh.submesh(0), jax.sharding.PartitionSpec(None, "tp")),
    )
    stage1.weight.value = jax.device_put(
        stage1.weight.value,
        jax.sharding.NamedSharding(mesh.submesh(1), jax.sharding.PartitionSpec(None, "tp")),
    )
    x, y = pipe_args[4], pipe_args[5]

    @sxjit(mesh=mesh, schedule=GPipe(microbatches=_M))
    def pipe_forward(stage0, stage1, x, y):
        h = stage0(x)
        h = sxstage_iter(h)
        h = stage1(h)
        return ((h - y) ** 2).mean()

    loss = pipe_forward(stage0, stage1, x, y)
    ref_h = jax.device_get(x) @ jax.device_get(stage0.weight.value)
    ref_h = ref_h @ jax.device_get(stage1.weight.value)
    ref = ((ref_h - jax.device_get(y)) ** 2).mean()
    assert jnp.allclose(loss, ref, atol=1e-2, rtol=3e-3)

    plan = pipe_forward._mpmd_state["schedule_plan"]
    for loc in ((0, 0), (1, 0)):
        weight_consts = [
            const
            for const in plan["per_loc_consts"][loc]
            if getattr(const, "shape", None) == (_D, _D) and getattr(const, "sharding", None) is not None
        ]
        assert weight_consts
        assert any(tuple(const.sharding.spec) == (None, "tp") for const in weight_consts)


def test_mpmd_jit_schedule_virtual_stage_owners_follow_schedule(pipe_args):
    """Virtual-stage schedules resolve variable owners through logical locations."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip("need 4 devices for virtual pp x tp schedule owner test")

    mesh = MpMdMesh(Mesh(np.array(devs).reshape(2, 2), ("pp", "tp")), "pp")
    schedule = KimiK2(microbatches=4, virtual_stages=2)
    blocks = []
    for logical in range(4):
        with spx.assign_stage(total=4, current=logical):
            blocks.append(_TpBlock(0.25 * (logical + 1)))
    x, y = pipe_args[4], pipe_args[5]

    @sxjit(mesh=mesh, schedule=schedule)
    def pipe_forward(b0, b1, b2, b3, x, y):
        h = b0(x)
        h = sxstage_iter(h)
        h = b1(h)
        h = sxstage_iter(h)
        h = b2(h)
        h = sxstage_iter(h)
        h = b3(h)
        return ((h - y) ** 2).mean()

    loss = pipe_forward(*blocks, x, y)
    ref = ((blocks[3](blocks[2](blocks[1](blocks[0](x)))) - y) ** 2).mean()
    assert jnp.allclose(loss, ref, atol=1e-2, rtol=1e-2)

    plan = pipe_forward._mpmd_state["schedule_plan"]
    flat_args = jax.tree.leaves((*blocks, x, y))
    expected_owner = {
        schedule.logical_at(rank, virt, mesh.mpmd_dim): rank
        for rank in range(mesh.mpmd_dim)
        for virt in range(schedule.virtual_stages_per_rank())
    }
    for logical, block in enumerate(blocks):
        flat_idx = next(i for i, leaf in enumerate(flat_args) if leaf is block.weight.value)
        assert plan["leaf_stage_owners"][flat_idx] == expected_owner[logical]


def test_mpmd_jit_schedule_accepts_physical_owner_stage_metadata(pipe_args):
    """Schedule owner resolution also accepts already-physical stage metadata."""
    devs = jax.devices()[:4]
    if len(devs) < 4:
        pytest.skip("need 4 devices for physical-owner metadata test")

    mesh = MpMdMesh(Mesh(np.array(devs).reshape(2, 2), ("pp", "tp")), "pp")
    schedule = KimiK2(microbatches=4, virtual_stages=2)
    owner_by_logical = {
        schedule.logical_at(rank, virt, mesh.mpmd_dim): rank
        for rank in range(mesh.mpmd_dim)
        for virt in range(schedule.virtual_stages_per_rank())
    }
    owners = tuple(owner_by_logical[logical] for logical in range(4))
    blocks = []
    for logical, owner in enumerate(owners):
        with spx.assign_stage(total=2, current=owner):
            blocks.append(_TpBlock(0.25 * (logical + 1)))
    x, y = pipe_args[4], pipe_args[5]

    @sxjit(mesh=mesh, schedule=schedule)
    def pipe_forward(b0, b1, b2, b3, x, y):
        h = b0(x)
        h = sxstage_iter(h)
        h = b1(h)
        h = sxstage_iter(h)
        h = b2(h)
        h = sxstage_iter(h)
        h = b3(h)
        return ((h - y) ** 2).mean()

    loss = pipe_forward(*blocks, x, y)
    ref = ((blocks[3](blocks[2](blocks[1](blocks[0](x)))) - y) ** 2).mean()
    assert jnp.allclose(loss, ref, atol=1e-2, rtol=1e-2)

    plan = pipe_forward._mpmd_state["schedule_plan"]
    flat_args = jax.tree.leaves((*blocks, x, y))
    for logical, block in enumerate(blocks):
        flat_idx = next(i for i, leaf in enumerate(flat_args) if leaf is block.weight.value)
        assert plan["leaf_stage_owners"][flat_idx] == owners[logical]


def test_mpmd_jit_schedule_grad_matches_reference(mpmd_mesh, pipe_args):
    """``jax.grad`` on schedule-driven ``sxjit`` matches single-device grads."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, Std1F1B)
    argnums = (0, 1, 2, 3)

    _ = pipe_forward(*pipe_args)

    pipe_grads = jax.grad(pipe_forward, argnums=argnums)(*pipe_args)
    ref_grads = jax.grad(_ref_forward, argnums=argnums)(*pipe_args)

    for pg, rg in zip(pipe_grads, ref_grads, strict=True):
        assert jnp.allclose(pg, rg, atol=1e-4, rtol=1e-4)

    faithful_grads = sxgrad(pipe_forward, argnums=argnums)(*pipe_args)
    for fg, rg in zip(faithful_grads, ref_grads, strict=True):
        assert jnp.allclose(fg, rg, atol=1e-4, rtol=1e-4)

    loss_vg, vg_grads = sxvalue_and_grad(pipe_forward, argnums=argnums)(*pipe_args)
    ref_loss = _ref_forward(*pipe_args)
    assert jnp.allclose(loss_vg, ref_loss, atol=1e-5)
    for vg, rg in zip(vg_grads, ref_grads, strict=True):
        assert jnp.allclose(vg, rg, atol=1e-4, rtol=1e-4)


def test_mpmd_jit_schedule_profiler_records_tasks(mpmd_mesh, pipe_args):
    """``collect_task_times_ms`` also observes schedule-driven ``sxjit``."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, Std1F1B)
    _ = pipe_forward(*pipe_args)

    with collect_task_times_ms() as times:
        loss, grads = sxvalue_and_grad(pipe_forward, argnums=(0, 1, 2, 3))(*pipe_args)
        jax.block_until_ready((loss, grads))

    assert times
    assert any(name.startswith("stage") for name in times)
    assert any(name.startswith("transfer_") for name in times)


def test_mpmd_jit_schedule_profiler_records_fused_tasks(mpmd_mesh, pipe_args):
    """Non-terminal ``FusedTask`` cells dispatch as one fused stage call."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, Std1F1B, microbatches=4)
    _ = pipe_forward(*pipe_args)

    with collect_task_times_ms() as times:
        loss, grads = sxvalue_and_grad(pipe_forward, argnums=(0, 1, 2, 3))(*pipe_args)
        jax.block_until_ready((loss, grads))

    ref_loss = _ref_forward(*pipe_args)
    assert jnp.allclose(loss, ref_loss, atol=1e-2, rtol=1e-2)
    assert any("_fused_fwd" in name for name in times)


def test_mpmd_schedule_default_uses_fused_async(mpmd_mesh, pipe_args):
    """Default schedule-driven ``sxjit`` keeps the measured fast dispatcher."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, Std1F1B, microbatches=4)
    _ = pipe_forward(*pipe_args)

    loss, grads = sxvalue_and_grad(pipe_forward, argnums=(0, 1, 2, 3))(*pipe_args)
    jax.block_until_ready((loss, grads))

    stats = pipe_forward._mpmd_state["schedule_plan"]["last_schedule_runtime_stats"]
    assert stats["dispatcher"] == "fused_async"
    assert stats["unit_count"] is not None
    assert stats["action_count"] >= stats["unit_count"]
    assert stats["fused_count"] >= 1
    assert stats["transfer_count"] > 0
    assert stats["transfer_bytes"] > 0
    assert stats["per_rank_launch_count"]
    assert stats["critical_path_ms"] >= 0.0


@pytest.mark.parametrize(
    ("schedule_cls", "kwargs"),
    [
        (KimiK2, {"virtual_stages": 1}),
        (ZeroBubbleH1, {}),
    ],
)
def test_mpmd_grad_flat_split_schedules_match_reference(schedule_cls, kwargs, mpmd_mesh, pipe_args):
    """Fused/async schedule path preserves gradients for flat split schedules."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, schedule_cls, microbatches=4, **kwargs)
    argnums = tuple(range(4))

    _ = pipe_forward(*pipe_args)
    faithful_grads = sxgrad(pipe_forward, argnums=argnums)(*pipe_args)
    ref_grads = jax.grad(_ref_forward, argnums=argnums)(*pipe_args)

    for fg, rg in zip(faithful_grads, ref_grads, strict=True):
        assert jnp.allclose(fg, rg, atol=1e-2, rtol=1e-2)


def test_mpmd_grad_virtual_kimi_matches_reference(mpmd_mesh, pipe_args_4stage):
    """Fused/async schedule path preserves gradients for Kimi virtual stages."""
    pipe_forward = _make_pipe_forward_4stage(mpmd_mesh, KimiK2, microbatches=4, virtual_stages=2)
    argnums = tuple(range(8))

    _ = pipe_forward(*pipe_args_4stage)
    faithful_grads = sxgrad(pipe_forward, argnums=argnums)(*pipe_args_4stage)
    ref_grads = jax.grad(_ref_forward_4stage, argnums=argnums)(*pipe_args_4stage)

    for fg, rg in zip(faithful_grads, ref_grads, strict=True):
        assert jnp.allclose(fg, rg, atol=1e-1, rtol=6e-2)


@pytest.mark.parametrize("schedule_cls", [GPipe, Std1F1B, Eager1F1B])
def test_mpmd_grad_across_schedules(schedule_cls, mpmd_mesh, pipe_args):
    """``sxgrad`` produces correct grads for multiple flat schedules."""
    pipe_forward = _make_pipe_forward(mpmd_mesh, schedule_cls)
    argnums = (0, 1, 2, 3)

    _ = pipe_forward(*pipe_args)
    faithful_grads = sxgrad(pipe_forward, argnums=argnums)(*pipe_args)
    ref_grads = jax.grad(_ref_forward, argnums=argnums)(*pipe_args)

    for fg, rg in zip(faithful_grads, ref_grads, strict=True):
        assert jnp.allclose(fg, rg, atol=1e-4, rtol=1e-4)


def test_mpmd_grad_lazy_bwd_batching(mpmd_mesh, pipe_args):
    """``lazy_bwd_batching=True`` yields numerically identical grads."""
    from spectrax.runtime.schedules import Std1F1B

    pipe_forward_std = _make_pipe_forward(mpmd_mesh, Std1F1B)
    _ = pipe_forward_std(*pipe_args)
    grads_std = sxgrad(pipe_forward_std, argnums=(0, 1, 2, 3))(*pipe_args)

    pipe_forward_lazy = _make_pipe_forward(mpmd_mesh, lambda **kw: Std1F1B(**kw, lazy_bwd_batching=True))
    _ = pipe_forward_lazy(*pipe_args)
    grads_lazy = sxgrad(pipe_forward_lazy, argnums=(0, 1, 2, 3))(*pipe_args)

    for g_std, g_lazy in zip(grads_std, grads_lazy, strict=True):
        assert jnp.allclose(g_std, g_lazy, atol=1e-5, rtol=1e-5)


def test_mpmd_grad_requires_schedule(mpmd_mesh, pipe_args):
    """``sxgrad`` raises on a forward-only ``sxjit`` wrapper."""

    @sxjit(mesh=mpmd_mesh)
    def fwd_only(w0, b0, w1, b1, x, y):
        h = jnp.maximum(x @ w0 + b0, 0)
        h = sxstage_iter(h)
        h = jnp.maximum(h @ w1 + b1, 0)
        return ((h - y) ** 2).mean()

    _ = fwd_only(*pipe_args)

    with pytest.raises(TypeError, match="schedule"):
        sxgrad(fwd_only)
