# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for MPMD validation and legacy compiler helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.runtime.mpmd.compiler import compile_ranked_executables, run_ranked_pipeline
from spectrax.runtime.mpmd.pscan_compiler import PscanPlan, _pack_grad_tree
from spectrax.runtime.mpmd.runtime import (
    _infer_schedule_static_argnums,
    _normalize_argnums,
    _resolve_explicit_shardings,
    sxcall,
)
from spectrax.runtime.schedules import Action, GPipe, Phase, Schedule


@dataclass
class _FwdOnly(Schedule):
    """Tiny forward-only schedule for compiler shape tests."""

    def build(self, n_stages: int):
        """Build helper."""
        return [[Action(Phase.FWD, microbatch=mb)] for mb in range(self.microbatches)]

    def peak_activations(self, n_stages: int) -> int:
        """Peak activation helper."""
        return self.microbatches


def test_infer_schedule_static_argnums_keeps_plain_arrays_dynamic():
    """Array batches stay dynamic even when no Module argument is present."""
    args = (jnp.ones((2, 3)), {"x": jnp.ones((1,))}, "metadata")

    assert _infer_schedule_static_argnums(args) == (2,)


def test_normalize_argnums_accepts_jax_style_negative_indices():
    """``sxgrad`` validation should match JAX's negative argnum convention."""
    assert _normalize_argnums((-1, 0), 2) == (1, 0)


def test_resolve_explicit_shardings_preserves_none_placeholders():
    """``None`` entries in ``in_shardings`` still consume positional slots."""
    sharding = object()

    assert _resolve_explicit_shardings((None, sharding, None), [1, 2, 3]) == {1: sharding}


def test_ranked_compiler_forward_outputs_can_change_shape():
    """Legacy rank programs must not assume stage output shape equals input shape."""

    def expand(x):
        """Expand helper."""
        return jnp.stack([x, x + 1], axis=0)

    cluster = jax.make_jaxpr(expand)(jnp.asarray(1.0))
    program = compile_ranked_executables([cluster], _FwdOnly(microbatches=2), n_stages=1)[0]

    _grads, outgoing_acts, _cots, _loss = program([()], jnp.asarray([1.0, 2.0]), jnp.zeros((2,)))

    assert outgoing_acts.shape == (2, 2)
    assert jnp.allclose(outgoing_acts, jnp.asarray([[1.0, 2.0], [2.0, 3.0]]))


def test_ranked_compiler_bwd_uses_incoming_cotangent_for_param_grads():
    """BWD must seed param gradients with ``g_y``, not JAX's implicit scalar one."""

    def stage(w, x):
        """Stage helper."""
        return w * x

    cluster = jax.make_jaxpr(stage)(jnp.asarray(2.0), jnp.asarray(3.0))
    program = compile_ranked_executables([cluster], GPipe(microbatches=1), n_stages=1)[0]

    grads, _outgoing_acts, outgoing_cots, _loss = program(
        [(jnp.asarray(2.0),)],
        jnp.asarray([3.0]),
        jnp.asarray([7.0]),
    )

    assert jnp.allclose(grads[0][0], 21.0)
    assert jnp.allclose(outgoing_cots[0], 14.0)


def test_run_ranked_pipeline_returns_mean_loss_and_mean_grads():
    """Legacy ranked pipeline helper should be numerically correct if used."""

    def stage(w, x):
        """Stage helper."""
        return w * x

    def loss_fn(y, target):
        """Compute the loss."""
        diff = y - target
        return 0.5 * jnp.sum(diff * diff)

    w0 = jnp.asarray(2.0)
    w1 = jnp.asarray(3.0)
    m = 4
    xs = jnp.arange(m * 2, dtype=jnp.float32).reshape(m, 2) + 1.0
    target = jnp.ones((m, 2), dtype=jnp.float32)
    cluster = jax.make_jaxpr(stage)(jnp.asarray(1.0), jnp.ones((2,), dtype=jnp.float32))

    loss, grads = run_ranked_pipeline(
        [cluster, cluster],
        [(w0,), (w1,)],
        GPipe(microbatches=m),
        n_stages=2,
        microbatches=m,
        xs=xs,
        target_args=(target,),
        loss_fn=loss_fn,
    )

    y0 = xs * w0
    y1 = y0 * w1
    diff = y1 - target
    ref_loss = 0.5 * jnp.sum(diff * diff) / m
    ref_g1 = (y0 * diff).sum() / m
    ref_g0 = (xs * diff * w1).sum() / m

    assert jnp.allclose(loss, ref_loss)
    assert jnp.allclose(grads[0][0], ref_g0)
    assert jnp.allclose(grads[1][0], ref_g1)


def test_pack_grad_tree_zeros_missing_const_grads():
    """No-producing-rank cases should return zeros, not index into an empty tuple."""
    plan = PscanPlan(
        n=1,
        v=1,
        n_logical=1,
        m=1,
        schedule=_FwdOnly(microbatches=1),
        ops=(),
        n_outs=0,
        n_outer_consts=1,
        body_mode="train",
        stage_shardings=[],
        rank_submeshes=[],
        mpmd_mesh=None,
        loc_for_logical=((0, 0),),
        logical_for_loc={(0, 0): 0},
        terminal_loc=(0, 0),
        per_loc_consts={},
        const_indices_per_loc={},
        n_invars_per_loc={},
        fwd_jits={},
        bwd_jits={},
        terminal_jit=lambda *args: args,
        init_state_template=[],
        grad_tree=jax.tree.structure({"w": jnp.ones((2,), dtype=jnp.float32)}),
        grad_const_indices=(0,),
        grad_template_leaves=(jnp.ones((2,), dtype=jnp.float32),),
        grad_output_sharding=jax.devices()[0],
    )

    out = _pack_grad_tree(plan, None)

    assert jnp.array_equal(out["w"], jnp.zeros((2,), dtype=jnp.float32))


def test_sxcall_rejects_invalid_mode_before_setup():
    """Unknown modes must not silently enter the train path."""
    mesh = spx.create_mesh(axis_dims=(-1,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="mode"):
        sxcall(object(), (jnp.ones((1,)),), mesh=mesh, schedule=_FwdOnly(microbatches=1), mode="bogus")


def test_sxgrad_argnums_validation_happens_at_call_time():
    """Out-of-range ``argnums`` should raise a friendly ``ValueError``."""

    def plain(x):
        """Plain reference implementation."""
        return x.sum()

    plain._mpmd_state = {"schedule_requested": True}

    from spectrax.runtime.mpmd.runtime import sxgrad

    with pytest.raises(ValueError, match="argnum"):
        sxgrad(plain, argnums=1)(jnp.ones((2,)))


def test_sxvalue_and_grad_argnums_validation_happens_at_call_time():
    """``sxvalue_and_grad`` uses the same friendly bounds validation."""

    def plain(x):
        """Plain reference implementation."""
        return x.sum()

    plain._mpmd_state = {"schedule_requested": True}

    from spectrax.runtime.mpmd.runtime import sxvalue_and_grad

    with pytest.raises(ValueError, match="argnum"):
        sxvalue_and_grad(plain, argnums=2)(jnp.ones((2,)))
