# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`make_scheduled_body` — shard_map body for arbitrary pipeline schedules.

Turns any :class:`~spectrax.runtime.schedules.Schedule`'s action grid
into a :func:`jax.shard_map` body with the ``pp`` axis **manual** and
cross-rank transport via :func:`jax.lax.ppermute`.

**Two compilation modes**:

* **Python-unrolled** (default when ``use_scan=False``): each time step
  is a separate HLO block with its own ``lax.cond`` + ``ppermute``.
  Fast compile at small scale; OOMs at 8B+ with virtual stages because
  the HLO graph grows linearly with ``T × n_actions``.

* **Scan-based** (``use_scan=True``): the schedule is encoded as
  integer arrays and ``lax.scan`` loops over time steps. ONE copy of
  the body in HLO regardless of ``T`` → compiles at any scale. Dynamic
  dispatch inside the scan via ``lax.switch`` over phase × ``lax.cond``
  over rank. No ``value_and_grad`` over the scan (explicit ``fwd_fn``
  + ``bwd_fn`` avoids the autograd-through-scan memory blowup).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp

from ..schedules import Action, FusedTask, Phase, Schedule

__all__ = ["make_scheduled_body"]

_PHASE_SKIP = 0
_PHASE_FWD = 1
_PHASE_BWD = 2
_PHASE_BWD_I = 3
_PHASE_BWD_W = 4

_PHASE_MAP = {
    Phase.FWD: _PHASE_FWD,
    Phase.BWD: _PHASE_BWD,
    Phase.BWD_I: _PHASE_BWD_I,
    Phase.BWD_W: _PHASE_BWD_W,
}


def _drop0(tree: Any) -> Any:
    """Strip a leading size-1 axis from every leaf."""
    return jax.tree.map(lambda a: a[0], tree)


def _add0(tree: Any) -> Any:
    """Add a leading size-1 axis to every leaf (inverse of :func:`_drop0`)."""
    return jax.tree.map(lambda a: a[None, ...], tree)


def _prev_logical_loc(schedule: Schedule, rank: int, virt: int, n_stages: int) -> tuple[int, int] | None:
    """Return the ``(rank, virt)`` hosting logical stage ``logical - 1``."""
    logical = schedule.logical_at(rank, virt, n_stages)
    if logical == 0:
        return None
    V = schedule.virtual_stages_per_rank()
    for r in range(n_stages):
        for v in range(V):
            if schedule.logical_at(r, v, n_stages) == logical - 1:
                return (r, v)
    return None


def _encode_grid(
    schedule: Schedule, n_stages: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Encode the schedule grid as integer arrays for :func:`lax.scan`.

    Returns ``(phase_grid, mb_grid, virt_grid, logical_grid,
    fwd_dest_grid, bwd_dest_grid, T)`` where each array has shape
    ``(T, n_stages)``.

    ``fwd_dest_grid[t, r]`` is the physical rank that this rank's FWD
    output should be transported to (``-1`` if no transport, i.e. the
    action is terminal or non-FWD). ``bwd_dest_grid[t, r]`` is where
    the BWD cotangent goes (the upstream rank, ``-1`` if first logical
    stage or non-BWD).
    """
    grid_raw = schedule.build(n_stages)
    T = len(grid_raw)
    phase_arr = [[_PHASE_SKIP] * n_stages for _ in range(T)]
    mb_arr = [[0] * n_stages for _ in range(T)]
    virt_arr = [[0] * n_stages for _ in range(T)]
    logical_arr = [[-1] * n_stages for _ in range(T)]
    fwd_dest_arr = [[-1] * n_stages for _ in range(T)]
    fwd_dest_virt_arr = [[0] * n_stages for _ in range(T)]
    bwd_dest_arr = [[-1] * n_stages for _ in range(T)]
    bwd_dest_virt_arr = [[0] * n_stages for _ in range(T)]
    for t, row in enumerate(grid_raw):
        for r, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, FusedTask):
                cell = cell.fwd
            phase_arr[t][r] = _PHASE_MAP.get(cell.phase, _PHASE_SKIP)
            mb_arr[t][r] = cell.microbatch
            virt_arr[t][r] = cell.virtual_stage
            v = cell.virtual_stage
            logical_arr[t][r] = schedule.logical_at(r, v, n_stages)
            if cell.phase == Phase.FWD:
                nxt = schedule.next_logical_loc(r, v, n_stages)
                if nxt is not None:
                    fwd_dest_arr[t][r] = nxt[0]
                    fwd_dest_virt_arr[t][r] = nxt[1]
            elif cell.phase in (Phase.BWD, Phase.BWD_I):
                prev = _prev_logical_loc(schedule, r, v, n_stages)
                if prev is not None:
                    bwd_dest_arr[t][r] = prev[0]
                    bwd_dest_virt_arr[t][r] = prev[1]
    return (
        jnp.array(phase_arr, dtype=jnp.int32),
        jnp.array(mb_arr, dtype=jnp.int32),
        jnp.array(virt_arr, dtype=jnp.int32),
        jnp.array(logical_arr, dtype=jnp.int32),
        jnp.array(fwd_dest_arr, dtype=jnp.int32),
        jnp.array(fwd_dest_virt_arr, dtype=jnp.int32),
        jnp.array(bwd_dest_arr, dtype=jnp.int32),
        jnp.array(bwd_dest_virt_arr, dtype=jnp.int32),
        T,
    )


def make_scheduled_body(
    *,
    schedule: Schedule,
    n_stages: int,
    microbatches: int,
    pp_axis: str,
    fwd_fn: Callable[[Any, Any], Any],
    bwd_fn: Callable[[Any, Any, Any], tuple[Any, Any]],
    loss_and_g_y: Callable[..., tuple[Any, Any]],
    mode: Literal["train"] = "train",
    checkpoint_policy: Callable[..., bool] | None = None,
    use_scan: bool = False,
) -> Callable[..., Any]:
    """Build a :func:`jax.shard_map` body that executes ``schedule``.

    Args:
        schedule: Flat or virtual-stage :class:`Schedule`.
        n_stages: Physical pipeline rank count.
        microbatches: ``M``.
        pp_axis: Manual pipeline-parallel mesh axis name.
        fwd_fn: ``(params, x) -> y`` per microbatch.
        bwd_fn: ``(params, x, g_y) -> (g_params, g_x)`` per microbatch.
        loss_and_g_y: ``(y, *targets) -> (loss, g_y)`` terminal rank.
        mode: Only ``"train"`` supported.
        checkpoint_policy: If truthy, wrap fwd/bwd in ``jax.checkpoint``.
        use_scan: Use ``lax.scan`` (compact HLO, scales to 8B+) or
            Python-unrolled (small-scale only). Default ``False``.
    """
    if mode != "train":
        raise NotImplementedError(f"make_scheduled_body only supports mode='train'; got {mode}.")

    V = schedule.virtual_stages_per_rank()
    n_logical = n_stages * V
    m = microbatches

    if checkpoint_policy is not None:
        _raw_fwd, _raw_bwd = fwd_fn, bwd_fn

        def fwd_fn(params, x):
            """Checkpointed forward."""

            @jax.checkpoint
            def _ckpt(p, xi):
                """Checkpoint boundary: ``_raw_fwd`` recomputed during backward instead of saved."""
                return _raw_fwd(p, xi)

            return _ckpt(params, x)

        def bwd_fn(params, x, g_y):
            """Checkpointed backward."""

            @jax.checkpoint
            def _ckpt(p, xi, g):
                """Checkpoint boundary: ``_raw_bwd`` recomputed during higher-order autodiff."""
                return _raw_bwd(p, xi, g)

            return _ckpt(params, x, g_y)

    if use_scan:
        return _make_scan_body(
            schedule=schedule,
            n_stages=n_stages,
            microbatches=m,
            pp_axis=pp_axis,
            fwd_fn=fwd_fn,
            bwd_fn=bwd_fn,
            loss_and_g_y=loss_and_g_y,
            V=V,
            n_logical=n_logical,
        )
    return _make_unrolled_body(
        schedule=schedule,
        n_stages=n_stages,
        microbatches=m,
        pp_axis=pp_axis,
        fwd_fn=fwd_fn,
        bwd_fn=bwd_fn,
        loss_and_g_y=loss_and_g_y,
        V=V,
        n_logical=n_logical,
    )


def _make_scan_body(
    *,
    schedule,
    n_stages,
    microbatches,
    pp_axis,
    fwd_fn,
    bwd_fn,
    loss_and_g_y,
    V,
    n_logical,
):
    """Scan-based body: ``lax.scan`` over time steps with dynamic dispatch.

    ONE copy of the step body in HLO. Schedule encoded as integer
    arrays. Phase dispatch via ``lax.switch``. Transport via
    ``lax.ppermute`` with all-pairs swap (covers both forward and
    backward transport directions for any ``n_stages``).
    """
    (
        phase_grid,
        mb_grid,
        virt_grid,
        logical_grid,
        fwd_dest_grid,
        fwd_dv_grid,
        bwd_dest_grid,
        bwd_dv_grid,
        T,
    ) = _encode_grid(schedule, n_stages)
    m = microbatches

    [*list(range(1, n_stages)), 0]
    [n_stages - 1, *list(range(n_stages - 1))]

    def body(stacked_params, xs, *targets):
        """Shard-map body: scan over schedule time steps."""
        rank = jax.lax.axis_index(pp_axis)
        p_per_rank = _drop0(stacked_params)
        mb_shape = xs.shape[1:]
        mb_dtype = xs.dtype

        def pick_virt(tree, v):
            """Select virt ``v`` from per-rank params."""
            if V == 1:
                return tree
            return jax.tree.map(lambda a: a[v], tree)

        if V == 1:
            saved_inputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
        else:
            saved_inputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)

        g_params = jax.tree.map(jnp.zeros_like, p_per_rank)
        loss_acc = jnp.zeros((), dtype=jnp.float32)

        my_phases = phase_grid[:, rank]
        my_mbs = mb_grid[:, rank]
        my_virts = virt_grid[:, rank]
        my_logicals = logical_grid[:, rank]
        my_fwd_dests = fwd_dest_grid[:, rank]
        my_bwd_dests = bwd_dest_grid[:, rank]

        def si_get(arr, v, mb_idx):
            """Read saved[v, mb] or saved[mb]."""
            return arr[mb_idx] if V == 1 else arr[v, mb_idx]

        def si_set(arr, v, mb_idx, val):
            """Write saved[v, mb] = val or saved[mb] = val."""
            return arr.at[mb_idx].set(val) if V == 1 else arr.at[v, mb_idx].set(val)

        def g_add(acc, v, upd):
            """Accumulate grad update at virt ``v``."""
            if V == 1:
                return jax.tree.map(lambda a, b: a + b, acc, upd)
            return jax.tree.map(lambda a, b: a.at[v].add(b), acc, upd)

        carry_init = (saved_inputs, saved_outputs, incoming_fwd, incoming_bwd, g_params, loss_acc)

        transfer_edges = tuple((src, dst) for src in range(n_stages) for dst in range(n_stages) if src != dst)

        def route_value(buffer, value, dest, dest_virt, mb_idx):
            """Route ``value`` to ``dest`` and write it into ``buffer[dest_virt, mb]``."""
            same_rank = dest == rank
            buffer = jnp.where(same_rank, si_set(buffer, dest_virt, mb_idx, value), buffer)
            for src, dst in transfer_edges:
                send = (rank == src) & (dest == dst)
                mb_tag = jnp.where(send, mb_idx, -1).astype(jnp.int32)
                v_tag = jnp.where(send, dest_virt, 0).astype(jnp.int32)
                recv_value = jax.lax.ppermute(value, pp_axis, perm=[(src, dst)])
                recv_mb = jax.lax.ppermute(mb_tag, pp_axis, perm=[(src, dst)])
                recv_v = jax.lax.ppermute(v_tag, pp_axis, perm=[(src, dst)])
                has_value = (rank == dst) & (recv_mb >= 0)
                buffer = jnp.where(has_value, si_set(buffer, recv_v, recv_mb, recv_value), buffer)
            return buffer

        def scan_step(carry, t_idx):
            """One time step: compute via lax.switch, then transport via ppermute OUTSIDE switch.

            Collectives (ppermute) must fire on every device every step —
            they can't live inside lax.switch branches (which only execute
            one branch per device). So the switch returns the value to
            transport, and ppermute runs unconditionally after.
            """
            si, so, ifwd, ibwd, gp, la = carry
            phase = my_phases[t_idx]
            mb_idx = my_mbs[t_idx]
            v = my_virts[t_idx]
            p_v = pick_virt(p_per_rank, v)

            logical = my_logicals[t_idx]
            is_first_logical = logical == 0
            is_terminal = logical == n_logical - 1

            x_from_xs = xs[mb_idx]
            x_from_incoming = si_get(ifwd, v, mb_idx)
            x_in = jnp.where(is_first_logical, x_from_xs, x_from_incoming)

            def do_fwd(args):
                """Forward: compute y, save input/output. Return y for transport."""
                si_, so_, gp_, la_ = args
                y = fwd_fn(p_v, x_in)
                si_ = si_set(si_, v, mb_idx, x_in)
                so_ = jnp.where(is_terminal, si_set(so_, v, mb_idx, y), so_)
                return si_, so_, gp_, la_, y, jnp.zeros_like(y)

            def do_bwd(args):
                """Full backward: accumulate parameter grads and return g_x."""
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    """Terminal-stage branch: compute ``loss`` and ``g_y`` from saved output + targets."""
                    y_out = si_get(so_, v, mb_idx)
                    loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return loss_mb.astype(jnp.float32), g_y_mb

                def _get_gy_incoming():
                    """Non-terminal branch: ``loss=0``, ``g_y`` = previously received cotangent buffer entry."""
                    return jnp.zeros((), dtype=jnp.float32), si_get(ibwd, v, mb_idx)

                loss_mb, g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                la_ = la_ + loss_mb
                g_p, g_x = bwd_fn(p_v, x_saved, g_y)
                gp_ = g_add(gp_, v, g_p)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), g_x

            def do_bwd_i(args):
                """Input-gradient half of ZeroBubble: send g_x, do not add g_params."""
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    y_out = si_get(so_, v, mb_idx)
                    loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return loss_mb.astype(jnp.float32), g_y_mb

                def _get_gy_incoming():
                    return jnp.zeros((), dtype=jnp.float32), si_get(ibwd, v, mb_idx)

                loss_mb, g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                la_ = la_ + loss_mb
                _g_p, g_x = bwd_fn(p_v, x_saved, g_y)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), g_x

            def do_bwd_w(args):
                """Weight-gradient half of ZeroBubble: add g_params, no upstream send."""
                si_, so_, gp_, la_ = args
                x_saved = si_get(si_, v, mb_idx)

                def _get_gy():
                    y_out = si_get(so_, v, mb_idx)
                    _loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[mb_idx] for t_arr in targets))
                    return g_y_mb

                def _get_gy_incoming():
                    return si_get(ibwd, v, mb_idx)

                g_y = jax.lax.cond(is_terminal, _get_gy, _get_gy_incoming)
                g_p, _g_x = bwd_fn(p_v, x_saved, g_y)
                gp_ = g_add(gp_, v, g_p)
                return si_, so_, gp_, la_, jnp.zeros_like(x_saved), jnp.zeros_like(x_saved)

            def do_skip(args):
                """Idle: zeros for transport, carry unchanged."""
                si_, so_, gp_, la_ = args
                z = jnp.zeros(mb_shape, dtype=mb_dtype)
                return si_, so_, gp_, la_, z, z

            args = (si, so, gp, la)
            si_, so_, gp_, la_, fwd_val, bwd_val = jax.lax.switch(
                phase,
                [do_skip, do_fwd, do_bwd, do_bwd_i, do_bwd_w],
                args,
            )

            fwd_dest = my_fwd_dests[t_idx]
            fwd_dv = fwd_dv_grid[:, rank][t_idx]
            bwd_dest = my_bwd_dests[t_idx]
            bwd_dv = bwd_dv_grid[:, rank][t_idx]

            ifwd_ = route_value(ifwd, fwd_val, fwd_dest, fwd_dv, mb_idx)
            ibwd_ = route_value(ibwd, bwd_val, bwd_dest, bwd_dv, mb_idx)

            return (si_, so_, ifwd_, ibwd_, gp_, la_), None

        (_si_f, _so_f, _ifwd_f, _ibwd_f, gp_f, la_f), _ = jax.lax.scan(scan_step, carry_init, jnp.arange(T))

        total_loss = jax.lax.psum(la_f, pp_axis)
        mean_loss = total_loss / jnp.asarray(m, dtype=total_loss.dtype)
        return mean_loss, _add0(gp_f)

    return body


def _make_unrolled_body(
    *,
    schedule,
    n_stages,
    microbatches,
    pp_axis,
    fwd_fn,
    bwd_fn,
    loss_and_g_y,
    V,
    n_logical,
):
    """Python-unrolled body: original implementation for small-scale configs."""
    m = microbatches

    grid_raw = schedule.build(n_stages)
    grid: list[list[Action | None]] = []
    for row in grid_raw:
        new_row: list[Action | None] = []
        for cell in row:
            if cell is None:
                new_row.append(None)
            elif isinstance(cell, FusedTask):
                new_row.append(cell.fwd)
                new_row.append(cell.bwd)
            else:
                new_row.append(cell)
        grid.append(new_row[:n_stages])

    def body(stacked_params, xs, *targets):
        """Shard-map body: Python-unrolled time steps."""
        rank = jax.lax.axis_index(pp_axis)
        p_per_rank = _drop0(stacked_params)
        mb_shape = xs.shape[1:]
        mb_dtype = xs.dtype

        def pick_virt(tree, v):
            """Select virt."""
            if V == 1:
                return tree
            return jax.tree.map(lambda a: a[v], tree)

        if V == 1:
            saved_inputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((m, *mb_shape), dtype=mb_dtype)
        else:
            saved_inputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            saved_outputs = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_fwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)
            incoming_bwd = jnp.zeros((V, m, *mb_shape), dtype=mb_dtype)

        g_params = jax.tree.map(jnp.zeros_like, p_per_rank)
        loss_acc = jnp.zeros((), dtype=jnp.float32)

        def si_get(arr, v, mb_idx):
            """Read."""
            return arr[mb_idx] if V == 1 else arr[v, mb_idx]

        def si_set(arr, v, mb_idx, val):
            """Write."""
            return arr.at[mb_idx].set(val) if V == 1 else arr.at[v, mb_idx].set(val)

        def g_add(acc, v, upd):
            """Accumulate."""
            if V == 1:
                return jax.tree.map(lambda a, b: a + b, acc, upd)
            return jax.tree.map(lambda a, b: a.at[v].add(b), acc, upd)

        for _t, row in enumerate(grid):
            for r, action in enumerate(row):
                if action is None:
                    continue
                mb = action.microbatch
                v = action.virtual_stage
                is_my = rank == r
                p_v = pick_virt(p_per_rank, v)
                logical = schedule.logical_at(r, v, n_stages)

                if action.phase == Phase.FWD:
                    x_in_source = xs[mb] if logical == 0 else si_get(incoming_fwd, v, mb)

                    def _fwd_branch(x_, _p=p_v):
                        """Active-rank forward: ``fwd_fn(params, x)``."""
                        return fwd_fn(_p, x_)

                    def _fwd_skip(x_):
                        """Inactive-rank forward: produce zeros so all ranks have a valid out value."""
                        return jnp.zeros_like(x_)

                    y = jax.lax.cond(is_my, _fwd_branch, _fwd_skip, x_in_source)
                    saved_inputs = jnp.where(is_my, si_set(saved_inputs, v, mb, x_in_source), saved_inputs)
                    if logical == n_logical - 1:
                        saved_outputs = jnp.where(is_my, si_set(saved_outputs, v, mb, y), saved_outputs)
                    next_loc = schedule.next_logical_loc(r, v, n_stages)
                    if next_loc is not None:
                        dp, dv = next_loc
                        if dp == r:
                            incoming_fwd = jnp.where(is_my, si_set(incoming_fwd, dv, mb, y), incoming_fwd)
                        else:
                            y_sent = jax.lax.ppermute(y, pp_axis, perm=[(r, dp)])
                            incoming_fwd = jnp.where(rank == dp, si_set(incoming_fwd, dv, mb, y_sent), incoming_fwd)

                elif action.phase in (Phase.BWD, Phase.BWD_I, Phase.BWD_W):
                    if logical == n_logical - 1:

                        def _loss_branch(_so=saved_outputs, _v=v, _mb=mb):
                            """Active terminal-rank branch: pull saved output, return ``(loss, g_y)``."""
                            y_out = si_get(_so, _v, _mb)
                            loss_mb, g_y_mb = loss_and_g_y(y_out, *(t_arr[_mb] for t_arr in targets))
                            return loss_mb.astype(jnp.float32), g_y_mb

                        def _loss_skip():
                            """Inactive-rank branch: zero loss and zero seed cotangent."""
                            return jnp.zeros((), dtype=jnp.float32), jnp.zeros(mb_shape, dtype=mb_dtype)

                        loss_mb, g_y_seed = jax.lax.cond(is_my, _loss_branch, _loss_skip)
                        if action.phase != Phase.BWD_W:
                            loss_acc = loss_acc + loss_mb
                        g_y = g_y_seed
                    else:
                        g_y = si_get(incoming_bwd, v, mb)
                    x_saved = si_get(saved_inputs, v, mb)

                    def _bwd_branch(args, _p=p_v):
                        """Active-rank backward: ``bwd_fn(params, x_saved, g_y) -> (g_p, g_x)``."""
                        x_, g_ = args
                        return bwd_fn(_p, x_, g_)

                    def _bwd_skip(args, _p=p_v):
                        """Inactive-rank backward: zero param-grads and zero input-grad."""
                        x_, _g = args
                        return (jax.tree.map(jnp.zeros_like, _p), jnp.zeros_like(x_))

                    g_p_virt, g_x = jax.lax.cond(is_my, _bwd_branch, _bwd_skip, (x_saved, g_y))
                    if action.phase != Phase.BWD_I:
                        gated = jax.tree.map(lambda u, _m=is_my: jnp.where(_m, u, jnp.zeros_like(u)), g_p_virt)
                        g_params = g_add(g_params, v, gated)
                    prev_loc = _prev_logical_loc(schedule, r, v, n_stages)
                    if action.phase != Phase.BWD_W and prev_loc is not None:
                        sp, sv = prev_loc
                        if sp == r:
                            incoming_bwd = jnp.where(is_my, si_set(incoming_bwd, sv, mb, g_x), incoming_bwd)
                        else:
                            g_sent = jax.lax.ppermute(g_x, pp_axis, perm=[(r, sp)])
                            incoming_bwd = jnp.where(rank == sp, si_set(incoming_bwd, sv, mb, g_sent), incoming_bwd)

        total_loss = jax.lax.psum(loss_acc, pp_axis)
        mean_loss = total_loss / jnp.asarray(m, dtype=total_loss.dtype)
        return mean_loss, _add0(g_params)

    return body
