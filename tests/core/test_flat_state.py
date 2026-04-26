# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :meth:`State.flatten` / :meth:`State.from_flat`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.core.graph import export
from spectrax.core.state import State
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_flatten_produces_slash_separated_keys():
    """Flat keys have form ``collection/path``."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)
    flat = state.flatten()
    for k in flat:
        assert "/" in k


def test_flatten_from_flat_roundtrip():
    """``from_flat(flatten(state))`` reconstructs the state."""
    m = Linear(2, 2, rngs=Rngs(0))
    _gdef, state = export(m)
    restored = State.from_flat(state.flatten())
    for c in state:
        assert c in restored
        for p, v in state.raw()[c].items():
            assert jnp.array_equal(v, restored.raw()[c][p])


def test_from_flat_rejects_malformed_keys():
    """Keys without ``/`` raise."""
    with pytest.raises(ValueError):
        State.from_flat({"bad_key": jnp.zeros(())})
