# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`PipelineSequential`: explicit stage container.

The primary way to declare pipeline stages in spectrax. Each positional
argument to the constructor is a distinct stage — the modules run
sequentially in eager mode (identical to :class:`spectrax.nn.Sequential`),
and under :func:`spectrax.pipeline.pipeline` each stage is assigned to
a separate device along the mesh's pipeline axis.

Declaring stages explicitly at the container level (rather than inline
via :func:`spectrax.pipeline.boundary` calls) makes the split
introspectable: :func:`spectrax.export` records the stage boundaries in
the :class:`~spectrax.GraphDef`, and ``model.stages`` returns the live
per-stage :class:`~spectrax.Module` instances for per-stage inspection
or surgery.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from spectrax.core.module import Module
from spectrax.core.variable import Variable

__all__ = ["PipelineSequential"]


class PipelineSequential(Module):
    """Sequential container that also declares pipeline stage boundaries.

    Constructed with a positional list of sub-modules; each one is a
    stage. In eager mode and on a single device, acts identically to
    :class:`spectrax.nn.Sequential` — calls each stage in order,
    threading the output of stage *k* into the input of stage *k+1*.

    Under :func:`spectrax.pipeline.pipeline`, the container's stage
    count must match the mesh's pipeline-axis size; the decorator uses
    the container's stage list to shard the model across that axis.

    Args:
        *stages: One :class:`Module` per pipeline stage, in
            application order. At least one stage is required.

    Attributes:
        num_stages: Integer count of pipeline stages.

    Example::

        class Block(spx.Module):
            def __init__(self, d, *, rngs):
                super().__init__()
                self.fc = nn.Linear(d, d, rngs=rngs)
            def forward(self, x):
                return jax.nn.gelu(self.fc(x))


        model = spx.pipeline.PipelineSequential(
            Block(128, rngs=spx.Rngs(0)),
            Block(128, rngs=spx.Rngs(1)),
            Block(128, rngs=spx.Rngs(2)),
            Block(128, rngs=spx.Rngs(3)),
        )

        # Eager:
        y = model(x)

        # Under pipeline:
        @spx.pipeline.pipeline(mesh=mesh, schedule=spx.pipeline.GPipe(microbatches=8))
        def step(m, x, y): ...
    """

    _spx_container_kind = "list"

    def __init__(self, *stages: Module) -> None:
        """Construct a pipeline container with ``len(stages)`` stages.

        The stages are stored on the private list attribute
        ``_spx_items`` so the framework's container traversal
        machinery (see :meth:`_spx_graph_children`) discovers them in
        order. The list is set via :func:`object.__setattr__` to
        bypass the standard module-field validation, which would
        otherwise reject a heterogeneous list of children.

        Args:
            *stages: One :class:`Module` per pipeline stage, in
                application order. At least one stage is required;
                all entries must be :class:`Module` instances.

        Raises:
            TypeError: If any positional argument is not a
                :class:`Module`.
            ValueError: If zero stages are passed.
        """
        super().__init__()
        if not stages:
            raise ValueError("PipelineSequential requires at least one stage.")
        for i, s in enumerate(stages):
            if not isinstance(s, Module):
                raise TypeError(f"PipelineSequential stage {i} must be a Module, got {type(s).__name__}.")
        object.__setattr__(self, "_spx_items", list(stages))

    @property
    def num_stages(self) -> int:
        """Number of pipeline stages currently held by the container."""
        return len(self._spx_items)

    @property
    def stages(self) -> list[Module]:
        """Snapshot list of per-stage :class:`Module` instances.

        The returned list is a fresh copy so callers can iterate or
        reorder without mutating the container's internal state.
        """
        return list(self._spx_items)

    def forward(self, x: Any) -> Any:
        """Apply every stage in order (eager / single-device path).

        Each stage is called as ``stage(x)`` and its return value
        becomes the input of the next stage. Under
        :func:`spectrax.pipeline.pipeline` the orchestrator calls
        each stage's ``forward`` on its assigned device instead of
        invoking this method directly — i.e. this code path is only
        used in eager / single-device execution.

        Args:
            x: Input passed to the first stage; subsequent stages
                receive the previous stage's output.

        Returns:
            The output of the final stage.
        """
        for stage in self._spx_items:
            x = stage(x)
        return x

    def _spx_graph_children(self) -> Iterator[tuple[int, Module | Variable]]:
        """Yield ``(index, stage)`` pairs in stage order.

        Used by :class:`~spectrax.Module`'s graph-traversal machinery
        to discover sub-modules and their state. Indexing by integer
        position keeps stage paths stable across pickling and
        container surgery.
        """
        yield from enumerate(self._spx_items)

    def __len__(self) -> int:
        """Return the number of stages (same as :attr:`num_stages`)."""
        return len(self._spx_items)

    def __getitem__(self, idx: int) -> Module:
        """Return the stage at integer position ``idx``.

        Args:
            idx: Zero-based stage index. Negative indices are
                resolved by Python's list semantics.

        Returns:
            The :class:`Module` registered at that position.
        """
        return self._spx_items[idx]

    def __iter__(self) -> Iterator[Module]:
        """Iterate over stages in application order."""
        return iter(self._spx_items)
