# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`MpMdMesh` — a :class:`jax.sharding.Mesh` with one axis tagged as MPMD.

The MPMD runtime (:func:`~spectrax.runtime.mpmd.sxcall`) needs to
know *which* axis of the mesh enumerates pipeline stages (each stage
is a separate compiled program — hence *multiple programs*). The
other axes stay SPMD: they're available for intra-stage FSDP / tensor
/ data parallelism.

This lets pipeline parallelism **compose** with the rest of spectrax's
sharding machinery. A model on a ``(pp=4, fsdp=2, tp=2)`` mesh gets a
4-stage pipeline where each stage is internally sharded over a 2x2
SPMD subgrid — no new infrastructure, just a ``MpMdMesh`` around the
existing :class:`~jax.sharding.Mesh`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

__all__ = ["MpMdMesh", "resolve_mpmd_mesh"]


def resolve_mpmd_mesh(mesh: Any) -> MpMdMesh:
    """Coerce ``mesh`` to a :class:`MpMdMesh`.

    Accepts:
    * :class:`~spectrax.sharding.SpxMesh` with ``mpmd_axis`` set
      (canonical, preferred).
    * :class:`MpMdMesh` (legacy direct usage; still supported).

    Raises if neither, or if an :class:`SpxMesh` was passed without an
    MPMD axis configured. The :class:`SpxMesh` import is performed
    lazily here to avoid a ``spectrax.sharding`` ↔ ``spectrax.runtime``
    module import cycle.
    """
    from ...sharding.mesh import SpxMesh

    if isinstance(mesh, SpxMesh):
        if not mesh.is_mpmd:
            raise ValueError(
                "MPMD pipeline runtimes need an SpxMesh built with "
                "create_mesh(..., mpmd_axis=<axis>); got an SpxMesh "
                "with no mpmd_axis set."
            )
        return mesh.mpmd_mesh
    if isinstance(mesh, MpMdMesh):
        return mesh
    raise TypeError(f"mesh must be SpxMesh or MpMdMesh, got {type(mesh).__name__}.")


@dataclass(frozen=True)
class MpMdMesh:
    """A :class:`~jax.sharding.Mesh` with one axis designated as MPMD.

    Wraps a regular JAX mesh plus a single axis name that enumerates
    pipeline stages. Every other axis remains SPMD, so intra-stage
    DP / FSDP / TP behave exactly as they do outside the pipeline.

    Example::

        import jax, numpy as np
        from jax.sharding import Mesh
        from spectrax.runtime.types import MpMdMesh

        # 2 pipeline stages, each sharded over 2 FSDP devices.
        devices = np.array(jax.devices()[:4]).reshape(2, 2)
        mm = MpMdMesh(Mesh(devices, ("pp", "fsdp")), "pp")

        assert mm.mpmd_dim == 2
        assert mm.spmd_axis_names == ("fsdp",)

    Attributes:
        jax_mesh: The underlying :class:`~jax.sharding.Mesh`.
        mpmd_axis_name: The axis along which pipeline stages are laid
            out. Must be one of ``jax_mesh.axis_names``.
    """

    jax_mesh: Mesh
    mpmd_axis_name: str

    def __post_init__(self) -> None:
        """Validate that the named MPMD axis exists and has positive size."""
        if self.mpmd_axis_name not in self.jax_mesh.axis_names:
            raise ValueError(f"mpmd_axis_name {self.mpmd_axis_name!r} must be one of {self.jax_mesh.axis_names}.")
        if self.mpmd_dim < 1:
            raise ValueError(f"MPMD axis {self.mpmd_axis_name!r} has size {self.mpmd_dim}; must be >= 1.")

    @property
    def mpmd_dim(self) -> int:
        """Number of pipeline stages (size along the MPMD axis)."""
        return int(self.jax_mesh.shape[self.mpmd_axis_name])

    @property
    def mpmd_axis(self) -> int:
        """Index of the MPMD axis within ``jax_mesh.axis_names``."""
        return self.jax_mesh.axis_names.index(self.mpmd_axis_name)

    @property
    def spmd_axis_names(self) -> tuple[str, ...]:
        """Axis names *other than* the MPMD axis — available for SPMD."""
        return tuple(n for n in self.jax_mesh.axis_names if n != self.mpmd_axis_name)

    def submesh(self, mpmd_idx: int) -> Mesh:
        """Return the sub-mesh for pipeline stage ``mpmd_idx``.

        The returned mesh keeps every axis of ``jax_mesh`` (including
        the MPMD axis, now size 1); only the devices laid out on the
        other axes remain. Specs meant to shard intra-stage should not
        mention :attr:`mpmd_axis_name`.

        Args:
            mpmd_idx: Which pipeline stage. Must satisfy
                ``0 <= mpmd_idx < mpmd_dim``.

        Returns:
            A :class:`~jax.sharding.Mesh` of the same rank as
            ``jax_mesh`` but with size 1 on the MPMD axis.
        """
        if not 0 <= mpmd_idx < self.mpmd_dim:
            raise IndexError(f"mpmd_idx {mpmd_idx} out of range [0, {self.mpmd_dim}).")
        devices = np.take(
            self.jax_mesh.devices,
            indices=[mpmd_idx],
            axis=self.mpmd_axis,
        )
        return Mesh(devices, self.jax_mesh.axis_names)

    def unstack(self) -> list[Mesh]:
        """Return one :class:`~jax.sharding.Mesh` per pipeline stage."""
        return [self.submesh(i) for i in range(self.mpmd_dim)]

    def sub_sharding(
        self,
        mpmd_idx: int,
        spec: PartitionSpec | Sequence[str | None] | None = None,
    ) -> NamedSharding:
        """Build a :class:`NamedSharding` on stage ``mpmd_idx``'s sub-mesh.

        The partition spec must not mention :attr:`mpmd_axis_name` —
        the sub-mesh has size 1 there so specs on that axis are
        meaningless (and XLA would reject them).

        Args:
            mpmd_idx: Which pipeline stage.
            spec: A :class:`PartitionSpec` or sequence of axis names
                describing intra-stage sharding. ``None`` (default) ->
                fully replicated within the sub-mesh.

        Returns:
            A :class:`~jax.sharding.NamedSharding` suitable for
            :func:`jax.device_put`.
        """
        if spec is None:
            spec = PartitionSpec()
        elif not isinstance(spec, PartitionSpec):
            spec = PartitionSpec(*spec)
        for axis in spec:
            if axis == self.mpmd_axis_name:
                raise ValueError(
                    f"PartitionSpec {spec!r} references the MPMD axis "
                    f"{self.mpmd_axis_name!r}, which has size 1 in each "
                    f"sub-mesh. Remove it from the spec."
                )
        return NamedSharding(self.submesh(mpmd_idx), spec)

    def device_mpmd_idx(self, device: Any) -> int:
        """Return the MPMD stage a device belongs to.

        Args:
            device: A :class:`jax.Device`.

        Raises:
            ValueError: If ``device`` is not part of ``jax_mesh``.
        """
        flat = self.jax_mesh.devices.reshape(-1)
        shape = self.jax_mesh.devices.shape
        for idx, d in enumerate(flat):
            if d is device:
                coords = np.unravel_index(idx, shape)
                return int(coords[self.mpmd_axis])
        raise ValueError(f"Device {device} is not part of this mesh.")

    def my_mpmd_axis_index(self) -> int | None:
        """Return the MPMD stage the current process owns, or ``None``.

        In a **single-process** run (the common case for local /
        tests), every MPMD group is visible from the one process, so
        there's no single "mine" — returns ``None``.

        In a **multi-process** run, returns the index of the single
        MPMD group whose devices live in the current process; raises
        :class:`ValueError` if the process straddles multiple groups
        (a misconfiguration).
        """
        if jax.process_count() == 1:
            return None
        local = set(jax.local_devices())
        found: set[int] = set()
        flat = self.jax_mesh.devices.reshape(-1)
        shape = self.jax_mesh.devices.shape
        for idx, d in enumerate(flat):
            if d in local:
                coords = np.unravel_index(idx, shape)
                found.add(int(coords[self.mpmd_axis]))
        if len(found) != 1:
            raise ValueError(
                f"Process owns devices across MPMD groups {sorted(found)}; "
                f"expected exactly one. Check mesh / process_id setup."
            )
        return next(iter(found))
