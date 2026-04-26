# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rich tabular summary of a module, including per-submodule parameter/byte counts."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..core.graph import iter_modules, live_variables
from ..core.module import Module
from ..core.variable import Variable

__all__ = ["count_bytes", "count_parameters", "hlo_cost", "tabulate"]


def _var_size(v: Variable) -> tuple[int, int]:
    """Return ``(element_count, bytes)`` for a variable."""
    shape = getattr(v, "shape", ())
    n = 1
    for s in shape:
        try:
            n *= int(s)
        except Exception:
            n = 0
            break
    dtype = getattr(v, "dtype", None)
    try:
        itemsize = jnp.dtype(dtype).itemsize if dtype is not None else 4
    except Exception:
        itemsize = 4
    return n, n * itemsize


def count_parameters(module: Module, *, collection: str = "parameters") -> int:
    """Return the total number of scalar values across a collection."""
    total = 0
    for _, v in live_variables(module):
        if v.kind == collection:
            n, _ = _var_size(v)
            total += n
    return total


def count_bytes(module: Module, *, collection: str = "parameters") -> int:
    """Return the total byte count across a collection."""
    total = 0
    for _, v in live_variables(module):
        if v.kind == collection:
            _, b = _var_size(v)
            total += b
    return total


def tabulate(
    module: Module,
    *example_args: Any,
    depth: int | None = None,
    **example_kwargs: Any,
) -> str:
    """Return a PyTorch-style tabulated per-submodule report.

    Columns are ``(path, class, parameters, bytes)``. When example inputs
    are supplied the output shape is reported after running
    :func:`jax.eval_shape` on the module.
    """
    rows: list[tuple[str, str, str, str]] = []
    for path, mod in iter_modules(module):
        if depth is not None and path.count(".") + (1 if path else 0) > depth:
            continue
        parameters_here = 0
        bytes_here = 0
        prefix = path + ("." if path else "")
        for p, v in live_variables(mod):
            n, b = _var_size(v)
            if v.kind == "parameters":
                parameters_here += n
                bytes_here += b
            _ = p
            _ = prefix
        cls = type(mod).__name__
        rows.append((path or "(root)", cls, f"{parameters_here:,}", f"{bytes_here:,}"))

    w0 = max((len(r[0]) for r in rows), default=4)
    w1 = max((len(r[1]) for r in rows), default=5)
    w2 = max((len(r[2]) for r in rows), default=6)
    w3 = max((len(r[3]) for r in rows), default=5)
    header = f"{'path':<{w0}}  {'class':<{w1}}  {'parameters':>{w2}}  {'bytes':>{w3}}"
    lines: list[str] = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:>{w2}}  {r[3]:>{w3}}")
    lines.append("-" * len(header))
    total_parameters = count_parameters(module)
    total_bytes = count_bytes(module)
    lines.append(f"Total parameters: {total_parameters:,}  bytes: {total_bytes:,}")
    if example_args or example_kwargs:
        try:
            out = jax.eval_shape(lambda: module(*example_args, **example_kwargs))
            lines.append(f"Output: {out}")
        except Exception as e:
            lines.append(f"Output: (eval_shape failed: {e})")
    return "\n".join(lines)


def hlo_cost(module: Module, *example_args: Any, **example_kwargs: Any) -> dict[str, float]:
    """Return a dict with ``flops`` and ``bytes_accessed`` derived from XLA's cost model.

    Uses ``jax.jit(...).lower(...).compile().cost_analysis()`` on a pure
    wrapping of ``module(*example_args, **example_kwargs)``. Failing
    compile returns ``{}``.
    """
    try:

        def run(*args: Any, **kwargs: Any) -> Any:
            """Closure: ``module(*args, **kwargs)`` for ``jax.jit(...).lower(...)``."""
            return module(*args, **kwargs)

        lowered = jax.jit(run).lower(*example_args, **example_kwargs)
        compiled = lowered.compile()
        analysis = compiled.cost_analysis()
        if analysis is None:
            return {}
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        return {
            "flops": float(analysis.get("flops", 0.0)),
            "bytes_accessed": float(analysis.get("bytes accessed", analysis.get("bytes_accessed", 0.0))),
        }
    except Exception:
        return {}
