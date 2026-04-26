# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto-split a single :class:`spectrax.Module` into pipeline stages.

The user writes one Module — typically ``embed`` + ``blocks: ModuleList``
+ ``head`` — and :func:`auto_split` slices it into ``n_pp`` per-rank
stages without any explicit pipeline annotations.

**Default behavior** (no annotations):

* Stage 0 gets everything declared before ``blocks`` in ``__init__``
  (``embed``, etc.).
* Stage ``n_pp - 1`` gets everything declared after ``blocks``
  (``head``, ``norm_f``, etc.).
* ``blocks`` is evenly sliced across all stages.

**Manual stage assignment** via ``pp_stage``:

Any :class:`Module` child, including individual blocks inside the repeated
``ModuleList``, can carry a ``pp_stage`` attribute to override automatic
placement::

    model.embed.pp_stage = 0        # explicit: rank 0
    model.blocks[0].pp_stage = 0    # explicit block placement
    model.head.pp_stage = "last"    # explicit: last rank
    model.aux_head.pp_stage = 2     # put on rank 2

Supported values:

* ``int`` — stage index (0-based). ``-1`` means last stage.
* ``"first"`` — alias for 0.
* ``"last"`` — alias for ``n_pp - 1``.

When ``pp_stage`` is set, that module is placed on the requested
stage regardless of whether it appears before or after ``blocks``
in ``__init__`` order. Modules without ``pp_stage`` fall back to
the default pre/post auto-detection.
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from inspect import Parameter

from ...core.containers import ModuleList
from ...core.module import Module

__all__ = ["auto_split", "split_block_stack"]


def _resolve_stage(pp_stage: int | str, n_pp: int) -> int:
    """Resolve a ``pp_stage`` annotation to a concrete stage index."""
    if isinstance(pp_stage, str):
        if pp_stage == "first":
            return 0
        if pp_stage == "last":
            return n_pp - 1
        raise ValueError(f"pp_stage must be an int, 'first', or 'last'; got {pp_stage!r}.")
    idx = int(pp_stage)
    if idx < 0:
        idx = n_pp + idx
    if not 0 <= idx < n_pp:
        raise ValueError(f"pp_stage={pp_stage} resolves to index {idx}, out of range for n_pp={n_pp}.")
    return idx


def _default_block_stage(index: int, n_blocks: int, n_pp: int) -> int:
    """Map block ``index`` to a balanced contiguous default pipeline rank."""
    return min(n_pp - 1, (index * n_pp) // n_blocks)


def _block_stage_indices(blocks: list[Module], n_pp: int) -> list[int]:
    """Resolve per-block stage ownership.

    Explicit ``block.pp_stage`` annotations win. Unannotated blocks use the
    balanced contiguous default mapping. The resolved sequence must be
    monotonic, otherwise stage order would no longer match model order.
    """
    n_blocks = len(blocks)
    stages: list[int] = []
    for i, block in enumerate(blocks):
        pp_stage = getattr(block, "pp_stage", None)
        stages.append(
            _resolve_stage(pp_stage, n_pp) if pp_stage is not None else _default_block_stage(i, n_blocks, n_pp)
        )
    for i, (prev, cur) in enumerate(itertools.pairwise(stages), start=1):
        if cur < prev:
            raise ValueError(
                "Block-level pp_stage annotations must be non-decreasing in block order; "
                f"block {i - 1} maps to stage {prev}, but block {i} maps to stage {cur}."
            )
    return stages


def _forward_positional_params(module: Module) -> list[inspect.Parameter]:
    """Return meaningful positional params from ``module.forward``.

    Wrappers such as ``spx.remat`` often expose ``forward(*args, **kwargs)``;
    those generic varargs should not trigger multi-carry warnings.
    """
    sig = inspect.signature(module.forward)
    return [
        p
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]


class _StageWrapper(Module):
    """One pipeline stage's worth of submodules, chained ``pre -> blocks -> post``.

    Supports both single-arg and multi-arg block signatures. When a
    block returns a tuple, the tuple is unpacked as positional args
    for the next block. When it returns a single value, it is wrapped
    in a 1-tuple. This lets blocks like::

        def forward(self, hidden, mask, pos_ids, kv_cache=None):
            ...
            return hidden, mask, pos_ids, updated_kv_cache

    chain naturally through the pipeline — all args flow through, only
    the ones the block modifies change between stages.
    """

    def __init__(
        self,
        pre: list[tuple[str, Module]],
        blocks: list[Module],
        post: list[tuple[str, Module]],
        carry_indices: tuple[int, ...] = (0,),
    ):
        """Store pre/blocks/post submodules as attributes."""
        super().__init__()
        for name, m in pre:
            setattr(self, name, m)
        self.blocks = ModuleList(blocks)
        for name, m in post:
            setattr(self, name, m)
        self.pre_names = tuple(name for name, _ in pre)
        self.post_names = tuple(name for name, _ in post)
        self.carry_indices = carry_indices

    def forward(self, *args, **kwargs):
        """Run pre -> blocks -> post, threading carry + broadcast args.

        Args are split into **carry** (change each block) and
        **broadcast** (same for all blocks) by carry_indices.
        Blocks receive (*carry, *broadcast) and return updated
        carry values. Pre/post modules receive only the first carry arg.

        Example: carry_indices=(0, 3) with args
        (hidden, mask, pos_ids, kv_cache) means carry is
        (hidden, kv_cache) and broadcast is (mask, pos_ids).
        Block called as blk(hidden, kv_cache, mask, pos_ids)
        returns (new_hidden, new_kv_cache) which replaces carry
        for the next block.
        """
        ci = self.carry_indices
        n_args = len(args)
        max_idx = max(ci) + 1 if ci else 1
        if n_args < max_idx:
            args = args + (None,) * (max_idx - n_args)

        for name in self.pre_names:
            first = getattr(self, name)(args[0], **kwargs)
            args = (first, *args[1:])
            kwargs = {}
        for blk in self.blocks:
            result = blk(*args)
            returned = result if isinstance(result, tuple) else (result,)
            args_list = list(args)
            for carry_pos, ret_val in zip(ci, returned, strict=False):
                args_list[carry_pos] = ret_val
            args = tuple(args_list)
        for name in self.post_names:
            first = getattr(self, name)(args[0])
            args = (first, *args[1:])

        return args if len(args) > 1 else args[0]


def split_block_stack(
    model: Module,
    n_pp: int,
    *,
    blocks_attr: str = "blocks",
    pre_attrs: list[str] | None = None,
    post_attrs: list[str] | None = None,
) -> list[Module]:
    """Split ``model`` into ``n_pp`` :class:`_StageWrapper` stages.

    Respects ``pp_stage`` annotations on Module children. Children
    with ``pp_stage`` set are placed on the specified stage; children
    without it fall back to auto-detection (before ``blocks`` → stage 0,
    after ``blocks`` → stage ``n_pp - 1``).

    Args:
        model: The full model with a ``blocks: ModuleList``.
        n_pp: Number of pipeline stages. Must divide ``len(blocks)``.
        blocks_attr: Name of the :class:`ModuleList` attribute.
        pre_attrs: Override auto-detected pre-block children.
        post_attrs: Override auto-detected post-block children.

    Returns:
        ``n_pp`` :class:`_StageWrapper` instances.
    """
    blocks = getattr(model, blocks_attr, None)
    if blocks is None:
        raise ValueError(
            f"model has no attribute {blocks_attr!r}; supply blocks_attr or define model.pipeline_split(n_pp)."
        )
    if not hasattr(blocks, "__len__"):
        raise TypeError(f"model.{blocks_attr} must be a ModuleList (or sized); got {type(blocks).__name__}.")
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError(f"model.{blocks_attr} must contain at least one block.")

    if pre_attrs is None or post_attrs is None:
        pre_auto, post_auto = _auto_pre_post(model, blocks_attr)
        if pre_attrs is None:
            pre_attrs = pre_auto
        if post_attrs is None:
            post_attrs = post_auto

    per_stage_extras: list[list[tuple[str, Module, str]]] = [[] for _ in range(n_pp)]

    for name in (*pre_attrs, *post_attrs):
        child = getattr(model, name)
        pp_stage = getattr(child, "pp_stage", None)
        if pp_stage is not None:
            stage_idx = _resolve_stage(pp_stage, n_pp)
            position = "pre" if name in pre_attrs else "post"
            per_stage_extras[stage_idx].append((name, child, position))
        else:
            if name in pre_attrs:
                per_stage_extras[0].append((name, child, "pre"))
            else:
                per_stage_extras[n_pp - 1].append((name, child, "post"))

    first_block = blocks[0]
    carry_indices = getattr(first_block, "block_carry", None)
    if carry_indices is None:
        params = _forward_positional_params(first_block)
        if len(params) > 1:
            param_names = [p.name for p in params]
            warnings.warn(
                f"{type(first_block).__name__}.forward takes {len(params)} "
                f"args ({', '.join(param_names)}) but has no `block_carry` "
                f"attribute. The pipeline will only pass the first arg "
                f"between blocks and discard the rest. Set "
                f"`block_carry = (0,)` for single-arg chains, or "
                f"`block_carry = (0, {len(params) - 1})` to carry the "
                f"first and last args through the pipeline. Example:\n"
                f"  class {type(first_block).__name__}(spx.Module):\n"
                f"      block_carry = (0, {len(params) - 1})",
                UserWarning,
                stacklevel=3,
            )
        carry_indices = (0,)
    if isinstance(carry_indices, int):
        carry_indices = (carry_indices,)
    carry_indices = tuple(carry_indices)

    block_list = list(blocks)
    block_stages = _block_stage_indices(block_list, n_pp)
    stages: list[Module] = []
    for r in range(n_pp):
        slab = [block for block, stage_idx in zip(block_list, block_stages, strict=True) if stage_idx == r]
        pre_pairs = [(name, mod) for name, mod, pos in per_stage_extras[r] if pos == "pre"]
        post_pairs = [(name, mod) for name, mod, pos in per_stage_extras[r] if pos == "post"]
        stages.append(_StageWrapper(pre=pre_pairs, blocks=slab, post=post_pairs, carry_indices=carry_indices))
    return stages


def auto_split(model: Module, n_pp: int) -> list[Module]:
    """Top-level entry: prefer ``model.pipeline_split(n_pp)`` if defined.

    Falls back to :func:`split_block_stack` which auto-detects
    ``blocks`` + pre/post Modules, respecting ``pp_stage`` annotations.
    """
    if hasattr(model, "pipeline_split"):
        out = model.pipeline_split(n_pp)
        if not isinstance(out, (list, tuple)) or len(out) != n_pp:
            raise ValueError(
                f"{type(model).__name__}.pipeline_split(n_pp={n_pp}) must "
                f"return a sequence of {n_pp} Modules; got {type(out).__name__} "
                f"of length {len(out) if hasattr(out, '__len__') else '?'}."
            )
        return list(out)
    return split_block_stack(model, n_pp)


def _auto_pre_post(model: Module, blocks_attr: str) -> tuple[list[str], list[str]]:
    """Walk ``model.__dict__`` in insertion order; classify each Module
    attribute as 'before blocks' or 'after blocks'."""
    pre: list[str] = []
    post: list[str] = []
    seen_blocks = False
    for name, val in vars(model).items():
        if name == blocks_attr:
            seen_blocks = True
            continue
        if isinstance(val, Module):
            (post if seen_blocks else pre).append(name)
    return pre, post
