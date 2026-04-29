# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weights & Biases backend for the unified logger."""

from __future__ import annotations

import typing as tp

import numpy as np

from .base import ArrayLike, BaseBackend, Scalar

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandBBackend(BaseBackend):
    """Backend that writes to Weights & Biases.

    Requires ``wandb`` to be installed and initialized (either externally
    or via the ``init_kwargs`` passed here).

    Args:
        project: W&B project name. If ``None``, assumes ``wandb.init()``
            has already been called elsewhere.
        init_kwargs: Extra keyword arguments forwarded to ``wandb.init()``
            when ``project`` is provided.
    """

    def __init__(
        self,
        project: str | None = None,
        *,
        init_kwargs: dict[str, tp.Any] | None = None,
    ):
        if not _WANDB_AVAILABLE:
            raise RuntimeError("WandBBackend requires wandb. Install it:  pip install wandb")
        if project is not None:
            wandb.init(project=project, **(init_kwargs or {}))
        elif wandb.run is None:
            raise RuntimeError(
                "WandBBackend: wandb is not initialized. Pass ``project=...`` or call ``wandb.init()`` first."
            )

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        wandb.log({tag: float(value)}, step=step)

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        wandb.log({tag: wandb.Histogram(np.asarray(values))}, step=step)

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        wandb.log({tag: wandb.Image(np.asarray(image))}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        wandb.log({tag: wandb.Html(text)}, step=step)

    def log_hparams(self, hparams: dict[str, tp.Any]) -> None:
        wandb.config.update(hparams)

    def log_summary(self, metrics: dict[str, tp.Any]) -> None:
        if wandb.run is not None:
            wandb.run.summary.update(metrics)

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[tp.Any]],
        step: int,
    ) -> None:
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*row)
        wandb.log({tag: table}, step=step)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        wandb.finish()
