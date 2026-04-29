# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Base backend protocol and unified logger implementation."""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

import jax
import numpy as np

from spectrax._internal.logging import get_logger

_logger = get_logger(__name__)

Scalar = float | int
ArrayLike = tp.Union[np.ndarray, "jax.Array"]  # type: ignore[name-defined]


class BaseBackend(ABC):
    """Abstract interface for a logging backend."""

    @abstractmethod
    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Log a scalar value."""

    @abstractmethod
    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Log a histogram of values."""

    @abstractmethod
    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Log an image."""

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string."""

    @abstractmethod
    def log_hparams(self, hparams: dict[str, tp.Any]) -> None:
        """Log hyper-parameters."""

    def log_summary(self, metrics: dict[str, tp.Any]) -> None:
        """Log summary-level metrics (optional; default no-op)."""
        return None

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[tp.Any]],
        step: int,
    ) -> None:
        """Log a table (optional; default no-op)."""
        return None

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered writes."""

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""


class _NullBackend(BaseBackend):
    """No-op backend used when no backends are configured."""

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        pass

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        pass

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        pass

    def log_hparams(self, hparams: dict[str, tp.Any]) -> None:
        pass

    def log_summary(self, metrics: dict[str, tp.Any]) -> None:
        pass

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[tp.Any]],
        step: int,
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class Logger:
    """Unified logger that multiplexes writes to multiple backends.

    In distributed JAX settings only process ``0`` performs actual I/O;
    calls on other ranks are silently dropped.

    This class also exposes EasyDeL-compatible aliases so it can be used
    as a drop-in replacement for ``tensorboardX.SummaryWriter`` and
    ``flax.metrics.tensorboard.SummaryWriter``:

    * ``add_scalar(tag, value, step)`` → ``log_scalar(... )``
    * ``add_histogram(tag, values, step)`` → ``log_histogram(... )``
    * ``scalar(tag, value, step)`` → ``log_scalar(... )``  (Flax style)
    * ``histogram(tag, values, step)`` → ``log_histogram(... )``  (Flax style)

    Args:
        backends: List of :class:`BaseBackend` instances. If empty, a no-op
            null backend is used.
        auto_flush: If ``True`` (default), :meth:`flush` is called after
            every logging operation.

    Example::

        from spectrax.loggers import Logger, TensorBoardBackend, ConsoleBackend

        logger = Logger([
            TensorBoardBackend("./runs"),
            ConsoleBackend(),
        ])
        logger.log_scalar("loss", 0.5, step=100)
        logger.close()
    """

    def __init__(
        self,
        backends: list[BaseBackend] | None = None,
        *,
        auto_flush: bool = True,
    ):
        self._backends = backends or [_NullBackend()]
        self._auto_flush = auto_flush
        self._closed = False
        self._is_main = self._is_main_process()

    @staticmethod
    def _is_main_process() -> bool:
        try:
            return jax.process_index() == 0
        except RuntimeError:
            return True

    def _dispatch(self, method: str, *args: tp.Any, **kwargs: tp.Any) -> None:
        if self._closed or not self._is_main:
            return
        for backend in self._backends:
            try:
                getattr(backend, method)(*args, **kwargs)
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.{method} failed: {e}")
        if self._auto_flush:
            self.flush()

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Log a scalar value to all backends.

        Args:
            tag: Metric identifier, e.g. ``"loss/train"``.
            value: Scalar number.
            step: Training step.
        """
        self._dispatch("log_scalar", tag, value, step)

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Log a histogram to all backends.

        Args:
            tag: Metric identifier.
            values: Array of values to histogram.
            step: Training step.
        """
        self._dispatch("log_histogram", tag, values, step)

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Log an image to all backends.

        Args:
            tag: Image identifier.
            image: Image array (format depends on backend).
            step: Training step.
        """
        self._dispatch("log_image", tag, image, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text to all backends.

        Args:
            tag: Text identifier.
            text: String content.
            step: Training step.
        """
        self._dispatch("log_text", tag, text, step)

    def log_hparams(self, hparams: dict[str, tp.Any]) -> None:
        """Log hyper-parameters to all backends.

        Args:
            hparams: Flat or nested dict of hyper-parameters.
        """
        self._dispatch("log_hparams", hparams)

    def log_summary(self, metrics: dict[str, tp.Any]) -> None:
        """Log summary-level metrics (e.g. WandB ``run.summary``).

        Not all backends support this; those that don't silently ignore it.

        Args:
            metrics: Dictionary of summary metrics.
        """
        self._dispatch("log_summary", metrics)

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[tp.Any]],
        step: int,
    ) -> None:
        """Log a table to all backends.

        Backends that do not support tables (e.g. TensorBoard) silently
        ignore this call.

        Args:
            tag: Table identifier.
            columns: List of column header strings.
            rows: List of row values; each row is a list of cell values.
            step: Training step.
        """
        self._dispatch("log_table", tag, columns, rows, step)

    def add_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Alias for :meth:`log_scalar`."""
        self.log_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Alias for :meth:`log_histogram`."""
        self.log_histogram(tag, values, step)

    def add_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Alias for :meth:`log_image`."""
        self.log_image(tag, image, step)

    def add_text(self, tag: str, text: str, step: int) -> None:
        """Alias for :meth:`log_text`."""
        self.log_text(tag, text, step)

    def add_hparams(self, hparams: dict[str, tp.Any]) -> None:
        """Alias for :meth:`log_hparams`."""
        self.log_hparams(hparams)

    def scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Flax-style alias for :meth:`log_scalar`."""
        self.log_scalar(tag, value, step)

    def histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Flax-style alias for :meth:`log_histogram`."""
        self.log_histogram(tag, values, step)

    def image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Flax-style alias for :meth:`log_image`."""
        self.log_image(tag, image, step)

    def text(self, tag: str, textdata: str, step: int) -> None:
        """Flax-style alias for :meth:`log_text`."""
        self.log_text(tag, textdata, step)

    def hparams(self, hparams: dict[str, tp.Any]) -> None:
        """Flax-style alias for :meth:`log_hparams`."""
        self.log_hparams(hparams)

    def flush(self) -> None:
        """Flush all backends."""
        if not self._is_main:
            return
        for backend in self._backends:
            try:
                backend.flush()
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.flush failed: {e}")

    def close(self) -> None:
        """Close all backends and release resources."""
        if self._closed:
            return
        self._closed = True
        if not self._is_main:
            return
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.close failed: {e}")

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc: tp.Any) -> None:
        self.close()
