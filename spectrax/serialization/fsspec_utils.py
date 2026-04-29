# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Minimal fsspec helpers for TensorStore checkpoint I/O."""

import jax

from ._fs import is_remote_path


def should_write_shared_checkpoint_files(path) -> bool:
    """Whether the current process should write shared checkpoint metadata.

    Local files keep the historical behavior where every process may perform the
    shared setup/writes. Remote/object-store paths are restricted to process 0 to
    avoid cross-host contention on shared metadata files.
    """
    return not is_remote_path(path) or jax.process_index() == 0
