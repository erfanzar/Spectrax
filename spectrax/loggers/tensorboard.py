# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native TensorBoard backend — no external TF/TB/Flax dependencies."""

from __future__ import annotations

import os
import struct
import time
import typing as tp
import zlib

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from spectrax.serialization._fs import _get_fs, joinpath, mkdir

from .base import ArrayLike, BaseBackend, Scalar

_ProtoCache: dict[str, tp.Any] = {}


def _masked_crc(data: bytes) -> int:
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8 & 0xFFFFFFFF


def _write_record(f: tp.BinaryIO, data: bytes) -> None:
    length = struct.pack("<Q", len(data))
    f.write(length)
    f.write(struct.pack("<I", _masked_crc(length)))
    f.write(data)
    f.write(struct.pack("<I", _masked_crc(data)))


def _get_protos() -> dict[str, tp.Any]:
    if _ProtoCache:
        return _ProtoCache

    pool = descriptor_pool.DescriptorPool()
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "tb.proto"
    file_proto.package = "tensorboard"
    file_proto.syntax = "proto3"

    def add_message(name: str, fields: list[tuple[int, str, str, int]]) -> None:
        msg = file_proto.message_type.add()
        msg.name = name
        for number, fname, ftype, label in fields:
            field = msg.field.add()
            field.number = number
            field.name = fname
            field.label = label
            if ftype.startswith("."):
                field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
                field.type_name = ftype
            else:
                field.type = getattr(descriptor_pb2.FieldDescriptorProto, f"TYPE_{ftype.upper()}")

    add_message(
        "TensorShapeProto_Dim",
        [(1, "size", "int64", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)],
    )
    add_message(
        "TensorShapeProto",
        [
            (1, "dim", ".tensorboard.TensorShapeProto_Dim", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (2, "unknown_rank", "bool", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )
    add_message(
        "TensorProto",
        [
            (1, "dtype", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "tensor_shape", ".tensorboard.TensorShapeProto", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "version_number", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (4, "tensor_content", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (5, "half_val", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (6, "float_val", "float", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (7, "double_val", "double", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (8, "int_val", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (9, "string_val", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (10, "scomplex_val", "float", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (11, "int64_val", "int64", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (12, "bool_val", "bool", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (13, "dcomplex_val", "double", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (14, "resource_handle_val", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (15, "variant_val", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (16, "uint32_val", "uint32", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (17, "uint64_val", "uint64", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
        ],
    )
    add_message(
        "SummaryMetadata_PluginData",
        [
            (1, "plugin_name", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "content", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )
    add_message(
        "SummaryMetadata",
        [
            (
                1,
                "plugin_data",
                ".tensorboard.SummaryMetadata_PluginData",
                descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            ),
            (2, "display_name", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "summary_description", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )
    add_message(
        "HistogramProto",
        [
            (1, "min", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "max", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "num", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (4, "sum", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (5, "sum_squares", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (6, "bucket_limit", "double", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
            (7, "bucket", "double", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED),
        ],
    )
    add_message(
        "Summary_Image",
        [
            (1, "height", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "width", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "colorspace", "int32", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (4, "encoded_image_string", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )
    add_message(
        "Summary_Value",
        [
            (7, "node_name", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (1, "tag", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (9, "metadata", ".tensorboard.SummaryMetadata", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "simple_value", "float", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "obsolete_old_style_histogram", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (4, "image", ".tensorboard.Summary_Image", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (5, "histo", ".tensorboard.HistogramProto", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (6, "audio", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (8, "tensor", ".tensorboard.TensorProto", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )
    add_message(
        "Summary",
        [(1, "value", ".tensorboard.Summary_Value", descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED)],
    )
    add_message(
        "Event",
        [
            (1, "wall_time", "double", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (2, "step", "int64", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (3, "file_version", "string", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (4, "graph_def", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (5, "summary", ".tensorboard.Summary", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (6, "log_message", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (7, "session_log", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (8, "tagged_run_metadata", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
            (9, "meta_graph_def", "bytes", descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL),
        ],
    )

    pool.Add(file_proto)
    factory = message_factory.GetMessageClassesForFiles(["tb.proto"], pool)
    _ProtoCache.update(factory)
    return _ProtoCache


class TensorBoardBackend(BaseBackend):
    """Backend that writes TensorBoard event files natively.

    No dependency on TensorFlow, tensorboardX, torch, or flax.
    Requires only ``google.protobuf`` (already pulled in by JAX).

    Args:
        log_dir: Directory to write TensorBoard event files.
    """

    def __init__(self, log_dir: str | os.PathLike[str]):
        log_dir = os.fspath(log_dir)
        mkdir(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._file: tp.BinaryIO | None = None
        self._open()

    def _open(self) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        path = joinpath(self._log_dir, f"events.out.tfevents.{ts}.{pid}")
        fs, plain = _get_fs(path)
        if fs is not None:
            self._file = fs.open(plain, "wb")
        else:
            self._file = open(path, "wb")

    def _ensure_open(self) -> tp.BinaryIO:
        if self._file is None or self._file.closed:
            self._open()
        return self._file

    def _write(self, data: bytes) -> None:
        f = self._ensure_open()
        _write_record(f, data)
        f.flush()

    def _event(self, step: int, summary: tp.Any) -> bytes:
        P = _get_protos()
        event = P["tensorboard.Event"]()
        event.wall_time = time.time()
        event.step = step
        event.summary.CopyFrom(summary)
        return event.SerializeToString()

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        P = _get_protos()
        summary = P["tensorboard.Summary"]()
        val = summary.value.add()
        val.tag = tag
        val.simple_value = float(value)
        self._write(self._event(step, summary))

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        P = _get_protos()
        arr = np.asarray(values).flatten()
        counts, edges = np.histogram(arr, bins=30)
        summary = P["tensorboard.Summary"]()
        val = summary.value.add()
        val.tag = tag
        histo = P["tensorboard.HistogramProto"]()
        histo.min = float(arr.min()) if arr.size else 0.0
        histo.max = float(arr.max()) if arr.size else 0.0
        histo.num = float(arr.size)
        histo.sum = float(arr.sum())
        histo.sum_squares = float(np.dot(arr, arr))
        histo.bucket_limit.extend(edges[1:].tolist())
        histo.bucket.extend(counts.astype(float).tolist())
        val.histo.CopyFrom(histo)
        self._write(self._event(step, summary))

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        height, width, channels = image.shape
        if channels != 3:
            raise ValueError("Image must have 1 or 3 channels")
        try:
            from PIL import Image as PILImage  #type:ignore
        except Exception as exc:
            raise RuntimeError("TensorBoardBackend image logging requires Pillow") from exc
        img = PILImage.fromarray(image)
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        P = _get_protos()
        summary = P["tensorboard.Summary"]()
        val = summary.value.add()
        val.tag = tag
        img_msg = P["tensorboard.Summary_Image"]()
        img_msg.height = height
        img_msg.width = width
        img_msg.colorspace = channels
        img_msg.encoded_image_string = png_bytes
        val.image.CopyFrom(img_msg)
        self._write(self._event(step, summary))

    def log_text(self, tag: str, text: str, step: int) -> None:
        P = _get_protos()
        summary = P["tensorboard.Summary"]()
        val = summary.value.add()
        val.tag = tag + "/text_summary"
        metadata = P["tensorboard.SummaryMetadata"]()
        plugin_data = P["tensorboard.SummaryMetadata_PluginData"]()
        plugin_data.plugin_name = "text"
        metadata.plugin_data.CopyFrom(plugin_data)
        val.metadata.CopyFrom(metadata)
        tensor = P["tensorboard.TensorProto"]()
        tensor.dtype = 7
        tensor.string_val.append(text.encode("utf-8"))
        shape = P["tensorboard.TensorShapeProto"]()
        dim = shape.dim.add()
        dim.size = 1
        tensor.tensor_shape.CopyFrom(shape)
        val.tensor.CopyFrom(tensor)
        self._write(self._event(step, summary))

    def log_hparams(self, hparams: dict[str, tp.Any]) -> None:
        P = _get_protos()
        summary = P["tensorboard.Summary"]()
        val = summary.value.add()
        val.tag = "_hparams_/session_start_info"
        metadata = P["tensorboard.SummaryMetadata"]()
        plugin_data = P["tensorboard.SummaryMetadata_PluginData"]()
        plugin_data.plugin_name = "hparams"
        metadata.plugin_data.CopyFrom(plugin_data)
        val.metadata.CopyFrom(metadata)
        tensor = P["tensorboard.TensorProto"]()
        tensor.dtype = 7
        tensor.string_val.append(b"")
        shape = P["tensorboard.TensorShapeProto"]()
        tensor.tensor_shape.CopyFrom(shape)
        val.tensor.CopyFrom(tensor)
        self._write(self._event(0, summary))
        for k, v in hparams.items():
            try:
                self.log_scalar(f"hparams/{k}", float(v), 0)
            except (TypeError, ValueError):
                self.log_text(f"hparams/{k}", str(v), 0)

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
            self._file = None
