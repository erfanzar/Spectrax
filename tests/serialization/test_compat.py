# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Comprehensive tests for spectrax.serialization._compat."""

from __future__ import annotations

import pytest

from spectrax.serialization._compat import PyTree, flatten_dict


class TestPyTree:
    """PyTree is simply a type alias for Any."""

    def test_is_any(self):
        import typing

        assert PyTree is typing.Any


class TestFlattenDict:
    """Tests for flatten_dict."""

    def test_flatten_simple_dict(self):
        xs = {"a": 1, "b": 2}
        result = flatten_dict(xs)
        assert result == {("a",): 1, ("b",): 2}

    def test_flatten_nested_dict(self):
        xs = {"a": {"b": 1, "c": 2}, "d": 3}
        result = flatten_dict(xs)
        assert result == {("a", "b"): 1, ("a", "c"): 2, ("d",): 3}

    def test_flatten_with_separator(self):
        xs = {"a": {"b": 1}}
        result = flatten_dict(xs, sep=".")
        assert result == {"a.b": 1}

    def test_flatten_deeply_nested(self):
        xs = {"a": {"b": {"c": {"d": 42}}}}
        result = flatten_dict(xs, sep="/")
        assert result == {"a/b/c/d": 42}

    def test_flatten_empty_dict(self):
        result = flatten_dict({})
        assert result == {}

    def test_flatten_keep_empty_nodes(self):
        xs = {"a": {}, "b": 1}
        result = flatten_dict(xs, keep_empty_nodes=True)
        assert result == {("a",): {}, ("b",): 1}

    def test_flatten_with_is_leaf(self):
        xs = {"a": {"b": [1, 2, 3]}, "c": 4}
        result = flatten_dict(xs, is_leaf=lambda _path, obj: isinstance(obj, list))
        assert result == {("a", "b"): [1, 2, 3], ("c",): 4}

    def test_flatten_non_dict_mapping(self):
        from collections import OrderedDict

        xs = OrderedDict([("a", 1), ("b", 2)])
        result = flatten_dict(xs, fumap=True)
        assert result == {("a",): 1, ("b",): 2}

    def test_flatten_non_dict_without_fumap_raises(self):
        with pytest.raises(TypeError):
            flatten_dict([1, 2, 3])

    def test_flatten_preserves_values(self):
        import numpy as np

        arr = np.array([1.0, 2.0])
        xs = {"layer": {"weight": arr}}
        result = flatten_dict(xs, sep=".")
        assert result["layer.weight"] is arr
