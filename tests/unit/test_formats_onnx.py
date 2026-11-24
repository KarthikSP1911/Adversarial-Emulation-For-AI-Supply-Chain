"""Unit tests for modelscan.static.formats.onnx (FMT-O-* rules).

We build minimal valid protobuf ModelProto payloads without installing the
``onnx`` package, using hand-encoded protobuf bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modelscan.static.formats.onnx import scan


# ---------------------------------------------------------------------------
# Minimal protobuf encoding helpers
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _field(number: int, wire: int, payload: bytes) -> bytes:
    tag = (number << 3) | wire
    if wire == 0:
        return _varint(tag) + payload
    if wire == 1:
        return _varint(tag) + payload
    if wire == 2:
        return _varint(tag) + _varint(len(payload)) + payload
    if wire == 5:
        return _varint(tag) + payload
    raise ValueError(f"Unknown wire type {wire}")


def _string_field(number: int, s: str) -> bytes:
    enc = s.encode("utf-8")
    return _field(number, 2, enc)


def _varint_field(number: int, value: int) -> bytes:
    return _field(number, 0, _varint(value))


def _make_model(
    *,
    opset_domain: str = "",
    extra_graph: bytes = b"",
) -> bytes:
    """Build a minimal ModelProto protobuf."""
    # field 1: ir_version = 7
    ir_version = _varint_field(1, 7)

    # field 2: opset_import = OperatorSetIdProto { domain="...", version=17 }
    opset = _string_field(1, opset_domain) + _varint_field(2, 17)
    opset_field = _field(2, 2, opset)

    # field 7: graph
    graph_field = _field(7, 2, extra_graph)

    return ir_version + opset_field + graph_field


def _make_onnx_file(tmp_path: Path, payload: bytes, name: str = "model.onnx") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnnxScanSkipsNonOnnx:
    def test_pt_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"\x08\x07")
        assert scan(f) == []


class TestFmtO004MagicCheck:
    """FMT-O-004: file doesn't start with protobuf header byte."""

    def test_non_proto_magic_flagged(self, tmp_path: Path) -> None:
        f = _make_onnx_file(tmp_path, b"\xff\xff\xff\xff")
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-004" in ids

    def test_valid_proto_magic_not_flagged(self, tmp_path: Path) -> None:
        payload = _make_model()
        f = _make_onnx_file(tmp_path, payload)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-004" not in ids


class TestFmtO001CustomDomain:
    """FMT-O-001: non-standard operator domain."""

    def test_standard_domain_no_flag(self, tmp_path: Path) -> None:
        payload = _make_model(opset_domain="ai.onnx")
        f = _make_onnx_file(tmp_path, payload)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-001" not in ids

    def test_custom_domain_flagged(self, tmp_path: Path) -> None:
        payload = _make_model(opset_domain="com.evil.malicious")
        f = _make_onnx_file(tmp_path, payload)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-001" in ids

    def test_empty_domain_no_flag(self, tmp_path: Path) -> None:
        payload = _make_model(opset_domain="")
        f = _make_onnx_file(tmp_path, payload)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-001" not in ids


class TestFmtO002ExternalData:
    """FMT-O-002: external tensor data reference."""

    def _make_external_tensor(self, path_str: str) -> bytes:
        """Build a TensorProto with data_location=EXTERNAL and path=path_str."""
        # field 9: data_location = 1 (EXTERNAL)
        data_loc = _varint_field(9, 1)
        # field 13: external_data = StringStringEntryProto { key="location", value=path_str }
        kv = _string_field(1, "location") + _string_field(2, path_str)
        ext_data = _field(13, 2, kv)
        tensor = data_loc + ext_data
        return tensor

    def test_external_data_flagged(self, tmp_path: Path) -> None:
        tensor_bytes = self._make_external_tensor("../../etc/passwd")
        # Build a graph with initializer (field 4)
        graph = _field(4, 2, tensor_bytes)
        payload = _make_model(extra_graph=graph)
        f = _make_onnx_file(tmp_path, payload)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-O-002" in ids
