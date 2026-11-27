"""Unit tests for modelscan.static.formats.gguf (FMT-G-* rules).

Builds minimal valid GGUF byte sequences without any llama.cpp dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modelscan.static.formats.gguf import scan

# ---------------------------------------------------------------------------
# GGUF value types
# ---------------------------------------------------------------------------
_T_UINT32  = 4
_T_FLOAT32 = 6
_T_STRING  = 8
_T_ARRAY   = 9
_T_UINT64  = 10


# ---------------------------------------------------------------------------
# Minimal GGUF builder
# ---------------------------------------------------------------------------


def _encode_string(s: str) -> bytes:
    enc = s.encode("utf-8")
    return struct.pack("<Q", len(enc)) + enc


def _encode_kv(key: str, vtype: int, value_bytes: bytes) -> bytes:
    return _encode_string(key) + struct.pack("<I", vtype) + value_bytes


def _gguf(
    version: int = 3,
    kvs: list[tuple[str, int, bytes]] | None = None,
    tensor_count: int = 0,
) -> bytes:
    """Return a minimal GGUF file as bytes."""
    if kvs is None:
        kvs = []
    header = b"GGUF"
    header += struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count)
    header += struct.pack("<Q", len(kvs))
    for key, vtype, val in kvs:
        header += _encode_kv(key, vtype, val)
    return header


def _write(tmp_path: Path, data: bytes, name: str = "model.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGgufScanSkipsNonGguf:
    def test_pt_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"PK\x03\x04")
        assert scan(f) == []

    def test_gguf_ext_required(self, tmp_path: Path) -> None:
        f = tmp_path / "model.bin"
        f.write_bytes(b"GGUF")
        assert scan(f) == []


class TestFmtG001MagicCheck:
    """FMT-G-001: wrong magic bytes."""

    def test_wrong_magic_flagged(self, tmp_path: Path) -> None:
        f = _write(tmp_path, b"AAAA" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-001" in ids

    def test_correct_magic_no_flag(self, tmp_path: Path) -> None:
        f = _write(tmp_path, _gguf())
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-001" not in ids


class TestFmtG005Version:
    """FMT-G-005: unsupported GGUF version."""

    def test_version_99_flagged(self, tmp_path: Path) -> None:
        f = _write(tmp_path, _gguf(version=99))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-005" in ids

    def test_version_3_no_flag(self, tmp_path: Path) -> None:
        f = _write(tmp_path, _gguf(version=3))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-005" not in ids


class TestFmtG002SuspiciousString:
    """FMT-G-002: suspicious string in metadata."""

    def test_shell_command_in_metadata(self, tmp_path: Path) -> None:
        val = _encode_string("/bin/bash -c 'id'")
        kvs = [("model.name", _T_STRING, val)]
        f = _write(tmp_path, _gguf(kvs=kvs))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-002" in ids

    def test_url_in_metadata(self, tmp_path: Path) -> None:
        val = _encode_string("http://evil.example.com/payload.sh")
        kvs = [("extra.url", _T_STRING, val)]
        f = _write(tmp_path, _gguf(kvs=kvs))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-002" in ids

    def test_clean_string_no_flag(self, tmp_path: Path) -> None:
        val = _encode_string("LLaMA-3-8B")
        kvs = [("general.name", _T_STRING, val)]
        f = _write(tmp_path, _gguf(kvs=kvs))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-002" not in ids


class TestFmtG003ChatTemplate:
    """FMT-G-003: Jinja2 control-flow in chat_template."""

    def test_for_loop_flagged(self, tmp_path: Path) -> None:
        tpl = "{% for x in messages %}{{ x.content }}{% endfor %}"
        val = _encode_string(tpl)
        kvs = [("tokenizer.chat_template", _T_STRING, val)]
        f = _write(tmp_path, _gguf(kvs=kvs))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-003" in ids

    def test_simple_expression_no_flag(self, tmp_path: Path) -> None:
        # Only {{ }} expressions, no {% %} control-flow
        tpl = "{{ message.content }}"
        val = _encode_string(tpl)
        kvs = [("tokenizer.chat_template", _T_STRING, val)]
        f = _write(tmp_path, _gguf(kvs=kvs))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-G-003" not in ids
