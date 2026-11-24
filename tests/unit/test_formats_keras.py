"""Unit tests for modelscan.static.formats.keras (FMT-K-* rules).

Tests cover .keras ZIP, SavedModel directory, and legacy .h5 detection without
any TensorFlow / Keras installation.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from modelscan.static.formats.keras import scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keras_zip(
    members: dict[str, bytes | str],
    tmp_path: Path,
    name: str = "model.keras",
) -> Path:
    target = tmp_path / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname, data in members.items():
            if isinstance(data, str):
                data = data.encode()
            zf.writestr(fname, data)
    target.write_bytes(buf.getvalue())
    return target


def _config(extra: dict) -> str:
    cfg = {"class_name": "Sequential", "config": {"layers": []}}
    cfg.update(extra)
    return json.dumps(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKerasScanSkipsNonKeras:
    def test_pt_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"PK\x03\x04")
        assert scan(f) == []

    def test_txt_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        assert scan(f) == []


class TestKerasCleanModel:
    def test_valid_keras_zip_no_findings(self, tmp_path: Path) -> None:
        cfg = _config({})
        keras = _make_keras_zip({"config.json": cfg}, tmp_path)
        findings = scan(keras)
        # may still have FMT-K-004 if saved_model.pb present, but here it's not
        assert all(f.rule_id not in {"FMT-K-001", "FMT-K-002", "FMT-K-003"} for f in findings)


class TestFmtK001LambdaKey:
    """FMT-K-001: lambda/function key in config.json."""

    def test_lambda_key_flagged(self, tmp_path: Path) -> None:
        cfg = json.dumps({
            "class_name": "Lambda",
            "config": {
                "function": "lambda x: __import__('os').system('id')"
            }
        })
        keras = _make_keras_zip({"config.json": cfg}, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-001" in ids

    def test_fn_key_flagged(self, tmp_path: Path) -> None:
        cfg = json.dumps({"fn": "lambda x: x"})
        keras = _make_keras_zip({"config.json": cfg}, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-001" in ids


class TestFmtK002DangerousCallable:
    """FMT-K-002: dangerous callable string in config."""

    def test_eval_string_flagged(self, tmp_path: Path) -> None:
        cfg = json.dumps({"activation": "eval('malicious code')"})
        keras = _make_keras_zip({"config.json": cfg}, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-002" in ids

    def test_exec_string_flagged(self, tmp_path: Path) -> None:
        cfg = json.dumps({"note": "exec(open('/etc/passwd').read())"})
        keras = _make_keras_zip({"config.json": cfg}, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-002" in ids


class TestFmtK003ExecutableMember:
    """FMT-K-003: executable member in .keras ZIP."""

    @pytest.mark.parametrize("ext", [".py", ".sh", ".so"])
    def test_exec_extension_flagged(self, tmp_path: Path, ext: str) -> None:
        cfg = _config({})
        members = {"config.json": cfg, f"exploit{ext}": b"#!/bin/sh\nid\n"}
        keras = _make_keras_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-003" in ids


class TestFmtK004SavedModelPb:
    """FMT-K-004: saved_model.pb presence (low severity)."""

    def test_saved_model_pb_flagged(self, tmp_path: Path) -> None:
        cfg = _config({})
        members = {"config.json": cfg, "saved_model.pb": b"\x0a\x0b"}
        keras = _make_keras_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-004" in ids


class TestFmtK005NoConfig:
    """FMT-K-005: .keras ZIP without config.json."""

    def test_no_config_flagged(self, tmp_path: Path) -> None:
        members = {"weights.h5": b"\x89HDF\r\n\x1a\n"}
        keras = _make_keras_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(keras)}
        assert "FMT-K-005" in ids


class TestH5MagicCheck:
    """Legacy .h5 files are checked for HDF5 magic."""

    def test_valid_hdf5_no_findings(self, tmp_path: Path) -> None:
        h5 = tmp_path / "model.h5"
        h5.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
        findings = scan(h5)
        # No FMT-K-005 for a valid HDF5 header
        assert all(f.rule_id != "FMT-K-005" for f in findings)

    def test_non_hdf5_flagged(self, tmp_path: Path) -> None:
        h5 = tmp_path / "model.h5"
        h5.write_bytes(b"\x80\x02}q\x00.")  # pickle, not HDF5
        ids = {f.rule_id for f in scan(h5)}
        assert "FMT-K-005" in ids
