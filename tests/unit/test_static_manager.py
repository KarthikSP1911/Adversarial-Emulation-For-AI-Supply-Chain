"""End-to-end tests of the static manager (no CLI needed)."""

import os
import struct
import zipfile

import pytest
from static_fixtures import (
    BENIGN_PICKLE,
    evil_pickle,
    safetensors_bytes,
    symlink_or_skip,
    torch_zip,
    write,
)

from modelscan.models import Severity, Status
from modelscan.static import scan


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())


def _rules(result):
    return {f.rule_id for f in result.findings}


def test_clean_model_folder(tmp_path):
    write(tmp_path / "config.json", b'{"model_type": "gpt2"}')
    write(tmp_path / "model.safetensors", safetensors_bytes())
    write(tmp_path / "vocab.txt", b"hello\nworld\n")
    result = scan(_files(tmp_path))
    assert result.status == Status.OK
    assert result.findings == []
    assert {r.path for r in result.manifest} == {"config.json", "model.safetensors", "vocab.txt"}
    assert all(r.sha256 and len(r.sha256) == 64 for r in result.manifest)


def test_pickle_disguised_as_safetensors_is_caught_twice(tmp_path):
    marker = tmp_path / "marker"
    write(tmp_path / "model.safetensors", evil_pickle(f"touch {marker}"))
    result = scan(_files(tmp_path))
    assert {"DET-001", "PKL-001"} <= _rules(result)
    assert not marker.exists()


def test_protocol0_pickle_in_bin_is_scanned(tmp_path):
    write(tmp_path / "pytorch_model.bin", b"cos\nsystem\n(S'touch /tmp/x'\ntR.")
    result = scan(_files(tmp_path))
    assert "PKL-001" in _rules(result)


def test_torch_zip_with_other_extension(tmp_path):
    torch_zip(tmp_path / "weights.zip", evil_pickle())
    result = scan(_files(tmp_path))
    assert "PKL-001" in _rules(result)
    assert all(f.file.startswith("weights.zip") for f in result.findings)


def test_benign_torch_zip_is_clean(tmp_path):
    torch_zip(tmp_path / "model.pt", BENIGN_PICKLE)
    result = scan(_files(tmp_path))
    assert result.findings == [], result.findings


def test_symlink_reported_not_followed(tmp_path):
    write(tmp_path / "model.safetensors", safetensors_bytes())
    symlink_or_skip(tmp_path / "evil.bin", "/etc/passwd")
    result = scan(_files(tmp_path))
    (f,) = [f for f in result.findings if f.rule_id == "IO-001"]
    assert f.file == "evil.bin"


def test_checker_crash_fails_closed(tmp_path, monkeypatch):
    from modelscan.static import formats

    def boom(_art):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(formats.safetensors, "check", boom)
    write(tmp_path / "model.safetensors", safetensors_bytes())
    result = scan(_files(tmp_path))
    assert result.status == Status.ERROR
    assert any(f.rule_id == "IO-090" and f.severity == Severity.HIGH for f in result.findings)


def test_pinned_manifest_mismatch(tmp_path):
    write(tmp_path / "model.safetensors", safetensors_bytes())
    result = scan(_files(tmp_path), pinned={"model.safetensors": "0" * 64})
    assert any(f.rule_id == "INT-001" for f in result.findings)


def test_broken_pin_file_fails_closed(tmp_path):
    write(tmp_path / "model.safetensors", safetensors_bytes())
    pins = write(tmp_path.parent / "pins.json", b"not json")
    result = scan(_files(tmp_path), root=tmp_path, pinned=pins)
    assert result.status == Status.ERROR
    assert "INT-006" in _rules(result)


def test_renamed_numpy_reported_as_not_inspected(tmp_path):
    write(tmp_path / "a.bin", b"\x93NUMPY\x01\x00" + b"\x00" * 10)
    result = scan(_files(tmp_path))
    assert "DET-004" in _rules(result)


@pytest.mark.parametrize(
    ("name", "data", "checker"),
    [
        ("a.npy", b"\x93NUMPY\x01\x00" + b"\x00" * 10, "numpy"),
        ("a.gguf", b"GGUF" + struct.pack("<IQQ", 3, 0, 0), "gguf"),
    ],
)
def test_inspected_format_has_no_det004(tmp_path, name, data, checker):
    write(tmp_path / name, data)
    result = scan(_files(tmp_path))
    assert "DET-004" not in _rules(result)
    assert checker in result.manifest[0].checkers


@pytest.mark.parametrize(
    ("name", "data", "checker", "rule"),
    [
        ("m.onnx", b"\xff\xff\xff\xff", "onnx", "FMT-O-004"),
        ("m.gguf", b"AAAA", "gguf", "FMT-G-001"),
        ("m.npy", b"\x00" * 24, "numpy", "FMT-N-005"),
        ("m.h5", b"\x00" * 24, "keras", "FMT-K-005"),
        ("chat.jinja", b"{{ x|attr('y') }}", "templates", "TPL-004"),
        ("modeling.py", b"import subprocess\n", "remote_code", "RCS-001"),
    ],
)
def test_path_scanner_runs_with_relative_name(tmp_path, name, data, checker, rule):
    write(tmp_path / "sub" / name, data)
    result = scan(_files(tmp_path), root=tmp_path)
    (f,) = [f for f in result.findings if f.rule_id == rule]
    assert f.file == f"sub/{name}"
    assert checker in result.manifest[0].checkers


def test_torch_scanner_runs_with_relative_name(tmp_path):
    torch_zip(tmp_path / "model.pt", BENIGN_PICKLE, extra={"archive/run.sh": b"id"})
    result = scan(_files(tmp_path))
    (f,) = [f for f in result.findings if f.rule_id == "FMT-T-001"]
    assert f.file == "model.pt"


def test_zip_slip_in_keras_like_zip(tmp_path):
    p = tmp_path / "model.keras"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("config.json", b"{}")
        zf.writestr("../../evil.py", b"import os")
    result = scan(_files(tmp_path))
    assert "ARC-001" in _rules(result)


def test_relative_names(tmp_path):
    sub = tmp_path / "sub"
    write(sub / "a.json", b"{}")
    write(tmp_path / "b.json", b"{}")
    result = scan(_files(tmp_path))
    assert {r.path for r in result.manifest} == {"sub/a.json", "b.json"}
    assert os.sep not in "".join(r.path for r in result.manifest if r.path != "sub/a.json")
