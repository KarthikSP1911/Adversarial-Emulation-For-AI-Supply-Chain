"""Unit tests for modelscan.static.formats.torch (FMT-T-* rules).

Tests verify structural detection without needing a real PyTorch installation.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from modelscan.static.formats.torch import scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip(members: dict[str, bytes], tmp_path: Path, name: str = "model.pt") -> Path:
    """Create a minimal ZIP archive at *tmp_path/name* containing *members*."""
    target = tmp_path / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname, data in members.items():
            zf.writestr(fname, data)
    target.write_bytes(buf.getvalue())
    return target


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTorchScanSkipsNonTorch:
    """Files with unsupported extensions are silently ignored."""

    def test_txt_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "model.txt"
        f.write_text("hello")
        assert scan(f) == []

    def test_onnx_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "model.onnx"
        f.write_bytes(b"\x08\x01")
        assert scan(f) == []


class TestTorchScanHappyPath:
    """A clean, well-formed PyTorch checkpoint produces no findings."""

    def test_valid_checkpoint_no_findings(self, tmp_path: Path) -> None:
        pt = _make_zip({"archive/data.pkl": b"clean"}, tmp_path)
        findings = scan(pt)
        assert findings == [], findings


class TestFmtT001ExecutableMember:
    """FMT-T-001: executable member inside ZIP."""

    @pytest.mark.parametrize("ext", [".py", ".sh", ".exe", ".dll", ".so"])
    def test_executable_extension_flagged(self, tmp_path: Path, ext: str) -> None:
        members = {
            "archive/data.pkl": b"clean",
            f"payload{ext}": b"malicious",
        }
        pt = _make_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(pt)}
        assert "FMT-T-001" in ids, f"expected FMT-T-001 for {ext}"


class TestFmtT002DirectoryTraversal:
    """FMT-T-002: directory traversal via member name."""

    def test_dotdot_member_flagged(self, tmp_path: Path) -> None:
        members = {
            "../etc/passwd": b"root:x:0:0:root:/root:/bin/bash",
            "archive/data.pkl": b"clean",
        }
        pt = _make_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(pt)}
        assert "FMT-T-002" in ids


class TestFmtT004NoWeightMember:
    """FMT-T-004: checkpoint with no recognised weight member."""

    def test_no_data_pkl_flagged(self, tmp_path: Path) -> None:
        members = {"metadata.json": b"{}"}
        pt = _make_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(pt)}
        assert "FMT-T-004" in ids

    def test_data_pkl_present_no_flag(self, tmp_path: Path) -> None:
        members = {"archive/data.pkl": b"clean"}
        pt = _make_zip(members, tmp_path)
        ids = {f.rule_id for f in scan(pt)}
        assert "FMT-T-004" not in ids


class TestTorchScanNotAZip:
    """Non-ZIP .pt files are not flagged for structure (pickle engine handles them)."""

    def test_raw_bytes_not_a_zip(self, tmp_path: Path) -> None:
        pt = tmp_path / "raw.pt"
        pt.write_bytes(b"\x80\x02}q\x00.")  # a small pickle
        # scan() should return empty list (not a ZIP)
        assert scan(pt) == []
