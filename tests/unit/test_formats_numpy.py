"""Unit tests for modelscan.static.formats.numpy (FMT-N-* rules).

Builds minimal .npy and .npz payloads in-memory without requiring NumPy.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from modelscan.static.formats.numpy import scan

# ---------------------------------------------------------------------------
# .npy builder
# ---------------------------------------------------------------------------


def _npy_header_v1(dtype_str: str = "float32", shape: tuple = (1,)) -> bytes:
    """Build a NumPy v1.0 .npy file header (magic + version + header dict)."""
    magic = b"\x93NUMPY"
    major, minor = 1, 0
    # The header dict as bytes
    header_dict = f"{{'descr': '<{dtype_str}', 'fortran_order': False, 'shape': {shape!s}, }}"
    # Pad to 64-byte alignment
    header_bytes = header_dict.encode("latin-1")
    pad = 64 - ((len(magic) + 4 + len(header_bytes)) % 64)
    if pad == 64:
        pad = 0
    header_bytes = header_bytes + b" " * pad + b"\n"
    hdr = magic + bytes([major, minor]) + struct.pack("<H", len(header_bytes)) + header_bytes
    return hdr


def _npy_object_header_v1() -> bytes:
    """Build a .npy header declaring object dtype."""
    magic = b"\x93NUMPY"
    major, minor = 1, 0
    header_dict = "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    header_bytes = header_dict.encode("latin-1")
    pad = 64 - ((len(magic) + 4 + len(header_bytes)) % 64)
    if pad == 64:
        pad = 0
    header_bytes = header_bytes + b" " * pad + b"\n"
    return magic + bytes([major, minor]) + struct.pack("<H", len(header_bytes)) + header_bytes


def _write_npy(tmp_path: Path, data: bytes, name: str = "arr.npy") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _write_npz(tmp_path: Path, members: dict[str, bytes], name: str = "data.npz") -> Path:
    p = tmp_path / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname, content in members.items():
            zf.writestr(fname, content)
    p.write_bytes(buf.getvalue())
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNumpyScanSkipsNonNumpy:
    def test_pt_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"PK\x03\x04")
        assert scan(f) == []

    def test_txt_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "text.txt"
        f.write_text("hello")
        assert scan(f) == []


class TestFmtN005BadMagic:
    """FMT-N-005: missing .npy magic bytes."""

    def test_wrong_magic_flagged(self, tmp_path: Path) -> None:
        f = _write_npy(tmp_path, b"\x00\x00\x00\x00" + b"\x00" * 20)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-005" in ids

    def test_valid_magic_no_flag(self, tmp_path: Path) -> None:
        header = _npy_header_v1("f4", (3, 3))
        # append float data
        data = header + struct.pack("<9f", *range(9))
        f = _write_npy(tmp_path, data)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-005" not in ids


class TestFmtN002ObjectDtype:
    """FMT-N-002: object dtype declared in header."""

    def test_object_dtype_flagged(self, tmp_path: Path) -> None:
        header = _npy_object_header_v1()
        f = _write_npy(tmp_path, header)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-002" in ids

    def test_float_dtype_no_flag(self, tmp_path: Path) -> None:
        header = _npy_header_v1("f4", (1,))
        f = _write_npy(tmp_path, header + struct.pack("<f", 1.0))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-002" not in ids


class TestFmtN001EmbeddedPickle:
    """FMT-N-001: embedded pickle stream in .npy data section."""

    def test_pickle_in_data_section_flagged(self, tmp_path: Path) -> None:
        header = _npy_object_header_v1()
        # Append a pickle protocol 2 marker in the data section
        pickle_marker = b"\x80\x02}q\x00."
        f = _write_npy(tmp_path, header + pickle_marker)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-001" in ids

    def test_no_pickle_no_flag(self, tmp_path: Path) -> None:
        header = _npy_header_v1("f4", (1,))
        f = _write_npy(tmp_path, header + struct.pack("<f", 42.0))
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-001" not in ids


class TestFmtN003TraversalInNpz:
    """FMT-N-003: path traversal in .npz member name."""

    def test_dotdot_flagged(self, tmp_path: Path) -> None:
        header = _npy_header_v1("f4", (1,))
        content = header + struct.pack("<f", 1.0)
        members = {"../etc/passwd.npy": content}
        f = _write_npz(tmp_path, members)
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-003" in ids


class TestFmtN004MemberCount:
    """FMT-N-004: excessive .npz member count."""

    def test_many_members_flagged(self, tmp_path: Path) -> None:
        # Create just above MAX_NPZ_MEMBERS
        from modelscan.static.formats.numpy import MAX_NPZ_MEMBERS
        buf = io.BytesIO()
        header = _npy_header_v1("f4", (1,))
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(MAX_NPZ_MEMBERS + 1):
                zf.writestr(f"arr_{i}.npy", header + struct.pack("<f", float(i)))
        f = tmp_path / "big.npz"
        f.write_bytes(buf.getvalue())
        ids = {fi.rule_id for fi in scan(f)}
        assert "FMT-N-004" in ids
