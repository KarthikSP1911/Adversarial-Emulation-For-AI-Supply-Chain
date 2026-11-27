import io
import stat
import tarfile
import zipfile

from static_fixtures import BENIGN_PICKLE, torch_zip

from modelscan.models import Severity
from modelscan.static import archive
from modelscan.static.base import Artifact, Kind


def _check(p, kind):
    return archive.check(Artifact(path=p, rel=p.name, size=p.stat().st_size, kind=kind))


def _rules(findings):
    return {f.rule_id for f in findings}


def test_clean_torch_zip(tmp_path):
    p = torch_zip(tmp_path / "m.pt", BENIGN_PICKLE)
    assert _check(p, Kind.TORCH_ZIP) == []


def test_traversal_and_symlink(tmp_path):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("../../.bashrc", b"x")
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, "/etc/passwd")
    findings = _check(p, Kind.ZIP)
    assert {"ARC-001", "ARC-002"} <= _rules(findings)
    assert any(f.rule_id == "ARC-001" and f.severity == Severity.CRITICAL for f in findings)


def test_duplicate_members(tmp_path):
    p = tmp_path / "m.pt"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("archive/data.pkl", BENIGN_PICKLE)
        zf.writestr("archive/data.pkl", b"\x80\x02N.")
    assert "ARC-004" in _rules(_check(p, Kind.TORCH_ZIP))


def test_zip_bomb_ratio(tmp_path):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big", b"\x00" * (20 * 1024 * 1024))
    assert "ARC-003" in _rules(_check(p, Kind.ZIP))


def test_prepended_data_and_executable(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("run.sh", b"#!/bin/sh\necho hi\n")
    p = tmp_path / "a.zip"
    p.write_bytes(b"JUNK" * 8 + buf.getvalue())
    assert {"ARC-008", "ARC-010"} <= _rules(_check(p, Kind.ZIP))


def test_local_header_name_mismatch(tmp_path):
    p = tmp_path / "m.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("aaaa.txt", b"hi")
    data = bytearray(p.read_bytes())
    i = data.index(b"aaaa.txt")  # first occurrence = local header
    data[i : i + 8] = b"bbbb.pkl"
    p.write_bytes(bytes(data))
    assert "ARC-007" in _rules(_check(p, Kind.ZIP))


def test_unexpected_member_in_torch_zip(tmp_path):
    p = torch_zip(tmp_path / "m.pt", BENIGN_PICKLE, extra={"archive/extra.bin": b"x"})
    assert "ARC-011" in _rules(_check(p, Kind.TORCH_ZIP))


def test_two_torch_folders(tmp_path):
    p = torch_zip(tmp_path / "m.pt", BENIGN_PICKLE, extra={"other/data.pkl": BENIGN_PICKLE})
    assert "ARC-011" in _rules(_check(p, Kind.TORCH_ZIP))


def test_tar_traversal_link_and_device(tmp_path):
    p = tmp_path / "a.tar"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("../evil")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
        sym = tarfile.TarInfo("ln")
        sym.type, sym.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(sym)
        dev = tarfile.TarInfo("dev")
        dev.type = tarfile.CHRTYPE
        tf.addfile(dev)
    findings = _check(p, Kind.TAR)
    assert {"ARC-001", "ARC-002"} <= _rules(findings)
    assert sum(f.rule_id == "ARC-002" for f in findings) == 2
