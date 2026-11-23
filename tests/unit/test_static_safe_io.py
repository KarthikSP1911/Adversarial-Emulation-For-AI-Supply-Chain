import os

import pytest

from modelscan.static import safe_io


def test_regular_file_ok(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    assert safe_io.inspect_path(p, tmp_path).ok
    assert safe_io.read_head(p, 2) == b"ab"


def test_symlink_not_followed(tmp_path):
    (tmp_path / "real").write_bytes(b"x")
    link = tmp_path / "model.bin"
    link.symlink_to("/etc/passwd")
    pc = safe_io.inspect_path(link, tmp_path)
    assert not pc.ok and pc.problem == "symlink"
    with pytest.raises(OSError):
        safe_io.open_regular(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_refused(tmp_path):
    p = tmp_path / "model.bin"
    os.mkfifo(p)
    pc = safe_io.inspect_path(p, tmp_path)
    assert not pc.ok and pc.problem == "special"


def test_size_limit(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"x" * 10)
    pc = safe_io.inspect_path(p, tmp_path, max_size=5)
    assert pc.problem == "too_large"


def test_read_range_limits(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"x" * 10)
    with safe_io.open_regular(p) as fh:
        with pytest.raises(safe_io.ScanLimitError):
            safe_io.read_range(fh, 0, 100, limit=50)
        with pytest.raises(safe_io.ScanLimitError):
            safe_io.read_range(fh, 5, 10, limit=50)


def test_deadline():
    d = safe_io.Deadline(0)
    with pytest.raises(safe_io.ScanLimitError):
        d.check()
