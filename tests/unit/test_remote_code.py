"""Unit tests for modelscan.static.remote_code (RCS-* rules).

Checks that static text-scanning of Python files correctly identifies
dangerous patterns without executing any code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelscan.static.remote_code import scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py(tmp_path: Path, source: str, name: str = "modeling.py") -> Path:
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRemoteCodeScanSkipsNonPython:
    """Non-.py files are silently skipped."""

    def test_txt_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("import os\nos.system('id')")
        assert scan(f) == []

    def test_gguf_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.gguf"
        f.write_bytes(b"GGUF\x03\x00\x00\x00")
        assert scan(f) == []


class TestRcs001HighRiskImport:
    """RCS-001: dangerous standard-library imports."""

    @pytest.mark.parametrize("module", [
        "subprocess", "socket", "ctypes", "pty", "multiprocessing",
        "importlib", "marshal", "ftplib",
    ])
    def test_import_flagged(self, tmp_path: Path, module: str) -> None:
        f = _write_py(tmp_path, f"import {module}\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-001" in ids, f"expected RCS-001 for 'import {module}'"

    def test_safe_import_no_flag(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "import math\nimport json\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-001" not in ids


class TestRcs002DynamicExec:
    """RCS-002: eval / exec / compile / __import__."""

    @pytest.mark.parametrize("snippet", [
        "eval('1+1')",
        "exec('import os')",
        "compile('x=1','<str>','exec')",
        "__import__('os')",
    ])
    def test_exec_pattern_flagged(self, tmp_path: Path, snippet: str) -> None:
        f = _write_py(tmp_path, f"result = {snippet}\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-002" in ids, f"expected RCS-002 for: {snippet}"

    def test_no_exec_no_flag(self, tmp_path: Path) -> None:
        f = _write_py(tmp_path, "x = 1 + 1\nprint(x)\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-002" not in ids


class TestRcs003NetworkConnectivity:
    """RCS-003: network calls."""

    @pytest.mark.parametrize("snippet", [
        "requests.get('http://evil.com')",
        "urllib.request.urlopen('http://evil.com')",
        "socket.connect(('evil.com', 443))",
    ])
    def test_network_call_flagged(self, tmp_path: Path, snippet: str) -> None:
        f = _write_py(tmp_path, snippet + "\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-003" in ids, f"expected RCS-003 for: {snippet}"


class TestRcs004SensitivePath:
    """RCS-004: reading sensitive file paths."""

    @pytest.mark.parametrize("snippet", [
        "open('/etc/passwd')",
        "open('.ssh/id_rsa')",
        "open('.aws/credentials')",
    ])
    def test_sensitive_path_flagged(self, tmp_path: Path, snippet: str) -> None:
        f = _write_py(tmp_path, snippet + "\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-004" in ids, f"expected RCS-004 for: {snippet}"


class TestRcs005Obfuscation:
    """RCS-005: base64/zlib/marshal decode chains."""

    @pytest.mark.parametrize("snippet", [
        "base64.b64decode(payload)",
        "zlib.decompress(data)",
        "marshal.loads(blob)",
        "codecs.decode(s, 'rot13')",
    ])
    def test_decode_chain_flagged(self, tmp_path: Path, snippet: str) -> None:
        f = _write_py(tmp_path, snippet + "\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-005" in ids, f"expected RCS-005 for: {snippet}"


class TestRcs006ShellInvocation:
    """RCS-006: os.system / subprocess / pty.spawn."""

    @pytest.mark.parametrize("snippet", [
        "os.system('id')",
        "os.popen('ls')",
        "subprocess.run(['ls', '-la'])",
        "subprocess.Popen(['nc', '-e', '/bin/sh'])",
        "pty.spawn('/bin/bash')",
    ])
    def test_shell_invocation_flagged(self, tmp_path: Path, snippet: str) -> None:
        f = _write_py(tmp_path, snippet + "\n")
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-006" in ids, f"expected RCS-006 for: {snippet}"

    def test_clean_code_no_flag(self, tmp_path: Path) -> None:
        source = (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "if __name__ == '__main__':\n"
            "    print(add(1, 2))\n"
        )
        f = _write_py(tmp_path, source)
        ids = {fi.rule_id for fi in scan(f)}
        assert "RCS-006" not in ids
