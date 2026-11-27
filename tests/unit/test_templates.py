"""Unit tests for modelscan.static.templates (TPL-* rules).

Exercises the chat-template / Jinja2 injection scanner against in-memory files
only – no tokenizers or ML frameworks are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelscan.static.templates import scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, cfg: dict, name: str = "tokenizer_config.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_jinja(tmp_path: Path, template: str, name: str = "chat.jinja") -> Path:
    p = tmp_path / name
    p.write_text(template, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemplateScanSkipsIrrelevant:
    """Files that are not config / template files are ignored."""

    def test_py_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.py"
        f.write_text("import os")
        assert scan(f) == []

    def test_pt_file_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "model.pt"
        f.write_bytes(b"PK\x03\x04")
        assert scan(f) == []


class TestTpl001MroGadget:
    """TPL-001: MRO / __subclasses__ gadget chain."""

    def test_mro_in_jinja_flagged(self, tmp_path: Path) -> None:
        tpl = "{% for c in [].class.mro()[1].__subclasses__() %}{{ c }}{% endfor %}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-001" in ids

    def test_globals_in_config_flagged(self, tmp_path: Path) -> None:
        cfg = {"chat_template": "{{ config.__class__.__init__.__globals__ }}"}
        f = _write_config(tmp_path, cfg)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-001" in ids

    def test_safe_config_no_flag(self, tmp_path: Path) -> None:
        cfg = {"chat_template": "{% for msg in messages %}{{ msg.content }}{% endfor %}"}
        f = _write_config(tmp_path, cfg)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-001" not in ids


class TestTpl002OsAccess:
    """TPL-002: os.popen or subprocess in template."""

    def test_os_popen_flagged(self, tmp_path: Path) -> None:
        tpl = "{{ cycler.__init__.__globals__['os'].popen('id').read() }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        # TPL-001 or TPL-002 depending on which pattern fires first
        assert ids & {"TPL-001", "TPL-002"}

    def test_subprocess_run_flagged(self, tmp_path: Path) -> None:
        tpl = "{{ subprocess.run(['id'], capture_output=True).stdout }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-002" in ids


class TestTpl003BuiltinAbuse:
    """TPL-003: cycler / joiner / namespace built-in abuse."""

    def test_cycler_flagged(self, tmp_path: Path) -> None:
        tpl = "{{ cycler('even', 'odd').next() }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-003" in ids

    def test_namespace_flagged(self, tmp_path: Path) -> None:
        tpl = "{% set ns = namespace(count=0) %}{{ ns.count }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-003" in ids


class TestTpl004AttrFilter:
    """TPL-004: attr() filter bypass."""

    def test_attr_filter_flagged(self, tmp_path: Path) -> None:
        tpl = "{{ ''|attr('__class__')|attr('__mro__')[1] }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-004" in ids


class TestTpl005ExternalUrl:
    """TPL-005: URL in template (medium severity)."""

    def test_http_url_flagged(self, tmp_path: Path) -> None:
        tpl = "{{ 'http://evil.example.com/exfil' }}"
        f = _write_jinja(tmp_path, tpl)
        ids = {fi.rule_id for fi in scan(f)}
        assert "TPL-005" in ids


class TestTokenizerJson:
    """tokenizer.json is also inspected if it contains a chat_template field."""

    def test_chat_template_in_tokenizer_json(self, tmp_path: Path) -> None:
        payload = {
            "added_tokens": [],
            "chat_template": "{{ ''|attr('__class__') }}",
        }
        f = tmp_path / "tokenizer.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        ids = {fi.rule_id for fi in scan(f)}
        assert ids  # at least one finding
