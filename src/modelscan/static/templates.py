"""Chat-template abuse scanner (TPL-*).

HuggingFace tokenizers ship a ``tokenizer_config.json`` that may include a
``chat_template`` field containing a **Jinja2 template**.  Many inference
frameworks (``transformers``, ``vllm``, ``llama-cpp-python``) render this
template at every request by passing it through Jinja2's ``Environment`` with
default settings that allow arbitrary Python execution via:

* ``{{ cycler.__init__.__globals__['os'].popen('cmd').read() }}``
* ``{% for x in [].class.mro()[1].subclasses() %}`` style gadget chains
* ``{{ namespace(x=config.__class__.__init__.__globals__) }}``

This module scans tokenizer config files statically – **without rendering any
template** – and flags patterns that indicate template-injection gadget usage.

Supported file names
--------------------
- ``tokenizer_config.json``  (HuggingFace standard)
- ``chat_template.jinja``    (standalone template files)
- ``*.jinja`` / ``*.j2``     (generic Jinja2 template files)
- ``tokenizer.json``         (may contain ``chat_template`` inside ``"added_tokens"``
                              or root-level ``"chat_template"`` key)

Rule IDs
--------
- TPL-001  Jinja2 gadget chain accessing __class__, __mro__, or __subclasses__
- TPL-002  Template calls os, subprocess, or other OS-interaction module
- TPL-003  Template contains cycler / joiner / namespace built-in abuse pattern
- TPL-004  Template contains a ``{{ ... | attr(...) }}`` filter bypass pattern
- TPL-005  Template embeds a suspicious URL or external resource reference
- TPL-006  Jinja2 raw-code execution via ``{% set ns = namespace() %}`` trick
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from modelscan.models import Finding, Severity

INSPECTOR = "static.templates"

# ---------------------------------------------------------------------------
# File selectors
# ---------------------------------------------------------------------------

#: Exact filenames that always carry a tokenizer config (checked case-insensitively).
_CONFIG_NAMES = frozenset({
    "tokenizer_config.json",
    "tokenizer.json",
})

#: Suffixes that are always Jinja2 templates.
_TEMPLATE_SUFFIXES = frozenset({".jinja", ".j2"})


def _is_template_file(path: Path) -> bool:
    name = path.name.lower()
    return name in _CONFIG_NAMES or path.suffix.lower() in _TEMPLATE_SUFFIXES


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

#: (rule_id, severity, name, compiled_pattern)
_RULES: list[tuple[str, Severity, str, re.Pattern[str]]] = [
    (
        "TPL-001", Severity.CRITICAL,
        "MRO/subclass gadget chain",
        re.compile(
            r"__(?:class__|mro__|subclasses__|bases__|base__|dict__|globals__"
            r"|builtins__|init__|code__|func__|self__)",
            re.IGNORECASE,
        ),
    ),
    (
        "TPL-002", Severity.CRITICAL,
        "OS / subprocess module access in template",
        re.compile(
            r"\b(?:os\.(?:popen|system|execv?e?[plr]*|spawn|getenv)|"
            r"subprocess\.(?:call|run|Popen|check_output)|"
            r"__import__\s*\(\s*['\"](?:os|subprocess|socket)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "TPL-003", Severity.HIGH,
        "Jinja2 built-in abuse (cycler / joiner / namespace)",
        re.compile(r"\b(?:cycler|joiner|namespace)\s*[\.\(]", re.IGNORECASE),
    ),
    (
        "TPL-004", Severity.HIGH,
        "attr() filter bypass pattern",
        re.compile(r"\|\s*attr\s*\(", re.IGNORECASE),
    ),
    (
        "TPL-005", Severity.MEDIUM,
        "External URL in template",
        re.compile(r"\bhttps?://[^\s'\"<>]+", re.IGNORECASE),
    ),
    (
        "TPL-006", Severity.HIGH,
        "namespace() globals extraction trick",
        re.compile(
            r"namespace\s*\(.*?config\s*\.\s*__class__"
            r"|__globals__\s*\[",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------


def _extract_templates(path: Path) -> list[tuple[str, str]]:
    """Return a list of ``(label, template_string)`` from *path*.

    For JSON files the template may be nested inside the config.  For raw
    Jinja2 files the whole file is the template.
    """
    suffix = path.suffix.lower()
    name   = path.name.lower()

    if suffix in _TEMPLATE_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return [(path.name, text)]
        except OSError:
            return []

    # JSON config files
    if name in _CONFIG_NAMES:
        try:
            raw = path.read_bytes()
            # Safety cap: don't parse huge files in full
            if len(raw) > 20 * 1024 * 1024:
                raw = raw[: 20 * 1024 * 1024]
            cfg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        templates: list[tuple[str, str]] = []

        def _recurse(obj: object, key_hint: str = "", depth: int = 0) -> None:
            if depth > 20:
                return
            if isinstance(obj, str):
                if "chat_template" in key_hint.lower() or "{%" in obj or "{{" in obj:
                    templates.append((f"{path.name}[{key_hint}]", obj))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _recurse(v, k, depth + 1)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _recurse(item, f"{key_hint}[{i}]", depth + 1)

        _recurse(cfg)
        return templates

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect a tokenizer config or Jinja2 template file for injection risks.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.  An empty list means no suspicious patterns
        were detected.
    """
    if not _is_template_file(path):
        return []

    findings: list[Finding] = []
    file = str(path)

    templates = _extract_templates(path)
    if not templates:
        return []

    reported: set[tuple[str, str]] = set()  # (rule_id, match) dedup set

    for label, template in templates:
        for rule_id, severity, name, pattern in _RULES:
            for m in pattern.finditer(template):
                dedup_key = (rule_id, m.group()[:80])
                if dedup_key in reported:
                    continue
                reported.add(dedup_key)
                # Provide a small window of context around the match.
                start = max(0, m.start() - 40)
                end   = min(len(template), m.end() + 40)
                context = template[start:end]
                findings.append(Finding(
                    rule_id=rule_id,
                    severity=severity,
                    inspector=INSPECTOR,
                    file=file,
                    message=f"Chat-template abuse: {name}",
                    offset=m.start(),
                    evidence={
                        "source": label,
                        "match": m.group()[:200],
                        "context": context,
                    },
                ))

    return findings
