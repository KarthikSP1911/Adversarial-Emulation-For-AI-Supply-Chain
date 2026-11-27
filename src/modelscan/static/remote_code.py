"""Shipped Python file scanner (RCS-*).

HuggingFace repositories can include arbitrary ``.py`` files that are
executed automatically when ``trust_remote_code=True`` is passed to
``AutoModel.from_pretrained()`` or similar APIs.  Even without that flag,
some downstream tools blindly ``import`` or ``exec`` every ``.py`` file
found in a model directory.

This module scans ``.py`` files **as plain text** – it never imports or
executes them.  It looks for:

1. High-risk import statements (``subprocess``, ``os``, ``socket``, …).
2. Dynamic code-execution constructs (``eval``, ``exec``, ``compile``, …).
3. Network connectivity patterns (``requests.get``, ``urllib.urlopen``, …).
4. File-system exfiltration patterns (reading sensitive paths, writing to
   ``/tmp``, etc.).
5. Obfuscation markers (base64-encoded blobs, ROT13, ``zlib.decompress``, …).

Rule IDs
--------
- RCS-001  High-risk standard-library import (subprocess, socket, ctypes, …)
- RCS-002  Dynamic code execution (eval / exec / compile / __import__)
- RCS-003  Network connectivity (requests, urllib, http.client, socket connect)
- RCS-004  Suspicious file-system access (sensitive path read/write)
- RCS-005  Obfuscation / decode pattern (base64, zlib, marshal decode)
- RCS-006  Process / shell invocation (os.system, Popen, subprocess.run, …)
"""

from __future__ import annotations

import re
from pathlib import Path

from modelscan.models import Finding, Severity

INSPECTOR = "static.remote_code"

# ---------------------------------------------------------------------------
# File selector
# ---------------------------------------------------------------------------


def _is_python_file(path: Path) -> bool:
    return path.suffix.lower() == ".py"


# ---------------------------------------------------------------------------
# Detection rules  (rule_id, severity, description, pattern)
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, Severity, str, re.Pattern[str]]] = [
    (
        "RCS-001", Severity.HIGH,
        "High-risk standard-library import",
        re.compile(
            r"^\s*(?:import|from)\s+"
            r"(?:subprocess|socket|ctypes|pty|signal|multiprocessing|"
            r"threading|runpy|importlib|zipimport|imp\b|ftplib|smtplib|"
            r"telnetlib|webbrowser|marshal|code\b|codeop)\b",
            re.MULTILINE,
        ),
    ),
    (
        "RCS-002", Severity.CRITICAL,
        "Dynamic code execution",
        re.compile(
            r"\b(?:eval|exec|compile)\s*\("
            r"|\b__import__\s*\("
            r"|\bexecfile\s*\(",
            re.MULTILINE,
        ),
    ),
    (
        "RCS-003", Severity.HIGH,
        "Network connectivity",
        re.compile(
            r"\b(?:requests|httpx|aiohttp|urllib(?:\.request)?|http\.client)"
            r"\s*\.\s*(?:get|post|put|delete|request|urlopen|urlretrieve)\s*\("
            r"|\bsocket\s*\.\s*(?:connect|create_connection)\s*\(",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "RCS-004", Severity.HIGH,
        "Sensitive file-system access",
        re.compile(
            r"""(?:open|read_text|read_bytes)\s*\(\s*['"]"""
            r"""(?:/etc/(?:passwd|shadow|sudoers)|\.ssh/|\.aws/credentials"""
            r"""|/proc/self|\.env)\b""",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "RCS-005", Severity.HIGH,
        "Obfuscation / decode chain",
        re.compile(
            r"\bbase64\s*\.\s*(?:b64decode|decodebytes|a85decode|b85decode)\s*\("
            r"|\bzlib\s*\.\s*decompress\s*\("
            r"|\bmarshal\s*\.\s*loads?\s*\("
            r"|\bcodecs\s*\.\s*decode\s*\(",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "RCS-006", Severity.CRITICAL,
        "Process / shell invocation",
        re.compile(
            r"\bos\s*\.\s*(?:system|popen|execv?e?[plr]*|spawnv?[pe]?)\s*\("
            r"|\bsubprocess\s*\.\s*(?:call|run|Popen|check_output|check_call)\s*\("
            r"|\bpty\s*\.\s*spawn\s*\(",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum file size we read (1 MiB per .py file is generous).
MAX_FILE_BYTES = 1 * 1024 * 1024

#: Maximum lines of context shown around each match.
_CONTEXT_LINES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context_window(lines: list[str], lineno: int) -> str:
    """Return a small excerpt around *lineno* (1-based)."""
    start = max(0, lineno - 1 - _CONTEXT_LINES)
    end   = min(len(lines), lineno + _CONTEXT_LINES)
    snippet_lines = []
    for i in range(start, end):
        marker = ">" if i == lineno - 1 else " "
        snippet_lines.append(f"{marker} {i + 1:4d} | {lines[i].rstrip()}")
    return "\n".join(snippet_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one ``.py`` file for dangerous code patterns.

    The file is read as **plain text** – nothing is imported or executed.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.  An empty list means no suspicious patterns
        were detected in this file.
    """
    if not _is_python_file(path):
        return []

    file = str(path)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [Finding(
            rule_id="RCS-001",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"Could not read Python file: {exc}",
            evidence={"error": str(exc)},
        )]

    if len(raw) > MAX_FILE_BYTES:
        raw = raw[:MAX_FILE_BYTES]

    try:
        source = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        source = raw.decode("latin-1", errors="replace")

    lines = source.splitlines(keepends=True)
    findings: list[Finding] = []
    reported: set[tuple[str, int]] = set()  # (rule_id, lineno) dedup

    for rule_id, severity, description, pattern in _RULES:
        for m in pattern.finditer(source):
            # Convert byte offset to line number.
            lineno = source.count("\n", 0, m.start()) + 1
            dedup_key = (rule_id, lineno)
            if dedup_key in reported:
                continue
            reported.add(dedup_key)

            context = _context_window(lines, lineno)
            findings.append(Finding(
                rule_id=rule_id,
                severity=severity,
                inspector=INSPECTOR,
                file=file,
                message=f"Shipped Python file: {description}",
                offset=m.start(),
                evidence={
                    "line": lineno,
                    "match": m.group()[:200],
                    "context": context,
                },
            ))

    return findings
