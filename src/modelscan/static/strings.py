"""Suspicious strings and dynamic inputs in a pickle (PKL-010, PKL-011)."""

from __future__ import annotations

import re

from modelscan.models import Finding, Severity

from .globals_policy import INSPECTOR
from .pickle_vm import CallEvent, Container, Glob, VMResult, iter_consts

MAX_EVIDENCE_CHARS = 200
MIN_BLOB_LEN = 80

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("shell-path", re.compile(r"(?:^|[\s;|&])/(?:usr/)?bin/(?:ba|z|da|k)?sh\b|\bcmd(?:\.exe)?\s*/c\b")),
    ("powershell", re.compile(r"\bpowershell(?:\.exe)?\b|\bInvoke-Expression\b|\bIEX\b", re.I)),
    ("download-exec", re.compile(r"\b(?:curl|wget)\b[^\n]*(?:\||-o|-O|http)", re.I)),
    ("python-exec", re.compile(r"__import__\s*\(|\bexec\s*\(|\beval\s*\(|\bimport\s+(?:os|subprocess|socket)\b")),
    ("reverse-shell", re.compile(r"/dev/tcp/|\bnc\s+-[a-z]*e\b|\bsocket\.socket\b|\bpty\.spawn\b")),
    ("url", re.compile(r"\bhttps?://[^\s'\"<>]+", re.I)),
    ("sensitive-path", re.compile(r"(?:\.ssh/|/etc/(?:passwd|shadow)|\.aws/credentials)")),
    ("base64-blob", re.compile(rf"^[A-Za-z0-9+/=\s]{{{MIN_BLOB_LEN},}}$")),
]  # fmt: skip


def _called_strings(result: VMResult) -> set[str]:
    """Strings that flow into the arguments of some call."""
    out: set[str] = set()
    for call in result.calls:
        for arg in call.args:
            for const in iter_consts(arg):
                if isinstance(const, str):
                    out.add(const)
                elif isinstance(const, (bytes, bytearray)):
                    out.add(bytes(const).decode("utf-8", errors="replace"))
    return out


def _describe_callee(call: CallEvent) -> str:
    if isinstance(call.callee, Glob):
        return f"{call.callee.module}.{call.callee.name}"
    if isinstance(call.callee, Container):
        return f"<{call.callee.kind}>"
    return "<computed>"


def check(result: VMResult, file: str) -> list[Finding]:
    findings: list[Finding] = []
    flowing = _called_strings(result)
    reported: set[tuple[str, str]] = set()

    for const in result.strings:
        for kind, pattern in _PATTERNS:
            if not pattern.search(const.value):
                continue
            key = (kind, const.value)
            if key in reported:
                continue
            reported.add(key)
            into_call = const.value in flowing
            findings.append(
                Finding(
                    rule_id="PKL-010",
                    severity=Severity.HIGH if into_call else Severity.MEDIUM,
                    inspector=INSPECTOR,
                    file=file,
                    message=f"Suspicious string ({kind})"
                    + (" passed to a call" if into_call else ""),
                    offset=const.offset,
                    evidence={
                        "pattern": kind,
                        "string": const.value[:MAX_EVIDENCE_CHARS],
                        "passed_to_call": into_call,
                    },
                )
            )
            break  # one finding per string

    for ref in result.dynamic:
        findings.append(
            Finding(
                rule_id="PKL-011",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message="Global resolved dynamically (module/name not a literal)",
                offset=ref.offset,
                evidence={"module": ref.module, "name": ref.name},
            )
        )

    for call in result.calls:
        if isinstance(call.callee, Glob):
            continue
        findings.append(
            Finding(
                rule_id="PKL-011",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message=f"Call target is computed at load time ({_describe_callee(call)})",
                offset=call.offset,
                evidence={"opcode": call.opcode},
            )
        )
    return findings
