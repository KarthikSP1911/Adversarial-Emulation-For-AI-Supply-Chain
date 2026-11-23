"""Pickle static analysis engine. Never loads or executes the file."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from modelscan.models import Finding, Severity

from . import globals_policy, lambdas, pickle_vm, strings
from .globals_policy import INSPECTOR

MAX_PICKLE_BYTES = 256 * 1024 * 1024  # read at most this much of one pickle
MAX_ZIP_MEMBERS = 10_000

# Always pickle containers: a parse failure is a finding.
PICKLE_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".joblib"}
# Only scanned when the content looks like a pickle or a zip.
AMBIGUOUS_SUFFIXES = {".bin", ".model", ".data"}


@dataclass
class PickleScan:
    findings: list[Finding] = field(default_factory=list)
    scanned: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def looks_like_pickle(head: bytes) -> bool:
    return len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5


def is_zip(head: bytes) -> bool:
    return head[:4] in (b"PK\x03\x04", b"PK\x05\x06")


def scan_pickle_bytes(data: bytes, file: str) -> tuple[list[Finding], pickle_vm.VMResult]:
    """Run every check over one pickle stream."""
    result = pickle_vm.run(data)
    findings: list[Finding] = []
    if result.error:
        findings.append(
            Finding(
                rule_id="PKL-090",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message="Pickle could not be parsed; file is not verified safe",
                offset=result.error_offset,
                evidence={"error": result.error},
            )
        )
        return findings, result
    findings += globals_policy.check(result, file)
    findings += lambdas.check(result, file)
    findings += strings.check(result, file)
    return findings, result


def _zip_pickles(path: Path, file: str, out: PickleScan) -> None:
    with zipfile.ZipFile(path) as zf:
        members = [i for i in zf.infolist() if i.filename.endswith((".pkl", ".pickle"))]
        if len(zf.infolist()) > MAX_ZIP_MEMBERS:
            raise ValueError("zip has too many members")
        if not members:
            out.notes.append(f"{file}: zip without pickle members")
            return
        for info in members:
            with zf.open(info) as fh:
                data = fh.read(MAX_PICKLE_BYTES + 1)
            if len(data) > MAX_PICKLE_BYTES:
                raise ValueError(f"{info.filename} exceeds {MAX_PICKLE_BYTES} bytes")
            findings, res = scan_pickle_bytes(data, f"{file}!{info.filename}")
            out.findings += findings
            out.scanned = True
            out.notes.append(f"{info.filename}: {res.pickles} pickle(s), {res.ops} opcode(s)")


def scan_file(path: Path) -> PickleScan:
    """Scan one file. ``scanned`` is False when it is not a pickle candidate."""
    out = PickleScan()
    file = str(path)
    suffix = path.suffix.lower()
    try:
        with path.open("rb") as fh:
            head = fh.read(4)
            candidate = suffix in PICKLE_SUFFIXES or (
                suffix in AMBIGUOUS_SUFFIXES and (looks_like_pickle(head) or is_zip(head))
            )
            if not candidate:
                return out
            if is_zip(head):
                _zip_pickles(path, file, out)
                return out
            fh.seek(0)
            data = fh.read(MAX_PICKLE_BYTES + 1)
        if len(data) > MAX_PICKLE_BYTES:
            data = data[:MAX_PICKLE_BYTES]
            out.notes.append(f"{file}: only first {MAX_PICKLE_BYTES} bytes scanned")
        findings, res = scan_pickle_bytes(data, file)
        out.findings += findings
        out.scanned = True
        out.notes.append(f"{file}: {res.pickles} pickle(s), {res.ops} opcode(s)")
        if res.trailing_bytes:
            out.notes.append(f"{file}: {res.trailing_bytes} trailing byte(s) not parsed")
    except Exception as exc:  # noqa: BLE001 - fail closed on any engine error
        out.scanned = True
        out.error = f"{file}: {type(exc).__name__}: {exc}"
    return out
