"""ONNX model scanner (FMT-O-*).

ONNX models are serialised as Protocol Buffers (protobuf).  Rather than
importing the ``onnx`` package (which may not be installed in a minimal
environment), this scanner uses a lightweight hand-rolled protobuf parser that
reads only the field tags and wire types it needs.

Threat model
------------
1. **Custom operators** – ``NodeProto.op_type`` can reference an operator
   implemented by a shared-library plugin loaded at runtime.  An attacker can
   ship a malicious ``*.so`` / ``*.dll`` alongside the ONNX file and have it
   executed via ``SessionOptions.RegisterCustomOpsLibrary``.
2. **External data** – ``TensorProto.data_location == EXTERNAL`` causes the
   runtime to read a file from an arbitrary path, which can be exploited for
   directory-traversal or SSRF.
3. **String initializer blobs** – large base64-encoded strings or shell
   commands embedded in string tensors.

Rule IDs
--------
- FMT-O-001  Custom (non-standard) ONNX operator domain detected
- FMT-O-002  External tensor data reference found (path-traversal / SSRF risk)
- FMT-O-003  Suspicious string in ONNX model (shell command, URL, etc.)
- FMT-O-004  File does not start with ONNX protobuf magic (format confusion)
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from modelscan.models import Finding, Severity

INSPECTOR = "static.formats.onnx"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONNX_SUFFIXES = frozenset({".onnx"})

# Standard ONNX operator domains that every compliant runtime ships.
_STANDARD_DOMAINS = frozenset({
    "",           # default ONNX domain
    "ai.onnx",
    "ai.onnx.ml",
    "ai.onnx.preview.training",
    "com.microsoft",          # widely deployed ONNX Runtime extensions
    "com.microsoft.experimental",
})

# Suspicious patterns inside ONNX string fields (same approach as strings.py).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("shell-path",    re.compile(r"(?:^|[\s;|&])/(?:usr/)?bin/(?:ba|z|da|k)?sh\b|\bcmd(?:\.exe)?\s*/c\b")),
    ("powershell",    re.compile(r"\bpowershell(?:\.exe)?\b|\bInvoke-Expression\b|\bIEX\b", re.I)),
    ("download-exec", re.compile(r"\b(?:curl|wget)\b[^\n]*(?:\||-o|-O|http)", re.I)),
    ("python-exec",   re.compile(r"__import__\s*\(|\bexec\s*\(|\beval\s*\(")),
    ("url",           re.compile(r"\bhttps?://[^\s'\"<>]+", re.I)),
    ("sensitive-path",re.compile(r"(?:\.ssh/|/etc/(?:passwd|shadow)|\.aws/credentials)")),
]

# ---------------------------------------------------------------------------
# Lightweight protobuf reader
# ---------------------------------------------------------------------------


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a protobuf varint starting at *pos*.  Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")
    return result, pos


def _iter_fields(data: bytes, start: int = 0, end: int | None = None):
    """Yield (field_number, wire_type, payload_bytes) for each protobuf field."""
    if end is None:
        end = len(data)
    pos = start
    while pos < end:
        try:
            tag, pos = _read_varint(data, pos)
        except (ValueError, IndexError):
            break
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:          # varint
            val, pos = _read_varint(data, pos)
            yield field_number, wire_type, val.to_bytes(8, "little")
        elif wire_type == 1:        # 64-bit
            yield field_number, wire_type, data[pos:pos + 8]
            pos += 8
        elif wire_type == 2:        # length-delimited
            length, pos = _read_varint(data, pos)
            yield field_number, wire_type, data[pos:pos + length]
            pos += length
        elif wire_type == 5:        # 32-bit
            yield field_number, wire_type, data[pos:pos + 4]
            pos += 4
        else:
            # Unknown wire type – we cannot parse further.
            break


# ---------------------------------------------------------------------------
# ONNX protobuf field numbers (ModelProto / NodeProto / TensorProto)
# ---------------------------------------------------------------------------
# ModelProto:
#   field 1  = ir_version (int64)
#   field 2  = opset_import (repeated OperatorSetIdProto)  field 1=domain, 2=version
#   field 7  = graph (GraphProto)
# GraphProto:
#   field 1  = node (repeated NodeProto)
#   field 4  = initializer (repeated TensorProto)
# NodeProto:
#   field 1  = input (repeated string)
#   field 2  = output (repeated string)
#   field 4  = op_type (string)
#   field 7  = domain (string)
# TensorProto:
#   field 1  = dims (repeated int64)
#   field 2  = data_type (int32)   STRING = 8
#   field 4  = float_data
#   field 7  = string_data (repeated bytes)
#   field 8  = name (string)
#   field 9  = data_location (int32)  EXTERNAL = 1
#  field 13  = external_data (repeated StringStringEntryProto: 1=key,2=value)


def _extract_strings_from_tensor(tensor_bytes: bytes) -> list[str]:
    """Extract all string_data entries from a TensorProto."""
    out: list[str] = []
    for fn, wt, payload in _iter_fields(tensor_bytes):
        if fn == 7 and wt == 2:  # string_data
            try:
                out.append(payload.decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                pass
    return out


def _extract_external_path(tensor_bytes: bytes) -> str | None:
    """Return the 'location' value from external_data if present."""
    data_location = None
    ext_path: str | None = None
    for fn, wt, payload in _iter_fields(tensor_bytes):
        if fn == 9 and wt == 0:   # data_location
            data_location = int.from_bytes(payload[:8], "little")
        if fn == 13 and wt == 2:  # external_data StringStringEntryProto
            # decode key/value
            key = val = None
            for sfn, swt, sp in _iter_fields(payload):
                if sfn == 1 and swt == 2:
                    key = sp.decode("utf-8", errors="replace")
                if sfn == 2 and swt == 2:
                    val = sp.decode("utf-8", errors="replace")
            if key == "location" and val:
                ext_path = val
    if data_location == 1:
        return ext_path or "<unknown>"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one ONNX model file and return structural / content findings.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.
    """
    if path.suffix.lower() not in ONNX_SUFFIXES:
        return []

    findings: list[Finding] = []
    file = str(path)

    # Read the whole file (ONNX models are typically < 2 GB).
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [Finding(
            rule_id="FMT-O-004",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"Could not read ONNX file: {exc}",
            evidence={"error": str(exc)},
        )]

    # FMT-O-004 – protobuf magic check.
    # A ModelProto always starts with field 1 (ir_version), which is a varint.
    # Field tag for field-1, wire-type-0 = 0x08.  Most ONNX files start with
    # 0x08.  We accept 0x08 or 0x0A (field-1, wire-type-2 for string fields).
    if len(data) < 2 or data[0] not in (0x08, 0x0A):
        findings.append(Finding(
            rule_id="FMT-O-004",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=(
                "File has .onnx extension but does not start with an ONNX "
                "protobuf header byte (0x08 or 0x0a); possible format confusion"
            ),
            evidence={"first_bytes": data[:4].hex()},
        ))
        return findings

    # --- Walk ModelProto top-level fields ---
    for fn, wt, payload in _iter_fields(data):
        # field 2 – opset_import (OperatorSetIdProto)
        if fn == 2 and wt == 2:
            domain = ""
            for sfn, swt, sp in _iter_fields(payload):
                if sfn == 1 and swt == 2:
                    domain = sp.decode("utf-8", errors="replace")
            if domain not in _STANDARD_DOMAINS:
                findings.append(Finding(
                    rule_id="FMT-O-001",
                    severity=Severity.HIGH,
                    inspector=INSPECTOR,
                    file=file,
                    message=(
                        f"Non-standard ONNX operator domain '{domain}' detected; "
                        "custom operator plugin may be loaded at inference time"
                    ),
                    evidence={"domain": domain},
                ))

        # field 7 – graph (GraphProto)
        if fn == 7 and wt == 2:
            # Walk GraphProto fields
            for gfn, gwt, gpayload in _iter_fields(payload):
                # field 4 – initializer (TensorProto)
                if gfn == 4 and gwt == 2:
                    ext_path = _extract_external_path(gpayload)
                    if ext_path is not None:
                        findings.append(Finding(
                            rule_id="FMT-O-002",
                            severity=Severity.HIGH,
                            inspector=INSPECTOR,
                            file=file,
                            message=(
                                "ONNX model references external tensor data at "
                                f"'{ext_path}'; path-traversal or SSRF risk"
                            ),
                            evidence={"external_path": ext_path},
                        ))

                    for s in _extract_strings_from_tensor(gpayload):
                        for kind, pattern in _PATTERNS:
                            if pattern.search(s):
                                findings.append(Finding(
                                    rule_id="FMT-O-003",
                                    severity=Severity.HIGH,
                                    inspector=INSPECTOR,
                                    file=file,
                                    message=(
                                        f"Suspicious string ({kind}) found in "
                                        "ONNX string tensor"
                                    ),
                                    evidence={
                                        "pattern": kind,
                                        "string": s[:200],
                                    },
                                ))
                                break

    return findings
