"""GGUF (GPT-Generated Unified Format) checkpoint scanner (FMT-G-*).

GGUF is the binary format used by llama.cpp and its derivatives.  The file
structure is:

    magic (4 bytes: "GGUF")  |  version (uint32)  |  tensor_count (uint64)
    metadata_kv_count (uint64)
    [metadata key-value pairs]
    [tensor info blocks]
    [tensor data]

Metadata KV pairs carry model hyper-parameters, tokenizer vocabulary, chat
templates, and other configuration.  An attacker can embed malicious content
in several places:

1. **Chat template strings** (key ``tokenizer.chat_template``) – Jinja2
   template code that is executed by the inference framework.
2. **Arbitrary string values** – URLs, shell commands, or base64 payloads
   hidden in any string metadata value.
3. **Oversized metadata** – a crafted file with an enormous ``metadata_count``
   can cause the reader to allocate gigabytes of memory (DoS / confusion).

This scanner reads the GGUF header and all metadata KV pairs **without**
executing any template or loading tensor data.

Rule IDs
--------
- FMT-G-001  File does not start with GGUF magic bytes (format confusion)
- FMT-G-002  Suspicious string in GGUF metadata (shell command, URL, etc.)
- FMT-G-003  Chat template contains potentially executable Jinja2 code blocks
- FMT-G-004  Metadata KV count exceeds safety threshold (possible DoS / bomb)
- FMT-G-005  Unsupported GGUF version (future-version confusion attack)
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from modelscan.models import Finding, Severity

INSPECTOR = "static.formats.gguf"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_SUFFIXES = frozenset({".gguf"})
GGUF_MAGIC = b"GGUF"

#: Maximum metadata KV pairs we will read.  A real model has hundreds at most.
MAX_KV_COUNT = 100_000

#: Metadata key that carries the Jinja2 chat template.
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

# Jinja2 control-flow tags that can execute arbitrary Python-like logic.
_JINJA_EXEC_RE = re.compile(r"\{%-?\s*(?:for|if|set|import|from|macro|call)\b")

# Suspicious patterns in string values.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("shell-path",    re.compile(r"(?:^|[\s;|&])/(?:usr/)?bin/(?:ba|z|da|k)?sh\b|\bcmd(?:\.exe)?\s*/c\b")),
    ("powershell",    re.compile(r"\bpowershell(?:\.exe)?\b|\bInvoke-Expression\b|\bIEX\b", re.I)),
    ("download-exec", re.compile(r"\b(?:curl|wget)\b[^\n]*(?:\||-o|-O|http)", re.I)),
    ("python-exec",   re.compile(r"__import__\s*\(|\bexec\s*\(|\beval\s*\(")),
    ("url",           re.compile(r"\bhttps?://[^\s'\"<>]+", re.I)),
    ("sensitive-path",re.compile(r"(?:\.ssh/|/etc/(?:passwd|shadow)|\.aws/credentials)")),
]


# ---------------------------------------------------------------------------
# GGUF value type identifiers
# ---------------------------------------------------------------------------

class _Vtype:
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12


_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _Vtype.UINT8:   ("<B",  1),
    _Vtype.INT8:    ("<b",  1),
    _Vtype.UINT16:  ("<H",  2),
    _Vtype.INT16:   ("<h",  2),
    _Vtype.UINT32:  ("<I",  4),
    _Vtype.INT32:   ("<i",  4),
    _Vtype.FLOAT32: ("<f",  4),
    _Vtype.BOOL:    ("<B",  1),
    _Vtype.UINT64:  ("<Q",  8),
    _Vtype.INT64:   ("<q",  8),
    _Vtype.FLOAT64: ("<d",  8),
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Reader:
    """Lightweight sequential reader over a bytes buffer."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos: self._pos + n]
        if len(chunk) < n:
            raise EOFError(f"expected {n} bytes, got {len(chunk)}")
        self._pos += n
        return chunk

    def read_u32(self) -> int:
        return struct.unpack_from("<I", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack_from("<Q", self.read(8))[0]

    def read_string(self) -> str:
        length = self.read_u64()
        if length > 10 * 1024 * 1024:  # 10 MB per string – sanity cap
            raise ValueError(f"string length {length} exceeds 10 MiB cap")
        raw = self.read(length)
        return raw.decode("utf-8", errors="replace")

    def read_value(self, vtype: int) -> object:
        """Read one GGUF value of the given type and return a Python object."""
        if vtype in _SCALAR_FMT:
            fmt, size = _SCALAR_FMT[vtype]
            return struct.unpack_from(fmt, self.read(size))[0]
        if vtype == _Vtype.STRING:
            return self.read_string()
        if vtype == _Vtype.ARRAY:
            elem_type = self.read_u32()
            count = self.read_u64()
            # Only collect string elements; skip others for performance.
            results = []
            for _ in range(count):
                v = self.read_value(elem_type)
                if isinstance(v, str):
                    results.append(v)
            return results
        raise ValueError(f"unknown GGUF value type {vtype}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one GGUF checkpoint file and return metadata findings.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.
    """
    if path.suffix.lower() not in GGUF_SUFFIXES:
        return []

    findings: list[Finding] = []
    file = str(path)

    # Read at most 128 MiB of metadata (tensor data follows later in the file).
    MAX_HEADER_READ = 128 * 1024 * 1024
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_HEADER_READ)
    except OSError as exc:
        return [Finding(
            rule_id="FMT-G-001",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"Could not read GGUF file: {exc}",
            evidence={"error": str(exc)},
        )]

    # FMT-G-001 – magic check
    if not data[:4] == GGUF_MAGIC:
        findings.append(Finding(
            rule_id="FMT-G-001",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=(
                "File has .gguf extension but does not start with the GGUF "
                "magic bytes ('GGUF'); possible format confusion or corruption"
            ),
            evidence={"first_bytes": data[:4].hex()},
        ))
        return findings

    rdr = _Reader(data)
    try:
        rdr.read(4)              # magic already checked
        version = rdr.read_u32()

        # FMT-G-005 – unsupported version
        if version not in (1, 2, 3):
            findings.append(Finding(
                rule_id="FMT-G-005",
                severity=Severity.LOW,
                inspector=INSPECTOR,
                file=file,
                message=(
                    f"GGUF version {version} is not recognised (supported: 1-3); "
                    "this scanner may miss threats in future-version files"
                ),
                evidence={"version": version},
            ))

        tensor_count   = rdr.read_u64()  # noqa: F841  (not used but must be consumed)
        metadata_count = rdr.read_u64()

        # FMT-G-004 – excessive metadata
        if metadata_count > MAX_KV_COUNT:
            findings.append(Finding(
                rule_id="FMT-G-004",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message=(
                    f"GGUF metadata KV count ({metadata_count}) exceeds safety "
                    f"threshold ({MAX_KV_COUNT}); possible DoS / parsing trap"
                ),
                evidence={"metadata_count": metadata_count},
            ))
            return findings

        reported_strings: set[str] = set()

        for _ in range(metadata_count):
            key   = rdr.read_string()
            vtype = rdr.read_u32()
            value = rdr.read_value(vtype)

            # Collect string values to check for suspicious content.
            str_values: list[str] = []
            if isinstance(value, str):
                str_values.append(value)
            elif isinstance(value, list):
                str_values.extend(v for v in value if isinstance(v, str))

            for s in str_values:
                # FMT-G-003 – chat template executable code blocks
                if key == CHAT_TEMPLATE_KEY and _JINJA_EXEC_RE.search(s):
                    findings.append(Finding(
                        rule_id="FMT-G-003",
                        severity=Severity.HIGH,
                        inspector=INSPECTOR,
                        file=file,
                        message=(
                            "Chat template contains Jinja2 control-flow blocks "
                            "that execute arbitrary logic during tokenisation"
                        ),
                        evidence={
                            "key": key,
                            "template_prefix": s[:300],
                        },
                    ))

                # FMT-G-002 – generic suspicious string patterns
                for kind, pattern in _PATTERNS:
                    if not pattern.search(s):
                        continue
                    dedup_key = f"{kind}:{s[:100]}"
                    if dedup_key in reported_strings:
                        break
                    reported_strings.add(dedup_key)
                    findings.append(Finding(
                        rule_id="FMT-G-002",
                        severity=Severity.HIGH,
                        inspector=INSPECTOR,
                        file=file,
                        message=(
                            f"Suspicious string ({kind}) in GGUF metadata "
                            f"key '{key}'"
                        ),
                        evidence={
                            "pattern": kind,
                            "key": key,
                            "string": s[:200],
                        },
                    ))
                    break  # one finding per string

    except (EOFError, ValueError, struct.error) as exc:
        findings.append(Finding(
            rule_id="FMT-G-001",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"GGUF file could not be fully parsed: {exc}",
            evidence={"error": str(exc)},
        ))

    return findings
