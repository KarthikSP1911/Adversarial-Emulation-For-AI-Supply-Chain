"""Structural validation of a .safetensors file, without loading any tensor.

Layout: 8-byte little-endian header length N, N bytes of JSON header, then the
raw tensor data. Safetensors cannot run code, so the risks are different:
bytes hidden in gaps or after the last tensor, overlapping tensors, headers
that disagree with themselves, and parser tricks (duplicate keys) that make
two loaders see two different models.
"""

from __future__ import annotations

import json
import math
import re
import struct

from modelscan.models import Finding, Severity

from .. import safe_io
from ..base import ATLAS_EMBED_MALWARE, ATLAS_SUPPLY_CHAIN_MODEL, Artifact, Kind, make_finding

CHECKER = "safetensors"
MAX_HEADER = 100 * 1024 * 1024
MAX_METADATA_VALUE = 64 * 1024
TRAILING_SNIFF = 16

# dtype -> size in bits (sub-byte types must pack to whole bytes)
DTYPE_BITS: dict[str, int] = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}  # fmt: skip

_SUSPICIOUS_META = re.compile(
    r"https?://|/bin/(?:ba)?sh|\bcurl\b|\bwget\b|__import__|\bexec\s*\(|\beval\s*\(|"
    r"powershell|/dev/tcp/|base64",
    re.I,
)


class _DuplicateKeyError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKeyError(key)
        out[key] = value
    return out


def applies(art: Artifact) -> bool:
    return art.kind == Kind.SAFETENSORS or art.path.suffix.lower() == ".safetensors"


def read_header(art: Artifact) -> tuple[int, dict[str, object]]:
    """Return (header_len, header). Raises ValueError on any structural problem."""
    with safe_io.open_regular(art.path) as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError("file shorter than the 8-byte header length")
        (n,) = struct.unpack("<Q", raw_len)
        if n > MAX_HEADER:
            raise ValueError(f"header length {n} exceeds {MAX_HEADER}")
        if 8 + n > art.size:
            raise ValueError(f"header length {n} runs past end of file ({art.size} bytes)")
        raw = safe_io.read_range(fh, 8, n, MAX_HEADER)
    text = raw.decode("utf-8")  # UnicodeDecodeError is a ValueError
    if not text.startswith("{"):
        raise ValueError("header does not start with '{'")
    stripped = text.rstrip(" ")  # the spec allows trailing space padding only
    header = json.loads(stripped, object_pairs_hook=_no_duplicates)
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return n, header


def check(art: Artifact) -> list[Finding]:
    findings: list[Finding] = []

    def add(rule: str, sev: Severity, msg: str, atlas: str | None = None, **ev: object) -> None:
        findings.append(make_finding(rule, sev, art, msg, checker=CHECKER, atlas=atlas, **ev))

    try:
        n, header = read_header(art)
    except _DuplicateKeyError as exc:
        add("SFT-002", Severity.HIGH,
            f"Duplicate key {str(exc)!r} in header: loaders may disagree on the tensor",
            ATLAS_SUPPLY_CHAIN_MODEL, key=str(exc))  # fmt: skip
        return findings
    except (ValueError, OSError, safe_io.ScanLimitError) as exc:
        add("SFT-001", Severity.HIGH, f"Malformed safetensors header: {exc}", error=str(exc))
        return findings

    data_len = art.size - 8 - n
    meta = header.pop("__metadata__", None)
    findings += _check_metadata(art, meta)

    spans: list[tuple[int, int, str]] = []
    for name, info in header.items():
        problem = _tensor_problem(info)
        if problem:
            add("SFT-003", Severity.HIGH, f"Tensor {name!r}: {problem}", tensor=name)
            continue
        begin, end = info["data_offsets"]
        if end > data_len:
            add("SFT-003", Severity.HIGH,
                f"Tensor {name!r} ends at {end}, past the data section ({data_len} bytes)",
                tensor=name)  # fmt: skip
            continue
        spans.append((begin, end, name))

    findings += _check_layout(art, spans, data_len, n)
    return findings


def _tensor_problem(info: object) -> str | None:
    if not isinstance(info, dict):
        return "entry is not an object"
    extra = set(info) - {"dtype", "shape", "data_offsets"}
    if extra:
        return f"unexpected fields {sorted(extra)}"
    dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
    if dtype not in DTYPE_BITS:
        return f"unknown dtype {dtype!r}"
    if not isinstance(shape, list) or not all(
        isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
    ):
        return f"invalid shape {shape!r}"
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
        or not 0 <= offsets[0] <= offsets[1]
    ):
        return f"invalid data_offsets {offsets!r}"
    bits = math.prod(shape) * DTYPE_BITS[dtype]
    if bits % 8:
        return f"{dtype} x {shape} is not a whole number of bytes"
    if offsets[1] - offsets[0] != bits // 8:
        return (
            f"byte span {offsets[1] - offsets[0]} does not match "
            f"{dtype} x {shape} = {bits // 8} bytes"
        )
    return None


def _check_layout(
    art: Artifact, spans: list[tuple[int, int, str]], data_len: int, header_len: int
) -> list[Finding]:
    """Tensors must tile the data section exactly: no gaps, overlaps or leftovers."""
    findings: list[Finding] = []
    spans.sort()
    cursor = 0
    prev_name = None
    for begin, end, name in spans:
        if begin < cursor:
            findings.append(make_finding(
                "SFT-004", Severity.HIGH, art,
                f"Tensor {name!r} overlaps {prev_name!r}",
                checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL,
                tensor=name, other=prev_name, begin=begin,
            ))  # fmt: skip
        elif begin > cursor:
            findings.append(make_finding(
                "SFT-005", Severity.HIGH, art,
                f"{begin - cursor} unreferenced byte(s) before tensor {name!r}",
                checker=CHECKER, atlas=ATLAS_EMBED_MALWARE,
                offset=8 + header_len + cursor, gap_bytes=begin - cursor,
            ))  # fmt: skip
        if end >= cursor:
            cursor, prev_name = end, name

    if cursor < data_len:
        leftover = data_len - cursor
        offset = 8 + header_len + cursor
        with safe_io.open_regular(art.path) as fh:
            fh.seek(offset)
            sample = fh.read(TRAILING_SNIFF)
        looks = _describe(sample)
        findings.append(make_finding(
            "SFT-006", Severity.CRITICAL if looks else Severity.HIGH, art,
            f"{leftover} trailing byte(s) after the last tensor"
            + (f" that look like {looks}" if looks else ""),
            checker=CHECKER, atlas=ATLAS_EMBED_MALWARE,
            offset=offset, trailing_bytes=leftover, starts_with=sample.hex(),
        ))  # fmt: skip
    return findings


def _describe(sample: bytes) -> str | None:
    if sample[:1] == b"\x80" and sample[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        return "a pickle"
    if sample[:4] == b"PK\x03\x04":
        return "a zip archive"
    if sample[:4] == b"\x7fELF" or sample[:2] == b"MZ":
        return "an executable"
    if sample[:2] == b"#!":
        return "a script"
    return None


def _check_metadata(art: Artifact, meta: object) -> list[Finding]:
    if meta is None:
        return []
    out: list[Finding] = []
    if not isinstance(meta, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
    ):
        return [make_finding(
            "SFT-007", Severity.MEDIUM, art,
            "__metadata__ must map strings to strings",
            checker=CHECKER,
        )]  # fmt: skip
    for key, value in meta.items():
        if len(value) > MAX_METADATA_VALUE:
            out.append(make_finding(
                "SFT-007", Severity.MEDIUM, art,
                f"__metadata__[{key!r}] is {len(value)} chars (possible smuggled payload)",
                checker=CHECKER, key=key, length=len(value),
            ))  # fmt: skip
        elif _SUSPICIOUS_META.search(value):
            out.append(make_finding(
                "SFT-008", Severity.MEDIUM, art,
                f"Suspicious text in __metadata__[{key!r}]",
                checker=CHECKER, key=key, value=value[:200],
            ))  # fmt: skip
    return out
