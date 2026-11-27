"""What is this file really? Decide from bytes, then compare with the name.

The extension is only a claim. A pickle renamed to ``model.safetensors`` or a
zip appended to a harmless file is exactly what an attacker would ship, so a
mismatch between claim and content is itself a finding.
"""

from __future__ import annotations

import pickletools
import re
import struct
import zipfile
from pathlib import Path

from modelscan.models import Finding, Severity

from . import safe_io
from .base import ATLAS_UNSAFE_ARTIFACT, CODE_CAPABLE, Artifact, Kind, make_finding

CHECKER = "detect"
SNIFF_BYTES = 4096
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024  # same limit as the reference parser

# What each extension claims the file is. A set means "any of these is fine".
CLAIMS: dict[str, frozenset[Kind]] = {
    ".safetensors": frozenset({Kind.SAFETENSORS}),
    ".gguf": frozenset({Kind.GGUF}),
    ".json": frozenset({Kind.JSON}),
    ".onnx": frozenset({Kind.ONNX}),
    ".h5": frozenset({Kind.HDF5}),
    ".hdf5": frozenset({Kind.HDF5}),
    ".keras": frozenset({Kind.ZIP}),
    ".npy": frozenset({Kind.NUMPY}),
    ".npz": frozenset({Kind.ZIP}),
    ".zip": frozenset({Kind.ZIP, Kind.TORCH_ZIP}),
    ".tar": frozenset({Kind.TAR}),
    ".gz": frozenset({Kind.GZIP}),
    ".txt": frozenset({Kind.TEXT, Kind.JSON, Kind.EMPTY}),
    ".md": frozenset({Kind.TEXT, Kind.EMPTY}),
    ".pkl": frozenset({Kind.PICKLE}),
    ".pickle": frozenset({Kind.PICKLE}),
    ".pt": frozenset({Kind.TORCH_ZIP, Kind.PICKLE, Kind.TAR}),
    ".pth": frozenset({Kind.TORCH_ZIP, Kind.PICKLE, Kind.TAR}),
    ".ckpt": frozenset({Kind.TORCH_ZIP, Kind.PICKLE, Kind.TAR}),
}

# Extensions people treat as "safe to load". Code-capable content behind one
# of these is a disguise, not an accident.
SAFE_CLAIMS = frozenset({".safetensors", ".gguf", ".json", ".onnx", ".txt", ".md", ".npy"})

_PICKLE_START_OPS = frozenset({
    "PROTO", "GLOBAL", "MARK", "EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE",
    "INST", "STRING", "UNICODE", "INT", "LONG", "NONE", "FRAME",
})  # fmt: skip
_GLOBAL_ARG = re.compile(r"^[A-Za-z_][\w.]* [A-Za-z_][\w.]*$")


def _looks_like_safetensors(head: bytes, size: int) -> bool:
    if len(head) < 9:
        return False
    (n,) = struct.unpack("<Q", head[:8])
    return 2 <= n <= min(MAX_SAFETENSORS_HEADER, size - 8) and head[8:9] == b"{"


def _looks_like_pickle(head: bytes) -> bool:
    """Protocol 2+ has a header byte; protocol 0/1 must be recognised by parsing."""
    if len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        return True
    ops: list[str] = []
    named_global = False
    try:
        for op, arg, _pos in pickletools.genops(head):
            ops.append(op.name)
            if op.name in ("GLOBAL", "INST") and _GLOBAL_ARG.match(str(arg)):
                named_global = True
            if op.name == "STOP" or len(ops) >= 64:
                break
    except Exception:  # noqa: BLE001 - truncated head or not a pickle
        pass
    if not ops or ops[0] not in _PICKLE_START_OPS:
        return False
    # Ordinary text often parses as a few opcodes ("Note" = NONE, OBJ, TUPLE...).
    # Demand a complete pickle, or a well-formed "module name" global.
    return ops[-1] == "STOP" or named_global


def _looks_like_tar(head: bytes) -> bool:
    return len(head) >= 262 and head[257:262] == b"ustar"


def _looks_like_text(head: bytes) -> bool:
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # The sniff window may split a multi-byte character; allow that.
        try:
            head[:-3].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return b"\x00" not in head


def _zip_kind(path: Path) -> Kind:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return Kind.ZIP  # archive.py reports the corruption
    if any(n.endswith("/data.pkl") or n == "data.pkl" for n in names):
        return Kind.TORCH_ZIP
    return Kind.ZIP


def sniff(path: Path, size: int) -> Kind:
    if size == 0:
        return Kind.EMPTY
    head = safe_io.read_head(path, SNIFF_BYTES)
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return _zip_kind(path)
    if head[:4] == b"GGUF":
        return Kind.GGUF
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return Kind.HDF5
    if head[:6] == b"\x93NUMPY":
        return Kind.NUMPY
    if head[:2] == b"\x1f\x8b":
        return Kind.GZIP
    if _looks_like_tar(head):
        return Kind.TAR
    if _looks_like_safetensors(head, size):
        return Kind.SAFETENSORS
    if _looks_like_pickle(head):
        return Kind.PICKLE
    stripped = head.lstrip()
    if stripped[:1] in (b"{", b"[") and _looks_like_text(head):
        return Kind.JSON
    if _looks_like_text(head):
        return Kind.TEXT
    if path.suffix.lower() == ".onnx" and head[:1] == b"\x08":
        return Kind.ONNX  # protobuf field 1 (ir_version); onnx checker verifies
    return Kind.UNKNOWN


def _has_appended_zip(path: Path, size: int, kind: Kind) -> bool:
    """A zip end-of-central-directory record inside a file that is not a zip."""
    if kind in (Kind.ZIP, Kind.TORCH_ZIP) or size < 22:
        return False
    tail = safe_io.read_tail(path, min(size, 65_557))  # max EOCD + comment
    return b"PK\x05\x06" in tail


def check(art: Artifact) -> list[Finding]:
    """Set ``art.kind`` and report disguised or polyglot files."""
    findings: list[Finding] = []
    art.kind = sniff(art.path, art.size)
    suffix = art.path.suffix.lower()
    claimed = CLAIMS.get(suffix)

    if claimed is not None and art.kind not in claimed and art.kind != Kind.EMPTY:
        disguised = suffix in SAFE_CLAIMS and art.kind in CODE_CAPABLE
        findings.append(
            make_finding(
                "DET-001",
                Severity.CRITICAL if disguised else Severity.HIGH,
                art,
                f"Extension {suffix} claims {'/'.join(sorted(k.value for k in claimed))} "
                f"but content is {art.kind.value}"
                + (" (code-capable format disguised as a safe one)" if disguised else ""),
                checker=CHECKER,
                atlas=ATLAS_UNSAFE_ARTIFACT,
                extension=suffix,
                detected=art.kind.value,
            )
        )

    if _has_appended_zip(art.path, art.size, art.kind):
        findings.append(
            make_finding(
                "DET-002",
                Severity.HIGH,
                art,
                f"Zip directory found inside a {art.kind.value} file (polyglot)",
                checker=CHECKER,
                atlas=ATLAS_UNSAFE_ARTIFACT,
                detected=art.kind.value,
            )
        )

    if art.kind == Kind.UNKNOWN:
        findings.append(
            make_finding(
                "DET-003",
                Severity.LOW,
                art,
                "Unrecognised binary format; not inspected",
                checker=CHECKER,
                extension=suffix,
            )
        )
    return findings
