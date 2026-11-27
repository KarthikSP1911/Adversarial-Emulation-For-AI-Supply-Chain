"""NumPy array file scanner (FMT-N-*).

Covers both the single-array ``.npy`` format and the multi-array ``.npz``
(ZIP of ``.npy`` files) format.

Threat model
------------
1. **Pickle inside npy** – NumPy's ``.npy`` format version 1.0 stores the
   array header as a Python ``dict`` literal that is evaluated with
   ``ast.literal_eval``.  *However*, if the ``allow_pickle=True`` flag is
   used, object-dtype arrays are stored as a raw pickle stream embedded inside
   the ``.npy`` file.  We detect these embedded pickles and flag them.
2. **Object-dtype header declaration** – even without an embedded pickle, a
   header that declares ``dtype=object`` signals that the file *requires*
   pickle to load, which is a risk indicator.
3. **Zip-bomb / path-traversal** – applied to ``.npz`` archives.
4. **Excessive member count** – ``.npz`` files should contain one ``.npy``
   per array, not thousands.

Rule IDs
--------
- FMT-N-001  Embedded pickle stream detected inside a .npy file
- FMT-N-002  Array header declares object dtype (requires allow_pickle=True)
- FMT-N-003  .npz member path contains directory-traversal sequence
- FMT-N-004  .npz member count exceeds safety threshold
- FMT-N-005  .npy magic bytes missing or version unsupported
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path, PurePosixPath

from modelscan.models import Finding, Severity

INSPECTOR = "static.formats.numpy"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMPY_SUFFIXES = frozenset({".npy", ".npz"})

#: Magic prefix for every .npy file: b'\x93NUMPY'
NPY_MAGIC = b"\x93NUMPY"

#: Maximum number of .npz members we enumerate.
MAX_NPZ_MEMBERS = 10_000

#: Pickle protocol opcode – every pickle starts with 0x80 followed by 2–5.
_PICKLE_MAGIC_BYTE = 0x80

#: Maximum header size we read from a .npy file (should be < 128 KiB).
MAX_HEADER_BYTES = 65536


# ---------------------------------------------------------------------------
# .npy helpers
# ---------------------------------------------------------------------------


def _read_npy_header(data: bytes) -> tuple[int, int, bytes]:
    """Parse the .npy file header.

    The NumPy binary format layout (all versions):
    - Bytes [0:5]  : magic ``\\x93NUMPY``
    - Byte  [5]    : major version (1, 2, or 3)
    - Byte  [6]    : minor version
    - Bytes [7:9]  : ``HEADER_LEN`` as uint16-LE  (version 1 only)
    - Bytes [7:11] : ``HEADER_LEN`` as uint32-LE  (version 2 / 3)
    - Bytes [9/11+]: Python dict header string

    Returns
    -------
    (major, minor, header_bytes)
        ``major`` and ``minor`` are the format version.
        ``header_bytes`` is the raw header dictionary bytes.

    Raises
    ------
    ValueError
        If the magic is wrong or the header is truncated.
    """
    if len(data) < 7:
        raise ValueError("file too short to be a .npy file")
    if data[:6] != NPY_MAGIC:
        raise ValueError("not a .npy file (magic mismatch)")

    major = data[6]
    minor = data[7]

    if major not in (1, 2, 3):
        raise ValueError(f"unsupported .npy format version {major}.{minor}")

    if major == 1:
        # Version 1: 2-byte header length at offset 8
        if len(data) < 10:
            raise ValueError(".npy version 1 header too short")
        header_len = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
    else:
        # Version 2 / 3: 4-byte header length at offset 8
        if len(data) < 12:
            raise ValueError(".npy version 2/3 header too short")
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_start = 12

    header_end = header_start + header_len
    header_bytes = data[header_start:header_end]
    return major, minor, header_bytes


def _scan_npy_bytes(data: bytes, file: str) -> list[Finding]:
    """Scan one .npy file content (already in memory)."""
    findings: list[Finding] = []

    try:
        major, minor, header_bytes = _read_npy_header(data)
    except ValueError as exc:
        findings.append(Finding(
            rule_id="FMT-N-005",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f".npy header could not be parsed: {exc}",
            evidence={"error": str(exc), "magic": data[:8].hex()},
        ))
        return findings

    # FMT-N-002 – object dtype declared in header
    header_str = header_bytes.decode("latin-1", errors="replace")
    # NumPy represents object dtype as 'O', '|O', or '>O'/'<O' in the header dict.
    if (
        "'O'" in header_str
        or "'|O'" in header_str
        or "\"O\"" in header_str
        or "\"O'" in header_str
        or "dtype('O')" in header_str
        or "dtype=\"O\"" in header_str
    ):
        findings.append(Finding(
            rule_id="FMT-N-002",
            severity=Severity.HIGH,
            inspector=INSPECTOR,
            file=file,
            message=(
                ".npy file declares object dtype ('O'); loading this file "
                "requires allow_pickle=True and may execute arbitrary code"
            ),
            evidence={"header": header_str[:200]},
        ))

    # FMT-N-001 – embedded pickle stream
    # After the header comes the raw data section.  For object arrays NumPy
    # writes individual pickled objects back-to-back.  We scan the data
    # section for the pickle protocol byte (0x80 0x02–0x05).
    header_size = (10 if major == 1 else 12) + len(header_bytes)
    data_section = data[header_size:]
    pos = 0
    while pos < len(data_section) - 1:
        if data_section[pos] == _PICKLE_MAGIC_BYTE and 2 <= data_section[pos + 1] <= 5:
            findings.append(Finding(
                rule_id="FMT-N-001",
                severity=Severity.CRITICAL,
                inspector=INSPECTOR,
                file=file,
                message=(
                    "Embedded pickle stream detected in .npy data section; "
                    "this file will execute pickle code when loaded"
                ),
                offset=header_size + pos,
                evidence={
                    "pickle_protocol": data_section[pos + 1],
                    "data_offset": header_size + pos,
                },
            ))
            break  # one finding per file
        pos += 1

    return findings


# ---------------------------------------------------------------------------
# .npz helpers
# ---------------------------------------------------------------------------


def _has_traversal(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return ".." in parts or name.startswith("/")


def _scan_npz(path: Path, file: str) -> list[Finding]:
    """Scan a .npz archive."""
    findings: list[Finding] = []

    if not zipfile.is_zipfile(path):
        findings.append(Finding(
            rule_id="FMT-N-005",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=".npz file is not a valid ZIP archive",
            evidence={},
        ))
        return findings

    with zipfile.ZipFile(path, "r") as zf:
        members = zf.infolist()

        # FMT-N-004 – too many members
        if len(members) > MAX_NPZ_MEMBERS:
            findings.append(Finding(
                rule_id="FMT-N-004",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message=(
                    f".npz archive has {len(members)} members, which exceeds "
                    f"the safety threshold of {MAX_NPZ_MEMBERS}"
                ),
                evidence={"member_count": len(members)},
            ))
            return findings

        for info in members:
            # FMT-N-003 – path traversal
            if _has_traversal(info.filename):
                findings.append(Finding(
                    rule_id="FMT-N-003",
                    severity=Severity.CRITICAL,
                    inspector=INSPECTOR,
                    file=file,
                    message=(
                        f".npz member '{info.filename}' contains a "
                        "directory-traversal sequence"
                    ),
                    evidence={"member": info.filename},
                ))

            # Scan each .npy member for embedded pickles
            if info.filename.endswith(".npy"):
                with zf.open(info) as fh:
                    npy_data = fh.read(MAX_HEADER_BYTES + 2048)
                member_findings = _scan_npy_bytes(
                    npy_data, f"{file}!{info.filename}"
                )
                findings.extend(member_findings)

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one NumPy array file and return findings.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.
    """
    if path.suffix.lower() not in NUMPY_SUFFIXES:
        return []

    file = str(path)

    if path.suffix.lower() == ".npz":
        return _scan_npz(path, file)

    # .npy – single-array file
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [Finding(
            rule_id="FMT-N-005",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"Could not read .npy file: {exc}",
            evidence={"error": str(exc)},
        )]

    return _scan_npy_bytes(data, file)
