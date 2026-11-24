"""Keras / TensorFlow SavedModel and Keras v3 scanner (FMT-K-*).

Keras models can be saved in several formats.  This scanner handles:

* ``.keras`` – a ZIP archive containing ``config.json``, ``model.weights.h5``
  and optionally a ``saved_model.pb`` or ``assets/`` directory.
* ``SavedModel`` directories – a folder with ``saved_model.pb`` and
  ``variables/``.
* Legacy ``.h5`` / ``.hdf5`` – single HDF5 files (structural check only;
  we do not parse HDF5 binary format, just flag known dangerous members).

The most dangerous vector in Keras models is **lambda layers**: the model
config (JSON) can embed arbitrary Python source code that is re-executed via
``eval()`` / ``exec()`` during ``model.load()``.  We detect this in the JSON
config without executing it.

Rule IDs
--------
- FMT-K-001  Lambda layer detected in Keras config JSON (code-execution risk)
- FMT-K-002  Custom-object string in config references a dangerous callable
- FMT-K-003  Unexpected executable file found inside ``.keras`` ZIP
- FMT-K-004  Keras config contains an ``assets`` or ``saved_model.pb`` reference
              that could load a TF-serving computation graph (medium risk)
- FMT-K-005  ``.keras`` ZIP contains no config.json (malformed / suspicious)
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from modelscan.models import Finding, Severity

INSPECTOR = "static.formats.keras"

# ---------------------------------------------------------------------------
# Supported suffixes
# ---------------------------------------------------------------------------

KERAS_SUFFIXES = frozenset({".keras", ".h5", ".hdf5"})

# ---------------------------------------------------------------------------
# Dangerous callable patterns that may appear as custom-object class names
# ---------------------------------------------------------------------------

_DANGEROUS_CALLABLE_RE = re.compile(
    r"\b(?:eval|exec|__import__|subprocess|os\.system|open|compile"
    r"|lambda\s*:|builtins)\b",
    re.IGNORECASE,
)

# Key names inside Keras config JSON that carry executable source code.
_LAMBDA_KEYS = frozenset({"function", "lambda", "fn"})

# Extensions that must not appear inside a .keras ZIP.
_EXEC_EXTS = frozenset({".py", ".pyc", ".so", ".dll", ".sh", ".bat", ".exe"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_keras_file(path: Path) -> bool:
    return path.suffix.lower() in KERAS_SUFFIXES


def _is_savedmodel_dir(path: Path) -> bool:
    return path.is_dir() and (path / "saved_model.pb").exists()


def _walk_config(obj: object, depth: int = 0) -> list[str]:
    """Recursively collect all string values from a decoded JSON object."""
    if depth > 30:
        return []
    strings: list[str] = []
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            strings.extend(_walk_config(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_walk_config(item, depth + 1))
    return strings


def _has_lambda_key(obj: object, depth: int = 0) -> bool:
    """Return True if any dict key is a known lambda carrier."""
    if depth > 30:
        return False
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in _LAMBDA_KEYS:
                return True
            if _has_lambda_key(v, depth + 1):
                return True
    elif isinstance(obj, list):
        return any(_has_lambda_key(i, depth + 1) for i in obj)
    return False


# ---------------------------------------------------------------------------
# Sub-scanners
# ---------------------------------------------------------------------------


def _scan_keras_zip(path: Path, file: str) -> list[Finding]:
    """Structural and config scan for .keras ZIP archives."""
    findings: list[Finding] = []

    if not zipfile.is_zipfile(path):
        return findings  # let the outer caller handle non-ZIP .keras files

    with zipfile.ZipFile(path, "r") as zf:
        names = {i.filename for i in zf.infolist()}

        # FMT-K-005 – no config.json
        if "config.json" not in names:
            findings.append(Finding(
                rule_id="FMT-K-005",
                severity=Severity.MEDIUM,
                inspector=INSPECTOR,
                file=file,
                message=(
                    ".keras archive contains no config.json; "
                    "file may be malformed or deliberately obfuscated"
                ),
                evidence={"members": sorted(names)[:20]},
            ))

        # FMT-K-003 – unexpected executable members
        for name in names:
            from pathlib import PurePosixPath
            ext = PurePosixPath(name).suffix.lower()
            if ext in _EXEC_EXTS:
                findings.append(Finding(
                    rule_id="FMT-K-003",
                    severity=Severity.HIGH,
                    inspector=INSPECTOR,
                    file=file,
                    message=(
                        f"Unexpected executable member '{name}' inside "
                        ".keras ZIP archive"
                    ),
                    evidence={"member": name, "extension": ext},
                ))

        # FMT-K-004 – saved_model.pb presence (medium risk, not automatically bad)
        if "saved_model.pb" in names:
            findings.append(Finding(
                rule_id="FMT-K-004",
                severity=Severity.LOW,
                inspector=INSPECTOR,
                file=file,
                message=(
                    ".keras archive contains saved_model.pb; "
                    "TF computation graph will be loaded at model.load() time"
                ),
                evidence={"member": "saved_model.pb"},
            ))

        # FMT-K-001/002 – parse config.json for lambda layers
        if "config.json" in names:
            with zf.open("config.json") as fh:
                raw = fh.read(2 * 1024 * 1024)  # read at most 2 MB
            findings.extend(_scan_config_bytes(raw, file))

    return findings


def _scan_config_bytes(raw: bytes, file: str) -> list[Finding]:
    """Inspect a Keras config.json payload for lambda/exec patterns."""
    findings: list[Finding] = []
    try:
        cfg = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Unparseable config – flag as suspicious
        findings.append(Finding(
            rule_id="FMT-K-005",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message="config.json could not be parsed as JSON",
            evidence={"raw_prefix": raw[:200].decode("utf-8", errors="replace")},
        ))
        return findings

    # FMT-K-001 – lambda-carrier key present
    if _has_lambda_key(cfg):
        findings.append(Finding(
            rule_id="FMT-K-001",
            severity=Severity.CRITICAL,
            inspector=INSPECTOR,
            file=file,
            message=(
                "Keras config.json contains a lambda/function key; "
                "arbitrary Python code will be executed when the model is loaded"
            ),
            evidence={"detail": "lambda key detected in config.json"},
        ))

    # FMT-K-002 – dangerous callable string values
    for s in _walk_config(cfg):
        m = _DANGEROUS_CALLABLE_RE.search(s)
        if m:
            findings.append(Finding(
                rule_id="FMT-K-002",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=file,
                message=(
                    f"Keras config.json contains a suspicious callable reference: "
                    f"'{m.group()}'"
                ),
                evidence={"match": m.group(), "context": s[:200]},
            ))
            break  # one finding per file is enough to flag the issue

    return findings


def _scan_savedmodel_dir(path: Path, file: str) -> list[Finding]:
    """Inspect a TF SavedModel directory for suspicious Python assets."""
    findings: list[Finding] = []
    for asset in path.rglob("*"):
        if not asset.is_file():
            continue
        if asset.suffix.lower() in {".py", ".pyc", ".sh"}:
            findings.append(Finding(
                rule_id="FMT-K-003",
                severity=Severity.HIGH,
                inspector=INSPECTOR,
                file=str(asset),
                message=(
                    f"Executable file '{asset.name}' found inside "
                    "SavedModel directory"
                ),
                evidence={"asset": str(asset.relative_to(path))},
            ))
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one Keras / TF model artifact and return structural findings.

    Parameters
    ----------
    path:
        Absolute path to a file **or** directory being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.  An empty list means no problems detected.
    """
    file = str(path)

    if path.is_dir():
        if _is_savedmodel_dir(path):
            return _scan_savedmodel_dir(path, file)
        return []

    if not _is_keras_file(path):
        return []

    if path.suffix.lower() == ".keras":
        return _scan_keras_zip(path, file)

    # Legacy .h5 / .hdf5 – basic magic-byte check only
    # (Full HDF5 parsing is out of scope; we just ensure it is not a pickle
    # or ZIP disguised with an .h5 extension.)
    findings: list[Finding] = []
    with path.open("rb") as fh:
        magic = fh.read(8)

    # HDF5 files begin with \x89HDF\r\n\x1a\n
    if not magic.startswith(b"\x89HDF"):
        findings.append(Finding(
            rule_id="FMT-K-005",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=(
                f"File has {path.suffix} extension but does not start with the "
                "HDF5 magic bytes; possible format confusion"
            ),
            evidence={"magic_hex": magic.hex()},
        ))

    return findings
