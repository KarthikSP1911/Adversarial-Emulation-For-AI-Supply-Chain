"""PyTorch checkpoint scanner (FMT-T-*).

Scans ``.pt``, ``.pth``, and ``.ckpt`` files **without** executing them.
PyTorch checkpoints are ZIP archives that contain one or more pickle streams
(``data.pkl``).  This module:

1. Verifies the outer ZIP structure is valid.
2. Enumerates every member file and flags suspicious names.
3. Checks for unexpected executable members (``*.py``, ``*.so``, etc.).
4. Counts how many pickle sub-streams exist and raises if the number is
   suspiciously large (possible zip-bomb / archive confusion attack).

The heavy pickle-level analysis (globals, strings, lambdas) is handled by
:mod:`modelscan.static.pickle_scan` which is called by the main engine for
every ``data.pkl`` member.  This module only does **structural** checks that
are specific to the PyTorch ZIP layout.

Rule IDs
--------
- FMT-T-001  Unexpected executable member inside checkpoint ZIP
- FMT-T-002  Member path contains a directory-traversal sequence
- FMT-T-003  ZIP member count exceeds safety threshold (possible zip-bomb)
- FMT-T-004  Checkpoint contains no recognised weight member (suspicious)
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from modelscan.models import Finding, Severity

INSPECTOR = "static.formats.torch"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Maximum number of ZIP members we are willing to enumerate.
MAX_MEMBERS = 10_000

#: Extensions that have no place inside a PyTorch checkpoint.
SUSPICIOUS_EXTENSIONS = frozenset({
    ".py", ".pyc", ".pyo", ".pyd",
    ".so", ".dll", ".dylib",
    ".sh", ".bat", ".cmd", ".ps1",
    ".exe", ".elf",
})

#: At least one of these names should appear in a valid PyTorch checkpoint.
WEIGHT_INDICATORS = frozenset({"data.pkl", "data/0", "archive/data.pkl"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TORCH_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})


def _is_torch_file(path: Path) -> bool:
    """Return True if *path* looks like a PyTorch checkpoint by extension."""
    return path.suffix.lower() in TORCH_SUFFIXES


def _has_traversal(name: str) -> bool:
    """Return True if the ZIP member name contains a path-traversal sequence."""
    # PurePosixPath collapses ``..`` segments; compare normalised vs raw.
    parts = PurePosixPath(name).parts
    return ".." in parts or name.startswith("/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(path: Path) -> list[Finding]:
    """Inspect one PyTorch checkpoint file and return structural findings.

    Parameters
    ----------
    path:
        Absolute path to the file being scanned.

    Returns
    -------
    list[Finding]
        Zero or more findings.  An empty list means no structural problems
        were detected (the pickle contents are checked separately).
    """
    if not _is_torch_file(path):
        return []

    findings: list[Finding] = []
    file = str(path)

    # --- guard: must be a valid ZIP -------------------------------------------
    if not zipfile.is_zipfile(path):
        # Not a ZIP at all – the pickle engine will handle raw pickle files.
        # We only do structural checks for the ZIP-based format.
        return []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = zf.infolist()

            # FMT-T-003 – too many members
            if len(members) > MAX_MEMBERS:
                findings.append(Finding(
                    rule_id="FMT-T-003",
                    severity=Severity.HIGH,
                    inspector=INSPECTOR,
                    file=file,
                    message=(
                        f"ZIP member count ({len(members)}) exceeds safety "
                        f"threshold ({MAX_MEMBERS}); possible zip-bomb"
                    ),
                    evidence={"member_count": len(members)},
                ))
                # Do not iterate further – we cannot safely enumerate this.
                return findings

            has_weight = False
            for info in members:
                name = info.filename

                # FMT-T-004 – track presence of weight members
                if name in WEIGHT_INDICATORS or name.endswith("/data.pkl"):
                    has_weight = True

                # FMT-T-002 – directory traversal
                if _has_traversal(name):
                    findings.append(Finding(
                        rule_id="FMT-T-002",
                        severity=Severity.CRITICAL,
                        inspector=INSPECTOR,
                        file=file,
                        message=(
                            f"ZIP member '{name}' contains a directory-traversal "
                            "sequence ('..') – extraction could overwrite host files"
                        ),
                        evidence={"member": name},
                    ))

                # FMT-T-001 – suspicious extension
                suffix = PurePosixPath(name).suffix.lower()
                if suffix in SUSPICIOUS_EXTENSIONS:
                    findings.append(Finding(
                        rule_id="FMT-T-001",
                        severity=Severity.HIGH,
                        inspector=INSPECTOR,
                        file=file,
                        message=(
                            f"Unexpected executable member '{name}' inside "
                            "PyTorch checkpoint ZIP"
                        ),
                        evidence={"member": name, "extension": suffix},
                    ))

            # FMT-T-004 – no weight member found
            if not has_weight:
                findings.append(Finding(
                    rule_id="FMT-T-004",
                    severity=Severity.MEDIUM,
                    inspector=INSPECTOR,
                    file=file,
                    message=(
                        "Checkpoint ZIP contains no recognised weight member "
                        "(data.pkl / data/0); file may be misclassified or corrupted"
                    ),
                    evidence={"members": [i.filename for i in members[:20]]},
                ))

    except zipfile.BadZipFile as exc:
        # Corrupt archive – flag but do not crash the whole scan.
        findings.append(Finding(
            rule_id="FMT-T-001",
            severity=Severity.MEDIUM,
            inspector=INSPECTOR,
            file=file,
            message=f"Could not open checkpoint as a ZIP: {exc}",
            evidence={"error": str(exc)},
        ))

    return findings
