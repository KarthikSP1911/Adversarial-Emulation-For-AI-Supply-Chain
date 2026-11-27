"""Shared pieces for every static checker.

* ``Artifact``  - one file being scanned, plus what we learned about it.
* ``Checker``   - the interface every static checker implements.
* ``make_finding`` - builds a ``Finding`` with a consistent shape.
* ``StaticResult`` - ``InspectorResult`` plus the file manifest.

Rule ID prefixes owned by this layer:

* ``IO-``  file system safety (symlinks, special files, size limits)
* ``DET-`` format detection (disguised or unrecognised files)
* ``SFT-`` safetensors structure
* ``ARC-`` zip / tar archive content
* ``INT-`` integrity (hashes, manifests, CRC, sharded index consistency)

The format scanners own ``FMT-``, ``templates`` owns ``TPL-`` and ``remote_code``
owns ``RCS-``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from modelscan.models import Finding, InspectorResult, Severity

INSPECTOR = "static"

# MITRE ATLAS technique IDs attached to findings as evidence["atlas"].
ATLAS_UNSAFE_ARTIFACT = "AML.T0011.000"  # User Execution: Unsafe AI Artifacts
ATLAS_SUPPLY_CHAIN_MODEL = "AML.T0010.003"  # AI Supply Chain Compromise: Model
ATLAS_EMBED_MALWARE = "AML.T0018.002"  # Manipulate AI Model: Embed Malware


class Kind(str, Enum):
    """What a file really is, decided from its bytes (never its name)."""

    PICKLE = "pickle"
    TORCH_ZIP = "torch_zip"  # zip with <prefix>/data.pkl (torch.save default)
    ZIP = "zip"
    TAR = "tar"
    GZIP = "gzip"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    HDF5 = "hdf5"
    NUMPY = "numpy"
    ONNX = "onnx"
    JSON = "json"
    TEXT = "text"
    EMPTY = "empty"
    UNKNOWN = "unknown"


# Formats that can run code when loaded with their usual loader.
CODE_CAPABLE = frozenset({Kind.PICKLE, Kind.TORCH_ZIP, Kind.NUMPY, Kind.HDF5})


@dataclass
class Artifact:
    path: Path
    rel: str  # path relative to the scan root, used in findings
    size: int
    kind: Kind = Kind.UNKNOWN
    sha256: str | None = None
    notes: list[str] = field(default_factory=list)


class Checker(Protocol):
    """A static checker. It must never import, load or execute the artifact."""

    name: str

    def applies(self, art: Artifact) -> bool: ...

    def check(self, art: Artifact) -> list[Finding]: ...


def make_finding(
    rule_id: str,
    severity: Severity,
    art: Artifact | str,
    message: str,
    *,
    checker: str,
    offset: int | None = None,
    atlas: str | None = None,
    **evidence: object,
) -> Finding:
    file = art.rel if isinstance(art, Artifact) else art
    if atlas:
        evidence["atlas"] = atlas
    return Finding(
        rule_id=rule_id,
        severity=severity,
        inspector=f"{INSPECTOR}.{checker}",
        file=file,
        message=message,
        offset=offset,
        evidence=evidence,
    )


class ArtifactRecord(BaseModel):
    """One manifest line: enough to pin and re-verify the file later."""

    path: str
    size: int
    kind: str
    sha256: str | None = None
    checkers: list[str] = Field(default_factory=list)


class StaticResult(InspectorResult):
    manifest: list[ArtifactRecord] = Field(default_factory=list)
