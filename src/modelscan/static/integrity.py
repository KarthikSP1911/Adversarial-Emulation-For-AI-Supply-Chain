"""Fingerprints, pinned manifests and internal consistency.

* Every file gets a SHA-256 and goes into the manifest.
* If the caller supplies a pinned manifest (known-good hashes), any change,
  missing file or extra file is reported.
* Zip members are streamed so their CRC-32 is verified (edits after packaging).
* Sharded checkpoints are cross-checked against their ``*.index.json``.

Comparing hashes against a hub or a trusted registry is provenance's job;
this module only provides the fingerprints and the local comparison.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from modelscan.models import Finding, Severity

from . import safe_io
from .base import (
    ATLAS_SUPPLY_CHAIN_MODEL,
    Artifact,
    ArtifactRecord,
    Kind,
    make_finding,
)
from .formats import safetensors as st

CHECKER = "integrity"
MAX_INDEX_BYTES = 64 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


# --- per file -------------------------------------------------------------


def fingerprint(art: Artifact, deadline: safe_io.Deadline | None = None) -> None:
    art.sha256 = safe_io.sha256_file(art.path, deadline)


def check_zip_crc(art: Artifact, deadline: safe_io.Deadline | None = None) -> list[Finding]:
    """Stream every member; zipfile raises on a CRC-32 mismatch at end of member."""
    if art.kind not in (Kind.ZIP, Kind.TORCH_ZIP):
        return []
    findings: list[Finding] = []
    try:
        with safe_io.open_regular(art.path) as raw, zipfile.ZipFile(raw) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.flag_bits & 0x1:
                    continue  # encrypted members are reported by archive.py
                try:
                    with zf.open(info) as fh:
                        for _ in safe_io.iter_chunks(fh, deadline):
                            pass
                except zipfile.BadZipFile as exc:
                    findings.append(make_finding(
                        "INT-003", Severity.HIGH, art,
                        f"Member {info.filename!r} fails its CRC check: modified after packaging",
                        checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL,
                        member=info.filename, error=str(exc),
                    ))  # fmt: skip
                except (NotImplementedError, OSError, ValueError, RuntimeError) as exc:
                    findings.append(make_finding(
                        "INT-003", Severity.HIGH, art,
                        f"Member {info.filename!r} cannot be read to verify it: {exc}",
                        checker=CHECKER, member=info.filename, error=str(exc),
                    ))  # fmt: skip
    except (zipfile.BadZipFile, OSError, ValueError):
        pass  # archive.py reports an unreadable archive
    return findings


# --- pinned manifest ------------------------------------------------------


def load_pinned(path: Path) -> dict[str, str]:
    """Accept ``{"file": "sha256"}`` or a manifest list of ``{"path", "sha256"}``.

    Raises ValueError for anything else, so a broken pin file never means "no pins".
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "manifest" in data:
        data = data["manifest"]
    if isinstance(data, list):
        data = {row["path"]: row["sha256"] for row in data}
    if not isinstance(data, dict):
        raise ValueError("pinned manifest must be an object or a list of records")
    pins: dict[str, str] = {}
    for name, digest in data.items():
        digest = str(digest).lower().removeprefix("sha256:")
        if len(digest) != 64 or not set(digest) <= HEX:
            raise ValueError(f"bad sha256 for {name!r}")
        pins[str(name)] = digest
    return pins


def check_pinned(records: list[ArtifactRecord], pins: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    actual = {r.path: r.sha256 for r in records}
    for name, expected in sorted(pins.items()):
        got = actual.get(name)
        if name not in actual:
            findings.append(make_finding(
                "INT-002", Severity.HIGH, name,
                "Pinned file is missing from the artifact",
                checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL, expected=expected,
            ))  # fmt: skip
        elif got != expected:
            findings.append(make_finding(
                "INT-001", Severity.CRITICAL, name,
                "SHA-256 does not match the pinned value",
                checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL,
                expected=expected, actual=got,
            ))  # fmt: skip
    for name in sorted(set(actual) - set(pins)):
        findings.append(make_finding(
            "INT-002", Severity.HIGH, name,
            "File is not in the pinned manifest",
            checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL, actual=actual[name],
        ))  # fmt: skip
    return findings


# --- folder consistency ---------------------------------------------------


def check_consistency(arts: list[Artifact]) -> list[Finding]:
    findings: list[Finding] = []
    by_rel = {a.rel: a for a in arts}
    for art in arts:
        if art.path.name.endswith(".index.json"):
            findings += _check_index(art, by_rel)

    names = {Path(a.rel).name for a in arts}
    has_safe = any(n.endswith(".safetensors") for n in names)
    pickles = sorted(
        a.rel for a in arts
        if a.kind in (Kind.PICKLE, Kind.TORCH_ZIP) and Path(a.rel).name.startswith(
            ("pytorch_model", "model", "adapter_model")
        )
    )  # fmt: skip
    if has_safe and pickles:
        findings.append(make_finding(
            "INT-005", Severity.MEDIUM, pickles[0],
            "Pickle weights ship next to safetensors weights; some loaders fall back "
            "to the pickle, so both copies must be trusted",
            checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL, pickle_files=pickles,
        ))  # fmt: skip
    return findings


def _check_index(index: Artifact, by_rel: dict[str, Artifact]) -> list[Finding]:
    """``weight_map`` must name real shards, and safetensors shards must agree."""
    findings: list[Finding] = []

    def add(rule: str, sev: Severity, msg: str, **ev: object) -> None:
        findings.append(make_finding(
            rule, sev, index, msg, checker=CHECKER, atlas=ATLAS_SUPPLY_CHAIN_MODEL, **ev
        ))  # fmt: skip

    try:
        if index.size > MAX_INDEX_BYTES:
            raise ValueError(f"index is {index.size} bytes")
        with safe_io.open_regular(index.path) as fh:
            data = json.loads(fh.read().decode("utf-8"))
        weight_map = data["weight_map"]
        if not isinstance(weight_map, dict):
            raise ValueError("weight_map is not an object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add("INT-004", Severity.HIGH, f"Shard index cannot be read: {exc}", error=str(exc))
        return findings

    folder = str(Path(index.rel).parent)
    folder = "" if folder == "." else folder + "/"
    expected: dict[str, set[str]] = {}
    for tensor, shard in weight_map.items():
        expected.setdefault(str(shard), set()).add(str(tensor))

    for shard, tensors in sorted(expected.items()):
        art = by_rel.get(folder + shard)
        if art is None:
            add("INT-004", Severity.HIGH, f"Index points to missing shard {shard!r}",
                shard=shard)  # fmt: skip
            continue
        if art.kind != Kind.SAFETENSORS:
            continue  # pickle shards can't be listed without loading them
        try:
            _, header = st.read_header(art)
        except Exception:  # noqa: BLE001 - safetensors checker reports it
            continue
        actual = {k for k in header if k != "__metadata__"}
        missing = sorted(tensors - actual)
        hidden = sorted(actual - tensors)
        if missing:
            add("INT-004", Severity.HIGH,
                f"{len(missing)} tensor(s) listed for {shard!r} are not in it",
                shard=shard, tensors=missing[:50])  # fmt: skip
        if hidden:
            add("INT-004", Severity.HIGH,
                f"{shard!r} holds {len(hidden)} tensor(s) the index does not list",
                shard=shard, tensors=hidden[:50])  # fmt: skip

    prefix = Path(index.path.name.removesuffix(".index.json")).stem  # model / pytorch_model
    for rel in sorted(by_rel):
        name = Path(rel).name
        in_folder = rel.startswith(folder) and "/" not in rel[len(folder) :]
        if in_folder and name.startswith(prefix + "-") and name not in expected:
            add("INT-004", Severity.MEDIUM, f"Shard {name!r} is not referenced by the index",
                shard=name)  # fmt: skip
    return findings
