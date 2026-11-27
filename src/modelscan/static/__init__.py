"""Static scan manager: runs every checker on every file and combines results.

Order per file: path safety -> format detection -> fingerprint -> safetensors and
archive checkers -> pickle engine -> format, chat-template and shipped-code
scanners. Then folder-level integrity checks.

Fail closed: a checker that crashes, a file that can't be opened safely, or a
format nobody inspects is reported. It never silently counts as clean.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Callable
from pathlib import Path

from modelscan.models import Finding, Severity, Status

from . import archive, detect, integrity, remote_code, safe_io, templates
from . import pickle_scan as pickle_engine
from .base import (
    ATLAS_UNSAFE_ARTIFACT,
    Artifact,
    ArtifactRecord,
    Kind,
    StaticResult,
    make_finding,
)
from .formats import gguf, keras, numpy, onnx, safetensors, torch

INSPECTOR = "static"

# Recognised formats that may go uninspected, and how much that matters.
NOT_YET_INSPECTED: dict[Kind, Severity] = {
    Kind.NUMPY: Severity.MEDIUM,  # object arrays are pickles
    Kind.HDF5: Severity.MEDIUM,  # Keras Lambda layers carry code
    Kind.GGUF: Severity.LOW,
    Kind.ONNX: Severity.LOW,
    Kind.GZIP: Severity.LOW,
}

# Scanners that take a path and choose files by name: (name, applies, scan).
_PATH_SCANNERS: tuple[
    tuple[str, Callable[[Artifact], bool], Callable[[Path], list[Finding]]], ...
] = (
    (
        "torch",
        lambda a: a.path.suffix.lower() in torch.TORCH_SUFFIXES
        and a.kind in (Kind.ZIP, Kind.TORCH_ZIP),
        torch.scan,
    ),
    ("keras", lambda a: a.path.suffix.lower() in keras.KERAS_SUFFIXES, keras.scan),
    ("onnx", lambda a: a.path.suffix.lower() in onnx.ONNX_SUFFIXES, onnx.scan),
    ("gguf", lambda a: a.path.suffix.lower() in gguf.GGUF_SUFFIXES, gguf.scan),
    ("numpy", lambda a: a.path.suffix.lower() in numpy.NUMPY_SUFFIXES, numpy.scan),
    ("templates", lambda a: templates._is_template_file(a.path), templates.scan),
    ("remote_code", lambda a: remote_code._is_python_file(a.path), remote_code.scan),
)

# The scanner that reads each kind's content. HDF5 and gzip have none: the
# keras scanner reads only the magic bytes of an HDF5 file.
_KIND_SCANNER = {Kind.NUMPY: "numpy", Kind.GGUF: "gguf", Kind.ONNX: "onnx"}

_PATH_PROBLEMS = {
    "symlink": "Symlink not followed",
    "special": "Not a regular file (device, FIFO or socket); not opened",
    "outside_root": "Resolves outside the scanned folder; not opened",
    "too_large": "File exceeds the size limit; not inspected",
    "unreadable": "File cannot be read",
}


def scan(
    files: list[Path],
    root: Path | None = None,
    pinned: Path | dict[str, str] | None = None,
) -> StaticResult:
    root = root or _common_root(files)
    findings: list[Finding] = []
    coverage: list[str] = []
    errors: list[str] = []
    records: list[ArtifactRecord] = []
    arts: list[Artifact] = []

    for path in files:
        rel = _rel(path, root)
        pc = safe_io.inspect_path(path, root)
        if not pc.ok:
            findings.append(make_finding(
                "IO-001", Severity.HIGH, rel, f"{_PATH_PROBLEMS[pc.problem or 'unreadable']}: "
                f"{pc.detail}", checker="safe_io", atlas=ATLAS_UNSAFE_ARTIFACT,
                problem=pc.problem, detail=pc.detail,
            ))  # fmt: skip
            records.append(ArtifactRecord(path=rel, size=pc.size, kind="skipped"))
            coverage.append(f"{rel}: skipped ({pc.problem})")
            continue

        art = Artifact(path=path, rel=rel, size=pc.size)
        ran, file_findings, file_errors = _scan_one(art)
        findings += file_findings
        errors += file_errors
        arts.append(art)
        records.append(ArtifactRecord(
            path=rel, size=art.size, kind=art.kind.value, sha256=art.sha256, checkers=ran,
        ))  # fmt: skip
        coverage.append(f"{rel}: {art.kind.value}; checked by {', '.join(ran) or 'nothing'}")
        coverage += art.notes

    try:
        findings += integrity.check_consistency(arts)
    except Exception as exc:  # noqa: BLE001 - fail closed
        errors.append(f"integrity.consistency: {type(exc).__name__}: {exc}")
        findings.append(_crash("integrity", "<folder>", exc))

    if pinned is not None:
        try:
            pins = integrity.load_pinned(pinned) if isinstance(pinned, Path) else pinned
            findings += integrity.check_pinned(records, pins)
            coverage.append(f"pinned manifest: {len(pins)} file(s) compared")
        except Exception as exc:  # noqa: BLE001 - unreadable pins must not mean "no pins"
            errors.append(f"pinned manifest: {type(exc).__name__}: {exc}")
            findings.append(make_finding(
                "INT-006", Severity.HIGH, str(pinned), f"Pinned manifest cannot be used: {exc}",
                checker="integrity",
            ))  # fmt: skip

    inspected = sum(1 for r in records if r.kind != "skipped")
    coverage.insert(0, f"static: {inspected} of {len(files)} file(s) inspected")
    coverage.append("No findings means none were found by these checks, not that none exist.")
    return StaticResult(
        inspector=INSPECTOR,
        status=Status.ERROR if errors else Status.OK,
        findings=findings,
        coverage=coverage,
        error="; ".join(errors) or None,
        manifest=records,
    )


def _scan_one(art: Artifact) -> tuple[list[str], list[Finding], list[str]]:
    ran: list[str] = []
    findings: list[Finding] = []
    errors: list[str] = []
    deadline = safe_io.Deadline()

    def step(name: str, fn: Callable[[], list[Finding] | None]) -> None:
        try:
            deadline.check()
            findings.extend(fn() or [])
            ran.append(name)
        except Exception as exc:  # noqa: BLE001 - one broken checker must not hide others
            errors.append(f"{art.rel}: {name}: {type(exc).__name__}: {exc}")
            findings.append(_crash(name, art.rel, exc))

    step("detect", lambda: detect.check(art))
    step("integrity", lambda: integrity.fingerprint(art, deadline))
    step("zip_crc", lambda: integrity.check_zip_crc(art, deadline))
    if safetensors.applies(art):
        step("safetensors", lambda: safetensors.check(art))
    if archive.applies(art):
        step("archive", lambda: archive.check(art))
    if art.kind in (Kind.PICKLE, Kind.TORCH_ZIP, Kind.ZIP):
        step("pickle", lambda: _pickle(art))

    selected: set[str] = set()
    for name, applies, run in _PATH_SCANNERS:
        if applies(art):
            selected.add(name)
            step(name, lambda run=run: [_relabel(f, art) for f in run(art.path)])

    # Scanners pick files by name, so content under another name is still uninspected.
    if art.kind in NOT_YET_INSPECTED and _KIND_SCANNER.get(art.kind) not in selected:
        findings.append(make_finding(
            "DET-004", NOT_YET_INSPECTED[art.kind], art,
            f"{art.kind.value} file recognised but no checker inspected its content",
            checker="detect", detected=art.kind.value,
        ))  # fmt: skip
    return ran, findings, errors


def _pickle(art: Artifact) -> list[Finding]:
    """Run the pickle engine, also on pickles whose file name hides them."""
    outcome = pickle_engine.scan_file(art.path)
    if outcome.scanned:
        if outcome.error:
            raise RuntimeError(outcome.error)
        art.notes += outcome.notes
        return [_relabel(f, art) for f in outcome.findings]

    # scan_file chose by extension; detection says there is pickle data here.
    findings: list[Finding] = []
    streams = 0
    if art.kind == Kind.PICKLE:
        with safe_io.open_regular(art.path) as fh:
            data = fh.read(pickle_engine.MAX_PICKLE_BYTES + 1)
        if len(data) > pickle_engine.MAX_PICKLE_BYTES:
            raise safe_io.ScanLimitError("pickle larger than the engine's read limit")
        found, _ = pickle_engine.scan_pickle_bytes(data, art.rel)
        findings += found
        streams += 1
    else:
        with safe_io.open_regular(art.path) as raw, zipfile.ZipFile(raw) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.flag_bits & 0x1:
                    continue
                with zf.open(info) as fh:
                    head = fh.read(2)
                if not (info.filename.endswith((".pkl", ".pickle")) or (
                    head[:1] == b"\x80" and head[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05")
                )):  # fmt: skip
                    continue
                with zf.open(info) as fh:
                    data = fh.read(pickle_engine.MAX_PICKLE_BYTES + 1)
                if len(data) > pickle_engine.MAX_PICKLE_BYTES:
                    raise safe_io.ScanLimitError(f"{info.filename} exceeds the pickle read limit")
                found, _ = pickle_engine.scan_pickle_bytes(data, f"{art.rel}!{info.filename}")
                findings += found
                streams += 1
    if streams:
        art.notes.append(
            f"{art.rel}: {streams} pickle stream(s) scanned despite the {art.path.suffix!r} name"
        )
    return findings


def _relabel(f: Finding, art: Artifact) -> Finding:
    """Some scanners name files by full path; use the scan-relative name."""
    full = str(art.path)
    if f.file == full or f.file.startswith(full + "!"):
        return f.model_copy(update={"file": art.rel + f.file[len(full) :]})
    return f


def _crash(checker: str, file: str, exc: Exception) -> Finding:
    return make_finding(
        "IO-090", Severity.HIGH, file,
        f"{checker} failed ({type(exc).__name__}: {exc}); file is not verified",
        checker=checker, error=f"{type(exc).__name__}: {exc}",
    )  # fmt: skip


def _common_root(files: list[Path]) -> Path | None:
    if not files:
        return None
    if len(files) == 1:
        return files[0].parent
    try:
        return Path(os.path.commonpath([str(f.absolute()) for f in files]))
    except ValueError:
        return None


def _rel(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            pass
    return path.as_posix()
