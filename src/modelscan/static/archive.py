"""Zip and tar traps: everything is read from the directory, nothing is extracted.

Model formats that are zips (torch.save, .keras, .npz) are loaded by readers
that differ from Python's ``zipfile``. Duplicate names, local headers that
disagree with the central directory, and prepended data are classic ways to
show the scanner one file and the loader another.
"""

from __future__ import annotations

import posixpath
import stat
import struct
import tarfile
import zipfile
from pathlib import Path

from modelscan.models import Finding, Severity

from . import safe_io
from .base import ATLAS_UNSAFE_ARTIFACT, Artifact, Kind, make_finding

CHECKER = "archive"
MAX_MEMBERS = 10_000
MAX_TOTAL_UNCOMPRESSED = 50 * 1024**3
MAX_RATIO = 100  # uncompressed / compressed for one member
RATIO_MIN_SIZE = 10 * 1024**2  # ignore ratio on small members
MAX_COMMENT = 1024

EXECUTABLE_SUFFIXES = frozenset({
    ".py", ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bat",
    ".cmd", ".ps1", ".sh", ".bash", ".jar", ".js", ".vbs", ".scr", ".msi",
})  # fmt: skip
EXECUTABLE_MAGIC = (b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"#!")
NESTED_MAGIC = (b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"\xfd7zXZ", b"7z\xbc\xaf")

# Members torch.save writes under its single top-level folder.
TORCH_MEMBERS = frozenset({
    "data.pkl", "version", ".format_version", "byteorder", ".storage_alignment",
    ".data/version", ".data/serialization_id",
})  # fmt: skip


def applies(art: Artifact) -> bool:
    return art.kind in (Kind.ZIP, Kind.TORCH_ZIP, Kind.TAR)


def check(art: Artifact) -> list[Finding]:
    if art.kind == Kind.TAR:
        return _check_tar(art)
    return _check_zip(art)


# --- shared ---------------------------------------------------------------


def unsafe_name(name: str) -> str | None:
    """Why a member name could escape an extraction folder, or None."""
    if "\x00" in name:
        return "contains a NUL byte"
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        return "absolute path"
    if ".." in norm.split("/"):
        return "parent directory (..) component"
    if posixpath.normpath(norm).startswith("../"):
        return "escapes the archive root"
    return None


def _executable(name: str, head: bytes) -> bool:
    return Path(name).suffix.lower() in EXECUTABLE_SUFFIXES or head.startswith(EXECUTABLE_MAGIC)


# --- zip ------------------------------------------------------------------


def _check_zip(art: Artifact) -> list[Finding]:
    findings: list[Finding] = []

    def add(rule: str, sev: Severity, msg: str, **ev: object) -> None:
        findings.append(
            make_finding(rule, sev, art, msg, checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, **ev)
        )

    try:
        zf = zipfile.ZipFile(safe_io.open_regular(art.path))
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        add("ARC-012", Severity.HIGH, f"Archive cannot be read: {exc}", error=str(exc))
        return findings

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            add("ARC-005", Severity.HIGH,
                f"{len(infos)} members (limit {MAX_MEMBERS}); not inspected further",
                members=len(infos))  # fmt: skip
            return findings

        seen: set[str] = set()
        total = 0
        torch_prefixes: set[str] = set()
        for info in infos:
            name = info.filename
            if name in seen:
                add("ARC-004", Severity.HIGH,
                    f"Duplicate member {name!r}: readers may pick different copies",
                    member=name)  # fmt: skip
            seen.add(name)

            why = unsafe_name(name)
            if why:
                add("ARC-001", Severity.CRITICAL, f"Path traversal in member {name!r}: {why}",
                    member=name)  # fmt: skip

            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                add("ARC-002", Severity.HIGH, f"Symlink member {name!r}", member=name)

            if info.flag_bits & 0x1:
                add("ARC-006", Severity.HIGH, f"Encrypted member {name!r} cannot be inspected",
                    member=name)  # fmt: skip

            total += info.file_size
            if info.file_size >= RATIO_MIN_SIZE and info.file_size > MAX_RATIO * max(
                info.compress_size, 1
            ):
                add("ARC-003", Severity.HIGH,
                    f"Member {name!r} expands {info.file_size // max(info.compress_size, 1)}x "
                    "(decompression bomb)",
                    member=name, size=info.file_size, compressed=info.compress_size)  # fmt: skip

            if name.endswith("/data.pkl"):
                torch_prefixes.add(name[: -len("/data.pkl")])

        if total > MAX_TOTAL_UNCOMPRESSED:
            add("ARC-003", Severity.HIGH,
                f"Archive expands to {total} bytes (limit {MAX_TOTAL_UNCOMPRESSED})",
                total=total)  # fmt: skip

        findings += _check_zip_layout(art, zf, infos)
        findings += _check_member_content(art, zf, infos)
        if art.kind == Kind.TORCH_ZIP:
            findings += _check_torch_layout(art, infos, torch_prefixes)
    return findings


def _check_zip_layout(
    art: Artifact, zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> list[Finding]:
    """Local headers must match the central directory; no hidden prefix data."""
    findings: list[Finding] = []
    fp = zf.fp
    assert fp is not None
    first = min((i.header_offset for i in infos), default=0)
    if first > 0:
        findings.append(make_finding(
            "ARC-008", Severity.HIGH, art,
            f"{first} byte(s) of data before the first zip entry",
            checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, prefix_bytes=first,
        ))  # fmt: skip
    if len(zf.comment) > MAX_COMMENT:
        findings.append(make_finding(
            "ARC-008", Severity.MEDIUM, art,
            f"Zip comment is {len(zf.comment)} bytes (possible hidden payload)",
            checker=CHECKER, comment_bytes=len(zf.comment),
        ))  # fmt: skip

    for info in infos:
        fp.seek(info.header_offset)
        local = fp.read(30)
        if len(local) < 30 or local[:4] != b"PK\x03\x04":
            findings.append(make_finding(
                "ARC-007", Severity.HIGH, art,
                f"Member {info.filename!r}: no local header at the recorded offset",
                checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, member=info.filename,
            ))  # fmt: skip
            continue
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        raw_name = fp.read(name_len)
        local_name = raw_name.decode("utf-8" if info.flag_bits & 0x800 else "cp437", "replace")
        if local_name != info.orig_filename:
            findings.append(make_finding(
                "ARC-007", Severity.HIGH, art,
                f"Local header names {local_name!r} but directory says {info.orig_filename!r}",
                checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT,
                member=info.filename, local_name=local_name,
            ))  # fmt: skip
    return findings


def _check_member_content(
    art: Artifact, zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> list[Finding]:
    """Look at the first bytes of each member for executables and nested archives."""
    findings: list[Finding] = []
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            continue
        try:
            with zf.open(info) as fh:
                head = fh.read(8)
        except (zipfile.BadZipFile, OSError, NotImplementedError, RuntimeError, ValueError):
            continue  # integrity.py reports unreadable members
        name = info.filename
        if _executable(name, head):
            torchscript = "/code/" in name and name.endswith(".py")
            findings.append(make_finding(
                "ARC-010",
                Severity.MEDIUM if torchscript else Severity.HIGH,
                art,
                f"Executable or script member {name!r}"
                + (" (TorchScript code; review before use)" if torchscript else ""),
                checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, member=name,
                starts_with=head.hex(),
            ))  # fmt: skip
        elif head.startswith(NESTED_MAGIC) or _is_tar_name(name):
            findings.append(make_finding(
                "ARC-009", Severity.MEDIUM, art,
                f"Nested archive {name!r} is not inspected",
                checker=CHECKER, member=name, starts_with=head.hex(),
            ))  # fmt: skip
    return findings


def _is_tar_name(name: str) -> bool:
    return name.lower().endswith((".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"))


def _check_torch_layout(
    art: Artifact, infos: list[zipfile.ZipInfo], prefixes: set[str]
) -> list[Finding]:
    findings: list[Finding] = []
    if len(prefixes) != 1:
        findings.append(make_finding(
            "ARC-011", Severity.HIGH, art,
            f"Torch archive has {len(prefixes)} data.pkl folders; loaders may pick different ones",
            checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, prefixes=sorted(prefixes),
        ))  # fmt: skip
        return findings
    (prefix,) = prefixes
    unexpected = []
    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        if not name.startswith(prefix + "/"):
            unexpected.append(name)
            continue
        inner = name[len(prefix) + 1 :]
        if inner in TORCH_MEMBERS or inner.startswith("data/"):
            continue
        unexpected.append(name)
    if unexpected:
        findings.append(make_finding(
            "ARC-011", Severity.MEDIUM, art,
            f"{len(unexpected)} member(s) torch.save would not write",
            checker=CHECKER, members=unexpected[:50],
        ))  # fmt: skip
    return findings


# --- tar ------------------------------------------------------------------


def _check_tar(art: Artifact) -> list[Finding]:
    findings: list[Finding] = []

    def add(rule: str, sev: Severity, msg: str, **ev: object) -> None:
        findings.append(
            make_finding(rule, sev, art, msg, checker=CHECKER, atlas=ATLAS_UNSAFE_ARTIFACT, **ev)
        )

    total = count = 0
    try:
        with safe_io.open_regular(art.path) as raw, tarfile.open(fileobj=raw, mode="r:") as tf:
            for member in tf:  # streams headers; never extracts
                count += 1
                if count > MAX_MEMBERS:
                    add("ARC-005", Severity.HIGH, f"More than {MAX_MEMBERS} members")
                    break
                name = member.name
                why = unsafe_name(name)
                if why:
                    add("ARC-001", Severity.CRITICAL, f"Path traversal in member {name!r}: {why}",
                        member=name)  # fmt: skip
                if member.issym() or member.islnk():
                    add("ARC-002", Severity.HIGH,
                        f"Link member {name!r} -> {member.linkname!r}",
                        member=name, target=member.linkname)  # fmt: skip
                if member.isdev() or member.isfifo():
                    add("ARC-002", Severity.HIGH, f"Device or FIFO member {name!r}", member=name)
                if member.mode & (stat.S_ISUID | stat.S_ISGID):
                    add("ARC-010", Severity.HIGH, f"setuid/setgid member {name!r}", member=name)
                elif member.isfile() and Path(name).suffix.lower() in EXECUTABLE_SUFFIXES:
                    add("ARC-010", Severity.HIGH, f"Executable or script member {name!r}",
                        member=name)  # fmt: skip
                total += member.size
    except (tarfile.TarError, OSError, ValueError) as exc:
        add("ARC-012", Severity.HIGH, f"Archive cannot be read: {exc}", error=str(exc))
    if total > MAX_TOTAL_UNCOMPRESSED:
        add("ARC-003", Severity.HIGH, f"Archive holds {total} bytes", total=total)
    return findings
