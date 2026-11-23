"""Bounded, non-following file access so the scanner itself can't be attacked.

Every read is capped, every scan has a deadline, symlinks are never followed
and special files (devices, FIFOs, sockets) are never opened. A FIFO named
``model.bin`` would otherwise hang the scanner; ``/dev/zero`` would fill memory.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from modelscan.config import MAX_FILE_SIZE_BYTES, SCAN_TIMEOUT_SECONDS

CHUNK = 1024 * 1024
HEAD_BYTES = 64


class ScanLimitError(Exception):
    """A safety limit was hit. The file is NOT verified."""


class Deadline:
    def __init__(self, seconds: float = SCAN_TIMEOUT_SECONDS) -> None:
        self.seconds = seconds
        self.end = time.monotonic() + seconds

    def check(self) -> None:
        # >= so a zero deadline is expired even on coarse clocks (Windows ~15 ms)
        if time.monotonic() >= self.end:
            raise ScanLimitError(f"scan exceeded {self.seconds:g}s time limit")


@dataclass
class PathCheck:
    """Result of inspecting a path *without* following it."""

    ok: bool
    size: int = 0
    problem: str | None = None  # "symlink", "special", "too_large", "outside_root"
    detail: str = ""


def inspect_path(path: Path, root: Path | None, max_size: int = MAX_FILE_SIZE_BYTES) -> PathCheck:
    try:
        st = path.lstat()
    except OSError as exc:
        return PathCheck(False, problem="unreadable", detail=str(exc))
    if stat.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(path)
        except OSError:
            target = "?"
        return PathCheck(False, problem="symlink", detail=target)
    if not stat.S_ISREG(st.st_mode):
        return PathCheck(False, problem="special", detail=stat.filemode(st.st_mode))
    if root is not None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return PathCheck(False, problem="outside_root", detail=str(path.resolve()))
    if st.st_size > max_size:
        return PathCheck(False, st.st_size, "too_large", f"{st.st_size} > {max_size} bytes")
    return PathCheck(True, st.st_size)


def open_regular(path: Path) -> BinaryIO:
    """Open for reading, refusing symlinks where the OS supports O_NOFOLLOW."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ScanLimitError("not a regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_head(path: Path, n: int = HEAD_BYTES) -> bytes:
    with open_regular(path) as fh:
        return fh.read(n)


def read_tail(path: Path, n: int) -> bytes:
    with open_regular(path) as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(size - n, 0))
        return fh.read(n)


def read_range(fh: BinaryIO, offset: int, length: int, limit: int) -> bytes:
    """Read exactly ``length`` bytes at ``offset``; refuse anything over ``limit``."""
    if length < 0 or length > limit:
        raise ScanLimitError(f"refusing to read {length} bytes (limit {limit})")
    fh.seek(offset)
    data = fh.read(length)
    if len(data) != length:
        raise ScanLimitError(f"short read at {offset}: wanted {length}, got {len(data)}")
    return data


def iter_chunks(fh: BinaryIO, deadline: Deadline | None = None) -> Iterator[bytes]:
    while chunk := fh.read(CHUNK):
        if deadline:
            deadline.check()
        yield chunk


def sha256_file(path: Path, deadline: Deadline | None = None) -> str:
    h = hashlib.sha256()
    with open_regular(path) as fh:
        for chunk in iter_chunks(fh, deadline):
            h.update(chunk)
    return h.hexdigest()
