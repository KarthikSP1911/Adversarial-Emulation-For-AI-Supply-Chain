"""Builders for test artifacts. Every file is written as raw bytes; nothing is ever loaded."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

DTYPE_BYTES = {"F32": 4, "F16": 2, "I64": 8, "U8": 1}


def safetensors_bytes(
    tensors: dict[str, tuple[str, list[int]]] | None = None,
    *,
    metadata: dict | None = None,
    header_override: str | None = None,
    trailing: bytes = b"",
) -> bytes:
    tensors = tensors if tensors is not None else {"w": ("F32", [2, 2]), "b": ("F32", [2])}
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = DTYPE_BYTES[dtype]
        for d in shape:
            size *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = (header_override if header_override is not None else json.dumps(header)).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * offset + trailing


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def torch_zip(path: Path, pickle_bytes: bytes, extra: dict[str, bytes] | None = None) -> Path:
    """Same member layout as torch.save (stored, no compression)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", pickle_bytes)
        zf.writestr("archive/byteorder", b"little")
        zf.writestr("archive/data/0", b"\x00" * 16)
        zf.writestr("archive/version", b"3\n")
        zf.writestr("archive/.data/serialization_id", b"1234")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def evil_pickle(cmd: str = "touch /tmp/never") -> bytes:
    """os.system(cmd) as protocol-2 bytes. Built by hand; never unpickled."""
    b = cmd.encode()
    return b"\x80\x02cos\nsystem\nX" + struct.pack("<I", len(b)) + b + b"\x85R."


BENIGN_PICKLE = b"\x80\x02}q\x00X\x01\x00\x00\x00aK\x01s."  # {"a": 1}
