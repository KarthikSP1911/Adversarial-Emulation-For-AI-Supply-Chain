# Static scan: foundation, integrity and archive rules

Owner: Susheel. Pickle rules (`PKL-*`) are documented with the pickle engine.

Every finding carries `evidence.atlas` where a MITRE ATLAS technique applies:
`AML.T0011.000` (Unsafe AI Artifacts), `AML.T0010.003` (Supply Chain: Model),
`AML.T0018.002` (Embed Malware).

## Modules

| File                                 | Job                                                            |
| ------------------------------------ | -------------------------------------------------------------- |
| `static/__init__.py`                 | Manager: runs every checker per file, then folder-level checks |
| `static/base.py`                     | `Artifact`, `Kind`, `Checker` interface, `StaticResult`        |
| `static/safe_io.py`                  | No symlink following, no special files, size and time limits   |
| `static/detect.py`                   | Real format from bytes; compares with the extension            |
| `static/formats/safetensors.py`      | Header and tensor layout validation                            |
| `static/archive.py`                  | Zip and tar traps, read from the directory only                |
| `static/integrity.py`                | SHA-256 manifest, pinned hashes, zip CRC, shard index checks   |

## Rules

| ID      | Severity        | Meaning                                                               |
| ------- | --------------- | --------------------------------------------------------------------- |
| IO-001  | HIGH            | Symlink, special file, outside root, too large or unreadable: skipped |
| IO-090  | HIGH            | A checker crashed; the file is not verified (status becomes `error`)  |
| DET-001 | CRITICAL / HIGH | Extension and content disagree; CRITICAL if code hides behind a safe name |
| DET-002 | HIGH            | Zip directory inside a non-zip file (polyglot)                        |
| DET-003 | LOW             | Unrecognised binary, not inspected                                    |
| DET-004 | MEDIUM / LOW    | Recognised format with no checker yet (numpy, HDF5, GGUF, ONNX, gzip) |
| SFT-001 | HIGH            | Malformed safetensors header                                          |
| SFT-002 | HIGH            | Duplicate keys in the header                                          |
| SFT-003 | HIGH            | Bad tensor entry: dtype, shape, offsets or size mismatch              |
| SFT-004 | HIGH            | Overlapping tensors                                                   |
| SFT-005 | HIGH            | Unreferenced bytes between tensors                                    |
| SFT-006 | CRITICAL / HIGH | Bytes after the last tensor; CRITICAL if they look like pickle, zip or executable |
| SFT-007 | MEDIUM          | `__metadata__` not string-to-string, or a huge value                  |
| SFT-008 | MEDIUM          | Suspicious text in `__metadata__`                                     |
| ARC-001 | CRITICAL        | Path traversal member name                                            |
| ARC-002 | HIGH            | Symlink, hard link, device or FIFO member                             |
| ARC-003 | HIGH            | Decompression bomb (ratio or total size)                              |
| ARC-004 | HIGH            | Duplicate member names                                                |
| ARC-005 | HIGH            | Too many members                                                      |
| ARC-006 | HIGH            | Encrypted member                                                      |
| ARC-007 | HIGH            | Local header disagrees with the central directory                     |
| ARC-008 | HIGH / MEDIUM   | Data before the first entry, or a large zip comment                   |
| ARC-009 | MEDIUM          | Nested archive, not inspected                                         |
| ARC-010 | HIGH / MEDIUM   | Executable or script member (MEDIUM for TorchScript `code/*.py`)      |
| ARC-011 | HIGH / MEDIUM   | Torch archive layout that `torch.save` would not produce              |
| ARC-012 | HIGH            | Archive cannot be read                                                |
| INT-001 | CRITICAL        | SHA-256 differs from the pinned value                                 |
| INT-002 | HIGH            | Pinned file missing, or file not in the pinned manifest               |
| INT-003 | HIGH            | Zip member fails its CRC check (modified after packaging)             |
| INT-004 | HIGH / MEDIUM   | Shard index points to missing shards or tensors, or misses some       |
| INT-005 | MEDIUM          | Pickle weights shipped next to safetensors weights                    |
| INT-006 | HIGH            | Pinned manifest file cannot be used                                   |

## Using it

```python
from pathlib import Path
from modelscan.static import scan

result = scan(files, pinned=Path("pins.json"))  # pinned is optional
result.manifest  # path, size, kind, sha256, checkers that ran
```

`pins.json` is either `{"file": "sha256"}` or a saved `manifest` list. Save one
`result.manifest` from a model you trust and pass it back later to detect changes.
