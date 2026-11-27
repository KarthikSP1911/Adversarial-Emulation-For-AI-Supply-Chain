import json
import zipfile

from static_fixtures import BENIGN_PICKLE, safetensors_bytes, write

from modelscan.models import Severity
from modelscan.static import integrity
from modelscan.static.base import Artifact, ArtifactRecord, Kind


def test_crc_tamper_detected(tmp_path):
    p = tmp_path / "m.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data/0", b"A" * 64)
    data = bytearray(p.read_bytes())
    i = data.index(b"A" * 64)
    data[i] = ord("B")  # change one byte, keep the stored CRC
    p.write_bytes(bytes(data))
    art = Artifact(path=p, rel="m.zip", size=len(data), kind=Kind.ZIP)
    (f,) = integrity.check_zip_crc(art)
    assert f.rule_id == "INT-003"


def test_pinned_manifest(tmp_path):
    good = "a" * 64
    records = [
        ArtifactRecord(path="model.safetensors", size=1, kind="safetensors", sha256="b" * 64),
        ArtifactRecord(path="extra.pkl", size=1, kind="pickle", sha256="c" * 64),
    ]
    pins = {"model.safetensors": good, "config.json": good}
    findings = integrity.check_pinned(records, pins)
    by_file = {f.file: f for f in findings}
    assert by_file["model.safetensors"].rule_id == "INT-001"
    assert by_file["model.safetensors"].severity == Severity.CRITICAL
    assert by_file["config.json"].rule_id == "INT-002"
    assert by_file["extra.pkl"].rule_id == "INT-002"


def test_load_pinned_formats(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"manifest": [{"path": "a", "sha256": "sha256:" + "A" * 64}]}))
    assert integrity.load_pinned(p) == {"a": "a" * 64}
    p.write_text(json.dumps({"a": "short"}))
    try:
        integrity.load_pinned(p)
    except ValueError:
        pass
    else:
        raise AssertionError("bad digest accepted")


def _arts(tmp_path, files):
    out = []
    for name, data in files.items():
        p = write(tmp_path / name, data)
        kind = Kind.SAFETENSORS if name.endswith(".safetensors") else Kind.JSON
        if name.endswith(".bin"):
            kind = Kind.PICKLE
        out.append(Artifact(path=p, rel=name, size=len(data), kind=kind))
    return out


def test_shard_index_consistency(tmp_path):
    index = {"weight_map": {
        "w": "model-00001-of-00002.safetensors",
        "missing_tensor": "model-00001-of-00002.safetensors",
        "x": "model-00002-of-00002.safetensors",
    }}  # fmt: skip
    arts = _arts(tmp_path, {
        "model.safetensors.index.json": json.dumps(index).encode(),
        "model-00001-of-00002.safetensors": safetensors_bytes({
            "w": ("F32", [1]), "hidden": ("F32", [1]),
        }),
        "model-00003-of-00003.safetensors": safetensors_bytes({"z": ("F32", [1])}),
    })  # fmt: skip
    msgs = [f.message for f in integrity.check_consistency(arts) if f.rule_id == "INT-004"]
    assert any("missing shard" in m for m in msgs)
    assert any("are not in it" in m for m in msgs)
    assert any("does not list" in m for m in msgs)
    assert any("not referenced" in m for m in msgs)


def test_pickle_next_to_safetensors(tmp_path):
    arts = _arts(tmp_path, {
        "model.safetensors": safetensors_bytes(),
        "pytorch_model.bin": BENIGN_PICKLE,
    })  # fmt: skip
    assert [f.rule_id for f in integrity.check_consistency(arts)] == ["INT-005"]
