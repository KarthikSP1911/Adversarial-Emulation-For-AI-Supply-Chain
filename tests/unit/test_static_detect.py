import zipfile

from static_fixtures import BENIGN_PICKLE, evil_pickle, safetensors_bytes, torch_zip, write

from modelscan.models import Severity
from modelscan.static import detect
from modelscan.static.base import Artifact, Kind


def _art(p):
    return Artifact(path=p, rel=p.name, size=p.stat().st_size)


def test_sniffs_by_content(tmp_path):
    cases = {
        "a.safetensors": (safetensors_bytes(), Kind.SAFETENSORS),
        "a.pkl": (BENIGN_PICKLE, Kind.PICKLE),
        "a.gguf": (b"GGUF" + b"\x00" * 20, Kind.GGUF),
        "a.npy": (b"\x93NUMPY\x01\x00" + b"\x00" * 10, Kind.NUMPY),
        "a.json": (b'{"x": 1}', Kind.JSON),
        "a.txt": (b"Note: hello world\n", Kind.TEXT),
        "empty.txt": (b"", Kind.EMPTY),
    }
    for name, (data, kind) in cases.items():
        p = write(tmp_path / name, data)
        assert detect.sniff(p, len(data)) == kind, name


def test_torch_zip_detected(tmp_path):
    p = torch_zip(tmp_path / "m.pt", BENIGN_PICKLE)
    assert detect.sniff(p, p.stat().st_size) == Kind.TORCH_ZIP


def test_protocol0_pickle_detected():
    head = b"cos\nsystem\n(S'touch /tmp/x'\ntR."
    assert detect._looks_like_pickle(head)


def test_pickle_disguised_as_safetensors_is_critical(tmp_path):
    art = _art(write(tmp_path / "model.safetensors", evil_pickle()))
    (f,) = detect.check(art)
    assert art.kind == Kind.PICKLE
    assert f.rule_id == "DET-001" and f.severity == Severity.CRITICAL


def test_bin_extension_is_not_a_claim(tmp_path):
    art = _art(write(tmp_path / "pytorch_model.bin", BENIGN_PICKLE))
    assert detect.check(art) == []
    assert art.kind == Kind.PICKLE


def test_zip_appended_to_safetensors_is_polyglot(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("evil.pkl", evil_pickle())
    art = _art(write(tmp_path / "model.safetensors", safetensors_bytes() + z.read_bytes()))
    assert "DET-002" in {f.rule_id for f in detect.check(art)}


def test_unknown_binary_reported(tmp_path):
    art = _art(write(tmp_path / "weights.dat", bytes(range(256))))
    (f,) = detect.check(art)
    assert f.rule_id == "DET-003" and art.kind == Kind.UNKNOWN
