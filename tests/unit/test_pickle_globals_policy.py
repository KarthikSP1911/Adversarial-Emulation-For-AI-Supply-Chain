import pickle
import subprocess
from collections import OrderedDict

import pytest

from modelscan.models import Severity
from modelscan.static import globals_policy as gp
from modelscan.static import pickle_vm


class _Sys:
    def __reduce__(self):
        import os

        return (os.system, ("echo x",))


class _Sub:
    def __reduce__(self):
        return (subprocess.Popen, (["echo", "x"],))


class _Eval:
    def __reduce__(self):
        return (eval, ("1+1",))


class _Custom:
    pass


@pytest.mark.parametrize("module,name", [("os", "system"), ("posix", "system"),
    ("nt", "system"), ("subprocess", "Popen"), ("__builtin__", "eval"),
    ("builtins", "exec"), ("socket", "socket"), ("os.path", "join")])
def test_denied(module, name):
    assert gp.classify(module, name) is gp.Verdict.DENIED


def test_allowed_and_unknown():
    assert gp.classify("collections", "OrderedDict") is gp.Verdict.ALLOWED
    assert gp.classify("torch._utils", "_rebuild_tensor_v2") is gp.Verdict.ALLOWED
    assert gp.classify("mypkg.mod", "Thing") is gp.Verdict.UNKNOWN


@pytest.mark.parametrize("cls", [_Sys, _Sub, _Eval])
def test_malicious_is_critical(cls):
    res = pickle_vm.run(pickle.dumps(cls()))
    findings = gp.check(res, "x.pkl")
    assert any(f.rule_id == "PKL-001" and f.severity is Severity.CRITICAL for f in findings)


def test_benign_ordered_dict_clean():
    res = pickle_vm.run(pickle.dumps(OrderedDict(a=1)))
    assert gp.check(res, "x.pkl") == []


def test_unknown_called_is_high():
    res = pickle_vm.run(pickle.dumps(_Custom()))
    findings = gp.check(res, "x.pkl")
    assert {f.rule_id for f in findings} <= {"PKL-002", "PKL-003"}
    assert findings
