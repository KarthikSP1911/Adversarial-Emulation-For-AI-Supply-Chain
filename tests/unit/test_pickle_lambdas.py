import pickle

from modelscan.models import Severity
from modelscan.static import lambdas, pickle_vm


class _Compile:
    def __reduce__(self):
        return (compile, ("print(1)", "<s>", "exec"))


def _check(data):
    return lambdas.check(pickle_vm.run(data), "x.pkl")


def test_benign_is_clean():
    assert _check(pickle.dumps([1, "a", {"k": 2.5}])) == []


def test_compile_reduce_is_critical():
    findings = _check(pickle.dumps(_Compile()))
    assert [f.rule_id for f in findings] == ["PKL-020"]
    assert findings[0].severity is Severity.CRITICAL
    assert findings[0].evidence["called"] is True


def test_function_type_global():
    data = b"ctypes\nFunctionType\n."
    findings = _check(data)
    assert findings and findings[0].evidence["called"] is False


def test_lambda_name_and_marshal():
    assert _check(b"c__main__\n<lambda>\n.")
    assert _check(b"cmarshal\nloads\n.")


def test_dill_function_constructor():
    assert _check(b"cdill._dill\n_create_function\n.")
