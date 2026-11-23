import pickle

from modelscan.static.pickle import pickle_vm


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_benign_dict_has_no_calls():
    res = pickle_vm.run(pickle.dumps({"a": [1, 2, 3], "b": "text"}))
    assert res.error is None
    assert res.pickles == 1
    assert res.calls == []


def test_reduce_records_global_and_call_without_executing():
    for proto in range(0, 6):
        res = pickle_vm.run(pickle.dumps(_Evil(), protocol=proto))
        assert res.error is None
        assert len(res.calls) == 1
        target = res.calls[0].target
        assert target is not None
        assert target.name == "system"
        assert target.module in {"os", "posix", "nt"}


def test_truncated_stream_is_an_error():
    data = pickle.dumps({"a": 1})[:-3]
    res = pickle_vm.run(data)
    assert res.error is not None


def test_dynamic_stack_global_is_flagged():
    # BINGET of a missing memo entry feeds STACK_GLOBAL with unknown operands
    data = b"\x80\x04h\x00h\x01\x93."
    res = pickle_vm.run(data)
    assert res.dynamic
