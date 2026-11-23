"""Policy for global references found in a pickle (PKL-001/002/003)."""

from __future__ import annotations

from enum import Enum

from modelscan.models import Finding, Severity

from .pickle_vm import CallEvent, GlobalRef, VMResult

INSPECTOR = "static.pickle"

_ALIASES = {
    "posix": "os", "nt": "os", "os.path": "os",
    "__builtin__": "builtins", "__builtins__": "builtins",
    "_posixsubprocess": "subprocess", "_winapi": "subprocess",
}  # fmt: skip


class Verdict(str, Enum):
    DENIED = "denied"
    ALLOWED = "allowed"
    UNKNOWN = "unknown"


# Whole modules that must never be referenced from a model file.
DENIED_MODULES = frozenset({
    "subprocess", "runpy", "socket", "pty", "shutil", "importlib", "ctypes",
    "webbrowser", "sys", "marshal", "code", "codeop", "commands", "pdb",
    "multiprocessing", "asyncio", "threading", "signal", "pickle", "_pickle",
    "http", "urllib", "ftplib", "smtplib", "telnetlib", "requests", "httplib",
    "pip", "platform", "tempfile", "zipimport", "imp", "bdb", "timeit",
})  # fmt: skip

# Specific dangerous names inside otherwise common modules.
DENIED_NAMES: dict[str, frozenset[str]] = {
    "os": frozenset({"*"}),
    "builtins": frozenset({
        "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
        "getattr", "setattr", "delattr", "globals", "locals", "vars", "execfile",
    }),
    "types": frozenset({"FunctionType", "LambdaType", "CodeType", "MethodType"}),
    "operator": frozenset({"attrgetter", "methodcaller"}),
    "functools": frozenset({"partial"}),
    "base64": frozenset({"b64decode", "decodebytes", "a85decode", "b85decode"}),
    "codecs": frozenset({"decode", "open"}),
    "io": frozenset({"open", "FileIO"}),
}  # fmt: skip

# Globals that legitimate ML checkpoints use.
ALLOWED: dict[str, frozenset[str]] = {
    "collections": frozenset({"OrderedDict", "defaultdict", "deque"}),
    "torch": frozenset({
        "FloatStorage", "HalfStorage", "DoubleStorage", "LongStorage",
        "IntStorage", "ShortStorage", "CharStorage", "ByteStorage",
        "BoolStorage", "BFloat16Storage", "Size", "device", "Tensor",
        "float16", "float32", "float64", "bfloat16", "int8", "int16",
        "int32", "int64", "uint8", "bool",
    }),
    "torch._utils": frozenset({
        "_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter",
        "_rebuild_qtensor", "_rebuild_device_tensor_from_numpy",
    }),
    "torch.storage": frozenset({"_load_from_bytes", "UntypedStorage", "TypedStorage"}),
    "torch.nn.parameter": frozenset({"Parameter"}),
    "numpy": frozenset({"dtype", "ndarray"}),
    "numpy.core.multiarray": frozenset({"_reconstruct", "scalar"}),
    "numpy._core.multiarray": frozenset({"_reconstruct", "scalar"}),
    "numpy.core.numeric": frozenset({"_frombuffer"}),
    "numpy._core.numeric": frozenset({"_frombuffer"}),
    "_codecs": frozenset({"encode"}),
    "copy_reg": frozenset({"_reconstructor"}),
    "copyreg": frozenset({"_reconstructor"}),
}  # fmt: skip


def normalize(module: str, name: str) -> tuple[str, str]:
    module = _ALIASES.get(module, module)
    root = module.split(".", 1)[0]
    if root in _ALIASES and module != root:
        module = _ALIASES[root] + module[len(root) :]
    return module, name


def classify(module: str, name: str) -> Verdict:
    module, name = normalize(module, name)
    root = module.split(".", 1)[0]
    if module in DENIED_MODULES or root in DENIED_MODULES:
        return Verdict.DENIED
    names = DENIED_NAMES.get(module)
    if names and ("*" in names or name in names):
        return Verdict.DENIED
    allowed = ALLOWED.get(module)
    if allowed and name in allowed:
        return Verdict.ALLOWED
    return Verdict.UNKNOWN


def check(result: VMResult, file: str) -> list[Finding]:
    """Turn VM global references and calls into findings."""
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()

    def emit(rule: str, sev: Severity, ref: GlobalRef, msg: str, **ev: object) -> None:
        key = (rule, ref.module, ref.name)
        if key in seen:
            return
        seen.add(key)
        evidence = {"module": ref.module, "name": ref.name, **ev}
        findings.append(
            Finding(
                rule_id=rule, severity=sev, inspector=INSPECTOR,
                file=file, message=msg, offset=ref.offset, evidence=evidence,
            )
        )  # fmt: skip

    called: dict[tuple[str, str], CallEvent] = {}
    for call in result.calls:
        if call.target is not None and call.target.resolved:
            called.setdefault((call.target.module, call.target.name), call)

    for ref in result.globals:
        if not ref.resolved:
            continue  # handled by strings.py (PKL-011)
        verdict = classify(ref.module, ref.name)
        if verdict is Verdict.ALLOWED:
            continue
        was_called = (ref.module, ref.name) in called
        label = f"{ref.module}.{ref.name}"
        if verdict is Verdict.DENIED:
            emit("PKL-001", Severity.CRITICAL, ref,
                 f"Dangerous global {label} referenced in pickle",
                 called=was_called)  # fmt: skip
        elif was_called:
            emit("PKL-002", Severity.HIGH, ref,
                 f"Unknown global {label} is called during unpickling")  # fmt: skip
        else:
            emit("PKL-003", Severity.MEDIUM, ref,
                 f"Unknown global {label} referenced in pickle")  # fmt: skip
    return findings
