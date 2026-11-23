"""Pickle-side dynamic code and lambda execution (PKL-020).

Covers pickles that rebuild functions or code objects, or that feed data to a
compiler. Keras ``Lambda`` layers live in other formats and are out of scope.
"""

from __future__ import annotations

from modelscan.models import Finding, Severity

from .globals_policy import INSPECTOR, normalize
from .pickle_vm import VMResult

CODE_GLOBALS: dict[str, frozenset[str]] = {
    "types": frozenset({"FunctionType", "LambdaType", "CodeType", "CellType"}),
    "marshal": frozenset({"*"}),
    "builtins": frozenset({"compile", "exec", "eval", "execfile"}),
    "dill._dill": frozenset({
        "_create_function", "_create_code", "_create_lambda", "_create_cell",
        "_load_type", "_create_type",
    }),
    "dill": frozenset({"loads", "load"}),
    "cloudpickle.cloudpickle": frozenset({
        "_make_function", "_make_skeleton_class", "_make_skeleton_enum",
        "_builtin_type", "_make_cell", "_function_setstate",
    }),
    "cloudpickle": frozenset({"loads", "load"}),
    "joblib": frozenset({"load"}),
    "code": frozenset({"*"}),
}  # fmt: skip


def is_code_global(module: str, name: str) -> bool:
    module, name = normalize(module, name)
    if "<lambda>" in name or "<lambda>" in module:
        return True
    names = CODE_GLOBALS.get(module)
    return bool(names and ("*" in names or name in names))


def check(result: VMResult, file: str) -> list[Finding]:
    findings: list[Finding] = []
    called = {
        (c.target.module, c.target.name)
        for c in result.calls
        if c.target is not None and c.target.resolved
    }
    seen: set[tuple[str, str]] = set()
    for ref in result.globals:
        if not ref.resolved or (ref.module, ref.name) in seen:
            continue
        if not is_code_global(ref.module, ref.name):
            continue
        seen.add((ref.module, ref.name))
        invoked = (ref.module, ref.name) in called
        label = f"{ref.module}.{ref.name}"
        findings.append(
            Finding(
                rule_id="PKL-020",
                severity=Severity.CRITICAL,
                inspector=INSPECTOR,
                file=file,
                message=f"Dynamic code construction/execution via {label}",
                offset=ref.offset,
                evidence={"module": ref.module, "name": ref.name, "called": invoked},
            )
        )
    return findings
