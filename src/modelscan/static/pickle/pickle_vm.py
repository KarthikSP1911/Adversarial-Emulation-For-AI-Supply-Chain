"""Safe symbolic pickle VM.

Walks the opcode stream with ``pickletools.genops`` and simulates the stack
and memo using *symbolic* values. Nothing is ever imported, unpickled or
executed. The VM only records what the pickle *would* do (global lookups,
calls, string constants) so that policy modules can judge it.
"""

from __future__ import annotations

import io
import pickletools
from dataclasses import dataclass, field

MAX_OPS = 2_000_000
MAX_STACK = 100_000
MAX_PICKLES = 16


# --- symbolic values ------------------------------------------------------


class Sym:
    """Base class for symbolic stack values."""

    __slots__ = ()


class Const(Sym):
    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value


class Glob(Sym):
    """A global lookup. ``resolved`` is False when module/name were computed."""

    __slots__ = ("module", "name", "resolved")

    def __init__(self, module: str, name: str, resolved: bool = True) -> None:
        self.module = module
        self.name = name
        self.resolved = resolved


class Container(Sym):
    __slots__ = ("kind", "items")

    def __init__(self, kind: str, items: list[Sym] | None = None) -> None:
        self.kind = kind
        self.items = items if items is not None else []


class Result(Sym):
    """Value produced by calling something (never actually evaluated)."""

    __slots__ = ("call",)

    def __init__(self, call: CallEvent) -> None:
        self.call = call


class Unknown(Sym):
    __slots__ = ()


# --- events ---------------------------------------------------------------


@dataclass
class GlobalRef:
    module: str
    name: str
    offset: int
    resolved: bool = True


@dataclass
class CallEvent:
    callee: Sym
    args: list[Sym]
    offset: int
    opcode: str

    @property
    def target(self) -> Glob | None:
        return self.callee if isinstance(self.callee, Glob) else None


@dataclass
class StringConst:
    value: str
    offset: int


@dataclass
class VMResult:
    globals: list[GlobalRef] = field(default_factory=list)
    calls: list[CallEvent] = field(default_factory=list)
    strings: list[StringConst] = field(default_factory=list)
    dynamic: list[GlobalRef] = field(default_factory=list)
    ops: int = 0
    pickles: int = 0
    error: str | None = None
    error_offset: int | None = None
    trailing_bytes: int = 0


# --- helpers --------------------------------------------------------------


def iter_consts(value: Sym, _depth: int = 0):
    """Yield every constant reachable inside a symbolic value."""
    if _depth > 8:
        return
    if isinstance(value, Const):
        yield value.value
    elif isinstance(value, Container):
        for item in value.items:
            yield from iter_consts(item, _depth + 1)
    elif isinstance(value, Result):
        for arg in value.call.args:
            yield from iter_consts(arg, _depth + 1)


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return None


# --- the VM ---------------------------------------------------------------

_SIMPLE_CONSTS = {
    "NONE", "NEWTRUE", "NEWFALSE", "INT", "BININT", "BININT1", "BININT2",
    "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
}  # fmt: skip
_STRING_OPS = {
    "STRING", "BINSTRING", "SHORT_BINSTRING", "BINBYTES", "SHORT_BINBYTES",
    "BINBYTES8", "BYTEARRAY8", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE",
    "BINUNICODE8",
}  # fmt: skip
_EMPTY = {
    "EMPTY_LIST": "list", "EMPTY_DICT": "dict",
    "EMPTY_TUPLE": "tuple", "EMPTY_SET": "set",
}  # fmt: skip
_FROM_MARK = {"LIST": "list", "TUPLE": "tuple", "FROZENSET": "frozenset"}
_PUT = {"PUT", "BINPUT", "LONG_BINPUT"}
_GET = {"GET", "BINGET", "LONG_BINGET"}
_NOOP = {"PROTO", "FRAME", "NEXT_BUFFER", "READONLY_BUFFER"}


class _Run:
    def __init__(self, result: VMResult) -> None:
        self.r = result
        self.stack: list[Sym] = []
        self.metastack: list[list[Sym]] = []
        self.memo: dict[object, Sym] = {}

    def pop(self) -> Sym:
        return self.stack.pop() if self.stack else Unknown()

    def popn(self, n: int) -> list[Sym]:
        items = [self.pop() for _ in range(n)]
        items.reverse()
        return items

    def pop_mark(self) -> list[Sym]:
        items = self.stack
        self.stack = self.metastack.pop() if self.metastack else []
        return items

    def call(self, callee: Sym, args: list[Sym], offset: int, opcode: str) -> None:
        event = CallEvent(callee, args, offset, opcode)
        self.r.calls.append(event)
        self.stack.append(Result(event))

    def glob(self, module: str, name: str, offset: int, resolved: bool = True) -> Glob:
        ref = GlobalRef(module, name, offset, resolved)
        self.r.globals.append(ref)
        if not resolved:
            self.r.dynamic.append(ref)
        return Glob(module, name, resolved)

    def step(self, name: str, arg: object, pos: int) -> None:
        s = self.stack
        if name in _NOOP:
            return
        if name in _SIMPLE_CONSTS:
            s.append(Const(arg))
        elif name in _STRING_OPS:
            s.append(Const(arg))
            text = _text(arg)
            if text:
                self.r.strings.append(StringConst(text, pos))
        elif name in _EMPTY:
            s.append(Container(_EMPTY[name]))
        elif name == "MARK":
            self.metastack.append(s)
            self.stack = []
        elif name in _FROM_MARK:
            items = self.pop_mark()
            self.stack.append(Container(_FROM_MARK[name], items))
        elif name == "DICT":
            self.stack.append(Container("dict", self.pop_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            s.append(Container("tuple", self.popn(int(name[-1]))))
        elif name in ("APPEND", "SETITEM"):
            self.popn(2 if name == "SETITEM" else 1)
        elif name in ("APPENDS", "SETITEMS", "ADDITEMS"):
            items = self.pop_mark()
            if self.stack and isinstance(self.stack[-1], Container):
                self.stack[-1].items.extend(items)
        elif name == "POP":
            self.pop()
        elif name == "POP_MARK":
            self.pop_mark()
        elif name == "DUP":
            s.append(s[-1] if s else Unknown())
        elif name in _PUT:
            self.memo[arg] = s[-1] if s else Unknown()
        elif name == "MEMOIZE":
            self.memo[len(self.memo)] = s[-1] if s else Unknown()
        elif name in _GET:
            s.append(self.memo.get(arg, Unknown()))
        elif name == "GLOBAL":
            module, _, attr = str(arg).partition(" ")
            s.append(self.glob(module, attr, pos))
        elif name == "STACK_GLOBAL":
            attr, module = self.pop(), self.pop()
            mod_v = module.value if isinstance(module, Const) else None
            attr_v = attr.value if isinstance(attr, Const) else None
            if isinstance(mod_v, str) and isinstance(attr_v, str):
                s.append(self.glob(mod_v, attr_v, pos))
            else:
                s.append(self.glob("<dynamic>", "<dynamic>", pos, resolved=False))
        elif name == "INST":
            module, _, attr = str(arg).partition(" ")
            args = self.pop_mark()
            self.call(self.glob(module, attr, pos), args, pos, name)
        elif name == "OBJ":
            items = self.pop_mark()
            callee = items[0] if items else Unknown()
            self.call(callee, items[1:], pos, name)
        elif name == "NEWOBJ":
            args, cls = self.pop(), self.pop()
            self.call(cls, [args], pos, name)
        elif name == "NEWOBJ_EX":
            kwargs, args, cls = self.pop(), self.pop(), self.pop()
            self.call(cls, [args, kwargs], pos, name)
        elif name == "REDUCE":
            args, func = self.pop(), self.pop()
            self.call(func, [args], pos, name)
        elif name == "BUILD":
            self.pop()  # state; the object stays on the stack
        elif name in ("PERSID", "BINPERSID"):
            if name == "BINPERSID":
                self.pop()
            s.append(Unknown())
        elif name in ("EXT1", "EXT2", "EXT4"):
            s.append(self.glob("<extension>", str(arg), pos, resolved=False))
        elif name == "STOP":
            return
        else:  # unmodelled opcode: stay conservative, keep the stack sane
            s.append(Unknown())
        if len(self.stack) > MAX_STACK:
            raise ValueError("stack limit exceeded")


def run(data: bytes) -> VMResult:
    """Symbolically execute every pickle found at the start of ``data``."""
    result = VMResult()
    stream = io.BytesIO(data)
    while result.pickles < MAX_PICKLES:
        start = stream.tell()
        if start >= len(data):
            break
        vm = _Run(result)
        try:
            for opcode, arg, pos in pickletools.genops(stream):
                result.ops += 1
                if result.ops > MAX_OPS:
                    raise ValueError("opcode limit exceeded")
                vm.step(opcode.name, arg, pos if pos is not None else stream.tell())
        except Exception as exc:  # noqa: BLE001 - any parse failure is reported
            if result.pickles == 0:
                result.error = f"{type(exc).__name__}: {exc}"
                result.error_offset = stream.tell()
            else:
                result.trailing_bytes = len(data) - start
            break
        result.pickles += 1
    else:
        result.trailing_bytes = max(len(data) - stream.tell(), 0)
    return result
