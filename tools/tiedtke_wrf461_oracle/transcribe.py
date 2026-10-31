"""Reproduce the cu6 CUDA routines from the SHA-pinned WRF source.

This is a deliberately narrow Fortran reader. Unsupported syntax fails with
the source line. It emits each statement in order and pins each real binary
operation. The WRF driver boundary is maintained separately in tiedtke.cu.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

SOURCE_SHA256 = "3514aaaaf6084edc0f6a6b2097e34607562f0b6f12934cc3c675a9e651566999"
SOURCE_URL = "https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/phys/module_cu_tiedtke.F"


def split(text, delimiter=","):
    depth = 0
    start = 0
    parts = []
    for i, c in enumerate(text):
        depth += (c == "(") - (c == ")")
        if c == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def statements(source):
    pending = ""
    start = 0
    alternate = False
    for line, raw in enumerate(source.splitlines(), 1):
        text = raw.split("!", 1)[0].strip().lower()
        if text == "#ifndef aarch64_x86_correctness_fix":
            continue
        if text == "#else":
            alternate = True
            continue
        if text == "#endif":
            alternate = False
            continue
        if alternate:
            continue
        if not text:
            continue
        if not pending:
            start = line
        pending += " " + text.lstrip("&").rstrip("&").strip()
        if text.endswith("&"):
            continue
        yield start, pending.strip()
        pending = ""
    if pending:
        raise ValueError("unterminated continuation")


@dataclass
class Variable:
    kind: str
    dimensions: tuple[str, ...] = ()


@dataclass
class Routine:
    line: int
    name: str
    args: list[str]
    result: str
    variables: dict[str, Variable]
    body: list[tuple[int, str]]
    writes: set[str]


def read_routines(source):
    routines = {}
    current = None
    for line, s in statements(source):
        m = re.fullmatch(r"(subroutine|real function)\s+(\w+)\s*\((.*)\)", s)
        if m:
            current = Routine(line, m[2], split(m[3]),
                              "float" if m[1] == "real function" else "void",
                              {}, [], set())
            if current.name not in ("cu_tiedtke", "tiedtkeinit"):
                routines[current.name] = current
            continue
        if not current or current.name in ("cu_tiedtke", "tiedtkeinit"):
            continue
        if s.startswith("end subroutine") or s.startswith("end function"):
            current = None
            continue
        if s == "implicit none":
            continue
        m = re.match(r"^(real|integer|logical)\s+(.+)$", s)
        if m:
            kind = {"real": "float", "integer": "int", "logical": "bool"}[m[1]]
            for declaration in split(m[2]):
                v = re.fullmatch(r"(\w+)(?:\((.*)\))?", declaration)
                if not v:
                    raise ValueError(f"declaration at {line}: {s}")
                current.variables[v[1]] = Variable(kind, tuple(split(v[2])) if v[2] else ())
            continue
        current.body.append((line, s))
        m = re.match(r"^(\w+)\s*=", s)
        if m:
            current.writes.add(m[1])
        m = re.match(r"^do\s+(\w+)\s*=", s)
        if m:
            current.writes.add(m[1])
        # A scalar assignment can also be under a single-line IF.
        if s.startswith("if"):
            end = close_paren(s, s.index("("))
            m = re.match(r"\s*(\w+)\s*=", s[end + 1:])
            if m:
                current.writes.add(m[1])
    for routine in routines.values():
        missing = set(routine.args) - set(routine.variables)
        if missing:
            raise ValueError(f"undeclared arguments in {routine.name}: {missing}")
    # Fortran scalar arguments retain changes made by child routines too.
    changed = True
    while changed:
        changed = False
        for routine in routines.values():
            for _, s in routine.body:
                m = re.fullmatch(r"call (\w+)\s*\((.*)\)", s)
                if not m:
                    continue
                child = routines[m[1]]
                for actual, formal in zip(split(m[2]), child.args, strict=True):
                    if (formal in child.writes and actual in routine.variables
                            and not routine.variables[actual].dimensions
                            and actual not in routine.writes):
                        routine.writes.add(actual)
                        changed = True
    return routines


def close_paren(s, start):
    depth = 0
    for i in range(start, len(s)):
        depth += (s[i] == "(") - (s[i] == ")")
        if depth == 0:
            return i
    raise ValueError(f"unclosed parenthesis: {s}")


TOKEN = re.compile(
    r"\s*(\.true\.|\.false\.|\.(?:and|or|not|eq|ne|gt|ge|lt|le)\.|"
    r"(?:\d+\.(?!(?:and|or|not|eq|ne|gt|ge|lt|le)\.)\d*|\.\d+|\d+)(?:[ed][+-]?\d+)?|"
    r"[a-z_]\w*|\*\*|==|/=|>=|<=|[()+*/=<>,-])")
PRECEDENCE = {".or.": 1, ".and.": 2,
              ".eq.": 3, ".ne.": 3, ".gt.": 3, ".ge.": 3,
              ".lt.": 3, ".le.": 3, "==": 3, "/=": 3,
              ">": 3, ">=": 3, "<": 3, "<=": 3,
              "+": 4, "-": 4, "*": 5, "/": 5, "**": 7}


class Expression:
    def __init__(self, text, variables):
        text = text.strip()
        self.variables = variables
        self.tokens = []
        pos = 0
        while pos < len(text):
            m = TOKEN.match(text, pos)
            if m is None:
                raise ValueError(f"expression token at {text[pos:]!r}: {text}")
            self.tokens.append(m[1])
            pos = m.end()
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ""

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self, minimum=0):
        token = self.take()
        if token in ("-", "+", ".not."):
            node = ("unary", token, self.parse(3 if token == ".not." else 6))
        elif token == "(":
            node = self.parse()
            if self.take() != ")":
                raise ValueError("expected closing parenthesis")
        elif token in (".true.", ".false.") or re.match(r"[\d.]", token):
            node = ("literal", token)
        elif re.fullmatch(r"[a-z_]\w*", token):
            if self.peek() == "(":
                self.take()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.parse())
                        if self.peek() != ",":
                            break
                        self.take()
                if self.take() != ")":
                    raise ValueError("expected closing call parenthesis")
                node = ("call", token, args)
            else:
                node = ("name", token)
        else:
            raise ValueError(f"unexpected expression token: {token!r}")
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()] >= minimum:
            op = self.take()
            prec = PRECEDENCE[op]
            rhs = self.parse(prec if op == "**" else prec + 1)
            node = ("binary", op, node, rhs)
        return node

    def all(self):
        node = self.parse()
        if self.pos != len(self.tokens):
            raise ValueError(f"unconsumed expression: {self.tokens[self.pos:]}")
        return node


def kind(node, variables):
    tag, name = node[:2]
    if tag == "literal":
        return "bool" if name in (".true.", ".false.") else (
            "float" if any(c in name for c in ".ed") else "int")
    if tag == "name":
        return variables[name].kind
    if tag == "unary":
        return "bool" if name == ".not." else kind(node[2], variables)
    if tag == "call":
        if name in variables:
            return variables[name].kind
        if name == "int":
            return "int"
        if name in ("min", "max", "abs"):
            return kind(node[2][0], variables)
        return "float"
    if name in (".and.", ".or.", ".eq.", ".ne.", ".gt.", ".ge.", ".lt.", ".le.",
                "==", "/=", ">", ">=", "<", "<="):
        return "bool"
    return "float" if "float" in (kind(node[2], variables), kind(node[3], variables)) else "int"


def emit(node, variables):
    tag, name = node[:2]
    if tag == "literal":
        if name in (".true.", ".false."):
            return "true" if name == ".true." else "false"
        if kind(node, variables) == "float":
            return name.replace("d", "e") + (".0f" if "." not in name and "e" not in name else "f")
        return name
    if tag == "name":
        return "tk_" + name if name in CONSTANTS else ("result" if variables.get(name) == Variable("result") else name)
    if tag == "unary":
        op = "!" if name == ".not." else name
        return f"({op}{emit(node[2], variables)})"
    if tag == "call":
        args = [emit(n, variables) for n in node[2]]
        if name in variables:
            return f"{name}({', '.join(args)})"
        if name in ("min", "max", "amin1", "amax1"):
            op = "tk_min" if name in ("min", "amin1") else "tk_max"
            answer = args[0]
            for arg in args[1:]:
                answer = f"{op}({answer}, {arg})"
            return answer
        if name == "ssum":
            base = node[2][1]
            if base[0] != "call" or base[1] not in variables:
                raise ValueError("SSUM needs an array section start")
            args[1] = f"{base[1]}.slice({emit(base[2][0], variables)})"
        if name in ("ssum", "tlucua", "tlucub", "tlucuc", "tlucua_tiecnv", "tan"):
            args.append("arena")
        fn = {"sqrt": "sqrtf", "exp": "gfk_exp", "log": "gfk_log", "tan": "tk_tan", "abs": "tk_abs", "int": "int"}.get(name, "tk_" + name)
        return f"{fn}({', '.join(args)})"
    lhs, rhs = emit(node[2], variables), emit(node[3], variables)
    if name == "**":
        exponent = node[3]
        if exponent[0] == "literal":
            value = float(exponent[1])
            integer = kind(exponent, variables) == "int"
            if value == 1:
                return lhs
            if value == 2:
                return f"tk_mul({lhs}, {lhs})"
            if integer and value > 0 and int(value) == value:
                return f"tk_powi<{int(value)}>({lhs})"
        if exponent == ("unary", "-", ("literal", "1.0")):
            return f"tk_div(1.0f, {lhs})"
        return f"gfk_pow({lhs}, {rhs})"
    if name in ("+", "-", "*", "/") and kind(node, variables) == "float":
        op = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[name]
        return f"tk_{op}({lhs}, {rhs})"
    op = {".or.": "|", ".and.": "&", ".eq.": "==", ".ne.": "!=", ".gt.": ">", ".ge.": ">=",
          ".lt.": "<", ".le.": "<=", "/=": "!="}.get(name, name)
    return f"({lhs} {op} {rhs})"


CONSTANTS = {"rd": "287.0", "rv": "461.6", "cpd": "7.*rd/2.",
             "alv": "2.5e6", "als": "2.85e6", "alf": "3.50e5", "g": "9.81"}


def constant_value(node, values):
    import numpy as np
    tag, name = node[:2]
    if tag == "literal":
        if name in (".true.", ".false."):
            return name == ".true."
        return np.float32(name) if any(c in name for c in ".ed") else int(name)
    if tag == "name":
        return values[name]
    if tag == "unary":
        v = constant_value(node[2], values)
        return -v if name == "-" else v
    a, b = constant_value(node[2], values), constant_value(node[3], values)
    return {"+": lambda: a + b, "-": lambda: a - b,
            "*": lambda: a * b, "/": lambda: a / b}[name]()


def signature(routine):
    args = []
    for name in routine.args:
        v = routine.variables[name]
        args.append(f"TkArray<{v.kind}> {name}" if v.dimensions else
                    f"{v.kind}{'&' if name in routine.writes else ''} {name}")
    args.append("TkArena arena")
    return f"__device__ {routine.result} tk_{routine.name}({', '.join(args)})"


def render_routine(routine, constants, routines):
    variables = {**constants, **routine.variables}
    if routine.result != "void":
        variables[routine.name] = Variable("float")
    def expr(text):
        return emit(Expression(text, variables).all(), variables)
    lines = [signature(routine) + " {", f"    // module_cu_tiedtke.F:{routine.line}"]
    if routine.result != "void":
        lines.append("    float result;")
    for name, v in routine.variables.items():
        if name in routine.args:
            continue
        if v.dimensions:
            dimensions = [expr(s) for s in v.dimensions]
            size = " * ".join(f"({s})" for s in dimensions)
            first = dimensions[0]
            lines.append(f"    TkArray<{v.kind}> {name} = arena.take<{v.kind}>({size}, {first});")
        else:
            lines.append(f"    {v.kind} {name};")
    depth = 1
    blocks = []
    def statement(s):
        nonlocal depth
        if routine.name == "tiecnv" and s == VECTOR_QSAT:
            # F:588 is the one exp in a loop the stock build vectorizes;
            # its column's lane decides libmvec or the scalar expf.
            s = s.replace("tlucua(", "tlucua_tiecnv(")
        if routine.name == "cumastr_new" and s == "if(icum.eq.0) return":
            return ("*arena.activity = icum; "
                    "if (icum == 0 && !arena.force_continue) { return; }")
        if s in ("end do", "enddo", "end if", "endif"):
            depth -= 1
            expected = "do" if s in ("end do", "enddo") else "if"
            if not blocks or blocks.pop() != expected:
                raise ValueError(f"unbalanced {s}")
            return "}"
        if s in ("else",) or re.match(r"else\s*if", s):
            depth -= 1
            if not blocks or blocks[-1] != "if":
                raise ValueError("ELSE without IF")
            if s == "else":
                return "} else {"
            begin = s.index("(")
            end = close_paren(s, begin)
            return f"}} else if ({expr(s[begin + 1:end])}) {{"
        if re.match(r"^if\s*\(", s):
            begin = s.index("(")
            end = close_paren(s, begin)
            condition = expr(s[begin + 1:end])
            tail = s[end + 1:].strip()
            if tail == "then":
                blocks.append("if")
                return f"if ({condition}) {{"
            return f"if ({condition}) {{ {statement(tail)} }}"
        m = re.fullmatch(r"do\s+(\w+)\s*=\s*(.*)", s)
        if m:
            bounds = split(m[2])
            if len(bounds) not in (2, 3):
                raise ValueError("DO bounds")
            lo, hi = map(expr, bounds[:2])
            step = expr(bounds[2]) if len(bounds) == 3 else "1"
            compare = ">=" if len(bounds) == 3 and bounds[2].startswith("-") else "<="
            blocks.append("do")
            # Pin the terminal bound at entry, as Fortran does.
            return (f"const int tk_limit_{line} = {hi};\n" + "    " * depth +
                    f"for ({m[1]} = {lo}; {m[1]} {compare} tk_limit_{line}; {m[1]} += {step}) {{")
        if s == "return":
            return "return;" if routine.result == "void" else "return result;"
        if s == "cycle":
            return "continue;"
        if s.startswith("call "):
            m = re.fullmatch(r"call (\w+)\s*\((.*)\)", s)
            if not m or m[1] not in routines:
                raise ValueError(f"unknown call {s}")
            return f"tk_{m[1]}({', '.join(expr(a) for a in split(m[2]))}, arena);"
        parts = split(s, "=")
        if len(parts) == 2:
            lhs = "result" if parts[0] == routine.name else expr(parts[0])
            traced = (parts[0] == routine.name or
                      kind(Expression(parts[0], variables).all(), variables) == "float")
            return f"{lhs} = {expr(parts[1])};" + (f" TK_TRACE({line}, {lhs});" if traced else "")
        raise ValueError(f"unsupported statement {s}")
    for line, s in routine.body:
        try:
            before = depth
            text = statement(s)
            lines.append("    " * depth + f"// F:{line}")
            lines.append("    " * depth + text)
            if s.startswith("do ") or (re.match(r"if\s*\(", s) and s.endswith("then")) or text.startswith("} else"):
                depth += 1
        except Exception as exc:
            raise ValueError(f"{routine.name} at source line {line}: {s}: {exc}") from exc
    if blocks or depth != 1:
        raise ValueError(f"unbalanced routine: {routine.name}")
    lines.append("}")
    return "\n".join(lines)


def transcribe(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("WRF cu6 source hash differs from v4.6.1")
    source = raw.decode("utf-8")
    constants = dict(CONSTANTS)
    for line, s in statements(source):
        if s == "contains":
            break
        if s.startswith("parameter("):
            for p in split(s[len("parameter("):-1]):
                name, value = p.split("=", 1)
                constants[name.strip()] = value.strip()
    CONSTANTS.update(constants)
    variables = {n: Variable("bool" if s in (".true.", ".false.") else "float") for n, s in constants.items()}
    values = {}
    lines = ["// Generated by tools/tiedtke_wrf461_oracle/transcribe.py.",
             f"// WRF v4.6.1 phys/module_cu_tiedtke.F SHA256: {SOURCE_SHA256}",
             "// Source line numbers refer to that exact file."]
    for name, text in constants.items():
        value = constant_value(Expression(text, variables).all(), values)
        values[name] = value
        literal = ("true" if value else "false") if isinstance(value, bool) else float(value).hex() + "f"
        lines.append(f"static constexpr {variables[name].kind} tk_{name} = {literal};")
    routines = read_routines(source)
    for routine in routines.values():
        lines.append(signature(routine) + ";")
    lines.append(lane_twin(signature(routines["tlucua"]) + ";"))
    for routine in routines.values():
        lines.append(render_routine(routine, variables, routines))
    lines.append(lane_twin(render_routine(routines["tlucua"], variables, routines)))
    return "\n\n".join(lines) + "\n"


#: module_cu_tiedtke.F:588 as statements() spells it.
VECTOR_QSAT = "zqsat(j,k)=tlucua(tt)/pap(j,k)"


def lane_twin(text):
    """tlucua as tiecnv's conversion loop runs it (F:575-593).

    gfortran 15.2 at WRF's stock -O2 -ftree-vectorize inlines tlucua there
    and evaluates the loop four columns at a time, binding its exp to
    glibc's libmvec _ZGVbN4v_expf; the last (ite-its+1) mod 4 columns of
    the row run the scalar loop and its expf.  Every other tlucua call
    (cuadjtq) stays scalar in that build.  The twin differs from tlucua in
    its name and its exp only; tk_lane_exp picks per column.
    """
    if text.count("tk_tlucua(") != 1 or text.count("gfk_exp(") > 1:
        raise ValueError("tlucua's shape changed; re-derive the lane twin")
    return (text.replace("tk_tlucua(", "tk_tlucua_tiecnv(")
                .replace("gfk_exp(", "tk_lane_exp(arena, "))


def workspace_peak(routines, nz):
    """Exact arena extents for the generated single-column call tree."""
    values = {"klon": 1, "lq": 1, "km": nz, "km1": nz + 1,
              "klev": nz, "klevp1": nz + 1, "klevm1": nz - 1}
    types = {n: Variable("int") for n in values}
    def peak(name):
        r = routines[name]
        own = {"float": 0, "int": 0, "bool": 0}
        for n, v in r.variables.items():
            if n in r.args or not v.dimensions:
                continue
            count = 1
            for d in v.dimensions:
                count *= int(constant_value(Expression(d, types).all(), values))
            own[v.kind] += count
        children = []
        for _, s in r.body:
            m = re.fullmatch(r"call (\w+)\s*\((.*)\)", s)
            if m:
                children.append(peak(m[1]))
        return {k: own[k] + max((c[k] for c in children), default=0) for k in own}
    p = peak("tiecnv")
    p["float"] += 15 * nz + 3
    p["int"] += 2
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(transcribe(args.source), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
