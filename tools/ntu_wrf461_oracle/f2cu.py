#!/usr/bin/env python3
"""Mechanical transcription of WRF v4.6.1 phys/module_mp_ntu.F into CUDA C++.

The transcription keeps WRF's operation order exactly: every Fortran binary
operation becomes one C++ binary operation on the same operand types, left to
right, with Fortran's precedence.  The lowering rules are gfortran's:

* REAL ** 2.0 -> x*x, REAL ** 1.0 -> x, REAL ** (-1.0) -> 1/x; every other
  REAL ** REAL is a pow call (the scheme's own WOOF math routine).
* A constant subexpression that contains a power or an intrinsic call is
  folded the way gfortran folds it (correctly rounded, at compile time); the
  folded value is read from a gfortran program built from the same text and
  emitted as a hexadecimal literal.  Pure + - * / constant subexpressions are
  left to the C++ compiler, whose IEEE folding is identical.
* MIN/MAX follow gfortran's NaN-aware left fold: m = a1; m = (ai op m || m!=m)
  ? ai : m.
* Branches on the scheme's compile-time PARAMETER switches are pruned.

Usage:
  f2cu.py SOURCE.F --params-out params.f90           (pass 1: constants program)
  f2cu.py SOURCE.F --params-in params.txt --out X.cuh (pass 2: emit)
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Lexing and logical lines
# --------------------------------------------------------------------------

DEFINES = {"DWORDSIZE": 8, "RWORDSIZE": 4}


def strip_comment(line: str) -> str:
    out = []
    q = None
    for ch in line:
        if q:
            out.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
            out.append(ch)
            continue
        if ch == "!":
            break
        out.append(ch)
    return "".join(out)


def eval_pp(cond: str) -> bool:
    c = cond
    for k, v in DEFINES.items():
        c = re.sub(r"\b%s\b" % k, str(v), c)
    c = c.replace("&&", " and ").replace("||", " or ")
    return bool(eval(c))


def logical_lines(text: str):
    raw = text.splitlines()
    # preprocessor
    keep = []
    stack = []  # (active, taken)
    active = True
    for ln, line in enumerate(raw, 1):
        s = line.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
            if not m:
                continue
            kw, rest = m.group(1), m.group(2)
            if kw in ("if", "ifdef", "ifndef"):
                stack.append((active, False))
                parent = active
                if kw == "if":
                    t = parent and eval_pp(rest)
                else:
                    defined = rest.strip() in DEFINES
                    t = parent and (defined if kw == "ifdef" else not defined)
                stack[-1] = (parent, t)
                active = t
            elif kw == "elif":
                parent, taken = stack[-1]
                t = parent and not taken and eval_pp(rest)
                stack[-1] = (parent, taken or t)
                active = t
            elif kw == "else":
                parent, taken = stack[-1]
                active = parent and not taken
                stack[-1] = (parent, True)
            else:
                parent, _ = stack.pop()
                active = parent
            continue
        if active:
            keep.append((ln, line))
    if stack:
        raise ValueError("unterminated preprocessor conditional")
    # continuation
    joined = []
    buf = ""
    start = None
    for ln, line in keep:
        s = strip_comment(line).rstrip()
        if not s.strip():
            continue
        st = s.strip()
        if buf:
            if st.startswith("&"):
                st = st[1:]
            buf += st if not buf.endswith(" ") else st
        else:
            buf = st
            start = ln
        if buf.endswith("&"):
            buf = buf[:-1]
            continue
        joined.append((start, buf))
        buf = ""
    # split ';'
    out = []
    for ln, s in joined:
        depth = 0
        q = None
        cur = ""
        for ch in s:
            if q:
                cur += ch
                if ch == q:
                    q = None
                continue
            if ch in "'\"":
                q = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth == 0:
                if cur.strip():
                    out.append((ln, cur.strip()))
                cur = ""
                continue
            cur += ch
        if cur.strip():
            out.append((ln, cur.strip()))
    # labels
    res = []
    for ln, s in out:
        m = re.match(r"^(\d+)\s+(.*)$", s)
        if m:
            res.append((ln, int(m.group(1)), m.group(2).strip()))
        else:
            res.append((ln, None, s))
    return res


OPWORDS = r"(?:EQ|NE|LT|LE|GT|GE|AND|OR|NOT|TRUE|FALSE|EQV|NEQV)\."
TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<real>(?:\d+\.(?!" + OPWORDS + r")\d*(?:[EeDd][+-]?\d+)?)|(?:\.\d+(?:[EeDd][+-]?\d+)?)|(?:\d+[EeDd][+-]?\d+))"
    r"|(?P<int>\d+)"
    r"|(?P<dotop>\.(?:EQ|NE|LT|LE|GT|GE|AND|OR|NOT|TRUE|FALSE|EQV|NEQV)\.)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<str>'[^']*'|\"[^\"]*\")"
    r"|(?P<op>\*\*|==|/=|<=|>=|\(/|/\)|[-+*/(),<>=:])"
    r")", re.I)


def tokenize(s: str):
    pos = 0
    toks = []
    s = s.rstrip()
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            if s[pos:].strip() == "":
                break
            raise SyntaxError(f"cannot tokenize at {s[pos:]!r} in {s!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "dotop":
            val = val.upper()
        elif kind == "name":
            val = val.upper()
        toks.append((kind, val))
    return toks


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------

@dataclass
class Num:
    text: str
    kind: str  # 'i','f','d'


@dataclass
class Bool:
    val: bool


@dataclass
class Name:
    name: str


@dataclass
class Ref:  # call or array element
    name: str
    args: list


@dataclass
class Un:
    op: str
    x: object


@dataclass
class Bin:
    op: str
    a: object
    b: object


@dataclass
class Str:
    s: str


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.t[self.i]
        if val is not None and tok[1] != val:
            raise SyntaxError(f"expected {val} got {tok}")
        self.i += 1
        return tok

    def at_end(self):
        return self.i >= len(self.t)

    def expr(self):
        return self.eqv()

    def eqv(self):
        a = self.orx()
        while self.peek()[1] in (".EQV.", ".NEQV."):
            op = self.take()[1]
            a = Bin(op, a, self.orx())
        return a

    def orx(self):
        a = self.andx()
        while self.peek()[1] == ".OR.":
            self.take()
            a = Bin(".OR.", a, self.andx())
        return a

    def andx(self):
        a = self.notx()
        while self.peek()[1] == ".AND.":
            self.take()
            a = Bin(".AND.", a, self.notx())
        return a

    def notx(self):
        if self.peek()[1] == ".NOT.":
            self.take()
            return Un(".NOT.", self.notx())
        return self.rel()

    RELMAP = {".LT.": "<", ".LE.": "<=", ".GT.": ">", ".GE.": ">=",
              ".EQ.": "==", ".NE.": "!=", "<": "<", "<=": "<=", ">": ">",
              ">=": ">=", "==": "==", "/=": "!="}

    def rel(self):
        a = self.add()
        if self.peek()[1] in self.RELMAP:
            op = self.RELMAP[self.take()[1]]
            b = self.add()
            return Bin(op, a, b)
        return a

    def add(self):
        if self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            a = self.mul()
            a = Un(op, a)
        else:
            a = self.mul()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            a = Bin(op, a, self.mul())
        return a

    def mul(self):
        a = self.power()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            a = Bin(op, a, self.power())
        return a

    def power(self):
        a = self.primary()
        if self.peek()[1] == "**":
            self.take()
            if self.peek()[1] in ("+", "-"):  # extension: x**-y
                op = self.take()[1]
                b = Un(op, self.power())
            else:
                b = self.power()
            return Bin("**", a, b)
        return a

    def primary(self):
        kind, val = self.peek()
        if kind == "real":
            self.take()
            k = "d" if re.search(r"[dD]", val) else "f"
            return Num(val, k)
        if kind == "int":
            self.take()
            return Num(val, "i")
        if kind == "dotop" and val in (".TRUE.", ".FALSE."):
            self.take()
            return Bool(val == ".TRUE.")
        if kind == "str":
            self.take()
            return Str(val)
        if val == "(":
            self.take()
            e = self.expr()
            self.take(")")
            return e
        if val == "(/":
            self.take()
            items = [self.expr()]
            while self.peek()[1] == ",":
                self.take()
                items.append(self.expr())
            self.take("/)")
            return Ref("__ARRAYCTOR__", items)
        if kind == "name":
            self.take()
            if self.peek()[1] == "(":
                self.take()
                args = []
                if self.peek()[1] != ")":
                    args.append(self.arg())
                    while self.peek()[1] == ",":
                        self.take()
                        args.append(self.arg())
                self.take(")")
                return Ref(val, args)
            return Name(val)
        raise SyntaxError(f"unexpected {kind} {val}")

    def arg(self):
        # range lo:hi in declarations
        e = self.expr()
        if self.peek()[1] == ":":
            self.take()
            hi = self.expr()
            return ("range", e, hi)
        return e


def parse_expr(s):
    p = Parser(tokenize(s))
    e = p.expr()
    if not p.at_end():
        raise SyntaxError(f"trailing tokens in {s!r}: {p.t[p.i:]}")
    return e


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------

@dataclass
class Sym:
    name: str
    type: str            # 'f','d','i','b'
    dims: list | None    # list of (lo_expr, hi_expr) ASTs
    param: object = None  # AST for PARAMETER value
    dummy: bool = False
    data: list | None = None   # literal init values (texts)
    module: bool = False
    save_runtime: bool = False  # module SAVE variable (lives in AC struct)
    intent: str | None = None


INTRINSICS = {
    "MIN", "MAX", "ABS", "SQRT", "EXP", "LOG", "LOG10", "ERF", "DERF",
    "NINT", "INT", "REAL", "DBLE", "SNGL", "AINT", "DEXP", "DLOG", "DSQRT",
    "DABS", "DACOS", "DMAX1", "COS", "SIN", "TANH", "ACOS",
}

TYPEMAP = {"REAL": "f", "INTEGER": "i", "LOGICAL": "b",
           "DOUBLE PRECISION": "d", "DOUBLEPRECISION": "d"}

CTYPE = {"f": "float", "d": "double", "i": "int", "b": "bool"}


@dataclass
class Unit:
    kind: str           # 'sub' or 'fn'
    name: str
    args: list
    rtype: str | None
    stmts: list = field(default_factory=list)   # (ln,label,text)
    syms: dict = field(default_factory=dict)
    body: list = field(default_factory=list)    # parsed tree


def split_top(s, sep=","):
    out, depth, cur, q = [], 0, "", None
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out]


DECL_RE = re.compile(
    r"^(REAL|INTEGER|LOGICAL|DOUBLE\s*PRECISION)\b(.*)$", re.I)


def parse_dims(dimtext):
    dims = []
    for d in split_top(dimtext):
        if ":" in d:
            lo, hi = d.split(":", 1)
            dims.append((parse_expr(lo), parse_expr(hi)))
        else:
            dims.append((Num("1", "i"), parse_expr(d)))
    return dims


def parse_decl(text, scope: dict, module: bool):
    m = DECL_RE.match(text)
    if not m:
        return False
    tname = re.sub(r"\s+", " ", m.group(1).upper())
    t = TYPEMAP[tname.replace(" ", "") if tname.startswith("DOUBLE") else tname]
    rest = m.group(2)
    attrs = []
    if "::" in rest:
        attrpart, entpart = rest.split("::", 1)
        attrs = [a.strip().upper() for a in split_top(attrpart.lstrip(","))]
    else:
        entpart = rest
    is_param = any(a == "PARAMETER" for a in attrs)
    dims = None
    intent = None
    save = any(a == "SAVE" for a in attrs)
    for a in attrs:
        if a.startswith("DIMENSION"):
            dims = parse_dims(a[a.index("(") + 1:a.rindex(")")])
        if a.startswith("INTENT"):
            intent = a[a.index("(") + 1:a.rindex(")")].strip()
    for ent in split_top(entpart):
        init = None
        if "=" in ent and not ent.strip().startswith("("):
            nm, init = ent.split("=", 1)
        else:
            nm = ent
        nm = nm.strip()
        edims = dims
        mm = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)$", nm)
        if mm:
            nm = mm.group(1)
            edims = parse_dims(mm.group(2))
        nm = nm.upper()
        if nm in INTRINSICS and not is_param:
            continue  # typing an intrinsic name
        sym = scope.get(nm) or Sym(nm, t, edims)
        sym.type = t
        sym.dims = edims
        sym.module = module
        sym.intent = intent
        if is_param:
            sym.param = parse_expr(init)
        elif init is not None:
            sym.data = parse_expr(init)
        scope[nm] = sym
    return True


# --------------------------------------------------------------------------
# Program structure
# --------------------------------------------------------------------------

def read_program(path):
    lines = logical_lines(open(path, encoding="latin-1").read())
    modsyms: dict[str, Sym] = {}
    units: list[Unit] = []
    i = 0
    cur = None
    in_contains = False
    while i < len(lines):
        ln, label, s = lines[i]
        i += 1
        su = s.upper()
        if su.startswith("MODULE ") or su.startswith("END MODULE"):
            continue
        if su == "CONTAINS":
            in_contains = True
            continue
        if not in_contains:
            if su.startswith(("USE ", "IMPLICIT", "PUBLIC", "PRIVATE ::")):
                continue
            if parse_decl(s, modsyms, True):
                continue
            if su.startswith("DATA"):
                apply_data(s, modsyms)
                continue
            raise SyntaxError(f"module-level {ln}: {s}")
        m = re.match(r"^(?:(REAL|DOUBLE\s*PRECISION|INTEGER)\s+)?(SUBROUTINE|FUNCTION)\s+(\w+)\s*\((.*)\)\s*$", s, re.I)
        if m and cur is None:
            rtype = TYPEMAP.get(re.sub(r"\s+", "", (m.group(1) or "").upper()).replace("DOUBLEPRECISION", "DOUBLEPRECISION"), None)
            if m.group(1) and m.group(1).upper().startswith("DOUBLE"):
                rtype = "d"
            cur = Unit("fn" if m.group(2).upper() == "FUNCTION" else "sub",
                       m.group(3).upper(),
                       [a.strip().upper() for a in split_top(m.group(4))],
                       rtype)
            cur.span = [ln, None]
            continue
        if re.match(r"^END\s*(SUBROUTINE|FUNCTION)\b", s, re.I):
            cur.span[1] = ln
            units.append(cur)
            cur = None
            continue
        cur.stmts.append((ln, label, s))
    return modsyms, units


def apply_data(s, scope, lookup=None):
    lookup = scope if lookup is None else {**lookup, **scope}
    body = s[4:].strip()
    # groups: names /values/ [,] names /values/
    groups = re.findall(r"([^/]+)/([^/]*)/\s*,?", body)
    for names, vals in groups:
        nlist = [n.strip().upper() for n in split_top(names) if n.strip()]
        vlist = [v.strip() for v in split_top(vals)]
        k = 0
        for nm in nlist:
            sym = scope[nm]
            n = 1
            if sym.dims:
                for lo, hi in sym.dims:
                    n *= const_int(hi, lookup) - const_int(lo, lookup) + 1
            chunk = vlist[k:k + n]
            k += n
            sym.data = chunk
        if k != len(vlist):
            raise SyntaxError(f"DATA count mismatch {s}")


def const_int(e, scope):
    v = const_eval(e, scope)
    if v is None or not isinstance(v, int):
        raise ValueError(f"not a constant int: {e}")
    return v


def const_eval(e, scope, modsyms=None):
    """Evaluate an integer/logical constant expression (PARAMETERs only)."""
    look = lambda n: (scope.get(n) or (modsyms or {}).get(n))
    if isinstance(e, Num):
        return int(e.text) if e.kind == "i" else None
    if isinstance(e, Bool):
        return e.val
    if isinstance(e, Name):
        s = look(e.name)
        if s is not None and s.param is not None and s.type in ("i", "b"):
            return const_eval(s.param, scope, modsyms)
        return None
    if isinstance(e, Un):
        v = const_eval(e.x, scope, modsyms)
        if v is None:
            return None
        if e.op == "-":
            return -v
        if e.op == "+":
            return v
        if e.op == ".NOT.":
            return not v
    if isinstance(e, Bin):
        a = const_eval(e.a, scope, modsyms)
        b = const_eval(e.b, scope, modsyms)
        if e.op == ".AND.":
            if a is False or b is False:
                return False
            if a is None or b is None:
                return None
            return a and b
        if e.op == ".OR.":
            if a is True or b is True:
                return True
            if a is None or b is None:
                return None
            return a or b
        if a is None or b is None:
            return None
        if e.op == "+": return a + b
        if e.op == "-": return a - b
        if e.op == "*": return a * b
        if e.op == "/": return int(a / b) if b else None
        if e.op == "<": return a < b
        if e.op == "<=": return a <= b
        if e.op == ">": return a > b
        if e.op == ">=": return a >= b
        if e.op == "==": return a == b
        if e.op == "!=": return a != b
    return None


# --------------------------------------------------------------------------
# Statement tree
# --------------------------------------------------------------------------

@dataclass
class If:
    branches: list  # [(cond_ast, [stmts])]
    els: list | None


@dataclass
class Do:
    var: str
    lo: object
    hi: object
    step: object
    body: list


@dataclass
class Simple:
    ln: int
    label: int | None
    text: str


def build_tree(stmts):
    pos = 0

    def block(terms):
        nonlocal pos
        out = []
        while pos < len(stmts):
            ln, label, s = stmts[pos]
            su = s.upper().replace(" ", "")
            if any(su == t or su.startswith(t) for t in terms):
                return out, su
            pos += 1
            m = re.match(r"^IF\s*\(", s, re.I)
            if m:
                cond, rest = split_if(s)
                if rest.upper().replace(" ", "") == "THEN":
                    branches = []
                    body, term = block(("ELSEIF", "ELSE", "ENDIF"))
                    branches.append((parse_expr(cond), body))
                    els = None
                    while True:
                        ln2, _, s2 = stmts[pos]
                        pos += 1
                        t2 = s2.upper().replace(" ", "")
                        if t2.startswith("ELSEIF"):
                            c2, r2 = split_if(re.sub(r"^ELSE\s*IF", "IF", s2, flags=re.I))
                            body2, _ = block(("ELSEIF", "ELSE", "ENDIF"))
                            branches.append((parse_expr(c2), body2))
                        elif t2 == "ELSE":
                            els, _ = block(("ENDIF",))
                        elif t2 == "ENDIF":
                            break
                    if label is not None:
                        out.append(Simple(ln, label, "CONTINUE"))
                    out.append(If(branches, els))
                else:
                    if label is not None:
                        out.append(Simple(ln, label, "CONTINUE"))
                    out.append(If([(parse_expr(cond), [Simple(ln, None, rest)])], None))
                continue
            m = re.match(r"^DO\s+(\w+)\s*=\s*(.*)$", s, re.I)
            if m:
                parts = split_top(m.group(2))
                lo, hi = parse_expr(parts[0]), parse_expr(parts[1])
                step = parse_expr(parts[2]) if len(parts) > 2 else Num("1", "i")
                body, _ = block(("ENDDO", "END DO".replace(" ", "")))
                pos += 1
                if label is not None:
                    out.append(Simple(ln, label, "CONTINUE"))
                out.append(Do(m.group(1).upper(), lo, hi, step, body))
                continue
            out.append(Simple(ln, label, s))
        return out, None

    tree, _ = block(("__never__",))
    return tree


def split_if(s):
    # s starts with IF (
    i = s.index("(")
    depth = 0
    q = None
    for j in range(i, len(s)):
        ch = s[j]
        if q:
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], s[j + 1:].strip()
    raise SyntaxError(s)


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

RESERVED = {"G", "RD", "RV", "CP", "CV", "P0", "T0", "GAMMA", "RCP", "RCV",
            "RVOVRD", "RERADIUS", "XLV", "SVP1", "SVP2", "SVP3", "SVPT0",
            "RHOWATER", "EP2", "I3", "I3S", "IDX3", "PERIODIC", "real",
            "INFINITY", "NAN", "EOF"}


def cname_local(n):
    return ("v_" + n) if n in RESERVED else n


class Emitter:
    def __init__(self, modsyms, units, folds_in=None):
        self.mod = modsyms
        self.units = {u.name: u for u in units}
        self.folds: list[tuple[str, str, str]] = []  # (cname, type, fortran expr text)
        self.fold_index: dict[str, str] = {}
        self.folds_in = folds_in or {}
        self.unit = None
        self.written: dict[str, set] = {}
        self.uses_ac: set[str] = set()

    # ---- symbol lookup
    def sym(self, n):
        if self.unit is not None and n in self.unit.syms:
            return self.unit.syms[n]
        return self.mod.get(n)

    # ---- constant detection
    def is_const(self, e):
        if isinstance(e, (Num, Bool)):
            return True
        if isinstance(e, Name):
            s = self.sym(e.name)
            return s is not None and s.param is not None
        if isinstance(e, Un):
            return self.is_const(e.x)
        if isinstance(e, Bin):
            return self.is_const(e.a) and self.is_const(e.b)
        if isinstance(e, Ref):
            if e.name in INTRINSICS:
                return all(not isinstance(a, tuple) and self.is_const(a) for a in e.args)
            return False
        return False

    def needs_fold(self, e):
        if isinstance(e, Bin):
            return e.op == "**" and not self.trivial_pow(e) or self.needs_fold(e.a) or self.needs_fold(e.b)
        if isinstance(e, Un):
            return self.needs_fold(e.x)
        if isinstance(e, Ref):
            return e.name in INTRINSICS and e.name not in ("MIN", "MAX", "DMAX1", "REAL", "DBLE", "SNGL", "INT", "ABS", "DABS")
        return False

    def trivial_pow(self, e):
        v = self.real_literal_value(e.b)
        return v in (2.0, 1.0, -1.0)

    def real_literal_value(self, e):
        neg = False
        if isinstance(e, Un) and e.op == "-":
            neg = True
            e = e.x
        if isinstance(e, Num) and e.kind in ("f", "d"):
            v = float(e.text.upper().replace("D", "E"))
            return -v if neg else v
        return None

    def fortran_text(self, e):
        """Re-emit an AST as Fortran (for the constants program)."""
        if isinstance(e, Num):
            return e.text
        if isinstance(e, Bool):
            return ".TRUE." if e.val else ".FALSE."
        if isinstance(e, Name):
            s = self.sym(e.name)
            if s is not None and s.param is not None and not s.module:
                return "(" + self.fortran_text(s.param) + ")"
            return e.name
        if isinstance(e, Un):
            return f"({e.op}{self.fortran_text(e.x)})" if e.op != ".NOT." else f"(.NOT.{self.fortran_text(e.x)})"
        if isinstance(e, Bin):
            op = e.op
            return f"({self.fortran_text(e.a)}{op}{self.fortran_text(e.b)})"
        if isinstance(e, Ref):
            return f"{e.name}(" + ",".join(self.fortran_text(a) for a in e.args) + ")"
        raise TypeError(e)

    def fold(self, e, t):
        key = self.fortran_text(e) + "::" + t
        if key in self.fold_index:
            return self.fold_index[key]
        cname = f"NTU_K{len(self.folds)}"
        self.folds.append((cname, t, self.fortran_text(e)))
        self.fold_index[key] = cname
        return cname

    # ---- typing
    def etype(self, e):
        return self.emit(e, dry=True)[1]

    @staticmethod
    def maxt(a, b):
        order = {"b": 0, "i": 1, "f": 2, "d": 3}
        return a if order[a] >= order[b] else b

    def conv(self, code, frm, to):
        if frm == to:
            return code
        if to == "f" and frm in ("i", "d"):
            return f"(float)({code})"
        if to == "d" and frm in ("i", "f"):
            return f"(double)({code})"
        if to == "i":
            return f"(int)({code})"
        raise TypeError(f"conv {frm}->{to}")

    def emit(self, e, dry=False):
        """Return (code, type)."""
        if isinstance(e, Num):
            if e.kind == "i":
                return e.text, "i"
            txt = e.text.upper()
            if e.kind == "d":
                txt = txt.replace("D", "E")
                if "." not in txt.split("E")[0]:
                    m = re.match(r"(\d+)(E.*)", txt)
                    txt = m.group(1) + "." + m.group(2)
                return txt, "d"
            if txt.endswith("."):
                txt = txt + "0"
            if "." not in txt and "E" in txt:
                m = re.match(r"(\d+)(E.*)", txt)
                txt = m.group(1) + ".0" + m.group(2)
            if txt.startswith("."):
                txt = "0" + txt
            txt = txt.replace(".E", ".0E")
            return txt + "f", "f"
        if isinstance(e, Bool):
            return ("true" if e.val else "false"), "b"
        if isinstance(e, Name):
            if self.unit is not None and self.unit.kind == "fn" and e.name == self.unit.name:
                return "NTU_RET", self.unit.rtype
            s = self.sym(e.name)
            if s is None:
                raise NameError(f"{e.name} in {self.unit.name if self.unit else '-'}")
            if s.param is not None:
                if s.module:
                    return f"NTU_{s.name}", s.type
                return cname_local(s.name), s.type
            if s.module and s.save_runtime:
                self.uses_ac.add(self.unit.name)
                return f"AC->{s.name}", s.type
            if s.module and s.data is not None:
                return f"NTU_{s.name}", s.type
            if self.unit and e.name == self.unit.name and self.unit.kind == "fn":
                return "NTU_RET", self.unit.rtype
            return cname_local(s.name), s.type
        if isinstance(e, Str):
            return e.s, "s"
        if isinstance(e, Un):
            x, t = self.emit(e.x, dry)
            if e.op == ".NOT.":
                return f"(!{x})", "b"
            if e.op == "+":
                return x, t
            return f"(-{x})", t
        if isinstance(e, Bin):
            return self.emit_bin(e, dry)
        if isinstance(e, Ref):
            return self.emit_ref(e, dry)
        raise TypeError(e)

    LOGOPS = {".AND.": "&&", ".OR.": "||", ".EQV.": "==", ".NEQV.": "!="}

    def emit_bin(self, e, dry):
        if e.op in self.LOGOPS:
            a, _ = self.emit(e.a, dry)
            b, _ = self.emit(e.b, dry)
            return f"({a} {self.LOGOPS[e.op]} {b})", "b"
        if e.op == "**":
            return self.emit_pow(e, dry)
        a, ta = self.emit(e.a, dry)
        b, tb = self.emit(e.b, dry)
        t = self.maxt(ta, tb)
        a = self.conv(a, ta, t)
        b = self.conv(b, tb, t)
        if e.op in ("<", "<=", ">", ">=", "==", "!="):
            return f"({a} {e.op} {b})", "b"
        return f"({a} {e.op} {b})", t

    def emit_pow(self, e, dry):
        ta = self.etype(e.a)
        tb = self.etype(e.b)
        if self.is_const(e) and self.needs_fold(e):
            t = self.maxt(ta, tb) if tb != "i" else ta
            return self.fold(e, t), t
        a, ta = self.emit(e.a, dry)
        lit = self.real_literal_value(e.b)
        if tb == "i":
            raise NotImplementedError(f"integer exponent: {self.fortran_text(e)}")
        t = self.maxt(ta, tb)
        a = self.conv(a, ta, t)
        if lit == 2.0:
            return (f"ntu_sq({a})" if t == "f" else f"ntu_sqd({a})"), t
        if lit == 1.0:
            return a, t
        if lit == -1.0:
            return (f"(1.0f / {a})" if t == "f" else f"(1.0 / {a})"), t
        b, tb = self.emit(e.b, dry)
        b = self.conv(b, tb, t)
        return (f"NTU_POWF({a}, {b})" if t == "f" else f"NTU_POW({a}, {b})"), t

    def emit_ref(self, e, dry):
        n = e.name
        s = self.sym(n)
        if s is not None and s.dims is not None and n not in INTRINSICS:
            return self.emit_index(s, e.args, dry), s.type
        if n in INTRINSICS:
            if self.is_const(e) and self.needs_fold(e):
                t = self.intrinsic_type(e)
                return self.fold(e, t), t
            return self.emit_intrinsic(e, dry)
        if n in self.units:
            u = self.units[n]
            return self.emit_call_args(u, e.args, dry, fn=True), u.rtype
        raise NameError(f"unknown ref {n}")

    def intrinsic_type(self, e):
        n = e.name
        ts = [self.etype(a) for a in e.args]
        if n in ("DEXP", "DLOG", "DSQRT", "DABS", "DACOS", "DMAX1", "DERF", "DBLE"):
            return "d"
        if n in ("NINT", "INT"):
            return "i"
        if n in ("REAL", "SNGL"):
            return "f"
        t = ts[0]
        for x in ts[1:]:
            t = self.maxt(t, x)
        return t

    def emit_intrinsic(self, e, dry):
        n = e.name
        args = [self.emit(a, dry) for a in e.args]
        if n in ("MIN", "MAX", "DMAX1"):
            t = args[0][1]
            for _, ta in args[1:]:
                t = self.maxt(t, ta)
            fn = {"f": "ntu_f", "d": "ntu_d", "i": "ntu_i"}[t] + ("min" if n == "MIN" else "max")
            acc = self.conv(args[0][0], args[0][1], t)
            for c, ta in args[1:]:
                acc = f"{fn}({acc}, {self.conv(c, ta, t)})"
            return acc, t
        (x, t), = args[:1]
        if n == "ABS":
            return {"f": f"fabsf({x})", "d": f"fabs({x})", "i": f"abs({x})"}[t], t
        if n == "DABS":
            return f"fabs({x})", "d"
        if n == "SQRT":
            return (f"sqrtf({x})", "f") if t == "f" else (f"sqrt({x})", "d")
        if n == "DSQRT":
            return f"sqrt({self.conv(x, t, 'd')})", "d"
        if n == "EXP":
            return (f"NTU_EXPF({x})", "f") if t == "f" else (f"NTU_EXP({x})", "d")
        if n == "DEXP":
            return f"NTU_EXP({self.conv(x, t, 'd')})", "d"
        if n == "LOG":
            return (f"NTU_LOGF({x})", "f") if t == "f" else (f"NTU_LOG({x})", "d")
        if n == "DLOG":
            return f"NTU_LOG({self.conv(x, t, 'd')})", "d"
        if n == "LOG10":
            assert t == "f"
            return f"NTU_LOG10F({x})", "f"
        if n == "ERF":
            assert t == "f"
            return f"NTU_ERFF({x})", "f"
        if n == "DERF":
            return f"NTU_ERF({self.conv(x, t, 'd')})", "d"
        if n == "COS":
            return f"NTU_COSF({x})", "f"
        if n == "SIN":
            return f"NTU_SINF({x})", "f"
        if n == "TANH":
            return f"NTU_TANHF({x})", "f"
        if n == "NINT":
            assert t == "f"
            return f"ntu_nint({x})", "i"
        if n == "INT":
            return (f"(int)({x})" if t != "i" else x), "i"
        if n in ("REAL", "SNGL"):
            return self.conv(x, t, "f"), "f"
        if n == "DBLE":
            return self.conv(x, t, "d"), "d"
        if n == "AINT":
            return f"truncf({x})", "f"
        raise NotImplementedError(n)

    def index_code(self, s, args, dry):
        dims = s.dims
        if len(args) != len(dims):
            raise IndexError(f"{s.name}: {len(args)} subscripts for {len(dims)} dims")
        terms = []
        for k, (a, (lo, hi)) in enumerate(zip(args, dims)):
            c, t = self.emit(a, dry)
            lc, _ = self.emit(lo, dry)
            if lc == "0":
                terms.append(f"{c}")
            elif lc == "1" and False:
                terms.append(f"{c} - 1")
            else:
                terms.append(f"({c}) - ({lc})")
        idx = terms[-1]
        for k in range(len(dims) - 2, -1, -1):
            ext = self.extent(s, k)
            idx = f"({terms[k]}) + {ext} * ({idx})"
        return idx

    def extent(self, s, k):
        lo, hi = s.dims[k]
        try:
            return str(const_int(hi, {**self.mod, **(self.unit.syms if self.unit else {})}) -
                       const_int(lo, {**self.mod, **(self.unit.syms if self.unit else {})}) + 1)
        except ValueError:
            return "NTU_KMAX"

    def array_base(self, s):
        if s.module and s.save_runtime:
            self.uses_ac.add(self.unit.name)
            return f"AC->{s.name}"
        if s.module:
            return f"NTU_{s.name}"
        return cname_local(s.name)

    def emit_index(self, s, args, dry):
        return f"{self.array_base(s)}[{self.index_code(s, args, dry)}]"

    def emit_call_args(self, u, args, dry, fn=False):
        out = []
        for k, a in enumerate(args):
            dname = u.args[k]
            dsym = u.syms[dname]
            if dsym.dims is not None:
                # array dummy: pass address
                if isinstance(a, Name):
                    s = self.sym(a.name)
                    out.append(self.array_base(s))
                elif isinstance(a, Ref) and self.sym(a.name) is not None and self.sym(a.name).dims is not None:
                    s = self.sym(a.name)
                    out.append(f"&{self.array_base(s)}[{self.index_code(s, a.args, dry)}]")
                else:
                    raise TypeError(f"array dummy {dname} of {u.name} given {a}")
            else:
                c, t = self.emit(a, dry)
                written = dname in self.written.get(u.name, set())
                if written:
                    if t != dsym.type:
                        raise TypeError(f"type mismatch passing {c} to written {u.name}:{dname}")
                    out.append(c)
                else:
                    out.append(self.conv(c, t, dsym.type))
        if u.name in self.uses_ac:
            out.insert(0, "AC")
        name = "ntu_" + u.name.lower()
        return f"{name}({', '.join(out)})"

    # ---- statements
    def stmt(self, node, ind):
        pad = "    " * ind
        if isinstance(node, If):
            return self.emit_if(node, ind)
        if isinstance(node, Do):
            v = cname_local(node.var)
            lo, tlo = self.emit(node.lo)
            hi, thi = self.emit(node.hi)
            st = const_eval(node.step, {}, self.mod)
            lines = []
            bound = f"ntu_hi_{node.var.lower()}"
            lines.append(f"{pad}{{ const int {bound} = {hi};")
            if st == 1:
                lines.append(f"{pad}for ({v} = {lo}; {v} <= {bound}; ++{v}) {{")
            elif st == -1:
                lines.append(f"{pad}for ({v} = {lo}; {v} >= {bound}; --{v}) {{")
            else:
                raise NotImplementedError("DO step")
            for b in node.body:
                lines.append(self.stmt(b, ind + 1))
            lines.append(f"{pad}}} }}")
            return "\n".join(l for l in lines if l)
        return self.simple(node, ind)

    def emit_if(self, node, ind):
        pad = "    " * ind
        kept = []
        els = node.els
        for cond, body in node.branches:
            v = const_eval(cond, self.unit.syms, self.mod)
            if v is False:
                continue
            if v is True:
                els = body
                break
            kept.append((cond, body))
        lines = []
        if not kept:
            if els:
                for b in els:
                    lines.append(self.stmt(b, ind))
            return "\n".join(l for l in lines if l)
        for k, (cond, body) in enumerate(kept):
            c, _ = self.emit(cond)
            lines.append(f"{pad}{'if' if k == 0 else '} else if'} ({c}) {{")
            for b in body:
                lines.append(self.stmt(b, ind + 1))
        if els:
            lines.append(f"{pad}}} else {{")
            for b in els:
                lines.append(self.stmt(b, ind + 1))
        lines.append(f"{pad}}}")
        return "\n".join(l for l in lines if l)

    def simple(self, node, ind):
        pad = "    " * ind
        s = node.text
        su = s.upper()
        lab = f"{pad}L_{node.label}:;\n" if node.label is not None else ""
        if su == "CONTINUE":
            return lab.rstrip("\n")
        m = re.match(r"^GO\s*TO\s+(\d+)$", s, re.I)
        if m:
            return f"{lab}{pad}goto L_{m.group(1)};"
        if su == "RETURN":
            if self.unit.kind == "fn":
                return f"{lab}{pad}return NTU_RET;"
            return f"{lab}{pad}return;"
        if su.startswith("PRINT") or su.startswith("WRITE"):
            return f"{lab}{pad}/* {s[:60]} */"
        if su == "STOP":
            return f"{lab}{pad}NTU_FATAL(NTU_ERR_STOP);"
        m = re.match(r"^CALL\s+(\w+)\s*(?:\((.*)\))?$", s, re.I)
        if m:
            cn = m.group(1).upper()
            if cn == "WRF_ERROR_FATAL":
                return f"{lab}{pad}NTU_FATAL(NTU_ERR_FATAL); /* {m.group(2)} */"
            if cn not in self.units:
                raise NameError(cn)
            args = [] if not m.group(2) else [parse_expr(x) for x in split_top(m.group(2))]
            return f"{lab}{pad}{self.emit_call_args(self.units[cn], args, False)};"
        # assignment
        lhs, rhs = split_assign(s)
        le = parse_lhs(lhs)
        re_ = parse_expr(rhs)
        if isinstance(re_, Ref) and re_.name == "__ARRAYCTOR__":
            s_ = self.sym(le.name)
            base = self.array_base(s_)
            parts = []
            for k, item in enumerate(re_.args):
                c, t = self.emit(item)
                parts.append(f"{base}[{k}] = {self.conv(c, t, s_.type)};")
            return lab + pad + " ".join(parts)
        rc, rt = self.emit(re_)
        if isinstance(le, Name):
            s_ = self.sym(le.name)
            if self.unit.kind == "fn" and le.name == self.unit.name:
                lc, lt = "NTU_RET", self.unit.rtype
            elif s_.dims is not None:
                # whole-array assignment
                n = 1
                for k in range(len(s_.dims)):
                    n = f"{n} * {self.extent(s_, k)}"
                base = self.array_base(s_)
                return f"{lab}{pad}for (int ntu_w = 0; ntu_w < {n}; ++ntu_w) {base}[ntu_w] = {self.conv(rc, rt, s_.type)};"
            else:
                lc, lt = self.emit(le)
        else:
            lc, lt = self.emit(le)
        return f"{lab}{pad}{lc} = {self.conv(rc, rt, lt) if rt != 'b' else rc};"


def split_assign(s):
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "=" and depth == 0 and s[i + 1:i + 2] != "=" and s[i - 1:i] not in "<>/=":
            return s[:i], s[i + 1:]
    raise SyntaxError(f"not an assignment: {s}")


def parse_lhs(s):
    return parse_expr(s)


# --------------------------------------------------------------------------
# Unit preparation: declarations, write analysis
# --------------------------------------------------------------------------

def prepare(units, modsyms):
    for s in modsyms.values():
        s.save_runtime = s.param is None and s.data is None
    for u in units:
        body = []
        for ln, label, s in u.stmts:
            su = s.upper()
            if su.startswith("IMPLICIT"):
                continue
            if su.startswith("SAVE"):
                continue
            if parse_decl(s, u.syms, False):
                continue
            if su.startswith("DATA "):
                apply_data(s, u.syms, modsyms)
                continue
            body.append((ln, label, s))
        for a in u.args:
            if a not in u.syms:
                raise NameError(f"undeclared dummy {a} in {u.name}")
            u.syms[a].dummy = True
        if u.kind == "fn" and u.rtype is None:
            u.rtype = u.syms[u.name].type
        u.stmts = body
        u.body = build_tree(body)


def apply_data_unit(s, syms):
    apply_data(s, syms)


def collect_writes(units):
    direct = {}
    calls = {}
    for u in units:
        w = set()
        c = []

        def walk(nodes):
            for n in nodes:
                if isinstance(n, If):
                    for _, b in n.branches:
                        walk(b)
                    if n.els:
                        walk(n.els)
                elif isinstance(n, Do):
                    w.add(n.var)
                    walk(n.body)
                else:
                    s = n.text
                    m = re.match(r"^CALL\s+(\w+)\s*(?:\((.*)\))?$", s, re.I)
                    if m:
                        if m.group(2):
                            args = [x.strip() for x in split_top(m.group(2))]
                        else:
                            args = []
                        c.append((m.group(1).upper(), args))
                        continue
                    try:
                        lhs, _ = split_assign(s)
                        nm = re.match(r"\s*(\w+)", lhs).group(1).upper()
                        w.add(nm)
                    except SyntaxError:
                        pass
        walk(u.body)
        direct[u.name] = w
        calls[u.name] = c
    written = {k: set(v) for k, v in direct.items()}
    umap = {u.name: u for u in units}
    changed = True
    while changed:
        changed = False
        for u in units:
            for cn, args in calls[u.name]:
                if cn not in umap:
                    continue
                cu = umap[cn]
                for k, a in enumerate(args):
                    if k < len(cu.args) and cu.args[k] in written[cn]:
                        m = re.match(r"\s*(\w+)", a)
                        if m and m.group(1).upper() not in written[u.name]:
                            written[u.name].add(m.group(1).upper())
                            changed = True
    return written


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

SKIP_UNITS = {"GAMMA", "GAMIN", "NTU_INIT", "MP_NTU", "INIT_AEROSOL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--params-out")
    ap.add_argument("--params-in")
    ap.add_argument("--out")
    args = ap.parse_args()
    modsyms, units = read_program(args.source)
    prepare(units, modsyms)
    units = [u for u in units if u.name not in SKIP_UNITS]
    em = Emitter(modsyms, units)
    em.written = collect_writes(units)
    # functions that touch module runtime state need AC (transitive)
    # first pass: dry emission to collect folds and AC users
    for _ in range(3):
        for u in units:
            em.unit = u
            for node in u.body:
                em.stmt(node, 1)
        # propagate AC through calls
        prev = set(em.uses_ac)
        for u in units:
            for cn, _ in []:
                pass
    # transitive AC
    callgraph = {u.name: set() for u in units}
    for u in units:
        txt = "\n".join(s for _, _, s in u.stmts).upper()
        for v in units:
            if re.search(r"\b%s\s*\(" % v.name, txt) and v.name != u.name:
                callgraph[u.name].add(v.name)
    changed = True
    while changed:
        changed = False
        for u in units:
            if u.name not in em.uses_ac and any(c in em.uses_ac for c in callgraph[u.name]):
                em.uses_ac.add(u.name)
                changed = True
    # module params program
    if args.params_out:
        write_params_program(args.params_out, args.source, modsyms, em)
        return
    vals = read_params(args.params_in)
    out = []
    out.append(emit_module_constants(modsyms, em, vals))
    # prototypes
    order = topo(units, callgraph)
    for u in order:
        out.append(proto(u, em) + ";")
    for u in order:
        out.append(emit_unit(u, em))
    open(args.out, "w").write("\n\n".join(out) + "\n")


def topo(units, cg):
    seen, out = set(), []
    umap = {u.name: u for u in units}

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for c in sorted(cg[n]):
            visit(c)
        out.append(umap[n])
    for u in units:
        visit(u.name)
    return out


def proto(u, em):
    params = []
    if u.name in em.uses_ac:
        params.append("NtuAeroConst* __restrict__ AC")
    for a in u.args:
        s = u.syms[a]
        ct = CTYPE[s.type]
        nm = cname_local(a)
        if s.dims is not None:
            params.append(f"{ct}* {nm}")
        elif a in em.written.get(u.name, set()):
            params.append(f"{ct}& {nm}")
        else:
            params.append(f"{ct} {nm}")
    rt = CTYPE[u.rtype] if u.kind == "fn" else "void"
    span = getattr(u, "span", None)
    big = span is not None and span[1] - span[0] > 40
    qual = "NTU_DEV_BIG" if big else "NTU_DEV"
    return f"{qual} {rt} ntu_{u.name.lower()}({', '.join(params)})"


def emit_unit(u, em):
    em.unit = u
    span = getattr(u, "span", None)
    head = f"// WRF {'FUNCTION' if u.kind == 'fn' else 'SUBROUTINE'} {u.name}"
    if span:
        head += f" (module_mp_ntu.F:{span[0]}-{span[1]})"
    lines = [head, proto(u, em) + " {"]
    scalars = {"f": [], "d": [], "i": [], "b": []}
    for n, s in u.syms.items():
        if s.dummy:
            continue
        if u.kind == "fn" and n == u.name:
            continue
        ct = CTYPE[s.type]
        nm = cname_local(n)
        if s.param is not None:
            c, t = em.emit(s.param)
            lines.append(f"    const {ct} {nm} = {em.conv(c, t, s.type)};")
            continue
        if s.data is not None and isinstance(s.data, list):
            vals = [lit(v, s.type) for v in s.data]
            if s.dims:
                lines.append(f"    const {ct} {nm}[{len(vals)}] = {{{', '.join(vals)}}};")
            else:
                lines.append(f"    const {ct} {nm} = {vals[0]};")
            continue
        if s.dims:
            exts = [em.extent(s, k) for k in range(len(s.dims))]
            lines.append(f"    {ct} {nm}[{' * '.join(exts)}] = {{}};")
        else:
            scalars[s.type].append(nm)
    for t, names in scalars.items():
        if not names:
            continue
        init = {"f": "0.0f", "d": "0.0", "i": "0", "b": "false"}[t]
        cur = f"    {CTYPE[t]} "
        parts = []
        for nm in names:
            item = f"{nm} = {init}"
            if len(cur) + len(item) + 2 > 96 and parts:
                lines.append(cur + ", ".join(parts) + ";")
                parts = []
            parts.append(item)
        if parts:
            lines.append(cur + ", ".join(parts) + ";")
    if u.kind == "fn":
        lines.append(f"    {CTYPE[u.rtype]} NTU_RET = 0;")
    for node in u.body:
        lines.append(em.stmt(node, 1))
    if u.kind == "fn":
        lines.append("    return NTU_RET;")
    lines.append("}")
    return "\n".join(l for l in lines if l)


def lit(v, t):
    v = v.strip().upper()
    if t == "f":
        return Emitter.emit(None, Num(v, "f"))[0] if False else float_lit(v)
    if t == "d":
        x = v.replace("D", "E")
        return x if any(ch in x for ch in ".E") else x + ".0"
    return v


def float_lit(v):
    neg = v.startswith("-")
    if neg or v.startswith("+"):
        v = v[1:]
    e = Emitter.__new__(Emitter)
    c, _ = Emitter.emit(e, Num(v, "f"))
    return ("-" if neg else "") + c


def module_param_names(modsyms):
    return [n for n, s in modsyms.items() if s.param is not None]


def write_params_program(path, source, modsyms, em):
    """A gfortran program that prints the exact bits of every module
    PARAMETER and every folded constant subexpression."""
    text = open(source, encoding="latin-1").read()
    # copy the module-level PARAMETER declarations verbatim, minus PRIVATE
    decl_lines = []
    for ln, label, s in logical_lines(text):
        su = s.upper()
        if su.startswith("CONTAINS"):
            break
        if "PARAMETER" in su and DECL_RE.match(s):
            decl_lines.append(re.sub(r",\s*PRIVATE", "", s, flags=re.I))
    out = ["PROGRAM NTU_PARAMS", "IMPLICIT NONE"]
    for d in decl_lines:
        # wrap long lines
        out.append(wrap_f90(d))
    for i, (cname, t, ftxt) in enumerate(em.folds):
        ft = {"f": "REAL", "d": "DOUBLE PRECISION"}[t]
        out.append(wrap_f90(f"{ft}, PARAMETER :: X_{cname} = {ftxt}"))
    for n in module_param_names(modsyms):
        s = modsyms[n]
        if s.type == "f":
            out.append(f"WRITE(*,'(A,1X,Z8.8)') 'f {n}', TRANSFER({n}, 0)")
        elif s.type == "d":
            out.append(f"WRITE(*,'(A,1X,Z16.16)') 'd {n}', TRANSFER({n}, 0_8)")
        elif s.type == "i":
            out.append(f"WRITE(*,'(A,1X,I0)') 'i {n}', {n}")
    for cname, t, ftxt in em.folds:
        if t == "f":
            out.append(f"WRITE(*,'(A,1X,Z8.8)') 'f {cname}', TRANSFER(X_{cname}, 0)")
        else:
            out.append(f"WRITE(*,'(A,1X,Z16.16)') 'd {cname}', TRANSFER(X_{cname}, 0_8)")
    out.append("END PROGRAM NTU_PARAMS")
    open(path, "w").write("\n".join(out) + "\n")
    with open(path + ".folds", "w") as f:
        for cname, t, ftxt in em.folds:
            f.write(f"{cname} {t} {ftxt}\n")


def wrap_f90(s):
    out = []
    while len(s) > 100:
        out.append(s[:100] + "&")
        s = "&" + s[100:]
    out.append(s)
    return "\n".join(out)


def read_params(path):
    vals = {}
    for line in open(path):
        p = line.split()
        if len(p) == 3:
            vals[p[1].upper()] = (p[0], p[2])
    return vals


def emit_module_constants(modsyms, em, vals):
    lines = ["// ---- module PARAMETERs, exact bits from gfortran's own folding ----"]
    for n in module_param_names(modsyms):
        s = modsyms[n]
        t, v = vals[n]
        if t == "f":
            lines.append(f"#define NTU_{n} ntu_bits_f(0x{v}u) /* {em.fortran_text(s.param)} */")
        elif t == "d":
            lines.append(f"#define NTU_{n} ntu_bits_d(0x{v}ull) /* {em.fortran_text(s.param)} */")
        else:
            lines.append(f"#define NTU_{n} {v}")
    lines.append("// ---- constant subexpressions gfortran folds at compile time ----")
    for cname, t, ftxt in em.folds:
        tt, v = vals[cname]
        if tt == "f":
            lines.append(f"#define {cname} ntu_bits_f(0x{v}u) /* {ftxt} */")
        else:
            lines.append(f"#define {cname} ntu_bits_d(0x{v}ull) /* {ftxt} */")
    lines.append("// ---- module DATA tables ----")
    for n, s in modsyms.items():
        if s.param is None and s.data is not None and not s.save_runtime:
            if isinstance(s.data, list):
                vals_ = [lit(v, s.type) for v in s.data]
                lines.append(f"NTU_CONST_TABLE {CTYPE[s.type]} NTU_{n}[{len(vals_)}] = {{" +
                             ", ".join(vals_) + "};")
            else:
                c, t = em.emit(s.data) if not isinstance(s.data, Ref) else (None, None)
                items = s.data.args if isinstance(s.data, Ref) else []
                cs = [em.emit(x)[0] for x in items]
                lines.append(f"NTU_CONST_TABLE {CTYPE[s.type]} NTU_{n}[{len(cs)}] = {{" + ", ".join(cs) + "};")
    lines.append("// ---- module SAVE state set by AERO_CONST (lives in NtuAeroConst) ----")
    lines.append("struct NtuAeroConst {")
    for n, s in modsyms.items():
        if s.save_runtime:
            ct = CTYPE[s.type]
            if s.dims:
                n_el = 1
                for lo, hi in s.dims:
                    n_el *= const_int(hi, modsyms) - const_int(lo, modsyms) + 1
                lines.append(f"    {ct} {n}[{n_el}];")
            else:
                lines.append(f"    {ct} {n};")
    lines.append("};")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
