"""Build-time translation of the pinned complete RRFS MYNN-EDMF source to CUDA.

This tool handles the Fortran subset present in the pinned fork modules. It
does not execute physics in Python. Unsupported syntax is an error with its
original source location. Generated routines retain file:line citations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re


def split_top(text, separator=","):
    result, start, depth, quote = [], 0, 0, None
    for index, char in enumerate(text):
        if char in "'\"":
            quote = None if quote == char else char if quote is None else quote
        if quote is None:
            depth += (char == "(") - (char == ")")
            if char == separator and depth == 0:
                result.append(text[start:index].strip())
                start = index + 1
    result.append(text[start:].strip())
    return result


def statements(path):
    """Preserve the first physical source line of every logical statement."""
    pending, start, enabled, stack = "", 0, True, []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line.startswith("#"):
            if line.startswith("#if"):
                assert "WRF" in line or "HARDCODE_VERTICAL" in line, (path, lineno, line)
                stack.append(enabled)
                enabled = False
            elif line.startswith("#else"):
                enabled = stack[-1] and not enabled
            elif line.startswith("#endif"):
                enabled = stack.pop()
            elif enabled:
                raise ValueError((path, lineno, line))
            continue
        if not enabled:
            continue
        quote = None
        for index, char in enumerate(line):
            if char in "'\"":
                quote = None if quote == char else char if quote is None else quote
            if char == "!" and quote is None:
                line = line[:index]
                break
        line = line.strip().lstrip("&").strip()
        if not line:
            continue
        if not pending:
            start = lineno
        pending += " " + line.removesuffix("&").strip()
        if not line.endswith("&"):
            for stmt in split_top(pending, ";"):
                if stmt:
                    yield start, re.sub(r"\benum\b", "mynn_enum", stmt.lower())
            pending = ""
    assert not pending and not stack


@dataclass
class Variable:
    name: str
    kind: str
    dimensions: list[str] = field(default_factory=list)
    intent: str = ""
    optional: bool = False
    parameter: bool = False
    initial: str | None = None
    line: int = 0


def declaration(stmt, line):
    match = re.match(r"(double\s+precision|real(?:\s*\([^)]*\))?|integer(?:\s*\([^)]*\))?|logical|character\s*\([^)]*\))\s*(.*)", stmt)
    if not match:
        return None
    dtype, rest = match.groups()
    kind = "double" if dtype.startswith("double") or "kind_phys8" in dtype or "kind=8" in dtype else "float" if dtype.startswith("real") else "int" if dtype.startswith("integer") else "bool" if dtype.startswith("logical") else "char"
    if "::" in rest:
        attrs, names = rest.split("::", 1)
    else:
        attrs, names = "", rest
    dim = re.search(r"dimension\s*\(([^)]*)\)", attrs)
    intent = re.search(r"intent\s*\(([^)]*)\)", attrs)
    output = []
    for item in split_top(names):
        initial = None
        if "=" in item:
            item, initial = item.split("=", 1)
        var = re.fullmatch(r"\s*(\w+)\s*(?:\((.*)\))?\s*", item)
        if not var:
            raise ValueError((line, stmt, item))
        dimensions = split_top(var[2]) if var[2] is not None else split_top(dim[1]) if dim else []
        output.append(Variable(var[1], kind, dimensions, intent[1] if intent else "", "optional" in attrs, "parameter" in attrs, initial, line))
    return output


@dataclass
class Routine:
    name: str
    arguments: list[str]
    function: bool
    source: str
    start: int
    end: int = 0
    variables: dict = field(default_factory=dict)
    body: list = field(default_factory=list)


def parse_modules(paths):
    globals_, routines, current = {}, {}, None
    for path in paths:
        for line, stmt in statements(path):
            begin = re.match(r"(subroutine|function)\s+(\w+)\s*\((.*)\)$", stmt)
            end = re.match(r"end\s*(subroutine|function)\b", stmt)
            if begin:
                assert current is None
                current = Routine(begin[2], split_top(begin[3]), begin[1] == "function", ('src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/' + ('MPAS/' if 'MPAS' in path.parts else '') + path.name), line)
                routines[current.name] = current
            elif end:
                assert current is not None
                current.end = line
                current = None
            elif (decl := declaration(stmt, line)) is not None:
                table = current.variables if current else globals_
                table.update((var.name, var) for var in decl)
            elif current:
                if not stmt.startswith(("implicit ", "use ", "save", "external ")):
                    current.body.append((line, stmt))
    assert current is None
    for routine in routines.values():
        assert set(routine.arguments) <= routine.variables.keys(), (routine.name, set(routine.arguments)-routine.variables.keys())
    return globals_, routines


TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?(?:[ed][+-]?\d+)?(?:_\w+)?|\.\d+(?:[ed][+-]?\d+)?(?:_\w+)?)|(\.[a-z]+\.)|(\w+)|('[^']*'|\"[^\"]*\")|(\*\*|<=|>=|==|/=|[+*/(),:\-=<>]))", re.I)
PRECEDENCE = {".or.": 1, ".and.": 2, ".eqv.": 0, ".neqv.": 0, "==": 3, "/=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3, ".eq.": 3, ".ne.": 3, ".lt.": 3, ".gt.": 3, ".le.": 3, ".ge.": 3, "+": 4, "-": 4, "*": 5, "/": 5, "**": 7}


class Expression:
    def __init__(self, source):
        self.tokens, pos = [], 0
        while pos < len(source):
            match = TOKEN.match(source, pos)
            if not match:
                raise ValueError(f"Cannot tokenize {source[pos:]!r} in {source!r}")
            self.tokens.append(next(s for s in match.groups() if s is not None))
            pos = match.end()
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def pop(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise ValueError((expected, token, self.tokens))
        self.index += 1
        return token

    def parse(self, minimum=0):
        token = self.pop()
        if token in ("+", "-", ".not."):
            lhs = ("unary", token, self.parse(6 if token != ".not." else 3))
        elif token == "(":
            lhs = self.parse()
            self.pop(")")
        elif token == ":":
            lhs = ("range", None, self.parse() if self.peek() not in (",", ")", "") else None)
        elif re.match(r"(?:\d|\.\d|'|\")", token) or token in (".true.", ".false."):
            lhs = ("literal", token)
        else:
            lhs = ("name", token)
            if self.peek() == "(":
                self.pop()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.parse())
                        if self.peek() != ",":
                            break
                        self.pop()
                self.pop(")")
                lhs = ("call", token, args)
        while True:
            op = self.peek()
            if op == ":" and minimum == 0:
                self.pop()
                rhs = self.parse() if self.peek() not in (",", ")", "") else None
                lhs = ("range", lhs, rhs)
                continue
            level = PRECEDENCE.get(op, -1)
            if level < minimum:
                break
            self.pop()
            lhs = ("binary", op, lhs, self.parse(level if op == "**" else level+1))
        return lhs

    @classmethod
    def tree(cls, source):
        parser = cls(source.strip())
        result = parser.parse()
        if parser.peek():
            raise ValueError((source, parser.tokens[parser.index:]))
        return result


def promote(a, b):
    return "double" if "double" in (a, b) else "float" if "float" in (a, b) else "int"


class Emitter:
    def __init__(self, globals_, routines, routine=None):
        self.globals = globals_
        self.routines = routines
        self.routine = routine
        self.variables = {**globals_, **(routine.variables if routine else {})}
        self.scalar_index = None
        self.constant = False
        self.selects = []
        self.lane_elementwise = False
        self.lane_plan = None
        self.lane_uid = 0

    def expression(self, source):
        return self.emit(Expression.tree(source))

    def constant_value(self, node):
        if node[0] == 'literal':
            try:
                return float(re.sub(r'_(kind_phys|\d+)$', '', node[1]).replace('d', 'e'))
            except ValueError:
                return None
        if node[0] == 'name':
            var = self.variables.get(node[1])
            if var and var.parameter and var.initial is not None:
                return self.constant_value(Expression.tree(var.initial))
        return None

    def emit(self, tree):
        # GNU strict applies the sign after multiplication/division. CUDA unary
        # minus alone may preserve a NaN sign, so emit the explicit sign binding.
        if (tree[0] == 'binary' and tree[1] in ('*', '/')
                and tree[2][0] == 'unary' and tree[2][1] == '-'):
            inner, kind = self.emit(('binary', tree[1], tree[2][2], tree[3]))
            if kind in ('float', 'double') and not self.constant:
                return f'r6_neg({inner})', kind
        if tree[0] == 'unary' and tree[1] == '-':
            value, kind = self.emit(tree[2])
            if kind in ('float', 'double') and not self.constant:
                return f'r6_neg({value})', kind
        if tree[0] == 'call' and tree[1] in ('min', 'max'):
            # MAXSS/MINSS return their second operand for an unordered pair.
            # Preserve the original GNU constant/expression order, including
            # sh background limits (F90:2903,3061,3064) and tsq (F90:3534).
            name, args = tree[1], tree[2]
            values = [self.emit(arg) for arg in args]
            kind, result = values[0][1], values[0][0]
            constant = self.constant_value(args[0])
            for node, (value, vk) in zip(args[1:], values[1:]):
                kind = promote(kind, vk)
                other = self.constant_value(node)
                a, b = result, value
                background_sh = (name == 'max' and self.routine
                                 and self.routine.name == 'mym_turbulence'
                                 and args[0][0] == 'call' and args[0][1] == 'sh'
                                 and (other == 0.004 or node[0] == 'binary'))
                if background_sh:
                    a, b = value, result
                elif name == 'min' and constant is not None and other is None:
                    a, b = b, a
                if name == 'max' and not background_sh:
                    if other == 0. and constant is None:
                        a, b = b, a
                    elif (constant is not None and other is None
                          and (constant != 0. or node[0] == 'binary')):
                        a, b = b, a
                result = f'mynn_{name}(({kind})({a}), ({kind})({b}))'
                constant = None
            return result, kind
        return self.emit_base(tree)

    def emit_base(self, tree):
        tag = tree[0]
        if tag == "literal":
            token = tree[1]
            if token in (".true.", ".false."):
                return ("true" if token == ".true." else "false"), "bool"
            if token[0] in "'\"":
                return json.dumps(token[1:-1]), "char"
            base, _, suffix = token.partition("_")
            kind = "double" if "d" in base or suffix == "kind_phys8" else "float" if "." in base or "e" in base or suffix == "kind_phys" else "int"
            value = base.replace("d", "e")
            if kind == "float":
                if "." not in value and "e" not in value:
                    value += ".0"
                value += "f"
            return value, kind
        if tag == "name":
            var = self.variables[tree[1]]
            code = tree[1]
            if self.scalar_index is not None and var.dimensions:
                code += ".linear(" + self.scalar_index + ")"
            return code, var.kind
        if tag == "unary":
            code, kind = self.emit(tree[2])
            op = "!" if tree[1] == ".not." else tree[1]
            return f"({op}({code}))", "bool" if op == "!" else kind
        if tag == "binary":
            _, op, left, right = tree
            a, ak = self.emit(left)
            b, bk = self.emit(right)
            kind = promote(ak, bk)
            comparisons = {".eq.": "==", ".ne.": "!=", ".gt.": ">", ".lt.": "<", ".ge.": ">=", ".le.": "<=", ".and.": "&&", ".or.": "||", ".eqv.": "==", ".neqv.": "!=", "/=": "!="}
            if op in comparisons or op in ("==", "<", ">", "<=", ">="):
                return f"(({a}) {comparisons.get(op, op)} ({b}))", "bool"
            if op == "**":
                if bk == "int" and right[0] == "literal":
                    power = int(right[1])
                    return f"mynn_ipow<{power}>({a})", ak
                return f"mynn_pow({a}, {b})", kind
            if self.constant or kind == "int":
                return f"(({a}) {op} ({b}))", kind
            helper = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]
            prefix = "mynn_d" if kind == "double" else "mpas_"
            return f"{prefix}{helper}({a}, {b})", kind
        if tag == "call":
            _, name, arguments = tree
            if name == "minloc":
                def array_name(node):
                    if node[0] == "name" and self.variables[node[1]].dimensions:
                        return node[1]
                    children = node[2:] if node[0] in ("binary", "unary") else node[2] if node[0] == "call" else []
                    for child in children:
                        if isinstance(child, tuple) and (found := array_name(child)):
                            return found
                    return None
                array = array_name(arguments[0])
                assert array and arguments[1] == ("literal", "1")
                previous = self.scalar_index
                self.scalar_index = "mynn_element"
                value, _ = self.emit(arguments[0])
                self.scalar_index = previous
                return f"mynn_minloc({array}.size(), [&] (int mynn_element) {{ return {value}; }})", "int"
            var = self.variables.get(name)
            if var is not None and var.dimensions:
                indices, ranges = [], False
                for dim, arg in enumerate(arguments):
                    if arg[0] == "range":
                        ranges = True
                        lo = self.emit(arg[1])[0] if arg[1] is not None else f"{name}.lo[{dim}]"
                        hi = self.emit(arg[2])[0] if arg[2] is not None else f"{name}.hi[{dim}]"
                        indices.append(f"MynnRange{{{lo}, {hi}}}")
                    else:
                        indices.append(self.emit(arg)[0])
                code = name + (".slice(" if ranges else "(") + ", ".join(indices) + ")"
                if ranges and self.scalar_index is not None:
                    code += f".linear({self.scalar_index})"
                return code, var.kind
            args = [self.emit(arg) for arg in arguments if not (arg[0] == "name" and arg[1] == "kind_phys")]
            if name == "present":
                return f"({args[0][0]}.present())", "bool"
            if name in ("real", "int", "dble"):
                kind = {"real": "float", "int": "int", "dble": "double"}[name]
                return f"(({kind})({args[0][0]}))", kind
            if name in ("maxval", "minval"):
                return f"mynn_{name}({args[0][0]})", args[0][1]
            if name in ("ceiling", "floor", "nint"):
                return f"mynn_{name}({args[0][0]})", "int"
            if name in self.routines:
                target = self.routines[name]
                for missing in target.arguments[len(args):]:
                    assert target.variables[missing].optional
                    args.append(("{}", target.variables[missing].kind))
                return "f_" + name + "(" + ", ".join(x[0] for x in args) + ")", target.variables[name].kind
            if name in ("min", "max"):
                kind = args[0][1]
                result = args[0][0]
                for value, vk in args[1:]:
                    kind = promote(kind, vk)
                    result = f"mynn_{name}(({kind})({result}), ({kind})({value}))"
                return result, kind
            if name in ("abs", "sqrt", "exp", "log", "log10", "atan", "tanh", "erf", "sign", "mod"):
                return f"mynn_{name}(" + ", ".join(x[0] for x in args) + ")", args[0][1]
            raise ValueError(("unknown function", name, tree))
        raise ValueError(tree)

    def is_array(self, tree):
        if tree[0] == "name":
            return bool(self.variables[tree[1]].dimensions)
        return tree[0] == "call" and tree[1] in self.variables and any(arg[0] == "range" for arg in tree[2])

    def assignment(self, lhs, rhs):
        tree = Expression.tree(lhs)
        if self.is_array(tree):
            view = self.emit(tree)[0]
            self.scalar_index = "mynn_element"
            a, _ = self.emit(tree)
            b, _ = self.expression(rhs)
            self.scalar_index = None
            if self.lane_elementwise:
                # lane_plan: elements split over the column's lanes (the right side does not read the array).
                return (f"mynn_lane_sync(); for (int mynn_element=lane_sub; mynn_element < ({view}).size(); "
                        f"mynn_element += lane_width) {{ {a} = {b}; }} mynn_lane_sync();")
            return f"for (int mynn_element=0; mynn_element < ({view}).size(); ++mynn_element) {{ {a} = {b}; }}"
        return self.expression(lhs)[0] + " = " + self.expression(rhs)[0] + ";"

    def statement(self, stmt):
        if match := re.match(r"select\s*case\s*\((.*)\)$", stmt):
            self.selects.append(False)
            return "switch (" + self.expression(match[1])[0] + ") {"
        if stmt.startswith("case"):
            prefix = "break; } " if self.selects[-1] else ""
            self.selects[-1] = True
            if stmt == "case default":
                return prefix + "default: {"
            match = re.fullmatch(r"case\s*\((.*)\)", stmt)
            return prefix + " ".join("case " + self.expression(x)[0] + ":" for x in split_top(match[1])) + " {"
        if stmt in ("endselect", "end select"):
            return ("break; } " if self.selects.pop() else "") + "}"
        if match := re.match(r"(\d+)\s+(.*)", stmt):
            return "mynn_label_" + match[1] + ": " + self.statement(match[2])
        if match := re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", stmt):
            return "} else if (" + self.expression(match[1])[0] + ") {"
        if stmt in ("else",):
            return "} else {"
        if stmt in ("endif", "end if", "enddo", "end do"):
            return "}"
        if stmt.startswith("if"):
            parser = Expression(stmt[2:].lstrip())
            parser.pop("(")
            condition = parser.parse()
            parser.pop(")")
            tail = " ".join(parser.tokens[parser.index:])
            # Reconstruct from the raw string to preserve quoted text and kinds.
            depth = 0
            for index, char in enumerate(stmt[stmt.index("("):], stmt.index("(")):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    tail = stmt[index+1:].strip()
                    break
            code = "if (" + self.emit(condition)[0] + ") "
            return code + ("{" if tail == "then" else "{ " + self.statement(tail) + " }")
        if match := re.match(r"do\s*while\s*\((.*)\)$", stmt):
            return "while (" + self.expression(match[1])[0] + ") {"
        if match := re.match(r"do\s+(\w+)\s*=\s*(.*)$", stmt):
            var, bounds = match[1], split_top(match[2])
            start, end = (self.expression(x)[0] for x in bounds[:2])
            if self.routine.name == "mynnedmf_driver" and var in ("i", "j") and bounds[:2] == (["its", "ite"] if var == "i" else ["jts", "jte"]):
                start = f"(active_{var} >= 0 ? active_{var} : {bounds[0]})"
                end = f"(active_{var} >= 0 ? active_{var} : {bounds[1]})"
            step = self.expression(bounds[2])[0] if len(bounds) == 3 else "1"
            return f"for ({var}={start}; ({step}) > 0 ? {var} <= ({end}) : {var} >= ({end}); {var} += ({step})) {{"
        if match := re.match(r"call\s+(\w+)\s*\((.*)\)$", stmt):
            target = self.routines[match[1]]
            supplied, positional = {}, 0
            for arg in split_top(match[2]):
                if keyword := re.match(r"(\w+)\s*=([^=].*)$", arg):
                    supplied[keyword[1]] = keyword[2]
                else:
                    supplied[target.arguments[positional]] = arg
                    positional += 1
            values = []
            for name in target.arguments:
                var = target.variables[name]
                if name in supplied:
                    arg = supplied[name]
                    arg = re.sub(r",\s*kind\s*=\s*kind_phys(?=\))", ",kind_phys", arg)
                    tree = Expression.tree(arg)
                    value = self.emit(tree)[0]
                    if var.dimensions and tree[0] == "name" and len(self.variables[tree[1]].dimensions) != len(var.dimensions):
                        # Original Fortran sequence association; in this pin
                        # the main driver passes its full 2D rthraten to DDMF.
                        lo, hi = Emitter(self.globals, self.routines, target).bounds(var)
                        for dummy, actual in supplied.items():
                            lo = re.sub(rf"\b{dummy}\b", actual, lo)
                            hi = re.sub(rf"\b{dummy}\b", actual, hi)
                        value += f".sequence<{len(var.dimensions)}>({lo}, {hi})"
                    values.append(value)
                elif var.optional:
                    values.append("{}")
                else:
                    raise ValueError((target.name, "missing argument", name, stmt))
            return f"f_{target.name}(" + ", ".join(values) + ");"
        if stmt == "return":
            return "return" + (" " + self.routine.name if self.routine.function else "") + ";"
        if stmt == "exit":
            return "break;"
        if stmt == "cycle":
            return "continue;"
        if stmt == "continue":
            return ";"
        if match := re.match(r"go\s*to\s+(\d+)$", stmt):
            return "goto mynn_label_" + match[1] + ";"
        if stmt.startswith(("print", "write")):
            return "; /* Native diagnostic print; no field arithmetic. */"
        if match := re.match(r"(.+?)\s*=\s*(.*)$", stmt):
            rhs = re.sub(r",\s*kind\s*=\s*kind_phys(?=\))", ",kind_phys", match[2])
            return self.assignment(match[1], rhs)
        raise ValueError((self.routine.name, stmt))

    def variable(self, var, argument=False):
        if var.dimensions:
            return f"MynnArray<{var.kind}, {len(var.dimensions)}> {var.name}"
        kind = "MynnText" if var.kind == "char" else var.kind
        if var.optional:
            kind = f"MynnOptional<{kind}>"
        written = var.intent in ("out", "inout") or (not var.intent and self.routine and any(re.match(rf"{var.name}\s*=", stmt) for _, stmt in self.routine.body))
        return kind + ("& " if argument and written else " ") + var.name

    def bounds(self, var):
        lows, highs = [], []
        for dim, bounds in enumerate(var.dimensions):
            edges = split_top(bounds, ":")
            if len(edges) == 1:
                lo, hi = "1", edges[0]
            else:
                lo, hi = edges
                lo = lo or "1"
                hi = hi or f"{var.name}.extent({dim}) + ({lo}) - 1"
            lows.append(self.expression(lo)[0])
            # Descriptor methods in assumed-shape upper bounds need no AST.
            highs.append(hi if ".extent(" in hi else self.expression(hi)[0])
        return "{" + ", ".join(lows) + "}", "{" + ", ".join(highs) + "}"

    def split_loop_header(self, loop, stmt):
        """A lane_plan loop: iteration it runs on lane it % lane_width with the serial loop's index value."""
        match = re.match(r"do\s+(\w+)\s*=\s*(.*)$", stmt)
        bounds = split_top(match[2])
        start, end = (self.expression(x)[0] for x in bounds[:2])
        step = self.expression(bounds[2])[0] if len(bounds) == 3 else "1"
        loop.uid = u = self.lane_uid
        self.lane_uid += 1
        return (f"{{ const int mynn_lo{u} = ({start}), mynn_hi{u} = ({end}), mynn_st{u} = ({step}); "
                f"const int mynn_n{u} = mynn_trip(mynn_lo{u}, mynn_hi{u}, mynn_st{u}); mynn_lane_sync(); "
                f"for (int mynn_it{u} = lane_sub; mynn_it{u} < mynn_n{u}; mynn_it{u} += lane_width) {{ "
                f"{loop.var} = mynn_lo{u} + mynn_it{u}*mynn_st{u};")

    def split_loop_footer(self, loop):
        """Join the lanes; the index takes its serial exit value, privates the last iteration's values."""
        u = loop.uid
        code = f"}} mynn_lane_sync(); {loop.var} = mynn_lo{u} + mynn_n{u}*mynn_st{u};"
        if loop.privates:
            code += (f" if (mynn_n{u} > 0) {{ const int mynn_src{u} = (mynn_n{u} - 1) % lane_width; "
                     + " ".join(f"{name} = mynn_lane_bcast({name}, mynn_src{u});" for name, _ in loop.privates)
                     + " }")
        return code + " }"

    def routine_code(self):
        routine = self.routine
        result = routine.variables[routine.name].kind if routine.function else "void"
        args = ", ".join(self.variable(routine.variables[name], True) for name in routine.arguments)
        output = [f"// RRFS 1d6e370e MPAS-Model-rrfs/{routine.source}:{routine.start}-{routine.end}", f"__device__ __noinline__ {result} f_{routine.name}({args}) {{", "MynnFrame mynn_frame(arena);"]
        for name in routine.arguments:
            var = routine.variables[name]
            if var.dimensions:
                lo, _ = self.bounds(var)
                output.append(f"{name}.rebase({lo});")
        for name, var in routine.variables.items():
            if name in routine.arguments:
                continue
            if var.dimensions:
                lo, hi = self.bounds(var)
                output.append(self.variable(var) + f" = mynn_array<{var.kind}, {len(var.dimensions)}, mynn_tile>(arena, {lo}, {hi});")
            else:
                self.constant = var.parameter
                initial = " = " + self.expression(var.initial)[0] if var.initial is not None else ""
                self.constant = False
                output.append(("constexpr " if var.parameter else "") + self.variable(var) + initial + ";")
        plan = self.lane_plan
        for index, (line, stmt) in enumerate(routine.body):
            try:
                if plan is not None and index in plan.loops:
                    code = self.split_loop_header(plan.loops[index], stmt)
                elif plan is not None and index in plan.ends:
                    code = self.split_loop_footer(plan.ends[index])
                else:
                    self.lane_elementwise = plan is not None and index in plan.elementwise
                    code = self.statement(stmt)
                    self.lane_elementwise = False
            except Exception as error:
                raise ValueError(f"{routine.source}:{line}: {stmt}") from error
            output.append(f"// MPAS-Model-rrfs/{routine.source}:{line}\n{code}")
        if routine.function:
            output.append(f"return {routine.name};")
        output.append("}")
        return "\n".join(output)


LANE_MEMBERS = r"""// Lane group of this column (lane_plan.py, rrfs/speed2-mynn-edmf): lane_width lanes run the column's scalar code
// together; split loops and array assignments give iteration it to lane it % lane_width. lane_width 1 is one
// thread per column. The width is a template argument: with it (and so the warp tile width and every automatic
// array's stride) known at compile time the kernel frame stays under 4 KiB (a run-time width spilled the strides:
// 4,544 B fast, over the 4 KiB at which device_memory_release gives the local-memory store back every phase).
static constexpr int lane_width = MYNN_LANES; static constexpr int mynn_tile = 32 / MYNN_LANES;
int lane_sub = 0; unsigned lane_mask = 0u;
__device__ __forceinline__ void mynn_lane_sync() const { __syncwarp(lane_mask); }
__device__ __forceinline__ int mynn_lane_bcast(int v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ float mynn_lane_bcast(float v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ double mynn_lane_bcast(double v, int src) const { return __shfl_sync(lane_mask, v, src, lane_width); }
__device__ __forceinline__ bool mynn_lane_bcast(bool v, int src) const { return __shfl_sync(lane_mask, (int)v, src, lane_width) != 0; }
// Trip count of the generated loop for (v=lo; st > 0 ? v <= hi : v >= hi; v += st), st a nonzero literal.
__device__ __forceinline__ static int mynn_trip(int lo, int hi, int st) {
    return st > 0 ? (hi >= lo ? (hi - lo)/st + 1 : 0) : (hi <= lo ? (lo - hi)/(-st) + 1 : 0);
}"""


def generate(paths, out, receipt):
    globals_, routines = parse_modules(paths)
    imported = {
        'r_d':'287.', 'r_v':'461.6', 'cp':'1004.5', 'cpv':'4.*r_v',
        'cliq':'4190.', 'cice':'2106.', 'grav':'9.80616', 'karman':'0.4',
        'p1000mb':'100000.', 'rcp':'r_d/cp', 'xlv':'2.50e6', 'xlf':'3.50e5',
        'xls':'xlv+xlf', 'svp1':'0.6112','svp2':'17.67','svp3':'29.65',
        'p608':'r_v/r_d-1.', 'ep_2':'r_d/r_v','rvovrd':'r_v/r_d','t0c':'273.15'}
    globals_ = {**{name:Variable(name,'float',parameter=True,initial=value)
                  for name,value in imported.items()}, **globals_}
    emitter = Emitter(globals_, routines)
    pieces = ["// Generated from pinned RRFS MYNN-EDMF c942418; see translation receipt.", "template<int MYNN_LANES> struct MynnColumn {", "MynnArena& arena;", "int active_i = -1; int active_j = -1;",
              LANE_MEMBERS]
    for var in globals_.values():
        emitter.constant = True
        initial = " = " + emitter.expression(var.initial)[0] if var.initial is not None else ""
        pieces.append(("static constexpr " if var.parameter else "") + emitter.variable(var) + initial + ";")
    from lane_plan import alias_classes, lane_safe_routines, plan_routine
    lane_safe = lane_safe_routines(globals_, routines)
    alias_find = alias_classes(globals_, routines, "mynnedmf_driver")
    plans = {}
    for routine in routines.values():
        routine_emitter = Emitter(globals_, routines, routine)
        routine_emitter.lane_plan = plans[routine.name] = plan_routine(routine_emitter, routine, lane_safe, alias_find)
        pieces.append(routine_emitter.routine_code())
    pieces.append("};")
    text = "\n".join(pieces) + "\n"
    out.write_text(text)
    result = {"scope": "whole-module mechanical translation; compilation and numerical proof separate",
              "preprocessor": {"WRF_CHEM": 0},
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "lane_plan": {"lane_safe_routines": sorted(lane_safe),
                            "split_loops": {n: [routines[n].body[i][0] for i in sorted(p.loops)] for n, p in plans.items() if p.loops},
                            "split_array_assignments": {n: [routines[n].body[i][0] for i in sorted(p.elementwise)] for n, p in plans.items() if p.elementwise},
                            "serial_loops": [{"routine": n, "line": line, "reason": why} for p in plans.values() for n, line, why in p.serial_reasons]},
              "routines": [{"name": r.name, "file": r.source, "lines": [r.start, r.end],
                            "arguments": r.arguments, "statements": len(r.body)} for r in routines.values()]}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return text, routines

