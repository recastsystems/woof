"""Lane plan for the RRFS MYNN-EDMF translation: which statements run lane-strided (rrfs/speed2-mynn-edmf, 2026-10-28).

Execution shape: each column is run by a group of W lanes of one warp (W = 1, 2, ..., 32, chosen per launch from
the column count; W = 1 is the old one-thread-per-column shape). Every lane of a group runs the column's scalar code
together, on identical values. Two kinds of statement are split across the group's lanes instead:

- a DO loop whose iterations are provably independent: iteration ``it`` runs on lane ``it % W``;
- an array assignment (``a(lo:hi) = ...``, ``a = ...``) whose right side does not read the array it writes.

Every operation and its operands are those of the serial loop, so the words are the same in fast and strict
numerics; only which lane performs an iteration changes. A loop is split only when ALL of these hold (anything
the planner cannot prove keeps the loop serial):

1. ``do v = lo, hi[, literal step]`` over an integer v; no ``exit`` that leaves it, no ``goto``, label or ``return``
   anywhere in it; v and every name in its bounds are not written inside it.
2. Every routine it calls (statement or function) is lane-safe: no array arguments or automatic arrays (so no
   arena allocation), writes no module variable, and calls only lane-safe routines.
3. Every array it writes is written and read only at one element per iteration along one dimension: the index there
   is ``v``, ``v + c`` or ``v - c`` with c invariant in the loop, the same text at every reference. Dummy arrays
   that may share storage (the entry routine's arguments, and any dummies a call site binds to one array) form one
   alias class and must all be referenced at that same index.
4. No scalar it writes is read before it is assigned in the same iteration (definite assignment over the
   iteration's IF/ELSE/SELECT structure). A scalar assigned on every path through the iteration (CYCLE included)
   takes the last iteration's value after the loop, broadcast from the lane that ran it; one assigned on only some
   paths must be referenced nowhere else in the routine.
5. It writes at least one array (a loop that only computes scalars is left serial).

After a split loop the group synchronises (``__syncwarp`` on the group's lanes) and the loop variable is set to its
serial exit value. Loops nested in a split loop run serially on their lane. The generation receipt lists every
split loop and every loop left serial with its reason. MYNN_LANE_PLAN_SERIAL=routine[,routine] at generation keeps
the named routines serial (bisecting a suspected split).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


def _tm():
    import translate_module
    return translate_module


KIND_RHS = re.compile(r",\s*kind\s*=\s*kind_phys(?=\))")
BROADCAST_KINDS = ("int", "float", "double", "bool")


@dataclass
class Event:
    kind: str            # 'r' read, 'w' write, 'f' routine call
    name: str
    args: tuple | None   # index trees for an array element/section reference; None for scalar or whole array
    chain: tuple         # block ids from the loop body root to the statement
    loop_level: bool     # not inside a loop nested in the analysed loop
    after_cycle: bool    # a loop-level CYCLE precedes this event in the body
    order: int


@dataclass
class LoopPlan:
    start: int
    end: int
    var: str
    privates: list = field(default_factory=list)   # (name, kind) broadcast from the last iteration's lane


@dataclass
class RoutinePlan:
    loops: dict = field(default_factory=dict)        # body index of DO -> LoopPlan
    ends: dict = field(default_factory=dict)         # body index of ENDDO -> LoopPlan
    elementwise: set = field(default_factory=set)    # body indices of array assignments split across lanes
    serial_reasons: list = field(default_factory=list)


def classify(stmt):
    """Statement kind, in the dispatch order of translate_module.Emitter.statement."""
    if m := re.match(r"select\s*case\s*\((.*)\)$", stmt):
        return ("select", m[1])
    if stmt.startswith("case"):
        return ("case",)
    if stmt in ("endselect", "end select"):
        return ("endselect",)
    if m := re.match(r"(\d+)\s+(.*)", stmt):
        return ("label", m[1], m[2])
    if m := re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", stmt):
        return ("elseif", m[1])
    if stmt == "else":
        return ("else",)
    if stmt in ("endif", "end if"):
        return ("endif",)
    if stmt in ("enddo", "end do"):
        return ("enddo",)
    if stmt.startswith("if"):
        open_ = stmt.index("(")
        depth = 0
        for index, char in enumerate(stmt[open_:], open_):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                cond, tail = stmt[open_ + 1:index], stmt[index + 1:].strip()
                break
        return ("ifthen", cond) if tail == "then" else ("if1", cond, tail)
    if m := re.match(r"do\s*while\s*\((.*)\)$", stmt):
        return ("dowhile", m[1])
    if m := re.match(r"do\s+(\w+)\s*=\s*(.*)$", stmt):
        return ("do", m[1], _tm().split_top(m[2]))
    if m := re.match(r"call\s+(\w+)\s*\((.*)\)$", stmt):
        return ("call", m[1], m[2])
    if stmt in ("return", "exit", "cycle", "continue"):
        return (stmt,)
    if re.match(r"go\s*to\s+(\d+)$", stmt):
        return ("goto",)
    if stmt.startswith(("print", "write")):
        return ("print",)
    if m := re.match(r"(.+?)\s*=\s*(.*)$", stmt):
        return ("assign", m[1], KIND_RHS.sub(",kind_phys", m[2]))
    raise ValueError(stmt)


def matching_ends(body):
    """Body index of each DO / DO WHILE -> index of its ENDDO."""
    stack, ends = [], {}
    for index, (_, stmt) in enumerate(body):
        kind = classify(stmt)[0]
        if kind in ("do", "dowhile"):
            stack.append(("do", index))
        elif kind in ("ifthen", "select"):
            stack.append((kind, index))
        elif kind in ("enddo", "endif", "endselect"):
            opener, start = stack.pop()
            assert (opener, kind) in (("do", "enddo"), ("ifthen", "endif"), ("select", "endselect")), (body[start], body[index])
            if opener == "do":
                ends[start] = index
    assert not stack
    return ends


def written_dummy(routine, name):
    """As Emitter.variable: intent out/inout, or no intent and assigned in the body."""
    var = routine.variables[name]
    return var.intent in ("out", "inout") or (
        not var.intent and any(re.match(rf"{name}\s*=", stmt) for _, stmt in routine.body))


class Walker:
    def __init__(self, variables, routines):
        self.variables = variables
        self.routines = routines

    def is_array(self, name):
        var = self.variables.get(name)
        return var is not None and bool(var.dimensions)

    def refs(self, tree, out):
        """Append ('r', name, args) reads, ('w', ...) writes through called routines, ('f', name) calls."""
        tag = tree[0]
        if tag == "literal":
            return
        if tag == "name":
            out.append(("r", tree[1], None))
            return
        if tag == "unary":
            self.refs(tree[2], out)
            return
        if tag == "binary":
            self.refs(tree[2], out)
            self.refs(tree[3], out)
            return
        if tag == "range":
            for sub in tree[1:]:
                if sub is not None:
                    self.refs(sub, out)
            return
        if tag == "call":
            name, args = tree[1], tree[2]
            if self.is_array(name):
                for arg in args:
                    self.refs(arg, out)
                out.append(("r", name, tuple(args)))
                return
            if name in self.routines:
                out.append(("f", name, None))
                actual = [a for a in args if not (a[0] == "name" and a[1] == "kind_phys")]
                target = self.routines[name]
                for dummy, arg in zip(target.arguments, actual):
                    self.refs(arg, out)
                    if written_dummy(target, dummy):
                        self.target_write(arg, out)
                return
            for arg in args:
                self.refs(arg, out)
            return
        raise ValueError(tree)

    def target_write(self, tree, out):
        if tree[0] == "name":
            out.append(("w", tree[1], None))
        elif tree[0] == "call" and self.is_array(tree[1]):
            out.append(("w", tree[1], tuple(tree[2])))
        else:
            out.append(("w", "<expression>", None))

    def statement_refs(self, kind, out):
        """Reads/writes of one classified simple statement, in evaluation order."""
        tree = _tm().Expression.tree
        if kind[0] == "assign":
            lhs, rhs = tree(kind[1]), tree(kind[2])
            self.refs(rhs, out)
            if lhs[0] == "call" and self.is_array(lhs[1]):
                for arg in lhs[2]:
                    self.refs(arg, out)
                out.append(("w", lhs[1], tuple(lhs[2])))
            elif lhs[0] == "name":
                out.append(("w", lhs[1], None))
            else:
                raise ValueError(kind)
        elif kind[0] == "call":
            target = self.routines[kind[1]]
            out.append(("f", kind[1], None))
            supplied, positional = {}, 0
            for arg in _tm().split_top(kind[2]):
                if keyword := re.match(r"(\w+)\s*=([^=].*)$", arg):
                    supplied[keyword[1]] = keyword[2]
                else:
                    supplied[target.arguments[positional]] = arg
                    positional += 1
            for name in target.arguments:
                if name in supplied:
                    arg = tree(KIND_RHS.sub(",kind_phys", supplied[name]))
                    self.refs(arg, out)
                    if written_dummy(target, name):
                        self.target_write(arg, out)


def lane_safe_routines(globals_, routines):
    """Routines a lane may call on its own inside a split loop: no arrays, no module writes, lane-safe callees."""
    module_vars = {n for n, v in globals_.items() if not v.parameter}
    safe = {}
    for name, routine in routines.items():
        safe[name] = not any(v.dimensions for v in routine.variables.values())
    changed = True
    while changed:
        changed = False
        for name, routine in routines.items():
            if not safe[name]:
                continue
            walker = Walker({**globals_, **routine.variables}, routines)
            ok = True
            for _, stmt in routine.body:
                events = []
                try:
                    collect_simple(walker, classify(stmt), events)
                except ValueError:
                    ok = False
                    break
                for kind, ref, _ in events:
                    if kind == "f" and not safe.get(ref, False):
                        ok = False
                    if kind == "w" and (ref in module_vars and ref not in routine.variables):
                        ok = False
            if not ok:
                safe[name] = False
                changed = True
    return {n for n, ok in safe.items() if ok}


def collect_simple(walker, kind, out):
    """Reads/writes of a statement's own expressions (conditions, bounds, simple statements)."""
    tree = _tm().Expression.tree
    tag = kind[0]
    if tag in ("assign", "call"):
        walker.statement_refs(kind, out)
    elif tag in ("ifthen", "elseif", "dowhile", "select"):
        walker.refs(tree(kind[1]), out)
    elif tag == "if1":
        walker.refs(tree(kind[1]), out)
        collect_simple(walker, classify(kind[2]), out)
    elif tag == "label":
        collect_simple(walker, classify(kind[2]), out)
    elif tag == "do":
        for bound in kind[2]:
            walker.refs(tree(bound), out)
        out.append(("w", kind[1], None))


def alias_classes(globals_, routines, entry):
    """Dummy arrays that may share storage: the entry's array arguments (bound by the host), and dummies a call
    site binds to arrays of one class. Automatic arrays are their own class."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    run = routines[entry]
    arrays = [n for n in run.arguments if run.variables[n].dimensions]
    for n in arrays[1:]:
        union((entry, arrays[0]), (entry, n))
    changed = True
    while changed:
        changed = False
        for caller in routines.values():
            walker = Walker({**globals_, **caller.variables}, routines)
            for _, stmt in caller.body:
                kind = classify(stmt)
                while kind[0] in ("if1", "label"):
                    kind = classify(kind[2])
                calls = []
                if kind[0] == "call":
                    target = routines[kind[1]]
                    supplied, positional = {}, 0
                    for arg in _tm().split_top(kind[2]):
                        if keyword := re.match(r"(\w+)\s*=([^=].*)$", arg):
                            supplied[keyword[1]] = keyword[2]
                        else:
                            supplied[target.arguments[positional]] = arg
                            positional += 1
                    calls.append((target, {d: _tm().Expression.tree(KIND_RHS.sub(",kind_phys", a)) for d, a in supplied.items()}))
                for target, bound in calls:
                    groups = {}
                    for dummy, tree in bound.items():
                        if not target.variables[dummy].dimensions:
                            continue
                        base = tree[1] if tree[0] in ("name", "call") else None
                        if base is None or not walker.is_array(base):
                            continue
                        key = find((caller.name, base)) if base in caller.arguments else (caller.name, base)
                        groups.setdefault(key, []).append(dummy)
                    for members in groups.values():
                        for m in members[1:]:
                            if find((target.name, m)) != find((target.name, members[0])):
                                union((target.name, m), (target.name, members[0]))
                                changed = True
    return find


class LoopAnalysis:
    def __init__(self, emitter, routine, lane_safe, alias_find):
        self.routine = routine
        self.variables = emitter.variables
        self.walker = Walker(emitter.variables, emitter.routines)
        self.lane_safe = lane_safe
        self.alias_find = alias_find
        self.module_vars = {n for n, v in emitter.globals.items() if not v.parameter}

    def array_class(self, name):
        if name in self.routine.arguments:
            return self.alias_find((self.routine.name, name))
        return ("auto", self.routine.name, name)

    def events(self, body, start, end):
        """Events of body[start+1:end] (the loop body) with block chains."""
        out, chain, ids, frames = [], [], [0], []
        state = {"cycle": False, "order": 0, "flags": set()}

        def new_id():
            ids[0] += 1
            return ids[0]

        def emit(raw, depth_inner_loop):
            for kind, name, args in raw:
                out.append(Event(kind, name, args, tuple(chain), depth_inner_loop == 0, state["cycle"], state["order"]))
                state["order"] += 1

        def simple(kind, inner_loops):
            tag = kind[0]
            if tag == "if1":
                raw = []
                self.walker.refs(_tm().Expression.tree(kind[1]), raw)
                emit(raw, inner_loops)
                chain.append(new_id())
                simple(classify(kind[2]), inner_loops)
                chain.pop()
                return
            if tag == "label":
                state["flags"].add("label")
                simple(classify(kind[2]), inner_loops)
                return
            if tag in ("goto", "return"):
                state["flags"].add(tag)
                return
            if tag == "exit":
                if inner_loops == 0:
                    state["flags"].add("exit")
                return
            if tag == "cycle":
                if inner_loops == 0:
                    state["cycle"] = True
                return
            if tag in ("continue", "print", "case"):
                return
            raw = []
            collect_simple(self.walker, kind, raw)
            emit(raw, inner_loops)

        inner_loops = 0
        for index in range(start + 1, end):
            kind = classify(body[index][1])
            tag = kind[0]
            if tag in ("do", "dowhile"):
                raw = []
                collect_simple(self.walker, kind, raw)
                emit(raw, inner_loops)
                frames.append("do")
                inner_loops += 1
                chain.append(new_id())
            elif tag == "ifthen":
                raw = []
                collect_simple(self.walker, kind, raw)
                emit(raw, inner_loops)
                frames.append("if")
                chain.append(new_id())
            elif tag == "select":
                raw = []
                collect_simple(self.walker, kind, raw)
                emit(raw, inner_loops)
                frames.append("select")
                chain.append(new_id())
            elif tag in ("elseif", "else", "case"):
                chain[-1] = new_id()
                if tag == "elseif":
                    raw = []
                    collect_simple(self.walker, kind, raw)
                    emit(raw, inner_loops)
            elif tag in ("enddo", "endif", "endselect"):
                if frames.pop() == "do":
                    inner_loops -= 1
                chain.pop()
            else:
                simple(kind, inner_loops)
        return out, state["flags"]

    def structure(self, body, start, end):
        """Nested nodes of body[start+1:end]: ('s', kind), ('if', [branch nodes...], has_else),
        ('select', head kind, [case nodes...], has_default), ('loop', head kind, nodes)."""
        root = []
        stack = [("root", root)]
        def current():
            top = stack[-1]
            if top[0] in ("root", "loop"):
                return top[1]
            return top[1][1][-1] if top[0] == "if" else top[1][2][-1]

        for index in range(start + 1, end):
            stmt = body[index][1]
            kind = classify(stmt)
            tag = kind[0]
            if tag in ("do", "dowhile"):
                node = ("loop", kind, [])
                current().append(node)
                stack.append(("loop", node[2]))
            elif tag == "ifthen":
                node = ("if", [[("s", ("cond", kind[1]))]], [False])
                current().append(node)
                stack.append(("if", node))
            elif tag in ("elseif", "else"):
                node = stack[-1][1]
                node[1].append([("s", ("cond", kind[1]))] if tag == "elseif" else [])
                if tag == "else":
                    node[2][0] = True
            elif tag == "select":
                node = ("select", kind, [], [False])
                current().append(node)
                stack.append(("select", node))
            elif tag == "case":
                node = stack[-1][1]
                node[2].append([])
                if stmt.replace(" ", "") == "casedefault":
                    node[3][0] = True
            elif tag in ("enddo", "endif", "endselect"):
                stack.pop()
            else:
                current().append(("s", kind))
        assert len(stack) == 1
        return root

    def definite_assignment(self, body, start, end, names):
        """(names read before assignment in an iteration, names assigned on every path through an iteration)."""
        exposed = set()
        exits = []   # D at every way out of an iteration (CYCLE or the end)
        tree = _tm().Expression.tree

        def reads_then_writes(kind, defined):
            raw = []
            if kind[0] == "cond":
                self.walker.refs(tree(kind[1]), raw)
            else:
                collect_simple(self.walker, kind, raw)
            for k, name, _ in raw:
                if k == "r" and name in names and name not in defined:
                    exposed.add(name)
                elif k == "w" and name in names:
                    defined.add(name)

        def walk(nodes, defined, depth):
            """Returns the definite set after nodes, or None if no path falls through."""
            for node in nodes:
                if defined is None:
                    return None
                if node[0] == "s":
                    kind = node[1]
                    while kind[0] in ("if1", "label"):
                        if kind[0] == "if1":
                            reads_then_writes(("cond", kind[1]), defined)
                            inner = walk([("s", classify(kind[2]))], set(defined), depth)
                            if inner is not None:
                                defined &= inner
                            kind = None
                            break
                        kind = classify(kind[2])
                    if kind is None:
                        continue
                    if kind[0] == "cycle":
                        if depth == 0:
                            exits.append(set(defined))
                        return None
                    if kind[0] in ("exit", "return", "goto"):
                        return None
                    if kind[0] in ("assign", "call", "cond"):
                        reads_then_writes(kind, defined)
                elif node[0] == "if":
                    outs = []
                    for branch in node[1]:
                        outs.append(walk(branch, set(defined), depth))
                    if not node[2][0]:
                        outs.append(set(defined))
                    live = [o for o in outs if o is not None]
                    defined = set.intersection(*live) if live else None
                elif node[0] == "select":
                    reads_then_writes(("cond", node[1][1]), defined)
                    outs = [walk(case, set(defined), depth) for case in node[2]]
                    if not node[3][0]:
                        outs.append(set(defined))
                    live = [o for o in outs if o is not None]
                    defined = set.intersection(*live) if live else None
                elif node[0] == "loop":
                    head = node[1]
                    if head[0] == "do":
                        reads_then_writes(head, defined)
                    else:
                        reads_then_writes(("cond", head[1]), defined)
                    walk(node[2], set(defined), depth + 1)
            return defined

        final = walk(self.structure(body, start, end), set(), 0)
        if final is not None:
            exits.append(final)
        definite = set.intersection(*exits) if exits else set()
        return exposed, definite

    def affine(self, tree, var, variant):
        """Index tree is var, var + c, c + var or var - c with c invariant -> canonical key, else None."""
        def invariant(node):
            raw = []
            self.walker.refs(node, raw)
            return all(not (k == "r" and n in variant) and k != "f" for k, n, _ in raw)
        if tree == ("name", var):
            return repr(tree)
        if tree[0] == "binary" and tree[1] in ("+", "-"):
            if tree[2] == ("name", var) and invariant(tree[3]):
                return repr(tree)
            if tree[1] == "+" and tree[3] == ("name", var) and invariant(tree[2]):
                return repr(tree)
        return None

    def loop(self, body, start, end):
        """LoopPlan if the loop body[start] .. body[end] may be split across lanes, else a reason string."""
        tm = _tm()
        kind = classify(body[start][1])
        if kind[0] != "do":
            return "do while"
        var, bounds = kind[1], kind[2]
        if self.variables.get(var) is None or self.variables[var].kind != "int" or self.variables[var].dimensions:
            return "non-integer loop variable"
        if self.routine.name == "mynnedmf_driver" and var in ("i", "j"):
            return "column loop"
        if len(bounds) == 3 and not re.fullmatch(r"-?\s*\d+", bounds[2].strip()):
            return "non-literal step"
        if len(bounds) == 3 and int(bounds[2].replace(" ", "")) == 0:
            return "zero step"
        events, flags = self.events(body, start, end)
        if flags:
            return "control flow: " + ",".join(sorted(flags))
        for e in events:
            if e.kind == "f" and e.name not in self.lane_safe:
                return f"calls {e.name}"
        writes = [e for e in events if e.kind == "w"]
        written = {e.name for e in writes}
        if "<expression>" in written:
            return "writes through an expression"
        if var in written:
            return "writes its loop variable"
        bound_names = []
        for bound in bounds:
            raw = []
            self.walker.refs(tm.Expression.tree(bound), raw)
            bound_names += [n for _, n, _ in raw]
        if set(bound_names) & written:
            return "writes a bound"
        array_writes = [e for e in writes if self.walker.is_array(e.name)]
        if not array_writes:
            return "writes no array"
        scalar_written = {e.name for e in writes if not self.walker.is_array(e.name)}
        variant = scalar_written | {e.name for e in array_writes} | {var}
        # scalars: no read may see a value from an earlier iteration (or from before the loop); a scalar assigned on
        # every path through the iteration takes the last iteration's value after the loop (broadcast); one assigned
        # on only some paths must be referenced nowhere else in the routine.
        exposed, definite = self.definite_assignment(body, start, end, scalar_written)
        privates = []
        for name in sorted(scalar_written):
            v = self.variables.get(name)
            if v is None or v.kind not in BROADCAST_KINDS or v.optional:
                return f"writes non-broadcast scalar {name}"
            if name in exposed:
                return f"carries scalar {name}"
            if name in definite:
                privates.append((name, v.kind))
                continue
            elsewhere = (name in self.routine.arguments or name in self.module_vars or any(
                re.search(rf"\b{name}\b", s) for i, (_, s) in enumerate(self.routine.body) if not start <= i <= end))
            if elsewhere:
                return f"assigns scalar {name} on some paths and it is used outside the loop"
        # arrays: one element per iteration along one dimension, per alias class
        classes = {}
        for e in events:
            if e.kind in ("r", "w") and self.walker.is_array(e.name):
                classes.setdefault(self.array_class(e.name), []).append(e)
        for cls, refs in classes.items():
            if not any(e.kind == "w" for e in refs):
                continue
            position, key = {}, None
            for e in refs:
                if e.kind != "w":
                    continue
                if e.args is None:
                    return f"writes whole array {e.name}"
                found = None
                for p, arg in enumerate(e.args):
                    if arg[0] != "range" and (k := self.affine(arg, var, variant)) is not None:
                        found = (p, k)
                        break
                if found is None:
                    return f"writes {e.name} off the loop index"
                if key is None:
                    key = found[1]
                if found[1] != key or position.setdefault(e.name, found[0]) != found[0]:
                    return f"writes {e.name} at two offsets"
            for e in refs:
                if e.args is None:
                    return f"reads whole array {e.name} it writes"
                if e.name in position:
                    p = position[e.name]
                    if p >= len(e.args) or e.args[p][0] == "range" or self.affine(e.args[p], var, variant) != key:
                        return f"reads {e.name} at another offset"
                elif not any(a[0] != "range" and self.affine(a, var, variant) == key for a in e.args):
                    return f"reads {e.name} (aliased) at another offset"
        return LoopPlan(start, end, var, privates)

    def elementwise(self, kind):
        """An array assignment whose right side does not read the array (class) it writes."""
        while kind[0] in ("if1", "label"):
            kind = classify(kind[2])
        if kind[0] != "assign":
            return False
        tm = _tm()
        lhs = tm.Expression.tree(kind[1])
        array = (lhs[0] == "name" and self.walker.is_array(lhs[1])) or (
            lhs[0] == "call" and self.walker.is_array(lhs[1]) and any(a[0] == "range" for a in lhs[2]))
        if not array:
            return False
        raw = []
        self.walker.statement_refs(kind, raw)
        target = self.array_class(lhs[1])
        for k, name, _ in raw:
            if k == "f" and name not in self.lane_safe:
                return False
            if k == "w" and name != lhs[1]:
                return False
            if k == "r" and self.walker.is_array(name) and self.array_class(name) == target:
                return False
        return True


def plan_routine(emitter, routine, lane_safe, alias_find):
    plan = RoutinePlan()
    import os
    if routine.name in lane_safe or routine.name in os.environ.get("MYNN_LANE_PLAN_SERIAL", "").split(","):
        return plan
    analysis = LoopAnalysis(emitter, routine, lane_safe, alias_find)
    body = routine.body
    ends = matching_ends(body)
    index = 0
    skip_to = -1
    # walk in order; a split loop's body is skipped (its statements run serially on their lane)
    while index < len(body):
        if index <= skip_to:
            index += 1
            continue
        kind = classify(body[index][1])
        if kind[0] == "do" and index in ends:
            result = analysis.loop(body, index, ends[index])
            if isinstance(result, LoopPlan):
                plan.loops[index] = result
                plan.ends[ends[index]] = result
                skip_to = ends[index]
            else:
                plan.serial_reasons.append((routine.name, body[index][0], result))
        elif kind[0] in ("assign", "if1", "label") and analysis.elementwise(kind):
            plan.elementwise.add(index)
        index += 1
    return plan
