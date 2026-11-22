"""Lay the routines' private column arrays out as a call stack.

The lowering gives every routine its own disjoint arena slice, so one
column touches ~83 KB of distinct arena per call although no two routines
that are not caller and callee are ever active together. Every dirty line of
a routine that has returned is dead but still written back to DRAM, and the
next routine starts on cold lines. Native WRF places these arrays on its
stack, where a routine reuses the memory its finished siblings used. This
pass does the same: a routine's frame (its arrays, in their native order
and sizes) starts where the deepest frame of any of its callers ends, so
a frame never overlaps a frame that is live at the same time (its callers',
transitively) and siblings share memory.

Values never flow through a dead frame: a routine's private array has no
value on entry (Fortran locals without SAVE), and the strict replay of
real native columns is the proof that none is read before written.

Offsets stay affine in nz (both lowerings' sizes are carried), and the
returned arena size replaces the schemas' ``scratch_bytes``.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deadstore import _routine_spans  # noqa: E402

DECL = re.compile(r'M471Ptr<(\w+)> (\w+)=m471_array<\w+>\(m471_work,([^;]*),m471_n\);')


def _at(expr, nz):
    assert re.fullmatch(r'[\d\s()*+\-MYNN471_Z]+', expr), expr
    return eval(expr.replace('MYNN471_NZ', str(nz)), {'__builtins__': {}}, {})


def _affine(v49, v50):
    return f'({v50 - v49}*(MYNN471_NZ-49)+{v49})' if v50 != v49 else str(v49)


def overlay(text, scratch49, scratch50):
    spans = _routine_spans(text)
    arrays = []                               # (routine, decl match span, typ, off49, off50)
    for name, a, b in spans:
        for m in DECL.finditer(text, a, b):
            arrays.append([name, m.start(3), m.end(3), m[1], _at(m[3], 49), _at(m[3], 50)])
    assert arrays and not any(t[3] == 'doublereal' for t in arrays)
    order = sorted(arrays, key=lambda t: t[4])
    assert [t[5] for t in order] == sorted(t[5] for t in order)
    for here, nxt in zip(order, order[1:] + [None]):
        here.append((nxt[4] if nxt else scratch49) - here[4])
        here.append((nxt[5] if nxt else scratch50) - here[5])
        assert here[6] > 0 and here[7] >= here[6]
    calls = {}
    for name, a, b in spans:
        calls[name] = sorted(set(re.findall(r'\b(\w+_)\(m471_work', text[a:b])) - {name})
    callers = {name: [c for c, cs in calls.items() if name in cs] for name in calls}
    frame = {name: [0, 0] for name in calls}
    local = {}
    for t in sorted(arrays, key=lambda t: t[4]):
        local[id(t)] = tuple(frame[t[0]])
        frame[t[0]][0] += t[6]; frame[t[0]][1] += t[7]
    base = {}

    def resolve(name, seen=()):
        if name in base:
            return base[name]
        assert name not in seen, 'recursive call graph'
        ups = callers[name]
        if name == 'mynnedmf_' or not ups:
            ends = [(0, 0)] if name == 'mynnedmf_' else [tuple(x + y for x, y in zip(resolve('mynnedmf_'), frame['mynnedmf_']))]
        else:
            ends = [tuple(x + y for x, y in zip(resolve(c, seen + (name,)), frame[c])) for c in ups]
        best49 = max(e[0] for e in ends); best50 = max(e[1] for e in ends)
        assert (best49, best50) in ends, (name, ends)     # one deepest caller at both nz
        base[name] = (best49, best50)
        return base[name]
    for name in calls:
        resolve(name)
    edits = []
    for t in arrays:
        l49, l50 = local[id(t)]
        b49, b50 = base[t[0]]
        edits.append((t[1], t[2], _affine(b49 + l49, b50 + l50)))
    for start, end, expr in sorted(edits, reverse=True):
        text = text[:start] + expr + text[end:]
    tops = [(base[n][0] + frame[n][0], base[n][1] + frame[n][1]) for n in calls]
    top49 = max(t[0] for t in tops); top50 = max(t[1] for t in tops)
    assert (top49, top50) in tops
    report = dict(arena_before=(scratch49, scratch50), arena_after=(top49, top50),
                  frames={n: dict(base=base[n], bytes=tuple(frame[n])) for n in calls if frame[n][0]})
    return text, top49, top50, report
