"""Skip assignments to routine-private values that no enabled statement reads.

Native Fortran keeps routine-private arrays and scalars on the stack, and
the compiler deletes work nothing reads. The lowering moves private arrays
into the escaping column arena, so the CUDA compiler must keep every store,
and each is a DRAM write (the arena is far larger than L2). This pass does
on the generated text what the native compiler does:

* A routine-private value (an ``m471_array`` array declared in the routine,
  or a scalar local) is *live* when an enabled statement reads it: any use
  outside a literal ``if (FALSE_)`` block that is not inside an assignment
  to another private value, plus every use inside an assignment whose target
  is live (iterated to a fixed point). Assignments to values that are not
  live are removed.
* For each routine option ``*opt`` that the routine never writes, the same
  analysis with every ``if (*opt > 0)`` / ``if (*opt)`` block disabled names
  the values that only those blocks need; their assignments are guarded by
  that same test.

Only statement-level ``x = ...;`` / ``x op= ...;`` statements (``x`` a
private scalar or ``x[...]`` a private array element) are ever removed or
guarded, and never one whose right-hand side calls a routine. Their results
are unobservable, so every output word is unchanged.
"""
import re

ROUTINE = re.compile(r'__device__ M471_CALL (?:/\* Subroutine \*/ )?(?:int|integer|real|doublereal) (\w+)\(')
TYPES = r'(?:real|integer|logical|doublereal)'


def _strip(text):
    return re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m[0]), text, flags=re.S)


def _routine_spans(text):
    spans = []
    for m in ROUTINE.finditer(text):
        name = m[1]
        close = text.find('} /* ' + name + ' */', m.end())
        semicolon = text.find(';', m.end()); brace = text.find('{', m.end())
        if close < 0 or semicolon < brace:
            continue                                  # a prototype
        spans.append((name, brace + 1, close))
    return spans


def _blocks(clean):
    out = []; stack = []
    for i, ch in enumerate(clean):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            a = stack.pop()
            start = max(clean.rfind(c, 0, a) for c in ';{}')
            out.append((a, i, re.sub(r'\s+', ' ', clean[start + 1:a]).strip()))
    assert not stack
    return sorted(out)


def _end(clean, pos):
    level = 0
    for i in range(pos, len(clean)):
        ch = clean[i]
        if ch in '([':
            level += 1
        elif ch in ')]':
            level -= 1
        elif ch == ';' and level == 0:
            return i
    raise ValueError('statement without end')


def _analyse(body):
    clean = _strip(body)
    code_at = body.find('/* Function Body */')
    if code_at < 0:
        return None
    blocks = _blocks(clean)
    arrays = {m[1] for m in re.finditer(r'M471Ptr<\w+> (\w+)=m471_array<\w+>\(m471_work,', clean)}
    scalars = set()
    for m in re.finditer(r'^\s*' + TYPES + r'\s+([^;{}()=]+);', clean[:code_at], flags=re.M):
        for item in m[1].split(','):
            item = item.strip()
            if re.fullmatch(r'\w+', item):
                scalars.add(item)
    candidates = arrays | scalars
    if not candidates:
        return None
    statements = []
    for m in re.finditer(r'(?<![\w.>&*])(\w+)(\s*\[)?', clean[code_at:]):
        start = m.start() + code_at
        target = m[1]
        if target not in candidates:
            continue
        head_at = max(clean.rfind(c, 0, start) for c in ';{}')
        if clean[head_at + 1:start].strip():
            continue
        pos = m.end() + code_at
        if m[2]:
            if target not in arrays:
                continue
            level = 1
            while level:
                level += (clean[pos] == '[') - (clean[pos] == ']'); pos += 1
        elif target not in scalars:
            continue
        if not re.match(r'\s*[-+*/]?=(?!=)', clean[pos:]):
            continue
        end = _end(clean, start)
        if '(m471_work' in clean[start:end]:
            continue
        statements.append((start, end, target))
    statements.sort()
    starts = [s for s, _, _ in statements]

    def owner(pos):
        import bisect
        i = bisect.bisect_right(starts, pos) - 1
        if i >= 0 and statements[i][0] <= pos <= statements[i][1]:
            return i
        return None

    names = re.compile(r'(?<![\w.>])(' + '|'.join(sorted(candidates, key=len, reverse=True)) + r')\b')
    uses = []
    for m in names.finditer(clean, code_at):
        index = owner(m.start())
        if index is not None and statements[index][0] == m.start():
            continue                                  # the assignment's own target
        uses.append((m.start(), m[1], index))
    return dict(clean=clean, candidates=candidates, statements=statements, uses=uses, blocks=blocks)


def _enclosing(blocks, pos):
    return [h for a, b, h in blocks if a < pos < b]


def _live(info, disabled):
    statements = info['statements']
    off = [(a, b) for a, b, h in info['blocks'] if h in disabled]

    def enabled(pos):
        return not any(a < pos < b for a, b in off)
    live = set(); pending = {}
    for pos, name, index in info['uses']:
        if not enabled(pos):
            continue
        if index is None:
            live.add(name)
        else:
            pending.setdefault(statements[index][2], set()).add(name)
    changed = True
    while changed:
        changed = False
        for target, needs in pending.items():
            if target in live and not needs <= live:
                live |= needs; changed = True
    return live


def _options(info):
    tests = {}
    for a, b, h in info['blocks']:
        m = re.fullmatch(r'if \((\*(\w+)(?: > 0)?)\)', h)
        if m and not re.search(r'\*' + m[2] + r'\s*[-+*/]?=(?!=)', info['clean']):
            tests[h] = m[1]
    return tests


def _one(body):
    info = _analyse(body)
    if info is None:
        return body, {}
    base = _live(info, {'if (FALSE_)'})
    dead = info['candidates'] - base
    guard = {}
    for header, test in sorted(_options(info).items()):
        live = _live(info, {'if (FALSE_)', header})
        for name in sorted(base - live):
            guard.setdefault(name, test)
    edits = []; report = {}
    for start, end, target in info['statements']:
        enclosing = _enclosing(info['blocks'], start)
        if 'if (FALSE_)' in enclosing:
            continue
        if target in dead:
            edits.append((start, end, None))
            report.setdefault(target, dict(action='removed', statements=0))['statements'] += 1
        elif target in guard:
            test = guard[target]
            if 'if (' + test + ')' in enclosing:
                continue                              # already under its test
            edits.append((start, end, test))
            report.setdefault(target, dict(action='guarded by ' + test, statements=0))['statements'] += 1
    out = body
    for start, end, test in sorted(edits, reverse=True):
        statement = out[start:end + 1]
        out = out[:start] + ('/* unread private value */;' if test is None
                             else 'if (' + test + ') /* m480 unread unless */ { ' + statement + ' }') + out[end + 1:]
    return out, report


def deadstore(text):
    """Return (text, {routine.value: {action, statements}})."""
    report = {}
    for _ in range(6):
        changed = False
        for name, a, b in reversed(_routine_spans(text)):
            body, r = _one(text[a:b])
            if r:
                changed = True
                for value, info in r.items():
                    entry = report.setdefault(name + '.' + value, dict(action=info['action'], statements=0))
                    entry['statements'] += info['statements']
                text = text[:a] + body + text[b:]
        if not changed:
            break
    return text, report
