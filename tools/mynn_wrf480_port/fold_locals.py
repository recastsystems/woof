"""Keep the native ordered plume folds in registers at their own level.

8f8e8b59's ``dmp_mf`` accumulates plume sums plume by plume
(``do ip; do k; s_aw(k) = s_aw(k) + ...``), so every column array element
is read and written back from the arena once per plume. Each element's sum
depends only on its own level, so walking levels outer and plumes inner
performs the same additions, to the same element, in the same plume order;
the running sum just stays in a register for the eight additions. This is
the 4.7.1 transform (tools/mynn_wrf471_port/fold_locals.py) on this text,
plus the two ``edmf_*`` folds 8f8e8b59 adds:

* the surface level (``k = kts``; plume loop only), and
* the interior levels up to each plume's own top (``kts+1 .. ktop(ip)-1``),
  walked as ``k`` outer over the deepest top with the native per-plume
  bound kept as a test, so every plume still adds to exactly the levels it
  adds to natively, in plume order.

Each loop body may touch its fold arrays only at its own level and may
carry no scalar between iterations; both are asserted.
"""
import re

MARKER = '__device__ M471_CALL /* Subroutine */ int dmp_mf__('
FOLD_ARRAYS = r'(?:s_aw\w+|edmf_(?:a|w|qt|thl|ent|qc)1__)'


def _close(body, at):
    depth = 1; pos = at
    while depth:
        depth += (body[pos] == '{') - (body[pos] == '}'); pos += 1
    return pos


def _registers(inner, level):
    """Replace this level's fold elements by registers; return (inner, before, after)."""
    targets = list(dict.fromkeys(re.findall(r'\b(' + FOLD_ARRAYS + r')\[' + level + r'\]\s*(?:[-+*/]?=)(?!=)', inner)))
    assert targets, inner
    for name in targets:
        others = re.findall(r'\b' + name + r'\[([^\]]*)\]', inner)
        assert set(others) == {level.replace('\\', '')}, (name, others)
    before = []; after = []
    for i, name in enumerate(targets):
        register = 'm480_fold' + str(i)
        before.append('real ' + register + '=' + name + '[' + level.replace('\\', '') + '];')
        after.append(name + '[' + level.replace('\\', '') + ']=' + register + ';')
        inner = re.sub(r'\b' + name + r'\[' + level + r'\]', register, inner)
    return inner, before, after


def _scalar_check(inner, allowed):
    """Scalars assigned in a fold body are written before any read there."""
    for name in set(re.findall(r'(?<![\w\]\[.>])([a-z]\w*)\s*=(?!=)', inner)) - {'k', 'ip'}:
        if name.startswith('m480_fold'):
            continue
        assert name in allowed, name
        first = re.search(r'\b' + name + r'\b', inner)
        assert re.match(r'\s*=(?!=)', inner[first.end():]), name


def fold_locals(s):
    start = s.index(MARKER, s.index(MARKER) + len(MARKER))
    end = s.index('} /* dmp_mf__ */', start)
    body = s[start:end]
    # 1. Plume-outer folds over every interior level (the 4.7.1 shape).
    pattern = re.compile(r'for \(ip = 1; ip <= 8; \+\+ip\) \{\s*i__1 = \*kte - 1;\s*'
                         r'#pragma unroll 1\s*for \(k = \*kts; k <= i__1; \+\+k\) \{')
    matches = list(pattern.finditer(body))
    counts = []
    for m in reversed(matches):
        pos = _close(body, m.end())
        inner = body[m.end():pos - 1]
        outer_end = body.index('}', pos) + 1
        assert re.sub(r'/\*.*?\*/|\s+', '', body[pos:outer_end], flags=re.S) == '}'
        inner, before, after = _registers(inner, 'k')
        _scalar_check(inner, {'qc_plume__'})
        replacement = ('for(k=*kts;k<=*kte-1;++k){\n' + '\n'.join(before)
                       + '\n#pragma unroll 1\nfor (ip=1;ip<=8;++ip) {\n' + inner + '\n}\n'
                       + '\n'.join(after) + '\n}')
        body = body[:m.start()] + replacement + body[outer_end:]
        counts.append(len(before))
    assert len(matches) == 6, (len(matches), counts)
    # 2. The surface-level edmf fold: k = kts, plumes only.
    m = re.search(r'\n\s*k = \*kts;\s*#pragma unroll 1\s*for \(ip = 1; ip <= 8; \+\+ip\) \{', body)
    pos = _close(body, m.end())
    inner, before, after = _registers(body[m.end():pos - 1], 'k')
    _scalar_check(inner, set())
    body = (body[:m.start()] + '\n    k = *kts;\n' + '\n'.join(before)
            + '\n#pragma unroll 1\nfor (ip = 1; ip <= 8; ++ip) {' + inner + '}\n'
            + '\n'.join(after) + '\n' + body[pos:])
    # 3. The interior edmf fold bounded by each plume's own top.
    m = re.search(r'for \(ip = 1; ip <= 8; \+\+ip\) \{\s*i__1 = ktop_plume__\[ip - 1\] - 1;\s*'
                  r'#pragma unroll 1\s*for \(k = \*kts \+ 1; k <= i__1; \+\+k\) \{', body)
    pos = _close(body, m.end())
    inner = body[m.end():pos - 1]
    outer_end = body.index('}', pos) + 1
    assert re.sub(r'/\*.*?\*/|\s+', '', body[pos:outer_end], flags=re.S) == '}'
    inner, before, after = _registers(inner, 'k')
    _scalar_check(inner, {'upak', 'r__1'})
    # The next statements rewrite ip, k and i__1 before reading them, and
    # upak is rewritten before its next read (asserted below).
    tail = body[outer_end:]
    assert re.match(r'\s*/\*[^*]*\*/\s*#pragma unroll 1\s*for \(ip = 1; ip <= 8; \+\+ip\) \{\s*k = ktop_plume__\[ip - 1\];', tail), tail[:200]
    for name in ('upak', 'i__1'):
        nxt = re.search(r'\b' + name + r'\b', tail)
        assert re.match(r'\s*=(?!=)', tail[nxt.end():]), name
    deepest = ('integer m480_top=0;\n#pragma unroll 1\nfor(ip=1;ip<=8;++ip){if(ktop_plume__[ip-1]-1>m480_top)'
               'm480_top=ktop_plume__[ip-1]-1;}\n')
    replacement = (deepest + 'for(k=*kts+1;k<=m480_top;++k){\n' + '\n'.join(before)
                   + '\n#pragma unroll 1\nfor (ip=1;ip<=8;++ip) {if(k<=ktop_plume__[ip-1]-1){\n' + inner
                   + '\n}}\n' + '\n'.join(after) + '\n}')
    body = body[:m.start()] + replacement + body[outer_end:]
    return s[:start] + body + s[end:]
