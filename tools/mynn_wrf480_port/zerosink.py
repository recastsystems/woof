"""Initialise dmp_mf's plume carriers only in columns that integrate plumes.

8f8e8b59's ``dmp_mf`` clears its eight-plume carrier arrays (``upw``,
``upthl``, ``upa``, ... and ``ent``, 8 x (nz+1) words each) on entry, for
every column. Every other use of those private arrays sits inside the
updraft branch (``if (flt > 0 .and. fltv2 > ... .and. superadiabatic)``)
or the plume folds behind ``nup2 > 0``, and ``nup2 > 0`` holds only after
that branch ran (its else sets ``nup2 = 0``). On the 1974 case 94% of
columns take neither, yet the clears were ~40% of the routine's time
(1.2 GB of zero stores a call on 50,200 columns). The clears are moved, in
their native order, to the start of the updraft branch: a column that
integrates plumes clears exactly the same words before any use, and a
column that does not never reads them. Every output word is unchanged.
"""
import re

MARKER = '__device__ M471_CALL /* Subroutine */ int dmp_mf__('
ACTIVE = re.compile(r'    if \(\*flt > zero && fltv2 > rc0032 && \*maxwidth > minwidth && \s*superadiabatic\) \{')
NUP2 = '    if (nup2 > 0.f) {'


def _strip(text):
    return re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m[0]), text, flags=re.S)


def _close(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return i
    raise ValueError('unbalanced')


def zerosink(s):
    start = s.index(MARKER, s.index(MARKER) + len(MARKER))
    end = s.index('} /* dmp_mf__ */', start)
    body = s[start:end]
    clean = _strip(body)
    private = set(re.findall(r'M471Ptr<\w+> (\w+)=m471_array<', clean))
    m = ACTIVE.search(clean)
    assert m and len(ACTIVE.findall(clean)) == 1
    active_open = m.end() - 1
    active_close = _close(clean, active_open)
    assert re.match(r'\s*else\s*\{\s*nup2 = rc0007;\s*\}', clean[active_close + 1:]), clean[active_close:active_close + 80]
    fold_at = clean.index(NUP2, active_close)
    assert clean.count(NUP2) == 1
    fold_close = _close(clean, fold_at + len(NUP2) - 1)
    regions = [(active_open, active_close), (fold_at, fold_close)]
    first = body.index('/* Function Body */')
    nests = []
    for n in re.finditer(r'\n#pragma unroll \d+\nfor \(ixarr2 = 1; ixarr2 <= 8; \+\+ixarr2\) \{', clean[:active_open]):
        if n.start() < first:
            continue
        before = clean[:n.start()]
        if before.count('{') - before.count('}') != body[:body.index('{') + 1].count('{'):
            continue                                       # not at routine depth
        close = _close(clean, n.end() - 1)
        inner = clean[n.end():close]
        targets = set(re.findall(r'\b(\w+)\[[^\]]*\]\s*=(?!=)', inner))
        statements = re.findall(r'(\w+)\[[^;]*\]\s*=\s*([^;]+);', inner)
        assert targets <= private, targets
        assert all(re.fullmatch(r'zero|rc\d+', rhs.strip()) for _, rhs in statements), statements
        if len(targets) > 1:
            continue
        nests.append((n.start() + 1, close + 1, targets))
    moved = []
    for a, b, targets in nests:
        for name in targets:
            for use in re.finditer(r'\b' + name + r'\b', clean):
                p = use.start()
                if a <= p < b or clean[max(0, p - 30):p].rstrip().endswith('>'):
                    continue                               # the clear itself / declaration
                assert any(lo < p < hi for lo, hi in regions), (name, clean[p - 80:p + 40])
        moved.append(body[a:b] + '\n')
    assert len(moved) >= 10, len(moved)
    out = body
    for a, b, _ in reversed(nests):
        out = out[:a] + out[b:]
    shift = sum(b - a for a, b, _ in nests)
    insert = active_open + 1 - shift
    out = out[:insert] + '\n' + ''.join(moved) + out[insert:]
    return s[:start] + out + s[end:], len(moved)
