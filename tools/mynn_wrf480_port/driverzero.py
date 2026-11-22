"""Skip the driver's clears of arrays dmp_mf clears again before any use.

8f8e8b59's driver zeroes the EDMF flux, plume and subsidence arrays on
entry, then (``bl_mynn_edmf > 0``) calls ``dmp_mf``, whose first statements
zero the same arrays over at least the same words. When nothing in the
driver touches an array between its clear and that call, the driver's clear
is dead whenever ``bl_mynn_edmf > 0`` and is guarded by the opposite test,
so ``bl_mynn_edmf = 0`` runs still see their zeros. Checked per array: the
word ranges (after both routines' parameter adjustments) and that no
statement in between names the array. Every output word is unchanged.
"""
import re

DRIVER = '__device__ M471_CALL /* Subroutine */ int mynnedmf_('
DMP = '__device__ M471_CALL /* Subroutine */ int dmp_mf__('


def _strip(text):
    return re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m[0]), text, flags=re.S)


def _span(s, marker, name):
    start = s.index(marker, s.index(marker) + len(marker))
    return start, s.index('} /* ' + name + ' */', start)


def _close(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return i
    raise ValueError('unbalanced')


def _offset(declarations, name):
    """Index of element 0 after f2c's parameter adjustment (kts = 1)."""
    if re.search(r'--' + re.escape(name) + r';', declarations):
        return 1
    m = re.search(re.escape(name) + r' -= (\w+);', declarations)
    if m:
        assert re.search(re.escape(m[1]) + r' = \*kts;', declarations), name
        return 1
    return 0


def _clears(code, nz):
    """Single-array clear loops at routine depth: (start, end, array, first, last)."""
    out = []
    pattern = re.compile(r'(?:\n    i__1 = ([^;]+);\s*)?\n#pragma unroll \d+\nfor \(ixarr1 = 1; ixarr1 <= ([^;]+); \+\+ixarr1\) \{\s*'
                         r'(\w+)\[([^\]]+)\] = zero;\s*\}')
    for m in pattern.finditer(code):
        before = code[:m.start()]
        if before.count('{') - before.count('}') != 0:
            continue
        env = {'MYNN471_NZ': nz}
        bound = m[2].strip()
        if bound == 'i__1':
            assert m[1], m[0]
            bound = m[1]
        expr = lambda e, i: eval(e.replace('*kts', '1').replace('*kte', str(nz)).replace('ixarr1', str(i)),
                                 {'__builtins__': {}}, env)
        trip = expr(bound, 0)
        out.append((m.start(), m.end(), m[3], expr(m[4], 1), expr(m[4], trip)))
    return out


def driverzero(s, nz=49):
    a, b = _span(s, DRIVER, 'mynnedmf_')
    driver = _strip(s[a:b])
    dmp_a, dmp_b = _span(s, DMP, 'dmp_mf__')
    dmp = _strip(s[dmp_a:dmp_b])
    dmp_params = re.findall(r'M471Ptr<\w+> (\w+)', dmp[:dmp.index('{')])
    call = re.search(r'dmp_mf__\(m471_work,m471_n,([^;]*)\);', driver)
    args = [x.strip() for x in re.sub(r'\s+', ' ', call[1]).split(',')]
    assert len(args) == len(dmp_params), (len(args), len(dmp_params))
    head = driver[:call.start()]
    assert re.search(r'if \(\*bl_mynn_edmf__ > 0\) \{\s*$', head), head[-200:]
    assert not re.search(r'\*bl_mynn_edmf__\s*=(?!=)', driver)
    dmp_body = s[dmp_a:dmp_b].index('/* Function Body */')
    dmp_decl, dmp_code = dmp[:dmp_body], dmp[dmp_body:]
    active = dmp_code.index('superadiabatic) {')
    dmp_clears = {}
    for start, end, name, first, last in _clears(dmp_code[:active], nz):
        uses = [m.start() for m in re.finditer(r'\b' + name + r'\b', dmp_code)]
        assert uses[0] >= start, name                    # cleared before any use
        o = _offset(dmp_decl, name)
        dmp_clears.setdefault(name, (first - o, last - o))
    driver_body = s[a:b].index('/* Function Body */')
    driver_decl = driver[:driver_body]
    guarded = []
    for start, end, name, first, last in _clears(driver[driver_body:call.start()], nz):
        start += driver_body; end += driver_body
        bound = {}
        for param, arg in zip(dmp_params, args):
            m = re.fullmatch(r'&?(\w+)(?:\[(\w+)\])?', arg)
            if m and m[1] == name:
                bound[param] = m[2]
        if len(bound) != 1:
            continue
        param, index = next(iter(bound.items()))
        if param not in dmp_clears:
            continue
        o = _offset(driver_decl, name)
        if index is not None:
            assert index == name.rstrip('_') + '_offset' or index.endswith('_offset'), index
        lo, hi = first - o, last - o
        dlo, dhi = dmp_clears[param]
        if not (dlo <= lo and hi <= dhi):
            continue
        between = driver[end:call.start()]
        if re.search(r'\b' + name + r'\b', between):
            continue
        guarded.append((start, end, name))
    text = s[a:b]
    for start, end, name in reversed(guarded):
        text = (text[:start] + '\n    if (!(*bl_mynn_edmf__ > 0)) { /* dmp_mf clears ' + name + ' */'
                + text[start:end] + '\n    }' + text[end:])
    return s[:a] + text + s[b:], [g[2] for g in guarded]
