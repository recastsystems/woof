"""Name the driver arguments the column clears before it reads them.

The pack fills every argument array of every column before the launch. The
8f8e8b59 driver's first statements zero a set of its own arguments (the
tendencies, the EDMF and subsidence outputs, ...) over their whole length
before anything reads them, so packing a constant into those words is work
the column throws away. This lists them (by schema argument name); the pack
skips such a field only when its value is a constant, never a state array.
Detected on the driver text before ``driverzero`` moves any of those clears
into ``dmp_mf`` (which then clears the same words before any read).
"""
import re

DRIVER = '__device__ M471_CALL /* Subroutine */ int mynnedmf_('


def _strip(text):
    return re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m[0]), text, flags=re.S)


def initialised(s, schema, nz=49):
    start = s.index(DRIVER, s.index(DRIVER) + len(DRIVER))
    end = s.index('} /* mynnedmf_ */', start)
    text = s[start:end]
    clean = _strip(text)
    params = re.findall(r'M471Ptr<\w+> (\w+)', clean[:clean.index('{')])
    names = [n for n, f in schema['arguments'].items() if f['type'] != 'character']
    assert len(params) == len(names), (len(params), len(names))
    body = text.index('/* Function Body */')
    declarations, code = clean[:body], clean[body:]
    found = []
    for param, name in zip(params, names):
        field = schema['arguments'][name]
        if not field['dims'] or field['intent'] == 'in':
            continue
        # Earlier occurrences may only be clears (writes of zero, e.g. the
        # cold-start block's); the first other use must follow the full clear.
        first = None
        for use in re.finditer(r'\b' + re.escape(param) + r'\b', code):
            if not re.match(re.escape(param) + r'\[[^\]]*\] = zero;', code[use.start():]):
                break
            loop = code.rfind('#pragma', 0, use.start())
            before = code[:loop]
            if before.count('{') == before.count('}'):
                first = use
                break
        if first is None:
            continue
        loop = code.rfind('#pragma', 0, first.start())
        head = code[loop:first.start()]
        m = re.fullmatch(r'#pragma unroll \d+\nfor \(ixarr1 = 1; ixarr1 <= i__1; \+\+ixarr1\) \{\s*', head)
        if not m:
            continue
        before = code[:loop]
        if before.count('{') != before.count('}'):
            continue                                   # not at routine depth
        bound = re.search(r'i__1 = ([^;]+);\s*$', before)
        statement = re.match(re.escape(param) + r'\[\*kts \+ ixarr1 - 1\] = zero;\s*\}', code[first.start():])
        if not (bound and statement and bound[1].strip() == '*kte - *kts + 1'):
            continue
        assert re.search(re.escape(param) + r' -= ' + re.escape(param.rstrip('_') + '_offset') + r';', declarations) \
            or re.search(re.escape(param) + r' -= \w+_offset;', declarations), param
        assert field['count'] == nz, (name, field['count'])
        found.append(name)
    return found
