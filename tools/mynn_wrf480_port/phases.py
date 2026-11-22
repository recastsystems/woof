"""Split the generated 8f8e8b59 driver at native call boundaries.

The 4.7.1 split (tools/mynn_wrf471_port/phases.py) cut that generation's
driver; this is the same cut on 8f8e8b59's text. Nothing is reordered: each
phase kernel runs a contiguous run of complete native driver statements, and
the phase-ordered launches on one stream execute every statement in native
order for every column.

A cut can only lose the driver's private scalar locals: argument arrays live
in the packed slabs and every routine's private arrays in the column arena,
which persist between launches. A scalar is kept in a bounded per-column
arena region (``reserve_base`` onward, 8 bytes a slot) when a phase after its
first may read it before a dominating write: a statement-level, unconditional
write earlier in an enclosing block (the driver has no goto, break or
continue). Every other scalar stays a register local. The single
``mynn471_step`` kernel (phase -1, the strict qualification path) keeps every
local private, exactly as before.
"""
import re

MARKERS = ['    /* Function Body */', '    mym_condensation__(',
           '    if (*bl_mynn_edmf__ > 0) {', '    mym_turbulence__(',
           '    mym_predict__(', '    mynn_tendencies__(',
           '    if (*mix_chem__) {', '    return 0;']
PHASE_COUNT = len(MARKERS) - 1
SLOT = 8
#: Fast-numerics minimum resident blocks per SM for a phase kernel (ptxas
#: caps its registers to fit). Measured on the 1974 case, 50,200 columns,
#: RTX 4090: phase 2 (EDMF) 3.62 -> 3.40 ms at 3, phase 3 (turbulence)
#: 1.61 -> 1.33 ms at 4; every other phase is fastest uncapped.
PHASE_MIN_BLOCKS = {2: 3, 3: 4}


def _strip(text):
    return re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m[0]), text, flags=re.S)


def _spans(clean):
    spans = []; stack = []
    for i, ch in enumerate(clean):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            spans.append((stack.pop(), i))
    assert not stack
    return spans


def _enclosing(spans, pos):
    best = (-1, 10**12)
    for a, b in spans:
        if a < pos < b and a > best[0]:
            best = (a, b)
    return best


def _writes(clean, name):
    """Unconditional statement-level writes: (effective position, enclosing block)."""
    spans = _spans(clean); result = []
    for m in re.finditer(r'(?<![&*.\w>\[])\b' + re.escape(name) + r'\s*=(?!=)', clean):
        q = m.start()
        before = clean[:q]
        if re.search(r'\bfor\s*\([^;{}]*;[^{}]*$', before[max(before.rfind(c) for c in '{}') + 1:]):
            continue                      # a for condition or increment clause
        start = max(before.rfind(c) for c in ';{}')
        head = re.sub(r'\n#[^\n]*', '\n', '\n' + before[start + 1:]).strip()
        if re.fullmatch(r'for\s*\(', head):
            effective = clean.index(';', q)
        elif head == '' or (re.fullmatch(r'(?:\w+\s*=[^;{}]*,\s*)+', head)
                            and head.count('(') == head.count(')')):
            level = 0; effective = None
            for i in range(m.end(), len(clean)):
                ch = clean[i]
                if ch == '(':
                    level += 1
                elif ch == ')':
                    level -= 1
                elif ch in ',;' and level == 0:
                    effective = i; break
            assert effective is not None
        else:
            continue                      # under if/else/while: not definite
        result.append((effective, _enclosing(spans, q)))
    return result


def live_in(covered, piece, name):
    """True when some read of ``name`` in ``piece`` has no dominating write."""
    if covered:
        return False
    clean = _strip(piece)
    writes = _writes(clean, name)
    for m in re.finditer(r'\b' + re.escape(name) + r'\b', clean):
        p = m.start()
        if re.match(r'\s*=(?!=)', clean[m.end():]) and not re.search(r'[&*.>\[]\s*$', clean[:p]):
            continue                      # a pure write
        if not any(eff < p and a < p < b for eff, (a, b) in writes):
            return True
    return False


def phases(text, reserve_base):
    """``reserve_base``: C expression of the first free per-column arena byte."""
    declaration = '__device__ M471_CALL /* Subroutine */ int mynnedmf_('
    assert text.count(declaration) == 2
    text = text.replace(declaration, 'template<int M471_PHASE>\n' + declaration)
    start = text.index(declaration, text.index(declaration) + len(declaration))
    stop = text.index('} /* mynnedmf_ */', start)
    driver = text[start:stop]
    assert not re.search(r'\b(goto|break|continue)\b', _strip(driver))
    body = driver.index('    /* Function Body */')
    declarations, code = driver[:body], driver[body:]
    indices = [code.index(m) for m in MARKERS]
    assert indices == sorted(indices)
    for m in MARKERS:
        assert code.count(m) == 1, m
    pieces = [code[a:b] for a, b in zip(indices, indices[1:])]
    for piece in pieces:
        clean = _strip(piece)
        assert clean.count('{') == clean.count('}')
    # Initialised locals are DATA constants that the code never writes.
    for name in re.findall(r'^    (?:real|integer|logical|doublereal) (\w+) = ', declarations, flags=re.M):
        assert not re.search(r'(?<![=!<>\w])' + name + r'\s*(?:[-+*/]?=)(?!=)', _strip(code)), name
    prefix = _strip(declarations[declarations.index('{') + 1:])
    saved = {}; local_only = []; undefined = []

    def scalars(m):
        typ, items = m[1], m[2]
        if '=' in items or '[' in items:
            return m[0]
        names = [n.strip() for n in items.split(',')]
        assert all(re.fullmatch(r'\w+', n) for n in names), items
        keep, lines = [], []
        for name in names:
            # Parameter adjustments run in every phase before its body.
            covered = any(a < 0 for _, (a, b) in _writes(prefix, name))
            used = [i for i, piece in enumerate(pieces)
                    if re.search(r'\b' + re.escape(name) + r'\b', _strip(piece))]
            live = [i for i in used if live_in(covered, pieces[i], name)]
            if used and live and live[0] == used[0]:
                undefined.append(name)
            if any(i != used[0] for i in live):
                offset = f'({reserve_base}+{len(saved) * SLOT})'
                saved[name] = (typ, used, live)
                lines.append(f'    {typ} local_{name}; {typ} &{name}=M471_PHASE<0 ? local_{name} : '
                             f'*m471_array<{typ}>(m471_work,{offset},m471_n);')
            else:
                keep.append(name)
                local_only.append(name)
        if keep:
            lines.insert(0, f'    {typ} {", ".join(keep)};')
        return '\n'.join(lines)

    declarations = re.sub(r'^    (real|integer|logical|doublereal)\s+([^;]+);', scalars,
                          declarations, flags=re.M)
    driver = declarations + '\n'.join(
        f'    if constexpr(M471_PHASE<0 || M471_PHASE=={i}) {{\n{piece}\n    }}'
        for i, piece in enumerate(pieces)) + '\n    return 0;\n'
    text = text[:start] + driver + text[stop:]
    index = text.index('extern "C" __global__ M471_LAUNCH void mynn471_step(')
    engine = text[index:]
    entry = engine.replace('extern "C" __global__ M471_LAUNCH void mynn471_step(',
                           'template<int M471_PHASE> __device__ __forceinline__ void m471_entry(')
    assert entry.count('mynnedmf_(') == 1
    entry = entry.replace('mynnedmf_(', 'mynnedmf_<M471_PHASE>(')
    signature = '(int n,float*f32,int*i32,unsigned char*arena)'
    kernels = ('#if GPUWM_WRF_EXACT\n#define M480_PHASE_LAUNCH(blocks)\n#else\n'
               '#define M480_PHASE_LAUNCH(blocks) __launch_bounds__(128,blocks)\n#endif\n'
               'extern "C" __global__ M471_LAUNCH void mynn471_step' + signature
               + '{m471_entry<-1>(n,f32,i32,arena);}\n')
    for phase in range(PHASE_COUNT):
        bound = ('M480_PHASE_LAUNCH(' + str(PHASE_MIN_BLOCKS[phase]) + ')'
                 if phase in PHASE_MIN_BLOCKS else 'M471_LAUNCH')
        kernels += ('extern "C" __global__ ' + bound + ' void mynn471_phase' + str(phase) + signature
                    + '{m471_entry<' + str(phase) + '>(n,f32,i32,arena);}\n')
    report = dict(saved={k: dict(type=v[0], phases=v[1], read_first=v[2]) for k, v in saved.items()},
                  local=local_only, read_before_write=undefined, reserve_bytes=len(saved) * SLOT)
    return text[:index] + entry + kernels, report
