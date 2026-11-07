"""Generate the BEP+BEM CUDA port from WRF v4.7.1's own Fortran.

usage: python tools/transcribe_urban_bem.py WRF_SOURCE_ROOT [--report DIR]

Writes ``woof/core/kernels/urban_bem.cuh`` (module_sf_bem.F and the libm
helpers of ``tools/urban_bem_support.cuh.in``), ``woof/core/kernels/urban_bep_bem.cu``
(module_sf_bep_bem.F and the two entry kernels) and
``woof/core/urban_bem_layout.py`` (table, class and workspace layouts).

The three WRF inputs are hash-pinned against
``tools/urban_wrf471_oracle/SOURCES.sha256``, so a regeneration from any
other source stops before writing anything.

This deliberately narrow translator rejects syntax it does not understand.
Generated arithmetic is typed and parenthesized before C++ emission, in
Fortran's evaluation order.  Written by the BEM port's implementer (an
automated port lane) and adapted for the repository by that lane; the port it
emits is graded bit for bit by ``tests/test_urban_bem_wrf471_parity.py``.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Set by main(): the WRF v4.7.1 tree's phys/.
WRF_PHYS: Path = ROOT

#: The WRF files this translator reads, as SOURCES.sha256 names them.
WRF_INPUTS = ("phys/module_sf_bem.F", "phys/module_sf_bep_bem.F",
              "phys/module_sf_urban.F")
def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin_inputs(wrf_root: Path) -> None:
    global WRF_PHYS
    pins = {}
    sums = ROOT / "tools/urban_wrf471_oracle/SOURCES.sha256"
    for line in sums.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split()
            pins[rel] = digest
    for rel in WRF_INPUTS:
        have = _sha256(wrf_root / rel)
        if have != pins[rel]:
            raise SystemExit(f"{wrf_root / rel}: sha256 {have}, pinned {pins[rel]}")
    WRF_PHYS = wrf_root / "phys"


def split(s, sep=','):
    out, start, depth = [], 0, 0
    for i, ch in enumerate(s):
        depth += (ch == '(') - (ch == ')')
        if ch == sep and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


def statements(path):
    result, pending, first = [], '', 0
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        if raw.lstrip().startswith('#'):
            continue
        # Comments, but preserve quoted diagnostic text.
        quote = None
        end = len(raw)
        for i, ch in enumerate(raw):
            if ch in "'\"":
                quote = None if quote == ch else ch if quote is None else quote
            if ch == '!' and quote is None:
                end = i
                break
        s = raw[:end].strip().lower()
        if not s:
            continue
        if not pending:
            first = line
        pending += ' ' + s.lstrip('&').rstrip('&').strip()
        if s.endswith('&'):
            continue
        for part in split(pending.strip(), ';'):
            result.append((first, part))
        pending = ''
    assert not pending
    return result


@dataclasses.dataclass
class Var:
    kind: str
    dims: list[str] = dataclasses.field(default_factory=list)
    init: str | None = None
    const: bool = False


@dataclasses.dataclass
class Routine:
    name: str
    prefix: str
    file: str
    line: int
    args: list[str]
    vars: dict[str, Var] = dataclasses.field(default_factory=dict)
    body: list[tuple[int, str]] = dataclasses.field(default_factory=list)
    data: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    saved: list[str] = dataclasses.field(default_factory=list)
    writes: set[str] = dataclasses.field(default_factory=set)


def declaration(s, vs):
    m = re.match(r'(double precision|real|integer|logical|character(?:\([^)]*\))?)\b(.*)', s)
    if not m:
        return False
    k, rest = m.groups()
    kind = {'real': 'float', 'integer': 'int', 'logical': 'bool', 'double precision': 'double'}.get(k, 'char')
    attrs, names = rest.split('::', 1) if '::' in rest else ('', rest)
    dm = re.search(r'dimension\((.*?)\)', attrs)
    dims = split(dm[1]) if dm else []
    for item in split(names):
        namepart, *init = split(item, '=')
        v = re.fullmatch(r'([a-z][a-z0-9_]*)(?:\((.*)\))?', namepart.strip())
        if not v:
            raise ValueError(('declaration', s, item))
        vs[v[1]] = Var(kind, split(v[2]) if v[2] else dims[:], init[0] if init else None, 'parameter' in attrs)
    return True


def parse(file, prefix):
    routines, globals_, cur = [], {}, None
    for ln, s in statements(WRF_PHYS / file):
        m = re.match(r'subroutine\s+(\w+)\s*\((.*)\)', s)
        if m:
            cur = Routine(m[1], prefix, file, ln, split(m[2]))
            routines.append(cur)
            continue
        if s.startswith('end') and re.match(r'end\s*subroutine\b', s):
            cur = None
            continue
        if cur is None:
            if s.startswith('contains'):
                continue
            if declaration(s, globals_):
                continue
            m = re.match(r'parameter\s*\((.*)\)', s)
            if m:
                for p in split(m[1]):
                    n, v = p.split('=', 1)
                    globals_[n.strip()].init = v.strip()
                    globals_[n.strip()].const = True
            continue
        if s == 'implicit none':
            continue
        if declaration(s, cur.vars):
            continue
        m = re.match(r'parameter\s*\((.*)\)', s)
        if m:
            for p in split(m[1]):
                n, v = p.split('=', 1)
                cur.vars[n.strip()].init = v.strip()
                cur.vars[n.strip()].const = True
            continue
        if s.startswith('save '):
            cur.saved.extend(split(s[5:]))
            continue
        m = re.match(r'data\s+(\w+)\s*/(.*)/', s)
        if m:
            cur.data[m[1]] = split(m[2])
            continue
        cur.body.append((ln, s))
    return routines, globals_


TOKEN = re.compile(r'\s*(\*\*|\.(?:or|and|eq|ne|gt|ge|lt|le|not|true|false)\.|\d+\.\d*[ed][+-]?\d+|\d+[ed][+-]?\d+|\d+\.(?![a-z])\d*|\.\d+(?:[ed][+-]?\d+)?|\d+|[a-z_]\w*|/=|<=|>=|==|[()+*/,:<>\-=])')
IEEE_OPS = {
    'float': {'+': '__fadd_rn', '-': '__fsub_rn', '*': '__fmul_rn', '/': '__fdiv_rn'},
    'double': {'+': '__dadd_rn', '-': '__dsub_rn', '*': '__dmul_rn', '/': '__ddiv_rn'},
}
PREC = {'.or.': 1, '.and.': 2, '.eq.': 3, '.ne.': 3, '.gt.': 3, '.ge.': 3, '.lt.': 3, '.le.': 3, '==': 3, '/=': 3, '<': 3, '>': 3, '<=': 3, '>=': 3, '+': 4, '-': 4, '*': 5, '/': 5, '**': 7}


def expression(s):
    tokens, pos = [], 0
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise ValueError(('token', s, s[pos:]))
        tokens.append(m[1])
        pos = m.end()
    i = 0

    def expr(p=0):
        nonlocal i
        t = tokens[i]
        i += 1
        if t in ('+', '-', '.not.'):
            a = ('unary', t, expr(6 if t != '.not.' else 3))
        elif t == '(':
            a = expr()
            assert tokens[i] == ')', (s, tokens[i:])
            i += 1
        elif re.match(r'[a-z_]', t):
            a = ('var', t)
            if i < len(tokens) and tokens[i] == '(':
                i += 1
                args = []
                while tokens[i] != ')':
                    if tokens[i] == ':':
                        x = None
                    else:
                        x = expr()
                    if tokens[i] == ':':
                        i += 1
                        y = None if tokens[i] in (',', ')') else expr()
                        x = ('range', x, y)
                    args.append(x)
                    if tokens[i] != ',':
                        break
                    i += 1
                assert tokens[i] == ')', (s, tokens[i:])
                i += 1
                a = ('call', t, args)
        else:
            a = ('lit', t)
        while i < len(tokens) and PREC.get(tokens[i], -1) >= p:
            op = tokens[i]
            i += 1
            a = ('op', op, a, expr(PREC[op] + (0 if op == '**' else 1)))
        return a
    a = expr()
    assert i == len(tokens), (s, tokens[i:])
    return a


def bounds(d):
    return tuple(split(d, ':')) if ':' in d else ('1', d)


ROUTINES, GLOBALS = {}, {}
ERRORS = {}
TABLE_LAYOUT = {}
CLASS_LAYOUT = {}


def product(shape):
    result=1
    for n in shape:
        result*=n
    return result


def numeric(s):
    # Fixed dimensions only. Arithmetic constants never enter host evaluation.
    env={'ndm':2,'nz_um':18,'nwr_u':10,'ng_u':10,'ngr_u':10,'nf_u':10,'ngb_u':10,'nbui_max':15,'nurbmax':11,'nurbm':11}
    return int(eval(s, {'__builtins__':{}}, env))


def layout_add(layout, name, v, offsets, shape=None):
    if shape is None:
        shape=tuple(numeric(b)-numeric(a)+1 for a,b in map(bounds,v.dims))
    layout[name]=(v.kind, offsets[v.kind], shape)
    offsets[v.kind]+=product(shape)


def prepare_column(original):
    r=dataclasses.replace(original,name='column',vars=dict(original.vars),body=[],saved=[])
    # The column list replaces the two horizontal loops. Bounds retain WRF's
    # one-based vertical and collapsed urban indexing.
    body=[(ln,s) for ln,s in original.body if 497<=ln<=651]
    body.extend([(658,'if (num_urban_hi.ge.nz_um) then'),(658,'stop'),(659,'endif')])
    body.extend([(667,'z1d(kts)=0.'),(670,'do iz=kts+1,kte+1'),(673,'z1d(iz)=z1d(iz-1)+dz8w(ix,iz-1,iy)'),(674,'enddo'),(675,'hi_urb1d=0.'),(676,'do iz_u=1,num_urban_hi'),(677,'hi_urb1d(iz_u)=hi_urb2d(ix,iz_u,iy)'),(678,'enddo')])
    # Keep the source condition and all its writes, including copy-in zeroing.
    body.extend((ln,re.sub(r'z\(ix,([^,]+),iy\)',r'z1d(\1)',s)) for ln,s in original.body if 710<=ln<=1146 and not 714<=ln<=718)
    # 1148 is the end of the urban condition in this pinned source.
    r.body=body
    for n in ('z','hi_urb','first','text','time_bep','t_phy','u','v','ust'):
        r.vars.pop(n,None)
    r.args=original.args+['ix','iy']
    return r


class Emitter:
    def __init__(self, routine):
        self.r = routine
        self.vs = GLOBALS | routine.vars
        self.q = None

    def kind(self, a):
        if a[0] == 'lit':
            return 'bool' if a[1] in ('.true.', '.false.') else 'double' if 'd' in a[1] else 'float' if any(c in a[1] for c in '.e') else 'int'
        if a[0] == 'var':
            if a[1] not in self.vs:
                raise ValueError(('undeclared', self.r.name, a[1]))
            return self.vs[a[1]].kind
        if a[0] == 'unary':
            return self.kind(a[2])
        if a[0] == 'op':
            if PREC[a[1]] <= 3:
                return 'bool'
            l, r = self.kind(a[2]), self.kind(a[3])
            return 'double' if 'double' in (l, r) else 'float' if 'float' in (l, r) else 'int'
        if a[0] == 'call':
            n = a[1]
            if n in self.vs:
                return self.vs[n].kind
            if n in ('int', 'nint'):
                return 'int'
            if n == 'real':
                return 'float'
            if n == 'sum':
                return self.kind(a[2][0])
            if n in ('min', 'max', 'abs', 'sign', 'mod'):
                return self.kind(a[2][0])
            return self.kind(a[2][0])
        raise ValueError(a)

    def cast(self, text, kind, target):
        return f'{target}({text})' if kind != target else text

    def expr(self, a):
        tag = a[0]
        if tag == 'lit':
            s, k = a[1], self.kind(a)
            if k == 'bool':
                return 'true' if s == '.true.' else 'false'
            if k == 'double':
                return s.replace('d', 'e')
            return (s + 'f') if k == 'float' else s
        if tag == 'var':
            n = a[1]
            if self.vs[n].dims and self.q is not None:
                return f'{n}.flat({self.q})'
            return n
        if tag == 'unary':
            return f'({"!" if a[1] == ".not." else a[1]}{self.expr(a[2])})'
        if tag == 'op':
            op, l, r = a[1:]
            ls, rs = self.expr(l), self.expr(r)
            lk, rk = self.kind(l), self.kind(r)
            if op == '**':
                if rk == 'int':
                    if r[0] == 'lit':
                        return f'ubm_powi_const<{int(r[1])}>({ls})'
                    return f'ubm_powi({ls}, {rs})'
                target = self.kind(a)
                return f'{"pow" if target == "double" else "gfk_pow"}({self.cast(ls, lk, target)}, {self.cast(rs, rk, target)})'
            if PREC[op] <= 3 and op not in ('.and.', '.or.'):
                target = 'double' if 'double' in (lk, rk) else 'float' if 'float' in (lk, rk) else 'int'
            else:
                target = self.kind(a)
            if target != 'bool':
                ls, rs = self.cast(ls, lk, target), self.cast(rs, rk, target)
            op = {'.or.': '||', '.and.': '&&', '.eq.': '==', '.ne.': '!=', '.gt.': '>', '.ge.': '>=', '.lt.': '<', '.le.': '<=', '/=': '!='}.get(op, op)
            if target in ('float', 'double') and op in IEEE_OPS[target] and not getattr(self, 'constant', False):
                # One rounded IEEE operation per Fortran operator, as an
                # intrinsic the compiler may not reassociate, contract or
                # fold: NVRTC 13.3 for sm_120 rewrote (c*ch)/cm as
                # c*(ch/cm) in flux_flat, one ULP off gfortran's -O0 word.
                return f'{IEEE_OPS[target][op]}({ls}, {rs})'
            return f'({ls} {op} {rs})'
        if tag == 'call':
            name, args = a[1:]
            if name in self.vs and self.vs[name].dims:
                if any(x[0] == 'range' for x in args):
                    assert self.q is not None
                    assert sum(x[0] == 'range' for x in args) == 1
                    args = [('op', '+', x[1] or expression(bounds(self.vs[name].dims[j])[0]), ('var', '_aq')) if x[0] == 'range' else x for j, x in enumerate(args)]
                return name + '(' + ', '.join(self.expr(x) for x in args) + ')'
            es = [self.expr(x) for x in args]
            k = self.kind(a)
            if name in ('real', 'int'):
                return f'{"float" if name == "real" else "int"}({es[0]})'
            if name == 'sum':
                return f'ubm_sum({es[0]})'
            if name in ('max', 'min'):
                acc = es[0]
                for x in es[1:]:
                    acc = f'ubm_{name}({acc}, {x})'
                return acc
            if name in ('abs', 'sign', 'mod', 'nint'):
                return f'ubm_{name}(' + ', '.join(es) + ')'
            libm = {'exp': 'gfk_exp', 'log': 'gfk_log', 'log10': 'ubm_log10f', 'sqrt': '__fsqrt_rn', 'sin': 'ubm_sinf', 'cos': 'ubm_cosf', 'acos': 'ubm_acosf', 'asin': 'ubm_asinf', 'atan': 'ubm_atanf', 'tan': 'ubm_tanf'}
            if name not in libm:
                raise ValueError(('intrinsic', self.r.name, name))
            return f'{libm[name]}(' + ', '.join(es) + ')'
        raise ValueError(a)

    def text(self, s):
        return self.expr(expression(s))

    def signature(self):
        params = ['ubm_Context& ctx']
        if self.r.name=='column':
            params.extend(['const float* clsf','const int* clsi'])
        for n in self.r.args:
            v = self.r.vars[n]
            params.append(f'ubm_Array<{v.kind}, {len(v.dims)}> {n}' if v.dims else f'{v.kind}{"&" if n in self.r.writes else ""} {n}')
        return f'__device__ void {self.r.prefix}{self.r.name}(' + ', '.join(params) + ')'

    def stmt(self, s, ln):
        m = re.match(r'(\d+)\s+(.*)', s)
        if m:
            return f'L{m[1]}: ;\n' + self.stmt(m[2], ln)
        if s.startswith('if') or s.startswith('else if') or s.startswith('elseif'):
            start = s.index('(')
            depth = 1
            end = start + 1
            while depth:
                depth += (s[end] == '(') - (s[end] == ')')
                end += 1
            cond, tail = self.text(s[start + 1:end - 1]), s[end:].strip()
            pref = '} else if' if s.startswith('else') else 'if'
            if tail == 'then':
                return f'{pref} ({cond}) {{'
            return f'{pref} ({cond}) {{ {self.stmt(tail, ln)} }}'
        if s in ('endif', 'end if', 'enddo', 'end do'):
            return '}'
        if s == 'else':
            return '} else {'
        if s.startswith('do '):
            m = re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)', s)
            if not m:
                raise ValueError(('do', s))
            n = m[1]
            vals = split(m[2])
            step = vals[2] if len(vals) == 3 else '1'
            # Bounds and step are evaluated once, as Fortran requires.
            return f'{{ const int _end{ln} = {self.text(vals[1])}; const int _step{ln} = {self.text(step)};\nfor ({n} = {self.text(vals[0])}; _step{ln} > 0 ? {n} <= _end{ln} : {n} >= _end{ln}; {n} += _step{ln}) {{'
        if re.match(r'(go to|goto)\s+\d+', s):
            return 'goto L' + re.search(r'\d+', s)[0] + ';'
        if s == 'continue':
            return ';'
        if s == 'return':
            return 'return;'
        if s == 'exit':
            return 'break;'
        if s.startswith(('write', 'print')):
            return '// WRF diagnostic emitted by the host on failure.'
        if s.startswith('fatal_error') or s.startswith('stop'):
            code = len(ERRORS) + 1
            key = f'{self.r.file}:{ln}'
            if key not in ERRORS:
                raw=(WRF_PHYS/self.r.file).read_text().splitlines()
                texts=[]
                for previous in reversed(raw[max(0,ln-10):ln]):
                    if 'FATAL_ERROR' in previous or re.search(r'\b(write|print)\b',previous,re.I):
                        texts[:0]=re.findall(r"'([^']*)'",previous)
                    elif texts and re.search(r'\b(endif|stop)\b',previous,re.I):
                        break
                ERRORS[key] = (code, ' '.join(texts) or s)
            return f'ctx.fail({ERRORS[key][0]}); return;'
        if s.startswith('call '):
            m = re.fullmatch(r'call\s+(\w+)\s*\((.*)\)', s)
            callee = ROUTINES[m[1]]
            actuals = split(m[2])
            assert len(actuals) == len(callee.args), (self.r.name, m[1], len(actuals), len(callee.args))
            es = []
            for actual, formal in zip(actuals, callee.args):
                v = callee.vars[formal]
                es.append(self.text(actual))
            return f'{callee.prefix}{callee.name}(ctx, ' + ', '.join(es) + ');\nif (ctx.error) return;'
        m = re.match(r'(.+?)\s*=\s*(.*)', s)
        if m:
            if m[2].startswith('(/'):
                vals = split(m[2][2:-2])
                return '\n'.join(f'{m[1].strip()}.flat({i}) = {self.text(x)};' for i, x in enumerate(vals))
            lhs, rhs = expression(m[1]), expression(m[2])
            if lhs[0] == 'var' and self.vs[lhs[1]].dims:
                count = f'{lhs[1]}.size()'
            elif lhs[0] == 'call' and any(x[0] == 'range' for x in lhs[2]):
                ranges = [(i, x) for i, x in enumerate(lhs[2]) if x[0] == 'range']
                assert len(ranges) == 1
                j, x = ranges[0]
                lo, hi = bounds(self.vs[lhs[1]].dims[j])
                count = '(' + self.expr(x[2] or expression(hi)) + ' - ' + self.expr(x[1] or expression(lo)) + ' + 1)'
            else:
                count = None
            if count:
                self.q = '_aq'
                self.vs['_aq'] = Var('int')
            l, r = self.expr(lhs), self.expr(rhs)
            self.q = None
            code = f'{l} = {r};'
            return f'for (int _aq=0; _aq < {count}; ++_aq) {{ {code} }}' if count else code
        raise ValueError(('statement', self.r.name, ln, s))

    def emit(self):
        out = [f'// WRF {self.r.file}:{self.r.line}', self.signature() + ' {', 'ubm_Frame _frame(ctx);', 'const int nurbm=ctx.nurbm;']
        if self.r.name == 'init_para':
            for n, (kind, off, shape) in TABLE_LAYOUT.items():
                ptr = 'ctx.tbli' if kind == 'int' else 'ctx.tblf'
                if not shape:
                    out.append(f'const {kind} {n} = {ptr}[{off}];')
                else:
                    lower = ', '.join('1' for _ in shape)
                    sizes = ', '.join(str(x) for x in shape)
                    out.append(f'auto {n} = ubm_packed<{kind}, {len(shape)}>(const_cast<{kind}*>({ptr}+{off}), {{{lower}}}, {{{sizes}}});')
                    out.append(f'{n}.sz[{len(shape)-1}] = nurbm;' if n not in ('hsequip_tbl', 'irho_tbl') else '')
        for n in self.r.args:
            v = self.r.vars[n]
            if v.dims:
                ds=[bounds(d) for d in v.dims]
                lo=', '.join(self.text(a) for a,b in ds)
                sz=', '.join(f'({self.text(b)} - {self.text(a)} + 1)' for a,b in ds)
                out.append(f'{n} = {n}.rebind({{{lo}}}, {{{sz}}});')
        for n, v in self.r.vars.items():
            if n in self.r.args or v.kind == 'char':
                continue
            if v.dims:
                dims = [bounds(d) for d in v.dims]
                lo = ', '.join(self.text(a) for a, b in dims)
                sz = ', '.join(f'({self.text(b)} - {self.text(a)} + 1)' for a, b in dims)
                if self.r.name in ('wall','wall_gr') and n in ('k1','k2','kc'):
                    off=(0 if self.r.name=='wall' else 60)+('k1','k2','kc').index(n)*20
                    out.append(f'// WRF SAVE coefficients persist across material calls in this column.\n'
                               f'auto {n} = ubm_packed<float,1>(ctx.ws+{off}*ctx.nthreads+ctx.tid, {{1}}, {{20}});\n{n}.st[0]=ctx.nthreads;')
                elif self.r.name=='column' and n in CLASS_LAYOUT:
                    kind,off,shape=CLASS_LAYOUT[n]
                    ptr='clsi' if kind=='int' else 'clsf'
                    out.append(f'auto {n} = ubm_packed<{kind},{len(shape)}>(const_cast<{kind}*>({ptr}+{off}), {{{lo}}}, {{{sz}}});')
                else:
                    out.append(f'auto {n} = ctx.array<{v.kind}, {len(dims)}>({{{lo}}}, {{{sz}}});')
                if n in self.r.data:
                    for i, x in enumerate(self.r.data[n]):
                        out.append(f'{n}.flat({i}) = {self.text(x)};')
            else:
                if self.r.name=='column' and n in CLASS_LAYOUT:
                    kind,off,shape=CLASS_LAYOUT[n]
                    out.append(f'{kind} {n} = {"clsi" if kind=="int" else "clsf"}[{off}];')
                else:
                    out.append(f'{"const " if v.const else ""}{v.kind} {n}' + (f' = {self.text(v.init)}' if v.init else '') + ';')
        # Each Fortran DO emits an outer bound scope and an inner loop scope.
        stack = []
        for ln, s in self.r.body:
            if s.startswith('do '):
                stack.append('do')
            elif re.match(r'(if|else if|elseif)', s) and s.endswith('then'):
                if s.startswith('if'):
                    stack.append('if')
            elif s in ('endif', 'end if', 'enddo', 'end do'):
                which = stack.pop()
                assert which == ('do' if 'do' in s else 'if'), (self.r.name, ln, stack)
                out.append(f'// {self.r.file}:{ln}\n' + ('}}' if which == 'do' else '}'))
                continue
            out.append(f'// {self.r.file}:{ln}\n' + self.stmt(s, ln))
        assert not stack, (self.r.name, stack)
        out.append('}')
        return '\n'.join(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wrf_root", type=Path)
    ap.add_argument("--report", type=Path, default=None,
                    help="also write the source inventory and layout JSON here")
    args = ap.parse_args(argv)
    _pin_inputs(args.wrf_root)
    all_routines = []
    for file, prefix in [('module_sf_bem.F', 'bem_'), ('module_sf_bep_bem.F', 'bb_')]:
        rs, gs = parse(file, prefix)
        all_routines.extend(rs)
        GLOBALS.update({k: v for k, v in gs.items() if v.const})
    GLOBALS['nurbm'] = Var('int', init='11', const=True)
    table_globals = {}
    for ln, s in statements(WRF_PHYS / 'module_sf_urban.F'):
        if s == 'contains':
            break
        declaration(s, table_globals)
    GLOBALS.update({k: v for k, v in table_globals.items() if k.endswith('_tbl') or k == 'icate'})
    ROUTINES.update({r.name: r for r in all_routines})
    for r in all_routines:
        for ln, s in r.body:
            m = re.match(r'(?:\d+\s+)?([a-z]\w*)(?:\(.*?\))?\s*=(?!=)', s)
            if m:
                r.writes.add(m[1])
            for n in re.findall(r'do\s+(\w+)\s*=', s):
                r.writes.add(n)
        # Explicit output intents are covered by assignments and callee propagation.
    changed = True
    while changed:
        changed = False
        for r in all_routines:
            for ln, s in r.body:
                m = re.search(r'call\s+(\w+)\s*\((.*)\)', s)
                if not m:
                    continue
                c = ROUTINES[m[1]]
                for actual, formal in zip(split(m[2]), c.args):
                    name = re.match(r'([a-z]\w*)', actual)
                    if formal in c.writes and name and name[1] not in r.writes:
                        r.writes.add(name[1])
                        changed = True
    info = {}
    for r in all_routines:
        info[r.name] = {'file': r.file, 'line': r.line, 'arguments': r.args, 'variables': {k: dataclasses.asdict(v) for k, v in r.vars.items()}, 'saved': r.saved}
    if args.report is not None:
        args.report.mkdir(parents=True, exist_ok=True)
        (args.report / 'bem_source_inventory.json').write_text(json.dumps(info, indent=2), newline='\n')
    original=ROUTINES['bep_bem']
    names=set(re.findall(r'\b(?:\w+_tbl|icate)\b', ' '.join(s for ln,s in ROUTINES['init_para'].body)))
    offsets={'float':0,'int':0}
    for n in sorted(names):
        v=GLOBALS[n]
        if n in ('hsequip_tbl','irho_tbl'): shape=(24,)
        elif n in ('street_direction_tbl','street_width_tbl','building_width_tbl'): shape=(3,11)
        elif n in ('height_bin_tbl','hpercent_bin_tbl'): shape=(50,11)
        elif v.dims: shape=(11,)
        else: shape=()
        layout_add(TABLE_LAYOUT,n,v,offsets,shape)
    offsets={'float':0,'int':1}  # clsi[0] is the first-call error code.
    class_names=[n for n in original.saved if n not in ('first','time_bep')]
    class_names+=['twini_u','trini_u','tgini_u']
    for n in class_names:
        layout_add(CLASS_LAYOUT,n,original.vars[n],offsets)
    layout_add(CLASS_LAYOUT,'_nurbm',Var('int'),offsets)
    r=prepare_column(original)
    ROUTINES['column']=r
    all_routines=[x for x in all_routines if x.name!='bep_bem']+[r]
    # Workspace bound.  Every routine opens a ubm_Frame that hands its local
    # arrays back on return, so the most a column ever holds is the deepest
    # call path: a routine's own locals plus the largest bound among the
    # routines it calls, from the column body down.  (The kernel still
    # checks the cursor against the capacity it was given and fails with
    # 9001 rather than write past it.)  Arrays over kms:kme add a linear
    # nz+1 term; the first 120 words hold the saved wall coefficients.
    dimenv={'ndm':2,'nz_um':18,'nwr_u':10,'ng_u':10,'ngr_u':10,'nf_u':10,'ngb_u':10,'nbui_max':15,'nurbmax':11,'nurbm':11,
            'nmax':150,'n':150,'np':150,'nz':150,'nlev':18,'ncan':18,'nzc':18,'nzcanm':18,'nwal':10,'nrof':10,'nflo':10,'ngrd':10}
    by_name={x.name: x for x in all_routines}

    def own_locals(x):
        f,pl=0,0
        for n,v in x.vars.items():
            if n in x.args or not v.dims or (x.name=='column' and n in CLASS_LAYOUT) or (x.name in ('wall','wall_gr') and n in ('k1','k2','kc')):
                continue
            if v.dims==['kms:kme']:
                pl+=1
                continue
            size=1
            for a,b in map(bounds,v.dims):
                size*=int(eval(b,{'__builtins__':{}},dimenv))-int(eval(a,{'__builtins__':{}},dimenv))+1
            f+=size
        return f,pl

    bound_memo={}

    def path_bound(name, stack=()):
        if name in bound_memo:
            return bound_memo[name]
        if name in stack:
            raise SystemExit(f'recursive call through {name}: no static workspace bound')
        x=by_name[name]
        f,pl=own_locals(x)
        callees=set(re.findall(r'(?:^|[^\w])call\s+(\w+)\s*\(', ' '.join(st for ln,st in x.body)))
        unknown=sorted(c for c in callees if c not in by_name)
        if unknown:
            raise SystemExit(f'{name} calls {unknown}, which the transcription does not contain')
        deepest_f,deepest_pl=0,0
        for c in sorted(callees):
            if c in by_name:
                cf,cpl=path_bound(c, stack+(name,))
                deepest_f=max(deepest_f,cf)
                deepest_pl=max(deepest_pl,cpl)
        bound_memo[name]=(f+deepest_f,pl+deepest_pl)
        return bound_memo[name]

    column_f,column_pl=path_bound('column')
    fixed,per_level=120+column_f,column_pl
    globals_code=[]
    for n,v in GLOBALS.items():
        if v.const and n!='nurbm':
            dummy=Emitter(r)
            # Module PARAMETERs are constant expressions gfortran folds at
            # compile time (correctly rounded); C++ constexpr folds them the
            # same way, and an intrinsic is not a constant expression.
            dummy.constant=True
            globals_code.append(f'constexpr {v.kind} {n} = {dummy.text(v.init)};')
    support=(ROOT/'tools/urban_bem_support.cuh.in').read_text()
    header='// Generated by tools/transcribe_urban_bem.py from the pinned WRF 4.7.1 sources.\n'
    bem=header+'\n'.join(globals_code)+'\n'+support+'\n'
    bem+='\n'.join(Emitter(x).signature()+';' for x in all_routines if x.prefix=='bem_')+'\n'
    bem+='\n\n'.join(Emitter(x).emit() for x in all_routines if x.prefix=='bem_')+'\n'
    bep=header+'\n'.join(Emitter(x).signature()+';' for x in all_routines if x.prefix=='bb_')+'\n'
    bep+='\n\n'.join(Emitter(x).emit() for x in all_routines if x.prefix=='bb_')+'\n'
    bep+=f'\nconstexpr int BB_WORKSPACE_FIXED_FLOATS={fixed};\nconstexpr int BB_WORKSPACE_FLOATS_PER_LEVEL={per_level};\n'
    bep+='extern "C" __global__ void bep_bem_class_init(const float* tblf,const int* tbli,float* clsf,int* clsi) {\n'
    bep+='if(blockIdx.x || threadIdx.x)return;\nubm_Context ctx={nullptr,tblf,tbli,1,0,120,0,0,tbli['+str(TABLE_LAYOUT['icate'][1])+']};\n'
    bep+='if(ctx.nurbm<1 || ctx.nurbm>11){clsi[0]=9002;return;}\n'
    for n,(kind,off,shape) in CLASS_LAYOUT.items():
        if n=='_nurbm':continue
        ptr='clsi' if kind=='int' else 'clsf'
        if shape:
            sz=', '.join(str(x) for x in shape)
            lo=', '.join('1' for x in shape)
            bep+=f'auto {n}=ubm_packed<{kind},{len(shape)}>({ptr}+{off}, {{{lo}}}, {{{sz}}});\n'
        else:
            bep+=f'{kind}& {n}={ptr}[{off}];\n'
    for name in ('init_para','icbep'):
        bep+=f'bb_{name}(ctx, '+', '.join(ROUTINES[name].args)+');\nif(ctx.error){clsi[0]=ctx.error;return;}\n'
    bep+=f'clsi[{CLASS_LAYOUT["_nurbm"][1]}]=ctx.nurbm;clsi[0]=0;\n}}\n'
    array_args=[n for n in original.args if original.vars[n].dims]
    scalar_args=['gmt','julday','declin_urb','dt','itimestep','nz','ny','nx','num_urban_hi']
    scalar_types={n:'int' if n in ('julday','itimestep','nz','ny','nx','num_urban_hi') else 'float' for n in scalar_args}
    params=['const int* cols','int ncol','const float* clsf','const int* clsi']
    params+=[f'{original.vars[n].kind}* dev_{n}' for n in array_args]
    params+=[f'{scalar_types[n]} {n}' for n in scalar_args]
    params+=['float* ws','int workspace_floats','int* err']
    bep+='extern "C" __global__ void bep_bem_columns('+', '.join(params)+') {\n'
    bep+='const int tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=ncol)return;\n'
    bep+=f'ubm_Context ctx={{ws,nullptr,nullptr,ncol,tid,120,workspace_floats,0,clsi[{CLASS_LAYOUT["_nurbm"][1]}]}};\n'
    bep+='int ix=cols[tid]%nx+1,iy=cols[tid]/nx+1;\n'
    for n in array_args:
        v=original.vars[n];rank=len(v.dims)
        if rank==2:
            bep+=f'ubm_Array<{v.kind},2> {n}={{dev_{n},{{1,1}},{{nx,ny}},{{1,nx}}}};\n'
        else:
            d=v.dims[1]
            layer=d.split(':')[-1]
            layer=layer.replace('num_urban_ndm','2').replace('kme','nz+1')
            maps={'zrd':360,'zwd':5400,'gd':20,'zd':540,'zdf':36,'bd':270,'wd':540,'gbd':300,'fbd':5100,'zgrd':360}
            layer=re.sub(r'urban_map_(\w+)',lambda m:str(maps[m[1]]),layer)
            bep+=f'ubm_Array<{v.kind},3> {n}={{dev_{n},{{1,1,1}},{{nx,{layer},ny}},{{1,nx*ny,nx}}}};\n'
    scalar_map={'num_urban_ndm':'2','ims':'1','ime':'nx','jms':'1','jme':'ny','kms':'1','kme':'nz+1','its':'1','ite':'nx','jts':'1','jte':'ny','kts':'1','kte':'nz','ids':'1','ide':'nx+1','jds':'1','jde':'ny+1','kds':'1','kde':'nz+1'}
    for k,num in {'zrd':360,'zwd':5400,'gd':20,'zd':540,'zdf':36,'bd':270,'wd':540,'gbd':300,'fbd':5100,'zgrd':360}.items():scalar_map['urban_map_'+k]=str(num)
    bep+='bb_column(ctx,clsf,clsi, '+', '.join(scalar_map.get(n,n) for n in r.args)+');\nerr[tid]=ctx.error;\n}\n'
    (ROOT/'woof/core/kernels/urban_bem.cuh').write_text(bem, encoding='utf-8', newline='\n')
    (ROOT/'woof/core/kernels/urban_bep_bem.cu').write_text(bep, encoding='utf-8', newline='\n')
    metadata={'table':TABLE_LAYOUT,'class':CLASS_LAYOUT,'fixed':fixed,'per_level':per_level,'array_arguments':array_args,'scalar_arguments':scalar_args,'errors':ERRORS}
    if args.report is not None:
        (args.report/'bem_layout.json').write_text(json.dumps(metadata,indent=2)+'\n', newline='\n')
    import pprint
    layout = ('"""BEP_BEM\'s packed-table, class-table and workspace layouts.\n\n'
              'Generated by ``tools/transcribe_urban_bem.py`` together with\n'
              '``kernels/urban_bem.cuh`` and ``kernels/urban_bep_bem.cu``; regenerate, do not\n'
              'edit.  Offsets are in words of the float32 / int32 buffer each entry names;\n'
              'shapes are Fortran shapes (column-major).\n"""\n\n'
              'METADATA = ' + pprint.pformat(metadata, width=78, sort_dicts=False) + '\n')
    (ROOT/'woof/core/urban_bem_layout.py').write_text(layout, encoding='utf-8', newline='\n')
    print(f'Generated {len(all_routines)} routines; workspace = 4*({fixed}+{per_level}*(nz+1)) bytes/column')


if __name__ == '__main__':
    main()
