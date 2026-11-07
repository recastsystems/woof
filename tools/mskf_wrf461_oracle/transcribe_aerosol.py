"""Closed WRF MSKF aerosol transcription generator, executed on S4."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import sys

from tools.lake_wrf461_oracle import translate as seed

split = seed.split
Variable = seed.Variable
TOKEN = re.compile(r"\s*(\d+(?:\.\d*)?(?:[ed][+-]?\d+)?(?:_r[84k])?|\.\d+(?:[ed][+-]?\d+)?(?:_r[84k])?|\.[a-z]+\.|[a-z_]\w*|\*\*|<=|>=|==|/=|[^\s])")


def statements(path):
    carry, start = '', 0
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split('!', 1)[0].strip().lower()
        if not line or line.startswith('#'):
            continue
        if not carry:
            start = n
        carry += ' ' + line.lstrip('&').rstrip('&').strip()
        if not line.endswith('&'):
            for item in split(carry.strip(), ';'):
                yield start, item
            carry = ''
    assert not carry


def declaration(s):
    s = re.sub(r'real\s*\(\s*(r8|rk)\s*\)', 'real(r8)', s)
    if not re.match(r'(real|integer|logical)\b', s):
        return None
    if '::' not in s:
        m = re.match(r'(real\(r8\)|real|integer|logical)\s+(.*)', s)
        if not m:
            raise ValueError(s)
        s = m[1] + ' :: ' + m[2]
    return seed.declaration(s)


def expr(source, symbols, slices=None):
    p = Expression(source, symbols, slices or {})
    value = p.parse()
    if p.i != len(p.tokens):
        raise ValueError((source, p.tokens[p.i:]))
    return value


class Expression(seed.Expression):
    def __init__(self, source, symbols, slices):
        source=re.sub(r'(\.[a-z]+\.)', r' \1 ', source)
        self.tokens = TOKEN.findall(source)
        self.i, self.symbols, self.slices = 0, symbols, slices

    def parse(self, minimum=0):
        tok = self.take()
        if tok in ('+', '-', '.not.'):
            left = '(' + ('!' if tok == '.not.' else tok) + self.parse(3 if tok == '.not.' else 6) + ')'
        elif tok == '(':
            left = self.parse()
            assert self.take() == ')'
            left = '(' + left + ')'
        elif re.match(r'\d|\.\d', tok):
            if '_r' in tok:
                left = re.sub(r'_r[84k]', '', tok).replace('d', 'e')
                if not any(c in left for c in '.e'):
                    left += '.0'
            elif any(c in tok for c in '.ed'):
                left = tok.replace('d', 'e') + ('' if 'd' in tok else 'f')
            else:
                left = tok
        elif tok in ('.true.', '.false.'):
            left = tok[1:-1]
        else:
            left = tok
            if self.peek() == '(':
                self.take()
                rawargs, begin, depth = [], self.i, 0
                while True:
                    t = self.take()
                    if t == '(':
                        depth += 1
                    elif t == ')' and depth:
                        depth -= 1
                    elif t in (',', ')') and depth == 0:
                        rawargs.append(''.join(self.tokens[begin:self.i-1]))
                        begin = self.i
                        if t == ')':
                            break
                args = []
                for j, arg in enumerate(rawargs):
                    if ':' in arg and tok in self.symbols:
                        var = self.symbols[tok]
                        lo = arg.split(':')[0] or seed.bounds(var.dims[j])[0]
                        args.append('(' + expr(lo, self.symbols) + '+_slice)' if tok in self.slices else ':')
                    else:
                        args.append(expr(arg, self.symbols))
                if tok in self.symbols and self.symbols[tok].dims:
                    left = tok + ('.slice(' + ','.join(args[:-1]) + ')' if args[-1]==':' else '(' + ','.join(args) + ')')
                elif tok in ('real', 'dble', 'int', 'nint', 'conv'):
                    dtype = ('double' if tok in ('dble', 'conv') or (tok == 'real' and len(rawargs)>1) else 'float' if tok=='real' else 'int')
                    left = '(' + dtype + ')(' + ('round('+args[0]+')' if tok=='nint' else args[0]) + ')'
                elif tok in ('min', 'max', 'exp', 'log', 'log10', 'abs', 'aint', 'sqrt'):
                    left = 'ma_' + tok + '(' + ','.join(args) + ')'
                elif tok in ('erf', 'erfc'):
                    left = 'cam_' + tok + '(' + ','.join(args) + ')'
                else:
                    left = tok + '(' + ','.join(args) + ')'
            elif tok in self.symbols and self.symbols[tok].dims and tok in self.slices:
                left = tok + '.at(_slice)'
        while self.peek() in seed.PREC and seed.PREC[self.peek()] >= minimum:
            op = self.take()
            begin = self.i
            right = self.parse(seed.PREC[op] + (0 if op == '**' else 1))
            rawright = ''.join(self.tokens[begin:self.i])
            if op == '**':
                if re.fullmatch(r'-?\d+', rawright):
                    left = 'ma_powi<' + rawright + '>(' + left + ')'
                else:
                    left = 'ma_pow(' + left + ',' + right + ')'
            elif op == '/':
                left = '(' + left + '/' + right + ')'
            else:
                left = '(' + left + seed.OPS.get(op, op) + right + ')'
        return left


def assignment(line, symbols):
    lhs, rhs = (x.strip() for x in line.split('=', 1))
    base = lhs.split('(')[0].strip()
    slices = {}
    if base in symbols and symbols[base].dims:
        var = symbols[base]
        if '(' not in lhs:
            slices[base] = True
            count = base + '.size()'
        else:
            indices = split(lhs[lhs.index('(')+1:-1])
            for j, dim in enumerate(indices):
                if ':' in dim:
                    lo, hi = dim.split(':')
                    dlo, dhi = seed.bounds(var.dims[j])
                    lo, hi = lo or dlo, hi or dhi
                    count = '(' + expr(hi, symbols) + ')-(' + expr(lo, symbols) + ')+1'
                    slices[base] = True
    if slices:
        for name, var in symbols.items():
            if var.dims:
                slices[name] = True
        return 'for(int _slice=0; _slice<' + count + '; ++_slice) ' + expr(lhs, symbols, slices) + ' = ' + expr(rhs, symbols, slices) + ';'
    return expr(lhs, symbols) + ' = ' + expr(rhs, symbols) + ';'


def data_line(line, symbols):
    rest, output = line[4:].strip(), []
    while rest:
        names, rest = rest.split('/', 1)
        values, rest = rest.split('/', 1)
        rest = rest.lstrip(' ,')
        names, vals = split(names), split(values)
        if len(names)==1 and names[0] in symbols and symbols[names[0]].dims:
            name = names[0]
            output += [f'{name}.at({i})={expr(v,symbols)};' for i,v in enumerate(vals)]
        else:
            assert len(names)==len(vals), (names,vals)
            output += [f'{k}={expr(v,symbols)};' for k,v in zip(names,vals)]
    return ' '.join(output)


def action(line, symbols, function, n):
    if line.startswith('call '):
        call = line[5:].strip()
        if call.startswith('wrf_error_fatal'):
            return 'error=32; return;'
        return expr(call, symbols) + '; if(error) return' + (' retval' if function else '') + ';'
    if line == 'return':
        return 'return' + (' retval' if function else '') + ';'
    if line == 'cycle':
        return 'continue;'
    if line == 'exit':
        return 'break;'
    if re.match(r'go\s*to\s+\d+',line):
        return 'goto ma_label_' + re.findall(r'\d+',line)[0] + ';'
    if '=' in line:
        if function:
            line = re.sub('^' + function + r'\s*=', 'retval=', line)
        return assignment(line,symbols)
    raise ValueError((n,line))


def body_line(n, line, symbols, function):
    line = re.sub(r"else\s+if", "else if", line)
    if line.startswith(('implicit ', 'use ', 'save', 'write', 'print', 'format')):
        return ''
    if line.startswith('data '):
        return data_line(line,symbols)
    label = re.match(r'(\d+)\s+(.*)',line)
    prefix = ''
    if label:
        line = label[2]
        if line == 'continue':
            return 'ma_label_'+label[1]+':;'
        prefix = 'ma_label_'+label[1]+': '
    if re.match(r'conv\(i\)\s*=',line):
        return ''
    if line in ('enddo','end do','endif','end if'):
        return '}'
    if line == 'else':
        return '} else {'
    if line.startswith(('if ', 'if(', 'else if', 'elseif')):
        start = line.index('(')
        depth=0
        for i in range(start,len(line)):
            depth += (line[i]=='(')-(line[i]==')')
            if depth==0:
                end=i
                break
        condition=expr(line[start+1:end],symbols)
        tail=line[end+1:].strip()
        head='} else if' if line.startswith('else') else 'if'
        if tail=='then':
            return head+'('+condition+') {'
        return head+'('+condition+') { '+action(tail,symbols,function,n)+' }'
    if line.startswith('do '):
        values=line[3:]
        values=re.sub(r'^\d+\s+','',values)
        v, values=values.split('=',1)
        v=v.strip()
        values=split(values)
        start,end=values[:2]
        step=values[2] if len(values)>2 else '1'
        op='>=' if step.startswith('-') else '<='
        return f'for({v}={expr(start,symbols)}; {v}{op}{expr(end,symbols)}; {v}+={expr(step,symbols)}) {{'
    return prefix+action(line,symbols,function,n)


SUPPORT=r'''
// Runtime arrays use one double-sized cell per element, interleaved across lanes.
template<class T> struct MaArray {
 double* ptr; int n1,n2,n3,l1,l2,l3,step=1;
 __device__ T& at(int x) const { return *reinterpret_cast<T*>(ptr+(size_t)x*step*32); }
 __device__ MaArray<T> slice(int i,int j) const { return {ptr+(size_t)((i-l1)+n1*(j-l2))*32,n3,1,1,l3,1,1,n1*n2}; }
 __device__ int size() const { return n1*n2*n3; }
 __device__ T& operator()(int i) const { return at(i-l1); }
 __device__ T& operator()(int i,int j) const { return at((i-l1)+n1*(j-l2)); }
 __device__ T& operator()(int i,int j,int k) const { return at((i-l1)+n1*((j-l2)+n2*(k-l3))); }
};
struct MaArena {
 double* ptr; int used;
 template<class T> __device__ MaArray<T> make(int a,int b,int c=1,int d=1,int e=1,int f=1) {
  MaArray<T> x{ptr+(size_t)used*32,b-a+1,d-c+1,f-e+1,a,c,e};
  used+=x.size(); return x;
 }
};
template<class A,class B> __device__ auto ma_max(A a,B b)->decltype(a+b) { using T=decltype(a+b); return a>b?(T)a:(T)b; }
template<class A,class B,class... C> __device__ auto ma_max(A a,B b,C... c)->decltype(a+b) { return ma_max(ma_max(a,b),c...); }
template<class A,class B> __device__ auto ma_min(A a,B b)->decltype(a+b) { using T=decltype(a+b); return a<b?(T)a:(T)b; }
template<class A,class B,class... C> __device__ auto ma_min(A a,B b,C... c)->decltype(a+b) { return ma_min(ma_min(a,b),c...); }
#ifdef GPUWM_MSKF_FAST
__device__ double ma_exp(double x) {return exp(x);}
__device__ float ma_exp(float x) {return gfk_exp(x);}
__device__ double ma_log(double x) {return log(x);}
__device__ float ma_log(float x) {return gfk_log(x);}
#else
__device__ double ma_exp(double x) {return g64_exp::exp(x);}
__device__ float ma_exp(float x) {return gfk_exp(x);}
__device__ double ma_log(double x) {return g64_log::log(x);}
__device__ float ma_log(float x) {return gfk_log(x);}
#endif
__device__ double ma_sqrt(double x) {return sqrt(x);}
__device__ float ma_sqrt(float x) {return sqrtf(x);}
__device__ double ma_abs(double x) {return fabs(x);}
__device__ float ma_abs(float x) {return fabsf(x);}
__device__ int ma_abs(int x) {return abs(x);}
__device__ double ma_aint(double x) {return trunc(x);}
__device__ float ma_aint(float x) {return truncf(x);}
__device__ double ma_log10(double x) {return log10(x);}
template<class A,class B> __device__ auto ma_pow(A x,B y)->decltype(x+y) {
 using T=decltype(x+y);
#ifdef GPUWM_MSKF_FAST
 if constexpr(sizeof(T)==8) return pow((double)x,(double)y);
 else return gfk_pow((float)x,(float)y);
#else
 if constexpr(sizeof(T)==8) return g64_pow::pow((double)x,(double)y);
 else return gfk_pow((float)x,(float)y);
#endif
}
template<int N,class T> __device__ T ma_powi(T x) {
 if constexpr(N==0) return T(1);
 else if constexpr(N<0) return T(1)/ma_powi<-N>(x);
 else if constexpr(N==1) return x;
 else if constexpr(N==2) return x*x;
 else if constexpr(N==3) return (x*x)*x;
 else if constexpr(N==4) {T a=x*x; return a*a;}
 else if constexpr(N==5) {T a=x*x;return (a*x)*a;}
 else if constexpr(N==6) {T a=(x*x)*x;return a*a;}
 else {T a=ma_powi<N/2>(x); if constexpr(N%2) return (a*a)*x; else return a*a;}
}
'''


def parse_source(path, wanted, module=False):
    mod, init, routines, current = {}, [], [], None
    for n,s in statements(path):
        if module and n>=3424:
            break
        begin=re.match(r'(?:real\s+)?(subroutine|function)\s+(\w+)\s*(?:\((.*)\))?$',s)
        if begin:
            name=begin[2]
            current={'name':name,'ret':'double' if begin[1]=='function' else 'void',
                     'args':split(begin[3]) if begin[3] else [],'vars':{},'body':[],'line':n} if name in wanted else None
            if current:
                routines.append(current)
            continue
        if s.startswith(('end subroutine','end function')):
            current=None
            continue
        # Skip routines not in the requested set and module directives.
        if current is None and (not module or n>=202):
            continue
        if s.startswith('parameter'):
            dest=mod if current is None else current['vars']
            for definition in split(s[s.index('(')+1:-1]):
                name,value=definition.split('=',1)
                dest[name.strip()].initial=value.strip()
                dest[name.strip()].const=True
            continue
        if s.startswith(('module ','end module','use ','implicit ','private','public','contains','save','character','parameter','data aername')):
            continue
        if s.startswith('dimension '):
            for d in split(s[len('dimension '):]):
                m=re.fullmatch(r'(\w+)\((.*)\)',d)
                current['vars'][m[1]].dims=split(m[2])
            continue
        dec=declaration(s)
        if dec is not None:
            dest=mod if current is None else current['vars']
            for var in dec:
                dest[var.name]=var
        elif current:
            current['body'].append((n,s))
        elif s.startswith('data ') and not s.startswith('data aername'):
            init.append((n,s))
        else:
            raise ValueError((n,s))
    return mod,init,routines


def definition(var,symbols,arena=True):
    if var.dims:
        dims=[expr(v,symbols) for dim in var.dims for v in seed.bounds(dim)]
        return f'MaArray<{var.typ}> {var.name}' + (f'=arena.make<{var.typ}>('+','.join(dims)+');' if arena else ';')
    const='const ' if var.const else ''
    initial='='+expr(var.initial,symbols) if var.initial else ''
    return const+var.typ+' '+var.name+initial+';'


def generate(source,cam):
    expected='4bca54b294def08ffc98ef69a30650c943e2c394f34802e75e50c7f1a2896fcb'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    wanted={'mskf_mphyi','mskf_mphy','mskf_gamma','mskf_activate','mskf_maxsat',
            'mskf_nucleati','mskf_hetero','mskf_hf','mskf_polysvp'}
    mod,init,rs=parse_source(source,wanted,True)
    _,_,cr=parse_source(cam,{'calerf_r8'})
    rs+=cr
    output=['// Generated closed transcription from WRF 4.6.1. Source SHA256: '+expected,
            '// Generation refuses unknown statements. Each emitted statement has its WRF line.',SUPPORT,
            'struct MaPhysics { MaArena arena; int error=0;']
    # Mutable module state stays per column. Fast callers may share only the
    # invariant initialization tables, keeping all later arena offsets intact.
    for v in mod.values():
        if v.name=='r8':
            continue
        if v.const and v.initial and not v.dims:
            output.append('static constexpr '+v.typ+' '+v.name+'='+expr(v.initial,mod)+';')
        else:
            output.append(definition(v,mod,False))
    output += ['__device__ MaPhysics(double* ptr, double* cached=nullptr):arena{ptr,0} {',
               'if(!ptr) return;',
               '// Only species constants use the shared table. Keep the original arena offset',
               '// for every mutable per-column array, including the strict replay layout.',
               'MaArena table_arena{cached ? cached : ptr,0};']
    for v in mod.values():
        if v.dims:
            dims=[expr(x,mod) for d in v.dims for x in seed.bounds(d)]
            output.append(v.name+'=table_arena.make<'+v.typ+'>('+','.join(dims)+');')
    output += ['arena.used=table_arena.used;', 'if(!cached) {']
    output += [body_line(n,s,mod,None) for n,s in init]
    for v in mod.values():
        if v.dims and v.initial:
            vals=split(v.initial.removeprefix('(/').removesuffix('/)'))
            output += [v.name+'.at('+str(i)+')='+expr(x,mod)+';' for i,x in enumerate(vals)]
    output += ['}', 'mskf_mphyi(cached != nullptr); }']
    for rt in rs:
        symbols=mod|rt['vars']
        args=[]
        for name in rt['args']:
            v=rt['vars'][name]
            outputs={'mskf_activate':{'nact'},'mskf_maxsat':{'smax'},'mskf_nucleati':{'nuci','onihf','oniimm','onidep','onimey'},'mskf_hetero':{'nis','nid'},'mskf_hf':{'ni'},'calerf_r8':{'result'}}
            args.append((f'MaArray<{v.typ}>' if v.dims else v.typ+('&' if name in outputs.get(rt['name'],set()) else ''))+' '+name)
        if rt['name']=='mskf_mphyi':
            args.append('bool cached=false')
        output.append('__device__ '+rt['ret']+' '+rt['name']+'('+','.join(args)+') {')
        output.append('int arena_saved=arena.used;')
        if rt['ret']!='void':
            output.append('double retval;')
        for v in rt['vars'].values():
            if v.name not in rt['args'] and v.name!=rt['name'] and v.name not in ('rk','r8'):
                output.append(definition(v,symbols))
                if v.dims and v.initial:
                    vals=split(v.initial.removeprefix('(/').removesuffix('/)'))
                    output += [v.name+'.at('+str(i)+')='+expr(x,symbols)+';' for i,x in enumerate(vals)]
        if rt['name']=='mskf_gamma':
            # END DO in the original labeled loops is represented by CONTINUE.
            pass
        for n,s in rt['body']:
            if rt['name']=='mskf_mphyi' and n==362:
                output.append('if(!cached) {')
            if rt['name']=='mskf_mphyi' and n==385:
                output.append('}')
            if rt['name']=='mskf_mphy' and n==2202:
                assert 'pgam(k)' in s
                s=s.replace('pgam(k)','pgam(k-1)')
            if re.fullmatch(r'(260|290|350)\s+continue',s) and rt['name']=='mskf_gamma':
                value='}'
            else:
                try:
                    value=body_line(n,s,symbols,rt['name'] if rt['ret']!='void' else None)
                except Exception as e:
                    raise ValueError((rt['name'],n,s)) from e
            if value:
                # A local arena is reclaimed on every return, including early returns.
                value=re.sub(r'\breturn( retval)?;',lambda m:'{ arena.used=arena_saved; return'+(m[1] or '')+'; }',value)
                output += [f'// WRF:{n}',value]
        output.append('arena.used=arena_saved;'+(' return retval;' if rt['ret']!='void' else ''))
        output.append('}')
    output += ['__device__ double cam_erf(double x) { double y;int j=0;calerf_r8(x,y,j);return y;}',
               '__device__ double cam_erfc(double x) { double y;int j=1;calerf_r8(x,y,j);return y;}',
               '};']
    notice=source.read_text()
    notice=notice[notice.index('!    Copyright(C) 1996'):notice.index('      function derf(x)')].rstrip()
    output += ['/* Upstream notice retained verbatim.',notice,'*/']
    return '\n'.join(output)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('cam',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text(generate(args.source,args.cam))
