"""Move generated automatic column arrays into an explicit device arena.

The transformation changes storage only. Every array keeps its original
extent, padding and index expressions, including f2c's shifted origins.
All callees receive disjoint slices so nested calls cannot alias locals.
"""
import re

POINTER = '''
// A pointer carries its column stride through f2c's shifted array origins.
// Warp neighbours address adjacent words, including inside nested calls.
template<class T> struct gsfc_ptr {
    T *p; int stride;
    __device__ gsfc_ptr(T *value, int step=1): p(value), stride(step) {}
    __device__ gsfc_ptr(const T *value): p(const_cast<T *>(value)), stride(1) {}
    __device__ T& operator*() const { return *p; }
    __device__ T& operator[](int i) const { return p[i*stride]; }
    __device__ gsfc_ptr operator+(int i) const { return gsfc_ptr(p+i*stride,stride); }
    __device__ gsfc_ptr operator-(int i) const { return gsfc_ptr(p-i*stride,stride); }
    __device__ gsfc_ptr& operator+=(int i) { p+=i*stride; return *this; }
    __device__ gsfc_ptr& operator-=(int i) { p-=i*stride; return *this; }
    __device__ gsfc_ptr& operator--() { p-=stride; return *this; }
    __device__ gsfc_ptr& operator++() { p+=stride; return *this; }
};
'''


def move_arrays(source):
    words = 0
    def array(match):
        nonlocal words
        kind, name, count = match.groups()
        offset = words
        words += int(count)
        return f'    {kind} *{name} = ({kind} *)(workspace + {offset});'
    source, count = re.subn(r'^    (real|integer|logical) (\w+)\[(\d+)\];', array, source, flags=re.M)
    assert count > 100, 'unexpected generated array layout'
    source = source.replace('(const real *tables,', '(real *workspace, const real *tables,')
    source = re.sub(r'\b(\w+_)\(tables,', r'\1(workspace, tables,', source)
    return f'#define GSFC_WORKSPACE_WORDS {words}\n' + source, words


def coalesce_arrays(source):
    """Transpose each 32-column arena tile, preserving every scalar operation."""
    source = source.replace('real *workspace', 'GSFC_WORKSPACE_ARGUMENT')
    source = source.replace('const real *tables', 'GSFC_TABLE_ARGUMENT')
    # Taking a shifted array address must retain its stride in the callee.
    address = r'(?<!&)&(?!&)\s*(\w+)\s*\[([^\[\]]*)\]'
    source = re.sub(address, r'(\1 + (\2))', source)
    assert not re.search(r'(?<!&)&(?!&)\s*\w+\s*\[', source), 'unhandled nested array address'
    source = re.sub(r'\b(real|integer|logical)\s*\*', r'gsfc_ptr<\1> ', source)
    source = re.sub(r'\(gsfc_ptr<(real|integer|logical)> \)\(workspace \+ (\d+)\)',
                    lambda m: f'gsfc_ptr<{m[1]}>(({m[1]} *)(workspace + {int(m[2])*32}),32)', source)
    source = source.replace('GSFC_WORKSPACE_ARGUMENT', 'real *workspace')
    source = source.replace('GSFC_TABLE_ARGUMENT', 'const real *tables')
    return POINTER + source


def single_column_fast(source):
    """Specialize integer dimensions and fuse independent flux-level passes.

    The packed entry calls sorad with m=n=ndim=1. Exposing these integers
    lets NVRTC remove the horizontal loops and scalarize five temporaries.
    cldflx still visits every sky state in WRF order. Its two level passes
    become one: each level's flux is accumulated immediately, without
    storing and reloading three intermediate profiles. No spectral term,
    cloud state or floating expression is removed.

    This generates the default image only. The strict translation remains
    byte-identical to the image used by the native captured-input replay.
    """
    assert '// Bottom-to-top packed layers.' not in source
    result, count = re.subn(r'\*(?:m|n|ndim)\b', '1', source)
    assert count == 434, 'unexpected GSFC dimension layout'
    for name in ('ch', 'cm', 'ct', 'fdndir', 'fdndif'):
        pattern = (rf'gsfc_ptr<real> {name}_storage = gsfc_ptr<real>\('
                   rf'\(real \*\)\(workspace \+ \d+\),32\);')
        result, changed = re.subn(
            pattern,
            rf'real {name}_local[18]; gsfc_ptr<real> {name}_storage = '
            rf'gsfc_ptr<real>({name}_local);', result)
        assert changed == 1, 'unexpected GSFC scalar layout: ' + name

    begin = result.index(' __device__ int cldflx_(')
    end = result.index(' __device__ int flxco2_(', begin)
    body = result[begin:end]
    first = body.index('i__1 = *np + 1;\n\t\tfor (k = 2;')
    compute_start = body.index('denm = 1.f /', first)
    compute_end = body.index('\n\t\t\t}', compute_start)
    top_start = body.index('flxdn[i__ + (j + flxdn_dim2)', compute_end)
    top_end = body.index('\n\t\t    }', top_start)
    sum_start = body.index('if (ih == 1 && im == 1 && is == 1)', top_end)
    sum_end = body.index('\n\t\t\t}', sum_start)
    last = body.index('i__1 = 1;', sum_end)

    def scalar_flux(text):
        names = {'flxdn': 'flux_net', 'flxdnu': 'flux_up',
                 'flxdnd': 'flux_down'}
        return re.sub(r'\b(flxdn|flxdnu|flxdnd)\[[^\]]+\]',
                      lambda match: names[match[1]], text)

    fused = ('i__=1; j=1;\nfor(k=1;k<=*np+1;++k){\n'
             'real flux_net,flux_up,flux_down;\nif(k==1){\n' +
             scalar_flux(body[top_start:top_end]) + '\n}else{\n' +
             scalar_flux(body[compute_start:compute_end]) + '\n}\n' +
             scalar_flux(body[sum_start:sum_end]) + '\n}\n\t\t')
    body = body[:first] + fused + body[last:]
    return result[:begin] + body + result[end:]
