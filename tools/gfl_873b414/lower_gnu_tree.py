"""Reproduce the complete GFL CUDA transcription from GNU 15.2's original tree.

Input is the unmodified GFL module compiled with pinned strict flags.
This tool is development-only; production does not parse/compile Fortran.
All procedure bodies remain in the generated source. Character diagnostics
are retained; Fortran formatted writes become diagnostic bits, because a
CUDA thread has no Fortran I/O unit. GFL uses the deterministic random inputs declared by its MPAS wrapper.
"""
from pathlib import Path
import re
import sys

KINDS = {'real(kind=4)': 'GFReal', 'integer(kind=4)': 'int',
         'integer(kind=8)': 'long long', 'logical(kind=4)': 'bool',
         'character(kind=1)': 'char'}


def lower(tree: str, authority: str, path='module_cu_gfl_deep.F90', call_names=None) -> str:
    # Original line ranges, including preprocessor alternate signatures.
    citations = {}
    starts = list(re.finditer(r'^[ \t]*(?:subroutine|(?:real|integer)[ \t]+function)[ \t]+(\w+)[ \t]*\(', authority, re.I | re.M))
    for m in starts:
        name = m[1].lower()
        if name in citations:
            continue
        end = re.search(r'^[ \t]*end[ \t]+(?:subroutine|function)\b[^\n]*', authority[m.end():], re.I | re.M)
        lo = authority[:m.start()].count('\n') + 1
        hi = authority[:m.end() + end.end()].count('\n') + 1
        citations[name] = (lo, hi)
    tree = re.sub(r'^__attribute__[^\n]*\n', '', tree, flags=re.M)
    names = re.findall(r'^(?:void|real\(kind=4\)|integer\(kind=4\)) (\w+) \(', tree, re.M)
    if set(names) != set(citations):
        raise ValueError(('incomplete procedure inventory', set(names) ^ set(citations)))
    # Scalar Fortran references are pointer parameters in the C ABI.
    tree = re.sub(r'(real\(kind=4\)|integer\(kind=[48]\)|logical\(kind=4\)) & restrict', r'\1 *', tree)
    # A character scalar is already addressed as a string, not dereferenced.
    tree = re.sub(r'character\(kind=1\)\[1:[^]]+\] & restrict', 'char *', tree)
    # Array-of-50-character diagnostics retain their element extent.
    tree = re.sub(r'character\(kind=1\)\[0:[^]]*\]\[1:50\]', 'GFText', tree)
    tree = re.sub(r'(real\(kind=4\)|integer\(kind=[48]\)|logical\(kind=4\))\[0:[^]]*\]', r'\1', tree)
    for src, dst in KINDS.items():
        tree = tree.replace(src, dst)
    tree = re.sub(r'struct array0([1234])_GFReal &', r'GFDescriptor<\1> *', tree)
    tree = re.sub(r'struct array0([1234])_GFReal', r'GFDescriptor<\1>', tree)
    tree = re.sub(r'GFReal\[\d+\] \* restrict', 'GFReal *', tree)
    tree = tree.replace('&zuh2', 'zuh2')
    tree = re.sub(r'char (\w+)\[1:(\d+)\]', r'char \1[\2]', tree)
    tree = tree.replace('&cumulus', 'cumulus')
    tree = re.sub(r'\w+ = \{CLOBBER\};', '; // GNU end-of-lifetime marker', tree)
    tree = tree.replace(' restrict', '').replace('static ', '')
    # fct1d3 returns before its later STOP. Its error SAVE can consequently
    # become true on a returning call and must survive subsequent calls.
    if path == 'module_cu_gfl_deep.F90':
        assert tree.count('bool error = 0;') == 1
        tree = tree.replace('bool error = 0;', 'bool &error = workspace.fct1d3_error;')
    tree = re.sub(r'\{\.elem_len=(\d+), \.version=(\d+), \.rank=(\d+), \.type=(\d+)\}', r'GFDType{\1,\2,\3,\4}', tree)
    tree = re.sub(r'\b0B\b', 'nullptr', tree)
    # GNU cleanup regions reclaim automatic arrays. GFFrame provides the
    # same lifetime without device malloc/free or a C++ exception region.
    tree = re.sub(r'^  (?:try|finally)\n', '', tree, flags=re.M)
    tree = re.sub(r'\b(\w+) = \{\};', r'gf_memset(\1, 0, sizeof(\1));', tree)
    tree = re.sub(r'\bInf\b', 'gf_infinity()', tree)
    tree = re.sub(r'\bNan\b', 'gf_nan()', tree)
    tree = tree.replace('(void *)', '(char *)')
    # GNU's POINTER_PLUS_EXPR prints a typed pointer plus a byte displacement.
    # C++ pointer addition would scale that displacement a second time. This
    # occurs throughout assumed-shape cloud-base and chemistry descriptors.
    pointer_plus = re.compile(r'\(\(GFReal \*\) ([A-Za-z_][\w.]*) \+ ')
    while (m := pointer_plus.search(tree)):
        depth, j = 1, m.end()
        while depth:
            if tree[j] == '(': depth += 1
            if tree[j] == ')': depth -= 1
            j += 1
        displacement = tree[m.end():j-1]
        tree = (tree[:m.start()] + '((GFReal *) ((char *) ' + m[1] +
                ' + ' + displacement + '))' + tree[j:])
    # GCC pointer-to-array notation becomes an ordinary flattened pointer.
    tree = re.sub(r'\(\*(\w+)\)\[', r'\1[', tree)
    tree = re.sub(r'&("(?:[^"\\]|\\.)*")\[1\]\{lb: 1 sz: 1\}', r'(char *) \1', tree)
    tree = tree.replace('(bitsizetype)', '(unsigned long long)').replace('(sizetype)', '(unsigned long long)')
    tree = re.sub(r'\b(?:bitsizetype|sizetype)\b', 'unsigned long long', tree)

    # Non-C original-tree expression nodes. Their argument order is preserved.
    pattern = re.compile(r'\b(MAX_EXPR|MIN_EXPR|ABS_EXPR|NON_LVALUE_EXPR|SAVE_EXPR) <')
    while (m := pattern.search(tree)):
        depth, j = 1, m.end()
        while depth:
            if tree[j] == '<': depth += 1
            if tree[j] == '>': depth -= 1
            j += 1
        body = tree[m.end():j-1]
        macro = {'MAX_EXPR': 'gf_max', 'MIN_EXPR': 'gf_min', 'ABS_EXPR': 'gf_abs',
                 'NON_LVALUE_EXPR': '', 'SAVE_EXPR': ''}[m[1]]
        tree = tree[:m.start()] + macro + '(' + body + ')' + tree[j:]

    # Rename compiler identifiers without changing decimal numbers or strings.
    tokens = re.compile(r'"(?:[^"\\]|\\.)*"|\b[A-Za-z_]\w*(?:\.\d+)+|(?<![\w.])(?:\d+\.\d*(?:e[+-]?\d+)?|\d+e[+-]?\d+)', re.I)
    def token(m):
        s = m[0]
        if s.startswith('"'): return s
        if s[0].isalpha() or s[0] == '_': return s.replace('.', '_')
        return s + 'f'
    tree = tokens.sub(token, tree)
    # GNU prints unsigned int constants without a C suffix. On an LP64 C++
    # host/NVRTC, an unsuffixed decimal above INT_MAX becomes signed long,
    # defeating the original modulo-2**32 range tests (including drafts 4/5).
    # Keep the unsigned 32-bit constant type from the original expression.
    tree = re.sub(r'(?<![\w.])\d{10}(?![\w.])',
                  lambda m: m[0] + 'u' if 2**31 <= int(m[0]) < 2**32 else m[0], tree)
    tree = re.sub(r'^\s*struct __st_parameter_dt .*?;\n', '', tree, flags=re.M)
    tree = re.sub(r'^\s*dt_parm_\d+\..*?;\n', '', tree, flags=re.M)
    tree = re.sub(r'^\s*_gfortran_(?:transfer_\w+|st_write_done) \(.*?;\n', '', tree, flags=re.M)
    tree = re.sub(r'_gfortran_st_write \(.*?\);', 'workspace.diagnostics |= 1;', tree)
    tree = tree.replace('_gfortran_compare_string', 'gf_compare_string')
    tree = tree.replace('_gfortran_stop_string', 'gf_stop_string')
    tree = tree.replace('__builtin_expect', 'gf_expect').replace('__builtin_memcmp', 'gf_memcmp')
    tree = tree.replace('__builtin_memmove', 'gf_memmove').replace('__builtin_memset', 'gf_memset')
    tree = tree.replace('__builtin_malloc (', 'workspace.allocate (')
    tree = re.sub(r'__builtin_free \([^;]*\);', '; // reclaimed at procedure return', tree)
    for src, dst in [('__builtin_atanf','gf_atan'), ('__builtin_sinf','gf_sin'), ('__builtin_tgammaf','gf_tgamma'), ('__builtin_expf','gf_exp'), ('__builtin_logf','gf_log'), ('__builtin_log10f','gf_log10'), ('__builtin_powf','gf_pow'),
                     ('__builtin_powif','gf_powi'), ('__builtin_sqrtf','gf_sqrt')]:
        tree = tree.replace(src, dst)
    # Every procedure receives the same column-owned scratch arena.
    for name in sorted(call_names or names, key=len, reverse=True):
        tree = re.sub(r'\b' + name + r' \(', name + ' (workspace, ', tree)
    prototypes = []
    for name in names:
        pat = re.compile(r'^((?:void|GFReal|int) ' + name + r' \()workspace, ([^\n]+)\n\{', re.M)
        m = pat.search(tree)
        signature = m[1] + 'GFWorkspace &workspace, ' + m[2]
        lo, hi = citations[name]
        citation=f'// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/{path}:{lo}-{hi}'
        prototypes.append(citation+'\n__device__ ' + signature + ';')
        replacement = citation+'\n__device__ ' + signature + '\n{\n  GFFrame frame(workspace);'
        tree = tree[:m.start()] + replacement + tree[m.end():]
    if '__builtin_' in tree or '_gfortran_' in tree or '_EXPR' in tree:
        Path('native/unhandled.cuh').write_text(tree)
        raise ValueError(('unhandled GNU tree construct', sorted(set(re.findall(r'(?:__builtin_|_gfortran_)\w+', tree)))))
    return '\n'.join(prototypes) + '\n\n' + tree


if __name__ == '__main__':
    Path(sys.argv[3]).write_text(lower(Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text(), Path(sys.argv[2]).name))
