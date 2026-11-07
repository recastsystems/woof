"""Expose the column ABI constants and scalar scratch to the CUDA compiler.

Every entry handles one column per worker and is specialized to CAM_LEVELS.
The F2C pointer ABI otherwise hides those facts from loop/index optimization.
No floating point expression or reduction order is changed.
"""
import re


def optimize_columns(source):
    pattern = re.compile(r'__device__\s+(?:int|integer|logical|doublereal)\s+\w+\(CamState\* state,[^;{]*\)\s*\{')
    # The generic standalone ABI keeps its native logical operands. Production
    # supplies immutable configuration defines, allowing dead spectra and their
    # device call frames to disappear before register/stack allocation.
    source = ('#ifdef CAM_LONGWAVE_MODE\n#define CAM_LONGWAVE (CAM_LONGWAVE_MODE==1)\n#else\n#define CAM_LONGWAVE (*dolw)\n#endif\n'
              '#ifdef CAM_SHORTWAVE_MODE\n#define CAM_SHORTWAVE (CAM_SHORTWAVE_MODE==1)\n#else\n#define CAM_SHORTWAVE (*dosw)\n#endif\n') + source
    edits = []
    for match in pattern.finditer(source):
        begin = match.end()
        end = begin
        depth = 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        body = source[begin:end-1]
        for name, value in {'pcols': '1', 'ncol': '1',
                            'pver': 'CAM_LEVELS', 'pverp': '(CAM_LEVELS+1)'}.items():
            if re.search(r'\*\s*' + name + r'\b', match[0]):
                # Native INTENT(IN), fixed by all four CUDA entry wrappers.
                body = re.sub(r'\*' + name + r'\b', value, body)
        for name, define in (('dolw','CAM_LONGWAVE'),('dosw','CAM_SHORTWAVE')):
            if re.search(r'\*\s*'+name+r'\b',match[0]):
                body=re.sub(r'\*\s*'+name+r'\b',define,body)
        # A one-element automatic array is a scalar, not an arena allocation.
        # Its {} initialization exactly preserves cam_alloc's zero fill.
        owns_operands = any(name+'(' in match[0] for name in
                            ('camrad_', 'radctl_', 'radclwmx_'))
        if not owns_operands:
            body = re.sub(r'\*(\w+) = cam_alloc<(?:doublereal|integer|real|logical)>\(state, 1\)',
                          r'\1[1] = {}', body)
        # CUDA local memory interleaves the same private-array index across
        # lanes. Small automatic arrays therefore avoid the column-strided
        # global arena transactions; constant indices can also become registers.
        # Keep larger spectral and layer-pair arrays in the bounded arena.
        arrays = []
        for allocation in re.finditer(r'\*(\w+) = cam_alloc<(\w+)>\(state, ', body):
            # These frames own operands exported to the absorption kernel.
            # Their pointers must remain in the global arena between launches.
            if owns_operands:
                continue
            start = stop = allocation.end()
            nesting = 1
            while nesting:
                nesting += (body[stop] == '(') - (body[stop] == ')')
                stop += 1
            count = body[start:stop-1]
            try:
                elements = eval(count, {'__builtins__': {}}, {'CAM_LEVELS': 32})
            except (NameError, SyntaxError):
                continue
            if 1 < elements <= 66:
                arrays.append((allocation.start(), stop,
                               allocation[1]+'['+count+'] = {}'))
        for start, stop, value in reversed(arrays):
            body = body[:start] + value + body[stop:]
        edits.append((begin, end-1, body))
    for begin, end, body in reversed(edits):
        source = source[:begin] + body + source[end:]
    return source
