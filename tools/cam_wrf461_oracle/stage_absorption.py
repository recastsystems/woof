"""Schedule independent native CAM absorption rows across GPU threads.

The preparation entry stops at radabs and leaves its array operands in the
column arena. The absorption entry evaluates disjoint rows, preserving every
expression and the within-row order. The finish entry rebuilds the inexpensive
prefix and consumes those binary64 results. All dependencies use stream order.
The ordinary entry remains the standalone native module replay interface.
"""
import re


def function(source, name):
    match = re.search(r'__device__ int '+name+r'\(CamState\* state,[^;{]*\)\s*\{', source)
    if match is None:
        raise ValueError(name)
    end = match.end()
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return match.start(), match.end(), end


def stop_after_calls(body, callee):
    positions = []
    for call in re.finditer(r'\b'+callee+r'\(', body):
        end = call.end()
        depth = 1
        while depth:
            depth += (body[end] == '(') - (body[end] == ')')
            end += 1
        assert body[end] == ';'
        positions.append(end+1)
    for end in reversed(positions):
        body = body[:end]+'\n    if (state->abs_phase == 1) return 0;'+body[end:]
    assert positions, callee
    return body


def stage_absorption(source, metadata):
    if 'void cam_absorption(' in source:
        raise ValueError('absorption schedule already generated')
    first, begin, end = function(source, 'radabs_')
    signature = source[first:begin]
    # All radabs operands after the five scalar dimensions are double arrays.
    params = re.findall(r'\*\s*(\w+)', signature)[6:]
    assert len(params) == 41 and params[-1] == 'aer_trn_ttl__', params
    body = source[begin:end-1]
    parallel = body
    init = parallel.index('    i__1 = state->cg_ntoplw__ - 1;')
    init_end = parallel.index('    i__1 = CAM_LEVELS;', init)
    parallel = parallel[:init]+'    if (state->abs_lane == 0) {\n'+parallel[init:init_end]+'    }\n'+parallel[init_end:]
    pairs = 'for (k1 = (CAM_LEVELS+1); k1 >= i__1; --k1)'
    neighbors = 'for (k2 = CAM_LEVELS; k2 >= i__1; --k2)'
    assert parallel.count(pairs) == parallel.count(neighbors) == 1
    parallel = parallel.replace(pairs, 'for (k1 = (CAM_LEVELS+1)-state->abs_lane; k1 >= i__1; k1 -= state->abs_stride)')
    parallel = parallel.replace(neighbors, 'for (k2 = CAM_LEVELS-state->abs_lane; k2 >= i__1; k2 -= state->abs_stride)')
    parallel = signature.replace('radabs_(', 'radabs_parallel_(')+parallel+'}\n'
    # Stop before automatic arrays are allocated: snapshot only arena pointers.
    dispatch = '\n    if (state->abs_phase == 1) {\n'
    dispatch += ''.join(f'        state->abs_args[{i}] = {name};\n' for i, name in enumerate(params))
    dispatch += '        return 0;\n    }\n'
    dispatch += '''    if (state->abs_phase == 2) {
        for (int i=0;i<(CAM_LEVELS+1)*(CAM_LEVELS+1);++i) abstot[i]=state->abs_cache[i];
        for (int i=0;i<CAM_LEVELS*4;++i) absnxt[i]=state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+i];
        return 0;
    }
'''
    source = source[:begin]+dispatch+body+source[end-1:end]+parallel+source[end:]
    for parent, child in [('radclwmx_', 'radabs_'), ('radctl_', 'radclwmx_'), ('camrad_', 'radctl_')]:
        _, begin, end = function(source, parent)
        source = source[:begin]+stop_after_calls(source[begin:end-1], child)+source[end-1:]
    # Preparation needs the LW prefix only. Finish retains the original SW
    # execution and obtains emissivity from the preparation pass as well.
    _, begin, end = function(source, 'radctl_')
    body = source[begin:end-1]
    assert body.count('if (CAM_SHORTWAVE) {') == 1
    body = body.replace('if (CAM_SHORTWAVE) {', 'if (CAM_SHORTWAVE && state->abs_phase != 1) {')
    source = source[:begin]+body+source[end-1:]
    _, begin, end = function(source, 'radclwmx_')
    body = source[begin:end-1]
    start = body.index('    radoz2_(state,')
    finish = body.index('    radabs_(state,', start)
    prefix = body[start:finish]
    body = body[:start]+'''    if (state->abs_phase != 2) {
'''+prefix+'''    }
    if (state->abs_phase == 1)
        for (int i=0;i<CAM_LEVELS+1;++i) state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+i]=emstot[emstot_offset+i];
    if (state->abs_phase == 2)
        for (int i=0;i<CAM_LEVELS+1;++i) emstot[emstot_offset+i]=state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+i];
'''+body[finish:]
    source = source[:begin]+body+source[end-1:]
    # Scalar fields are private to the worker; the helper owns its own copy.
    source = source.replace('unsigned long long scratch_offset;',
        'unsigned long long scratch_offset;\nvoid** abs_args; double* abs_cache; int abs_phase, abs_lane, abs_stride, abs_padding;')
    source = source.replace('state->scratch_offset=0;',
        'state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_padding=0;')
    wrapper_start = source.index('extern "C" __global__ __launch_bounds__(32) void cam_driver(')
    wrapper_end = source.index('extern "C" __global__', wrapper_start+1)
    staged = source[wrapper_start:wrapper_end]
    staged = staged.replace('void cam_driver(', 'void cam_driver_stage(')
    staged = staged.replace('unsigned char* workspace,',
        'unsigned char* workspace,int phase,void** descriptors,double* absorption,')
    staged = staged.replace('state->abs_stride=1;',
        'state->abs_stride=1; state->abs_phase=phase; state->abs_args=descriptors+worker*41; state->abs_cache=absorption+worker*CAM_ABS_VALUES;')
    # Include every arena declaration in radabs and its two trace-gas callees.
    # The sum bounds the deepest call chain without depending on nz=32.
    sizes = []
    for name in ('radabs_parallel_', 'trcab_', 'trcabn_'):
        _, begin, end = function(source, name)
        for match in re.finditer(r'cam_alloc<(\w+)>\(state, ', source[begin:end]):
            text = source[begin:end]
            pos = match.end(); stop = pos; depth = 1
            while depth:
                depth += (text[stop] == '(') - (text[stop] == ')')
                stop += 1
            count = text[pos:stop-1]
            sizes.append(f'(({count})*{8 if match[1]=="doublereal" else 4}+7)')
    budget = '(2048+(('+'+'.join(['64', *sizes, '7'])+')&~7))'
    definitions = '\n#define CAM_ABS_VALUES ((CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+CAM_LEVELS+1)\n'
    definitions += '#define CAM_ABS_WORK_BYTES '+budget+'\n'
    # One independent row per lane avoids a long tail on the final row. Eight
    # resident two-warp blocks cap register use at 128 on the target cards.
    helper = '\nextern "C" __global__ __launch_bounds__(64,8) void cam_absorption(int columns,const unsigned char* parents,void** descriptors,double* absorption,unsigned char* workspace){\n'
    helper += '''int thread=blockIdx.x*blockDim.x+threadIdx.x;
int column=thread/(CAM_LEVELS+1),lane=thread%(CAM_LEVELS+1);
if(column>=columns)return;
CamState* state=(CamState*)(workspace+thread*CAM_ABS_WORK_BYTES);
*state=*(const CamState*)(parents+column*CAM_WORK_BYTES);
state->scratch=workspace+thread*CAM_ABS_WORK_BYTES+2048;
state->scratch_offset=0; state->abs_phase=0; state->abs_lane=lane; state->abs_stride=CAM_LEVELS+1;
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
double* cache=absorption+column*CAM_ABS_VALUES;
void** operands=descriptors+column*41;
'''
    args = []
    for i, name in enumerate(params):
        args.append('cache' if name == 'abstot' else 'cache+(CAM_LEVELS+1)*(CAM_LEVELS+1)' if name == 'absnxt' else f'(double*)operands[{i}]')
    helper += 'radabs_parallel_(state,&lchnk,&ncol,&pcols,&pver,&pverp,'+','.join(args)+');\n}\n'
    # Bound all automatic arrays across all device frames, plus 16 KiB for
    # scalar locals, ABI arguments and compiler bookkeeping. This deliberately
    # sums mutually exclusive calls; it is a conservative source bound, not a
    # measurement advertised as a cross-compiler ceiling.
    local_counts = re.findall(r'\b\w+\[([^\]]+)\]\s*=\s*\{\}', source)
    # Eight bytes also bounds integer/real arrays in comma declarations.
    local_terms = [f'(({count})*8)' for count in local_counts]
    metadata['camrad']['local_frame_bound_expression'] = '16384+'+'+'.join(local_terms)
    source = source[:wrapper_start]+definitions+source[wrapper_start:]+staged+helper
    metadata['camrad']['absorption_workspace_expression'] = budget
    metadata['camrad']['absorption_lanes_expression'] = 'CAM_LEVELS+1'
    metadata['camrad']['absorption_operands'] = 41
    return source, metadata
