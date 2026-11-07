"""Build-time CUDA-to-HIP translation. Original kernel text remains authoritative.

Only the PTX forms present in the engine are accepted. New inline assembly fails
with its statement, rather than silently producing an incorrect AMD kernel.
"""
from __future__ import annotations
import ast
import re
from pathlib import Path

REVISION = 'gpuwm-hip-source-v5'


def _operands(text):
    # GCC constraint operand expressions may contain nested parentheses.
    result = []
    for match in re.finditer(r'"[^"\n]*"\s*\(', text):
        start = match.end(); depth = 1; end = start
        while end < len(text) and depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        result.append(text[start:end-1])
    return result


def _assembly(match):
    statement = match.group(0)
    strings = re.match(r'asm\s*(?:volatile\s*)?\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)', statement)
    if strings is None:
        raise ValueError('HIP cannot translate assembly: ' + statement[:180])
    op = ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', strings[1]))
    args = _operands(statement[strings.end():])
    normalized = re.sub(r'\s+', '', op)
    arithmetic = re.fullmatch(r'(add|sub|mul|div)\.rn\.f(32|64)%0,%1,%2;', normalized)
    if arithmetic:
        operation, bits = arithmetic.groups()
        return f'{args[0]} = __{"f" if bits == "32" else "d"}{operation}_rn({args[1]}, {args[2]});'
    if normalized.startswith('cvt.rn.f32.f64%0,%1;'):
        return f'{args[0]} = __double2float_rn({args[1]});'
    comparison = re.search(r'setp\.(lt|le|gt|ge|eq|ne)\.f(?:32|64)[^,]+,%1,%2;', normalized)
    if comparison:
        symbol = {'lt':'<','le':'<=','gt':'>','ge':'>=','eq':'==','ne':'!='}[comparison[1]]
        condition = f'({args[1]} {symbol} {args[2]})'
        if 'selp.u32%0,1,0,' in normalized:
            return f'{args[0]} = {condition};'
        if 'selp.f32%0,0f3F800000,0f00000000,' in normalized:
            return f'{args[0]} = {condition} ? 1.0f : 0.0f;'
        if re.search(r'selp\.(?:f32|f64)%0,%1,%2,', normalized):
            return f'{args[0]} = {condition} ? {args[1]} : {args[2]};'
        if 'selp.b32%0,%3,%4,' in normalized:
            return f'{args[0]} = {condition} ? {args[3]} : {args[4]};'
    if normalized == 'min.f32%0,%1,%2;':
        return f'{args[0]} = fminf({args[1]}, {args[2]});'
    if normalized == 'mov.b32%0,%1;':
        # This form in hex explicitly requests a bit-cast, not a conversion.
        return f'__builtin_memcpy(&({args[0]}), &({args[1]}), 4);'
    if normalized == 'xor.b32%0,%1,0x80000000;':
        return f'{args[0]} = ({args[1]}) ^ 0x80000000u;'
    if normalized == 'xor.b64%0,%1,0x8000000000000000;':
        return f'{args[0]} = ({args[1]}) ^ 0x8000000000000000ull;'
    if 'xor.b32bits,bits,0x80000000;' in normalized:
        return f'{args[0]} = -({args[1]});'
    if 'and.b32bits,bits,0x7fffffff;' in normalized:
        return f'{args[0]} = fabsf({args[1]});'
    if normalized == '' and '"memory"' in statement:
        return statement # Empty compiler barrier is portable GNU assembly.
    if normalized == 'trap;':
        return '__builtin_trap();'
    if normalized == 'mov.u64%0,%%globaltimer;':
        return f'{args[0]} = gpuwm_globaltimer_ns();'
    raise ValueError('HIP cannot translate assembly: ' + op)


# Balanced operand parentheses; semicolons within the PTX strings are ignored.
def _translate_assembly(source, *, nvidia=False):
    pattern = re.compile(r'\basm\s*(?:volatile\s*)?\(')
    output=[]; pos=0
    for match in pattern.finditer(source):
        if match.start() < pos: continue
        # Ignore assembly mentioned in line comments.
        line = source[source.rfind('\n', 0, match.start())+1:match.start()]
        if '//' in line: continue
        depth=1; quoted=False; escaped=False; end=match.end()
        while end<len(source) and depth:
            c=source[end]
            if escaped: escaped=False
            elif c=='\\' and quoted: escaped=True
            elif c=='"': quoted=not quoted
            elif not quoted: depth += (c=='(')-(c==')')
            end+=1
        while end<len(source) and source[end].isspace():end+=1
        if end>=len(source) or source[end]!=';':
            raise ValueError('Unterminated assembly at '+str(match.start()))
        statement=source[match.start():end+1]
        portable = _assembly(re.match(r'[\s\S]+',statement))
        # HIP's NVIDIA compiler accepts the original PTX. Keep its optimizer
        # barriers as well as its rounded operations, so its optimizer sees
        # the same boundaries as CUDA. AMD receives the validated portable form.
        output.extend((source[pos:match.start()], statement if nvidia else portable))
        pos=end+1
    return ''.join(output)+source[pos:]


def translate(source: str, *, nvidia=False) -> str:
    if source.startswith('// '+REVISION): return source
    source = expand_local_includes(source)
    alias = None
    depth = 0
    offset = 0
    for line in source.splitlines(keepends=True):
        if re.match(r'\s*#\s*(?:if|ifdef|ifndef)\b', line):
            depth += 1
        elif re.match(r'\s*#\s*endif\b', line):
            depth -= 1
        match = re.search(r'\btypedef\s+long\s+long\s+int64_t\s*;', line)
        if depth == 0 and match is not None:
            alias = offset + match.start()
            break
        offset += len(line)
    if not nvidia and alias is not None:
        # Some NVRTC-only units declare their own 64-bit index alias. HIP's
        # runtime already declares int64_t as long on LP64 hosts. Keep the
        # canonical long-long arithmetic under a private alias, avoiding a
        # conflicting redeclaration without changing any CUDA source.
        # RTC-guarded typedefs instead select stdint.h under HIPCC. Leave
        # those and any earlier uses of the standard type untouched.
        source=source[:alias]+re.sub(r'\bint64_t\b','gpuwm_cuda_int64_t',source[alias:])
    # CUDA's integer warp reduction is not a HIP builtin. All named lanes
    # receive the reduction of their logical 32-lane mask, including the
    # upper half of a CDNA wave. NVIDIA keeps its original intrinsic.
    if '__reduce_max_sync' in source:
        source=source.replace('__reduce_max_sync(', 'gpuwm_reduce_max_sync(')
        source='''
#if defined(__HIP_PLATFORM_AMD__)
template<class T> __device__ __forceinline__ T gpuwm_reduce_max_sync(unsigned mask,T value) {
    T result=value;
    for(int lane=0;lane<32;++lane) {
        T other=gpuwm_shfl_sync(mask,value,lane,32);
        if((mask&(1u<<lane)) && other>result) result=other;
    }
    return result;
}
#else
#define gpuwm_reduce_max_sync __reduce_max_sync
#endif
''' + source
    source = re.sub(r"\b__CUDACC__\b", "__HIPCC__", source)
    # Positive RTC-only guards omit device entry points from serial CPU
    # oracles; HIPCC AOT must retain those entry points too. Other RTC tests
    # still describe the actual compiler: notably !defined(RTC) chooses
    # stdint.h over runtime-only typedefs, which conflict under offline NVCC.
    source = re.sub(r'#\s*ifdef\s+__CUDACC_RTC__',
                    '#if defined(__HIPCC_RTC__) || defined(__HIPCC__)', source)
    source = re.sub(r"\b__CUDACC_RTC__\b", "__HIPCC_RTC__", source)
    # Fused RUC temporarily redefines __global__ and restores CUDA's private
    # __location__ spelling. Restore the public HIP attribute on that route.
    source = re.sub(r'(#\s*define\s+__global__\s+)__location__\(global\)',
                    r'\1__attribute__((global))',source)
    source = re.sub(r'#\s*ifdef\s+__CUDA_ARCH__',
                    '#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)', source)
    source = _translate_assembly(source, nvidia=nvidia)
    source=re.sub(r"\b__trap\s*\(","__builtin_trap(",source)
    source = re.sub(r'#\s*include\s*[<"]cuda_runtime(?:_api)?\.h[>"]', '#include <hip/hip_runtime.h>', source)
    source = re.sub(r'#\s*include\s*[<"]cub/[^>"]+[>"]', lambda m:'#if defined(__HIP_PLATFORM_AMD__)\n#include <hipcub/hipcub.hpp>\n#else\n'+m[0]+'\n#endif', source)
    source = re.sub(r'\bcub::', 'GPUWM_CUB_NAMESPACE::', source)
    source = re.sub(r'#\s*include\s*[<"]cuComplex\.h[>"]', '#include <hip/hip_complex.h>', source)
    for old,new in {'cuFloatComplex':'hipFloatComplex','cuDoubleComplex':'hipDoubleComplex',
                    'make_cuFloatComplex':'make_hipFloatComplex','make_cuDoubleComplex':'make_hipDoubleComplex',
                    'cuCrealf':'hipCrealf','cuCimagf':'hipCimagf','cuCreal':'hipCreal','cuCimag':'hipCimag'}.items():
        source=re.sub(r'\b'+old+r'\b',new,source)
    source=re.sub(r'\bcudaDeviceProp\b','hipDeviceProp_t',source)
    source=re.sub(r"#ifdef __HIP_DEVICE_COMPILE__\s*\n#define cub hipcub", "#if defined(__HIP_PLATFORM_AMD__)\n#define cub hipcub",source)
    # Native host launchers use this closed set of CUDA runtime calls. The
    # CUDA path still compiles its original source; HIP takes runtime twins.
    for name in ('DeviceProp','DeviceScheduleBlockingSync','DeviceSynchronize',
                 'Error_t','Free','GetDevice','GetDeviceProperties','GetErrorString',
                 'GetLastError','Malloc','Memcpy','MemcpyAsync','MemcpyDeviceToDevice',
                 'MemcpyDeviceToHost','MemcpyHostToDevice','SetDeviceFlags',
                 'StreamCreate','StreamDestroy','StreamSynchronize','Stream_t','Success'):
        source=re.sub(r'\bcuda'+name+r'\b','hip'+name,source)
    # CUDA masks are uint32 and refer to a logical 32-lane group. HIP's
    # native ballot is uint64 on gfx942. Wrappers shift the upper group
    # before narrowing, and shuffles explicitly use width=32 on both GPUs.
    for name in ('shfl_sync','shfl_up_sync','shfl_down_sync','shfl_xor_sync','ballot_sync','activemask','syncwarp','any_sync','all_sync','ldcg'):
        source = re.sub(r'\b__'+name+r'\s*\(', 'gpuwm_'+name+'(', source)
    for prefix in ('f','d'):
        for op in ('add','sub','mul','div','sqrt'):
            source=re.sub(r'\b__'+prefix+op+r'_rn\s*\(', 'gpuwm_'+prefix+op+'_rn(',source)
    source=re.sub(r'\b__launch_bounds__\s*\(', 'GPUWM_LAUNCH_BOUNDS(', source)
    return '// '+REVISION+'\n' + Path(__file__).with_name('hip_portable.h').read_text() + '\n' + source


def expand_local_includes(source, directory=None, stack=()):
    """Bind quoted package includes into the compilation and cache identity.

    CUDA and HIP still read the same files. Ambiguous basenames are refused;
    no arbitrary caller or working-directory files enter the source digest.
    """
    root=Path(__file__).parent
    def include(match):
        name=match[1]
        if name.startswith(('cupy/','hip/')):return match[0]
        candidates=([Path(directory)/name] if directory is not None else [])
        if not any(p.is_file() for p in candidates): candidates=list(root.rglob(name))
        candidates=list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
        if not candidates:return match[0]
        if len(candidates)!=1:raise ValueError('Ambiguous HIP include: '+name)
        path=candidates[0].resolve()
        if path in stack:raise ValueError('Recursive HIP include: '+name)
        text=expand_local_includes(path.read_text(encoding='utf-8'),path.parent,(*stack,path))
        return '\n// included '+str(path.relative_to(root))+'\n'+text+'\n'
    return re.sub(r'^\s*#\s*include\s*"([^"]+)"[^\n]*$',include,source,flags=re.M)


def options_for_amd(options):
    result=['-fhip-fp32-correctly-rounded-divide-sqrt']
    for option in options:
        if option in ('--fmad=false','-fmad=false'): result.append('-ffp-contract=off')
        elif option in ('--fmad=true','-fmad=true'): result.append('-ffp-contract=fast')
        elif option in ('--ftz=false','-ftz=false'): result.append('-fno-cuda-flush-denormals-to-zero')
        elif option in ('--ftz=true','-ftz=true'): result.append('-fcuda-flush-denormals-to-zero')
        # The default above supplies correctly rounded division/sqrt. NVCC's
        # false setting permits approximation; keeping stronger rounding is
        # valid on AMD and does not enable unrelated fast-math transforms.
        elif option.lstrip('-') in ('prec-div=true','prec-sqrt=true','prec-div=false','prec-sqrt=false'): pass
        # NVIDIA's register-count limit is an occupancy hint, not an AMD
        # register allocation contract. AMD uses its own resource planner.
        elif option.lstrip('-').startswith('maxrregcount='): pass
        elif option.startswith(('-arch=', '--gpu-architecture=', '--generate-code=', '-gencode=')): pass
        # NVCC's host compiler scheduling controls have no arithmetic or
        # device-code semantics. HIPCC parallelism belongs to its job pool.
        elif option in ('--split-compile=0', '--ptxas-options=--split-compile=0'): pass
        elif option.startswith(('--std=',)): result.append(option[1:])
        else: result.append(option)
    return tuple(result)
