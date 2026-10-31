"""Regenerate the complete backend from the three pristine fork modules."""
import argparse,ast,json,re
from pathlib import Path
from translate_module import generate


def lower_gnu_operand_order(text):
    """Preserve GNU 15.2 strict option-loop ordering at native F90:1250,2869,3005,3015,3122-3124."""
    for old,new in (
        ('qbuoy1(k) = mpas_add(mpas_add(mpas_mul(elq, mpas_add(mpas_mul(sh(k), gh(k)), gamv)), mpas_mul(p5, tkeprod_dn(k))), mpas_mul(p5, tkeprod_up(k)));',
         'qbuoy1(k) = mpas_add(mpas_add(mpas_mul(elq, mpas_add(mpas_mul(gh(k), sh(k)), gamv)), mpas_mul(p5, tkeprod_dn(k))), mpas_mul(p5, tkeprod_up(k)));'),
        ('mynn_dmul(mynn_dmul(q3sq, mpas_mul(a2, a2fac)), mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel)))',
         'mynn_dmul((tke_budget == 1 ? mynn_dmul(q3sq, mpas_mul(a2, a2fac)) : mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel))), (tke_budget == 1 ? mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel)) : mynn_dmul(q3sq, mpas_mul(a2, a2fac))))'),
        ('mpas_div(r6_neg(mpas_mul(mynn_ipow<3>(ust), zet)), mpas_mul(karman, dz1(k)))',
         'r6_neg(mpas_div(mpas_mul(mynn_ipow<3>(ust), zet), mpas_mul(karman, dz1(k))))'),
        ('mynn_dadd(e2, mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel))',
         'mynn_dadd(mynn_dmul(mpas_mul(mpas_mul(three, c1), e5c), gmel), e2)'),
        ('gamt = mynn_ddiv(r6_neg(mynn_dmul(e1, mynn_enum)), eden);',
         'gamt = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));'),
        ('gamq = mynn_ddiv(r6_neg(mynn_dmul(e1, mynn_enum)), eden);',
         'gamq = r6_neg((float)mynn_ddiv(mynn_dmul(mynn_enum, e1), eden));'),
    ):
        assert old in text, old
        text = text.replace(old, new)
    for name in ("add","sub","mul","div"):
        text=text.replace("mpas_"+name+"(","r6_"+name+"(")
    return text


def inline_entry_routines(text):
    """Inline the once-per-column entry routines into the kernel (rrfs/mynn-speed, 2026-10-28).

    Their ~100-array by-value parameter blocks otherwise sit on the per-thread stack: the kernel frame was
    4,432 B fast / 4,688 B strict, over the 4 KiB at which device_memory_release returns the local-memory store
    after every PBL phase (two device syncs a step). Inlined: 2,640 B / 2,816 B. Words are unchanged (no
    contraction is possible: fast operations are round-to-nearest intrinsics, strict compiles --fmad=false).
    """
    for name in ("f_mynnedmf_driver", "f_mynnedmf"):
        old = "__device__ __noinline__ void " + name + "("
        assert text.count(old) == 1, old
        text = text.replace(old, "__device__ __forceinline__ void " + name + "(")
    return text


KERNEL_TILE_LINE='MynnArena arena{workspace+(unsigned long long)(column_index-column_index%tile_columns)*workspace_stride,0,tile_columns,column_index%tile_columns};'
#: Lane widths compiled into the image, one kernel each (rrfs_mynnedmf_run_l<W>): 1 is the one-thread-per-column
#: shape a 3 km mesh fills the card with; 4 and 16 fill it at about 10,000 and 2,500 columns.
LANE_WIDTHS=(1,4,16)


def workspace_frames(routines):
    """Arena frames and calls, including function references inside expressions.

    Each translated routine allocates its locals on entry and MynnFrame restores
    the cursor on return. Sibling calls never need storage at the same time.
    Keep all branches, chemistry and optional routines in the bound.
    """
    return {name: {
        'arrays': [{'kind': v.kind, 'dimensions': v.dimensions}
                   for n, v in routine.variables.items()
                   if n not in routine.arguments and v.dimensions],
        'calls': sorted({called for _, statement in routine.body
                         for called in re.findall(r'\b(\w+)\s*\(', statement)
                         if called in routines}),
    } for name, routine in routines.items()}


def emit(root,target,receipt, *, keep_intermediate=False, model_pin="1d6e370e", scheme_pin="c942418"):
    module=root/'src/core_atmosphere/physics/physics_noaa/MYNN-EDMF'
    paths=[module/'MPAS/module_bl_mynnedmf_common.F90',module/'module_bl_mynnedmf.F90',module/'MPAS/module_bl_mynnedmf_driver.F90']
    temporary=receipt.with_suffix('.cuh')
    text,routines=generate(paths,temporary,receipt)
    if not keep_intermediate:
        temporary.unlink()
    text = text.replace('RRFS 1d6e370e ', 'RRFS ' + model_pin + ' ').replace('MYNN-EDMF c942418;', 'MYNN-EDMF ' + scheme_pin + ';')
    run=routines['mynnedmf_driver']
    params=['int batch_first','int batch_count','unsigned char* workspace','unsigned long long workspace_stride','int* error_flags']
    bindings=[];values=[]
    for n in run.arguments:
        v=run.variables[n]
        if n=='errmsg':values.append('message');continue
        if n=='errflg':values.append('error_flag');continue
        kind='int' if v.kind=='bool' else v.kind
        params.append(f'{kind}{"*" if v.dimensions else ""} {n}')
        if v.dimensions:
            bindings.append(f'MynnArray<{v.kind},{len(v.dimensions)}> a_{n};a_{n}.data={n};')
            stride='1'
            for d,dim in enumerate(v.dimensions):
                edges=dim.split(':');lo=edges[0] if len(edges)==2 else '1';hi=edges[-1]
                bindings.append(f'a_{n}.lo[{d}]={lo};a_{n}.hi[{d}]={hi};a_{n}.stride[{d}]={stride};')
                stride+=f'*({hi}-{lo}+1)'
            values.append('a_'+n)
        elif v.optional:
            # All optional scalar values are part of the MPAS actual argument list.
            params.append('int present_'+n)
            values.append(f'(present_{n} ? MynnOptional<{v.kind}>({n}) : MynnOptional<{v.kind}>())')
        elif v.kind=='bool':values.append(f'({n}!=0)')
        else:values.append(n)
    names=[p.split()[-1].lstrip('*') for p in params]
    # W=4 is the throughput kernel. Cap only this width: capping W=1
    # regressed tiled call24, while W=16 already wins the small shape.
    # The strict image keeps its original register allocation.
    bounds = '#ifdef HEX_NUMERICS_FAST\n#define MYNN_L4_REGS __maxnreg__(160)\n#else\n#define MYNN_L4_REGS\n#endif\n'
    kernel=bounds+'''
// RRFS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/MYNN-EDMF/MPAS/module_bl_mynnedmf_driver.F90:17-653
template<int MYNN_LANES> __device__ __forceinline__ void rrfs_mynnedmf_column('''+','.join(params)+''') {
constexpr int lane_width=MYNN_LANES;
// Lane groups (lane_plan.py, rrfs/speed2-mynn-edmf): lane_width lanes (1..32, a power of two) per column, so a
// warp holds 32/lane_width columns; a column's automatic arrays live in a tile shared with the warp's other columns,
// word w of column c at lane c % (32/lane_width), so a group's lanes on consecutive elements read one line.
int thread=blockIdx.x*blockDim.x+threadIdx.x;
int column_index=thread/lane_width;
if(column_index>=batch_count) return;
int index=batch_first+column_index;
constexpr int tile_columns=32/lane_width;
MynnArena arena{workspace+(unsigned long long)(column_index-column_index%tile_columns)*workspace_stride,0,tile_columns,column_index%tile_columns};
MynnColumn<MYNN_LANES> column{arena};
column.lane_sub=thread%lane_width;
column.lane_mask=lane_width==32 ? 0xffffffffu : ((1u<<(lane_width%32))-1u)<<((threadIdx.x&31)/lane_width*lane_width);
column.active_i=its+index%(ite-its+1);
column.active_j=jts+index/(ite-its+1);
MynnText message;int error_flag=0;
'''+ '\n'.join(bindings)+'\ncolumn.f_mynnedmf_driver('+','.join(values)+');\nerror_flags[index]=error_flag;\n}\n'+''.join(
        f'extern "C" __global__ {"MYNN_L4_REGS " if w == 4 else ""}void rrfs_mynnedmf_run_l{w}('+','.join(params)+f') {{ rrfs_mynnedmf_column<{w}>('+','.join(names)+'); }\n'
        for w in LANE_WIDTHS)
    kernel = kernel.replace('RRFS 1d6e370e ', 'RRFS ' + model_pin + ' ').replace(':17-653', f':{run.start}-{run.end}')
    schema={n:{k:getattr(v,k) for k in ('kind','dimensions','intent','optional')} for n in run.arguments for v in [run.variables[n]]}
    # Sum of all automatic arrays is a conservative allocation bound: frames
    # restore the cursor on return, so no call path can exceed this sum.
    arrays=[{'kind':v.kind,'dimensions':v.dimensions} for r in routines.values() for n,v in r.variables.items() if n not in r.arguments and v.dimensions]
    old=target.read_text()
    prefix=old.split('# BEGIN GENERATED RRFS MYNN-EDMF')[0]
    suffix=old.split('# END GENERATED RRFS MYNN-EDMF')[-1] if '# END GENERATED RRFS MYNN-EDMF' in old else ''
    result=prefix+'# BEGIN GENERATED RRFS MYNN-EDMF\nMODULE_CUDA = r\'\'\'\n'+inline_entry_routines(lower_gnu_operand_order(text+kernel))+"'''\n\nARGUMENTS = "+repr(schema)+'\nWORKSPACE_ARRAYS = '+repr(arrays)+'\nWORKSPACE_FRAMES = '+repr(workspace_frames(routines))+'\n# END GENERATED RRFS MYNN-EDMF'+suffix
    target.write_text(result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--target',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--keep-intermediate',action='store_true')
    p.add_argument('--model-pin',default='1d6e370e')
    p.add_argument('--scheme-pin',default='c942418')
    a=p.parse_args();emit(a.root,a.target,a.receipt,keep_intermediate=a.keep_intermediate,model_pin=a.model_pin,scheme_pin=a.scheme_pin)
