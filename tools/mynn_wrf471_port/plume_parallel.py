"""Split independent native updraft integrations; keep native ordered folds.

Plume-local scalars stay local. The otherwise dead UPQV vector holds28 words
of cross-launch state, within its 8*(nz+1) words for every admitted nz>=4.
"""
from pathlib import Path
import re
from prune_private import prune_private

def plume_parallel(s):
    s,_=prune_private(s)
    marker='__device__ M471_CALL /* Subroutine */ int dmp_mf__('
    assert s.count(marker)==2
    start=s.index(marker,s.index(marker)+len(marker))
    end=s.index('} /* dmp_mf__ */',start)
    fn=s[start:end]
    at=fn.index('    /* Function Body */')
    decl,code=fn[:at],fn[at:]
    loop=code.index('for (ip = 1; ip <= 8; ++ip) {\n\t    qcn = zero;')
    loop=code.rfind('#pragma unroll 1',0,loop)
    loop_end=code.index('/* L9029: */',loop)
    # The extracted loop is balanced and contains no shared environmental
    # writes after prune_private. Only ktop and nup2 need exact scalar folds.
    pre,main,tail=code[:loop],code[loop:loop_end],code[loop_end:]
    assert pre.count('{')-pre.count('}')==1
    assert main.count('{')==main.count('}')
    tail_start=tail.index('/* Computing MIN */')
    close=tail[:tail_start]
    assert re.sub(r'/\*.*?\*/|\s+','',close,flags=re.S)=='}else{nup2=rc0007;}'
    tail=tail[tail_start:]
    assert tail.count('{')==tail.count('}')
    names=['minwidth','pgfac','dzpmax','psig_w__','fluxportion','cf_thresh__','atot','cdet','csub','dl','dxsa']
    state='upqv'
    # Fixed values are defined before the active test. dl/dxsa exist only
    # inside it, so their stores remain inside the same native branch.
    active='    if (fltv2 > rc0002 && *maxwidth > minwidth && superadiabatic) {'
    assert pre.count(active)==1
    fixed='\n'.join(f'    {state}[{i}]={name};' for i,name in enumerate(names[:9]))
    init=f'    {state}[11]=0.f;\n    for(int m471_p=0;m471_p<8;++m471_p){{{state}[12+m471_p]=0.f;{state}[20+m471_p]=0.f;}}\n'
    pre=pre.replace(active,fixed+'\n'+init+active)
    pre+='\n    upqv[9]=dl;upqv[10]=dxsa;upqv[11]=1.f;\n    }\n'
    restore='\n'.join(f'    {name}={state}[{i}];' for i,name in enumerate(names))
    main=main.replace('for (ip = 1; ip <= 8; ++ip) {','for (ip = int(blockIdx.y)+1; ip <= int(blockIdx.y)+1; ++ip) {',1)
    # These pointers initially refer to the driver's public ktop local.
    # Use a private scalar until the ordered post-integration max fold.
    main='    integer m471_local_top=0;ktop=M471Ptr<integer>(&m471_local_top);nup2=8.f;\n'+main
    main+='\n    upqv[12+int(blockIdx.y)]=nup2;upqv[20+int(blockIdx.y)]=(real)m471_local_top;\n'
    fold='    nup2=8.f;*ktop=0;\n    for(int m471_p=0;m471_p<8;++m471_p){if(upqv[12+m471_p]==0.f)nup2=0.f;*ktop=max(*ktop,(integer)(real)upqv[20+m471_p]);}\n'
    # Inactive columns never read the conditional dl/dxsa state.
    main=restore.rsplit('    dl=',1)[0]+'\n    if(upqv[11]>0.f){\n    dl=upqv[9];dxsa=upqv[10];\n'+main+'\n    }\n'
    # tail doesn't use dl/dxsa, and compiler drops their dead restores.
    finish=restore.rsplit('    dl=',1)[0]+'\n'+fold+tail
    # C++ labels have function scope, including discarded constexpr arms.
    # Give the phased copy distinct names from the retained single routine.
    pre=re.sub(r'\bL(\d+)\b',r'Lsplit\1',pre)
    main=re.sub(r'\bL(\d+)\b',r'Lsplit\1',main)
    finish=re.sub(r'\bL(\d+)\b',r'Lsplit\1',finish)
    code2=('    if constexpr(M471_PHASE<0 || M471_PHASE==2){\n'+code+'\n    }else{\n'
           '    if constexpr(M471_PHASE==7){\n'+pre+'\n    }\n'
           '    if constexpr(M471_PHASE==8){\n'+main+'\n    }\n'
           '    if constexpr(M471_PHASE==9){\n'+finish+'\n    }\n    }\n    return 0;\n')
    s=s[:start]+decl+code2+s[end:]
    s=s.replace(marker,'template<int M471_PHASE>\n'+marker)
    assert s.count('\tdmp_mf__(')==1
    s=s.replace('\tdmp_mf__(', '\tdmp_mf__<M471_PHASE>(')
    s=s.replace('if constexpr(M471_PHASE<0 || M471_PHASE==2) {',
                'if constexpr(M471_PHASE<0 || M471_PHASE==2 || M471_PHASE==7 || M471_PHASE==8 || M471_PHASE==9) {',1)
    # The main launch performs only independent plume work. Setup and main
    # return before the rest of driver phase2 writes per-column outputs.
    needle='    if (FALSE_) {\n\tddmp_mf__('
    assert s.count(needle)==1
    s=s.replace(needle,'    if constexpr(M471_PHASE==7 || M471_PHASE==8)return 0;\n'+needle,1)
    entry=s.index('template<int M471_PHASE> __device__ __forceinline__ void m471_entry(')
    call=s.index(' mynnedmf_<M471_PHASE>(',entry)
    s=s[:call]+' if constexpr(M471_PHASE==8){if(a_bl_mynn_edmf==0)return;}\n'+s[call:]
    stores=s.index('f32[(44*(MYNN471_NZ-49)+2173)*n+c]=a_pblh;',entry)
    end_entry=s.index('extern "C" __global__ M471_LAUNCH void mynn471_step',stores)
    part=s[stores:end_entry]
    assert part.rstrip().endswith('}')
    # Prevent the eight plume workers from duplicating public scalar stores.
    cut=part.rfind('}')
    part='if constexpr(M471_PHASE!=8){\n'+part[:cut]+'}\n'+part[cut:]
    s=s[:stores]+part+s[end_entry:]
    signature='(int n,float*f32,int*i32,unsigned char*arena)'
    for phase in (7,8,9):
        s+='extern "C" __global__ M471_LAUNCH void mynn471_phase'+str(phase)+signature+'{m471_entry<'+str(phase)+'>(n,f32,i32,arena);}\n'
    return s

if __name__=='__main__':
    import sys
    Path(sys.argv[2]).write_text(plume_parallel(Path(sys.argv[1]).read_text()))
