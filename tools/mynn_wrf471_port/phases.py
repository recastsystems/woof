"""Split the generated driver at native call boundaries without reordering work."""
import re


def phases(text):
    declaration='__device__ M471_CALL /* Subroutine */ int mynnedmf_('
    assert text.count(declaration)==2
    text=text.replace(declaration,'template<int M471_PHASE>\n'+declaration)
    start=text.index(declaration,text.index(declaration)+len(declaration))
    stop=text.index('} /* mynnedmf_ */',start)
    driver=text[start:stop]
    body=driver.index('    /* Function Body */')
    declarations,code=driver[:body],driver[body:]
    slot=0
    def saved(m):
        nonlocal slot
        typ=m[1];items=m[2].strip()
        if '=' in items:return m[0]
        names=[n.strip() for n in items.split(',')]
        if any(not re.fullmatch(r'\w+',n) for n in names):raise ValueError(items)
        result=[]
        for name in names:
            offset=f'(1184*(MYNN471_NZ-49)+85800+{slot*4})'
            result.append(f'    {typ} local_{name}; {typ} &{name}=M471_PHASE<0 ? local_{name} : *m471_array<{typ}>(m471_work,{offset},m471_n);')
            slot+=1
        return '\n'.join(result)
    declarations=re.sub(r'^    (real)\s+([^;]+);',saved,declarations,flags=re.M)
    declarations=declarations.replace('    real rmol = 0.f;',saved(re.match(r'(real) (.+)', 'real rmol'))+'\n    if constexpr(M471_PHASE<=0) rmol=0.f;')
    declarations=declarations.replace('    integer ktop_plume__;',saved(re.match(r'(integer) (.+)','integer ktop_plume__')))
    assert slot*4<1008
    markers=['    /* Function Body */','    mym_condensation__(',
             '    if (*bl_mynn_edmf__ > 0) {','    mym_turbulence__(',
             '    mym_predict__(', '    mynn_tendencies__(',
             '    if (*mix_chem__) {','    return 0;']
    indices=[code.index(m) for m in markers]
    assert indices==sorted(indices)
    pieces=[code[a:b] for a,b in zip(indices,indices[1:])]
    # Every cut is between complete native statements, at driver brace depth zero.
    for piece in pieces:
        clean=re.sub(r'/\*.*?\*/','',piece,flags=re.S)
        assert clean.count('{')==clean.count('}')
    driver=declarations+'\n'.join(f'    if constexpr(M471_PHASE<0 || M471_PHASE=={i}) {{\n{piece}\n    }}' for i,piece in enumerate(pieces))+'\n    return 0;\n'
    text=text[:start]+driver+text[stop:]
    index=text.index('extern "C" __global__ M471_LAUNCH void mynn471_step(')
    engine=text[index:]
    entry=engine.replace('extern "C" __global__ M471_LAUNCH void mynn471_step(',
                         'template<int M471_PHASE> __device__ __forceinline__ void m471_entry(')
    entry=entry.replace('mynnedmf_(', 'mynnedmf_<M471_PHASE>(')
    signature='(int n,float*f32,int*i32,unsigned char*arena)'
    kernels='extern "C" __global__ M471_LAUNCH void mynn471_step'+signature+'{m471_entry<-1>(n,f32,i32,arena);}\n'
    for phase in range(7):
        kernels+='extern "C" __global__ M471_LAUNCH void mynn471_phase'+str(phase)+signature+'{m471_entry<'+str(phase)+'>(n,f32,i32,arena);}\n'
    return text[:index]+entry+kernels
