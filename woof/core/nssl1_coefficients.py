"""Cold CUDA specialization of WRF's one-moment initialization.

Native authority: module_mp_nssl_2mom.F:1283,1413,1487-1521,
1865-1952,1997,2128-2146. No coefficient calculation or readback in a step.
The unchanged default header keeps its literal native initializer words.
"""
from __future__ import annotations
import re
import numpy as np


def specialize(source, cccn, shapes, hail):
    from woof.core.nssl2_contract import nssl2_cccn_concentration
    cccn = float(nssl2_cccn_concentration(cccn))
    if cccn == 0.5e9 and tuple(shapes) == (0.,1.):
        return source, ''
    def scalar(name, value):
        nonlocal source
        source, count = re.subn(r'(?:static )?constexpr (?:float|int) '+name+r'=[^;]+;',
                                f'/* native init {name} */\n#define {name} ({value})\n', source)
        if count != 1:raise ValueError(f'missing native coefficient {name}')
    scalar('ccn', float(np.float32(cccn)).hex()+'f')
    scalar('cwccn', 'ccn')
    scalar('qccn', 'lake_div(ccn,rho00)')
    scalar('alphah', 'NSSL2_ALPHA_G');scalar('alphahl', 'NSSL2_ALPHA_H')
    scalar('icdx','NSSL2_DRAG_G');scalar('icdxhl','NSSL2_DRAG_H')
    scalar('rz','RZ');scalar('rzhl','RZHL')
    # These initialised array words also enter single-moment source processes,
    # not just terminal velocities. Preserve native expression ordering.
    for name, g, h in (
        ('dnu','NSSL2_ALPHA_G','NSSL2_ALPHA_H'),
        ('xnu','lake_div(NSSL2_ALPHA_G-2.f,3.f)','lake_div(NSSL2_ALPHA_H-2.f,3.f)'),
        ('ax','NSSL2_AX_G','NSSL2_AX_H'),('bx','NSSL2_BX_G','NSSL2_BX_H')):
        pat=r'(struct coeff_'+name+r' \{.*?\n)'
        m=re.search(pat,source);assert m,name
        code=m[0]
        for index,value in ((4,g),(5,h)):
            code,n=re.subn(r'case '+str(index)+r':return [^;]+;',f'case {index}:return {value};',code)
            assert n==1
        source=source[:m.start()]+code+source[m.end():]
    # delbk/delabk use the single-moment snow distribution; borrowing the
    # two-moment table would silently use xnu=-0.8,xmu=1 instead of -2/3,1/3.
    if tuple(shapes[:2]) == (0.,1.):
        return source, ''
    table='__device__ float nssl1_init_da0[9],nssl1_init_da1[9],nssl1_init_dab0[9][9],nssl1_init_dab1[9][9];\n'
    source=table+source
    for name in ('da0','da1','dab0','dab1'):
        args='int i0,int i1' if name.startswith('dab') else 'int i0'
        lookup='[i0][i1]' if name.startswith('dab') else '[i0]'
        code=f'struct coeff_{name} {{ __device__ __forceinline__ float operator()({args}) const {{return nssl1_init_{name}{lookup};}}}};\n'
        source,n=re.subn(r'struct coeff_'+name+r' \{[^\n]*\n',code,source);assert n==1
    last=8 if hail else 7
    initialization='''
extern "C" __global__ void nssl1_init_coefficients() {
    using C = nssl1::Column<1>;
    const int n=threadIdx.x, il=3+n/6, ic=3+n%6;
    if(n>=36 || il>LAST || ic>LAST)return;
    const float bb=C::coeff_bb{}(il), nu=C::coeff_xnu{}(il), mu=C::coeff_xmu{}(il);
    if(il==ic) {
        nssl1_init_da0[il]=delbk(bb,nu,mu,0);
        nssl1_init_da1[il]=delbk(bb,nu,mu,1);
        nssl1_init_dab0[il][ic]=0.f;nssl1_init_dab1[il][ic]=0.f;
    } else {
        nssl1_init_dab0[il][ic]=delabk(bb,C::coeff_bb{}(ic),nu,C::coeff_xnu{}(ic),mu,C::coeff_xmu{}(ic),0);
        nssl1_init_dab1[il][ic]=delabk(bb,C::coeff_bb{}(ic),nu,C::coeff_xnu{}(ic),mu,C::coeff_xmu{}(ic),1);
    }
}
'''.replace('LAST',str(last))
    return source, initialization
