"""Diagnostic word dump from the production WDM5 reflectivity kernel."""
from pathlib import Path
import json
import sys
import numpy as np
import cupy as cp
from woof.wrf_exact import install
install()
import woof.core.kernels as loader
from oracle_io import read_input
from run_woof import run
inp=Path(sys.argv[1]);out=Path(sys.argv[2]);case=read_input(inp)
source=loader.module_source("wdm5")
source=source.replace("float *refl, int nz, int ny, int nx)","float *refl, unsigned int *debug_words, int nz, int ny, int nx)")
anchor="        float dbz = (float)(10.0 * log10("
extra="""        int dd = (k * stride + col) * 16;
        debug_words[dd] = __float_as_uint(temp[k]);
        debug_words[dd+1] = __float_as_uint(rr[k]);
        debug_words[dd+2] = __float_as_uint(rs[k]);
        debug_words[dd+3] = __float_as_uint(nr[k]);
        debug_words[dd+4] = __float_as_uint(ze_rain[k]);
        debug_words[dd+5] = __float_as_uint(ze_snow[k]);
        float cc = __fmul_rn(0.189247310f,__fdiv_rn(6.0f,WDM_PI));
        cc=__fmul_rn(cc,__fdiv_rn(6.0f,WDM_PI));
        cc=__fmul_rn(cc,__fdiv_rn(WDM5_XAM_S,900.0f));
        cc=__fmul_rn(cc,__fdiv_rn(WDM5_XAM_S,900.0f));
        debug_words[dd+6] = __float_as_uint(cc);
        double dv[4] = {n0_r[k],n0_s[k],ilamr[k],ilams[k]};
        for (int q=0;q<4;++q) {
            unsigned long long word = __double_as_longlong(dv[q]);
            debug_words[dd+7+2*q] = (unsigned int)(word >> 32);
            debug_words[dd+8+2*q] = (unsigned int)word;
        }
"""
assert source.count(anchor)==1
source=source.replace(anchor,extra+anchor)
module=loader.compile_wdm5_no_ftz(source,"wdm5-reflectivity-diagnostic")
debug=cp.zeros(case.nz*case.ncol*16,cp.uint32)
original=loader.get_kernel
def get(unit,symbol):
    if unit=="wdm5" and symbol=="refl10cm_wdm5_column":
        kernel=module.get_function(symbol)
        return lambda grid,block,args:kernel(grid,block,args[:-3]+(debug,)+args[-3:])
    return original(unit,symbol)
loader.get_kernel=get
run(inp,out)
a=cp.asnumpy(debug).reshape(case.nz,case.ncol,16)
rows=[]
for col in [0,48,85]:
 for k in range(case.nz):
  words=[f"{int(v):08X}" for v in a[k,col,:7]]
  words += [f"{int(a[k,col,7+2*q]):08X}{int(a[k,col,8+2*q]):08X}" for q in range(4)]
  rows.append(dict(column=col,level=k,words=words))
out.with_suffix(".json").write_text(json.dumps(rows,indent=1))
