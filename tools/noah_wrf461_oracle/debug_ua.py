"""Read actual UA kernel intermediate words for one fixture column."""
from pathlib import Path
import cupy as cp
from woof.core import kernels
from woof.verify.noah_oracle import load_noah_oracle,noah_port_outputs
source=kernels.module_source("noah_ua")
# Print operation results, never substitute a numerical answer.
needle="    // canopy resistance"
trace=r'''
    if(idx==75)printf("DBG_GPU %08x %08x %08x %08x %08x %08x %08x %08x\\n",
        __float_as_uint(etpn),__float_as_uint(etp),__float_as_uint(rch),
        __float_as_uint(rr),__float_as_uint(th2),__float_as_uint(sfctmp),
        __float_as_uint(q2sat),__float_as_uint(dqsdt2));
'''
source=source.replace(needle,trace+needle)
needle="    real q1 = q2k + eta_kin * CP / rch;"
source=source.replace(needle,r'''
    if(idx==75)printf("DBG_SNOW_GPU %08x %08x %08x %08x %08x\\n",
        __float_as_uint(esnow),__float_as_uint(etns),__float_as_uint(dew),
        __float_as_uint(pc),__float_as_uint(sncovr));
'''+needle)
module=cp.RawModule(code=source,options=kernels.module_options("noah_ua"))
original=kernels.get_kernel
def traced(name,function):
    if name=="noah_ua":return module.get_function(function)
    return original(name,function)
kernels.get_kernel=traced
fixture=load_noah_oracle("noah-lsm-ua.csv",Path("woof/data/noah/ua_oracle"))
noah_port_outputs(fixture)
cp.cuda.runtime.deviceSynchronize()
