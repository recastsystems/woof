import os,json,hashlib
import cupy as cp
from cupy.cuda import nvrtc
from woof.core.kernels import get_kernel,module_source,module_options
from woof.wrf_exact import ENABLED,effective_options
result={'strict':os.environ.get('GPUWM_WRF_EXACT')=='1','device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'compute_capability':cp.cuda.Device().compute_capability,'nvrtc':nvrtc.getVersion(),'modules':{}}
for unit,names in {'cldfra3':['cal_cldfra3_column','cldfra3_column_max'],'farms':['farms_column'],'cu_radiation':['cu_radiation_merge'],'wrf_stock_aer3':['wrf_stock_aer3_optics','wrf_stock_aer3_diagnostics','wrf_stock_aer3_diagnostics_fast'],'solar_eclipse':['solar_eclipse']}.items():
 result['modules'][unit]={'source_sha256':hashlib.sha256(module_source(unit).encode()).hexdigest(),'options':effective_options(module_options(unit)) if ENABLED else module_options(unit),'kernels':{name:get_kernel(unit,name).attributes for name in names}}
print(json.dumps(result,indent=2))
