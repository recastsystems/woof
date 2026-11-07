"""Read production-loader CUDA function attributes; launch no forecast."""
import json
from cupy_backends.cuda.libs import nvrtc
from woof.core.kernels import load_module, kernel_local_size_bytes
from woof.numerics import receipt as numerics_receipt
from pathlib import Path
import cupy as cp
from woof.certify.compile_platform import compile_platform_fingerprint
r={'compile_platform':compile_platform_fingerprint(),'nvrtc':nvrtc.getVersion(),'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'numerics':numerics_receipt(),'frames':{}}
for name,func in [('cammgmp','cammgmp_columns'),('camzm','camzm_columns'),('uwshcu','uwshcu_driver_columns')]:
 f=load_module(name).get_function(func)
 r['frames'][name]={'local_size_bytes':kernel_local_size_bytes(f)}
Path(__import__('sys').argv[1]).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
