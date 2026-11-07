"""Measure every shallow-kernel entry through the production loader."""
import json
import argparse
from pathlib import Path
import sys
import cupy as cp
from cupy_backends.cuda.libs import nvrtc
from woof.core.kernels import load_module, kernel_local_size_bytes
from woof.certify.compile_platform import compile_platform_fingerprint
from woof.numerics import receipt
from woof import nvrtc_ptx_cache
from woof.kernel_aot import compiler_identity, direct_metadata, effective_options
from woof.kernel_pack import write

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
parser.add_argument('--recipes',type=Path)
args=parser.parse_args()
requests=[]
original=nvrtc_ptx_cache.compile_using_nvrtc
def recorded(source,options=(),arch=None,filename='kern.cu',**kwargs):
    # Record the production request before its persistent image cache lookup.
    # A warm cache must not leave an empty AOT catalog.
    requests.append(dict(source=source,options=list(effective_options(options)),
                         **direct_metadata(filename)))
    return original(source,options,arch,filename,**kwargs)
if args.recipes:
    nvrtc_ptx_cache.compile_using_nvrtc=recorded

r = dict(compile_platform=compile_platform_fingerprint(), nvrtc=nvrtc.getVersion(),
         device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
         numerics=receipt(), frames={})
for name, entry in [('grims', 'grims_column'), ('shcu_nscv', 'shcu_nscv_column'),
                    ('deng', 'deng_column')]:
    f = load_module(name).get_function(entry)
    r['frames'][name] = dict(local_size_bytes=kernel_local_size_bytes(f))
args.output.write_text(json.dumps(r, indent=2)+'\n')
if args.recipes:
    write(args.recipes,compiler_identity(),requests)
print(json.dumps(r))
