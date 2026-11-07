"""Measure selected production images, warming their source-bound cache in parallel."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import json
import re
import sys

UNITS=('noah','noah_ua','noah_mosaic_ua','noah_ua_sfcdiags','noah_seaice_options',
       'noah_fractional','irrigation','sfclay_dx2d')
def measure(item):
    name,exact=item
    os.environ['GPUWM_WRF_EXACT']=str(exact)
    import cupy as cp
    # Record the production request, including its direct-NVRTC arithmetic
    # options, when building compiler-bound images for a cold installation.
    from woof.kernel_aot import install
    install(cp)
    from woof.core.kernels import load_module,module_source,kernel_local_size_bytes
    from woof.certify.compile_platform import compile_platform_fingerprint
    module=load_module(name)
    source=module_source(name)
    symbols=set(re.findall(r'extern\s+"C"\s+__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(\w+)',source))
    # Preprocessor selection sits between the sea-ice export and its name.
    symbols.update(re.findall(r'void\s+(noah_seaice(?:_options)?_column)\s*\(',source))
    frames={}
    missing=[]
    for symbol in sorted(symbols):
        try: function=module.get_function(symbol)
        except cp.cuda.driver.CUDADriverError:
            missing.append(symbol)
            continue
        frames[symbol]=kernel_local_size_bytes(function)
    if not frames:raise RuntimeError((name,'no exported functions',symbols))
    from woof.kernel_aot import receipt
    from woof.nvrtc_ptx_cache import cache_events
    return dict(module=name,exact=exact,frames=frames,missing=missing,
                fingerprint=compile_platform_fingerprint(),
                aot=receipt(), direct_cache=cache_events())
if __name__=='__main__':
    # Numerics and WRF_EXACT are fixed at import. A worker must never switch
    # modes between tasks, or its catalogue describes a mixed-mode process.
    with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=1,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(measure,[(n,e) for n in UNITS for e in (0,1)]))
    Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
