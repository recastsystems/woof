"""Load the production shallow kernels from shipped images with JIT refused."""
import json
import os
from pathlib import Path
import tempfile


with tempfile.TemporaryDirectory(prefix='shallow-aot-') as cache:
    os.environ['CUPY_CACHE_DIR']=cache
    import cupy as cp
    from cupy.cuda import compiler
    from woof.kernel_aot import install, receipt
    install(cp)
    def refuse(*args,**kwargs):
        raise AssertionError('a shallow kernel requested NVRTC instead of its shipped image')
    compiler._compile_using_nvrtc_no_warning=refuse
    from woof.core.kernels import load_module, kernel_local_size_bytes
    frames={}
    for name,symbol in [('grims','grims_column'),('shcu_nscv','shcu_nscv_column'),
                        ('deng','deng_column')]:
        function=load_module(name).get_function(symbol)
        frames[name]=kernel_local_size_bytes(function)
    events=receipt()
    assert len(events)==3 and all(row['source']=='aot' for row in events),events
    print(json.dumps(dict(frames=frames,events=events),sort_keys=True))
