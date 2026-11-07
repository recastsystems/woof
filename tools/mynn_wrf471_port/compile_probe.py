import os,time
from pathlib import Path
from woof.core.kernels import module_source,compile_module_no_ftz
from source import cuda_source
work=Path(os.environ.get('MYNN471_WORK','.'))
src=cuda_source(work)
t=time.monotonic();module=compile_module_no_ftz('mynn471',src)
k=module.get_function('mynn471_step')
print('compiled_seconds',time.monotonic()-t,flush=True)
print('attributes',getattr(k,'attributes',None),flush=True)
