"""Explicit qualification route that compiles the actual pre-port CUDA body."""
import os
from pathlib import Path
source_path=os.environ.get("WOOF_NOAH_BEFORE_SOURCE")
if source_path:
    from woof.core import kernels
    from woof.certify.kernel_manifest import record_module
    original=kernels.get_kernel
    modules={}
    def before_kernel(name,function):
        if name!="noah": return original(name,function)
        import cupy as cp
        device=int(cp.cuda.Device().id)
        if device not in modules:
            source=kernels._preamble()+Path(source_path).read_text(encoding="utf-8")
            options=kernels.module_options("noah")
            module=cp.RawModule(code=source,options=options)
            key=kernels.MODULE_KEY_ROOT+":noah"
            kernels._compile_observed(module,key)
            record_module(key,source=source,options=options,module=module)
            modules[device]=module
        return modules[device].get_function(function)
    kernels.get_kernel=before_kernel
