"""Direct NVRTC route with fast defaults and strict proof options."""
from __future__ import annotations
from types import SimpleNamespace
import re
from woof.core.device_cache import cuda_cache

def module_recipe_no_ftz(name, defines=()):
    """The identical source/options request for runtime and CPU-only AOT builds."""
    from woof.core.kernels import module_source
    from woof.wrf_exact import ENABLED, effective_options
    for key, value in defines:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None or type(value) is not int or value < 1:
            raise ValueError("kernel defines must be uppercase names and positive integers")
    source = "".join(f"#define {key} {value}\n" for key,value in defines) + module_source(name)
    if name in {"mskf", "mskf_aerosol_replay"}:
        from woof.core.kernels import module_options
        options = module_options(name)
        from woof.numerics import is_strict
        if (name == "mskf" and ("GPUWM_MSKF_NO_AEROSOL", 1) in defines
                and not is_strict()):
            # The contraction restriction protects aerosol number moments.
            # The aerosol-off specialization uses normal fast arithmetic.
            options = tuple(option for option in options if option not in
                            ("--fmad=false", "-DGPUWM_MSKF_OPERATION_ORDER=1"))
    else:
        options = ("-std=c++17", "--fmad=true", "--ftz=true")
    if ENABLED:
        options = effective_options(options)
    return source, options


@cuda_cache(maxsize=None)
def load_module_no_ftz(name, defines=()):
    import cupy as cp
    from woof.core.kernels import MODULE_KEY_ROOT
    from woof.nvrtc_ptx_cache import compile_using_nvrtc
    from woof.certify.kernel_manifest import record_module
    source, options = module_recipe_no_ftz(name, defines)
    image, _ = compile_using_nvrtc(source, options, None, name + ".cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image,str) else image)
    tier = ",".join(f"{key}={value}" for key,value in defines)
    key = f"{MODULE_KEY_ROOT}:{name}[{tier}]"
    # This is the image passed to Module.load, including cached images.
    record_module(key, source=source, options=options,
                  module=SimpleNamespace(**{"cubin" if isinstance(image,bytes) and image.startswith(b"\x7fELF") else "ptx":image}))
    return module

@cuda_cache(maxsize=None)
def get_kernel_no_ftz(name, symbol, defines=()):
    return load_module_no_ftz(name, defines).get_function(symbol)
