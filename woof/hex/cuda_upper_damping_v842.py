"""Native upper damping with direct NVRTC and gradual underflow.

MPAS-A v8.4.2 mpas_atm_time_integration.F:6390-6408,6610-6629.
The arithmetic sources are shared with v841; only compilation differs.
"""
from woof.hex import numerics as _hex_numerics
from functools import lru_cache
import hashlib

CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")


class DirectNvrtcCache:
    """Compile saved dissipation sources without RawModule's forced FTZ."""
    def __init__(self):
        self.modules = {}
        self.kernels = {}
        self.inventory = []

    def raw_kernel(self, name, source, *, module_key, options=()):
        import cupy as cp
        from cupy.cuda import compiler
        if options:
            raise ValueError("the native dissipation compiler options are fixed")
        device = cp.cuda.Device()
        digest = hashlib.sha256(source.encode()).hexdigest()
        key = (device.id, module_key, digest)
        if key not in self.modules:
            binary, _ = compiler.compile_using_nvrtc(
                source, options=_hex_numerics.options(CUDA_OPTIONS), arch=__import__("woof.backend", fromlist=["compilation_target"]).compilation_target(xp=cp))
            module = cp.cuda.function.Module()
            module.load(binary)
            self.modules[key] = module
            self.inventory.append({"module_key": module_key,
                "source_sha256": digest, "binary_sha256": hashlib.sha256(binary).hexdigest(),
                "device_id": device.id, "architecture": __import__("woof.backend", fromlist=["compilation_target"]).compilation_target(xp=cp),
                "compiler": "direct HIP" if cp.cuda.runtime.is_hip else "direct NVRTC", "options": list(_hex_numerics.options(CUDA_OPTIONS))})
        function_key = (*key, name)
        if function_key not in self.kernels:
            self.kernels[function_key] = self.modules[key].get_function(name)
        return self.kernels[function_key]


@lru_cache(maxsize=None)
def _cache(device_id):
    return DirectNvrtcCache()


def direct_cache():
    import cupy as cp
    return _cache(cp.cuda.Device().id)


def apply_cam_floor_cuda(values, *, cache=None, **kwargs):
    """mpas_atm_time_integration.F:6390-6408, unchanged native floor."""
    from .cuda_upper_damping_v841 import apply_cam_floor_cuda as apply
    return apply(values, cache=direct_cache(), **kwargs)


def apply_rayleigh_u_cuda(tendency, rho, velocity, *, cache=None, **kwargs):
    """mpas_atm_time_integration.F:6610-6629, unchanged native tendency."""
    from .cuda_upper_damping_v841 import apply_rayleigh_u_cuda as apply
    return apply(tendency, rho, velocity, cache=direct_cache(), **kwargs)
