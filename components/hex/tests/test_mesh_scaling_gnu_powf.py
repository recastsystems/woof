"""meshScalingDel2/Del4 compile on GNU powf on every route.

Prevents: CUDA's powf in mixing_scaling_f32 differs from the GNU libm powf native MPAS uses
(mpas_atm_core.F atm_compute_mesh_scaling) in the last bit for some densities; on a variable-resolution
mesh that moved every mixed edge (storm box 2024-05-21 15Z against native 44c0849 strict: 986,078 history
words after one step, 0 with config_h_ScaleWithMesh false on both sides).
"""
from woof.hex import cuda_horizontal, cuda_horizontal_rrfs


class _Recorder:
    def __init__(self):
        self.calls = []

    def raw_kernel(self, name, source, module_key=None):
        self.calls.append((name, source, module_key))
        return object()


def _scaling_body(source):
    return source.split('void mixing_scaling_f32(', 1)[1].split('extern "C"', 1)[0]


def test_scaling_source_calls_gnu_powf_only():
    body = _scaling_body(cuda_horizontal._SCALING_GNU_SOURCE)
    assert body.count('mpas_native_powf_positive(mean_density,') == 2
    assert body.replace('mpas_native_powf_positive(', '').count('powf(') == 0
    assert '__device__ float mpas_native_powf_positive' in cuda_horizontal._SCALING_GNU_SOURCE


def test_base_route_compiles_scaling_from_gnu_source_and_others_unchanged():
    horizontal = object.__new__(cuda_horizontal.CudaHorizontal)
    horizontal._kernels = {}
    horizontal.kernel_cache = _Recorder()
    horizontal._kernel('mixing_scaling_f32')
    horizontal._kernel('divergence_damping_f32')
    (n1, s1, _), (n2, s2, _) = horizontal.kernel_cache.calls
    assert s1 is cuda_horizontal._SCALING_GNU_SOURCE
    assert s2 is cuda_horizontal._CUDA_SOURCE


def test_rrfs_route_uses_the_same_source():
    horizontal = object.__new__(cuda_horizontal_rrfs.CudaHorizontalRRFS)
    horizontal.kernel_cache = _Recorder()
    horizontal._kernel('mixing_scaling_f32')
    assert horizontal.kernel_cache.calls[0][1] is cuda_horizontal._SCALING_GNU_SOURCE


def test_host_scalings_use_gnu_powf_and_it_equals_this_hosts_libm():
    """Regional relaxation scaling and the host mixing scaling: the correctly rounded hook differed from GNU
    powf on 64 edges and 33 cells of the storm mesh, which moved the relaxation zone after one step."""
    import ctypes
    import inspect
    import numpy as np
    from woof.hex import mixing, regional_v841
    from woof.hex.native_powf import powf_native_gnu
    for function in (regional_v841.compute_mesh_scaling_regional, mixing.compute_mesh_mixing_scaling):
        text = inspect.getsource(function)
        assert "powf_native_gnu(" in text and "powf_rkind(" not in text
    try:
        libm = ctypes.CDLL("libm.so.6")
    except OSError:
        return
    powf = libm.powf
    powf.restype = ctypes.c_float
    powf.argtypes = [ctypes.c_float, ctypes.c_float]
    x = np.random.default_rng(7).uniform(1e-3, 1.0, 20000).astype(np.float32)
    for e in (0.25, 0.75):
        gnu = np.array([powf(float(v), e) for v in x], dtype=np.float32)
        assert np.array_equal(powf_native_gnu(x, e).view(np.uint32), gnu.view(np.uint32))


def test_fast_mode_keeps_the_gnu_scaling():
    """Prevents: fast mode compiles with --fmad=true; a plain double a*b+c in the GNU powf would contract into an
    FMA there and move meshScalingDel2/Del4 off native's on a variable-resolution mesh in the default run. Every
    double operation in the device powf is an explicitly rounded intrinsic (never contracted), and no mode
    defines USE_FMA, so fast and strict compute the same scaling."""
    import re
    from woof.hex import numerics
    from woof.hex.native_powf import CUDA_NATIVE_POWF
    body = CUDA_NATIVE_POWF.split("__device__ float mpas_native_powf_positive", 1)[1].split(chr(10) + "}", 1)[0]
    assert "#define MUL(a,b) __dmul_rn((a),(b))" in CUDA_NATIVE_POWF
    assert "#define ADD(a,b) __dadd_rn((a),(b))" in CUDA_NATIVE_POWF
    assert "#define SUB(a,b) __dsub_rn((a),(b))" in CUDA_NATIVE_POWF
    # Integer index arithmetic (unsigned/int lines) and the non-normal fallback (if) are not double operations.
    arithmetic = [line for line in body.splitlines()[1:] if not line.strip().startswith(("unsigned", "int ", "if("))]
    assert len(arithmetic) >= 10
    for line in arithmetic:
        code = re.sub(r"0x[0-9a-fA-Fp.+-]+", "C", line)
        assert not re.search(r"[A-Za-z0-9_)\]]\s*[*+]\s*[A-Za-z0-9_(]", code.split("//")[0].replace("(double)", "")), line
    for mode in ("fast", "strict"):
        assert not any("USE_FMA" in option for option in numerics.kernel_base(mode))
    assert not any("USE_FMA" in option for option in numerics.FAST_FLAGS + numerics.FAST_DEFINES)
