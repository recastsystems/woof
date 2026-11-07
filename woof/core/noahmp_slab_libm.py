"""The FP32 primitives the vectorised Noah-MP composition is allowed to use.

Noah-MP's column composition -- ENERGY's own arithmetic and NOAHMP_SFLX's
prefix and postfix -- is transcribed in :mod:`woof.core.noahmp_energy` and
:mod:`woof.core.noahmp_sflx` as scalar CPython: Python floats with an explicit
``f32`` rounding call after every operation.  That transcription is what the
unmodified-WRF fixtures pin, and it is also, measured, essentially the entire
cost of a Noah-MP land-surface call.

Evaluating it for every land column at once needs the same arithmetic over
arrays.  This module states, in one place, which array operation reproduces
which scalar one **bitwise**, and refuses to guess:

``+ - * /`` and ``sqrt``
    The ordinary CuPy float32 route agrees over the normal arguments tested
    by ``test_ieee_agreement`` and ``test_sqrt_agreement``. Its default FTZ
    route can flush subnormal operands or results. Arithmetic crossing that
    range uses the generated CUDA kernels with FTZ disabled. In particular,
    ``slab_add_product`` retains the ground-humidity blend between leaf batches.

``min`` / ``max``
    **Not** ``cupy.minimum``/``cupy.maximum``.  ENERGY spells them
    ``a if a < b else b`` (:func:`woof.core.noahmp_energy._mn`), which returns
    the *second* argument on a tie; ``cupy.minimum(-0.0, +0.0)`` returns
    ``-0.0`` where ``_mn(-0.0, +0.0)`` returns ``+0.0``.  gfortran does not
    flush, so signed zeros are exactly the input class this project has lost
    bugs to.  :func:`fmn` and :func:`fmx` are the ``where`` forms that match.

``powf`` / ``expf`` / ``logf`` / ``tanhf``
    These use WOOF's own float32 routines, tested against the WRF oracle.
    They go through
    ``woof/core/kernels/noahmp_libm_slab.cu``, which is a bare elementwise
    wrapper around WOOF's own float32 routines already in this
    tree (``r_pow``/``r_exp``/``r_log`` in ``noahmp_leaves.cu`` and
    ``nmpe_tanhf`` in ``noahmp_energy.cu``).  No second copy is created here,
    which is also why the entry points are spelled ``slab_powf`` and not
    ``powf``: ``tests/test_noahmp_radiation.py`` treats a module under
    ``gpuwm/`` that *defines* ``powf`` as a fork of the transcription, and it
    is right to.

``tests/test_noahmp_slab_libm.py`` holds every claim above against
:mod:`woof.core.noahmp_libm`, with four negative controls: ``cupy.minimum``,
``numpy.power`` and ``cupy.exp`` are each shown failing a gate the wrapper
passes, and ``__double2float_rn`` is shown still flushing the subnormals that
``nmp_d2f_rn`` exists to recover.
"""

from __future__ import annotations

import numpy as np

from woof.core.noahmp_kernel_sources import (compile_runtime_unit,
                                              compile_generated_slab_kernel)

#: One CUDA thread per column; these kernels are memory bound and elementwise.
THREADS = 128

_ADD_PRODUCT_KERNELS = {}

# WRF-lineage Noah-MP 848f54ad ENERGY :2195-2197, opt_rsf=4.
# Original notice: licenses/LICENSE-WRF-public-domain.txt. No Noah-MP 5 source.
RSURF_SNOW_SOURCE = r'''extern "C" __global__ void noahmp_rsurf_snow(
const float* soil,const float* fsno,const float* snow,float* out,int n){
int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
float r=soil[i]>0.001f?soil[i]:0.001f;
float a=__fmul_rn(fsno[i],__fdiv_rn(1.0f,snow[i]));
float b=__fmul_rn(__fsub_rn(1.0f,fsno[i]),__fdiv_rn(1.0f,r));
out[i]=__fdiv_rn(1.0f,__fadd_rn(a,b));
}'''
_RSURF_SNOW_KERNELS = {}


def snow_surface_resistance(soil, fsno, snow):
    """Native RSF=4 harmonic blend, before the dry/urban resistance caps.

    Inputs are actual f32 soil resistance, snow fraction and transferred
    RSURF_SNOW slabs. One thread per element, no host field arithmetic.
    """
    import cupy as cp

    arrays = cp.broadcast_arrays(cp.asarray(soil, dtype=cp.float32),
                                 cp.asarray(fsno, dtype=cp.float32),
                                 cp.asarray(snow, dtype=cp.float32))
    values = tuple(cp.ascontiguousarray(a) for a in arrays)
    out = cp.empty(arrays[0].shape, dtype=cp.float32)
    n = int(out.size)
    if n:
        key = cp.cuda.runtime.getDevice()
        kernel = _RSURF_SNOW_KERNELS.get(key)
        if kernel is None:
            kernel = compile_generated_slab_kernel(RSURF_SNOW_SOURCE,
                                                   "noahmp_rsurf_snow")
            _RSURF_SNOW_KERNELS[key] = kernel
        kernel((_blocks(n),), (THREADS,), (*values, out, np.int32(n)))
    return out


def slab_add_product(a, b, c):
    """Two binary32 roundings for a + b*c, retaining subnormal words."""
    import cupy as cp
    arrays = cp.broadcast_arrays(cp.asarray(a, dtype=cp.float32),
                                 cp.asarray(b, dtype=cp.float32),
                                 cp.asarray(c, dtype=cp.float32))
    shape = arrays[0].shape
    values = [cp.ascontiguousarray(v) for v in arrays]
    out = cp.empty(shape, dtype=cp.float32)
    n = int(out.size)
    if n:
        key = cp.cuda.runtime.getDevice()
        kernel = _ADD_PRODUCT_KERNELS.get(key)
        if kernel is None:
            source = '''extern "C" __global__ void noahmp_add_product(
                const float* a,const float* b,const float* c,float* o,int n) {
                int i=blockDim.x*blockIdx.x+threadIdx.x;
                if(i<n)o[i]=__fadd_rn(a[i],__fmul_rn(b[i],c[i]));
            }'''
            kernel = compile_generated_slab_kernel(source, 'noahmp_add_product')
            _ADD_PRODUCT_KERNELS[key] = kernel
        kernel(((n+THREADS-1)//THREADS,), (THREADS,), (*values, out, np.int32(n)))
    return out

_MODULE_NAME = "noahmp_libm_slab"


def _blocks(n: int) -> int:
    return (n + THREADS - 1) // THREADS


def _module():
    """Compile ``noahmp_libm_slab`` as the three-part unit it is, per card.

    ``compile_runtime_unit`` loads the image with ``cupy.cuda.function.Module``
    into the CURRENT device's context only (it is not a device-aware
    ``RawModule``).  One module held per process handed card 1 card 0's
    function handles: CUDA_ERROR_INVALID_HANDLE at the first ``slab_expf``
    of RRTMG's cloud fraction on the two-card HRRR start (2026-10-16).
    """
    import cupy as cp

    device = cp.cuda.runtime.getDevice()
    module = _MODULE_CACHE.get(device)
    if module is None:
        module = compile_runtime_unit(
            _MODULE_NAME,
            module_key=f"woof.core.noahmp_slab_libm:{_MODULE_NAME}")
        _MODULE_CACHE[device] = module
    return module


#: Keyed by device: the loaded module lives in one card's context.
_MODULE_CACHE: dict[int, object] = {}
#: Keyed by ``(device, name)``: a ``Function`` handle is bound to the module
#: of the card it was taken from, and calling it on another card fails.
_KERNEL_CACHE: dict[tuple[int, str], object] = {}


def _kernel(name: str):
    import cupy as cp

    key = (cp.cuda.runtime.getDevice(), name)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _module().get_function(name)
    return _KERNEL_CACHE[key]


def _contiguous(array):
    """A raw kernel argument carries a pointer, not a stride.

    Handing a strided view to a raw kernel is the defect that fed column 0's
    neighbouring slots to PRECIP_HEAT on every column; it is silent, it
    survives every per-leaf oracle, and it is one call away every time an
    array is sliced.  Every argument below goes through here.
    """
    import cupy as cp

    return cp.ascontiguousarray(cp.asarray(array, dtype=cp.float32))


def _unary(name, x):
    import cupy as cp

    x = _contiguous(x)
    out = cp.empty_like(x)
    n = int(x.size)
    if n:
        _kernel(name)((_blocks(n),), (THREADS,), (x, out, np.int32(n)))
    return out


def slab_powf(x, y):
    """WOOF's float32 power over arrays, elementwise.

    ``y`` may be a scalar; it is broadcast to ``x``'s shape *before* the
    launch, because the kernel reads one exponent per element.
    """
    import cupy as cp

    x = _contiguous(x)
    if isinstance(y, cp.ndarray) or np.ndim(y) != 0:
        y = _contiguous(cp.broadcast_to(cp.asarray(y, dtype=cp.float32), x.shape))
    else:
        # A host scalar fills on the card: cp.asarray of it was one pinned
        # staged upload (and one CUDA event) per call, 62 per legacy
        # radiation call (rrtmg_mcica._window_tables names the breakage).
        y = cp.full(x.shape, np.float32(y), dtype=cp.float32)
    out = cp.empty_like(x)
    n = int(x.size)
    if n:
        _kernel("nmp_slab_powf")((_blocks(n),), (THREADS,),
                                 (x, y, out, np.int32(n)))
    return out


def slab_expf(x):
    """WOOF's float32 exponential over arrays, elementwise."""
    return _unary("nmp_slab_expf", x)


def slab_logf(x):
    """WOOF's float32 logarithm over arrays, elementwise."""
    return _unary("nmp_slab_logf", x)


_SELECTED_TANH_KERNELS = {}


def slab_tanhf(x, *, profile="glibc_2_39"):
    """Explicit native tanhf generation; existing oracle remains the default.

    The newer native proof uses the existing MIT CORE-MATH implementation.
    Selection never follows the execution host or the field values.
    """
    if profile == "glibc_2_39":
        return _unary("nmp_slab_tanhf", x)
    if profile != "glibc_2_43":
        raise ValueError("unknown Noah-MP tanh profile")
    import cupy as cp
    from woof.core.noahmp_kernel_sources import KERNEL_DIR
    x = _contiguous(x)
    out = cp.empty_like(x)
    n = int(x.size)
    if n:
        key = cp.cuda.runtime.getDevice()
        kernel = _SELECTED_TANH_KERNELS.get(key)
        if kernel is None:
            source = (KERNEL_DIR / "sfire_libm.cuh").read_text(encoding="ascii")
            source += r'''
extern "C" __global__ void noahmp_tanhf_glibc243(
const float* x, float* out, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<n) out[i]=sfire_tanhf(x[i]);
}
'''
            kernel = compile_generated_slab_kernel(source, "noahmp_tanhf_glibc243")
            _SELECTED_TANH_KERNELS[key] = kernel
        kernel((_blocks(n),), (THREADS,), (x, out, np.int32(n)))
    return out


def slab_sqrtf(x):
    """IEEE-754 square root with binary32 rounding."""
    import cupy as cp

    return cp.sqrt(cp.asarray(x, dtype=cp.float32))


def fmn(a, b):
    """``woof.core.noahmp_energy._mn`` over arrays.

    ``a if a < b else b``.  On a tie -- including ``(-0.0, +0.0)`` -- this
    returns ``b``, which ``cupy.minimum`` does not.
    """
    import cupy as cp

    a = cp.asarray(a, dtype=cp.float32)
    b = cp.asarray(b, dtype=cp.float32)
    return cp.where(a < b, a, b)


def fmx(a, b):
    """``woof.core.noahmp_energy._mx`` over arrays: ``a if a > b else b``."""
    import cupy as cp

    a = cp.asarray(a, dtype=cp.float32)
    b = cp.asarray(b, dtype=cp.float32)
    return cp.where(a > b, a, b)


def slab_fabsf(x):
    """Fortran ``ABS`` on ``REAL(4)``: a sign-bit clear, ``-0.0 -> +0.0``."""
    import cupy as cp

    return cp.abs(cp.asarray(x, dtype=cp.float32))


__all__ = [
    "fmn",
    "fmx",
    "slab_expf",
    "slab_fabsf",
    "slab_logf",
    "slab_powf",
    "slab_sqrtf",
    "slab_tanhf",
]


_COPY_SLOT_KERNELS = {}


def copy_slab_slots(out, blocks):
    """Copy validated column slabs into row slots with one integer-only launch.

    Strides remain explicit so broadcast rows and sliced columns do not need
    temporary copies. Integer loads and stores preserve every floating bit.
    """
    import cupy as cp

    n, stride = out.shape
    arrays, layout = [], []
    for start, value in blocks:
        value = cp.asarray(value, dtype=out.dtype)
        if value.ndim == 0:
            value = cp.broadcast_to(value, (n,))
        width = 1 if value.ndim == 1 else value.shape[1]
        row = value.strides[0] // 4
        col = 0 if value.ndim == 1 else value.strides[1] // 4
        arrays.append(value)
        layout.append((int(start), int(width), int(row), int(col)))
    if not n or not arrays:
        return out
    key = (cp.cuda.runtime.getDevice(), int(stride), tuple(layout))
    kernel = _COPY_SLOT_KERNELS.get(key)
    if kernel is None:
        args = ", ".join(f"const unsigned int* v{k}" for k in range(len(arrays)))
        lines = []
        for k, (start, width, row, col) in enumerate(layout):
            lines.extend(f"out[(long long)c*{stride}+{start+j}] = "
                         f"v{k}[(long long)c*({row})+({j*col})];"
                         for j in range(width))
        source = (f'extern "C" __global__ void noahmp_copy_slots('
                  f'unsigned int* out, int n, {args}) {{ '
                  'int c = blockDim.x*blockIdx.x+threadIdx.x; if(c>=n) return; '
                  + "\n".join(lines) + '}')
        kernel = compile_generated_slab_kernel(source, "noahmp_copy_slots")
        _COPY_SLOT_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,), (out, np.int32(n), *arrays))
    return out


_SPLIT_SLOT_KERNELS = {}


def split_slab_slots(matrix, specs):
    """Copy row slots to owned contiguous slabs with one integer-only launch."""
    import cupy as cp

    n, stride = matrix.shape
    result = {name: cp.empty((n,) if width == 1 else (n, width),
                             dtype=matrix.dtype)
              for name, (start, width) in specs.items()}
    if not n or not result:
        return result
    layout = tuple(specs.values())
    key = (cp.cuda.runtime.getDevice(), int(stride), layout)
    kernel = _SPLIT_SLOT_KERNELS.get(key)
    if kernel is None:
        args = ", ".join(f"unsigned int* v{k}" for k in range(len(result)))
        lines = []
        for k, (start, width) in enumerate(layout):
            lines.extend(f"v{k}[(long long)c*{width}+{j}] = "
                         f"src[(long long)c*{stride}+{start+j}];"
                         for j in range(width))
        source = (f'extern "C" __global__ void noahmp_split_slots('
                  f'const unsigned int* src, int n, {args}) {{ '
                  'int c = blockDim.x*blockIdx.x+threadIdx.x; if(c>=n) return; '
                  + "\n".join(lines) + '}')
        kernel = compile_generated_slab_kernel(source, "noahmp_split_slots")
        _SPLIT_SLOT_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,), (matrix, np.int32(n), *result.values()))
    return result


_SCATTER_SLOT_KERNELS = {}


def scatter_slab_fields(j, i, blocks, *, active=None):
    """Store independent land-column outputs together without floating arithmetic."""
    import cupy as cp

    n = int(j.size)
    if not n:
        return
    j = cp.ascontiguousarray(j, dtype=cp.int64)
    i = cp.ascontiguousarray(i, dtype=cp.int64)
    if active is not None and any(target.dtype.itemsize != 4 for target, _ in blocks):
        raise TypeError("masked Noah-MP scatter requires 32-bit carriers")
    arrays, layout = [], []
    for target, value in blocks:
        value = cp.asarray(value, dtype=target.dtype)
        if target.dtype.itemsize != 4 and active is None:
            # Wider unmasked carriers retain the typed assignment conversion.
            for destination, source in blocks:
                if destination.ndim == 2:
                    destination[j, i] = source
                else:
                    destination[:, j, i] = source.T
            return
        width = 1 if value.ndim == 1 else value.shape[1]
        row = value.strides[0] // 4
        col = 0 if value.ndim == 1 else value.strides[1] // 4
        plane = 0 if target.ndim == 2 else target.strides[0] // 4
        ys, xs = target.strides[-2] // 4, target.strides[-1] // 4
        # The grid strides (plane, ys, xs) are kernel arguments: baked into the
        # source they made a new module for every mesh and partition size, which
        # no prebuilt kernel pack can cover (x1.163842 split in two, 2026-10-16).
        layout.append((width, row, col))
        arrays.extend((target, value, np.int64(plane), np.int64(ys), np.int64(xs)))
    key = (cp.cuda.runtime.getDevice(), tuple(layout), active is not None)
    kernel = _SCATTER_SLOT_KERNELS.get(key)
    if kernel is None:
        args = ", ".join(f"unsigned int* o{k}, const unsigned int* v{k}, "
                         f"long long p{k}, long long y{k}, long long x{k}"
                         for k in range(len(layout)))
        lines = []
        for k, (width, row, col) in enumerate(layout):
            lines.extend(f"o{k}[j[c]*y{k}+i[c]*x{k}+{layer}LL*p{k}] = "
                         f"v{k}[(long long)c*({row})+({layer*col})];"
                         for layer in range(width))
        source = (f'extern "C" __global__ void noahmp_scatter_fields('
                  f'const long long* j, const long long* i, int n, {args}) {{ '
                  'int c = blockDim.x*blockIdx.x+threadIdx.x; if(c>=n) return; '
                  + ("if(!active[c])return;" if active is not None else "")
                  + "\n".join(lines) + '}')
        if active is not None:
            source = source.replace("int n,", "int n, const bool* active,", 1)
        kernel = compile_generated_slab_kernel(source, "noahmp_scatter_fields")
        _SCATTER_SLOT_KERNELS[key] = kernel
    mask_args = () if active is None else (active,)
    kernel((_blocks(n),), (THREADS,), (j, i, np.int32(n), *mask_args, *arrays))


_GATHER_SLOT_KERNELS = {}


def gather_slab_fields(j, i, sources, grid_shape):
    """Gather known-valid grid coordinates together with integer-only transfers."""
    import cupy as cp

    n = int(j.size)
    if any(value.dtype.itemsize != 4 or value.shape[-2:] != grid_shape
           for value in sources.values()):
        return {name: value[j, i] if value.ndim == 2 else value[:, j, i].T
                for name, value in sources.items()}
    result = {name: cp.empty((n,) if value.ndim == 2 else (n, value.shape[0]),
                             dtype=value.dtype)
              for name, value in sources.items()}
    if not n or not result:
        return result
    j = cp.ascontiguousarray(j, dtype=cp.int64)
    i = cp.ascontiguousarray(i, dtype=cp.int64)
    # Grid strides are kernel arguments (see scatter_slab_fields); only the
    # layer counts shape the source.
    strides = tuple((0 if value.ndim == 2 else value.strides[0] // 4,
                     value.strides[-2] // 4, value.strides[-1] // 4)
                    for value in sources.values())
    layout = tuple(1 if value.ndim == 2 else value.shape[0] for value in sources.values())
    key = (cp.cuda.runtime.getDevice(), layout)
    kernel = _GATHER_SLOT_KERNELS.get(key)
    if kernel is None:
        args = ", ".join(f"const unsigned int* v{k}, unsigned int* o{k}, "
                         f"long long p{k}, long long y{k}, long long x{k}"
                         for k in range(len(layout)))
        lines = []
        for k, width in enumerate(layout):
            lines.extend(f"o{k}[(long long)c*{width}+{layer}] = "
                         f"v{k}[j[c]*y{k}+i[c]*x{k}+{layer}LL*p{k}];"
                         for layer in range(width))
        source = (f'extern "C" __global__ void noahmp_gather_fields('
                  f'const long long* j, const long long* i, int n, {args}) {{ '
                  'int c = blockDim.x*blockIdx.x+threadIdx.x; if(c>=n) return; '
                  + "\n".join(lines) + '}')
        kernel = compile_generated_slab_kernel(source, "noahmp_gather_fields")
        _GATHER_SLOT_KERNELS[key] = kernel
    arrays = tuple(item for value, out, (plane, ys, xs)
                   in zip(sources.values(), result.values(), strides)
                   for item in (value, out, np.int64(plane), np.int64(ys), np.int64(xs)))
    kernel((_blocks(n),), (THREADS,), (j, i, np.int32(n), *arrays))
    return result


_TAKE_ROW_KERNELS = {}


def take_slab_rows(index, sources, n):
    """Compact a proven in-range row selection with integer-only transfers."""
    import cupy as cp

    count = int(index.size)
    if any(value.dtype.itemsize != 4 or value.shape[0] != n
           for value in sources.values()):
        return {name: value[index] for name, value in sources.items()}
    result = {name: cp.empty((count,) if value.ndim == 1
                             else (count, value.shape[1]), dtype=value.dtype)
              for name, value in sources.items()}
    if not count or not result:
        return result
    index = cp.ascontiguousarray(index, dtype=cp.int64)
    layout = tuple((1 if value.ndim == 1 else value.shape[1],
                    value.strides[0] // 4,
                    0 if value.ndim == 1 else value.strides[1] // 4)
                   for value in sources.values())
    key = (cp.cuda.runtime.getDevice(), layout)
    kernel = _TAKE_ROW_KERNELS.get(key)
    if kernel is None:
        args = ", ".join(f"const unsigned int* v{k}, unsigned int* o{k}"
                         for k in range(len(layout)))
        lines = []
        for k, (width, row, col) in enumerate(layout):
            lines.extend(f"o{k}[(long long)c*{width}+{layer}] = "
                         f"v{k}[index[c]*({row})+({layer*col})];"
                         for layer in range(width))
        source = (f'extern "C" __global__ void noahmp_take_rows('
                  f'const long long* index, int n, {args}) {{ '
                  'int c = blockDim.x*blockIdx.x+threadIdx.x; if(c>=n) return; '
                  + "\n".join(lines) + '}')
        kernel = compile_generated_slab_kernel(source, "noahmp_take_rows")
        _TAKE_ROW_KERNELS[key] = kernel
    arrays = tuple(array for pair in zip(sources.values(), result.values())
                   for array in pair)
    kernel((_blocks(count),), (THREADS,), (index, np.int32(count), *arrays))
    return result


_WRITE_ARITHMETIC_KERNELS = {}


def write_slab_arithmetic(fields, j, i, values, *, nsoil, dt, active=None):
    """Preserve the separate CuPy roundings while reducing arithmetic launches."""
    import cupy as cp

    outputs = ("qfx", "lh", "smstav", "smstot", "sfcrunoff", "udrunoff",
               "albedo", "canwat", "acsnow", "acsnom", "pondingxy",
               "q2mvxy", "q2mbxy", "rs", "soilenergy", "snowenergy")
    inputs = ("ecan", "edir", "etran", "fcev", "fgev", "fctr", "runsrf",
              "runsub", "albedo", "canliq", "canice", "fpice", "qmelt",
              "ponding", "ponding1", "ponding2", "q2v", "q2b", "laisun",
              "laisha", "rb", "rssun", "rssha", "stc", "zsnso", "hcpct")
    if (any(fields[name].dtype != cp.float32 or not fields[name].flags.c_contiguous
            for name in (*outputs, "rainbl"))
            or any(values[name].dtype != cp.float32 for name in inputs)):
        return False
    n = int(j.size)
    if not n:
        return True
    arrays = tuple(cp.ascontiguousarray(values[name]) for name in inputs)
    j = cp.ascontiguousarray(j, dtype=cp.int64)
    i = cp.ascontiguousarray(i, dtype=cp.int64)
    isnow = cp.ascontiguousarray(values["isnow"], dtype=cp.int32)
    nx = fields["qfx"].shape[1]
    key = (cp.cuda.runtime.getDevice(), nsoil, active is not None)
    kernel = _WRITE_ARITHMETIC_KERNELS.get(key)
    if kernel is None:
        args = (", ".join(f"float* o_{name}" for name in outputs) + ", "
                + ", ".join(f"const float* v_{name}" for name in inputs))
        source = r'''__device__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ float D(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void noahmp_write_arithmetic(
const long long* j,const long long* i,const int* isnow,
const float* rainbl,float dt,int n,long long nx,ARGS){
int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=n)return;
long long p=j[c]*nx+i[c];
o_qfx[p]=A(A(v_ecan[c],v_edir[c]),v_etran[c]);
o_lh[p]=A(A(v_fcev[c],v_fgev[c]),v_fctr[c]);
o_smstav[p]=0.0f;o_smstot[p]=0.0f;
o_sfcrunoff[p]=A(o_sfcrunoff[p],v_runsrf[c]);
o_udrunoff[p]=A(o_udrunoff[p],v_runsub[c]);
if(v_albedo[c]>-999.0f)o_albedo[p]=v_albedo[c];
o_canwat[p]=A(v_canliq[c],v_canice[c]);
o_acsnow[p]=A(o_acsnow[p],M(rainbl[p],v_fpice[c]));
float pond=A(A(v_ponding[c],v_ponding1[c]),v_ponding2[c]);
o_acsnom[p]=A(A(A(A(o_acsnom[p],M(v_qmelt[c],dt)),v_ponding[c]),v_ponding1[c]),v_ponding2[c]);
o_pondingxy[p]=pond;
o_q2mvxy[p]=D(v_q2v[c],S(1.0f,v_q2v[c]));
o_q2mbxy[p]=D(v_q2b[c],S(1.0f,v_q2b[c]));
float sun=0.0f>v_laisun[c]?0.0f:v_laisun[c];
float shade=0.0f>v_laisha[c]?0.0f:v_laisha[c];
float rb=0.0f>v_rb[c]?0.0f:v_rb[c];
bool closed=v_rssun[c]<=0.0f||v_rssha[c]<=0.0f||sun==0.0f||shade==0.0f;
float inv=A(M(D(1.0f,A(v_rssun[c],rb)),sun),M(D(1.0f,A(v_rssha[c],rb)),shade));
o_rs[p]=closed?0.0f:D(1.0f,inv);
float soil=0.0f,snow=0.0f;
for(int k=-2;k<=NSOIL;k++){
 int slot=k+2;float z=v_zsnso[c*NLAY+slot];
 float above=slot>0?v_zsnso[c*NLAY+slot-1]:0.0f;
 float thick=k==isnow[c]+1?-z:S(above,z);
 float term=M(M(M(thick,v_hcpct[c*NLAY+slot]),S(v_stc[c*NLAY+slot],273.16f)),0.001f);
 float contribution=k>=isnow[c]+1?term:0.0f;
 if(k>=1)soil=A(soil,contribution);else snow=A(snow,contribution);
}
o_soilenergy[p]=soil;o_snowenergy[p]=snow;
}'''
        source = (source.replace("ARGS", args).replace("NSOIL", str(nsoil))
                  .replace("NLAY", str(nsoil + 3)))
        if active is not None:
            source = source.replace("int n,long long nx,", "int n,long long nx,const bool* active,")
            source = source.replace("if(c>=n)return;", "if(c>=n || !active[c])return;", 1)
        kernel = compile_generated_slab_kernel(source, "noahmp_write_arithmetic")
        _WRITE_ARITHMETIC_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,),
           (j, i, isnow, fields["rainbl"], np.float32(dt), np.int32(n), np.int64(nx),
            *((active,) if active is not None else ()),
            *(fields[name] for name in outputs), *arrays))
    return True


_ENERGY_AVERAGE_KERNELS = {}


def energy_average_slabs(s, g, e, v, b, sb):
    """Keep each tile-average operation rounded as in the separate CuPy calls."""
    import cupy as cp

    spec = {
        "s": ("fveg", "pahg", "pahb", "pahv", "sfcprs", "lwdn", "parsun",
              "laisun", "parsha", "laisha", "acc_ssoil", "dt"),
        "g": ("z0m", "z0mg"), "e": ("emv", "emg"),
        "v": ("tauxv", "tauyv", "irg", "irc", "shg", "shc", "evg", "ghv",
              "evc", "tr", "tgv", "t2mv", "tv", "cmv", "chv", "eah",
              "q2v", "rssun", "rssha", "psnsun", "psnsha"),
        "b": ("tauxb", "tauyb", "irb", "shb", "evb", "ghb", "tgb", "t2mb",
              "cmb", "chb", "qsfc", "q2b"),
    }
    tables = {"s": s, "g": g, "e": e, "v": v, "b": b}
    inputs = [(f"{prefix}_{name}", tables[prefix][name])
              for prefix, names in spec.items() for name in names]
    names = ("taux", "tauy", "fira", "fsh", "fgev", "ssoil", "fcev", "fctr",
             "pah", "tg", "t2m", "ts", "cm", "ch", "q1", "q2e", "z0wrf",
             "rssun", "rssha", "tgv", "chv", "emissi", "apar", "psn",
             "acc_ssoil", "ssoil_avg", "dt_soil", "_fire", "_trad_base")
    n = int(s["fveg"].size)
    output = {name: cp.empty(n, dtype=cp.float32) for name in names}
    if not n:
        return output
    arrays = [cp.ascontiguousarray(value, dtype=cp.float32) for _, value in inputs]
    key = cp.cuda.runtime.getDevice()
    kernel = _ENERGY_AVERAGE_KERNELS.get(key)
    if kernel is None:
        args = (", ".join(f"const float* {name}" for name, _ in inputs) + ", "
                + ", ".join(f"float* o_{name}" for name in names))
        code = r'''__device__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ float D(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void noahmp_energy_average(const bool* tile,int n,float sb,float steps,ARGS){
int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=n)return;
bool t=tile[c];float f=s_fveg[c],one=S(1.0f,f);
BODY
float fire=A(s_lwdn[c],o_fira[c]);o__fire[c]=fire;
float om=S(1.0f,e_emv[c]);
float emiss=A(M(f,A(A(M(e_emg[c],om),e_emv[c]),M(M(e_emv[c],om),S(1.0f,e_emg[c])))),M(one,e_emg[c]));
o_emissi[c]=emiss;
o__trad_base[c]=D(S(fire,M(S(1.0f,emiss),s_lwdn[c])),M(emiss,sb));
o_apar[c]=A(M(s_parsun[c],s_laisun[c]),M(s_parsha[c],s_laisha[c]));
o_psn[c]=A(M(v_psnsun[c],s_laisun[c]),M(v_psnsha[c],s_laisha[c]));
float accum=A(s_acc_ssoil[c],o_ssoil[c]);o_acc_ssoil[c]=accum;
o_ssoil_avg[c]=D(accum,steps);o_dt_soil[c]=M(s_dt[c],steps);
}'''
        pairs = {"taux": ("v_tauxv", "b_tauxb"), "tauy": ("v_tauyv", "b_tauyb"),
                 "fgev": ("v_evg", "b_evb"), "ssoil": ("v_ghv", "b_ghb"),
                 "tg": ("v_tgv", "b_tgb"), "t2m": ("v_t2mv", "b_t2mb"),
                 "cm": ("v_cmv", "b_cmb"), "ch": ("v_chv", "b_chb"),
                 "q2e": ("v_q2v", "b_q2b")}
        body = [f"o_{name}[c]=t?A(M(f,{veg}[c]),M(one,{bare}[c])):{bare}[c];"
                for name, (veg, bare) in pairs.items()]
        body += [
            "o_fira[c]=t?A(A(M(f,v_irg[c]),M(one,b_irb[c])),v_irc[c]):b_irb[c];",
            "o_fsh[c]=t?A(A(M(f,v_shg[c]),M(one,b_shb[c])),v_shc[c]):b_shb[c];",
            "o_fcev[c]=t?v_evc[c]:0.0f;o_fctr[c]=t?v_tr[c]:0.0f;",
            "o_pah[c]=t?A(A(M(f,s_pahg[c]),M(one,s_pahb[c])),s_pahv[c]):s_pahb[c];",
            "o_ts[c]=t?A(M(f,v_tv[c]),M(one,b_tgb[c])):o_tg[c];",
            "o_q1[c]=t?A(M(f,D(M(v_eah[c],0.622f),S(s_sfcprs[c],M(0.378f,v_eah[c])))),M(one,b_qsfc[c])):b_qsfc[c];",
            "o_z0wrf[c]=t?g_z0m[c]:g_z0mg[c];",
            "o_rssun[c]=t?v_rssun[c]:0.0f;o_rssha[c]=t?v_rssha[c]:0.0f;",
            "o_tgv[c]=t?v_tgv[c]:b_tgb[c];o_chv[c]=t?v_chv[c]:b_chb[c];",
        ]
        code = code.replace("ARGS", args).replace("BODY", "\n".join(body))
        kernel = compile_generated_slab_kernel(code, "noahmp_energy_average")
        _ENERGY_AVERAGE_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,),
           (cp.ascontiguousarray(e["tile_veg"]), np.int32(n), np.float32(sb),
            np.float32(s["soil_update_steps"]), *arrays, *output.values()))
    return output


_ROOT_FRACTION_KERNELS = {}


def root_fraction_slabs(s, nsnow, nsoil, mpe, *, opt_btr=1):
    """Accumulate root fractions in layer order with explicit FP32 rounding."""
    if type(opt_btr) is not int or opt_btr not in (1, 2, 3):
        raise ValueError("ENERGY opt_btr must be scalar int in (1, 2, 3)")
    if opt_btr != 1:
        return _selected_root_fraction_slabs(s, nsnow, nsoil, mpe, opt_btr)
    import cupy as cp

    inputs = tuple(cp.ascontiguousarray(s[name], dtype=cp.float32)
                   for name in ("sh2o", "smcwlt", "smcref", "zsoil", "dzsnso"))
    roots = cp.ascontiguousarray(s["nroot"], dtype=cp.int32)
    n = int(roots.size)
    total = cp.empty(n, dtype=cp.float32)
    fractions = cp.empty((n, nsoil), dtype=cp.float32)
    if not n:
        return total, fractions
    key = (cp.cuda.runtime.getDevice(), nsnow, nsoil)
    kernel = _ROOT_FRACTION_KERNELS.get(key)
    if kernel is None:
        source = r'''extern "C" __global__ void noahmp_root_fractions(
const float* water,const float* wilt,const float* ref,const float* zsoil,
const float* dz,const int* roots,int n,float mpe,float* total,float* fractions){
int c=blockDim.x*blockIdx.x+threadIdx.x;if(c>=n)return;
int root=roots[c],index=(root-1)%NSOIL;if(index<0)index+=NSOIL;
float depth=-zsoil[c*NSOIL+index],sum=0.0f;
for(int layer=0;layer<NSOIL;layer++){
 int slot=c*NSOIL+layer;
 float gx=__fdiv_rn(__fsub_rn(water[slot],wilt[slot]),__fsub_rn(ref[slot],wilt[slot]));
 gx=0.0f>gx?0.0f:gx;gx=1.0f<gx?1.0f:gx;
 float value=__fmul_rn(__fdiv_rn(dz[c*NLAY+NSNOW+layer],depth),gx);
 value=mpe>value?mpe:value;value=root>=layer+1?value:0.0f;
 fractions[slot]=value;sum=__fadd_rn(sum,value);
}
sum=mpe>sum?mpe:sum;total[c]=sum;
for(int layer=0;layer<NSOIL;layer++){
 int slot=c*NSOIL+layer;
 float value=fractions[slot];
 fractions[slot]=root>=layer+1?__fdiv_rn(value,sum):value;
}}
'''
        source = (source.replace("NSOIL", str(nsoil)).replace("NSNOW", str(nsnow))
                  .replace("NLAY", str(nsnow + nsoil)))
        kernel = compile_generated_slab_kernel(source, "noahmp_root_fractions")
        _ROOT_FRACTION_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,),
           (*inputs, roots, np.int32(n), np.float32(mpe), total, fractions))
    return total, fractions


_SELECTED_ROOT_FRACTION_KERNELS = {}


def _selected_root_fraction_slabs(s, nsnow, nsoil, mpe, opt_btr):
    """WRF CLM/SSiB BTRAN with device-resident soil and root carriers.

    Fast CUDA math is the default. GPUWM_WRF_EXACT=1 selects the existing
    native-libm transcription and separate FP32 rounding for proof. Neither
    path reads a device value on the host or iterates over columns in Python.
    """
    import cupy as cp
    from woof.core.kernels import _preamble
    from woof.core.noahmp_kernel_sources import KERNEL_DIR

    inputs = tuple(cp.ascontiguousarray(s[name], dtype=cp.float32)
                   for name in ("sh2o", "psisat", "smcmax", "bexp",
                                "zsoil", "dzsnso"))
    roots = cp.ascontiguousarray(s["nroot"], dtype=cp.int32)
    n = int(roots.size)
    total = cp.empty(n, dtype=cp.float32)
    fractions = cp.empty((n, nsoil), dtype=cp.float32)
    if not n:
        return total, fractions
    key = (cp.cuda.runtime.getDevice(), nsnow, nsoil, opt_btr)
    kernel = _SELECTED_ROOT_FRACTION_KERNELS.get(key)
    if kernel is None:
        # Reuse the audited math source. Do not fork its tables or alter the
        # established BTR1 unit. The compiler records the assembled source
        # and applies the process's strict options to the image-cache key.
        source = (_preamble() + f"#define NMP_BT_NSNOW {int(nsnow)}\n"
                  f"#define NMP_BT_NSOIL {int(nsoil)}\n"
                  f"#define NMP_BT_NLAY {int(nsnow + nsoil)}\n"
                  f"#define NMP_BT_OPTION {opt_btr}\n"
                  + (KERNEL_DIR / "noahmp_leaves.cu").read_text(encoding="ascii")
                  + "\n" + (KERNEL_DIR / "noahmp_btran_options.cuh").read_text(
                      encoding="ascii"))
        kernel = compile_generated_slab_kernel(
            source, "noahmp_root_fractions_selected")
        _SELECTED_ROOT_FRACTION_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,),
           (*inputs, roots, np.int32(n), np.float32(mpe), total, fractions))
    return total, fractions


_SELLERS_RESISTANCE_KERNELS = {}


def sellers_surface_resistance(water, smcmax, fsno, *, opt_rsf):
    """Native Sellers/adjusted Sellers resistance, before dry/urban caps.

    BEVAP has a lower bound of zero and no upper bound in WRF. Snow cover
    blends resistance linearly. The harmonic RSF4 blend is a separate branch.
    """
    if type(opt_rsf) is not int or opt_rsf not in (2, 3):
        raise ValueError("Sellers opt_rsf must be scalar int 2 or 3")
    import cupy as cp
    from woof.core.kernels import _preamble
    from woof.core.noahmp_kernel_sources import KERNEL_DIR

    arrays = cp.broadcast_arrays(*(cp.asarray(v, dtype=cp.float32)
                                   for v in (water, smcmax, fsno)))
    output = cp.empty(arrays[0].shape, dtype=cp.float32)
    n = int(output.size)
    if not n:
        return output
    key = (cp.cuda.runtime.getDevice(), opt_rsf)
    kernel = _SELLERS_RESISTANCE_KERNELS.get(key)
    if kernel is None:
        coefficient = "4.225f" if opt_rsf == 2 else "6.0f"
        source = (_preamble() + f"#define NMP_SELLERS_COEFF {coefficient}\n"
                  + (KERNEL_DIR / "noahmp_leaves.cu").read_text(encoding="ascii")
                  + "\n" + (KERNEL_DIR / "noahmp_sellers_options.cuh").read_text(
                      encoding="ascii"))
        kernel = compile_generated_slab_kernel(source, "noahmp_sellers_resistance")
        _SELLERS_RESISTANCE_KERNELS[key] = kernel
    kernel((_blocks(n),), (THREADS,),
           (*(cp.ascontiguousarray(v) for v in arrays), output, np.int32(n)))
    return output
