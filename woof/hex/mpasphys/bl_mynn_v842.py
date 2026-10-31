# Modified 2026-10-28: experimental HRRR-like profile and aerosol route support.
"""Pinned MPAS-A 8.4.2 MMM MYNN driver boundary.

The adapter is separate from WOOF's WRF/GSL implementation. The registration
entry binds f4's slot registry; the forecast provider supplies original driver
arguments and persistent pool synchronization.
All weather-field arithmetic below executes in CUDA; Python handles arguments.
"""
from __future__ import annotations
from woof.hex.device_memory_release import shared_card_memory
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from typing import Mapping

from ..cuda_fp32 import CUDA_FTZ_HELPERS
from ..native_powf import CUDA_NATIVE_POWF


@dataclass(frozen=True)
class MynnOptions:
    """Registry.xml:2434-2517; module_bl_mynn.F:307-335 option conversion."""

    tkeadvect: bool = False
    tkebudget: int = 0
    closure: float = 2.5
    cloudpdf: int = 2
    mixlength: int = 2
    stfunc: int = 1
    topdown: int = 0
    scaleaware: int = 1
    dheat_opt: int = 1
    edmf: int = 1
    edmf_dd: int = 0
    edmf_mom: int = 0
    edmf_tke: int = 0
    edmf_output: int = 0
    mixscalars: int = 1
    mixclouds: int = 1
    mixqt: int = 0

    @classmethod
    def from_namelist(cls, values: Mapping):
        """mpas_atmphys_driver_pbl.F:855-871, all 17 pinned MYNN options."""
        supplied = {}
        for name, value in values.items():
            if not name.startswith("config_mynn_"):
                continue
            field = name.removeprefix("config_mynn_")
            if field not in cls.__dataclass_fields__:
                raise ValueError(f"{name} is absent from the pinned MPAS v8.4.2 Registry")
            expected = type(getattr(cls(), field))
            if expected is float:
                valid = isinstance(value, (float, int)) and not isinstance(value, bool)
            else:
                valid = isinstance(value, expected) and (expected is bool or not isinstance(value, bool))
            if not valid:
                raise TypeError(f"{name} expects {expected.__name__}, got {type(value).__name__}")
            supplied[field] = value
        return cls(**supplied)

    def mmm_arguments(self):
        """module_bl_mynn.F:307-335, integer switches are enabled only by 1."""
        result = {}
        raw = {"closure", "cloudpdf", "mixlength", "stfunc", "tkeadvect"}
        renamed = {"dheat_opt": "dheatopt", "edmf_output": "output", "mixclouds": "cloudmix"}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            result["bl_mynn_" + renamed.get(name, name)] = value if name in raw else value == 1
        return result


# MPAS physical arrays (k,i) -> driver arrays (i,k), j=1. Layout conversion
# uses explicit device copies. All tensors are contiguous in Fortran order.
# mpas_atmphys_driver_pbl.F:454-507, 661-712.
_DIAGNOSTICS = {
    "el_pbl": "el_pbl", "cov": "cov", "qke": "qke", "qsq": "qsq", "tsq": "tsq",
    "tke_pbl": "tke_pbl", "qke_adv": "qke_adv", "sh3d": "sh3d", "sm3d": "sm3d",
    "cldfrac_bl": "cldfra_bl", "qc_bl": "qc_bl", "qi_bl": "qi_bl", "edmf_a": "edmf_a",
    "edmf_ent": "edmf_ent", "edmf_qc": "edmf_qc", "edmf_qt": "edmf_qt",
    "edmf_thl": "edmf_thl", "edmf_w": "edmf_w", "sub_thl": "sub_thl",
    "sub_qv": "sub_sqv", "det_thl": "det_thl", "det_qv": "det_sqv",
}
_BUDGETS = ("dqke", "qbuoy", "qdiss", "qshear", "qwt")
_ZERO_TENDENCIES = ("rqsblten", "rncblten", "rniblten", "rnifablten", "rnwfablten")

# MPAS bl_mynn_pre.F:61-143, bl_mynn_post.F:61-154 and MMM arithmetic.
# Share the measured GNU word semantics across both the scheme and adapters.
MYNN_IEEE_ARITHMETIC_SUPPORT = r'''
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU ADDSS/SUBSS/MULSS/DIVSS
// quiet the first NaN operand and preserve its payload; invalid non-NaN inputs
// produce the negative default NaN. CUDA arithmetic canonicalizes these bits.
// Only a NaN result is selected here. Finite arithmetic remains in the shared
// device helpers. Strict primitive evidence: tests/mpas_mynn_v842/prepare_ieee_probe.py.
#ifdef __CUDA_ARCH__
__device__ __forceinline__ float mynn_ieee_result(float a,float b,float result) {
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
    return result;
#else
    unsigned int r=__float_as_uint(result);
    if ((r & 0x7fffffffu)<=0x7f800000u) return result;
    unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
    if ((x & 0x7fffffffu)>0x7f800000u) return __uint_as_float(x|0x00400000u);
    if ((y & 0x7fffffffu)>0x7f800000u) return __uint_as_float(y|0x00400000u);
    return __uint_as_float(0xffc00000u);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict primitive NaN selection.
__device__ __forceinline__ double mynn_ieee_result(double a,double b,double result) {
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
    return result;
#else
    unsigned long long r=(unsigned long long)__double_as_longlong(result);
    if ((r & 0x7fffffffffffffffULL)<=0x7ff0000000000000ULL) return result;
    unsigned long long x=(unsigned long long)__double_as_longlong(a);
    unsigned long long y=(unsigned long long)__double_as_longlong(b);
    if ((x & 0x7fffffffffffffffULL)>0x7ff0000000000000ULL)
        return __longlong_as_double(x|0x0008000000000000ULL);
    if ((y & 0x7fffffffffffffffULL)>0x7ff0000000000000ULL)
        return __longlong_as_double(y|0x0008000000000000ULL);
    return __longlong_as_double(0xfff8000000000000ULL);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict primitive NaN selection.
__device__ __forceinline__ float mynn_ieee_add(float a,float b) {
    return mynn_ieee_result(a,b,mpas_add(a,b));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict primitive NaN selection.
__device__ __forceinline__ float mynn_ieee_sub(float a,float b) {
    return mynn_ieee_result(a,b,mpas_sub(a,b));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict primitive NaN selection.
__device__ __forceinline__ float mynn_ieee_mul(float a,float b) {
    return mynn_ieee_result(a,b,mpas_mul(a,b));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict primitive NaN selection.
__device__ __forceinline__ float mynn_ieee_div(float a,float b) {
    return mynn_ieee_result(a,b,mpas_div(a,b));
}
#endif
#ifdef __CUDA_ARCH__
#define mpas_add mynn_ieee_add
#define mpas_sub mynn_ieee_sub
#define mpas_mul mynn_ieee_mul
#define mpas_div mynn_ieee_div
#endif
'''

ADAPTER_CUDA_SOURCE = CUDA_FTZ_HELPERS + CUDA_NATIVE_POWF + MYNN_IEEE_ARITHMETIC_SUPPORT + r'''
// MPAS physics_wrf/bl_mynn_pre.F:61-143, complete numerical conversion.
extern "C" __global__ void mynn_pre(
    const int count, const int f_qc, const int f_qi, const int f_qs,
    const float* qv, const float* qc, const float* qi, const float* qs,
    float* sqv, float* sqc, float* sqi, float* sqs)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const float denom = mpas_add(1.0f, qv[n]);
    sqv[n] = mpas_div(qv[n], denom);
    sqc[n] = f_qc ? mpas_div(qc[n], denom) : 0.0f;
    sqi[n] = f_qi ? mpas_div(qi[n], denom) : 0.0f;
    // Pinned MPAS deliberately uses snow in this denominator. When the
    // species is absent its output is left as provided by the caller.
    if (f_qs) sqs[n] = mpas_div(qs[n], mpas_add(1.0f, qs[n]));
}

// MPAS physics_wrf/bl_mynn_post.F:61-154, complete numerical conversion.
extern "C" __global__ void mynn_post(
    const int count, const int f_qc, const int f_qi, const int f_qs,
    const float delt, const float* qv, const float* qc, const float* qi,
    const float* qs, float* dqv, float* dqc, float* dqi, float* dqs)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    // Preserve the original strict GNU operand destinations, including first
    // NaN selection: product first for vapor, factor first for condensates.
    // Unchanged native bl_mynn_post_run: 2cbc0, 2ccb7, 2cdb9, 2ce88;
    // witness receipts/whole-post-native-lowering-v94/original-post.txt.
    const float denom = mpas_add(qv[n], 1.0f);
    const float sqv = mpas_add(mpas_mul(dqv[n], delt), mpas_div(qv[n], denom));
    const float rqv = mpas_div(sqv, mpas_sub(1.0f, sqv));
    dqv[n] = mpas_div(mpas_sub(rqv, qv[n]), delt);
    // Preserve MPAS's literal 1+updated_specific_vapor condensate factor.
    const float factor = mpas_add(sqv, 1.0f);
    if (f_qc) {
        const float sqc = mpas_add(mpas_div(qc[n], denom), mpas_mul(dqc[n], delt));
        dqc[n] = mpas_div(mpas_sub(mpas_mul(factor, sqc), qc[n]), delt);
    }
    if (f_qi) {
        const float sqi = mpas_add(mpas_div(qi[n], denom), mpas_mul(dqi[n], delt));
        dqi[n] = mpas_div(mpas_sub(mpas_mul(factor, sqi), qi[n]), delt);
    }
    if (f_qs) {
        const float sqs = mpas_add(mpas_div(qs[n], denom), mpas_mul(dqs[n], delt));
        dqs[n] = mpas_div(mpas_sub(mpas_mul(factor, sqs), qs[n]), delt);
    }
}

// RP1 (correct mode): the same conversion with snow over 1+qv, as cloud water and ice.
extern "C" __global__ void mynn_pre_rp1(
    const int count, const int f_qc, const int f_qi, const int f_qs,
    const float* qv, const float* qc, const float* qi, const float* qs,
    float* sqv, float* sqc, float* sqi, float* sqs)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const float denom = mpas_add(1.0f, qv[n]);
    sqv[n] = mpas_div(qv[n], denom);
    sqc[n] = f_qc ? mpas_div(qc[n], denom) : 0.0f;
    sqi[n] = f_qi ? mpas_div(qi[n], denom) : 0.0f;
    if (f_qs) sqs[n] = mpas_div(qs[n], denom);
}

// RP1 (correct mode): condensate returns to mixing ratio by 1/(1-sqv), the exact inverse of the
// 1/(1+qv) forward conversion, as water vapor already does; zero tendencies leave every species unchanged.
extern "C" __global__ void mynn_post_rp1(
    const int count, const int f_qc, const int f_qi, const int f_qs,
    const float delt, const float* qv, const float* qc, const float* qi,
    const float* qs, float* dqv, float* dqc, float* dqi, float* dqs)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    // Water vapor is the unchanged pristine loop in native correct mode: mynn_post's operand order.
    const float denom = mpas_add(qv[n], 1.0f);
    const float sqv = mpas_add(mpas_mul(dqv[n], delt), mpas_div(qv[n], denom));
    const float rest = mpas_sub(1.0f, sqv);
    const float rqv = mpas_div(sqv, rest);
    dqv[n] = mpas_div(mpas_sub(rqv, qv[n]), delt);
    if (f_qc) {
        const float sqc = mpas_add(mpas_div(qc[n], denom), mpas_mul(dqc[n], delt));
        dqc[n] = mpas_div(mpas_sub(mpas_div(sqc, rest), qc[n]), delt);
    }
    if (f_qi) {
        const float sqi = mpas_add(mpas_div(qi[n], denom), mpas_mul(dqi[n], delt));
        dqi[n] = mpas_div(mpas_sub(mpas_div(sqi, rest), qi[n]), delt);
    }
    if (f_qs) {
        const float sqs = mpas_add(mpas_div(qs[n], denom), mpas_mul(dqs[n], delt));
        dqs[n] = mpas_div(mpas_sub(mpas_div(sqs, rest), qs[n]), delt);
    }
}

// MPAS mpas_atmphys_driver_pbl.F:454-485 and 661-694.
extern "C" __global__ void mynn_layout(
    const int ncol, const int nlev, const int to_mpas,
    const float* input, float* output)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= ncol * nlev) return;
    const int c = n / nlev;
    const int k = n % nlev;
    if (to_mpas) output[n] = input[c + ncol*k];
    else output[c + ncol*k] = input[n];
}

// MPAS mpas_atmphys_driver_pbl.F:384-389; original LW+SW order.
extern "C" __global__ void mynn_radiation(
    const int count, const float* lw, const float* sw, float* result)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < count) result[n] = mpas_add(lw[n], sw[n]);
}

// MPAS driver_pbl.F:384-389,454-485; sum directly into MMM layout.
// Input strides retain solve-cell slices and avoid packed temporaries.
extern "C" __global__ void mynn_radiation_layout(
    const int ncol, const int nlev, const float* lw, const float* sw,
    const int lwr, const int lwc, const int swr, const int swc, float* result)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= ncol*nlev) return;
    const int c = n % ncol, k = n / ncol;
    result[n] = mpas_add(lw[k*lwr+c*lwc], sw[k*swr+c*swc]);
}

// MPAS mpas_atmphys_driver_pbl.F:454-462; shared GNU device powf.
extern "C" __global__ void mynn_dx(
    const int ncol, const float len_disp, const float* mesh_density, float* dx)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < ncol)
        dx[c] = mpas_div(len_disp, mpas_native_powf_positive(mesh_density[c], 0.25f));
}
'''


MYNN_CUDA_OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true",
                     "--prec-sqrt=true", "--std=c++17")


def _compile_device_module(source, extra_options=()):
    """bl_mynn.F90:129-1242; preserve native subnormal arithmetic.

    Direct NVRTC avoids CuPy RawModule's forced flush-to-zero compiler option.
    All entry points use extern C names, so no expression mapping is needed.
    Compilation and device loading run under the caller's build/GPU slots.
    """
    import cupy as cp
    import hashlib
    import json
    import os
    from pathlib import Path
    import uuid
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc

    arch = cp.cuda.Device().compute_capability
    options = (*_hex_numerics.options(MYNN_CUDA_OPTIONS), *extra_options)
    identity = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "options": list(options), "arch": arch,
                "nvrtc": list(nvrtc.getVersion()), "cupy": cp.__version__}
    cache_root = os.environ.get("CUPY_CACHE_DIR")
    cache = Path(cache_root) / "mpas_mynn_v842" if cache_root else None
    binary = None
    if cache is not None:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        image_path, record_path = cache / (key + ".bin"), cache / (key + ".json")
        try:
            record = json.loads(record_path.read_text())
            candidate = image_path.read_bytes()
            if (record["identity"] == identity and record["binary_sha256"] ==
                    hashlib.sha256(candidate).hexdigest()):
                binary = candidate
        except (OSError, ValueError, KeyError):
            pass
    if binary is None:
        from ..cuda_backend.aot import direct_image
        binary = direct_image(source, options, arch)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            suffix = "." + uuid.uuid4().hex
            image_tmp = image_path.with_name(image_path.name + suffix)
            record_tmp = record_path.with_name(record_path.name + suffix)
            image_tmp.write_bytes(binary)
            record_tmp.write_text(json.dumps({"identity": identity,
                "binary_sha256": hashlib.sha256(binary).hexdigest()}, sort_keys=True) + "\n")
            os.replace(image_tmp, image_path)
            os.replace(record_tmp, record_path)
    module = cp.cuda.function.Module()
    module.load(binary)
    return module


class MynnDriverAdapter:
    """MPAS pre/post and PBL driver slice; does not register a PBL backend."""

    def __init__(self):
        """bl_mynn_pre.F:61-143; bl_mynn_post.F:61-154 CUDA launch setup."""
        import cupy as cp
        self.cp = cp
        self._module = None

    @property
    def module(self):
        """bl_mynn_pre.F:61-143; compile at first device execution."""
        if self._module is None:
            self._module = _compile_device_module(ADAPTER_CUDA_SOURCE)
        return self._module

    def _checked(self, arrays, shape):
        """module_bl_mynn.F:340-444, single precision and Fortran tensor layout."""
        for name, array in arrays.items():
            if not isinstance(array, self.cp.ndarray):
                raise TypeError(f"{name} must be a CUDA array")
            if array.dtype != self.cp.float32 or array.shape != shape or not array.flags.f_contiguous:
                raise ValueError(f"{name} must be Fortran contiguous float32 with shape {shape}")

    def _launch(self, name, count, args):
        """module_bl_mynn.F:506-618, horizontal/vertical adapter launch."""
        import numpy as np
        self.module.get_function(name)(((count + 127) // 128,), (128,),
                                        tuple(np.int32(a) if isinstance(a, (int, bool)) else a for a in args))

    def pre(self, mixing, specific, flags, *, exact_inverse=False):
        """physics_wrf/bl_mynn_pre.F:61-143, mixing -> specific contents (RP1 kernel in correct mode)."""
        shape = mixing["qv"].shape
        self._checked(mixing, shape)
        self._checked(specific, shape)
        count = mixing["qv"].size
        self._launch("mynn_pre_rp1" if exact_inverse else "mynn_pre", count,
                     (count, *flags, *(mixing[s] for s in ("qv", "qc", "qi", "qs")),
                      *(specific[s] for s in ("sqv", "sqc", "sqi", "sqs"))))

    def post(self, mixing, tendencies, flags, delt, *, exact_inverse=False):
        """physics_wrf/bl_mynn_post.F:61-154, specific -> mixing tendencies (RP1 kernel in correct mode)."""
        import numpy as np
        shape = mixing["qv"].shape
        self._checked(mixing, shape)
        self._checked(tendencies, shape)
        count = mixing["qv"].size
        self._launch("mynn_post_rp1" if exact_inverse else "mynn_post", count,
                     (count, *flags, np.float32(delt),
                      *(mixing[s] for s in ("qv", "qc", "qi", "qs")),
                      *(tendencies[s] for s in ("dqv", "dqc", "dqi", "dqs"))))

    def layout(self, array, *, to_mpas=False):
        """mpas_atmphys_driver_pbl.F:454-485, 661-694, transpose on device."""
        self._checked({"array": array}, array.shape)
        if array.ndim != 2:
            raise ValueError("MYNN driver layout requires a two-dimensional array")
        ncol, nlev = array.shape if to_mpas else array.shape[::-1]
        output = self.cp.empty(array.shape[::-1], dtype=self.cp.float32, order="F")
        self._launch("mynn_layout", array.size, (ncol, nlev, to_mpas, array, output))
        return output

    def radiation(self, lw, sw):
        """driver_pbl.F:384-389,454-485; one sum/layout launch, no staging."""
        if (not isinstance(lw, self.cp.ndarray) or not isinstance(sw, self.cp.ndarray)
                or lw.ndim != 2 or sw.shape != lw.shape or lw.dtype != self.cp.float32
                or sw.dtype != self.cp.float32):
            raise ValueError("MYNN radiation requires matching float32 (level,column) arrays")
        nlev, ncol = lw.shape
        result = self.cp.empty((ncol, nlev), dtype=self.cp.float32, order="F")
        if result.size:
            self._launch("mynn_radiation_layout", result.size,
                         (ncol, nlev, lw, sw, *(s // 4 for s in (*lw.strides, *sw.strides)), result))
        return result

    def from_mpas(self, diagnostics, surface, mesh_density, len_disp, *, extra_copies=()):
        """mpas_atmphys_driver_pbl.F:419-507, scheme-specific input slice.

        Shared PBL input fields are assembled by the common physics seam.
        Returned fields use the MPAS wrapper names and (column,level) Fortran layout.
        Independent binder copies and clears share this launch via extra_copies.
        """
        import numpy as np
        # A column-fastest (level, column) pool array is already the MMM
        # (column, level) Fortran array: MMM then updates the pool in place, as
        # native's by-reference call does, and the commit has nothing to copy.
        fields, copies = {}, list(extra_copies)
        for src, dst in _DIAGNOSTICS.items():
            view = diagnostics[src].T
            if view.flags.f_contiguous and view.dtype == self.cp.float32:
                fields[dst] = view
            else:
                fields[dst] = self.cp.empty(view.shape, dtype=self.cp.float32, order="F")
                copies.append((fields[dst], view))
        shape = fields["qke"].shape
        ncol, _ = shape
        self._checked({"meshDensity": mesh_density}, (ncol,))
        for src, dst in (("ch", "ch"), ("qsfc", "qsfc"), ("rmol", "rmol"), ("skintemp", "ts")):
            self._checked({src: surface[src]}, (ncol,))
            fields[dst] = self.cp.empty(ncol, dtype=self.cp.float32)
            copies.append((fields[dst], surface[src]))
        fields["dx"] = self.cp.empty(ncol, dtype=self.cp.float32)
        self._launch("mynn_dx", ncol, (ncol, np.float32(len_disp), mesh_density, fields["dx"]))
        for name in (*_BUDGETS, *_ZERO_TENDENCIES, "pattern_spp_pbl"):
            fields[name] = self.cp.empty(shape, dtype=self.cp.float32, order="F")
            copies.append((fields[name], None))
        for name in ("maxwidth", "maxmf", "ztop_plume"):
            fields[name] = self.cp.empty(ncol, dtype=self.cp.float32)
            copies.append((fields[name], None))
        fields["ktop_plume"] = self.cp.empty(ncol, dtype=self.cp.int32)
        copies.append((fields["ktop_plume"], None))
        self.copy_many(copies)
        return fields

    def copy_many(self, pairs):
        """driver_pbl.F:454-507,628-725; parallel pool layout transfers."""
        from .mynn_copies_v842 import MynnCopies
        copies = getattr(self, '_copies', None)
        if copies is None:
            copies = self._copies = MynnCopies(self.cp)
        copies.copy_many(pairs)

    def to_mpas(self, fields, *, f_ni, f_nc, f_nifa, f_nwfa):
        """mpas_atmphys_driver_pbl.F:628-725, scheme-specific output slice."""
        diagnostics = {src: self.layout(fields[dst], to_mpas=True) for src, dst in _DIAGNOSTICS.items()}
        diagnostics.update({s: self.layout(fields[s], to_mpas=True) for s in _BUDGETS})
        names = ["rqsblten"]
        if f_ni:
            names.append("rniblten")
        if f_nc and f_nifa and f_nwfa:
            names.extend(("rncblten", "rnifablten", "rnwfablten"))
        tendencies = {s: self.layout(fields[s], to_mpas=True) for s in names}
        return diagnostics, tendencies


# CUDA descriptor support for the original automatic arrays and array sections.
# MPAS physics_mmm/bl_mynn.F90:197-435 and bl_mynn_subroutines.F90:488-6470.
# This supplies storage/indexing only. Numerical routines are generated below.
MYNN_ARRAY_SUPPORT = MYNN_IEEE_ARITHMETIC_SUPPORT + r'''

// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU CVTSS2SD preserves
// NaN sign/payload and quiets signaling inputs. Finite conversion is unchanged.
__device__ __forceinline__ double mynn_real64(float a) {
#if defined(__CUDA_ARCH__) && !(defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST)
    unsigned int x=__float_as_uint(a);
    if ((x & 0x7fffffffu)>0x7f800000u) {
        unsigned long long bits=((unsigned long long)(x & 0x80000000u)<<32)
            |0x7ff8000000000000ULL|((unsigned long long)(x & 0x007fffffu)<<29);
        return __longlong_as_double(bits);
    }
#endif
    return (double)a;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU CVTSD2SS truncates
// the NaN payload, preserves sign and sets the binary32 quiet bit.
__device__ __forceinline__ float mynn_real32(double a) {
#if defined(__CUDA_ARCH__) && !(defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST)
    unsigned long long x=(unsigned long long)__double_as_longlong(a);
    if ((x & 0x7fffffffffffffffULL)>0x7ff0000000000000ULL) {
        unsigned int bits=(unsigned int)((x>>32)&0x80000000ULL)|0x7fc00000u
            |(unsigned int)((x & 0x000fffffffffffffULL)>>29);
        return __uint_as_float(bits);
    }
#endif
    return (float)a;
}
#ifdef __CUDA_ARCH__
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve measured GNU
// NaN results at the approved shared libm boundary; finite libm is unchanged.
__device__ __forceinline__ float mynn_ieee_exp(float a) { return mynn_ieee_result(a,a,mynn_exp(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ double mynn_ieee_exp(double a) { return mynn_ieee_result(a,a,mynn_exp(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ float mynn_ieee_log(float a) { return mynn_ieee_result(a,a,mynn_log(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ double mynn_ieee_log(double a) { return mynn_ieee_result(a,a,mynn_log(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ float mynn_ieee_log10(float a) { return mynn_ieee_result(a,a,mynn_log10(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ float mynn_ieee_atan(float a) { return mynn_ieee_result(a,a,mynn_atan(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ float mynn_ieee_tanh(float a) { return mynn_ieee_result(a,a,mynn_tanh(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict libm NaN result.
__device__ __forceinline__ float mynn_ieee_erf(float a) { return mynn_ieee_result(a,a,mynn_erf(a)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU POW propagates a
// signaling NaN even for x**0 and 1**y. The shared finite POW path is retained.
__device__ __forceinline__ float mynn_ieee_pow(float a,float b) {
#if !(defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST)
    unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
    // The pinned GNU/libm primitive bank measures positive quiet NaNs for
    // NaN**(+/-1), including negative signaling inputs; finite POW is shared.
    if ((x&0x7fffffffu)>0x7f800000u && (y&0x7fffffffu)==0x3f800000u)
        return __uint_as_float((x&0x7fffffffu)|0x00400000u);
    bool signaling_x=(x&0x7fc00000u)==0x7f800000u && (x&0x003fffffu)!=0;
    bool signaling_y=(y&0x7fc00000u)==0x7f800000u && (y&0x003fffffu)!=0;
    if(signaling_x || signaling_y) return mynn_ieee_result(a,b,__uint_as_float(0xffc00000u));
#endif
    return mynn_ieee_result(a,b,mynn_pow(a,b));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU binary64 POW NaNs.
__device__ __forceinline__ double mynn_ieee_pow(double a,double b) {
#if !(defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST)
    unsigned long long x=(unsigned long long)__double_as_longlong(a),y=(unsigned long long)__double_as_longlong(b);
    if ((x&0x7fffffffffffffffULL)>0x7ff0000000000000ULL &&
        (y&0x7fffffffffffffffULL)==0x3ff0000000000000ULL)
        return __longlong_as_double((x&0x7fffffffffffffffULL)|0x0008000000000000ULL);
    bool signaling_x=(x&0x7ff8000000000000ULL)==0x7ff0000000000000ULL && (x&0x0007ffffffffffffULL)!=0;
    bool signaling_y=(y&0x7ff8000000000000ULL)==0x7ff0000000000000ULL && (y&0x0007ffffffffffffULL)!=0;
    if(signaling_x || signaling_y) return mynn_ieee_result(a,b,__longlong_as_double(0xfff8000000000000ULL));
#endif
    return mynn_ieee_result(a,b,mynn_pow(a,b));
}
#define mynn_exp mynn_ieee_exp
#define mynn_log mynn_ieee_log
#define mynn_log10 mynn_ieee_log10
#define mynn_atan mynn_ieee_atan
#define mynn_tanh mynn_ieee_tanh
#define mynn_erf mynn_ieee_erf
#define mynn_pow mynn_ieee_pow
#endif
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; GNU unary minus flips
// the sign bit, including NaN payloads and signaling inputs, without arithmetic.
__device__ __forceinline__ float mynn_neg(float a) {
#ifdef __CUDA_ARCH__
    unsigned int bits;
    asm volatile("xor.b32 %0,%1,0x80000000;" : "=r"(bits) : "r"(__float_as_uint(a)));
    return __uint_as_float(bits);
#else
    return -a;
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; binary64 unary minus.
__device__ __forceinline__ double mynn_neg(double a) {
#ifdef __CUDA_ARCH__
    unsigned long long bits;
    asm volatile("xor.b64 %0,%1,0x8000000000000000;" : "=l"(bits) : "l"((unsigned long long)__double_as_longlong(a)));
    return __longlong_as_double(bits);
#else
    return -a;
#endif
}
template<int R> struct MynnIndices { int value[R]; };
struct MynnRange { int lo, hi; };
template<class I> struct MynnIsRange { static constexpr int value = 0; };
template<> struct MynnIsRange<MynnRange> { static constexpr int value = 1; };
struct MynnArena {
    unsigned char* data;
    unsigned long long offset;
    // One warp shares a tile; each logical element has one word per lane.
    // Width one retains the ordinary host witness layout.
    int width = 1;
    int lane = 0;
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ void* allocate(unsigned long long bytes, int itemsize) {
        offset = (offset + 7ULL) & ~7ULL;
        void* p = data + offset*width + lane*itemsize;
        offset += bytes;
        return p;
    }
};
struct MynnFrame {
    MynnArena& arena;
    unsigned long long saved;
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnFrame(MynnArena& a) : arena(a), saved(a.offset) {}
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ ~MynnFrame() { arena.offset = saved; }
};
template<class T, int R> struct MynnArray {
    T* data = nullptr;
    int lo[R], hi[R], stride[R];
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ bool present() const { return data != nullptr; }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ int extent(int d) const { return hi[d] - lo[d] + 1; }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ int size() const {
        int n = 1;
        for (int d=0; d<R; ++d) n *= extent(d);
        return n;
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ void rebase(MynnIndices<R> lower) {
        for (int d=0; d<R; ++d) {
            int n = extent(d);
            lo[d] = lower.value[d]; hi[d] = lo[d]+n-1;
        }
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ T& linear(int index) const {
        int offset = 0;
        for (int d=0; d<R; ++d) {
            offset += (index % extent(d))*stride[d];
            index /= extent(d);
        }
        return data[offset];
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    template<class... I> __device__ T& operator()(I... indices) const {
        static_assert(sizeof...(I) == R, "Fortran array rank");
        int ix[] = {int(indices)...};
        int offset = 0;
        for (int d=0; d<R; ++d) offset += (ix[d]-lo[d])*stride[d];
        return data[offset];
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    template<int N> __device__ void select(MynnArray<T,N>& out, int& d, int& od, int& offset, MynnRange r) const {
        offset += (r.lo-lo[d])*stride[d];
        out.lo[od] = 1; out.hi[od] = r.hi-r.lo+1;
        out.stride[od++] = stride[d++];
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    template<int N> __device__ void select(MynnArray<T,N>& out, int& d, int& od, int& offset, int i) const {
        offset += (i-lo[d])*stride[d++];
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    template<class... I> __device__ auto slice(I... indices) const {
        constexpr int N = (MynnIsRange<I>::value + ... + 0);
        static_assert(N > 0, "Array section must retain a dimension");
        MynnArray<T,N> out;
        int d=0, od=0, offset=0;
        (select(out, d, od, offset, indices), ...);
        out.data = data + offset;
        return out;
    }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    template<int N> __device__ MynnArray<T,N> sequence(MynnIndices<N> lower,MynnIndices<N> upper) const {
        MynnArray<T,N> out;
        out.data=data;
        int step=1;
        for (int d=0; d<N; ++d) {
            out.lo[d]=lower.value[d]; out.hi[d]=upper.value[d];
            out.stride[d]=step; step *= out.extent(d);
        }
        return out;
    }
};
// MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
template<class T,int R> __device__ MynnArray<T,R> mynn_array(MynnArena& arena, MynnIndices<R> lo, MynnIndices<R> hi) {
    static_assert(sizeof(T) == 4, "MYNN warp scratch requires four-byte elements");
    MynnArray<T,R> result;
    int count=1;
    for (int d=0; d<R; ++d) {
        result.lo[d]=lo.value[d]; result.hi[d]=hi.value[d];
        result.stride[d]=count*arena.width;
        count *= result.extent(d);
    }
    result.data = static_cast<T*>(arena.allocate(sizeof(T)*count, sizeof(T)));
    return result;
}
struct MynnText {
    const char* text = "";
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnText() = default;
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnText(const char* s) : text(s) {}
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnText& operator=(const char* s) { text=s; return *this; }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ bool operator==(const MynnText& b) const {
        int i=0;
        while (text[i] && text[i] == b.text[i]) ++i;
        return text[i] == b.text[i];
    }
};
template<class T> struct MynnOptional {
    T value{};
    bool supplied = false;
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnOptional() = default;
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ MynnOptional(T v) : value(v), supplied(true) {}
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ operator T() const { return value; }
    // MPAS physics_mmm/bl_mynn.F90:197-435; represent original array bounds, sections and optional presence.
    __device__ bool present() const { return supplied; }
};
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
// Strict GNU's MINSS returns operand two on ties/unordered comparisons.
template<class T> __device__ T mynn_min(T a,T b) { return a<b ? a:b; }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
// Strict GNU's MAXSS returns operand two on ties/unordered comparisons.
template<class T> __device__ T mynn_max(T a,T b) { return a>b ? a:b; }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict MINSS/MAXSS selection.
// Keep binary32 MIN/MAX calls separate. The admitted sm_89 compiler changes
// 52 NaN words in the inlined clamp while each isolated operation matches GNU;
// separate calls reproduce all 7813 strict primitive words (minmax-noinline-v49).
#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST
// The isolated-call workaround protects strict NaN words. Fast numerics
// keeps the ordered comparison but inlines the per-level clamp operations.
#define MYNN_MINMAX_ATTR __forceinline__
#else
#define MYNN_MINMAX_ATTR __noinline__
#endif
__device__ MYNN_MINMAX_ATTR float mynn_min(float a,float b) {
#ifdef __CUDA_ARCH__
    float result;
    // A comparison/select cannot become CUDA's different NaN MIN/MAX.
    asm volatile("{ .reg .pred p; setp.lt.f32 p,%1,%2; selp.f32 %0,%1,%2,p; }"
                 : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    return a<b ? a:b;
#endif
}

// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict MINSS/MAXSS selection.
__device__ MYNN_MINMAX_ATTR float mynn_max(float a,float b) {
#ifdef __CUDA_ARCH__
    float result;
    // A comparison/select cannot become CUDA's different NaN MIN/MAX.
    asm volatile("{ .reg .pred p; setp.gt.f32 p,%1,%2; selp.f32 %0,%1,%2,p; }"
                 : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    return a>b ? a:b;
#endif
}

#undef MYNN_MINMAX_ATTR
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict MINSS/MAXSS selection.
__device__ double mynn_min(double a,double b) {
#ifdef __CUDA_ARCH__
    double result;
    // A comparison/select cannot become CUDA's different NaN MIN/MAX.
    asm volatile("{ .reg .pred p; setp.lt.f64 p,%1,%2; selp.f64 %0,%1,%2,p; }"
                 : "=d"(result) : "d"(a), "d"(b));
    return result;
#else
    return a<b ? a:b;
#endif
}

// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; strict MINSS/MAXSS selection.
__device__ double mynn_max(double a,double b) {
#ifdef __CUDA_ARCH__
    double result;
    // A comparison/select cannot become CUDA's different NaN MIN/MAX.
    asm volatile("{ .reg .pred p; setp.gt.f64 p,%1,%2; selp.f64 %0,%1,%2,p; }"
                 : "=d"(result) : "d"(a), "d"(b));
    return result;
#else
    return a>b ? a:b;
#endif
}

// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
template<class T,int R> __device__ T mynn_minval(MynnArray<T,R> a) {
    T value=a.linear(0);
    for (int i=1; i<a.size(); ++i) value=mynn_min(value,a.linear(i));
    return value;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
template<class T,int R> __device__ T mynn_maxval(MynnArray<T,R> a) {
    T value=a.linear(0);
    for (int i=1; i<a.size(); ++i) value=mynn_max(value,a.linear(i));
    return value;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:6067; MINLOC first minimum and Fortran indexing.
template<class F> __device__ int mynn_minloc(int count,F function) {
    int index=0;
    auto value=function(0);
    for (int i=1; i<count; ++i) {
        auto next=function(i);
        if (next<value) { value=next; index=i; }
    }
    return index+1;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dadd(double a,double b) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,b,__dadd_rn(a,b));
#else
    return __dadd_rn(a,b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dsub(double a,double b) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,b,__dsub_rn(a,b));
#else
    return __dsub_rn(a,b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_dmul(double a,double b) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,b,__dmul_rn(a,b));
#else
    return __dmul_rn(a,b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_ddiv(double a,double b) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,b,__ddiv_rn(a,b));
#else
    return __ddiv_rn(a,b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ float mynn_sqrt(float a) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,a,mpas_sqrt(a));
#else
    return mpas_sqrt(a);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_sqrt(double a) {
#ifdef __CUDA_ARCH__
    return mynn_ieee_result(a,a,__dsqrt_rn(a));
#else
    return __dsqrt_rn(a);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ float mynn_abs(float a) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(__float_as_uint(a)&0x7fffffffu);
#else
    return mpas_abs(a);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ double mynn_abs(double a) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double((unsigned long long)__double_as_longlong(a)&0x7fffffffffffffffULL);
#else
    return std::fabs(a);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ int mynn_abs(int a) { return a<0 ? -a:a; }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ float mynn_sign(float a,float b) {
#ifdef __CUDA_ARCH__
    return __uint_as_float((__float_as_uint(a)&0x7fffffffu)|(__float_as_uint(b)&0x80000000u));
#else
    return copysignf(mynn_abs(a),b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:1866,2680; SIGN preserves binary64 covariance.
__device__ double mynn_sign(double a,double b) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(((unsigned long long)__double_as_longlong(a)&0x7fffffffffffffffULL)
        |((unsigned long long)__double_as_longlong(b)&0x8000000000000000ULL));
#else
    return copysign(a,b);
#endif
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ int mynn_mod(int a,int b) { return a%b; }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ int mynn_ceiling(float a) { return __float2int_ru(a); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ int mynn_floor(float a) { return __float2int_rd(a); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
__device__ int mynn_nint(float a) { return (int)(a<0 ? mpas_sub(a,0.5f):mpas_add(a,0.5f)); }
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
template<int N> __device__ float mynn_ipow(float x) {
    static_assert(N==2 || N==3, "Pinned MYNN integer powers");
    float xx=mpas_mul(x,x);
    if constexpr(N==2) return xx;
    return mpas_mul(xx,x);
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-6560; preserve intrinsic/operator kind and operation order.
template<int N> __device__ double mynn_ipow(double x) {
    static_assert(N==2 || N==3, "Pinned MYNN integer powers");
    double xx=mynn_dmul(x,x);
    if constexpr(N==2) return xx;
    return mynn_dmul(xx,x);
}
'''
#: Worker scratch of the MYNN kernel, one per (device, stream, per-worker stride): (workspace, workers).
#: THE COST THIS REMOVES (MPAS-route LTS memory, 2026-10-27): every rate class's MYNN module sized its own scratch
#: from the card's free memory (up to one eighth of it, 99 KB per worker), so four classes on one card held 5.6 GB
#: of it on a 99,898-cell mesh although only one class integrates at a time.  The kernel's words do not depend on
#: the worker count (each column is independent, in its worker's private arena), so a shared block changes none.
_SHARED_WORKSPACE: dict = {}
_SHARED_LOCK = __import__("threading").Lock()


def _column_batch(ncol, stride, processors, free_bytes, *, shared_card=False):
    """Resident workers: four blocks per SM, at most one eighth of free memory.

    Round storage to full blocks because the final partial warp still uses
    32-wide scratch tiles. Columns beyond the resident width use grid strides.
    """
    budget = max(64 * stride, free_bytes // 8)
    affordable = max(64, (budget // stride // 64) * 64)
    return min(((ncol + 63) // 64) * 64, max(64, (64 if shared_card else 256) * processors), affordable)


_STATIC_COLUMNS = (
    "const int worker = blockIdx.x*blockDim.x+threadIdx.x;\n"
    "for (int thread = worker; thread < batch_count; thread += blockDim.x*gridDim.x) {\n")
_DYNAMIC_COLUMNS = (
    "const int worker = blockIdx.x*blockDim.x+threadIdx.x;\n"
    "const int mynn_lane = threadIdx.x & 31;\n"
    "int mynn_base = worker & ~31;\n"
    "for (;;) {\n"
    "if (mynn_base >= batch_count) break;\n"
    "{ const int thread = mynn_base + mynn_lane;\n"
    "if (thread < batch_count) {\n")
_STATIC_END = "error_flags[batch_first+thread]=errflg;\n}\n}\n"
_DYNAMIC_END = (
    "error_flags[batch_first+thread]=errflg;\n} }\n"
    "int mynn_next = 0;\n"
    "if (mynn_lane == 0) mynn_next = (int)atomicAdd(&g_mynn_next_column, 32u);\n"
    "mynn_next = __shfl_sync(0xffffffffu, mynn_next, 0);\n"
    "mynn_base = mynn_next + (int)(blockDim.x*gridDim.x);\n"
    "}\n}\n")


def _dynamic_columns(source):
    """Fast numerics: warps take 32-column chunks from a device counter.

    The static grid stride gave every worker the same column count, so the
    few warps holding EDMF plume columns (dmp_mf plus condensation_edmf are
    ~40% of the kernel on 5% of columns) set the kernel's end while the rest
    of the card idled. Each warp still starts on its own first 32 columns and
    then takes the next free chunk. Columns are independent and use only their
    worker's private arena, so the words do not depend on which warp computes
    a column (measured: 0 of 711,471,929 output words differ on a captured
    216,187-column call). Every lane of a 64-thread block reaches the shuffle.
    """
    if source.count(_STATIC_COLUMNS) != 1 or not source.endswith(_STATIC_END):
        raise ValueError("MYNN column loop changed; dynamic column scheduling needs review")
    head = source[:-len(_STATIC_END)].replace(_STATIC_COLUMNS, _DYNAMIC_COLUMNS, 1)
    return "__device__ unsigned int g_mynn_next_column;\n" + head + _DYNAMIC_END


class MynnCudaModule:
    """Whole MMM module using MPAS's bl_mynn_run argument names.

    Shared device maths supplies mynn_pow/exp/log/log10/atan/tanh/erf overloads.
    It is supplied separately so the same generated numerical text can be
    checked with the strict host-glibc witness and the shared CUDA header.
    This class is not yet registered as a proven forecast backend.
    """

    def __init__(self, *, shared_math_source: str, corrections=frozenset()):
        """bl_mynn.F90:24-107,129-1242; whole numerical module compilation.

        ``corrections`` are the correct-mode findings whose kernel text this module carries
        (MYNN_TEXT_CORRECTIONS); none compiles the exact text byte for byte."""
        import cupy as cp
        from .driver_v842 import correct_source
        self.cp = cp
        numerical = correct_source(MYNN_MODULE_CUDA, MYNN_TEXT_CORRECTIONS, corrections,
                                   "bl_mynn hex-cuda")
        # Fast numerics balances columns across warps (G4, MPAS-ROUTE-SPEED-2026-10-24).
        # Strict numerics keeps the proven image byte for byte.
        self._dynamic = not _hex_numerics.is_strict()
        if self._dynamic:
            numerical = _dynamic_columns(numerical)
        self._source = CUDA_FTZ_HELPERS + shared_math_source + MYNN_ARRAY_SUPPORT + numerical
        self._module = None
        self._resident = None
        self._workers = None
        self._shared_card_memory = shared_card_memory()
        self._next_column = None

    def _storage(self, ncol, stride):
        """Worker scratch allocated once per device and shape; error flags per call.

        The kernel's words do not depend on the worker count (each column is
        independent; the grid strides over columns), so the first call's
        budget is kept. Allocating per call cost cudaMalloc/cudaFree pairs
        (each free a device sync) on every step once the slot's pool blocks
        were released (MPAS-ROUTE-SPEED-2026-10-24 G4: 70 mallocs + 45 frees).
        """
        if self._shared_card_memory:
            return self._borrow_storage(ncol, stride)
        cp = self.cp
        device = cp.cuda.runtime.getDevice()
        # Error flags stay per call: callers may read a step's flags later.
        errors = cp.empty(ncol, dtype=cp.int32)
        # The scratch is shared by every MYNN module on this device and stream (_SHARED_WORKSPACE): kernels on one
        # stream run in order, so they can never use it at once.
        key = (device, int(cp.cuda.get_current_stream().ptr), int(stride))
        held = self._resident
        if held is not None and held[0] == (device, ncol, stride, key[1]):
            shared = _SHARED_WORKSPACE.get(key)
            if shared is not None and shared[0] is held[1]:
                return held[1], errors, held[2]
        self._resident = None
        with _SHARED_LOCK:
            shared = _SHARED_WORKSPACE.get(key)
            properties = cp.cuda.runtime.getDeviceProperties(device)
            processors = properties['multiProcessorCount']
            wanted = min(((ncol + 63) // 64) * 64, max(64, 256 * processors))
            if shared is None or shared[1] < wanted:
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                if shared is not None:
                    free_bytes += shared[0].nbytes     # the block this replaces is given back first
                batch = _column_batch(ncol, stride, processors, free_bytes)
                if shared is None or shared[1] < batch:
                    shared = None
                    _SHARED_WORKSPACE.pop(key, None)
                    shared = (cp.empty(batch * stride, dtype=cp.uint8), batch)
                    _SHARED_WORKSPACE[key] = shared
            workspace, capacity = shared
        batch = min(wanted, capacity)
        self._resident = ((device, ncol, stride, key[1]), workspace, batch)
        return workspace, errors, batch

    def _borrow_storage(self, ncol, stride):
        """Borrow worker scratch from the stream's pool only for this call.

        Retaining a strong reference across steps prevented the dycore from
        reusing over a GiB per member. CuPy's stream-ordered pool can reuse the
        allocation after this launch without a host wait. Cache the worker
        count so device properties and free memory are read only once per shape.
        Each column still uses its private arena and unchanged arithmetic.
        """
        cp = self.cp
        device = cp.cuda.runtime.getDevice()
        key = (device, int(cp.cuda.get_current_stream().ptr), ncol, stride)
        held = self._workers
        if held is None or held[0] != key:
            processors = cp.cuda.runtime.getDeviceProperties(device)['multiProcessorCount']
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            held = self._workers = (key, _column_batch(ncol, stride, processors, free_bytes, shared_card=True))
        batch = held[1]
        return (cp.empty(batch * stride, dtype=cp.uint8),
                cp.empty(ncol, dtype=cp.int32), batch)

    @property
    def module(self):
        """bl_mynn.F90:129-1242; compile at first device execution."""
        if self._module is None:
            self._module = _compile_device_module(self._source)
        return self._module

    def run(self, arguments: Mapping, constants):
        """bl_mynn.F90:129-435, original argument/option contract and column loop.

        Arrays have the MPAS MMM (column,level) Fortran layout. Constants are
        the 18 con_* arguments to bl_mynn_init, in their original order. All
        field arithmetic executes on the device. Python packs arguments and
        allocates storage for the source's automatic arrays.
        """
        import numpy as np
        cp = self.cp
        its, ite, kts, kte, kme = (int(arguments[n]) for n in ("its", "ite", "kts", "kte", "kme"))
        ncol, nlev = ite-its+1, kte-kts+1
        if constants.dtype != cp.float32 or constants.shape != (18,):
            raise ValueError("MYNN constants require 18 float32 bl_mynn_init arguments")
        values = []
        for name, spec in MYNN_ARGUMENTS.items():
            if name in ("errmsg", "errflg"):
                continue
            value = arguments.get(name)
            if spec["dimensions"]:
                if value is None and spec["optional"]:
                    values.append(np.uint64(0))
                    continue
                dtype = cp.int32 if spec["kind"] == "int" else cp.float32
                shape = (ncol,) if len(spec["dimensions"]) == 1 else (ncol, kme-kts+1 if name == "w" else nlev)
                if not isinstance(value, cp.ndarray) or value.dtype != dtype or value.shape != shape or not value.flags.f_contiguous:
                    raise ValueError(f"{name} requires a Fortran contiguous {dtype} CUDA array with shape {shape}")
                values.append(value)
            elif spec["kind"] in ("int", "bool"):
                values.append(np.int32(value))
            else:
                values.append(np.float32(value))
        # The generator sums every automatic array in every routine. Recursion
        # is absent, so this bounds the active call tree including alignment.
        stride = MYNN_WORKSPACE_LEVEL_BYTES*kte + MYNN_WORKSPACE_FIXED_BYTES
        stride = (stride+7)//8*8
        workspace, errors, batch = self._storage(ncol, stride)
        kernel = self.module.get_function("mynn_mmm_run")
        if self._dynamic:
            # The column counter restarts on the launch stream; no host wait.
            if self._next_column is None:
                self._next_column = int(self.module.get_global_var("g_mynn_next_column"))
            cp.cuda.runtime.memsetAsync(self._next_column, 0, 4, cp.cuda.get_current_stream().ptr)
        kernel((batch // 64,), (64,), (np.int32(0), np.int32(ncol), constants,
                             workspace, np.uint64(stride), errors, *values))
        return errors


# MPAS physics_wrf/module_bl_mynn.F:543-598, complete non-chemistry call map.
WRAPPER_TO_MMM = {
    "restart": "do_restart", "cycling": "do_dacycling", "qnc": "nc",
    "qni": "ni", "qnwfa": "nwfa", "qnifa": "nifa", "qnbca": "nbca",
    "qozone": "qoz", "rqncblten": "rncblten", "rqniblten": "rniblten",
    "rqnwfablten": "rnwfablten", "rqnifablten": "rnifablten",
    "rqnbcablten": "rnbcablten", "sh": "sh3d", "sm": "sm3d",
    "pattern_spp_pbl": "pattern_spp", "flag_qc": "f_qc", "flag_qi": "f_qi",
    "flag_qs": "f_qs", "flag_qnc": "f_nc", "flag_qni": "f_ni",
    "flag_qnwfa": "f_nwfa", "flag_qnifa": "f_nifa", "flag_qnbca": "f_nbca",
    "flag_qoz": "f_qoz",
}
_OPTIONAL_SPECIES = {"sqc": ("qc", "f_qc"), "sqi": ("qi", "f_qi"),
    "sqs": ("qs", "f_qs"), "qnc": ("nc", "f_nc"), "qni": ("ni", "f_ni"),
    "qnwfa": ("nwfa", "f_nwfa"), "qnifa": ("nifa", "f_nifa"),
    "qnbca": ("nbca", "f_nbca"), "qozone": ("qoz", "f_qoz")}
_OPTIONAL_TENDENCIES = {"rqcblten": "f_qc", "rqiblten": "f_qi",
    "rqsblten": "f_qs", "rqozblten": "f_qoz", "rqncblten": "f_nc",
    "rqniblten": "f_ni", "rqnifablten": "f_nifa", "rqnwfablten": "f_nwfa",
    "rqnbcablten": "f_nbca"}


class MynnBlDriver:
    """MPAS mynn_bl_driver on the driver's horizontal/vertical tile.

    The common MPAS driver supplies its j=1 tile as (column,level) Fortran
    arrays, and surface arrays as (column,). Argument names are the original
    wrapper names; every numerical conversion and copy stays on the GPU.
    """

    def __init__(self, *, shared_math_source: str):
        """module_bl_mynn.F:18-58; original driver argument boundary."""
        self.adapter = MynnDriverAdapter()
        self._shared_math_source = shared_math_source
        self.numerical = MynnCudaModule(shared_math_source=shared_math_source)
        self._corrected = {}

    def numerical_for(self, corrections):
        """The numerical module for this run's correct-mode findings (exact text when it applies none)."""
        applied = frozenset(corrections) & frozenset(MYNN_TEXT_CORRECTIONS)
        if not applied:
            return self.numerical
        if not hasattr(self, "_corrected"):
            self._corrected = {}
        module = self._corrected.get(applied)
        if module is None:
            module = self._corrected[applied] = MynnCudaModule(
                shared_math_source=self._shared_math_source, corrections=applied)
        return module

    def run(self, arguments: Mapping, constants, *, options: MynnOptions,
            corrections=frozenset()):
        """module_bl_mynn.F:307-741; complete non-chemistry wrapper slice.

        ``corrections`` are the correct-mode finding ids of the run; RP1 selects the exact-inverse
        moisture conversion (bl_mynn_pre.F:135, bl_mynn_post.F:123,134,145 in matched MPAS).
        """
        cp = self.adapter.cp
        exact_inverse = "RP1" in corrections
        # module_bl_mynn.F:307-308 initializes the character status to blanks.
        message = arguments.get("errmsg")
        if isinstance(message, cp.ndarray):
            message.fill(32)
        # module_bl_mynn.F:347-376,478-525: locals cover only kts:kte;
        # MPAS actual storage can also contain an unwritten interface level.
        # Pack the active section on-device and preserve that entry storage
        # when the wrapper copies its outputs back at :602-741.
        original = arguments
        arguments = dict(arguments)
        shape = (int(arguments["ite"])-int(arguments["its"])+1,
                 int(arguments["kte"])-int(arguments["kts"])+1)
        for name, field in arguments.items():
            if isinstance(field, cp.ndarray) and field.ndim == 2:
                levels = int(arguments["kme"])-int(arguments["kts"])+1 if name == "w" else shape[1]
                if field.shape[0] != shape[0] or field.shape[1] < levels:
                    raise ValueError(f"{name} storage does not contain the original MYNN tile")
                arguments[name] = field[:, :levels]
        values = {}
        direct = set()
        mixing = {"qv": arguments["qv"]}
        copies = []
        if not mixing['qv'].flags.f_contiguous:
            mixing['qv'] = cp.empty(shape, dtype=cp.float32, order='F')
            copies.append((mixing['qv'], arguments['qv']))
        # module_bl_mynn.F:378-452: zero local species first, then copy only
        # those whose flag is true and whose optional argument is present.
        for destination, (source, flag) in _OPTIONAL_SPECIES.items():
            supplied = arguments.get(source)
            field = cp.empty(shape, dtype=cp.float32, order="F")
            copies.append((field, supplied if arguments[flag] and supplied is not None else None))
            if destination.startswith("sq"):
                mixing[source] = field
            else:
                values[destination] = field
        specific = {name: cp.empty(shape, dtype=cp.float32, order="F")
                    for name in ("sqv", "sqc", "sqi", "sqs")}
        flags = tuple(bool(arguments[n]) for n in ("f_qc", "f_qi", "f_qs"))
        values.update(specific)
        # module_bl_mynn.F:462-476: stochastic input is zero unless both the
        # forcing switch and optional pattern argument are supplied.
        pattern = arguments.get("pattern_spp")
        values["pattern_spp_pbl"] = cp.empty(shape, dtype=cp.float32, order="F")
        copies.append((values["pattern_spp_pbl"], pattern
                       if arguments["spp_pbl"] > 0 and pattern is not None else None))
        # module_bl_mynn.F:527-540: these local tendencies start at zero.
        for name in _OPTIONAL_TENDENCIES:
            values[name] = cp.empty(shape, dtype=cp.float32, order="F")
            copies.append((values[name], None))
        for name, spec in MYNN_ARGUMENTS.items():
            if name in values or name in ("errmsg", "errflg") or name.startswith("bl_mynn_"):
                continue
            # module_bl_mynn.F:543-598 omits all six optional DD diagnostics.
            # This reproduces the actual call, independent of the DD option.
            if name.startswith("edmf_") and name.endswith("_dd"):
                values[name] = None
                continue
            source = WRAPPER_TO_MMM.get(name, name)
            value = arguments.get(source)
            if spec["dimensions"]:
                dtype = cp.int32 if spec["kind"] == "int" else cp.float32
                target_shape = (shape[0],) if len(spec["dimensions"]) == 1 else shape
                # Native always supplies its five local budget arrays to MMM.
                # MMM defines them before the wrapper's optional copy-out.
                if name in _BUDGETS:
                    values[name] = cp.empty(target_shape, dtype=dtype, order="F")
                elif (spec["intent"] in ("out", "inout") and isinstance(value, cp.ndarray)
                      and value.shape == target_shape and value.dtype == dtype
                      and value.flags.f_contiguous):
                    # The wrapper's private copy and its copy-back move the same
                    # words; MMM updates the caller's storage in place instead.
                    values[name] = value
                    direct.add(name)
                elif spec["intent"] == "out":
                    values[name] = cp.empty(target_shape, dtype=dtype, order="F")
                elif spec["intent"] == "in":
                    # The generated MMM descriptor reads these arrays only.
                    # Keep the wrapper's private copies for INOUT arguments.
                    if value is None or value.flags.f_contiguous:
                        values[name] = value
                    else:
                        values[name] = cp.empty(value.shape, dtype=dtype, order='F')
                        copies.append((values[name], value))
                else:
                    values[name] = cp.empty(target_shape, dtype=dtype, order="F")
                    copies.append((values[name], value))
            else:
                values[name] = value
        values.update(options.mmm_arguments())
        # In-place storage must not alias any other argument; such a name keeps
        # the wrapper's private copy (module_bl_mynn.F:543-598 copies locals).
        seen = {}
        for name, array in values.items():
            if isinstance(array, cp.ndarray) and array.size:
                seen.setdefault(_address(array), []).append(name)
        for names in seen.values():
            for name in names if len(names) > 1 else ():
                if name in direct:
                    direct.discard(name)
                    private = cp.empty(values[name].shape, dtype=values[name].dtype, order="F")
                    copies.append((private, values[name]))
                    values[name] = private
        self.adapter.copy_many(copies)
        self.adapter.pre(mixing, specific, flags, exact_inverse=exact_inverse)
        errors = self.numerical_for(corrections).run(values, constants)
        post = {name: values[source] for name, source in
                (("dqv", "rqvblten"), ("dqc", "rqcblten"),
                 ("dqi", "rqiblten"), ("dqs", "rqsblten"))}
        self.adapter.post(mixing, post, flags, arguments["delt"], exact_inverse=exact_inverse)
        # module_bl_mynn.F:602-741: optional species tendencies copy out only
        # with their species flag; all budgets copy out as a complete group.
        budgets_present = all(arguments.get(name) is not None for name in _BUDGETS)
        copies = []
        for name, spec in MYNN_ARGUMENTS.items():
            if not spec["dimensions"] or spec["intent"] not in ("out", "inout") or name in direct:
                continue
            source = WRAPPER_TO_MMM.get(name, name)
            target = original.get(source)
            if target is None:
                continue
            if name in _OPTIONAL_TENDENCIES and not arguments[_OPTIONAL_TENDENCIES[name]]:
                continue
            if name in _BUDGETS and not budgets_present:
                continue
            if target.ndim == 2:
                target = target[:, :shape[1]]
            copies.append((target, values[name]))
        self.adapter.copy_many(copies)
        return errors


def register_mynn_backend(registry, *, arguments, after_call, constants,
                          shared_math_source: str):
    """mpas_atmphys_driver_pbl.F:854-960; bind only the MYNN PBL slot.

    ``arguments(call)`` supplies CUDA arrays named by mynn_bl_driver after the
    MPAS X_from_MPAS slice. ``after_call(call, fields, errors)`` commits its
    changed arrays through X_to_MPAS and records native error status. Constants
    are the original 18 single-precision bl_mynn_init arguments on the device.
    The registry follows f4's register(slot, scheme, backend, callback) API.
    This binding performs no default-suite or backend selection.
    """
    driver = MynnBlDriver(shared_math_source=shared_math_source)

    def invoke(call):
        """MPAS mpas_atmphys_driver_pbl.F:855-960; selected slot callback."""
        # MPAS mpas_atmphys_driver_pbl.F:855-871; current namelist options.
        fields = dict(arguments(call))
        errors = driver.run(fields, constants,
                            options=MynnOptions.from_namelist(call.options),
                            corrections=getattr(call, "corrections", frozenset()))
        return after_call(call, fields, errors)

    registry.register("pbl", "bl_mynn", "hex-cuda", invoke)
    return driver


# Native preparation from MPAS_to_physics and driver_pbl, before MYNN's own
# X_from_MPAS. mpas_atmphys_driver_pbl.F:906-934, no interpolated replacements.
_MYNN_PREPARED = {
    "ps": "psfc_p", "dz": "dz_p", "u": "u_p", "v": "v_p",
    "w": "w_p", "th": "th_p", "tt": "t_p", "p": "pres_hyd_p",
    "exner": "pi_p", "rho": "rho_p", "qv": "qv_p", "qc": "qc_p",
    "qi": "qi_p", "qs": "qs_p", "nc": "nc_p", "ni": "ni_p",
    "nifa": "nifa_p", "nwfa": "nwfa_p", "rublten": "rublten_p",
    "rvblten": "rvblten_p", "rthblten": "rthblten_p",
    "rqvblten": "rqvblten_p", "rqcblten": "rqcblten_p",
    "rqiblten": "rqiblten_p", "exch_h": "kzh_p", "exch_m": "kzm_p",
}
# module_bl_mynn.F:114-152 declares these prepared arguments INTENT(IN).
# The moisture pre/post kernels take const pointers; the numerical wrapper
# retains private MMM INOUT storage and copies back only OUT/INOUT arguments.
_MYNN_PREPARED_INPUTS = (
    "ps", "dz", "u", "v", "w", "th", "tt", "p", "exner", "rho",
    "qv", "qc", "qi", "qs", "nc", "ni", "nifa", "nwfa",
)


def _discard_native_inputs(provider):
    """module_bl_mynn.F:114-152; retire only this slot's input-only copies.

    The host-backed path reloads these words next slot. A resident provider
    owns the authoritative device values and must retain them.
    """
    if getattr(provider, "resident", False):
        return
    for name in _MYNN_PREPARED_INPUTS:
        provider.buffers.pop(("prepared", _MYNN_PREPARED[name]), None)


# MPAS mpas_atmphys_driver_pbl.F:746-747; exact compiled constant arguments.
_MYNN_CONSTANTS = ("cp", "cpv", "cice", "cliq", "ep_1", "ep_2", "gravity",
                   "karman", "p0", "r_d", "r_v", "svp1", "svp2", "svp3",
                   "svpt0", "xlf", "xls", "xlv")
# MPAS mpas_atmphys_vars.F:153-155,249; native manager tile and species flags.
_MYNN_FLAGS = ("f_qc", "f_qi", "f_qs", "f_qoz", "f_nc", "f_ni",
               "f_nifa", "f_nwfa", "f_nbca")
_MYNN_BOUNDS = ("ids", "ide", "jds", "jde", "kds", "kde", "ims", "ime",
                "jms", "jme", "kms", "kme", "its", "ite", "jts", "jte",
                "kts", "kte")


def load_mynn_resources(resources, device):
    """MPAS mpas_atmphys_driver_pbl.F:746-747,855-960; bind certified shared maths.

    f4 supplies a provenance packet, rather than choosing a math fallback.
    Loading source text performs no field arithmetic or device initialization.
    """
    from .table_resources_v842 import certified_path

    return {"device_math_source": certified_path(resources["device_math_source"]).read_text()}


def bind_native_pools(provider, call):
    """MPAS mpas_atmphys_driver_pbl.F:306-507,818-960; whole driver binding.

    Native manager/MPAS_to_physics own common prepared input storage. This
    scheme supplies its complete X_from_MPAS and X_to_MPAS conversions on
    CUDA. The session installs provider.mynn_v842_resources with approved
    device_math_source; the provider commits modified solve cells and halos.
    """
    if call.slot != "pbl" or call.scheme != "bl_mynn" or str(call.backend) != "hex-cuda":
        raise ValueError("MYNN native binder requires its exact registered PBL slot")
    if provider.level != 1:
        raise ValueError("MPAS PBL reads state at time level 1")
    resources = getattr(provider, "mynn_v842_resources", None)
    if resources is None:
        raise RuntimeError("native session must supply approved MYNN shared device maths")
    maths = resources["device_math_source"]
    if not isinstance(maths, str) or not maths.strip():
        raise ValueError("MYNN resources require the approved shared device maths source")
    cp = provider.cp
    driver = resources.get("driver")
    if driver is None:
        driver = resources["driver"] = MynnBlDriver(shared_math_source=maths)
    constants = cp.asarray([provider["constants"][name] for name in _MYNN_CONSTANTS],
                           dtype=cp.float32)
    scalar = provider["scalar_state"]
    fields = {name: int(scalar[name]) for name in _MYNN_BOUNDS}
    fields.update({name: bool(scalar[name]) for name in _MYNN_FLAGS})
    from .radshared_v842 import icloud_bl_enabled
    configs = provider["configs"]
    fields.update(delt=scalar["dt_pbl"], do_restart=bool(configs["config_do_restart"]),
                  do_dacycling=bool(configs["config_do_DAcycling"]),
                  initflag=int(not configs["config_do_restart"] and call.itimestep <= 1),
                  icloud_bl=int(icloud_bl_enabled(call.options)), spp_pbl=0, errflg=0)
    # Native mpas_atmphys_vars.F:417-418 fixes both at 0; the experimental
    # radiation coupling opts into icloud_bl=1. The default remains native.
    # driver_pbl.F:813-816 initializes StrKIND status, not a diagnostic refusal.
    # MynnBlDriver.run initializes the status once before its numerical call.
    fields["errmsg"] = cp.empty(512, dtype=cp.uint8)
    # MPAS framework/mpas_kind_types.F:32 declares StrKIND=512.
    prepared = provider["prepared"]
    copies = []
    for name, native in _MYNN_PREPARED.items():
        value = prepared[native]
        view = value.T if value.ndim == 2 else value
        if view.flags.f_contiguous and (value.ndim == 2 or name in _MYNN_PREPARED_INPUTS):
            fields[name] = view
        else:
            fields[name] = cp.empty(view.shape, dtype=view.dtype, order='F')
            copies.append((fields[name], view))
    diag = provider["diag_physics"]
    tendencies = provider["tend_physics"]
    surface = {name: diag[name] for name in ("ch", "qsfc", "rmol")}
    surface["skintemp"] = provider["sfc_input"]["skintemp"]
    diagnostics = {name: diag[name] for name in _DIAGNOSTICS}
    for name in ("hfx", "qfx", "ust", "wspd", "znt"):
        fields[name] = cp.empty_like(diag[name])
        copies.append((fields[name], diag[name]))
    for name, source in (("pblh", diag["hpbl"]), ("xland", provider["sfc_input"]["xland"])):
        fields[name] = cp.empty_like(source)
        copies.append((fields[name], source))
    count = fields["ite"] - fields["its"] + 1
    fields["kpbl"] = cp.empty(count, dtype=cp.int32)
    copies.append((fields["kpbl"], 1))
    for name in ("uoce", "voce"):
        fields[name] = cp.empty(count, dtype=cp.float32)
        copies.append((fields[name], None))
    fields.update(driver.adapter.from_mpas(diagnostics, surface,
                  provider["mesh"]["meshDensity"], provider["configs"]["config_len_disp"],
                  extra_copies=copies))
    nz = fields["kte"] - fields["kts"] + 1
    fields["rthraten"] = driver.adapter.radiation(tendencies["rthratenlw"][:nz],
                                                tendencies["rthratensw"][:nz])
    return {"mynn_v842": {"provider": provider, "driver": driver,
                         "arguments": fields, "constants": constants}}


def run_slot(call):
    """MPAS mpas_atmphys_driver_pbl.F:906-960,532-724; full MYNN transaction."""
    if call.slot != "pbl" or call.scheme != "bl_mynn" or str(call.backend) != "hex-cuda":
        raise ValueError("MYNN callback requires its exact registered PBL slot")
    packet = call.pools["mynn_v842"]
    provider, driver = packet["provider"], packet["driver"]
    fields = packet["arguments"]
    errors = driver.run(fields, packet["constants"], options=MynnOptions.from_namelist(call.options),
                        corrections=getattr(call, "corrections", frozenset()))
    if getattr(provider, 'wind_farm', None) is not None:
        from ..wind_farm import apply_native
        apply_native(provider, fields)
    commit_native_outputs(provider, driver, fields)
    # Native driver_pbl.F:969-975 always copies output; it never aborts on errflg.
    return errors


def _address(array):
    """First-word device (or host witness) address of an array view."""
    face = getattr(array, "__cuda_array_interface__", None) or array.__array_interface__
    return face["data"][0]


def commit_native_outputs(provider, driver, fields):
    """MPAS mpas_atmphys_driver_pbl.F:628-725,969-975; complete output transaction.

    The numerical wrapper and native-pool interface replay share this exact
    copy-out boundary. Unwritten levels and non-solve cells remain in storage.
    """
    nz = fields["kte"] - fields["kts"] + 1
    diag, tend = provider["diag_physics"], provider["tend_physics"]
    pairs = [(diag[name][:nz], fields[source][:, :nz].T)
             for name, source in (*_DIAGNOSTICS.items(), *((n, n) for n in _BUDGETS))]
    extra = ["rqsblten"]
    if fields["f_ni"]:
        extra.append("rniblten")
    if fields["f_nc"] and fields["f_nifa"] and fields["f_nwfa"]:
        extra.extend(("rncblten", "rnifablten", "rnwfablten"))
    pairs.extend((tend[name][:nz], fields[name][:, :nz].T) for name in extra)
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten"):
        pairs.append((tend[name][:nz], fields[name][:, :nz].T))
    for name, source in (("kzh", "exch_h"), ("kzm", "exch_m"), ("kzq", "exch_h")):
        pairs.append((diag[name][:nz], fields[source][:, :nz].T))
    pairs.extend(((diag["hpbl"], fields["pblh"]), (diag["kpbl"], fields["kpbl"])))
    # Fields MMM updated in place in the pool need no copy (from_mpas views).
    pairs = [(d, s) for d, s in pairs if not (_address(d) == _address(s) and d.shape == s.shape
                                               and d.strides == s.strides)]
    driver.adapter.copy_many(pairs)
    _discard_native_inputs(provider)


# BEGIN GENERATED MPAS MMM MYNN
MYNN_MODULE_CUDA = r'''
// Generated from pinned MPAS MMM; see translation receipt.
struct MynnColumn {
MynnArena& arena;
int active_column = -1;
static constexpr float j0 = .611583699e03f;
static constexpr float j1 = .444606896e02f;
static constexpr float j2 = .143177157e01f;
static constexpr float j3 = .264224321e-1f;
static constexpr float j4 = .299291081e-3f;
static constexpr float j5 = .203154182e-5f;
static constexpr float j6 = .702620698e-8f;
static constexpr float j7 = .379534310e-11f;
static constexpr float j8 = (-(.321582393e-13f));
static constexpr float k0 = .609868993e03f;
static constexpr float k1 = .499320233e02f;
static constexpr float k2 = .184672631e01f;
static constexpr float k3 = .402737184e-1f;
static constexpr float k4 = .565392987e-3f;
static constexpr float k5 = .521693933e-5f;
static constexpr float k6 = .307839583e-7f;
static constexpr float k7 = .105785160e-9f;
static constexpr float k8 = .161444444e-12f;
float cp;
float cpv;
float cice;
float cliq;
float ep_1;
float ep_2;
float grav;
float karman;
float p1000mb;
float rcp;
float r_d;
float r_v;
float rvovrd;
float svp1;
float svp2;
float svp3;
float svpt0;
float xlf;
float xlv;
float xls;
float ep_3;
float gtr;
float p608;
float t0c;
float tv0;
float xlscp;
float xlvcp;
static constexpr float tice = 240.0f;
static constexpr float tkmin = 253.0f;
static constexpr float tref = 300.0f;
static constexpr float onethird = ((1.f) / (3.f));
static constexpr float twothirds = ((2.f) / (3.f));
static constexpr float zero = 0.f;
static constexpr float b1 = 24.0f;
static constexpr float b2 = 15.0f;
static constexpr float cphh_st = 5.0f;
static constexpr float cphm_st = 5.0f;
static constexpr float cphh_unst = 16.0f;
static constexpr float cphm_unst = 16.0f;
static constexpr float pr = 0.74f;
static constexpr float g1 = 0.235f;
static constexpr float c2 = 0.729f;
static constexpr float c3 = 0.340f;
static constexpr float c4 = 0.0f;
static constexpr float c5 = 0.2f;
static constexpr float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
static constexpr float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
static constexpr float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
static constexpr float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
static constexpr float cc2 = ((1.0f) - (c2));
static constexpr float cc3 = ((1.0f) - (c3));
static constexpr float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3));
static constexpr float e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5))));
static constexpr float e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
static constexpr float e5c = ((((6.0f) * (a1))) * (a1));
static constexpr float qmin = 0.0f;
static constexpr float zmax = 1.0f;
static constexpr float sqfac = 3.0f;
static constexpr float gpw = ((5.f) / (3.f));
static constexpr float qcgmin = 1.e-8f;
static constexpr float qkemin = 1.e-12f;
static constexpr float tliq = 269.f;
static constexpr float rr2 = 0.7071068f;
static constexpr float rrp = 0.3989423f;
static constexpr float ckmod = 1.f;
static constexpr bool env_subs = false;
static constexpr bool debug_code = false;
static constexpr int idbg = 23;
// MPAS physics_mmm/mynn_shared.F90:44-65
__device__ __noinline__ float f_esat_blend(float t, float t0c, float tice) {
MynnFrame mynn_frame(arena);
float esat_blend;
float xc;
float esl;
float esi;
float chi;
// mynn_shared.F90:50
xc = mynn_max(((float)(mynn_neg(80.f))), ((float)(mpas_sub(t, t0c))));
// mynn_shared.F90:54
if (((t) >= (t0c))) {
// mynn_shared.F90:55
esat_blend = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:56
} else if (((t) <= (tice))) {
// mynn_shared.F90:57
esat_blend = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:58
} else {
// mynn_shared.F90:59
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:60
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:61
chi = mpas_div(mpas_sub(273.16f, t), 20.16f);
// mynn_shared.F90:62
esat_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), esl), mpas_mul(chi, esi));
// mynn_shared.F90:63
}
return esat_blend;
}
// MPAS physics_mmm/mynn_shared.F90:72-103
__device__ __noinline__ float f_qsat_blend(float t, float t0c, float tice, float p, MynnOptional<MynnText> waterice) {
MynnFrame mynn_frame(arena);
MynnText wrt;
float qsat_blend;
float xc;
float esl;
float esi;
float rslf;
float rsif;
float chi;
// mynn_shared.F90:80
if ((!((waterice.present())))) {
// mynn_shared.F90:81
wrt = "b";
// mynn_shared.F90:82
} else {
// mynn_shared.F90:83
wrt = waterice;
// mynn_shared.F90:84
}
// mynn_shared.F90:86
xc = mynn_max(((float)(mynn_neg(80.f))), ((float)(mpas_sub(t, t0c))));
// mynn_shared.F90:88
if (((((t) >= (t0c))) || (((wrt) == ("w"))))) {
// mynn_shared.F90:89
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:90
qsat_blend = mpas_div(mpas_mul(0.622f, esl), mpas_sub(p, esl));
// mynn_shared.F90:91
} else if (((t) <= (tice))) {
// mynn_shared.F90:92
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:93
qsat_blend = mpas_div(mpas_mul(0.622f, esi), mpas_sub(p, esi));
// mynn_shared.F90:94
} else {
// mynn_shared.F90:95
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:96
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:97
rslf = mpas_div(mpas_mul(0.622f, esl), mpas_sub(p, esl));
// mynn_shared.F90:98
rsif = mpas_div(mpas_mul(0.622f, esi), mpas_sub(p, esi));
// mynn_shared.F90:99
chi = mpas_div(mpas_sub(t0c, t), mpas_sub(t0c, tice));
// mynn_shared.F90:100
qsat_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), rslf), mpas_mul(chi, rsif));
// mynn_shared.F90:101
}
return qsat_blend;
}
// MPAS physics_mmm/mynn_shared.F90:111-129
__device__ __noinline__ float f_xl_blend(float t, float t0c, float tice, float cice, float cliq, float cpv, float xls, float xlv) {
MynnFrame mynn_frame(arena);
float xl_blend;
float xlvt;
float xlst;
float chi;
// mynn_shared.F90:118
if (((t) >= (t0c))) {
// mynn_shared.F90:119
xl_blend = mpas_add(xlv, mpas_mul(mpas_sub(cpv, cliq), mpas_sub(t, t0c)));
// mynn_shared.F90:120
} else if (((t) <= (tice))) {
// mynn_shared.F90:121
xl_blend = mpas_add(xls, mpas_mul(mpas_sub(cpv, cice), mpas_sub(t, t0c)));
// mynn_shared.F90:122
} else {
// mynn_shared.F90:123
xlvt = mpas_add(xlv, mpas_mul(mpas_sub(cpv, cliq), mpas_sub(t, t0c)));
// mynn_shared.F90:124
xlst = mpas_add(xls, mpas_mul(mpas_sub(cpv, cice), mpas_sub(t, t0c)));
// mynn_shared.F90:125
chi = mpas_div(mpas_sub(t0c, t), mpas_sub(t0c, tice));
// mynn_shared.F90:126
xl_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), xlvt), mpas_mul(chi, xlst));
// mynn_shared.F90:127
}
return xl_blend;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:473-633
__device__ __noinline__ void f_mym_initialize(int kts, int kte, float xland, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> qw, float zi, MynnArray<float, 1> theta, MynnArray<float, 1> thetav, MynnArray<float, 1> sh, MynnArray<float, 1> sm, float ust, float rmo, MynnArray<float, 1> el, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, float psig_bl, MynnArray<float, 1> cldfra_bl1d, int bl_mynn_mixlength, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, bool initialize_qke, int spp_pbl, MynnArray<float, 1> rstoch_col) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
qw.rebase({kts});
theta.rebase({kts});
thetav.rebase({kts});
sh.rebase({kts});
sm.rebase({kts});
el.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
cldfra_bl1d.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
rstoch_col.rebase({kts});
MynnArray<float, 1> ql = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdk = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdt = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdq = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dtl = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dtv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> gm = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> gh = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qkw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> vt = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> vq = mynn_array<float, 1>(arena, {kts}, {kte});
int k;
int l;
int lmax;
float phm;
float vkz;
float elq;
float elv;
float b1l;
float b2l;
float pmz = 1.f;
float phh = 1.f;
float flt = 0.f;
float fltv = 0.f;
float flq = 0.f;
float tmpq;
// bl_mynn_subroutines.F90:511
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:512
ql(k) = 0.0f;
// bl_mynn_subroutines.F90:513
vt(k) = 0.0f;
// bl_mynn_subroutines.F90:514
vq(k) = 0.0f;
// bl_mynn_subroutines.F90:515
}
// bl_mynn_subroutines.F90:518
f_mym_level2(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// bl_mynn_subroutines.F90:526
el(kts) = 0.0f;
// bl_mynn_subroutines.F90:527
if (initialize_qke) {
// bl_mynn_subroutines.F90:529
qke(kts) = mpas_mul(mpas_mul(1.5f, mynn_ipow<2>(ust)), mynn_pow(((float)(mpas_mul(b1, pmz))), ((float)(mpas_div(2.0f, 3.0f)))));
// bl_mynn_subroutines.F90:530
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:533
qke(k) = mpas_mul(qke(kts), mynn_max(((float)(mpas_div(mpas_sub(mpas_mul(ust, 700.f), zw(k)), mpas_mul(mynn_max(((float)(ust)), ((float)(0.01f))), 700.f)))), ((float)(0.01f))));
// bl_mynn_subroutines.F90:534
}
// bl_mynn_subroutines.F90:535
}
// bl_mynn_subroutines.F90:537
phm = mpas_div(mpas_mul(phh, b2), mynn_pow(((float)(mpas_mul(b1, pmz))), ((float)(mpas_div(1.0f, 3.0f)))));
// bl_mynn_subroutines.F90:538
tsq(kts) = mpas_mul(phm, mynn_ipow<2>(mpas_div(flt, ust)));
// bl_mynn_subroutines.F90:539
qsq(kts) = mpas_mul(phm, mynn_ipow<2>(mpas_div(flq, ust)));
// bl_mynn_subroutines.F90:540
cov(kts) = mpas_mul(mpas_mul(phm, mpas_div(flt, ust)), mpas_div(flq, ust));
// bl_mynn_subroutines.F90:542
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:543
vkz = mpas_mul(karman, zw(k));
// bl_mynn_subroutines.F90:544
el(k) = mpas_div(vkz, mpas_add(1.0f, mpas_div(vkz, 100.0f)));
// bl_mynn_subroutines.F90:547
tsq(k) = 0.0f;
// bl_mynn_subroutines.F90:548
qsq(k) = 0.0f;
// bl_mynn_subroutines.F90:549
cov(k) = 0.0f;
// bl_mynn_subroutines.F90:550
}
// bl_mynn_subroutines.F90:554
lmax = 5;
// bl_mynn_subroutines.F90:556
for (l=1; (1) > 0 ? l <= (lmax) : l >= (lmax); l += (1)) {
// bl_mynn_subroutines.F90:559
f_mym_length(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, vt, vq, u, v, qke, dtv, el, zi, theta, qkw, psig_bl, cldfra_bl1d, bl_mynn_mixlength, edmf_w1, edmf_a1);
// bl_mynn_subroutines.F90:572
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:573
elq = mpas_mul(el(k), qkw(k));
// bl_mynn_subroutines.F90:574
pdk(k) = mpas_mul(elq, mpas_add(mpas_mul(sm(k), gm(k)), mpas_mul(sh(k), gh(k))));
// bl_mynn_subroutines.F90:576
pdt(k) = mpas_mul(mpas_mul(elq, sh(k)), mynn_ipow<2>(dtl(k)));
// bl_mynn_subroutines.F90:577
pdq(k) = mpas_mul(mpas_mul(elq, sh(k)), mynn_ipow<2>(dqw(k)));
// bl_mynn_subroutines.F90:578
pdc(k) = mpas_mul(mpas_mul(mpas_mul(elq, sh(k)), dtl(k)), dqw(k));
// bl_mynn_subroutines.F90:579
}
// bl_mynn_subroutines.F90:582
vkz = mpas_mul(mpas_mul(karman, 0.5f), dz(kts));
// bl_mynn_subroutines.F90:583
elv = mpas_div(mpas_mul(0.5f, mpas_add(el(((kts) + (1))), el(kts))), vkz);
// bl_mynn_subroutines.F90:584
if (initialize_qke) {
// bl_mynn_subroutines.F90:586
qke(kts) = mpas_mul(mpas_mul(1.0f, mynn_ipow<2>(mynn_max(((float)(ust)), ((float)(0.02f))))), mynn_pow(((float)(mpas_mul(mpas_mul(b1, pmz), elv))), ((float)(mpas_div(2.0f, 3.0f)))));
// bl_mynn_subroutines.F90:587
}
// bl_mynn_subroutines.F90:589
phm = mpas_div(mpas_mul(phh, b2), mynn_pow(((float)(mpas_div(mpas_mul(b1, pmz), mynn_ipow<2>(elv)))), ((float)(mpas_div(1.0f, 3.0f)))));
// bl_mynn_subroutines.F90:590
tsq(kts) = mpas_mul(phm, mynn_ipow<2>(mpas_div(flt, ust)));
// bl_mynn_subroutines.F90:591
qsq(kts) = mpas_mul(phm, mynn_ipow<2>(mpas_div(flq, ust)));
// bl_mynn_subroutines.F90:592
cov(kts) = mpas_mul(mpas_mul(phm, mpas_div(flt, ust)), mpas_div(flq, ust));
// bl_mynn_subroutines.F90:594
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:595
b1l = mpas_mul(mpas_mul(b1, 0.25f), mpas_add(el(((k) + (1))), el(k)));
// bl_mynn_subroutines.F90:598
tmpq = mynn_min(((float)(mynn_max(((float)(mpas_mul(b1l, mpas_add(pdk(((k) + (1))), pdk(k))))), ((float)(qkemin))))), ((float)(125.f)));
// bl_mynn_subroutines.F90:600
if (initialize_qke) {
// bl_mynn_subroutines.F90:601
qke(k) = mynn_pow(((float)(tmpq)), ((float)(twothirds)));
// bl_mynn_subroutines.F90:602
}
// bl_mynn_subroutines.F90:604
if (((qke(k)) <= (0.0f))) {
// bl_mynn_subroutines.F90:605
b2l = 0.0f;
// bl_mynn_subroutines.F90:606
} else {
// bl_mynn_subroutines.F90:607
b2l = mpas_div(mpas_mul(b2, mpas_div(b1l, b1)), mynn_sqrt(qke(k)));
// bl_mynn_subroutines.F90:608
}
// bl_mynn_subroutines.F90:610
tsq(k) = mpas_mul(b2l, mpas_add(pdt(((k) + (1))), pdt(k)));
// bl_mynn_subroutines.F90:611
qsq(k) = mpas_mul(b2l, mpas_add(pdq(((k) + (1))), pdq(k)));
// bl_mynn_subroutines.F90:612
cov(k) = mpas_mul(b2l, mpas_add(pdc(((k) + (1))), pdc(k)));
// bl_mynn_subroutines.F90:613
}
// bl_mynn_subroutines.F90:615
}
// bl_mynn_subroutines.F90:622
if (initialize_qke) {
// bl_mynn_subroutines.F90:623
qke(kts) = mpas_mul(0.5f, mpas_add(qke(kts), qke(((kts) + (1)))));
// bl_mynn_subroutines.F90:624
qke(kte) = qke(((kte) - (1)));
// bl_mynn_subroutines.F90:625
}
// bl_mynn_subroutines.F90:626
tsq(kte) = tsq(((kte) - (1)));
// bl_mynn_subroutines.F90:627
qsq(kte) = qsq(((kte) - (1)));
// bl_mynn_subroutines.F90:628
cov(kte) = cov(((kte) - (1)));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:677-788
__device__ __noinline__ void f_mym_level2(int kts, int kte, MynnArray<float, 1> dz, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thetav, MynnArray<float, 1> qw, MynnArray<float, 1> ql, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> dtl, MynnArray<float, 1> dqw, MynnArray<float, 1> dtv, MynnArray<float, 1> gm, MynnArray<float, 1> gh, MynnArray<float, 1> sm, MynnArray<float, 1> sh) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
thetav.rebase({kts});
qw.rebase({kts});
ql.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
dtl.rebase({kts});
dqw.rebase({kts});
dtv.rebase({kts});
gm.rebase({kts});
gh.rebase({kts});
sm.rebase({kts});
sh.rebase({kts});
int k;
float rfc;
float f1;
float f2;
float rf1;
float rf2;
float smc;
float shc;
float ri1;
float ri2;
float ri3;
float ri4;
float duz;
float dtz;
float dqz;
float vtt;
float vqq;
float dtq;
float dzk;
float afk;
float abk;
float ri;
float rf;
float a2fac;
// bl_mynn_subroutines.F90:711
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:712
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(3.0f, a2), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:714
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:715
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:716
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:717
smc = mpas_div(mpas_mul(mpas_div(a1, a2), f1), f2);
// bl_mynn_subroutines.F90:718
shc = mpas_mul(mpas_mul(3.0f, a2), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:720
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:721
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:722
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:723
ri4 = mynn_ipow<2>(ri2);
// bl_mynn_subroutines.F90:725
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:726
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:727
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:728
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:729
duz = mpas_add(mynn_ipow<2>(mpas_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(mpas_sub(v(k), v(((k) - (1))))));
// bl_mynn_subroutines.F90:730
duz = mpas_div(duz, mynn_ipow<2>(dzk));
// bl_mynn_subroutines.F90:731
dtz = mpas_div(mpas_sub(thl(k), thl(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:732
dqz = mpas_div(mpas_sub(qw(k), qw(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:734
vtt = mpas_add(mpas_add(1.0f, mpas_mul(vt(k), abk)), mpas_mul(vt(((k) - (1))), afk));
// bl_mynn_subroutines.F90:735
vqq = mpas_add(mpas_add(tv0, mpas_mul(vq(k), abk)), mpas_mul(vq(((k) - (1))), afk));
// bl_mynn_subroutines.F90:736
dtq = mpas_add(mpas_mul(vtt, dtz), mpas_mul(vqq, dqz));
// bl_mynn_subroutines.F90:740
dtl(k) = dtz;
// bl_mynn_subroutines.F90:741
dqw(k) = dqz;
// bl_mynn_subroutines.F90:742
dtv(k) = dtq;
// bl_mynn_subroutines.F90:747
gm(k) = duz;
// bl_mynn_subroutines.F90:748
gh(k) = mpas_mul(mynn_neg(dtq), gtr);
// bl_mynn_subroutines.F90:751
if (k == kte && (kte-kts)%2 == 0) { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(1.0e-10f)), ((float)(duz)))); } else { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(duz)), ((float)(1.0e-10f)))); }
// bl_mynn_subroutines.F90:754
if (((ckmod) == (1))) {
// bl_mynn_subroutines.F90:755
if (k == kts + 2*((kte-kts-1)/2) + 1) { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(ri)), ((float)(0.0f))))); } else { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(0.0f)), ((float)(ri))))); }
// bl_mynn_subroutines.F90:756
} else {
// bl_mynn_subroutines.F90:757
a2fac = 1.f;
// bl_mynn_subroutines.F90:758
}
// bl_mynn_subroutines.F90:760
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:761
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(3.0f, a2), a2fac), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:763
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:764
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:765
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:766
smc = mpas_div(mpas_mul(mpas_div(a1, mpas_mul(a2, a2fac)), f1), f2);
// bl_mynn_subroutines.F90:767
shc = mpas_mul(mpas_mul(3.0f, mpas_mul(a2, a2fac)), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:769
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:770
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:771
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:772
ri4 = mynn_ipow<2>(ri2);
// bl_mynn_subroutines.F90:775
rf = mynn_min(((float)(mpas_mul(ri1, mpas_sub(mpas_add(ri, ri2), mynn_sqrt(mpas_add(mpas_sub(mynn_ipow<2>(ri), mpas_mul(ri3, ri)), ri4)))))), ((float)(rfc)));
// bl_mynn_subroutines.F90:777
sh(k) = mpas_div(mpas_mul(shc, mpas_sub(rfc, rf)), mpas_sub(1.0f, rf));
// bl_mynn_subroutines.F90:778
sm(k) = mpas_mul(mpas_div(mpas_mul(smc, mpas_sub(rf1, rf)), mpas_sub(rf2, rf)), sh(k));
// bl_mynn_subroutines.F90:779
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:808-1202
__device__ __noinline__ void f_mym_length(int kts, int kte, float xland, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, float rmo, float flt, float fltv, float flq, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> u1, MynnArray<float, 1> v1, MynnArray<float, 1> qke, MynnArray<float, 1> dtv, MynnArray<float, 1> el, float zi, MynnArray<float, 1> theta, MynnArray<float, 1> qkw, float psig_bl, MynnArray<float, 1> cldfra_bl1d, int bl_mynn_mixlength, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
u1.rebase({kts});
v1.rebase({kts});
qke.rebase({kts});
dtv.rebase({kts});
el.rebase({kts});
theta.rebase({kts});
qkw.rebase({kts});
cldfra_bl1d.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
float elt;
float vsc;
MynnArray<float, 1> qtke = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> elblmin = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> elblavg = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> thetaw = mynn_array<float, 1>(arena, {kts}, {kte});
float wt;
float wt2;
float zi2;
float h1;
float h2;
float hs;
float elblmin0;
float elblavg0;
float cldavg;
float cns;
float alp1;
float alp2;
float alp3;
float alp4;
float alp5;
float alp6;
constexpr float minzi = 300.f;
constexpr float maxdz = 750.f;
constexpr float mindz = 300.f;
constexpr float zslh = 100.f;
constexpr float csl = 2.f;
int i;
int j;
int k;
float afk;
float abk;
float zwk;
float zwk1;
float dzk;
float qdz;
float vflx;
float bv;
float tau_cloud;
float wstar;
float elb;
float els;
float elf;
float el_stab;
float el_mf;
float el_stab_mf;
float elb_mf;
float pblh_plus_ent;
float uonset;
float ugrid;
float wt_u;
float el_les;
constexpr float ctau = 1000.f;
// bl_mynn_subroutines.F90:878
switch (bl_mynn_mixlength) {
// bl_mynn_subroutines.F90:880
case 0: {
// bl_mynn_subroutines.F90:882
cns = 2.7f;
// bl_mynn_subroutines.F90:883
alp1 = 0.23f;
// bl_mynn_subroutines.F90:884
alp2 = 1.0f;
// bl_mynn_subroutines.F90:885
alp3 = 5.0f;
// bl_mynn_subroutines.F90:886
alp4 = 100.f;
// bl_mynn_subroutines.F90:887
alp5 = 0.3f;
// bl_mynn_subroutines.F90:890
zi2 = mynn_min(((float)(10000.f)), ((float)(zw(((kte) - (2))))));
// bl_mynn_subroutines.F90:891
h1 = mynn_max(((float)(mpas_mul(0.3f, zi2))), ((float)(mindz)));
// bl_mynn_subroutines.F90:892
h1 = mynn_min(((float)(h1)), ((float)(maxdz)));
// bl_mynn_subroutines.F90:893
h2 = mpas_div(h1, 2.0f);
// bl_mynn_subroutines.F90:895
qkw(kts) = mynn_sqrt(mynn_max(((float)(qke(kts))), ((float)(1.0e-10f))));
// bl_mynn_subroutines.F90:896
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:897
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:898
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:899
qkw(k) = mynn_sqrt(mynn_max(((float)(mpas_add(mpas_mul(qke(k), abk), mpas_mul(qke(((k) - (1))), afk)))), ((float)(1.0e-3f))));
// bl_mynn_subroutines.F90:900
}
// bl_mynn_subroutines.F90:902
elt = 1.0e-5f;
// bl_mynn_subroutines.F90:903
vsc = 1.0e-5f;
// bl_mynn_subroutines.F90:906
k = ((kts) + (1));
// bl_mynn_subroutines.F90:907
zwk = zw(k);
// bl_mynn_subroutines.F90:908
while (((zwk) <= (mpas_add(zi2, h1)))) {
// bl_mynn_subroutines.F90:909
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:910
qdz = mpas_mul(mynn_max(((float)(mpas_sub(qkw(k), qmin))), ((float)(0.03f))), dzk);
// bl_mynn_subroutines.F90:911
elt = mpas_add(elt, mpas_mul(qdz, zwk));
// bl_mynn_subroutines.F90:912
vsc = mpas_add(vsc, qdz);
// bl_mynn_subroutines.F90:913
k = ((k) + (1));
// bl_mynn_subroutines.F90:914
zwk = zw(k);
// bl_mynn_subroutines.F90:915
}
// bl_mynn_subroutines.F90:917
elt = mpas_div(mpas_mul(alp1, elt), vsc);
// bl_mynn_subroutines.F90:918
vflx = mpas_add(mpas_mul(mpas_add(vt(kts), 1.0f), flt), mpas_mul(mpas_add(vq(kts), tv0), flq));
// bl_mynn_subroutines.F90:919
vsc = mynn_pow(((float)(mpas_mul(mpas_mul(gtr, elt), mynn_max(((float)(vflx)), ((float)(0.0f)))))), ((float)(mpas_div(1.0f, 3.0f))));
// bl_mynn_subroutines.F90:922
el(kts) = 0.0f;
// bl_mynn_subroutines.F90:923
zwk1 = zw(((kts) + (1)));
// bl_mynn_subroutines.F90:925
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:926
zwk = zw(k);
// bl_mynn_subroutines.F90:929
if (((dtv(k)) > (0.0f))) {
// bl_mynn_subroutines.F90:930
bv = mynn_sqrt(mpas_mul(gtr, dtv(k)));
// bl_mynn_subroutines.F90:931
elb = mpas_mul(mpas_div(mpas_mul(alp2, qkw(k)), bv), mpas_add(1.0f, mpas_mul(mpas_div(alp3, alp2), mynn_sqrt(mpas_div(vsc, mpas_mul(bv, elt))))));
// bl_mynn_subroutines.F90:934
elf = mpas_div(mpas_mul(alp2, qkw(k)), bv);
// bl_mynn_subroutines.F90:936
} else {
// bl_mynn_subroutines.F90:937
elb = 1.0e10f;
// bl_mynn_subroutines.F90:938
elf = elb;
// bl_mynn_subroutines.F90:939
}
// bl_mynn_subroutines.F90:942
if (((rmo) > (0.0f))) {
// bl_mynn_subroutines.F90:943
els = mpas_div(mpas_mul(karman, zwk), mpas_add(1.0f, mpas_mul(cns, mynn_min(((float)(mpas_mul(zwk, rmo))), ((float)(zmax))))));
// bl_mynn_subroutines.F90:944
} else {
// bl_mynn_subroutines.F90:945
els = mpas_mul(mpas_mul(karman, zwk), mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(mpas_mul(alp4, zwk), rmo)))), ((float)(0.2f))));
// bl_mynn_subroutines.F90:946
}
// bl_mynn_subroutines.F90:952
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:954
el(k) = mynn_min(((float)(mpas_div(elb, mpas_add(mpas_add(mpas_div(elb, elt), mpas_div(elb, els)), 1.0f)))), ((float)(elf)));
// bl_mynn_subroutines.F90:956
}
// bl_mynn_subroutines.F90:958
break; } case 1: {
// bl_mynn_subroutines.F90:960
ugrid = mynn_sqrt(mpas_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// bl_mynn_subroutines.F90:961
uonset = 15.f;
// bl_mynn_subroutines.F90:962
wt_u = mpas_sub(1.0f, mynn_min(((float)(mpas_div(mynn_max(((float)(mpas_sub(ugrid, uonset))), ((float)(0.0f))), 30.0f))), ((float)(0.5f))));
// bl_mynn_subroutines.F90:963
cns = 2.7f;
// bl_mynn_subroutines.F90:964
alp1 = 0.23f;
// bl_mynn_subroutines.F90:965
alp2 = 0.3f;
// bl_mynn_subroutines.F90:966
alp3 = mpas_mul(2.5f, wt_u);
// bl_mynn_subroutines.F90:967
alp4 = 5.0f;
// bl_mynn_subroutines.F90:968
alp5 = 0.3f;
// bl_mynn_subroutines.F90:969
alp6 = 50.f;
// bl_mynn_subroutines.F90:972
zi2 = mynn_max(((float)(zi)), ((float)(300.f)));
// bl_mynn_subroutines.F90:973
h1 = mynn_max(((float)(mpas_mul(0.3f, zi2))), ((float)(300.f)));
// bl_mynn_subroutines.F90:974
h1 = mynn_min(((float)(h1)), ((float)(600.f)));
// bl_mynn_subroutines.F90:975
h2 = mpas_div(h1, 2.0f);
// bl_mynn_subroutines.F90:977
qtke(kts) = mynn_max(((float)(mpas_mul(0.5f, qke(kts)))), ((float)(0.01f)));
// bl_mynn_subroutines.F90:978
thetaw(kts) = theta(kts);
// bl_mynn_subroutines.F90:979
qkw(kts) = mynn_sqrt(mynn_max(((float)(qke(kts))), ((float)(1.0e-10f))));
// bl_mynn_subroutines.F90:981
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:982
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:983
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:984
qkw(k) = mynn_sqrt(mynn_max(((float)(mpas_add(mpas_mul(qke(k), abk), mpas_mul(qke(((k) - (1))), afk)))), ((float)(1.0e-3f))));
// bl_mynn_subroutines.F90:985
qtke(k) = mpas_mul(0.5f, mynn_ipow<2>(qkw(k)));
// bl_mynn_subroutines.F90:986
thetaw(k) = mpas_add(mpas_mul(theta(k), abk), mpas_mul(theta(((k) - (1))), afk));
// bl_mynn_subroutines.F90:987
}
// bl_mynn_subroutines.F90:989
elt = 1.0e-5f;
// bl_mynn_subroutines.F90:990
vsc = 1.0e-5f;
// bl_mynn_subroutines.F90:993
k = ((kts) + (1));
// bl_mynn_subroutines.F90:994
zwk = zw(k);
// bl_mynn_subroutines.F90:995
while (((zwk) <= (mpas_add(zi2, h1)))) {
// bl_mynn_subroutines.F90:996
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:997
qdz = mpas_mul(mynn_min(((float)(mynn_max(((float)(mpas_sub(qkw(k), qmin))), ((float)(0.03f))))), ((float)(30.0f))), dzk);
// bl_mynn_subroutines.F90:998
elt = mpas_add(elt, mpas_mul(qdz, zwk));
// bl_mynn_subroutines.F90:999
vsc = mpas_add(vsc, qdz);
// bl_mynn_subroutines.F90:1000
k = ((k) + (1));
// bl_mynn_subroutines.F90:1001
zwk = zw(k);
// bl_mynn_subroutines.F90:1002
}
// bl_mynn_subroutines.F90:1004
elt = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(alp1, elt), vsc))), ((float)(10.f))))), ((float)(400.f)));
// bl_mynn_subroutines.F90:1007
vflx = fltv;
// bl_mynn_subroutines.F90:1008
vsc = mynn_pow(((float)(mpas_mul(mpas_mul(gtr, elt), mynn_max(((float)(vflx)), ((float)(0.0f)))))), ((float)(onethird)));
// bl_mynn_subroutines.F90:1011
el(kts) = 0.0f;
// bl_mynn_subroutines.F90:1012
zwk1 = zw(((kts) + (1)));
// bl_mynn_subroutines.F90:1015
f_boulac_length(kts, kte, zw, dz, qtke, thetaw, elblmin, elblavg);
// bl_mynn_subroutines.F90:1017
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1018
zwk = zw(k);
// bl_mynn_subroutines.F90:1021
if (((dtv(k)) > (0.0f))) {
// bl_mynn_subroutines.F90:1022
bv = mynn_max(((float)(mynn_sqrt(mpas_mul(gtr, dtv(k))))), ((float)(0.0001f)));
// bl_mynn_subroutines.F90:1023
elb = mpas_mul(mpas_div(mynn_max(((float)(mpas_mul(alp2, qkw(k)))), ((float)(mpas_mul(mpas_mul(alp6, edmf_a1(((k) - (1)))), edmf_w1(((k) - (1))))))), bv), mpas_add(1.0f, mpas_mul(alp3, mynn_sqrt(mpas_div(vsc, mpas_mul(bv, elt))))));
// bl_mynn_subroutines.F90:1026
elb = mynn_min(((float)(elb)), ((float)(zwk)));
// bl_mynn_subroutines.F90:1027
elf = mpas_div(mpas_mul(1.0f, qkw(k)), bv);
// bl_mynn_subroutines.F90:1028
elblavg(k) = mynn_max(((float)(elblavg(k))), ((float)(mpas_div(mpas_mul(mpas_mul(alp6, edmf_a1(((k) - (1)))), edmf_w1(((k) - (1)))), bv))));
// bl_mynn_subroutines.F90:1029
} else {
// bl_mynn_subroutines.F90:1030
elb = 1.0e10f;
// bl_mynn_subroutines.F90:1031
elf = elb;
// bl_mynn_subroutines.F90:1032
}
// bl_mynn_subroutines.F90:1035
if (((rmo) > (0.0f))) {
// bl_mynn_subroutines.F90:1036
els = mpas_div(mpas_mul(karman, zwk), mpas_add(1.0f, mpas_mul(cns, mynn_min(((float)(mpas_mul(zwk, rmo))), ((float)(zmax))))));
// bl_mynn_subroutines.F90:1037
} else {
// bl_mynn_subroutines.F90:1038
els = mpas_mul(mpas_mul(karman, zwk), mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(mpas_mul(alp4, zwk), rmo)))), ((float)(0.2f))));
// bl_mynn_subroutines.F90:1039
}
// bl_mynn_subroutines.F90:1042
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1049
el(k) = mynn_sqrt(mpas_div(mynn_ipow<2>(els), mpas_add(1.f, mpas_div(mynn_ipow<2>(els), mynn_ipow<2>(elt)))));
// bl_mynn_subroutines.F90:1050
el(k) = mynn_min(((float)(el(k))), ((float)(elb)));
// bl_mynn_subroutines.F90:1051
el(k) = mynn_min(((float)(el(k))), ((float)(elf)));
// bl_mynn_subroutines.F90:1052
el(k) = mpas_add(mpas_mul(el(k), mpas_sub(1.f, wt)), mpas_mul(mpas_mul(alp5, elblavg(k)), wt));
// bl_mynn_subroutines.F90:1055
el(k) = mpas_mul(el(k), psig_bl);
// bl_mynn_subroutines.F90:1057
}
// bl_mynn_subroutines.F90:1059
break; } case 2: {
// bl_mynn_subroutines.F90:1061
uonset = mpas_add(3.5f, mpas_mul(dz(kts), 0.1f));
// bl_mynn_subroutines.F90:1062
ugrid = mynn_sqrt(mpas_add(mynn_ipow<2>(u1(kts)), mynn_ipow<2>(v1(kts))));
// bl_mynn_subroutines.F90:1063
cns = 3.5f;
// bl_mynn_subroutines.F90:1064
alp1 = 0.22f;
// bl_mynn_subroutines.F90:1065
alp2 = 0.30f;
// bl_mynn_subroutines.F90:1066
alp3 = 2.0f;
// bl_mynn_subroutines.F90:1067
alp4 = 5.0f;
// bl_mynn_subroutines.F90:1068
alp5 = alp2;
// bl_mynn_subroutines.F90:1069
alp6 = 50.0f;
// bl_mynn_subroutines.F90:1073
zi2 = mynn_max(((float)(zi)), ((float)(300.f)));
// bl_mynn_subroutines.F90:1076
h1 = mynn_max(((float)(mpas_mul(0.3f, zi2))), ((float)(300.f)));
// bl_mynn_subroutines.F90:1077
h1 = mynn_min(((float)(h1)), ((float)(600.f)));
// bl_mynn_subroutines.F90:1078
h2 = mpas_mul(h1, 0.5f);
// bl_mynn_subroutines.F90:1080
qtke(kts) = mynn_max(((float)(mpas_mul(0.5f, qke(kts)))), ((float)(0.01f)));
// bl_mynn_subroutines.F90:1081
qkw(kts) = mynn_sqrt(mynn_max(((float)(qke(kts))), ((float)(1.0e-4f))));
// bl_mynn_subroutines.F90:1083
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1084
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1085
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:1086
qkw(k) = mynn_sqrt(mynn_max(((float)(mpas_add(mpas_mul(qke(k), abk), mpas_mul(qke(((k) - (1))), afk)))), ((float)(1.0e-3f))));
// bl_mynn_subroutines.F90:1087
qtke(k) = mpas_mul(0.5f, mynn_ipow<2>(qkw(k)));
// bl_mynn_subroutines.F90:1088
}
// bl_mynn_subroutines.F90:1090
elt = 1.0e-5f;
// bl_mynn_subroutines.F90:1091
vsc = 1.0e-5f;
// bl_mynn_subroutines.F90:1094
pblh_plus_ent = mynn_max(((float)(mpas_add(zi, h1))), ((float)(100.f)));
// bl_mynn_subroutines.F90:1095
k = ((kts) + (1));
// bl_mynn_subroutines.F90:1096
zwk = zw(k);
// bl_mynn_subroutines.F90:1097
while (((zwk) <= (pblh_plus_ent))) {
// bl_mynn_subroutines.F90:1098
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1099
qdz = mpas_mul(mynn_min(((float)(mynn_max(((float)(mpas_sub(qkw(k), qmin))), ((float)(0.03f))))), ((float)(30.0f))), dzk);
// bl_mynn_subroutines.F90:1100
elt = mpas_add(elt, mpas_mul(qdz, zwk));
// bl_mynn_subroutines.F90:1101
vsc = mpas_add(vsc, qdz);
// bl_mynn_subroutines.F90:1102
k = ((k) + (1));
// bl_mynn_subroutines.F90:1103
zwk = zw(k);
// bl_mynn_subroutines.F90:1104
}
// bl_mynn_subroutines.F90:1106
elt = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(alp1, elt), vsc))), ((float)(10.f))))), ((float)(400.f)));
// bl_mynn_subroutines.F90:1109
vflx = fltv;
// bl_mynn_subroutines.F90:1110
vsc = mynn_pow(((float)(mpas_mul(mpas_mul(gtr, elt), mynn_max(((float)(vflx)), ((float)(0.0f)))))), ((float)(onethird)));
// bl_mynn_subroutines.F90:1113
el(kts) = 0.0f;
// bl_mynn_subroutines.F90:1114
zwk1 = zw(((kts) + (1)));
// bl_mynn_subroutines.F90:1116
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1117
zwk = zw(k);
// bl_mynn_subroutines.F90:1118
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1119
cldavg = mpas_mul(0.5f, mpas_add(cldfra_bl1d(((k) - (1))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:1122
if (((dtv(k)) > (0.0f))) {
// bl_mynn_subroutines.F90:1124
bv = mynn_max(((float)(mynn_sqrt(mpas_mul(gtr, dtv(k))))), ((float)(0.001f)));
// bl_mynn_subroutines.F90:1126
elb_mf = mpas_mul(mpas_div(mynn_max(((float)(mpas_mul(alp2, qkw(k)))), ((float)(mpas_mul(mpas_mul(alp6, edmf_a1(((k) - (1)))), edmf_w1(((k) - (1))))))), bv), mpas_add(1.0f, mpas_mul(alp3, mynn_sqrt(mpas_div(vsc, mpas_mul(bv, elt))))));
// bl_mynn_subroutines.F90:1129
elb = mynn_min(((float)(mpas_div(mynn_max(((float)(mpas_mul(alp5, qkw(k)))), ((float)(mpas_mul(mpas_mul(alp6, edmf_a1(k)), edmf_w1(k))))), bv))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1132
wstar = mpas_mul(1.25f, mynn_pow(((float)(mpas_mul(mpas_mul(gtr, zi), mynn_max(((float)(vflx)), ((float)(1.0e-4f)))))), ((float)(onethird))));
// bl_mynn_subroutines.F90:1133
tau_cloud = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(ctau, wstar), grav))), ((float)(30.f))))), ((float)(150.f)));
// bl_mynn_subroutines.F90:1135
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1136
tau_cloud = mpas_add(mpas_mul(tau_cloud, mpas_sub(1.f, wt)), mpas_mul(50.f, wt));
// bl_mynn_subroutines.F90:1137
elf = mynn_min(((float)(mynn_max(((float)(mpas_mul(tau_cloud, mynn_sqrt(mynn_min(((float)(qtke(k))), ((float)(40.f))))))), ((float)(mpas_div(mpas_mul(mpas_mul(alp6, edmf_a1(k)), edmf_w1(k)), bv)))))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1147
} else {
// bl_mynn_subroutines.F90:1159
wstar = mpas_mul(1.25f, mynn_pow(((float)(mpas_mul(mpas_mul(gtr, zi), mynn_max(((float)(vflx)), ((float)(1.0e-4f)))))), ((float)(onethird))));
// bl_mynn_subroutines.F90:1160
tau_cloud = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(ctau, wstar), grav))), ((float)(50.f))))), ((float)(200.f)));
// bl_mynn_subroutines.F90:1162
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1164
tau_cloud = mpas_add(mpas_mul(tau_cloud, mpas_sub(1.f, wt)), mpas_mul(mynn_max(((float)(100.f)), ((float)(mpas_mul(dzk, 0.25f)))), wt));
// bl_mynn_subroutines.F90:1166
elb = mynn_min(((float)(mpas_mul(tau_cloud, mynn_sqrt(mynn_min(((float)(qtke(k))), ((float)(40.f))))))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1168
elf = elb;
// bl_mynn_subroutines.F90:1169
elb_mf = elb;
// bl_mynn_subroutines.F90:1170
}
// bl_mynn_subroutines.F90:1171
elf = mpas_div(elf, mpas_add(1.f, mpas_div(elf, 800.f)));
// bl_mynn_subroutines.F90:1172
elb_mf = mynn_max(((float)(elb_mf)), ((float)(0.01f)));
// bl_mynn_subroutines.F90:1175
if (((rmo) > (0.0f))) {
// bl_mynn_subroutines.F90:1176
els = mpas_div(mpas_mul(karman, zwk), mpas_add(1.0f, mpas_mul(cns, mynn_min(((float)(mpas_mul(zwk, rmo))), ((float)(zmax))))));
// bl_mynn_subroutines.F90:1177
} else {
// bl_mynn_subroutines.F90:1178
els = mpas_mul(mpas_mul(karman, zwk), mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(mpas_mul(alp4, zwk), rmo)))), ((float)(0.2f))));
// bl_mynn_subroutines.F90:1179
}
// bl_mynn_subroutines.F90:1182
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1185
el(k) = mynn_sqrt(mpas_div(mynn_ipow<2>(els), mpas_add(mpas_add(1.f, mpas_div(mynn_ipow<2>(els), mynn_ipow<2>(elt))), mpas_div(mynn_ipow<2>(els), mynn_ipow<2>(elb_mf)))));
// bl_mynn_subroutines.F90:1186
el(k) = mpas_add(mpas_mul(el(k), mpas_sub(1.f, wt)), mpas_mul(elf, wt));
// bl_mynn_subroutines.F90:1189
el_les = mynn_min(((float)(mpas_div(els, mpas_add(1.f, mpas_div(els, 12.f))))), ((float)(elb_mf)));
// bl_mynn_subroutines.F90:1190
el(k) = mpas_add(mpas_mul(el(k), psig_bl), mpas_mul(mpas_sub(1.f, psig_bl), el_les));
// bl_mynn_subroutines.F90:1192
}
// bl_mynn_subroutines.F90:1194
break; } }
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:1217-1365
__device__ __noinline__ void f_boulac_length0(int k, int kts, int kte, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> qtke, MynnArray<float, 1> theta, float& lb1, float& lb2) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
qtke.rebase({kts});
theta.rebase({kts});
int izz;
int found;
float dlu;
float dld;
float dzt;
float zup;
float beta;
float zup_inf;
float bbb;
float tl;
float zdo;
float zdo_sup;
float zzz;
// bl_mynn_subroutines.F90:1249
zup = 0.f;
// bl_mynn_subroutines.F90:1250
dlu = mpas_sub(mpas_sub(zw(((kte) + (1))), zw(k)), mpas_mul(dz(k), 0.5f));
// bl_mynn_subroutines.F90:1251
zzz = 0.f;
// bl_mynn_subroutines.F90:1252
zup_inf = 0.f;
// bl_mynn_subroutines.F90:1253
beta = gtr;
// bl_mynn_subroutines.F90:1257
if (((k) < (kte))) {
// bl_mynn_subroutines.F90:1258
found = 0;
// bl_mynn_subroutines.F90:1259
izz = k;
// bl_mynn_subroutines.F90:1260
while (((found) == (0))) {
// bl_mynn_subroutines.F90:1262
if (((izz) < (kte))) {
// bl_mynn_subroutines.F90:1263
dzt = dz(izz);
// bl_mynn_subroutines.F90:1264
zup = mpas_sub(zup, mpas_mul(mpas_mul(beta, theta(k)), dzt));
// bl_mynn_subroutines.F90:1266
zup = mpas_add(zup, mpas_mul(mpas_mul(mpas_mul(beta, mpas_add(theta(((izz) + (1))), theta(izz))), dzt), 0.5f));
// bl_mynn_subroutines.F90:1267
zzz = mpas_add(zzz, dzt);
// bl_mynn_subroutines.F90:1269
if (((((qtke(k)) < (zup))) && (((qtke(k)) >= (zup_inf))))) {
// bl_mynn_subroutines.F90:1270
bbb = mpas_div(mpas_sub(theta(((izz) + (1))), theta(izz)), dzt);
// bl_mynn_subroutines.F90:1271
if (((bbb) != (0.f))) {
// bl_mynn_subroutines.F90:1273
tl = mpas_div(mpas_div(mpas_add(mpas_mul(mynn_neg(beta), mpas_sub(theta(izz), theta(k))), mynn_sqrt(mynn_max(((float)(0.f)), ((float)(mpas_add(mynn_ipow<2>(mpas_mul(beta, mpas_sub(theta(izz), theta(k)))), mpas_mul(mpas_mul(mpas_mul(2.f, bbb), beta), mpas_sub(qtke(k), zup_inf)))))))), bbb), beta);
// bl_mynn_subroutines.F90:1276
} else {
// bl_mynn_subroutines.F90:1277
if (((theta(izz)) != (theta(k)))) {
// bl_mynn_subroutines.F90:1278
tl = mpas_div(mpas_sub(qtke(k), zup_inf), mpas_mul(beta, mpas_sub(theta(izz), theta(k))));
// bl_mynn_subroutines.F90:1279
} else {
// bl_mynn_subroutines.F90:1280
tl = 0.f;
// bl_mynn_subroutines.F90:1281
}
// bl_mynn_subroutines.F90:1282
}
// bl_mynn_subroutines.F90:1283
dlu = mpas_add(mpas_sub(zzz, dzt), tl);
// bl_mynn_subroutines.F90:1285
found = 1;
// bl_mynn_subroutines.F90:1286
}
// bl_mynn_subroutines.F90:1287
zup_inf = zup;
// bl_mynn_subroutines.F90:1288
izz = ((izz) + (1));
// bl_mynn_subroutines.F90:1289
} else {
// bl_mynn_subroutines.F90:1290
found = 1;
// bl_mynn_subroutines.F90:1291
}
// bl_mynn_subroutines.F90:1293
}
// bl_mynn_subroutines.F90:1295
}
// bl_mynn_subroutines.F90:1300
zdo = 0.f;
// bl_mynn_subroutines.F90:1301
zdo_sup = 0.f;
// bl_mynn_subroutines.F90:1302
dld = zw(k);
// bl_mynn_subroutines.F90:1303
zzz = 0.f;
// bl_mynn_subroutines.F90:1306
if (((k) > (kts))) {
// bl_mynn_subroutines.F90:1308
found = 0;
// bl_mynn_subroutines.F90:1309
izz = k;
// bl_mynn_subroutines.F90:1310
while (((found) == (0))) {
// bl_mynn_subroutines.F90:1312
if (((izz) > (kts))) {
// bl_mynn_subroutines.F90:1313
dzt = dz(((izz) - (1)));
// bl_mynn_subroutines.F90:1314
zdo = mpas_add(zdo, mpas_mul(mpas_mul(beta, theta(k)), dzt));
// bl_mynn_subroutines.F90:1316
zdo = mpas_sub(zdo, mpas_mul(mpas_mul(mpas_mul(beta, mpas_add(theta(((izz) - (1))), theta(izz))), dzt), 0.5f));
// bl_mynn_subroutines.F90:1317
zzz = mpas_add(zzz, dzt);
// bl_mynn_subroutines.F90:1319
if (((((qtke(k)) < (zdo))) && (((qtke(k)) >= (zdo_sup))))) {
// bl_mynn_subroutines.F90:1320
bbb = mpas_div(mpas_sub(theta(izz), theta(((izz) - (1)))), dzt);
// bl_mynn_subroutines.F90:1321
if (((bbb) != (0.f))) {
// bl_mynn_subroutines.F90:1322
tl = mpas_div(mpas_div(mpas_add(mpas_mul(beta, mpas_sub(theta(izz), theta(k))), mynn_sqrt(mynn_max(((float)(0.f)), ((float)(mpas_add(mynn_ipow<2>(mpas_mul(beta, mpas_sub(theta(izz), theta(k)))), mpas_mul(mpas_mul(mpas_mul(2.f, bbb), beta), mpas_sub(qtke(k), zdo_sup)))))))), bbb), beta);
// bl_mynn_subroutines.F90:1325
} else {
// bl_mynn_subroutines.F90:1326
if (((theta(izz)) != (theta(k)))) {
// bl_mynn_subroutines.F90:1327
tl = mpas_div(mpas_sub(qtke(k), zdo_sup), mpas_mul(beta, mpas_sub(theta(izz), theta(k))));
// bl_mynn_subroutines.F90:1328
} else {
// bl_mynn_subroutines.F90:1329
tl = 0.f;
// bl_mynn_subroutines.F90:1330
}
// bl_mynn_subroutines.F90:1331
}
// bl_mynn_subroutines.F90:1332
dld = mpas_add(mpas_sub(zzz, dzt), tl);
// bl_mynn_subroutines.F90:1334
found = 1;
// bl_mynn_subroutines.F90:1335
}
// bl_mynn_subroutines.F90:1336
zdo_sup = zdo;
// bl_mynn_subroutines.F90:1337
izz = ((izz) - (1));
// bl_mynn_subroutines.F90:1338
} else {
// bl_mynn_subroutines.F90:1339
found = 1;
// bl_mynn_subroutines.F90:1340
}
// bl_mynn_subroutines.F90:1341
}
// bl_mynn_subroutines.F90:1343
}
// bl_mynn_subroutines.F90:1350
dld = mynn_min(((float)(dld)), ((float)(zw(((k) + (1))))));
// bl_mynn_subroutines.F90:1351
lb1 = mynn_min(((float)(dlu)), ((float)(dld)));
// bl_mynn_subroutines.F90:1353
dlu = mynn_max(((float)(0.1f)), ((float)(mynn_min(((float)(dlu)), ((float)(1000.f))))));
// bl_mynn_subroutines.F90:1354
dld = mynn_max(((float)(0.1f)), ((float)(mynn_min(((float)(dld)), ((float)(1000.f))))));
// bl_mynn_subroutines.F90:1355
lb2 = mynn_sqrt(mpas_mul(dlu, dld));
// bl_mynn_subroutines.F90:1358
if (((k) == (kte))) {
// bl_mynn_subroutines.F90:1359
lb1 = 0.f;
// bl_mynn_subroutines.F90:1360
lb2 = 0.f;
// bl_mynn_subroutines.F90:1361
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:1376-1526
__device__ __noinline__ void f_boulac_length(int kts, int kte, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> qtke, MynnArray<float, 1> theta, MynnArray<float, 1> lb1, MynnArray<float, 1> lb2) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
qtke.rebase({kts});
theta.rebase({kts});
lb1.rebase({kts});
lb2.rebase({kts});
int iz;
int izz;
int found;
MynnArray<float, 1> dlu = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dld = mynn_array<float, 1>(arena, {kts}, {kte});
constexpr float lmax = 2000.f;
float dzt;
float zup;
float beta;
float zup_inf;
float bbb;
float tl;
float zdo;
float zdo_sup;
float zzz;
// bl_mynn_subroutines.F90:1398
for (iz=kts; (1) > 0 ? iz <= (kte) : iz >= (kte); iz += (1)) {
// bl_mynn_subroutines.F90:1403
zup = 0.f;
// bl_mynn_subroutines.F90:1404
dlu(iz) = mpas_sub(mpas_sub(zw(((kte) + (1))), zw(iz)), mpas_mul(dz(iz), 0.5f));
// bl_mynn_subroutines.F90:1405
zzz = 0.f;
// bl_mynn_subroutines.F90:1406
zup_inf = 0.f;
// bl_mynn_subroutines.F90:1407
beta = gtr;
// bl_mynn_subroutines.F90:1411
if (((iz) < (kte))) {
// bl_mynn_subroutines.F90:1413
found = 0;
// bl_mynn_subroutines.F90:1414
izz = iz;
// bl_mynn_subroutines.F90:1415
while (((found) == (0))) {
// bl_mynn_subroutines.F90:1417
if (((izz) < (kte))) {
// bl_mynn_subroutines.F90:1418
dzt = dz(izz);
// bl_mynn_subroutines.F90:1419
zup = mpas_sub(zup, mpas_mul(mpas_mul(beta, theta(iz)), dzt));
// bl_mynn_subroutines.F90:1421
zup = mpas_add(zup, mpas_mul(mpas_mul(mpas_mul(beta, mpas_add(theta(((izz) + (1))), theta(izz))), dzt), 0.5f));
// bl_mynn_subroutines.F90:1422
zzz = mpas_add(zzz, dzt);
// bl_mynn_subroutines.F90:1424
if (((((qtke(iz)) < (zup))) && (((qtke(iz)) >= (zup_inf))))) {
// bl_mynn_subroutines.F90:1425
bbb = mpas_div(mpas_sub(theta(((izz) + (1))), theta(izz)), dzt);
// bl_mynn_subroutines.F90:1426
if (((bbb) != (0.f))) {
// bl_mynn_subroutines.F90:1428
tl = mpas_div(mpas_div(mpas_add(mpas_mul(mynn_neg(beta), mpas_sub(theta(izz), theta(iz))), mynn_sqrt(mynn_max(((float)(0.f)), ((float)(mpas_add(mynn_ipow<2>(mpas_mul(beta, mpas_sub(theta(izz), theta(iz)))), mpas_mul(mpas_mul(mpas_mul(2.f, bbb), beta), mpas_sub(qtke(iz), zup_inf)))))))), bbb), beta);
// bl_mynn_subroutines.F90:1431
} else {
// bl_mynn_subroutines.F90:1432
if (((theta(izz)) != (theta(iz)))) {
// bl_mynn_subroutines.F90:1433
tl = mpas_div(mpas_sub(qtke(iz), zup_inf), mpas_mul(beta, mpas_sub(theta(izz), theta(iz))));
// bl_mynn_subroutines.F90:1434
} else {
// bl_mynn_subroutines.F90:1435
tl = 0.f;
// bl_mynn_subroutines.F90:1436
}
// bl_mynn_subroutines.F90:1437
}
// bl_mynn_subroutines.F90:1438
dlu(iz) = mpas_add(mpas_sub(zzz, dzt), tl);
// bl_mynn_subroutines.F90:1440
found = 1;
// bl_mynn_subroutines.F90:1441
}
// bl_mynn_subroutines.F90:1442
zup_inf = zup;
// bl_mynn_subroutines.F90:1443
izz = ((izz) + (1));
// bl_mynn_subroutines.F90:1444
} else {
// bl_mynn_subroutines.F90:1445
found = 1;
// bl_mynn_subroutines.F90:1446
}
// bl_mynn_subroutines.F90:1448
}
// bl_mynn_subroutines.F90:1450
}
// bl_mynn_subroutines.F90:1455
zdo = 0.f;
// bl_mynn_subroutines.F90:1456
zdo_sup = 0.f;
// bl_mynn_subroutines.F90:1457
dld(iz) = zw(iz);
// bl_mynn_subroutines.F90:1458
zzz = 0.f;
// bl_mynn_subroutines.F90:1461
if (((iz) > (kts))) {
// bl_mynn_subroutines.F90:1463
found = 0;
// bl_mynn_subroutines.F90:1464
izz = iz;
// bl_mynn_subroutines.F90:1465
while (((found) == (0))) {
// bl_mynn_subroutines.F90:1467
if (((izz) > (kts))) {
// bl_mynn_subroutines.F90:1468
dzt = dz(((izz) - (1)));
// bl_mynn_subroutines.F90:1469
zdo = mpas_add(zdo, mpas_mul(mpas_mul(beta, theta(iz)), dzt));
// bl_mynn_subroutines.F90:1471
zdo = mpas_sub(zdo, mpas_mul(mpas_mul(mpas_mul(beta, mpas_add(theta(((izz) - (1))), theta(izz))), dzt), 0.5f));
// bl_mynn_subroutines.F90:1472
zzz = mpas_add(zzz, dzt);
// bl_mynn_subroutines.F90:1474
if (((((qtke(iz)) < (zdo))) && (((qtke(iz)) >= (zdo_sup))))) {
// bl_mynn_subroutines.F90:1475
bbb = mpas_div(mpas_sub(theta(izz), theta(((izz) - (1)))), dzt);
// bl_mynn_subroutines.F90:1476
if (((bbb) != (0.f))) {
// bl_mynn_subroutines.F90:1477
tl = mpas_div(mpas_div(mpas_add(mpas_mul(beta, mpas_sub(theta(izz), theta(iz))), mynn_sqrt(mynn_max(((float)(0.f)), ((float)(mpas_add(mynn_ipow<2>(mpas_mul(beta, mpas_sub(theta(izz), theta(iz)))), mpas_mul(mpas_mul(mpas_mul(2.f, bbb), beta), mpas_sub(qtke(iz), zdo_sup)))))))), bbb), beta);
// bl_mynn_subroutines.F90:1480
} else {
// bl_mynn_subroutines.F90:1481
if (((theta(izz)) != (theta(iz)))) {
// bl_mynn_subroutines.F90:1482
tl = mpas_div(mpas_sub(qtke(iz), zdo_sup), mpas_mul(beta, mpas_sub(theta(izz), theta(iz))));
// bl_mynn_subroutines.F90:1483
} else {
// bl_mynn_subroutines.F90:1484
tl = 0.f;
// bl_mynn_subroutines.F90:1485
}
// bl_mynn_subroutines.F90:1486
}
// bl_mynn_subroutines.F90:1487
dld(iz) = mpas_add(mpas_sub(zzz, dzt), tl);
// bl_mynn_subroutines.F90:1489
found = 1;
// bl_mynn_subroutines.F90:1490
}
// bl_mynn_subroutines.F90:1491
zdo_sup = zdo;
// bl_mynn_subroutines.F90:1492
izz = ((izz) - (1));
// bl_mynn_subroutines.F90:1493
} else {
// bl_mynn_subroutines.F90:1494
found = 1;
// bl_mynn_subroutines.F90:1495
}
// bl_mynn_subroutines.F90:1496
}
// bl_mynn_subroutines.F90:1498
}
// bl_mynn_subroutines.F90:1505
dld(iz) = mynn_min(((float)(dld(iz))), ((float)(zw(((iz) + (1))))));
// bl_mynn_subroutines.F90:1506
lb1(iz) = mynn_min(((float)(dlu(iz))), ((float)(dld(iz))));
// bl_mynn_subroutines.F90:1508
dlu(iz) = mynn_max(((float)(0.1f)), ((float)(mynn_min(((float)(dlu(iz))), ((float)(1000.f))))));
// bl_mynn_subroutines.F90:1509
dld(iz) = mynn_max(((float)(0.1f)), ((float)(mynn_min(((float)(dld(iz))), ((float)(1000.f))))));
// bl_mynn_subroutines.F90:1510
lb2(iz) = mynn_sqrt(mpas_mul(dlu(iz), dld(iz)));
// bl_mynn_subroutines.F90:1514
lb1(iz) = mpas_div(lb1(iz), mpas_add(1.f, mpas_div(lb1(iz), lmax)));
// bl_mynn_subroutines.F90:1515
lb2(iz) = mpas_div(lb2(iz), mpas_add(1.f, mpas_div(lb2(iz), lmax)));
// bl_mynn_subroutines.F90:1517
if (((iz) == (kte))) {
// bl_mynn_subroutines.F90:1518
lb1(kte) = lb1(((kte) - (1)));
// bl_mynn_subroutines.F90:1519
lb2(kte) = lb2(((kte) - (1)));
// bl_mynn_subroutines.F90:1520
}
// bl_mynn_subroutines.F90:1524
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:1577-2108
__device__ __noinline__ void f_mym_turbulence(int kts, int kte, float xland, float closure, MynnArray<float, 1> dz, float dx, MynnArray<float, 1> zw, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> thl, MynnArray<float, 1> thetav, MynnArray<float, 1> ql, MynnArray<float, 1> qw, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> vt, MynnArray<float, 1> vq, float rmo, float flt, float fltv, float flq, float zi, MynnArray<float, 1> theta, MynnArray<float, 1> sh, MynnArray<float, 1> sm, MynnArray<float, 1> el, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qwt1d, MynnArray<float, 1> qshear1d, MynnArray<float, 1> qbuoy1d, MynnArray<float, 1> qdiss1d, bool bl_mynn_tkebudget, float psig_bl, float psig_shcu, MynnArray<float, 1> cldfra_bl1d, int bl_mynn_mixlength, MynnArray<float, 1> edmf_w1, MynnArray<float, 1> edmf_a1, MynnArray<float, 1> tkeprodtd, int spp_pbl, MynnArray<float, 1> rstoch_col) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
u.rebase({kts});
v.rebase({kts});
thl.rebase({kts});
thetav.rebase({kts});
ql.rebase({kts});
qw.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
theta.rebase({kts});
sh.rebase({kts});
sm.rebase({kts});
el.rebase({kts});
dfm.rebase({kts});
dfh.rebase({kts});
dfq.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
pdk.rebase({kts});
pdt.rebase({kts});
pdq.rebase({kts});
pdc.rebase({kts});
qwt1d.rebase({kts});
qshear1d.rebase({kts});
qbuoy1d.rebase({kts});
qdiss1d.rebase({kts});
cldfra_bl1d.rebase({kts});
edmf_w1.rebase({kts});
edmf_a1.rebase({kts});
tkeprodtd.rebase({kts});
rstoch_col.rebase({kts});
float q3sq_old;
float dlsq1;
float qwtp_old;
float qwtp_new;
float dudz;
float dvdz;
float dtdz;
float upwp;
float vpwp;
float tpwp;
MynnArray<float, 1> qkw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dtl = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dtv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> gm = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> gh = mynn_array<float, 1>(arena, {kts}, {kte});
int k;
float e6c;
float dzk;
float afk;
float abk;
float vtt;
float vqq;
float cw25;
float clow;
float cupp;
float gamt;
float gamq;
float smd;
float gamv;
float elq;
float elh;
float cldavg;
float a2fac;
float duz;
float ri;
float auh;
float aum;
float adh;
float adm;
float aeh;
float aem;
float req;
float rsl;
float rsl2;
float gmelq;
float sm20;
float sh20;
float sm25max;
float sh25max;
float sm25min;
float sh25min;
float sm_pbl;
float sh_pbl;
float zi2;
float wt;
float slht;
float wtpr;
double q2sq;
double t2sq;
double r2sq;
double c2sq;
double elsq;
double gmel;
double ghel;
double q3sq;
double t3sq;
double r3sq;
double c3sq;
double dlsq;
double qdiv;
double e1;
double e2;
double e3;
double e4;
double mynn_enum;
double eden;
double wden;
float prnum;
float shb;
constexpr float prlimit = 5.0f;
// bl_mynn_subroutines.F90:1664
f_mym_level2(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq, dtl, dqw, dtv, gm, gh, sm, sh);
// bl_mynn_subroutines.F90:1670
f_mym_length(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, vt, vq, u, v, qke, dtv, el, zi, theta, qkw, psig_bl, cldfra_bl1d, bl_mynn_mixlength, edmf_w1, edmf_a1);
// bl_mynn_subroutines.F90:1684
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1685
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1686
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1687
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:1688
elsq = mynn_real64(mynn_ipow<2>(el(k)));
// bl_mynn_subroutines.F90:1689
q3sq = mynn_real64(mynn_ipow<2>(qkw(k)));
// bl_mynn_subroutines.F90:1690
q2sq = mynn_dmul(mynn_dmul(mynn_real64(b1), elsq), mynn_real64(mpas_add(mpas_mul(sm(k), gm(k)), mpas_mul(sh(k), gh(k)))));
// bl_mynn_subroutines.F90:1692
sh20 = mynn_max(((float)(sh(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1693
sm20 = mynn_max(((float)(sm(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1694
sh(k) = mynn_max(((float)(sh(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1697
duz = mpas_add(mynn_ipow<2>(mpas_sub(u(k), u(((k) - (1))))), mynn_ipow<2>(mpas_sub(v(k), v(((k) - (1))))));
// bl_mynn_subroutines.F90:1698
duz = mpas_div(duz, mynn_ipow<2>(dzk));
// bl_mynn_subroutines.F90:1700
ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(duz)), ((float)(1.0e-10f))));
// bl_mynn_subroutines.F90:1701
if (((ckmod) == (1))) {
// bl_mynn_subroutines.F90:1702
a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(ri)), ((float)(0.0f)))));
// bl_mynn_subroutines.F90:1703
} else {
// bl_mynn_subroutines.F90:1704
a2fac = 1.f;
// bl_mynn_subroutines.F90:1705
}
// bl_mynn_subroutines.F90:1713
prnum = mynn_min(((float)(mpas_add(0.76f, mpas_mul(4.0f, mynn_max(((float)(ri)), ((float)(0.0f))))))), ((float)(prlimit)));
// bl_mynn_subroutines.F90:1717
gmel = mynn_dmul(mynn_real64(gm(k)), elsq);
// bl_mynn_subroutines.F90:1718
ghel = mynn_dmul(mynn_real64(gh(k)), elsq);
// bl_mynn_subroutines.F90:1722
if (debug_code) {
// bl_mynn_subroutines.F90:1723
if (((((sh(k)) < (0.0f))) || (((sm(k)) < (0.0f))))) {
// bl_mynn_subroutines.F90:1724
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1725
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1726
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1727
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1728
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1729
}
// bl_mynn_subroutines.F90:1730
}
// bl_mynn_subroutines.F90:1736
dlsq = elsq;
// bl_mynn_subroutines.F90:1737
if (((mynn_ddiv(q3sq, dlsq)) < (mynn_neg(gh(k))))) { q3sq = mynn_dmul(mynn_neg(dlsq), mynn_real64(gh(k))); }
// bl_mynn_subroutines.F90:1739
if (((q3sq) < (q2sq))) {
// bl_mynn_subroutines.F90:1741
qdiv = mynn_sqrt(mynn_ddiv(q3sq, q2sq));
// bl_mynn_subroutines.F90:1758
sh(k) = mynn_real32(mynn_dmul(mynn_real64(sh(k)), qdiv));
// bl_mynn_subroutines.F90:1759
sm(k) = mynn_real32(mynn_dmul(mynn_real64(sm(k)), qdiv));
// bl_mynn_subroutines.F90:1776
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e1c), ghel), mynn_real64(a2fac)), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1777
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e2c), ghel), mynn_real64(a2fac)), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1778
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e3c), ghel), mynn_real64(mynn_ipow<2>(a2fac))), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1779
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e4c), ghel), mynn_real64(a2fac)), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1780
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, mynn_real64(e5c)), gmel), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1781
eden = mynn_max(((double)(eden)), ((double)(1.0e-20)));
// bl_mynn_subroutines.F90:1786
} else {
// bl_mynn_subroutines.F90:1792
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_real64(e1c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1793
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_real64(e2c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1794
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_real64(e3c), ghel), mynn_real64(mynn_ipow<2>(a2fac))));
// bl_mynn_subroutines.F90:1795
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_real64(e4c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1796
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(e3, mynn_real64(e5c)), gmel));
// bl_mynn_subroutines.F90:1797
eden = mynn_max(((double)(eden)), ((double)(1.0e-20)));
// bl_mynn_subroutines.F90:1799
qdiv = mynn_real64(1.0f);
// bl_mynn_subroutines.F90:1801
sm(k) = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(a1)), mynn_dsub(e3, mynn_dmul(mynn_real64(mpas_mul(3.0f, c1)), e4))), eden));
// bl_mynn_subroutines.F90:1805
sh(k) = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(mpas_mul(a2, a2fac))), mynn_dadd(e2, mynn_dmul(mynn_real64(mpas_mul(mpas_mul(3.0f, c1), e5c)), gmel))), eden));
// bl_mynn_subroutines.F90:1812
}
// bl_mynn_subroutines.F90:1815
gmelq = mynn_real32(mynn_max(((double)(mynn_ddiv(gmel, q3sq))), ((double)(1e-8))));
// bl_mynn_subroutines.F90:1816
sm25max = 4.f;
// bl_mynn_subroutines.F90:1817
sh25max = 4.f;
// bl_mynn_subroutines.F90:1818
sm25min = 0.0f;
// bl_mynn_subroutines.F90:1819
sh25min = 0.0f;
// bl_mynn_subroutines.F90:1823
if (debug_code) {
// bl_mynn_subroutines.F90:1824
if (((((((((sh(k)) < (sh25min))) || (((sm(k)) < (sm25min))))) || (((sh(k)) > (sh25max))))) || (((sm(k)) > (sm25max))))) {
// bl_mynn_subroutines.F90:1826
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1827
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1828
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1829
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1830
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1831
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1832
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1833
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1834
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1836
}
// bl_mynn_subroutines.F90:1837
}
// bl_mynn_subroutines.F90:1840
if (((sh(k)) > (sh25max))) { sh(k) = sh25max; }
// bl_mynn_subroutines.F90:1841
if (((sh(k)) < (sh25min))) { sh(k) = sh25min; }
// bl_mynn_subroutines.F90:1853
shb = mynn_max(((float)(0.002f)), ((float)(sh(k))));
// bl_mynn_subroutines.F90:1854
if (q3sq < q2sq) { sm(k) = mynn_min(((float)(sm(k))), ((float)(mpas_mul(prlimit, shb)))); } else { sm(k) = mynn_min(((float)(mpas_mul(prlimit, shb))), ((float)(sm(k)))); }
// bl_mynn_subroutines.F90:1857
if (((closure) >= (3.0f))) {
// bl_mynn_subroutines.F90:1858
t2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, mynn_real64(b2)), elsq), mynn_real64(sh(k))), mynn_real64(mynn_ipow<2>(dtl(k))));
// bl_mynn_subroutines.F90:1859
r2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, mynn_real64(b2)), elsq), mynn_real64(sh(k))), mynn_real64(mynn_ipow<2>(dqw(k))));
// bl_mynn_subroutines.F90:1860
c2sq = mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(mynn_dmul(qdiv, mynn_real64(b2)), elsq), mynn_real64(sh(k))), mynn_real64(dtl(k))), mynn_real64(dqw(k)));
// bl_mynn_subroutines.F90:1861
t3sq = mynn_real64(mynn_max(((float)(mpas_add(mpas_mul(tsq(k), abk), mpas_mul(tsq(((k) - (1))), afk)))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:1862
r3sq = mynn_real64(mynn_max(((float)(mpas_add(mpas_mul(qsq(k), abk), mpas_mul(qsq(((k) - (1))), afk)))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:1863
c3sq = mynn_real64(mpas_add(mpas_mul(cov(k), abk), mpas_mul(cov(((k) - (1))), afk)));
// bl_mynn_subroutines.F90:1866
c3sq = mynn_sign(mynn_min(((double)(mynn_abs(c3sq))), ((double)(mynn_sqrt(mynn_dmul(t3sq, r3sq))))), c3sq);
// bl_mynn_subroutines.F90:1868
vtt = mpas_add(mpas_add(1.0f, mpas_mul(vt(k), abk)), mpas_mul(vt(((k) - (1))), afk));
// bl_mynn_subroutines.F90:1869
vqq = mpas_add(mpas_add(tv0, mpas_mul(vq(k), abk)), mpas_mul(vq(((k) - (1))), afk));
// bl_mynn_subroutines.F90:1871
t2sq = mynn_dadd(mynn_dmul(mynn_real64(vtt), t2sq), mynn_dmul(mynn_real64(vqq), c2sq));
// bl_mynn_subroutines.F90:1872
r2sq = mynn_dadd(mynn_dmul(mynn_real64(vtt), c2sq), mynn_dmul(mynn_real64(vqq), r2sq));
// bl_mynn_subroutines.F90:1873
c2sq = mynn_max(((double)(mynn_dadd(mynn_dmul(mynn_real64(vtt), t2sq), mynn_dmul(mynn_real64(vqq), r2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1874
t3sq = mynn_dadd(mynn_dmul(mynn_real64(vtt), t3sq), mynn_dmul(mynn_real64(vqq), c3sq));
// bl_mynn_subroutines.F90:1875
r3sq = mynn_dadd(mynn_dmul(mynn_real64(vtt), c3sq), mynn_dmul(mynn_real64(vqq), r3sq));
// bl_mynn_subroutines.F90:1876
c3sq = mynn_max(((double)(mynn_dadd(mynn_dmul(mynn_real64(vtt), t3sq), mynn_dmul(mynn_real64(vqq), r3sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1878
cw25 = mynn_real32(mynn_ddiv(mynn_dmul(e1, mynn_dadd(e2, mynn_dmul(mynn_dmul(mynn_real64(mpas_mul(mpas_mul(3.0f, c1), e5c)), gmel), mynn_ipow<2>(qdiv)))), mynn_dmul(mynn_real64(3.0f), eden)));
// bl_mynn_subroutines.F90:1881
dlsq = elsq;
// bl_mynn_subroutines.F90:1882
if (((mynn_ddiv(q3sq, dlsq)) < (mynn_neg(gh(k))))) { q3sq = mynn_dmul(mynn_neg(dlsq), mynn_real64(gh(k))); }
// bl_mynn_subroutines.F90:1887
auh = mpas_mul(mpas_mul(mpas_mul(mpas_mul(27.f, a1), mynn_ipow<2>(mpas_mul(a2, a2fac))), b2), mynn_ipow<2>(gtr));
// bl_mynn_subroutines.F90:1888
aum = mpas_mul(mpas_mul(mpas_mul(mpas_mul(mpas_mul(54.f, mynn_ipow<2>(a1)), mpas_mul(a2, a2fac)), b2), c1), gtr);
// bl_mynn_subroutines.F90:1889
adh = mpas_mul(mpas_mul(mpas_mul(mpas_mul(9.f, a1), mynn_ipow<2>(mpas_mul(a2, a2fac))), mpas_add(mpas_mul(12.f, a1), mpas_mul(3.f, b2))), mynn_ipow<2>(gtr));
// bl_mynn_subroutines.F90:1890
adm = mpas_mul(mpas_mul(mpas_mul(mpas_mul(18.f, mynn_ipow<2>(a1)), mpas_mul(a2, a2fac)), mpas_sub(b2, mpas_mul(3.f, mpas_mul(a2, a2fac)))), gtr);
// bl_mynn_subroutines.F90:1892
aeh = mpas_mul(mpas_add(mpas_mul(mpas_mul(mpas_mul(9.f, a1), mynn_ipow<2>(mpas_mul(a2, a2fac))), b1), mpas_mul(mpas_mul(mpas_mul(9.f, a1), mynn_ipow<2>(mpas_mul(a2, a2fac))), mpas_add(mpas_mul(12.f, a1), mpas_mul(3.f, b2)))), gtr);
// bl_mynn_subroutines.F90:1894
aem = mpas_add(mpas_mul(mpas_mul(mpas_mul(mpas_mul(3.f, a1), mpas_mul(a2, a2fac)), b1), mpas_add(mpas_add(mpas_mul(3.f, mpas_mul(a2, a2fac)), mpas_mul(mpas_mul(3.f, b2), c1)), mpas_sub(mpas_mul(mpas_mul(18.f, a1), c1), b2))), mpas_mul(mpas_mul(mpas_mul(18.f, mynn_ipow<2>(a1)), mpas_mul(a2, a2fac)), mpas_sub(b2, mpas_mul(3.f, mpas_mul(a2, a2fac)))));
// bl_mynn_subroutines.F90:1898
req = mpas_div(mynn_neg(aeh), aem);
// bl_mynn_subroutines.F90:1899
rsl = mpas_div(mpas_add(auh, mpas_mul(aum, req)), mpas_add(mpas_mul(3.f, adh), mpas_mul(mpas_mul(3.f, adm), req)));
// bl_mynn_subroutines.F90:1901
rsl = .12f;
// bl_mynn_subroutines.F90:1902
rsl2 = mpas_sub(1.0f, mpas_mul(2.f, rsl));
// bl_mynn_subroutines.F90:1914
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e2c), ghel), mynn_real64(a2fac)), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1915
e3 = mynn_dadd(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e3c), ghel), mynn_real64(mynn_ipow<2>(a2fac))), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1916
e4 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e4c), ghel), mynn_real64(a2fac)), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1917
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, mynn_real64(e5c)), gmel), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1922
wden = mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(mynn_real64(mpas_mul(cc3, mynn_ipow<2>(gtr))), mynn_ipow<2>(dlsq)), elsq), mynn_ipow<2>(qdiv)), mynn_dsub(mynn_dmul(mynn_dmul(e2, mynn_real64(e4c)), mynn_real64(a2fac)), mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(mpas_mul(e3c, e5c)), gmel), mynn_real64(mynn_ipow<2>(a2fac))), mynn_ipow<2>(qdiv))));
// bl_mynn_subroutines.F90:1925
if (((wden) != (0.0f))) {
// bl_mynn_subroutines.F90:1927
clow = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(mpas_sub(0.12f, cw25))), eden), wden));
// bl_mynn_subroutines.F90:1928
cupp = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(mpas_sub(0.76f, cw25))), eden), wden));
// bl_mynn_subroutines.F90:1932
if (((wden) > (0.0f))) {
// bl_mynn_subroutines.F90:1933
c3sq = mynn_min(((double)(mynn_max(((double)(c3sq)), ((double)(mynn_dadd(c2sq, mynn_real64(clow))))))), ((double)(mynn_dadd(c2sq, mynn_real64(cupp)))));
// bl_mynn_subroutines.F90:1934
} else {
// bl_mynn_subroutines.F90:1935
c3sq = mynn_max(((double)(mynn_min(((double)(c3sq)), ((double)(mynn_dadd(c2sq, mynn_real64(clow))))))), ((double)(mynn_dadd(c2sq, mynn_real64(cupp)))));
// bl_mynn_subroutines.F90:1936
}
// bl_mynn_subroutines.F90:1937
}
// bl_mynn_subroutines.F90:1939
e1 = mynn_dadd(e2, mynn_dmul(mynn_dmul(mynn_real64(e5c), gmel), mynn_ipow<2>(qdiv)));
// bl_mynn_subroutines.F90:1940
eden = mynn_max(((double)(eden)), ((double)(1.0e-20)));
// bl_mynn_subroutines.F90:1945
e6c = mynn_real32(mynn_ddiv(mynn_dmul(mynn_real64(mpas_mul(mpas_mul(mpas_mul(3.0f, mpas_mul(a2, a2fac)), cc3), gtr)), dlsq), elsq));
// bl_mynn_subroutines.F90:1950
if (((t2sq) >= (0.0f))) {
// bl_mynn_subroutines.F90:1951
mynn_enum = mynn_max(((double)(mynn_dmul(mynn_dmul(qdiv, mynn_real64(e6c)), mynn_dsub(t3sq, t2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1952
} else {
// bl_mynn_subroutines.F90:1953
mynn_enum = mynn_min(((double)(mynn_dmul(mynn_dmul(qdiv, mynn_real64(e6c)), mynn_dsub(t3sq, t2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1954
}
// bl_mynn_subroutines.F90:1955
gamt = mynn_real32(mynn_ddiv(mynn_dmul(mynn_neg(e1), mynn_enum), eden));
// bl_mynn_subroutines.F90:1960
if (((r2sq) >= (0.0f))) {
// bl_mynn_subroutines.F90:1961
mynn_enum = mynn_max(((double)(mynn_dmul(mynn_dmul(qdiv, mynn_real64(e6c)), mynn_dsub(r3sq, r2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1962
} else {
// bl_mynn_subroutines.F90:1963
mynn_enum = mynn_min(((double)(mynn_dmul(mynn_dmul(qdiv, mynn_real64(e6c)), mynn_dsub(r3sq, r2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1964
}
// bl_mynn_subroutines.F90:1965
gamq = mynn_real32(mynn_ddiv(mynn_dmul(mynn_neg(e1), mynn_enum), eden));
// bl_mynn_subroutines.F90:1970
mynn_enum = mynn_max(((double)(mynn_dmul(mynn_dmul(qdiv, mynn_real64(e6c)), mynn_dsub(c3sq, c2sq)))), ((double)(0.0e0)));
// bl_mynn_subroutines.F90:1974
smd = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(mynn_dmul(mynn_ddiv(mynn_dmul(mynn_dmul(dlsq, mynn_enum), mynn_real64(gtr)), eden), mynn_ipow<2>(qdiv)), mynn_real64(mpas_add(mpas_mul(e3c, mynn_ipow<2>(a2fac)), mpas_mul(e4c, a2fac)))), mynn_real64(a1)), mynn_real64(mpas_mul(a2, a2fac))));
// bl_mynn_subroutines.F90:1977
gamv = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(e1, mynn_enum), mynn_real64(gtr)), eden));
// bl_mynn_subroutines.F90:1978
sm(k) = mpas_add(sm(k), smd);
// bl_mynn_subroutines.F90:1982
qdiv = mynn_real64(1.0f);
// bl_mynn_subroutines.F90:1985
if (debug_code) {
// bl_mynn_subroutines.F90:1986
if (((((((((sh(k)) < (mynn_neg(0.3f)))) || (((sm(k)) < (mynn_neg(0.3f)))))) || (((qke(k)) < (mynn_neg(0.1f)))))) || (((mynn_abs(smd)) > (2.0f))))) {
// bl_mynn_subroutines.F90:1988
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1989
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1990
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1991
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1992
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:1993
}
// bl_mynn_subroutines.F90:1994
}
// bl_mynn_subroutines.F90:1998
} else {
// bl_mynn_subroutines.F90:2000
gamt = 0.0f;
// bl_mynn_subroutines.F90:2001
gamq = 0.0f;
// bl_mynn_subroutines.F90:2002
gamv = 0.0f;
// bl_mynn_subroutines.F90:2003
}
// bl_mynn_subroutines.F90:2007
cldavg = mpas_mul(0.5f, mpas_add(cldfra_bl1d(((k) - (1))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:2008
if (((((edmf_a1(k)) > (0.001f))) || (((cldavg) > (0.02f))))) {
// bl_mynn_subroutines.F90:2010
sm(k) = mynn_max(((float)(sm(k))), ((float)(mpas_mul(0.03f, mynn_min(((float)(mpas_mul(mpas_mul(10.f, edmf_a1(k)), edmf_w1(k)))), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2011
sh(k) = mynn_max(((float)(sh(k))), ((float)(mpas_mul(0.03f, mynn_min(((float)(mpas_mul(mpas_mul(10.f, edmf_a1(k)), edmf_w1(k)))), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2013
sm(k) = mynn_max(((float)(sm(k))), ((float)(mpas_mul(0.05f, mynn_min(((float)(cldavg)), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2014
sh(k) = mynn_max(((float)(sh(k))), ((float)(mpas_mul(0.05f, mynn_min(((float)(cldavg)), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2015
}
// bl_mynn_subroutines.F90:2017
elq = mpas_mul(el(k), qkw(k));
// bl_mynn_subroutines.F90:2018
elh = mynn_real32(mynn_dmul(mynn_real64(elq), qdiv));
// bl_mynn_subroutines.F90:2022
pdk(k) = mpas_add(mpas_mul(elq, mpas_add(mpas_add(mpas_mul(sm(k), gm(k)), mpas_mul(sh(k), gh(k))), gamv)), mpas_mul(0.5f, tkeprodtd(k)));
// bl_mynn_subroutines.F90:2025
pdt(k) = mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dtl(k)), gamt)), dtl(k));
// bl_mynn_subroutines.F90:2026
pdq(k) = mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dqw(k)), gamq)), dqw(k));
// bl_mynn_subroutines.F90:2027
pdc(k) = mpas_add(mpas_mul(mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dtl(k)), gamt)), dqw(k)), 0.5f), mpas_mul(mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dqw(k)), gamq)), dtl(k)), 0.5f));
// bl_mynn_subroutines.F90:2032
tcd(k) = mpas_mul(elq, gamt);
// bl_mynn_subroutines.F90:2033
qcd(k) = mpas_mul(elq, gamq);
// bl_mynn_subroutines.F90:2036
dfm(k) = mpas_div(mpas_mul(elq, sm(k)), dzk);
// bl_mynn_subroutines.F90:2037
dfh(k) = mpas_div(mpas_mul(elq, sh(k)), dzk);
// bl_mynn_subroutines.F90:2041
dfq(k) = dfm(k);
// bl_mynn_subroutines.F90:2044
if (bl_mynn_tkebudget) {
// bl_mynn_subroutines.F90:2060
qshear1d(k) = mpas_mul(mpas_mul(elq, sm(k)), gm(k));
// bl_mynn_subroutines.F90:2068
qbuoy1d(k) = mpas_add(mpas_mul(elq, mpas_add(mpas_mul(sh(k), gh(k)), gamv)), mpas_mul(0.5f, tkeprodtd(k)));
// bl_mynn_subroutines.F90:2074
}
// bl_mynn_subroutines.F90:2076
}
// bl_mynn_subroutines.F90:2079
dfm(kts) = 0.0f;
// bl_mynn_subroutines.F90:2080
dfh(kts) = 0.0f;
// bl_mynn_subroutines.F90:2081
dfq(kts) = 0.0f;
// bl_mynn_subroutines.F90:2082
tcd(kts) = 0.0f;
// bl_mynn_subroutines.F90:2083
qcd(kts) = 0.0f;
// bl_mynn_subroutines.F90:2085
tcd(kte) = 0.0f;
// bl_mynn_subroutines.F90:2086
qcd(kte) = 0.0f;
// bl_mynn_subroutines.F90:2089
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2090
dzk = dz(k);
// bl_mynn_subroutines.F90:2091
tcd(k) = mpas_div(mpas_sub(tcd(((k) + (1))), tcd(k)), dzk);
// bl_mynn_subroutines.F90:2092
qcd(k) = mpas_div(mpas_sub(qcd(((k) + (1))), qcd(k)), dzk);
// bl_mynn_subroutines.F90:2093
}
// bl_mynn_subroutines.F90:2095
if (((spp_pbl) == (1))) {
// bl_mynn_subroutines.F90:2096
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2097
dfm(k) = mpas_add(dfm(k), mpas_mul(mpas_mul(mpas_mul(dfm(k), rstoch_col(k)), 1.5f), mynn_max(((float)(mynn_exp(mpas_div(mynn_neg(mynn_max(((float)(mpas_sub(zw(k), 8000.f))), ((float)(0.0f)))), 2000.f)))), ((float)(0.001f)))));
// bl_mynn_subroutines.F90:2098
dfh(k) = mpas_add(dfh(k), mpas_mul(mpas_mul(mpas_mul(dfh(k), rstoch_col(k)), 1.5f), mynn_max(((float)(mynn_exp(mpas_div(mynn_neg(mynn_max(((float)(mpas_sub(zw(k), 8000.f))), ((float)(0.0f)))), 2000.f)))), ((float)(0.001f)))));
// bl_mynn_subroutines.F90:2099
}
// bl_mynn_subroutines.F90:2100
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:2156-2526
__device__ __noinline__ void f_mym_predict(int kts, int kte, float closure, float delt, MynnArray<float, 1> dz, float ust, float flt, float flq, float pmz, float phh, MynnArray<float, 1> el, MynnArray<float, 1> dfq, MynnArray<float, 1> rho, MynnArray<float, 1> pdk, MynnArray<float, 1> pdt, MynnArray<float, 1> pdq, MynnArray<float, 1> pdc, MynnArray<float, 1> qke, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> s_aw, MynnArray<float, 1> s_awqke, bool bl_mynn_edmf_tke, MynnArray<float, 1> qwt1d, MynnArray<float, 1> qdiss1d, bool bl_mynn_tkebudget) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
el.rebase({kts});
dfq.rebase({kts});
rho.rebase({kts});
pdk.rebase({kts});
pdt.rebase({kts});
pdq.rebase({kts});
pdc.rebase({kts});
qke.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
s_aw.rebase({kts});
s_awqke.rebase({kts});
qwt1d.rebase({kts});
qdiss1d.rebase({kts});
MynnArray<float, 1> tke_up = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dzinv = mynn_array<float, 1>(arena, {kts}, {kte});
int k;
MynnArray<float, 1> qkw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> bp = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rp = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> df3q = mynn_array<float, 1>(arena, {kts}, {kte});
float vkz;
float pdk1;
float phm;
float pdt1;
float pdq1;
float pdc1;
float b1l;
float b2l;
float onoff;
MynnArray<float, 1> dtz = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoinv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kqdz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kmdz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
// bl_mynn_subroutines.F90:2200
if ((!(bl_mynn_edmf_tke))) {
// bl_mynn_subroutines.F90:2201
onoff = 0.0f;
// bl_mynn_subroutines.F90:2202
} else {
// bl_mynn_subroutines.F90:2203
onoff = 1.0f;
// bl_mynn_subroutines.F90:2204
}
// bl_mynn_subroutines.F90:2207
vkz = mpas_mul(mpas_mul(karman, 0.5f), dz(kts));
// bl_mynn_subroutines.F90:2211
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2213
qkw(k) = mynn_sqrt(mynn_max(((float)(qke(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2214
df3q(k) = mpas_mul(sqfac, dfq(k));
// bl_mynn_subroutines.F90:2215
dtz(k) = mpas_div(delt, dz(k));
// bl_mynn_subroutines.F90:2216
}
// bl_mynn_subroutines.F90:2221
rhoz(kts) = rho(kts);
// bl_mynn_subroutines.F90:2222
rhoinv(kts) = mpas_div(1.f, rho(kts));
// bl_mynn_subroutines.F90:2223
kqdz(kts) = mpas_mul(rhoz(kts), df3q(kts));
// bl_mynn_subroutines.F90:2224
kmdz(kts) = mpas_mul(rhoz(kts), dfq(kts));
// bl_mynn_subroutines.F90:2225
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2226
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) - (1)))), mpas_mul(rho(((k) - (1))), dz(k))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:2227
rhoz(k) = mynn_max(((float)(rhoz(k))), ((float)(1e-4f)));
// bl_mynn_subroutines.F90:2228
rhoinv(k) = mpas_div(1.f, mynn_max(((float)(rho(k))), ((float)(1e-4f))));
// bl_mynn_subroutines.F90:2229
kqdz(k) = mpas_mul(rhoz(k), df3q(k));
// bl_mynn_subroutines.F90:2230
kmdz(k) = mpas_mul(rhoz(k), dfq(k));
// bl_mynn_subroutines.F90:2231
}
// bl_mynn_subroutines.F90:2232
rhoz(((kte) + (1))) = rhoz(kte);
// bl_mynn_subroutines.F90:2233
kqdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), df3q(kte));
// bl_mynn_subroutines.F90:2234
kmdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfq(kte));
// bl_mynn_subroutines.F90:2237
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2238
kqdz(k) = mynn_max(((float)(kqdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:2239
kqdz(k) = mynn_max(((float)(kqdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:2240
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:2241
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:2242
}
// bl_mynn_subroutines.F90:2245
pdk1 = mpas_div(mpas_mul(mpas_mul(2.0f, mynn_ipow<3>(ust)), pmz), vkz);
// bl_mynn_subroutines.F90:2246
phm = mpas_div(mpas_mul(mpas_div(2.0f, ust), phh), vkz);
// bl_mynn_subroutines.F90:2247
pdt1 = mpas_mul(phm, mynn_ipow<2>(flt));
// bl_mynn_subroutines.F90:2248
pdq1 = mpas_mul(phm, mynn_ipow<2>(flq));
// bl_mynn_subroutines.F90:2249
pdc1 = mpas_mul(mpas_mul(phm, flt), flq);
// bl_mynn_subroutines.F90:2252
pdk(kts) = mpas_sub(pdk1, pdk(((kts) + (1))));
// bl_mynn_subroutines.F90:2257
pdt(kts) = pdt(((kts) + (1)));
// bl_mynn_subroutines.F90:2258
pdq(kts) = pdq(((kts) + (1)));
// bl_mynn_subroutines.F90:2259
pdc(kts) = pdc(((kts) + (1)));
// bl_mynn_subroutines.F90:2263
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2264
b1l = mpas_mul(mpas_mul(b1, 0.5f), mpas_add(el(((k) + (1))), el(k)));
// bl_mynn_subroutines.F90:2265
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b1l);
// bl_mynn_subroutines.F90:2266
rp(k) = mpas_add(pdk(((k) + (1))), pdk(k));
// bl_mynn_subroutines.F90:2267
}
// bl_mynn_subroutines.F90:2275
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2287
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kqdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff));
// bl_mynn_subroutines.F90:2289
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2292
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kqdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:2294
d(k) = mpas_add(mpas_add(mpas_mul(rp(k), delt), qke(k)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqke(k), s_awqke(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:2296
}
// bl_mynn_subroutines.F90:2311
a(kte) = 0.f;
// bl_mynn_subroutines.F90:2312
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2313
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2314
d(kte) = qke(kte);
// bl_mynn_subroutines.F90:2317
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:2319
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2321
qke(k) = mynn_max(((float)(x(k))), ((float)(1.e-4f)));
// bl_mynn_subroutines.F90:2322
qke(k) = mynn_min(((float)(qke(k))), ((float)(150.f)));
// bl_mynn_subroutines.F90:2323
}
// bl_mynn_subroutines.F90:2327
if (bl_mynn_tkebudget) {
// bl_mynn_subroutines.F90:2329
for (int mynn_element=0; mynn_element < (tke_up).size(); ++mynn_element) { tke_up.linear(mynn_element) = mpas_mul(0.5f, qke.linear(mynn_element)); }
// bl_mynn_subroutines.F90:2330
for (int mynn_element=0; mynn_element < (dzinv).size(); ++mynn_element) { dzinv.linear(mynn_element) = mpas_div(1.f, dz.linear(mynn_element)); }
// bl_mynn_subroutines.F90:2331
k = kts;
// bl_mynn_subroutines.F90:2332
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_sub(mpas_mul(kqdz(((k) + (1))), mpas_sub(tke_up(((k) + (1))), tke_up(k))), mpas_mul(kqdz(k), tke_up(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_add(mpas_mul(s_aw(((k) + (1))), tke_up(((k) + (1)))), mpas_mul(mpas_sub(s_aw(((k) + (1))), s_aw(k)), tke_up(k))), mpas_sub(s_awqke(k), s_awqke(((k) + (1)))))), onoff)));
// bl_mynn_subroutines.F90:2337
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2338
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_sub(mpas_mul(kqdz(((k) + (1))), mpas_sub(tke_up(((k) + (1))), tke_up(k))), mpas_mul(kqdz(k), mpas_sub(tke_up(k), tke_up(((k) - (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_sub(mpas_add(mpas_mul(s_aw(((k) + (1))), tke_up(((k) + (1)))), mpas_mul(mpas_sub(s_aw(((k) + (1))), s_aw(k)), tke_up(k))), mpas_mul(s_aw(k), tke_up(((k) - (1))))), mpas_sub(s_awqke(k), s_awqke(((k) + (1)))))), onoff)));
// bl_mynn_subroutines.F90:2344
}
// bl_mynn_subroutines.F90:2345
k = kte;
// bl_mynn_subroutines.F90:2346
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_mul(mynn_neg(kqdz(k)), mpas_sub(tke_up(k), tke_up(((k) - (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_sub(mpas_mul(mynn_neg(s_aw(k)), tke_up(k)), mpas_mul(s_aw(k), tke_up(((k) - (1))))), s_awqke(k))), onoff)));
// bl_mynn_subroutines.F90:2349
for (int mynn_element=0; mynn_element < (qdiss1d).size(); ++mynn_element) { qdiss1d.linear(mynn_element) = mpas_mul(bp.linear(mynn_element), tke_up.linear(mynn_element)); }
// bl_mynn_subroutines.F90:2350
}
// bl_mynn_subroutines.F90:2353
if (((closure) > (2.5f))) {
// bl_mynn_subroutines.F90:2356
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2357
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(((k) + (1))), el(k)));
// bl_mynn_subroutines.F90:2358
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2359
rp(k) = mpas_add(pdq(((k) + (1))), pdq(k));
// bl_mynn_subroutines.F90:2360
}
// bl_mynn_subroutines.F90:2369
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2370
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2371
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2372
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2373
d(k) = mpas_add(mpas_mul(rp(k), delt), qsq(k));
// bl_mynn_subroutines.F90:2374
}
// bl_mynn_subroutines.F90:2376
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2377
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2378
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2379
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2382
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:2384
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2386
qsq(k) = mynn_max(((float)(x(k))), ((float)(1e-17f)));
// bl_mynn_subroutines.F90:2387
}
// bl_mynn_subroutines.F90:2388
} else {
// bl_mynn_subroutines.F90:2390
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2391
if (((qkw(k)) <= (0.0f))) {
// bl_mynn_subroutines.F90:2392
b2l = 0.0f;
// bl_mynn_subroutines.F90:2393
} else {
// bl_mynn_subroutines.F90:2394
b2l = mpas_div(mpas_mul(mpas_mul(b2, 0.25f), mpas_add(el(((k) + (1))), el(k))), qkw(k));
// bl_mynn_subroutines.F90:2395
}
// bl_mynn_subroutines.F90:2396
qsq(k) = mpas_mul(b2l, mpas_add(pdq(((k) + (1))), pdq(k)));
// bl_mynn_subroutines.F90:2397
}
// bl_mynn_subroutines.F90:2398
qsq(kte) = qsq(((kte) - (1)));
// bl_mynn_subroutines.F90:2399
}
// bl_mynn_subroutines.F90:2402
if (((closure) >= (3.0f))) {
// bl_mynn_subroutines.F90:2408
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2409
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(k), el(((k) + (1)))));
// bl_mynn_subroutines.F90:2410
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2411
rp(k) = mpas_add(pdt(k), pdt(((k) + (1))));
// bl_mynn_subroutines.F90:2412
}
// bl_mynn_subroutines.F90:2422
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2428
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2429
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2430
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2431
d(k) = mpas_add(mpas_mul(rp(k), delt), tsq(k));
// bl_mynn_subroutines.F90:2432
}
// bl_mynn_subroutines.F90:2441
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2442
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2443
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2444
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2447
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:2449
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2451
tsq(k) = x(k);
// bl_mynn_subroutines.F90:2452
}
// bl_mynn_subroutines.F90:2456
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2457
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(k), el(((k) + (1)))));
// bl_mynn_subroutines.F90:2458
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2459
rp(k) = mpas_add(pdc(k), pdc(((k) + (1))));
// bl_mynn_subroutines.F90:2460
}
// bl_mynn_subroutines.F90:2470
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2476
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2477
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2478
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2479
d(k) = mpas_add(mpas_mul(rp(k), delt), cov(k));
// bl_mynn_subroutines.F90:2480
}
// bl_mynn_subroutines.F90:2489
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2490
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2491
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2492
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2495
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:2497
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2499
cov(k) = x(k);
// bl_mynn_subroutines.F90:2500
}
// bl_mynn_subroutines.F90:2502
} else {
// bl_mynn_subroutines.F90:2505
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2506
if (((qkw(k)) <= (0.0f))) {
// bl_mynn_subroutines.F90:2507
b2l = 0.0f;
// bl_mynn_subroutines.F90:2508
} else {
// bl_mynn_subroutines.F90:2509
b2l = mpas_div(mpas_mul(mpas_mul(b2, 0.25f), mpas_add(el(((k) + (1))), el(k))), qkw(k));
// bl_mynn_subroutines.F90:2510
}
// bl_mynn_subroutines.F90:2512
tsq(k) = mpas_mul(b2l, mpas_add(pdt(((k) + (1))), pdt(k)));
// bl_mynn_subroutines.F90:2513
cov(k) = mpas_mul(b2l, mpas_add(pdc(((k) + (1))), pdc(k)));
// bl_mynn_subroutines.F90:2514
}
// bl_mynn_subroutines.F90:2516
tsq(kte) = tsq(((kte) - (1)));
// bl_mynn_subroutines.F90:2517
cov(kte) = cov(((kte) - (1)));
// bl_mynn_subroutines.F90:2519
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:2562-2981
__device__ __noinline__ void f_mym_condensation(int kts, int kte, float dx, MynnArray<float, 1> dz, MynnArray<float, 1> zw, float xland, MynnArray<float, 1> thl, MynnArray<float, 1> qw, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> sh, MynnArray<float, 1> el, int bl_mynn_cloudpdf, MynnArray<float, 1> qc_bl1d, MynnArray<float, 1> qi_bl1d, MynnArray<float, 1> cldfra_bl1d, float pblh1, float hfx1, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> th, MynnArray<float, 1> sgm, float rmo, int spp_pbl, MynnArray<float, 1> rstoch_col) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
zw.rebase({kts});
thl.rebase({kts});
qw.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qi.rebase({kts});
qs.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
sh.rebase({kts});
el.rebase({kts});
qc_bl1d.rebase({kts});
qi_bl1d.rebase({kts});
cldfra_bl1d.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
th.rebase({kts});
sgm.rebase({kts});
rstoch_col.rebase({kts});
MynnArray<float, 1> alp = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> bet = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> ql = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> q1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rh = mynn_array<float, 1>(arena, {kts}, {kte});
double t3sq;
double r3sq;
double c3sq;
float qsl;
float esat;
float qsat;
float dqsl;
float cld0;
float q1k;
float qlk;
float eq1;
float qll;
float q2p;
float pt;
float rac;
float qt;
float t;
float xl;
float rsl;
float cpm;
float fng;
float qww;
float alpha;
float beta;
float bb;
float ls;
float wt;
float wt2;
float qpct;
float cld_factor;
float fac_damp;
float liq_frac;
float ql_ice;
float ql_water;
float qmq;
float qsat_tk;
float q1_rh;
float rh_hack;
float dzm1;
float zsl;
float maxqc;
constexpr float qpct_sfc = 0.025f;
constexpr float qpct_pbl = 0.030f;
constexpr float qpct_trp = 0.040f;
constexpr float rhcrit = 0.83f;
constexpr float rhmax = 1.02f;
int i;
int j;
int k;
float erf;
float dth;
float dtl;
float dqw;
float dzk;
float els;
float zagl;
float damp;
float pblh2;
float cfmax;
float theta1;
float theta2;
float ht1;
float ht2;
int k_tropo;
float qw_pert;
// bl_mynn_subroutines.F90:2632
for (k=((kte) - (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// bl_mynn_subroutines.F90:2633
theta1 = th(k);
// bl_mynn_subroutines.F90:2634
theta2 = th(((k) + (2)));
// bl_mynn_subroutines.F90:2635
ht1 = mpas_mul(44307.692f, mpas_sub(1.0f, mynn_pow(((float)(mpas_div(p(k), 101325.f))), ((float)(0.190f)))));
// bl_mynn_subroutines.F90:2636
ht2 = mpas_mul(44307.692f, mpas_sub(1.0f, mynn_pow(((float)(mpas_div(p(((k) + (2))), 101325.f))), ((float)(0.190f)))));
// bl_mynn_subroutines.F90:2637
if (((((((mpas_div(mpas_sub(theta2, theta1), mpas_sub(ht2, ht1))) < (mpas_div(10.f, 1500.f)))) && (((ht1) < (19000.f))))) && (((ht1) > (4000.f))))) {
// bl_mynn_subroutines.F90:2639
goto mynn_label_86;
// bl_mynn_subroutines.F90:2640
}
// bl_mynn_subroutines.F90:2641
}
// bl_mynn_subroutines.F90:2642
mynn_label_86: ;
// bl_mynn_subroutines.F90:2643
k_tropo = mynn_max(((int)(((kts) + (2)))), ((int)(((k) + (2)))));
// bl_mynn_subroutines.F90:2645
zagl = 0.f;
// bl_mynn_subroutines.F90:2647
switch (bl_mynn_cloudpdf) {
// bl_mynn_subroutines.F90:2649
case 0: {
// bl_mynn_subroutines.F90:2651
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2652
t = mpas_mul(th(k), exner(k));
// bl_mynn_subroutines.F90:2665
esat = f_esat_blend(t, t0c, tice);
// bl_mynn_subroutines.F90:2668
qsl = mpas_div(mpas_mul(ep_2, esat), mynn_max(((float)(1.e-4f)), ((float)(mpas_sub(p(k), mpas_mul(ep_3, esat))))));
// bl_mynn_subroutines.F90:2670
dqsl = mpas_div(mpas_mul(mpas_mul(qsl, ep_2), xlv), mpas_mul(r_d, mynn_ipow<2>(t)));
// bl_mynn_subroutines.F90:2672
alp(k) = mpas_div(1.0f, mpas_add(1.0f, mpas_mul(dqsl, xlvcp)));
// bl_mynn_subroutines.F90:2673
bet(k) = mpas_mul(dqsl, exner(k));
// bl_mynn_subroutines.F90:2677
t3sq = mynn_real64(mynn_max(((float)(tsq(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2678
r3sq = mynn_real64(mynn_max(((float)(qsq(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2679
c3sq = mynn_real64(cov(k));
// bl_mynn_subroutines.F90:2680
c3sq = mynn_sign(mynn_min(((double)(mynn_abs(c3sq))), ((double)(mynn_sqrt(mynn_dmul(t3sq, r3sq))))), c3sq);
// bl_mynn_subroutines.F90:2681
r3sq = mynn_dsub(mynn_dadd(r3sq, mynn_dmul(mynn_real64(mynn_ipow<2>(bet(k))), t3sq)), mynn_dmul(mynn_real64(mpas_mul(2.0f, bet(k))), c3sq));
// bl_mynn_subroutines.F90:2683
qmq = mpas_sub(qw(k), qsl);
// bl_mynn_subroutines.F90:2685
sgm(k) = mynn_real32(mynn_sqrt(mynn_max(((double)(r3sq)), ((double)(1.0e-10)))));
// bl_mynn_subroutines.F90:2687
q1(k) = mpas_div(qmq, sgm(k));
// bl_mynn_subroutines.F90:2689
cldfra_bl1d(k) = mpas_mul(0.5f, mpas_add(1.0f, mynn_erf(mpas_mul(q1(k), rr2))));
// bl_mynn_subroutines.F90:2691
q1k = q1(k);
// bl_mynn_subroutines.F90:2692
eq1 = mpas_mul(rrp, mynn_exp(mpas_mul(mpas_mul(mynn_neg(0.5f), q1k), q1k)));
// bl_mynn_subroutines.F90:2693
qll = mynn_max(((float)(mpas_add(mpas_mul(cldfra_bl1d(k), q1k), eq1))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:2695
ql(k) = mpas_mul(mpas_mul(alp(k), sgm(k)), qll);
// bl_mynn_subroutines.F90:2697
liq_frac = mynn_min(((float)(1.0f)), ((float)(mynn_max(((float)(0.0f)), ((float)(mpas_div(mpas_sub(t, 240.0f), 29.0f)))))));
// bl_mynn_subroutines.F90:2698
qc_bl1d(k) = mpas_mul(liq_frac, ql(k));
// bl_mynn_subroutines.F90:2699
qi_bl1d(k) = mpas_mul(mpas_sub(1.0f, liq_frac), ql(k));
// bl_mynn_subroutines.F90:2702
q2p = mpas_div(xlvcp, exner(k));
// bl_mynn_subroutines.F90:2703
pt = mpas_add(thl(k), mpas_mul(q2p, ql(k)));
// bl_mynn_subroutines.F90:2706
qt = mpas_sub(mpas_add(1.0f, mpas_mul(p608, qw(k))), mpas_mul(mpas_mul(mpas_add(1.f, p608), mpas_add(qc_bl1d(k), qi_bl1d(k))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:2707
rac = mpas_mul(mpas_mul(alp(k), mpas_sub(cldfra_bl1d(k), mpas_mul(qll, eq1))), mpas_sub(mpas_mul(q2p, qt), mpas_mul(mpas_add(1.f, p608), pt)));
// bl_mynn_subroutines.F90:2712
vt(k) = mpas_sub(mpas_sub(qt, 1.0f), mpas_mul(rac, bet(k)));
// bl_mynn_subroutines.F90:2713
vq(k) = mpas_add(mpas_sub(mpas_mul(p608, pt), tv0), rac);
// bl_mynn_subroutines.F90:2715
}
// bl_mynn_subroutines.F90:2717
break; } case 1: case (-(1)): {
// bl_mynn_subroutines.F90:2719
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2720
t = mpas_mul(th(k), exner(k));
// bl_mynn_subroutines.F90:2722
esat = f_esat_blend(t, t0c, tice);
// bl_mynn_subroutines.F90:2725
qsl = mpas_div(mpas_mul(ep_2, esat), mynn_max(((float)(1.e-4f)), ((float)(mpas_sub(p(k), mpas_mul(ep_3, esat))))));
// bl_mynn_subroutines.F90:2727
dqsl = mpas_div(mpas_mul(mpas_mul(qsl, ep_2), xlv), mpas_mul(r_d, mynn_ipow<2>(t)));
// bl_mynn_subroutines.F90:2729
alp(k) = mpas_div(1.0f, mpas_add(1.0f, mpas_mul(dqsl, xlvcp)));
// bl_mynn_subroutines.F90:2730
bet(k) = mpas_mul(dqsl, exner(k));
// bl_mynn_subroutines.F90:2732
if (((k) == (kts))) {
// bl_mynn_subroutines.F90:2733
dzk = mpas_mul(0.5f, dz(k));
// bl_mynn_subroutines.F90:2734
} else {
// bl_mynn_subroutines.F90:2735
dzk = dz(k);
// bl_mynn_subroutines.F90:2736
}
// bl_mynn_subroutines.F90:2737
dth = mpas_sub(mpas_mul(0.5f, mpas_add(thl(((k) + (1))), thl(k))), mpas_mul(0.5f, mpas_add(thl(k), thl(mynn_max(((int)(((k) - (1)))), ((int)(kts)))))));
// bl_mynn_subroutines.F90:2738
dqw = mpas_sub(mpas_mul(0.5f, mpas_add(qw(((k) + (1))), qw(k))), mpas_mul(0.5f, mpas_add(qw(k), qw(mynn_max(((int)(((k) - (1)))), ((int)(kts)))))));
// bl_mynn_subroutines.F90:2739
sgm(k) = mynn_sqrt(mynn_max(((float)(mpas_mul(mpas_div(mpas_mul(mpas_mul(mpas_mul(mynn_ipow<2>(alp(k)), mynn_max(((float)(mynn_ipow<2>(el(k)))), ((float)(0.1f)))), b2), mynn_max(((float)(0.03f)), ((float)(sh(k))))), 4.f), mynn_ipow<2>(mpas_sub(mpas_div(dqw, dzk), mpas_mul(bet(k), mpas_div(dth, dzk))))))), ((float)(1.0e-10f))));
// bl_mynn_subroutines.F90:2742
qmq = mpas_sub(qw(k), qsl);
// bl_mynn_subroutines.F90:2743
q1(k) = mpas_div(qmq, sgm(k));
// bl_mynn_subroutines.F90:2744
cldfra_bl1d(k) = mpas_mul(0.5f, mpas_add(1.0f, mynn_erf(mpas_mul(q1(k), rr2))));
// bl_mynn_subroutines.F90:2749
q1k = q1(k);
// bl_mynn_subroutines.F90:2750
eq1 = mpas_mul(rrp, mynn_exp(mpas_mul(mpas_mul(mynn_neg(0.5f), q1k), q1k)));
// bl_mynn_subroutines.F90:2751
qll = mynn_max(((float)(mpas_add(mpas_mul(cldfra_bl1d(k), q1k), eq1))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:2753
ql(k) = mpas_mul(mpas_mul(alp(k), sgm(k)), qll);
// bl_mynn_subroutines.F90:2754
liq_frac = mynn_min(((float)(1.0f)), ((float)(mynn_max(((float)(0.0f)), ((float)(mpas_div(mpas_sub(t, 240.0f), 29.0f)))))));
// bl_mynn_subroutines.F90:2755
qc_bl1d(k) = mpas_mul(liq_frac, ql(k));
// bl_mynn_subroutines.F90:2756
qi_bl1d(k) = mpas_mul(mpas_sub(1.0f, liq_frac), ql(k));
// bl_mynn_subroutines.F90:2759
q2p = mpas_div(xlvcp, exner(k));
// bl_mynn_subroutines.F90:2760
pt = mpas_add(thl(k), mpas_mul(q2p, ql(k)));
// bl_mynn_subroutines.F90:2763
qt = mpas_sub(mpas_add(1.0f, mpas_mul(p608, qw(k))), mpas_mul(mpas_mul(mpas_add(1.f, p608), mpas_add(qc_bl1d(k), qi_bl1d(k))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:2764
rac = mpas_mul(mpas_mul(alp(k), mpas_sub(cldfra_bl1d(k), mpas_mul(qll, eq1))), mpas_sub(mpas_mul(q2p, qt), mpas_mul(mpas_add(1.f, p608), pt)));
// bl_mynn_subroutines.F90:2769
vt(k) = mpas_sub(mpas_sub(qt, 1.0f), mpas_mul(rac, bet(k)));
// bl_mynn_subroutines.F90:2770
vq(k) = mpas_add(mpas_sub(mpas_mul(p608, pt), tv0), rac);
// bl_mynn_subroutines.F90:2772
}
// bl_mynn_subroutines.F90:2774
break; } case 2: case (-(2)): {
// bl_mynn_subroutines.F90:2778
pblh2 = mynn_max(((float)(10.f)), ((float)(pblh1)));
// bl_mynn_subroutines.F90:2779
zagl = 0.f;
// bl_mynn_subroutines.F90:2780
dzm1 = 0.f;
// bl_mynn_subroutines.F90:2781
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2782
zagl = mpas_add(zagl, mpas_mul(0.5f, mpas_add(dz(k), dzm1)));
// bl_mynn_subroutines.F90:2783
dzm1 = dz(k);
// bl_mynn_subroutines.F90:2785
t = mpas_mul(th(k), exner(k));
// bl_mynn_subroutines.F90:2786
xl = f_xl_blend(t, t0c, tice, cice, cliq, cpv, xls, xlv);
// bl_mynn_subroutines.F90:2787
qsat_tk = f_qsat_blend(t, t0c, tice, p(k), {});
// bl_mynn_subroutines.F90:2788
rh(k) = mynn_max(((float)(mynn_min(((float)(mpas_div(qw(k), mynn_max(((float)(1.e-10f)), ((float)(qsat_tk)))))), ((float)(rhmax))))), ((float)(0.001f)));
// bl_mynn_subroutines.F90:2791
dqsl = mpas_div(mpas_mul(mpas_mul(qsat_tk, ep_2), xlv), mpas_mul(r_d, mynn_ipow<2>(t)));
// bl_mynn_subroutines.F90:2792
alp(k) = mpas_div(1.0f, mpas_add(1.0f, mpas_mul(dqsl, xlvcp)));
// bl_mynn_subroutines.F90:2793
bet(k) = mpas_mul(dqsl, exner(k));
// bl_mynn_subroutines.F90:2795
rsl = mpas_div(mpas_mul(xl, qsat_tk), mpas_mul(r_v, mynn_ipow<2>(t)));
// bl_mynn_subroutines.F90:2797
cpm = mpas_add(cp, mpas_mul(qw(k), cpv));
// bl_mynn_subroutines.F90:2798
a(k) = mpas_div(1.f, mpas_add(1.f, mpas_div(mpas_mul(xl, rsl), cpm)));
// bl_mynn_subroutines.F90:2799
b(k) = mpas_mul(a(k), rsl);
// bl_mynn_subroutines.F90:2802
qw_pert = mpas_add(qw(k), mpas_mul(mpas_mul(mpas_mul(qw(k), 0.5f), rstoch_col(k)), ((float)(spp_pbl))));
// bl_mynn_subroutines.F90:2805
qmq = mpas_sub(qw_pert, qsat_tk);
// bl_mynn_subroutines.F90:2809
r3sq = mynn_real64(mynn_max(((float)(qsq(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2811
sgm(k) = mynn_real32(mynn_sqrt(r3sq));
// bl_mynn_subroutines.F90:2813
sgm(k) = mynn_min(((float)(mpas_mul(qsat_tk, 0.666f))), ((float)(sgm(k))));
// bl_mynn_subroutines.F90:2817
wt = mpas_div(mynn_max(((float)(mpas_sub(500.f, mynn_max(((float)(mpas_sub(dz(k), 100.f))), ((float)(0.0f)))))), ((float)(0.0f))), 500.f);
// bl_mynn_subroutines.F90:2818
sgm(k) = mpas_add(sgm(k), mpas_mul(mpas_mul(sgm(k), 0.2f), mpas_sub(1.0f, wt)));
// bl_mynn_subroutines.F90:2821
qpct = mpas_add(mpas_mul(qpct_pbl, wt), mpas_mul(qpct_trp, mpas_sub(1.0f, wt)));
// bl_mynn_subroutines.F90:2822
qpct = mynn_min(((float)(qpct)), ((float)(mynn_max(((float)(qpct_sfc)), ((float)(mpas_div(mpas_mul(qpct_pbl, zagl), 500.f)))))));
// bl_mynn_subroutines.F90:2823
sgm(k) = mynn_max(((float)(mpas_mul(qsat_tk, qpct))), ((float)(sgm(k))));
// bl_mynn_subroutines.F90:2825
q1(k) = mpas_div(qmq, sgm(k));
// bl_mynn_subroutines.F90:2829
rh_hack = rh(k);
// bl_mynn_subroutines.F90:2830
wt2 = mynn_min(((float)(mpas_div(mynn_max(((float)(mpas_sub(zagl, pblh2))), ((float)(0.0f))), 300.f))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2832
if (((((mpas_add(qi(k), qs(k))) > (1.e-9f))) && (((zagl) > (pblh2))))) {
// bl_mynn_subroutines.F90:2833
rh_hack = mynn_min(((float)(rhmax)), ((float)(mpas_add(rhcrit, mpas_mul(mpas_mul(wt2, 0.045f), mpas_add(9.0f, mynn_log10(mpas_add(qi(k), qs(k)))))))));
// bl_mynn_subroutines.F90:2834
rh(k) = mynn_max(((float)(rh(k))), ((float)(rh_hack)));
// bl_mynn_subroutines.F90:2836
q1_rh = mpas_add(mynn_neg(3.f), mpas_div(mpas_mul(3.f, mpas_sub(rh(k), rhcrit)), mpas_sub(1.f, rhcrit)));
// bl_mynn_subroutines.F90:2837
q1(k) = mynn_max(((float)(q1_rh)), ((float)(q1(k))));
// bl_mynn_subroutines.F90:2838
}
// bl_mynn_subroutines.F90:2840
if (((((qc(k)) > (1.e-6f))) && (((zagl) > (pblh2))))) {
// bl_mynn_subroutines.F90:2841
rh_hack = mynn_min(((float)(rhmax)), ((float)(mpas_add(rhcrit, mpas_mul(mpas_mul(wt2, 0.08f), mpas_add(6.0f, mynn_log10(qc(k))))))));
// bl_mynn_subroutines.F90:2842
rh(k) = mynn_max(((float)(rh(k))), ((float)(rh_hack)));
// bl_mynn_subroutines.F90:2844
q1_rh = mpas_add(mynn_neg(3.f), mpas_div(mpas_mul(3.f, mpas_sub(rh(k), rhcrit)), mpas_sub(1.f, rhcrit)));
// bl_mynn_subroutines.F90:2845
q1(k) = mynn_max(((float)(q1_rh)), ((float)(q1(k))));
// bl_mynn_subroutines.F90:2846
}
// bl_mynn_subroutines.F90:2848
q1k = q1(k);
// bl_mynn_subroutines.F90:2856
cldfra_bl1d(k) = mynn_max(((float)(mynn_min(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.8f, mpas_add(q1(k), 0.2f))))))), ((float)(1.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:2861
maxqc = mynn_max(((float)(mpas_sub(qw(k), qsat_tk))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:2862
if (((q1k) < (0.f))) {
// bl_mynn_subroutines.F90:2863
ql_water = mpas_mul(sgm(k), mynn_exp(mpas_sub(mpas_mul(1.2f, q1k), 1.f)));
// bl_mynn_subroutines.F90:2864
ql_ice = mpas_mul(sgm(k), mynn_exp(mpas_sub(mpas_mul(1.2f, q1k), 1.f)));
// bl_mynn_subroutines.F90:2865
} else if (((q1k) > (2.f))) {
// bl_mynn_subroutines.F90:2866
ql_water = mynn_min(((float)(maxqc)), ((float)(mpas_mul(sgm(k), q1k))));
// bl_mynn_subroutines.F90:2867
ql_ice = mpas_mul(sgm(k), q1k);
// bl_mynn_subroutines.F90:2868
} else {
// bl_mynn_subroutines.F90:2869
ql_water = mynn_min(((float)(maxqc)), ((float)(mpas_mul(sgm(k), mpas_add(mpas_add(mynn_exp(mynn_neg(1.f)), mpas_mul(0.66f, q1k)), mpas_mul(0.086f, mynn_ipow<2>(q1k)))))));
// bl_mynn_subroutines.F90:2870
ql_ice = mpas_mul(sgm(k), mpas_add(mpas_add(mynn_exp(mynn_neg(1.f)), mpas_mul(0.66f, q1k)), mpas_mul(0.086f, mynn_ipow<2>(q1k))));
// bl_mynn_subroutines.F90:2871
}
// bl_mynn_subroutines.F90:2878
if (((cldfra_bl1d(k)) < (0.001f))) {
// bl_mynn_subroutines.F90:2879
ql_ice = 0.0f;
// bl_mynn_subroutines.F90:2880
ql_water = 0.0f;
// bl_mynn_subroutines.F90:2881
cldfra_bl1d(k) = 0.0f;
// bl_mynn_subroutines.F90:2882
}
// bl_mynn_subroutines.F90:2884
liq_frac = mynn_min(((float)(1.0f)), ((float)(mynn_max(((float)(0.0f)), ((float)(mpas_div(mpas_sub(t, tice), mpas_sub(tliq, tice))))))));
// bl_mynn_subroutines.F90:2885
qc_bl1d(k) = mpas_mul(liq_frac, ql_water);
// bl_mynn_subroutines.F90:2886
qi_bl1d(k) = mpas_mul(mpas_sub(1.0f, liq_frac), ql_ice);
// bl_mynn_subroutines.F90:2890
if (((k) >= (k_tropo))) {
// bl_mynn_subroutines.F90:2891
cldfra_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2892
qc_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2893
qi_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2894
}
// bl_mynn_subroutines.F90:2899
if (((mpas_sub(xland, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:2900
q1k = mynn_max(((float)(q1(k))), ((float)(mynn_neg(2.5f))));
// bl_mynn_subroutines.F90:2901
} else {
// bl_mynn_subroutines.F90:2902
q1k = mynn_max(((float)(q1(k))), ((float)(mynn_neg(2.0f))));
// bl_mynn_subroutines.F90:2903
}
// bl_mynn_subroutines.F90:2916
if (((q1k) >= (1.0f))) {
// bl_mynn_subroutines.F90:2917
fng = 1.0f;
// bl_mynn_subroutines.F90:2918
} else if (((((q1k) >= (mynn_neg(1.7f)))) && (((q1k) < (1.0f))))) {
// bl_mynn_subroutines.F90:2919
fng = mynn_exp(mpas_mul(mynn_neg(0.4f), mpas_sub(q1k, 1.0f)));
// bl_mynn_subroutines.F90:2920
} else if (((((q1k) >= (mynn_neg(2.5f)))) && (((q1k) < (mynn_neg(1.7f)))))) {
// bl_mynn_subroutines.F90:2921
fng = mpas_add(3.0f, mynn_exp(mpas_mul(mynn_neg(3.8f), mpas_add(q1k, 1.7f))));
// bl_mynn_subroutines.F90:2922
} else {
// bl_mynn_subroutines.F90:2923
fng = mynn_min(((float)(mpas_add(23.9f, mynn_exp(mpas_mul(mynn_neg(1.6f), mpas_add(q1k, 2.5f)))))), ((float)(60.f)));
// bl_mynn_subroutines.F90:2924
}
// bl_mynn_subroutines.F90:2926
cfmax = mynn_min(((float)(cldfra_bl1d(k))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:2928
zsl = mynn_min(((float)(mynn_max(((float)(25.f)), ((float)(mpas_mul(0.1f, pblh2)))))), ((float)(100.f)));
// bl_mynn_subroutines.F90:2929
wt = mynn_min(((float)(mpas_div(zagl, zsl))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2930
cfmax = mpas_mul(cfmax, wt);
// bl_mynn_subroutines.F90:2932
bb = mpas_div(mpas_mul(b(k), t), th(k));
// bl_mynn_subroutines.F90:2938
qww = mpas_add(1.f, mpas_mul(0.61f, qw(k)));
// bl_mynn_subroutines.F90:2939
alpha = mpas_mul(0.61f, th(k));
// bl_mynn_subroutines.F90:2940
beta = mpas_sub(mpas_mul(mpas_div(th(k), t), mpas_div(xl, cp)), mpas_mul(1.61f, th(k)));
// bl_mynn_subroutines.F90:2941
vt(k) = mpas_sub(mpas_sub(qww, mpas_mul(mpas_mul(mpas_mul(cfmax, beta), bb), fng)), 1.f);
// bl_mynn_subroutines.F90:2942
vq(k) = mpas_sub(mpas_add(alpha, mpas_mul(mpas_mul(mpas_mul(cfmax, beta), a(k)), fng)), tv0);
// bl_mynn_subroutines.F90:2950
fac_damp = mynn_min(((float)(mpas_mul(zagl, 0.0025f))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2953
cld_factor = mpas_add(1.0f, mpas_mul(fac_damp, mynn_min(((float)(mynn_ipow<2>(mpas_div(mynn_max(((float)(0.0f)), ((float)(mpas_sub(rh(k), 0.92f)))), 0.145f)))), ((float)(0.37f)))));
// bl_mynn_subroutines.F90:2954
cldfra_bl1d(k) = mynn_min(((float)(1.f)), ((float)(mpas_mul(cld_factor, cldfra_bl1d(k)))));
// bl_mynn_subroutines.F90:2955
}
// bl_mynn_subroutines.F90:2957
break; } }
// bl_mynn_subroutines.F90:2960
if (((bl_mynn_cloudpdf) < (0))) {
// bl_mynn_subroutines.F90:2961
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2962
cldfra_bl1d(k) = 0.0f;
// bl_mynn_subroutines.F90:2963
qc_bl1d(k) = 0.0f;
// bl_mynn_subroutines.F90:2964
qi_bl1d(k) = 0.0f;
// bl_mynn_subroutines.F90:2965
}
// bl_mynn_subroutines.F90:2966
}
// bl_mynn_subroutines.F90:2968
ql(kte) = ql(((kte) - (1)));
// bl_mynn_subroutines.F90:2969
vt(kte) = vt(((kte) - (1)));
// bl_mynn_subroutines.F90:2970
vq(kte) = vq(((kte) - (1)));
// bl_mynn_subroutines.F90:2971
qc_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2972
qi_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2973
cldfra_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2974
return;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:2987-4092
__device__ __noinline__ void f_mynn_tendencies(int kts, int kte, float delt, MynnArray<float, 1> dz, MynnArray<float, 1> rho, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> tk, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, float psfc, MynnArray<float, 1> p, MynnArray<float, 1> exner, MynnArray<float, 1> thl, MynnArray<float, 1> sqv, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqs, MynnArray<float, 1> sqw, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qnbca, MynnArray<float, 1> ozone, float ust, float flt, float flq, float flqv, float flqc, float wspd, float uoce, float voce, MynnArray<float, 1> tsq, MynnArray<float, 1> qsq, MynnArray<float, 1> cov, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dfq, MynnArray<float, 1> du, MynnArray<float, 1> dv, MynnArray<float, 1> dth, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dqnc, MynnArray<float, 1> dqni, MynnArray<float, 1> dqnwfa, MynnArray<float, 1> dqnifa, MynnArray<float, 1> dqnbca, MynnArray<float, 1> dozone, MynnArray<float, 1> diss_heat, MynnArray<float, 1> s_aw, MynnArray<float, 1> s_awthl, MynnArray<float, 1> s_awqt, MynnArray<float, 1> s_awqv, MynnArray<float, 1> s_awqc, MynnArray<float, 1> s_awu, MynnArray<float, 1> s_awv, MynnArray<float, 1> s_awqnc, MynnArray<float, 1> s_awqni, MynnArray<float, 1> s_awqnwfa, MynnArray<float, 1> s_awqnifa, MynnArray<float, 1> s_awqnbca, MynnArray<float, 1> sd_aw, MynnArray<float, 1> sd_awthl, MynnArray<float, 1> sd_awqt, MynnArray<float, 1> sd_awqv, MynnArray<float, 1> sd_awqc, MynnArray<float, 1> sd_awu, MynnArray<float, 1> sd_awv, MynnArray<float, 1> sub_thl, MynnArray<float, 1> sub_sqv, MynnArray<float, 1> sub_u, MynnArray<float, 1> sub_v, MynnArray<float, 1> det_thl, MynnArray<float, 1> det_sqv, MynnArray<float, 1> det_sqc, MynnArray<float, 1> det_u, MynnArray<float, 1> det_v, bool flag_qc, bool flag_qi, bool flag_qnc, bool flag_qni, bool flag_qs, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_ozone, MynnArray<float, 1> cldfra_bl1d, bool bl_mynn_cloudmix, bool bl_mynn_mixqt, bool bl_mynn_edmf_mom, bool bl_mynn_mixscalars) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
rho.rebase({kts});
u.rebase({kts});
v.rebase({kts});
th.rebase({kts});
tk.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qi.rebase({kts});
qs.rebase({kts});
qnc.rebase({kts});
qni.rebase({kts});
p.rebase({kts});
exner.rebase({kts});
thl.rebase({kts});
sqv.rebase({kts});
sqc.rebase({kts});
sqi.rebase({kts});
sqs.rebase({kts});
sqw.rebase({kts});
qnwfa.rebase({kts});
qnifa.rebase({kts});
qnbca.rebase({kts});
ozone.rebase({kts});
tsq.rebase({kts});
qsq.rebase({kts});
cov.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
dfm.rebase({kts});
dfh.rebase({kts});
dfq.rebase({kts});
du.rebase({kts});
dv.rebase({kts});
dth.rebase({kts});
dqv.rebase({kts});
dqc.rebase({kts});
dqi.rebase({kts});
dqs.rebase({kts});
dqnc.rebase({kts});
dqni.rebase({kts});
dqnwfa.rebase({kts});
dqnifa.rebase({kts});
dqnbca.rebase({kts});
dozone.rebase({kts});
diss_heat.rebase({kts});
s_aw.rebase({kts});
s_awthl.rebase({kts});
s_awqt.rebase({kts});
s_awqv.rebase({kts});
s_awqc.rebase({kts});
s_awu.rebase({kts});
s_awv.rebase({kts});
s_awqnc.rebase({kts});
s_awqni.rebase({kts});
s_awqnwfa.rebase({kts});
s_awqnifa.rebase({kts});
s_awqnbca.rebase({kts});
sd_aw.rebase({kts});
sd_awthl.rebase({kts});
sd_awqt.rebase({kts});
sd_awqv.rebase({kts});
sd_awqc.rebase({kts});
sd_awu.rebase({kts});
sd_awv.rebase({kts});
sub_thl.rebase({kts});
sub_sqv.rebase({kts});
sub_u.rebase({kts});
sub_v.rebase({kts});
det_thl.rebase({kts});
det_sqv.rebase({kts});
det_sqc.rebase({kts});
det_u.rebase({kts});
det_v.rebase({kts});
cldfra_bl1d.rebase({kts});
float wsp;
float wsp2;
float tk2;
float th2;
bool problem;
int kproblem;
MynnArray<float, 1> dtz = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dfhc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dfmc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> delp = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqv2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqc2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqi2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqs2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqw2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qni2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnc2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnwfa2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnifa2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnbca2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> ozone2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> zfac = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> plumekh = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoinv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> khdz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> kmdz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
float rhs;
float gfluxm;
float gfluxp;
float dztop;
float maxdfh;
float mindfh;
float maxcf;
float maxkh;
float zw;
float t;
float esat;
float qsl;
float onoff;
float kh;
float km;
float dzk;
float rhosfc;
float ustdrag;
float ustdiff;
float qvflux;
float th_new;
float portion_qc;
float portion_qi;
float condensate;
float qsat;
int k;
int kk;
constexpr float nonloc = 1.0f;
// bl_mynn_subroutines.F90:3083
dztop = mpas_mul(.5f, mpas_add(dz(kte), dz(((kte) - (1)))));
// bl_mynn_subroutines.F90:3088
if ((!(bl_mynn_edmf_mom))) {
// bl_mynn_subroutines.F90:3089
onoff = 0.0f;
// bl_mynn_subroutines.F90:3090
} else {
// bl_mynn_subroutines.F90:3091
onoff = 1.0f;
// bl_mynn_subroutines.F90:3092
}
// bl_mynn_subroutines.F90:3096
rhosfc = mpas_div(psfc, mpas_mul(r_d, mpas_add(tk(kts), mpas_mul(p608, qv(kts)))));
// bl_mynn_subroutines.F90:3097
dtz(kts) = mpas_div(delt, dz(kts));
// bl_mynn_subroutines.F90:3098
rhoz(kts) = rho(kts);
// bl_mynn_subroutines.F90:3099
rhoinv(kts) = mpas_div(1.f, rho(kts));
// bl_mynn_subroutines.F90:3100
khdz(kts) = mpas_mul(rhoz(kts), dfh(kts));
// bl_mynn_subroutines.F90:3101
kmdz(kts) = mpas_mul(rhoz(kts), dfm(kts));
// bl_mynn_subroutines.F90:3102
delp(kts) = mpas_sub(psfc, mpas_div(mpas_add(mpas_mul(p(((kts) + (1))), dz(kts)), mpas_mul(p(kts), dz(((kts) + (1))))), mpas_add(dz(kts), dz(((kts) + (1))))));
// bl_mynn_subroutines.F90:3103
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3104
dtz(k) = mpas_div(delt, dz(k));
// bl_mynn_subroutines.F90:3105
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) - (1)))), mpas_mul(rho(((k) - (1))), dz(k))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:3106
rhoz(k) = mynn_max(((float)(rhoz(k))), ((float)(1e-4f)));
// bl_mynn_subroutines.F90:3107
rhoinv(k) = mpas_div(1.f, mynn_max(((float)(rho(k))), ((float)(1e-4f))));
// bl_mynn_subroutines.F90:3108
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:3109
khdz(k) = mpas_mul(rhoz(k), dfh(k));
// bl_mynn_subroutines.F90:3110
kmdz(k) = mpas_mul(rhoz(k), dfm(k));
// bl_mynn_subroutines.F90:3111
}
// bl_mynn_subroutines.F90:3112
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3113
delp(k) = mpas_sub(mpas_div(mpas_add(mpas_mul(p(k), dz(((k) - (1)))), mpas_mul(p(((k) - (1))), dz(k))), mpas_add(dz(k), dz(((k) - (1))))), mpas_div(mpas_add(mpas_mul(p(((k) + (1))), dz(k)), mpas_mul(p(k), dz(((k) + (1))))), mpas_add(dz(k), dz(((k) + (1))))));
// bl_mynn_subroutines.F90:3115
}
// bl_mynn_subroutines.F90:3116
delp(kte) = delp(((kte) - (1)));
// bl_mynn_subroutines.F90:3117
rhoz(((kte) + (1))) = rhoz(kte);
// bl_mynn_subroutines.F90:3118
khdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfh(kte));
// bl_mynn_subroutines.F90:3119
kmdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfm(kte));
// bl_mynn_subroutines.F90:3122
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3123
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:3124
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:3125
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:3126
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:3127
}
// bl_mynn_subroutines.F90:3129
ustdrag = mpas_div(mynn_min(((float)(mpas_mul(ust, ust))), ((float)(0.99f))), wspd);
// bl_mynn_subroutines.F90:3130
ustdiff = mpas_div(mynn_min(((float)(mpas_mul(ust, ust))), ((float)(0.01f))), wspd);
// bl_mynn_subroutines.F90:3131
for (int mynn_element=0; mynn_element < (dth.slice(MynnRange{kts, kte})).size(); ++mynn_element) { dth.slice(MynnRange{kts, kte}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:3137
k = kts;
// bl_mynn_subroutines.F90:3140
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3141
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(((k) + (1))), mpas_div(mpas_mul(rhosfc, mynn_ipow<2>(ust)), wspd))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3144
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3147
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_add(u(k), mpas_div(mpas_mul(mpas_mul(dtz(k), uoce), mynn_ipow<2>(ust)), wspd)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awu(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awu(((k) + (1)))), onoff)), mpas_mul(sub_u(k), delt)), mpas_mul(det_u(k), delt));
// bl_mynn_subroutines.F90:3152
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3153
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)), onoff));
// bl_mynn_subroutines.F90:3156
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:3159
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3162
d(k) = mpas_add(mpas_add(mpas_sub(mpas_add(u(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awu(k), s_awu(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awu(k), sd_awu(((k) + (1))))), onoff)), mpas_mul(sub_u(k), delt)), mpas_mul(det_u(k), delt));
// bl_mynn_subroutines.F90:3165
}
// bl_mynn_subroutines.F90:3180
a(kte) = 0;
// bl_mynn_subroutines.F90:3181
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3182
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3183
d(kte) = u(kte);
// bl_mynn_subroutines.F90:3186
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3189
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3191
du(k) = mpas_div(mpas_sub(x(k), u(k)), delt);
// bl_mynn_subroutines.F90:3192
}
// bl_mynn_subroutines.F90:3198
k = kts;
// bl_mynn_subroutines.F90:3201
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3202
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(((k) + (1))), mpas_div(mpas_mul(rhosfc, mynn_ipow<2>(ust)), wspd))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3205
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3208
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_add(v(k), mpas_div(mpas_mul(mpas_mul(dtz(k), voce), mynn_ipow<2>(ust)), wspd)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awv(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awv(((k) + (1)))), onoff)), mpas_mul(sub_v(k), delt)), mpas_mul(det_v(k), delt));
// bl_mynn_subroutines.F90:3213
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3214
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)), onoff));
// bl_mynn_subroutines.F90:3217
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:3220
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3223
d(k) = mpas_add(mpas_add(mpas_sub(mpas_add(v(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awv(k), s_awv(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awv(k), sd_awv(((k) + (1))))), onoff)), mpas_mul(sub_v(k), delt)), mpas_mul(det_v(k), delt));
// bl_mynn_subroutines.F90:3226
}
// bl_mynn_subroutines.F90:3241
a(kte) = 0;
// bl_mynn_subroutines.F90:3242
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3243
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3244
d(kte) = v(kte);
// bl_mynn_subroutines.F90:3247
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3250
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3252
dv(k) = mpas_div(mpas_sub(x(k), v(k)), delt);
// bl_mynn_subroutines.F90:3253
}
// bl_mynn_subroutines.F90:3258
k = kts;
// bl_mynn_subroutines.F90:3277
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3278
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3279
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3280
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(thl(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flt), rhoinv(k))), mpas_mul(tcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awthl(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awthl(((k) + (1))))), mpas_mul(diss_heat(k), delt)), mpas_mul(sub_thl(k), delt)), mpas_mul(det_thl(k), delt));
// bl_mynn_subroutines.F90:3284
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3285
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3286
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3288
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3289
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(thl(k), mpas_mul(tcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awthl(k), s_awthl(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awthl(k), sd_awthl(((k) + (1)))))), mpas_mul(diss_heat(k), delt)), mpas_mul(sub_thl(k), delt)), mpas_mul(det_thl(k), delt));
// bl_mynn_subroutines.F90:3293
}
// bl_mynn_subroutines.F90:3309
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3310
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3311
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3312
d(kte) = thl(kte);
// bl_mynn_subroutines.F90:3315
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3318
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3320
thl(k) = x(k);
// bl_mynn_subroutines.F90:3321
}
// bl_mynn_subroutines.F90:3323
if (bl_mynn_mixqt) {
// bl_mynn_subroutines.F90:3331
k = kts;
// bl_mynn_subroutines.F90:3347
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3348
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3349
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3350
d(k) = mpas_sub(mpas_sub(mpas_add(mpas_add(sqw(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flq), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqt(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqt(((k) + (1)))));
// bl_mynn_subroutines.F90:3352
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3353
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3354
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3356
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3357
d(k) = mpas_add(mpas_add(mpas_add(sqw(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqt(k), s_awqt(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqt(k), sd_awqt(((k) + (1))))));
// bl_mynn_subroutines.F90:3358
}
// bl_mynn_subroutines.F90:3372
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3373
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3374
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3375
d(kte) = sqw(kte);
// bl_mynn_subroutines.F90:3378
f_tridiag2(kte, a, b, c, d, sqw2);
// bl_mynn_subroutines.F90:3384
} else {
// bl_mynn_subroutines.F90:3385
for (int mynn_element=0; mynn_element < (sqw2).size(); ++mynn_element) { sqw2.linear(mynn_element) = sqw.linear(mynn_element); }
// bl_mynn_subroutines.F90:3386
}
// bl_mynn_subroutines.F90:3388
if ((!(bl_mynn_mixqt))) {
// bl_mynn_subroutines.F90:3393
if (((bl_mynn_cloudmix) && (flag_qc))) {
// bl_mynn_subroutines.F90:3395
k = kts;
// bl_mynn_subroutines.F90:3412
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3413
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3414
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3415
d(k) = mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(sqc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flqc), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqc(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqc(((k) + (1))))), mpas_mul(det_sqc(k), delt));
// bl_mynn_subroutines.F90:3419
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3420
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3421
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3423
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3424
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(sqc(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqc(k), s_awqc(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqc(k), sd_awqc(((k) + (1)))))), mpas_mul(det_sqc(k), delt));
// bl_mynn_subroutines.F90:3426
}
// bl_mynn_subroutines.F90:3429
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3430
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3431
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3432
d(kte) = sqc(kte);
// bl_mynn_subroutines.F90:3435
f_tridiag2(kte, a, b, c, d, sqc2);
// bl_mynn_subroutines.F90:3441
} else {
// bl_mynn_subroutines.F90:3443
for (int mynn_element=0; mynn_element < (sqc2).size(); ++mynn_element) { sqc2.linear(mynn_element) = sqc.linear(mynn_element); }
// bl_mynn_subroutines.F90:3444
}
// bl_mynn_subroutines.F90:3445
}
// bl_mynn_subroutines.F90:3447
if ((!(bl_mynn_mixqt))) {
// bl_mynn_subroutines.F90:3453
k = kts;
// bl_mynn_subroutines.F90:3470
qvflux = flqv;
// bl_mynn_subroutines.F90:3471
if (((qvflux) < (0.0f))) {
// bl_mynn_subroutines.F90:3473
qvflux = mynn_max(((float)(qvflux)), ((float)(mpas_div(mynn_min(((float)(mpas_sub(mpas_mul(0.9f, sqv(kts)), 1e-8f))), ((float)(0.0f))), dtz(kts)))));
// bl_mynn_subroutines.F90:3474
}
// bl_mynn_subroutines.F90:3477
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3478
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3479
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3480
d(k) = mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(sqv(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqv(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqv(((k) + (1))))), mpas_mul(sub_sqv(k), delt)), mpas_mul(det_sqv(k), delt));
// bl_mynn_subroutines.F90:3484
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3485
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3486
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3488
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3489
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(sqv(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqv(k), s_awqv(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqv(k), sd_awqv(((k) + (1)))))), mpas_mul(sub_sqv(k), delt)), mpas_mul(det_sqv(k), delt));
// bl_mynn_subroutines.F90:3491
}
// bl_mynn_subroutines.F90:3507
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3508
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3509
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3510
d(kte) = sqv(kte);
// bl_mynn_subroutines.F90:3513
f_tridiag2(kte, a, b, c, d, sqv2);
// bl_mynn_subroutines.F90:3519
} else {
// bl_mynn_subroutines.F90:3520
for (int mynn_element=0; mynn_element < (sqv2).size(); ++mynn_element) { sqv2.linear(mynn_element) = sqv.linear(mynn_element); }
// bl_mynn_subroutines.F90:3521
}
// bl_mynn_subroutines.F90:3526
if (((bl_mynn_cloudmix) && (flag_qi))) {
// bl_mynn_subroutines.F90:3528
k = kts;
// bl_mynn_subroutines.F90:3530
a(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(k)), rhoinv(k)));
// bl_mynn_subroutines.F90:3531
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3532
c(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// bl_mynn_subroutines.F90:3533
d(k) = sqi(k);
// bl_mynn_subroutines.F90:3535
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3536
a(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(k)), rhoinv(k)));
// bl_mynn_subroutines.F90:3537
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3538
c(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// bl_mynn_subroutines.F90:3539
d(k) = sqi(k);
// bl_mynn_subroutines.F90:3540
}
// bl_mynn_subroutines.F90:3556
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3557
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3558
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3559
d(kte) = sqi(kte);
// bl_mynn_subroutines.F90:3562
f_tridiag2(kte, a, b, c, d, sqi2);
// bl_mynn_subroutines.F90:3568
} else {
// bl_mynn_subroutines.F90:3569
for (int mynn_element=0; mynn_element < (sqi2).size(); ++mynn_element) { sqi2.linear(mynn_element) = sqi.linear(mynn_element); }
// bl_mynn_subroutines.F90:3570
}
// bl_mynn_subroutines.F90:3576
if (((bl_mynn_cloudmix) && (false))) {
// bl_mynn_subroutines.F90:3578
k = kts;
// bl_mynn_subroutines.F90:3580
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3581
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3582
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3583
d(k) = sqs(k);
// bl_mynn_subroutines.F90:3585
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3586
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3587
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3588
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3589
d(k) = sqs(k);
// bl_mynn_subroutines.F90:3590
}
// bl_mynn_subroutines.F90:3593
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3594
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3595
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3596
d(kte) = sqs(kte);
// bl_mynn_subroutines.F90:3599
f_tridiag2(kte, a, b, c, d, sqs2);
// bl_mynn_subroutines.F90:3605
} else {
// bl_mynn_subroutines.F90:3606
for (int mynn_element=0; mynn_element < (sqs2).size(); ++mynn_element) { sqs2.linear(mynn_element) = sqs.linear(mynn_element); }
// bl_mynn_subroutines.F90:3607
}
// bl_mynn_subroutines.F90:3612
if (((((bl_mynn_cloudmix) && (flag_qni))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3615
k = kts;
// bl_mynn_subroutines.F90:3617
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3618
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3619
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3620
d(k) = mpas_sub(qni(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqni(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3622
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3623
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3624
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3626
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3627
d(k) = mpas_add(qni(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqni(k), s_awqni(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3628
}
// bl_mynn_subroutines.F90:3631
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3632
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3633
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3634
d(kte) = qni(kte);
// bl_mynn_subroutines.F90:3637
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3640
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3642
qni2(k) = x(k);
// bl_mynn_subroutines.F90:3643
}
// bl_mynn_subroutines.F90:3645
} else {
// bl_mynn_subroutines.F90:3646
for (int mynn_element=0; mynn_element < (qni2).size(); ++mynn_element) { qni2.linear(mynn_element) = qni.linear(mynn_element); }
// bl_mynn_subroutines.F90:3647
}
// bl_mynn_subroutines.F90:3653
if (((((bl_mynn_cloudmix) && (flag_qnc))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3656
k = kts;
// bl_mynn_subroutines.F90:3658
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3659
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3660
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3661
d(k) = mpas_sub(qnc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnc(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3663
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3664
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3665
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3667
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3668
d(k) = mpas_add(qnc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnc(k), s_awqnc(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3669
}
// bl_mynn_subroutines.F90:3672
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3673
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3674
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3675
d(kte) = qnc(kte);
// bl_mynn_subroutines.F90:3678
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3681
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3683
qnc2(k) = x(k);
// bl_mynn_subroutines.F90:3684
}
// bl_mynn_subroutines.F90:3686
} else {
// bl_mynn_subroutines.F90:3687
for (int mynn_element=0; mynn_element < (qnc2).size(); ++mynn_element) { qnc2.linear(mynn_element) = qnc.linear(mynn_element); }
// bl_mynn_subroutines.F90:3688
}
// bl_mynn_subroutines.F90:3693
if (((((bl_mynn_cloudmix) && (flag_qnwfa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3696
k = kts;
// bl_mynn_subroutines.F90:3698
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3699
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3701
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3702
d(k) = mpas_sub(qnwfa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnwfa(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3704
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3705
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3706
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3708
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3709
d(k) = mpas_add(qnwfa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnwfa(k), s_awqnwfa(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3710
}
// bl_mynn_subroutines.F90:3713
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3714
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3715
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3716
d(kte) = qnwfa(kte);
// bl_mynn_subroutines.F90:3719
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3722
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3724
qnwfa2(k) = x(k);
// bl_mynn_subroutines.F90:3725
}
// bl_mynn_subroutines.F90:3727
} else {
// bl_mynn_subroutines.F90:3729
for (int mynn_element=0; mynn_element < (qnwfa2).size(); ++mynn_element) { qnwfa2.linear(mynn_element) = qnwfa.linear(mynn_element); }
// bl_mynn_subroutines.F90:3730
}
// bl_mynn_subroutines.F90:3735
if (((((bl_mynn_cloudmix) && (flag_qnifa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3738
k = kts;
// bl_mynn_subroutines.F90:3740
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3741
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3743
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3744
d(k) = mpas_sub(qnifa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnifa(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3746
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3747
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3748
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3750
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3751
d(k) = mpas_add(qnifa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnifa(k), s_awqnifa(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3752
}
// bl_mynn_subroutines.F90:3755
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3756
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3757
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3758
d(kte) = qnifa(kte);
// bl_mynn_subroutines.F90:3761
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3764
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3766
qnifa2(k) = x(k);
// bl_mynn_subroutines.F90:3767
}
// bl_mynn_subroutines.F90:3769
} else {
// bl_mynn_subroutines.F90:3771
for (int mynn_element=0; mynn_element < (qnifa2).size(); ++mynn_element) { qnifa2.linear(mynn_element) = qnifa.linear(mynn_element); }
// bl_mynn_subroutines.F90:3772
}
// bl_mynn_subroutines.F90:3777
if (((((bl_mynn_cloudmix) && (flag_qnbca))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3780
k = kts;
// bl_mynn_subroutines.F90:3782
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3783
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3785
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3786
d(k) = mpas_sub(qnbca(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnbca(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3788
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3789
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3790
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3792
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3793
d(k) = mpas_add(qnbca(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnbca(k), s_awqnbca(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3794
}
// bl_mynn_subroutines.F90:3797
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3798
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3799
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3800
d(kte) = qnbca(kte);
// bl_mynn_subroutines.F90:3803
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3806
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3808
qnbca2(k) = x(k);
// bl_mynn_subroutines.F90:3809
}
// bl_mynn_subroutines.F90:3811
} else {
// bl_mynn_subroutines.F90:3813
for (int mynn_element=0; mynn_element < (qnbca2).size(); ++mynn_element) { qnbca2.linear(mynn_element) = qnbca.linear(mynn_element); }
// bl_mynn_subroutines.F90:3814
}
// bl_mynn_subroutines.F90:3819
if (flag_ozone) {
// bl_mynn_subroutines.F90:3820
k = kts;
// bl_mynn_subroutines.F90:3823
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3824
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3825
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3826
d(k) = ozone(k);
// bl_mynn_subroutines.F90:3828
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3829
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3830
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3831
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3832
d(k) = ozone(k);
// bl_mynn_subroutines.F90:3833
}
// bl_mynn_subroutines.F90:3836
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3837
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3838
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3839
d(kte) = ozone(kte);
// bl_mynn_subroutines.F90:3842
f_tridiag2(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:3845
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3847
dozone(k) = mpas_div(mpas_sub(x(k), ozone(k)), delt);
// bl_mynn_subroutines.F90:3848
}
// bl_mynn_subroutines.F90:3849
} else {
// bl_mynn_subroutines.F90:3850
for (int mynn_element=0; mynn_element < (dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]})).size(); ++mynn_element) { dozone.slice(MynnRange{dozone.lo[0], dozone.hi[0]}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:3851
}
// bl_mynn_subroutines.F90:3858
if (bl_mynn_mixqt) {
// bl_mynn_subroutines.F90:3859
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3861
th_new = mpas_add(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc(k))), mpas_mul(mpas_div(xlscp, exner(k)), sqi(k)));
// bl_mynn_subroutines.F90:3864
t = mpas_mul(th_new, exner(k));
// bl_mynn_subroutines.F90:3865
qsat = f_qsat_blend(t, t0c, tice, p(k), {});
// bl_mynn_subroutines.F90:3872
if (((((sqc(k)) > (0.0f))) || (((sqi(k)) > (0.0f))))) {
// bl_mynn_subroutines.F90:3873
sqv2(k) = mynn_min(((float)(sqw2(k))), ((float)(qsat)));
// bl_mynn_subroutines.F90:3874
portion_qc = mpas_div(sqc(k), mpas_add(sqc(k), sqi(k)));
// bl_mynn_subroutines.F90:3875
portion_qi = mpas_div(sqi(k), mpas_add(sqc(k), sqi(k)));
// bl_mynn_subroutines.F90:3876
condensate = mynn_max(((float)(mpas_sub(sqw2(k), qsat))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:3877
sqc2(k) = mpas_mul(condensate, portion_qc);
// bl_mynn_subroutines.F90:3878
sqi2(k) = mpas_mul(condensate, portion_qi);
// bl_mynn_subroutines.F90:3879
} else {
// bl_mynn_subroutines.F90:3880
sqv2(k) = sqw2(k);
// bl_mynn_subroutines.F90:3881
sqi2(k) = 0.0f;
// bl_mynn_subroutines.F90:3882
sqc2(k) = 0.0f;
// bl_mynn_subroutines.F90:3883
}
// bl_mynn_subroutines.F90:3884
}
// bl_mynn_subroutines.F90:3885
}
// bl_mynn_subroutines.F90:3891
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3892
dqv(k) = mpas_div(mpas_sub(sqv2(k), sqv(k)), delt);
// bl_mynn_subroutines.F90:3894
}
// bl_mynn_subroutines.F90:3896
if (bl_mynn_cloudmix) {
// bl_mynn_subroutines.F90:3901
if (flag_qc) {
// bl_mynn_subroutines.F90:3902
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3903
dqc(k) = mpas_div(mpas_sub(sqc2(k), sqc(k)), delt);
// bl_mynn_subroutines.F90:3905
}
// bl_mynn_subroutines.F90:3906
} else {
// bl_mynn_subroutines.F90:3907
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3908
dqc(k) = 0.f;
// bl_mynn_subroutines.F90:3909
}
// bl_mynn_subroutines.F90:3910
}
// bl_mynn_subroutines.F90:3915
if (((flag_qnc) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3916
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3917
dqnc(k) = mpas_div(mpas_sub(qnc2(k), qnc(k)), delt);
// bl_mynn_subroutines.F90:3919
}
// bl_mynn_subroutines.F90:3920
} else {
// bl_mynn_subroutines.F90:3921
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3922
dqnc(k) = 0.f;
// bl_mynn_subroutines.F90:3923
}
// bl_mynn_subroutines.F90:3924
}
// bl_mynn_subroutines.F90:3929
if (flag_qi) {
// bl_mynn_subroutines.F90:3930
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3931
dqi(k) = mpas_div(mpas_sub(sqi2(k), sqi(k)), delt);
// bl_mynn_subroutines.F90:3933
}
// bl_mynn_subroutines.F90:3934
} else {
// bl_mynn_subroutines.F90:3935
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3936
dqi(k) = 0.f;
// bl_mynn_subroutines.F90:3937
}
// bl_mynn_subroutines.F90:3938
}
// bl_mynn_subroutines.F90:3943
if (false) {
// bl_mynn_subroutines.F90:3944
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3945
dqs(k) = mpas_div(mpas_sub(sqs2(k), sqs(k)), delt);
// bl_mynn_subroutines.F90:3946
}
// bl_mynn_subroutines.F90:3947
} else {
// bl_mynn_subroutines.F90:3948
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3949
dqs(k) = 0.f;
// bl_mynn_subroutines.F90:3950
}
// bl_mynn_subroutines.F90:3951
}
// bl_mynn_subroutines.F90:3956
if (((flag_qni) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3957
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3958
dqni(k) = mpas_div(mpas_sub(qni2(k), qni(k)), delt);
// bl_mynn_subroutines.F90:3960
}
// bl_mynn_subroutines.F90:3961
} else {
// bl_mynn_subroutines.F90:3962
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3963
dqni(k) = 0.f;
// bl_mynn_subroutines.F90:3964
}
// bl_mynn_subroutines.F90:3965
}
// bl_mynn_subroutines.F90:3966
} else {
// bl_mynn_subroutines.F90:3968
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3969
dqc(k) = 0.f;
// bl_mynn_subroutines.F90:3970
dqnc(k) = 0.f;
// bl_mynn_subroutines.F90:3971
dqi(k) = 0.f;
// bl_mynn_subroutines.F90:3972
dqni(k) = 0.f;
// bl_mynn_subroutines.F90:3973
dqs(k) = 0.f;
// bl_mynn_subroutines.F90:3974
}
// bl_mynn_subroutines.F90:3975
}
// bl_mynn_subroutines.F90:3978
f_moisture_check(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl, dqv, dqc, dqi, dqs, dth);
// bl_mynn_subroutines.F90:3985
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3986
if (((mpas_add(mpas_mul(dozone(k), delt), ozone(k))) < (0.f))) {
// bl_mynn_subroutines.F90:3987
dozone(k) = mpas_div(mpas_mul(mynn_neg(ozone(k)), 0.99f), delt);
// bl_mynn_subroutines.F90:3988
}
// bl_mynn_subroutines.F90:3989
}
// bl_mynn_subroutines.F90:3994
if (flag_qi) {
// bl_mynn_subroutines.F90:3995
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3996
dth(k) = mpas_div(mpas_sub(mpas_add(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc2(k))), mpas_mul(mpas_div(xlscp, exner(k)), mpas_add(sqi2(k), sqs(k)))), th(k)), delt);
// bl_mynn_subroutines.F90:4004
}
// bl_mynn_subroutines.F90:4005
} else {
// bl_mynn_subroutines.F90:4006
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4007
dth(k) = mpas_div(mpas_sub(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc2(k))), th(k)), delt);
// bl_mynn_subroutines.F90:4012
}
// bl_mynn_subroutines.F90:4013
}
// bl_mynn_subroutines.F90:4018
if (((((flag_qnwfa) && (flag_qnifa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:4020
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4024
dqnwfa(k) = mpas_div(mpas_sub(qnwfa2(k), qnwfa(k)), delt);
// bl_mynn_subroutines.F90:4028
dqnifa(k) = mpas_div(mpas_sub(qnifa2(k), qnifa(k)), delt);
// bl_mynn_subroutines.F90:4029
}
// bl_mynn_subroutines.F90:4030
} else {
// bl_mynn_subroutines.F90:4031
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4032
dqnwfa(k) = 0.f;
// bl_mynn_subroutines.F90:4033
dqnifa(k) = 0.f;
// bl_mynn_subroutines.F90:4034
}
// bl_mynn_subroutines.F90:4035
}
// bl_mynn_subroutines.F90:4040
if (((flag_qnbca) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:4041
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4042
dqnbca(k) = mpas_div(mpas_sub(qnbca2(k), qnbca(k)), delt);
// bl_mynn_subroutines.F90:4043
}
// bl_mynn_subroutines.F90:4044
} else {
// bl_mynn_subroutines.F90:4045
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4046
dqnbca(k) = 0.f;
// bl_mynn_subroutines.F90:4047
}
// bl_mynn_subroutines.F90:4048
}
// bl_mynn_subroutines.F90:4057
if (debug_code) {
// bl_mynn_subroutines.F90:4058
problem = false;
// bl_mynn_subroutines.F90:4059
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4060
wsp = mynn_sqrt(mpas_add(mynn_ipow<2>(u(k)), mynn_ipow<2>(v(k))));
// bl_mynn_subroutines.F90:4061
wsp2 = mynn_sqrt(mpas_add(mynn_ipow<2>(mpas_add(u(k), mpas_mul(du(k), delt))), mynn_ipow<2>(mpas_add(v(k), mpas_mul(du(k), delt)))));
// bl_mynn_subroutines.F90:4062
th2 = mpas_add(th(k), mpas_mul(dth(k), delt));
// bl_mynn_subroutines.F90:4063
tk2 = mpas_mul(th2, exner(k));
// bl_mynn_subroutines.F90:4064
if (((((((wsp2) > (200.f))) || (((tk2) > (360.f))))) || (((tk2) < (160.f))))) {
// bl_mynn_subroutines.F90:4065
problem = true;
// bl_mynn_subroutines.F90:4066
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4067
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4068
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4069
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4070
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4071
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4072
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4073
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4074
kproblem = k;
// bl_mynn_subroutines.F90:4075
}
// bl_mynn_subroutines.F90:4076
}
// bl_mynn_subroutines.F90:4077
if (problem) {
// bl_mynn_subroutines.F90:4078
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4079
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4080
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4081
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4082
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4083
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:4084
}
// bl_mynn_subroutines.F90:4085
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4095-4182
__device__ __noinline__ void f_moisture_check(int kte, float delt, MynnArray<float, 1> dp, MynnArray<float, 1> exner, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qi, MynnArray<float, 1> qs, MynnArray<float, 1> th, MynnArray<float, 1> dqv, MynnArray<float, 1> dqc, MynnArray<float, 1> dqi, MynnArray<float, 1> dqs, MynnArray<float, 1> dth) {
MynnFrame mynn_frame(arena);
dp.rebase({1});
exner.rebase({1});
qv.rebase({1});
qc.rebase({1});
qi.rebase({1});
qs.rebase({1});
th.rebase({1});
dqv.rebase({1});
dqc.rebase({1});
dqi.rebase({1});
dqs.rebase({1});
dth.rebase({1});
int k;
float dqc2;
float dqi2;
float dqs2;
float dqv2;
float sum;
float aa;
float dum;
constexpr float qvmin = 1e-20f;
constexpr float qcmin = 0.0f;
constexpr float qimin = 0.0f;
// bl_mynn_subroutines.F90:4124
for (k=kte; ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// bl_mynn_subroutines.F90:4125
dqc2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qcmin, qc(k)))));
// bl_mynn_subroutines.F90:4126
dqi2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, qi(k)))));
// bl_mynn_subroutines.F90:4127
dqs2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, qs(k)))));
// bl_mynn_subroutines.F90:4130
dqc(k) = mpas_add(dqc(k), mpas_div(dqc2, delt));
// bl_mynn_subroutines.F90:4131
dqi(k) = mpas_add(dqi(k), mpas_div(dqi2, delt));
// bl_mynn_subroutines.F90:4132
dqs(k) = mpas_add(dqs(k), mpas_div(dqs2, delt));
// bl_mynn_subroutines.F90:4133
dqv(k) = mpas_sub(dqv(k), mpas_div(mpas_add(mpas_add(dqc2, dqi2), dqs2), delt));
// bl_mynn_subroutines.F90:4134
dth(k) = mpas_add(mpas_add(dth(k), mpas_mul(mpas_div(xlvcp, exner(k)), mpas_div(dqc2, delt))), mpas_mul(mpas_div(xlscp, exner(k)), mpas_div(mpas_add(dqi2, dqs2), delt)));
// bl_mynn_subroutines.F90:4137
qc(k) = mpas_add(qc(k), dqc2);
// bl_mynn_subroutines.F90:4138
qi(k) = mpas_add(qi(k), dqi2);
// bl_mynn_subroutines.F90:4139
qs(k) = mpas_add(qs(k), dqs2);
// bl_mynn_subroutines.F90:4140
qv(k) = mpas_sub(mpas_sub(mpas_sub(qv(k), dqc2), dqi2), dqs2);
// bl_mynn_subroutines.F90:4141
th(k) = mpas_add(mpas_add(th(k), mpas_mul(mpas_div(xlvcp, exner(k)), dqc2)), mpas_mul(mpas_div(xlscp, exner(k)), mpas_add(dqi2, dqs2)));
// bl_mynn_subroutines.F90:4145
dqv2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qvmin, qv(k)))));
// bl_mynn_subroutines.F90:4146
dqv(k) = mpas_add(dqv(k), mpas_div(dqv2, delt));
// bl_mynn_subroutines.F90:4147
qv(k) = mpas_add(qv(k), dqv2);
// bl_mynn_subroutines.F90:4148
if (((k) != (1))) {
// bl_mynn_subroutines.F90:4149
qv(((k) - (1))) = mpas_sub(qv(((k) - (1))), mpas_div(mpas_mul(dqv2, dp(k)), dp(((k) - (1)))));
// bl_mynn_subroutines.F90:4150
dqv(((k) - (1))) = mpas_sub(dqv(((k) - (1))), mpas_div(mpas_div(mpas_mul(dqv2, dp(k)), dp(((k) - (1)))), delt));
// bl_mynn_subroutines.F90:4151
}
// bl_mynn_subroutines.F90:4152
qv(k) = mynn_max(((float)(qv(k))), ((float)(qvmin)));
// bl_mynn_subroutines.F90:4153
qc(k) = mynn_max(((float)(qc(k))), ((float)(qcmin)));
// bl_mynn_subroutines.F90:4154
qi(k) = mynn_max(((float)(qi(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4155
qs(k) = mynn_max(((float)(qs(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4156
}
// bl_mynn_subroutines.F90:4160
if (((dqv2) > (1.e-20f))) {
// bl_mynn_subroutines.F90:4161
sum = 0.0f;
// bl_mynn_subroutines.F90:4162
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4163
if (((qv(k)) > (mpas_mul(2.0f, qvmin)))) { sum = mpas_add(sum, mpas_mul(qv(k), dp(k))); }
// bl_mynn_subroutines.F90:4164
}
// bl_mynn_subroutines.F90:4165
aa = mpas_div(mpas_mul(dqv2, dp(1)), mynn_max(((float)(1.e-20f)), ((float)(sum))));
// bl_mynn_subroutines.F90:4166
if (((aa) < (0.5f))) {
// bl_mynn_subroutines.F90:4167
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4168
if (((qv(k)) > (mpas_mul(2.0f, qvmin)))) {
// bl_mynn_subroutines.F90:4169
dum = mpas_mul(aa, qv(k));
// bl_mynn_subroutines.F90:4170
qv(k) = mpas_sub(qv(k), dum);
// bl_mynn_subroutines.F90:4171
dqv(k) = mpas_sub(dqv(k), mpas_div(dum, delt));
// bl_mynn_subroutines.F90:4172
}
// bl_mynn_subroutines.F90:4173
}
// bl_mynn_subroutines.F90:4174
} else {
// bl_mynn_subroutines.F90:4177
}
// bl_mynn_subroutines.F90:4178
}
// bl_mynn_subroutines.F90:4180
return;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4186-4312
__device__ __noinline__ void f_mynn_mix_chem(int kts, int kte, float delt, MynnArray<float, 1> dz, float pblh, int nchem, int kdvel, int ndvel, MynnArray<float, 2> chem1, MynnArray<float, 1> vd1, MynnArray<float, 1> rho, float flt, MynnArray<float, 1> tcd, MynnArray<float, 1> qcd, MynnArray<float, 1> dfh, MynnArray<float, 1> s_aw, MynnArray<float, 2> s_awchem, float emis_ant_no, float frp, bool rrfs_sd, bool enh_mix, bool smoke_dbg) {
MynnFrame mynn_frame(arena);
dz.rebase({kts});
chem1.rebase({kts, 1});
vd1.rebase({1});
rho.rebase({kts});
tcd.rebase({kts});
qcd.rebase({kts});
dfh.rebase({kts});
s_aw.rebase({kts});
s_awchem.rebase({kts, 1});
MynnArray<float, 1> dtz = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> a = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> b = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> c = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> d = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> x = mynn_array<float, 1>(arena, {kts}, {kte});
float rhs;
float dztop;
float t;
float dzk;
float hght;
float khdz_old;
float khdz_back;
int k;
int kk;
int kmaxfire;
int ic;
int icall;
MynnArray<float, 1> rhoinv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> khdz = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
constexpr float no_threshold = 10.0f;
constexpr float frp_threshold = 10.0f;
constexpr float pblh_threshold = 100.0f;
// bl_mynn_subroutines.F90:4229
dztop = mpas_mul(.5f, mpas_add(dz(kte), dz(((kte) - (1)))));
// bl_mynn_subroutines.F90:4231
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4232
dtz(k) = mpas_div(delt, dz(k));
// bl_mynn_subroutines.F90:4233
}
// bl_mynn_subroutines.F90:4237
rhoz(kts) = rho(kts);
// bl_mynn_subroutines.F90:4238
rhoinv(kts) = mpas_div(1.f, rho(kts));
// bl_mynn_subroutines.F90:4239
khdz(kts) = mpas_mul(rhoz(kts), dfh(kts));
// bl_mynn_subroutines.F90:4241
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4242
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) - (1)))), mpas_mul(rho(((k) - (1))), dz(k))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:4243
rhoz(k) = mynn_max(((float)(rhoz(k))), ((float)(1e-4f)));
// bl_mynn_subroutines.F90:4244
rhoinv(k) = mpas_div(1.f, mynn_max(((float)(rho(k))), ((float)(1e-4f))));
// bl_mynn_subroutines.F90:4245
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:4246
khdz(k) = mpas_mul(rhoz(k), dfh(k));
// bl_mynn_subroutines.F90:4247
}
// bl_mynn_subroutines.F90:4248
rhoz(((kte) + (1))) = rhoz(kte);
// bl_mynn_subroutines.F90:4249
khdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfh(kte));
// bl_mynn_subroutines.F90:4252
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4253
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:4254
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:4255
}
// bl_mynn_subroutines.F90:4258
if (((rrfs_sd) && (enh_mix))) {
// bl_mynn_subroutines.F90:4259
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4260
khdz_old = khdz(k);
// bl_mynn_subroutines.F90:4261
khdz_back = mpas_div(mpas_mul(pblh, 0.15f), dz(k));
// bl_mynn_subroutines.F90:4263
if (((pblh) < (pblh_threshold))) {
// bl_mynn_subroutines.F90:4264
if (((emis_ant_no) > (no_threshold))) {
// bl_mynn_subroutines.F90:4265
khdz(k) = mynn_max(((float)(mpas_mul(1.1f, khdz(k)))), ((float)(mpas_mul(mpas_div(mynn_sqrt(mpas_div(emis_ant_no, no_threshold)), dz(k)), rhoz(k)))));
// bl_mynn_subroutines.F90:4267
}
// bl_mynn_subroutines.F90:4268
if (((frp) > (frp_threshold))) {
// bl_mynn_subroutines.F90:4269
kmaxfire = mynn_ceiling(mynn_log(frp));
// bl_mynn_subroutines.F90:4270
khdz(k) = mynn_max(((float)(mpas_mul(1.1f, khdz(k)))), ((float)(mpas_mul(mpas_div(mpas_mul(mpas_sub(1.f, mpas_div(k, mpas_mul(kmaxfire, 2.f))), mpas_sub(mynn_ipow<2>(mynn_log(frp)), mpas_mul(2.f, mynn_log(frp)))), dz(k)), rhoz(k)))));
// bl_mynn_subroutines.F90:4272
}
// bl_mynn_subroutines.F90:4273
}
// bl_mynn_subroutines.F90:4274
}
// bl_mynn_subroutines.F90:4275
}
// bl_mynn_subroutines.F90:4281
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:4282
k = kts;
// bl_mynn_subroutines.F90:4284
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:4285
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:4286
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:4287
d(k) = mpas_sub(mpas_sub(chem1(k, ic), mpas_mul(mpas_mul(dtz(k), vd1(ic)), chem1(k, ic))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awchem(((k) + (1)), ic)));
// bl_mynn_subroutines.F90:4291
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4292
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)));
// bl_mynn_subroutines.F90:4293
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:4295
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:4296
d(k) = mpas_add(chem1(k, ic), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awchem(k, ic), s_awchem(((k) + (1)), ic))));
// bl_mynn_subroutines.F90:4297
}
// bl_mynn_subroutines.F90:4300
a(kte) = 0.f;
// bl_mynn_subroutines.F90:4301
b(kte) = 1.f;
// bl_mynn_subroutines.F90:4302
c(kte) = 0.f;
// bl_mynn_subroutines.F90:4303
d(kte) = chem1(kte, ic);
// bl_mynn_subroutines.F90:4305
f_tridiag3(kte, a, b, c, d, x);
// bl_mynn_subroutines.F90:4307
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4308
chem1(k, ic) = x(k);
// bl_mynn_subroutines.F90:4309
}
// bl_mynn_subroutines.F90:4310
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4316-4340
__device__ __noinline__ void f_retrieve_exchange_coeffs(int kts, int kte, MynnArray<float, 1> dfm, MynnArray<float, 1> dfh, MynnArray<float, 1> dz, MynnArray<float, 1> k_m, MynnArray<float, 1> k_h) {
MynnFrame mynn_frame(arena);
dfm.rebase({kts});
dfh.rebase({kts});
dz.rebase({kts});
k_m.rebase({kts});
k_h.rebase({kts});
int k;
float dzk;
// bl_mynn_subroutines.F90:4331
k_m(kts) = 0.f;
// bl_mynn_subroutines.F90:4332
k_h(kts) = 0.f;
// bl_mynn_subroutines.F90:4334
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4335
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:4336
k_m(k) = mpas_mul(dfm(k), dzk);
// bl_mynn_subroutines.F90:4337
k_h(k) = mpas_mul(dfh(k), dzk);
// bl_mynn_subroutines.F90:4338
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4344-4376
__device__ __noinline__ void f_tridiag(int n, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
int i;
float p;
MynnArray<float, 1> q = mynn_array<float, 1>(arena, {1}, {n});
// bl_mynn_subroutines.F90:4362
c(n) = 0.f;
// bl_mynn_subroutines.F90:4363
q(1) = mpas_div(mynn_neg(c(1)), b(1));
// bl_mynn_subroutines.F90:4364
d(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4366
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4367
p = mpas_div(1.f, mpas_add(b(i), mpas_mul(a(i), q(((i) - (1))))));
// bl_mynn_subroutines.F90:4368
q(i) = mpas_mul(mynn_neg(c(i)), p);
// bl_mynn_subroutines.F90:4369
d(i) = mpas_mul(mpas_sub(d(i), mpas_mul(a(i), d(((i) - (1))))), p);
// bl_mynn_subroutines.F90:4370
}
// bl_mynn_subroutines.F90:4372
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4373
d(i) = mpas_add(d(i), mpas_mul(q(i), d(((i) + (1)))));
// bl_mynn_subroutines.F90:4374
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4380-4412
__device__ __noinline__ void f_tridiag2(int n, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
MynnArray<float, 1> cp = mynn_array<float, 1>(arena, {1}, {n});
MynnArray<float, 1> dp = mynn_array<float, 1>(arena, {1}, {n});
float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4415-4453
__device__ __noinline__ void f_tridiag3(int kte, MynnArray<float, 1> a, MynnArray<float, 1> b, MynnArray<float, 1> c, MynnArray<float, 1> d, MynnArray<float, 1> x) {
MynnFrame mynn_frame(arena);
a.rebase({1});
b.rebase({1});
c.rebase({1});
d.rebase({1});
x.rebase({1});
constexpr int kts = 1;
int in;
// bl_mynn_subroutines.F90:4439
for (in=((kte) - (1)); ((-(1))) > 0 ? in <= (kts) : in >= (kts); in += ((-(1)))) {
// bl_mynn_subroutines.F90:4440
d(in) = mpas_sub(d(in), mpas_div(mpas_mul(c(in), d(((in) + (1)))), b(((in) + (1)))));
// bl_mynn_subroutines.F90:4441
b(in) = mpas_sub(b(in), mpas_div(mpas_mul(c(in), a(((in) + (1)))), b(((in) + (1)))));
// bl_mynn_subroutines.F90:4442
}
// bl_mynn_subroutines.F90:4444
for (in=((kts) + (1)); (1) > 0 ? in <= (kte) : in >= (kte); in += (1)) {
// bl_mynn_subroutines.F90:4445
d(in) = mpas_sub(d(in), mpas_div(mpas_mul(a(in), d(((in) - (1)))), b(((in) - (1)))));
// bl_mynn_subroutines.F90:4446
}
// bl_mynn_subroutines.F90:4448
for (in=kts; (1) > 0 ? in <= (kte) : in >= (kte); in += (1)) {
// bl_mynn_subroutines.F90:4449
x(in) = mpas_div(d(in), b(in));
// bl_mynn_subroutines.F90:4450
}
// bl_mynn_subroutines.F90:4452
return;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4475-4615
__device__ __noinline__ void f_get_pblh(int kts, int kte, float& zi, MynnArray<float, 1> thetav1d, MynnArray<float, 1> qke1d, MynnArray<float, 1> zw1d, MynnArray<float, 1> dz1d, float landsea, int& kzi) {
MynnFrame mynn_frame(arena);
thetav1d.rebase({kts});
qke1d.rebase({kts});
zw1d.rebase({kts});
dz1d.rebase({kts});
float pblh_tke;
float qtke;
float qtkem1;
float wt;
float maxqke;
float tkeeps;
float minthv;
float delt_thv;
constexpr float sbl_lim = 200.f;
constexpr float sbl_damp = 400.f;
int i;
int j;
int k;
int kthv;
int ktke;
// bl_mynn_subroutines.F90:4513
kzi = 2;
// bl_mynn_subroutines.F90:4516
k = ((kts) + (1));
// bl_mynn_subroutines.F90:4517
kthv = 1;
// bl_mynn_subroutines.F90:4518
minthv = 9.e9f;
// bl_mynn_subroutines.F90:4519
while (((zw1d(k)) <= (200.f))) {
// bl_mynn_subroutines.F90:4521
if (((minthv) > (thetav1d(k)))) {
// bl_mynn_subroutines.F90:4522
minthv = thetav1d(k);
// bl_mynn_subroutines.F90:4523
kthv = k;
// bl_mynn_subroutines.F90:4524
}
// bl_mynn_subroutines.F90:4525
k = ((k) + (1));
// bl_mynn_subroutines.F90:4527
}
// bl_mynn_subroutines.F90:4530
zi = 0.f;
// bl_mynn_subroutines.F90:4531
k = ((kthv) + (1));
// bl_mynn_subroutines.F90:4532
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:4534
delt_thv = 1.0f;
// bl_mynn_subroutines.F90:4535
} else {
// bl_mynn_subroutines.F90:4537
delt_thv = 1.25f;
// bl_mynn_subroutines.F90:4538
}
// bl_mynn_subroutines.F90:4540
zi = 0.f;
// bl_mynn_subroutines.F90:4541
k = ((kthv) + (1));
// bl_mynn_subroutines.F90:4543
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4544
if (((thetav1d(k)) >= (mpas_add(minthv, delt_thv)))) {
// bl_mynn_subroutines.F90:4545
zi = mpas_sub(zw1d(k), mpas_mul(dz1d(((k) - (1))), mynn_min(((float)(mpas_div(mpas_sub(thetav1d(k), mpas_add(minthv, delt_thv)), mynn_max(((float)(mpas_sub(thetav1d(k), thetav1d(((k) - (1)))))), ((float)(1e-6f)))))), ((float)(1.0f)))));
// bl_mynn_subroutines.F90:4548
}
// bl_mynn_subroutines.F90:4550
if (((k) == (((kte) - (1))))) { zi = zw1d(((kts) + (1))); }
// bl_mynn_subroutines.F90:4551
if (((zi) != (0.0f))) { break; }
// bl_mynn_subroutines.F90:4552
}
// bl_mynn_subroutines.F90:4559
ktke = 1;
// bl_mynn_subroutines.F90:4560
maxqke = mynn_max(((float)(qke1d(kts))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4563
tkeeps = mpas_div(maxqke, 40.f);
// bl_mynn_subroutines.F90:4564
tkeeps = mynn_max(((float)(tkeeps)), ((float)(0.02f)));
// bl_mynn_subroutines.F90:4565
pblh_tke = 0.f;
// bl_mynn_subroutines.F90:4567
k = ((ktke) + (1));
// bl_mynn_subroutines.F90:4569
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4571
qtke = mynn_max(((float)(mpas_div(qke1d(k), 2.f))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4572
qtkem1 = mynn_max(((float)(mpas_div(qke1d(((k) - (1))), 2.f))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4573
if (((qtke) <= (tkeeps))) {
// bl_mynn_subroutines.F90:4574
pblh_tke = mpas_sub(zw1d(k), mpas_mul(dz1d(((k) - (1))), mynn_min(((float)(mpas_div(mpas_sub(tkeeps, qtke), mynn_max(((float)(mpas_sub(qtkem1, qtke))), ((float)(1e-6f)))))), ((float)(1.0f)))));
// bl_mynn_subroutines.F90:4577
pblh_tke = mynn_max(((float)(pblh_tke)), ((float)(zw1d(((kts) + (1))))));
// bl_mynn_subroutines.F90:4579
}
// bl_mynn_subroutines.F90:4581
if (((k) == (((kte) - (1))))) { pblh_tke = zw1d(((kts) + (1))); }
// bl_mynn_subroutines.F90:4582
if (((pblh_tke) != (0.f))) { break; }
// bl_mynn_subroutines.F90:4583
}
// bl_mynn_subroutines.F90:4591
pblh_tke = mynn_min(((float)(pblh_tke)), ((float)(mpas_add(zi, 350.f))));
// bl_mynn_subroutines.F90:4592
pblh_tke = mynn_max(((float)(pblh_tke)), ((float)(mynn_max(((float)(mpas_sub(zi, 350.f))), ((float)(10.f))))));
// bl_mynn_subroutines.F90:4594
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zi, sbl_lim), sbl_damp))), .5f);
// bl_mynn_subroutines.F90:4595
if (((maxqke) <= (0.05f))) {
// bl_mynn_subroutines.F90:4597
} else {
// bl_mynn_subroutines.F90:4599
zi = mpas_add(mpas_mul(pblh_tke, mpas_sub(1.f, wt)), mpas_mul(zi, wt));
// bl_mynn_subroutines.F90:4600
}
// bl_mynn_subroutines.F90:4603
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4604
if (((zw1d(k)) >= (zi))) {
// bl_mynn_subroutines.F90:4605
kzi = ((k) - (1));
// bl_mynn_subroutines.F90:4606
break;
// bl_mynn_subroutines.F90:4607
}
// bl_mynn_subroutines.F90:4608
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:4637-5785
__device__ __noinline__ void f_dmp_mf(int ii, int kts, int kte, float dt, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> p, MynnArray<float, 1> rho, bool momentum_opt, bool tke_opt, bool scalar_opt, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> w, MynnArray<float, 1> th, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> tk, MynnArray<float, 1> qt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> qke, MynnArray<float, 1> qnc, MynnArray<float, 1> qni, MynnArray<float, 1> qnwfa, MynnArray<float, 1> qnifa, MynnArray<float, 1> qnbca, MynnArray<float, 1> exner, MynnArray<float, 1> vt, MynnArray<float, 1> vq, MynnArray<float, 1> sgm, float ust, float flt, float fltv, float flq, float flqv, float pblh, int kpbl, float dx, float landsea, float ts, MynnArray<float, 1> edmf_a, MynnArray<float, 1> edmf_w, MynnArray<float, 1> edmf_qt, MynnArray<float, 1> edmf_thl, MynnArray<float, 1> edmf_ent, MynnArray<float, 1> edmf_qc, MynnArray<float, 1> s_aw, MynnArray<float, 1> s_awthl, MynnArray<float, 1> s_awqt, MynnArray<float, 1> s_awqv, MynnArray<float, 1> s_awqc, MynnArray<float, 1> s_awu, MynnArray<float, 1> s_awv, MynnArray<float, 1> s_awqke, MynnArray<float, 1> s_awqnc, MynnArray<float, 1> s_awqni, MynnArray<float, 1> s_awqnwfa, MynnArray<float, 1> s_awqnifa, MynnArray<float, 1> s_awqnbca, MynnArray<float, 1> sub_thl, MynnArray<float, 1> sub_sqv, MynnArray<float, 1> sub_u, MynnArray<float, 1> sub_v, MynnArray<float, 1> det_thl, MynnArray<float, 1> det_sqv, MynnArray<float, 1> det_sqc, MynnArray<float, 1> det_u, MynnArray<float, 1> det_v, int nchem, MynnArray<float, 2> chem1, MynnArray<float, 2> s_awchem, bool mix_chem, MynnArray<float, 1> qc_bl1d, MynnArray<float, 1> cldfra_bl1d, MynnArray<float, 1> qc_bl1d_old, MynnArray<float, 1> cldfra_bl1d_old, MynnOptional<bool> f_qc, MynnOptional<bool> f_qi, MynnOptional<bool> f_qnc, MynnOptional<bool> f_qni, MynnOptional<bool> f_qnwfa, MynnOptional<bool> f_qnifa, MynnOptional<bool> f_qnbca, float psig_shcu, float& maxwidth, int& ktop, float& maxmf, float& ztop, int spp_pbl, MynnArray<float, 1> rstoch_col) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
p.rebase({kts});
rho.rebase({kts});
u.rebase({kts});
v.rebase({kts});
w.rebase({kts});
th.rebase({kts});
thl.rebase({kts});
thv.rebase({kts});
tk.rebase({kts});
qt.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
qke.rebase({kts});
qnc.rebase({kts});
qni.rebase({kts});
qnwfa.rebase({kts});
qnifa.rebase({kts});
qnbca.rebase({kts});
exner.rebase({kts});
vt.rebase({kts});
vq.rebase({kts});
sgm.rebase({kts});
edmf_a.rebase({kts});
edmf_w.rebase({kts});
edmf_qt.rebase({kts});
edmf_thl.rebase({kts});
edmf_ent.rebase({kts});
edmf_qc.rebase({kts});
s_aw.rebase({kts});
s_awthl.rebase({kts});
s_awqt.rebase({kts});
s_awqv.rebase({kts});
s_awqc.rebase({kts});
s_awu.rebase({kts});
s_awv.rebase({kts});
s_awqke.rebase({kts});
s_awqnc.rebase({kts});
s_awqni.rebase({kts});
s_awqnwfa.rebase({kts});
s_awqnifa.rebase({kts});
s_awqnbca.rebase({kts});
sub_thl.rebase({kts});
sub_sqv.rebase({kts});
sub_u.rebase({kts});
sub_v.rebase({kts});
det_thl.rebase({kts});
det_sqv.rebase({kts});
det_sqc.rebase({kts});
det_u.rebase({kts});
det_v.rebase({kts});
chem1.rebase({1, 1});
s_awchem.rebase({kts, 1});
qc_bl1d.rebase({kts});
cldfra_bl1d.rebase({kts});
qc_bl1d_old.rebase({kts});
cldfra_bl1d_old.rebase({kts});
rstoch_col.rebase({kts});
MynnArray<float, 1> edmf_th = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> s_aw2 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
constexpr int nup = 8;
constexpr int debug_mf = 0;
float nup2;
MynnArray<float, 2> upw = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upthl = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqt = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqc = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqv = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upa = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upu = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upv = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upthv = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqke = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnc = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqni = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnwfa = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnifa = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> upqnbca = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nup});
MynnArray<float, 2> ent = mynn_array<float, 2>(arena, {kts, 1}, {kte, nup});
MynnArray<float, 2> entf = mynn_array<float, 2>(arena, {kts, 1}, {kte, nup});
MynnArray<int, 2> enti = mynn_array<int, 2>(arena, {kts, 1}, {kte, nup});
int k;
int i;
int k50;
float fltv2;
float wstar;
float qstar;
float thstar;
float sigmaw;
float sigmaqt;
float sigmath;
float z0;
float pwmin;
float pwmax;
float wmin;
float wmax;
float wlv;
float psig_w;
float maxw;
float maxqc;
float wpbl;
float b;
float qtn;
float thln;
float thvn;
float qcn;
float un;
float vn;
float qken;
float qncn;
float qnin;
float qnwfan;
float qnifan;
float qnbcan;
float wn2;
float wn;
float entexp;
float entexm;
float entw;
float bcoeff;
float thvkm1;
float thvk;
float pk;
float rho_int;
constexpr float wa = ((2.f) / (3.f));
constexpr float wb = 0.002f;
constexpr float wc = 1.5f;
constexpr float l0 = 100.f;
constexpr float ent0 = 0.1f;
constexpr float atot = 0.10f;
constexpr float lmax = 1000.f;
constexpr float lmin = 300.f;
constexpr float dlmin = 0.f;
float minwidth;
float dl;
constexpr float dcut = 1.2f;
float d;
float cn;
float c;
float l;
float n;
float an2;
float hux;
float wspd_pbl;
float cloud_base;
float width_flx;
MynnArray<float, 1> chemn = mynn_array<float, 1>(arena, {1}, {nchem});
MynnArray<float, 3> upchem = mynn_array<float, 3>(arena, {kts, 1, 1}, {((kte) + (1)), nup, nchem});
int ic;
MynnArray<float, 2> edmf_chem = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nchem});
float erf;
bool superadiabatic;
float sigq;
float xl;
float rsl;
float cpm;
float a;
float qmq;
float mf_cf;
float aup;
float q1;
float diffqt;
float qsat_tk;
float fng;
float qww;
float alpha;
float beta;
float bb;
float f;
float pt;
float t;
float q2p;
float b9;
float satvp;
float rhgrid;
float ac_mf;
float ac_strat;
float qc_mf;
constexpr float cf_thresh = 0.5f;
MynnArray<float, 1> exneri = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dzi = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rhoz = mynn_array<float, 1>(arena, {kts}, {kte});
float thp;
float qtp;
float qcp;
float qcs;
float esat;
float qsl;
float csigma;
float acfac;
float ac_wsp;
int overshoot;
float bvf;
float frz;
float dzp;
float adjustment;
float flx1;
constexpr float fluxportion = 0.75f;
MynnArray<float, 1> envm_a = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_w = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_thl = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_sqv = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_sqc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_u = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envm_v = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> envi_a = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> envi_w = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
float temp;
float sublim;
float qc_ent;
float qv_ent;
float qt_ent;
float thl_ent;
float detrate;
float detrateuv;
float oow;
float exc_fac;
float aratio;
float detturb;
float qc_grid;
float qc_sgs;
float qc_plume;
float exc_heat;
float exc_moist;
float tk_int;
float tvs;
constexpr float cdet = ((1.f) / (45.f));
constexpr float dzpmax = 300.f;
constexpr float csub = 0.25f;
constexpr float pgfac = 0.00f;
float uk;
float ukm1;
float vk;
float vkm1;
float dxsa;
// bl_mynn_subroutines.F90:4836
for (int mynn_element=0; mynn_element < (upw).size(); ++mynn_element) { upw.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4837
for (int mynn_element=0; mynn_element < (upthl).size(); ++mynn_element) { upthl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4838
for (int mynn_element=0; mynn_element < (upthv).size(); ++mynn_element) { upthv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4839
for (int mynn_element=0; mynn_element < (upqt).size(); ++mynn_element) { upqt.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4840
for (int mynn_element=0; mynn_element < (upa).size(); ++mynn_element) { upa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4841
for (int mynn_element=0; mynn_element < (upu).size(); ++mynn_element) { upu.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4842
for (int mynn_element=0; mynn_element < (upv).size(); ++mynn_element) { upv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4843
for (int mynn_element=0; mynn_element < (upqc).size(); ++mynn_element) { upqc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4844
for (int mynn_element=0; mynn_element < (upqv).size(); ++mynn_element) { upqv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4845
for (int mynn_element=0; mynn_element < (upqke).size(); ++mynn_element) { upqke.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4846
for (int mynn_element=0; mynn_element < (upqnc).size(); ++mynn_element) { upqnc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4847
for (int mynn_element=0; mynn_element < (upqni).size(); ++mynn_element) { upqni.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4848
for (int mynn_element=0; mynn_element < (upqnwfa).size(); ++mynn_element) { upqnwfa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4849
for (int mynn_element=0; mynn_element < (upqnifa).size(); ++mynn_element) { upqnifa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4850
for (int mynn_element=0; mynn_element < (upqnbca).size(); ++mynn_element) { upqnbca.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4851
if (mix_chem) {
// bl_mynn_subroutines.F90:4852
for (int mynn_element=0; mynn_element < (upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem})).size(); ++mynn_element) { upchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nup}, MynnRange{1, nchem}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:4853
}
// bl_mynn_subroutines.F90:4855
for (int mynn_element=0; mynn_element < (ent).size(); ++mynn_element) { ent.linear(mynn_element) = 0.001f; }
// bl_mynn_subroutines.F90:4857
for (int mynn_element=0; mynn_element < (edmf_a).size(); ++mynn_element) { edmf_a.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4858
for (int mynn_element=0; mynn_element < (edmf_w).size(); ++mynn_element) { edmf_w.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4859
for (int mynn_element=0; mynn_element < (edmf_qt).size(); ++mynn_element) { edmf_qt.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4860
for (int mynn_element=0; mynn_element < (edmf_thl).size(); ++mynn_element) { edmf_thl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4861
for (int mynn_element=0; mynn_element < (edmf_ent).size(); ++mynn_element) { edmf_ent.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4862
for (int mynn_element=0; mynn_element < (edmf_qc).size(); ++mynn_element) { edmf_qc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4863
if (mix_chem) {
// bl_mynn_subroutines.F90:4864
for (int mynn_element=0; mynn_element < (edmf_chem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); ++mynn_element) { edmf_chem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:4865
}
// bl_mynn_subroutines.F90:4868
for (int mynn_element=0; mynn_element < (s_aw).size(); ++mynn_element) { s_aw.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4869
for (int mynn_element=0; mynn_element < (s_awthl).size(); ++mynn_element) { s_awthl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4870
for (int mynn_element=0; mynn_element < (s_awqt).size(); ++mynn_element) { s_awqt.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4871
for (int mynn_element=0; mynn_element < (s_awqv).size(); ++mynn_element) { s_awqv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4872
for (int mynn_element=0; mynn_element < (s_awqc).size(); ++mynn_element) { s_awqc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4873
for (int mynn_element=0; mynn_element < (s_awu).size(); ++mynn_element) { s_awu.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4874
for (int mynn_element=0; mynn_element < (s_awv).size(); ++mynn_element) { s_awv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4875
for (int mynn_element=0; mynn_element < (s_awqke).size(); ++mynn_element) { s_awqke.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4876
for (int mynn_element=0; mynn_element < (s_awqnc).size(); ++mynn_element) { s_awqnc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4877
for (int mynn_element=0; mynn_element < (s_awqni).size(); ++mynn_element) { s_awqni.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4878
for (int mynn_element=0; mynn_element < (s_awqnwfa).size(); ++mynn_element) { s_awqnwfa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4879
for (int mynn_element=0; mynn_element < (s_awqnifa).size(); ++mynn_element) { s_awqnifa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4880
for (int mynn_element=0; mynn_element < (s_awqnbca).size(); ++mynn_element) { s_awqnbca.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4881
if (mix_chem) {
// bl_mynn_subroutines.F90:4882
for (int mynn_element=0; mynn_element < (s_awchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem})).size(); ++mynn_element) { s_awchem.slice(MynnRange{kts, ((kte) + (1))}, MynnRange{1, nchem}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:4883
}
// bl_mynn_subroutines.F90:4886
for (int mynn_element=0; mynn_element < (sub_thl).size(); ++mynn_element) { sub_thl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4887
for (int mynn_element=0; mynn_element < (sub_sqv).size(); ++mynn_element) { sub_sqv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4888
for (int mynn_element=0; mynn_element < (sub_u).size(); ++mynn_element) { sub_u.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4889
for (int mynn_element=0; mynn_element < (sub_v).size(); ++mynn_element) { sub_v.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4890
for (int mynn_element=0; mynn_element < (det_thl).size(); ++mynn_element) { det_thl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4891
for (int mynn_element=0; mynn_element < (det_sqv).size(); ++mynn_element) { det_sqv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4892
for (int mynn_element=0; mynn_element < (det_sqc).size(); ++mynn_element) { det_sqc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4893
for (int mynn_element=0; mynn_element < (det_u).size(); ++mynn_element) { det_u.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4894
for (int mynn_element=0; mynn_element < (det_v).size(); ++mynn_element) { det_v.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:4895
nup2 = nup;
// bl_mynn_subroutines.F90:4899
maxw = 0.0f;
// bl_mynn_subroutines.F90:4900
cloud_base = 9000.0f;
// bl_mynn_subroutines.F90:4901
for (k=1; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4902
if (((zw(k)) > (mpas_add(pblh, 500.f)))) { break; }
// bl_mynn_subroutines.F90:4904
wpbl = w(k);
// bl_mynn_subroutines.F90:4905
if (((w(k)) < (0.f))) { wpbl = mpas_mul(2.f, w(k)); }
// bl_mynn_subroutines.F90:4906
maxw = mynn_max(((float)(maxw)), ((float)(mynn_abs(wpbl))));
// bl_mynn_subroutines.F90:4909
if (((zw(k)) <= (50.f))) { k50 = k; }
// bl_mynn_subroutines.F90:4912
qc_sgs = mynn_max(((float)(qc(k))), ((float)(qc_bl1d(k))));
// bl_mynn_subroutines.F90:4913
if (((((((qc_sgs) > (1e-5f))) && (((cldfra_bl1d(k)) >= (0.5f))))) && (((cloud_base) == (9000.0f))))) {
// bl_mynn_subroutines.F90:4914
cloud_base = mpas_mul(0.5f, mpas_add(zw(k), zw(((k) + (1)))));
// bl_mynn_subroutines.F90:4915
}
// bl_mynn_subroutines.F90:4916
}
// bl_mynn_subroutines.F90:4919
maxw = mynn_max(((float)(0.f)), ((float)(mpas_sub(maxw, 1.0f))));
// bl_mynn_subroutines.F90:4920
psig_w = mynn_max(((float)(0.0f)), ((float)(mpas_sub(1.0f, maxw))));
// bl_mynn_subroutines.F90:4921
psig_w = mynn_min(((float)(psig_w)), ((float)(psig_shcu)));
// bl_mynn_subroutines.F90:4924
fltv2 = fltv;
// bl_mynn_subroutines.F90:4925
if (((((psig_w) == (0.0f))) && (((fltv) > (0.0f))))) { fltv2 = mpas_mul(mynn_neg(1.f), fltv); }
// bl_mynn_subroutines.F90:4929
superadiabatic = false;
// bl_mynn_subroutines.F90:4930
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:4931
hux = mynn_neg(0.001f);
// bl_mynn_subroutines.F90:4932
} else {
// bl_mynn_subroutines.F90:4933
hux = mynn_neg(0.005f);
// bl_mynn_subroutines.F90:4934
}
// bl_mynn_subroutines.F90:4935
tvs = mpas_mul(ts, mpas_add(1.0f, mpas_mul(p608, qv(kts))));
// bl_mynn_subroutines.F90:4936
for (k=1; (1) > 0 ? k <= (mynn_max(((int)(1)), ((int)(((k50) - (1)))))) : k >= (mynn_max(((int)(1)), ((int)(((k50) - (1)))))); k += (1)) {
// bl_mynn_subroutines.F90:4937
if (((k) == (1))) {
// bl_mynn_subroutines.F90:4938
if (((mpas_div(mpas_sub(thv(k), tvs), mpas_mul(0.5f, dz(k)))) < (hux))) {
// bl_mynn_subroutines.F90:4939
superadiabatic = true;
// bl_mynn_subroutines.F90:4940
} else {
// bl_mynn_subroutines.F90:4941
superadiabatic = false;
// bl_mynn_subroutines.F90:4942
break;
// bl_mynn_subroutines.F90:4943
}
// bl_mynn_subroutines.F90:4944
} else {
// bl_mynn_subroutines.F90:4945
if (((mpas_div(mpas_sub(thv(k), thv(((k) - (1)))), mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1))))))) < (hux))) {
// bl_mynn_subroutines.F90:4946
superadiabatic = true;
// bl_mynn_subroutines.F90:4947
} else {
// bl_mynn_subroutines.F90:4948
superadiabatic = false;
// bl_mynn_subroutines.F90:4949
break;
// bl_mynn_subroutines.F90:4950
}
// bl_mynn_subroutines.F90:4951
}
// bl_mynn_subroutines.F90:4952
}
// bl_mynn_subroutines.F90:4963
maxwidth = mynn_min(((float)(mpas_mul(dx, dcut))), ((float)(lmax)));
// bl_mynn_subroutines.F90:4965
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(1.1f, pblh))));
// bl_mynn_subroutines.F90:4967
if (((mpas_sub(landsea, 1.5f)) < (0))) {
// bl_mynn_subroutines.F90:4968
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(0.5f, cloud_base))));
// bl_mynn_subroutines.F90:4969
} else {
// bl_mynn_subroutines.F90:4970
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(0.9f, cloud_base))));
// bl_mynn_subroutines.F90:4971
}
// bl_mynn_subroutines.F90:4973
wspd_pbl = mynn_sqrt(mynn_max(((float)(mpas_add(mynn_ipow<2>(u(kts)), mynn_ipow<2>(v(kts))))), ((float)(0.01f))));
// bl_mynn_subroutines.F90:4976
if (((mpas_sub(landsea, 1.5f)) < (0))) {
// bl_mynn_subroutines.F90:4977
width_flx = mynn_max(((float)(mynn_min(((float)(mpas_mul(1000.f, mpas_add(mpas_mul(0.6f, mynn_tanh(mpas_div(mpas_sub(fltv, 0.040f), 0.04f))), .5f)))), ((float)(1000.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4978
} else {
// bl_mynn_subroutines.F90:4979
width_flx = mynn_max(((float)(mynn_min(((float)(mpas_mul(1000.f, mpas_add(mpas_mul(0.6f, mynn_tanh(mpas_div(mpas_sub(fltv, 0.007f), 0.02f))), .5f)))), ((float)(1000.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4980
}
// bl_mynn_subroutines.F90:4981
maxwidth = mynn_min(((float)(maxwidth)), ((float)(width_flx)));
// bl_mynn_subroutines.F90:4982
minwidth = lmin;
// bl_mynn_subroutines.F90:4985
if (((((maxwidth) >= (mpas_sub(lmax, 1.0f)))) && (((fltv) > (0.2f))))) { minwidth = mpas_add(lmin, mpas_mul(dlmin, mynn_min(((float)(mpas_div(mpas_sub(fltv, 0.2f), 0.3f))), ((float)(1.f))))); }
// bl_mynn_subroutines.F90:4987
if (((maxwidth) <= (minwidth))) {
// bl_mynn_subroutines.F90:4988
nup2 = 0;
// bl_mynn_subroutines.F90:4989
maxwidth = 0.0f;
// bl_mynn_subroutines.F90:4990
}
// bl_mynn_subroutines.F90:4993
ktop = 0;
// bl_mynn_subroutines.F90:4994
ztop = 0.0f;
// bl_mynn_subroutines.F90:4995
maxmf = 0.0f;
// bl_mynn_subroutines.F90:4998
if (((((((fltv2) > (0.002f))) && (((maxwidth) > (minwidth))))) && (superadiabatic))) {
// bl_mynn_subroutines.F90:5001
cn = 0.f;
// bl_mynn_subroutines.F90:5002
d = mynn_neg(1.9f);
// bl_mynn_subroutines.F90:5003
dl = mpas_div(mpas_sub(maxwidth, minwidth), ((float)(((nup) - (1)))));
// bl_mynn_subroutines.F90:5004
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5006
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5007
cn = mpas_add(cn, mpas_mul(mpas_div(mpas_mul(mynn_pow(((float)(l)), ((float)(d))), mpas_mul(l, l)), mpas_mul(dx, dx)), dl));
// bl_mynn_subroutines.F90:5008
}
// bl_mynn_subroutines.F90:5009
c = mpas_div(atot, cn);
// bl_mynn_subroutines.F90:5012
if (((mpas_sub(landsea, 1.5f)) < (0))) {
// bl_mynn_subroutines.F90:5013
acfac = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(fltv2, 0.02f), 0.05f))), .5f);
// bl_mynn_subroutines.F90:5014
} else {
// bl_mynn_subroutines.F90:5015
acfac = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(fltv2, 0.01f), 0.03f))), .5f);
// bl_mynn_subroutines.F90:5016
}
// bl_mynn_subroutines.F90:5021
if (((wspd_pbl) <= (10.f))) {
// bl_mynn_subroutines.F90:5022
ac_wsp = 1.0f;
// bl_mynn_subroutines.F90:5023
} else {
// bl_mynn_subroutines.F90:5024
ac_wsp = mpas_sub(1.0f, mynn_min(((float)(mpas_div(mpas_sub(wspd_pbl, 10.0f), 15.f))), ((float)(1.0f))));
// bl_mynn_subroutines.F90:5025
}
// bl_mynn_subroutines.F90:5026
acfac = mpas_mul(acfac, ac_wsp);
// bl_mynn_subroutines.F90:5029
an2 = 0.f;
// bl_mynn_subroutines.F90:5030
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5032
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5033
n = mpas_mul(c, mynn_pow(((float)(l)), ((float)(d))));
// bl_mynn_subroutines.F90:5034
upa(1, i) = mpas_mul(mpas_div(mpas_mul(mpas_mul(n, l), l), mpas_mul(dx, dx)), dl);
// bl_mynn_subroutines.F90:5036
upa(1, i) = mpas_mul(upa(1, i), acfac);
// bl_mynn_subroutines.F90:5037
an2 = mpas_add(an2, upa(1, i));
// bl_mynn_subroutines.F90:5039
}
// bl_mynn_subroutines.F90:5042
z0 = 50.f;
// bl_mynn_subroutines.F90:5043
pwmin = 0.1f;
// bl_mynn_subroutines.F90:5044
pwmax = 0.4f;
// bl_mynn_subroutines.F90:5046
wstar = mynn_max(((float)(1.e-2f)), ((float)(mynn_pow(((float)(mpas_mul(mpas_mul(gtr, fltv2), pblh))), ((float)(onethird))))));
// bl_mynn_subroutines.F90:5047
qstar = mpas_div(mynn_max(((float)(flq)), ((float)(1.0e-5f))), wstar);
// bl_mynn_subroutines.F90:5048
thstar = mpas_div(flt, wstar);
// bl_mynn_subroutines.F90:5050
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:5051
csigma = 1.34f;
// bl_mynn_subroutines.F90:5052
} else {
// bl_mynn_subroutines.F90:5053
csigma = 1.34f;
// bl_mynn_subroutines.F90:5054
}
// bl_mynn_subroutines.F90:5056
if (env_subs) {
// bl_mynn_subroutines.F90:5057
exc_fac = 0.0f;
// bl_mynn_subroutines.F90:5058
} else {
// bl_mynn_subroutines.F90:5059
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:5061
exc_fac = mpas_mul(0.58f, 4.0f);
// bl_mynn_subroutines.F90:5062
} else {
// bl_mynn_subroutines.F90:5064
exc_fac = 0.58f;
// bl_mynn_subroutines.F90:5065
}
// bl_mynn_subroutines.F90:5066
}
// bl_mynn_subroutines.F90:5068
exc_fac = mpas_mul(exc_fac, ac_wsp);
// bl_mynn_subroutines.F90:5071
sigmaw = mpas_mul(mpas_mul(mpas_mul(csigma, wstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird)))), mpas_sub(1, mpas_div(mpas_mul(0.8f, z0), pblh)));
// bl_mynn_subroutines.F90:5072
sigmaqt = mpas_mul(mpas_mul(csigma, qstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird))));
// bl_mynn_subroutines.F90:5073
sigmath = mpas_mul(mpas_mul(csigma, thstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird))));
// bl_mynn_subroutines.F90:5077
wmin = mynn_min(((float)(mpas_mul(sigmaw, pwmin))), ((float)(0.1f)));
// bl_mynn_subroutines.F90:5078
wmax = mynn_min(((float)(mpas_mul(sigmaw, pwmax))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:5081
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5082
wlv = mpas_add(wmin, mpas_mul(mpas_div(mpas_sub(wmax, wmin), nup2), ((i) - (1))));
// bl_mynn_subroutines.F90:5085
upw(1, i) = mpas_add(wmin, mpas_mul(mpas_div(((float)(i)), ((float)(nup))), mpas_sub(wmax, wmin)));
// bl_mynn_subroutines.F90:5086
upu(1, i) = mpas_div(mpas_add(mpas_mul(u(kts), dz(((kts) + (1)))), mpas_mul(u(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5087
upv(1, i) = mpas_div(mpas_add(mpas_mul(v(kts), dz(((kts) + (1)))), mpas_mul(v(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5088
upqc(1, i) = 0.0f;
// bl_mynn_subroutines.F90:5091
exc_heat = mpas_div(mpas_mul(mpas_mul(exc_fac, upw(1, i)), sigmath), sigmaw);
// bl_mynn_subroutines.F90:5092
upthv(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(thv(kts), dz(((kts) + (1)))), mpas_mul(thv(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_heat);
// bl_mynn_subroutines.F90:5094
upthl(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(thl(kts), dz(((kts) + (1)))), mpas_mul(thl(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_heat);
// bl_mynn_subroutines.F90:5098
exc_moist = mpas_div(mpas_mul(mpas_mul(exc_fac, upw(1, i)), sigmaqt), sigmaw);
// bl_mynn_subroutines.F90:5099
upqt(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(qt(kts), dz(((kts) + (1)))), mpas_mul(qt(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_moist);
// bl_mynn_subroutines.F90:5102
upqke(1, i) = mpas_div(mpas_add(mpas_mul(qke(kts), dz(((kts) + (1)))), mpas_mul(qke(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5103
upqnc(1, i) = mpas_div(mpas_add(mpas_mul(qnc(kts), dz(((kts) + (1)))), mpas_mul(qnc(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5104
upqni(1, i) = mpas_div(mpas_add(mpas_mul(qni(kts), dz(((kts) + (1)))), mpas_mul(qni(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5105
upqnwfa(1, i) = mpas_div(mpas_add(mpas_mul(qnwfa(kts), dz(((kts) + (1)))), mpas_mul(qnwfa(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5106
upqnifa(1, i) = mpas_div(mpas_add(mpas_mul(qnifa(kts), dz(((kts) + (1)))), mpas_mul(qnifa(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5107
upqnbca(1, i) = mpas_div(mpas_add(mpas_mul(qnbca(kts), dz(((kts) + (1)))), mpas_mul(qnbca(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5108
}
// bl_mynn_subroutines.F90:5110
if (mix_chem) {
// bl_mynn_subroutines.F90:5111
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5112
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5113
upchem(1, i, ic) = mpas_div(mpas_add(mpas_mul(chem1(kts, ic), dz(((kts) + (1)))), mpas_mul(chem1(((kts) + (1)), ic), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5114
}
// bl_mynn_subroutines.F90:5115
}
// bl_mynn_subroutines.F90:5116
}
// bl_mynn_subroutines.F90:5119
for (int mynn_element=0; mynn_element < (envm_thl.slice(MynnRange{kts, kte})).size(); ++mynn_element) { envm_thl.slice(MynnRange{kts, kte}).linear(mynn_element) = thl.slice(MynnRange{kts, kte}).linear(mynn_element); }
// bl_mynn_subroutines.F90:5120
for (int mynn_element=0; mynn_element < (envm_sqv.slice(MynnRange{kts, kte})).size(); ++mynn_element) { envm_sqv.slice(MynnRange{kts, kte}).linear(mynn_element) = qv.slice(MynnRange{kts, kte}).linear(mynn_element); }
// bl_mynn_subroutines.F90:5121
for (int mynn_element=0; mynn_element < (envm_sqc.slice(MynnRange{kts, kte})).size(); ++mynn_element) { envm_sqc.slice(MynnRange{kts, kte}).linear(mynn_element) = qc.slice(MynnRange{kts, kte}).linear(mynn_element); }
// bl_mynn_subroutines.F90:5122
for (int mynn_element=0; mynn_element < (envm_u.slice(MynnRange{kts, kte})).size(); ++mynn_element) { envm_u.slice(MynnRange{kts, kte}).linear(mynn_element) = u.slice(MynnRange{kts, kte}).linear(mynn_element); }
// bl_mynn_subroutines.F90:5123
for (int mynn_element=0; mynn_element < (envm_v.slice(MynnRange{kts, kte})).size(); ++mynn_element) { envm_v.slice(MynnRange{kts, kte}).linear(mynn_element) = v.slice(MynnRange{kts, kte}).linear(mynn_element); }
// bl_mynn_subroutines.F90:5124
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5125
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) + (1)))), mpas_mul(rho(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5126
}
// bl_mynn_subroutines.F90:5127
rhoz(kte) = rho(kte);
// bl_mynn_subroutines.F90:5130
dxsa = mpas_sub(1.f, mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_sub(12000.0f, dx), mpas_sub(12000.0f, 3000.0f)))), ((float)(0.f))))), ((float)(1.f))));
// bl_mynn_subroutines.F90:5133
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5134
qcn = 0.f;
// bl_mynn_subroutines.F90:5135
overshoot = 0;
// bl_mynn_subroutines.F90:5136
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5137
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5140
wmin = mpas_add(0.3f, mpas_mul(l, 0.0005f));
// bl_mynn_subroutines.F90:5141
ent(k, i) = mpas_div(0.33f, mpas_mul(mynn_min(((float)(mynn_max(((float)(upw(((k) - (1)), i))), ((float)(wmin))))), ((float)(0.9f))), l));
// bl_mynn_subroutines.F90:5149
ent(k, i) = mynn_max(((float)(ent(k, i))), ((float)(0.0003f)));
// bl_mynn_subroutines.F90:5153
if (((zw(k)) >= (mynn_min(((float)(mpas_add(pblh, 1500.f))), ((float)(4000.f)))))) {
// bl_mynn_subroutines.F90:5154
ent(k, i) = mpas_add(ent(k, i), mpas_mul(mpas_sub(zw(k), mynn_min(((float)(mpas_add(pblh, 1500.f))), ((float)(4000.f)))), 5.0e-6f));
// bl_mynn_subroutines.F90:5155
}
// bl_mynn_subroutines.F90:5158
ent(k, i) = mpas_mul(ent(k, i), mpas_sub(1.0f, rstoch_col(k)));
// bl_mynn_subroutines.F90:5160
ent(k, i) = mynn_min(((float)(ent(k, i))), ((float)(mpas_div(0.9f, mpas_sub(zw(((k) + (1))), zw(k))))));
// bl_mynn_subroutines.F90:5163
uk = mpas_div(mpas_add(mpas_mul(u(k), dz(((k) + (1)))), mpas_mul(u(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5164
ukm1 = mpas_div(mpas_add(mpas_mul(u(((k) - (1))), dz(k)), mpas_mul(u(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5165
vk = mpas_div(mpas_add(mpas_mul(v(k), dz(((k) + (1)))), mpas_mul(v(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5166
vkm1 = mpas_div(mpas_add(mpas_mul(v(((k) - (1))), dz(k)), mpas_mul(v(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5169
entexp = mpas_mul(ent(k, i), mpas_sub(zw(((k) + (1))), zw(k)));
// bl_mynn_subroutines.F90:5170
entexm = mpas_mul(entexp, 0.3333f);
// bl_mynn_subroutines.F90:5171
qtn = mpas_add(mpas_mul(upqt(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qt(k), entexp));
// bl_mynn_subroutines.F90:5172
thln = mpas_add(mpas_mul(upthl(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(thl(k), entexp));
// bl_mynn_subroutines.F90:5173
un = mpas_add(mpas_add(mpas_mul(upu(((k) - (1)), i), mpas_sub(1.f, entexm)), mpas_mul(u(k), entexm)), mpas_mul(mpas_mul(dxsa, pgfac), mpas_sub(uk, ukm1)));
// bl_mynn_subroutines.F90:5174
vn = mpas_add(mpas_add(mpas_mul(upv(((k) - (1)), i), mpas_sub(1.f, entexm)), mpas_mul(v(k), entexm)), mpas_mul(mpas_mul(dxsa, pgfac), mpas_sub(vk, vkm1)));
// bl_mynn_subroutines.F90:5175
qken = mpas_add(mpas_mul(upqke(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qke(k), entexp));
// bl_mynn_subroutines.F90:5176
qncn = mpas_add(mpas_mul(upqnc(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnc(k), entexp));
// bl_mynn_subroutines.F90:5177
qnin = mpas_add(mpas_mul(upqni(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qni(k), entexp));
// bl_mynn_subroutines.F90:5178
qnwfan = mpas_add(mpas_mul(upqnwfa(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnwfa(k), entexp));
// bl_mynn_subroutines.F90:5179
qnifan = mpas_add(mpas_mul(upqnifa(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnifa(k), entexp));
// bl_mynn_subroutines.F90:5180
qnbcan = mpas_add(mpas_mul(upqnbca(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnbca(k), entexp));
// bl_mynn_subroutines.F90:5184
qc_ent = qcn;
// bl_mynn_subroutines.F90:5185
qt_ent = qtn;
// bl_mynn_subroutines.F90:5186
thl_ent = thln;
// bl_mynn_subroutines.F90:5196
if (mix_chem) {
// bl_mynn_subroutines.F90:5197
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5201
chemn(ic) = mpas_add(mpas_mul(upchem(((k) - (1)), i, ic), mpas_sub(1.f, entexp)), mpas_mul(chem1(k, ic), entexp));
// bl_mynn_subroutines.F90:5202
}
// bl_mynn_subroutines.F90:5203
}
// bl_mynn_subroutines.F90:5206
pk = mpas_div(mpas_add(mpas_mul(p(k), dz(((k) + (1)))), mpas_mul(p(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5208
f_condensation_edmf(qtn, thln, pk, zw(((k) + (1))), thvn, qcn);
// bl_mynn_subroutines.F90:5211
thvk = mpas_div(mpas_add(mpas_mul(thv(k), dz(((k) + (1)))), mpas_mul(thv(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5212
thvkm1 = mpas_div(mpas_add(mpas_mul(thv(((k) - (1))), dz(k)), mpas_mul(thv(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5215
b = mpas_mul(grav, mpas_sub(mpas_div(thvn, thvk), 1.0f));
// bl_mynn_subroutines.F90:5216
if (((b) > (0.f))) {
// bl_mynn_subroutines.F90:5217
bcoeff = 0.15f;
// bl_mynn_subroutines.F90:5218
} else {
// bl_mynn_subroutines.F90:5219
bcoeff = 0.2f;
// bl_mynn_subroutines.F90:5220
}
// bl_mynn_subroutines.F90:5230
if (((upw(((k) - (1)), i)) < (0.2f))) {
// bl_mynn_subroutines.F90:5231
wn = mpas_add(upw(((k) - (1)), i), mpas_mul(mpas_add(mpas_mul(mpas_mul(mynn_neg(2.f), ent(k, i)), upw(((k) - (1)), i)), mpas_div(mpas_mul(bcoeff, b), mynn_max(((float)(upw(((k) - (1)), i))), ((float)(0.2f))))), mynn_min(((float)(mpas_sub(zw(k), zw(((k) - (1)))))), ((float)(250.f)))));
// bl_mynn_subroutines.F90:5232
} else {
// bl_mynn_subroutines.F90:5233
wn = mpas_add(upw(((k) - (1)), i), mpas_mul(mpas_add(mpas_mul(mpas_mul(mynn_neg(2.f), ent(k, i)), upw(((k) - (1)), i)), mpas_div(mpas_mul(bcoeff, b), upw(((k) - (1)), i))), mynn_min(((float)(mpas_sub(zw(k), zw(((k) - (1)))))), ((float)(250.f)))));
// bl_mynn_subroutines.F90:5234
}
// bl_mynn_subroutines.F90:5237
if (((wn) > (mpas_add(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))))))) {
// bl_mynn_subroutines.F90:5238
wn = mpas_add(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))));
// bl_mynn_subroutines.F90:5239
}
// bl_mynn_subroutines.F90:5241
if (((wn) < (mpas_sub(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))))))) {
// bl_mynn_subroutines.F90:5242
wn = mpas_sub(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))));
// bl_mynn_subroutines.F90:5243
}
// bl_mynn_subroutines.F90:5244
wn = mynn_min(((float)(mynn_max(((float)(wn)), ((float)(0.0f))))), ((float)(3.0f)));
// bl_mynn_subroutines.F90:5248
if (((((k) == (((kts) + (1))))) && (((wn) == (0.f))))) {
// bl_mynn_subroutines.F90:5249
nup2 = 0;
// bl_mynn_subroutines.F90:5250
break;
// bl_mynn_subroutines.F90:5251
}
// bl_mynn_subroutines.F90:5253
if (((debug_mf) == (1))) {
// bl_mynn_subroutines.F90:5254
if (((wn) >= (3.0f))) {
// bl_mynn_subroutines.F90:5256
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5257
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5258
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5259
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5260
}
// bl_mynn_subroutines.F90:5261
}
// bl_mynn_subroutines.F90:5264
if (((((wn) <= (0.0f))) && (((overshoot) == (0))))) {
// bl_mynn_subroutines.F90:5265
overshoot = 1;
// bl_mynn_subroutines.F90:5266
if (((mpas_sub(thvk, thvkm1)) > (0.0f))) {
// bl_mynn_subroutines.F90:5267
bvf = mynn_sqrt(mpas_div(mpas_mul(gtr, mpas_sub(thvk, thvkm1)), dz(k)));
// bl_mynn_subroutines.F90:5269
frz = mpas_div(upw(((k) - (1)), i), mpas_mul(bvf, dz(k)));
// bl_mynn_subroutines.F90:5271
dzp = mpas_mul(dz(k), mynn_max(((float)(mynn_min(((float)(frz)), ((float)(1.0f))))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:5272
}
// bl_mynn_subroutines.F90:5273
} else {
// bl_mynn_subroutines.F90:5274
dzp = dz(k);
// bl_mynn_subroutines.F90:5275
}
// bl_mynn_subroutines.F90:5285
aratio = mynn_min(((float)(mpas_div(upa(((k) - (1)), i), mpas_sub(1.f, upa(((k) - (1)), i))))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:5286
detturb = 0.00008f;
// bl_mynn_subroutines.F90:5287
oow = mpas_div(mynn_neg(0.060f), mynn_max(((float)(1.0f)), ((float)(mpas_mul(0.5f, mpas_add(wn, upw(((k) - (1)), i)))))));
// bl_mynn_subroutines.F90:5288
detrate = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(oow, mpas_sub(wn, upw(((k) - (1)), i))), dz(k)))), ((float)(detturb))))), ((float)(.0002f)));
// bl_mynn_subroutines.F90:5289
detrateuv = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(oow, mpas_sub(wn, upw(((k) - (1)), i))), dz(k)))), ((float)(detturb))))), ((float)(.0001f)));
// bl_mynn_subroutines.F90:5290
envm_thl(k) = mpas_add(envm_thl(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(thl_ent, upthl(((k) - (1)), i))), thl(k)), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5291
qv_ent = mpas_mul(0.5f, mpas_add(mynn_max(((float)(mpas_sub(qt_ent, qc_ent))), ((float)(0.f))), mynn_max(((float)(mpas_sub(upqt(((k) - (1)), i), upqc(((k) - (1)), i)))), ((float)(0.f)))));
// bl_mynn_subroutines.F90:5292
envm_sqv(k) = mpas_add(envm_sqv(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(qv_ent, qv(k)), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5293
if (((upqc(((k) - (1)), i)) > (1e-8f))) {
// bl_mynn_subroutines.F90:5294
if (((qc(k)) > (1e-6f))) {
// bl_mynn_subroutines.F90:5295
qc_grid = qc(k);
// bl_mynn_subroutines.F90:5296
} else {
// bl_mynn_subroutines.F90:5297
qc_grid = mpas_mul(cldfra_bl1d(k), qc_bl1d(k));
// bl_mynn_subroutines.F90:5298
}
// bl_mynn_subroutines.F90:5299
envm_sqc(k) = mpas_add(envm_sqc(k), mpas_mul(mpas_mul(mpas_mul(mynn_max(((float)(mpas_sub(mpas_mul(mpas_mul(upa(((k) - (1)), i), 0.5f), mpas_add(qcn, upqc(((k) - (1)), i))), qc_grid))), ((float)(0.0f))), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5300
}
// bl_mynn_subroutines.F90:5301
envm_u(k) = mpas_add(envm_u(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(un, upu(((k) - (1)), i))), u(k)), detrateuv), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5302
envm_v(k) = mpas_add(envm_v(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(vn, upv(((k) - (1)), i))), v(k)), detrateuv), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5304
if (((wn) > (0.f))) {
// bl_mynn_subroutines.F90:5306
upw(k, i) = wn;
// bl_mynn_subroutines.F90:5307
upthv(k, i) = thvn;
// bl_mynn_subroutines.F90:5308
upthl(k, i) = thln;
// bl_mynn_subroutines.F90:5309
upqt(k, i) = qtn;
// bl_mynn_subroutines.F90:5310
upqc(k, i) = qcn;
// bl_mynn_subroutines.F90:5311
upu(k, i) = un;
// bl_mynn_subroutines.F90:5312
upv(k, i) = vn;
// bl_mynn_subroutines.F90:5313
upqke(k, i) = qken;
// bl_mynn_subroutines.F90:5314
upqnc(k, i) = qncn;
// bl_mynn_subroutines.F90:5315
upqni(k, i) = qnin;
// bl_mynn_subroutines.F90:5316
upqnwfa(k, i) = qnwfan;
// bl_mynn_subroutines.F90:5317
upqnifa(k, i) = qnifan;
// bl_mynn_subroutines.F90:5318
upqnbca(k, i) = qnbcan;
// bl_mynn_subroutines.F90:5319
upa(k, i) = upa(((k) - (1)), i);
// bl_mynn_subroutines.F90:5320
if (mix_chem) {
// bl_mynn_subroutines.F90:5321
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5322
upchem(k, i, ic) = chemn(ic);
// bl_mynn_subroutines.F90:5323
}
// bl_mynn_subroutines.F90:5324
}
// bl_mynn_subroutines.F90:5325
ktop = mynn_max(((int)(ktop)), ((int)(k)));
// bl_mynn_subroutines.F90:5326
} else {
// bl_mynn_subroutines.F90:5327
break;
// bl_mynn_subroutines.F90:5328
}
// bl_mynn_subroutines.F90:5329
}
// bl_mynn_subroutines.F90:5331
if (((debug_mf) == (1))) {
// bl_mynn_subroutines.F90:5332
if (((((((((mynn_maxval(upw.slice(MynnRange{upw.lo[0], upw.hi[0]}, i))) > (10.0f))) || (((mynn_minval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, i))) < (0.0f))))) || (((mynn_maxval(upa.slice(MynnRange{upa.lo[0], upa.hi[0]}, i))) > (atot))))) || (((nup2) > (10))))) {
// bl_mynn_subroutines.F90:5335
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5336
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5337
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5339
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5340
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5341
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5342
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5343
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5344
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5345
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5346
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5347
}
// bl_mynn_subroutines.F90:5348
}
// bl_mynn_subroutines.F90:5349
}
// bl_mynn_subroutines.F90:5350
} else {
// bl_mynn_subroutines.F90:5352
nup2 = 0.f;
// bl_mynn_subroutines.F90:5353
}
// bl_mynn_subroutines.F90:5355
ktop = mynn_min(((int)(ktop)), ((int)(((kte) - (1)))));
// bl_mynn_subroutines.F90:5356
if (((ktop) == (0))) {
// bl_mynn_subroutines.F90:5357
ztop = 0.0f;
// bl_mynn_subroutines.F90:5358
} else {
// bl_mynn_subroutines.F90:5359
ztop = zw(ktop);
// bl_mynn_subroutines.F90:5360
}
// bl_mynn_subroutines.F90:5362
if (((nup2) > (0))) {
// bl_mynn_subroutines.F90:5365
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5366
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5367
s_aw(((k) + (1))) = mpas_add(s_aw(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), psig_w));
// bl_mynn_subroutines.F90:5368
s_awthl(((k) + (1))) = mpas_add(s_awthl(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upthl(k, i)), psig_w));
// bl_mynn_subroutines.F90:5369
s_awqt(((k) + (1))) = mpas_add(s_awqt(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqt(k, i)), psig_w));
// bl_mynn_subroutines.F90:5374
qc_plume = upqc(k, i);
// bl_mynn_subroutines.F90:5378
s_awqc(((k) + (1))) = mpas_add(s_awqc(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), qc_plume), psig_w));
// bl_mynn_subroutines.F90:5379
s_awqv(((k) + (1))) = mpas_sub(s_awqt(((k) + (1))), s_awqc(((k) + (1))));
// bl_mynn_subroutines.F90:5380
}
// bl_mynn_subroutines.F90:5381
}
// bl_mynn_subroutines.F90:5383
if (momentum_opt) {
// bl_mynn_subroutines.F90:5384
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5385
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5386
s_awu(((k) + (1))) = mpas_add(s_awu(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upu(k, i)), psig_w));
// bl_mynn_subroutines.F90:5387
s_awv(((k) + (1))) = mpas_add(s_awv(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upv(k, i)), psig_w));
// bl_mynn_subroutines.F90:5388
}
// bl_mynn_subroutines.F90:5389
}
// bl_mynn_subroutines.F90:5390
}
// bl_mynn_subroutines.F90:5392
if (tke_opt) {
// bl_mynn_subroutines.F90:5393
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5394
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5395
s_awqke(((k) + (1))) = mpas_add(s_awqke(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqke(k, i)), psig_w));
// bl_mynn_subroutines.F90:5396
}
// bl_mynn_subroutines.F90:5397
}
// bl_mynn_subroutines.F90:5398
}
// bl_mynn_subroutines.F90:5400
if (mix_chem) {
// bl_mynn_subroutines.F90:5401
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:5402
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5403
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5404
s_awchem(((k) + (1)), ic) = mpas_add(s_awchem(((k) + (1)), ic), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upchem(k, i, ic)), psig_w));
// bl_mynn_subroutines.F90:5405
}
// bl_mynn_subroutines.F90:5406
}
// bl_mynn_subroutines.F90:5407
}
// bl_mynn_subroutines.F90:5408
}
// bl_mynn_subroutines.F90:5410
if (scalar_opt) {
// bl_mynn_subroutines.F90:5411
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:5412
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5413
s_awqnc(((k) + (1))) = mpas_add(s_awqnc(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnc(k, i)), psig_w));
// bl_mynn_subroutines.F90:5414
s_awqni(((k) + (1))) = mpas_add(s_awqni(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqni(k, i)), psig_w));
// bl_mynn_subroutines.F90:5415
s_awqnwfa(((k) + (1))) = mpas_add(s_awqnwfa(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnwfa(k, i)), psig_w));
// bl_mynn_subroutines.F90:5416
s_awqnifa(((k) + (1))) = mpas_add(s_awqnifa(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnifa(k, i)), psig_w));
// bl_mynn_subroutines.F90:5417
s_awqnbca(((k) + (1))) = mpas_add(s_awqnbca(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnbca(k, i)), psig_w));
// bl_mynn_subroutines.F90:5418
}
// bl_mynn_subroutines.F90:5419
}
// bl_mynn_subroutines.F90:5420
}
// bl_mynn_subroutines.F90:5425
if (((s_aw(((kts) + (1)))) != (0.f))) {
// bl_mynn_subroutines.F90:5426
dzi(kts) = mpas_mul(0.5f, mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5427
flx1 = mynn_max(((float)(mpas_div(mpas_mul(s_aw(((kts) + (1))), mpas_sub(th(kts), th(((kts) + (1))))), dzi(kts)))), ((float)(1.0e-5f)));
// bl_mynn_subroutines.F90:5428
} else {
// bl_mynn_subroutines.F90:5429
flx1 = 0.0f;
// bl_mynn_subroutines.F90:5432
}
// bl_mynn_subroutines.F90:5433
adjustment = 1.0f;
// bl_mynn_subroutines.F90:5436
if (((((flx1) > (mpas_div(mpas_mul(fluxportion, flt), dz(kts))))) && (((flx1) > (0.0f))))) {
// bl_mynn_subroutines.F90:5437
adjustment = mpas_div(mpas_div(mpas_mul(fluxportion, flt), dz(kts)), flx1);
// bl_mynn_subroutines.F90:5438
for (int mynn_element=0; mynn_element < (s_aw).size(); ++mynn_element) { s_aw.linear(mynn_element) = mpas_mul(s_aw.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5439
for (int mynn_element=0; mynn_element < (s_awthl).size(); ++mynn_element) { s_awthl.linear(mynn_element) = mpas_mul(s_awthl.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5440
for (int mynn_element=0; mynn_element < (s_awqt).size(); ++mynn_element) { s_awqt.linear(mynn_element) = mpas_mul(s_awqt.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5441
for (int mynn_element=0; mynn_element < (s_awqc).size(); ++mynn_element) { s_awqc.linear(mynn_element) = mpas_mul(s_awqc.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5442
for (int mynn_element=0; mynn_element < (s_awqv).size(); ++mynn_element) { s_awqv.linear(mynn_element) = mpas_mul(s_awqv.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5443
for (int mynn_element=0; mynn_element < (s_awqnc).size(); ++mynn_element) { s_awqnc.linear(mynn_element) = mpas_mul(s_awqnc.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5444
for (int mynn_element=0; mynn_element < (s_awqni).size(); ++mynn_element) { s_awqni.linear(mynn_element) = mpas_mul(s_awqni.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5445
for (int mynn_element=0; mynn_element < (s_awqnwfa).size(); ++mynn_element) { s_awqnwfa.linear(mynn_element) = mpas_mul(s_awqnwfa.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5446
for (int mynn_element=0; mynn_element < (s_awqnifa).size(); ++mynn_element) { s_awqnifa.linear(mynn_element) = mpas_mul(s_awqnifa.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5447
for (int mynn_element=0; mynn_element < (s_awqnbca).size(); ++mynn_element) { s_awqnbca.linear(mynn_element) = mpas_mul(s_awqnbca.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5448
if (momentum_opt) {
// bl_mynn_subroutines.F90:5449
for (int mynn_element=0; mynn_element < (s_awu).size(); ++mynn_element) { s_awu.linear(mynn_element) = mpas_mul(s_awu.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5450
for (int mynn_element=0; mynn_element < (s_awv).size(); ++mynn_element) { s_awv.linear(mynn_element) = mpas_mul(s_awv.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5451
}
// bl_mynn_subroutines.F90:5452
if (tke_opt) {
// bl_mynn_subroutines.F90:5453
for (int mynn_element=0; mynn_element < (s_awqke).size(); ++mynn_element) { s_awqke.linear(mynn_element) = mpas_mul(s_awqke.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5454
}
// bl_mynn_subroutines.F90:5455
if (mix_chem) {
// bl_mynn_subroutines.F90:5456
for (int mynn_element=0; mynn_element < (s_awchem).size(); ++mynn_element) { s_awchem.linear(mynn_element) = mpas_mul(s_awchem.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5457
}
// bl_mynn_subroutines.F90:5458
for (int mynn_element=0; mynn_element < (upa).size(); ++mynn_element) { upa.linear(mynn_element) = mpas_mul(upa.linear(mynn_element), adjustment); }
// bl_mynn_subroutines.F90:5459
}
// bl_mynn_subroutines.F90:5464
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5465
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5466
edmf_a(k) = mpas_add(edmf_a(k), upa(k, i));
// bl_mynn_subroutines.F90:5467
edmf_w(k) = mpas_add(edmf_w(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)));
// bl_mynn_subroutines.F90:5468
edmf_qt(k) = mpas_add(edmf_qt(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upqt(k, i)));
// bl_mynn_subroutines.F90:5469
edmf_thl(k) = mpas_add(edmf_thl(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upthl(k, i)));
// bl_mynn_subroutines.F90:5470
edmf_ent(k) = mpas_add(edmf_ent(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), ent(k, i)));
// bl_mynn_subroutines.F90:5471
edmf_qc(k) = mpas_add(edmf_qc(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upqc(k, i)));
// bl_mynn_subroutines.F90:5472
}
// bl_mynn_subroutines.F90:5473
}
// bl_mynn_subroutines.F90:5474
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5477
if (((edmf_a(k)) > (0.f))) {
// bl_mynn_subroutines.F90:5478
edmf_w(k) = mpas_div(edmf_w(k), edmf_a(k));
// bl_mynn_subroutines.F90:5479
edmf_qt(k) = mpas_div(edmf_qt(k), edmf_a(k));
// bl_mynn_subroutines.F90:5480
edmf_thl(k) = mpas_div(edmf_thl(k), edmf_a(k));
// bl_mynn_subroutines.F90:5481
edmf_ent(k) = mpas_div(edmf_ent(k), edmf_a(k));
// bl_mynn_subroutines.F90:5482
edmf_qc(k) = mpas_div(edmf_qc(k), edmf_a(k));
// bl_mynn_subroutines.F90:5483
edmf_a(k) = mpas_mul(edmf_a(k), psig_w);
// bl_mynn_subroutines.F90:5485
if (((mpas_mul(edmf_a(k), edmf_w(k))) > (maxmf))) { maxmf = mpas_mul(edmf_a(k), edmf_w(k)); }
// bl_mynn_subroutines.F90:5486
}
// bl_mynn_subroutines.F90:5487
}
// bl_mynn_subroutines.F90:5490
if (mix_chem) {
// bl_mynn_subroutines.F90:5491
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5492
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5493
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5494
edmf_chem(k, ic) = mpas_add(edmf_chem(k, ic), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upchem(k, i, ic)));
// bl_mynn_subroutines.F90:5495
}
// bl_mynn_subroutines.F90:5496
}
// bl_mynn_subroutines.F90:5497
}
// bl_mynn_subroutines.F90:5498
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5499
if (((edmf_a(k)) > (0.f))) {
// bl_mynn_subroutines.F90:5500
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn_subroutines.F90:5501
edmf_chem(k, ic) = mpas_div(edmf_chem(k, ic), edmf_a(k));
// bl_mynn_subroutines.F90:5502
}
// bl_mynn_subroutines.F90:5503
}
// bl_mynn_subroutines.F90:5504
}
// bl_mynn_subroutines.F90:5505
}
// bl_mynn_subroutines.F90:5509
if (env_subs) {
// bl_mynn_subroutines.F90:5510
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5515
envi_w(k) = mpas_mul(onethird, mpas_add(mpas_add(edmf_w(((k) - (1))), edmf_w(k)), edmf_w(((k) + (1)))));
// bl_mynn_subroutines.F90:5516
envi_a(k) = mpas_mul(mpas_mul(onethird, mpas_add(mpas_add(edmf_a(((k) - (1))), edmf_a(k)), edmf_a(((k) + (1))))), adjustment);
// bl_mynn_subroutines.F90:5517
}
// bl_mynn_subroutines.F90:5519
envi_w(kts) = edmf_w(kts);
// bl_mynn_subroutines.F90:5520
envi_a(kts) = edmf_a(kts);
// bl_mynn_subroutines.F90:5522
envi_w(kte) = 0.0f;
// bl_mynn_subroutines.F90:5523
envi_a(kte) = edmf_a(kte);
// bl_mynn_subroutines.F90:5525
envi_w(((kte) + (1))) = 0.0f;
// bl_mynn_subroutines.F90:5526
envi_a(((kte) + (1))) = edmf_a(kte);
// bl_mynn_subroutines.F90:5530
if (((envi_w(kts)) > (mpas_div(mpas_mul(0.9f, dz(kts)), dt)))) {
// bl_mynn_subroutines.F90:5531
sublim = mpas_div(mpas_div(mpas_mul(0.9f, dz(kts)), dt), envi_w(kts));
// bl_mynn_subroutines.F90:5532
} else {
// bl_mynn_subroutines.F90:5533
sublim = 1.0f;
// bl_mynn_subroutines.F90:5534
}
// bl_mynn_subroutines.F90:5536
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:5537
temp = envi_a(k);
// bl_mynn_subroutines.F90:5538
envi_a(k) = mpas_sub(1.0f, temp);
// bl_mynn_subroutines.F90:5539
envi_w(k) = mpas_div(mpas_mul(mpas_mul(mpas_mul(csub, sublim), envi_w(k)), temp), mpas_sub(1.f, temp));
// bl_mynn_subroutines.F90:5540
}
// bl_mynn_subroutines.F90:5543
dzi(kts) = mpas_mul(0.5f, mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5544
sub_thl(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), thl(((kts) + (1)))), mpas_mul(rho(kts), thl(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5546
sub_sqv(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), qv(((kts) + (1)))), mpas_mul(rho(kts), qv(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5548
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5549
dzi(k) = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) + (1)))));
// bl_mynn_subroutines.F90:5550
sub_thl(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), thl(((k) + (1)))), mpas_mul(rho(k), thl(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5552
sub_sqv(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), qv(((k) + (1)))), mpas_mul(rho(k), qv(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5554
}
// bl_mynn_subroutines.F90:5556
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5557
det_thl(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_thl(k), thl(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5558
det_sqv(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_sqv(k), qv(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5559
det_sqc(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_sqc(k), qc(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5560
}
// bl_mynn_subroutines.F90:5562
if (momentum_opt) {
// bl_mynn_subroutines.F90:5563
sub_u(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), u(((kts) + (1)))), mpas_mul(rho(kts), u(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5565
sub_v(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), v(((kts) + (1)))), mpas_mul(rho(kts), v(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5567
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5568
sub_u(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), u(((k) + (1)))), mpas_mul(rho(k), u(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5570
sub_v(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), v(((k) + (1)))), mpas_mul(rho(k), v(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5572
}
// bl_mynn_subroutines.F90:5574
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5575
det_u(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_u(k), u(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5576
det_v(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_v(k), v(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5577
}
// bl_mynn_subroutines.F90:5578
}
// bl_mynn_subroutines.F90:5579
}
// bl_mynn_subroutines.F90:5584
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5585
exneri(k) = mpas_div(mpas_add(mpas_mul(exner(k), dz(((k) + (1)))), mpas_mul(exner(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5586
edmf_th(k) = mpas_add(edmf_thl(k), mpas_mul(mpas_div(xlvcp, exneri(k)), edmf_qc(k)));
// bl_mynn_subroutines.F90:5587
dzi(k) = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) + (1)))));
// bl_mynn_subroutines.F90:5588
}
// bl_mynn_subroutines.F90:5593
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// bl_mynn_subroutines.F90:5594
if (((k) > (ktop))) { break; }
// bl_mynn_subroutines.F90:5595
if (((((mpas_mul(0.5f, mpas_add(edmf_qc(k), edmf_qc(((k) - (1)))))) > (0.0f))) && (((cldfra_bl1d(k)) < (cf_thresh))))) {
// bl_mynn_subroutines.F90:5597
aup = mpas_div(mpas_add(mpas_mul(edmf_a(k), dzi(((k) - (1)))), mpas_mul(edmf_a(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5598
thp = mpas_div(mpas_add(mpas_mul(edmf_th(k), dzi(((k) - (1)))), mpas_mul(edmf_th(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5599
qtp = mpas_div(mpas_add(mpas_mul(edmf_qt(k), dzi(((k) - (1)))), mpas_mul(edmf_qt(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5603
esat = f_esat_blend(tk(k), t0c, tice);
// bl_mynn_subroutines.F90:5605
qsl = mpas_div(mpas_mul(ep_2, esat), mynn_max(((float)(1.e-7f)), ((float)(mpas_sub(p(k), mpas_mul(ep_3, esat))))));
// bl_mynn_subroutines.F90:5608
if (((((edmf_qc(k)) > (0.0f))) && (((edmf_qc(((k) - (1)))) > (0.0f))))) {
// bl_mynn_subroutines.F90:5609
qcp = mpas_div(mpas_add(mpas_mul(edmf_qc(k), dzi(((k) - (1)))), mpas_mul(edmf_qc(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5610
} else {
// bl_mynn_subroutines.F90:5611
qcp = mynn_max(((float)(edmf_qc(k))), ((float)(edmf_qc(((k) - (1))))));
// bl_mynn_subroutines.F90:5612
}
// bl_mynn_subroutines.F90:5615
xl = f_xl_blend(tk(k), t0c, tice, cice, cliq, cpv, xls, xlv);
// bl_mynn_subroutines.F90:5616
qsat_tk = f_qsat_blend(tk(k), t0c, tice, p(k), {});
// bl_mynn_subroutines.F90:5618
rsl = mpas_div(mpas_mul(xl, qsat_tk), mpas_mul(r_v, mynn_ipow<2>(tk(k))));
// bl_mynn_subroutines.F90:5620
cpm = mpas_add(cp, mpas_mul(qt(k), cpv));
// bl_mynn_subroutines.F90:5621
a = mpas_div(1.f, mpas_add(1.f, mpas_div(mpas_mul(xl, rsl), cpm)));
// bl_mynn_subroutines.F90:5622
b9 = mpas_mul(a, rsl);
// bl_mynn_subroutines.F90:5624
q2p = mpas_div(xlvcp, exner(k));
// bl_mynn_subroutines.F90:5625
pt = mpas_add(thl(k), mpas_mul(mpas_mul(q2p, qcp), aup));
// bl_mynn_subroutines.F90:5626
bb = mpas_div(mpas_mul(b9, tk(k)), pt);
// bl_mynn_subroutines.F90:5632
qww = mpas_add(1.f, mpas_mul(0.61f, qt(k)));
// bl_mynn_subroutines.F90:5633
alpha = mpas_mul(0.61f, pt);
// bl_mynn_subroutines.F90:5634
beta = mpas_sub(mpas_div(mpas_mul(pt, xl), mpas_mul(tk(k), cp)), mpas_mul(1.61f, pt));
// bl_mynn_subroutines.F90:5638
if (((a) > (0.0f))) {
// bl_mynn_subroutines.F90:5639
f = mynn_min(((float)(mpas_div(1.0f, a))), ((float)(4.0f)));
// bl_mynn_subroutines.F90:5640
} else {
// bl_mynn_subroutines.F90:5641
f = 1.0f;
// bl_mynn_subroutines.F90:5642
}
// bl_mynn_subroutines.F90:5649
sigq = mpas_mul(mpas_mul(10.f, aup), mpas_sub(qtp, qt(k)));
// bl_mynn_subroutines.F90:5651
sigq = mynn_max(((float)(sigq)), ((float)(mpas_mul(qsat_tk, 0.02f))));
// bl_mynn_subroutines.F90:5652
sigq = mynn_min(((float)(sigq)), ((float)(mpas_mul(qsat_tk, 0.25f))));
// bl_mynn_subroutines.F90:5654
qmq = mpas_mul(a, mpas_sub(qt(k), qsat_tk));
// bl_mynn_subroutines.F90:5655
q1 = mpas_div(qmq, sigq);
// bl_mynn_subroutines.F90:5657
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:5661
mf_cf = mynn_min(((float)(mynn_max(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.55f, q1)))))), ((float)(0.01f))))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:5662
mf_cf = mynn_max(((float)(mf_cf)), ((float)(mpas_mul(1.2f, aup))));
// bl_mynn_subroutines.F90:5663
mf_cf = mynn_min(((float)(mf_cf)), ((float)(mpas_mul(5.0f, aup))));
// bl_mynn_subroutines.F90:5664
} else {
// bl_mynn_subroutines.F90:5668
mf_cf = mynn_min(((float)(mynn_max(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.55f, q1)))))), ((float)(0.01f))))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:5669
mf_cf = mynn_max(((float)(mf_cf)), ((float)(mpas_mul(1.8f, aup))));
// bl_mynn_subroutines.F90:5670
mf_cf = mynn_min(((float)(mf_cf)), ((float)(mpas_mul(5.0f, aup))));
// bl_mynn_subroutines.F90:5671
}
// bl_mynn_subroutines.F90:5684
if (((mpas_sub(landsea, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:5685
if (((mpas_mul(qcp, aup)) > (5e-5f))) {
// bl_mynn_subroutines.F90:5686
qc_bl1d(k) = mpas_sub(mpas_mul(1.86f, mpas_mul(qcp, aup)), 2.2e-5f);
// bl_mynn_subroutines.F90:5687
} else {
// bl_mynn_subroutines.F90:5688
qc_bl1d(k) = mpas_mul(1.18f, mpas_mul(qcp, aup));
// bl_mynn_subroutines.F90:5689
}
// bl_mynn_subroutines.F90:5690
cldfra_bl1d(k) = mf_cf;
// bl_mynn_subroutines.F90:5691
ac_mf = mf_cf;
// bl_mynn_subroutines.F90:5692
} else {
// bl_mynn_subroutines.F90:5693
if (((mpas_mul(qcp, aup)) > (5e-5f))) {
// bl_mynn_subroutines.F90:5694
qc_bl1d(k) = mpas_sub(mpas_mul(1.86f, mpas_mul(qcp, aup)), 2.2e-5f);
// bl_mynn_subroutines.F90:5695
} else {
// bl_mynn_subroutines.F90:5696
qc_bl1d(k) = mpas_mul(1.18f, mpas_mul(qcp, aup));
// bl_mynn_subroutines.F90:5697
}
// bl_mynn_subroutines.F90:5698
cldfra_bl1d(k) = mf_cf;
// bl_mynn_subroutines.F90:5699
ac_mf = mf_cf;
// bl_mynn_subroutines.F90:5700
}
// bl_mynn_subroutines.F90:5708
q1 = mynn_max(((float)(q1)), ((float)(mynn_neg(2.25f))));
// bl_mynn_subroutines.F90:5713
if (((q1) >= (1.0f))) {
// bl_mynn_subroutines.F90:5714
fng = 1.0f;
// bl_mynn_subroutines.F90:5715
} else if (((((q1) >= (mynn_neg(1.7f)))) && (((q1) < (1.0f))))) {
// bl_mynn_subroutines.F90:5716
fng = mynn_exp(mpas_mul(mynn_neg(0.4f), mpas_sub(q1, 1.0f)));
// bl_mynn_subroutines.F90:5717
} else if (((((q1) >= (mynn_neg(2.5f)))) && (((q1) < (mynn_neg(1.7f)))))) {
// bl_mynn_subroutines.F90:5718
fng = mpas_add(3.0f, mynn_exp(mpas_mul(mynn_neg(3.8f), mpas_add(q1, 1.7f))));
// bl_mynn_subroutines.F90:5719
} else {
// bl_mynn_subroutines.F90:5720
fng = mynn_min(((float)(mpas_add(23.9f, mynn_exp(mpas_mul(mynn_neg(1.6f), mpas_add(q1, 2.5f)))))), ((float)(60.f)));
// bl_mynn_subroutines.F90:5721
}
// bl_mynn_subroutines.F90:5724
vt(k) = mpas_sub(mpas_sub(qww, mpas_mul(mpas_mul(mpas_mul(mpas_mul(1.5f, aup), beta), bb), fng)), 1.f);
// bl_mynn_subroutines.F90:5725
vq(k) = mpas_sub(mpas_add(alpha, mpas_mul(mpas_mul(mpas_mul(mpas_mul(1.5f, aup), beta), a), fng)), tv0);
// bl_mynn_subroutines.F90:5726
}
// bl_mynn_subroutines.F90:5727
}
// bl_mynn_subroutines.F90:5729
}
// bl_mynn_subroutines.F90:5732
if (((ktop) > (0))) {
// bl_mynn_subroutines.F90:5733
maxqc = mynn_maxval(edmf_qc.slice(MynnRange{1, ktop}));
// bl_mynn_subroutines.F90:5734
if (((maxqc) < (1.e-8f))) { maxmf = mpas_mul(mynn_neg(1.0f), maxmf); }
// bl_mynn_subroutines.F90:5735
}
// bl_mynn_subroutines.F90:5740
if (((edmf_w(1)) > (4.0f))) {
// bl_mynn_subroutines.F90:5742
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5743
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5744
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5772
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5773
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5774
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5775
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5777
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:5789-5846
__device__ __noinline__ void f_condensation_edmf(float qt, float thl, float p, float zagl, float& thv, float& qc) {
MynnFrame mynn_frame(arena);
int niter;
int i;
float diff;
float exn;
float t;
float th;
float qs;
float qcold;
// bl_mynn_subroutines.F90:5808
niter = 50;
// bl_mynn_subroutines.F90:5810
diff = 1.e-6f;
// bl_mynn_subroutines.F90:5812
exn = mynn_pow(((float)(mpas_div(p, p1000mb))), ((float)(rcp)));
// bl_mynn_subroutines.F90:5814
for (i=1; (1) > 0 ? i <= (niter) : i >= (niter); i += (1)) {
// bl_mynn_subroutines.F90:5815
t = mpas_add(mpas_mul(exn, thl), mpas_mul(xlvcp, qc));
// bl_mynn_subroutines.F90:5816
qs = f_qsat_blend(t, t0c, tice, p, {});
// bl_mynn_subroutines.F90:5817
qcold = qc;
// bl_mynn_subroutines.F90:5818
qc = mpas_add(mpas_mul(0.5f, qc), mpas_mul(0.5f, mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)))));
// bl_mynn_subroutines.F90:5819
if (((mynn_abs(mpas_sub(qc, qcold))) < (diff))) { break; }
// bl_mynn_subroutines.F90:5820
}
// bl_mynn_subroutines.F90:5822
t = mpas_add(mpas_mul(exn, thl), mpas_mul(xlvcp, qc));
// bl_mynn_subroutines.F90:5823
qs = f_qsat_blend(t, t0c, tice, p, {});
// bl_mynn_subroutines.F90:5824
qc = mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)));
// bl_mynn_subroutines.F90:5827
if (((zagl) < (100.f))) { qc = 0.f; }
// bl_mynn_subroutines.F90:5830
thv = mpas_mul(mpas_add(thl, mpas_mul(xlvcp, qc)), mpas_sub(mpas_add(1.f, mpas_mul(qt, mpas_sub(rvovrd, 1.f))), mpas_mul(rvovrd, qc)));
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:5850-5883
__device__ __noinline__ void f_condensation_edmf_r(float qt, float& thl, float p, float zagl, float thv, float& qc) {
MynnFrame mynn_frame(arena);
int niter;
int i;
float diff;
float exn;
float t;
float th;
float qs;
float qcold;
// bl_mynn_subroutines.F90:5862
niter = 50;
// bl_mynn_subroutines.F90:5864
diff = 2.e-5f;
// bl_mynn_subroutines.F90:5866
exn = mynn_pow(((float)(mpas_div(p, p1000mb))), ((float)(rcp)));
// bl_mynn_subroutines.F90:5868
t = mpas_mul(thv, exn);
// bl_mynn_subroutines.F90:5872
qc = 0.f;
// bl_mynn_subroutines.F90:5874
for (i=1; (1) > 0 ? i <= (niter) : i >= (niter); i += (1)) {
// bl_mynn_subroutines.F90:5875
qcold = qc;
// bl_mynn_subroutines.F90:5876
t = mpas_div(mpas_mul(exn, thv), mpas_sub(mpas_add(1.f, mpas_mul(qt, mpas_sub(rvovrd, 1.f))), mpas_mul(rvovrd, qc)));
// bl_mynn_subroutines.F90:5877
qs = f_qsat_blend(t, t0c, tice, p, {});
// bl_mynn_subroutines.F90:5878
qc = mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)));
// bl_mynn_subroutines.F90:5879
if (((mynn_abs(mpas_sub(qc, qcold))) < (diff))) { break; }
// bl_mynn_subroutines.F90:5880
}
// bl_mynn_subroutines.F90:5881
thl = mpas_div(mpas_sub(t, mpas_mul(mpas_div(xlv, cp), qc)), exn);
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:5892-6269
__device__ __noinline__ void f_ddmf_jpl(int kts, int kte, float dt, MynnArray<float, 1> zw, MynnArray<float, 1> dz, MynnArray<float, 1> p, MynnArray<float, 1> u, MynnArray<float, 1> v, MynnArray<float, 1> th, MynnArray<float, 1> thl, MynnArray<float, 1> thv, MynnArray<float, 1> tk, MynnArray<float, 1> qt, MynnArray<float, 1> qv, MynnArray<float, 1> qc, MynnArray<float, 1> rho, MynnArray<float, 1> exner, float ust, float wthl, float wqt, float pblh, int kpbl, MynnArray<float, 1> edmf_a_dd, MynnArray<float, 1> edmf_w_dd, MynnArray<float, 1> edmf_qt_dd, MynnArray<float, 1> edmf_thl_dd, MynnArray<float, 1> edmf_ent_dd, MynnArray<float, 1> edmf_qc_dd, MynnArray<float, 1> sd_aw, MynnArray<float, 1> sd_awthl, MynnArray<float, 1> sd_awqt, MynnArray<float, 1> sd_awqv, MynnArray<float, 1> sd_awqc, MynnArray<float, 1> sd_awu, MynnArray<float, 1> sd_awv, MynnArray<float, 1> sd_awqke, MynnArray<float, 1> qc_bl1d, MynnArray<float, 1> cldfra_bl1d, MynnArray<float, 1> rthraten) {
MynnFrame mynn_frame(arena);
zw.rebase({kts});
dz.rebase({kts});
p.rebase({kts});
u.rebase({kts});
v.rebase({kts});
th.rebase({kts});
thl.rebase({kts});
thv.rebase({kts});
tk.rebase({kts});
qt.rebase({kts});
qv.rebase({kts});
qc.rebase({kts});
rho.rebase({kts});
exner.rebase({kts});
edmf_a_dd.rebase({kts});
edmf_w_dd.rebase({kts});
edmf_qt_dd.rebase({kts});
edmf_thl_dd.rebase({kts});
edmf_ent_dd.rebase({kts});
edmf_qc_dd.rebase({kts});
sd_aw.rebase({kts});
sd_awthl.rebase({kts});
sd_awqt.rebase({kts});
sd_awqv.rebase({kts});
sd_awqc.rebase({kts});
sd_awu.rebase({kts});
sd_awv.rebase({kts});
sd_awqke.rebase({kts});
qc_bl1d.rebase({kts});
cldfra_bl1d.rebase({kts});
rthraten.rebase({kts});
MynnArray<float, 1> sd_aw2 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
constexpr int ndown = 5;
MynnArray<int, 1> dd_initk = mynn_array<int, 1>(arena, {1}, {ndown});
MynnArray<float, 1> randnum = mynn_array<float, 1>(arena, {1}, {ndown});
MynnArray<float, 2> downw = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downthl = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downqt = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downqc = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downa = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downu = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downv = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> downthv = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> ent = mynn_array<float, 2>(arena, {((kts) + (1)), 1}, {((kte) + (1)), ndown});
MynnArray<float, 2> entf = mynn_array<float, 2>(arena, {((kts) + (1)), 1}, {((kte) + (1)), ndown});
MynnArray<int, 2> enti = mynn_array<int, 2>(arena, {((kts) + (1)), 1}, {((kte) + (1)), ndown});
int k;
int i;
int ki;
int kminrad;
int qltop;
int p700_ind;
int qlbase;
float wthv;
float wstar;
float qstar;
float thstar;
float sigmaw;
float sigmaqt;
float sigmath;
float z0;
float pwmin;
float pwmax;
float wmin;
float wmax;
float wlv;
float wtv;
float went;
float mindownw;
float b;
float qtn;
float thln;
float thvn;
float qcn;
float un;
float vn;
float qken;
float wn2;
float wn;
float thvk;
float pk;
float entexp;
float entw;
float beta_dm;
float entexp_m;
float rho_int;
float jump_thetav;
float jump_qt;
float jump_thetal;
float refthl;
float refthv;
float refqt;
float minrad;
float zminrad;
float radflux;
float f0;
float wst_rad;
float wst_dd;
bool cloudflg;
float sigq;
float xl;
float rsl;
float cpm;
float a;
float mf_cf;
float diffqt;
float fng;
float qww;
float alpha;
float beta;
float bb;
float f;
float pt;
float t;
float q2p;
float b9;
float satvp;
float rhgrid;
constexpr float wa = 1.f;
constexpr float wb = 1.5f;
constexpr float z00 = 100.f;
constexpr float bcoeff = 0.2f;
constexpr float l0 = 80;
constexpr float ent0 = 0.2f;
float dp;
float dl;
float adn;
constexpr int debug_mf = 0;
// bl_mynn_subroutines.F90:5965
dl = mpas_div(mpas_sub(1000.f, 500.f), ((float)(ndown)));
// bl_mynn_subroutines.F90:5966
pwmin = mynn_neg(3.f);
// bl_mynn_subroutines.F90:5967
pwmax = mynn_neg(1.f);
// bl_mynn_subroutines.F90:5970
for (int mynn_element=0; mynn_element < (downw).size(); ++mynn_element) { downw.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5971
for (int mynn_element=0; mynn_element < (downthl).size(); ++mynn_element) { downthl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5972
for (int mynn_element=0; mynn_element < (downthv).size(); ++mynn_element) { downthv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5973
for (int mynn_element=0; mynn_element < (downqt).size(); ++mynn_element) { downqt.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5974
for (int mynn_element=0; mynn_element < (downa).size(); ++mynn_element) { downa.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5975
for (int mynn_element=0; mynn_element < (downu).size(); ++mynn_element) { downu.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5976
for (int mynn_element=0; mynn_element < (downv).size(); ++mynn_element) { downv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5977
for (int mynn_element=0; mynn_element < (downqc).size(); ++mynn_element) { downqc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5978
for (int mynn_element=0; mynn_element < (ent).size(); ++mynn_element) { ent.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5979
for (int mynn_element=0; mynn_element < (dd_initk).size(); ++mynn_element) { dd_initk.linear(mynn_element) = 0; }
// bl_mynn_subroutines.F90:5981
for (int mynn_element=0; mynn_element < (edmf_a_dd).size(); ++mynn_element) { edmf_a_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5982
for (int mynn_element=0; mynn_element < (edmf_w_dd).size(); ++mynn_element) { edmf_w_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5983
for (int mynn_element=0; mynn_element < (edmf_qt_dd).size(); ++mynn_element) { edmf_qt_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5984
for (int mynn_element=0; mynn_element < (edmf_thl_dd).size(); ++mynn_element) { edmf_thl_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5985
for (int mynn_element=0; mynn_element < (edmf_ent_dd).size(); ++mynn_element) { edmf_ent_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5986
for (int mynn_element=0; mynn_element < (edmf_qc_dd).size(); ++mynn_element) { edmf_qc_dd.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5988
for (int mynn_element=0; mynn_element < (sd_aw).size(); ++mynn_element) { sd_aw.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5989
for (int mynn_element=0; mynn_element < (sd_awthl).size(); ++mynn_element) { sd_awthl.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5990
for (int mynn_element=0; mynn_element < (sd_awqt).size(); ++mynn_element) { sd_awqt.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5991
for (int mynn_element=0; mynn_element < (sd_awqv).size(); ++mynn_element) { sd_awqv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5992
for (int mynn_element=0; mynn_element < (sd_awqc).size(); ++mynn_element) { sd_awqc.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5993
for (int mynn_element=0; mynn_element < (sd_awu).size(); ++mynn_element) { sd_awu.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5994
for (int mynn_element=0; mynn_element < (sd_awv).size(); ++mynn_element) { sd_awv.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5995
for (int mynn_element=0; mynn_element < (sd_awqke).size(); ++mynn_element) { sd_awqke.linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:5998
cloudflg = false;
// bl_mynn_subroutines.F90:5999
minrad = 100.f;
// bl_mynn_subroutines.F90:6000
kminrad = kpbl;
// bl_mynn_subroutines.F90:6001
zminrad = pblh;
// bl_mynn_subroutines.F90:6002
qltop = 1;
// bl_mynn_subroutines.F90:6003
qlbase = 1;
// bl_mynn_subroutines.F90:6004
wthv = mpas_add(wthl, mpas_mul(svp1, wqt));
// bl_mynn_subroutines.F90:6005
for (k=mynn_max(((int)(3)), ((int)(((kpbl) - (2))))); (1) > 0 ? k <= (((kpbl) + (3))) : k >= (((kpbl) + (3))); k += (1)) {
// bl_mynn_subroutines.F90:6006
if (((((qc(k)) > (1.e-6f))) && (((cldfra_bl1d(k)) > (0.5f))))) {
// bl_mynn_subroutines.F90:6007
cloudflg = true;
// bl_mynn_subroutines.F90:6008
qltop = k;
// bl_mynn_subroutines.F90:6009
}
// bl_mynn_subroutines.F90:6010
}
// bl_mynn_subroutines.F90:6012
for (k=qltop; ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// bl_mynn_subroutines.F90:6013
if (((qc(k)) > (1e-6f))) {
// bl_mynn_subroutines.F90:6014
qlbase = k;
// bl_mynn_subroutines.F90:6015
}
// bl_mynn_subroutines.F90:6016
}
// bl_mynn_subroutines.F90:6017
qlbase = ((((qltop) + (qlbase))) / (2));
// bl_mynn_subroutines.F90:6021
for (i=1; (1) > 0 ? i <= (ndown) : i >= (ndown); i += (1)) {
// bl_mynn_subroutines.F90:6024
dd_initk(i) = qltop;
// bl_mynn_subroutines.F90:6025
}
// bl_mynn_subroutines.F90:6028
f0 = 0.f;
// bl_mynn_subroutines.F90:6029
for (k=1; (1) > 0 ? k <= (qltop) : k >= (qltop); k += (1)) {
// bl_mynn_subroutines.F90:6030
radflux = mpas_mul(rthraten(k), exner(k));
// bl_mynn_subroutines.F90:6031
radflux = mpas_mul(mpas_div(mpas_mul(radflux, cp), grav), mpas_sub(p(k), p(((k) + (1)))));
// bl_mynn_subroutines.F90:6032
if (((radflux) < (0.0f))) { f0 = mpas_add(mynn_abs(radflux), f0); }
// bl_mynn_subroutines.F90:6033
}
// bl_mynn_subroutines.F90:6034
f0 = mynn_max(((float)(f0)), ((float)(1.0f)));
// bl_mynn_subroutines.F90:6041
adn = mynn_min(((float)(mpas_add(0.05f, mpas_mul(f0, 0.001f)))), ((float)(0.3f)));
// bl_mynn_subroutines.F90:6044
if (cloudflg) {
// bl_mynn_subroutines.F90:6067
p700_ind = mynn_minloc(p.size(), [&] (int mynn_element) { return mynn_abs(mpas_sub(p.linear(mynn_element), 70000)); });
// bl_mynn_subroutines.F90:6068
jump_thetav = mpas_sub(mpas_sub(thv(p700_ind), thv(1)), mpas_mul(mpas_div(mpas_sub(thv(p700_ind), thv(((qltop) + (3)))), mpas_sub(zw(p700_ind), zw(((qltop) + (3))))), mpas_sub(zw(p700_ind), zw(qltop))));
// bl_mynn_subroutines.F90:6069
jump_qt = mpas_sub(mpas_sub(mpas_add(qc(p700_ind), qv(p700_ind)), qc(1)), qv(1));
// bl_mynn_subroutines.F90:6070
jump_thetal = mpas_sub(mpas_sub(thl(p700_ind), thl(1)), mpas_mul(mpas_div(mpas_sub(thl(p700_ind), thl(((qltop) + (3)))), mpas_sub(zw(p700_ind), zw(((qltop) + (3))))), mpas_sub(zw(p700_ind), zw(qltop))));
// bl_mynn_subroutines.F90:6072
refthl = thl(qltop);
// bl_mynn_subroutines.F90:6073
refthv = thv(qltop);
// bl_mynn_subroutines.F90:6074
refqt = qt(qltop);
// bl_mynn_subroutines.F90:6077
wst_rad = mynn_pow(((float)(mpas_div(mpas_mul(mpas_mul(grav, zw(qltop)), f0), mpas_mul(mpas_mul(refthl, rho(qltop)), cp)))), ((float)(0.333f)));
// bl_mynn_subroutines.F90:6078
wst_rad = mynn_max(((float)(wst_rad)), ((float)(0.1f)));
// bl_mynn_subroutines.F90:6079
wstar = mynn_max(((float)(mynn_pow(((float)(mpas_mul(mpas_mul(mpas_div(grav, thv(1)), wthv), pblh))), ((float)(onethird))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:6080
went = mpas_mul(mpas_div(thv(1), mpas_mul(mpas_mul(grav, jump_thetav), zw(qltop))), mpas_add(mpas_mul(0.15f, mpas_add(mynn_ipow<3>(wstar), mpas_mul(5, mynn_ipow<3>(ust)))), mpas_mul(0.35f, mynn_ipow<3>(wst_rad))));
// bl_mynn_subroutines.F90:6082
qstar = mynn_abs(mpas_div(mpas_mul(went, jump_qt), wst_rad));
// bl_mynn_subroutines.F90:6083
thstar = mpas_sub(mpas_div(mpas_div(mpas_div(f0, rho(qltop)), cp), wst_rad), mpas_div(mpas_mul(went, jump_thetav), wst_rad));
// bl_mynn_subroutines.F90:6085
wst_dd = mynn_pow(((float)(mpas_add(mpas_mul(0.15f, mpas_add(mynn_ipow<3>(wstar), mpas_mul(5, mynn_ipow<3>(ust)))), mpas_mul(0.35f, mynn_ipow<3>(wst_rad))))), ((float)(0.333f)));
// bl_mynn_subroutines.F90:6087
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:6088
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:6089
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:6091
sigmaw = mpas_mul(0.2f, wst_dd);
// bl_mynn_subroutines.F90:6092
sigmaqt = mpas_mul(40, qstar);
// bl_mynn_subroutines.F90:6093
sigmath = mpas_mul(1.0f, thstar);
// bl_mynn_subroutines.F90:6095
wmin = mpas_mul(sigmaw, pwmin);
// bl_mynn_subroutines.F90:6096
wmax = mpas_mul(sigmaw, pwmax);
// bl_mynn_subroutines.F90:6099
for (i=1; (1) > 0 ? i <= (ndown) : i >= (ndown); i += (1)) {
// bl_mynn_subroutines.F90:6100
ki = dd_initk(i);
// bl_mynn_subroutines.F90:6102
wlv = mpas_add(wmin, mpas_mul(mpas_div(mpas_sub(wmax, wmin), ((float)(ndown))), ((i) - (1))));
// bl_mynn_subroutines.F90:6103
wtv = mpas_add(wmin, mpas_mul(mpas_div(mpas_sub(wmax, wmin), ((float)(ndown))), i));
// bl_mynn_subroutines.F90:6106
downw(ki, i) = wlv;
// bl_mynn_subroutines.F90:6110
downa(ki, i) = mpas_div(adn, ((float)(ndown)));
// bl_mynn_subroutines.F90:6111
downu(ki, i) = mpas_div(mpas_add(mpas_mul(u(((ki) - (1))), dz(ki)), mpas_mul(u(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6112
downv(ki, i) = mpas_div(mpas_add(mpas_mul(v(((ki) - (1))), dz(ki)), mpas_mul(v(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6119
refthl = mpas_div(mpas_add(mpas_mul(thl(((ki) - (1))), dz(ki)), mpas_mul(thl(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6120
refthv = mpas_div(mpas_add(mpas_mul(thv(((ki) - (1))), dz(ki)), mpas_mul(thv(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6121
refqt = mpas_div(mpas_add(mpas_mul(qt(((ki) - (1))), dz(ki)), mpas_mul(qt(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6124
downqc(ki, i) = mpas_div(mpas_add(mpas_mul(qc(((ki) - (1))), dz(ki)), mpas_mul(qc(ki), dz(((ki) - (1))))), mpas_add(dz(ki), dz(((ki) - (1)))));
// bl_mynn_subroutines.F90:6125
downqt(ki, i) = refqt;
// bl_mynn_subroutines.F90:6126
downthv(ki, i) = mpas_add(refthv, mpas_div(mpas_mul(mpas_mul(0.01f, downw(ki, i)), sigmath), sigmaw));
// bl_mynn_subroutines.F90:6127
downthl(ki, i) = mpas_add(refthl, mpas_div(mpas_mul(mpas_mul(0.01f, downw(ki, i)), sigmath), sigmaw));
// bl_mynn_subroutines.F90:6135
}
// bl_mynn_subroutines.F90:6138
for (i=1; (1) > 0 ? i <= (ndown) : i >= (ndown); i += (1)) {
// bl_mynn_subroutines.F90:6139
dp = mpas_add(500.f, mpas_mul(dl, ((float)(i))));
// bl_mynn_subroutines.F90:6141
for (k=((dd_initk(i)) - (1)); ((-(1))) > 0 ? k <= (((kts) + (1))) : k >= (((kts) + (1))); k += ((-(1)))) {
// bl_mynn_subroutines.F90:6144
wmin = mpas_add(0.3f, mpas_mul(dp, 0.0005f));
// bl_mynn_subroutines.F90:6145
ent(k, i) = mpas_div(0.33f, mpas_mul(mynn_min(((float)(mynn_max(((float)(mpas_mul(mynn_neg(1.0f), downw(((k) + (1)), i)))), ((float)(wmin))))), ((float)(0.9f))), dp));
// bl_mynn_subroutines.F90:6150
entexp = mpas_mul(ent(k, i), dz(k));
// bl_mynn_subroutines.F90:6151
entexp_m = mpas_mul(mpas_mul(ent(k, i), 0.333f), dz(k));
// bl_mynn_subroutines.F90:6153
qtn = mpas_add(mpas_mul(downqt(((k) + (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qt(k), entexp));
// bl_mynn_subroutines.F90:6154
thln = mpas_add(mpas_mul(downthl(((k) + (1)), i), mpas_sub(1.f, entexp)), mpas_mul(thl(k), entexp));
// bl_mynn_subroutines.F90:6155
un = mpas_add(mpas_mul(downu(((k) + (1)), i), mpas_sub(1.f, entexp)), mpas_mul(u(k), entexp_m));
// bl_mynn_subroutines.F90:6156
vn = mpas_add(mpas_mul(downv(((k) + (1)), i), mpas_sub(1.f, entexp)), mpas_mul(v(k), entexp_m));
// bl_mynn_subroutines.F90:6165
pk = mpas_div(mpas_add(mpas_mul(p(((k) - (1))), dz(k)), mpas_mul(p(k), dz(((k) - (1))))), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:6166
f_condensation_edmf(qtn, thln, pk, zw(k), thvn, qcn);
// bl_mynn_subroutines.F90:6168
thvk = mpas_div(mpas_add(mpas_mul(thv(((k) - (1))), dz(k)), mpas_mul(thv(k), dz(((k) - (1))))), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:6169
b = mpas_mul(grav, mpas_sub(mpas_div(thvn, thvk), 1.0f));
// bl_mynn_subroutines.F90:6173
entw = entexp;
// bl_mynn_subroutines.F90:6180
mindownw = mynn_min(((float)(downw(((k) + (1)), i))), ((float)(mynn_neg(0.2f))));
// bl_mynn_subroutines.F90:6181
wn = mpas_add(downw(((k) + (1)), i), mpas_mul(mpas_sub(mpas_mul(mpas_mul(mynn_neg(2.f), ent(k, i)), downw(((k) + (1)), i)), mpas_div(mpas_mul(bcoeff, b), mindownw)), mynn_min(((float)(dz(k))), ((float)(250.f)))));
// bl_mynn_subroutines.F90:6186
if (((wn) < (mpas_sub(downw(((k) + (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, dz(k)), 200.f))), ((float)(mynn_neg(2.0f)))))))) {
// bl_mynn_subroutines.F90:6187
wn = mpas_sub(downw(((k) + (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, dz(k)), 200.f))), ((float)(mynn_neg(2.0f)))));
// bl_mynn_subroutines.F90:6188
}
// bl_mynn_subroutines.F90:6190
if (((wn) > (mpas_add(downw(((k) + (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, dz(k)), 200.f))), ((float)(2.0f))))))) {
// bl_mynn_subroutines.F90:6191
wn = mpas_add(downw(((k) + (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, dz(k)), 200.f))), ((float)(2.0f))));
// bl_mynn_subroutines.F90:6192
}
// bl_mynn_subroutines.F90:6193
wn = mynn_max(((float)(mynn_min(((float)(wn)), ((float)(0.0f))))), ((float)(mynn_neg(3.0f))));
// bl_mynn_subroutines.F90:6202
if (((wn) < (0.f))) {
// bl_mynn_subroutines.F90:6203
downw(k, i) = wn;
// bl_mynn_subroutines.F90:6204
downthv(k, i) = thvn;
// bl_mynn_subroutines.F90:6205
downthl(k, i) = thln;
// bl_mynn_subroutines.F90:6206
downqt(k, i) = qtn;
// bl_mynn_subroutines.F90:6207
downqc(k, i) = qcn;
// bl_mynn_subroutines.F90:6208
downu(k, i) = un;
// bl_mynn_subroutines.F90:6209
downv(k, i) = vn;
// bl_mynn_subroutines.F90:6210
downa(k, i) = downa(((k) + (1)), i);
// bl_mynn_subroutines.F90:6211
} else {
// bl_mynn_subroutines.F90:6213
if (((((dd_initk(i)) - (k))) < (2))) {
// bl_mynn_subroutines.F90:6214
for (int mynn_element=0; mynn_element < (downw.slice(MynnRange{downw.lo[0], downw.hi[0]}, i)).size(); ++mynn_element) { downw.slice(MynnRange{downw.lo[0], downw.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6215
for (int mynn_element=0; mynn_element < (downthv.slice(MynnRange{downthv.lo[0], downthv.hi[0]}, i)).size(); ++mynn_element) { downthv.slice(MynnRange{downthv.lo[0], downthv.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6216
for (int mynn_element=0; mynn_element < (downthl.slice(MynnRange{downthl.lo[0], downthl.hi[0]}, i)).size(); ++mynn_element) { downthl.slice(MynnRange{downthl.lo[0], downthl.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6217
for (int mynn_element=0; mynn_element < (downqt.slice(MynnRange{downqt.lo[0], downqt.hi[0]}, i)).size(); ++mynn_element) { downqt.slice(MynnRange{downqt.lo[0], downqt.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6218
for (int mynn_element=0; mynn_element < (downqc.slice(MynnRange{downqc.lo[0], downqc.hi[0]}, i)).size(); ++mynn_element) { downqc.slice(MynnRange{downqc.lo[0], downqc.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6219
for (int mynn_element=0; mynn_element < (downu.slice(MynnRange{downu.lo[0], downu.hi[0]}, i)).size(); ++mynn_element) { downu.slice(MynnRange{downu.lo[0], downu.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6220
for (int mynn_element=0; mynn_element < (downv.slice(MynnRange{downv.lo[0], downv.hi[0]}, i)).size(); ++mynn_element) { downv.slice(MynnRange{downv.lo[0], downv.hi[0]}, i).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6221
}
// bl_mynn_subroutines.F90:6222
break;
// bl_mynn_subroutines.F90:6223
}
// bl_mynn_subroutines.F90:6224
}
// bl_mynn_subroutines.F90:6225
}
// bl_mynn_subroutines.F90:6226
}
// bl_mynn_subroutines.F90:6228
for (int mynn_element=0; mynn_element < (downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]})).size(); ++mynn_element) { downw.slice(1, MynnRange{downw.lo[1], downw.hi[1]}).linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:6229
for (int mynn_element=0; mynn_element < (downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]})).size(); ++mynn_element) { downa.slice(1, MynnRange{downa.lo[1], downa.hi[1]}).linear(mynn_element) = 0.f; }
// bl_mynn_subroutines.F90:6233
for (k=qltop; ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// bl_mynn_subroutines.F90:6234
for (i=1; (1) > 0 ? i <= (ndown) : i >= (ndown); i += (1)) {
// bl_mynn_subroutines.F90:6235
edmf_a_dd(k) = mpas_add(edmf_a_dd(k), downa(((k) - (1)), i));
// bl_mynn_subroutines.F90:6236
edmf_w_dd(k) = mpas_add(edmf_w_dd(k), mpas_mul(downa(((k) - (1)), i), downw(((k) - (1)), i)));
// bl_mynn_subroutines.F90:6237
edmf_qt_dd(k) = mpas_add(edmf_qt_dd(k), mpas_mul(downa(((k) - (1)), i), downqt(((k) - (1)), i)));
// bl_mynn_subroutines.F90:6238
edmf_thl_dd(k) = mpas_add(edmf_thl_dd(k), mpas_mul(downa(((k) - (1)), i), downthl(((k) - (1)), i)));
// bl_mynn_subroutines.F90:6239
edmf_ent_dd(k) = mpas_add(edmf_ent_dd(k), mpas_mul(downa(((k) - (1)), i), ent(((k) - (1)), i)));
// bl_mynn_subroutines.F90:6240
edmf_qc_dd(k) = mpas_add(edmf_qc_dd(k), mpas_mul(downa(((k) - (1)), i), downqc(((k) - (1)), i)));
// bl_mynn_subroutines.F90:6241
}
// bl_mynn_subroutines.F90:6243
if (((edmf_a_dd(k)) > (0.f))) {
// bl_mynn_subroutines.F90:6244
edmf_w_dd(k) = mpas_div(edmf_w_dd(k), edmf_a_dd(k));
// bl_mynn_subroutines.F90:6245
edmf_qt_dd(k) = mpas_div(edmf_qt_dd(k), edmf_a_dd(k));
// bl_mynn_subroutines.F90:6246
edmf_thl_dd(k) = mpas_div(edmf_thl_dd(k), edmf_a_dd(k));
// bl_mynn_subroutines.F90:6247
edmf_ent_dd(k) = mpas_div(edmf_ent_dd(k), edmf_a_dd(k));
// bl_mynn_subroutines.F90:6248
edmf_qc_dd(k) = mpas_div(edmf_qc_dd(k), edmf_a_dd(k));
// bl_mynn_subroutines.F90:6249
}
// bl_mynn_subroutines.F90:6250
}
// bl_mynn_subroutines.F90:6256
for (k=kts; (1) > 0 ? k <= (qltop) : k >= (qltop); k += (1)) {
// bl_mynn_subroutines.F90:6257
rho_int = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) + (1)))), mpas_mul(rho(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:6258
for (i=1; (1) > 0 ? i <= (ndown) : i >= (ndown); i += (1)) {
// bl_mynn_subroutines.F90:6259
sd_aw(k) = mpas_add(sd_aw(k), mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)));
// bl_mynn_subroutines.F90:6260
sd_awthl(k) = mpas_add(sd_awthl(k), mpas_mul(mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)), downthl(k, i)));
// bl_mynn_subroutines.F90:6261
sd_awqt(k) = mpas_add(sd_awqt(k), mpas_mul(mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)), downqt(k, i)));
// bl_mynn_subroutines.F90:6262
sd_awqc(k) = mpas_add(sd_awqc(k), mpas_mul(mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)), downqc(k, i)));
// bl_mynn_subroutines.F90:6263
sd_awu(k) = mpas_add(sd_awu(k), mpas_mul(mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)), downu(k, i)));
// bl_mynn_subroutines.F90:6264
sd_awv(k) = mpas_add(sd_awv(k), mpas_mul(mpas_mul(mpas_mul(rho_int, downa(k, i)), downw(k, i)), downv(k, i)));
// bl_mynn_subroutines.F90:6265
}
// bl_mynn_subroutines.F90:6266
sd_awqv(k) = mpas_sub(sd_awqt(k), sd_awqc(k));
// bl_mynn_subroutines.F90:6267
}
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:6273-6344
__device__ __noinline__ void f_scale_aware(float dx, float pbl1, float& psig_bl, float& psig_shcu) {
MynnFrame mynn_frame(arena);
float dxdh;
// bl_mynn_subroutines.F90:6288
psig_bl = 1.0f;
// bl_mynn_subroutines.F90:6289
psig_shcu = 1.0f;
// bl_mynn_subroutines.F90:6290
dxdh = mpas_div(mynn_max(((float)(mpas_mul(2.5f, dx))), ((float)(10.f))), mynn_min(((float)(pbl1)), ((float)(3000.f))));
// bl_mynn_subroutines.F90:6298
psig_bl = mpas_div(mpas_add(mynn_ipow<2>(dxdh), mpas_mul(0.106f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), mpas_add(mpas_add(mynn_ipow<2>(dxdh), mpas_mul(0.066f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), 0.071f));
// bl_mynn_subroutines.F90:6301
dxdh = mpas_div(mynn_max(((float)(mpas_mul(2.5f, dx))), ((float)(10.f))), mynn_min(((float)(mpas_add(pbl1, 500.f))), ((float)(3500.f))));
// bl_mynn_subroutines.F90:6328
psig_shcu = mpas_div(mpas_add(mynn_ipow<2>(dxdh), mpas_mul(0.145f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), mpas_add(mpas_add(mynn_ipow<2>(dxdh), mpas_mul(0.172f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), 0.170f));
// bl_mynn_subroutines.F90:6338
if (((psig_bl) > (1.0f))) { psig_bl = 1.0f; }
// bl_mynn_subroutines.F90:6339
if (((psig_bl) < (0.0f))) { psig_bl = 0.0f; }
// bl_mynn_subroutines.F90:6341
if (((psig_shcu) > (1.0f))) { psig_shcu = 1.0f; }
// bl_mynn_subroutines.F90:6342
if (((psig_shcu) < (0.0f))) { psig_shcu = 0.0f; }
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:6346-6395
__device__ __noinline__ float f_phim(float zet) {
MynnFrame mynn_frame(arena);
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phi_m;
float phim;
// bl_mynn_subroutines.F90:6362
if (((zet) >= (0.0f))) {
// bl_mynn_subroutines.F90:6363
dummy_0 = mpas_add(1, mynn_pow(((float)(zet)), ((float)(bm_st))));
// bl_mynn_subroutines.F90:6364
dummy_1 = mpas_add(zet, mynn_pow(((float)(dummy_0)), ((float)(rbm_st))));
// bl_mynn_subroutines.F90:6365
dummy_11 = mpas_add(1, mpas_mul(mynn_pow(((float)(dummy_0)), ((float)(mpas_sub(rbm_st, 1)))), mynn_pow(((float)(zet)), ((float)(mpas_sub(bm_st, 1))))));
// bl_mynn_subroutines.F90:6366
dummy_2 = mpas_mul(mpas_div(mynn_neg(am_st), dummy_1), dummy_11);
// bl_mynn_subroutines.F90:6367
phi_m = mpas_sub(1, mpas_mul(zet, dummy_2));
// bl_mynn_subroutines.F90:6368
} else {
// bl_mynn_subroutines.F90:6369
dummy_0 = mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(cphm_unst, zet)))), ((float)(0.25f)));
// bl_mynn_subroutines.F90:6370
phi_m = mpas_div(1.f, dummy_0);
// bl_mynn_subroutines.F90:6371
dummy_psi = mpas_add(mpas_sub(mpas_add(mpas_mul(2.f, mynn_log(mpas_mul(0.5f, mpas_add(1.f, dummy_0)))), mynn_log(mpas_mul(0.5f, mpas_add(1.f, mynn_ipow<2>(dummy_0))))), mpas_mul(2.f, mynn_atan(dummy_0))), 1.570796f);
// bl_mynn_subroutines.F90:6373
dummy_0 = mpas_sub(1.f, mpas_mul(am_unst, zet));
// bl_mynn_subroutines.F90:6374
dummy_1 = mynn_pow(((float)(dummy_0)), ((float)(0.333333f)));
// bl_mynn_subroutines.F90:6375
dummy_11 = mpas_mul(mpas_mul(mynn_neg(0.33333f), am_unst), mynn_pow(((float)(dummy_0)), ((float)(mynn_neg(0.6666667f)))));
// bl_mynn_subroutines.F90:6376
dummy_2 = mpas_mul(0.33333f, mpas_add(mpas_add(mynn_ipow<2>(dummy_1), dummy_1), 1.f));
// bl_mynn_subroutines.F90:6377
dummy_22 = mpas_mul(mpas_mul(0.3333f, dummy_11), mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6378
dummy_3 = mpas_mul(0.57735f, mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6379
dummy_33 = mpas_mul(1.1547f, dummy_11);
// bl_mynn_subroutines.F90:6380
dummy_4 = mpas_add(mpas_sub(mpas_mul(1.5f, mynn_log(dummy_2)), mpas_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// bl_mynn_subroutines.F90:6381
dummy_44 = mpas_sub(mpas_mul(mpas_div(1.5f, dummy_2), dummy_22), mpas_div(mpas_mul(1.73205f, dummy_33), mpas_add(1.f, mynn_ipow<2>(dummy_3))));
// bl_mynn_subroutines.F90:6383
dummy_0 = mynn_ipow<2>(zet);
// bl_mynn_subroutines.F90:6384
dummy_1 = mpas_div(1.f, mpas_add(1.f, dummy_0));
// bl_mynn_subroutines.F90:6385
dummy_11 = mpas_mul(2.f, zet);
// bl_mynn_subroutines.F90:6386
dummy_2 = mpas_mul(mpas_add(mpas_add(mpas_div(mpas_sub(1, phi_m), zet), mpas_mul(dummy_11, dummy_4)), mpas_mul(dummy_0, dummy_44)), dummy_1);
// bl_mynn_subroutines.F90:6387
dummy_22 = mpas_mul(mpas_mul(mynn_neg(dummy_11), mpas_add(dummy_psi, mpas_mul(dummy_0, dummy_4))), mynn_ipow<2>(dummy_1));
// bl_mynn_subroutines.F90:6389
phi_m = mpas_sub(1.f, mpas_mul(zet, mpas_add(dummy_2, dummy_22)));
// bl_mynn_subroutines.F90:6390
}
// bl_mynn_subroutines.F90:6393
phim = phi_m;
return phim;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:6398-6444
__device__ __noinline__ float f_phih(float zet) {
MynnFrame mynn_frame(arena);
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phh;
float phih;
// bl_mynn_subroutines.F90:6414
if (((zet) >= (0.0f))) {
// bl_mynn_subroutines.F90:6415
dummy_0 = mpas_add(1, mynn_pow(((float)(zet)), ((float)(bh_st))));
// bl_mynn_subroutines.F90:6416
dummy_1 = mpas_add(zet, mynn_pow(((float)(dummy_0)), ((float)(rbh_st))));
// bl_mynn_subroutines.F90:6417
dummy_11 = mpas_add(1, mpas_mul(mynn_pow(((float)(dummy_0)), ((float)(mpas_sub(rbh_st, 1)))), mynn_pow(((float)(zet)), ((float)(mpas_sub(bh_st, 1))))));
// bl_mynn_subroutines.F90:6418
dummy_2 = mpas_mul(mpas_div(mynn_neg(ah_st), dummy_1), dummy_11);
// bl_mynn_subroutines.F90:6419
phih = mpas_sub(1, mpas_mul(zet, dummy_2));
// bl_mynn_subroutines.F90:6420
} else {
// bl_mynn_subroutines.F90:6421
dummy_0 = mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(cphh_unst, zet)))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:6422
phh = mpas_div(1.f, dummy_0);
// bl_mynn_subroutines.F90:6423
dummy_psi = mpas_mul(2.f, mynn_log(mpas_mul(0.5f, mpas_add(1.f, dummy_0))));
// bl_mynn_subroutines.F90:6425
dummy_0 = mpas_sub(1.f, mpas_mul(ah_unst, zet));
// bl_mynn_subroutines.F90:6426
dummy_1 = mynn_pow(((float)(dummy_0)), ((float)(0.333333f)));
// bl_mynn_subroutines.F90:6427
dummy_11 = mpas_mul(mpas_mul(mynn_neg(0.33333f), ah_unst), mynn_pow(((float)(dummy_0)), ((float)(mynn_neg(0.6666667f)))));
// bl_mynn_subroutines.F90:6428
dummy_2 = mpas_mul(0.33333f, mpas_add(mpas_add(mynn_ipow<2>(dummy_1), dummy_1), 1.f));
// bl_mynn_subroutines.F90:6429
dummy_22 = mpas_mul(mpas_mul(0.3333f, dummy_11), mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6430
dummy_3 = mpas_mul(0.57735f, mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6431
dummy_33 = mpas_mul(1.1547f, dummy_11);
// bl_mynn_subroutines.F90:6432
dummy_4 = mpas_add(mpas_sub(mpas_mul(1.5f, mynn_log(dummy_2)), mpas_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// bl_mynn_subroutines.F90:6433
dummy_44 = mpas_sub(mpas_mul(mpas_div(1.5f, dummy_2), dummy_22), mpas_div(mpas_mul(1.73205f, dummy_33), mpas_add(1.f, mynn_ipow<2>(dummy_3))));
// bl_mynn_subroutines.F90:6435
dummy_0 = mynn_ipow<2>(zet);
// bl_mynn_subroutines.F90:6436
dummy_1 = mpas_div(1.f, mpas_add(1.f, dummy_0));
// bl_mynn_subroutines.F90:6437
dummy_11 = mpas_mul(2.f, zet);
// bl_mynn_subroutines.F90:6438
dummy_2 = mpas_mul(mpas_add(mpas_add(mpas_div(mpas_sub(1, phh), zet), mpas_mul(dummy_11, dummy_4)), mpas_mul(dummy_0, dummy_44)), dummy_1);
// bl_mynn_subroutines.F90:6439
dummy_22 = mpas_mul(mpas_mul(mynn_neg(dummy_11), mpas_add(dummy_psi, mpas_mul(dummy_0, dummy_4))), mynn_ipow<2>(dummy_1));
// bl_mynn_subroutines.F90:6441
phih = mpas_sub(1.f, mpas_mul(zet, mpas_add(dummy_2, dummy_22)));
// bl_mynn_subroutines.F90:6442
}
return phih;
}
// MPAS physics_mmm/bl_mynn_subroutines.F90:6446-6560
__device__ __noinline__ void f_topdown_cloudrad(int kts, int kte, MynnArray<float, 1> dz1, MynnArray<float, 1> zw, float fltv, float xland, int kpbl, float pblh, MynnArray<float, 1> sqc, MynnArray<float, 1> sqi, MynnArray<float, 1> sqw, MynnArray<float, 1> thl, MynnArray<float, 1> th1, MynnArray<float, 1> ex1, MynnArray<float, 1> p1, MynnArray<float, 1> rho1, MynnArray<float, 1> thetav, MynnArray<float, 1> cldfra_bl1d, MynnArray<float, 1> rthraten, float& maxkhtopdown, MynnArray<float, 1> khtopdown, MynnArray<float, 1> tkeprodtd) {
MynnFrame mynn_frame(arena);
dz1.rebase({kts});
zw.rebase({kts});
sqc.rebase({kts});
sqi.rebase({kts});
sqw.rebase({kts});
thl.rebase({kts});
th1.rebase({kts});
ex1.rebase({kts});
p1.rebase({kts});
rho1.rebase({kts});
thetav.rebase({kts});
cldfra_bl1d.rebase({kts});
rthraten.rebase({kts});
khtopdown.rebase({kts});
tkeprodtd.rebase({kts});
MynnArray<float, 1> zfac = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> wscalek2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> zfacent = mynn_array<float, 1>(arena, {kts}, {kte});
float bfx0;
float wm2;
float wm3;
float bfxpbl;
float dthvx;
float tmp1;
float temps;
float templ;
float zl1;
float wstar3_2;
float ent_eff;
float radsum;
float radflux;
float we;
float rcldb;
float rvls;
float minrad;
float zminrad;
constexpr float pfac = 2.0f;
constexpr float zfmin = 0.01f;
constexpr float phifac = 8.0f;
int k;
int kk;
int kminrad;
bool cloudflg;
// bl_mynn_subroutines.F90:6473
cloudflg = false;
// bl_mynn_subroutines.F90:6474
minrad = 100.f;
// bl_mynn_subroutines.F90:6475
kminrad = kpbl;
// bl_mynn_subroutines.F90:6476
zminrad = pblh;
// bl_mynn_subroutines.F90:6477
for (int mynn_element=0; mynn_element < (khtopdown.slice(MynnRange{kts, kte})).size(); ++mynn_element) { khtopdown.slice(MynnRange{kts, kte}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6478
for (int mynn_element=0; mynn_element < (tkeprodtd.slice(MynnRange{kts, kte})).size(); ++mynn_element) { tkeprodtd.slice(MynnRange{kts, kte}).linear(mynn_element) = 0.0f; }
// bl_mynn_subroutines.F90:6479
maxkhtopdown = 0.0f;
// bl_mynn_subroutines.F90:6482
for (kk=mynn_max(((int)(1)), ((int)(((kpbl) - (2))))); (1) > 0 ? kk <= (((kpbl) + (3))) : kk >= (((kpbl) + (3))); kk += (1)) {
// bl_mynn_subroutines.F90:6483
if (((((((sqc(kk)) > (1.e-6f))) || (((sqi(kk)) > (1.e-6f))))) || (((cldfra_bl1d(kk)) > (0.5f))))) {
// bl_mynn_subroutines.F90:6485
cloudflg = true;
// bl_mynn_subroutines.F90:6486
}
// bl_mynn_subroutines.F90:6487
if (((rthraten(kk)) < (minrad))) {
// bl_mynn_subroutines.F90:6488
minrad = rthraten(kk);
// bl_mynn_subroutines.F90:6489
kminrad = kk;
// bl_mynn_subroutines.F90:6490
zminrad = mpas_add(zw(kk), mpas_mul(0.5f, dz1(kk)));
// bl_mynn_subroutines.F90:6491
}
// bl_mynn_subroutines.F90:6492
}
// bl_mynn_subroutines.F90:6494
if (((mynn_max(((int)(kminrad)), ((int)(kpbl)))) < (2))) { cloudflg = false; }
// bl_mynn_subroutines.F90:6495
if (cloudflg) {
// bl_mynn_subroutines.F90:6496
zl1 = dz1(kts);
// bl_mynn_subroutines.F90:6497
k = mynn_max(((int)(((kpbl) - (1)))), ((int)(((kminrad) - (1)))));
// bl_mynn_subroutines.F90:6501
templ = mpas_mul(thl(k), ex1(k));
// bl_mynn_subroutines.F90:6503
rvls = mpas_mul(mpas_mul(mpas_mul(100.f, 6.112f), mynn_exp(mpas_div(mpas_mul(17.67f, mpas_sub(templ, 273.16f)), mpas_sub(templ, 29.65f)))), mpas_div(ep_2, p1(((k) + (1)))));
// bl_mynn_subroutines.F90:6504
temps = mpas_add(templ, mpas_div(mpas_sub(sqw(k), rvls), mpas_add(mpas_div(cp, xlv), mpas_div(mpas_mul(mpas_mul(ep_2, xlv), rvls), mpas_mul(r_d, mynn_ipow<2>(templ))))));
// bl_mynn_subroutines.F90:6505
rvls = mpas_mul(mpas_mul(mpas_mul(100.f, 6.112f), mynn_exp(mpas_div(mpas_mul(17.67f, mpas_sub(temps, 273.15f)), mpas_sub(temps, 29.65f)))), mpas_div(ep_2, p1(((k) + (1)))));
// bl_mynn_subroutines.F90:6506
rcldb = mynn_max(((float)(mpas_sub(sqw(k), rvls))), ((float)(0.f)));
// bl_mynn_subroutines.F90:6509
dthvx = mpas_sub(mpas_add(thl(((k) + (2))), mpas_mul(mpas_mul(th1(((k) + (2))), p608), sqw(((k) + (2))))), mpas_add(thl(k), mpas_mul(mpas_mul(th1(k), p608), sqw(k))));
// bl_mynn_subroutines.F90:6511
dthvx = mynn_max(((float)(dthvx)), ((float)(0.1f)));
// bl_mynn_subroutines.F90:6512
tmp1 = mpas_div(mpas_mul(xlvcp, rcldb), mpas_mul(ex1(k), dthvx));
// bl_mynn_subroutines.F90:6515
ent_eff = mpas_add(0.2f, mpas_mul(mpas_mul(0.2f, 8.f), tmp1));
// bl_mynn_subroutines.F90:6517
radsum = 0.f;
// bl_mynn_subroutines.F90:6518
for (kk=mynn_max(((int)(1)), ((int)(((kpbl) - (3))))); (1) > 0 ? kk <= (((kpbl) + (3))) : kk >= (((kpbl) + (3))); kk += (1)) {
// bl_mynn_subroutines.F90:6519
radflux = mpas_mul(rthraten(kk), ex1(kk));
// bl_mynn_subroutines.F90:6520
radflux = mpas_mul(mpas_div(mpas_mul(radflux, cp), grav), mpas_sub(p1(kk), p1(((kk) + (1)))));
// bl_mynn_subroutines.F90:6521
if (((radflux) < (0.0f))) { radsum = mpas_add(mynn_abs(radflux), radsum); }
// bl_mynn_subroutines.F90:6522
}
// bl_mynn_subroutines.F90:6525
if (((mpas_sub(xland, 1.5f)) >= (0))) {
// bl_mynn_subroutines.F90:6526
radsum = mynn_min(((float)(radsum)), ((float)(90.0f)));
// bl_mynn_subroutines.F90:6527
bfx0 = mynn_max(((float)(mpas_div(mpas_div(radsum, rho1(k)), cp))), ((float)(0.f)));
// bl_mynn_subroutines.F90:6528
} else {
// bl_mynn_subroutines.F90:6529
radsum = mynn_min(((float)(mpas_mul(0.25f, radsum))), ((float)(30.0f)));
// bl_mynn_subroutines.F90:6530
bfx0 = mynn_max(((float)(mpas_sub(mpas_div(mpas_div(radsum, rho1(k)), cp), mynn_max(((float)(fltv)), ((float)(0.0f)))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:6531
}
// bl_mynn_subroutines.F90:6534
wm3 = mpas_mul(mpas_mul(mpas_div(grav, thetav(k)), bfx0), mynn_min(((float)(pblh)), ((float)(1500.f))));
// bl_mynn_subroutines.F90:6536
bfxpbl = mpas_mul(mynn_neg(ent_eff), bfx0);
// bl_mynn_subroutines.F90:6537
dthvx = mynn_max(((float)(mpas_sub(thetav(((k) + (1))), thetav(k)))), ((float)(0.1f)));
// bl_mynn_subroutines.F90:6538
we = mynn_max(((float)(mpas_div(bfxpbl, dthvx))), ((float)(mynn_neg(mynn_sqrt(mynn_pow(((float)(wm3)), ((float)(twothirds))))))));
// bl_mynn_subroutines.F90:6540
for (kk=kts; (1) > 0 ? kk <= (((kpbl) + (3))) : kk >= (((kpbl) + (3))); kk += (1)) {
// bl_mynn_subroutines.F90:6542
zfac(kk) = mynn_min(((float)(mynn_max(((float)(mpas_sub(1.f, mpas_div(mpas_sub(zw(((kk) + (1))), zl1), mpas_sub(zminrad, zl1))))), ((float)(zfmin))))), ((float)(1.f)));
// bl_mynn_subroutines.F90:6543
zfacent(kk) = mpas_mul(mpas_mul(10.f, mynn_max(((float)(mpas_div(mpas_sub(zminrad, zw(((kk) + (1)))), zminrad))), ((float)(0.0f)))), mynn_ipow<3>(mpas_sub(1.f, zfac(kk))));
// bl_mynn_subroutines.F90:6546
wscalek2(kk) = mynn_pow(((float)(mpas_mul(mpas_mul(mpas_mul(phifac, karman), wm3), zfac(kk)))), ((float)(onethird)));
// bl_mynn_subroutines.F90:6548
khtopdown(kk) = mpas_mul(mpas_mul(mpas_mul(wscalek2(kk), karman), mpas_sub(zminrad, zw(((kk) + (1))))), mynn_ipow<3>(mpas_sub(1.f, zfac(kk))));
// bl_mynn_subroutines.F90:6549
khtopdown(kk) = mynn_max(((float)(khtopdown(kk))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:6554
tkeprodtd(kk) = mpas_mul(mpas_div(mpas_mul(mpas_mul(2.f, ent_eff), wm3), mynn_max(((float)(pblh)), ((float)(100.f)))), zfacent(kk));
// bl_mynn_subroutines.F90:6555
tkeprodtd(kk) = mynn_max(((float)(tkeprodtd(kk))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:6556
}
// bl_mynn_subroutines.F90:6557
}
// bl_mynn_subroutines.F90:6558
maxkhtopdown = mynn_maxval(khtopdown.slice(MynnRange{khtopdown.lo[0], khtopdown.hi[0]}));
}
// MPAS physics_mmm/bl_mynn.F90:24-109
__device__ __noinline__ void f_bl_mynn_init(float con_cp, float con_cpv, float con_cice, float con_cliq, float con_ep1, float con_ep2, float con_grav, float con_karman, float con_p0, float con_rd, float con_rv, float con_svp1, float con_svp2, float con_svp3, float con_svpt0, float con_xlf, float con_xls, float con_xlv, MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// bl_mynn.F90:70
cp = con_cp;
// bl_mynn.F90:71
cpv = con_cpv;
// bl_mynn.F90:72
cliq = con_cliq;
// bl_mynn.F90:73
cice = con_cice;
// bl_mynn.F90:74
ep_1 = con_ep1;
// bl_mynn.F90:75
ep_2 = con_ep2;
// bl_mynn.F90:76
grav = con_grav;
// bl_mynn.F90:77
karman = con_karman;
// bl_mynn.F90:78
p1000mb = con_p0;
// bl_mynn.F90:79
r_d = con_rd;
// bl_mynn.F90:80
r_v = con_rv;
// bl_mynn.F90:81
rvovrd = mpas_div(r_v, r_d);
// bl_mynn.F90:82
svp1 = con_svp1;
// bl_mynn.F90:83
svp2 = con_svp2;
// bl_mynn.F90:84
svp3 = con_svp3;
// bl_mynn.F90:85
svpt0 = con_svpt0;
// bl_mynn.F90:86
xlf = con_xlf;
// bl_mynn.F90:87
xls = con_xls;
// bl_mynn.F90:88
xlv = con_xlv;
// bl_mynn.F90:91
ep_3 = mpas_sub(1.f, ep_2);
// bl_mynn.F90:92
gtr = mpas_div(grav, tref);
// bl_mynn.F90:93
p608 = ep_1;
// bl_mynn.F90:94
rcp = mpas_div(r_d, cp);
// bl_mynn.F90:95
t0c = svpt0;
// bl_mynn.F90:96
tv0 = mpas_mul(p608, tref);
// bl_mynn.F90:97
xlscp = mpas_div(mpas_add(xlv, xlf), cp);
// bl_mynn.F90:98
xlvcp = mpas_div(xlv, cp);
// bl_mynn.F90:106
errmsg = " ";
// bl_mynn.F90:107
errflg = 0;
}
// MPAS physics_mmm/bl_mynn.F90:112-124
__device__ __noinline__ void f_bl_mynn_finalize(MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
// bl_mynn.F90:121
errmsg = " ";
// bl_mynn.F90:122
errflg = 0;
}
// MPAS physics_mmm/bl_mynn.F90:127-1240
__device__ __noinline__ void f_bl_mynn_run(int initflag, bool restart, bool cycling, float delt, MynnArray<float, 2> dz, MynnArray<float, 1> dx, MynnArray<float, 1> znt, MynnArray<float, 2> u, MynnArray<float, 2> v, MynnArray<float, 2> w, MynnArray<float, 2> th, MynnArray<float, 2> sqv, MynnArray<float, 2> sqc, MynnArray<float, 2> sqi, MynnArray<float, 2> sqs, MynnArray<float, 2> qnc, MynnArray<float, 2> qni, MynnArray<float, 2> qnwfa, MynnArray<float, 2> qnifa, MynnArray<float, 2> qnbca, MynnArray<float, 2> qozone, MynnArray<float, 2> p, MynnArray<float, 2> exner, MynnArray<float, 2> rho, MynnArray<float, 2> tt, MynnArray<float, 1> xland, MynnArray<float, 1> ts, MynnArray<float, 1> qsfc, MynnArray<float, 1> ps, MynnArray<float, 1> ust, MynnArray<float, 1> ch, MynnArray<float, 1> hfx, MynnArray<float, 1> qfx, MynnArray<float, 1> rmol, MynnArray<float, 1> wspd, MynnArray<float, 1> uoce, MynnArray<float, 1> voce, MynnArray<float, 2> qke, MynnArray<float, 2> qke_adv, MynnArray<float, 2> tsq, MynnArray<float, 2> qsq, MynnArray<float, 2> cov, MynnArray<float, 2> rublten, MynnArray<float, 2> rvblten, MynnArray<float, 2> rthblten, MynnArray<float, 2> rqvblten, MynnArray<float, 2> rqcblten, MynnArray<float, 2> rqiblten, MynnArray<float, 2> rqsblten, MynnArray<float, 2> rqncblten, MynnArray<float, 2> rqniblten, MynnArray<float, 2> rqnwfablten, MynnArray<float, 2> rqnifablten, MynnArray<float, 2> rqnbcablten, MynnArray<float, 2> rqozblten, MynnArray<float, 2> exch_h, MynnArray<float, 2> exch_m, MynnArray<float, 1> pblh, MynnArray<int, 1> kpbl, MynnArray<float, 2> el_pbl, MynnArray<float, 2> dqke, MynnArray<float, 2> qwt, MynnArray<float, 2> qshear, MynnArray<float, 2> qbuoy, MynnArray<float, 2> qdiss, MynnArray<float, 2> sh, MynnArray<float, 2> sm, MynnArray<float, 2> qc_bl, MynnArray<float, 2> qi_bl, MynnArray<float, 2> cldfra_bl, int icloud_bl, bool bl_mynn_tkeadvect, bool bl_mynn_tkebudget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_stfunc, bool bl_mynn_topdown, bool bl_mynn_edmf, bool bl_mynn_edmf_dd, bool bl_mynn_edmf_mom, bool bl_mynn_edmf_tke, bool bl_mynn_mixscalars, bool bl_mynn_output, bool bl_mynn_cloudmix, bool bl_mynn_mixqt, bool bl_mynn_scaleaware, bool bl_mynn_dheatopt, MynnArray<float, 2> edmf_a, MynnArray<float, 2> edmf_w, MynnArray<float, 2> edmf_qt, MynnArray<float, 2> edmf_thl, MynnArray<float, 2> edmf_ent, MynnArray<float, 2> edmf_qc, MynnArray<float, 2> sub_thl, MynnArray<float, 2> sub_sqv, MynnArray<float, 2> det_thl, MynnArray<float, 2> det_sqv, MynnArray<float, 2> edmf_a_dd, MynnArray<float, 2> edmf_w_dd, MynnArray<float, 2> edmf_qt_dd, MynnArray<float, 2> edmf_thl_dd, MynnArray<float, 2> edmf_ent_dd, MynnArray<float, 2> edmf_qc_dd, MynnArray<float, 1> maxwidth, MynnArray<float, 1> maxmf, MynnArray<float, 1> ztop_plume, MynnArray<int, 1> ktop_plume, int spp_pbl, MynnArray<float, 2> pattern_spp_pbl, MynnArray<float, 2> rthraten, bool flag_qc, bool flag_qi, bool flag_qs, bool flag_qnc, bool flag_qni, bool flag_qnwfa, bool flag_qnifa, bool flag_qnbca, bool flag_qoz, int its, int ite, int kts, int kte, int kme, MynnText& errmsg, int& errflg) {
MynnFrame mynn_frame(arena);
dz.rebase({its, 1});
dx.rebase({its});
znt.rebase({its});
u.rebase({its, 1});
v.rebase({its, 1});
w.rebase({its, 1});
th.rebase({its, 1});
sqv.rebase({its, 1});
sqc.rebase({its, 1});
sqi.rebase({its, 1});
sqs.rebase({its, 1});
qnc.rebase({its, 1});
qni.rebase({its, 1});
qnwfa.rebase({its, 1});
qnifa.rebase({its, 1});
qnbca.rebase({its, 1});
qozone.rebase({its, 1});
p.rebase({its, 1});
exner.rebase({its, 1});
rho.rebase({its, 1});
tt.rebase({its, 1});
xland.rebase({its});
ts.rebase({its});
qsfc.rebase({its});
ps.rebase({its});
ust.rebase({its});
ch.rebase({its});
hfx.rebase({its});
qfx.rebase({its});
rmol.rebase({its});
wspd.rebase({its});
uoce.rebase({its});
voce.rebase({its});
qke.rebase({its, 1});
qke_adv.rebase({its, 1});
tsq.rebase({its, 1});
qsq.rebase({its, 1});
cov.rebase({its, 1});
rublten.rebase({its, 1});
rvblten.rebase({its, 1});
rthblten.rebase({its, 1});
rqvblten.rebase({its, 1});
rqcblten.rebase({its, 1});
rqiblten.rebase({its, 1});
rqsblten.rebase({its, 1});
rqncblten.rebase({its, 1});
rqniblten.rebase({its, 1});
rqnwfablten.rebase({its, 1});
rqnifablten.rebase({its, 1});
rqnbcablten.rebase({its, 1});
rqozblten.rebase({its, 1});
exch_h.rebase({its, 1});
exch_m.rebase({its, 1});
pblh.rebase({its});
kpbl.rebase({its});
el_pbl.rebase({its, 1});
dqke.rebase({its, 1});
qwt.rebase({its, 1});
qshear.rebase({its, 1});
qbuoy.rebase({its, 1});
qdiss.rebase({its, 1});
sh.rebase({its, 1});
sm.rebase({its, 1});
qc_bl.rebase({its, 1});
qi_bl.rebase({its, 1});
cldfra_bl.rebase({its, 1});
edmf_a.rebase({its, 1});
edmf_w.rebase({its, 1});
edmf_qt.rebase({its, 1});
edmf_thl.rebase({its, 1});
edmf_ent.rebase({its, 1});
edmf_qc.rebase({its, 1});
sub_thl.rebase({its, 1});
sub_sqv.rebase({its, 1});
det_thl.rebase({its, 1});
det_sqv.rebase({its, 1});
edmf_a_dd.rebase({its, 1});
edmf_w_dd.rebase({its, 1});
edmf_qt_dd.rebase({its, 1});
edmf_thl_dd.rebase({its, 1});
edmf_ent_dd.rebase({its, 1});
edmf_qc_dd.rebase({its, 1});
maxwidth.rebase({1});
maxmf.rebase({1});
ztop_plume.rebase({1});
ktop_plume.rebase({its});
pattern_spp_pbl.rebase({its, 1});
rthraten.rebase({its, 1});
bool initialize_qke;
int i;
int k;
float qc_bl2;
float qi_bl2;
float cpm;
float exnerg;
float flq;
float flqc;
float flqv;
float flt;
float fltv;
float phh;
float pmz;
float psig_bl;
float psig_shcu;
float sqcg;
float phi_m;
float th_sfc;
float zet;
float ts_decay;
MynnArray<float, 1> cldfra_bl1_old = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qc_bl1_old = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qi_bl1_old = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qi1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qs1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> det_sqc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> det_u = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> det_v = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sub_u = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sub_v = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdc = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdk = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdq = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> pdt = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sgm = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqw = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> thetav = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> thl = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> vq = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> vt = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> kzero = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dfh = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dfm = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dfq = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qcd = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> tcd = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> diss_heat = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rstoch_col = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> zw = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_aw1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awthl1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqt1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqv1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqc1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awu1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awv1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqke1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnc1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqni1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnwfa1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnifa1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> s_awqnbca1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_aw1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awthl1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqt1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqv1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqc1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awu1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awv1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
MynnArray<float, 1> sd_awqke1 = mynn_array<float, 1>(arena, {kts}, {((kte) + (1))});
bool cloudflg;
int kk;
int kminrad;
constexpr float pfac = 2.0f;
constexpr float zfmin = 0.01f;
constexpr float phifac = 8.0f;
float maxkhtopdown;
float bfxpbl;
float dthvx;
float tmp1;
float temps;
float templ;
float zl1;
float wstar3_2;
float ent_eff;
float radsum;
float radflux;
float we;
float rcldb;
float rvls;
float minrad;
float zminrad;
MynnArray<float, 1> khtopdown = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> zfac = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> wscalek2 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> zfacent = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> tkeprodtd = mynn_array<float, 1>(arena, {kts}, {kte});
float dx1;
float xland1;
float ps1;
float ts1;
float qsfc1;
float ust1;
float ch1;
float hfx1;
float qfx1;
float rmol1;
float wspd1;
float uoce1;
float voce1;
float znt1;
MynnArray<float, 1> dz1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> u1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> v1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> th1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> tk1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> p1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> ex1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rho1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qni1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnifa1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnwfa1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qnbca1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qozone1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> rthraten1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqi1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sqs1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> w1 = mynn_array<float, 1>(arena, {kts}, {kme});
int kpbl1;
int ktop_plume1;
float pblh1;
MynnArray<float, 1> cldfra_bl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qc_bl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qi_bl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> el_pbl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qke1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qke_adv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> cov1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qsq1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> tsq1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sh1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sm1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> du1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dth1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqi1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqs1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqnc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqni1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqnifa1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqnwfa1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqnbca1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqozone1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_a1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_w1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qt1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_thl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_ent1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qc1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sub_thl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> sub_sqv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> det_thl1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> det_sqv1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_a_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_w_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qt_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_thl_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_ent_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> edmf_qc_dd1 = mynn_array<float, 1>(arena, {kts}, {kte});
float maxwidth1;
float maxmf1;
float ztop_plume1;
MynnArray<float, 1> exch_h1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> exch_m1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> dqke1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qwt1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qshear1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qbuoy1 = mynn_array<float, 1>(arena, {kts}, {kte});
MynnArray<float, 1> qdiss1 = mynn_array<float, 1>(arena, {kts}, {kte});
int nsub;
float delt2;
constexpr bool mix_chem = false;
constexpr int nchem = 1;
constexpr int kdvel = 1;
constexpr int ndvel = 1;
constexpr bool rrfs_sd = false;
constexpr bool smoke_dbg = false;
constexpr bool enh_mix = false;
int ic;
float emis_ant_no1;
float frp1;
MynnArray<float, 1> vd1 = mynn_array<float, 1>(arena, {1}, {ndvel});
MynnArray<float, 2> chem1 = mynn_array<float, 2>(arena, {kts, 1}, {kte, nchem});
MynnArray<float, 2> s_awchem1 = mynn_array<float, 2>(arena, {kts, 1}, {((kte) + (1)), nchem});
// bl_mynn.F90:434
errmsg = " ";
// bl_mynn.F90:435
errflg = 0;
// bl_mynn.F90:437
for (i=(active_column >= 0 ? active_column : its); (1) > 0 ? i <= ((active_column >= 0 ? active_column : ite)) : i >= ((active_column >= 0 ? active_column : ite)); i += (1)) {
// bl_mynn.F90:439
if ((dqke.present())) {
// bl_mynn.F90:440
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:441
dqke(i, k) = qke(i, k);
// bl_mynn.F90:442
}
// bl_mynn.F90:443
}
// bl_mynn.F90:446
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:447
rublten(i, k) = 0.f;
// bl_mynn.F90:448
rvblten(i, k) = 0.f;
// bl_mynn.F90:449
rthblten(i, k) = 0.f;
// bl_mynn.F90:450
rqvblten(i, k) = 0.f;
// bl_mynn.F90:451
rqcblten(i, k) = 0.f;
// bl_mynn.F90:452
rqiblten(i, k) = 0.f;
// bl_mynn.F90:453
rqsblten(i, k) = 0.f;
// bl_mynn.F90:454
rqncblten(i, k) = 0.f;
// bl_mynn.F90:455
rqniblten(i, k) = 0.f;
// bl_mynn.F90:456
rqnifablten(i, k) = 0.f;
// bl_mynn.F90:457
rqnwfablten(i, k) = 0.f;
// bl_mynn.F90:458
rqnbcablten(i, k) = 0.f;
// bl_mynn.F90:459
rqozblten(i, k) = 0.f;
// bl_mynn.F90:460
}
// bl_mynn.F90:463
ktop_plume(i) = 0;
// bl_mynn.F90:464
maxwidth(i) = 0.f;
// bl_mynn.F90:465
maxmf(i) = 0.f;
// bl_mynn.F90:466
ztop_plume(i) = 0.f;
// bl_mynn.F90:469
dx1 = dx(i);
// bl_mynn.F90:470
xland1 = xland(i);
// bl_mynn.F90:471
ps1 = ps(i);
// bl_mynn.F90:472
ts1 = ts(i);
// bl_mynn.F90:473
qsfc1 = qsfc(i);
// bl_mynn.F90:474
ust1 = ust(i);
// bl_mynn.F90:475
ch1 = ch(i);
// bl_mynn.F90:476
hfx1 = hfx(i);
// bl_mynn.F90:477
qfx1 = qfx(i);
// bl_mynn.F90:478
rmol1 = rmol(i);
// bl_mynn.F90:479
wspd1 = wspd(i);
// bl_mynn.F90:480
uoce1 = uoce(i);
// bl_mynn.F90:481
voce1 = voce(i);
// bl_mynn.F90:482
znt1 = znt(i);
// bl_mynn.F90:484
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:485
dz1(k) = dz(i, k);
// bl_mynn.F90:486
u1(k) = u(i, k);
// bl_mynn.F90:487
v1(k) = v(i, k);
// bl_mynn.F90:488
w1(k) = w(i, k);
// bl_mynn.F90:489
th1(k) = th(i, k);
// bl_mynn.F90:490
tk1(k) = tt(i, k);
// bl_mynn.F90:491
p1(k) = p(i, k);
// bl_mynn.F90:492
ex1(k) = exner(i, k);
// bl_mynn.F90:493
rho1(k) = rho(i, k);
// bl_mynn.F90:494
sh1(k) = sh(i, k);
// bl_mynn.F90:495
sm1(k) = sm(i, k);
// bl_mynn.F90:496
rthraten1(k) = rthraten(i, k);
// bl_mynn.F90:497
sqv1(k) = sqv(i, k);
// bl_mynn.F90:498
sqc1(k) = sqc(i, k);
// bl_mynn.F90:499
sqi1(k) = sqi(i, k);
// bl_mynn.F90:500
sqs1(k) = sqs(i, k);
// bl_mynn.F90:501
qnc1(k) = qnc(i, k);
// bl_mynn.F90:502
qni1(k) = qni(i, k);
// bl_mynn.F90:503
qnifa1(k) = qnifa(i, k);
// bl_mynn.F90:504
qnwfa1(k) = qnwfa(i, k);
// bl_mynn.F90:505
qnbca1(k) = qnbca(i, k);
// bl_mynn.F90:506
qozone1(k) = qozone(i, k);
// bl_mynn.F90:507
kzero(k) = 0.f;
// bl_mynn.F90:508
}
// bl_mynn.F90:509
for (k=kte; (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:510
w1(k) = w(i, k);
// bl_mynn.F90:511
}
// bl_mynn.F90:514
if (((spp_pbl) == (1))) {
// bl_mynn.F90:515
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:516
rstoch_col(k) = pattern_spp_pbl(i, k);
// bl_mynn.F90:517
}
// bl_mynn.F90:518
} else {
// bl_mynn.F90:519
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:520
rstoch_col(k) = 0.f;
// bl_mynn.F90:521
}
// bl_mynn.F90:522
}
// bl_mynn.F90:526
kpbl1 = kpbl(i);
// bl_mynn.F90:527
pblh1 = pblh(i);
// bl_mynn.F90:529
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:530
cldfra_bl1(k) = cldfra_bl(i, k);
// bl_mynn.F90:531
qc_bl1(k) = qc_bl(i, k);
// bl_mynn.F90:532
qi_bl1(k) = qi_bl(i, k);
// bl_mynn.F90:533
}
// bl_mynn.F90:535
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:536
el_pbl1(k) = el_pbl(i, k);
// bl_mynn.F90:537
qke1(k) = qke(i, k);
// bl_mynn.F90:538
qke_adv1(k) = qke_adv(i, k);
// bl_mynn.F90:539
cov1(k) = cov(i, k);
// bl_mynn.F90:540
qsq1(k) = qsq(i, k);
// bl_mynn.F90:541
tsq1(k) = tsq(i, k);
// bl_mynn.F90:542
sh1(k) = sh(i, k);
// bl_mynn.F90:543
sm1(k) = sm(i, k);
// bl_mynn.F90:544
}
// bl_mynn.F90:547
ktop_plume1 = 0;
// bl_mynn.F90:548
maxwidth1 = 0.f;
// bl_mynn.F90:549
maxmf1 = 0.f;
// bl_mynn.F90:550
ztop_plume1 = 0.f;
// bl_mynn.F90:551
maxkhtopdown = 0.f;
// bl_mynn.F90:553
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:554
du1(k) = 0.f;
// bl_mynn.F90:555
dv1(k) = 0.f;
// bl_mynn.F90:556
dth1(k) = 0.f;
// bl_mynn.F90:557
dqv1(k) = 0.f;
// bl_mynn.F90:558
dqc1(k) = 0.f;
// bl_mynn.F90:559
dqi1(k) = 0.f;
// bl_mynn.F90:560
dqs1(k) = 0.f;
// bl_mynn.F90:561
dqnc1(k) = 0.f;
// bl_mynn.F90:562
dqni1(k) = 0.f;
// bl_mynn.F90:563
dqnifa1(k) = 0.f;
// bl_mynn.F90:564
dqnwfa1(k) = 0.f;
// bl_mynn.F90:565
dqnbca1(k) = 0.f;
// bl_mynn.F90:566
dqozone1(k) = 0.f;
// bl_mynn.F90:567
}
// bl_mynn.F90:568
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:569
edmf_a1(k) = 0.f;
// bl_mynn.F90:570
edmf_w1(k) = 0.f;
// bl_mynn.F90:571
edmf_qc1(k) = 0.f;
// bl_mynn.F90:572
edmf_ent1(k) = 0.f;
// bl_mynn.F90:573
edmf_qt1(k) = 0.f;
// bl_mynn.F90:574
edmf_thl1(k) = 0.f;
// bl_mynn.F90:575
sub_thl1(k) = 0.f;
// bl_mynn.F90:576
sub_sqv1(k) = 0.f;
// bl_mynn.F90:577
det_thl1(k) = 0.f;
// bl_mynn.F90:578
det_sqv1(k) = 0.f;
// bl_mynn.F90:580
edmf_a_dd1(k) = 0.f;
// bl_mynn.F90:581
edmf_w_dd1(k) = 0.f;
// bl_mynn.F90:582
edmf_qc_dd1(k) = 0.f;
// bl_mynn.F90:583
edmf_ent_dd1(k) = 0.f;
// bl_mynn.F90:584
edmf_qt_dd1(k) = 0.f;
// bl_mynn.F90:585
edmf_thl_dd1(k) = 0.f;
// bl_mynn.F90:586
}
// bl_mynn.F90:587
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:588
dqke1(k) = 0.f;
// bl_mynn.F90:589
qwt1(k) = 0.f;
// bl_mynn.F90:590
qshear1(k) = 0.f;
// bl_mynn.F90:591
qbuoy1(k) = 0.f;
// bl_mynn.F90:592
qdiss1(k) = 0.f;
// bl_mynn.F90:593
exch_h1(k) = 0.f;
// bl_mynn.F90:594
exch_m1(k) = 0.f;
// bl_mynn.F90:595
}
// bl_mynn.F90:596
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:597
sub_u(k) = 0.f;
// bl_mynn.F90:598
sub_v(k) = 0.f;
// bl_mynn.F90:599
det_sqc(k) = 0.f;
// bl_mynn.F90:600
det_u(k) = 0.f;
// bl_mynn.F90:601
det_v(k) = 0.f;
// bl_mynn.F90:602
}
// bl_mynn.F90:603
for (k=kts; (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:604
s_aw1(k) = 0.f;
// bl_mynn.F90:605
s_awthl1(k) = 0.f;
// bl_mynn.F90:606
s_awqt1(k) = 0.f;
// bl_mynn.F90:607
s_awqv1(k) = 0.f;
// bl_mynn.F90:608
s_awqc1(k) = 0.f;
// bl_mynn.F90:609
s_awu1(k) = 0.f;
// bl_mynn.F90:610
s_awv1(k) = 0.f;
// bl_mynn.F90:611
s_awqke1(k) = 0.f;
// bl_mynn.F90:612
s_awqnc1(k) = 0.f;
// bl_mynn.F90:613
s_awqni1(k) = 0.f;
// bl_mynn.F90:614
s_awqnwfa1(k) = 0.f;
// bl_mynn.F90:615
s_awqnifa1(k) = 0.f;
// bl_mynn.F90:616
s_awqnbca1(k) = 0.f;
// bl_mynn.F90:617
}
// bl_mynn.F90:618
for (k=kts; (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:619
sd_aw1(k) = 0.f;
// bl_mynn.F90:620
sd_awthl1(k) = 0.f;
// bl_mynn.F90:621
sd_awqt1(k) = 0.f;
// bl_mynn.F90:622
sd_awqv1(k) = 0.f;
// bl_mynn.F90:623
sd_awqc1(k) = 0.f;
// bl_mynn.F90:624
sd_awu1(k) = 0.f;
// bl_mynn.F90:625
sd_awv1(k) = 0.f;
// bl_mynn.F90:626
sd_awqke1(k) = 0.f;
// bl_mynn.F90:627
}
// bl_mynn.F90:628
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:629
cldfra_bl1_old(k) = 0.f;
// bl_mynn.F90:630
qc_bl1_old(k) = 0.f;
// bl_mynn.F90:631
qi_bl1_old(k) = 0.f;
// bl_mynn.F90:632
}
// bl_mynn.F90:634
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:635
qv1(k) = mpas_div(sqv1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:636
qc1(k) = mpas_div(sqc1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:637
qi1(k) = mpas_div(sqi1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:638
qs1(k) = mpas_div(sqs1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:639
}
// bl_mynn.F90:641
k = kts;
// bl_mynn.F90:642
zw(k) = 0.f;
// bl_mynn.F90:643
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:644
zw(k) = mpas_add(zw(((k) - (1))), dz1(((k) - (1))));
// bl_mynn.F90:645
}
// bl_mynn.F90:664
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn.F90:665
vd1(ic) = 0.f;
// bl_mynn.F90:666
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:667
chem1(k, ic) = 0.f;
// bl_mynn.F90:668
}
// bl_mynn.F90:669
}
// bl_mynn.F90:670
emis_ant_no1 = 0.f;
// bl_mynn.F90:671
frp1 = 0.f;
// bl_mynn.F90:673
for (ic=1; (1) > 0 ? ic <= (nchem) : ic >= (nchem); ic += (1)) {
// bl_mynn.F90:674
for (k=kts; (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:675
s_awchem1(k, ic) = 0.f;
// bl_mynn.F90:676
}
// bl_mynn.F90:677
}
// bl_mynn.F90:681
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:683
sqw(k) = mpas_add(mpas_add(sqv1(k), sqc1(k)), sqi1(k));
// bl_mynn.F90:684
thl(k) = mpas_sub(mpas_sub(th1(k), mpas_mul(mpas_div(xlvcp, ex1(k)), sqc1(k))), mpas_mul(mpas_div(xlscp, ex1(k)), sqi1(k)));
// bl_mynn.F90:685
thetav(k) = mpas_mul(th1(k), mpas_add(1.f, mpas_mul(0.608f, sqv1(k))));
// bl_mynn.F90:691
}
// bl_mynn.F90:696
if (((((initflag) > (0))) && ((!(restart))))) {
// bl_mynn.F90:699
if (((restart) || (cycling))) {
// bl_mynn.F90:700
if (((qke1(kts)) < (0.0002f))) {
// bl_mynn.F90:701
initialize_qke = true;
// bl_mynn.F90:702
} else {
// bl_mynn.F90:703
initialize_qke = false;
// bl_mynn.F90:704
}
// bl_mynn.F90:705
} else {
// bl_mynn.F90:706
initialize_qke = true;
// bl_mynn.F90:707
}
// bl_mynn.F90:709
if ((((!(restart))) || ((!(cycling))))) {
// bl_mynn.F90:710
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:711
sh1(k) = 0.f;
// bl_mynn.F90:712
sm1(k) = 0.f;
// bl_mynn.F90:713
el_pbl1(k) = 0.f;
// bl_mynn.F90:714
tsq1(k) = 0.f;
// bl_mynn.F90:715
qsq1(k) = 0.f;
// bl_mynn.F90:716
cov1(k) = 0.f;
// bl_mynn.F90:717
cldfra_bl1(k) = 0.f;
// bl_mynn.F90:718
qc_bl1(k) = 0.f;
// bl_mynn.F90:719
qi_bl1(k) = 0.f;
// bl_mynn.F90:720
qke1(k) = 0.f;
// bl_mynn.F90:721
}
// bl_mynn.F90:722
}
// bl_mynn.F90:723
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:724
cldfra_bl1_old(k) = 0.f;
// bl_mynn.F90:725
qc_bl1_old(k) = 0.f;
// bl_mynn.F90:726
qi_bl1_old(k) = 0.f;
// bl_mynn.F90:727
}
// bl_mynn.F90:729
if (initialize_qke) {
// bl_mynn.F90:730
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:731
qke1(k) = mpas_mul(mpas_mul(5.f, ust1), mynn_max(((float)(mpas_div(mpas_sub(mpas_mul(ust1, 700.f), zw(k)), mpas_mul(mynn_max(((float)(ust1)), ((float)(0.01f))), 700.f)))), ((float)(0.01f))));
// bl_mynn.F90:732
}
// bl_mynn.F90:733
}
// bl_mynn.F90:736
f_get_pblh(kts, kte, pblh1, thetav, qke1, zw, dz1, xland1, kpbl1);
// bl_mynn.F90:739
if (bl_mynn_scaleaware) {
// bl_mynn.F90:740
f_scale_aware(dx1, pblh1, psig_bl, psig_shcu);
// bl_mynn.F90:741
} else {
// bl_mynn.F90:742
psig_bl = 1.f;
// bl_mynn.F90:743
psig_shcu = 1.f;
// bl_mynn.F90:744
}
// bl_mynn.F90:747
f_mym_initialize(kts, kte, xland1, dz1, dx1, zw, u1, v1, thl, sqv1, pblh1, th1, thetav, sh1, sm1, ust1, rmol1, el_pbl1, qke1, tsq1, qsq1, cov1, psig_bl, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, initialize_qke, spp_pbl, rstoch_col);
// bl_mynn.F90:760
}
// bl_mynn.F90:766
if (bl_mynn_tkeadvect) {
// bl_mynn.F90:767
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:768
qke1(k) = qke_adv1(k);
// bl_mynn.F90:769
}
// bl_mynn.F90:770
}
// bl_mynn.F90:772
if (bl_mynn_tkebudget) {
// bl_mynn.F90:773
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:774
dqke1(k) = qke1(k);
// bl_mynn.F90:775
}
// bl_mynn.F90:776
}
// bl_mynn.F90:777
if (((icloud_bl) > (0))) {
// bl_mynn.F90:778
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:779
cldfra_bl1_old(k) = cldfra_bl1(k);
// bl_mynn.F90:780
qc_bl1_old(k) = qc_bl1(k);
// bl_mynn.F90:781
qi_bl1_old(k) = qi_bl1(k);
// bl_mynn.F90:782
}
// bl_mynn.F90:783
}
// bl_mynn.F90:786
f_get_pblh(kts, kte, pblh1, thetav, qke1, zw, dz1, xland1, kpbl1);
// bl_mynn.F90:789
if (bl_mynn_scaleaware) {
// bl_mynn.F90:790
f_scale_aware(dx1, pblh1, psig_bl, psig_shcu);
// bl_mynn.F90:791
} else {
// bl_mynn.F90:792
psig_bl = 1.f;
// bl_mynn.F90:793
psig_shcu = 1.f;
// bl_mynn.F90:794
}
// bl_mynn.F90:796
sqcg = 0.0f;
// bl_mynn.F90:797
cpm = mpas_mul(cp, mpas_add(1.f, mpas_mul(0.84f, qv1(kts))));
// bl_mynn.F90:798
exnerg = mynn_pow(((float)(mpas_div(ps1, p1000mb))), ((float)(rcp)));
// bl_mynn.F90:807
flqv = mpas_div(qfx1, rho1(kts));
// bl_mynn.F90:808
flqc = 0.0f;
// bl_mynn.F90:809
th_sfc = mpas_div(ts1, ex1(kts));
// bl_mynn.F90:812
flq = mpas_add(flqv, flqc);
// bl_mynn.F90:813
flt = mpas_sub(mpas_div(hfx1, mpas_mul(rho1(kts), cpm)), mpas_div(mpas_mul(xlvcp, flqc), ex1(kts)));
// bl_mynn.F90:814
fltv = mpas_add(flt, mpas_mul(mpas_mul(flqv, p608), th_sfc));
// bl_mynn.F90:817
rmol1 = mpas_div(mpas_mul(mpas_mul(mynn_neg(karman), gtr), fltv), mynn_max(((float)(mynn_ipow<3>(ust1))), ((float)(1.0e-6f))));
// bl_mynn.F90:818
zet = mpas_mul(mpas_mul(0.5f, dz1(kts)), rmol1);
// bl_mynn.F90:819
zet = mynn_max(((float)(zet)), ((float)(mynn_neg(20.f))));
// bl_mynn.F90:820
zet = mynn_min(((float)(zet)), ((float)(20.f)));
// bl_mynn.F90:823
if (((bl_mynn_stfunc) == (0))) {
// bl_mynn.F90:825
if (((zet) >= (0.0f))) {
// bl_mynn.F90:826
pmz = mpas_add(1.0f, mpas_mul(mpas_sub(cphm_st, 1.0f), zet));
// bl_mynn.F90:827
phh = mpas_add(1.0f, mpas_mul(cphh_st, zet));
// bl_mynn.F90:828
} else {
// bl_mynn.F90:829
pmz = mpas_sub(mpas_div(1.0f, mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(cphm_unst, zet)))), ((float)(0.25f)))), zet);
// bl_mynn.F90:830
phh = mpas_div(1.0f, mynn_sqrt(mpas_sub(1.0f, mpas_mul(cphh_unst, zet))));
// bl_mynn.F90:831
}
// bl_mynn.F90:832
phi_m = mpas_add(pmz, zet);
// bl_mynn.F90:833
} else {
// bl_mynn.F90:835
phi_m = f_phim(zet);
// bl_mynn.F90:836
pmz = mpas_sub(phi_m, zet);
// bl_mynn.F90:837
phh = f_phih(zet);
// bl_mynn.F90:838
}
// bl_mynn.F90:843
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:844
vt(k) = 0.f;
// bl_mynn.F90:845
vq(k) = 0.f;
// bl_mynn.F90:846
sgm(k) = 0.f;
// bl_mynn.F90:847
}
// bl_mynn.F90:849
f_mym_condensation(kts, kte, dx1, dz1, zw, xland1, thl, sqw, sqv1, sqc1, sqi1, sqs1, p1, ex1, tsq1, qsq1, cov1, sh1, el_pbl1, bl_mynn_cloudpdf, qc_bl1, qi_bl1, cldfra_bl1, pblh1, hfx1, vt, vq, th1, sgm, rmol1, spp_pbl, rstoch_col);
// bl_mynn.F90:862
if (bl_mynn_topdown) {
// bl_mynn.F90:863
f_topdown_cloudrad(kts, kte, dz1, zw, fltv, xland1, kpbl1, pblh1, sqc1, sqi1, sqw, thl, th1, ex1, p1, rho1, thetav, cldfra_bl1, rthraten1, maxkhtopdown, khtopdown, tkeprodtd);
// bl_mynn.F90:868
} else {
// bl_mynn.F90:869
maxkhtopdown = 0.f;
// bl_mynn.F90:870
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:871
khtopdown(k) = 0.f;
// bl_mynn.F90:872
tkeprodtd(k) = 0.f;
// bl_mynn.F90:873
}
// bl_mynn.F90:874
}
// bl_mynn.F90:878
if (bl_mynn_edmf) {
// bl_mynn.F90:879
f_dmp_mf(i, kts, kte, delt, zw, dz1, p1, rho1, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, u1, v1, w1, th1, thl, thetav, tk1, sqw, sqv1, sqc1, qke1, qnc1, qni1, qnwfa1, qnifa1, qnbca1, ex1, vt, vq, sgm, ust1, flt, fltv, flq, flqv, pblh1, kpbl1, dx1, xland1, th_sfc, edmf_a1, edmf_w1, edmf_qt1, edmf_thl1, edmf_ent1, edmf_qc1, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqke1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sub_thl1, sub_sqv1, sub_u, sub_v, det_thl1, det_sqv1, det_sqc, det_u, det_v, nchem, chem1, s_awchem1, mix_chem, qc_bl1, cldfra_bl1, qc_bl1_old, cldfra_bl1_old, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa, flag_qnbca, psig_shcu, maxwidth1, ktop_plume1, maxmf1, ztop_plume1, spp_pbl, rstoch_col);
// bl_mynn.F90:919
if (bl_mynn_edmf_dd) {
// bl_mynn.F90:920
f_ddmf_jpl(kts, kte, delt, zw, dz1, p1, u1, v1, th1, thl, thetav, tk1, sqw, sqv1, sqc1, rho1, ex1, ust1, flt, flq, pblh1, kpbl1, edmf_a_dd1, edmf_w_dd1, edmf_qt_dd1, edmf_thl_dd1, edmf_ent_dd1, edmf_qc_dd1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awu1, sd_awv1, sd_awqke1, qc_bl1, cldfra_bl1, rthraten.sequence<1>({kts}, {kte}));
// bl_mynn.F90:933
}
// bl_mynn.F90:934
}
// bl_mynn.F90:939
delt2 = delt;
// bl_mynn.F90:941
f_mym_turbulence(kts, kte, xland1, bl_mynn_closure, dz1, dx1, zw, u1, v1, thl, thetav, sqc1, sqw, qke1, tsq1, qsq1, cov1, vt, vq, rmol1, flt, fltv, flq, pblh1, th1, sh1, sm1, el_pbl1, dfm, dfh, dfq, tcd, qcd, pdk, pdt, pdq, pdc, qwt1, qshear1, qbuoy1, qdiss1, bl_mynn_tkebudget, psig_bl, psig_shcu, cldfra_bl1, bl_mynn_mixlength, edmf_w1, edmf_a1, tkeprodtd, spp_pbl, rstoch_col);
// bl_mynn.F90:963
f_mym_predict(kts, kte, bl_mynn_closure, delt2, dz1, ust1, flt, flq, pmz, phh, el_pbl1, dfq, rho1, pdk, pdt, pdq, pdc, qke1, tsq1, qsq1, cov1, s_aw1, s_awqke1, bl_mynn_edmf_tke, qwt1, qdiss1, bl_mynn_tkebudget);
// bl_mynn.F90:972
if (bl_mynn_dheatopt) {
// bl_mynn.F90:973
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn.F90:975
diss_heat(k) = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_div(mpas_mul(1.0f, mynn_pow(((float)(qke1(k))), ((float)(1.5f)))), mpas_mul(b1, mynn_max(((float)(mpas_mul(0.5f, mpas_add(el_pbl1(k), el_pbl1(((k) + (1))))))), ((float)(1.f))))), cp))), ((float)(0.0f))))), ((float)(0.002f)));
// bl_mynn.F90:978
diss_heat(k) = mpas_mul(diss_heat(k), mynn_exp(mpas_div(mynn_neg(10000.f), mynn_max(((float)(p1(k))), ((float)(1.f))))));
// bl_mynn.F90:979
}
// bl_mynn.F90:980
diss_heat(kte) = 0.f;
// bl_mynn.F90:981
} else {
// bl_mynn.F90:982
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:983
diss_heat(k) = 0.f;
// bl_mynn.F90:984
}
// bl_mynn.F90:985
}
// bl_mynn.F90:989
f_mynn_tendencies(kts, kte, delt, dz1, rho1, u1, v1, th1, tk1, qv1, qc1, qi1, kzero, qnc1, qni1, ps1, p1, ex1, thl, sqv1, sqc1, sqi1, kzero, sqw, qnwfa1, qnifa1, qnbca1, qozone1, ust1, flt, flq, flqv, flqc, wspd1, uoce1, voce1, tsq1, qsq1, cov1, tcd, qcd, dfm, dfh, dfq, du1, dv1, dth1, dqv1, dqc1, dqi1, dqs1, dqnc1, dqni1, dqnwfa1, dqnifa1, dqnbca1, dqozone1, diss_heat, s_aw1, s_awthl1, s_awqt1, s_awqv1, s_awqc1, s_awu1, s_awv1, s_awqnc1, s_awqni1, s_awqnwfa1, s_awqnifa1, s_awqnbca1, sd_aw1, sd_awthl1, sd_awqt1, sd_awqv1, sd_awqc1, sd_awu1, sd_awv1, sub_thl1, sub_sqv1, sub_u, sub_v, det_thl1, det_sqv1, det_sqc, det_u, det_v, flag_qc, flag_qi, flag_qnc, flag_qni, flag_qs, flag_qnwfa, flag_qnifa, flag_qnbca, flag_qoz, cldfra_bl1, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_edmf_mom, bl_mynn_mixscalars);
// bl_mynn.F90:1031
if (mix_chem) {
// bl_mynn.F90:1032
if (rrfs_sd) {
// bl_mynn.F90:1033
f_mynn_mix_chem(kts, kte, delt, dz1, pblh1, nchem, kdvel, ndvel, chem1, vd1, rho1, flt, tcd, qcd, dfh, s_aw1, s_awchem1, emis_ant_no1, frp1, rrfs_sd, enh_mix, smoke_dbg);
// bl_mynn.F90:1044
} else {
// bl_mynn.F90:1045
f_mynn_mix_chem(kts, kte, delt, dz1, pblh1, nchem, kdvel, ndvel, chem1, vd1, rho1, flt, tcd, qcd, dfh, s_aw1, s_awchem1, zero, zero, rrfs_sd, enh_mix, smoke_dbg);
// bl_mynn.F90:1056
}
// bl_mynn.F90:1057
}
// bl_mynn.F90:1069
f_retrieve_exchange_coeffs(kts, kte, dfm, dfh, dz1, exch_m1, exch_h1);
// bl_mynn.F90:1076
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1077
rublten(i, k) = du1(k);
// bl_mynn.F90:1078
rvblten(i, k) = dv1(k);
// bl_mynn.F90:1079
rthblten(i, k) = dth1(k);
// bl_mynn.F90:1080
rqvblten(i, k) = dqv1(k);
// bl_mynn.F90:1081
}
// bl_mynn.F90:1082
if (((bl_mynn_cloudmix) && (flag_qc))) {
// bl_mynn.F90:1083
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1084
rqcblten(i, k) = dqc1(k);
// bl_mynn.F90:1085
}
// bl_mynn.F90:1086
}
// bl_mynn.F90:1087
if (((bl_mynn_cloudmix) && (flag_qi))) {
// bl_mynn.F90:1088
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1089
rqiblten(i, k) = dqi1(k);
// bl_mynn.F90:1090
}
// bl_mynn.F90:1091
}
// bl_mynn.F90:1092
if (((bl_mynn_cloudmix) && (flag_qs))) {
// bl_mynn.F90:1093
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1094
rqsblten(i, k) = dqs1(k);
// bl_mynn.F90:1095
}
// bl_mynn.F90:1096
}
// bl_mynn.F90:1097
if (((((bl_mynn_cloudmix) && (bl_mynn_mixscalars))) && (flag_qnc))) {
// bl_mynn.F90:1098
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1099
rqncblten(i, k) = dqnc1(k);
// bl_mynn.F90:1100
}
// bl_mynn.F90:1101
}
// bl_mynn.F90:1102
if (((((bl_mynn_cloudmix) && (bl_mynn_mixscalars))) && (flag_qni))) {
// bl_mynn.F90:1103
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1104
rqniblten(i, k) = dqni1(k);
// bl_mynn.F90:1105
}
// bl_mynn.F90:1106
}
// bl_mynn.F90:1107
if (((((bl_mynn_cloudmix) && (bl_mynn_mixscalars))) && (flag_qnifa))) {
// bl_mynn.F90:1108
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1109
rqnifablten(i, k) = dqnifa1(k);
// bl_mynn.F90:1110
}
// bl_mynn.F90:1111
}
// bl_mynn.F90:1112
if (((((bl_mynn_cloudmix) && (bl_mynn_mixscalars))) && (flag_qnwfa))) {
// bl_mynn.F90:1113
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1114
rqnwfablten(i, k) = dqnwfa1(k);
// bl_mynn.F90:1115
}
// bl_mynn.F90:1116
}
// bl_mynn.F90:1117
if (((((bl_mynn_cloudmix) && (bl_mynn_mixscalars))) && (flag_qnbca))) {
// bl_mynn.F90:1118
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1119
rqnbcablten(i, k) = dqnbca1(k);
// bl_mynn.F90:1120
}
// bl_mynn.F90:1121
}
// bl_mynn.F90:1122
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1123
rqozblten(i, k) = 0.f;
// bl_mynn.F90:1124
}
// bl_mynn.F90:1127
kpbl(i) = kpbl1;
// bl_mynn.F90:1128
ktop_plume(i) = ktop_plume1;
// bl_mynn.F90:1130
pblh(i) = pblh1;
// bl_mynn.F90:1132
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1133
cldfra_bl(i, k) = cldfra_bl1(k);
// bl_mynn.F90:1134
qc_bl(i, k) = qc_bl1(k);
// bl_mynn.F90:1135
qi_bl(i, k) = qi_bl1(k);
// bl_mynn.F90:1136
}
// bl_mynn.F90:1138
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1139
el_pbl(i, k) = el_pbl1(k);
// bl_mynn.F90:1140
qke(i, k) = qke1(k);
// bl_mynn.F90:1141
qke_adv(i, k) = qke_adv1(k);
// bl_mynn.F90:1142
cov(i, k) = cov1(k);
// bl_mynn.F90:1143
qsq(i, k) = qsq1(k);
// bl_mynn.F90:1144
tsq(i, k) = tsq1(k);
// bl_mynn.F90:1145
sh(i, k) = sh1(k);
// bl_mynn.F90:1146
sm(i, k) = sm1(k);
// bl_mynn.F90:1147
}
// bl_mynn.F90:1151
if ((((((((((qwt.present())) && ((qbuoy.present())))) && ((qshear.present())))) && ((qdiss.present())))) && ((dqke.present())))) {
// bl_mynn.F90:1153
if (bl_mynn_tkebudget) {
// bl_mynn.F90:1155
k = kts;
// bl_mynn.F90:1156
qshear1(k) = mpas_sub(mpas_mul(4.f, mpas_div(mpas_mul(mynn_ipow<3>(ust1), phi_m), mpas_mul(karman, dz1(k)))), qshear1(((k) + (1))));
// bl_mynn.F90:1157
qbuoy1(k) = mpas_sub(mpas_mul(4.f, mpas_div(mpas_mul(mynn_neg(mynn_ipow<3>(ust1)), zet), mpas_mul(karman, dz1(k)))), qbuoy1(((k) + (1))));
// bl_mynn.F90:1160
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn.F90:1161
qshear(i, k) = mpas_mul(0.5f, mpas_add(qshear1(k), qshear1(((k) + (1)))));
// bl_mynn.F90:1162
qbuoy(i, k) = mpas_mul(0.5f, mpas_add(qbuoy1(k), qbuoy1(((k) + (1)))));
// bl_mynn.F90:1163
qwt(i, k) = qwt1(k);
// bl_mynn.F90:1164
qdiss(i, k) = qdiss1(k);
// bl_mynn.F90:1165
dqke(i, k) = mpas_div(mpas_mul(mpas_sub(qke1(k), dqke(i, k)), 0.5f), delt);
// bl_mynn.F90:1166
}
// bl_mynn.F90:1168
k = kte;
// bl_mynn.F90:1169
qshear(i, k) = 0.f;
// bl_mynn.F90:1170
qbuoy(i, k) = 0.f;
// bl_mynn.F90:1171
qwt(i, k) = 0.f;
// bl_mynn.F90:1172
qdiss(i, k) = 0.f;
// bl_mynn.F90:1173
dqke(i, k) = 0.f;
// bl_mynn.F90:1174
} else {
// bl_mynn.F90:1175
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1176
qshear(i, k) = 0.f;
// bl_mynn.F90:1177
qbuoy(i, k) = 0.f;
// bl_mynn.F90:1178
qwt(i, k) = 0.f;
// bl_mynn.F90:1179
qdiss(i, k) = 0.f;
// bl_mynn.F90:1180
dqke(i, k) = 0.f;
// bl_mynn.F90:1181
}
// bl_mynn.F90:1182
}
// bl_mynn.F90:1183
}
// bl_mynn.F90:1187
if (((bl_mynn_edmf) && (bl_mynn_output))) {
// bl_mynn.F90:1188
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1189
edmf_a(i, k) = edmf_a1(k);
// bl_mynn.F90:1190
edmf_w(i, k) = edmf_w1(k);
// bl_mynn.F90:1191
edmf_qt(i, k) = edmf_qt1(k);
// bl_mynn.F90:1192
edmf_thl(i, k) = edmf_thl1(k);
// bl_mynn.F90:1193
edmf_ent(i, k) = edmf_ent1(k);
// bl_mynn.F90:1194
edmf_qc(i, k) = edmf_qc1(k);
// bl_mynn.F90:1195
sub_thl(i, k) = sub_thl1(k);
// bl_mynn.F90:1196
sub_sqv(i, k) = sub_sqv1(k);
// bl_mynn.F90:1197
det_thl(i, k) = det_thl1(k);
// bl_mynn.F90:1198
det_sqv(i, k) = det_sqv1(k);
// bl_mynn.F90:1199
}
// bl_mynn.F90:1200
} else {
// bl_mynn.F90:1201
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1202
edmf_a(i, k) = 0.f;
// bl_mynn.F90:1203
edmf_w(i, k) = 0.f;
// bl_mynn.F90:1204
edmf_qt(i, k) = 0.f;
// bl_mynn.F90:1205
edmf_thl(i, k) = 0.f;
// bl_mynn.F90:1206
edmf_ent(i, k) = 0.f;
// bl_mynn.F90:1207
edmf_qc(i, k) = 0.f;
// bl_mynn.F90:1208
sub_thl(i, k) = 0.f;
// bl_mynn.F90:1209
sub_sqv(i, k) = 0.f;
// bl_mynn.F90:1210
det_thl(i, k) = 0.f;
// bl_mynn.F90:1211
det_sqv(i, k) = 0.f;
// bl_mynn.F90:1212
}
// bl_mynn.F90:1213
}
// bl_mynn.F90:1214
if (((bl_mynn_edmf_dd) && (bl_mynn_output))) {
// bl_mynn.F90:1215
if ((((((((((((edmf_a_dd.present())) && ((edmf_w_dd.present())))) && ((edmf_qt_dd.present())))) && ((edmf_thl_dd.present())))) && ((edmf_ent_dd.present())))) && ((edmf_qc_dd.present())))) {
// bl_mynn.F90:1217
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1218
edmf_a_dd(i, k) = edmf_a_dd1(k);
// bl_mynn.F90:1219
edmf_w_dd(i, k) = edmf_w_dd1(k);
// bl_mynn.F90:1220
edmf_qt_dd(i, k) = edmf_qt_dd1(k);
// bl_mynn.F90:1221
edmf_thl_dd(i, k) = edmf_thl_dd1(k);
// bl_mynn.F90:1222
edmf_ent_dd(i, k) = edmf_ent_dd1(k);
// bl_mynn.F90:1223
edmf_qc_dd(i, k) = edmf_qc_dd1(k);
// bl_mynn.F90:1224
}
// bl_mynn.F90:1225
}
// bl_mynn.F90:1226
}
// bl_mynn.F90:1229
maxwidth(i) = maxwidth1;
// bl_mynn.F90:1230
maxmf(i) = maxmf1;
// bl_mynn.F90:1231
ztop_plume(i) = ztop_plume1;
// bl_mynn.F90:1233
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:1234
exch_h(i, k) = exch_h1(k);
// bl_mynn.F90:1235
exch_m(i, k) = exch_m1(k);
// bl_mynn.F90:1236
}
// bl_mynn.F90:1238
}
}
};
// MPAS physics_mmm/bl_mynn.F90:24-107 and 129-1242, one original horizontal column per CUDA thread.
extern "C" __global__ void mynn_mmm_run(const int batch_first, const int batch_count, const float* constants, unsigned char* workspace, const unsigned long long workspace_stride, int* error_flags, int initflag, int restart, int cycling, float delt, float* dz, float* dx, float* znt, float* u, float* v, float* w, float* th, float* sqv, float* sqc, float* sqi, float* sqs, float* qnc, float* qni, float* qnwfa, float* qnifa, float* qnbca, float* qozone, float* p, float* exner, float* rho, float* tt, float* xland, float* ts, float* qsfc, float* ps, float* ust, float* ch, float* hfx, float* qfx, float* rmol, float* wspd, float* uoce, float* voce, float* qke, float* qke_adv, float* tsq, float* qsq, float* cov, float* rublten, float* rvblten, float* rthblten, float* rqvblten, float* rqcblten, float* rqiblten, float* rqsblten, float* rqncblten, float* rqniblten, float* rqnwfablten, float* rqnifablten, float* rqnbcablten, float* rqozblten, float* exch_h, float* exch_m, float* pblh, int* kpbl, float* el_pbl, float* dqke, float* qwt, float* qshear, float* qbuoy, float* qdiss, float* sh, float* sm, float* qc_bl, float* qi_bl, float* cldfra_bl, int icloud_bl, int bl_mynn_tkeadvect, int bl_mynn_tkebudget, int bl_mynn_cloudpdf, int bl_mynn_mixlength, float bl_mynn_closure, int bl_mynn_stfunc, int bl_mynn_topdown, int bl_mynn_edmf, int bl_mynn_edmf_dd, int bl_mynn_edmf_mom, int bl_mynn_edmf_tke, int bl_mynn_mixscalars, int bl_mynn_output, int bl_mynn_cloudmix, int bl_mynn_mixqt, int bl_mynn_scaleaware, int bl_mynn_dheatopt, float* edmf_a, float* edmf_w, float* edmf_qt, float* edmf_thl, float* edmf_ent, float* edmf_qc, float* sub_thl, float* sub_sqv, float* det_thl, float* det_sqv, float* edmf_a_dd, float* edmf_w_dd, float* edmf_qt_dd, float* edmf_thl_dd, float* edmf_ent_dd, float* edmf_qc_dd, float* maxwidth, float* maxmf, float* ztop_plume, int* ktop_plume, int spp_pbl, float* pattern_spp_pbl, float* rthraten, int flag_qc, int flag_qi, int flag_qs, int flag_qnc, int flag_qni, int flag_qnwfa, int flag_qnifa, int flag_qnbca, int flag_qoz, int its, int ite, int kts, int kte, int kme) {
const int worker = blockIdx.x*blockDim.x+threadIdx.x;
for (int thread = worker; thread < batch_count; thread += blockDim.x*gridDim.x) {
MynnArena arena{workspace+workspace_stride*(worker & ~31),0,32,worker & 31};
MynnColumn column{arena};
column.active_column = its + batch_first + thread;
MynnText errmsg;
int errflg;
column.f_bl_mynn_init(constants[0], constants[1], constants[2], constants[3], constants[4], constants[5], constants[6], constants[7], constants[8], constants[9], constants[10], constants[11], constants[12], constants[13], constants[14], constants[15], constants[16], constants[17],errmsg,errflg);
MynnArray<float,2> a_dz; a_dz.data=dz;
a_dz.lo[0]=1; a_dz.hi[0]=ite-its+1; a_dz.stride[0]=1;
a_dz.lo[1]=1; a_dz.hi[1]=kte-kts+1; a_dz.stride[1]=ite-its+1;
MynnArray<float,1> a_dx; a_dx.data=dx;
a_dx.lo[0]=1; a_dx.hi[0]=ite-its+1; a_dx.stride[0]=1;
MynnArray<float,1> a_znt; a_znt.data=znt;
a_znt.lo[0]=1; a_znt.hi[0]=ite-its+1; a_znt.stride[0]=1;
MynnArray<float,2> a_u; a_u.data=u;
a_u.lo[0]=1; a_u.hi[0]=ite-its+1; a_u.stride[0]=1;
a_u.lo[1]=1; a_u.hi[1]=kte-kts+1; a_u.stride[1]=ite-its+1;
MynnArray<float,2> a_v; a_v.data=v;
a_v.lo[0]=1; a_v.hi[0]=ite-its+1; a_v.stride[0]=1;
a_v.lo[1]=1; a_v.hi[1]=kte-kts+1; a_v.stride[1]=ite-its+1;
MynnArray<float,2> a_w; a_w.data=w;
a_w.lo[0]=1; a_w.hi[0]=ite-its+1; a_w.stride[0]=1;
a_w.lo[1]=1; a_w.hi[1]=kme-kts+1; a_w.stride[1]=ite-its+1;
MynnArray<float,2> a_th; a_th.data=th;
a_th.lo[0]=1; a_th.hi[0]=ite-its+1; a_th.stride[0]=1;
a_th.lo[1]=1; a_th.hi[1]=kte-kts+1; a_th.stride[1]=ite-its+1;
MynnArray<float,2> a_sqv; a_sqv.data=sqv;
a_sqv.lo[0]=1; a_sqv.hi[0]=ite-its+1; a_sqv.stride[0]=1;
a_sqv.lo[1]=1; a_sqv.hi[1]=kte-kts+1; a_sqv.stride[1]=ite-its+1;
MynnArray<float,2> a_sqc; a_sqc.data=sqc;
a_sqc.lo[0]=1; a_sqc.hi[0]=ite-its+1; a_sqc.stride[0]=1;
a_sqc.lo[1]=1; a_sqc.hi[1]=kte-kts+1; a_sqc.stride[1]=ite-its+1;
MynnArray<float,2> a_sqi; a_sqi.data=sqi;
a_sqi.lo[0]=1; a_sqi.hi[0]=ite-its+1; a_sqi.stride[0]=1;
a_sqi.lo[1]=1; a_sqi.hi[1]=kte-kts+1; a_sqi.stride[1]=ite-its+1;
MynnArray<float,2> a_sqs; a_sqs.data=sqs;
a_sqs.lo[0]=1; a_sqs.hi[0]=ite-its+1; a_sqs.stride[0]=1;
a_sqs.lo[1]=1; a_sqs.hi[1]=kte-kts+1; a_sqs.stride[1]=ite-its+1;
MynnArray<float,2> a_qnc; a_qnc.data=qnc;
a_qnc.lo[0]=1; a_qnc.hi[0]=ite-its+1; a_qnc.stride[0]=1;
a_qnc.lo[1]=1; a_qnc.hi[1]=kte-kts+1; a_qnc.stride[1]=ite-its+1;
MynnArray<float,2> a_qni; a_qni.data=qni;
a_qni.lo[0]=1; a_qni.hi[0]=ite-its+1; a_qni.stride[0]=1;
a_qni.lo[1]=1; a_qni.hi[1]=kte-kts+1; a_qni.stride[1]=ite-its+1;
MynnArray<float,2> a_qnwfa; a_qnwfa.data=qnwfa;
a_qnwfa.lo[0]=1; a_qnwfa.hi[0]=ite-its+1; a_qnwfa.stride[0]=1;
a_qnwfa.lo[1]=1; a_qnwfa.hi[1]=kte-kts+1; a_qnwfa.stride[1]=ite-its+1;
MynnArray<float,2> a_qnifa; a_qnifa.data=qnifa;
a_qnifa.lo[0]=1; a_qnifa.hi[0]=ite-its+1; a_qnifa.stride[0]=1;
a_qnifa.lo[1]=1; a_qnifa.hi[1]=kte-kts+1; a_qnifa.stride[1]=ite-its+1;
MynnArray<float,2> a_qnbca; a_qnbca.data=qnbca;
a_qnbca.lo[0]=1; a_qnbca.hi[0]=ite-its+1; a_qnbca.stride[0]=1;
a_qnbca.lo[1]=1; a_qnbca.hi[1]=kte-kts+1; a_qnbca.stride[1]=ite-its+1;
MynnArray<float,2> a_qozone; a_qozone.data=qozone;
a_qozone.lo[0]=1; a_qozone.hi[0]=ite-its+1; a_qozone.stride[0]=1;
a_qozone.lo[1]=1; a_qozone.hi[1]=kte-kts+1; a_qozone.stride[1]=ite-its+1;
MynnArray<float,2> a_p; a_p.data=p;
a_p.lo[0]=1; a_p.hi[0]=ite-its+1; a_p.stride[0]=1;
a_p.lo[1]=1; a_p.hi[1]=kte-kts+1; a_p.stride[1]=ite-its+1;
MynnArray<float,2> a_exner; a_exner.data=exner;
a_exner.lo[0]=1; a_exner.hi[0]=ite-its+1; a_exner.stride[0]=1;
a_exner.lo[1]=1; a_exner.hi[1]=kte-kts+1; a_exner.stride[1]=ite-its+1;
MynnArray<float,2> a_rho; a_rho.data=rho;
a_rho.lo[0]=1; a_rho.hi[0]=ite-its+1; a_rho.stride[0]=1;
a_rho.lo[1]=1; a_rho.hi[1]=kte-kts+1; a_rho.stride[1]=ite-its+1;
MynnArray<float,2> a_tt; a_tt.data=tt;
a_tt.lo[0]=1; a_tt.hi[0]=ite-its+1; a_tt.stride[0]=1;
a_tt.lo[1]=1; a_tt.hi[1]=kte-kts+1; a_tt.stride[1]=ite-its+1;
MynnArray<float,1> a_xland; a_xland.data=xland;
a_xland.lo[0]=1; a_xland.hi[0]=ite-its+1; a_xland.stride[0]=1;
MynnArray<float,1> a_ts; a_ts.data=ts;
a_ts.lo[0]=1; a_ts.hi[0]=ite-its+1; a_ts.stride[0]=1;
MynnArray<float,1> a_qsfc; a_qsfc.data=qsfc;
a_qsfc.lo[0]=1; a_qsfc.hi[0]=ite-its+1; a_qsfc.stride[0]=1;
MynnArray<float,1> a_ps; a_ps.data=ps;
a_ps.lo[0]=1; a_ps.hi[0]=ite-its+1; a_ps.stride[0]=1;
MynnArray<float,1> a_ust; a_ust.data=ust;
a_ust.lo[0]=1; a_ust.hi[0]=ite-its+1; a_ust.stride[0]=1;
MynnArray<float,1> a_ch; a_ch.data=ch;
a_ch.lo[0]=1; a_ch.hi[0]=ite-its+1; a_ch.stride[0]=1;
MynnArray<float,1> a_hfx; a_hfx.data=hfx;
a_hfx.lo[0]=1; a_hfx.hi[0]=ite-its+1; a_hfx.stride[0]=1;
MynnArray<float,1> a_qfx; a_qfx.data=qfx;
a_qfx.lo[0]=1; a_qfx.hi[0]=ite-its+1; a_qfx.stride[0]=1;
MynnArray<float,1> a_rmol; a_rmol.data=rmol;
a_rmol.lo[0]=1; a_rmol.hi[0]=ite-its+1; a_rmol.stride[0]=1;
MynnArray<float,1> a_wspd; a_wspd.data=wspd;
a_wspd.lo[0]=1; a_wspd.hi[0]=ite-its+1; a_wspd.stride[0]=1;
MynnArray<float,1> a_uoce; a_uoce.data=uoce;
a_uoce.lo[0]=1; a_uoce.hi[0]=ite-its+1; a_uoce.stride[0]=1;
MynnArray<float,1> a_voce; a_voce.data=voce;
a_voce.lo[0]=1; a_voce.hi[0]=ite-its+1; a_voce.stride[0]=1;
MynnArray<float,2> a_qke; a_qke.data=qke;
a_qke.lo[0]=1; a_qke.hi[0]=ite-its+1; a_qke.stride[0]=1;
a_qke.lo[1]=1; a_qke.hi[1]=kte-kts+1; a_qke.stride[1]=ite-its+1;
MynnArray<float,2> a_qke_adv; a_qke_adv.data=qke_adv;
a_qke_adv.lo[0]=1; a_qke_adv.hi[0]=ite-its+1; a_qke_adv.stride[0]=1;
a_qke_adv.lo[1]=1; a_qke_adv.hi[1]=kte-kts+1; a_qke_adv.stride[1]=ite-its+1;
MynnArray<float,2> a_tsq; a_tsq.data=tsq;
a_tsq.lo[0]=1; a_tsq.hi[0]=ite-its+1; a_tsq.stride[0]=1;
a_tsq.lo[1]=1; a_tsq.hi[1]=kte-kts+1; a_tsq.stride[1]=ite-its+1;
MynnArray<float,2> a_qsq; a_qsq.data=qsq;
a_qsq.lo[0]=1; a_qsq.hi[0]=ite-its+1; a_qsq.stride[0]=1;
a_qsq.lo[1]=1; a_qsq.hi[1]=kte-kts+1; a_qsq.stride[1]=ite-its+1;
MynnArray<float,2> a_cov; a_cov.data=cov;
a_cov.lo[0]=1; a_cov.hi[0]=ite-its+1; a_cov.stride[0]=1;
a_cov.lo[1]=1; a_cov.hi[1]=kte-kts+1; a_cov.stride[1]=ite-its+1;
MynnArray<float,2> a_rublten; a_rublten.data=rublten;
a_rublten.lo[0]=1; a_rublten.hi[0]=ite-its+1; a_rublten.stride[0]=1;
a_rublten.lo[1]=1; a_rublten.hi[1]=kte-kts+1; a_rublten.stride[1]=ite-its+1;
MynnArray<float,2> a_rvblten; a_rvblten.data=rvblten;
a_rvblten.lo[0]=1; a_rvblten.hi[0]=ite-its+1; a_rvblten.stride[0]=1;
a_rvblten.lo[1]=1; a_rvblten.hi[1]=kte-kts+1; a_rvblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rthblten; a_rthblten.data=rthblten;
a_rthblten.lo[0]=1; a_rthblten.hi[0]=ite-its+1; a_rthblten.stride[0]=1;
a_rthblten.lo[1]=1; a_rthblten.hi[1]=kte-kts+1; a_rthblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqvblten; a_rqvblten.data=rqvblten;
a_rqvblten.lo[0]=1; a_rqvblten.hi[0]=ite-its+1; a_rqvblten.stride[0]=1;
a_rqvblten.lo[1]=1; a_rqvblten.hi[1]=kte-kts+1; a_rqvblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqcblten; a_rqcblten.data=rqcblten;
a_rqcblten.lo[0]=1; a_rqcblten.hi[0]=ite-its+1; a_rqcblten.stride[0]=1;
a_rqcblten.lo[1]=1; a_rqcblten.hi[1]=kte-kts+1; a_rqcblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqiblten; a_rqiblten.data=rqiblten;
a_rqiblten.lo[0]=1; a_rqiblten.hi[0]=ite-its+1; a_rqiblten.stride[0]=1;
a_rqiblten.lo[1]=1; a_rqiblten.hi[1]=kte-kts+1; a_rqiblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqsblten; a_rqsblten.data=rqsblten;
a_rqsblten.lo[0]=1; a_rqsblten.hi[0]=ite-its+1; a_rqsblten.stride[0]=1;
a_rqsblten.lo[1]=1; a_rqsblten.hi[1]=kte-kts+1; a_rqsblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqncblten; a_rqncblten.data=rqncblten;
a_rqncblten.lo[0]=1; a_rqncblten.hi[0]=ite-its+1; a_rqncblten.stride[0]=1;
a_rqncblten.lo[1]=1; a_rqncblten.hi[1]=kte-kts+1; a_rqncblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqniblten; a_rqniblten.data=rqniblten;
a_rqniblten.lo[0]=1; a_rqniblten.hi[0]=ite-its+1; a_rqniblten.stride[0]=1;
a_rqniblten.lo[1]=1; a_rqniblten.hi[1]=kte-kts+1; a_rqniblten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqnwfablten; a_rqnwfablten.data=rqnwfablten;
a_rqnwfablten.lo[0]=1; a_rqnwfablten.hi[0]=ite-its+1; a_rqnwfablten.stride[0]=1;
a_rqnwfablten.lo[1]=1; a_rqnwfablten.hi[1]=kte-kts+1; a_rqnwfablten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqnifablten; a_rqnifablten.data=rqnifablten;
a_rqnifablten.lo[0]=1; a_rqnifablten.hi[0]=ite-its+1; a_rqnifablten.stride[0]=1;
a_rqnifablten.lo[1]=1; a_rqnifablten.hi[1]=kte-kts+1; a_rqnifablten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqnbcablten; a_rqnbcablten.data=rqnbcablten;
a_rqnbcablten.lo[0]=1; a_rqnbcablten.hi[0]=ite-its+1; a_rqnbcablten.stride[0]=1;
a_rqnbcablten.lo[1]=1; a_rqnbcablten.hi[1]=kte-kts+1; a_rqnbcablten.stride[1]=ite-its+1;
MynnArray<float,2> a_rqozblten; a_rqozblten.data=rqozblten;
a_rqozblten.lo[0]=1; a_rqozblten.hi[0]=ite-its+1; a_rqozblten.stride[0]=1;
a_rqozblten.lo[1]=1; a_rqozblten.hi[1]=kte-kts+1; a_rqozblten.stride[1]=ite-its+1;
MynnArray<float,2> a_exch_h; a_exch_h.data=exch_h;
a_exch_h.lo[0]=1; a_exch_h.hi[0]=ite-its+1; a_exch_h.stride[0]=1;
a_exch_h.lo[1]=1; a_exch_h.hi[1]=kte-kts+1; a_exch_h.stride[1]=ite-its+1;
MynnArray<float,2> a_exch_m; a_exch_m.data=exch_m;
a_exch_m.lo[0]=1; a_exch_m.hi[0]=ite-its+1; a_exch_m.stride[0]=1;
a_exch_m.lo[1]=1; a_exch_m.hi[1]=kte-kts+1; a_exch_m.stride[1]=ite-its+1;
MynnArray<float,1> a_pblh; a_pblh.data=pblh;
a_pblh.lo[0]=1; a_pblh.hi[0]=ite-its+1; a_pblh.stride[0]=1;
MynnArray<int,1> a_kpbl; a_kpbl.data=kpbl;
a_kpbl.lo[0]=1; a_kpbl.hi[0]=ite-its+1; a_kpbl.stride[0]=1;
MynnArray<float,2> a_el_pbl; a_el_pbl.data=el_pbl;
a_el_pbl.lo[0]=1; a_el_pbl.hi[0]=ite-its+1; a_el_pbl.stride[0]=1;
a_el_pbl.lo[1]=1; a_el_pbl.hi[1]=kte-kts+1; a_el_pbl.stride[1]=ite-its+1;
MynnArray<float,2> a_dqke; a_dqke.data=dqke;
a_dqke.lo[0]=1; a_dqke.hi[0]=ite-its+1; a_dqke.stride[0]=1;
a_dqke.lo[1]=1; a_dqke.hi[1]=kte-kts+1; a_dqke.stride[1]=ite-its+1;
MynnArray<float,2> a_qwt; a_qwt.data=qwt;
a_qwt.lo[0]=1; a_qwt.hi[0]=ite-its+1; a_qwt.stride[0]=1;
a_qwt.lo[1]=1; a_qwt.hi[1]=kte-kts+1; a_qwt.stride[1]=ite-its+1;
MynnArray<float,2> a_qshear; a_qshear.data=qshear;
a_qshear.lo[0]=1; a_qshear.hi[0]=ite-its+1; a_qshear.stride[0]=1;
a_qshear.lo[1]=1; a_qshear.hi[1]=kte-kts+1; a_qshear.stride[1]=ite-its+1;
MynnArray<float,2> a_qbuoy; a_qbuoy.data=qbuoy;
a_qbuoy.lo[0]=1; a_qbuoy.hi[0]=ite-its+1; a_qbuoy.stride[0]=1;
a_qbuoy.lo[1]=1; a_qbuoy.hi[1]=kte-kts+1; a_qbuoy.stride[1]=ite-its+1;
MynnArray<float,2> a_qdiss; a_qdiss.data=qdiss;
a_qdiss.lo[0]=1; a_qdiss.hi[0]=ite-its+1; a_qdiss.stride[0]=1;
a_qdiss.lo[1]=1; a_qdiss.hi[1]=kte-kts+1; a_qdiss.stride[1]=ite-its+1;
MynnArray<float,2> a_sh; a_sh.data=sh;
a_sh.lo[0]=1; a_sh.hi[0]=ite-its+1; a_sh.stride[0]=1;
a_sh.lo[1]=1; a_sh.hi[1]=kte-kts+1; a_sh.stride[1]=ite-its+1;
MynnArray<float,2> a_sm; a_sm.data=sm;
a_sm.lo[0]=1; a_sm.hi[0]=ite-its+1; a_sm.stride[0]=1;
a_sm.lo[1]=1; a_sm.hi[1]=kte-kts+1; a_sm.stride[1]=ite-its+1;
MynnArray<float,2> a_qc_bl; a_qc_bl.data=qc_bl;
a_qc_bl.lo[0]=1; a_qc_bl.hi[0]=ite-its+1; a_qc_bl.stride[0]=1;
a_qc_bl.lo[1]=1; a_qc_bl.hi[1]=kte-kts+1; a_qc_bl.stride[1]=ite-its+1;
MynnArray<float,2> a_qi_bl; a_qi_bl.data=qi_bl;
a_qi_bl.lo[0]=1; a_qi_bl.hi[0]=ite-its+1; a_qi_bl.stride[0]=1;
a_qi_bl.lo[1]=1; a_qi_bl.hi[1]=kte-kts+1; a_qi_bl.stride[1]=ite-its+1;
MynnArray<float,2> a_cldfra_bl; a_cldfra_bl.data=cldfra_bl;
a_cldfra_bl.lo[0]=1; a_cldfra_bl.hi[0]=ite-its+1; a_cldfra_bl.stride[0]=1;
a_cldfra_bl.lo[1]=1; a_cldfra_bl.hi[1]=kte-kts+1; a_cldfra_bl.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_a; a_edmf_a.data=edmf_a;
a_edmf_a.lo[0]=1; a_edmf_a.hi[0]=ite-its+1; a_edmf_a.stride[0]=1;
a_edmf_a.lo[1]=1; a_edmf_a.hi[1]=kte-kts+1; a_edmf_a.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_w; a_edmf_w.data=edmf_w;
a_edmf_w.lo[0]=1; a_edmf_w.hi[0]=ite-its+1; a_edmf_w.stride[0]=1;
a_edmf_w.lo[1]=1; a_edmf_w.hi[1]=kte-kts+1; a_edmf_w.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_qt; a_edmf_qt.data=edmf_qt;
a_edmf_qt.lo[0]=1; a_edmf_qt.hi[0]=ite-its+1; a_edmf_qt.stride[0]=1;
a_edmf_qt.lo[1]=1; a_edmf_qt.hi[1]=kte-kts+1; a_edmf_qt.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_thl; a_edmf_thl.data=edmf_thl;
a_edmf_thl.lo[0]=1; a_edmf_thl.hi[0]=ite-its+1; a_edmf_thl.stride[0]=1;
a_edmf_thl.lo[1]=1; a_edmf_thl.hi[1]=kte-kts+1; a_edmf_thl.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_ent; a_edmf_ent.data=edmf_ent;
a_edmf_ent.lo[0]=1; a_edmf_ent.hi[0]=ite-its+1; a_edmf_ent.stride[0]=1;
a_edmf_ent.lo[1]=1; a_edmf_ent.hi[1]=kte-kts+1; a_edmf_ent.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_qc; a_edmf_qc.data=edmf_qc;
a_edmf_qc.lo[0]=1; a_edmf_qc.hi[0]=ite-its+1; a_edmf_qc.stride[0]=1;
a_edmf_qc.lo[1]=1; a_edmf_qc.hi[1]=kte-kts+1; a_edmf_qc.stride[1]=ite-its+1;
MynnArray<float,2> a_sub_thl; a_sub_thl.data=sub_thl;
a_sub_thl.lo[0]=1; a_sub_thl.hi[0]=ite-its+1; a_sub_thl.stride[0]=1;
a_sub_thl.lo[1]=1; a_sub_thl.hi[1]=kte-kts+1; a_sub_thl.stride[1]=ite-its+1;
MynnArray<float,2> a_sub_sqv; a_sub_sqv.data=sub_sqv;
a_sub_sqv.lo[0]=1; a_sub_sqv.hi[0]=ite-its+1; a_sub_sqv.stride[0]=1;
a_sub_sqv.lo[1]=1; a_sub_sqv.hi[1]=kte-kts+1; a_sub_sqv.stride[1]=ite-its+1;
MynnArray<float,2> a_det_thl; a_det_thl.data=det_thl;
a_det_thl.lo[0]=1; a_det_thl.hi[0]=ite-its+1; a_det_thl.stride[0]=1;
a_det_thl.lo[1]=1; a_det_thl.hi[1]=kte-kts+1; a_det_thl.stride[1]=ite-its+1;
MynnArray<float,2> a_det_sqv; a_det_sqv.data=det_sqv;
a_det_sqv.lo[0]=1; a_det_sqv.hi[0]=ite-its+1; a_det_sqv.stride[0]=1;
a_det_sqv.lo[1]=1; a_det_sqv.hi[1]=kte-kts+1; a_det_sqv.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_a_dd; a_edmf_a_dd.data=edmf_a_dd;
a_edmf_a_dd.lo[0]=1; a_edmf_a_dd.hi[0]=ite-its+1; a_edmf_a_dd.stride[0]=1;
a_edmf_a_dd.lo[1]=1; a_edmf_a_dd.hi[1]=kte-kts+1; a_edmf_a_dd.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_w_dd; a_edmf_w_dd.data=edmf_w_dd;
a_edmf_w_dd.lo[0]=1; a_edmf_w_dd.hi[0]=ite-its+1; a_edmf_w_dd.stride[0]=1;
a_edmf_w_dd.lo[1]=1; a_edmf_w_dd.hi[1]=kte-kts+1; a_edmf_w_dd.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_qt_dd; a_edmf_qt_dd.data=edmf_qt_dd;
a_edmf_qt_dd.lo[0]=1; a_edmf_qt_dd.hi[0]=ite-its+1; a_edmf_qt_dd.stride[0]=1;
a_edmf_qt_dd.lo[1]=1; a_edmf_qt_dd.hi[1]=kte-kts+1; a_edmf_qt_dd.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_thl_dd; a_edmf_thl_dd.data=edmf_thl_dd;
a_edmf_thl_dd.lo[0]=1; a_edmf_thl_dd.hi[0]=ite-its+1; a_edmf_thl_dd.stride[0]=1;
a_edmf_thl_dd.lo[1]=1; a_edmf_thl_dd.hi[1]=kte-kts+1; a_edmf_thl_dd.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_ent_dd; a_edmf_ent_dd.data=edmf_ent_dd;
a_edmf_ent_dd.lo[0]=1; a_edmf_ent_dd.hi[0]=ite-its+1; a_edmf_ent_dd.stride[0]=1;
a_edmf_ent_dd.lo[1]=1; a_edmf_ent_dd.hi[1]=kte-kts+1; a_edmf_ent_dd.stride[1]=ite-its+1;
MynnArray<float,2> a_edmf_qc_dd; a_edmf_qc_dd.data=edmf_qc_dd;
a_edmf_qc_dd.lo[0]=1; a_edmf_qc_dd.hi[0]=ite-its+1; a_edmf_qc_dd.stride[0]=1;
a_edmf_qc_dd.lo[1]=1; a_edmf_qc_dd.hi[1]=kte-kts+1; a_edmf_qc_dd.stride[1]=ite-its+1;
MynnArray<float,1> a_maxwidth; a_maxwidth.data=maxwidth;
a_maxwidth.lo[0]=1; a_maxwidth.hi[0]=ite-its+1; a_maxwidth.stride[0]=1;
MynnArray<float,1> a_maxmf; a_maxmf.data=maxmf;
a_maxmf.lo[0]=1; a_maxmf.hi[0]=ite-its+1; a_maxmf.stride[0]=1;
MynnArray<float,1> a_ztop_plume; a_ztop_plume.data=ztop_plume;
a_ztop_plume.lo[0]=1; a_ztop_plume.hi[0]=ite-its+1; a_ztop_plume.stride[0]=1;
MynnArray<int,1> a_ktop_plume; a_ktop_plume.data=ktop_plume;
a_ktop_plume.lo[0]=1; a_ktop_plume.hi[0]=ite-its+1; a_ktop_plume.stride[0]=1;
MynnArray<float,2> a_pattern_spp_pbl; a_pattern_spp_pbl.data=pattern_spp_pbl;
a_pattern_spp_pbl.lo[0]=1; a_pattern_spp_pbl.hi[0]=ite-its+1; a_pattern_spp_pbl.stride[0]=1;
a_pattern_spp_pbl.lo[1]=1; a_pattern_spp_pbl.hi[1]=kte-kts+1; a_pattern_spp_pbl.stride[1]=ite-its+1;
MynnArray<float,2> a_rthraten; a_rthraten.data=rthraten;
a_rthraten.lo[0]=1; a_rthraten.hi[0]=ite-its+1; a_rthraten.stride[0]=1;
a_rthraten.lo[1]=1; a_rthraten.hi[1]=kte-kts+1; a_rthraten.stride[1]=ite-its+1;
column.f_bl_mynn_run(initflag, restart, cycling, delt, a_dz, a_dx, a_znt, a_u, a_v, a_w, a_th, a_sqv, a_sqc, a_sqi, a_sqs, a_qnc, a_qni, a_qnwfa, a_qnifa, a_qnbca, a_qozone, a_p, a_exner, a_rho, a_tt, a_xland, a_ts, a_qsfc, a_ps, a_ust, a_ch, a_hfx, a_qfx, a_rmol, a_wspd, a_uoce, a_voce, a_qke, a_qke_adv, a_tsq, a_qsq, a_cov, a_rublten, a_rvblten, a_rthblten, a_rqvblten, a_rqcblten, a_rqiblten, a_rqsblten, a_rqncblten, a_rqniblten, a_rqnwfablten, a_rqnifablten, a_rqnbcablten, a_rqozblten, a_exch_h, a_exch_m, a_pblh, a_kpbl, a_el_pbl, a_dqke, a_qwt, a_qshear, a_qbuoy, a_qdiss, a_sh, a_sm, a_qc_bl, a_qi_bl, a_cldfra_bl, icloud_bl, bl_mynn_tkeadvect, bl_mynn_tkebudget, bl_mynn_cloudpdf, bl_mynn_mixlength, bl_mynn_closure, bl_mynn_stfunc, bl_mynn_topdown, bl_mynn_edmf, bl_mynn_edmf_dd, bl_mynn_edmf_mom, bl_mynn_edmf_tke, bl_mynn_mixscalars, bl_mynn_output, bl_mynn_cloudmix, bl_mynn_mixqt, bl_mynn_scaleaware, bl_mynn_dheatopt, a_edmf_a, a_edmf_w, a_edmf_qt, a_edmf_thl, a_edmf_ent, a_edmf_qc, a_sub_thl, a_sub_sqv, a_det_thl, a_det_sqv, a_edmf_a_dd, a_edmf_w_dd, a_edmf_qt_dd, a_edmf_thl_dd, a_edmf_ent_dd, a_edmf_qc_dd, a_maxwidth, a_maxmf, a_ztop_plume, a_ktop_plume, spp_pbl, a_pattern_spp_pbl, a_rthraten, flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa, flag_qnbca, flag_qoz, its, ite, kts, kte, kme, errmsg, errflg);
error_flags[batch_first+thread]=errflg;
}
}
'''

#: Correct-mode kernel text of MYNN (driver_v842.correct_source rows: exact text, corrected text, count).
#: RP7: mynn_tendencies (bl_mynn_subroutines.F90:3102) sets the lowest layer's pressure thickness to
#: psfc minus the hydrostatic pressure interpolated to the top of layer 1, i.e. MPAS's full (non-
#: hydrostatic) surface pressure psfc_p against its hydrostatic pres_hyd_p.  A near-surface
#: non-hydrostatic deficit larger than half the layer makes it zero or negative, and moisture_check
#: (F90:4149-4150) divides by it: 0/0 at delp(kts) = 0 made rqvblten (and, through the moisture
#: post-conversion, rqc/rqi/rqs) NaN at level 1 of one column, and the dycore then refused the step
#: with ~4,100-4,600 whole columns non-finite (MYNN-NAN-1: x1.40962 plainsst 2021-07-01 at t=1658400 s
#: and 2020-01-01 at t=6672600 s, delp(kts) = 0.0 in both).  The corrected thickness is the layer's
#: own hydrostatic weight rho*g*dz, the formula the NOAA-GSL MYNN-EDMF uses for every level
#: (module_bl_mynnedmf.F90:4270), so it is positive with the hydrostatic interior thicknesses.
MYNN_TEXT_CORRECTIONS = {
    "RP7": (
        ("delp(kts) = mpas_sub(psfc, mpas_div(mpas_add(mpas_mul(p(((kts) + (1))), dz(kts)), "
         "mpas_mul(p(kts), dz(((kts) + (1))))), mpas_add(dz(kts), dz(((kts) + (1))))));",
         "delp(kts) = mpas_mul(mpas_mul(rho(kts), grav), dz(kts));", 1),
    ),
}

MYNN_ARGUMENTS = {'initflag': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'restart': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'cycling': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'delt': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'dz': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'dx': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'znt': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'u': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'v': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'w': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'th': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'sqv': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'sqc': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'sqi': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'sqs': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qnc': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qni': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qnwfa': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qnifa': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qnbca': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'qozone': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'p': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'exner': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'rho': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'tt': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'xland': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'ts': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'qsfc': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'ps': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'ust': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'ch': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'hfx': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'qfx': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'rmol': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'wspd': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'uoce': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'voce': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'in', 'optional': False}, 'qke': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'qke_adv': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'tsq': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'qsq': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'cov': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rublten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rvblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rthblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqvblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqcblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqiblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqsblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqncblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqniblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqnwfablten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqnifablten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqnbcablten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'rqozblten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'exch_h': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': False}, 'exch_m': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': False}, 'pblh': {'kind': 'float', 'dimensions': ['its:'], 'intent': 'inout', 'optional': False}, 'kpbl': {'kind': 'int', 'dimensions': ['its:'], 'intent': 'inout', 'optional': False}, 'el_pbl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'dqke': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': True}, 'qwt': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': True}, 'qshear': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': True}, 'qbuoy': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': True}, 'qdiss': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'out', 'optional': True}, 'sh': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'sm': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'qc_bl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'qi_bl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'cldfra_bl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'icloud_bl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_tkeadvect': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_tkebudget': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudpdf': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixlength': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_closure': {'kind': 'float', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_stfunc': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_topdown': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_dd': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_mom': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_edmf_tke': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixscalars': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_output': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_cloudmix': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_mixqt': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_scaleaware': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'bl_mynn_dheatopt': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'edmf_a': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_w': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_qt': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_thl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_ent': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_qc': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'sub_thl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'sub_sqv': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'det_thl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'det_sqv': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': False}, 'edmf_a_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'edmf_w_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'edmf_qt_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'edmf_thl_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'edmf_ent_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'edmf_qc_dd': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'inout', 'optional': True}, 'maxwidth': {'kind': 'float', 'dimensions': [':'], 'intent': 'out', 'optional': False}, 'maxmf': {'kind': 'float', 'dimensions': [':'], 'intent': 'out', 'optional': False}, 'ztop_plume': {'kind': 'float', 'dimensions': [':'], 'intent': 'out', 'optional': False}, 'ktop_plume': {'kind': 'int', 'dimensions': ['its:'], 'intent': 'inout', 'optional': False}, 'spp_pbl': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'pattern_spp_pbl': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'rthraten': {'kind': 'float', 'dimensions': ['its:', ':'], 'intent': 'in', 'optional': False}, 'flag_qc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qi': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qs': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qnc': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qni': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qnwfa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qnifa': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qnbca': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'flag_qoz': {'kind': 'bool', 'dimensions': [], 'intent': 'in', 'optional': False}, 'its': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'ite': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kts': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kte': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'kme': {'kind': 'int', 'dimensions': [], 'intent': 'in', 'optional': False}, 'errmsg': {'kind': 'char', 'dimensions': [], 'intent': 'out', 'optional': False}, 'errflg': {'kind': 'int', 'dimensions': [], 'intent': 'out', 'optional': False}}
MYNN_WORKSPACE_LEVEL_BYTES = 1752
MYNN_WORKSPACE_FIXED_BYTES = 2723
# END GENERATED MPAS MMM MYNN
