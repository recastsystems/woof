"""MPAS-A v8.4.2 New Tiedtke at the convection driver boundary.

The entire MMM cu_ntiedtke.F90:1-3594 is byte-identical to the WRF 4.6.1
authority of woof.core.kernels.ntiedtke. Reuse its complete transcription,
workspace and call order, with an independent NVRTC module compiled without
contraction. Do not enter the engine's WRF physics adapter: its forcing sum,
constants, timestep flags and precipitation coupling differ from MPAS.

Arrays are float32 CUDA arrays in (level, cell) order, bottom level first;
interface arrays have one extra level. All physics arithmetic runs on CUDA.
MPAS framework packing into this layout belongs to the common physics seam.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

SCHEME = "cu_ntiedtke"
BACKEND = "hex-cuda"
MPAS_SCHEME_SHA256 = "e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1"
CUDA_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")

# MPAS src/core_atmosphere/physics/physics_mmm/cu_ntiedtke.F90.
# These citations cover every reused routine, including inlined helpers.
MPAS_ROUTINES = {
    "cu_ntiedtke_init": (81, 123),  # cu_ntiedtke.F90:81-123
    "cu_ntiedtke_finalize": (129, 141),  # cu_ntiedtke.F90:129-141
    "cu_ntiedtke_run": (148, 332),  # cu_ntiedtke.F90:148-332
    "cumastrn": (342, 1085),  # cu_ntiedtke.F90:342-1085
    "cuinin": (1091, 1216),  # cu_ntiedtke.F90:1091-1216
    "cutypen": (1221, 1750),  # cu_ntiedtke.F90:1221-1750
    "cuascn": (1755, 2258),  # cu_ntiedtke.F90:1755-2258
    "cudlfsn": (2262, 2487),  # cu_ntiedtke.F90:2262-2487
    "cuddrafn": (2495, 2721),  # cu_ntiedtke.F90:2495-2721
    "cuflxn": (2725, 3060),  # cu_ntiedtke.F90:2725-3060
    "cudtdqn": (3064, 3148),  # cu_ntiedtke.F90:3064-3148
    "cududvn": (3152, 3252),  # cu_ntiedtke.F90:3152-3252
    "cuadjtqn": (3256, 3402),  # cu_ntiedtke.F90:3256-3402
    "cubasmcn": (3406, 3485),  # cu_ntiedtke.F90:3406-3485
    "cuentrn": (3489, 3538),  # cu_ntiedtke.F90:3489-3538
    "foealfa": (3542, 3557),  # cu_ntiedtke.F90:3542-3557
    "foelhm": (3559, 3564),  # cu_ntiedtke.F90:3559-3564
    "foeewm": (3566, 3573),  # cu_ntiedtke.F90:3566-3573
    "foedem": (3575, 3581),  # cu_ntiedtke.F90:3575-3581
    "foeldcpm": (3583, 3589),  # cu_ntiedtke.F90:3583-3589
}
LEVEL_INPUTS = ("t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d",
                "pcps", "dz8w", "rho3d", "exner", "qvften", "thften")
INTERFACE_INPUTS = ("p8w", "w")
SURFACE_INPUTS = ("xland", "hfx", "qfx", "dx")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                 "rucuten", "rvcuten")
SURFACE_OUTPUTS = ("raincv", "pratec")


@dataclass(frozen=True)
class Constants:
    """mpas_constants.F:46-50.

    src/core_atmosphere/physics/mpas_atmphys_constants.F:42,60-62.
    """
    cp: float = 1004.5
    rd: float = 287.0
    rv: float = 461.6
    xlv: float = 2.50e6
    xls: float = 2.85e6
    xlf: float = 3.50e5
    grav: float = 9.80616


# MPAS mpas_atmphys_driver_convection.F:827-850. Every operation has an
# explicit rounding boundary; radiation cannot be pre-summed as in WRF.
DRIVER_CUDA = r'''
// mpas_atmphys_driver_convection.F:827-850.
extern "C" __global__ void mpas_ntiedtke_forcing(
    const float *rthdynten, const float *rthblten,
    const float *rthratenlw, const float *rthratensw, const float *exner,
    const float *rqvdynten, const float *rqvblten,
    float *thften, float *qvften, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= size) return;
    float s = __fadd_rn(rthdynten[j], rthblten[j]);
    s = __fadd_rn(s, rthratenlw[j]);
    s = __fadd_rn(s, rthratensw[j]);
    thften[j] = __fmul_rn(s, exner[j]);
    qvften[j] = __fadd_rn(rqvdynten[j], rqvblten[j]);
}
// mpas_atmphys_driver_convection.F:825-826.
extern "C" __global__ void mpas_ntiedtke_dx(
    const float *meshDensity, float len_disp, float *dx, int ncol) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ncol) dx[j] = __fdiv_rn(len_disp, gfk_pow(meshDensity[j], 0.25f));
}
// cu_ntiedtke.F90:100-118: materialize all initialized module variables.
extern "C" __global__ void mpas_ntiedtke_init_state(
    float cp, float rd, float rv, float xlv, float xls, float xlf,
    float grav, float *out) {
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);
    out[0]=c.alf; out[1]=c.als; out[2]=c.alv; out[3]=c.c2es;
    out[4]=c.c5ies; out[5]=c.c5les; out[6]=c.cpd; out[7]=c.g;
    out[8]=c.r5alscp; out[9]=c.r5alvcp; out[10]=c.ralfdcp;
    out[11]=c.ralsdcp; out[12]=c.ralvdcp; out[13]=c.rcpd;
    out[14]=c.rd; out[15]=rv; out[16]=c.vtmpc1; out[17]=c.zrg;
}
// cu_ntiedtke.F90:1903-2012,3406-3485. Find the first level at which the
// slab-wide ascent flag becomes true. Before that level no ascent updates
// occur: labels propagate zeros upward, and cubasmcn reads only the fixed
// environment. Its first eligible level therefore determines the latch.
extern "C" __global__ void mpas_ntiedtke_ascent_start(
    const int *ldcum, const int *ktype, const int *klab,
    const float *pqen, const float *pqsen, const float *pgeo,
    int *start, int ncol, int klev, float grav) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    int first = 0;
    if (ldcum[i] && ktype[i] != 3) {
        if (klab[(size_t)klev * ncol + i] > 0) first = klev - 1;
    } else {
        float zrg = __fdiv_rn(1.0f, grav);
        for (int jk = klev - 1; jk >= 3; --jk) {
            size_t j = (size_t)jk * ncol + i;
            float h = __fmul_rn(pgeo[j], zrg);
            if (pqen[j] > __fmul_rn(0.80f, pqsen[j]) && h > 500.0f && h < 10000.0f) {
                first = jk;
                break;
            }
        }
    }
    atomicMax(start, first);
}
'''


# cu_ntiedtke.F90:148-332,342-1085,3064-3252. The strict native
# binary propagates SSE operand NaNs; CUDA arithmetic canonicalizes them.
# Keep finite operations at their original explicit rounding boundaries and
# implement the measured operand/invalid words before they enter the scheme.
NATIVE_ARITHMETIC_CUDA = r'''
// cu_ntiedtke.F90:1300,1449,1670: strict CVTTSS2SI returns the
// integer-indefinite word for an unrepresentable cloud-base pressure.
// CUDA's cast saturates positive overflow instead. Keep the native word.
__device__ __forceinline__ int mpas_nt_int(float x) {
    if (!(x >= -2147483648.0f && x < 2147483648.0f))
        return -2147483647 - 1;
    return (int)x;
}
__device__ __forceinline__ bool mpas_nt_nan(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) > 0x7f800000u;
}
__device__ __forceinline__ float mpas_nt_quiet(float x) {
    return __uint_as_float(__float_as_uint(x) | 0x00400000u);
}
__device__ __forceinline__ float mpas_nt_neg(float x) {
    return __uint_as_float(__float_as_uint(x) ^ 0x80000000u);
}
__device__ __forceinline__ float mpas_nt_invalid(float x) {
    return mpas_nt_nan(x) ? __uint_as_float(0xffc00000u) : x;
}
__device__ __forceinline__ float mpas_nt_add(float a, float b) {
    if (mpas_nt_nan(a)) return mpas_nt_quiet(a);
    if (mpas_nt_nan(b)) return mpas_nt_quiet(b);
    return mpas_nt_invalid(__fadd_rn(a, b));
}
__device__ __forceinline__ float mpas_nt_sub(float a, float b) {
    if (mpas_nt_nan(a)) return mpas_nt_quiet(a);
    if (mpas_nt_nan(b)) return mpas_nt_quiet(b);
    return mpas_nt_invalid(__fsub_rn(a, b));
}
__device__ __forceinline__ float mpas_nt_mul(float a, float b) {
    if (mpas_nt_nan(a)) return mpas_nt_quiet(a);
    if (mpas_nt_nan(b)) return mpas_nt_quiet(b);
    return mpas_nt_invalid(__fmul_rn(a, b));
}
__device__ __forceinline__ float mpas_nt_div(float a, float b) {
    if (mpas_nt_nan(a)) return mpas_nt_quiet(a);
    if (mpas_nt_nan(b)) return mpas_nt_quiet(b);
    return mpas_nt_invalid(__fdiv_rn(a, b));
}
__device__ __forceinline__ float mpas_nt_sqrt(float a) {
    if (mpas_nt_nan(a)) return mpas_nt_quiet(a);
    return mpas_nt_invalid(__fsqrt_rn(a));
}
'''


@lru_cache(maxsize=1)
def cuda_source() -> str:
    """cu_ntiedtke.F90:81-123,1-3594; reuse the identical full scheme.

    The shared glibc_flt32.cuh is prepended by module_source. Propagate xls
    explicitly: MPAS init accepts it separately from xlv and xlf.
    """
    from woof.core.kernels import module_source

    source = _preserve_native_float_words(
        _adapt_scheme_source(module_source("ntiedtke")) + DRIVER_CUDA)
    # Retain the shared glibc algorithms. Its invalid/NaN special cases
    # must preserve the native operand and invalid sign as well.
    source = source.replace(
        "if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)\n"
        "            return __int_as_float(0x7fc00000);",
        "if (mpas_nt_nan(x)) return mpas_nt_quiet(x);\n"
        "        if (ix & 0x80000000u) return __int_as_float(0xffc00000);")
    source = source.replace(
        "if (yint == 0) return __int_as_float(0x7fc00000);",
        "if (yint == 0) return __int_as_float(0xffc00000);")
    return NATIVE_ARITHMETIC_CUDA + source


def _preserve_native_float_words(source: str) -> str:
    """cu_ntiedtke.F90:1-3594; native words at every arithmetic boundary."""
    source = _preserve_float_negations(_preserve_float_subtractions(source))
    for intrinsic, operation in (("__fadd_rn", "add"), ("__fsub_rn", "sub"),
                                 ("__fmul_rn", "mul"), ("__fdiv_rn", "div"),
                                 ("__fsqrt_rn", "sqrt")):
        source = source.replace(intrinsic, "mpas_nt_" + operation)
    return source


def _preserve_float_subtractions(source: str) -> str:
    """cu_ntiedtke.F90:3064-3252; retain SUBSS NaN operand semantics.

    The reused transcription spells subtraction as add(a,-b). Native
    subtraction preserves b's NaN sign rather than negating it first.
    All finite rounding boundaries are identical in the two forms.
    """
    marker = "__fadd_rn("
    result, offset = [], 0
    while (start := source.find(marker, offset)) >= 0:
        result.append(source[offset:start])
        first, end, comma, depth = start + len(marker), start + len(marker), None, 1
        while depth:
            char = source[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 1:
                comma = end
            end += 1
        if comma is None:
            raise ValueError("rounded addition must have two operands")
        a = _preserve_float_subtractions(source[first:comma])
        b = _preserve_float_subtractions(source[comma + 1:end - 1])
        operand = b.lstrip()
        if operand.startswith("-"):
            result.append("__fsub_rn(" + a + "," + b[:len(b) - len(operand)] + operand[1:] + ")")
        else:
            result.append(marker + a + "," + b + ")")
        offset = end
    result.append(source[offset:])
    return "".join(result)


def _preserve_float_negations(source: str) -> str:
    """cu_ntiedtke.F90:3064-3252; negate by sign bit, including NaNs.

    Unary CUDA negation also canonicalizes NaNs. Apply the sign-bit operation
    to float intrinsics, field access macros and declared float scalars;
    integer indexing and compile-time numeric literals retain their syntax.
    """
    import re

    scalars = set(re.findall(r"\bfloat\s+(\w+)", source))
    pattern = re.compile(r"-(?P<name>\w+(?:\.\w+)?)")
    offset = 0
    while match := pattern.search(source, offset):
        start, end = match.span()
        name = match.group("name")
        callable_float = (name.startswith("__f") and name.endswith("_rn")
                          or name.startswith(("nt_foe", "gfk_"))
                          or name in ("NTC", "NTK", "A", "NT_IN", "NT_OUT"))
        scalar_float = name in scalars or name.startswith("c.")
        if not (callable_float or scalar_float):
            offset = end
            continue
        previous = source[:start].rstrip()
        if previous and previous[-1] not in "(,=:+-*/!?{;":
            offset = end
            continue
        if callable_float:
            opening = end
            while source[opening].isspace():
                opening += 1
            if source[opening] != "(":
                offset = end
                continue
            end, depth = opening + 1, 1
            while depth:
                depth += (source[end] == "(") - (source[end] == ")")
                end += 1
        expression = _preserve_float_negations(source[start + 1:end])
        replacement = "mpas_nt_neg(" + expression + ")"
        source = source[:start] + replacement + source[end:]
        offset = start + len(replacement)
    return source


def _adapt_scheme_source(source: str) -> str:
    """cu_ntiedtke.F90:81-123,1903-2012; MPAS constants and ascent latch."""
    source = source.replace("float xlv, float xlf", "float xlv, float xls, float xlf")
    source = source.replace("nt_init(cp, rd, rv, xlv, xlf, grav)",
                            "nt_init(cp, rd, rv, xlv, xls, xlf, grav)")
    # cuddrafn:2616-2650 leaves itopde unassigned on a malformed pressure
    # ladder. Both strict native hosts have variable exits there; preserve
    # that undefined read rather than impose the reused kernel's early exit.
    # Completed malformed-column words are diagnostic, never acceptance.
    source = source.replace("int itopde = -1;", "int itopde;")
    source = source.replace(
        "if (itopde < 0) return;          /* never set: nothing to descend into */", "")
    source = source.replace("const float als = __fadd_rn(xlv, xlf);",
                            "const float als = xls;")
    # cu_ntiedtke.F90:2020: strict GNU emits MINSS with zdmfde as
    # its second operand. An unordered comparison retains that operand,
    # including its NaN word; fminf instead chooses the finite cap.
    # Preserve the measured native operation, including equal signed zeros.
    source = source.replace(
        "zdmfde = fminf(zdmfde, __fmul_rn(0.75f, NTC(pmfu, jk + 1)));",
        "const float mpas_detrain_bound = __fmul_rn(0.75f, NTC(pmfu, jk + 1));\n"
        "            if (mpas_detrain_bound < zdmfde) zdmfde = mpas_detrain_bound;")
    # cu_ntiedtke.F90:1173-1183: strict MAXSS retains z1 when the
    # smoothing comparison is unordered; fmaxf discards its NaN and changes
    # the subsequent parcel temperature and finite moisture tendencies.
    source = source.replace(
        "__fadd_rn(fmaxf(z1, z2), -A(pgeoh, k))",
        "__fadd_rn((z2 > z1 ? z2 : z1), -A(pgeoh, k))")
    # cu_ntiedtke.F90:3023-3026: the pinned strict executable evaluates
    # MAX(zrnew,0) before MAX(...,zrmin), then MIN(...,0). Its original
    # MAXSS retains zrmin on unordered comparison. Preserve the measured
    # grouping; dropping either NaN instead removes finite precipitation.
    source = source.replace(
        "zrnew = fmaxf(zrnew, zrmin);\n"
        "            const float zrfln = fmaxf(zrnew, 0.0f);",
        "const float mpas_precip_nonnegative = (zrnew > 0.0f ? zrnew : 0.0f);\n"
        "            const float zrfln = (mpas_precip_nonnegative > zrmin ? "
        "mpas_precip_nonnegative : zrmin);")
    # cu_ntiedtke.F90:528,684-687,715-716: the strict MINSS operations
    # retain the second mass-flux cap when unordered. Original instructions
    # 0xdf93 and 0xf73e were restored and single-stepped on a NaN timestep;
    # every final pristine output byte remained unchanged.
    source = source.replace("mfub = fminf(mfub, zmfmax);",
                            "mfub = (mfub < zmfmax ? mfub : zmfmax);")
    source = source.replace("zmfub1[i] = fminf(b1, zmfmax);",
                            "zmfub1[i] = (b1 < zmfmax ? b1 : zmfmax);")
    # cu_ntiedtke.F90:319-320: native loads the timestep as the first
    # multiplication operand. Preserve this order for exceptional words.
    for name in ("pvom", "pvol"):
        source = source.replace("__fmul_rn(NTC(" + name + ", jk), delt)",
                                "__fmul_rn(delt, NTC(" + name + ", jk))")
    # cu_ntiedtke.F90:1054-1055: the strict heating ADDSS retains the
    # dissipation operand when both terms are NaNs. Finite addition has
    # the same rounding in either order; preserve the measured NaN order.
    source = source.replace(
        "NTC(ptte, jk) = __fadd_rn(NTC(ptte, jk), ztdis);",
        "NTC(ptte, jk) = __fadd_rn(ztdis, NTC(ptte, jk));")
    # Retire the engine's guarded hoist: a slab may first trigger at a
    # midlevel. Preserve the native per-level latching flag in that case.
    source = source.replace("int llo3,\n", "const int *mpas_llo3_start,\n")
    source = source.replace("if (llo3 < 0) llo3 = llo3_mask[0] & 1;", "int llo3 = 0;")
    source = source.replace("for (int jk = klevm1; jk >= 3; --jk) {",
        "for (int jk = klevm1; jk >= 3; --jk) {\n"
        "        if (jk <= mpas_llo3_start[0]) llo3 = 1;")
    # cumastrn:656-659: (thermal + vapor) - liquid. The reused kernel
    # groups vapor minus liquid before adding thermal, changing rounding.
    source = source.replace('''__fadd_rn(
                    __fdiv_rn(__fadd_rn(NTK(ptu, jk), -NTK(ptenh, jk)),
                              NTK(ptenh, jk)),
                    __fadd_rn(__fmul_rn(c.vtmpc1,
                                        __fadd_rn(NTK(pqu, jk),
                                                  -NTK(pqenh, jk))),
                              -NTK(plu, jk)))''', '''__fadd_rn(__fadd_rn(
                    __fdiv_rn(__fadd_rn(NTK(ptu, jk), -NTK(ptenh, jk)),
                              NTK(ptenh, jk)),
                    __fmul_rn(c.vtmpc1,
                              __fadd_rn(NTK(pqu, jk), -NTK(pqenh, jk)))),
                    -NTK(plu, jk))''')
    # cuflxn:3017-3025: preserve the left-associated multiply/divide
    # chains in precipitation evaporation, including the humidity limit.
    source = source.replace('''const float zbase = __fmul_rn(
                __fdiv_rn(__fsqrt_rn(__fdiv_rn(NTC(paph, jk),
                                               NTC(paph, klev + 1))),
                          5.09e-3f),
                __fdiv_rn(zrfl, NT_ZCUCOV));''', '''const float zbase = __fdiv_rn(__fmul_rn(
                __fdiv_rn(__fsqrt_rn(__fdiv_rn(NTC(paph, jk),
                                               NTC(paph, klev + 1))),
                          5.09e-3f), zrfl), NT_ZCUCOV);''')
    source = source.replace('''-__fmul_rn(__fmul_rn(NT_ZCUCOV,
                    fmaxf(0.0f, __fadd_rn(
                        __fmul_rn(rhevap, NTC(pqsen, jk)), -NTC(pqen, jk)))),
                    __fmul_rn(zcons2, zdp))''', '''-__fmul_rn(__fmul_rn(__fmul_rn(NT_ZCUCOV,
                    fmaxf(0.0f, __fadd_rn(
                        __fmul_rn(rhevap, NTC(pqsen, jk)), -NTC(pqen, jk)))),
                    zcons2), zdp)''')
    # cutypen:1300,1449-1452,1670-1673: zcbase is INTEGER in MPAS.
    # Truncate the pressure before choosing the nearest half level.
    # Treating it as real changes the chosen cloud base in regression 44.
    source = source.replace(
        "const float zcbase = __fadd_rn(NTC(paph, ik), zdp);",
        "const float zcbase = (float)mpas_nt_int(__fadd_rn(NTC(paph, ik), zdp));")
    # cutypen:1540-1555 resets every trial's scratch, including rows that
    # the new parcel does not visit. Preserve that native initialization.
    source = source.replace('''        // Inactive trials cannot update outputs. Keep the final scratch reset.
        if ((deepflag || levels < itoppacel) && levels > klev / 2 + 1)
            continue;
        // Only the final trial needs to leave every unused scratch row clear.
        if (levels == klev / 2 + 1) {''', '''        // MPAS cutypen:1540-1555: clear scratch for every parcel trial.
        {''')
    return source


@lru_cache(maxsize=None)
def _module(device: int):
    """cu_ntiedtke.F90:1-3594; independent strict CUDA compilation."""
    import cupy as cp
    from cupy.cuda import compiler
    import json
    import os
    from pathlib import Path
    with cp.cuda.Device(device):
        # RawModule's cache compiler appends FTZ after caller options.
        # Compile directly so native subnormal words survive unchanged.
        arch = cp.cuda.Device(device).compute_capability
        artifact = os.environ.get('MPAS_NTIEDTKE_ARTIFACT')
        if artifact:
            root = Path(artifact)
            metadata = json.loads((root / 'receipt.json').read_text())
            binary = (root / 'ntiedtke.cubin').read_bytes()
            expected = dict(schema='mpas-ntiedtke-nvrtc-artifact/v1', arch=arch,
                            options=list(_hex_numerics.options(CUDA_OPTIONS)),
                            cuda_sha256=sha256(cuda_source().encode()).hexdigest(),
                            binary_sha256=sha256(binary).hexdigest())
            if any(metadata.get(key) != value for key, value in expected.items()):
                # Prevent stale source, wrong precision or another card's
                # binary from silently invalidating native word identity.
                raise ValueError('New Tiedtke NVRTC artifact source/options/architecture/hash mismatch')
        else:
            binary, _ = compiler.compile_using_nvrtc(
                cuda_source(), options=_hex_numerics.options(CUDA_OPTIONS), arch=arch)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def source_receipt() -> dict[str, Any]:
    """cu_ntiedtke.F90:1-3594; identify the source this port executes."""
    from woof.core import ntiedtke
    from woof.core.kernels import module_source
    from pathlib import Path
    import json
    import os
    root = Path(ntiedtke.__file__).parent
    result = {
        "scheme": SCHEME, "backend": BACKEND,
        "mpas_scheme_sha256": MPAS_SCHEME_SHA256,
        "cuda_sha256": sha256(cuda_source().encode()).hexdigest(),
        "reused_cuda_sha256": sha256((root / "kernels/ntiedtke.cu").read_bytes()).hexdigest(),
        "reused_launcher_sha256": sha256(Path(ntiedtke.__file__).read_bytes()).hexdigest(),
        "shared_math_sha256": sha256((root / "kernels/glibc_flt32.cuh").read_bytes()).hexdigest(),
        "reused_translation_unit_sha256": sha256(module_source("ntiedtke").encode()).hexdigest(),
        "options": list(_hex_numerics.options(CUDA_OPTIONS)), "rkind_bytes": 4,
    }
    artifact = os.environ.get('MPAS_NTIEDTKE_ARTIFACT')
    if artifact:
        result['precompiled_nvrtc'] = json.loads((Path(artifact) / 'receipt.json').read_text())
    return result


def convection_from_MPAS(fields: Mapping[str, Any], *, len_disp: float) -> dict[str, Any]:
    """mpas_atmphys_driver_convection.F:790-851, bottom-up (level,cell)."""
    import cupy as cp
    exner = fields["exner"]
    if exner.dtype != cp.float32 or exner.ndim != 2 or not exner.flags.c_contiguous:
        raise ValueError("MPAS forcing kernels require contiguous float32 (level,cell) arrays")
    shape = exner.shape
    for name in ("rthdynten", "rthblten", "rthratenlw", "rthratensw", "rqvdynten", "rqvblten"):
        a = fields[name]
        if a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous:
            raise ValueError(f"{name} must match the MPAS forcing shape and single precision")
    density = fields["meshDensity"]
    if density.shape != (shape[1],) or density.dtype != cp.float32 or not density.flags.c_contiguous:
        raise ValueError("meshDensity must be contiguous float32 with one value per cell")
    output = {n: fields[n] for n in LEVEL_INPUTS + INTERFACE_INPUTS + SURFACE_INPUTS
              if n not in ("qvften", "thften", "dx")}
    output["thften"] = cp.empty_like(exner)
    output["qvften"] = cp.empty_like(exner)
    output["dx"] = cp.empty(shape[1], dtype=cp.float32)
    module = _module(cp.cuda.Device().id)
    size = exner.size
    module.get_function("mpas_ntiedtke_forcing")(((size + 127) // 128,), (128,),
        tuple(fields[n] for n in ("rthdynten", "rthblten", "rthratenlw", "rthratensw", "exner", "rqvdynten", "rqvblten"))
        + (output["thften"], output["qvften"], np.int32(size)))
    ncol = shape[1]
    module.get_function("mpas_ntiedtke_dx")(((ncol + 127) // 128,), (128,),
        (density, np.float32(len_disp), output["dx"], np.int32(ncol)))
    return output


def convection_to_MPAS(output: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    """mpas_atmphys_driver_convection.F:891-917,971-983; overwrite rates."""
    for name in LEVEL_OUTPUTS + ("raincv",):
        target[name][...] = output[name]
    target["cuprec"][...] = output["pratec"]


def _pipeline(ncol: int, nz: int, dt: float, stepcu: int, initflag: int, constants: Constants):
    """cu_ntiedtke.F90:148-332,342-1085; full scheme call sequence.

    cu_ntiedtke_pre.F:105-182 and cu_ntiedtke_post.F:88-115 are identical
    arithmetic to the engine's prep/post kernels. Custom stages compile
    in this module and propagate the separate sublimation constant.
    """
    from woof.core.ntiedtke import NtPipeline, NtStages, NT_STAGE_SIGNATURE

    class MpasStages(NtStages):
        def launch(self, kernel_name, args):
            """cu_ntiedtke.F90:148-332; dispatch the native module sequence."""
            fn = self._functions.get(kernel_name)
            if fn is None:
                fn = _module(self._cp.cuda.Device().id).get_function(kernel_name)
                self._functions[kernel_name] = fn
            fn(self.geometry.grid, self.geometry.block, tuple(args) + self._tail())
            return self

    class MpasPipeline(NtPipeline):
        def args_for(self, stage):
            """cu_ntiedtke.F90:148-332; bind native arguments and timestep."""
            args = list(super().args_for(stage))
            signature = NT_STAGE_SIGNATURE[stage]
            if "xlv" in signature:
                args.insert(signature.index("xlv") + 1, np.float32(constants.xls))
            if stage == "ntiedtke_cuascn":
                args[signature.index("llo3")] = self._ascent_start
            if stage in ("ntiedtke_mfub", "ntiedtke_closure"):
                # woof 2.9.9 calls this argument ztmst; newer launchers use
                # dt. Both kernels receive the MPAS convection timestep.
                timestep = next((name for name in ("dt", "ztmst") if name in signature), None)
                if timestep is None:
                    raise ValueError(f"{stage}: engine signature lacks timestep argument dt or ztmst")
                args[signature.index(timestep)] = self.scalars["ztmst"]
            return tuple(args)

        def reduce_llo3(self):
            """cu_ntiedtke.F90:1903-2012; preserve the native slab ascent latch."""
            import cupy as cp
            self._ascent_start.fill(0)
            names = ("ldcum", "ktype", "klab", "pqen", "pqsen", "pgeo")
            _module(cp.cuda.Device().id).get_function("mpas_ntiedtke_ascent_start")(
                self.geometry.grid, self.geometry.block,
                tuple(self.w.bind(n, 1) for n in names) +
                (self._ascent_start, np.int32(ncol), np.int32(nz), np.float32(constants.grav)))
            return None

        def run_chunk(self):
            """cu_ntiedtke.F90:148-332; execute the complete module call order."""
            # Use the complete original call order; the engine's fused
            # translation unit would bypass this strict compilation.
            from woof.core.ntiedtke import _nt_walk
            _nt_walk(self)

    p = MpasPipeline(ncol=ncol, nz=nz, dt=dt, stepcu=stepcu, itimestep=initflag,
        cp=constants.cp, rd=constants.rd, rv=constants.rv, xlv=constants.xlv,
        xlf=constants.xlf, grav=constants.grav)
    # cu_ntiedtke_pre.F:105 rounds dt to RKIND before multiplication.
    delt = np.float32(np.float32(dt) * np.float32(stepcu))
    p.scalars["delt"] = delt
    p.scalars["ztmst"] = delt
    p.stages = MpasStages(p.geometry)
    p._ascent_start = p.stages._cp.zeros(1, dtype=p.stages._cp.int32)
    return p


class NewTiedtkeV842:
    """module_cu_ntiedtke.F:20-289; cu_ntiedtke.F90:1-3594, whole module."""
    def __init__(self, *, constants: Constants = Constants()):
        """module_cu_ntiedtke.F:20-289; retain native module constants."""
        self.constants = constants
        # One pipeline slot per (card, ncol, nz). The slot launcher hands one instance to every same-host rank, so
        # a single slot is rebuilt on every call once two cards alternate, and a rank can pick up another card's
        # pipeline between the key check and its use.
        self._cached = {}

    def __call__(self, fields: Mapping[str, Any], *, dt: float, itimestep: int,
                 restart: bool = False, stepcu: int = 1, f_qc: bool = True,
                 f_qi: bool = True, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """mpas_atmphys_driver_convection.F:476-478,580-608.

        This entry accepts packed driver fields, including already folded
        qvften/thften and per-cell dx. itimestep is the MPAS model counter;
        the native wrapper's initflag is derived here. False tracer flags
        carry the existing condensate tendency as module_cu_ntiedtke does.
        """
        import cupy as cp
        shape = fields["t3d"].shape
        if len(shape) != 2 or shape[0] < 4 or shape[1] < 1:
            raise ValueError("New Tiedtke requires at least four levels and one cell")
        nz, ncol = shape
        for name in LEVEL_INPUTS + INTERFACE_INPUTS + SURFACE_INPUTS:
            a = fields[name]
            expected = (nz + 1, ncol) if name in INTERFACE_INPUTS else ((ncol,) if name in SURFACE_INPUTS else shape)
            if not isinstance(a, cp.ndarray) or a.shape != expected or a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise ValueError(f"{name}: kernels require contiguous float32 CUDA arrays with shape {expected}")
        if isinstance(stepcu, bool) or int(stepcu) != stepcu or stepcu < 1:
            raise ValueError("integer stepcu must be positive, as in the MPAS convection controller")
        # The MPAS convection controller passes this derived initflag to
        # the native wrapper, rather than its model timestep counter.
        initflag = 0 if restart or itimestep > 1 else 1
        # cu_ntiedtke_pre.F:105: dt is an RKIND word. Python equality
        # conflates signed zeros, but they produce different native words.
        # Cache the actual argument word, including its NaN sign/payload.
        dt_word = int(np.float32(dt).view(np.uint32))
        key = (cp.cuda.Device().id, ncol, nz, dt_word, stepcu, initflag)
        cached = self._cached.get(key[:3])
        if cached is None or cached[0] != key:
            cached = (key, _pipeline(ncol, nz, dt, stepcu, initflag, self.constants))
            self._cached[key[:3]] = cached
        p = cached[1]
        p.stages._llo3_mask.fill(0)
        p.stages._geom_report.fill(-1)
        p.stages._order_report.fill(-1)
        p.stages._ticket.fill(0)
        for name in LEVEL_INPUTS:
            p.w.bind(name, 0)[:nz, :] = fields[name]
        for name in INTERFACE_INPUTS:
            p.w.bind(name, 0)[:nz + 1, :] = fields[name]
        for name in SURFACE_INPUTS:
            p.w.bind(name, 1)[:] = fields[name]
        p.run_chunk()
        output = {n: p.w.bind(n, 0)[:nz, :].copy() for n in LEVEL_OUTPUTS}
        output.update({n: p.w.bind(n, 1).copy() for n in SURFACE_OUTPUTS})
        output["cu_act_flag"] = cp.ones(ncol, dtype=cp.bool_)
        for enabled, name in ((f_qc, "rqccuten"), (f_qi, "rqicuten")):
            if not enabled:
                if previous is None or name not in previous:
                    raise ValueError(f"{name} must be supplied to preserve the disabled tracer tendency")
                output[name] = previous[name].copy()
        return output


def ntiedtkeinit(target: Mapping[str, Any], *, restart: bool, p_qc: int,
                 p_qi: int, p_first_scalar: int) -> None:
    """module_cu_ntiedtke.F:292-351; preserve every restart value."""
    if restart:
        return
    for name in ("rthcuten", "rqvcuten", "rucuten", "rvcuten", "rthften", "rqvften"):
        target[name].fill(0)
    for index, name in ((p_qc, "rqccuten"), (p_qi, "rqicuten")):
        if index >= p_first_scalar:
            target[name].fill(0)


def ntiedtkefinalize() -> tuple[str, int]:
    """cu_ntiedtke.F90:129-141; native finalization changes no fields."""
    return "cu_ntiedtke_finalize OK", 0


@dataclass(frozen=True)
class DriverBinding:
    """mpas_atmphys_driver_convection.F:580-608,790-851,891-983.

    Common seam packing supplies solve extents and species mappings. Fields
    carry the prepared state plus named dynamics/PBL/radiation forcing and
    meshDensity. Target contains six tendencies, raincv and cuprec. The
    dynamical dt and native n_cu must remain separate from SlotCall's dt_cu.
    """
    fields: Mapping[str, Any]
    target: Mapping[str, Any]
    dt_dyn: float
    stepcu: int
    len_disp: float
    restart: bool = False
    f_qc: bool = True
    f_qi: bool = True


def bind_native_pools(provider, call):
    """Bind the complete native driver boundary to f4's solve-cell views.

    MPAS mpas_atmphys_driver_convection.F:476-478,580-608,790-851,891-983;
    mpas_atmphys_vars.F:179-249,303-306.
    The common provider owns packing, halos and native prepared top entries.
    No field arithmetic or timestep rounding occurs in this binder.
    """
    if call.slot != "convection" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("New Tiedtke binder requires its registered convection slot")
    nz = provider.session.layout.dimensions["nVertLevels"]
    prepared = provider["prepared"]
    native_names = dict(t3d="t_p", qv3d="qv_p", qc3d="qc_p", qi3d="qi_p",
                        u3d="u_p", v3d="v_p", pcps="pres_hyd_p", dz8w="dz_p",
                        rho3d="rho_p", exner="pi_p")
    fields = {name: prepared[native][:nz] for name, native in native_names.items()}
    fields.update(p8w=prepared["pres2_hyd_p"][:nz+1], w=prepared["w_p"][:nz+1])
    fields["xland"] = provider["sfc_input"]["xland"]
    fields.update({name: provider["diag_physics"][name] for name in ("hfx", "qfx")})
    fields["meshDensity"] = provider["mesh"]["meshDensity"]
    tendencies = provider["tend_physics"]
    fields.update({name: tendencies[name] for name in
                   ("rthdynten", "rthblten", "rthratenlw", "rthratensw",
                    "rqvdynten", "rqvblten")})
    target = {name: tendencies[name] for name in LEVEL_OUTPUTS}
    target.update({name: provider["diag_physics"][name] for name in ("raincv", "cuprec")})
    target["cu_act_flag"] = prepared["cu_act_flag"]
    scalars = provider["scalar_state"]
    binding = DriverBinding(fields=fields, target=target, dt_dyn=scalars["dt_dyn"],
                            stepcu=int(scalars["n_cu"]),
                            len_disp=provider["configs"]["config_len_disp"],
                            restart=bool(provider["configs"]["config_do_restart"]),
                            f_qc=bool(scalars["f_qc"]), f_qi=bool(scalars["f_qi"]))
    aliases = dict(rd="r_d", rv="r_v", grav="gravity")
    constants = Constants(**{name: provider["constants"][aliases.get(name, name)]
                            for name in Constants.__dataclass_fields__})
    return {"ntiedtke_v842": dict(binding=binding, constants=constants)}


@lru_cache(maxsize=4)
def _slot_launcher(constants: Constants):
    """cu_ntiedtke.F90:148-332; cache temporary workspace, not model state."""
    return NewTiedtkeV842(constants=constants)


def _execute_binding(call, binding: DriverBinding, port: NewTiedtkeV842):
    """mpas_atmphys_driver_convection.F:476-478,580-608,790-851,891-983."""
    fields = convection_from_MPAS(binding.fields, len_disp=binding.len_disp)
    result = port(fields, dt=binding.dt_dyn, stepcu=binding.stepcu,
                  itimestep=call.itimestep, restart=binding.restart,
                  f_qc=binding.f_qc, f_qi=binding.f_qi, previous=binding.target)
    convection_to_MPAS(result, binding.target)
    if "cu_act_flag" in binding.target:
        binding.target["cu_act_flag"][...] = result["cu_act_flag"]
    return result


def run_slot(call):
    """mpas_atmphys_driver_convection.F:580-608,790-983; complete callback."""
    if call.slot != "convection" or call.scheme != SCHEME or str(call.backend) != BACKEND:
        raise ValueError("New Tiedtke callback requires its registered convection slot")
    bound = call.pools["ntiedtke_v842"]
    return _execute_binding(call, bound["binding"], _slot_launcher(bound["constants"]))


def register_backend(registry, bind, *, constants: Constants = Constants()):
    """mpas_atmphys_driver_convection.F:580-608; f4 SlotRegistry callback.

    bind(call) is the framework adapter's packing operation. The cache holds
    temporary workspace only; all MPAS persistent arrays stay in call.pools.
    """
    port = NewTiedtkeV842(constants=constants)

    def launch(call):
        """mpas_atmphys_driver_convection.F:580-608; bound complete callback."""
        b = bind(call)
        return _execute_binding(call, b, port)

    registry.register("convection", SCHEME, BACKEND, launch)
    return launch
