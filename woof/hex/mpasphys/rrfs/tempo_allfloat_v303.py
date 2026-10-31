"""TEMPO fast numerics with no binary64 anywhere in the kernel (rrfs/tempo-yolo-allfloat).

Applied by ``tempo_v303`` in FAST numerics only, over ``tempo_fastdp_v303.fast_dp_source`` (TEMPO's own
real(dp) arithmetic already binary32) and in place of ``tempo_fastmath_v303.fast_source``. Strict numerics
never sees this module: the strict column is ``tempo_v303._COLUMN`` byte for byte.

What changes in fast mode, on top of the binary32 arithmetic:

* the table blob is binary32 (``TempoBlob`` members ``float``): the binary64 blob is converted once, flushing
  binary32 subnormals to zero, which is the value every fast read of it already produced
  (``cvt.rn.ftz.f32.f64``); table reads go through ``__ldg`` with 32-bit index arithmetic;
* the scalar parameters (every TEMPO_FIELDS entry of extent 1) sit in ``__constant__ TempoScalarsF tempo_k``,
  written once per launch, so a parameter read is a constant-bank operand, not a load and a 1/64-rate
  binary64 -> binary32 conversion;
* the maths shim is CUDA's approximate binary32 intrinsics: ``__expf``, ``__logf``, ``__log10f``,
  ``pow(x, y) = exp2f(y * __log2f(x))`` for positive bases (zero, negative and NaN bases go to binary32
  ``powf`` in one out-of-line copy), ``tgammaf``; a shared pow base keeps its ``__log2f`` once;
* the column body's direct glibc replicas (``mpas_cosf``, ``mpas_sin``, ``mpas_cos``, ``mpas_hypot``,
  ``mpas_atan2``) become CUDA binary32 functions, and so do the MPAS slice kernels' (``tempo_to``'s exner
  ``mpas_powf``, the aerosol initialisation's ``mpas_cosf``/``mpas_logf``/``mpas_expf``).

THE BREAKAGE THIS ENDS: after the binary32 arithmetic (tempo3km-a1) and the binary32 maths (tempo3km-a2), the
3 km TEMPO slot still ran at the binary64 pipe's pace (a1 + a2 together saved no more than either alone):
the remaining binary64 was the blob reads (1,829 static F2F.F32.F64), the maths helpers' binary64 range
checks and glibc fallbacks, and the direct replicas. Fast words change by design; the fast referee is the
forecast (maps, rain and echo-area totals), not words.
"""
from __future__ import annotations

import re

from woof.hex.mpasphys.rrfs.tempo_fastdp_v303 import _POW_F as _FASTDP_POW_F

_BLOB_OLD = "#define TEMPO_TX_MEMBER(name, d1, d2, d3, d4, d5) double name[(d1) * (d2) * (d3) * (d4) * (d5)];"
_BLOB_NEW = "#define TEMPO_TX_MEMBER(name, d1, d2, d3, d4, d5) float name[(d1) * (d2) * (d3) * (d4) * (d5)];"
_BLOB_END = "  TEMPO_FIELDS(TEMPO_TX_MEMBER)\n#undef TEMPO_TX_MEMBER\n};\n"
_FIELDS = re.compile(r"TX\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)")

_ACCESSORS_OLD = r"""#define PS(name) ((TK_##name)P.L->name[0])
#define PI(name) ((int)(P.L->name[0]))
#define PA1(name, i) ((TK_##name)P.L->name[(long long)(i) - 1])
#define PA2(name, i, j) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * ((long long)(j) - 1)])
#define PA3(name, i, j, k) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * ((long long)(k) - 1))])
#define PA4(name, i, j, k, l) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * ((long long)(l) - 1)))])
#define PA5(name, i, j, k, l, m) ((TK_##name)P.L->name[((long long)(i) - 1) + TD1_##name * (((long long)(j) - 1) + \
    TD2_##name * (((long long)(k) - 1) + TD3_##name * (((long long)(l) - 1) + \
    TD4_##name * ((long long)(m) - 1))))])"""

# Fast (all binary32): scalars from the constant bank, tables through the read-only path, 32-bit indices
# (the largest table, 37*37*9*37*37 words, is far below 2^31).
_ACCESSORS_NEW = r"""#define PS(name) ((TK_##name)tempo_k.name)
#define PI(name) ((int)tempo_k.name)
#define TEMPO_TI(i) ((int)(i) - 1)
#define TEMPO_TD(d, name) ((int)d##_##name)
#define PA1(name, i) ((TK_##name)__ldg(&P.L->name[TEMPO_TI(i)]))
#define PA2(name, i, j) ((TK_##name)__ldg(&P.L->name[TEMPO_TI(i) + TEMPO_TD(TD1, name) * TEMPO_TI(j)]))
#define PA3(name, i, j, k) ((TK_##name)__ldg(&P.L->name[TEMPO_TI(i) + TEMPO_TD(TD1, name) * (TEMPO_TI(j) + \
    TEMPO_TD(TD2, name) * TEMPO_TI(k))]))
#define PA4(name, i, j, k, l) ((TK_##name)__ldg(&P.L->name[TEMPO_TI(i) + TEMPO_TD(TD1, name) * (TEMPO_TI(j) + \
    TEMPO_TD(TD2, name) * (TEMPO_TI(k) + TEMPO_TD(TD3, name) * TEMPO_TI(l)))]))
#define PA5(name, i, j, k, l, m) ((TK_##name)__ldg(&P.L->name[TEMPO_TI(i) + TEMPO_TD(TD1, name) * (TEMPO_TI(j) + \
    TEMPO_TD(TD2, name) * (TEMPO_TI(k) + TEMPO_TD(TD3, name) * (TEMPO_TI(l) + \
    TEMPO_TD(TD4, name) * TEMPO_TI(m))))]))"""

_TNINT_DOUBLE = "TEMPO_HD int tnint(double x) {return (int)lround(x);}\n"
_SHIM_START = "TEMPO_HD float tempo_exp(float x) {return mpas_expf(x);}\n"
_SHIM_END_MARK = "TEMPO_HD double tempo_pow_with(const TempoPowBase& b, double y) {return mpas_pow(b.x, y);}\n#endif\n"

_SHIM_NEW = r"""// Fast numerics, all binary32 (tempo_allfloat_v303): CUDA's approximate binary32 intrinsics.
// Positive bases: pow = exp2(y log2 x).  Zero, negative and NaN bases: binary32 powf, one out-of-line copy.
__device__ __noinline__ float tempo_pow_slow(float x, float y) {return powf(x, y);}
TEMPO_HD float tempo_exp(float x) {return __expf(x);}
TEMPO_HD float tempo_log(float x) {return __logf(x);}
TEMPO_HD float tempo_log10(float x) {return __log10f(x);}
TEMPO_HD float tempo_pow(float x,float y) {return x > 0.0f ? exp2f(y * __log2f(x)) : tempo_pow_slow(x, y);}
template<class A,class B> TEMPO_HD auto tempo_pow(A x,B y)->decltype(x+y) {using T=decltype(x+y);return tempo_pow(T(x),T(y));}
TEMPO_HD float tempo_gamma(float x) {return tgammaf(x);}
"""

#: A shared pow base keeps its log2 (snow moments: 10^loga for every moment, smo2^b for several b).
_POW_F_NEW = r"""
// Fast numerics, all binary32 (tempo_allfloat_v303): pow of one base, several exponents, log2 taken once.
struct TempoPowBaseF { float x, lg; bool pos; };
TEMPO_HD TempoPowBaseF tempo_pow_base_f(float x) {
  TempoPowBaseF b; b.x = x; b.pos = x > 0.0f; b.lg = b.pos ? __log2f(x) : 0.0f; return b;
}
TEMPO_HD float tempo_pow_with_f(const TempoPowBaseF& b, float y) {
  return b.pos ? exp2f(y * b.lg) : tempo_pow_slow(b.x, y);
}
"""

#: The column body's direct glibc replicas, and how many of each the pinned body holds.
_BODY_CALLS = (("mpas_cosf(", "__cosf(", 2), ("mpas_hypot(", "hypotf(", 3), ("mpas_atan2(", "atan2f(", 1),
               ("mpas_sin(", "__sinf(", 2), ("mpas_cos(", "__cosf(", 1))
#: The MPAS slice kernels' glibc replicas (tempo_to's exner pow, tempo_init_aerosols).
_MPAS_CALLS = (("mpas_powf(", "tempo_pow(", 1), ("mpas_cosf(", "__cosf(", 1), ("mpas_logf(", "__logf(", 2),
               ("mpas_expf(", "__expf(", 2))


def fields(source: str):
    """The blob fields in layout order, [(name, words)], read from the column's TEMPO_FIELDS table."""
    start = source.index("#define TEMPO_FIELDS(TX)")
    end = source.index("struct TempoBlob {", start)
    out = []
    for match in _FIELDS.finditer(source[start:end]):
        words = 1
        for d in match.groups()[1:]:
            words *= int(d)
        out.append((match.group(1), words))
    return out


def scalar_names(source: str):
    """The extent-1 fields, in layout order: the members of the constant-bank parameter block."""
    return [name for name, words in fields(source) if words == 1]


def _replace_counted(text: str, old: str, new: str, count: int, what: str) -> str:
    found = len(re.findall(re.escape(old), text))
    if found != count:
        raise ValueError(f"TEMPO all-float: expected {count} {what} '{old}', found {found}")
    return text.replace(old, new)


def all_float_source(source: str) -> str:
    """The fast column support with no binary64 type: apply to ``fast_dp_source(_COLUMN)``.

    Fails on source drift (a replaced block missing or duplicated) and when applied twice.
    """
    if source.count(_BLOB_OLD) != 1:
        raise ValueError(f"TEMPO all-float: expected one binary64 blob member macro, found {source.count(_BLOB_OLD)}")
    if source.count(_ACCESSORS_OLD) != 1:
        raise ValueError("TEMPO all-float: expected one copy of the blob accessor macros")
    if source.count(_SHIM_START) != 1 or source.count(_SHIM_END_MARK) != 1:
        raise ValueError("TEMPO all-float: expected one copy of the maths shim")
    if source.count(_FASTDP_POW_F) != 1:
        raise ValueError("TEMPO all-float: apply over tempo_fastdp_v303.fast_dp_source (its binary32 pow base)")
    if source.count(_TNINT_DOUBLE) != 1:
        raise ValueError("TEMPO all-float: expected one binary64 tnint")
    names = scalar_names(source)
    struct = ("\n// Fast numerics (tempo_allfloat_v303): the extent-1 parameters, binary32, in the constant bank;\n"
              "// written by the host once per launch from the binary32 blob.\n"
              "struct TempoScalarsF {\n" + "".join(f"  float {n};\n" for n in names) + "};\n"
              "__constant__ TempoScalarsF tempo_k;\n")
    source = source.replace(_BLOB_OLD, _BLOB_NEW)
    if source.count(_BLOB_END) != 1:
        raise ValueError("TEMPO all-float: expected one TempoBlob struct end")
    source = source.replace(_BLOB_END, _BLOB_END + struct)
    source = source.replace(_ACCESSORS_OLD, _ACCESSORS_NEW)
    source = source.replace(_TNINT_DOUBLE, "")
    start = source.index(_SHIM_START)
    end = source.index(_SHIM_END_MARK) + len(_SHIM_END_MARK)
    source = source[:start] + _SHIM_NEW + source[end:]
    source = source.replace(_FASTDP_POW_F, _POW_F_NEW)
    for old, new, count in _BODY_CALLS:
        source = _replace_counted(source, old, new, count, "column call")
    return source


def all_float_mpas(mpas: str) -> str:
    """The MPAS slice kernels for the all-float fast module: their glibc replicas become CUDA binary32."""
    for old, new, count in _MPAS_CALLS:
        mpas = _replace_counted(mpas, old, new, count, "MPAS kernel call")
    return mpas
