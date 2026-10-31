# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
# For the two libm grants the text also sits beside the code, in
# woof/core/kernels/LICENSE-third-party.txt.
#
#   Arm optimized-routines -- the logf, expf, exp2f and powf cores and
#   their data tables github.com/ARM-software/optimized-routines:
#   math/logf.c, math/expf.c, math/exp2f.c, math/powf.c and the matching
#   math/*_data.c, published August 2017 and imported into glibc for
#   2.27/2.28 by their own author; transcribed here from glibc 2.39
#   sysdeps/ieee754/flt-32/.
#
#       Copyright (c) 2017-2018, Arm Limited.
#       SPDX-License-Identifier: MIT
#
#   Taken under the MIT branch of Arm's grant.  MIT requires the copyright
#   notice above and its permission notice to travel with every copy; the
#   permission notice is reproduced in full in the files named above.
#
#   FDLIBM -- expm1f, tanhf, atanf, log10f and log1pf. Developed at SunPro and
#   converted to single precision at Cygnus Support; glibc carries it
#   substantially unmodified and presents it, in its own LICENSES file, as
#   Sun's code.  The notice below is the whole of the licence: its one
#   condition is that it be preserved.
#
#       Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#
#       Developed at SunPro, a Sun Microsystems, Inc. business.
#       Permission to use, copy, modify, and distribute this
#       software is freely granted, provided that this notice
#       is preserved.
# ======================================================================
"""Bit-faithful reproductions of the glibc 2.39 FP32 libm calls WRF makes.

Not Noah-MP-specific despite the module name (the Noah-MP lane owns the
``noahmp_*`` namespace; promote this to a shared module when another lane needs
it).

Why this exists
---------------
The oracle is gfortran 13.3.0 linked against glibc 2.39, so ``EXP``, ``LOG``,
``LOG10`` and ``**`` on ``REAL(4)`` become calls to glibc's ``expf``, ``logf``,
``log10f`` and ``powf``.  **None of those four is correctly rounded.**  Rounding
the FP64 result once -- the shim the MYNN lane uses, and the obvious thing to
do -- disagrees with glibc on a large fraction of the domain:

| call      | domain swept                    | disagreement with FP64-then-round |
|-----------|---------------------------------|----------------------------------:|
| ``log10f``| every FP32 in (0.1, 1.0]        |            5,266,896 / 28,521,268 = **18.47%** |
| ``expf``  | FP32 grid over \\|x\\| <= 25      |               21,750 / 34,902,602 = 0.062% |
| ``powf``  | 4e6 samples over Noah-MP shapes |                       2,618 / 4e6 = 0.065% |

So TDFCND's ``LOG10(SATRATIO)`` cannot reach ``max_ulp 0`` through an FP64
shim: roughly one soil column in five lands on the wrong side of the rounding
boundary.  That is not a tolerance to loosen, it is the wrong function.

What this is
------------
Direct transcriptions of the glibc 2.39 implementations:

* ``logf``   -- ``sysdeps/ieee754/flt-32/e_logf.c`` + ``e_logf_data.c``
  (ARM optimized-routines: 16-entry table, cubic in ``r``, FP64 intermediate,
  one rounding to FP32 at the end).
* ``log10f`` -- ``sysdeps/ieee754/flt-32/e_log10f.c``, which is still the 1993
  SunPro reduction ``log10(x) = k*log10_2 + log10(e)*logf(z)`` evaluated in
  **FP32** on top of ``logf``.  This is why ``log10f`` is so much less accurate
  than ``logf``.
* ``expf``   -- ``sysdeps/ieee754/flt-32/e_expf.c`` + ``e_exp2f_data.c``
  (32-entry ``2^(i/32)`` table, the shift-trick round-to-int, cubic in ``r``).
* ``powf``   -- ``sysdeps/ieee754/flt-32/e_powf.c`` + ``e_powf_log2_data.c``
  (a 16-entry ``log2`` table feeding the same ``exp2`` core).

``TOINT_INTRINSICS`` is 0 on x86-64, so ``POWF_SCALE`` is 1.0 and the shift
trick is used in both ``exp2`` paths; both are reproduced as glibc compiles
them for this target.

Verification (all against the live glibc 2.39 on the oracle host, C
transcription compiled with ``-ffp-contract=off`` and again with
``-mfma -ffp-contract=fast``, identical results):

* ``logf``  : 222,414,918 FP32 inputs swept over (0.1,1], [1,100], [1e-6,0.1]
  -- **0 mismatches**.
* ``log10f``: the same 222,414,918 inputs -- **0 mismatches**.
* ``powf``  : 6,000,000 samples over the Noah-MP bases (7.7, 2.0, 2.2, 0.57)
  and broad random ``(base, exponent)`` -- **0 mismatches**.
* ``expf``  : 146,800,642 FP32 inputs over \\|x\\| <= 80 -- **2 mismatches**,
  both 1 ULP: ``x = 0x4202422F`` (32.5646324) and ``x = 0xC27C65D9``
  (-63.0994606).  Residual rate 1.4e-8, a 4,400x improvement on the FP64
  shim.  Diagnosis: glibc's x86-64 multiarch ``expf`` is selected at run time
  and is not byte-identical to the generic C at those two points; enabling FMA
  contraction in the transcription does not move either, so it is not a
  contraction artefact.  Any Noah-MP column that lands on one of those two
  exact FP32 arguments will miss ``max_ulp 0`` by 1 ULP and must be reported,
  not absorbed.

The CUDA halves of these functions live in
``woof/core/kernels/noahmp_leaves.cu`` and must stay in step with this file.

Two independent transcriptions, and what they measured about each other
-----------------------------------------------------------------------
The five Noah-MP leaf lanes (vegprecip, radiation, bareflux, vegeflux, snow)
arrived carrying a second transcription of ``logf``/``expf``/``powf``/``atanf``
and a third of ``logf``/``atanf``, written independently from the same glibc
2.39 sources.  They differed in one interesting way: the second transcription
put an explicit :func:`math.fma` at every ``a*b + c`` site in ``logf``,
``expf`` and ``powf``'s ``log2``, on the argument that glibc ifunc-selects the
``-mfma`` rebuild (``__logf_fma``, ``__powf_fma``) on an FMA host, and that
``logc + k*Ln2`` in particular is a single FMA in the disassembly.  This file
contracts nothing.

Both were run against each other over 10,500,000 argument pairs and both
against the 30,000 pinned glibc words in
``woof/data/noahmp/oracle/glibc-libm-fp32.csv`` and the 4,256 in
``glibc-atanf-fp32.csv``: **zero disagreements, on every function, in every
sweep**.  The FMA sites are real in the disassembly and they do change the
binary64 intermediate, but not by enough to move the single rounding to
binary32 at the end on any argument either lane could find.  That is a
measured null result, not a proof; it is recorded here because it is the only
independent check these transcriptions will ever get.  This file keeps the
non-FMA form for a second reason as well: :func:`math.fma` is CPython 3.13+,
and the GPU host this project validates on runs 3.12.

Only this file survived the merge.  ``sqrtf`` and :func:`f32` came across from
the leaf lanes; ``log10f`` exists only here.
"""

from __future__ import annotations

import math
import struct

import numpy as np

__all__ = [
    "GLIBC_VERSION",
    "f32",
    "logf",
    "logf_array",
    "log10f",
    "expf",
    "expf_array",
    "powf",
    "powf_array",
    "atanf",
    "atanf_correctly_rounded",
    "sqrtf",
    "expm1f",
    "tanhf",
    "log1pf",
    "log1pf_array",
]

#: The glibc these transcriptions reproduce.  Gates that pin a libm answer
#: assert against this so a host upgrade cannot silently move the reference.
GLIBC_VERSION = "2.39"

F = np.float32
_H = float.fromhex

_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


def _asuint(value) -> int:
    return struct.unpack("<I", struct.pack("<f", F(value)))[0]


def _asfloat(bits: int) -> np.float32:
    return F(struct.unpack("<f", struct.pack("<I", bits & _U32))[0])


def _asuint64(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _asdouble(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & _U64))[0]


def _as_int32(bits: int) -> int:
    bits &= _U32
    return bits - 0x100000000 if bits & 0x80000000 else bits


#: ``struct.Struct("<f")``, built once.  ``struct.pack("<f", x)`` re-resolves
#: the module global and looks the format up in struct's internal cache on
#: every call; binding the compiled struct's methods below skips both.  This
#: is the hottest function in the Noah-MP host column -- 5,888 calls per land
#: column, 282,634 per 48-column LSM step -- so the two lookups are worth
#: removing.
_F32_STRUCT = struct.Struct("<f")

#: The two halves of :func:`f32`, published so a caller in an inner loop can
#: inline the rounding instead of paying a Python call frame for it.  The only
#: caller that does is :class:`woof.core.noahmp_vegeflux.R4`, whose operators
#: are the hottest code in the host column; ``_unpack(_pack(x))[0]`` there is
#: character for character what ``f32`` evaluates, so inlining cannot move an
#: answer.  Do not reach for these anywhere a function call is not measurably
#: the cost.
F32_PACK = _F32_STRUCT.pack
F32_UNPACK = _F32_STRUCT.unpack


def f32(x, _pack=_F32_STRUCT.pack, _unpack=_F32_STRUCT.unpack) -> float:
    """Round to float32 and hand it back as a Python float.

    Round-to-nearest-even, which is what a C ``(float)`` cast and a Fortran
    ``REAL(4)`` assignment both do.  The leaf lanes use this rather than
    ``np.float32`` to keep every intermediate a Python float: a chain of
    Python-float operations rounded once per Fortran statement reproduces
    float32 arithmetic exactly, because binary32 -> binary64 -> binary32 is an
    innocuous double rounding for ``+ - * /`` (53 >= 2*24 + 2).

    An argument outside the float32 range raises ``OverflowError`` rather than
    saturating to an infinity.  That is a guard, not a hazard: a Fortran
    ``REAL(4)`` overflow is a physics bug, and none of the leaves reach one.

    ``_pack``/``_unpack`` are default arguments, not globals, so the body does
    no name lookup outside the frame.  Both are bound methods of the module
    ``Struct`` above; a caller that passes them explicitly gets the same
    function, and a caller that rebinds them is out of contract.

    Four alternative carriers were measured on the reference box against a
    21-value probe set (both zeros, the smallest subnormal, the largest finite
    binary32, the first value that overflows it, both infinities and a NaN).
    Two of them are faster and none of them is this function:

    ======================  ========  =========================================
    carrier                 ns/call   how it differs
    ======================  ========  =========================================
    ``struct`` (this one)     229.8   --
    ``ctypes.c_float``        133.7   saturates an overflow to an infinity
    ``array('f')``            162.2   saturates an overflow to an infinity
    ``array('f')`` + guard    194.2   identical, but reads and writes one
                                      module-level buffer, so two threads in
                                      ``f32`` can interleave and return each
                                      other's value
    ``float(np.float32(x))``  481.4   saturates, and warns while doing it
    ======================  ========  =========================================

    The guarded ``array`` variant is the only bit-identical one, and 15% is
    not worth a carrier that is silently wrong under concurrency.  What was
    taken instead is :data:`F32_PACK`/:data:`F32_UNPACK`: the same two calls,
    inlined into the one caller hot enough to notice the frame.
    """
    return _unpack(_pack(x))[0]


def sqrtf(x) -> np.float32:
    """``SQRT`` on ``REAL(4)``, which is not a libm call at all.

    gfortran emits ``sqrtss`` inline, and the hardware square root is
    correctly rounded, so unlike its neighbours in this file this one needs no
    transcription -- computing in binary64 and rounding once is the same
    function (again, 53 >= 2*24 + 2).  It lives here so that a leaf reaching
    for a float32 elementary function has exactly one place to look.
    """
    value = F(x)
    if value != value:                                  # NaN, payload intact
        return value
    if value < F(0.0):
        return F(np.nan)
    return F(math.sqrt(float(value)))


# ---------------------------------------------------------------------------
# e_logf_data.c -- 16-entry table, ln2, and the three polynomial coefficients.
# ---------------------------------------------------------------------------
_LOGF_TAB = (
    (_H("0x1.661ec79f8f3bep+0"), _H("-0x1.57bf7808caadep-2")),
    (_H("0x1.571ed4aaf883dp+0"), _H("-0x1.2bef0a7c06ddbp-2")),
    (_H("0x1.49539f0f010bp+0"), _H("-0x1.01eae7f513a67p-2")),
    (_H("0x1.3c995b0b80385p+0"), _H("-0x1.b31d8a68224e9p-3")),
    (_H("0x1.30d190c8864a5p+0"), _H("-0x1.6574f0ac07758p-3")),
    (_H("0x1.25e227b0b8eap+0"), _H("-0x1.1aa2bc79c81p-3")),
    (_H("0x1.1bb4a4a1a343fp+0"), _H("-0x1.a4e76ce8c0e5ep-4")),
    (_H("0x1.12358f08ae5bap+0"), _H("-0x1.1973c5a611cccp-4")),
    (_H("0x1.0953f419900a7p+0"), _H("-0x1.252f438e10c1ep-5")),
    (_H("0x1p+0"), _H("0x0p+0")),
    (_H("0x1.e608cfd9a47acp-1"), _H("0x1.aa5aa5df25984p-5")),
    (_H("0x1.ca4b31f026aap-1"), _H("0x1.c5e53aa362eb4p-4")),
    (_H("0x1.b2036576afce6p-1"), _H("0x1.526e57720db08p-3")),
    (_H("0x1.9c2d163a1aa2dp-1"), _H("0x1.bc2860d22477p-3")),
    (_H("0x1.886e6037841edp-1"), _H("0x1.1058bc8a07ee1p-2")),
    (_H("0x1.767dcf5534862p-1"), _H("0x1.4043057b6ee09p-2")),
)
_LOGF_LN2 = _H("0x1.62e42fefa39efp-1")
_LOGF_A = (
    _H("-0x1.00ea348b88334p-2"),
    _H("0x1.5575b0be00b6ap-2"),
    _H("-0x1.ffffef20a4123p-2"),
)
_LOGF_OFF = 0x3F330000

# ---------------------------------------------------------------------------
# e_exp2f_data.c -- shared by expf, exp2f and powf.  EXP2F_TABLE_BITS = 5.
# ---------------------------------------------------------------------------
_EXP2F_N = 32
_EXP2F_TAB = (
    0x3FF0000000000000, 0x3FEFD9B0D3158574, 0x3FEFB5586CF9890F,
    0x3FEF9301D0125B51, 0x3FEF72B83C7D517B, 0x3FEF54873168B9AA,
    0x3FEF387A6E756238, 0x3FEF1E9DF51FDEE1, 0x3FEF06FE0A31B715,
    0x3FEEF1A7373AA9CB, 0x3FEEDEA64C123422, 0x3FEECE086061892D,
    0x3FEEBFDAD5362A27, 0x3FEEB42B569D4F82, 0x3FEEAB07DD485429,
    0x3FEEA47EB03A5585, 0x3FEEA09E667F3BCD, 0x3FEE9F75E8EC5F74,
    0x3FEEA11473EB0187, 0x3FEEA589994CCE13, 0x3FEEACE5422AA0DB,
    0x3FEEB737B0CDC5E5, 0x3FEEC49182A3F090, 0x3FEED503B23E255D,
    0x3FEEE89F995AD3AD, 0x3FEEFF76F2FB5E47, 0x3FEF199BDD85529C,
    0x3FEF3720DCEF9069, 0x3FEF5818DCFBA487, 0x3FEF7C97337B9B5F,
    0x3FEFA4AFA2A490DA, 0x3FEFD0765B6E4540,
)
_EXP2F_POLY = (
    _H("0x1.c6af84b912394p-5"),
    _H("0x1.ebfce50fac4f3p-3"),
    _H("0x1.62e42ff0c52d6p-1"),
)
_EXP2F_SHIFT = _H("0x1.8p+52")
_EXP2F_SHIFT_SCALED = _H("0x1.8p+52") / _EXP2F_N
_EXP2F_POLY_SCALED = (
    _EXP2F_POLY[0] / _EXP2F_N / _EXP2F_N / _EXP2F_N,
    _EXP2F_POLY[1] / _EXP2F_N / _EXP2F_N,
    _EXP2F_POLY[2] / _EXP2F_N,
)
_EXP2F_INVLN2_SCALED = _H("0x1.71547652b82fep+0") * _EXP2F_N

# ---------------------------------------------------------------------------
# e_powf_log2_data.c -- POWF_SCALE is 1.0 because TOINT_INTRINSICS is 0.
# ---------------------------------------------------------------------------
_POWF_TAB = (
    (_H("0x1.661ec79f8f3bep+0"), _H("-0x1.efec65b963019p-2")),
    (_H("0x1.571ed4aaf883dp+0"), _H("-0x1.b0b6832d4fca4p-2")),
    (_H("0x1.49539f0f010bp+0"), _H("-0x1.7418b0a1fb77bp-2")),
    (_H("0x1.3c995b0b80385p+0"), _H("-0x1.39de91a6dcf7bp-2")),
    (_H("0x1.30d190c8864a5p+0"), _H("-0x1.01d9bf3f2b631p-2")),
    (_H("0x1.25e227b0b8eap+0"), _H("-0x1.97c1d1b3b7afp-3")),
    (_H("0x1.1bb4a4a1a343fp+0"), _H("-0x1.2f9e393af3c9fp-3")),
    (_H("0x1.12358f08ae5bap+0"), _H("-0x1.960cbbf788d5cp-4")),
    (_H("0x1.0953f419900a7p+0"), _H("-0x1.a6f9db6475fcep-5")),
    (_H("0x1p+0"), _H("0x0p+0")),
    (_H("0x1.e608cfd9a47acp-1"), _H("0x1.338ca9f24f53dp-4")),
    (_H("0x1.ca4b31f026aap-1"), _H("0x1.476a9543891bap-3")),
    (_H("0x1.b2036576afce6p-1"), _H("0x1.e840b4ac4e4d2p-3")),
    (_H("0x1.9c2d163a1aa2dp-1"), _H("0x1.40645f0c6651cp-2")),
    (_H("0x1.886e6037841edp-1"), _H("0x1.88e9c2c1b9ff8p-2")),
    (_H("0x1.767dcf5534862p-1"), _H("0x1.ce0a44eb17bccp-2")),
)
_POWF_A = (
    _H("0x1.27616c9496e0bp-2"),
    _H("-0x1.71969a075c67ap-2"),
    _H("0x1.ec70a6ca7baddp-2"),
    _H("-0x1.7154748bef6c8p-1"),
    _H("0x1.71547652ab82bp0"),
)
_POWF_OFF = 0x3F330000
_POWF_SIGN_BIAS = 1 << (5 + 11)

# e_log10f.c FP32 constants.
_IVLN10 = _asfloat(0x3EDE5BD9)
_LOG10_2HI = _asfloat(0x3E9A2080)
_LOG10_2LO = _asfloat(0x355427DB)
_TWO25 = _asfloat(0x4C000000)


def logf(x) -> np.float32:
    """glibc 2.39 ``logf`` (sysdeps/ieee754/flt-32/e_logf.c)."""
    ix = _asuint(x)
    if ix == 0x3F800000:
        return F(0.0)
    if (ix - 0x00800000) & _U32 >= 0x7F800000 - 0x00800000:
        if ix * 2 & _U32 == 0:
            return F(-np.inf)
        if ix == 0x7F800000:
            return F(x)
        if (ix & 0x80000000) or (ix * 2) & _U32 >= 0xFF000000:
            return F(np.nan)
        ix = _asuint(F(F(x) * F(_H("0x1p23"))))
        ix = (ix - (23 << 23)) & _U32
    tmp = (ix - _LOGF_OFF) & _U32
    i = (tmp >> (23 - 4)) % 16
    k = _as_int32(tmp) >> 23
    iz = (ix - (tmp & 0xFF800000)) & _U32
    invc, logc = _LOGF_TAB[i]
    z = float(_asfloat(iz))
    r = z * invc - 1.0
    y0 = logc + float(k) * _LOGF_LN2
    r2 = r * r
    y = _LOGF_A[1] * r + _LOGF_A[2]
    y = _LOGF_A[0] * r2 + y
    y = y * r2 + (y0 + r)
    return F(y)


#: Elements per block of :func:`logf_array` scratch.  Bounds the binary64
#: temporaries at a few MiB whatever the field's size.
_LOGF_ARRAY_BLOCK = 1 << 20
_LOGF_TAB_INVC = np.asarray([row[0] for row in _LOGF_TAB], dtype=np.float64)
_LOGF_TAB_LOGC = np.asarray([row[1] for row in _LOGF_TAB], dtype=np.float64)


def logf_array(x) -> np.ndarray:
    """:func:`logf` over a NumPy array, the same bits element for element.

    e_logf.c's main path evaluated in binary64 and integer array arithmetic,
    operation for operation as :func:`logf` evaluates it: each product and
    sum rounded once, as in the C, and NumPy's elementwise loops never
    contract a multiply and an add into one rounding.  Elements the C
    handles specially (exactly 1, zero, negative, subnormal, infinite, NaN)
    go through :func:`logf` itself; a physical field has none of them, so
    the field pays the array path only.

    Written for ``ruclsminit``'s freezing curve (:func:`woof.core.ruc.
    ruc_initialize_cold_start`), which took :func:`logf` once per soil level
    per column: about 30 microseconds of CPython per column, which is a
    minute of card idle on HRRR's 1.9 million columns at every launch.
    """
    value = np.asarray(x, dtype=np.float32)
    out = np.empty(value.shape, dtype=np.float32)
    flat = np.ascontiguousarray(value).reshape(-1)
    flat_out = out.reshape(-1)
    if flat.size == 0:
        return out
    block = min(_LOGF_ARRAY_BLOCK, flat.size)
    for start in range(0, flat.size, block):
        stop = min(start + block, flat.size)
        _logf_block(flat[start:stop], flat_out[start:stop])
    return out


def _logf_block(flat_x: np.ndarray, out: np.ndarray) -> None:
    """:func:`logf` of one contiguous 1-D float32 block, into ``out``."""
    ix = flat_x.view(np.uint32)
    # e_logf.c: ``if (ix - 0x00800000 >= 0x7f800000 - 0x00800000)`` takes
    # the special path; ``ix == 0x3f800000`` answers exactly zero.
    ordinary = np.subtract(ix, np.uint32(0x00800000)) < np.uint32(
        0x7F800000 - 0x00800000)
    ordinary &= ix != np.uint32(0x3F800000)
    if ordinary.all():
        special = None
        ixf = ix
    else:
        special = np.flatnonzero(~ordinary)
        ixf = ix[ordinary]
    tmp = np.subtract(ixf, np.uint32(_LOGF_OFF))
    i = (tmp >> np.uint32(23 - 4)) & np.uint32(15)
    k = tmp.view(np.int32) >> np.int32(23)
    iz = np.subtract(ixf, tmp & np.uint32(0xFF800000))
    invc = _LOGF_TAB_INVC[i]
    logc = _LOGF_TAB_LOGC[i]
    z = iz.view(np.float32).astype(np.float64)
    r = z * invc
    r -= 1.0
    y0 = k.astype(np.float64) * _LOGF_LN2
    y0 += logc
    r2 = r * r
    y = _LOGF_A[1] * r
    y += _LOGF_A[2]
    y = _logf_fma_free(_LOGF_A[0], r2, y)
    y0 += r
    y = _logf_fma_free(y, r2, y0)
    if special is None:
        out[...] = y.astype(np.float32)
    else:
        out[ordinary] = y.astype(np.float32)
        for index in special:
            out[index] = logf(flat_x[index])


def _logf_fma_free(a, b, c):
    """``a * b + c`` as two roundings, the way the C and :func:`logf` do it."""
    product = a * b
    product += c
    return product


def log10f(x) -> np.float32:
    """glibc 2.39 ``log10f`` (still the 1993 SunPro FP32 reduction)."""
    value = F(x)
    hx = _asuint(value)
    k = 0
    if _as_int32(hx) < 0x00800000:
        if hx & 0x7FFFFFFF == 0:
            # glibc spells log10(+-0) as ``-two25 / fabsf(x)``, a division
            # whose only purpose is to raise divide-by-zero on the way to
            # -inf; FDLIBM divides by a ``zero`` variable instead.  ArWen
            # returns the -inf.  Dropped at 2.7.0 -- the two forms were
            # compared on all 4,294,967,296 float32 bit patterns and differ
            # on none.
            return F(-np.inf)
        if _as_int32(hx) < 0:
            return F(np.nan)
        k -= 25
        value = F(value * _TWO25)
        hx = _asuint(value)
    if hx >= 0x7F800000:
        return F(value + value)
    k += (hx >> 23) - 127
    i = 1 if k < 0 else 0
    hx = (hx & 0x007FFFFF) | ((0x7F - i) << 23)
    y = F(k + i)
    value = _asfloat(hx)
    z = F(F(y * _LOG10_2LO) + F(_IVLN10 * logf(value)))
    return F(z + F(y * _LOG10_2HI))


def _exp2_core(xd: float, shift: float, poly, sign_bias: int) -> float:
    """The shared 32-entry exp2 core of glibc's expf / exp2f / powf."""
    kd = xd + shift
    ki = _asuint64(kd)
    kd -= shift
    r = xd - kd
    t = _EXP2F_TAB[ki % _EXP2F_N]
    t = (t + (((ki + sign_bias) & _U64) << (52 - 5))) & _U64
    s = _asdouble(t)
    z = poly[0] * r + poly[1]
    r2 = r * r
    y = poly[2] * r + 1.0
    y = z * r2 + y
    return y * s


def expf(x) -> np.float32:
    """glibc 2.39 ``expf`` (sysdeps/ieee754/flt-32/e_expf.c)."""
    value = F(x)
    abstop = (_asuint(value) >> 20) & 0x7FF
    if abstop >= (_asuint(F(88.0)) >> 20):
        if _asuint(value) == _asuint(F(-np.inf)):
            return F(0.0)
        if abstop >= (_asuint(F(np.inf)) >> 20):
            return F(value + value)
        if value > _asfloat(0x42B17218):        # 0x1.62e42ep6f
            return F(np.inf)
        if value < -_asfloat(0x42CFF1B4):       # -0x1.9fe368p6f
            return F(0.0)
    xd = float(value)
    z = _EXP2F_INVLN2_SCALED * xd
    return F(_exp2_core(z, _EXP2F_SHIFT, _EXP2F_POLY_SCALED, 0))


def expf_array(x) -> np.ndarray:
    """:func:`expf` over a NumPy array, the same bits element for element.

    The main path of :func:`expf` evaluated in binary64 and integer array
    arithmetic, operation for operation as :func:`expf` and
    :func:`_exp2_core` evaluate it: each product and sum rounded once, and
    NumPy's elementwise loops
    never contract a multiply and an add into one rounding.  The two
    saturating branches (overflow to +inf above 0x1.62e42ep6, underflow to
    +0 below -0x1.9fe368p6) are applied as masks, exactly the C's tests;
    infinities and NaN go through :func:`expf` itself.

    Written for P3's rain-table generation (woof/core/p3_tables.py), which
    takes ``exp(-lamr*dia)`` three million times at start-up; NumPy's own
    float32 ``exp`` is a different function there and moved the tables
    (module_mp_p3.F:652-656).
    """
    value = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    flat = value.reshape(-1)
    out = np.empty(flat.shape, dtype=np.float32)
    if flat.size == 0:
        return out.reshape(value.shape)
    finite = np.isfinite(flat)
    over = finite & (flat > _asfloat(0x42B17218))
    under = finite & (flat < -_asfloat(0x42CFF1B4))
    core = finite & ~over & ~under
    xd = flat[core].astype(np.float64)
    z = _EXP2F_INVLN2_SCALED * xd
    kd = z + _EXP2F_SHIFT                       # kd = xd + shift
    ki = kd.view(np.uint64).copy()
    kd = kd - _EXP2F_SHIFT                      # kd -= shift
    r = z - kd
    t = _EXP2F_TAB_U64[(ki & np.uint64(_EXP2F_N - 1)).astype(np.intp)]
    t = t + (ki << np.uint64(52 - 5))           # sign_bias 0
    sd = t.view(np.float64)
    zz = _EXP2F_POLY_SCALED[0] * r
    zz = zz + _EXP2F_POLY_SCALED[1]
    r2 = r * r
    y = _EXP2F_POLY_SCALED[2] * r
    y = y + 1.0
    zz = zz * r2
    y = zz + y
    out[core] = (y * sd).astype(np.float32)
    out[over] = np.float32(np.inf)
    out[under] = np.float32(0.0)
    for position in np.flatnonzero(~finite):
        out[position] = expf(flat[position])
    return out.reshape(value.shape)


def _powf_log2(ix: int) -> float:
    tmp = (ix - _POWF_OFF) & _U32
    i = (tmp >> (23 - 4)) % 16
    top = tmp & 0xFF800000
    iz = (ix - top) & _U32
    k = _as_int32(top) >> 23
    invc, logc = _POWF_TAB[i]
    z = float(_asfloat(iz))
    r = z * invc - 1.0
    y0 = logc + float(k)
    r2 = r * r
    y = _POWF_A[0] * r + _POWF_A[1]
    p = _POWF_A[2] * r + _POWF_A[3]
    r4 = r2 * r2
    q = _POWF_A[4] * r + y0
    q = p * r2 + q
    return y * r4 + q


def _checkint(iy: int) -> int:
    exponent = (iy >> 23) & 0xFF
    if exponent < 0x7F:
        return 0
    if exponent > 0x7F + 23:
        return 2
    if iy & ((1 << (0x7F + 23 - exponent)) - 1):
        return 0
    if iy & (1 << (0x7F + 23 - exponent)):
        return 1
    return 2


def _zeroinfnan(ix: int) -> bool:
    return ((2 * ix - 1) & _U32) >= (2 * 0x7F800000 - 1) & _U32


def powf(x, y) -> np.float32:
    """glibc 2.39 ``powf`` (sysdeps/ieee754/flt-32/e_powf.c)."""
    base = F(x)
    exponent = F(y)
    sign_bias = 0
    ix = _asuint(base)
    iy = _asuint(exponent)
    if (ix - 0x00800000) & _U32 >= 0x7F800000 - 0x00800000 or _zeroinfnan(iy):
        if _zeroinfnan(iy):
            if (2 * iy) & _U32 == 0:
                return F(1.0)
            if ix == 0x3F800000:
                return F(1.0)
            if (2 * ix) & _U32 > (2 * 0x7F800000) & _U32 \
                    or (2 * iy) & _U32 > (2 * 0x7F800000) & _U32:
                return F(base + exponent)
            if (2 * ix) & _U32 == 2 * 0x3F800000:
                return F(1.0)
            if ((2 * ix) & _U32 < 2 * 0x3F800000) == (not (iy & 0x80000000)):
                return F(0.0)
            return F(exponent * exponent)
        if _zeroinfnan(ix):
            squared = F(base * base)
            if ix & 0x80000000 and _checkint(iy) == 1:
                squared = F(-squared)
            return F(F(1.0) / squared) if iy & 0x80000000 else squared
        if ix & 0x80000000:
            yint = _checkint(iy)
            if yint == 0:
                return F(np.nan)
            if yint == 1:
                sign_bias = _POWF_SIGN_BIAS
            ix &= 0x7FFFFFFF
        if ix < 0x00800000:
            ix = _asuint(F(base * F(_H("0x1p23")))) & 0x7FFFFFFF
            ix = (ix - (23 << 23)) & _U32
    logx = _powf_log2(ix)
    ylogx = float(exponent) * logx
    if (_asuint64(ylogx) >> 47) & 0xFFFF >= (_asuint64(126.0) >> 47):
        if ylogx > _H("0x1.fffffffd1d571p+6"):
            return F(-np.inf) if sign_bias else F(np.inf)
        if ylogx <= -150.0:
            return F(-0.0) if sign_bias else F(0.0)
    return F(_exp2_core(ylogx, _EXP2F_SHIFT_SCALED, _EXP2F_POLY, sign_bias))


_POWF_INVC = np.array([row[0] for row in _POWF_TAB], dtype=np.float64)
_POWF_LOGC = np.array([row[1] for row in _POWF_TAB], dtype=np.float64)
_EXP2F_TAB_U64 = np.array(_EXP2F_TAB, dtype=np.uint64)
_POWF_BIG_TOP = np.uint64(_asuint64(126.0) >> 47)

#: Elements :func:`powf_array` evaluates at a time.  A call allocates one
#: block of scratch (about 54 bytes per element: three uint32, one index and
#: four binary64 arrays, 3.4 MiB) and reuses it for every block, so it holds
#: its float32 result plus that, whatever the field's size, as NumPy's own
#: float32 ``power`` holds its result.  The equation of state of every CPU
#: preparation calls it on a whole nested child, and the CPU preparation
#: host-RAM model (``woof.core.preflight``) prices no per-cell transient
#: there: evaluated in one piece, a 76 x 480 x 600 child's call held 3.2 GiB
#: more than NumPy's power did, enough to put a deep nested preparation past
#: its estimate.  Reusing the scratch, rather than letting each block
#: allocate its own, also keeps the allocator from returning and refaulting
#: it every block, which tripled the time on glibc.
_POWF_ARRAY_BLOCK = 1 << 16


def _flat_blocks(array: np.ndarray, shape: tuple, size: int):
    """``[start, stop)`` of ``array`` broadcast to ``shape`` and flattened in
    C order, never materializing the broadcast: a contiguous operand is
    sliced, a one-element operand (a scalar exponent) is repeated once into
    one block, and any other broadcast is gathered one block at a time."""
    if array.shape == shape and array.flags.c_contiguous:
        flat = array.reshape(-1)
        return lambda start, stop: flat[start:stop]
    if array.size == 1:
        repeated = np.repeat(array.reshape(-1), size)
        return lambda start, stop: repeated[:stop - start]
    wide = np.broadcast_to(array, shape)
    return lambda start, stop: wide.flat[start:stop]


def powf_array(x, y) -> np.ndarray:
    """The pinned FP32 power over arrays on the explicit Rust worker pool.

    Arithmetic, including zero and special operands, is in Rust. Each
    worker retains only scalar scratch, and array broadcasting is marshalled
    in bounded blocks. Older staged libraries keep the same reference bits.
    """
    from woof.core import portable_math as pm
    library = pm._load()
    entry = getattr(library, "gpuwm_glibc239_powf_array_f32", None)
    if entry is None:
        return _powf_array_reference(x, y)
    import ctypes
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, pointer, pointer, size, size, size, size]
    entry.restype = ctypes.c_int32
    base = np.asarray(x, dtype=np.float32)
    exponent = np.asarray(y, dtype=np.float32)
    shape = np.broadcast_shapes(base.shape, exponent.shape)
    out = np.empty(shape, dtype=np.float32)
    if not out.size:
        return out
    workers = pm._workers(None, out.size)

    def apply(left, right, target):
        left = np.ascontiguousarray(left, dtype=np.float32)
        right = np.ascontiguousarray(right, dtype=np.float32)
        code = entry(pointer(left.ctypes.data), pointer(right.ctypes.data),
                     pointer(target.ctypes.data), target.size,
                     left.size, right.size, workers)
        if code:
            raise RuntimeError(f"native pinned FP32 power failed with code {code}")

    if (base.size == 1 or (base.shape == shape and base.flags.c_contiguous)) and (
            exponent.size == 1 or (exponent.shape == shape and exponent.flags.c_contiguous)):
        apply(base, exponent, out)
        return out
    block = min(_POWF_ARRAY_BLOCK, out.size)
    base_block = _flat_blocks(base, shape, block)
    exponent_block = _flat_blocks(exponent, shape, block)
    flat_out = out.reshape(-1)
    for start in range(0, out.size, block):
        stop = min(start + block, out.size)
        apply(base_block(start, stop), exponent_block(start, stop), flat_out[start:stop])
    return out


def _powf_array_reference(x, y) -> np.ndarray:
    """:func:`powf` over NumPy arrays, the same bits element for element.

    NumPy's own float32 ``power`` is whatever the host provides: the MSVC
    runtime's ``powf`` on Windows, glibc's on Linux, and on an AVX-512 Linux
    machine NumPy 2.5's vector loop, which rounds differently from both.
    A value derived through it therefore depends on the machine it was
    computed on.  This is glibc's algorithm evaluated in binary64 and
    integer array arithmetic, operation for operation as :func:`powf`
    evaluates it, so every host gets glibc's answer.  It runs over the
    flattened operands :data:`_POWF_ARRAY_BLOCK` elements at a time in one
    reused block of scratch, so its memory is its result plus that block,
    not a multiple of the field.

    The array path covers a positive normal base with a finite nonzero
    exponent whose result lies inside the float32 range, which is every
    element of a physical field.  Any other element (zero, negative,
    subnormal, infinite or NaN operands, overflow and underflow) goes
    through :func:`powf` itself.
    """
    base = np.asarray(x, dtype=np.float32)
    exponent = np.asarray(y, dtype=np.float32)
    out = np.empty(np.broadcast_shapes(base.shape, exponent.shape),
                   dtype=np.float32)
    flat_out = out.reshape(-1)
    if flat_out.size == 0:
        return out
    block = min(_POWF_ARRAY_BLOCK, flat_out.size)
    base_block = _flat_blocks(base, out.shape, block)
    exponent_block = _flat_blocks(exponent, out.shape, block)
    work = _PowfScratch(block)
    for start in range(0, flat_out.size, block):
        stop = min(start + block, flat_out.size)
        _powf_block(base_block(start, stop), exponent_block(start, stop),
                    flat_out[start:stop], work)
    return out


class _PowfScratch:
    """One block of :func:`_powf_block` scratch, reused block after block."""

    def __init__(self, size: int):
        self.u32 = np.empty((3, size), dtype=np.uint32)
        self.index = np.empty(size, dtype=np.intp)
        self.f64 = np.empty((4, size), dtype=np.float64)
        self.mask = np.empty((2, size), dtype=bool)


def _powf_block(flat_base: np.ndarray, flat_exponent: np.ndarray,
                out: np.ndarray, work: _PowfScratch) -> None:
    """:func:`powf` of two contiguous 1-D float32 blocks, into ``out``.

    Every value is the one :func:`powf` computes, by the same binary64 or
    integer operation on the same operands (each product and sum rounded
    once, as in the C), written into ``work``; only where the values live,
    and the order of steps that do not depend on each other, differ from
    the scalar transcription.
    """
    n = flat_base.size
    ix = flat_base.view(np.uint32)
    iy = flat_exponent.view(np.uint32)
    ua, ub, uc = work.u32[:, :n]
    ordinary, spare = work.mask[:, :n]
    np.subtract(ix, np.uint32(0x00800000), out=ua)
    np.less(ua, np.uint32(0x7F800000 - 0x00800000), out=ordinary)
    np.multiply(iy, np.uint32(2), out=ua)
    np.subtract(ua, np.uint32(1), out=ua)
    np.less(ua, np.uint32((2 * 0x7F800000 - 1) & _U32), out=spare)
    np.logical_and(ordinary, spare, out=ordinary)
    # A field is ordinary throughout; only a field that is not pays for
    # gathering its ordinary elements out and scattering them back.
    if ordinary.all():
        fast = None
        ixf, yf = ix, flat_exponent
    else:
        fast = np.flatnonzero(ordinary)
        ixf, yf = ix[fast], flat_exponent[fast]
    m = ixf.size

    # _powf_log2, vectorized.
    tmp, i, top = ua[:m], ub[:m], uc[:m]
    index = work.index[:m]
    f0, f1, f2, f3 = work.f64[:, :m]
    np.subtract(ixf, np.uint32(_POWF_OFF), out=tmp)
    np.right_shift(tmp, np.uint32(23 - 4), out=i)
    np.bitwise_and(i, np.uint32(15), out=index)
    np.bitwise_and(tmp, np.uint32(0xFF800000), out=top)
    iz = tmp
    np.subtract(ixf, top, out=iz)                      # iz = ix - top
    k = top.view(np.int32)
    np.right_shift(k, np.int32(23), out=k)
    np.copyto(f0, k)                                   # k, exact
    np.copyto(f1, iz.view(np.float32))                 # z, exact
    r = f2
    np.take(_POWF_INVC, index, out=r, mode="clip")
    np.multiply(f1, r, out=r)
    np.subtract(r, 1.0, out=r)                         # r = z*invc - 1
    q = f3
    np.take(_POWF_LOGC, index, out=q, mode="clip")
    np.add(q, f0, out=q)                               # y0 = logc + k
    np.multiply(r, _POWF_A[4], out=f0)
    np.add(f0, q, out=q)                               # q = A4*r + y0
    r2 = f1
    np.multiply(r, r, out=r2)                          # r2 = r*r
    np.multiply(r, _POWF_A[2], out=f0)
    np.add(f0, _POWF_A[3], out=f0)                     # p = A2*r + A3
    np.multiply(f0, r2, out=f0)
    np.add(f0, q, out=q)                               # q = p*r2 + q
    np.multiply(r, _POWF_A[0], out=f0)
    np.add(f0, _POWF_A[1], out=f0)                     # y = A0*r + A1
    np.multiply(r2, r2, out=r2)                        # r4 = r2*r2
    np.multiply(f0, r2, out=f0)
    np.add(f0, q, out=f0)                              # logx = y*r4 + q
    ylogx = f1
    np.copyto(ylogx, yf)
    np.multiply(ylogx, f0, out=ylogx)                  # ylogx = y * logx

    # Overflow and underflow candidates take the scalar branch.
    top16 = f0.view(np.uint64)
    np.right_shift(ylogx.view(np.uint64), np.uint64(47), out=top16)
    np.bitwise_and(top16, np.uint64(0xFFFF), out=top16)
    inside = spare[:m]
    np.less(top16, _POWF_BIG_TOP, out=inside)
    if not inside.all():
        keep = np.flatnonzero(inside)
        fast = keep if fast is None else fast[keep]
        ylogx = ylogx[keep]
        m = ylogx.size
        index = index[:m]
        f0, f2, f3 = f0[:m], f2[:m], f3[:m]

    # _exp2_core with sign_bias 0, vectorized.
    ki = f0.view(np.uint64)
    np.add(ylogx, _EXP2F_SHIFT_SCALED, out=f0)         # kd = ylogx + shift
    kd = f2
    np.subtract(f0, _EXP2F_SHIFT_SCALED, out=kd)       # kd -= shift
    r = kd
    np.subtract(ylogx, kd, out=r)                      # r = ylogx - kd
    t = f3.view(np.uint64)
    np.bitwise_and(ki, np.uint64(_EXP2F_N - 1), out=t)
    np.copyto(index, t, casting="unsafe")              # 0 to 31
    np.take(_EXP2F_TAB_U64, index, out=t, mode="clip")
    np.left_shift(ki, np.uint64(52 - 5), out=ki)
    np.add(t, ki, out=t)                               # t = tab + (ki << 47)
    s = f3
    zz = f0
    np.multiply(r, _EXP2F_POLY[0], out=zz)
    np.add(zz, _EXP2F_POLY[1], out=zz)                 # z = C0*r + C1
    r2 = ylogx
    np.multiply(r, r, out=r2)                          # r2 = r*r
    yy = r
    np.multiply(r, _EXP2F_POLY[2], out=yy)
    np.add(yy, 1.0, out=yy)                            # y = C2*r + 1
    np.multiply(zz, r2, out=zz)
    np.add(zz, yy, out=yy)                             # y = z*r2 + y
    np.multiply(yy, s, out=yy)                         # y * s
    if fast is None:
        np.copyto(out, yy)                             # one rounding to f32
        return
    out[fast] = yy
    rest = np.ones(n, dtype=bool)
    rest[fast] = False
    for position in np.flatnonzero(rest):
        out[position] = powf(flat_base[position], flat_exponent[position])


# ---------------------------------------------------------------------------
# atanf -- glibc 2.39 sysdeps/ieee754/flt-32/s_atanf.c, which is still the
# fdlibm reduction: five argument ranges, one 11-term odd/even polynomial, and
# a hi/lo pair per range.  Every operation is FP32; there is no FP64
# intermediate anywhere, which is why an FP64-then-round shim is a different
# function here as well.  SFCDIF1 (module_sf_noahmplsm.F:4691, 4698) is the
# only Noah-MP caller.
#
# Two facts measured against the live glibc 2.39 on the oracle host, not
# assumed:
#
#   * the large-argument shortcut fires at |x| >= 2**25 (0x4C000000), not at
#     the 2**26 or 2**34 thresholds other fdlibm descendants use.  Above it
#     glibc returns the constant 0x3FC90FDB for every argument;
#   * with that threshold, this transcription reproduces glibc on **all
#     4,278,190,082 non-NaN FP32 inputs** -- the entire domain, swept
#     exhaustively -- with 0 mismatches.  Rounding the FP64 result once instead
#     disagrees on 823,767 of the 16,777,216 inputs in [0.5, 1) alone.
# ---------------------------------------------------------------------------
_ATANF_HI = tuple(_asfloat(b) for b in
                  (0x3EED6338, 0x3F490FDA, 0x3F7B985E, 0x3FC90FDA))
_ATANF_LO = tuple(_asfloat(b) for b in
                  (0x31AC3769, 0x33222168, 0x33140FB4, 0x33A22168))
_ATANF_T = tuple(_asfloat(b) for b in (
    0x3EAAAAAB, 0xBE4CCCCD, 0x3E124925, 0xBDE38E38, 0x3DBA2E6E, 0xBD9D8795,
    0x3D886B35, 0xBD6EF16B, 0x3D4BDA59, 0xBD15A221, 0x3C8569D7))


def atanf(x) -> np.float32:
    """glibc 2.39 ``atanf`` (sysdeps/ieee754/flt-32/s_atanf.c)."""
    value = F(x)
    hx = _as_int32(_asuint(value))
    ix = hx & 0x7FFFFFFF
    if ix >= 0x4C000000:                                # |x| >= 2**25
        if ix > 0x7F800000:
            return F(value + value)                     # NaN
        if hx > 0:
            return F(_ATANF_HI[3] + _ATANF_LO[3])
        return F(F(-_ATANF_HI[3]) - _ATANF_LO[3])
    if ix < 0x3EE00000:                                 # |x| < 0.4375
        if ix < 0x31000000:                             # |x| < 2**-29
            return value                                # huge + x > 1 always
        idx = -1
    else:
        value = F(abs(value))
        if ix < 0x3F980000:                             # |x| < 1.1875
            if ix < 0x3F300000:                         # 0.4375 <= |x| < 0.6875
                idx = 0
                value = F(F(F(F(2.0) * value) - F(1.0))
                          / F(F(2.0) + value))
            else:                                       # 0.6875 <= |x| < 1.1875
                idx = 1
                value = F(F(value - F(1.0)) / F(value + F(1.0)))
        elif ix < 0x401C0000:                           # |x| < 2.4375
            idx = 2
            value = F(F(value - F(1.5))
                      / F(F(1.0) + F(F(1.5) * value)))
        else:                                           # 2.4375 <= |x| < 2**25
            idx = 3
            value = F(F(-1.0) / value)
    z = F(value * value)
    w = F(z * z)
    s1 = F(z * F(_ATANF_T[0] + F(w * F(_ATANF_T[2] + F(w * F(
        _ATANF_T[4] + F(w * F(_ATANF_T[6] + F(w * F(
            _ATANF_T[8] + F(w * _ATANF_T[10])))))))))))
    s2 = F(w * F(_ATANF_T[1] + F(w * F(_ATANF_T[3] + F(w * F(
        _ATANF_T[5] + F(w * F(_ATANF_T[7] + F(w * _ATANF_T[9])))))))))
    if idx < 0:
        return F(value - F(value * F(s1 + s2)))
    z = F(_ATANF_HI[idx]
          - F(F(F(value * F(s1 + s2)) - _ATANF_LO[idx]) - value))
    return F(-z) if hx < 0 else z


# ---------------------------------------------------------------------------
# atanf, correctly rounded -- CORE-MATH src/binary32/atan/atanf.c (a8066a5),
# the routine glibc 2.41 and later ship as sysdeps/ieee754/flt-32/s_atanf.c.
# MIT; notice in woof/core/kernels/LICENSE-third-party.txt section 6, the
# same routine glibc_trig_flt32.cuh's glibc_atanf transcribes for the device.
#
# A WRF built on glibc 2.41 or later takes this atanf where one built on
# glibc 2.39 takes :func:`atanf` above; the two disagree on 816 of
# MYJSFCINIT's 10001 PSIM arguments (woof.core.myjsfc_tables).  Python
# floats are IEEE binary64 and every operation below rounds once, as the C
# does with each double operation separately rounded.
# ---------------------------------------------------------------------------
_ATANF_CR_CN = tuple(float.fromhex(h) for h in (
    "0x1.51eccde075d67p-2", "0x1.a76bb5637f2f2p-1", "0x1.81e0eed20de88p-1",
    "0x1.376c8ca67d11dp-2", "0x1.aec7b69202ac6p-5", "0x1.9561899acc73ep-9",
    "0x1.bf9fa5b67e6p-16"))
_ATANF_CR_CD = tuple(float.fromhex(h) for h in (
    "0x1.51eccde075d66p-2", "0x1.dfbdd7b392d28p-1", "0x1p+0",
    "0x1.fd22bf0e89b54p-2", "0x1.d91ff8b576282p-4", "0x1.653ea99fc9bbp-7",
    "0x1.1e7fcc202340ap-12"))
_ATANF_CR_PI2 = float.fromhex("0x1.921fb54442d18p+0")


def _fmaf_exact(a, b, c) -> np.float32:
    """float32 fmaf: a*b+c formed exactly, rounded once to binary32."""
    from fractions import Fraction
    return F(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def atanf_correctly_rounded(x) -> np.float32:
    """Correctly rounded binary32 ``atanf`` (glibc 2.41 and later)."""
    value = F(x)
    t = int(_asuint(value))
    e = (t >> 23) & 0xFF
    ta = t & 0x7FFFFFFF
    if ta >= 0x4C700518:
        if ta > 0x7F800000:
            with np.errstate(invalid="ignore"):
                return F(value + value)                 # NaN, quieted
        return F(math.copysign(_ATANF_CR_PI2, float(value)))
    if e < 127 - 13:
        if e < 127 - 25:
            if not (t << 1) & 0xFFFFFFFF:
                return value
            return _fmaf_exact(-value, abs(value), value)
        return _fmaf_exact(F(F(-float.fromhex("0x1.555556p-2")) * value),
                           F(value * value), value)
    gt = e >= 127
    z = float(value)
    if gt:
        z = 1.0 / z
    z2 = z * z
    z4 = z2 * z2
    z8 = z4 * z4
    cn, cd = _ATANF_CR_CN, _ATANF_CR_CD
    cn0 = cn[0] + z2 * cn[1]
    cn2 = cn[2] + z2 * cn[3]
    cn4 = cn[4] + z2 * cn[5]
    cn0 = cn0 + z4 * cn2
    cn4 = cn4 + z4 * cn[6]
    cn0 = (cn0 + z8 * cn4) * z
    cd0 = cd[0] + z2 * cd[1]
    cd2 = cd[2] + z2 * cd[3]
    cd4 = cd[4] + z2 * cd[5]
    cd0 = cd0 + z4 * cd2
    cd4 = cd4 + z4 * cd[6]
    cd0 = cd0 + z8 * cd4
    r = cn0 / cd0
    if not gt:
        return F(r)
    r = (math.copysign(float.fromhex("0x1.0fdaa22168c23p-7"), z) - r) \
        + math.copysign(float.fromhex("0x1.9p0"), z)
    return F(r)


# ---------------------------------------------------------------------------
# expm1f / tanhf -- sysdeps/ieee754/flt-32/s_expm1f.c and s_tanhf.c
#
# Noah-MP's ENERGY reaches TANH exactly once, in the snow-cover fraction
#   FSNO = TANH( SNOWH / (parameters%SCFFAC * FMELT) )       (:2072)
# and that single call is enough to put every snow column's answer through
# glibc's tanhf.  It is *not* interchangeable with an FP64 shim: over the
# 204,064,836 FP32 inputs in [1e-6, 22], `(float)tanh((double)x)` disagrees
# with glibc's `tanhf` on 48,501,304 of them -- 23.8%, three orders of
# magnitude worse than expf's 0.062%.  These are still the 1993 SunPro
# routines in glibc 2.39, unlike logf/expf/powf above, which is why they are
# so much less accurate.
#
# Verified against the live glibc 2.39 on the oracle host (gcc -O2
# -ffp-contract=off): 1,106,247,680 FP32 inputs over (0, 30] -- **0
# mismatches**.  The CUDA half lives in woof/core/kernels/noahmp_energy.cu
# and must stay in step with this.
# ---------------------------------------------------------------------------

_EXPM1F_ONE = F(1.0)
_EXPM1F_HUGE = F(1.0e30)
_EXPM1F_TINY = F(1.0e-30)
_EXPM1F_O_THRESHOLD = F(8.8721679688e+01)
_EXPM1F_LN2_HI = F(6.9313812256e-01)
_EXPM1F_LN2_LO = F(9.0580006145e-06)
_EXPM1F_INVLN2 = F(1.4426950216e+00)
_EXPM1F_Q = (
    F(-3.3333335072e-02),
    F(1.5873016091e-03),
    F(-7.9365076090e-05),
    F(4.0082177293e-06),
    F(-2.0109921195e-07),
)


def expm1f(x) -> np.float32:
    """glibc 2.39 ``expm1f``.  Only reached through :func:`tanhf`."""

    x = F(x)
    one = _EXPM1F_ONE
    hx = _asuint(x)
    xsb = hx & 0x80000000
    hx &= 0x7FFFFFFF
    c = F(0.0)

    if hx >= 0x4195B844:                                # |x| >= 27*ln2
        if hx >= 0x42B17218:                            # |x| >= 88.721...
            if hx > 0x7F800000:
                return F(x + x)                         # NaN
            if hx == 0x7F800000:
                return x if xsb == 0 else F(-1.0)
            if x > _EXPM1F_O_THRESHOLD:
                # glibc returns huge*huge, i.e. +inf with the overflow flag
                # raised.  numpy would warn about the multiply; the value is
                # the point, not the warning.
                with np.errstate(over="ignore"):
                    return F(_EXPM1F_HUGE * _EXPM1F_HUGE)
        if xsb != 0:
            return F(_EXPM1F_TINY - one)                # x < -27*ln2 -> -1

    if hx > 0x3EB17218:                                 # |x| > 0.5*ln2
        if hx < 0x3F851592:                             # |x| < 1.5*ln2
            if xsb == 0:
                hi = F(x - _EXPM1F_LN2_HI)
                lo = _EXPM1F_LN2_LO
                k = 1
            else:
                hi = F(x + _EXPM1F_LN2_HI)
                lo = F(-_EXPM1F_LN2_LO)
                k = -1
        else:
            k = int(F(F(_EXPM1F_INVLN2 * x)
                      + (F(0.5) if xsb == 0 else F(-0.5))))
            t = F(k)
            hi = F(x - F(t * _EXPM1F_LN2_HI))           # t*ln2_hi is exact
            lo = F(t * _EXPM1F_LN2_LO)
        x = F(hi - lo)
        c = F(F(hi - x) - lo)
    elif hx < 0x33000000:                               # |x| < 2**-25
        t = F(_EXPM1F_HUGE + x)
        return F(x - F(t - F(_EXPM1F_HUGE + x)))
    else:
        k = 0

    hfx = F(F(0.5) * x)
    hxs = F(x * hfx)
    q1, q2, q3, q4, q5 = _EXPM1F_Q
    r1 = F(hxs * q5)
    r1 = F(hxs * F(q4 + r1))
    r1 = F(hxs * F(q3 + r1))
    r1 = F(hxs * F(q2 + r1))
    r1 = F(hxs * F(q1 + r1))
    r1 = F(one + r1)
    t = F(F(3.0) - F(r1 * hfx))
    e = F(hxs * F(F(r1 - t) / F(F(6.0) - F(x * t))))
    if k == 0:
        return F(x - F(F(x * e) - hxs))

    e = F(F(x * F(e - c)) - c)
    e = F(e - hxs)
    if k == -1:
        return F(F(F(0.5) * F(x - e)) - F(0.5))
    if k == 1:
        if x < F(-0.25):
            return F(F(-2.0) * F(e - F(x + F(0.5))))
        return F(one + F(F(2.0) * F(x - e)))
    if k <= -2 or k > 56:                               # exp(x)-1 suffices
        y = F(one - F(e - x))
        y = _asfloat((_asuint(y) + (k << 23)) & _U32)
        return F(y - one)
    if k < 23:
        t = _asfloat(0x3F800000 - (0x1000000 >> k))     # t = 1 - 2**-k
        y = F(t - F(e - x))
    else:
        t = _asfloat(((0x7F - k) << 23) & _U32)         # t = 2**-k
        y = F(x - F(e + t))
        y = F(y + one)
    return _asfloat((_asuint(y) + (k << 23)) & _U32)


def tanhf(x) -> np.float32:
    """glibc 2.39 ``tanhf``.  Noah-MP reaches it only through ENERGY's FSNO."""

    x = F(x)
    one = F(1.0)
    jx = _as_int32(_asuint(x))
    ix = jx & 0x7FFFFFFF
    if ix >= 0x7F800000:                                # inf or NaN
        return F(F(one / x) + one) if jx >= 0 else F(F(one / x) - one)
    if ix < 0x41B00000:                                 # |x| < 22
        # glibc guards zero here with ``if (ix == 0) return x;``.  FDLIBM does
        # not, and it is redundant: the |x| < 2**-55 branch below returns
        # x*(1+x), which is x for both signed zeros.  Dropped at 2.7.0 -- the
        # two forms were compared on all 4,294,967,296 float32 bit patterns
        # and differ on none.
        if ix < 0x24000000:                             # |x| < 2**-55
            return F(x * F(one + x))
        ax = F(abs(x))
        if ix >= 0x3F800000:                            # |x| >= 1
            t = expm1f(F(F(2.0) * ax))
            z = F(one - F(F(2.0) / F(t + F(2.0))))
        else:
            t = expm1f(F(F(-2.0) * ax))
            z = F(F(-t) / F(t + F(2.0)))
    else:
        z = F(one - _EXPM1F_TINY)
    return z if jx >= 0 else F(-z)


# ---------------------------------------------------------------------------
# log1pf -- glibc 2.39 sysdeps/ieee754/flt-32/s_log1pf.c, the FDLIBM routine
# (float conversion by Ian Lance Taylor, Cygnus Support).  Every operation is
# FP32: the argument is reduced to 1+f with f in [sqrt(2)/2 - 1, sqrt(2) - 1]
# and a correction term c, and log(1+f) is the seven-term series in
# s = f/(2+f).  Arguments in (-0.2929, 0.41422) skip the reduction (k = 0).
#
# The CPU equation of state reaches it with hypsometric_opt = 2, as
# log1p((pfd - pfu)/pfu) (woof.core.diagnostics and the two preparation
# paths that spell the same operator), and NumPy's float32 ``log1p`` is not
# one function across hosts: NumPy 2.5 on an AVX-512 Linux machine runs its
# own vector loop, which returns other words than glibc 2.39 on about a
# fifth of the ratios the equation of state takes, and glibc 2.41 replaced
# this routine with CORE-MATH's correctly rounded one, so even NumPy's
# scalar path moves with the host's glibc.  This is the one log1pf ArWen's
# CPU paths take, whatever the host.
#
# Verified exhaustively on WSL Ubuntu 24.04 (glibc 2.39, NumPy 2.5.3, an
# AVX-512 CPU): log1pf_array against NumPy's float32 log1p with
# NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL" (which is libm's log1pf:
# 0 of 200,000 random words differ through ctypes) on all 4,294,967,296
# float32 bit patterns -- 0 mismatching words, NaN words included -- and
# the same digest over all of them with NumPy's AVX-512 loops on, where
# NumPy's own log1p differs from glibc on 51,886,787 words, 431,795 of
# 2,000,000 ratios in [0.0005, 0.2].  log1pf itself matched libm on the
# 500,028 words of the same session's smoke sweep.
# ---------------------------------------------------------------------------

_LOG1PF_LN2_HI = _asfloat(0x3F317180)       # 6.9313812256e-01
_LOG1PF_LN2_LO = _asfloat(0x3717F7D1)       # 9.0580006145e-06
_LOG1PF_LP = tuple(_asfloat(b) for b in (
    0x3F2AAAAB,                             # Lp1 = 6.6666668653e-01
    0x3ECCCCCD,                             # Lp2 = 3.9999997616e-01
    0x3E924925,                             # Lp3 = 2.8571429849e-01
    0x3E638E29,                             # Lp4 = 2.2222198546e-01
    0x3E3A3325,                             # Lp5 = 1.8183572590e-01
    0x3E1CD04F,                             # Lp6 = 1.5313838422e-01
    0x3E178897,                             # Lp7 = 1.4798198640e-01
))
_LOG1PF_TWO_THIRDS = F(0.66666666666666666)
#: glibc returns ``(x-x)/(x-x)`` for x < -1 and x = -inf, which on x86-64 is
#: the default QNaN with the sign bit SET.  Taken as a constant so the answer
#: is the reference host's word on every host.
_LOG1PF_INVALID = _asfloat(0xFFC00000)


def _log1pf_series(f):
    """``R`` and ``s`` of s_log1pf.c's main path, in FP32 scalars."""
    lp1, lp2, lp3, lp4, lp5, lp6, lp7 = _LOG1PF_LP
    s = F(f / F(F(2.0) + f))
    z = F(s * s)
    r = F(z * F(lp1 + F(z * F(lp2 + F(z * F(lp3 + F(z * F(lp4 + F(z * F(
        lp5 + F(z * F(lp6 + F(z * lp7)))))))))))))
    return r, s


def log1pf(x) -> np.float32:
    """glibc 2.39 ``log1pf`` (sysdeps/ieee754/flt-32/s_log1pf.c)."""
    x = F(x)
    hx = _as_int32(_asuint(x))
    ax = hx & 0x7FFFFFFF
    k = 1
    if hx < 0x3ED413D7:                                 # x < 0.41422
        if ax >= 0x3F800000:                            # x <= -1.0
            if x == F(-1.0):
                return F(-np.inf)
            if ax > 0x7F800000:
                return F(x + x)                         # a negative NaN, quiet
            return _LOG1PF_INVALID
        if ax < 0x31000000:                             # |x| < 2**-29
            # glibc returns x below 2**-54 and x - x*x*0.5 above it; the
            # correction is under a quarter ULP of x, so both are x.
            return x
        if hx > 0 or hx <= _as_int32(0xBE95F61F):       # -0.2929 < x < 0.41422
            k = 0
            f = x
            hu = 1
    elif hx >= 0x7F800000:
        return F(x + x)                                 # +inf or NaN
    c = F(0.0)
    if k != 0:
        if hx < 0x5A000000:
            u = F(F(1.0) + x)
            hu = _as_int32(_asuint(u))
            k = (hu >> 23) - 127
            # correction term
            c = F(F(1.0) - F(u - x)) if k > 0 else F(x - F(u - F(1.0)))
            c = F(c / u)
        else:
            u = x
            hu = _as_int32(_asuint(u))
            k = (hu >> 23) - 127
        hu &= 0x007FFFFF
        if hu < 0x3504F7:
            u = _asfloat(hu | 0x3F800000)               # normalize u
        else:
            k += 1
            u = _asfloat(hu | 0x3F000000)               # normalize u/2
            hu = (0x00800000 - hu) >> 2
        f = F(u - F(1.0))
    hfsq = F(F(F(0.5) * f) * f)
    kf = F(k)
    if hu == 0:                                         # |f| < 2**-20
        if f == F(0.0):
            if k == 0:
                return F(0.0)
            c = F(c + F(kf * _LOG1PF_LN2_LO))
            return F(F(kf * _LOG1PF_LN2_HI) + c)
        r = F(hfsq * F(F(1.0) - F(_LOG1PF_TWO_THIRDS * f)))
        if k == 0:
            return F(f - r)
        return F(F(kf * _LOG1PF_LN2_HI)
                 - F(F(r - F(F(kf * _LOG1PF_LN2_LO) + c)) - f))
    r, s = _log1pf_series(f)
    if k == 0:
        return F(f - F(hfsq - F(s * F(hfsq + r))))
    return F(F(kf * _LOG1PF_LN2_HI)
             - F(F(hfsq - F(F(s * F(hfsq + r))
                            + F(F(kf * _LOG1PF_LN2_LO) + c))) - f))


#: Elements :func:`log1pf_array` evaluates at a time, through one reused
#: block of scratch (six float32 and one uint32 array and two masks, about
#: 30 bytes per element, 1.9 MiB), so a call on a physical field holds its
#: result plus that block, as NumPy's own float32 ``log1p`` holds its
#: result.  The same reason as :data:`_POWF_ARRAY_BLOCK`: the CPU equation
#: of state calls it on a whole nested child, and the CPU preparation
#: host-RAM model prices no per-cell transient there.
_LOG1PF_ARRAY_BLOCK = 1 << 16


class _Log1pfScratch:
    """One block of :func:`_log1pf_block` scratch, reused block after block."""

    def __init__(self, size: int):
        self.u32 = np.empty(size, dtype=np.uint32)
        self.f32 = np.empty((6, size), dtype=np.float32)
        self.mask = np.empty((2, size), dtype=bool)


def log1pf_array(x) -> np.ndarray:
    """:func:`log1pf` over a NumPy array, the same bits element for element.

    NumPy's own float32 ``log1p`` is the host's: the C library's
    ``log1pf`` (glibc 2.39's FDLIBM routine, glibc 2.41 and later a
    correctly rounded one, the MSVC runtime's on Windows), and on an
    AVX-512 Linux machine NumPy 2.5's vector loop, which rounds a fifth of
    physical ratios differently from glibc.  This is glibc 2.39's algorithm
    as float32 and integer array arithmetic, operation for operation as
    :func:`log1pf` evaluates it (each operation rounded once, as the C
    does), so every host gets the same answer.  It runs over the flattened
    argument :data:`_LOG1PF_ARRAY_BLOCK` elements at a time in one reused
    block of scratch.

    Arguments in [-0.2929, 0.41422) with magnitude at least 2**-29, which
    covers every ratio the equation of state takes, are a dozen float32
    operations each.  The rest of the line is array arithmetic too, and
    every special argument is a constant or ``x + x``; only the reduced
    arguments whose mantissa lands on 1 (``hu == 0`` in the C, a handful
    per binade) go through :func:`log1pf` itself.
    """
    values = np.asarray(x, dtype=np.float32)
    out = np.empty(values.shape, dtype=np.float32)
    flat_out = out.reshape(-1)
    if flat_out.size == 0:
        return out
    block = min(_LOG1PF_ARRAY_BLOCK, flat_out.size)
    source = _flat_blocks(values, values.shape, block)
    work = _Log1pfScratch(block)
    for start in range(0, flat_out.size, block):
        stop = min(start + block, flat_out.size)
        _log1pf_block(source(start, stop), flat_out[start:stop], work)
    return out


def _series_into(f, s, z, r) -> None:
    """``s = f/(2+f)`` and ``R`` of the seven-term series, into s and r."""
    lp1, lp2, lp3, lp4, lp5, lp6, lp7 = _LOG1PF_LP
    np.add(f, F(2.0), out=s)
    np.divide(f, s, out=s)                             # s = f/(2+f)
    np.multiply(s, s, out=z)                           # z = s*s
    np.multiply(z, lp7, out=r)
    for coefficient in (lp6, lp5, lp4, lp3, lp2):
        np.add(r, coefficient, out=r)
        np.multiply(z, r, out=r)
    np.add(r, lp1, out=r)
    np.multiply(z, r, out=r)                           # R


def _log1pf_k0(f: np.ndarray, out: np.ndarray, rows) -> None:
    """s_log1pf.c's ``k == 0`` path (``f = x``, no reduction), into out."""
    m = f.size
    hfsq, s, z, r = (row[:m] for row in rows[:4])
    np.multiply(f, F(0.5), out=hfsq)
    np.multiply(hfsq, f, out=hfsq)                     # hfsq = 0.5*f*f
    _series_into(f, s, z, r)
    np.add(hfsq, r, out=r)
    np.multiply(s, r, out=r)
    np.subtract(hfsq, r, out=r)
    np.subtract(f, r, out=out)                         # f-(hfsq-s*(hfsq+R))


def _log1pf_block(flat_x: np.ndarray, out: np.ndarray, work) -> None:
    """:func:`log1pf` of one contiguous 1-D float32 block, into ``out``."""
    n = flat_x.size
    hx = flat_x.view(np.uint32)
    ua = work.u32[:n]
    ordinary, spare = work.mask[:, :n]
    # k == 0 and |x| >= 2**-29: 2**-29 <= x < 0.41422 (0x31000000 up to
    # 0x3ED413D7) or -0.2929 <= x <= -2**-29 (0xB1000000 up to 0xBE95F61F).
    np.subtract(hx, np.uint32(0x31000000), out=ua)
    np.less(ua, np.uint32(0x3ED413D7 - 0x31000000), out=ordinary)
    np.subtract(hx, np.uint32(0xB1000000), out=ua)
    np.less_equal(ua, np.uint32(0xBE95F61F - 0xB1000000), out=spare)
    np.logical_or(ordinary, spare, out=ordinary)
    # A physical field is ordinary throughout; only a block that is not
    # pays for gathering its elements by path and scattering them back.
    if ordinary.all():
        _log1pf_k0(flat_x, out, work.f32)
        return
    where = np.flatnonzero(ordinary)
    if where.size:
        result = work.f32[5, :where.size]
        _log1pf_k0(flat_x[where], result, work.f32)
        out[where] = result
    rest = np.flatnonzero(~ordinary)
    out[rest] = _log1pf_outside(flat_x[rest])


def _log1pf_outside(x: np.ndarray) -> np.ndarray:
    """:func:`log1pf` of arguments outside the ``k == 0`` band."""
    bits = x.view(np.uint32)
    magnitude = bits & np.uint32(0x7FFFFFFF)
    negative = bits >= np.uint32(0x80000000)
    value = np.empty_like(x)
    tiny = magnitude < np.uint32(0x31000000)
    value[tiny] = x[tiny]                              # |x| < 2**-29
    quiet = (magnitude > np.uint32(0x7F800000)) | (
        bits == np.uint32(0x7F800000))
    with np.errstate(invalid="ignore"):
        value[quiet] = x[quiet] + x[quiet]             # NaN, +inf
    value[bits == np.uint32(0xBF800000)] = F(-np.inf)  # x = -1
    invalid = negative & (magnitude > np.uint32(0x3F800000)) & (
        magnitude <= np.uint32(0x7F800000))
    value[invalid] = _LOG1PF_INVALID                   # x < -1 or -inf
    # k != 0: u = 1+x with a correction term for 0.41422 <= x < 2**53 and
    # for -1 < x < -0.2929, u = x for 2**53 <= x < inf.
    through_sum = (~negative & (bits >= np.uint32(0x3ED413D7))
                   & (bits < np.uint32(0x5A000000))) | (
        negative & (bits > np.uint32(0xBE95F61F))
        & (bits < np.uint32(0xBF800000)))
    through_x = ~negative & (bits >= np.uint32(0x5A000000)) & (
        bits < np.uint32(0x7F800000))
    reduced = through_sum | through_x
    if reduced.any():
        value[reduced] = _log1pf_reduced(x[reduced], through_sum[reduced])
    return value


def _log1pf_reduced(x: np.ndarray, through_sum: np.ndarray) -> np.ndarray:
    """s_log1pf.c's ``k != 0`` path; ``through_sum`` marks ``u = 1+x``."""
    one = F(1.0)
    u = np.where(through_sum, x + one, x)
    hu = u.view(np.int32)
    k = (hu >> 23) - np.int32(127)
    # correction term, (k>0) ? 1-(u-x) : x-(u-1), over u; none for u = x.
    c = np.where(k > 0, one - (u - x), x - (u - one))
    c = np.where(through_sum, c / u, F(0.0))
    hu = hu & np.int32(0x007FFFFF)
    high = hu >= np.int32(0x3504F7)
    k = k + high.astype(np.int32)
    u = (hu | np.where(high, np.int32(0x3F000000), np.int32(0x3F800000))
         ).view(np.float32)                            # u or u/2, normalized
    hu = np.where(high, (np.int32(0x00800000) - hu) >> 2, hu)
    f = u - one
    hfsq = (F(0.5) * f) * f
    s = np.empty_like(f)
    z = np.empty_like(f)
    r = np.empty_like(f)
    _series_into(f, s, z, r)
    kf = k.astype(np.float32)
    t = s * (hfsq + r)
    # The C tests the reduced k: an argument whose 1+x normalizes with
    # k = 0 takes the k == 0 return and drops the correction term.
    value = np.where(
        k == 0, f - (hfsq - t),
        kf * _LOG1PF_LN2_HI - ((hfsq - (t + (kf * _LOG1PF_LN2_LO + c))) - f))
    # |f| < 2**-20, the mantissa landed on 1: the scalar routine's branch.
    for position in np.flatnonzero(hu == 0):
        value[position] = log1pf(x[position])
    return value
