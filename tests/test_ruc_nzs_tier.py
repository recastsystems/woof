"""The ``RUC_NZS`` tier ladder, and the proof it changed nothing at nine.

``woof/core/kernels/ruc.cu`` sized every per-thread soil scratch array with a
bare literal -- ``zshalf[9]``, ``cotso[9]``, ``dtdzs[14]`` -- and read its
level table from ``__constant__ real ruc_soil_layer_depth[9]``.  That was the
whole nine-level pin on the forecast column.  The lift compiles the module at
a geometry chosen by the caller, through the same
``woof.core.kernels.get_kernel_int_defines`` loader ``acoustic.cu``'s
``WPHI_MAX_LEV`` ladder uses.

The acceptance that matters is NEGATIVE: every nine-level configuration that
ran before must compile the same translation unit it compiled before.  Four
independent instruments say so here, and every one of them is CPU-only:

1. **The geometry mechanism is inert at nine.**  With snow explicitly at
   ``wrf_45``, :func:`ruc_module_defines` is EMPTY at 9, so the launcher takes
   the unspecialized loader and the string handed to
   NVRTC is byte-identical to ``module_source("ruc")`` -- the exact string the
   pre-ladder launcher produced.  Digested and compared.

2. **The lift is exactly a macro-for-literal substitution, PLUS one named
   fix.**  The shipped source is run BACKWARDS -- the sentinel-delimited
   blocks are removed or restored, and the five macros are replaced by the
   bare literals they expand to at nine -- and the result is hashed against
   :data:`PRE_LIFT_FILE_SHA256`, the digest ``FROZEN_MODULE_DIGESTS['ruc'][0]``
   carried on 29c337754 *before* this lane touched anything.  This test
   authenticates itself against the tree's own record.  If it is red, the lift
   changed something other than a level count and the re-pinned freeze digest
   is no longer justified.

   The "plus one named fix" is the ``RUC_NZS DZSTOP`` block, kept in its own
   sentinel and its own inversion step precisely so that it cannot hide
   inside the substitution.  ``ruc_soil_finalize`` computed ``dzstop = 1 /
   (0.01f - 0.0f)`` -- WRF's NINE-level ``zsmain(2) - zsmain(1)`` written out
   as a literal instead of read from the table -- and the macro sweep walked
   past it because it is a DEPTH, not an extent.  At six levels that divides
   by 0.01 where the geometry is 0.05, and the kernel returned a ground heat
   flux five times too large: measured, before the fix, as grdflx
   -337.1 W m-2 against the host lane's -67.4 on the same column.  It now
   reads ``ruc_soil_layer_depth[1] - [0]`` like every other site in the file.
   At nine those ARE 0.01f and 0.00f, so no number moves --
   ``tests/test_ruc_nzs_device.py`` measures that on the hardware -- but the
   PTX does move, a ``__constant__`` load where an immediate was.  That is
   why the ladder's no-op and identical-PTX claims below are measured
   against a source that already carries the fix, and why the fix has a
   generated-code test of its own.

3. **The generated code is identical**, measured twice on real tools: token
   streams out of a host C preprocessor, and PTX out of ``nvcc -ptx``.  Each
   has a negative control at ``-DRUC_NZS=6`` that must DIFFER, so neither
   comparison passes merely because it cannot fail.

4. **The no-arithmetic rule is mechanized.**  Every macro the ladder defines
   expands to a BARE DECIMAL LITERAL -- ``#define RUC_NZS_M2 7``, never
   ``#define RUC_NZS_M2 (RUC_NZS - 2)``.  The second form is correct
   arithmetic and expands to ``(9 - 2)`` where the pre-lift source had the
   single token ``7``, which is what would make (2) impossible.  Note the
   rule is about the DEFINITIONS: use sites such as ``RUC_NZS - step`` are
   required by the lift and expand to the same ``9 - step`` they always did.

Every test in this file is CPU-only and imports no CuPy -- which is why
:mod:`woof.core.ruc_tier` exists as its own module rather than living in
``woof.core.ruc_gpu``, whose module-scope ``import cupy`` would make this
whole file unimportable on a box with no card.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from woof.core.kernels import module_source
from woof.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     WRF_SUPPORTED_NUM_SOIL_LAYERS)
from woof.core.ruc_tier import ruc_kernel_source, ruc_module_defines

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "woof" / "core" / "kernels" / "ruc.cu"

#: ``sha256`` of ``woof/core/kernels/ruc.cu`` as it stood on 29c337754,
#: BEFORE the RUC_NZS lift.  Written out here rather than imported from
#: ``tests/test_mp8_frozen.py``, so that moving the freeze pin cannot make
#: this file agree with itself.  It is the digest the mp=8 freeze carried,
#: which is what makes the reconstruction below self-authenticating.
PRE_LIFT_FILE_SHA256 = (
    "d446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7")

#: The four lines the depth table was, before the ladder selected it.
PRE_LIFT_DEPTH_TABLE = """__constant__ real ruc_soil_layer_depth[9] = {
    0.00f, 0.01f, 0.04f, 0.10f, 0.30f,
    0.60f, 1.00f, 1.60f, 3.00f
};
"""

#: Each derived macro and the bare decimal literal it expands to at the
#: shipped geometry.  LONGEST FIRST: substituting ``RUC_NZS`` before
#: ``RUC_NZS_M1`` would turn ``RUC_NZS_M1`` into ``9_M1``.
SHIPPED_MACRO_LITERALS = (
    ("RUC_NZS_M1", "8"),
    ("RUC_NZS_M2", "7"),
    ("RUC_NZS_M3", "6"),
    ("RUC_DTDZS_LEN", "14"),
    ("RUC_NZS", "9"),
)

TIER_LADDER = "RUC_NZS TIER LADDER"
DEPTH_TABLE = "RUC_NZS DEPTH TABLE"
DZSTOP = "RUC_NZS DZSTOP"

#: The one line ``ruc_soil_finalize``'s ``dzstop`` was before the fix.
#: Inverted SEPARATELY from the ladder, because it is the only edit this lane
#: made to ``ruc.cu`` that is not a macro-for-literal substitution: it changes
#: the generated code at nine while leaving every number identical.  Folding
#: it into :func:`_reconstruct_pre_lift` would let the ladder's "preprocessor
#: no-op" and "identical PTX" claims quietly cover a change that is neither.
PRE_FIX_DZSTOP = (
    "    const real dzstop = __fdiv_rn(one, __fsub_rn(0.01f, 0.0f));" + chr(10))


def _shipped() -> str:
    """``ruc.cu`` exactly as it sits on disk, with no newline translation."""
    # read_bytes rather than read_text(newline=""): the keyword arrived in
    # Python 3.13, and the bytes route keeps every newline exactly as well.
    return KERNEL.read_bytes().decode("utf-8")


def _block(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"// >>> {re.escape(name)} >>>\n.*?// <<< {re.escape(name)} <<<\n",
        re.S)


def _drop_sentinel_block(text: str, name: str) -> str:
    """Remove a sentinel block AND the blank line that separates it."""
    pattern = re.compile(_block(name).pattern + "\n", re.S)
    out, count = pattern.subn("", text)
    assert count == 1, f"expected exactly one {name} block, found {count}"
    return out


def _replace_sentinel_block(text: str, name: str, body: str) -> str:
    out, count = _block(name).subn(lambda _match: body, text)
    assert count == 1, f"expected exactly one {name} block, found {count}"
    return out


def _reconstruct_pre_lift(text: str) -> str:
    """Invert the LIFT only: sentinels out, macros back to bare literals.

    Deliberately does NOT invert the ``RUC_NZS DZSTOP`` fix, so that
    the token-stream and PTX comparisons below compare two sources
    that both carry it.  What they measure is then the ladder, which
    is what they claim to measure.
    """
    text = _drop_sentinel_block(text, TIER_LADDER)
    text = _replace_sentinel_block(text, DEPTH_TABLE, PRE_LIFT_DEPTH_TABLE)
    for macro, literal in SHIPPED_MACRO_LITERALS:
        text = re.sub(rf"\b{macro}\b", literal, text)
    return text


#: The SOILPROP lineage switch (ruc_soilprop, 2.8.5), each edit with the
#: v4.6.1 text it replaced.  Oracle-verified on its own
#: (tests/test_ruc_soilprop.py, the v4.6.1 name against the unmodified WRF
#: oracle), so it is undone here before the historical geometry proof.
SOILPROP_LINEAGE_EDITS = (
    ("""#ifdef GPUWM_SOILPROP_WRF461
    real mineral = qwrtz > 0.2f ? 2.0f : 3.0f;
#else
    real mineral = 2.0f;
#endif
""", """    real mineral = qwrtz > 0.2f ? 2.0f : 3.0f;
"""),
    ("""#ifdef GPUWM_SOILPROP_WRF461
            real h = fmaxf(
                0.0f,
                __fdiv_rn(
                    __fsub_rn(__fadd_rn(middle_moisture, qmin), ice),
                    fmaxf(minimum, __fsub_rn(ws, ice))));
            real porosity = ws;
#else
            real h = fmaxf(
                0.0f,
                __fdiv_rn(
                    __fsub_rn(middle_moisture, ice),
                    fmaxf(minimum, __fsub_rn(dqm, ice))));
            real porosity = dqm;
#endif
""", """            real h = fmaxf(
                0.0f,
                __fdiv_rn(
                    __fsub_rn(__fadd_rn(middle_moisture, qmin), ice),
                    fmaxf(minimum, __fsub_rn(ws, ice))));
"""),
    ("""            real ame = fmaxf(minimum, __fsub_rn(porosity, ice));
""", """            real ame = fmaxf(minimum, __fsub_rn(ws, ice));
"""),
    ("""                diffusivity, ruc_powf_rn(__fdiv_rn(porosity, ame), 3.0f));
""", """                diffusivity, ruc_powf_rn(__fdiv_rn(ws, ame), 3.0f));
"""),
    ("""#ifdef GPUWM_SOILPROP_WRF461
        real am = fmaxf(minimum, __fsub_rn(ws, ice));
#else
        real am = fmaxf(minimum, __fsub_rn(dqm, ice));
#endif
""", """        real am = fmaxf(minimum, __fsub_rn(ws, ice));
"""),
)


#: RUC's float32 libm words (development branch): every EXP, LOG, LOG10,
#: TANH, COS and REAL**REAL in ruc.cu moved from float64-rounded-once (and
#: CUDA's logf/cosf) to WOOF's float32 routines, each hunk with the text it
#: replaced, generated from the diff.  Oracle-verified on its own
#: (tools/ruc_lsm_gpu_oracle, tests/test_ruc_gpu_column_oracle.py and
#: tests/test_ruc_libm_words.py), so it is undone here, first, before every
#: historical reconstruction.
LIBM_WORDS_EDITS = (
    ("\n// WOOF's float32 libm words for RUC (lane/verify-ruc-lsm).  gfortran lowers\n// `**`, EXP, LOG and LOG10 on a default REAL to the C library's float32\n// powf/expf/logf/log10f, and the RUC column oracle against WRF v4.6.1\n// (tools/ruc_lsm_gpu_oracle) measured the earlier float64-rounded-once\n// stand-ins missing the reference words by 1 ULP on scattered columns every\n// step (grdflx, hfx, qfx, the saturation humidities).  These are WOOF's own\n// float32 routines -- gfk_pow / gfk_exp / gfk_log from glibc_flt32.cuh, and\n// the log10f, expm1f and tanhf reductions the MYNN unit already grades\n// bitwise against its gfortran oracle -- so RUC calls the same functions\n// every other bitwise-graded unit calls.  The names are kept so the host\n// mirror in gpuwm/core/ruc.py and the generated fused sources still bind.\n__device__ __forceinline__\n",
     '\n__device__ __forceinline__\n'),
    ('{\n    return gfk_pow(base, exponent);\n}\n',
     '{\n    return __double2float_rn(pow((double)base, (double)exponent));\n}\n'),
    ("\n// log10f on WOOF's float32 logf: the exponent split, then\n// y*log10_2lo + ivln10*log(m) + y*log10_2hi, each step rounded.\n__device__ real ruc_log10f_rn(real x)\n{\n",
     '\n__device__ __forceinline__\nreal ruc_log10f_rn(real value)\n{\n'),
    ('{\n    const real ivln10 = __int_as_float(0x3ede5bd9);\n    const real log10_2hi = __int_as_float(0x3e9a2080);\n    const real log10_2lo = __int_as_float(0x355427db);\n    const real two25 = __int_as_float(0x4c000000);\n    unsigned int hx = __float_as_uint(x);\n    int k = 0;\n    if ((int)hx < 0x00800000) {\n        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);\n        if ((int)hx < 0) return __int_as_float(0x7fc00000);\n        k -= 25;\n        x = FMUL(x, two25);\n        hx = __float_as_uint(x);\n    }\n    if (hx >= 0x7f800000u) return FADD(x, x);\n    k += (int)(hx >> 23) - 127;\n    int i = (k < 0) ? 1 : 0;\n    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);\n    real y = (real)(k + i);\n    x = __uint_as_float(hx);\n    real z = FADD(FMUL(y, log10_2lo), FMUL(ivln10, gfk_log(x)));\n    return FADD(z, FMUL(y, log10_2hi));\n}\n',
     '{\n    return __double2float_rn(log10((double)value));\n}\n'),
    ('{\n    return gfk_exp(value);\n}\n',
     '{\n    return __double2float_rn(exp((double)value));\n}\n'),
    ('        real temperature = tso[index];\n        real tln = gfk_log(__fdiv_rn(temperature, freeze));\n        if (tln < zero) {\n',
     '        real temperature = tso[index];\n        real tln = logf(__fdiv_rn(temperature, freeze));\n        if (tln < zero) {\n'),
    ("            base = __fdiv_rn(base, psis);\n            // WOOF's float32 powf (gfk_pow), not CUDA powf: CUDA's drifts\n            // ~4 ULP into soilice on cold deep layers relative to gfortran.\n            real liquid = __fsub_rn(\n",
     '            base = __fdiv_rn(base, psis);\n            // ruc_powf_rn, not powf: plain CUDA powf drifts ~4 ULP into\n            // soilice on cold deep layers relative to gfortran.  See the\n            // provisional-transcendental note at the top of this file.\n            real liquid = __fsub_rn(\n'),
    ('                __fmul_rn(\n                    maximum,\n                    gfk_pow(base, exponent)),\n                qmin);\n            liquid = fmaxf(zero, liquid);\n',
     '                __fmul_rn(\n                    maximum,\n                    __double2float_rn(pow((double)base, (double)exponent))),\n                qmin);\n            liquid = fmaxf(zero, liquid);\n'),
    ('        soilmoism[index] = middle_moisture;\n        real tln = gfk_log(__fdiv_rn(middle_temperature, freeze));\n        if (tln < zero) {\n',
     '        soilmoism[index] = middle_moisture;\n        real tln = logf(__fdiv_rn(middle_temperature, freeze));\n        if (tln < zero) {\n'),
    ('            base = __fdiv_rn(base, psis);\n            // The same float32 powf as the full-level loop above.\n            real liquid = __fsub_rn(\n',
     '            base = __fdiv_rn(base, psis);\n            // Same ruc_powf_rn substitution as the full-level loop above.\n            real liquid = __fsub_rn(\n'),
    ('                __fmul_rn(\n                    maximum,\n                    gfk_pow(base, exponent)),\n                qmin);\n            fwsat[index] = __fsub_rn(dqm, liquid);\n',
     '                __fmul_rn(\n                    maximum,\n                    __double2float_rn(pow((double)base, (double)exponent))),\n                qmin);\n            fwsat[index] = __fsub_rn(dqm, liquid);\n'),
    ("        fex = fmaxf(0.01f, fminf(one, fex));\n        // COS on a default REAL is the C library's float32 cosf; CUDA's\n        // cosf is a different function (the RUC oracle's dry desert column\n        // took a 1 ULP soilres miss into mavail, qfx and the top soil water).\n        // glibc_cosf is WOOF's float32 cosf (glibc_trig_flt32.cuh).\n        real resistance = __fsub_rn(\n",
     '        fex = fmaxf(0.01f, fminf(one, fex));\n        real resistance = __fsub_rn(\n'),
    ('        real resistance = __fsub_rn(\n            one, glibc_cosf(__fmul_rn(3.141592653589793f, fex)));\n        resistance = __fmul_rn(resistance, resistance);\n',
     '        real resistance = __fsub_rn(\n            one, cosf(__fmul_rn(3.141592653589793f, fex)));\n        resistance = __fmul_rn(resistance, resistance);\n'),
    ('{\n    return gfk_exp(x);\n}\n',
     '{\n    return (real)exp((double)x);\n}\n'),
    ("\n// expm1f in float32: the k*ln2 split, the five-term rational in hxs and the\n// exponent rebuild, every operation rounded.  The same routine as MYNN's\n// mynn_expm1f.\n__device__ __forceinline__ real ruc_scale_exponent(real y, int k)\n{\n",
     '\n__device__ __forceinline__\nreal ruc_expm1f_glibc(real x)\n{\n'),
    ('{\n    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));\n}\n',
     '{\n    return (real)expm1((double)x);\n}\n'),
    ("\n__device__ real ruc_expm1f_glibc(real x)\n{\n    const real ln2_hi = __uint_as_float(0x3F317180u);\n    const real ln2_lo = __uint_as_float(0x3717F7D1u);\n    const real invln2 = __uint_as_float(0x3FB8AA3Bu);\n    const real q1 = __uint_as_float(0xBD088889u);\n    const real q2 = __uint_as_float(0x3AD00D01u);\n    const real q3 = __uint_as_float(0xB8A670CDu);\n    const real q4 = __uint_as_float(0x36867E54u);\n    const real q5 = __uint_as_float(0xB457EDBBu);\n    const real tiny = 1.0e-30f;\n\n    unsigned word = __float_as_uint(x);\n    unsigned sign = word & 0x80000000u;\n    unsigned magnitude = word & 0x7FFFFFFFu;\n    if (magnitude >= 0x4195B844u) {              // |x| >= 27*ln2\n        if (magnitude >= 0x42B17218u) {          // |x| >= 88.72\n            if (magnitude > 0x7F800000u) return FADD(x, x);\n            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;\n            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);\n        }\n        if (sign != 0u) return FSUB(tiny, 1.0f);\n    }\n    int k;\n    real correction;\n    if (magnitude > 0x3EB17218u) {               // |x| > 0.5*ln2\n        real hi, lo;\n        if (magnitude < 0x3F851592u) {           // |x| < 1.5*ln2\n            if (sign == 0u) {\n                hi = FSUB(x, ln2_hi); lo = ln2_lo; k = 1;\n            } else {\n                hi = FADD(x, ln2_hi); lo = -ln2_lo; k = -1;\n            }\n        } else {\n            k = (int)FADD(FMUL(invln2, x), sign == 0u ? 0.5f : -0.5f);\n            real scale = (real)k;\n            hi = FSUB(x, FMUL(scale, ln2_hi));\n            lo = FMUL(scale, ln2_lo);\n        }\n        x = FSUB(hi, lo);\n        correction = FSUB(FSUB(hi, x), lo);\n    } else if (magnitude < 0x33000000u) {        // |x| < 2**-25\n        return x;\n    } else {\n        k = 0;\n        correction = 0.0f;\n    }\n\n    real hfx = FMUL(0.5f, x);\n    real hxs = FMUL(x, hfx);\n    real r1 = FADD(1.0f, FMUL(hxs, FADD(q1, FMUL(hxs,\n        FADD(q2, FMUL(hxs, FADD(q3, FMUL(hxs,\n            FADD(q4, FMUL(hxs, q5))))))))));\n    real t = FSUB(3.0f, FMUL(r1, hfx));\n    real e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));\n    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));\n    e = FSUB(FMUL(x, FSUB(e, correction)), correction);\n    e = FSUB(e, hxs);\n    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);\n    if (k == 1) {\n        if (x < -0.25f)\n            return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));\n        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));\n    }\n    real y;\n    if (k <= -2 || k > 56) {\n        y = FSUB(1.0f, FSUB(e, x));\n        y = ruc_scale_exponent(y, k);\n        return FSUB(y, 1.0f);\n    }\n    if (k < 23) {\n        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));\n        y = FSUB(t, FSUB(e, x));\n    } else {\n        t = __uint_as_float((unsigned)(0x7F - k) << 23);\n        y = FSUB(x, FADD(e, t));\n        y = FADD(y, 1.0f);\n    }\n    return ruc_scale_exponent(y, k);\n}\n\n// tanhf in float32 on ruc_expm1f_glibc: 1 - 2/(expm1(2|x|)+2) at |x| >= 1,\n// -t/(t+2) with t = expm1(-2|x|) below, x*(1+x) under 2**-55 and 1 beyond\n// 22.  The same routine as MYNN's mynn_tanhf.  The earlier body ran this\n// reduction on a float64 expm1 rounded once, which is a different expm1f:\n// the RUC oracle measured it 1-2 ULP off the reference TANH in the new-snow\n// density (module_sf_ruclsm.F:1520-1521) and so in rhosnf and snowfallac.\n// gpuwm.core.ruc._f32_tanh is the host mirror.\n__device__ __forceinline__\n",
     "\n// glibc's tanhf, unlike its expf, is still fdlibm's expm1-based reduction\n// evaluated in float32 and is NOT correctly rounded - on this lane's fixture\n// it lands 2 ULP above the correctly rounded value for one snow-fraction\n// argument.  The reduction is therefore spelled out here exactly as in\n// gpuwm.core.ruc._f32_tanh, so host and device share one definition instead of\n// inheriting two different libm implementations.\n__device__ __forceinline__\n"),
    ('{\n    const real tiny = 1.0e-30f;\n    unsigned word = __float_as_uint(x);\n    unsigned magnitude = word & 0x7FFFFFFFu;\n    real z;\n',
     '{\n    const real one = 1.0f;\n    const real two = 2.0f;\n    real magnitude = fabsf(x);\n    real z;\n'),
    ('    real z;\n    if (magnitude >= 0x7F800000u)                // inf or NaN: one/x +- one\n        return (word & 0x80000000u) == 0u ? FADD(FDIV(1.0f, x), 1.0f)\n                                          : FSUB(FDIV(1.0f, x), 1.0f);\n    if (magnitude < 0x41B00000u) {               // |x| < 22\n        if (magnitude < 0x24000000u)             // |x| < 2**-55\n            return FMUL(x, FADD(1.0f, x));\n        real ax = __uint_as_float(magnitude);\n        if (magnitude >= 0x3F800000u) {          // |x| >= 1\n            real t = ruc_expm1f_glibc(FMUL(2.0f, ax));\n            z = FSUB(1.0f, FDIV(2.0f, FADD(t, 2.0f)));\n        } else {\n',
     "    real z;\n    if (magnitude < 22.0f) {\n        if (magnitude < 3.7252902984619141e-09f) {\n            // tanh(tiny) == tiny, in fdlibm's inexact-flag form.\n            return __fmul_rn(x, __fadd_rn(one, x));\n        }\n        real doubled = __fmul_rn(two, magnitude);\n        real t;\n        if (magnitude >= one) {\n            t = ruc_expm1f_glibc(doubled);\n            z = __fsub_rn(one, __fdiv_rn(two, __fadd_rn(t, two)));\n        } else {\n"),
    ('        } else {\n            real t = ruc_expm1f_glibc(FMUL(-2.0f, ax));\n            z = FDIV(-t, FADD(t, 2.0f));\n        }\n',
     '        } else {\n            t = ruc_expm1f_glibc(-doubled);\n            z = __fdiv_rn(-t, __fadd_rn(t, two));\n        }\n'),
    ('    } else {\n        z = FSUB(1.0f, tiny);\n    }\n',
     '    } else {\n        // fdlibm returns one-tiny here, which rounds to exactly one.\n        z = one;\n    }\n'),
    ('    }\n    return (word & 0x80000000u) == 0u ? z : -z;\n}\n',
     '    }\n    return (x >= 0.0f) ? z : -z;\n}\n'),
)



#: The snow lineage switch (ruc_snow, 2.8.6), each hunk with the v4.6.1
#: text it replaced, generated from the diff and kept verbatim.  Both arms
#: are oracle-verified on their own (tests/test_ruc.py and test_ruc_gpu.py
#: under wrf_461, tests/test_ruc_fork_oracle.py under wrf_45), so the switch
#: is undone here before the historical geometry proof.
SNOW_LINEAGE_EDITS = (
    ("""// module_sf_ruclsm.F:63-69 sncovfac, read only when isncovr_opt==3.
// The snow scheme by WRF lineage (ruc_snow, gpuwm/core/ruc_tier.py).  The
// default compile is WRF v4.0-4.5, which the operational RAP/HRRR branch
// carries; GPUWM_SNOW_WRF461 compiles WRF v4.6.1.  GPUWM_RUC_SNOW_V461 is the same
// choice as a value, for the arms written as conditions; the generated
// fused sfctmp kernel reads it too.
#ifdef GPUWM_SNOW_WRF461
#define GPUWM_RUC_SNOW_V461 true
#else
#define GPUWM_RUC_SNOW_V461 false
#endif

__device__ static const real ruc_sncovfac[30] = {
""",
     """// module_sf_ruclsm.F:63-69 sncovfac, read only when isncovr_opt==3.
__device__ static const real ruc_sncovfac[30] = {
"""),
    ("""    // :1504 - the mosaic flag from the previous step's snow fraction.
    if (GPUWM_RUC_SNOW_V461 && snowfrac < 0.75f) snow_mosaic = one;

""",
     """    // :1504 - the mosaic flag from the previous step's snow fraction.
    if (snowfrac < 0.75f) snow_mosaic = one;

"""),
    ("""
#ifndef GPUWM_SNOW_WRF461
    // Branch :1675-1679: the critical depths from the density this step
    // built, and the mosaic flag from the depth before new snow.
    snhei_crit = __fdiv_rn(critical_depth_coefficient, rhosn);
    snhei_crit_newsn = __fdiv_rn(new_snow_depth_coefficient, rhosn);
    snowfrac = fminf(one, __fdiv_rn(snhei, __fmul_rn(2.0f, snhei_crit)));
    if (snowfrac < 0.75f) snow_mosaic = one;
#endif

    // :1580-1598 fresh snow onto the ground.
""",
     """
    // :1580-1598 fresh snow onto the ground.
"""),
    ("""        iland = isice;
#ifndef GPUWM_SNOW_WRF461
        // Branch :1708-1719.  isncovr_opt is not read.
        snowfrac = fminf(one, __fdiv_rn(snhei, __fmul_rn(2.0f, snhei_crit)));
        if (ivgtyp == urban) snowfrac = fminf(0.75f, snowfrac);
        if (snowfrac < 0.75f) snow_mosaic = one;
        if (newsn > zero) {
            snowfracnewsn = fminf(one, __fdiv_rn(snhei, snhei_crit_newsn));
        }
        keep_snow_albedo = zero;
        if (newsn > zero && snowfracnewsn > 0.99f) {
            keep_snow_albedo = one;
            snow_mosaic = zero;
        }
#else
        if (isncovr_opt == 1) {
""",
     """        iland = isice;
        if (isncovr_opt == 1) {
"""),
    ("""        }
#endif
        // :1672-1680 roughness blend toward the snow/ice class.
""",
     """        }
        // :1672-1680 roughness blend toward the snow/ice class.
"""),
    ("""                // pass.  Transcribed to stay faithful.
                // The branch has no 0.7 floor, here or below.
                if (GPUWM_RUC_SNOW_V461 && keep_snow_albedo > 0.9f && albsn < 0.4f) {
                    albsn = 0.7f;
                }
                emiss = emissn;
""",
     """                // pass.  Transcribed to stay faithful.
                if (keep_snow_albedo > 0.9f && albsn < 0.4f) albsn = 0.7f;
                emiss = emissn;
"""),
    ("""                        alb_snow));
                if (GPUWM_RUC_SNOW_V461 && newsn > zero && keep_snow_albedo > 0.9f
                    && albsn < 0.4f) {
                    albsn = 0.7f;
""",
     """                        alb_snow));
                if (newsn > zero && keep_snow_albedo > 0.9f && albsn < 0.4f) {
                    albsn = 0.7f;
"""),
    ("""
// module_sf_ruclsm.F:5046-5072, repeated verbatim at :5587-5610.  The snow
// scheme by WRF lineage (ruc_snow, gpuwm/core/ruc_tier.py): the default is
// WRF v4.0-4.5, which the operational RAP/HRRR branch carries (branch
// :5235, :5738): a constant conductivity, thdifsn = 0.265/rhocsn.
// GPUWM_SNOW_WRF461 compiles WRF v4.6.1, whose :49 fixes isncond_opt = 2:
// the Sturm et al. (1997) effective conductivity.
__device__ __forceinline__
""",
     """
// module_sf_ruclsm.F:5046-5072, repeated verbatim at :5587-5610.  :49 fixes
// isncond_opt = 2, so the constant 0.265/rhocsn branch is dead and the Sturm
// et al. (1997) effective conductivity always applies.
__device__ __forceinline__
"""),
    ("""{
#ifndef GPUWM_SNOW_WRF461
    return __fdiv_rn(0.265f, rhocsn);
#endif
    const real fact = 1.0f;
""",
     """{
    const real fact = 1.0f;
"""),
    ("""        soilt = ts1;
        // wrf_45 (the branch) has neither freezing clamp on the second pass.
        if (GPUWM_RUC_SNOW_V461 && nmelt == 1 && snowfrac == one && snwe > zero
            && soilt > freeze) {
            soilt = fminf(freeze, soilt);
""",
     """        soilt = ts1;
        if (nmelt == 1 && snowfrac == one && snwe > zero && soilt > freeze) {
            soilt = fminf(freeze, soilt);
"""),
    ("""        }
        if (GPUWM_RUC_SNOW_V461 && nmelt == 1 && snowfrac == one) {
            soilt1 = fminf(freeze, soilt1);
""",
     """        }
        if (nmelt == 1 && snowfrac == one) {
            soilt1 = fminf(freeze, soilt1);
"""),
    ("""
#ifdef GPUWM_SNOW_WRF461
        bool melts = soilt > freeze && beta == one && snhei > zero;
#else
        // Branch :5586: melt while the pack outlasts the step's evaporation.
        bool melts = soilt > freeze
            && __fsub_rn(
                   snwepr,
                   __fmul_rn(__fmul_rn(__fmul_rn(beta, epot), ras), delt))
               > zero
            && snhei > zero;
#endif
        if (melts) {
            // :5414-5553 top melt.
""",
     """
        if (soilt > freeze && beta == one && snhei > zero) {
            // :5414-5553 top melt.
"""),
    ("""                qsg, __fdiv_rn(ruc_qsn_lookup(soiltfrac, tbq), pp));
#ifdef GPUWM_SNOW_WRF461
            qvg = __fadd_rn(
""",
     """                qsg, __fdiv_rn(ruc_qsn_lookup(soiltfrac, tbq), pp));
            qvg = __fadd_rn(
"""),
    ("""                __fmul_rn(__fsub_rn(one, snowfrac), qvg));
#else
            qvg = qsg;  // branch :5590, saturated at melt
#endif
            // :5419-5421 t3/upflux/xinet are dead: xinet is never read.
""",
     """                __fmul_rn(__fsub_rn(one, snowfrac), qvg));
            // :5419-5421 t3/upflux/xinet are dead: xinet is never read.
"""),
    ("""            smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), milli);
            // :5530 the Koren et al. (1999) retained fraction, both lineages.
            real rsmfrac = fminf(
                0.18f,
                fmaxf(
                    0.08f,
                    __fmul_rn(__fdiv_rn(snwepr, 0.10f), 0.13f)));
#ifndef GPUWM_SNOW_WRF461
            {
                // Branch :5652-5698, straight-line: the cap does not scale
                // with the step or depend on density, and liquid is
                // retained whenever the pack is deeper than 1 cm.
                real available = __fsub_rn(
                    __fdiv_rn(snwepr, delt),
                    __fmul_rn(__fmul_rn(beta, epot), ras));
                smelt = fminf(smelt, available);
                smelt = fmaxf(zero, smelt);
                real limit = __fmul_rn(5.6e-8f, meltfactor);
                limit = __fmul_rn(
                    limit, fmaxf(one, __fsub_rn(soilt, freeze)));
                smelt = fminf(smelt, limit);
                real rr = fmaxf(zero, available);
                smelt = fminf(smelt, rr);
                snoh = __fmul_rn(__fmul_rn(smelt, xlmelt), thousand);
                if (snhei > 0.01f) {
                    rsm = __fmul_rn(__fmul_rn(rsmfrac, smelt), delt);
                } else {
                    rsm = zero;
                }
                smelt = fmaxf(zero, __fsub_rn(smelt, __fdiv_rn(rsm, delt)));
                snwe = fmaxf(
                    zero,
                    __fsub_rn(
                        snwepr,
                        __fmul_rn(
                            __fadd_rn(
                                smelt,
                                __fmul_rn(__fmul_rn(beta, epot), ras)),
                            delt)));
            }
#else
            real potential = __fmul_rn(__fmul_rn(epot, ras), delt);
""",
     """            smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), milli);
            real potential = __fmul_rn(__fmul_rn(epot, ras), delt);
"""),
    ("""                // :5529-5543 Koren et al. (1999) liquid retention.
                if (snhei > 0.01f && rhosn < 350.0f) {
""",
     """                // :5529-5543 Koren et al. (1999) liquid retention.
                real rsmfrac = fminf(
                    0.18f,
                    fmaxf(
                        0.08f,
                        __fmul_rn(__fdiv_rn(snwepr, 0.10f), 0.13f)));
                if (snhei > 0.01f && rhosn < 350.0f) {
"""),
    ("""            }
#endif
        } else {
            // :5557-5567 no melt: sublimation or condensation only.  The
            // branch updates any pack and never zeroes one here.
            if (snhei != zero && (beta == one || !GPUWM_RUC_SNOW_V461)) {
                epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
""",
     """            }
        } else {
            // :5557-5567 no melt: sublimation or condensation only.
            if (snhei != zero && beta == one) {
                epot = -__fmul_rn(qkms, __fsub_rn(qvatm, qsg));
"""),
    ("""                            __fmul_rn(__fmul_rn(beta, epot), ras), delt)));
            } else if (GPUWM_RUC_SNOW_V461) {
                snwe = zero;
""",
     """                            __fmul_rn(__fmul_rn(beta, epot), ras), delt)));
            } else {
                snwe = zero;
"""),
    ("""        real smeltg = __fmul_rn(__fdiv_rn(snohg, xlmelt), milli);
        // :5658-5660 the Egglston bottom-melt limit; unconditional in the
        // branch.
        if (!GPUWM_RUC_SNOW_V461
            || ((rhosn < 350.0f || (newsnow > zero && rhonewsn < 450.0f))
                && soilt < 283.0f)) {
            smeltg = fminf(smeltg, 5.8e-9f);
""",
     """        real smeltg = __fmul_rn(__fdiv_rn(snohg, xlmelt), milli);
        // :5658-5660 the Egglston bottom-melt limit.
        if ((rhosn < 350.0f || (newsnow > zero && rhonewsn < 450.0f))
            && soilt < 283.0f) {
            smeltg = fminf(smeltg, 5.8e-9f);
"""),
    ("""        snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);
        // The branch keeps the bottom melt water out of smelt.
        if (GPUWM_RUC_SNOW_V461) smelt = __fadd_rn(smelt, smeltg);
        if (snhei > zero) tso[0] = soiltfrac;
""",
     """        snhei = __fdiv_rn(__fmul_rn(snwe, thousand), rhosn);
        smelt = __fadd_rn(smelt, smeltg);
        if (snhei > zero) tso[0] = soiltfrac;
"""),
)


def _reconstruct_pre_libm_words(text: str) -> str:
    """Undo the libm-words edit, hunk by hunk."""
    for shipped, original in LIBM_WORDS_EDITS:
        assert text.count(shipped) == 1, shipped
        text = text.replace(shipped, original)
    return text


def _reconstruct_pre_snow(text: str) -> str:
    """Undo the libm-words edit, then the snow lineage switch, hunk by hunk."""
    text = _reconstruct_pre_libm_words(text)
    for shipped, original in SNOW_LINEAGE_EDITS:
        assert text.count(shipped) == 1, shipped
        text = text.replace(shipped, original)
    return text


def _reconstruct_pre_soilprop(text: str) -> str:
    """Undo the SOILPROP lineage switch: its comment block and five edits."""
    start = text.index("    // SOILPROP by WRF lineage (ruc_soilprop")
    end = text.index("#ifdef GPUWM_SOILPROP_WRF461", start)
    text = text[:start] + text[end:]
    for shipped, original in SOILPROP_LINEAGE_EDITS:
        assert text.count(shipped) == 1, shipped
        text = text.replace(shipped, original)
    return text


def _reconstruct_pre_mosaic(text: str) -> str:
    """Undo the separately oracle-verified mosaic surface port before history checks.

    The rest of the historical geometry proof remains pinned to its original
    hash. This prevents unrelated leaf edits hiding inside a mosaic update.
    """
    text = _reconstruct_pre_soilprop(_reconstruct_pre_snow(text))
    previous = (ROOT / "tests/data/ruc_surface_pre_mosaic.cu").read_text()
    text = _replace_sentinel_block(text, "RUC MOSAIC SURFACE", previous + "\n")
    current = (
        "// WRF v4.6.1 RUC LSM surface/soil parameter setup.\n"
        "// Public-domain WRF transcription: licenses/LICENSE-WRF-public-domain.txt.\n"
        "// One thread transcribes one call to module_sf_ruclsm.F:soilvegin.  Explicit round-to-nearest intrinsics keep\n")
    original = (
        "// WRF v4.6.1 RUC LSM dominant-category surface/soil parameter setup.\n"
        "// One thread transcribes one call to module_sf_ruclsm.F:soilvegin with\n"
        "// mosaic_lu=0 and mosaic_soil=0.  Explicit round-to-nearest intrinsics keep\n")
    assert text.startswith(current)
    return original + text[len(current):]


def _reconstruct_pre_fix(text: str) -> str:
    """Undo the one named non-substitution edit, and only it.

    Runs BEFORE :func:`_reconstruct_pre_lift` on any composed inversion:
    the macro pass rewrites every ``RUC_NZS`` token in the file, the
    sentinel names included, so a block looked up by name has to be
    resolved while the name is still spelled the way the source spells it.
    """
    return _replace_sentinel_block(text, DZSTOP, PRE_FIX_DZSTOP)


# ---------------------------------------------------------------------------
# 1. The tier itself
# ---------------------------------------------------------------------------

def test_the_shipped_geometry_injects_no_define_at_all():
    """Isolate the geometry ladder from the separately selected snow form."""
    assert ruc_module_defines(NUM_SOIL_LAYERS, snow="wrf_45") == ()
    assert NUM_SOIL_LAYERS == 9


def test_the_six_level_geometry_asks_for_exactly_one_define():
    assert ruc_module_defines(6, snow="wrf_45") == (("RUC_NZS", 6),)


@pytest.mark.parametrize("nzs", [0, 1, 4, 5, 7, 8, 10, 12, -9])
def test_a_geometry_wrf_does_not_define_is_refused(nzs):
    with pytest.raises(ValueError, match="is not one of"):
        ruc_module_defines(nzs)


def test_every_admitted_geometry_has_a_tier():
    for count in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        defines = ruc_module_defines(count, snow="wrf_45")
        assert defines == (() if count == NUM_SOIL_LAYERS
                           else (("RUC_NZS", count),))


def test_the_nine_level_source_is_the_unspecialized_module_byte_for_byte():
    """Leg 1: geometry adds no byte at nine under the same snow selection."""
    generated = ruc_kernel_source(NUM_SOIL_LAYERS, snow="wrf_45")
    unspecialized = module_source("ruc")
    assert generated == unspecialized
    assert (hashlib.sha256(generated.encode("utf-8")).hexdigest()
            == hashlib.sha256(unspecialized.encode("utf-8")).hexdigest())
    assert "#define RUC_NZS 6" not in generated


def test_a_six_level_source_adds_exactly_one_line_and_nothing_else():
    """Mutation control: the comparison above CAN fail.

    Removing the one injected define must recover the unspecialized string
    byte for byte, so a tiered compile cannot smuggle in any other edit.
    """
    generated = ruc_kernel_source(6, snow="wrf_45")
    unspecialized = module_source("ruc")
    assert generated != unspecialized
    injected = "#define RUC_NZS 6\n"
    assert generated.count(injected) == 1
    assert generated.replace(injected, "", 1) == unspecialized


# ---------------------------------------------------------------------------
# 2. The lift is exactly a macro-for-literal substitution
# ---------------------------------------------------------------------------

def test_the_lift_is_exactly_a_macro_for_literal_substitution():
    """Reconstruct the pre-lift ruc.cu FROM the shipped one and hash it.

    The lift is a pure textual substitution: five macros, each expanding to
    one bare decimal literal, plus two sentinel-delimited blocks.  Inverting
    it must reproduce the file the mp=8 freeze pinned BEFORE the lift, byte
    for byte.

    If this is red, the lift changed something other than a level count --
    reformatting, re-wrapping, a "while I'm here" fix -- and the re-pinned
    freeze digest is no longer justified by anything.
    """
    reconstructed = _reconstruct_pre_lift(_reconstruct_pre_fix(_reconstruct_pre_mosaic(_shipped())))
    digest = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    assert digest == PRE_LIFT_FILE_SHA256, (
        "the shipped ruc.cu does not invert to the pre-lift file.  Either a "
        "non-substitution edit entered the lift without a sentinel of its "
        "own, or a macro maps to a different literal than "
        "SHIPPED_MACRO_LITERALS claims")


def test_the_named_fix_is_the_only_non_substitution_edit():
    """Inverting the ladder ALONE must NOT reach the pre-lift file.

    The digest test above passes through two inversions.  Without this, a
    second undeclared edit could be hiding inside the DZSTOP sentinel and
    the pair would still agree.  This pins the split itself: the ladder
    inversion leaves exactly one difference, and it is the dzstop line.
    """
    shipped = _shipped()
    ladder_only = _reconstruct_pre_lift(shipped)
    assert PRE_FIX_DZSTOP not in ladder_only
    assert hashlib.sha256(
        ladder_only.encode("utf-8")).hexdigest() != PRE_LIFT_FILE_SHA256, (
        "inverting the ladder alone reached the pre-lift file, so the "
        "dzstop fix is not in the shipped source at all")

    fix_only = _reconstruct_pre_fix(shipped)
    assert PRE_FIX_DZSTOP in fix_only
    assert hashlib.sha256(_reconstruct_pre_lift(_reconstruct_pre_mosaic(fix_only)).encode(
        "utf-8")).hexdigest() == PRE_LIFT_FILE_SHA256
    # The difference between the shipped file and that one is EXACTLY the
    # sentinel block -- nothing was smuggled in beside it.
    assert shipped.replace(
        _block(DZSTOP).search(shipped).group(0), PRE_FIX_DZSTOP) == fix_only


def test_the_reconstruction_can_fail():
    """Negative control for the test above.

    A one-character change to the shipped source must break the digest.
    Without this, a reconstruction that accidentally normalised the file
    would pass and prove nothing.

    The mutation is deliberately made OUTSIDE both sentinel blocks: a change
    inside one of them is *supposed* to vanish under reconstruction, so
    mutating there would test nothing.
    """
    shipped = _shipped()
    anchor = "const real* zsmain = ruc_soil_layer_depth;"
    assert anchor in shipped
    mutated = shipped.replace(anchor, anchor + " ", 1)
    assert mutated != shipped, "the mutation control mutated nothing"
    digest = hashlib.sha256(_reconstruct_pre_lift(
        _reconstruct_pre_fix(_reconstruct_pre_mosaic(mutated))).encode("utf-8")).hexdigest()
    assert digest != PRE_LIFT_FILE_SHA256


def test_the_pre_lift_depth_table_is_the_ingest_tables_nine_level_row():
    """The literals carried inline above are not a third transcription.

    They are pinned against the table that is oracle-matched to WRF's
    ``init_soil_depth_3``, so the two cannot drift.
    """
    import numpy as np

    from woof.ingest.ruc_soil import ruc_soil_depths

    literals = [float(token) for token in
                re.findall(r"(-?\d+\.\d+)f", PRE_LIFT_DEPTH_TABLE)]
    expected = np.asarray(ruc_soil_depths(9)[0], dtype=np.float32)
    assert len(literals) == 9
    assert np.array_equal(np.asarray(literals, dtype=np.float32), expected)


def _depth_table_arm(count: int) -> list[float]:
    """The float literals of one ``#if`` arm of the depth table."""
    block = _block(DEPTH_TABLE).search(_shipped())
    assert block is not None
    arm = re.search(
        rf"#(?:if|elif) RUC_NZS == {count}\n(.*?)\n#(?:elif|endif)",
        block.group(0), re.S)
    assert arm is not None, f"no depth-table arm for RUC_NZS == {count}"
    body = arm.group(1)
    initializer = re.search(r"=\s*\{(.*?)\}", body, re.S)
    assert initializer is not None, f"arm {count} has no initializer"
    return [float(token)
            for token in re.findall(r"(-?\d+\.\d+)f", initializer.group(1))]


@pytest.mark.parametrize("count", [6, 9])
def test_the_kernel_depth_table_is_the_ingest_tables_row(count):
    """Every arm's literals ARE the oracle-matched table, bit for bit.

    ``woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M`` is the one transcription of
    WRF's ``init_soil_depth_3`` that is checked against real.exe's ZS/DZS.
    The device cannot import it -- ``__constant__`` needs literals -- so the
    literals are pinned against it here.  This is strictly stronger than the
    length check it replaces in ``tests/test_soil_layer_geometry.py``: it
    pins the VALUES, not just the count, and a length check would have been
    perfectly happy with six wrong depths.
    """
    import numpy as np

    from woof.ingest.ruc_soil import ruc_soil_depths

    literals = np.asarray(_depth_table_arm(count), dtype=np.float32)
    expected = np.asarray(ruc_soil_depths(count)[0], dtype=np.float32)
    assert len(literals) == count
    assert np.array_equal(literals.view(np.uint32), expected.view(np.uint32)), (
        f"ruc.cu's {count}-level depth table is "
        f"{[hex(v) for v in literals.view(np.uint32)]}, the ingest table is "
        f"{[hex(v) for v in expected.view(np.uint32)]}")


def test_the_depth_table_has_an_arm_for_every_admitted_geometry():
    for count in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        assert len(_depth_table_arm(count)) == count


def test_the_sentinels_are_present_exactly_once_each():
    """The reconstruction slices on these; a duplicate would slice wrong."""
    text = _shipped()
    for name in (TIER_LADDER, DEPTH_TABLE):
        assert text.count(f"// >>> {name} >>>\n") == 1, name
        assert text.count(f"// <<< {name} <<<\n") == 1, name


def test_the_ladder_guards_the_shipped_literal_and_derives_the_rest():
    """The ``#ifndef`` triple and the ``#if`` ladder, read structurally."""
    text = _shipped()
    lines = text.splitlines()
    defines = [i for i, line in enumerate(lines)
               if line.strip().startswith("#define RUC_NZS ")]
    assert len(defines) == 1, "exactly one RUC_NZS definition"
    i = defines[0]
    assert lines[i].strip() == "#define RUC_NZS 9"
    assert lines[i - 1].strip() == "#ifndef RUC_NZS"
    assert lines[i + 1].strip() == "#endif"
    assert lines[i + 2].strip() == "#if RUC_NZS == 9"
    assert "#elif RUC_NZS == 6" in text
    assert "#error" in text


# ---------------------------------------------------------------------------
# 3. The no-arithmetic rule, mechanized
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


#: Every macro the ladder defines, and the bare literal it must expand to at
#: the shipped geometry.
LADDER_DEFINE = re.compile(
    r"^#define (RUC_NZS|RUC_NZS_M[123]|RUC_DTDZS_LEN) +(.*)$", re.M)


def test_every_derived_macro_expands_to_a_bare_decimal_literal():
    """``RUC_NZS_M2`` is defined as ``7``, never as ``RUC_NZS - 2``.

    THIS is the no-arithmetic rule, and it is a rule about the DEFINITIONS,
    not about the use sites.  ``#define RUC_NZS_M2 (RUC_NZS - 2)`` would be
    perfectly correct arithmetic and would expand to ``(9 - 2)`` where the
    pre-lift source had the single token ``7`` -- so the preprocessed
    translation unit at nine would no longer be token-for-token what it was,
    and the cheap reconstruction proof above would die with it.

    Use sites are a different matter and are deliberately NOT restricted: the
    source says ``int kn = RUC_NZS - step;`` where it used to say
    ``int kn = 9 - step;``, and that expands to exactly the same tokens.  A
    guard that banned an operator NEXT TO a macro would forbid the fifteen
    index-arithmetic sites this lift is required to produce, while catching
    nothing the reconstruction test does not already catch exhaustively.
    """
    defines = LADDER_DEFINE.findall(_strip_comments(_shipped()))
    offenders = [(name, body) for name, body in defines
                 if not re.fullmatch(r"\d+", body.strip())]
    assert not offenders, (
        f"these ladder macros are not bare decimal literals: {offenders}.  "
        "An expression here is correct arithmetic and still breaks the "
        "token-identity of the nine-level translation unit")

    # RUC_NZS is defined once, under the #ifndef guard.  The four derived
    # macros are defined once per admitted geometry, so twice each.
    from collections import Counter
    counts = Counter(name for name, _ in defines)
    assert counts == {"RUC_NZS": 1, "RUC_NZS_M1": 2, "RUC_NZS_M2": 2,
                      "RUC_NZS_M3": 2, "RUC_DTDZS_LEN": 2}, counts

    # And each arm's values are the ones the geometry actually implies.
    body = _strip_comments(_shipped())
    for count, expected in ((9, ("8", "7", "6", "14")),
                            (6, ("5", "4", "3", "8"))):
        arm = re.search(
            rf"#(?:if|elif) RUC_NZS == {count}\n(.*?)\n#(?:elif|else|endif)",
            body, re.S)
        assert arm is not None, f"no ladder arm for RUC_NZS == {count}"
        got = tuple(m.group(2).strip() for m in
                    LADDER_DEFINE.finditer(arm.group(1)))
        assert got == expected, (
            f"RUC_NZS == {count} derives {got}, expected {expected}: "
            f"M1/M2/M3 are n-1/n-2/n-3 and DTDZS_LEN is 2*(n-2)")


def test_only_the_ladders_own_macros_are_used_in_the_body():
    """No RUC_NZS-family name may be used that the ladder does not define.

    A typo -- ``RUC_NZS_M4``, ``RUC_NZS_MI`` -- would silently preprocess to
    itself and then fail to compile only for whoever next builds the module.
    """
    defined = {name for name, _ in LADDER_DEFINE.findall(_shipped())}
    used = set(re.findall(r"\bRUC_[A-Z0-9_]+\b",
                          _strip_comments(_shipped())))
    assert used <= defined, f"undefined RUC_ macros used: {sorted(used - defined)}"


# ---------------------------------------------------------------------------
# 4. The generated code is identical, measured on real tools
# ---------------------------------------------------------------------------

def _host_preprocessor() -> list[str] | None:
    for root in (Path("C:/Program Files/Microsoft Visual Studio"),
                 Path("C:/Program Files (x86)/Microsoft Visual Studio")):
        if not root.is_dir():
            continue
        found = sorted(root.glob(
            "*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if found:
            return [str(found[-1]), "-nologo", "-EP", "-TP"]
    for name in ("g++", "clang++"):
        compiler = shutil.which(name)
        if compiler is not None:
            return [compiler, "-E", "-P", "-x", "c++"]
    return None


def _preprocess(source: str, tmp_path: Path, name: str,
                extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    command = _host_preprocessor()
    assert command is not None
    path = tmp_path / name
    path.write_text(source, encoding="utf-8", newline="")
    return subprocess.run(command + list(extra) + [str(path)],
                          capture_output=True, text=True, cwd=tmp_path)


def _token_stream(source: str, tmp_path: Path, name: str,
                  extra: tuple[str, ...] = ()) -> list[str]:
    """Preprocessed non-blank lines.

    ``ruc.cu`` has no ``#include``, so the preprocessor needs no header
    search path and its output is a pure macro expansion of this one file.
    ``-EP`` keeps the vertical whitespace a removed directive or comment left
    behind, which is invisible to the compiler, so blank lines are dropped.
    """
    result = _preprocess(source, tmp_path, name, extra)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_ladder_is_a_preprocessor_no_op_at_nine(tmp_path):
    """The whole claim, run through a real preprocessor."""
    shipped = _shipped()
    assert (_token_stream(shipped, tmp_path, "shipped.cpp")
            == _token_stream(_reconstruct_pre_lift(shipped), tmp_path,
                             "pre_lift.cpp"))


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
def test_the_preprocessor_comparison_can_fail(tmp_path):
    """Negative control: selecting six levels must move the token stream."""
    shipped = _shipped()
    assert (_token_stream(shipped, tmp_path, "six.cpp", ("-DRUC_NZS=6",))
            != _token_stream(_reconstruct_pre_lift(shipped), tmp_path,
                             "pre_lift.cpp"))


@pytest.mark.skipif(_host_preprocessor() is None,
                    reason="no host C preprocessor installed")
@pytest.mark.parametrize("nzs", [4, 5, 7, 8, 12])
def test_an_unadmitted_geometry_stops_the_compile(tmp_path, nzs):
    """``#error`` fires, so a bad tier is a build failure not a bad column."""
    result = _preprocess(_shipped(), tmp_path, f"bad{nzs}.cpp",
                         (f"-DRUC_NZS={nzs}",))
    assert result.returncode != 0
    assert "RUC_NZS must be 6 or 9" in (result.stdout + result.stderr)


def _nvcc() -> list[str] | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    command = [nvcc, "-ptx", "-std=c++17", "-arch=sm_120"]
    host = _host_preprocessor()
    if host is not None:
        command += ["-ccbin", str(Path(host[0]).parent)]
    return command


def _ptx(source: str, tmp_path: Path, name: str,
         extra: tuple[str, ...] = ()) -> str:
    command = _nvcc()
    assert command is not None
    src = tmp_path / f"{name}.cu"
    out = tmp_path / f"{name}.ptx"
    src.write_text(source, encoding="utf-8", newline="")
    result = subprocess.run(command + list(extra) + ["-o", str(out), str(src)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    return re.sub(r"^//.*$", "", out.read_text(encoding="utf-8"), flags=re.M)


def _assembled(cu_text: str) -> str:
    """``module_source``'s preamble in front of an arbitrary ruc.cu body."""
    assembled = module_source("ruc")
    shipped = _shipped()
    assert assembled.endswith(shipped)
    return assembled[:len(assembled) - len(shipped)] + cu_text


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_ladder_generates_identical_ptx_at_nine(tmp_path):
    """Stronger than the token comparison, and still device-free."""
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "shipped")
            == _ptx(_assembled(_reconstruct_pre_lift(shipped)), tmp_path,
                    "pre_lift"))


#: A kernel that is not in ``ruc.cu``, appended to make a source that must
#: compile to different PTX.  The six-level tier is NOT used as the negative
#: control here: until the depth table grows its ``#elif RUC_NZS == 6`` arm,
#: ``-DRUC_NZS=6`` does not compile at all, and "it failed to build" is not
#: evidence that ``_ptx`` can tell two working sources apart.  The six-level
#: PTX comparison lives in
#: :func:`test_the_six_level_tier_generates_different_ptx` instead.
_PTX_PROBE = """
__global__ void ruc_ptx_instrument_probe(float* out) { out[0] = 1.0f; }
"""


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_ptx_comparison_can_fail(tmp_path):
    """Negative control: the instrument distinguishes two sources.

    Without this, ``test_the_ladder_generates_identical_ptx_at_nine`` could
    be passing because ``_ptx`` returns the same string for everything.
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "shipped")
            != _ptx(_assembled(shipped + _PTX_PROBE), tmp_path, "probed"))


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_named_fix_is_a_real_change_to_the_generated_code(tmp_path):
    """The dzstop fix is NOT a preprocessor no-op, and must not read as one.

    Everything else in this file argues that the lift generates identical
    code at nine.  The fix does not: reading ``ruc_soil_layer_depth[1]``
    where an immediate stood is a ``__constant__`` load, and if the PTX came
    out identical it would mean the compiler folded the table back into an
    immediate -- which is exactly the ptxas folding this file's own depth
    table is ``__constant__`` to prevent.

    So this asserts the change is visible in the generated code at nine,
    while ``tests/test_ruc_nzs_device.py`` asserts no NUMBER moves there.
    Two different claims, and conflating them is what would let a real fix
    be waved through as "inert".
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "fixed")
            != _ptx(_assembled(_reconstruct_pre_fix(shipped)), tmp_path,
                    "pre_fix"))


@pytest.mark.skipif(_nvcc() is None, reason="no CUDA toolkit installed")
def test_the_six_level_tier_compiles_and_generates_different_ptx(tmp_path):
    """The six-level module is a real translation unit, not a hope.

    Two claims in one compile, both device-free: ``-DRUC_NZS=6`` BUILDS --
    every scratch extent, every derived bound and the ``#elif`` depth-table
    arm agree well enough for nvcc's front end and ptxas -- and what it
    builds is genuinely different code, which is the negative control the
    nine-level identity test needs.
    """
    shipped = _shipped()
    assert (_ptx(_assembled(shipped), tmp_path, "six", ("-DRUC_NZS=6",))
            != _ptx(_assembled(shipped), tmp_path, "nine"))


# ---------------------------------------------------------------------------
# 5. The host lane's geometry closures
# ---------------------------------------------------------------------------

def test_wrfs_default_root_count_is_in_range_at_every_admitted_geometry():
    """``woof.core.ruc``'s ``chosen = 4`` is WRF's :797 nroot fallback.

    It is a fixed level index, not a count that scales with the column, so it
    is only safe while ``4 <= n - 1`` for every admitted ``n``.  That holds at
    6 and at 9.  If the admitted set ever gains a shorter geometry, this fails
    here rather than by ``_root_count_field`` rejecting an nroot the physics
    itself produced.
    """
    assert min(WRF_SUPPORTED_NUM_SOIL_LAYERS) - 1 >= 4


@pytest.mark.parametrize("count", [6, 9])
def test_the_host_and_device_zshalf_derivations_agree(count):
    """One transcription on the host; the same expression on the device.

    ``ruc_zshalf`` replaced three identical loops in ``woof.core.ruc`` and a
    fourth in ``woof.core.ruc_gpu``.  The device builds the same interface
    depths from ``__constant__ ruc_soil_layer_depth`` in seven kernels.  Both
    are ``fadd_rn`` then a multiply by 0.5, which is exact for finite float32,
    so they must agree bit for bit -- and the literals the device uses are the
    ones this file already pins against the ingest table.
    """
    import numpy as np

    from woof.core.ruc import ruc_soil_geometry, ruc_zshalf

    zs, _ = ruc_soil_geometry(count)
    host = ruc_zshalf(zs)

    device_literals = np.asarray(_depth_table_arm(count), dtype=np.float32)
    device = np.zeros(count, dtype=np.float32)
    for level in range(1, count):
        device[level] = np.float32(
            np.float32(device_literals[level - 1] + device_literals[level])
            * np.float32(0.5))

    assert np.array_equal(host.view(np.uint32), device.view(np.uint32))
    assert host[0] == np.float32(0.0)
