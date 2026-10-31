"""TEMPO's own binary64 (``real(dp)``) arithmetic in binary32, for fast numerics only.

The source owner applies :func:`fast_dp_source` to the column support in both the serial and the warp
builders, never in strict mode, so the strict source stays byte for byte the one every identity proof
measured (TEMPO module replay 0 words, x1.2562 6 h strict 0 of 29,644,818 against native).

What changes in fast mode (the TEMPO_DP narrowing of the 3 km profile, attack 1):

* the process rates (``TempoTend``), the ``ilam*`` arrays, the snow moments and every ``double`` local of
  the process, check-and-update, fall-speed and diagnostic routines become ``float``;
* their unsuffixed (binary64) literals become binary32 literals, so float operands are not widened back to
  binary64 by a literal;
* the binary64 tables (``TK_* = double``) are read as ``float`` (one conversion per read, as every other
  TEMPO table already is), so a table read does not widen the arithmetic either;
* the shared-log pow of one base (``TempoPowBase``) is replaced in the column body by a binary32
  ``tempo_pow`` of the same operands (one base, several exponents; same values as plain powf).

Untouched: the blob storage (``TempoBlob`` stays binary64, the tables are uploaded as they are), the
elementary maths shim (``tempo_exp`` .. ``tempo_pow``, ``tempo_pow_base/with``, ``tnint``; owned by the
fast-maths transform ``tempo_fastmath_v303``, which composes with this one in either order), the parameter
and table accessor macros, the driver, the warp driver and the MPAS slice kernels.

THE BREAKAGE THIS ENDS: at 3 km TEMPO ran at the binary64 pipe's pace (1/64 of the FP32 rate on the
RTX PRO 6000): 18.4 k static FP64 instructions remained after the fast-maths swap, and TEMPO cost 7x
Thompson's 22 ms on mv3km. Fast words change by design; the fast referee is the forecast by eye
against native and MRMS plus domain stats, not words.
"""
from __future__ import annotations

import re

#: Where the column body starts: the first routine after the maths shim (calc_rslf).  Every ``double``
#: from here to the end of the column support is TEMPO's own dp arithmetic.
_BODY_MARKER = ("// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/TEMPO/src/"
                "module_mp_tempo_utils.F90:256-282 @ 1d6e370e; TEMPO 9adf9ef.\nTEMPO_HD float calc_rslf")
_TEND_OLD = "#define TEMPO_TT_MEMBER(name) double name[TEMPO_NZP];"
_TEND_NEW = "#define TEMPO_TT_MEMBER(name) float name[TEMPO_NZP];"
_TABLE_KIND = re.compile(r"^using (TK_\w+) = double;$", re.MULTILINE)
#: An unsuffixed decimal floating literal (1.0, .5, 2.e-3, 1e5); hex floats and suffixed literals do not match.
_LITERAL = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])")
_FLT_MAX = 3.4028234663852886e38
_FLT_MIN = 1.1754943508222875e-38

#: The binary32 shared pow of one base, used by the column body in place of TempoPowBase.
_POW_F = r"""
// Fast numerics (tempo_fastdp_v303): binary32 pow of one base, several exponents.
struct TempoPowBaseF { float x; };
TEMPO_HD TempoPowBaseF tempo_pow_base_f(float x) {TempoPowBaseF b; b.x = x; return b;}
TEMPO_HD float tempo_pow_with_f(const TempoPowBaseF& b, float y) {return tempo_pow(b.x, y);}
"""


def _float_literal(match: re.Match) -> str:
    text = match.group(1)
    value = abs(float(text))
    if value != 0.0 and (value > _FLT_MAX or value < _FLT_MIN):
        return text  # not representable as a normal binary32: keep the binary64 literal
    return text + "f"


def fast_dp_source(source: str) -> str:
    """Narrow TEMPO's binary64 arithmetic to binary32 in ``tempo_v303._COLUMN`` (fast numerics only).

    Accepts the column support alone or a module that contains it once. Fails on source drift (a marker
    missing or duplicated) or when applied twice. No environment or numerics side effects: the caller
    selects fast mode.
    """
    if source.count(_BODY_MARKER) != 1:
        raise ValueError(f"TEMPO fast dp: expected one column body marker, found {source.count(_BODY_MARKER)}")
    if source.count(_TEND_OLD) != 1:
        raise ValueError(f"TEMPO fast dp: expected one TempoTend member macro, found {source.count(_TEND_OLD)}")
    cut = source.index(_BODY_MARKER)
    head, body = source[:cut], source[cut:]
    head, tables = _TABLE_KIND.subn(r"using \1 = float;", head)
    if tables == 0:
        raise ValueError("TEMPO fast dp: no binary64 table kinds found")
    head = head.replace(_TEND_OLD, _TEND_NEW)
    body = re.sub(r"\bTempoPowBase\b", "TempoPowBaseF", body)
    body = re.sub(r"\btempo_pow_base\(", "tempo_pow_base_f(", body)
    body = re.sub(r"\btempo_pow_with\(", "tempo_pow_with_f(", body)
    body = re.sub(r"\bdouble\b", "float", body)
    body = _LITERAL.sub(_float_literal, body)
    return head + _POW_F + body
