"""CUDA checks against the unmodified WRF v4.6.1 MYNN surface oracles."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from test_mynn_surface import (
    _assert_ulp, _budget, INPUT_ALIASES, INPUT_NAMES, ISFFLX0_ZEROED,
    NARROW_CASES, WIDE_CASES,
)
from _toolchain_rows import toolchain_row



ORACLE_DIR = Path(__file__).parents[1] / "tests" / "data" / "oracles" / "mynn"
ORACLE = ORACLE_DIR / "surface-layer.csv"
WIDE_ORACLE = ORACLE_DIR / "surface-layer-wide.csv"
WRAPPER_ORACLE = ORACLE_DIR / "surface-layer-wrapper.csv"


# The FP32 ULP distance of the CUDA kernel from the unmodified WRF v4.6.1
# oracle, measured per (stage, output, column) -- one integer per compared
# number, in fixture column order (``NARROW_CASES`` / ``WIDE_CASES`` above).
# An output a stage does not name is bitwise on every column of that stage and
# is required to stay bitwise: the lookup returns zeros.
#
# There is no margin here to justify, because there is no margin: each entry is
# the residue that was measured at that element, and the gate is
# ``residue <= entry`` column by column.
#
# Sliced per column, the numbers stop being alarming and start being a
# diagnosis.  Read down any row of ``WIDE_ULP``: nine of the ten columns are
# 0-5 ULP and one is not.  ``thin_land_log10_wind``'s P1 is 99850 Pa, so the
# level-1 Exner base is 100/99.85 = 1.001502275, and CUDA's ``powf`` puts that
# ``**ROVCP`` one ULP away from the value gfortran and NumPy agree on.  That
# single ULP lands in TH1 (~1.8e-5 K on 296.13 K); DTHVDZ = THV1-THVGB
# (module_sf_mynn.F:559-560) is only 5.14 K on a 297 K THV, so the ~58x
# cancellation turns it into 92 ULP of BR, the zolri/zolrib solve carries that
# into ZOL (87), RMOL (116), PSIM (118) and PSIH (68), and PSIM/PSIH carry it
# into every exchange coefficient and flux below.  The CPU reference is 0 ULP
# on BR for the same fixture, so this is the CUDA ``powf`` shim, not the
# transcription.  It is not FMA contraction either: recompiling this module
# with ``-fmad=false`` reproduces all of these numbers exactly.
#
# The previous revision could not say any of that, because it budgeted per
# *fixture*: one table maxed over the widened fixture's three stages and all
# ten columns, so every column drew ``thin_land_log10_wind``'s 118.  Measured
# on this machine that left 59408 ULP of unearned margin, 53 of 205
# (stage, output) comparisons unable to fail on a 1-ULP regression, and single
# elements as much as 118 ULP loose.  Budgeting per (stage, output) and per
# (stage, column) and taking the smaller of the two was measured too: still
# 2488 ULP, and only 37% of single-element 1-ULP moves caught, for the same
# table volume as this.
#
# These numbers are Windows NumPy 2.2.6 / CPython 3.13.7 with cupy on the
# RTX 5090; unlike the CPU reference in ``test_mynn_surface.py`` there is no
# second platform to union with, because cupy only runs there.
#
# These are ratchets: lower them as the FP32 shims are unified, never raise.
NARROW_ULP = {
    "zol": (1, 1, 0, 0, 1, 1),
    "rmol": (1, 2, 0, 0, 1, 1),
    "ust": (1, 0, 1, 0, 0, 0),
    "ustm": (1, 0, 1, 0, 0, 0),
    "psim": (2, 4, 0, 0, 1, 5),
    "psih": (2, 0, 0, 0, 0, 1),
    "chs": (2, 0, 1, 0, 0, 0),
    "chs2": (1, 0, 1, 0, 1, 0),
    "cqs2": (1, 0, 1, 0, 0, 0),
    "ch": (0, 0, 4, 0, 0, 0),
    "flhc": (0, 0, 2, 0, 0, 0),
    "flqc": (2, 0, 1, 0, 0, 0),
    "qgh": (1, 1, 1, 1, 1, 3),
    "qsfc": (0, 0, 0, 0, 1, 0),
    "qfx": (2, 0, 0, 0, 5, 0),
    "lh": (2, 0, 0, 0, 6, 0),
    "u10": (1, 0, 2, 0, 0, 0),
    "v10": (1, 0, 0, 0, 0, 0),
    "gz1oz0": (0, 0, 1, 0, 0, 0),
    "cka": (4, 0, 1, 0, 0, 0),
    "cda": (2, 0, 2, 0, 0, 0),
    "qstar": (0, 0, 0, 0, 3, 0),
}

WIDE_ULP = {
    (1, 1): {
        "zol": (2, 0, 3, 0, 1, 1, 1, 87, 1, 0),
        "rmol": (2, 0, 3, 0, 1, 1, 2, 116, 1, 0),
        "ust": (1, 0, 1, 1, 0, 1, 1, 21, 0, 0),
        "ustm": (1, 0, 1, 1, 0, 1, 1, 21, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 36, 0, 0),
        "psim": (1, 0, 1, 0, 1, 2, 2, 118, 1, 0),
        "psih": (1, 0, 0, 0, 0, 1, 1, 68, 1, 1),
        "chs": (1, 0, 1, 2, 0, 1, 2, 77, 0, 0),
        "chs2": (1, 0, 2, 2, 0, 1, 2, 69, 0, 0),
        "cqs2": (1, 0, 2, 1, 0, 1, 1, 69, 0, 0),
        "ch": (1, 0, 1, 1, 0, 2, 1, 77, 0, 0),
        "flhc": (1, 0, 1, 1, 0, 2, 1, 44, 0, 0),
        "flqc": (1, 0, 1, 0, 0, 1, 3, 45, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
        "hfx": (1, 0, 1, 0, 0, 1, 1, 16, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 1, 3, 45, 5, 2),
        "lh": (1, 0, 0, 0, 0, 2, 4, 54, 3, 1),
        "u10": (1, 0, 0, 2, 1, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 1, 2, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (1, 0, 1, 0, 1, 1, 4, 62, 0, 0),
        "cka": (1, 0, 1, 1, 0, 1, 1, 74, 0, 0),
        "cd": (3, 0, 2, 0, 1, 0, 5, 70, 0, 0),
        "cda": (2, 0, 1, 2, 0, 1, 2, 41, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 3),
    },
    (2, 1): {
        "zol": (0, 0, 2, 0, 1, 2, 0, 87, 1, 0),
        "rmol": (0, 0, 2, 0, 1, 3, 0, 116, 0, 0),
        "ust": (0, 0, 2, 2, 0, 0, 0, 20, 0, 0),
        "ustm": (0, 0, 2, 2, 0, 0, 0, 20, 0, 0),
        "mol": (0, 0, 2, 0, 0, 0, 0, 36, 0, 0),
        "psim": (0, 0, 2, 0, 2, 4, 0, 117, 2, 2),
        "psih": (0, 0, 2, 0, 0, 1, 0, 69, 1, 0),
        "chs": (0, 0, 2, 1, 0, 0, 0, 47, 0, 0),
        "chs2": (0, 0, 2, 1, 0, 0, 0, 42, 0, 0),
        "cqs2": (0, 0, 1, 1, 0, 0, 0, 42, 0, 0),
        "ch": (0, 0, 4, 0, 0, 0, 0, 47, 0, 0),
        "flhc": (0, 0, 4, 0, 0, 0, 0, 54, 0, 0),
        "flqc": (0, 0, 2, 2, 0, 0, 0, 55, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (0, 0, 5, 0, 0, 0, 0, 20, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 55, 4, 1),
        "lh": (0, 0, 0, 0, 0, 0, 0, 33, 5, 1),
        "u10": (0, 0, 2, 2, 1, 0, 0, 0, 2, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 2, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (0, 0, 0, 0, 1, 2, 1, 64, 0, 0),
        "cka": (0, 0, 0, 1, 1, 0, 0, 72, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 3, 72, 0, 0),
        "cda": (0, 0, 2, 2, 1, 0, 0, 39, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 2),
    },
}

WRAPPER_ULP = {
    1: {
        "zol": (1, 0, 3, 0, 0, 2, 0, 83, 1, 2),
        "rmol": (1, 0, 3, 0, 0, 2, 0, 111, 1, 2),
        "ust": (1, 0, 2, 1, 0, 2, 0, 20, 0, 0),
        "ustm": (1, 0, 2, 1, 0, 2, 0, 20, 0, 0),
        "mol": (1, 1, 2, 0, 1, 0, 0, 37, 0, 0),
        "psim": (0, 0, 2, 0, 0, 2, 0, 111, 1, 4),
        "psih": (1, 0, 2, 0, 0, 0, 0, 66, 2, 1),
        "chs": (3, 1, 2, 1, 1, 1, 0, 47, 0, 0),
        "chs2": (1, 0, 1, 1, 0, 1, 0, 43, 0, 0),
        "cqs2": (1, 0, 1, 1, 0, 1, 0, 42, 0, 0),
        "ch": (3, 1, 3, 4, 2, 2, 0, 47, 0, 0),
        "flhc": (2, 1, 4, 2, 2, 3, 0, 55, 0, 0),
        "flqc": (0, 0, 3, 1, 0, 3, 0, 56, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (2, 1, 5, 0, 2, 3, 0, 23, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 56, 3, 2),
        "lh": (0, 0, 0, 0, 0, 0, 0, 33, 4, 1),
        "u10": (0, 0, 0, 2, 0, 2, 0, 0, 0, 0),
        "v10": (1, 0, 0, 0, 0, 4, 0, 0, 1, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (1, 0, 2, 0, 0, 0, 2, 62, 0, 0),
        "cka": (1, 0, 1, 1, 0, 3, 0, 70, 0, 0),
        "cd": (2, 0, 3, 0, 0, 0, 2, 69, 0, 0),
        "cda": (2, 0, 2, 2, 0, 5, 0, 42, 0, 0),
    },
    2: {
        "zol": (1, 0, 3, 0, 1, 3, 1, 84, 2, 0),
        "rmol": (1, 0, 2, 0, 1, 5, 1, 112, 2, 0),
        "ust": (0, 0, 1, 0, 0, 0, 0, 19, 1, 0),
        "ustm": (0, 0, 1, 0, 0, 0, 0, 19, 1, 0),
        "mol": (0, 0, 1, 0, 2, 0, 0, 37, 0, 0),
        "psim": (0, 0, 1, 0, 1, 0, 0, 110, 3, 0),
        "psih": (0, 0, 2, 0, 1, 1, 0, 65, 5, 0),
        "chs": (0, 0, 3, 1, 2, 0, 0, 49, 1, 0),
        "chs2": (0, 0, 1, 0, 0, 0, 1, 46, 1, 0),
        "cqs2": (0, 0, 3, 0, 0, 0, 0, 46, 2, 0),
        "ch": (0, 0, 2, 1, 2, 0, 0, 48, 1, 0),
        "flhc": (0, 0, 3, 1, 1, 0, 0, 28, 1, 0),
        "flqc": (0, 0, 3, 0, 0, 0, 0, 30, 1, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (0, 0, 2, 0, 1, 0, 0, 29, 1, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 30, 4, 2),
        "lh": (0, 0, 0, 0, 0, 0, 0, 36, 5, 1),
        "u10": (0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 0, 2, 0, 0, 1, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (2, 0, 2, 0, 0, 0, 1, 64, 2, 0),
        "cka": (0, 0, 3, 1, 0, 2, 0, 69, 0, 0),
        "cd": (0, 0, 2, 0, 0, 0, 0, 71, 2, 0),
        "cda": (0, 0, 3, 2, 0, 3, 0, 40, 0, 0),
        "znt": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    },
}

#: ``NARROW_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 4 rows read more than ``NARROW_ULP``.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
NARROW_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **NARROW_ULP,
        "lh": (2, 0, 0, 0, 0, 1),
        "qfx": (2, 0, 0, 0, 0, 2),
        "qsfc": (0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 3),
    },
}
#: A167: NVRTC 12.9.86, the compiler of the default recast-woof[gpu] extra
#: (cupy-cuda12x), reads the sm_120 row A146 re-recorded under 13.4: every
#: reading this file's tests take, and the device result behind each, is
#: bit-identical under the two compilers.  Before this row 12.9.86 failed
#: here by name (tests/_toolchain_rows.py).  MEASURED 2026-10-01 on test host 4's
#: RTX 5070 Ti and test host 2's RTX 5090, two processes per compiler, at
#: integrate/2.8 9dbb4a2db.
NARROW_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    NARROW_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89 reads the sm_120 row.  Since the cross-architecture rounding pins
#: (xnode-identity, ecbd106a9 and the A146 __fdiv_rn spellings) the RTX 4090
#: returns the RTX 5090's words for every output these gates compare:
#: 16,518 words, zero differing, NVRTC 13.4, 2026-10-05.  The default row,
#: recorded before those pins, no longer describes it: it failed 31 gates
#: on the RTX 4090 with no kernel change.  MEASURED on test host 1's RTX 4090
#: under NVRTC 12.9, 13.3 and 13.4: every gate passes on the sm_120 row.
NARROW_ULP_BY_TOOLCHAIN[("89", (12, 9))] = NARROW_ULP_BY_TOOLCHAIN[("120", (13, 4))]
NARROW_ULP_BY_TOOLCHAIN[("89", (13, 3))] = NARROW_ULP_BY_TOOLCHAIN[("120", (13, 4))]
NARROW_ULP_BY_TOOLCHAIN[("89", (13, 4))] = NARROW_ULP_BY_TOOLCHAIN[("120", (13, 4))]


#: ``WIDE_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 10 rows read more than ``WIDE_ULP``.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
WIDE_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **WIDE_ULP,
        (1, 1): {
            **WIDE_ULP[(1, 1)],
            "lh": (1, 0, 0, 0, 0, 0, 4, 3, 0, 2),
            "qfx": (1, 0, 0, 0, 0, 0, 3, 2, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        (2, 1): {
            **WIDE_ULP[(2, 1)],
            "lh": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
            "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
    },
}
#: A167: as for NARROW_ULP_BY_TOOLCHAIN above.
WIDE_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    WIDE_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89: as for NARROW_ULP_BY_TOOLCHAIN above.
WIDE_ULP_BY_TOOLCHAIN[("89", (12, 9))] = WIDE_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WIDE_ULP_BY_TOOLCHAIN[("89", (13, 3))] = WIDE_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WIDE_ULP_BY_TOOLCHAIN[("89", (13, 4))] = WIDE_ULP_BY_TOOLCHAIN[("120", (13, 4))]


#: ``WRAPPER_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 18 rows read more than ``WRAPPER_ULP``;
#: the reciprocal-multiply compiler's one row (psim on free_convective_land)
#: reads at or below ``WRAPPER_ULP`` now and is gone.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
WRAPPER_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **WRAPPER_ULP,
        1: {
            **WRAPPER_ULP[1],
            "cd": (2, 0, 3, 0, 1, 0, 2, 0, 0, 0),
            "ck": (1, 0, 2, 0, 2, 0, 2, 1, 0, 0),
            "lh": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
            "psih": (1, 0, 2, 0, 2, 0, 0, 1, 2, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 0, 1, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "rmol": (1, 0, 3, 0, 3, 2, 0, 2, 1, 2),
            "u10": (0, 0, 0, 2, 1, 2, 0, 0, 0, 0),
            "v10": (1, 0, 0, 0, 1, 4, 0, 0, 1, 0),
            "zol": (1, 0, 3, 0, 2, 2, 0, 1, 1, 2),
        },
        2: {
            **WRAPPER_ULP[2],
            "cd": (0, 0, 2, 0, 0, 0, 0, 2, 2, 2),
            "ck": (2, 0, 2, 0, 0, 0, 1, 2, 2, 1),
            "lh": (0, 0, 0, 0, 0, 0, 0, 0, 2, 2),
            "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 1, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "v10": (0, 0, 1, 0, 0, 2, 0, 0, 1, 2),
        },
    },
}
#: A167: as for NARROW_ULP_BY_TOOLCHAIN above.
WRAPPER_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    WRAPPER_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89: as for NARROW_ULP_BY_TOOLCHAIN above.
WRAPPER_ULP_BY_TOOLCHAIN[("89", (12, 9))] = WRAPPER_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WRAPPER_ULP_BY_TOOLCHAIN[("89", (13, 3))] = WRAPPER_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WRAPPER_ULP_BY_TOOLCHAIN[("89", (13, 4))] = WRAPPER_ULP_BY_TOOLCHAIN[("120", (13, 4))]


#: module_sf_mynn.F:1027-1044.  With ISFFLX<1 WRF assigns these thirteen
#: outputs the literal constant 0 and leaves the other twenty-two on the same
#: code path the ISFFLX=1 first step takes.  So the ISFFLX=0 stage needs no
#: table of its own: it reuses ``WIDE_ULP[(1, 1)]`` for the twenty-two, which
#: is exact because the outputs are bit-identical, and 0 for these thirteen,
#: which is exact because they are 0.0 on both sides.  Both halves are pinned
#: by ``test_the_isfflx0_stage_zeroes_thirteen_outputs_in_the_kernel_too``.
#:
#: This is where the old per-fixture table was worst: it handed these thirteen
#: the ISFFLX=1 budgets, so CH and CHS measured 0 ULP against a budget of 77.



def _rows(path: Path):
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _fields(rows, shape):
    skip = ("case", "itimestep", "isfflx")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(shape)
        for key in rows[0] if key not in skip
    }


def _oracle_fields():
    rows = _rows(ORACLE)
    return rows, _fields(rows, (2, 3))


def _wide_stage(path: Path, itimestep: int, isfflx: int | None = None):
    rows = [
        row for row in _rows(path)
        if int(row["itimestep"]) == itimestep
        and (isfflx is None or int(row["isfflx"]) == isfflx)
    ]
    assert tuple(row["case"] for row in rows) == WIDE_CASES
    return _fields(rows, (2, 5))


def _device_inputs(fields):
    import cupy as cp

    return {
        name: cp.asarray(fields[INPUT_ALIASES.get(name, name)].copy())
        for name in INPUT_NAMES
    }






def _assert_parity(actual, fields, names, table, cases, zeroed=()):
    import cupy as cp

    np.testing.assert_array_equal(
        cp.asnumpy(actual.regime), fields["regime"]
    )
    for name in names:
        if name == "regime":
            continue
        _assert_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name], name,
            _budget(table, name, cases, zeroed), cases,
        )


@pytest.mark.gpu
@requires_gpu
def test_mynn_surface_cuda_matches_official_wrf_oracle():
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
    )

    rows, fields = _oracle_fields()
    actual = mynn_surface_layer(_device_inputs(fields))
    cp.cuda.get_current_stream().synchronize()

    assert tuple(row["case"] for row in rows) == NARROW_CASES
    _assert_parity(
        actual, fields,
        tuple(name for name in MYNN_SURFACE_OUTPUTS if name != "znt"),
        toolchain_row(NARROW_ULP_BY_TOOLCHAIN, NARROW_ULP,
                      "NARROW_ULP_BY_TOOLCHAIN"), NARROW_CASES,
    )
    land = fields["xland"] < 1.5
    znt = cp.asnumpy(actual.znt)
    np.testing.assert_array_equal(znt[land], fields["znt_input"][land])
    assert np.all(znt[~land] != fields["znt_input"][~land])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(
    ("itimestep", "isfflx"), ((1, 1), (2, 1), (1, 0)),
)
def test_mynn_surface_cuda_matches_widened_wrf_oracle(itimestep, isfflx):
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
    )

    fields = _wide_stage(WIDE_ORACLE, itimestep, isfflx)
    actual = mynn_surface_layer(
        _device_inputs(fields), itimestep=itimestep, isfflx=isfflx,
        mol=cp.asarray(fields["mol_input"].copy()),
        ustm=cp.asarray(fields["ustm_input"].copy()),
    )
    cp.cuda.get_current_stream().synchronize()
    _assert_parity(
        actual, fields, MYNN_SURFACE_OUTPUTS,
        toolchain_row(WIDE_ULP_BY_TOOLCHAIN, WIDE_ULP,
                      "WIDE_ULP_BY_TOOLCHAIN")[(itimestep, 1)],
        WIDE_CASES, zeroed=ISFFLX0_ZEROED if isfflx == 0 else (),
    )


@pytest.mark.gpu
@requires_gpu
def test_the_isfflx0_stage_zeroes_thirteen_outputs_in_the_kernel_too():
    """What licenses reusing ``WIDE_ULP[(1, 1)]`` for the ISFFLX=0 stage.

    module_sf_mynn.F:1027-1044 makes ISFFLX<1 a pure post-processing branch:
    thirteen outputs become the constant 0 and nothing else changes.  Pinning
    both halves is what makes the reuse a measured consequence of the branch
    rather than a borrowed budget -- and it is a real gate, because a kernel
    that computed the coefficients anyway would land far from zero.
    """

    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
    )

    off = _wide_stage(WIDE_ORACLE, 1, 0)
    on = _wide_stage(WIDE_ORACLE, 1, 1)
    assert len(ISFFLX0_ZEROED) == 13
    for name in ISFFLX0_ZEROED:
        np.testing.assert_array_equal(
            off[name], np.zeros_like(off[name]), err_msg=f"oracle {name}"
        )
        assert np.any(on[name] != 0.0), name

    results = {}
    for isfflx in (1, 0):
        result = mynn_surface_layer(
            _device_inputs(off), itimestep=1, isfflx=isfflx,
            mol=cp.asarray(off["mol_input"].copy()),
            ustm=cp.asarray(off["ustm_input"].copy()),
        )
        cp.cuda.get_current_stream().synchronize()
        results[isfflx] = {
            name: cp.asnumpy(getattr(result, name))
            for name in MYNN_SURFACE_OUTPUTS
        }
    for name in MYNN_SURFACE_OUTPUTS:
        if name in ISFFLX0_ZEROED:
            np.testing.assert_array_equal(
                results[0][name], np.zeros_like(off[name]),
                err_msg=f"kernel {name}",
            )
            continue
        np.testing.assert_array_equal(off[name], on[name],
                                      err_msg=f"oracle {name}")
        np.testing.assert_array_equal(
            results[0][name], results[1][name], err_msg=f"kernel {name}"
        )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("itimestep", (1, 2))
def test_mynn_surface_cuda_matches_sfclay_wrapper_oracle(itimestep):
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
        seed_mynn_surface_first_step,
    )

    fields = _wide_stage(WRAPPER_ORACLE, itimestep)
    inputs = _device_inputs(fields)
    mol = cp.asarray(fields["mol_input"].copy())
    qstar = cp.zeros_like(mol)
    if itimestep == 1:
        assert not np.allclose(
            np.maximum(0.04 * np.hypot(fields["u1"], fields["v1"]), 0.001),
            fields["ust_input"],
        )
        seed_mynn_surface_first_step(
            inputs["u1"], inputs["v1"], inputs["qv1"],
            ust=inputs["ust"], mol=mol, qsfc=inputs["qsfc"], qstar=qstar,
        )
        assert bool((qstar == 0.0).all()) and bool((mol == 0.0).all())
    actual = mynn_surface_layer(
        inputs, itimestep=itimestep, isfflx=1, mol=mol,
        ustm=cp.asarray(fields["ustm_input"].copy()),
    )
    cp.cuda.get_current_stream().synchronize()
    # SFCLAY_mynn keeps wstar/qstar as wrapper locals and never returns them.
    _assert_parity(
        actual, fields,
        tuple(n for n in MYNN_SURFACE_OUTPUTS if n not in ("wstar", "qstar")),
        toolchain_row(WRAPPER_ULP_BY_TOOLCHAIN, WRAPPER_ULP,
                      "WRAPPER_ULP_BY_TOOLCHAIN")[itimestep],
        WIDE_CASES,
    )


@pytest.mark.gpu
@requires_gpu
def test_mynn_surface_cuda_evolves_water_znt_across_two_steps():
    import cupy as cp

    from woof.core.mynn_sfclay import mynn_surface_layer

    step1 = _wide_stage(WIDE_ORACLE, 1, 1)
    step2 = _wide_stage(WIDE_ORACLE, 2, 1)
    actual = mynn_surface_layer(
        _device_inputs(step1), itimestep=1,
        mol=cp.asarray(step1["mol_input"].copy()),
        ustm=cp.asarray(step1["ustm_input"].copy()),
    )
    cp.cuda.get_current_stream().synchronize()
    znt = cp.asnumpy(actual.znt)
    water = step1["xland"] > 1.5
    assert water.any()
    # The oracle carries WRF's own updated ZNT into step 2 and the kernel
    # reproduces it bitwise on every column, so this chain is pinned exact.
    _assert_ulp(znt, step2["znt_input"], "carried znt", 0, WIDE_CASES)
    assert np.all(znt[water] != step1["znt_input"][water])
    np.testing.assert_array_equal(
        znt[~water], step1["znt_input"][~water]
    )


def test_every_table_row_is_the_right_width_and_carries_a_measurement():
    """A row of the wrong width, or of zeros, is a table that lost meaning.

    ``_budget`` asserts the width when it is used, but only for outputs a test
    actually compares; this covers every row in the file, and rejects an
    all-zero row -- which would mean the output is bitwise and the row should
    have been deleted rather than left as decoration.  It needs no GPU.
    """

    tables = [(NARROW_ULP, NARROW_CASES)]
    tables += [(WIDE_ULP[key], WIDE_CASES) for key in WIDE_ULP]
    tables += [(WRAPPER_ULP[key], WIDE_CASES) for key in WRAPPER_ULP]
    assert len(tables) == 5
    for rows in WRAPPER_ULP_BY_TOOLCHAIN.values():
        assert set(rows) == set(WRAPPER_ULP)
        tables += [(rows[key], WIDE_CASES) for key in rows]
    for rows in WIDE_ULP_BY_TOOLCHAIN.values():
        assert set(rows) == set(WIDE_ULP)
        tables += [(rows[key], WIDE_CASES) for key in rows]
    tables += [(rows, NARROW_CASES)
               for rows in NARROW_ULP_BY_TOOLCHAIN.values()]
    for table, cases in tables:
        assert table
        for name, row in table.items():
            assert len(row) == len(cases), name
            assert all(isinstance(v, int) and v >= 0 for v in row), name
            assert any(row), f"{name} is all zero; delete the row"


@pytest.mark.gpu
@requires_gpu
def test_mynn_surface_cuda_rejects_selector_and_shape_drift():
    from woof.core.mynn_sfclay import mynn_surface_layer

    _, fields = _oracle_fields()
    inputs = _device_inputs(fields)
    with pytest.raises(ValueError, match="isfflx"):
        mynn_surface_layer(inputs, isfflx=2)
    bad = dict(inputs)
    bad["v1"] = bad["v1"][:, :2]
    with pytest.raises(ValueError, match="not broadcastable"):
        mynn_surface_layer(bad)


@pytest.mark.gpu
@requires_gpu
def test_first_step_seed_rejects_mismatched_device_arrays():
    import cupy as cp

    from woof.core.mynn_sfclay import seed_mynn_surface_first_step
    from woof.core.state import DTYPE

    ones = cp.ones((2, 3), dtype=DTYPE)
    with pytest.raises(ValueError, match="same-shape float32"):
        seed_mynn_surface_first_step(
            ones, ones, ones, ust=ones, mol=ones, qsfc=ones,
            qstar=cp.ones((2, 2), dtype=DTYPE),
        )
