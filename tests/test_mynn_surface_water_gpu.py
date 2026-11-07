"""CUDA parity for the over-water ISFTCFLX branches of MYNN.

The fixtures, the branch map and the CPU side of this comparison are in
``tests/test_mynn_surface_water.py``; this module is separate only because
``tests/conftest.py`` marks every cupy-importing module ``gpu`` wholesale, and
the CPU half has to stay runnable on a machine with no device.

``WATER_CUDA_ULP`` is measured once on the RTX 5090 (cupy 14.1.1), because
cupy runs nowhere else -- there is no second platform to union with, unlike
the CPU table.  One integer per (stage, output, column), each the residue
measured at that element, gated as ``residue <= entry``.

It is much larger than the CPU table, and the fixture says why rather than
leaving it to be argued: ``br`` is bitwise on every CPU row and reaches 696
ULP on ``gale_water`` here.  That column has TSK - T1 = 1 K, so
DTHVDZ = THV1 - THVGB (module_sf_mynn.F:559-560) is a ~1 K difference of
~289 K numbers -- a 300x cancellation -- and one ULP of disagreement between
CUDA's ``powf`` and glibc's in the Exner factor above it moves BR by hundreds
of ULP, which ZOL, PSIM, PSIH and HFX then inherit.  HFX is worst at
ISFTCFLX=0 (2,390) and *smaller* at 1/2/3 (286-539), because the dissipative
term the non-default identities add is large next to the cancelling
difference: the ported branches reduce this residue rather than cause it.

ZNT, the direct output of the ported leaves, is 0 ULP on the CPU on all three
platforms and at most 10 here, all of it in the ISFTCFLX=3 arm where
``powf(hs/Lp, 4.5)`` amplifies the device shim's error by 4.5x.

These are ratchets: lower them as the FP32 shims are unified, never raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from _toolchain_rows import toolchain_row

from woof.core.fp32_ulp import fp32_ulp_distance
from test_mynn_surface_water import (
    CASES,
    INPUT_ALIASES,
    INPUT_NAMES,
    ISFTCFLX_SWEEP,
    OUTPUT_NAMES,
    STAGES,
    _budget,
    _column_rows,
    _f32,
)


WATER_CUDA_ULP = {
    (0, 1, 1): {
        "br": (0, 3, 0, 0, 0, 689, 0, 0, 3, 0, 0, 0),
        "cd": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 2, 0),
        "ch": (0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 1),
        "chs": (0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 4, 1),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0),
        "ck": (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2),
        "cka": (0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0),
        "flhc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 3, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),
        "hfx": (0, 0, 0, 0, 0, 2390, 0, 0, 0, 0, 2, 1),
        "lh": (1, 1, 3, 3, 5, 1, 0, 4, 6, 1, 0, 0),
        "mol": (0, 0, 0, 0, 0, 812, 0, 0, 0, 0, 2, 1),
        "psih": (0, 0, 1, 0, 0, 784, 0, 1, 2, 1, 2, 1),
        "psim": (0, 1, 0, 0, 32, 744, 32, 1, 13, 1, 3, 1),
        "qfx": (3, 2, 3, 2, 4, 1, 0, 6, 5, 1, 0, 0),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 3, 0, 1, 2, 1, 1, 4, 10, 0, 0, 0),
        "rmol": (0, 1, 0, 0, 0, 664, 0, 1, 1, 2, 3, 4),
        "u10": (1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1),
        "ust": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "v10": (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "zol": (0, 1, 0, 0, 0, 830, 0, 1, 1, 1, 3, 2),
    },
    (0, 2, 1): {
        "br": (0, 3, 0, 1, 0, 696, 0, 0, 0, 0, 0, 0),
        "cd": (0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7),
        "cda": (0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 3, 5),
        "ch": (0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 3, 4),
        "chs": (0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 3),
        "chs2": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2),
        "ck": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7),
        "cka": (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 3),
        "cqs2": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2),
        "flhc": (0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 3, 4),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3),
        "gz1oz0": (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 1, 0, 2176, 0, 0, 0, 0, 1, 3),
        "lh": (1, 1, 4, 3, 4, 1, 0, 4, 7, 0, 0, 1),
        "mol": (0, 0, 0, 0, 0, 817, 0, 0, 0, 0, 0, 3),
        "psih": (0, 2, 1, 0, 0, 799, 1, 0, 0, 0, 0, 3),
        "psim": (0, 2, 0, 0, 29, 759, 32, 0, 9, 0, 0, 3),
        "qfx": (1, 1, 3, 3, 3, 1, 0, 6, 6, 0, 0, 3),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 2, 0, 1, 1, 2, 1, 4, 11, 0, 0, 0),
        "rmol": (0, 2, 0, 0, 0, 677, 1, 0, 0, 0, 0, 6),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "ust": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1),
        "ustm": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "v10": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "wspd": (0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "znt": (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "zol": (0, 2, 0, 1, 1, 846, 1, 0, 0, 0, 0, 4),
    },
    (1, 1, 1): {
        "br": (0, 3, 0, 0, 0, 689, 0, 0, 3, 0, 0, 0),
        "cd": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0),
        "ch": (0, 2, 0, 2, 0, 3, 0, 0, 0, 0, 3, 1),
        "chs": (0, 2, 0, 2, 0, 4, 0, 0, 0, 0, 4, 1),
        "chs2": (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0),
        "ck": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "cka": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0),
        "cqs2": (0, 2, 0, 0, 0, 1, 0, 1, 0, 0, 3, 0),
        "flhc": (0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 4, 1),
        "flqc": (0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 1, 0, 0, 0, 298, 0, 0, 0, 0, 2, 1),
        "lh": (1, 1, 3, 4, 4, 0, 0, 3, 6, 0, 0, 0),
        "mol": (0, 0, 0, 0, 0, 815, 0, 0, 0, 0, 2, 1),
        "psih": (0, 0, 0, 1, 1, 812, 0, 0, 2, 1, 2, 1),
        "psim": (0, 4, 0, 9, 23, 770, 32, 3, 13, 0, 3, 1),
        "qfx": (3, 1, 3, 3, 4, 0, 0, 5, 5, 0, 0, 0),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 2, 0, 2, 1, 2, 1, 5, 10, 0, 0, 0),
        "rmol": (0, 2, 0, 0, 1, 688, 0, 1, 1, 0, 3, 4),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1),
        "ust": (0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "v10": (0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1, 1),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "znt": (0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0),
        "zol": (0, 2, 0, 0, 1, 860, 0, 1, 1, 0, 3, 2),
    },
    (1, 2, 1): {
        "br": (0, 0, 0, 0, 0, 694, 0, 1, 1, 0, 0, 0),
        "cd": (0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 7),
        "cda": (0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3, 5),
        "ch": (0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 3, 4),
        "chs": (0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 3),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 2),
        "ck": (0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 7),
        "cka": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 3),
        "cqs2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2),
        "flhc": (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3, 4),
        "flqc": (0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 3),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 0, 0, 539, 0, 1, 0, 0, 1, 3),
        "lh": (2, 2, 2, 3, 4, 1, 0, 3, 6, 0, 0, 1),
        "mol": (0, 0, 0, 0, 0, 838, 0, 0, 0, 0, 0, 3),
        "psih": (0, 0, 1, 1, 0, 471, 1, 1, 3, 0, 0, 3),
        "psim": (0, 0, 2, 9, 18, 447, 31, 1, 13, 0, 0, 3),
        "qfx": (2, 1, 2, 2, 3, 1, 0, 4, 5, 0, 0, 3),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 2, 0, 2, 1, 1, 1, 4, 11, 0, 0, 0),
        "rmol": (0, 0, 1, 1, 0, 797, 2, 0, 1, 0, 0, 6),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "v10": (0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0),
        "wspd": (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "znt": (0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0),
        "zol": (0, 0, 2, 1, 0, 498, 1, 0, 1, 0, 0, 4),
    },
    (2, 1, 1): {
        "br": (0, 3, 0, 0, 0, 689, 0, 0, 3, 0, 0, 0),
        "cd": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0),
        "ch": (0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 3, 1),
        "chs": (0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 4, 1),
        "chs2": (0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0),
        "ck": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2),
        "cka": (0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0),
        "flhc": (0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 4, 1),
        "flqc": (0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 0, 0, 286, 1, 0, 0, 0, 2, 1),
        "lh": (2, 3, 3, 1, 4, 0, 1, 3, 5, 0, 0, 0),
        "mol": (0, 0, 0, 0, 1, 779, 0, 0, 0, 0, 2, 1),
        "psih": (0, 1, 2, 0, 0, 774, 0, 1, 4, 1, 2, 1),
        "psim": (0, 2, 2, 10, 24, 734, 27, 0, 27, 1, 3, 1),
        "qfx": (3, 2, 3, 2, 4, 0, 1, 5, 4, 0, 0, 0),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 3, 0, 1, 2, 1, 1, 4, 8, 0, 0, 0),
        "rmol": (0, 2, 1, 0, 0, 656, 0, 1, 2, 0, 3, 4),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (0, 0, 0, 0, 0, 2, 2, 1, 0, 1, 1, 1),
        "ust": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0),
        "ustm": (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0),
        "v10": (0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 1),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "znt": (0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0),
        "zol": (0, 3, 1, 0, 0, 820, 0, 1, 3, 0, 3, 2),
    },
    (2, 2, 1): {
        "br": (0, 0, 0, 2, 0, 694, 0, 0, 0, 0, 0, 0),
        "cd": (0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 7),
        "cda": (0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 3, 5),
        "ch": (0, 0, 0, 2, 0, 3, 0, 0, 0, 2, 3, 4),
        "chs": (0, 0, 0, 1, 0, 3, 0, 0, 0, 2, 1, 3),
        "chs2": (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2),
        "ck": (0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 7),
        "cka": (0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 3),
        "cqs2": (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2),
        "flhc": (0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 3, 4),
        "flqc": (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 3),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),
        "hfx": (0, 0, 0, 1, 1, 537, 1, 0, 0, 0, 1, 3),
        "lh": (3, 2, 2, 3, 3, 0, 0, 3, 6, 0, 0, 1),
        "mol": (0, 0, 0, 0, 0, 830, 0, 0, 0, 1, 0, 3),
        "psih": (0, 1, 0, 3, 2, 467, 0, 1, 0, 2, 0, 3),
        "psim": (0, 1, 0, 10, 20, 443, 29, 1, 20, 1, 0, 3),
        "qfx": (2, 3, 2, 2, 3, 0, 0, 5, 5, 0, 0, 3),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 3, 0, 2, 2, 0, 1, 4, 9, 0, 0, 0),
        "rmol": (0, 1, 1, 2, 1, 789, 0, 1, 0, 3, 0, 6),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0),
        "ust": (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1),
        "ustm": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1),
        "v10": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "wspd": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "wstar": (0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0),
        "znt": (0, 1, 1, 1, 2, 0, 0, 1, 0, 2, 0, 0),
        "zol": (0, 1, 1, 2, 1, 493, 0, 1, 0, 2, 0, 4),
    },
    (3, 1, 1): {
        "br": (0, 3, 0, 0, 0, 689, 0, 0, 3, 0, 0, 0),
        "cd": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 2, 0),
        "ch": (0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 3, 1),
        "chs": (0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 4, 1),
        "chs2": (0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0),
        "ck": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2),
        "cka": (0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0),
        "cqs2": (0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 3, 0),
        "flhc": (0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 3, 0),
        "gz1oz0": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1),
        "hfx": (0, 1, 0, 0, 0, 297, 0, 0, 0, 0, 2, 1),
        "lh": (1, 1, 3, 3, 3, 1, 0, 3, 6, 2, 0, 0),
        "mol": (0, 1, 0, 0, 0, 821, 0, 0, 0, 0, 2, 1),
        "psih": (0, 1, 0, 0, 0, 847, 0, 0, 2, 1, 2, 1),
        "psim": (0, 0, 0, 1, 29, 805, 32, 0, 13, 1, 3, 1),
        "qfx": (2, 2, 3, 3, 3, 1, 0, 5, 5, 1, 0, 0),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 3, 0, 1, 2, 1, 1, 4, 10, 0, 0, 0),
        "rmol": (0, 2, 2, 0, 0, 716, 0, 0, 1, 3, 3, 4),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1),
        "ust": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0),
        "v10": (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "znt": (0, 7, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0),
        "zol": (0, 2, 1, 0, 0, 448, 0, 0, 1, 2, 3, 2),
    },
    (3, 2, 1): {
        "br": (0, 0, 0, 0, 0, 696, 0, 0, 1, 0, 0, 0),
        "cd": (0, 0, 3, 0, 2, 1, 0, 0, 0, 0, 0, 7),
        "cda": (0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 5),
        "ch": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 4),
        "chs": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 3),
        "chs2": (0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2),
        "ck": (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 7),
        "cka": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 3),
        "cqs2": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2),
        "flhc": (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 4),
        "flqc": (0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 3),
        "gz1oz0": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1),
        "hfx": (0, 0, 0, 1, 1, 539, 0, 0, 0, 0, 1, 3),
        "lh": (2, 2, 3, 3, 4, 0, 0, 3, 6, 0, 0, 1),
        "mol": (0, 0, 0, 0, 0, 828, 0, 0, 0, 0, 0, 3),
        "psih": (0, 1, 3, 0, 0, 870, 1, 0, 3, 0, 0, 3),
        "psim": (0, 2, 2, 0, 28, 826, 31, 0, 13, 0, 0, 3),
        "qfx": (2, 1, 3, 2, 3, 0, 0, 5, 5, 0, 0, 3),
        "qgh": (2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0),
        "qsfc": (1, 2, 0, 1, 1, 1, 0, 1, 3, 0, 0, 0),
        "qstar": (1, 3, 0, 1, 2, 0, 1, 5, 11, 0, 0, 0),
        "rmol": (0, 0, 3, 0, 0, 736, 2, 0, 1, 0, 0, 6),
        "t2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        "u10": (0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        "v10": (0, 0, 0, 0, 1, 2, 0, 1, 0, 2, 1, 0),
        "wspd": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "znt": (0, 0, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0),
        "zol": (0, 0, 2, 0, 0, 460, 1, 0, 1, 0, 0, 4),
    },
}

#: ``WATER_CUDA_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 77 rows read more than ``WATER_CUDA_ULP``; the
#: override the reciprocal-multiply compiler needed (the ISFTCFLX=0 second
#: step's light_water column) is replaced by this measurement.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
WATER_CUDA_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **WATER_CUDA_ULP,
        (0, 1, 1): {
            **WATER_CUDA_ULP[(0, 1, 1)],
            "br": (0, 3, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0),
            "ck": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
            "cka": (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
            "flqc": (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0),
            "lh": (0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0),
            "psih": (0, 1, 0, 0, 0, 0, 3, 1, 2, 1, 2, 1),
            "psim": (0, 1, 0, 0, 32, 47, 35, 1, 13, 1, 3, 1),
            "qfx": (0, 1, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 2, 0, 1, 2, 0, 6, 0, 0, 0),
            "rmol": (0, 1, 0, 0, 0, 0, 5, 1, 1, 2, 3, 4),
            "wspd": (0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
            "zol": (0, 1, 0, 0, 0, 0, 4, 1, 1, 1, 3, 2),
        },
        (0, 2, 1): {
            **WATER_CUDA_ULP[(0, 2, 1)],
            "lh": (0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 1, 0, 1, 2, 0, 7, 0, 0, 0),
        },
        (1, 1, 1): {
            **WATER_CUDA_ULP[(1, 1, 1)],
            "br": (0, 3, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0),
            "cd": (0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 2),
            "ck": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2),
            "lh": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "psih": (0, 0, 0, 1, 1, 2, 3, 0, 2, 0, 2, 1),
            "psim": (0, 4, 0, 9, 23, 50, 35, 3, 13, 0, 3, 1),
            "qfx": (0, 1, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 0, 0, 1, 2, 0, 6, 0, 0, 0),
            "rmol": (0, 2, 0, 0, 1, 1, 5, 1, 1, 0, 3, 4),
            "wspd": (0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
            "znt": (0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),
            "zol": (0, 2, 0, 0, 1, 1, 4, 1, 1, 0, 3, 2),
        },
        (1, 2, 1): {
            **WATER_CUDA_ULP[(1, 2, 1)],
            "lh": (0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 0, 1),
            "qfx": (0, 0, 1, 0, 0, 0, 2, 2, 2, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 0, 0, 1, 0, 1, 2, 0, 7, 0, 0, 0),
        },
        (2, 1, 1): {
            **WATER_CUDA_ULP[(2, 1, 1)],
            "br": (0, 3, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0),
            "lh": (0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0),
            "psih": (0, 1, 2, 0, 0, 1, 4, 1, 4, 1, 2, 1),
            "psim": (0, 2, 2, 10, 24, 49, 32, 0, 27, 1, 3, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 1, 0, 1, 2, 0, 5, 0, 0, 0),
            "rmol": (0, 2, 1, 0, 0, 1, 5, 1, 2, 0, 3, 4),
            "wspd": (0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
            "znt": (0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),
            "zol": (0, 3, 1, 0, 0, 1, 4, 1, 3, 0, 3, 2),
        },
        (2, 2, 1): {
            **WATER_CUDA_ULP[(2, 2, 1)],
            "lh": (0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 1, 0, 2, 2, 0, 5, 0, 0, 0),
        },
        (3, 1, 1): {
            **WATER_CUDA_ULP[(3, 1, 1)],
            "br": (0, 3, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0),
            "lh": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "psih": (0, 1, 0, 0, 0, 0, 3, 0, 2, 2, 2, 1),
            "psim": (0, 0, 0, 1, 29, 51, 35, 0, 13, 2, 3, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 1, 0, 1, 2, 0, 6, 0, 0, 0),
            "rmol": (0, 2, 2, 0, 0, 0, 5, 0, 1, 1, 3, 4),
            "wspd": (0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
            "zol": (0, 2, 1, 0, 0, 0, 4, 0, 1, 1, 3, 2),
        },
        (3, 2, 1): {
            **WATER_CUDA_ULP[(3, 2, 1)],
            "cda": (0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 5),
            "chs2": (0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 3, 2),
            "cqs2": (0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 2),
            "lh": (0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "qsfc": (0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0),
            "qstar": (0, 1, 0, 1, 0, 2, 2, 0, 7, 0, 0, 0),
            "u10": (0, 1, 0, 0, 2, 1, 0, 1, 0, 2, 1, 0),
            "ust": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1),
            "znt": (0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0),
        },
    },
}
#: A167: NVRTC 12.9.86, the compiler of the default recast-woof[gpu] extra
#: (cupy-cuda12x), reads the sm_120 row A146 re-recorded under 13.4: every
#: reading this file's tests take, and the device result behind each, is
#: bit-identical under the two compilers.  Before this row 12.9.86 failed
#: here by name (tests/_toolchain_rows.py).  MEASURED 2026-10-01 on test host 4's
#: RTX 5070 Ti and test host 2's RTX 5090, two processes per compiler, at
#: integrate/2.8 9dbb4a2db.
WATER_CUDA_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    WATER_CUDA_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89 reads the sm_120 row.  Since the cross-architecture rounding pins
#: (xnode-identity, ecbd106a9 and the A146 __fdiv_rn spellings) the RTX 4090
#: returns the RTX 5090's words for every output these gates compare:
#: 16,518 words, zero differing, NVRTC 13.4, 2026-10-05.  The default row,
#: recorded before those pins, no longer describes it: it failed 31 gates
#: on the RTX 4090 with no kernel change.  MEASURED on test host 1's RTX 4090
#: under NVRTC 12.9, 13.3 and 13.4: every gate passes on the sm_120 row.
WATER_CUDA_ULP_BY_TOOLCHAIN[("89", (12, 9))] = WATER_CUDA_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WATER_CUDA_ULP_BY_TOOLCHAIN[("89", (13, 3))] = WATER_CUDA_ULP_BY_TOOLCHAIN[("120", (13, 4))]
WATER_CUDA_ULP_BY_TOOLCHAIN[("89", (13, 4))] = WATER_CUDA_ULP_BY_TOOLCHAIN[("120", (13, 4))]


@requires_gpu
@pytest.mark.parametrize("isftcflx", ISFTCFLX_SWEEP)
@pytest.mark.parametrize("itimestep,isfflx", STAGES)
def test_cuda_matches_the_water_oracle(isftcflx, itimestep, isfflx):
    import cupy as cp

    from woof.core.mynn_sfclay import mynn_surface_layer

    key = (isftcflx, itimestep, isfflx)
    rows = _column_rows()[key]
    values = {
        name: cp.asarray(
            _f32(rows, INPUT_ALIASES.get(name, name)).reshape(1, -1)
        )
        for name in INPUT_NAMES
    }
    actual = mynn_surface_layer(
        values,
        dx=float(np.float32(rows[0]["dx"])),
        itimestep=itimestep,
        isfflx=isfflx,
        isftcflx=isftcflx,
        mol=cp.asarray(_f32(rows, "mol_input").reshape(1, -1)),
        ustm=cp.asarray(_f32(rows, "ustm_input").reshape(1, -1)),
    )
    for name in OUTPUT_NAMES:
        expected = _f32(rows, name)
        have = cp.asnumpy(getattr(actual, name)).reshape(-1)
        residue = fp32_ulp_distance(have, expected)
        budget = _budget(
            toolchain_row(WATER_CUDA_ULP_BY_TOOLCHAIN, WATER_CUDA_ULP,
                          "WATER_CUDA_ULP_BY_TOOLCHAIN"),
            key, name, len(rows))
        over = np.nonzero(residue > budget)[0]
        assert not over.size, (
            f"{name} at isftcflx={isftcflx} step={itimestep} "
            f"isfflx={isfflx} exceeds its measured CUDA budget on "
            + ", ".join(
                f"{CASES[i]} ({int(residue[i])} > {int(budget[i])})"
                for i in over
            )
        )


@requires_gpu
@pytest.mark.parametrize("isftcflx", [4, 5, -1, True])
def test_undefined_isftcflx_is_refused_before_the_kernel_launches(isftcflx):
    import cupy as cp

    from woof.core.mynn_sfclay import mynn_surface_layer

    rows = _column_rows()[(0, 1, 1)]
    values = {
        name: cp.asarray(
            _f32(rows, INPUT_ALIASES.get(name, name)).reshape(1, -1)
        )
        for name in INPUT_NAMES
    }
    with pytest.raises(ValueError, match="isftcflx"):
        mynn_surface_layer(values, isftcflx=isftcflx)
