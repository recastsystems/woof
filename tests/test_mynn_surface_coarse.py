"""WRF-oracle parity for the MYNN surface layer above the 5 km DX threshold.

module_sf_mynn.F:584 adds a Beljaars (1995) subgrid-scale gust velocity to
the wind speed that every stability calculation below it is built on::

    VSGD = 0.32 * (max(DX/5000. - 1., 0.))**.33
    WSPD = SQRT(WSPD*WSPD + WSTAR*WSTAR + VSGD*VSGD)

``max(DX/5000.-1., 0.)`` is identically zero for every DX <= 5000 m, so the
narrow and widened fixtures -- both of which hardcode DX = 3000.0 in their
Fortran harnesses -- pin only VSGD == 0.  The live half of the branch was
never compared against WRF at all, and every 12 km parent domain runs it on
every column of every timestep.

``tests/data/oracles/mynn/surface-layer-coarse.csv`` closes that: the same
unmodified SFCLAY1D_mynn, the same ten columns as the widened fixture, swept
across DX = 3000 / 5000 / 5001 / 12000 / 27000 m over the widened fixture's
three stages -- (itimestep, isfflx) = (1, 1), (2, 1) and (1, 0)
(``tools/mynn_wrf461_oracle/run_surface_layer_coarse.F90``, built by
``build_coarse.sh`` against the same pinned WRF commit and source hash).

The ISFFLX=0 stage was the hole in the previous revision of this sweep: the
fixture carried only (1, 1) and (2, 1), so the arm of :1027-1044 that leaves
HFX/QFX and the exchange coefficients at zero instead of diagnosing them was
never run at any spacing above the threshold.  It is now.

The DX = 3000 rows of the fixture reproduce the pinned widened fixture bit for
bit -- all three stages -- which is what makes DX the only variable that
changed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from _toolchain_rows import toolchain_row
from test_mynn_surface import WIDE_ULP as _CPU_WIDE_ULP
from test_mynn_surface_gpu import WIDE_ULP as _GPU_WIDE_ULP

from woof.core.fp32_ulp import fp32_ulp_distance
from woof.core.mynn_surface import mynn_surface_layer_default


ORACLE_DIR = Path(__file__).parents[1] / "tests" / "data" / "oracles" / "mynn"
COARSE_ORACLE = ORACLE_DIR / "surface-layer-coarse.csv"
WIDE_ORACLE = ORACLE_DIR / "surface-layer-wide.csv"
HARNESS = (
    Path(__file__).parents[1] / "tools" / "mynn_wrf461_oracle"
    / "run_surface_layer_coarse.F90"
)
COARSE_VALIDATOR = (
    Path(__file__).parents[1] / "tools" / "mynn_wrf461_oracle"
    / "validate_coarse_oracle.py"
)

#: Identity of the fixture and of the harness that produced it.  Both are
#: pinned so a silently regenerated or hand-edited oracle cannot be passed
#: off as WRF's answer.  ``build_coarse.sh`` reproduces the CSV byte for
#: byte from the pinned WRF v4.6.1 commit d66e442f.
#:
#: The CSV is covered by ``woof/data/** -text`` so its bytes survive a
#: Windows checkout; the .F90 is not, and git hands it back with CRLF there,
#: so its hash is taken over LF-normalized bytes.
COARSE_ORACLE_SHA256 = (
    "1ee119c7c4f46b1b0c433db30264c782e36a612c7fc49eeb2bf106b6c34b20c4"
)
HARNESS_LF_SHA256 = (
    "7c297a30daf7bc70bc40aaed39af64bec9c5a72aff04b4c860212591dbfe37d4"
)

#: The sweep the Fortran harness writes.  5000 is on the threshold, where
#: ``max()`` still clamps VSGD to zero; 5001 is the smallest live VSGD.
DX_SWEEP = (3000.0, 5000.0, 5001.0, 12000.0, 27000.0)
#: The three SFCLAY1D_mynn stages, now at every spacing.
STAGES = ((1, 1), (2, 1), (1, 0))
CASES = (
    "strong_stable_land", "clipped_stable_land", "damped_stable_land",
    "neutral_land", "free_convective_land", "land_qsfc_unset",
    "thin_land_level2_wind", "thin_land_log10_wind", "midres_water",
    "coarse_water",
)

INPUT_ALIASES = {
    "hfx": "hfx_input", "qfx": "qfx_input", "znt": "znt_input",
    "qsfc": "qsfc_input", "ust": "ust_input",
}
INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)
OUTPUT_NAMES = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm", "znt",
)

#: module_sf_mynn.F:1027-1044.  With ISFFLX<1 WRF assigns these thirteen
#: outputs the literal constant 0 and leaves the other twenty-two on the same
#: code path the ISFFLX=1 first step takes.  So the ISFFLX=0 stages need no
#: tables of their own: each reuses its ``(dx, 1)`` table for the twenty-two,
#: which is exact because the outputs are bit-identical, and 0 for these
#: thirteen, which is exact because they are 0.0 on both sides.  Both halves
#: are pinned at every spacing by
#: ``test_the_isfflx0_stage_zeroes_thirteen_outputs_at_every_spacing``.
ISFFLX0_ZEROED = (
    "hfx", "qfx", "flhc", "flqc", "lh", "chs", "ch", "chs2", "cqs2",
    "ck", "cd", "cka", "cda",
)

#: DX = 5000 is on the threshold, so ``max()`` clamps VSGD to zero and WRF
#: writes byte-identical rows to DX = 3000 -- as do both ports.  One table
#: therefore covers both spacings exactly, with no borrowed margin.  The
#: identity is asserted, not assumed:
#: ``test_the_threshold_spacing_is_bitwise_the_3000_metre_spacing`` and
#: ``test_the_cuda_kernel_also_clamps_vsgd_at_the_threshold``.
DX_TABLE_ALIAS = {5000.0: 3000.0}

# The FP32 ULP distance from this oracle, measured per (stage, output,
# column) -- one integer per compared number, in ``CASES`` order -- where a
# stage is one (dx, itimestep), not one dx.  An output a stage does not name is
# bitwise on every column there and is required to stay bitwise: the lookup
# returns zeros.  There is no margin to justify, because there is no margin:
# each entry is the residue measured at that element and the gate is
# ``residue <= entry`` column by column.
#
# The previous revision keyed one table per DX, maxed over both timesteps and
# all ten columns.  Measured on this machine that left 190744 ULP of unearned
# margin and 322 of 1050 (stage, output) comparisons unable to fail on a 1-ULP
# regression, the worst by +118.
#
# The residue has the same two causes it has at DX = 3000 and no third one:
#
#   * CPU -- QFX/LH lead at 10-11 ULP because ``FLQC*(QSFCMR-QV1)`` cancels
#     three or four significant digits; the columns show it sitting in
#     ``midres_water`` and ``land_qsfc_unset``.  The rest tracks NumPy's
#     float32 exp/log/pow against the glibc functions gfortran linked.
#   * CUDA -- every large entry is in the ``thin_land_log10_wind`` column,
#     eighth of ten, documented in ``test_mynn_surface_gpu.py``: ``powf`` is
#     one ULP off gfortran on its level-1 Exner factor and the DTHVDZ
#     cancellation amplifies that into BR/ZOL/RMOL/PSIM and everything below.
#
# What matters for *this* fixture is that neither grows with DX.  WSPD, the
# output VSGD actually enters, is bitwise on the CPU and 1 ULP on CUDA at every
# spacing in the sweep, and BR is bitwise on the CPU throughout.  The DX > 5 km
# branch is reproduced exactly by both ports.
#
# ``CPU_ULP`` is the elementwise maximum over THREE NumPy builds -- Windows
# NumPy 2.2.6 / CPython 3.13.7, which runs pytest on this box; the WSL
# Ubuntu-24.04 NumPy 2.4.3 / CPython 3.12.3 that ``build_coarse.sh`` runs
# ``validate_coarse_oracle.py`` under; and Ubuntu-22.04 NumPy 2.5.1 / CPython
# 3.12.13 on the second RTX 5090 host.  The full derivation is in the note in
# ``tests/test_mynn_surface.py``; in short, the three disagree at 212 of the
# 3750 elements measured here, and for exactly two reasons: NumPy's float32
# ``arctan``, which 2.5.1 makes bit-identical to the glibc ``atanf`` gfortran
# linked and the two older builds do not, and NumPy's float32 ``**``, which is
# the platform ``powf`` -- glibc's on both Linux builds, the MSVC runtime's on
# Windows.  A table measured on one platform alone is red on the others: the
# previous revision's was Windows-only and had been failing 13 comparisons
# under WSL, and the two-platform revision before this one failed six
# comparisons here under NumPy 2.5.1.
#
# Three entries below are marked ``+2.5.1``: they are the only ones in this
# file that the third build raised, each by exactly 1 ULP, and they cost the
# other two builds 6 ULP of bite between them.  NumPy 2.5.1 is nonetheless the
# build closest to WRF overall -- 1540 ULP summed over the 5150 CPU elements
# of this file and ``test_mynn_surface.py``, against 1753 for 2.2.6 and 1866
# for 2.4.3.
#
# ``GPU_ULP`` is unchanged by any of this: it was measured on Windows / NVRTC
# 13.0, and it is green unaltered on the second host's NVRTC 12.8 / driver
# 570.133.20, so the kernel's residue does not carry a platform axis the way
# the NumPy reference does.
#
# These are ratchets: lower them as the FP32 shims are unified, never raise.
CPU_ULP = {
    (3000.0, 1): dict(_CPU_WIDE_ULP[(1, 1)]),
    (3000.0, 2): dict(_CPU_WIDE_ULP[(2, 1)]),
    (5001.0, 1): {
        "zol": (4, 0, 0, 0, 0, 2, 0, 0, 1, 0),
        "rmol": (3, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "ust": (3, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "ustm": (3, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "mol": (1, 0, 0, 0, 0, 2, 1, 0, 0, 0),
        "psim": (2, 0, 0, 0, 1, 4, 0, 0, 1, 1),
        "psih": (1, 0, 0, 0, 0, 1, 1, 0, 1, 0),
        "chs": (3, 0, 0, 0, 1, 2, 1, 0, 0, 0),
        "chs2": (2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "cqs2": (2, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ch": (3, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "flhc": (4, 0, 0, 0, 1, 1, 1, 0, 0, 0),
        "flqc": (3, 0, 0, 0, 1, 0, 1, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (4, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "qfx": (3, 0, 0, 0, 0, 2, 1, 0, 11, 0),
        "lh": (4, 0, 0, 0, 0, 2, 1, 0, 7, 0),
        # free_convective_land: 0 on NumPy 2.2.6/2.4.3, 1 on 2.5.1.   +2.5.1
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "v10": (1, 0, 0, 0, 1, 2, 0, 0, 0, 0),
        "ck": (1, 0, 0, 0, 2, 1, 1, 0, 0, 0),
        "cka": (3, 0, 0, 0, 1, 1, 1, 0, 0, 0),
        "cd": (4, 0, 0, 0, 3, 0, 2, 0, 0, 0),
        "cda": (5, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 5, 0),
    },
    (5001.0, 2): {
        "zol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "rmol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "mol": (0, 0, 2, 0, 0, 1, 0, 0, 1, 0),
        "psim": (0, 0, 2, 0, 0, 1, 0, 0, 1, 1),
        "psih": (0, 0, 3, 0, 0, 0, 0, 0, 1, 0),
        "chs": (0, 0, 1, 0, 0, 2, 0, 0, 1, 0),
        "chs2": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        "cqs2": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "flhc": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "flqc": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 0, 0, 0, 10, 0),
        "u10": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "ck": (0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
        "cka": (0, 0, 2, 0, 0, 0, 0, 0, 0, 0),
        "cd": (0, 0, 2, 0, 0, 0, 0, 0, 2, 0),
        "cda": (0, 0, 3, 0, 0, 0, 0, 0, 0, 0),
        # midres_water re-measured 4 -> 5 when the psi tables moved onto the
        # glibc transcriptions; identical on all three builds.       +psi_libm
        "qstar": (0, 0, 0, 0, 0, 1, 0, 0, 5, 0),
    },
    (12000.0, 1): {
        "zol": (0, 0, 0, 0, 1, 3, 1, 2, 0, 0),
        "rmol": (0, 0, 0, 0, 1, 3, 1, 3, 0, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "mol": (0, 0, 0, 0, 1, 0, 3, 1, 1, 0),
        # land_qsfc_unset: 1 on NumPy 2.2.6/2.4.3, 2 on 2.5.1.        +2.5.1
        "psim": (0, 0, 0, 0, 1, 2, 0, 2, 1, 1),
        "psih": (2, 0, 0, 0, 1, 2, 3, 4, 1, 0),
        "chs": (0, 0, 0, 0, 1, 0, 2, 2, 1, 0),
        "chs2": (2, 0, 0, 0, 0, 0, 0, 4, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 0, 1, 2, 0, 0),
        "ch": (0, 0, 0, 0, 2, 0, 2, 2, 1, 0),
        "flhc": (0, 0, 0, 0, 2, 0, 2, 2, 1, 0),
        "flqc": (1, 0, 0, 0, 1, 0, 2, 2, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (0, 0, 0, 0, 2, 0, 2, 3, 1, 0),
        "qfx": (1, 0, 0, 0, 0, 2, 2, 2, 10, 0),
        "lh": (1, 0, 0, 0, 0, 3, 3, 2, 6, 0),
        "u10": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ck": (0, 0, 0, 0, 2, 3, 1, 1, 0, 0),
        "cka": (1, 0, 0, 0, 1, 0, 1, 4, 2, 0),
        "cd": (0, 0, 0, 0, 2, 2, 3, 0, 0, 0),
        "cda": (0, 0, 0, 0, 0, 0, 0, 2, 4, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (12000.0, 2): {
        "zol": (0, 0, 2, 0, 1, 0, 0, 1, 1, 0),
        "rmol": (0, 0, 2, 0, 1, 0, 0, 2, 1, 0),
        "ust": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "psim": (0, 0, 1, 0, 1, 1, 1, 2, 1, 1),
        "psih": (0, 0, 1, 0, 0, 0, 0, 1, 1, 0),
        "chs": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "chs2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "ch": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "flhc": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 0, 0, 0, 10, 0),
        "u10": (0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 1, 0),
        "ck": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "cka": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "cd": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "cda": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (27000.0, 1): {
        "zol": (3, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "rmol": (2, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "ust": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "ustm": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "psim": (1, 0, 0, 0, 0, 6, 0, 3, 1, 0),
        "psih": (1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
        "chs": (0, 0, 0, 0, 0, 2, 0, 2, 0, 0),
        # strong_stable_land 1 -> 2, gale_land 1 -> 0: the same re-measurement,
        # net zero budget on this row.                               +psi_libm
        "chs2": (2, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 1, 0, 3, 0, 2),
        "ch": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "flhc": (1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 3, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (1, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        # land_qsfc_unset 1 -> 2, strong_stable_land 1 -> 0; net zero on
        # this row.                                                  +psi_libm
        "qfx": (0, 0, 0, 0, 0, 2, 0, 1, 11, 0),
        # land_qsfc_unset 1 -> 2, strong_stable_land 1 -> 0; net zero on
        # this row.                                                  +psi_libm
        "lh": (0, 0, 0, 0, 0, 2, 0, 2, 6, 0),
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "v10": (1, 0, 0, 0, 1, 2, 0, 0, 0, 0),
        "ck": (2, 0, 0, 0, 1, 0, 1, 1, 0, 0),
        "cka": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cd": (3, 0, 0, 0, 2, 0, 0, 1, 0, 0),
        "cda": (2, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (27000.0, 2): {
        "zol": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "rmol": (0, 0, 1, 0, 0, 0, 0, 2, 0, 0),
        "ust": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        "ustm": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "psim": (0, 0, 1, 0, 0, 3, 0, 4, 1, 1),
        "psih": (0, 0, 0, 0, 0, 0, 0, 2, 1, 0),
        "chs": (0, 0, 1, 0, 0, 1, 0, 2, 0, 0),
        "chs2": (0, 0, 1, 0, 0, 1, 0, 1, 0, 0),
        "cqs2": (0, 0, 1, 0, 0, 1, 1, 1, 0, 0),
        "ch": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0),
        "flhc": (0, 0, 2, 0, 0, 1, 0, 1, 0, 0),
        # thin_land_log10_wind 0 -> 1, and two columns down; net -2 ULP on
        # this row.                                                  +psi_libm
        "flqc": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 3, 0, 0, 1, 0, 1, 0, 0),
        # QFX/LH at the 27 km spacing are the FLQC*(QSFCMR-QV1) and XLV*QFX
        # cancellation columns named in the note above; re-measuring them on
        # the glibc psi tables moves midres_water 6 -> 8 and
        # thin_land_log10_wind 0 -> 1.  Identical on all three builds, and
        # paid for on this stage by cka below.                       +psi_libm
        "qfx": (0, 0, 0, 0, 0, 0, 0, 1, 8, 0),
        # Same two columns as qfx: 7 -> 9 and 0 -> 1.                +psi_libm
        "lh": (0, 0, 0, 0, 0, 0, 0, 1, 9, 0),
        "u10": (0, 0, 1, 0, 1, 1, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 1, 2, 0, 0, 0, 0),
        "ck": (2, 0, 0, 0, 1, 0, 0, 2, 0, 0),
        # Re-measured 6 -> 2 ULP summed.                             +psi_libm
        "cka": (0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 1, 0, 0),
        "cda": (0, 0, 2, 0, 0, 2, 0, 0, 2, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
}

GPU_ULP = {
    (3000.0, 1): dict(_GPU_WIDE_ULP[(1, 1)]),
    (3000.0, 2): dict(_GPU_WIDE_ULP[(2, 1)]),
    (5001.0, 1): {
        "zol": (1, 0, 2, 0, 0, 1, 2, 87, 0, 1),
        "rmol": (1, 0, 2, 0, 0, 1, 3, 116, 0, 1),
        "ust": (0, 1, 0, 1, 1, 0, 1, 21, 0, 0),
        "ustm": (0, 1, 0, 1, 1, 0, 1, 21, 0, 0),
        "mol": (0, 0, 0, 0, 0, 2, 1, 36, 0, 0),
        "psim": (0, 0, 0, 0, 1, 4, 3, 118, 0, 1),
        "psih": (0, 0, 0, 0, 0, 3, 1, 68, 0, 0),
        "chs": (0, 2, 0, 0, 1, 2, 2, 76, 0, 0),
        "chs2": (0, 1, 0, 1, 0, 0, 3, 69, 0, 0),
        "cqs2": (0, 2, 0, 1, 1, 0, 2, 69, 0, 0),
        "ch": (0, 1, 0, 0, 0, 0, 2, 74, 0, 0),
        "flhc": (0, 1, 0, 0, 1, 1, 2, 43, 0, 0),
        "flqc": (0, 1, 0, 1, 1, 0, 2, 44, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
        "hfx": (0, 1, 0, 0, 0, 0, 2, 17, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 1, 2, 44, 6, 2),
        "lh": (0, 0, 0, 0, 0, 1, 2, 52, 4, 1),
        "u10": (0, 0, 1, 2, 0, 1, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 0, 1, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (0, 0, 0, 0, 1, 1, 4, 63, 0, 0),
        "cka": (0, 0, 1, 1, 1, 0, 2, 73, 0, 0),
        "cd": (0, 0, 0, 0, 2, 2, 4, 68, 0, 0),
        "cda": (0, 0, 2, 2, 1, 0, 2, 41, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 5, 2),
    },
    (5001.0, 2): {
        "zol": (0, 0, 4, 0, 1, 3, 3, 87, 0, 1),
        "rmol": (0, 0, 3, 0, 1, 5, 4, 116, 0, 1),
        "ust": (0, 0, 2, 1, 0, 2, 0, 22, 0, 0),
        "ustm": (0, 0, 2, 1, 0, 2, 0, 22, 0, 0),
        "mol": (0, 0, 2, 0, 0, 1, 1, 36, 0, 0),
        "psim": (0, 0, 3, 0, 0, 3, 3, 118, 0, 3),
        "psih": (0, 0, 3, 0, 1, 3, 2, 68, 0, 0),
        "chs": (0, 0, 3, 1, 0, 0, 0, 47, 0, 0),
        "chs2": (0, 0, 3, 1, 0, 2, 1, 43, 0, 0),
        "cqs2": (0, 0, 2, 1, 1, 2, 0, 43, 0, 0),
        "ch": (0, 0, 2, 1, 0, 1, 0, 48, 0, 0),
        "flhc": (0, 0, 3, 1, 0, 1, 1, 55, 0, 0),
        "flqc": (0, 0, 5, 1, 0, 0, 1, 55, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (0, 0, 4, 0, 0, 1, 1, 19, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 0, 1, 55, 4, 1),
        "lh": (0, 0, 0, 0, 0, 0, 1, 32, 5, 1),
        "u10": (0, 0, 1, 2, 0, 1, 0, 0, 1, 0),
        "v10": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (0, 0, 1, 0, 0, 1, 2, 61, 1, 0),
        "cka": (0, 0, 3, 1, 0, 1, 1, 73, 0, 0),
        "cd": (0, 0, 2, 0, 0, 0, 2, 69, 2, 0),
        "cda": (0, 0, 5, 2, 0, 2, 0, 44, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 5, 2),
    },
    (12000.0, 1): {
        "zol": (2, 0, 3, 0, 1, 1, 1, 81, 0, 2),
        "rmol": (1, 0, 2, 0, 1, 0, 1, 108, 0, 2),
        "ust": (2, 1, 1, 0, 0, 1, 0, 20, 0, 0),
        "ustm": (2, 1, 1, 0, 0, 1, 0, 20, 0, 0),
        "mol": (0, 0, 0, 0, 1, 0, 0, 38, 0, 0),
        "psim": (1, 0, 3, 0, 1, 2, 0, 108, 0, 3),
        "psih": (1, 0, 2, 0, 1, 0, 1, 63, 0, 1),
        "chs": (1, 0, 2, 0, 1, 2, 0, 37, 0, 0),
        "chs2": (2, 1, 2, 0, 1, 1, 0, 68, 0, 0),
        "cqs2": (2, 1, 2, 0, 0, 2, 0, 69, 0, 0),
        "ch": (1, 1, 2, 0, 2, 3, 0, 36, 0, 0),
        "flhc": (1, 1, 2, 0, 2, 3, 0, 42, 0, 0),
        "flqc": (2, 1, 1, 0, 1, 1, 0, 44, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
        "hfx": (1, 1, 1, 0, 2, 1, 0, 21, 0, 0),
        "qfx": (2, 0, 0, 0, 0, 1, 0, 44, 5, 2),
        "lh": (2, 0, 0, 0, 0, 1, 0, 52, 3, 1),
        "u10": (1, 0, 2, 2, 0, 1, 0, 0, 0, 0),
        "v10": (1, 0, 1, 0, 1, 2, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 90, 0, 0),
        "ck": (0, 0, 0, 0, 1, 0, 0, 62, 0, 0),
        "cka": (1, 0, 1, 1, 1, 1, 0, 71, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 70, 0, 0),
        "cda": (2, 0, 3, 2, 0, 2, 0, 41, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 2),
    },
    (12000.0, 2): {
        "zol": (2, 0, 1, 0, 2, 2, 1, 83, 0, 1),
        "rmol": (1, 0, 1, 0, 3, 4, 2, 111, 0, 1),
        "ust": (1, 1, 0, 0, 0, 2, 0, 21, 0, 0),
        "ustm": (1, 1, 0, 0, 0, 2, 0, 21, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 37, 0, 0),
        "psim": (1, 0, 0, 0, 2, 2, 2, 111, 0, 2),
        "psih": (0, 0, 0, 0, 1, 1, 1, 66, 0, 0),
        "chs": (1, 1, 0, 0, 0, 3, 0, 48, 0, 0),
        "chs2": (2, 0, 0, 0, 0, 2, 0, 43, 0, 0),
        "cqs2": (2, 1, 0, 0, 0, 3, 1, 44, 0, 0),
        "ch": (2, 4, 0, 0, 0, 0, 0, 47, 0, 0),
        "flhc": (1, 2, 0, 0, 0, 0, 0, 55, 0, 0),
        "flqc": (2, 1, 0, 0, 1, 1, 1, 56, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (1, 2, 0, 0, 0, 0, 0, 21, 0, 0),
        "qfx": (2, 0, 0, 0, 0, 1, 1, 56, 4, 2),
        "lh": (2, 0, 0, 0, 0, 1, 1, 34, 5, 2),
        "u10": (0, 0, 1, 2, 1, 1, 0, 0, 0, 0),
        "v10": (0, 0, 1, 0, 1, 2, 0, 0, 1, 2),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 90, 0, 0),
        "ck": (2, 0, 0, 0, 1, 2, 1, 66, 0, 0),
        "cka": (2, 0, 1, 1, 1, 1, 1, 74, 0, 1),
        "cd": (3, 0, 0, 0, 1, 0, 2, 75, 0, 0),
        "cda": (2, 0, 1, 2, 0, 2, 2, 43, 0, 2),
        "wstar": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 2),
    },
    (27000.0, 1): {
        "zol": (3, 0, 4, 0, 0, 3, 0, 75, 0, 0),
        "rmol": (2, 0, 3, 0, 0, 2, 0, 100, 0, 0),
        "ust": (1, 1, 1, 1, 0, 0, 0, 19, 0, 0),
        "ustm": (1, 1, 1, 1, 0, 0, 0, 19, 0, 0),
        "mol": (0, 0, 0, 0, 0, 0, 0, 40, 0, 0),
        "psim": (2, 0, 2, 0, 1, 4, 0, 101, 0, 1),
        "psih": (1, 0, 1, 0, 0, 1, 0, 60, 0, 1),
        "chs": (0, 1, 1, 2, 0, 0, 0, 36, 0, 0),
        "chs2": (3, 1, 2, 1, 0, 0, 0, 66, 0, 0),
        "cqs2": (3, 2, 2, 2, 0, 0, 0, 66, 0, 0),
        "ch": (1, 2, 1, 1, 0, 0, 0, 35, 0, 0),
        "flhc": (1, 2, 2, 1, 0, 0, 0, 41, 0, 0),
        "flqc": (1, 1, 1, 3, 0, 2, 0, 43, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
        "hfx": (1, 2, 2, 0, 0, 0, 0, 25, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 0, 0, 43, 6, 2),
        "lh": (1, 0, 0, 0, 0, 0, 0, 52, 3, 2),
        "u10": (0, 0, 1, 2, 1, 0, 0, 0, 0, 0),
        "v10": (1, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 86, 0, 0),
        "ck": (2, 0, 2, 0, 1, 1, 1, 63, 0, 0),
        "cka": (1, 0, 1, 1, 0, 0, 0, 68, 0, 0),
        "cd": (3, 0, 3, 0, 2, 0, 3, 72, 0, 0),
        "cda": (2, 0, 3, 2, 0, 0, 0, 39, 0, 0),
        "wstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 2),
    },
    (27000.0, 2): {
        "zol": (4, 0, 3, 0, 1, 1, 0, 76, 0, 1),
        "rmol": (3, 0, 2, 0, 1, 2, 0, 101, 0, 1),
        "ust": (1, 0, 0, 1, 0, 0, 0, 19, 0, 0),
        "ustm": (1, 0, 0, 1, 0, 0, 0, 19, 0, 0),
        "mol": (2, 0, 1, 0, 0, 0, 0, 39, 0, 1),
        "psim": (2, 0, 0, 0, 0, 3, 0, 101, 0, 1),
        "psih": (2, 0, 2, 0, 1, 1, 0, 61, 0, 1),
        "chs": (4, 0, 1, 1, 0, 0, 0, 46, 0, 0),
        "chs2": (2, 0, 0, 1, 0, 0, 0, 42, 0, 0),
        "cqs2": (1, 0, 0, 1, 0, 0, 1, 43, 0, 0),
        "ch": (4, 0, 1, 2, 0, 0, 0, 45, 0, 1),
        "flhc": (2, 0, 1, 1, 0, 0, 0, 52, 0, 1),
        "flqc": (1, 0, 1, 0, 0, 0, 0, 52, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 2),
        "hfx": (2, 0, 1, 0, 0, 0, 0, 25, 0, 1),
        "qfx": (1, 0, 0, 0, 0, 0, 0, 52, 4, 2),
        "lh": (1, 0, 0, 0, 0, 0, 0, 31, 4, 3),
        "u10": (0, 0, 1, 2, 0, 1, 0, 0, 0, 0),
        "v10": (1, 0, 1, 0, 0, 2, 0, 0, 0, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 86, 0, 0),
        "ck": (4, 0, 1, 0, 0, 1, 0, 64, 0, 0),
        "cka": (2, 0, 1, 1, 0, 0, 0, 67, 0, 0),
        "cd": (3, 0, 2, 0, 0, 1, 0, 73, 0, 0),
        "cda": (3, 0, 0, 2, 0, 0, 0, 36, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 3),
        "znt": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    },
}


def _rows(path: Path):
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _stage(dx: float, itimestep: int, isfflx: int = 1):
    rows = [
        row for row in _rows(COARSE_ORACLE)
        if float(row["dx"]) == dx and int(row["itimestep"]) == itimestep
        and int(row["isfflx"]) == isfflx
    ]
    assert tuple(row["case"] for row in rows) == CASES
    skip = ("case", "itimestep", "isfflx", "dx")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0] if key not in skip
    }


#: ``GPU_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 41 rows read more than ``GPU_ULP``, by 1 to 2 ULP, all on
#: qgh, qsfc, qstar, lh and qfx at the last two columns, zol/rmol/psim/psih and
#: the exchange coefficients at a few; every other row reads at or below it,
#: most of the thin_land_log10_wind column's 36 to 118 ULP now 0 to 3.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
GPU_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **GPU_ULP,
        (3000.0, 1): {
            **GPU_ULP[(3000.0, 1)],
            "lh": (1, 0, 0, 0, 0, 0, 4, 3, 0, 2),
            "qfx": (1, 0, 0, 0, 0, 0, 3, 2, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        (3000.0, 2): {
            **GPU_ULP[(3000.0, 2)],
            "lh": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
            "qfx": (0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        (5001.0, 1): {
            **GPU_ULP[(5001.0, 1)],
            "lh": (0, 0, 0, 0, 0, 0, 2, 0, 0, 2),
            "qfx": (0, 0, 0, 0, 0, 0, 2, 0, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 0, 3),
        },
        (5001.0, 2): {
            **GPU_ULP[(5001.0, 2)],
            "lh": (0, 0, 0, 0, 0, 0, 1, 0, 0, 3),
            "psim": (0, 0, 3, 0, 1, 2, 3, 1, 0, 3),
            "qfx": (0, 0, 0, 0, 0, 0, 1, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 2, 0, 0, 0, 3),
        },
        (12000.0, 1): {
            **GPU_ULP[(12000.0, 1)],
            "lh": (2, 0, 0, 0, 0, 0, 0, 2, 0, 2),
            "qfx": (2, 0, 0, 0, 0, 0, 0, 2, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        (12000.0, 2): {
            **GPU_ULP[(12000.0, 2)],
            "lh": (2, 0, 0, 0, 0, 1, 1, 0, 0, 4),
            "qfx": (2, 0, 0, 0, 0, 1, 1, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        (27000.0, 1): {
            **GPU_ULP[(27000.0, 1)],
            "lh": (1, 0, 0, 0, 0, 2, 0, 2, 0, 3),
            "qfx": (1, 0, 0, 0, 0, 1, 0, 1, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
        },
        (27000.0, 2): {
            **GPU_ULP[(27000.0, 2)],
            "lh": (1, 0, 0, 0, 0, 0, 0, 0, 0, 4),
            "qfx": (1, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
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
GPU_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    GPU_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89 reads the sm_120 row.  Since the cross-architecture rounding pins
#: (xnode-identity, ecbd106a9 and the A146 __fdiv_rn spellings) the RTX 4090
#: returns the RTX 5090's words for every output these gates compare:
#: 16,518 words, zero differing, NVRTC 13.4, 2026-10-05.  The default row,
#: recorded before those pins, no longer describes it: it failed 31 gates
#: on the RTX 4090 with no kernel change.  MEASURED on test host 1's RTX 4090
#: under NVRTC 12.9, 13.3 and 13.4: every gate passes on the sm_120 row.
GPU_ULP_BY_TOOLCHAIN[("89", (12, 9))] = GPU_ULP_BY_TOOLCHAIN[("120", (13, 4))]
GPU_ULP_BY_TOOLCHAIN[("89", (13, 3))] = GPU_ULP_BY_TOOLCHAIN[("120", (13, 4))]
GPU_ULP_BY_TOOLCHAIN[("89", (13, 4))] = GPU_ULP_BY_TOOLCHAIN[("120", (13, 4))]


def _inputs(fields):
    return {name: fields[INPUT_ALIASES.get(name, name)] for name in INPUT_NAMES}


def _table(tables, dx: float, itimestep: int):
    return tables[(DX_TABLE_ALIAS.get(dx, dx), itimestep)]


def _budget(table, name, zeroed=()):
    """The measured per-column residue for ``name``, or zeros."""

    row = () if name in zeroed else table.get(name, ())
    if not row:
        return np.zeros(len(CASES), dtype=np.int64)
    assert len(row) == len(CASES), name
    return np.asarray(row, dtype=np.int64)


def _max_ulp(actual, expected):
    return int(
        fp32_ulp_distance(np.ravel(actual), np.ravel(expected))
        .max(initial=0)
    )


def _assert_ulp(actual, expected, name, budget):
    """Fail when any column of ``name`` passes its measured ULP residue.

    ``budget`` is per column, so the message names the case that drifted:
    a field-wide number cannot say which of the ten columns moved.
    """

    residue = fp32_ulp_distance(np.ravel(actual), np.ravel(expected))
    budget = np.broadcast_to(
        np.asarray(budget, dtype=np.int64), residue.shape
    )
    over = np.flatnonzero(residue > budget)
    if over.size:
        worst = int(over[np.argmax(residue[over] - budget[over])])
        raise AssertionError(
            f"{name}[{CASES[worst]}]: {int(residue[worst])} ULP from the "
            f"unmodified WRF oracle exceeds the measured {int(budget[worst])} "
            f"({over.size} of {residue.size} columns over)"
        )


def _expected_vsgd(dx: float) -> np.float32:
    """module_sf_mynn.F:584, in FP32."""

    f = np.float32
    return f(f(0.32) * max(f(dx) / f(5000.0) - f(1.0), f(0.0)) ** f(0.33))


def test_the_coarse_fixture_and_its_harness_are_the_pinned_bytes():
    payload = COARSE_ORACLE.read_bytes()
    assert b"\r" not in payload
    assert hashlib.sha256(payload).hexdigest() == COARSE_ORACLE_SHA256
    harness = HARNESS.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(harness).hexdigest() == HARNESS_LF_SHA256


def test_the_coarse_fixture_is_the_widened_fixture_plus_a_dx_sweep():
    """DX must be the only thing that changed, or the sweep proves nothing."""

    rows = _rows(COARSE_ORACLE)
    assert len(rows) == len(DX_SWEEP) * len(STAGES) * len(CASES)
    assert tuple(sorted({float(r["dx"]) for r in rows})) == DX_SWEEP
    assert {
        (int(r["itimestep"]), int(r["isfflx"])) for r in rows
    } == set(STAGES)

    wide = {
        (r["case"], r["itimestep"], r["isfflx"]): r for r in _rows(WIDE_ORACLE)
    }
    shared = [key for key in rows[0] if key != "dx"]
    compared = 0
    for row in rows:
        if float(row["dx"]) != 3000.0:
            continue
        reference = wide[(row["case"], row["itimestep"], row["isfflx"])]
        for key in shared:
            assert row[key] == reference[key], (row["case"], key)
        compared += 1
    assert compared == len(STAGES) * len(CASES)


#: Outputs WRF leaves untouched when VSGD switches on, measured per stage.
#:
#: At the first step REGIME/QGH/QSFC/Q2/GZ1OZ0/CPM have no WSPD dependence at
#: all, WSTAR is computed at :578 -- one line *before* VSGD is added at :584 --
#: and the two water ZNTs move by less than an FP32 ULP: 27 of the 35 outputs
#: move.  At the second step even GZ1OZ0/WSTAR/ZNT move, because the DX branch
#: has by then propagated into the carried UST/MOL that charnock_1955 (:657)
#: and the WSTAR scale read: 30 move.  With ISFFLX=0 the thirteen outputs of
#: :1027-1044 are the literal constant 0 and cannot respond to anything, so
#: only 14 move -- which is the accurate size of the hole this stage closes.
DX_INSENSITIVE = {
    (1, 1): ("regime", "qgh", "qsfc", "q2", "gz1oz0", "wstar", "cpm", "znt"),
    (2, 1): ("regime", "qgh", "qsfc", "q2", "cpm"),
    (1, 0): (
        "regime", "qgh", "qsfc", "q2", "gz1oz0", "wstar", "cpm", "znt",
    ) + ISFFLX0_ZEROED,
}


@pytest.mark.parametrize(("itimestep", "isfflx"), STAGES)
def test_the_wrf_oracle_itself_shows_the_dx_branch_is_live(itimestep, isfflx):
    """Below 5 km VSGD is dead; one metre above it, WRF's own outputs move."""

    base = _stage(3000.0, itimestep, isfflx)

    # max(DX/5000.-1., 0.) clamps at exactly 5000 m, so nothing may move.
    assert _expected_vsgd(5000.0) == 0.0
    at_threshold = _stage(5000.0, itimestep, isfflx)
    for name in OUTPUT_NAMES:
        np.testing.assert_array_equal(
            at_threshold[name], base[name], err_msg=f"dx=5000 {name}"
        )

    frozen = DX_INSENSITIVE[(itimestep, isfflx)]
    for dx in (5001.0, 12000.0, 27000.0):
        stage = _stage(dx, itimestep, isfflx)
        assert _expected_vsgd(dx) > 0.0
        still = tuple(
            name for name in OUTPUT_NAMES
            if np.array_equal(stage[name], base[name])
        )
        assert set(still) == set(frozen), (dx, itimestep, isfflx)
        assert len(OUTPUT_NAMES) - len(still) == {
            (1, 1): 27, (2, 1): 30, (1, 0): 14,
        }[(itimestep, isfflx)]

    # WSPD is where VSGD enters, so it is the output that pins the magnitude.
    # module_sf_mynn.F:556 + :585-586 in full, because at itimestep=2 WSTAR
    # itself has moved with DX and quadrature against the DX=3000 WSPD no
    # longer closes -- it misses by 2e-4 on the three convective columns.
    coarse = _stage(12000.0, itimestep, isfflx)
    quadrature = np.maximum(
        np.sqrt(
            coarse["u1"].astype(np.float64) ** 2
            + coarse["v1"].astype(np.float64) ** 2
            + coarse["wstar"].astype(np.float64) ** 2
            + float(_expected_vsgd(12000.0)) ** 2
        ),
        0.1,   # wmin, module_sf_mynn.F:82
    )
    np.testing.assert_allclose(coarse["wspd"], quadrature, rtol=2.0e-7)
    # ...and the same identity without VSGD is what the DX=3000 rows satisfy,
    # so the term really is the only difference.
    np.testing.assert_allclose(
        base["wspd"],
        np.maximum(
            np.sqrt(
                base["u1"].astype(np.float64) ** 2
                + base["v1"].astype(np.float64) ** 2
                + base["wstar"].astype(np.float64) ** 2
            ),
            0.1,
        ),
        rtol=2.0e-7,
    )


@pytest.mark.parametrize(("itimestep", "isfflx"), STAGES)
def test_the_threshold_spacing_is_bitwise_the_3000_metre_spacing(
    itimestep, isfflx
):
    """What licenses ``DX_TABLE_ALIAS``.

    DX = 5000 and DX = 3000 both clamp VSGD to zero, so WRF writes identical
    rows and the CPU reference produces identical output -- which makes the
    residue identical and one table exact for both, rather than one spacing
    borrowing the other's margin.
    """

    base = _stage(3000.0, itimestep, isfflx)
    threshold = _stage(5000.0, itimestep, isfflx)
    near = mynn_surface_layer_default(
        _inputs(base), dx=3000.0, itimestep=itimestep, isfflx=isfflx,
        mol=base["mol_input"], ustm=base["ustm_input"],
    )
    at = mynn_surface_layer_default(
        _inputs(threshold), dx=5000.0, itimestep=itimestep, isfflx=isfflx,
        mol=threshold["mol_input"], ustm=threshold["ustm_input"],
    )
    for name in OUTPUT_NAMES:
        np.testing.assert_array_equal(at[name], near[name], err_msg=name)
    assert _table(CPU_ULP, 5000.0, itimestep) is _table(
        CPU_ULP, 3000.0, itimestep
    )


@pytest.mark.parametrize("dx", DX_SWEEP)
def test_the_isfflx0_stage_zeroes_thirteen_outputs_at_every_spacing(dx):
    """What licenses reusing the ``(dx, 1)`` table for the ISFFLX=0 stage.

    module_sf_mynn.F:1027-1044 makes ISFFLX<1 a pure post-processing branch:
    thirteen outputs become the constant 0 and the other twenty-two stay on
    the ISFFLX=1 code path.  Pinning both halves is what makes the reuse a
    measured consequence of the branch rather than a borrowed budget -- and
    the ISFFLX=1 stage is checked to be nonzero in those thirteen, so the zero
    test discriminates something.
    """

    off = _stage(dx, 1, 0)
    on = _stage(dx, 1, 1)
    assert len(ISFFLX0_ZEROED) == 13
    for name in ISFFLX0_ZEROED:
        np.testing.assert_array_equal(
            off[name], np.zeros_like(off[name]), err_msg=f"oracle {name}"
        )
        assert np.any(on[name] != 0.0), name
    for name in OUTPUT_NAMES:
        if name not in ISFFLX0_ZEROED:
            np.testing.assert_array_equal(off[name], on[name], err_msg=name)

    inputs = _inputs(off)
    kwargs = dict(dx=dx, itimestep=1,
                  mol=off["mol_input"], ustm=off["ustm_input"])
    diagnosed = mynn_surface_layer_default(inputs, isfflx=1, **kwargs)
    prescribed = mynn_surface_layer_default(inputs, isfflx=0, **kwargs)
    for name in OUTPUT_NAMES:
        if name in ISFFLX0_ZEROED:
            np.testing.assert_array_equal(
                prescribed[name], np.zeros_like(off[name]),
                err_msg=f"reference {name}",
            )
            continue
        np.testing.assert_array_equal(
            prescribed[name], diagnosed[name], err_msg=f"reference {name}"
        )


@pytest.mark.parametrize("dx", DX_SWEEP)
@pytest.mark.parametrize(("itimestep", "isfflx"), STAGES)
def test_cpu_reference_matches_the_coarse_wrf_oracle(dx, itimestep, isfflx):
    fields = _stage(dx, itimestep, isfflx)
    actual = mynn_surface_layer_default(
        _inputs(fields), dx=dx, itimestep=itimestep, isfflx=isfflx,
        mol=fields["mol_input"], ustm=fields["ustm_input"],
    )
    table = _table(CPU_ULP, dx, itimestep)
    zeroed = ISFFLX0_ZEROED if isfflx == 0 else ()
    np.testing.assert_array_equal(actual["regime"], fields["regime"])
    for name in OUTPUT_NAMES:
        if name == "regime":
            continue
        _assert_ulp(
            actual[name], fields[name], name, _budget(table, name, zeroed)
        )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("dx", DX_SWEEP)
@pytest.mark.parametrize(("itimestep", "isfflx"), STAGES)
def test_cuda_kernel_matches_the_coarse_wrf_oracle(dx, itimestep, isfflx):
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
    )

    fields = _stage(dx, itimestep, isfflx)
    inputs = {
        name: cp.asarray(
            fields[INPUT_ALIASES.get(name, name)].copy().reshape(2, 5)
        )
        for name in INPUT_NAMES
    }
    actual = mynn_surface_layer(
        inputs, dx=dx, itimestep=itimestep, isfflx=isfflx,
        mol=cp.asarray(fields["mol_input"].copy().reshape(2, 5)),
        ustm=cp.asarray(fields["ustm_input"].copy().reshape(2, 5)),
    )
    cp.cuda.get_current_stream().synchronize()
    # The compiler pair is read here, not in a helper: a module-level
    # function importing cupy would mark every CPU test in this file gpu.
    toolchain = (cp.cuda.Device().compute_capability,
                 tuple(cp.cuda.nvrtc.getVersion()))
    table = _table(toolchain_row(GPU_ULP_BY_TOOLCHAIN, GPU_ULP,
                                 "GPU_ULP_BY_TOOLCHAIN", toolchain), dx,
                   itimestep)
    zeroed = ISFFLX0_ZEROED if isfflx == 0 else ()
    np.testing.assert_array_equal(
        cp.asnumpy(actual.regime).ravel(), fields["regime"]
    )
    for name in MYNN_SURFACE_OUTPUTS:
        if name == "regime":
            continue
        _assert_ulp(
            cp.asnumpy(getattr(actual, name)), fields[name], name,
            _budget(table, name, zeroed),
        )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(("itimestep", "isfflx"), STAGES)
def test_the_cuda_kernel_also_clamps_vsgd_at_the_threshold(itimestep, isfflx):
    """The kernel half of what licenses ``DX_TABLE_ALIAS``."""

    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
    )

    fields = _stage(3000.0, itimestep, isfflx)
    results = []
    for dx in (3000.0, 5000.0):
        inputs = {
            name: cp.asarray(
                fields[INPUT_ALIASES.get(name, name)].copy().reshape(2, 5)
            )
            for name in INPUT_NAMES
        }
        results.append(mynn_surface_layer(
            inputs, dx=dx, itimestep=itimestep, isfflx=isfflx,
            mol=cp.asarray(fields["mol_input"].copy().reshape(2, 5)),
            ustm=cp.asarray(fields["ustm_input"].copy().reshape(2, 5)),
        ))
    cp.cuda.get_current_stream().synchronize()
    for name in MYNN_SURFACE_OUTPUTS:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(results[1], name)),
            cp.asnumpy(getattr(results[0], name)), err_msg=name,
        )


def test_every_table_row_is_the_right_width_and_carries_a_measurement():
    """A row of the wrong width, or of zeros, is a table that lost meaning.

    ``_budget`` asserts the width when it is used, but only for outputs a test
    actually compares; this covers every row in the file, and rejects an
    all-zero row -- which would mean the output is bitwise and the row should
    have been deleted rather than left as decoration.
    """

    expected = {
        (dx, itimestep)
        for dx in DX_SWEEP if dx not in DX_TABLE_ALIAS
        for itimestep in (1, 2)
    }
    for tables in (CPU_ULP, GPU_ULP,
                   *GPU_ULP_BY_TOOLCHAIN.values()):
        assert set(tables) == expected
        for key, table in tables.items():
            assert table, key
            for name, row in table.items():
                assert name in OUTPUT_NAMES, name
                assert len(row) == len(CASES), (key, name)
                assert all(isinstance(v, int) and v >= 0 for v in row), name
                assert any(row), f"{key} {name} is all zero; delete the row"


def test_the_coarse_oracle_validator_gates_on_these_same_measurements():
    """One measurement, two consumers.

    ``build_coarse.sh`` runs
    ``tools/mynn_wrf461_oracle/validate_coarse_oracle.py`` against the CSV it
    just produced, and that script is standalone: it cannot import a test
    module, so it carries its own copy of the CPU tables.  Two copies of a
    measurement drift; this pins them equal instead of hoping.
    """

    spec = importlib.util.spec_from_file_location(
        "_mynn_coarse_oracle_validator", COARSE_VALIDATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CPU_ULP == CPU_ULP
    assert module.DX_TABLE_ALIAS == DX_TABLE_ALIAS
    assert module.ISFFLX0_ZEROED == ISFFLX0_ZEROED
    assert module.DX_INSENSITIVE == DX_INSENSITIVE
    assert module.EXPECTED_DX == DX_SWEEP
    assert module.EXPECTED_STAGES == STAGES
    assert module.EXPECTED_CASES == CASES


def test_the_coarse_branch_is_actually_reached_by_both_ports():
    """A guard against the gate above passing because VSGD stayed zero."""

    fields = _stage(12000.0, 1)
    inputs = _inputs(fields)
    near = mynn_surface_layer_default(
        inputs, dx=3000.0, itimestep=1, isfflx=1,
        mol=fields["mol_input"], ustm=fields["ustm_input"],
    )
    far = mynn_surface_layer_default(
        inputs, dx=12000.0, itimestep=1, isfflx=1,
        mol=fields["mol_input"], ustm=fields["ustm_input"],
    )
    assert not np.array_equal(near["wspd"], far["wspd"])
    # ...and the coarse answer is the one WRF wrote, not the near one.
    assert _max_ulp(far["wspd"], fields["wspd"]) == 0
    assert _max_ulp(near["wspd"], fields["wspd"]) > 0
