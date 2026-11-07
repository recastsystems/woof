"""GPU integration gate for the coupled MYNN surface-driver seam.

These tests exercise the exact entry points ``PhysicsDriver._run_sfclay``
uses -- ``seed_mynn_surface_first_step`` followed by
``launch_mynn_surface_layer`` into a ``MynnSurfaceResult`` that aliases the
driver's own persistent field arrays -- and pin them against the unmodified
WRF v4.6.1 oracles rather than against another CUDA call.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from _toolchain_rows import toolchain_row

from woof.core.fp32_ulp import fp32_ulp_distance


ORACLE_DIR = Path(__file__).parents[1] / "tests" / "data" / "oracles" / "mynn"
WIDE_ORACLE = ORACLE_DIR / "surface-layer-wide.csv"
WRAPPER_ORACLE = ORACLE_DIR / "surface-layer-wrapper.csv"

CASES = (
    "strong_stable_land", "clipped_stable_land", "damped_stable_land",
    "neutral_land", "free_convective_land", "land_qsfc_unset",
    "thin_land_level2_wind", "thin_land_log10_wind", "midres_water",
    "coarse_water",
)

# The FP32 ULP distance of the aliased driver launch from the unmodified WRF
# v4.6.1 oracle, measured per (launch, output, column) -- one integer per
# compared number, in ``CASES`` order.  An output a launch does not name is
# bitwise on every column there and is required to stay bitwise: the lookup
# returns zeros.  There is no margin to justify, because there is no margin:
# each entry is the residue measured at that element and the gate is
# ``residue <= entry`` column by column.
#
# The two chained steps get their own tables rather than a shared one, because
# the compounding is the whole point of this file: step 2 runs off step 1's own
# device output instead of re-reading the oracle's inputs, so the carried
# UST/MOL error grows -- FLHC/FLQC/QFX 65/66/66 at step 2 against 44/45/45 at
# step 1, UST/USTM 31 against 21, all of it in ``thin_land_log10_wind``.
#
# That column is the whole story here as it is in ``test_mynn_surface_gpu.py``:
# CUDA's ``powf`` puts the level-1 Exner factor of ``thin_land_log10_wind``
# (P1 = 99850 Pa) one ULP off gfortran, the DTHVDZ cancellation at
# module_sf_mynn.F:559-560 amplifies it ~58x into BR, and the zolri/zolrib
# solve carries it into ZOL/RMOL/PSIM/PSIH and every coefficient below.  The
# other nine columns are 0-5 ULP.  The previous revision budgeted per launch
# only, so all ten drew the 118: 36604 ULP of unearned margin, 19 of 107
# (launch, output) comparisons unable to fail on a 1-ULP regression, and
# single elements as much as 118 loose.
#
# Measured with Windows NumPy 2.2.6 / CPython 3.13.7 and cupy on the RTX 5090.
#
# These are ratchets: lower them as the FP32 shims are unified, never raise.
CHAINED_ULP = {
    1: {
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
    2: {
        "zol": (0, 0, 2, 0, 1, 2, 0, 87, 1, 0),
        "rmol": (0, 0, 2, 0, 1, 3, 0, 116, 0, 0),
        "ust": (0, 0, 2, 2, 0, 0, 0, 31, 0, 0),
        "ustm": (0, 0, 2, 2, 0, 0, 0, 31, 0, 0),
        "mol": (0, 0, 2, 0, 0, 0, 0, 37, 0, 0),
        "psim": (0, 0, 2, 0, 2, 4, 0, 117, 2, 2),
        "psih": (0, 0, 2, 0, 0, 1, 0, 69, 1, 0),
        "chs": (0, 0, 2, 1, 0, 0, 0, 56, 0, 0),
        "chs2": (0, 0, 2, 1, 0, 0, 0, 51, 0, 0),
        "cqs2": (0, 0, 1, 1, 0, 0, 0, 51, 0, 0),
        "ch": (0, 0, 4, 0, 0, 0, 0, 56, 0, 0),
        "flhc": (0, 0, 4, 0, 0, 0, 0, 65, 0, 0),
        "flqc": (0, 0, 2, 2, 0, 0, 0, 66, 0, 0),
        "qgh": (1, 1, 1, 1, 1, 1, 0, 1, 2, 1),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
        "hfx": (0, 0, 5, 0, 0, 0, 0, 13, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 1, 0, 66, 4, 1),
        "lh": (0, 0, 0, 0, 0, 1, 0, 40, 5, 1),
        "u10": (0, 0, 2, 2, 1, 0, 0, 0, 2, 0),
        "v10": (0, 0, 1, 0, 1, 0, 0, 0, 2, 0),
        "th2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "t2": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "gz1oz0": (1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
        "wspd": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        "br": (0, 0, 0, 0, 0, 0, 0, 92, 0, 0),
        "ck": (0, 0, 0, 0, 1, 2, 1, 64, 0, 0),
        "cka": (0, 0, 0, 1, 1, 0, 0, 71, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 3, 72, 0, 0),
        "cda": (0, 0, 2, 2, 1, 0, 0, 39, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 2),
    },
}

SEEDED_ULP = {
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
}

#: ``SEEDED_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 11 rows read more than ``SEEDED_ULP``;
#: the reciprocal-multiply compiler's psim row reads at or below it now.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
SEEDED_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **SEEDED_ULP,
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
}
#: A167: NVRTC 12.9.86, the compiler of the default recast-woof[gpu] extra
#: (cupy-cuda12x), reads the sm_120 row A146 re-recorded under 13.4: every
#: reading this file's tests take, and the device result behind each, is
#: bit-identical under the two compilers.  Before this row 12.9.86 failed
#: here by name (tests/_toolchain_rows.py).  MEASURED 2026-10-01 on test host 4's
#: RTX 5070 Ti and test host 2's RTX 5090, two processes per compiler, at
#: integrate/2.8 9dbb4a2db.
SEEDED_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    SEEDED_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89 reads the sm_120 row.  Since the cross-architecture rounding pins
#: (xnode-identity, ecbd106a9 and the A146 __fdiv_rn spellings) the RTX 4090
#: returns the RTX 5090's words for every output these gates compare:
#: 16,518 words, zero differing, NVRTC 13.4, 2026-10-05.  The default row,
#: recorded before those pins, no longer describes it: it failed 31 gates
#: on the RTX 4090 with no kernel change.  MEASURED on test host 1's RTX 4090
#: under NVRTC 12.9, 13.3 and 13.4: every gate passes on the sm_120 row.
SEEDED_ULP_BY_TOOLCHAIN[("89", (12, 9))] = SEEDED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
SEEDED_ULP_BY_TOOLCHAIN[("89", (13, 3))] = SEEDED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
SEEDED_ULP_BY_TOOLCHAIN[("89", (13, 4))] = SEEDED_ULP_BY_TOOLCHAIN[("120", (13, 4))]


#: ``CHAINED_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 10 rows read more than ``CHAINED_ULP``.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
CHAINED_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **CHAINED_ULP,
        1: {
            **CHAINED_ULP[1],
            "lh": (1, 0, 0, 0, 0, 0, 4, 3, 0, 2),
            "qfx": (1, 0, 0, 0, 0, 0, 3, 2, 0, 4),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
        2: {
            **CHAINED_ULP[2],
            "lh": (0, 0, 0, 0, 0, 0, 0, 1, 0, 2),
            "qfx": (0, 0, 0, 0, 0, 0, 0, 1, 0, 2),
            "qgh": (0, 0, 0, 0, 2, 0, 0, 0, 0, 2),
            "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
            "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
        },
    },
}
#: A167: as for SEEDED_ULP_BY_TOOLCHAIN above.
CHAINED_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    CHAINED_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89: as for SEEDED_ULP_BY_TOOLCHAIN above.
CHAINED_ULP_BY_TOOLCHAIN[("89", (12, 9))] = CHAINED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
CHAINED_ULP_BY_TOOLCHAIN[("89", (13, 3))] = CHAINED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
CHAINED_ULP_BY_TOOLCHAIN[("89", (13, 4))] = CHAINED_ULP_BY_TOOLCHAIN[("120", (13, 4))]


CARRIED_ULP = {
    "ust": (1, 0, 1, 1, 0, 1, 1, 21, 0, 0),
    "ustm": (1, 0, 1, 1, 0, 1, 1, 21, 0, 0),
    "mol": (0, 0, 0, 0, 0, 0, 0, 36, 0, 0),
    "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 2),
}

#: ``CARRIED_ULP`` per compiler where a compiler reads it differently, keyed
#: on (compute capability, NVRTC major.minor), the pair measured.
#: A146 re-recorded this table: NVRTC had compiled every float division
#: by a compile-time constant as a multiply by the rounded reciprocal on
#: Blackwell, and the kernels now spell those divisions ``__fdiv_rn``, the
#: IEEE quotient.  On sm_120 1 row reads more than ``CARRIED_ULP``.
#: MEASURED 2026-09-30 on the RTX 5070 Ti (sm_120, NVRTC 13.4.92) at the
#: A146 review repair, every output of every stage compared.
CARRIED_ULP_BY_TOOLCHAIN = {
    ("120", (13, 4)): {
        **CARRIED_ULP,
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 0, 3),
    },
}
#: A167: as for SEEDED_ULP_BY_TOOLCHAIN above.
CARRIED_ULP_BY_TOOLCHAIN[("120", (12, 9))] = (
    CARRIED_ULP_BY_TOOLCHAIN[("120", (13, 4))])
#: sm_89: as for SEEDED_ULP_BY_TOOLCHAIN above.
CARRIED_ULP_BY_TOOLCHAIN[("89", (12, 9))] = CARRIED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
CARRIED_ULP_BY_TOOLCHAIN[("89", (13, 3))] = CARRIED_ULP_BY_TOOLCHAIN[("120", (13, 4))]
CARRIED_ULP_BY_TOOLCHAIN[("89", (13, 4))] = CARRIED_ULP_BY_TOOLCHAIN[("120", (13, 4))]


INPUT_ALIASES = {
    "hfx": "hfx_input", "qfx": "qfx_input", "znt": "znt_input",
    "qsfc": "qsfc_input", "ust": "ust_input",
}
INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)


def _stage(path: Path, itimestep: int, isfflx: int | None = None):
    with path.open(newline="", encoding="ascii") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["itimestep"]) == itimestep
            and (isfflx is None or int(row["isfflx"]) == isfflx)
        ]
    assert tuple(row["case"] for row in rows) == CASES
    skip = ("case", "itimestep", "isfflx")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(2, 5)
        for key in rows[0] if key not in skip
    }


def _driver_fields(fields):
    """Allocate the driver's inout aliasing: one array per named field."""

    import cupy as cp

    from woof.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS

    arrays = {}
    for name in INPUT_NAMES:
        arrays[name] = cp.ascontiguousarray(
            cp.asarray(fields[INPUT_ALIASES.get(name, name)].copy())
        )
    for name in MYNN_SURFACE_OUTPUTS:
        if name not in arrays:
            arrays[name] = cp.zeros((2, 5), dtype=arrays["u1"].dtype)
    arrays["mol"] = cp.ascontiguousarray(
        cp.asarray(fields["mol_input"].copy())
    )
    arrays["ustm"] = cp.ascontiguousarray(
        cp.asarray(fields["ustm_input"].copy())
    )
    return arrays


def _launch(arrays, *, itimestep, isfflx):
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        MynnSurfaceResult,
        launch_mynn_surface_layer,
    )

    result = MynnSurfaceResult(
        **{name: arrays[name] for name in MYNN_SURFACE_OUTPUTS}
    )
    launch_mynn_surface_layer(
        {name: arrays[name] for name in INPUT_NAMES},
        arrays["mol"], arrays["ustm"], result,
        dx=3000.0, itimestep=itimestep, isfflx=isfflx,
    )
    cp.cuda.get_current_stream().synchronize()
    return result


def _budget(table, name):
    """The measured per-column residue for ``name``, or zeros."""

    row = table.get(name, ())
    if not row:
        return np.zeros(len(CASES), dtype=np.int64)
    assert len(row) == len(CASES), name
    return np.asarray(row, dtype=np.int64)


def _assert_ulp(actual, expected, name, budget):
    """Fail when any column of ``name`` passes its measured ULP residue.

    ``budget`` is per column, so the message names the case that drifted
    instead of only the output: a field-wide number cannot say which of the
    ten columns moved, and that is the information a regression needs.
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


def _assert_oracle(arrays, fields, names, table):
    import cupy as cp

    np.testing.assert_array_equal(
        cp.asnumpy(arrays["regime"]), fields["regime"]
    )
    for name in names:
        if name == "regime":
            continue
        _assert_ulp(
            cp.asnumpy(arrays[name]), fields[name], name,
            _budget(table, name),
        )


@pytest.mark.gpu
@requires_gpu
def test_driver_launch_path_matches_wrf_across_two_timesteps():
    """The aliased inout launch must reproduce WRF at step 1 and step 2."""

    import cupy as cp

    from woof.core.mynn_sfclay import MYNN_SURFACE_OUTPUTS

    step1 = _stage(WIDE_ORACLE, 1, 1)
    step2 = _stage(WIDE_ORACLE, 2, 1)
    arrays = _driver_fields(step1)

    _launch(arrays, itimestep=1, isfflx=1)
    chained = toolchain_row(CHAINED_ULP_BY_TOOLCHAIN, CHAINED_ULP,
                            "CHAINED_ULP_BY_TOOLCHAIN")
    _assert_oracle(arrays, step1, MYNN_SURFACE_OUTPUTS, chained[1])

    # WRF carries ZNT/UST/MOL/QSFC forward; the driver holds them in the same
    # device arrays, so step 2 runs straight off the step-1 results.  ZNT is
    # bitwise on every column, so CARRIED_ULP does not name it.
    for name, key in (("znt", "znt_input"), ("ust", "ust_input"),
                      ("qsfc", "qsfc_input"), ("mol", "mol_input"),
                      ("ustm", "ustm_input")):
        _assert_ulp(
            cp.asnumpy(arrays[name]), step2[key], f"carried {name}",
            _budget(toolchain_row(CARRIED_ULP_BY_TOOLCHAIN, CARRIED_ULP,
                                  "CARRIED_ULP_BY_TOOLCHAIN"),
                    name),
        )
    # Only the externally supplied forcing is refreshed between steps.
    for name in ("hfx", "qfx"):
        arrays[name][...] = cp.asarray(step2[f"{name}_input"].copy())

    _launch(arrays, itimestep=2, isfflx=1)
    _assert_oracle(arrays, step2, MYNN_SURFACE_OUTPUTS, chained[2])

    water = step1["xland"] > 1.5
    assert np.all(
        cp.asnumpy(arrays["znt"])[water] != step1["znt_input"][water]
    )


@pytest.mark.gpu
@requires_gpu
def test_driver_first_step_seeding_matches_the_wrf_wrapper():
    """module_sf_mynn.F:329-337 must run before the first column solve."""

    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        seed_mynn_surface_first_step,
    )

    fields = _stage(WRAPPER_ORACLE, 1)
    arrays = _driver_fields(fields)
    unseeded = cp.asnumpy(arrays["ust"]).copy()

    seed_mynn_surface_first_step(
        arrays["u1"], arrays["v1"], arrays["qv1"],
        ust=arrays["ust"], mol=arrays["mol"], qsfc=arrays["qsfc"],
        qstar=arrays["qstar"],
    )
    seeded = cp.asnumpy(arrays["ust"])
    assert not np.allclose(seeded, unseeded), "seeding must not be a no-op"
    # module_sf_mynn.F:330-337 is straight-line FP32 arithmetic with no
    # transcendentals, so the seeding is bitwise on both sides and is pinned
    # exact.  A budget here would only hide a transcription error.
    _assert_ulp(
        seeded,
        np.maximum(0.04 * np.hypot(fields["u1"], fields["v1"]), 0.001)
        .astype(np.float32),
        "seeded ust", 0,
    )
    _assert_ulp(
        cp.asnumpy(arrays["qsfc"]),
        (fields["qv1"] / (1.0 + fields["qv1"])).astype(np.float32),
        "seeded qsfc", 0,
    )
    assert bool((arrays["mol"] == 0.0).all())
    assert bool((arrays["qstar"] == 0.0).all())

    _launch(arrays, itimestep=1, isfflx=1)
    # SFCLAY_mynn keeps wstar/qstar as wrapper locals and never returns them.
    _assert_oracle(
        arrays, fields,
        tuple(n for n in MYNN_SURFACE_OUTPUTS if n not in ("wstar", "qstar")),
        toolchain_row(SEEDED_ULP_BY_TOOLCHAIN, SEEDED_ULP,
                      "SEEDED_ULP_BY_TOOLCHAIN"),
    )


def test_every_table_row_is_the_right_width_and_carries_a_measurement():
    """A row of the wrong width, or of zeros, is a table that lost meaning.

    ``_budget`` asserts the width when it is used, but only for outputs a test
    actually compares; this covers every row in the file, and rejects an
    all-zero row -- which would mean the output is bitwise and the row should
    have been deleted rather than left as decoration.  It needs no GPU.
    """

    tables = [CHAINED_ULP[1], CHAINED_ULP[2], SEEDED_ULP, CARRIED_ULP]
    tables += list(SEEDED_ULP_BY_TOOLCHAIN.values())
    tables += list(CARRIED_ULP_BY_TOOLCHAIN.values())
    for rows in CHAINED_ULP_BY_TOOLCHAIN.values():
        assert set(rows) == set(CHAINED_ULP)
        tables += [rows[1], rows[2]]
    for table in tables:
        assert table
        for name, row in table.items():
            assert len(row) == len(CASES), name
            assert all(isinstance(v, int) and v >= 0 for v in row), name
            assert any(row), f"{name} is all zero; delete the row"


@pytest.mark.gpu
@requires_gpu
def test_physics_driver_routes_exact_fields_to_mynn_surface_kernel():
    import cupy as cp

    from woof.core.mynn_sfclay import (
        MYNN_SURFACE_OUTPUTS,
        mynn_surface_layer,
        seed_mynn_surface_first_step,
    )
    from woof.core.physics import _prepare_atmosphere
    from test_physics_driver import _full_state

    state, cfg, driver = _full_state(
        sf_sfclay_physics=5,
        sf_surface_physics=0,
        bl_pbl_physics=0,
    )
    atmosphere = _prepare_atmosphere(state)
    fields = driver.fields
    seeded_ust = fields["ust"].copy()
    seeded_mol = fields["mol"].copy()
    seeded_qsfc = fields["qsfc"].copy()
    seeded_qstar = fields["qstar"].copy()
    seed_mynn_surface_first_step(
        atmosphere["u"][0], atmosphere["v"][0], atmosphere["qv"][0],
        ust=seeded_ust, mol=seeded_mol, qsfc=seeded_qsfc,
        qstar=seeded_qstar,
    )
    # The cold-start driver state is UST=1e-4 / QSFC=0, so the seeding is
    # observable at this seam rather than a no-op.
    assert not bool(cp.allclose(seeded_ust, fields["ust"]))
    assert not bool(cp.allclose(seeded_qsfc, fields["qsfc"]))

    inputs = {
        "u1": atmosphere["u"][0],
        "v1": atmosphere["v"][0],
        "t1": atmosphere["temperature"][0],
        "qv1": atmosphere["qv"][0],
        "p1": atmosphere["pressure"][0],
        "rho1": atmosphere["rho"][0],
        "dz1": atmosphere["dz"][0],
        "u2": atmosphere["u"][1],
        "v2": atmosphere["v"][1],
        "dz2": atmosphere["dz"][1],
        "psfc": atmosphere["p_interface"][0],
        "tsk": fields["tsk"],
        "pblh": fields["pblh"],
        "mavail": fields["mavail"],
        "hfx": fields["hfx"].copy(),
        "qfx": fields["qfx"].copy(),
        "znt": fields["znt"].copy(),
        "qsfc": seeded_qsfc.copy(),
        "ust": seeded_ust.copy(),
        "xland": fields["xland"],
        "snowh": fields["snowh"],
    }
    expected = mynn_surface_layer(
        inputs,
        dx=cfg.dx,
        itimestep=1,
        mol=seeded_mol.copy(),
        ustm=fields["ustm"].copy(),
    )

    driver.compute(state, cfg)

    assert driver.call_counts["sfclay"] == 1
    for name in MYNN_SURFACE_OUTPUTS:
        actual = fields[name]
        reference = getattr(expected, name)
        assert bool(cp.isfinite(actual).all()), name
        cp.testing.assert_array_equal(actual, reference, err_msg=name)
    # ZNT is INTENT(INOUT) in WRF: the driver must hold the updated value.
    water = fields["xland"] > 1.5
    if bool(water.any()):
        assert bool(
            (fields["znt"][water] != inputs["znt"][water]).all()
        )
