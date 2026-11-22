"""Noah-MP snow compaction option 2 (``opt_snow_compact=2``).

Re-implemented from published descriptions, not bitwise to upstream:
Abolafia-Rosenzweig, He, Chen & Barlage (2024), JAMES 16, e2023MS003869,
doi:10.1029/2023MS003869 (CC BY), Section 2.4, Eq. 11-12 and the 0.0315
upper constraint.  No NCAR/noahmp or other UCAR-licensed file was read.

What this file holds:

* the C_bd formula against hand-computed values, including the two
  statements the paper itself makes about it (C_bd equals WRF's 0.021 near
  266.3 K, and the 0.0315 cap binds below about 251 K);
* one COMPACT layer against the Anderson (1976) overburden equation the
  paper prints as Eq. 5, worked by hand for both options;
* option 1 is the unchanged default: the reference ``compact`` with no C_bd
  and with WRF's own 0.021 give the same bytes;
* the CUDA option copies of COMPACT/SNOWWATER differ from the imported
  default code by exactly the one documented edit each;
* on the GPU, the option kernel against the float32 reference on the 36
  WATER fixture columns, bit for bit, and option 2 really changes layered
  snow while leaving snow-free columns alone.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import noahmp_snow
from woof.core.noahmp_snow import SnowColumn, compact, compaction_cbd

KERNELS = Path(noahmp_snow.__file__).resolve().parent / "kernels"


# ---------------------------------------------------------------------------
# Eq. 11-12 and the cap, by hand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tair, psfc, want", [
    # Eq. 11 interior: -0.00069503 * 260 + 0.20606699 = 0.02535919
    (260.0, 95000.0, 0.02535919),
    (260.0, 70000.0, 0.02535919),
    # warm snow: Eq. 11 gives 0.01493374 at 275 K, so each Eq. 12 floor binds
    (275.0, 70000.0, 0.019),      # psfc < 80 kPa
    (275.0, 80000.0, 0.018),      # 80 kPa is in the middle bin
    (275.0, 84999.0, 0.018),
    (275.0, 85000.0, 0.017),      # 85 kPa is in the high-pressure bin
    (275.0, 101325.0, 0.017),
    # cold snow: Eq. 11 gives 0.03161446 at 251 K, above the 0.0315 cap
    (251.0, 95000.0, 0.0315),
    (240.0, 70000.0, 0.0315),
])
def test_cbd_against_hand_values(tair, psfc, want):
    got = compaction_cbd(tair, psfc)
    assert got.dtype == np.float32
    assert math.isclose(float(got), want, rel_tol=2e-6), (tair, psfc, got)


def test_cbd_meets_wrf_default_where_the_paper_says():
    # Section 4.2: the enhanced C_bd "equals the default value at ~266.3 K".
    t_equal = (0.20606699 - 0.021) / 0.00069503          # 266.27 K
    assert 266.2 < t_equal < 266.4
    assert math.isclose(float(compaction_cbd(t_equal, 95000.0)), 0.021,
                        rel_tol=1e-4)
    # Colder than that compacts less than WRF, warmer compacts more.
    assert compaction_cbd(262.0, 95000.0) > np.float32(0.021)
    assert compaction_cbd(270.0, 95000.0) < np.float32(0.021)


def test_floors_follow_figure_2_corners():
    # Figure 2: each pressure line leaves Eq. 11 where it meets its floor.
    for floor, psfc in ((0.019, 70000.0), (0.018, 82000.0), (0.017, 95000.0)):
        corner = (0.20606699 - floor) / 0.00069503
        assert float(compaction_cbd(corner - 0.5, psfc)) > floor
        assert math.isclose(float(compaction_cbd(corner + 0.5, psfc)), floor,
                            rel_tol=1e-6)


# ---------------------------------------------------------------------------
# one COMPACT layer worked by hand (Eq. 1-5, 7 of the paper)
# ---------------------------------------------------------------------------
def _one_layer():
    # isnow=-1: only the bottom snow layer (Fortran j=0) is live.
    col = SnowColumn(isnow=-1)
    col.snice[2] = 100.0          # kg/m2
    col.snliq[2] = 0.0
    col.dzsnso[2] = 0.5           # m, so bi = 200 kg/m3
    col.stc[2] = 273.16           # TFRZ, so td = 0
    return col


def _hand_dz(cbd, dt=3600.0):
    bi, wx, td = 200.0, 100.0, 0.0
    ddz1 = -2.5e-6 * math.exp(-0.04 * td)                 # Eq. 2
    ddz1 *= math.exp(-0.046 * (bi - 100.0))               # Eq. 3
    ddz2 = -(0.0 + 0.5 * wx) * math.exp(-0.08 * td - cbd * bi) / 1.33e6  # Eq. 5
    pdz = max(-0.5, (ddz1 + ddz2) * dt)                    # Eq. 1
    return 0.5 * (1.0 + pdz)                               # Eq. 7


@pytest.mark.parametrize("cbd", [None, 0.017, 0.019, 0.0315])
def test_one_layer_compaction_by_hand(cbd):
    col = _one_layer()
    compact(col, 3600.0, np.zeros(3, np.int32), np.zeros(3, np.float32), cbd)
    want = _hand_dz(0.021 if cbd is None else cbd)
    assert math.isclose(float(col.dzsnso[2]), want, rel_tol=1e-6)


def test_warm_option_compacts_more_than_wrf():
    # 275 K at 95 kPa: C_bd floors at 0.017, so the overburden rate is
    # exp((0.021 - 0.017) * 200) = 2.23x WRF's and the layer ends thinner.
    wrf, warm = _one_layer(), _one_layer()
    zeros = (np.zeros(3, np.int32), np.zeros(3, np.float32))
    compact(wrf, 3600.0, *zeros)
    compact(warm, 3600.0, *zeros, compaction_cbd(275.0, 95000.0))
    assert float(warm.dzsnso[2]) < float(wrf.dzsnso[2])
    assert math.isclose(float(wrf.dzsnso[2]), _hand_dz(0.021), rel_tol=1e-6)
    assert math.isclose(float(warm.dzsnso[2]), _hand_dz(0.017), rel_tol=1e-6)
    # The hand numbers: 0.49894 m (WRF) and 0.49770 m (option 2).
    assert abs(_hand_dz(0.021) - 0.49894) < 1e-5
    assert abs(_hand_dz(0.017) - 0.49770) < 1e-5


def test_option_one_is_wrfs_bytes():
    a, b = _one_layer(), _one_layer()
    zeros = (np.zeros(3, np.int32), np.zeros(3, np.float32))
    compact(a, 3600.0, *zeros)
    compact(b, 3600.0, *zeros, np.float32(0.021))
    assert a.dzsnso.tobytes() == b.dzsnso.tobytes()


def test_option_two_needs_pressure():
    with pytest.raises(ValueError, match="surface pressure"):
        noahmp_snow.snowwater(_one_layer(), 3600.0, np.zeros(4, np.float32),
                              np.zeros(3, np.int32), np.zeros(3, np.float32),
                              270.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              compaction=2)


# ---------------------------------------------------------------------------
# the CUDA option copies are the default code plus one edit each
# ---------------------------------------------------------------------------
def _block(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def test_cuda_option_copies_are_one_edit_from_default():
    water = (KERNELS / "noahmp_water.cu").read_text(encoding="ascii")
    options = (KERNELS / "noahmp_water_options.cuh").read_text(encoding="ascii")
    head = ("__device__ void compact(Col *c, float dt, const int *imelt, "
            "const float *ficeold)\n")
    base_compact = _block(water, head, "\n// ==========")
    base_sw = _block(water, "__device__ void snowwater(Col *c, float dt, const float *zsoil,",
                     "\n// <<< END imported section: noahmp_snow.cu")
    want_compact = base_compact.replace(
        head, "__device__ void compact_cbd(Col *c, float dt, const int *imelt, "
              "const float *ficeold,\n                            float cbd)\n"
    ).replace("SN_K_C2", "cbd")
    want_sw = base_sw.replace(
        "__device__ void snowwater(", "__device__ void snowwater_cbd(").replace(
        "                          float *ponding1, float *ponding2)\n{",
        "                          float *ponding1, float *ponding2, float cbd)\n{", 1
    ).replace("compact(c, dt, imelt, ficeold)", "compact_cbd(c, dt, imelt, ficeold, cbd)")
    assert want_compact in options
    assert want_sw in options
    # The default unit is untouched: C2 is still WRF's 0.021 constant there.
    assert base_compact.count("SN_K_C2") == 1


def test_cuda_constants_are_the_paper_values():
    options = (KERNELS / "noahmp_water_options.cuh").read_text(encoding="ascii")
    words = re.findall(r"0x([0-9A-F]{8})u", _block(options, "C_AR24_F32[8]", "};"))
    got = np.array([int(w, 16) for w in words], dtype=np.uint32).view(np.float32)
    want = np.array([-0.00069503, 0.20606699, 80000.0, 85000.0,
                     0.019, 0.018, 0.017, 0.0315], dtype=np.float32)
    assert got.tobytes() == want.tobytes()


# ---------------------------------------------------------------------------
# GPU: the option kernel against the float32 reference, bit for bit
# ---------------------------------------------------------------------------
PSFC_BINS = (70000.0, 82000.0, 95000.0)


@requires_gpu
@pytest.mark.gpu
def test_option_kernel_matches_reference_on_water_fixture(monkeypatch):
    import cupy as cp

    import woof.core.noahmp_water as water_mod
    from test_noahmp_water import CASES, _call
    from test_noahmp_water_slab import _host_outputs, _same_bits, _slab_fields
    from woof.core.noahmp_water_slab import evaluate_water_slab

    psfc = np.asarray([PSFC_BINS[k % 3] for k in range(len(CASES))], np.float32)
    current = {}
    real = water_mod.snowwater

    def snowwater_opt2(*args, **kwargs):
        return real(*args, compaction=2, psfc=current["p"], **kwargs)

    monkeypatch.setattr(water_mod, "snowwater", snowwater_opt2)
    calls, fluxes = [], []
    for k, case in enumerate(CASES):
        args = _call(case)
        current["p"] = psfc[k]
        fluxes.append(water_mod.water(*args))
        calls.append((args, {}))
    want = _host_outputs(calls, fluxes)

    fields = _slab_fields(cp)
    fields["psfc"] = cp.asarray(psfc)
    got = evaluate_water_slab(fields, len(CASES), opt_snow_compact=2)
    for name in sorted(want):
        _same_bits(cp.asnumpy(got[name]), want[name], name)

    # Option 2 changes layered snow and nothing else.
    base = evaluate_water_slab(_slab_fields(cp), len(CASES))
    d_opt = cp.asnumpy(got["dzsnso"])
    d_def = cp.asnumpy(base["dzsnso"])
    changed = np.any(d_opt != d_def, axis=1)
    assert changed.any(), "option 2 changed no fixture column"
    # A column with no layered snow on entry or exit never reaches COMPACT.
    entry_isnow = np.asarray([_call(c)[1].isnow for c in CASES])
    exit_isnow = cp.asnumpy(base["isnow"])
    assert not changed[(entry_isnow == 0) & (exit_isnow == 0)].any()
