"""GFS first-guess hydrometeors and the Thompson start-frame reflectivity (hex-meshes-sources review, 2026-10-29).

THE BREAKAGE: the default hex route (convection_permitting) started from GFS with no condensate (the gfs row's
Vtable decoded none of GFS's CLMR/ICMR/RWMR/SNMR/GRLE), and its f000 composite reflectivity read "No values
to plot" even when the init held rain (the start frame is written before any microphysics).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex import init_sources as src
from woof.hex.init_door import FIRST_GUESS_SPECIES
from woof.hex.mpasphys import thompson_refl_start_v842 as start
from woof.hex.output import HistoryField

GFS_IDX = "\n".join((
    "1:0:d=2023033112:PRMSL:mean sea level:anl:",
    "2:100:d=2023033112:CLMR:1 hybrid level:anl:",
    "3:180:d=2023033112:CLMR:850 mb:anl:",
    "4:260:d=2023033112:RWMR:850 mb:anl:",
    "5:300:d=2023033112:SNMR:500 mb:anl:",
    "6:350:d=2023033112:GRLE:500 mb:anl:",
    "7:400:d=2023033112:ICMR:300 mb:anl:",
    "8:450:d=2023033112:TMP:850 mb:anl:",
))


def _gfs_rows():
    rows = {r["id"]: r for r in json.loads(src.TABLE_PATH.read_text())["sources"]}
    return rows["gfs"], rows["gfs-gdex"], rows["gefs"]


def test_the_gfs_rows_decode_the_first_guess_hydrometeors_and_vtable_gfs_is_unchanged():
    gfs, gdex, gefs = _gfs_rows()
    for row in (gfs, gdex):
        vtable = src.resolve_vtable(row)
        names = {line.split("|")[4].strip() for line in vtable.read_text().splitlines()
                 if line.count("|") >= 10 and line.split("|")[1].strip() == "100"}
        # The hex init door hands exactly these to rw_mpas_init --first-guess-hydrometeors.
        assert {"QC", "QR", "QI", "QS", "QG"} <= names <= {"TT", "UU", "VV", "RH", "HGT", "QC", "QR", "QI", "QS", "QG"}
        assert {"QC", "QR", "QI", "QS", "QG"} <= set(FIRST_GUESS_SPECIES)
    # GEFS, the Earth-2 rows and woof's WRF path share Vtable.GFS: it carries none of them.
    shared = src.resolve_vtable(gefs).read_text()
    assert "| QC " not in shared and "| QR " not in shared


def test_the_gfs_index_selection_takes_the_isobaric_hydrometeors_only():
    gfs, _, _ = _gfs_rows()
    select = gfs["products"][0]["index"]["select"]
    chosen = [r["record"] for r in src.select_ncep_idx_records(GFS_IDX, select)]
    assert chosen == ["PRMSL:mean sea level", "CLMR:850 mb", "RWMR:850 mb", "SNMR:500 mb", "GRLE:500 mb",
                      "ICMR:300 mb", "TMP:850 mb"]


def _snapshot(nz=55, ncells=3, rain_cell=1):
    zgrid = np.repeat((np.arange(nz + 1, dtype=np.float32) * 250.0)[:, None], ncells, axis=1)
    p = (1.0e5 * np.exp(-0.5 * (zgrid[:-1] + zgrid[1:]) / 8000.0)).astype(np.float32)
    theta = (290.0 + 0.004 * 0.5 * (zgrid[:-1] + zgrid[1:])).astype(np.float32)
    zero = np.zeros((nz, ncells), np.float32)
    qr, nr = zero.copy(), zero.copy()
    qr[:12, rain_cell] = 1.0e-3     # 1 g/kg of rain in the lowest 3 km of one column
    nr[:12, rain_cell] = 3.0e3
    cell = ("nCells", "nVertLevels")
    fields = dict(qv=np.full((nz, ncells), 8.0e-3, np.float32), qc=zero, qr=qr, nr=nr, qs=zero, qg=zero,
                  theta=theta, pressure=p)
    snap = {k: HistoryField(np.ascontiguousarray(v.T[None]), ("Time",) + cell, {}) for k, v in fields.items()}
    snap["zgrid"] = HistoryField(np.ascontiguousarray(zgrid.T), ("nCells", "nVertLevelsP1"), {})
    for name in ("refl10cm_max", "refl10cm_1km", "refl10cm_1km_max"):
        snap[name] = HistoryField(np.zeros((1, ncells), np.float32), ("Time", "nCells"), {})
    return snap


def test_only_thompson_runs_form_it_and_a_frame_without_inputs_names_why():
    assert start.start_frame_reflectivity({"config_microp_scheme": "mp_tempo"}, {}, device=0) is None
    assert start.start_frame_reflectivity({"config_microp_scheme": "mp_wsm6"}, {}, device=0) is None
    snap = _snapshot()
    del snap["theta"]
    pending = start.start_frame_reflectivity({"config_microp_scheme": "mp_thompson"}, snap, device=0)
    assert pending.thread is None and pending.apply(snap, HistoryField)["applied"] is False
    assert "theta" in pending.record["reason"]


@pytest.mark.gpu
def test_the_start_frame_carries_the_rain_it_starts_with():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    snap = _snapshot()
    pending = start.start_frame_reflectivity({"config_microp_scheme": "mp_thompson"}, snap, device=0)
    record = pending.apply(snap, HistoryField)
    assert record["applied"] is True, record
    composite = np.asarray(snap["refl10cm_max"].values)[0]
    low = np.asarray(snap["refl10cm_1km"].values)[0]
    assert composite[1] >= 30.0 and low[1] >= 30.0          # the rain column has echo
    assert composite[0] == composite[2] == -35.0            # dry columns sit on MPAS's floor
    assert np.asarray(snap["refl10cm_1km_max"].values)[0][1] == low[1]


@pytest.mark.gpu
def test_slabs_give_the_same_words_as_one_whole_state_call():
    # THE BREAKAGE: a multi-card f000 frame is the global frame on rank 0; slabs bound the card's share and must
    # not change a word.
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from woof.hex.mpasphys.rrfs.tempo_refl_v303 import _logical
    snap = _snapshot(ncells=5, rain_cell=3)
    state = {n: _logical(snap[n])[0] for n in start.NEED}
    state["p"] = _logical(snap["pressure"])[0]
    zgrid = _logical(snap["zgrid"])[0]
    maths = start.device_math_source()
    whole = {k: cp.asnumpy(v) for k, v in start.radar(state, zgrid, maths=maths, cp=cp).items()}
    slabs = start.radar_staged(state, zgrid, maths=maths, cp=cp, columns=2)
    for name in start.START_TARGETS:
        assert np.array_equal(whole[name].view(np.uint32), slabs[name].view(np.uint32)), name


# WSM6 (mesoscale_reference): the same f000 breakage, carried to the third MPAS-A microphysics route
# (hex-meshes-sources review 2026-10-29: msr's f000 map still read "No values to plot" after ee94097c).
from woof.hex.mpasphys import wsm6_refl_start_v842 as wsm6_start
from woof.hex.mpasphys.wsm6_v842 import NATIVE_CONSTANT_NAMES

# mpas_atmphys_constants.F words, in wsm6_v842.NATIVE_CONSTANT_NAMES order (tests/wsm6_oracle/compare_driver.py).
WSM6_CONSTANTS = dict(zip(NATIVE_CONSTANT_NAMES, np.array(
    [9.80616, 1004.5, 1846.4, 287, 461.6, 273.15, np.float32(461.6) / np.float32(287) - np.float32(1),
     np.float32(287) / np.float32(461.6), 0, 0, 0, 1e-15, 2.85e6, 2.5e6, 3.5e5, 1.28, 1000, 4190, 2106, 610.78,
     2.49e-6, 4.99e-6, 9.99e-6, 50e-6, 125e-6, 999e-6], np.float32)))
WSM6_CONSTANTS.pop(None, None)


def test_wsm6_runs_form_it_and_the_others_do_not():
    assert wsm6_start.start_frame_reflectivity({"config_microp_scheme": "mp_thompson"}, {}, device=0) is None
    assert wsm6_start.start_frame_reflectivity({"config_microp_scheme": "mp_tempo"}, {}, device=0) is None
    snap = _snapshot()
    del snap["zgrid"]
    pending = wsm6_start.start_frame_reflectivity({"config_microp_scheme": "mp_wsm6"}, snap, device=0,
                                                  constants=WSM6_CONSTANTS)
    assert pending.thread is None and pending.apply(snap, HistoryField)["applied"] is False
    assert "zgrid" in pending.record["reason"]
    assert list(wsm6_start.host_constants(WSM6_CONSTANTS)[[2, 15, 16, 17]]) == [np.float32(1846.4), 1.28, 1000, 4190]


def _wsm6_rain_dbz(snap, cell):
    """Rain-only WSM6 Ze (mp_wsm6.F90:2316-2377, mu_r = 0): n0r * Gamma(7) / lamr**7, in float64."""
    from woof.hex.mpasphys.rrfs.tempo_refl_v303 import _logical
    p = _logical(snap["pressure"])[0][:, cell].astype(np.float64)
    t = _logical(snap["theta"])[0][:, cell] * (p / 1.0e5) ** (287.0 / 1004.5)
    qv = np.maximum(_logical(snap["qv"])[0][:, cell].astype(np.float64), 1e-10)
    qr = _logical(snap["qr"])[0][:, cell].astype(np.float64)
    rho = 0.622 * p / (287.0 * t * (qv + 0.622))
    wet = qr > 1e-9
    lam = (np.pi * 1000.0 * 8.0e6 / (rho[wet] * qr[wet])) ** 0.25
    return 10.0 * np.log10(8.0e6 * 720.0 / lam ** 7 * 1e18)


@pytest.mark.gpu
def test_the_wsm6_start_frame_carries_the_rain_it_starts_with():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    snap = _snapshot()
    expected = _wsm6_rain_dbz(snap, 1)
    pending = wsm6_start.start_frame_reflectivity({"config_microp_scheme": "mp_wsm6"}, snap, device=0,
                                                  constants=WSM6_CONSTANTS)
    record = pending.apply(snap, HistoryField)
    assert record["applied"] is True, record
    composite = np.asarray(snap["refl10cm_max"].values)[0]
    low = np.asarray(snap["refl10cm_1km"].values)[0]
    assert abs(composite[1] - expected.max()) < 0.05, (composite[1], expected.max())   # WSM6's own rain Ze
    assert low[1] >= 30.0
    assert composite[0] == composite[2] == -35.0            # dry columns sit on MPAS's floor
    assert np.asarray(snap["refl10cm_1km_max"].values)[0][1] == low[1]


@pytest.mark.gpu
def test_wsm6_slabs_give_the_same_words_as_one_whole_state_call():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    from woof.hex.mpasphys.rrfs.tempo_refl_v303 import _logical
    snap = _snapshot(ncells=5, rain_cell=3)
    state = {n: _logical(snap[n])[0] for n in wsm6_start.NEED}
    state["p"] = _logical(snap["pressure"])[0]
    zgrid = _logical(snap["zgrid"])[0]
    words = wsm6_start.host_constants(WSM6_CONSTANTS)
    whole = {k: cp.asnumpy(v) for k, v in wsm6_start.radar(state, zgrid, constants=words, cp=cp).items()}
    slabs = wsm6_start.radar_staged(state, zgrid, constants=words, cp=cp, columns=2)
    for name in wsm6_start.START_TARGETS:
        assert np.array_equal(whole[name].view(np.uint32), slabs[name].view(np.uint32)), name
