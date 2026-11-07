"""RTE+RRTMGP radiation (Phase 4 Task 3).

Reference authority: earth-system-radiation/rte-rrtmgp commit
``fa107a16120051c4124305c6b3d4c87059119f58`` and rrtmgp-data v1.9
commit ``eff0433faf9cbac3ad14fbf608bef0c26ebc4c79``.  Tests are ordered like
the plan deliverables: tables, gas optics, RTE, cloud optics, column driver.
"""

import os
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof import data_assets


#: Resolved, never joined.  These NetCDFs ship in the ``recast-woof-data``
#: companion distribution since 2.5.0, and re-deriving the path here from
#: the repository root would test a directory that no longer exists.
DATA_DIR = data_assets.rrtmgp_data_dir()


def _rfmip(name):
    """A pinned RFMIP file, fetched from upstream at test time.

    The five RFMIP NetCDFs are not in the tree or in any artifact
    (woof/core/rfmip_upstream.py says why), so a test that reads one
    fetches it, SHA-256 verified, into the RFMIP cache.  Without network
    and without a cached copy the test skips and says so.
    """
    from woof.core.rfmip_upstream import RfmipUnavailable, fetch_rfmip

    try:
        return fetch_rfmip(name)
    except RfmipUnavailable as error:
        pytest.skip(f"RFMIP upstream file unavailable: {error}")


_RFMIP_NAMES = {
    "co2": "carbon_dioxide", "n2o": "nitrous_oxide",
    "co": "carbon_monoxide", "ch4": "methane", "o2": "oxygen",
    "n2": "nitrogen", "ccl4": "carbon_tetrachloride",
    "cfc11": "cfc11", "cfc12": "cfc12", "cfc22": "hcfc22",
    "hfc143a": "hfc143a", "hfc125": "hfc125", "hfc23": "hfc23",
    "hfc32": "hfc32", "hfc134a": "hfc134a", "cf4": "cf4",
}


def test_trace_gases_selects_documented_annual_co2_and_override():
    from datetime import datetime
    from woof.core.rrtmgp import trace_gases

    may1999 = trace_gases(datetime(1999, 5, 3, 12))
    modern = trace_gases(datetime(2024, 7, 1))
    assert may1999 == {"co2": pytest.approx(367.80e-6)}
    assert modern == {"co2": pytest.approx(422.79e-6)}
    assert modern["co2"] > may1999["co2"]
    # The NOAA global series starts in 1979; earlier dates hold the first
    # documented annual value rather than inheriting the 2014 RFMIP baseline.
    assert trace_gases(datetime(1974, 4, 3, 12)) == {
        "co2": pytest.approx(336.85e-6)}
    assert trace_gases(
        datetime(1974, 4, 3, 12), {"co2": 330.0e-6}) == {
            "co2": pytest.approx(330.0e-6)}
    with pytest.raises(ValueError, match="unknown trace gas"):
        trace_gases(datetime(2024, 1, 1), {"mystery": 1.0e-6})


def test_trace_gas_table_provenance_is_committed_and_runtime_offline():
    from woof.core import rrtmgp

    table = rrtmgp._NOAA_GML_CO2_ANNUAL_PPM
    assert (min(table), table[min(table)]) == (1979, 336.85)
    assert (max(table), table[max(table)]) == (2025, 425.64)


def _cloud_reference_output(kind):
    rows = list(csv.DictReader(
        (DATA_DIR / f"cloud-optics-reference-{kind}.csv").open(
            encoding="ascii", newline="")))
    ncol = 1 + max(int(row["column"]) for row in rows)
    nband = 1 + max(int(row["band"]) for row in rows)
    inputs = {name: np.empty(ncol, np.float64)
              for name in ("lwp", "iwp", "reliq", "dgice")}
    expected = {name: np.empty((ncol, nband), np.float64)
                for name in ("tau", "ssa", "g")}
    for row in rows:
        column, band = int(row["column"]), int(row["band"])
        for name in inputs:
            inputs[name][column] = float(row[name])
        for name in expected:
            expected[name][column, band] = float(row[name])
    return inputs, expected


def _rfmip_columns(tables, sites=(0, 17), experiment=0):
    """Read selected upstream RFMIP columns in kernel ``(col,lay)`` order."""
    from netCDF4 import Dataset

    sites = np.asarray(sites, dtype=np.intp)
    with Dataset(_rfmip("rfmip-clear-sky-inputs.nc")) as nc:
        play = np.asarray(nc["pres_layer"][sites], np.float64)
        plev = np.asarray(nc["pres_level"][sites], np.float64)
        tlay = np.asarray(nc["temp_layer"][experiment, sites], np.float64)
        vmr = np.zeros((sites.size, play.shape[1], tables.ngas + 1),
                       np.float64)
        vmr[:, :, tables.gas_index["h2o"]] = np.asarray(
            nc["water_vapor"][experiment, sites], np.float64)
        vmr[:, :, tables.gas_index["o3"]] = np.asarray(
            nc["ozone"][experiment, sites], np.float64)
        for gas, rfmip_name in _RFMIP_NAMES.items():
            values = nc[rfmip_name + "_GM"]
            scale = float(getattr(values, "units", "1").replace(" ", ""))
            vmr[:, :, tables.gas_index[gas]] = \
                float(values[experiment]) * scale
        # RFMIP has no NO2 field; the reference example explicitly sets zero.
        return play, plev, tlay, vmr


# ---------------------------------------------------------------------------
# (a) coefficient loading and packing
# ---------------------------------------------------------------------------

def test_vendored_rrtmgp_v19_data_and_provenance():
    expected_hashes = {
        "rrtmgp-gas-lw-g256.nc": "4048360199d1917ed8f2ccaae2ec097d0f990da3bbad9830337b739b4fa01be7",
        "rrtmgp-gas-sw-g224.nc": "584f1dd41ea9fc07d4ee3754eb1dafbd46ad3161cd6fd20fa06b6922b6f0702e",
        "rrtmgp-clouds-lw-bnd.nc": "09d6704c5b863b4c3ceb417d20bb3076ec492e6bf2dfbcc9f3c5996a3706f0b0",
        "rrtmgp-clouds-sw-bnd.nc": "7671835992a45afe66244b591a02c0b3df73d7d59ecb746bbffd9763497651cd",
        "rrtmgp-trace-gas-climatology.json": "71d7f85758fda8cf05df66100ccd0a974fed71216a1b7facad2083bc3dd3b70e",
        "cloud-optics-reference-driver.F90": "3996197f0e712f4f0cb881d954140e63b527a2f4eb207d031d4ce3853b4906cc",
        "cloud-optics-reference-lw.csv": "654ee18ac84d92d0471bc80862af389f85506882a730d18e6ed920e24b97043d",
        "cloud-optics-reference-sw.csv": "3f065cba7546a783d3736e5ef17a42ebfa8e3c08c42f25ae529c4385a1603399",
    }
    expected_licenses = {"LICENSE", "LICENSE-CC-BY-SA-4.0"}
    present = {p.name for p in DATA_DIR.iterdir()}
    assert expected_hashes.keys() | expected_licenses | {"PROVENANCE.md"} \
        <= present
    # The RFMIP files, and the two licences only they needed, are fetched
    # from upstream by the tests that read them and never carried here.
    from woof.core.rfmip_upstream import RFMIP_FILES
    assert not present & (set(RFMIP_FILES) | {
        "LICENSE-CC-BY-4.0", "LICENSE-CC-BY-NC-SA-4.0"})
    for name, expected in expected_hashes.items():
        assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() \
            == expected
    provenance = (DATA_DIR / "PROVENANCE.md").read_text(encoding="utf8")
    assert "v1.9" in provenance
    assert "eff0433faf9cbac3ad14fbf608bef0c26ebc4c79" in provenance
    assert "fa107a16120051c4124305c6b3d4c87059119f58" in provenance
    assert "BSD 3-Clause" in provenance
    assert 'Attribution "Share Alike" 4.0 International License' in provenance
    assert "http://creativecommons.org/licenses/by/4.0/" in provenance
    assert "genuinely ambiguous" in provenance
    assert "CC-BY-SA-4.0" in provenance
    assert "CC-BY-NC-SA-4.0" in provenance
    assert "Fast math" in provenance and "OFF" in provenance
    assert "FMA" in provenance and "Reduction order" in provenance
    assert "Attribution-ShareAlike 4.0 International Public License" \
        in (DATA_DIR / "LICENSE-CC-BY-SA-4.0").read_text(encoding="utf8")


def test_load_gas_tables_dimensions_names_and_packed_indices():
    from woof.core.rrtmgp import load_gas_tables

    lw = load_gas_tables("lw")
    sw = load_gas_tables("sw")
    assert (lw.nband, lw.ngpt, lw.ntemp, lw.npres, lw.neta) == (
        16, 256, 14, 59, 9)
    assert (sw.nband, sw.ngpt, sw.ntemp, sw.npres, sw.neta) == (
        14, 224, 14, 59, 9)
    assert lw.gas_names[:8] == (
        "h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2")
    assert lw.gas_index["h2o"] == 1  # slot zero is dry air, as upstream
    assert lw.kmajor.shape == (14, 9, 60, 256)
    assert sw.rayleigh.shape == (2, 14, 9, 224)
    assert lw.band_lims_gpt.min() == 0
    assert lw.band_lims_gpt.max() == 255
    assert np.array_equal(lw.band_lims_gpt[:, 0],
                          np.r_[0, lw.band_lims_gpt[:-1, 1] + 1])
    assert lw.kminor_start_lower.min() == 0
    assert lw.gpoint_flavor.min() == 0
    assert lw.flavor.min() >= 0
    for array in lw.packed_arrays().values():
        assert array.flags.c_contiguous
        assert array.dtype in (np.dtype("float64"), np.dtype("int32"),
                               np.dtype("bool"))


@pytest.mark.parametrize("name,dimensions,permutation", [
    ("kmajor",
     ("temperature", "mixing_fraction", "pressure_interp", "gpt"),
     (3, 0, 2, 1)),
    ("plank_fraction",
     ("temperature", "mixing_fraction", "pressure_interp", "gpt"),
     (1, 3, 0, 2)),
    ("kminor_lower",
     ("temperature", "mixing_fraction", "contributors_lower"),
     (2, 0, 1)),
    ("kminor_upper",
     ("temperature", "mixing_fraction", "contributors_upper"),
     (1, 2, 0)),
])
def test_rrtmgp_named_dimension_packing_is_permutation_invariant(
        name, dimensions, permutation):
    from woof.core.rrtmgp import _packed_variable

    class SyntheticVariable:
        def __init__(self, data, dims):
            self.name = name
            self._data = data
            self.dimensions = dims

        def __getitem__(self, key):
            return self._data[key]

    shape = tuple(range(2, len(dimensions) + 2))
    canonical = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    baseline = _packed_variable(
        SyntheticVariable(canonical, dimensions), dimensions, np.float64)
    permuted = _packed_variable(
        SyntheticVariable(np.transpose(canonical, permutation),
                          tuple(dimensions[i] for i in permutation)),
        dimensions, np.float64)
    np.testing.assert_array_equal(permuted, baseline)
    assert permuted.flags.c_contiguous


def test_load_cloud_tables_reference_bounds_and_layout():
    from woof.core.rrtmgp import load_cloud_tables

    lw = load_cloud_tables("lw")
    sw = load_cloud_tables("sw")
    assert (lw.nband, lw.nsize_liq, lw.nsize_ice, lw.nrghice) == (
        16, 20, 18, 3)
    assert (sw.nband, sw.nsize_liq, sw.nsize_ice, sw.nrghice) == (
        14, 20, 18, 3)
    assert lw.extliq.shape == (20, 16)
    assert lw.extice.shape == (18, 16, 3)
    assert lw.radliq_lwr < lw.radliq_upr
    assert lw.diamice_lwr < lw.diamice_upr
    assert np.all(lw.extliq >= 0.0)
    assert np.all((lw.ssaliq >= 0.0) & (lw.ssaliq <= 1.0))


@pytest.mark.gpu
@requires_gpu
def test_tables_upload_once_as_fp32_device_tables():
    import cupy as cp
    from woof.core.rrtmgp import load_gas_tables

    host = load_gas_tables("sw")
    first = host.to_device()
    second = host.to_device()
    assert first is second
    assert first.kmajor.dtype == cp.float32
    assert first.kmajor.shape == host.kmajor.shape
    assert first.band_lims_gpt.dtype == cp.int32
    assert bool(cp.isfinite(first.kmajor).all())


# ---------------------------------------------------------------------------
# (b) gas optics
# ---------------------------------------------------------------------------

def test_rrtmgp_dry_column_number_matches_reference_equation():
    from woof.verify.npref import np_rrtmgp_col_dry

    plev = np.array([[100.0, 20000.0, 100000.0]], np.float64)
    h2o = np.array([[0.0, 0.02]], np.float64)
    got = np_rrtmgp_col_dry(h2o, plev)
    avogad, m_dry, m_h2o, grav = (
        6.02214076e23, 0.028964, 0.018016, 9.80665)
    fact = 1.0 / (1.0 + h2o)
    m_air = (m_dry + m_h2o * h2o) * fact
    expected = (np.abs(np.diff(plev, axis=1)) * avogad * fact
                / (10000.0 * m_air * grav))
    np.testing.assert_allclose(got, expected, rtol=2e-15)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_interface_temperatures_match_reference_pressure_scheme():
    import cupy as cp
    from woof.core.rrtmgp import _interface_temperatures

    play = np.array([[90000.0, 60000.0, 30000.0],
                     [85000.0, 50000.0, 20000.0]])
    plev = np.array([[100000.0, 75000.0, 45000.0, 10000.0],
                     [95000.0, 67500.0, 35000.0, 5000.0]])
    tlay = np.array([[290.0, 260.0, 230.0],
                     [285.0, 250.0, 220.0]])
    expected = np.empty((2, 4), np.float64)
    expected[:, 0] = tlay[:, 0] + (plev[:, 0] - play[:, 0]) * (
        tlay[:, 1] - tlay[:, 0]) / (play[:, 1] - play[:, 0])
    expected[:, -1] = tlay[:, -1] + (plev[:, -1] - play[:, -1]) * (
        tlay[:, -1] - tlay[:, -2]) / (play[:, -1] - play[:, -2])
    expected[:, 1:-1] = (
        play[:, :-1] * tlay[:, :-1] * (plev[:, 1:-1] - play[:, 1:])
        + play[:, 1:] * tlay[:, 1:] * (play[:, :-1] - plev[:, 1:-1])
    ) / (plev[:, 1:-1] * (play[:, :-1] - play[:, 1:]))
    np.testing.assert_allclose(
        cp.asnumpy(_interface_temperatures(cp.asarray(play), cp.asarray(plev),
                                           cp.asarray(tlay))),
        expected, rtol=2e-7, atol=2e-5)


def _wrf_lw_upper_temperature_oracle(upper_plev_pa, model_top_temperature,
                                     model_top_pressure_pa):
    """Independent transcription of WRF v4.6.1's 60-level LW table."""
    pprof_hpa = np.array([
        1000.00, 855.47, 731.82, 626.05, 535.57, 458.16,
        391.94, 335.29, 286.83, 245.38, 209.91, 179.57,
        153.62, 131.41, 112.42, 96.17, 82.27, 70.38,
        60.21, 51.51, 44.06, 37.69, 32.25, 27.59,
        23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
        9.25, 7.91, 6.77, 5.79, 4.95, 4.24,
        3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
        1.42, 1.22, 1.04, 0.89, 0.76, 0.65,
        0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
        0.22, 0.19, 0.16, 0.14, 0.12, 0.10,
    ], np.float64)
    tprof = np.array([
        286.96, 281.07, 275.16, 268.11, 260.56, 253.02,
        245.62, 238.41, 231.57, 225.91, 221.72, 217.79,
        215.06, 212.74, 210.25, 210.16, 210.69, 212.14,
        213.74, 215.37, 216.82, 217.94, 219.03, 220.18,
        221.37, 222.64, 224.16, 225.88, 227.63, 229.51,
        231.50, 233.73, 236.18, 238.78, 241.60, 244.44,
        247.35, 250.33, 253.32, 256.30, 259.22, 262.12,
        264.80, 266.50, 267.59, 268.44, 268.69, 267.76,
        266.13, 263.96, 261.54, 258.93, 256.15, 253.23,
        249.89, 246.67, 243.48, 240.25, 236.66, 233.86,
    ], np.float64)
    order = np.argsort(pprof_hpa)
    def climo(pressure_pa):
        return np.interp(
            np.asarray(pressure_pa) * 0.01, pprof_hpa[order], tprof[order])
    shift = model_top_temperature - climo(model_top_pressure_pa)
    return climo(upper_plev_pa) + shift


def test_above_model_profiles_match_wrf_v461_lw_and_sw_construction():
    """The 100-hPa real74 top becomes WRF's +25 LW / +1 SW layers."""
    from woof.core.rrtmgp import (
        RRTMGP_TOA_PRESSURE_PA, _extend_above_model_profile)

    ncol, nz = 2, 49
    plev = np.broadcast_to(
        np.linspace(100000.0, 10000.0, nz + 1, dtype=np.float32),
        (ncol, nz + 1)).copy()
    play = 0.5 * (plev[:, :-1] + plev[:, 1:])
    tlay = np.broadcast_to(
        np.linspace(292.0, 218.0, nz, dtype=np.float32),
        (ncol, nz)).copy()
    tlev = np.broadcast_to(
        np.linspace(293.0, 215.0, nz + 1, dtype=np.float32),
        (ncol, nz + 1)).copy()
    tlev[1, -1] = 223.0
    qv = np.broadcast_to(
        np.geomspace(1.0e-2, 2.0e-6, nz).astype(np.float32),
        (ncol, nz)).copy()

    lw = _extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=10000.0, kind="lw",
        pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=np)
    assert lw.model_nlay == 49
    assert lw.upper_nlay == 25
    assert lw.play.shape == lw.tlay.shape == lw.qv.shape == (2, 74)
    assert lw.plev.shape == lw.tlev.shape == (2, 75)
    expected_upper_plev = np.r_[
        np.arange(9600.0, 0.0, -400.0), RRTMGP_TOA_PRESSURE_PA]
    np.testing.assert_allclose(lw.plev[0, 50:], expected_upper_plev,
                               rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(
        lw.play[0, 49:],
        0.5 * (np.r_[10000.0, expected_upper_plev[:-1]]
               + expected_upper_plev), rtol=0.0, atol=1.0e-4)
    for col in range(ncol):
        expected_tlev = _wrf_lw_upper_temperature_oracle(
            expected_upper_plev, tlev[col, -1], 10000.0)
        np.testing.assert_allclose(lw.tlev[col, 50:], expected_tlev,
                                   rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(
            lw.tlay[col, 49:],
            0.5 * (np.r_[tlev[col, -1], expected_tlev[:-1]]
                   + expected_tlev), rtol=0.0, atol=2.0e-5)
        np.testing.assert_array_equal(lw.qv[col, 49:], qv[col, -1])

    sw = _extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=10000.0, kind="sw",
        pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=np)
    assert sw.model_nlay == 49
    assert sw.upper_nlay == 1
    assert sw.play.shape == sw.tlay.shape == sw.qv.shape == (2, 50)
    assert sw.plev.shape == sw.tlev.shape == (2, 51)
    np.testing.assert_array_equal(sw.play[:, -1], np.float32(5000.0))
    np.testing.assert_array_equal(sw.tlay[:, -1], tlev[:, -1])
    np.testing.assert_array_equal(sw.tlev[:, -1], tlev[:, -1])
    np.testing.assert_array_equal(sw.qv[:, -1], qv[:, -1])
    np.testing.assert_allclose(sw.plev[:, -1], RRTMGP_TOA_PRESSURE_PA,
                               rtol=0.0, atol=1.0e-7)


def test_above_model_ladder_stays_above_the_floor_for_a_low_column_top():
    """A column top below the declared p_top's ladder keeps positive layers.

    Breakage: the MPAS column seam hands a top pressure per column and one
    declared p_top.  On the global 30 km mesh (2026-10-11) the columns whose
    top sat more than 4 hPa below the declared value got a negative
    above-model interface and RRTMGP longwave returned non-finite heating.
    """
    from woof.core.rrtmgp import (
        RRTMGP_TOA_PRESSURE_PA, _extend_above_model_profile)

    tops = np.array([1150.0, 700.0, 801.0, 1600.0], np.float32)
    ncol, nz = tops.size, 6
    plev = np.stack([np.linspace(100000.0, top, nz + 1) for top in tops]).astype(np.float32)
    play = (0.5 * (plev[:, :-1] + plev[:, 1:])).astype(np.float32)
    tlay = np.broadcast_to(np.linspace(288.0, 225.0, nz, dtype=np.float32), (ncol, nz)).copy()
    tlev = np.broadcast_to(np.linspace(290.0, 224.0, nz + 1, dtype=np.float32), (ncol, nz + 1)).copy()
    qv = np.full((ncol, nz), 3.0e-6, np.float32)

    lw = _extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=1150.0, kind="lw",
        pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=np, validate_top=False)
    assert lw.upper_nlay == 3
    upper = lw.plev[:, nz:]
    assert np.all(np.isfinite(lw.plev)) and np.all(np.isfinite(lw.tlev))
    assert np.all(np.diff(upper, axis=1) < 0.0)
    assert np.all(upper[:, -1] == np.float32(RRTMGP_TOA_PRESSURE_PA))
    assert np.all(lw.play[:, nz:] > RRTMGP_TOA_PRESSURE_PA)
    # WRF's 4-hPa ladder is untouched wherever it stays above the floor.
    for col in (0, 3):
        np.testing.assert_array_equal(
            upper[col, 1:3], np.float32(tops[col]) - np.array([400.0, 800.0], np.float32))
    # The low tops share their own span evenly instead.
    np.testing.assert_allclose(
        upper[1, 1:3], 700.0 + (RRTMGP_TOA_PRESSURE_PA - 700.0) * np.array([1, 2]) / 3.0,
        rtol=1.0e-6)


def test_above_model_profile_guard_and_model_flux_slice():
    from woof.core.rrtmgp import (
        RRTMGP_TOA_PRESSURE_PA, _extend_above_model_profile,
        _model_flux_interfaces)

    play = np.array([[80000.0, 30000.0]], np.float32)
    plev = np.array([[100000.0, 50000.0, 10000.0]], np.float32)
    tlay = np.array([[285.0, 235.0]], np.float32)
    tlev = np.array([[290.0, 260.0, 220.0]], np.float32)
    qv = np.array([[8.0e-3, 1.0e-5]], np.float32)
    with pytest.raises(ValueError, match="workspace top pressure"):
        _extend_above_model_profile(
            play, plev, tlay, tlev, qv, p_top=5000.0, kind="lw",
            pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=np)
    with pytest.raises(ValueError, match="128"):
        _extend_above_model_profile(
            np.broadcast_to(play[:, :1], (1, 120)),
            np.linspace(100000.0, 10000.0, 121)[None],
            np.broadcast_to(tlay[:, :1], (1, 120)),
            np.linspace(290.0, 220.0, 121)[None],
            np.broadcast_to(qv[:, :1], (1, 120)),
            p_top=10000.0, kind="lw",
            pressure_floor=RRTMGP_TOA_PRESSURE_PA, xp=np)

    full_flux = np.arange(2 * 28, dtype=np.float32).reshape(2, 28)
    model_flux = _model_flux_interfaces(full_flux, model_nlay=2, xp=np)
    np.testing.assert_array_equal(model_flux, full_flux[:, :3])
    assert model_flux[0, -1] == full_flux[0, 2]  # model top, not TOA
    with pytest.raises(ValueError, match="does not reach model top"):
        _model_flux_interfaces(full_flux[:, :2], model_nlay=2, xp=np)


def test_above_model_cap_and_metadata_are_exact_chunk_local_with_tail(
        monkeypatch):
    """Host proof that 3+3+1 chunks reproduce one full 7-column cap.

    Spies pin the allocation-facing bounds too: profile extension, all five
    clear-layer fields, and all five interpolation-coordinate arrays see the
    real one-column tail, never a padded or full-domain extent.
    """
    from types import SimpleNamespace
    from woof.core import rrtmgp

    ncol, nz, chunk_size = 7, 4, 3
    plev = np.broadcast_to(
        np.linspace(100000.0, 10000.0, nz + 1, dtype=np.float32),
        (ncol, nz + 1)).copy()
    play = np.float32(0.5) * (plev[:, :-1] + plev[:, 1:])
    tlay = np.broadcast_to(
        np.linspace(291.0, 221.0, nz, dtype=np.float32),
        (ncol, nz)).copy()
    tlay += np.arange(ncol, dtype=np.float32)[:, None]
    tlev = np.broadcast_to(
        np.linspace(293.0, 216.0, nz + 1, dtype=np.float32),
        (ncol, nz + 1)).copy()
    tlev[:, -1] += np.arange(ncol, dtype=np.float32)
    qv = np.broadcast_to(
        np.geomspace(8.0e-3, 2.0e-6, nz).astype(np.float32),
        (ncol, nz)).copy()
    qv *= (np.arange(ncol, dtype=np.float32)[:, None] + np.float32(1.0))
    base = np.arange(ncol * nz, dtype=np.float32).reshape(ncol, nz)
    paths = rrtmgp.HydrometeorPaths(
        base + np.float32(1.0), base + np.float32(2.0),
        base + np.float32(3.0), base + np.float32(4.0))
    cldfra = np.float32(0.01) * base

    expected = {}
    for kind in ("lw", "sw"):
        profile = rrtmgp._extend_above_model_profile(
            play, plev, tlay, tlev, qv, p_top=10000.0, kind=kind, xp=np)
        expected[kind] = (
            profile,
            rrtmgp.HydrometeorPaths(*(
                rrtmgp._append_clear_upper_layers(
                    value, profile.upper_nlay, xp=np)
                for value in (paths.clwp, paths.ciwp,
                              paths.reliq, paths.dgice))),
            rrtmgp._append_clear_upper_layers(
                cldfra, profile.upper_nlay, xp=np))

    original_extend = rrtmgp._extend_above_model_profile
    original_append = rrtmgp._append_clear_upper_layers
    extended_rows = []
    clear_rows = []
    metadata_shapes = []

    def extend_spy(play_chunk, *args, kind, **kwargs):
        extended_rows.append((kind, play_chunk.shape[0]))
        return original_extend(play_chunk, *args, kind=kind, **kwargs)

    def append_spy(value, upper_nlay, **kwargs):
        clear_rows.append((upper_nlay, value.shape[0]))
        return original_append(value, upper_nlay, **kwargs)

    def metadata_spy(tables, play_chunk, tlay_chunk, *, validate,
                     scratch=None):
        assert not validate
        assert play_chunk.shape == tlay_chunk.shape
        metadata_shapes.append((tables.kind, play_chunk.shape))
        integer = tuple(np.zeros(play_chunk.shape, np.int32) for _ in range(3))
        fraction = tuple(np.zeros(play_chunk.shape, np.float32)
                         for _ in range(2))
        return rrtmgp._InterpolationMetadata(*integer, *fraction)

    monkeypatch.setattr(rrtmgp, "_extend_above_model_profile", extend_spy)
    monkeypatch.setattr(rrtmgp, "_append_clear_upper_layers", append_spy)
    monkeypatch.setattr(rrtmgp, "_interpolation_metadata", metadata_spy)

    for kind, nlay in (("lw", nz + 25), ("sw", nz + 1)):
        pieces = []
        for start in range(0, ncol, chunk_size):
            sl = slice(start, min(start + chunk_size, ncol))
            pieces.append(rrtmgp._prepare_above_model_chunk(
                tables=SimpleNamespace(kind=kind), play=play, plev=plev,
                tlay=tlay, tlev=tlev, qv=qv, paths=paths, cldfra=cldfra,
                columns=sl, p_top=10000.0, kind=kind, xp=np,
                validate=False))

        full_profile, full_paths, full_cldfra = expected[kind]
        for name in ("play", "plev", "tlay", "tlev", "qv"):
            np.testing.assert_array_equal(
                np.concatenate([getattr(piece.profile, name)
                                for piece in pieces]),
                getattr(full_profile, name))
        for name in ("clwp", "ciwp", "reliq", "dgice"):
            np.testing.assert_array_equal(
                np.concatenate([getattr(piece.paths, name)
                                for piece in pieces]),
                getattr(full_paths, name))
        np.testing.assert_array_equal(
            np.concatenate([piece.cldfra for piece in pieces]), full_cldfra)
        assert [piece.profile.play.shape for piece in pieces] == [
            (3, nlay), (3, nlay), (1, nlay)]
        assert [piece.metadata.jt.shape for piece in pieces] == [
            (3, nlay), (3, nlay), (1, nlay)]

    assert extended_rows == [
        ("lw", 3), ("lw", 3), ("lw", 1),
        ("sw", 3), ("sw", 3), ("sw", 1)]
    assert metadata_shapes == [
        ("lw", (3, 29)), ("lw", (3, 29)), ("lw", (1, 29)),
        ("sw", (3, 5)), ("sw", (3, 5)), ("sw", (1, 5))]
    assert [rows for _, rows in clear_rows] == [
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 1]


def test_uniform_top_above_model_fast_path_is_bitwise_identical():
    """The cap above the model is built from `plev[:, -1]`, and WRF's mass
    coordinate makes that ONE number for the whole grid.  `uniform_top`
    exploits that: the pressure ladder and the two climatology lookups on
    it become cached (1, upper) rows instead of (ncol, upper) arrays
    rebuilt per chunk.

    The claim is checked on device once per firing by the driver, so what
    has to hold here is that granting it changes nothing -- every field of
    the extended profile, bit for bit, across model tops that give
    different upper-layer counts.
    """
    cp = pytest.importorskip("cupy")
    from woof.core.rrtmgp import (DTYPE, _extend_above_model_profile,
                                   rrtmgp_above_model_layer_counts)

    rng = np.random.default_rng(11)
    seen = set()
    for p_top in (10000.0, 5000.0, 1000.0, 20000.0):
        seen.add(rrtmgp_above_model_layer_counts(p_top)[0])
        for kind in ("lw", "sw"):
            for ncol, nlay in ((1024, 49), (377, 49), (1, 49)):
                edge = np.linspace(101325.0, p_top, nlay + 1)
                plev = cp.asarray(np.tile(edge, (ncol, 1)), dtype=DTYPE)
                plev[:, -1] = DTYPE(p_top)
                play = cp.asarray(
                    np.tile(0.5 * (edge[:-1] + edge[1:]), (ncol, 1)),
                    dtype=DTYPE)
                tlay = cp.asarray(rng.uniform(190, 310, (ncol, nlay)),
                                  dtype=DTYPE)
                tlev = cp.asarray(rng.uniform(190, 310, (ncol, nlay + 1)),
                                  dtype=DTYPE)
                qv = cp.asarray(rng.uniform(1e-7, 3e-2, (ncol, nlay)),
                                dtype=DTYPE)
                kwargs = dict(p_top=p_top, kind=kind, xp=cp,
                              validate_top=False)
                slow = _extend_above_model_profile(
                    play, plev, tlay, tlev, qv, uniform_top=False, **kwargs)
                fast = _extend_above_model_profile(
                    play, plev, tlay, tlev, qv, uniform_top=True, **kwargs)
                assert slow.upper_nlay == fast.upper_nlay
                for field in ("play", "plev", "tlay", "tlev", "qv"):
                    a = getattr(slow, field)
                    b = getattr(fast, field)
                    assert a.shape == b.shape, (kind, field, p_top)
                    assert bool(cp.all(a.view(cp.uint32)
                                       == b.view(cp.uint32))), (
                        f"{kind} {field} p_top={p_top} {ncol}x{nlay}")
    # The tops chosen actually exercise different ladder lengths, so this is
    # not four repeats of one shape.
    assert len(seen) == 4


def test_the_driver_checks_the_uniform_top_claim_it_passes_down():
    """`uniform_top` is a promise about the data, and the chunk loop is
    bit-exact only while it holds -- so the driver must MEASURE it, not
    assume it.  Exact equality, because the tolerance in
    `_validate_model_top_interface` would wave through a one-ULP spread
    that still changes the synthetic cap."""
    import inspect
    from woof.core.rrtmgp import RRTMGPRadiation

    source = inspect.getsource(RRTMGPRadiation.__call__)
    assert "uniform_top = bool(cp.all(plev[:, -1] == plev[0, -1]))" in source
    assert source.count("uniform_top=uniform_top") == 2


def test_rte_tile_scales_with_the_kernel_frame_not_a_fixed_thread_count():
    """The g-point tile trades parallelism against the per-thread column
    frame the solver spills to local memory.  It was tuned when BOTH
    solvers carried 1.2-2.7 kB; halving LW to 592 B moved its optimum to
    twice the SM coverage while SW, still at 2088 B, stayed put.

    Asserting the split rather than the numbers: LW must get a bigger tile
    than SW at the same column count, and the fallback for a caller with no
    kernel in hand must be the conservative one.
    """
    cp = pytest.importorskip("cupy")
    del cp
    from woof.core.rrtmgp import _rte_gpt_tile, _rte_kernel

    lw = _rte_kernel("rrtmgp_lw_noscat", 74)
    sw = _rte_kernel("rrtmgp_sw_2stream", 74)
    assert lw.attributes["local_size_bytes"] <= 1024
    assert sw.attributes["local_size_bytes"] > 1024

    lw_tile = _rte_gpt_tile(lw, 1024, 256)
    sw_tile = _rte_gpt_tile(sw, 1024, 224)
    assert lw_tile == 2 * sw_tile, (lw_tile, sw_tile)
    # No kernel to inspect -> the frame is unknown -> take the small tile.
    assert _rte_gpt_tile(None, 1024, 256) == sw_tile
    # Still bounded by the g-point count and never zero.
    assert _rte_gpt_tile(lw, 1024, 8) <= 8
    assert _rte_gpt_tile(lw, 10 ** 9, 256) >= 1


@pytest.mark.parametrize("sm_count", [20, 128, 170])
def test_buffered_fold_tiles_are_card_independent(monkeypatch, sm_count):
    import woof.core.rrtmgp as r

    monkeypatch.setattr(r, "_rte_sm_count", lambda: sm_count)
    assert r._rte_gpt_tile(None, 3125, 256, fold=True, longwave=True) == 128
    assert r._rte_gpt_tile(None, 3125, 224, fold=True) == 32
    assert r._rte_gpt_tile(None, 971, 256, fold=True, longwave=True) == 128
    assert r._rte_gpt_tile(None, 3125, 16, fold=True, longwave=True) == 16
    assert r._rte_gpt_tile(None, 10 ** 9, 256, fold=True, longwave=True) == 128


def test_rte_tile_width_override_is_measurement_only(monkeypatch):
    import woof.core.rrtmgp as r

    monkeypatch.setenv("WOOF_RTE_TILE_WIDTH", "16")
    assert r._rte_gpt_tile(None, 3125, 256, fold=True, longwave=True) == 16
    assert r._rte_gpt_tile(None, 3125, 224, fold=True) == 16
    monkeypatch.setenv("WOOF_RTE_LW_TILE_WIDTH", "64")
    monkeypatch.setenv("WOOF_RTE_SW_TILE_WIDTH", "32")
    assert r._rte_gpt_tile(None, 3125, 256, fold=True, longwave=True) == 64
    assert r._rte_gpt_tile(None, 3125, 224, fold=True) == 32
    monkeypatch.delenv("WOOF_RTE_LW_TILE_WIDTH")
    monkeypatch.delenv("WOOF_RTE_SW_TILE_WIDTH")
    monkeypatch.setenv("WOOF_RTE_TILE_WIDTH", "0")
    with pytest.raises(ValueError, match="from 1 through 1024"):
        r._rte_gpt_tile(None, 3125, 256, fold=True, longwave=True)


@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rte_measurement_overrides_refuse_unsafe_groups(kind, monkeypatch):
    import cupy as cp
    import woof.core.rrtmgp as r

    tau = cp.full((3, 4, 16), 0.1, dtype=r.DTYPE)
    surface = cp.full((3, 16), 0.8, dtype=r.DTYPE)
    lev = cp.full((3, 5, 16), 10, dtype=r.DTYPE)

    def run():
        if kind == "lw":
            return r._lw_rte(tau, tau, lev, surface, surface,
                             top_at_1=False, out=None, incident_out=None)
        return r._sw_rte(tau, tau, tau, cp.ones(3, dtype=r.DTYPE),
                         surface, surface, surface, top_at_1=False, out=None)

    monkeypatch.setenv("WOOF_RTE_TILE_WIDTH", "8")
    monkeypatch.setenv("WOOF_RTE_FOLD", "1")
    with pytest.raises(ValueError, match="full tile of at least 16"):
        run()
    monkeypatch.setenv("WOOF_RTE_TILE_WIDTH", "16")
    monkeypatch.setenv("WOOF_RTE_COLUMNS_PER_BLOCK", "0")
    with pytest.raises(ValueError, match="must be positive"):
        run()
    monkeypatch.delenv("WOOF_RTE_COLUMNS_PER_BLOCK")
    monkeypatch.setenv("WOOF_RTE_FOLD", "yes")
    with pytest.raises(ValueError, match="must be auto, 0 or 1"):
        run()


def test_rte_tile_coverage_override_is_measurement_only(monkeypatch):
    """The override exists so the two settings can be INTERLEAVED in one
    session.  Comparing a block of runs against a block taken later gave a
    wrong answer here: the control moved 7.7% between the groups and the
    normaliser does not survive a regime change (HOWTO 13.5e)."""
    pytest.importorskip("cupy")
    from woof.core.rrtmgp import _rte_gpt_tile, _rte_kernel

    lw = _rte_kernel("rrtmgp_lw_noscat", 74)
    default = _rte_gpt_tile(lw, 1024, 256)
    monkeypatch.setenv("WOOF_RTE_TILE_COVERAGE", "1")
    assert _rte_gpt_tile(lw, 1024, 256) == default // 2
    monkeypatch.delenv("WOOF_RTE_TILE_COVERAGE")
    assert _rte_gpt_tile(lw, 1024, 256) == default


def test_chunk_prep_scratch_reuses_buffers_only_when_asked():
    """`_prepare_above_model_chunk` allocated ~15 fresh arrays per call,
    2364 times an hour, and profiles at 2.78 s of HOST against 0.98 s of
    device.  `scratch=True` hands back views of reused buffers instead.

    That is a CONTRACT, not just an optimisation: the caller must finish
    with a chunk before asking for the next of the same kind and shape.
    Only the driver's chunk loop opts in.  Both halves are asserted here --
    reuse when asked, independence when not -- because a caller that
    collects chunks and compares them afterwards is exactly what this
    would silently corrupt, and one already exists in this file.
    """
    cp = pytest.importorskip("cupy")
    from woof.core.rrtmgp import (DTYPE, HydrometeorPaths,
                                   _prepare_above_model_chunk,
                                   load_gas_tables)

    ncol, nz = 64, 20
    edge = np.linspace(101325.0, 10000.0, nz + 1)
    kw = dict(
        tables=load_gas_tables("lw"),
        play=cp.asarray(np.tile(0.5 * (edge[:-1] + edge[1:]), (ncol, 1)),
                        dtype=DTYPE),
        plev=cp.asarray(np.tile(edge, (ncol, 1)), dtype=DTYPE),
        tlay=cp.asarray(np.full((ncol, nz), 250.0), dtype=DTYPE),
        tlev=cp.asarray(np.full((ncol, nz + 1), 250.0), dtype=DTYPE),
        qv=cp.asarray(np.full((ncol, nz), 1.0e-3), dtype=DTYPE),
        cldfra=cp.asarray(np.full((ncol, nz), 0.25), dtype=DTYPE),
        paths=HydrometeorPaths(*(cp.full((ncol, nz), DTYPE(1.0e-3),
                                         dtype=DTYPE) for _ in range(4))),
        p_top=10000.0, kind="lw", xp=cp, validate_top=False, validate=False)

    lo, hi = slice(0, 32), slice(32, 64)
    a = _prepare_above_model_chunk(columns=lo, scratch=True, **kw)
    b = _prepare_above_model_chunk(columns=hi, scratch=True, **kw)
    assert a.cldfra.data.ptr == b.cldfra.data.ptr
    assert a.paths.clwp.data.ptr == b.paths.clwp.data.ptr
    assert a.metadata.jt.data.ptr == b.metadata.jt.data.ptr
    # Five live-at-once clear-layer buffers must never share bytes.
    ptrs = {v.data.ptr for v in (b.paths.clwp, b.paths.ciwp, b.paths.reliq,
                                 b.paths.dgice, b.cldfra)}
    assert len(ptrs) == 5

    c = _prepare_above_model_chunk(columns=lo, scratch=False, **kw)
    d = _prepare_above_model_chunk(columns=hi, scratch=False, **kw)
    assert c.cldfra.data.ptr != d.cldfra.data.ptr
    assert c.metadata.jt.data.ptr != d.metadata.jt.data.ptr

    # Reuse must not change a single bit, and the clear tail the first call
    # zeroed must still be zero after the buffer has been recycled.
    for field in ("play", "plev", "tlay", "tlev", "qv"):
        assert bool(cp.all(getattr(b.profile, field).view(cp.uint32)
                           == getattr(d.profile, field).view(cp.uint32)))
    assert bool(cp.all(b.cldfra.view(cp.uint32) == d.cldfra.view(cp.uint32)))
    assert bool(cp.all(b.cldfra[:, nz:] == DTYPE(0.0)))
    assert bool(cp.all(b.paths.clwp[:, nz:] == DTYPE(0.0)))


def test_gas_vmr_fused_kernel_matches_the_expression_reference():
    """`_gas_vmr` builds the whole VMR cube in one kernel; the expression
    sequence it replaced survives as `_gas_vmr_expressions`, and the two
    must agree BITWISE -- this feeds the gas optics, so a last-bit drift is
    a physics change.

    The ozone slot is the reason this test exists.  `cupy.interp` returns
    float64 even from float32 tables, and the CuPy path narrowed it on the
    store; the kernel therefore takes a double and narrows on its own
    store.  Read as float32 instead, every cell of that one slot is
    garbage and nothing else moves -- which is exactly what this caught.
    """
    cp = pytest.importorskip("cupy")
    from woof.core.rrtmgp import RRTMGPRadiation, load_gas_tables

    adapter = object.__new__(RRTMGPRadiation)
    pressure = np.array([0.5, 1.0, 5.0, 20.0, 1.0e2, 3.0e2, 7.0e2, 1013.0])
    pressure = pressure * 100.0
    ozone = np.array([2e-6, 4e-6, 8e-6, 6e-6, 1e-6, 8e-8, 5e-8, 3e-8])
    order = np.argsort(pressure)
    adapter._ozone_logp = cp.asarray(np.log(pressure[order]), dtype=np.float32)
    adapter._ozone_vmr = cp.asarray(ozone[order], dtype=np.float32)

    rng = np.random.default_rng(7)
    checked = 0
    for kind in ("lw", "sw"):
        tables = load_gas_tables(kind)
        for trace in ({"co2": 4.2e-4},
                      {"co2": 4.2e-4, "ch4": 1.9e-6, "n2o": 3.3e-7},
                      {"h2o": 1.25e-3, "o3": 2.5e-7},
                      {}):
            adapter.trace_vmr = trace
            for ncol, nlay in ((1024, 74), (377, 50), (1, 4)):
                play = cp.asarray(
                    np.exp(rng.uniform(np.log(2.0), np.log(1.0e5),
                                       (ncol, nlay))), dtype=np.float32)
                qv = cp.asarray(rng.uniform(0.0, 3.0e-2, (ncol, nlay)),
                                dtype=np.float32)
                want = adapter._gas_vmr_expressions(
                    cp, tables, play, qv, validate=False, out=None)
                # A DIRTY workspace buffer, so a slot the kernel forgets to
                # write shows up instead of reading back a helpful zero.
                dirty = cp.full((ncol, nlay, tables.ngas + 1),
                                np.float32(-7.5), dtype=np.float32)
                got = adapter._gas_vmr(tables, play, qv, validate=False,
                                       out=dirty)
                assert bool(cp.all(want.view(cp.uint32)
                                   == got.view(cp.uint32))), (
                    f"{kind} trace={len(trace)} {ncol}x{nlay} differs")
                checked += 1
    assert checked == 24


def test_above_model_gas_and_ozone_fill_every_appended_layer(monkeypatch):
    """The existing RFMIP climatology is evaluated on, not below, the cap."""
    import sys
    from types import SimpleNamespace
    from woof.core.rrtmgp import RRTMGPRadiation

    monkeypatch.setitem(sys.modules, "cupy", np)
    radiation = object.__new__(RRTMGPRadiation)
    radiation._ozone_logp = np.log(
        np.array([1.0, 100.0, 10000.0, 100000.0], np.float32))
    radiation._ozone_vmr = np.array(
        [5.0e-7, 7.0e-6, 2.0e-7, 4.0e-8], np.float32)
    radiation.trace_vmr = {"co2": 330.0e-6}
    tables = SimpleNamespace(
        ngas=3, gas_index={"h2o": 1, "o3": 2, "co2": 3})
    play = np.array([[80000.0, 10000.0, 5000.0, 200.0]], np.float32)
    qv = np.array([[8.0e-3, 1.0e-5, 1.0e-5, 1.0e-5]], np.float32)

    vmr = radiation._gas_vmr(tables, play, qv)
    assert vmr.shape == (1, 4, 4)
    np.testing.assert_array_equal(vmr[0, :, 3], np.float32(330.0e-6))
    np.testing.assert_array_equal(
        vmr[0, :, 1], qv[0] * np.float32(0.028964 / 0.018016))
    expected_o3 = np.interp(
        np.log(play[0]), radiation._ozone_logp, radiation._ozone_vmr)
    np.testing.assert_array_equal(vmr[0, :, 2], expected_o3.astype(np.float32))


def test_column_driver_routes_extended_profiles_through_every_solver_stage():
    """Static guard: no cap may be constructed and then bypassed downstream."""
    import inspect
    from woof.core.rrtmgp import RRTMGPRadiation

    source = inspect.getsource(RRTMGPRadiation.__call__)
    assert source.count("_prepare_above_model_chunk(") == 2
    assert source.count("columns=sl") == 2
    assert "_extend_above_model_profile(" not in source
    assert "_interpolation_metadata(" not in source
    for name in ("lw", "sw"):
        assert f'kind="{name}"' in source
        for field in ("play", "plev", "tlay", "qv"):
            assert source.count(f"{name}_profile.{field}") >= 2
        assert f"{name}_profile.play, {name}_cldfra" in source
        assert f"{name}_paths.clwp" in source
    assert "lw_profile.tlev, tsfc[sl]" in source
    # LW, SW over full chunks (daylight merge), SW over compacted daylit
    # columns (scattered back; no merge needed).
    assert source.count("_store_model_flux_pair(") == 3
    # The surface direct beam BEP+BEM reads (sf_urban_physics = 3), through
    # the same capped interfaces.
    assert source.count("_model_flux_interfaces(") == 1
    assert "flux.flux_dir, nz, xp=cp)" in source


def test_real74_model_top_downward_flux_cpu_reference(monkeypatch):
    """Independent float64 RRTMGP mirror pins the 100-hPa downward flux."""
    import sys
    from datetime import datetime
    from woof.core import rrtmgp
    from woof.verify.npref import (
        np_rrtmgp_delta_scale, np_rrtmgp_gas_optics,
        np_rrtmgp_lw_rte, np_rrtmgp_planck_sources, np_rrtmgp_sw_rte)

    monkeypatch.setitem(sys.modules, "cupy", np)
    nz = 49
    plev = np.geomspace(100000.0, 10000.0, nz + 1)[None].astype(np.float32)
    play = np.sqrt(plev[:, :-1] * plev[:, 1:]).astype(np.float32)
    tlay = np.linspace(292.0, 215.0, nz, dtype=np.float32)[None]
    # Independent NumPy transcription of the pinned interface-temperature
    # equations; the above-model construction itself is separately checked
    # against literal WRF pressure/temperature tables above.
    tlev = np.empty((1, nz + 1), np.float32)
    tlev[:, 0] = tlay[:, 0] + (plev[:, 0] - play[:, 0]) * (
        tlay[:, 1] - tlay[:, 0]) / (play[:, 1] - play[:, 0])
    tlev[:, -1] = tlay[:, -1] + (plev[:, -1] - play[:, -1]) * (
        tlay[:, -1] - tlay[:, -2]) / (play[:, -1] - play[:, -2])
    tlev[:, 1:-1] = (
        play[:, :-1] * tlay[:, :-1] * (plev[:, 1:-1] - play[:, 1:])
        + play[:, 1:] * tlay[:, 1:] * (play[:, :-1] - plev[:, 1:-1])
    ) / (plev[:, 1:-1] * (play[:, :-1] - play[:, 1:]))
    qv = np.geomspace(8.0e-3, 2.0e-6, nz).astype(np.float32)[None]

    radiation = object.__new__(rrtmgp.RRTMGPRadiation)
    climatology = rrtmgp.load_trace_climatology()
    radiation.trace_vmr = dict(climatology.trace_vmr)
    pressure = climatology.pressure_layer_pa
    ozone = climatology.ozone_vmr
    radiation.trace_vmr.update(rrtmgp.trace_gases(
        datetime(1974, 4, 3), {"co2": 330.0e-6}))
    order = np.argsort(pressure)
    radiation._ozone_logp = np.log(pressure[order]).astype(np.float32)
    radiation._ozone_vmr = ozone[order].astype(np.float32)

    lw = rrtmgp._extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=10000.0, kind="lw", xp=np)
    lw_tables = rrtmgp.load_gas_tables("lw")
    lw_vmr = radiation._gas_vmr(lw_tables, lw.play, lw.qv)
    lw_gas = np_rrtmgp_gas_optics(
        lw_tables, lw.play, lw.plev, lw.tlay, lw_vmr)
    lw_source = np_rrtmgp_planck_sources(
        lw_tables, lw.play, lw.plev, lw.tlay, lw.tlev,
        np.array([288.0]), lw_vmr)
    with np.errstate(invalid="ignore", divide="ignore"):
        lw_flux = np_rrtmgp_lw_rte(
            lw_gas.tau, lw_source.lay_source, lw_source.lev_source,
            lw_source.sfc_source, np.full((1, lw_tables.ngpt), 0.96),
            top_at_1=False)

    sw = rrtmgp._extend_above_model_profile(
        play, plev, tlay, tlev, qv, p_top=10000.0, kind="sw", xp=np)
    sw_tables = rrtmgp.load_gas_tables("sw")
    sw_vmr = radiation._gas_vmr(sw_tables, sw.play, sw.qv)
    sw_gas = np_rrtmgp_gas_optics(
        sw_tables, sw.play, sw.plev, sw.tlay, sw_vmr)
    tau, ssa, asym = np_rrtmgp_delta_scale(
        sw_gas.tau, sw_gas.ssa, sw_gas.g)
    # np.array (copy=True): solar_source is the lru-cached GasTables
    # array and is already float64, so np.asarray would alias it and the
    # in-place scale below would poison every later test in the process
    # (surfaced as an order-dependent zenith/solcon failure once the
    # legacy-RRTMG test files changed collection order).
    solar = np.array(sw_tables.solar_source, np.float64)
    solar *= (rrtmgp.RRTMGPRadiation._solar_constant(
        datetime(1974, 4, 3, 18)) / np.sum(solar))
    sw_flux = np_rrtmgp_sw_rte(
        tau, ssa, asym, np.array([0.5]),
        np.full((1, sw_tables.ngpt), 0.18),
        np.full((1, sw_tables.ngpt), 0.18), solar[None], top_at_1=False)

    # Interface 49 is the retained model top; interfaces 74/50 are TOA.
    assert lw_flux.flux_dn[0, -1] == 0.0
    assert lw_flux.flux_dn[0, 49] == pytest.approx(
        14.954168109470562, rel=2.0e-13)
    # Re-pinned from 684.8520566535782 (TOA) and 659.7770259926568 (100 hPa):
    # SOLCON is WRF's REAL(4) radconst word (woof.core.rrtmg_legacy.radconst,
    # the device kernel's own float32 operation order) since the WRF 4.8
    # solar generation became the default (7c936f94a0a); the float64 formula
    # it replaced sat 8.6e-9 relative above the word WRF computes.  The SW
    # solve is linear in the solar source, so both fluxes scale by that
    # ratio; the longwave is untouched.
    assert sw_flux.flux_dn[0, -1] == pytest.approx(
        684.8520507812499, rel=2.0e-13)
    # Re-pinned by development branch from 659.7770262288319: the SW
    # solar source table is now set_solar_variability evaluated in float32
    # (as the single-precision reference does), which moves 64 of its 224
    # g-point weights by 1 ULP; the spectral redistribution moves the
    # 100-hPa flux by 2.4e-7 W m-2 (3.6e-10 relative).  TOA is unchanged.
    assert sw_flux.flux_dn[0, 49] == pytest.approx(
        659.7770205715113, rel=2.0e-13)


@pytest.mark.gpu
@requires_gpu
def test_real74_top_extension_reaches_gpu_optics_mcica_and_flux_mapping(
        monkeypatch):
    """One-column integration guard for the exact 49+25 / 49+1 extents."""
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.model import SharedRRTMGPChunkWorkspace
    from woof.core import rrtmgp

    nz, ny, nx = 49, 1, 1
    plev_col = np.geomspace(100000.0, 10000.0, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(292.0, 215.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)
    def expand(value):
        return cp.asarray(
            np.broadcast_to(value[:, None, None], shape).copy(), cp.float32)
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(
            plev_col[:, None, None], dtype=cp.float32),
        "temperature": expand(t_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 2.0e-6, nz)),
        "qc": cp.zeros(shape, cp.float32),
        "qi": cp.zeros(shape, cp.float32),
    }
    fields = {
        "tsk": cp.full((1, 1), 288.0, cp.float32),
        "albedo": cp.full((1, 1), 0.18, cp.float32),
        "emiss": cp.full((1, 1), 0.96, cp.float32),
    }
    state = SimpleNamespace(
        elapsed_seconds=0.0, p_top=10000.0,
        qc=atmosphere["qc"], qr=cp.zeros(shape, cp.float32))
    cfg = SimpleNamespace(
        mp_physics=1, dt=60.0, radt=12.0, radt_minutes=12.0)
    radiation = rrtmgp.RRTMGPRadiation(
        datetime(1974, 4, 3, 18), cp.asarray([[40.0]]),
        cp.asarray([[-100.0]]), column_chunk=1,
        trace_gas_overrides={"co2": 330.0e-6})
    radiation.chunk_workspace = SharedRRTMGPChunkWorkspace(
        nz=49, column_chunk=1, p_top=10000.0)

    observed = {"gas": [], "mcica": [], "mapped": None}
    original_gas = rrtmgp._gas_optics
    original_mcica = rrtmgp._mcica_cloud_masks
    original_mapping = rrtmgp._fluxes_to_radiation

    def gas_spy(tables, play, *args, **kwargs):
        observed["gas"].append((tables.kind, tuple(play.shape)))
        return original_gas(tables, play, *args, **kwargs)

    def mcica_spy(play, *args, **kwargs):
        observed["mcica"].append(tuple(play.shape))
        return original_mcica(play, *args, **kwargs)

    def mapping_spy(lw_up, lw_dn, sw_up, sw_dn, *args, **kwargs):
        observed["mapped"] = tuple(
            tuple(value.shape) for value in (lw_up, lw_dn, sw_up, sw_dn))
        return original_mapping(
            lw_up, lw_dn, sw_up, sw_dn, *args, **kwargs)

    monkeypatch.setattr(rrtmgp, "_gas_optics", gas_spy)
    monkeypatch.setattr(rrtmgp, "_mcica_cloud_masks", mcica_spy)
    monkeypatch.setattr(rrtmgp, "_fluxes_to_radiation", mapping_spy)
    result = radiation(
        atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)

    assert observed["gas"] == [("lw", (1, 74)), ("sw", (1, 50))]
    assert observed["mcica"] == [(1, 74), (1, 50)]
    assert observed["mapped"] == ((1, 50),) * 4
    assert result.rthratenlw.shape == result.rthratensw.shape == shape
    assert bool(cp.isfinite(result.rthratenlw).all())
    assert bool(cp.isfinite(result.rthratensw).all())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("entrypoint", ["gas_optics", "planck_sources"])
def test_rrtmgp_rejects_temperature_below_table_before_kernel(entrypoint):
    import cupy as cp
    from woof.core.rrtmgp import gas_optics, load_gas_tables, planck_sources

    tables = load_gas_tables("lw")
    play = cp.asarray([[90000.0, 50000.0]], dtype=cp.float32)
    plev = cp.asarray([[100000.0, 70000.0, 30000.0]], dtype=cp.float32)
    tlay = cp.asarray([[tables.temp_ref.min() - 1.0, 250.0]],
                      dtype=cp.float32)
    vmr = cp.zeros((1, 2, tables.ngas + 1), dtype=cp.float32)
    with pytest.raises(ValueError, match=r"tlay range .* K .*allowed range"):
        if entrypoint == "gas_optics":
            gas_optics(tables, play, plev, tlay, vmr)
        else:
            planck_sources(
                tables, play, plev, tlay,
                cp.asarray([[280.0, 240.0, 200.0]], dtype=cp.float32),
                cp.asarray([285.0], dtype=cp.float32), vmr)


@pytest.mark.gpu
@requires_gpu
def test_planck_sources_refuses_a_column_taller_than_its_kernel_holds():
    """The 128-layer ceiling's OWNER refuses loudly, not silently.

    The RTE solvers compile ``RRTMGP_MAX_LAYERS`` to this run's nlay, so
    the ceiling is NOT theirs; it belongs to the unspecialized
    Planck-source kernel, whose per-thread ``pfrac[128]`` is fixed and
    whose device-side guard is a bare ``return`` -- with no host raise,
    a 129-layer call came back as uninitialized source arrays that read
    as valid radiation (stale-guard audit 2026-08-25, RRTMGP nlay
    adjudication).
    """
    import cupy as cp
    from woof.core.rrtmgp import load_gas_tables, planck_sources

    tables = load_gas_tables("lw")
    ncol, nlay = 1, 129
    play = cp.full((ncol, nlay), 50000.0, dtype=cp.float32)
    plev = cp.full((ncol, nlay + 1), 50000.0, dtype=cp.float32)
    tlay = cp.full((ncol, nlay), 250.0, dtype=cp.float32)
    tlev = cp.full((ncol, nlay + 1), 250.0, dtype=cp.float32)
    tsfc = cp.asarray([285.0], dtype=cp.float32)
    vmr = cp.zeros((ncol, nlay, tables.ngas + 1), dtype=cp.float32)
    with pytest.raises(ValueError, match=r"128 layers.*pfrac"):
        planck_sources(tables, play, plev, tlay, tlev, tsfc, vmr)


@pytest.mark.gpu
@requires_gpu
def test_sw_rte_runs_a_column_taller_than_the_retired_ceiling():
    """The SW solver has no 128-layer ceiling left to enforce.

    The 128 ceiling belongs to the Planck-source kernel's fixed
    ``pfrac[128]`` -- an LW-only stage.  The SW chain's kernels either
    compile ``RRTMGP_MAX_LAYERS`` to this run's nlay (the two-stream
    solver) or hold no per-layer thread storage at all (delta scaling,
    gas optics, cloud optics, McICA), so refusing a 129+ layer SW column
    prevented nothing (stale-guard audit 2026-08-25).  Proven on a
    transparent 160-layer atmosphere, where the direct beam is the
    analytic answer.
    """
    import cupy as cp
    from woof.core.rrtmgp import sw_rte

    ncol, nlay, ngpt = 2, 160, 4
    inc = 1000.0
    tau = cp.full((ncol, nlay, ngpt), 1e-6, dtype=cp.float32)
    ssa = cp.zeros((ncol, nlay, ngpt), dtype=cp.float32)
    g = cp.zeros((ncol, nlay, ngpt), dtype=cp.float32)
    mu0 = cp.ones((ncol,), dtype=cp.float32)
    alb = cp.zeros((ncol, ngpt), dtype=cp.float32)
    result = sw_rte(tau, ssa, g, mu0, alb, alb,
                    cp.full((ncol, ngpt), inc, dtype=cp.float32),
                    top_at_1=True)
    down = cp.asnumpy(result.flux_dn)
    up = cp.asnumpy(result.flux_up)
    assert down.shape == (ncol, nlay + 1)
    # A transparent atmosphere delivers the incident beam to the surface
    # (ngpt g-points of `inc` each) and reflects nothing off a black one.
    expected = ngpt * inc
    assert abs(float(down[:, -1].min()) - expected) < 1e-3 * expected
    assert float(abs(up).max()) < 1e-3 * expected


def test_rrtmgp_production_validation_mode_keeps_public_entrypoints_stable():
    from inspect import signature
    from woof.core.rrtmgp import (
        RRTMGPRadiation, _validation_error_messages, gas_optics,
        planck_sources,
    )

    assert signature(RRTMGPRadiation).parameters["validation_mode"].default \
        == "fused"
    assert tuple(signature(gas_optics).parameters) \
        == ("tables", "play", "plev", "tlay", "vmr")
    assert tuple(signature(planck_sources).parameters) == (
        "tables", "play", "plev", "tlay", "tlev", "tsfc", "vmr")
    messages = _validation_error_messages((1 << 0) | (1 << 5) | (1 << 19))
    assert len(messages) == 3
    assert messages[0].startswith("play ")
    assert messages[1].startswith("qv ")
    assert messages[2].startswith("surface emissivity ")


@pytest.mark.parametrize(
    ("bad_field", "match"),
    [("exner", "exner must have shape"),
     ("tsk", "tsfc must have 6 surface values")],
)
def test_rrtmgp_malformed_exner_and_tsfc_fail_before_device_packing(
        bad_field, match):
    from woof.core.rrtmgp import RRTMGPRadiation

    radiation = object.__new__(RRTMGPRadiation)
    radiation.validation_mode = "fused"
    radiation.latitude_deg = np.zeros((2, 3))
    called = []
    radiation._columns = lambda value: called.append(value)
    shape = (4, 2, 3)
    atmosphere = {
        "pressure": np.empty(shape),
        "exner": np.empty((3, 2, 3) if bad_field == "exner" else shape),
    }
    fields = {"tsk": np.empty(5 if bad_field == "tsk" else (2, 3))}
    with pytest.raises(ValueError, match=match):
        radiation(atmosphere=atmosphere, fields=fields, state=None, cfg=None)
    assert called == []


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("invalid_field", ["qc", "play"])
def test_rrtmgp_fused_and_full_invalid_inputs_keep_legacy_diagnostics(
        invalid_field):
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    nz = 4
    plev = np.array([100000.0, 80000.0, 60000.0, 40000.0, 20000.0])
    play = np.array([90000.0, 70000.0, 50000.0, 30000.0])
    if invalid_field == "play":
        play[0] = 120000.0
    temp = np.array([290.0, 270.0, 250.0, 230.0])
    shape = (nz, 1, 1)
    def column(value):
        return cp.asarray(
            np.asarray(value).reshape(nz, 1, 1), dtype=cp.float32)
    qc = cp.zeros(shape, cp.float32)
    if invalid_field == "qc":
        qc[0, 0, 0] = -1.0e-6
    atmosphere = {
        "pressure": column(play),
        "p_interface": cp.asarray(plev.reshape(nz + 1, 1, 1), cp.float32),
        "temperature": column(temp),
        "exner": column((play / 100000.0) ** (287.0 / 1004.0)),
        "qv": column([0.01, 0.005, 0.001, 1.0e-4]),
        "qc": qc,
        "qi": cp.zeros(shape, cp.float32),
    }
    fields = {
        "tsk": cp.full((1, 1), 288.0, cp.float32),
        "albedo": cp.full((1, 1), 0.18, cp.float32),
        "emiss": cp.full((1, 1), 0.96, cp.float32),
    }
    state = SimpleNamespace(elapsed_seconds=0.0, qc=qc,
                            qr=cp.zeros(shape, cp.float32))
    cfg = SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0,
                          radt_minutes=12.0)
    messages = []
    for mode in ("full", "fused"):
        radiation = RRTMGPRadiation(
            datetime(1974, 4, 3, 18), cp.zeros((1, 1)), cp.zeros((1, 1)),
            validation_mode=mode,
            trace_gas_overrides={"co2": 330.0e-6})
        with pytest.raises(ValueError) as excinfo:
            radiation(atmosphere=atmosphere, fields=fields,
                      state=state, cfg=cfg)
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1]
    if invalid_field == "qc":
        assert messages[0].startswith("qc must be finite and non-negative: ")
        assert "first_index=(0, 0)" in messages[0]
        assert "negative_count=" in messages[0]
        assert "nonfinite_count=" in messages[0]
    else:
        assert "play range [" in messages[0]
        assert " Pa is outside allowed range [" in messages[0]


def _validation_inputs(nband=16):
    cell = (1, 4)
    return {
        "play": np.array([[90000.0, 70000.0, 50000.0, 30000.0]],
                         np.float32),
        "plev": np.array([[100000.0, 80000.0, 60000.0, 40000.0, 20000.0]],
                         np.float32),
        "tlay": np.full(cell, 270.0, np.float32),
        "tlev": np.full((1, 5), 270.0, np.float32),
        "tsfc": np.full((1,), 285.0, np.float32),
        "exner": np.ones(cell, np.float32),
        "qv": np.zeros(cell, np.float32),
        "qc": np.zeros(cell, np.float32),
        "qr": np.zeros(cell, np.float32),
        "qi": np.zeros(cell, np.float32),
        "qs": np.zeros(cell, np.float32),
        "cldfra": np.zeros(cell, np.float32),
        "numbers": {name: np.ones(cell, np.float32)
                    for name in ("nc", "nr", "ni", "ns")},
        "effective": {name: np.ones(cell, np.float32)
                      for name in ("effc", "effr", "effi", "effs")},
        "emiss": np.ones((1, nband), np.float32),
    }


def test_rrtmgp_fused_validation_rejects_single_layer_before_kernel_lookup(
        monkeypatch):
    import woof.core.kernels as kernels
    from woof.core.rrtmgp import _validation_flags_device

    called = []
    monkeypatch.setattr(
        kernels, "get_kernel", lambda *args: called.append(args))
    values = _validation_inputs()
    for name in ("play", "tlay", "exner", "qv", "qc", "qr", "qi", "qs",
                 "cldfra"):
        values[name] = values[name][:, :1]
    for fields in (values["numbers"], values["effective"]):
        for name in fields:
            fields[name] = fields[name][:, :1]
    values["plev"] = values["plev"][:, :2]
    values["tlev"] = values["tlev"][:, :2]

    with pytest.raises(ValueError, match="at least one column and two layers"):
        _validation_flags_device(
            **values, tables_lw=None, tables_sw=None)
    assert called == []


def test_rrtmgp_legacy_diagnostic_replay_preserves_chunk_order_and_extrema(
        monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.core.rrtmgp import _raise_full_call_validation_error

    fake_cupy = SimpleNamespace(
        abs=np.abs, any=np.any, asnumpy=np.asarray, isfinite=np.isfinite)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    ncol, nlay = 4, 4
    cell = (ncol, nlay)
    play = np.tile(
        np.array([90000.0, 70000.0, 50000.0, 30000.0], np.float32),
        (ncol, 1))
    play[0, 0] = 120000.0
    play[2:, -1] = 10000.0
    qv = np.zeros(cell, np.float32)
    qv[2, 0] = -1.0
    plev = np.tile(
        np.array([100000.0, 80000.0, 60000.0, 40000.0, 20000.0],
                 np.float32), (ncol, 1))
    tables = SimpleNamespace(
        press_ref=np.array([1.0, 110000.0]),
        temp_ref=np.array([160.0, 355.0]))
    zeros = np.zeros(cell, np.float32)

    with pytest.raises(ValueError) as excinfo:
        _raise_full_call_validation_error(
            play=play, plev=plev, tlay=np.full(cell, 270.0, np.float32),
            tlev=np.full((ncol, nlay + 1), 270.0, np.float32),
            tsfc=np.full(ncol, 285.0, np.float32),
            exner=np.ones(cell, np.float32), qv=qv,
            qc=zeros, qr=zeros, qi=zeros, qs=zeros,
            cldfra=zeros, emiss=np.ones((ncol, 16), np.float32),
            numbers={}, effective={}, tables_lw=tables,
            tables_sw=tables, column_chunk=2)
    assert str(excinfo.value) == (
        "play range [30000, 120000] Pa is outside allowed range "
        "[1, 110000] Pa")


def _invalidate_validation_bit(values, bit):
    fields = ("play", "plev", "tlay", "tlev", "tsfc", "qv", "qc",
              "qr", "qi", "qs", "cldfra")
    if bit < len(fields):
        values[fields[bit]].flat[0] = np.nan if bit < 5 else -1.0
    elif bit < 15:
        values["numbers"][("nc", "nr", "ni", "ns")[bit - 11]].flat[0] = -1
    elif bit < 19:
        # NaN violates the finite/non-negative predicate without also
        # tripping the plausibility band (NaN comparisons are false), so
        # bits 15-18 stay single-bit next to the band bits 22-24.
        values["effective"][("effc", "effr", "effi", "effs")[bit - 15]] \
            .flat[0] = np.nan
    elif bit == 19:
        values["emiss"].flat[0] = -1
    elif bit == 20:
        values["play"][0, 0] = 60000.0
    elif bit == 21:
        values["exner"].flat[0] = 0.0
    elif bit == 22:
        # Metre-scale liquid radius: finite and positive, out of unit.
        values["effective"]["effc"].flat[0] = 2.49e-6
    elif bit == 23:
        values["effective"]["effi"].flat[0] = 4.99e-6
    else:
        # Nanometre-scale snow radius: finite and positive, out of unit.
        values["effective"]["effs"].flat[0] = 9990.0


def _validation_flags_oracle(values, play_bounds=(1.0, 110000.0),
                             temp_bounds=(160.0, 355.0)):
    flags = 0

    def outside(value, bounds):
        return (np.any(~np.isfinite(value)) or np.any(value < bounds[0])
                or np.any(value > bounds[1]))

    if outside(values["play"], play_bounds):
        flags |= 1 << 0
    if np.any(~np.isfinite(values["plev"])) or np.any(values["plev"] < 0):
        flags |= 1 << 1
    for bit, name in ((2, "tlay"), (3, "tlev"), (4, "tsfc")):
        if outside(values[name], temp_bounds):
            flags |= 1 << bit
    for bit, name in enumerate(("qv", "qc", "qr", "qi", "qs"), 5):
        value = values[name]
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            flags |= 1 << bit
    if outside(values["cldfra"], (0.0, 1.0)):
        flags |= 1 << 10
    for offset, group in ((11, values["numbers"]),
                          (15, values["effective"])):
        for bit, value in enumerate(group.values(), offset):
            if np.any(~np.isfinite(value)) or np.any(value < 0):
                flags |= 1 << bit
    if outside(values["emiss"], (0.0, 1.0)):
        flags |= 1 << 19
    if np.any(values["play"][:, 0] < values["play"][:, 1]):
        flags |= 1 << 20
    dp = np.abs(np.diff(values["plev"], axis=1))
    if np.any(dp <= 0) or np.any(values["exner"] <= 0):
        flags |= 1 << 21
    # Independent restatement of EFFECTIVE_RADIUS_PLAUSIBLE_UM: the micron
    # contract's physical bands (effr is interface parity, not gated).
    for bit, name, bounds in ((22, "effc", (0.5, 100.0)),
                              (23, "effi", (1.0, 600.0)),
                              (24, "effs", (1.0, 5000.0))):
        value = values["effective"][name]
        if np.any(value < bounds[0]) or np.any(value > bounds[1]):
            flags |= 1 << bit
    return flags


def test_rrtmgp_validation_cpu_oracle_discriminates_all_25_predicates():
    from woof.core.rrtmgp import _VALIDATION_MESSAGES

    assert len(_VALIDATION_MESSAGES) == 25
    for bit in range(25):
        values = _validation_inputs()
        _invalidate_validation_bit(values, bit)
        assert _validation_flags_oracle(values) == 1 << bit


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_fused_validation_matches_independent_predicate_oracle():
    import cupy as cp
    from woof.core.rrtmgp import (
        _validation_flags_device, load_gas_tables,
    )

    lw = load_gas_tables("lw")
    sw = load_gas_tables("sw")
    play_bounds = (
        max(float(np.min(lw.press_ref)), float(np.min(sw.press_ref))),
        min(float(np.max(lw.press_ref)), float(np.max(sw.press_ref))),
    )
    temp_bounds = (
        max(float(np.min(lw.temp_ref)), float(np.min(sw.temp_ref))),
        min(float(np.max(lw.temp_ref)), float(np.max(sw.temp_ref))),
    )
    for bit in range(25):
        host = _validation_inputs(lw.nband)
        _invalidate_validation_bit(host, bit)
        expected = _validation_flags_oracle(host, play_bounds, temp_bounds)
        device = {
            name: cp.asarray(value)
            for name, value in host.items()
            if name not in ("numbers", "effective")
        }
        device["numbers"] = {
            name: cp.asarray(value) for name, value in host["numbers"].items()}
        device["effective"] = {
            name: cp.asarray(value)
            for name, value in host["effective"].items()}
        observed = _validation_flags_device(
            **device, tables_lw=lw, tables_sw=sw)
        assert observed == expected == 1 << bit


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_gas_optics_mirror_rfmip_spot_columns(kind):
    from woof.core.rrtmgp import load_gas_tables
    from woof.verify.npref import np_rrtmgp_gas_optics

    tables = load_gas_tables(kind)
    play, plev, tlay, vmr = _rfmip_columns(tables)
    out = np_rrtmgp_gas_optics(tables, play, plev, tlay, vmr)
    assert out.tau.shape == (2, 60, tables.ngpt)
    assert np.all(np.isfinite(out.tau)) and np.all(out.tau >= 0.0)
    if kind == "lw":
        assert out.ssa is None and out.g is None
    else:
        assert np.all(np.isfinite(out.ssa))
        assert np.all((out.ssa >= 0.0) & (out.ssa <= 1.0))
        assert np.all(out.g == 0.0)
        assert np.any(out.ssa > 0.0)
    # Transcription-freeze change detectors, not independent correctness
    # oracles: the same pipeline is separately gated end-to-end against the
    # module's hashed upstream RFMIP reference flux files.
    sample = out.tau[[0, 0, 1, 1], [0, 59, 12, 47],
                     [0, tables.ngpt - 1, tables.ngpt // 3,
                      2 * tables.ngpt // 3]]
    expected = {
        "lw": np.array([1.7831777766687745e-08, 5.4327605236965733,
                        7.1027085950978617e-06, 1.9392695176931772e02]),
        "sw": np.array([3.0817831625259552e-08, 1.1499826797030600e-01,
                        9.5488193518374828e-03, 1.4833544139222820e-03]),
    }[kind]
    np.testing.assert_allclose(sample, expected, rtol=2e-9, atol=1e-13)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_minor_interval_first_last_gpoints_align_with_raw_start():
    """Both interval edges use raw one-based kminor_start minus exactly one."""
    import cupy as cp
    from netCDF4 import Dataset
    from woof.core.rrtmgp import gas_optics, load_gas_tables
    from woof.verify.npref import (
        _rrtmgp_interp2, _rrtmgp_interpolation, np_rrtmgp_col_dry,
    )

    tables = load_gas_tables("lw")
    interval = 19  # lower-atmosphere CFC11, g-points 81:96 in Fortran
    gfirst, glast = tables.minor_limits_gpt_lower[interval]
    assert (gfirst, glast) == (80, 95)
    assert tables.idx_minor_lower[interval] == tables.gas_index["cfc11"]
    with Dataset(DATA_DIR / "rrtmgp-gas-lw-g256.nc") as nc:
        raw_start = int(nc["kminor_start_lower"][interval]) - 1
    assert raw_start == 304
    assert int(tables.kminor_start_lower[interval]) == raw_start

    width = int(glast - gfirst + 1)
    kminor = np.zeros_like(tables.kminor_lower)
    kminor[:, :, raw_start:raw_start + width] = \
        tables.kminor_lower[:, :, raw_start:raw_start + width]
    isolated = replace(
        tables, kmajor=np.zeros_like(tables.kmajor),
        kminor_lower=kminor, kminor_upper=np.zeros_like(tables.kminor_upper))

    jt = 6
    play = np.array([[tables.press_ref[10]]], np.float64)
    plev = np.array([[play[0, 0] * 1.1, play[0, 0] * 0.9]], np.float64)
    tlay = np.array([[tables.temp_ref[jt]]], np.float64)
    vmr = np.zeros((1, 1, tables.ngas + 1), np.float64)
    ratio = tables.vmr_ref[0, tables.gas_index["h2o"], jt]
    vmr[0, 0, tables.gas_index["h2o"]] = (2.0 / 3.0) * ratio
    vmr[0, 0, tables.gas_index["cfc11"]] = 1.0e-6

    col_dry = np_rrtmgp_col_dry(
        vmr[:, :, tables.gas_index["h2o"]], plev)
    col_gas = vmr * col_dry[:, :, None]
    col_gas[:, :, 0] = col_dry
    (jtemp, _jpress, _tropo, jeta, _col_mix,
     fminor, _fmajor) = _rrtmgp_interpolation(
        tables, play, tlay, col_gas)
    flavor = int(tables.gpoint_flavor[0, gfirst])
    expected = []
    for gpoint in (gfirst, glast):
        coefficient_index = raw_start + int(gpoint - gfirst)
        coefficient = _rrtmgp_interp2(
            tables.kminor_lower, coefficient_index, coefficient_index,
            jtemp[0, 0], jeta[:, 0, 0, flavor],
            fminor[:, :, 0, 0, flavor])[0]
        expected.append(
            vmr[0, 0, tables.gas_index["cfc11"]] * col_dry[0, 0]
            * coefficient)
    assert np.all(np.asarray(expected) > 0.0)

    actual = gas_optics(
        isolated, cp.asarray(play, dtype=cp.float32),
        cp.asarray(plev, dtype=cp.float32),
        cp.asarray(tlay, dtype=cp.float32),
        cp.asarray(vmr, dtype=cp.float32))
    edge_values = cp.asnumpy(actual.tau[0, 0, [gfirst, glast]])
    np.testing.assert_allclose(edge_values, expected, rtol=3e-5, atol=2e-7)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_gas_cuda_matches_float64_mirror(kind):
    import cupy as cp
    from woof.core.rrtmgp import gas_optics, load_gas_tables
    from woof.verify.npref import np_rrtmgp_gas_optics

    tables = load_gas_tables(kind)
    play, plev, tlay, vmr = _rfmip_columns(tables, sites=(4, 31, 88))
    ref = np_rrtmgp_gas_optics(tables, play, plev, tlay, vmr)
    got = gas_optics(tables, cp.asarray(play, dtype=cp.float32),
                     cp.asarray(plev, dtype=cp.float32),
                     cp.asarray(tlay, dtype=cp.float32),
                     cp.asarray(vmr, dtype=cp.float32))
    np.testing.assert_allclose(cp.asnumpy(got.tau), ref.tau,
                               rtol=8e-5, atol=2e-6)
    if kind == "sw":
        np.testing.assert_allclose(cp.asnumpy(got.ssa), ref.ssa,
                                   rtol=1.5e-4, atol=3e-6)
        assert got.g.shape == got.tau.shape
        assert bool(cp.all(got.g == 0.0))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_interpolation_prepass_matches_retained_inline_boundaries():
    import cupy as cp
    from woof.core.rrtmgp import (
        DTYPE, _interpolation_metadata, load_gas_tables,
    )
    from woof.core.kernels import get_kernel

    tables = load_gas_tables("lw")
    p_bottom = np.float32(tables.press_ref[0])
    p_top = np.float32(tables.press_ref[-1])
    p_trop = np.float32(tables.press_ref_trop)
    pressures = np.array([
        p_bottom, np.nextafter(p_bottom, np.float32(0.0)),
        np.nextafter(p_trop, np.float32(-np.inf)), p_trop,
        np.nextafter(p_trop, np.float32(np.inf)),
        np.nextafter(p_top, np.float32(np.inf)), p_top,
    ], np.float32)
    t_bottom = np.float32(tables.temp_ref[0])
    t_top = np.float32(tables.temp_ref[-1])
    temperatures = np.array([
        t_bottom, np.nextafter(t_bottom, np.float32(np.inf)),
        np.float32(tables.temp_ref[1]),
        np.nextafter(t_top, np.float32(-np.inf)), t_top,
    ], np.float32)
    pgrid, tgrid = np.meshgrid(pressures, temperatures)
    play = cp.asarray(pgrid.reshape(1, -1))
    tlay = cp.asarray(tgrid.reshape(1, -1))
    actual = _interpolation_metadata(tables, play, tlay, validate=False)

    d = tables.to_device()
    integer = tuple(cp.empty(play.shape, dtype=cp.int32) for _ in range(3))
    fraction = tuple(cp.empty(play.shape, dtype=cp.float32) for _ in range(2))
    n = play.size
    threads = 256
    get_kernel("rrtmgp_gas", "rrtmgp_interpolation_inline_reference")(
        ((n + threads - 1) // threads,), (threads,), (
            play, tlay, d.press_ref, d.temp_ref,
            DTYPE(tables.press_ref_trop), *integer, *fraction,
            np.int32(n), np.int32(tables.ntemp), np.int32(tables.npres)))
    for got, expected in zip(
            (actual.iatm, actual.jt, actual.jp,
             actual.ftemp, actual.fpress), (*integer, *fraction)):
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_driver_owned_metadata_reuse_is_bit_identical():
    import cupy as cp
    from woof.core.rrtmgp import (
        _gas_optics, _interface_temperatures, _interpolation_metadata,
        _planck_sources, gas_optics, load_gas_tables, planck_sources,
    )

    tables = load_gas_tables("lw")
    play = cp.asarray([[90000.0, 45000.0, 8000.0],
                       [85000.0, 30000.0, 2500.0]], dtype=cp.float32)
    plev = cp.asarray([[100000.0, 65000.0, 20000.0, 1000.0],
                       [95000.0, 55000.0, 12000.0, 800.0]], dtype=cp.float32)
    tlay = cp.asarray([[290.0, 255.0, 220.0],
                       [285.0, 245.0, 205.0]], dtype=cp.float32)
    vmr = cp.zeros((2, 3, tables.ngas + 1), dtype=cp.float32)
    vmr[:, :, tables.gas_index["h2o"]] = cp.asarray(
        [[0.01, 0.002, 1.0e-5], [0.008, 8.0e-4, 2.0e-6]],
        dtype=cp.float32)
    metadata = _interpolation_metadata(tables, play, tlay)
    fresh = gas_optics(tables, play, plev, tlay, vmr)
    reused = _gas_optics(
        tables, play, plev, tlay, vmr, metadata=metadata,
        validate=False, zero_g_sentinel=True)
    np.testing.assert_array_equal(cp.asnumpy(reused.tau),
                                  cp.asnumpy(fresh.tau))

    tlev = _interface_temperatures(play, plev, tlay)
    tsfc = cp.asarray([294.0, 289.0], dtype=cp.float32)
    fresh_source = planck_sources(
        tables, play, plev, tlay, tlev, tsfc, vmr)
    reused_source = _planck_sources(
        tables, play, plev, tlay, tlev, tsfc, vmr,
        metadata=metadata, validate=False)
    for name in ("lay_source", "lev_source", "sfc_source"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(reused_source, name)),
            cp.asnumpy(getattr(fresh_source, name)))


# ---------------------------------------------------------------------------
# (c) RTE solvers
# ---------------------------------------------------------------------------

def test_rrtmgp_lw_noscat_one_layer_isothermal_analytic():
    from woof.verify.npref import np_rrtmgp_lw_rte

    tau = np.array([[[0.7]]])
    source = 17.0
    got = np_rrtmgp_lw_rte(
        tau, np.full_like(tau, source),
        np.full((1, 2, 1), source), np.full((1, 1), source),
        np.ones((1, 1)), top_at_1=True)
    d = 1.0 / 0.6096748751
    trans = np.exp(-0.7 * d)
    np.testing.assert_allclose(got.flux_dn[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(got.flux_dn[0, 1],
                               np.pi * source * (1.0 - trans), rtol=2e-15)
    np.testing.assert_allclose(got.flux_up[0], np.pi * source, rtol=2e-15)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_lw_small_tau_source_branch_matches_float64_mirror():
    """FP32 source term stays accurate across its working-real branch point."""
    import cupy as cp
    from woof.core.rrtmgp import lw_rte
    from woof.verify.npref import np_rrtmgp_lw_rte

    tau = np.logspace(-5, -1, 81, dtype=np.float64)[:, None, None]
    lay_source = np.ones_like(tau)
    lev_source = np.zeros((tau.shape[0], 2, 1), np.float64)
    sfc_source = np.zeros((tau.shape[0], 1), np.float64)
    sfc_emis = np.ones_like(sfc_source)
    reference = np_rrtmgp_lw_rte(
        tau, lay_source, lev_source, sfc_source, sfc_emis, top_at_1=True)
    actual = lw_rte(
        cp.asarray(tau, dtype=cp.float32),
        cp.asarray(lay_source, dtype=cp.float32),
        cp.asarray(lev_source, dtype=cp.float32),
        cp.asarray(sfc_source, dtype=cp.float32),
        cp.asarray(sfc_emis, dtype=cp.float32), top_at_1=True)

    # The stale float64 threshold produced about 1.38e-3 worst source-flux
    # error near tau=1e-4; the float32-derived branch stays below this bound.
    for name in ("flux_up", "flux_dn"):
        residual = cp.asnumpy(getattr(actual, name)) - getattr(reference, name)
        assert np.max(np.abs(residual)) <= 2.0e-5


def test_rrtmgp_sw_delta_scaling_reference_equations():
    from woof.verify.npref import np_rrtmgp_delta_scale

    tau = np.array([[[2.0, 0.5]]])
    ssa = np.array([[[0.8, 0.2]]])
    asym = np.array([[[0.75, 0.1]]])
    got_tau, got_ssa, got_g = np_rrtmgp_delta_scale(tau, ssa, asym)
    f = asym ** 2
    wf = ssa * f
    np.testing.assert_allclose(got_tau, (1.0 - wf) * tau)
    np.testing.assert_allclose(got_ssa, (ssa - wf) / (1.0 - wf))
    np.testing.assert_allclose(got_g, (asym - f) / (1.0 - f))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_public_delta_scale_returns_independent_arrays_and_ieee_edges():
    import cupy as cp
    from woof.core.rrtmgp import delta_scale

    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    tau = cp.asarray([[[1.25, -0.0, tiny, 2.0]]], dtype=cp.float32)
    ssa = cp.asarray([[[0.5, 0.0, tiny, np.inf]]], dtype=cp.float32)
    asym = cp.asarray([[[0.0, -0.0, 0.0, 0.0]]], dtype=cp.float32)
    out_tau, out_ssa, out_g = delta_scale(tau, ssa, asym)
    for source in (tau, ssa, asym):
        for result in (out_tau, out_ssa, out_g):
            assert source.data.ptr != result.data.ptr
    assert out_tau.shape == out_ssa.shape == out_g.shape == tau.shape

    host_tau, host_ssa, host_g = (
        cp.asnumpy(value).reshape(-1) for value in (out_tau, out_ssa, out_g))
    np.testing.assert_array_equal(
        [host_tau[0], host_ssa[0], host_g[0]], [1.25, 0.5, 0.0])
    assert np.signbit(host_tau[1])
    assert not np.signbit(host_ssa[1])
    assert np.signbit(host_g[1])
    np.testing.assert_array_equal(
        [host_tau[2], host_ssa[2], host_g[2]], [0.0, 0.0, 0.0])
    assert np.isnan(host_tau[3]) and np.isnan(host_ssa[3])
    assert host_g[3] == 0.0


@pytest.mark.parametrize("top_at_1", [True, False])
def test_rrtmgp_sw_two_stream_conservative_column_energy(top_at_1):
    from woof.verify.npref import np_rrtmgp_delta_scale, np_rrtmgp_sw_rte

    tau = np.array([[[0.15], [0.5], [1.2]]])
    ssa = np.ones_like(tau)
    asym = np.full_like(tau, 0.7)
    if not top_at_1:
        tau, ssa, asym = tau[:, ::-1], ssa[:, ::-1], asym[:, ::-1]
    tau, ssa, asym = np_rrtmgp_delta_scale(tau, ssa, asym)
    got = np_rrtmgp_sw_rte(
        tau, ssa, asym, np.array([0.65]), np.array([[0.0]]),
        np.array([[0.0]]), np.array([[1000.0]]), top_at_1=top_at_1)
    top = 0 if top_at_1 else -1
    sfc = -1 if top_at_1 else 0
    incoming = 650.0
    # Conservative scattering atmosphere + black surface: TOA net equals
    # radiation reaching and absorbed at the surface.
    np.testing.assert_allclose(
        incoming - got.flux_up[0, top], got.flux_dn[0, sfc],
        # RRTMGP's FP32 epsilon floor bounds the near-conservative solution
        # to better than 0.02% while avoiding catastrophic cancellation.
        rtol=2e-4, atol=1e-5)
    assert np.all(got.flux_up >= 0.0) and np.all(got.flux_dn >= 0.0)


@pytest.mark.gpu
@requires_gpu
def test_rte_warp_fold_matches_the_reduce_pass_it_replaces(monkeypatch):
    """Force tile-16 partials as the oracle for the wider buffered folds.

    The tile only partitions an ascending sum. Every folded pass starts
    from the running flux, and every row follows the same FMA sequence as
    the reduction kernel. Both paths use the shipped launchers, including
    the measurement-only fold override. The separate geometry test also
    covers the 16-lane buffered path and packed columns.
    """
    import cupy as cp
    import woof.core.rrtmgp as rrtmgp

    original = rrtmgp._rte_gpt_tile

    def with_tile(forced, run):
        rrtmgp._rte_gpt_tile = (
            lambda kernel, ncol, ngpt, _t=forced, **kw: min(_t, int(ngpt)))
        try:
            with monkeypatch.context() as context:
                context.setenv("WOOF_RTE_FOLD", "0" if forced == 16 else "1")
                return run()
        finally:
            rrtmgp._rte_gpt_tile = original

    for ncol, nlay in ((176, 40), (37, 12), (1, 8)):
        rng = np.random.default_rng(ncol * 7 + nlay)
        nlev = nlay + 1
        for kind, ngpt, fold_tiles in (("lw", 256, (32, 64, 128)),
                                       ("sw", 224, (32,))):
            tau = cp.asarray(rng.uniform(0, 4, (ncol, nlay, ngpt)), np.float32)
            if kind == "lw":
                lay_s = cp.asarray(
                    rng.uniform(0, 20, (ncol, nlay, ngpt)), np.float32)
                lev_s = cp.asarray(
                    rng.uniform(0, 20, (ncol, nlev, ngpt)), np.float32)
                sfc_s = cp.asarray(
                    rng.uniform(0, 20, (ncol, ngpt)), np.float32)
                emis = cp.asarray(
                    rng.uniform(0.7, 1.0, (ncol, ngpt)), np.float32)

                def run(top):
                    return rrtmgp._lw_rte(
                        tau, lay_s, lev_s, sfc_s, emis, top_at_1=top,
                        out=None, incident_out=None)

                fields = ("flux_up", "flux_dn")
            else:
                ssa = cp.asarray(
                    rng.uniform(0, 1, (ncol, nlay, ngpt)), np.float32)
                asym = cp.asarray(
                    rng.uniform(0, 0.9, (ncol, nlay, ngpt)), np.float32)
                mu0 = cp.asarray(
                    rng.uniform(0.05, 1.0, (ncol, nlay)), np.float32)
                alb = cp.asarray(rng.uniform(0, 1, (ncol, ngpt)), np.float32)
                inc = cp.asarray(rng.uniform(1, 100, (ncol, ngpt)), np.float32)

                def run(top):
                    return rrtmgp._sw_rte(
                        tau, ssa, asym, mu0, alb, alb, inc, top_at_1=top,
                        out=None)

                fields = ("flux_up", "flux_dn", "flux_dir")
            for top_at_1 in (False, True):
                reduced = with_tile(16, lambda: run(top_at_1))
                for fold_tile in fold_tiles:
                    folded = with_tile(fold_tile, lambda: run(top_at_1))
                    for name in fields:
                        want = getattr(reduced, name)
                        got = getattr(folded, name)
                        np.testing.assert_array_equal(
                            cp.asnumpy(want).view(np.uint32),
                            cp.asnumpy(got).view(np.uint32),
                            err_msg=(f"{kind} {name} ncol={ncol} nlay={nlay} "
                                     f"top_at_1={top_at_1} is not bit-identical"))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_rte_cuda_matches_float64_mirrors():
    import cupy as cp
    from woof.core.rrtmgp import delta_scale, lw_rte, sw_rte
    from woof.verify.npref import (
        np_rrtmgp_delta_scale, np_rrtmgp_lw_rte, np_rrtmgp_sw_rte,
    )

    rng = np.random.default_rng(7441)
    ncol, nlay, ngpt = 3, 7, 5
    tau = rng.uniform(1e-4, 3.0, (ncol, nlay, ngpt))
    lay = rng.uniform(0.1, 30.0, tau.shape)
    lev = rng.uniform(0.1, 30.0, (ncol, nlay + 1, ngpt))
    sfc = rng.uniform(5.0, 35.0, (ncol, ngpt))
    emis = rng.uniform(0.9, 1.0, (ncol, ngpt))
    lw_ref = np_rrtmgp_lw_rte(tau, lay, lev, sfc, emis, top_at_1=False)
    lw_got = lw_rte(cp.asarray(tau, dtype=cp.float32),
                    cp.asarray(lay, dtype=cp.float32),
                    cp.asarray(lev, dtype=cp.float32),
                    cp.asarray(sfc, dtype=cp.float32),
                    cp.asarray(emis, dtype=cp.float32), top_at_1=False)
    np.testing.assert_allclose(cp.asnumpy(lw_got.flux_up), lw_ref.flux_up,
                               rtol=2e-5, atol=8e-5)
    np.testing.assert_allclose(cp.asnumpy(lw_got.flux_dn), lw_ref.flux_dn,
                               rtol=2e-5, atol=8e-5)

    ssa = rng.uniform(0.0, 1.0, tau.shape)
    asym = rng.uniform(0.0, 0.9, tau.shape)
    dt, ds, dg = np_rrtmgp_delta_scale(tau, ssa, asym)
    dev_tau, dev_ssa, dev_g = delta_scale(
        cp.asarray(tau, dtype=cp.float32), cp.asarray(ssa, dtype=cp.float32),
        cp.asarray(asym, dtype=cp.float32))
    np.testing.assert_allclose(cp.asnumpy(dev_tau), dt, rtol=2e-6, atol=2e-6)
    mu0 = rng.uniform(0.2, 0.95, ncol)
    alb_dir = rng.uniform(0.02, 0.35, (ncol, ngpt))
    alb_dif = rng.uniform(0.02, 0.35, (ncol, ngpt))
    inc = rng.uniform(0.1, 25.0, (ncol, ngpt))
    sw_ref = np_rrtmgp_sw_rte(dt, ds, dg, mu0, alb_dir, alb_dif, inc,
                              top_at_1=True)
    sw_got = sw_rte(dev_tau, dev_ssa, dev_g,
                    cp.asarray(mu0, dtype=cp.float32),
                    cp.asarray(alb_dir, dtype=cp.float32),
                    cp.asarray(alb_dif, dtype=cp.float32),
                    cp.asarray(inc, dtype=cp.float32), top_at_1=True)
    for name in ("flux_up", "flux_dn", "flux_dir"):
        np.testing.assert_allclose(cp.asnumpy(getattr(sw_got, name)),
                                   getattr(sw_ref, name),
                                   rtol=8e-5, atol=2e-4)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_band_emissivity_expands_unchanged_and_matches_mirror():
    import cupy as cp
    from woof.core.rrtmgp import (
        _expand_band_to_gpoint, _surface_emissivity_bands,
        load_gas_tables, lw_rte,
    )
    from woof.verify.npref import np_rrtmgp_lw_rte

    tables = load_gas_tables("lw")
    band_emissivity = np.full(tables.nband, 0.6, np.float64)
    band_emissivity[0] = 0.2
    band_emissivity[1] = 0.85
    bands = _surface_emissivity_bands(
        cp.asarray(band_emissivity[:, None, None]), tables, 1, 1)
    emissivity = _expand_band_to_gpoint(bands, tables, "test emissivity")
    expected_emissivity = band_emissivity[tables.gpoint_bands][None, :]
    np.testing.assert_array_equal(
        cp.asnumpy(emissivity), expected_emissivity.astype(np.float32))
    scalar_bands = _surface_emissivity_bands(cp.float32(0.73), tables, 1, 1)
    np.testing.assert_array_equal(
        cp.asnumpy(scalar_bands),
        np.full((1, tables.nband), np.float32(0.73), np.float32))

    tau = np.full((1, 2, tables.ngpt), 0.25, np.float64)
    lay_source = np.full_like(tau, 1.7)
    lev_source = np.full((1, 3, tables.ngpt), 1.2, np.float64)
    sfc_source = np.linspace(1.0, 3.0, tables.ngpt)[None, :]
    reference = np_rrtmgp_lw_rte(
        tau, lay_source, lev_source, sfc_source, expected_emissivity,
        top_at_1=True)
    actual = lw_rte(
        cp.asarray(tau, dtype=cp.float32),
        cp.asarray(lay_source, dtype=cp.float32),
        cp.asarray(lev_source, dtype=cp.float32),
        cp.asarray(sfc_source, dtype=cp.float32), emissivity, top_at_1=True)
    np.testing.assert_allclose(
        cp.asnumpy(actual.flux_up), reference.flux_up,
        rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(
        cp.asnumpy(actual.flux_dn), reference.flux_dn,
        rtol=2e-5, atol=2e-4)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_planck_sources_cuda_match_float64_mirror():
    import cupy as cp
    from netCDF4 import Dataset
    from woof.core.rrtmgp import load_gas_tables, planck_sources
    from woof.verify.npref import np_rrtmgp_planck_sources

    tables = load_gas_tables("lw")
    sites = np.array([3, 44, 79])
    play, plev, tlay, vmr = _rfmip_columns(tables, sites=sites)
    with Dataset(_rfmip("rfmip-clear-sky-inputs.nc")) as nc:
        tlev = np.asarray(nc["temp_level"][0, sites], np.float64)
        tsfc = np.asarray(nc["surface_temperature"][0, sites], np.float64)
    ref = np_rrtmgp_planck_sources(
        tables, play, plev, tlay, tlev, tsfc, vmr)
    got = planck_sources(
        tables, cp.asarray(play, dtype=cp.float32),
        cp.asarray(plev, dtype=cp.float32),
        cp.asarray(tlay, dtype=cp.float32),
        cp.asarray(tlev, dtype=cp.float32),
        cp.asarray(tsfc, dtype=cp.float32),
        cp.asarray(vmr, dtype=cp.float32))
    for name in ("lay_source", "lev_source", "sfc_source"):
        np.testing.assert_allclose(cp.asnumpy(getattr(got, name)),
                                   getattr(ref, name), rtol=8e-5, atol=2e-7)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_solar_normalization_uses_float64_reduction(monkeypatch):
    import cupy as cp
    from woof.core.rrtmgp import _normalized_solar_incident

    def fp32_reduction_is_forbidden(*args, **kwargs):
        raise AssertionError("solar normalization called cupy.sum")

    monkeypatch.setattr(cp, "sum", fp32_reduction_is_forbidden)
    solar = np.array([1.0e8, 3.0, 5.0, 7.0, 11.0], dtype=np.float64)
    tsi = np.array([1360.2, 1361.7], dtype=np.float64)
    got = cp.asnumpy(_normalized_solar_incident(solar, tsi))
    norm = np.sum(solar, dtype=np.float64)
    expected = solar.astype(np.float32)[None, :] \
        * (tsi / norm).astype(np.float32)[:, None]
    np.testing.assert_array_equal(got, expected)


@pytest.mark.gpu
@requires_gpu
def test_rfmip_clear_sky_full_profile_acceptance():
    """Ratified gate over all 100 sites by all 18 RFMIP experiments.

    sm_86 provenance, measured 2026-08-31: a report circulated that two
    rfmip tests were red on the desktop RTX 3080 at the RRTMGP
    optimisation (51819ed0f).  Re-run on that card (sm_86, NVRTC 13.0.48
    CL-36260728, driver 13030, cupy 14.0.1), the whole of this file is
    green at the optimisation, at its parent a744904667, and at the
    2.6.0-era tip 40b9878c0 -- 107/97/109 passed, 0 failed, 1
    environmental skip each.  The reds do not reproduce on the current
    desktop stack at any of the three refs, so they were a property of
    some earlier environment, not of the optimisation and not of this
    source; nothing here is skipped, pinned, or loosened on their
    account.
    """
    import cupy as cp
    from netCDF4 import Dataset
    from woof.core.rrtmgp import rfmip_clear_sky

    nsite, nexperiment, nlevel = 100, 18, 61
    got = rfmip_clear_sky(inputs=_rfmip("rfmip-clear-sky-inputs.nc"))
    with Dataset(_rfmip("rfmip-clear-sky-reference-lw-down.nc")) as nc:
        nc.set_auto_mask(False)
        lw_dn = np.asarray(nc["rld"][:], np.float64)
    with Dataset(_rfmip("rfmip-clear-sky-reference-lw-up.nc")) as nc:
        nc.set_auto_mask(False)
        lw_up = np.asarray(nc["rlu"][:], np.float64)
    with Dataset(_rfmip("rfmip-clear-sky-reference-sw-down.nc")) as nc:
        nc.set_auto_mask(False)
        sw_dn = np.asarray(nc["rsd"][:], np.float64)
    with Dataset(_rfmip("rfmip-clear-sky-reference-sw-up.nc")) as nc:
        nc.set_auto_mask(False)
        sw_up = np.asarray(nc["rsu"][:], np.float64)
    with Dataset(_rfmip("rfmip-clear-sky-inputs.nc")) as nc:
        nc.set_auto_mask(False)
        plev = np.asarray(nc["pres_level"][:], np.float64)
    refs = {"lw_dn": lw_dn, "lw_up": lw_up, "sw_dn": sw_dn, "sw_up": sw_up}
    actuals = {}
    for name, reference in refs.items():
        assert reference.shape == (nexperiment, nsite, nlevel)
        actual = cp.asnumpy(getattr(got, name)).reshape(reference.shape)
        actuals[name] = actual
        residual = actual - reference
        assert np.sqrt(np.mean(residual * residual)) <= 0.01, name
        # 0.06, not 0.05: the reference files are the double-precision
        # RTE-RRTMGP-181204 run, and the upstream code WOOF matches word
        # for word (fa107a1, single precision; tests/
        # test_rrtmgp_upstream_oracle_gpu.py) itself reaches 0.05188 W m-2 on
        # rsd against them (its own rrtmgp_rfmip_sw example, gfortran 13.3,
        # measured on W1 by development branch; rsu 0.0253, rld 0.00058,
        # rlu 0.00061).  The bound still refuses a real defect: any error
        # above single-precision rounding of this size fails it.
        assert np.max(np.abs(residual)) <= 0.06, name

    # Heating is diagnosed from the same broadband level fluxes as the driver.
    dp = np.abs(np.diff(plev, axis=1))[None]
    for wave in ("lw", "sw"):
        actual_net = actuals[f"{wave}_dn"] - actuals[f"{wave}_up"]
        reference_net = refs[f"{wave}_dn"] - refs[f"{wave}_up"]
        residual = (9.80665 / 1004.64) * np.diff(
            actual_net - reference_net, axis=2) / dp * 86400.0
        assert np.sqrt(np.mean(residual * residual)) <= 0.01, wave
        assert np.max(np.abs(residual)) <= 0.10, wave


@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_cloud_optics_float64_mirror_spot(kind):
    from woof.core.rrtmgp import load_cloud_tables
    from woof.verify.npref import np_rrtmgp_cloud_optics

    tables = load_cloud_tables(kind)
    clwp = np.array([[12.0, 0.0]])
    ciwp = np.array([[0.0, 8.5]])
    reliq = np.array([[tables.radliq_lwr + 2.25 * tables.liq_step_size,
                       tables.radliq_lwr]])
    dgice = np.array([[tables.diamice_lwr,
                       tables.diamice_lwr + 4.5 * tables.ice_step_size]])
    got = np_rrtmgp_cloud_optics(tables, clwp, ciwp, reliq, dgice)
    liq_ext = 0.75 * tables.extliq[2] + 0.25 * tables.extliq[3]
    ice_ext = 0.5 * tables.extice[4, :, 1] + 0.5 * tables.extice[5, :, 1]
    np.testing.assert_allclose(got.tau[0, 0], 12.0 * liq_ext,
                               rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(got.tau[0, 1], 8.5 * ice_ext,
                               rtol=0.0, atol=1e-13)
    assert np.all((got.ssa >= 0.0) & (got.ssa <= 1.0))
    assert np.all((got.g >= 0.0) & (got.g <= 1.0))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_cloud_optics_matches_fortran_reference_outputs(kind):
    """Gate production optics against the pinned Fortran example outputs."""
    import cupy as cp
    from woof.core.rrtmgp import cloud_optics, load_cloud_tables

    inputs, expected = _cloud_reference_output(kind)
    got = cloud_optics(
        load_cloud_tables(kind),
        *(cp.asarray(inputs[name][:, None])
          for name in ("lwp", "iwp", "reliq", "dgice")))
    limits = {
        "tau": (2.0e-7, 8.0e-7),
        "ssa": (1.0e-7, 4.0e-7),
        "g": (1.2e-7, 5.0e-7),
    }
    for name, (rmse_limit, max_limit) in limits.items():
        residual = cp.asnumpy(getattr(got, name))[:, 0, :] - expected[name]
        assert np.sqrt(np.mean(residual * residual)) <= rmse_limit, name
        assert np.max(np.abs(residual)) <= max_limit, name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_cloud_optics_cuda_matches_float64_mirror(kind):
    import cupy as cp
    from woof.core.rrtmgp import cloud_optics, load_cloud_tables
    from woof.verify.npref import np_rrtmgp_cloud_optics

    tables = load_cloud_tables(kind)
    clwp = np.array([[0.0, 4.0, 13.0], [2.0, 7.0, 0.0]])
    ciwp = np.array([[6.0, 0.0, 5.0], [0.0, 3.0, 9.0]])
    reliq = np.array([[2.5, 7.1, 18.4], [4.0, 12.2, 21.5]])
    dgice = np.array([[31.0, 10.0, 95.0], [10.0, 140.0, 178.0]])
    ref = np_rrtmgp_cloud_optics(tables, clwp, ciwp, reliq, dgice)
    got = cloud_optics(tables, cp.asarray(clwp), cp.asarray(ciwp),
                       cp.asarray(reliq), cp.asarray(dgice))
    for name in ("tau", "ssa", "g"):
        np.testing.assert_allclose(cp.asnumpy(getattr(got, name)),
                                   getattr(ref, name), rtol=8e-5, atol=2e-6)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
def test_rrtmgp_cloud_optics_preserves_reference_nonpositive_path_mask(kind):
    """RRTMGP reference masks non-positive water paths as clear sky."""
    import cupy as cp
    from woof.core.rrtmgp import cloud_optics, load_cloud_tables

    tables = load_cloud_tables(kind)
    paths = cp.asarray([[-1.0, 0.0]], dtype=cp.float32)
    result = cloud_optics(
        tables, paths, paths,
        cp.full_like(paths, tables.radliq_lwr),
        cp.full_like(paths, tables.diamice_lwr))
    for name in ("tau", "ssa", "g"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(result, name)), 0.0)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_hydrometeor_paths_kessler_and_morrison_interfaces():
    import cupy as cp
    from woof.core.rrtmgp import hydrometeor_paths

    plev = cp.asarray([[100000.0, 90000.0]], dtype=cp.float32)
    qc = cp.asarray([[1.0e-3]], dtype=cp.float32)
    qr = cp.asarray([[2.0e-4]], dtype=cp.float32)
    zero = cp.zeros_like(qc)
    fixed = hydrometeor_paths(plev, qc, qr, zero, zero,
                              microphysics="kessler")
    # WRF's radiation liquid path is cloud water only -- rain is excluded
    # (module_ra_rrtmg_sw.F:11031, module_ra_rrtmg_lw.F:12490).
    np.testing.assert_allclose(cp.asnumpy(fixed.clwp),
                               10000.0 / 9.80665 * 1.0e-3 * 1000.0,
                               rtol=2e-6)
    np.testing.assert_array_equal(cp.asnumpy(fixed.reliq), [[10.0]])
    np.testing.assert_array_equal(cp.asnumpy(fixed.dgice), [[50.0]])

    play = cp.asarray([[95000.0]], dtype=cp.float32)
    tlay = cp.asarray([[280.0]], dtype=cp.float32)
    morr = hydrometeor_paths(
        plev, qc, qr, cp.asarray([[4.0e-4]], dtype=cp.float32),
        cp.asarray([[1.0e-4]], dtype=cp.float32), microphysics="morrison",
        play=play, tlay=tlay,
        nc=cp.asarray([[8.0e7]], dtype=cp.float32),
        nr=cp.asarray([[2.0e5]], dtype=cp.float32),
        ni=cp.asarray([[5.0e5]], dtype=cp.float32),
        ns=cp.asarray([[1.0e4]], dtype=cp.float32))
    assert 2.5 <= float(morr.reliq[0, 0]) <= 21.5
    assert 10.0 <= float(morr.dgice[0, 0]) <= 180.0
    assert float(morr.ciwp[0, 0]) > 0.0

    # The integrated seam consumes Morrison's post-update EFFC/EFFR/EFFI/
    # EFFS diagnostics (computed before fixed-Nc restore), rather than
    # reconstructing EFFC from the restored 250 cm-3 state.  The liquid
    # radius is EFFC alone: rain carries no radiative mass in WRF
    # (module_ra_rrtmg_sw.F:11029-11034), so EFFR is accepted and ignored.
    coupled = hydrometeor_paths(
        plev, qc, qr, cp.asarray([[4.0e-4]], dtype=cp.float32),
        cp.asarray([[1.0e-4]], dtype=cp.float32), microphysics="morrison",
        play=play, tlay=tlay,
        nc=cp.asarray([[8.0e7]], dtype=cp.float32),
        nr=cp.asarray([[2.0e5]], dtype=cp.float32),
        ni=cp.asarray([[5.0e5]], dtype=cp.float32),
        ns=cp.asarray([[1.0e4]], dtype=cp.float32),
        effc=cp.asarray([[6.0]], dtype=cp.float32),
        effr=cp.asarray([[20.0]], dtype=cp.float32),
        effi=cp.asarray([[30.0]], dtype=cp.float32),
        effs=cp.asarray([[80.0]], dtype=cp.float32))
    expected_liquid = 6.0
    expected_ice_diameter = 2.0 * (
        (5.0e5 * 30.0 + 1.0e4 * 80.0) / (5.0e5 + 1.0e4))
    assert float(coupled.reliq[0, 0]) == pytest.approx(
        expected_liquid, rel=2e-6)
    assert float(coupled.dgice[0, 0]) == pytest.approx(
        expected_ice_diameter, rel=2e-6)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_nssl_paths_use_scheme_native_micron_radii_and_wrf_masses():
    import cupy as cp
    from woof.core.rrtmgp import hydrometeor_paths
    from woof.verify.npref import np_rrtmgp_hydrometeor_paths

    plev = np.asarray([[100000.0, 90000.0]], dtype=np.float64)
    qc = np.asarray([[1.0e-3]], dtype=np.float64)
    # Deliberately large rain proves WRF's radiation mass path remains qc,
    # not qc + qr, for NSSL too.
    qr = np.asarray([[2.0e-2]], dtype=np.float64)
    qi = np.asarray([[4.0e-4]], dtype=np.float64)
    qs = np.asarray([[1.0e-4]], dtype=np.float64)
    radii = {
        "effc": np.asarray([[6.0]], dtype=np.float64),
        "effi": np.asarray([[30.0]], dtype=np.float64),
        "effs": np.asarray([[80.0]], dtype=np.float64),
    }
    ref = np_rrtmgp_hydrometeor_paths(
        plev, qc, qr, qi, qs, microphysics="nssl", **radii)
    got = hydrometeor_paths(
        cp.asarray(plev), cp.asarray(qc), cp.asarray(qr), cp.asarray(qi),
        cp.asarray(qs), microphysics="nssl",
        **{name: cp.asarray(value) for name, value in radii.items()})

    expected_mass_path = 10000.0 * 1000.0 / 9.80665
    assert ref.clwp[0, 0] == pytest.approx(qc[0, 0] * expected_mass_path)
    assert ref.ciwp[0, 0] == pytest.approx(
        (qi[0, 0] + qs[0, 0]) * expected_mass_path)
    assert ref.reliq[0, 0] == pytest.approx(6.0)
    assert ref.dgice[0, 0] == pytest.approx(80.0)
    for name in ("clwp", "ciwp", "reliq", "dgice"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(got, name)), getattr(ref, name), rtol=2.0e-6)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("invalid", [-1.0e-6, np.nan])
def test_rrtmgp_rejects_invalid_hydrometeor_masses_and_numbers(invalid):
    import cupy as cp
    from woof.core.rrtmgp import hydrometeor_paths

    plev = cp.asarray([[100000.0, 90000.0]], dtype=cp.float32)
    zero = cp.zeros((1, 1), dtype=cp.float32)
    bad_mass = cp.asarray([[invalid]], dtype=cp.float32)
    with pytest.raises(
            ValueError, match="qc must be finite and non-negative") as error:
        hydrometeor_paths(plev, bad_mass, zero, zero, zero)
    assert "first_index=(0, 0)" in str(error.value)
    assert ("negative_count=1, nonfinite_count=0" if invalid < 0.0 else
            "negative_count=0, nonfinite_count=1") in str(error.value)

    one = cp.ones((1, 1), dtype=cp.float32)
    bad_number = cp.asarray([[invalid]], dtype=cp.float32)
    with pytest.raises(ValueError, match="nr must be finite and non-negative"):
        hydrometeor_paths(
            plev, one * 1.0e-3, one * 1.0e-4, zero, zero,
            microphysics="morrison", play=one * 95000.0,
            tlay=one * 280.0, nc=one * 8.0e7, nr=bad_number,
            ni=zero, ns=zero)


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_rejects_invalid_water_vapor_in_driver_coupling():
    from datetime import datetime
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 12), cp.asarray([[40.0]]), cp.asarray([[-100.0]]),
        trace_gas_overrides={"co2": 330.0e-6})
    with pytest.raises(ValueError, match="qv must be finite and non-negative"):
        radiation._gas_vmr(
            radiation.lw_tables, cp.asarray([[90000.0]], dtype=cp.float32),
            cp.asarray([[-1.0e-5]], dtype=cp.float32))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_hydrometeor_paths_cuda_matches_float64_mirror_regimes():
    """Clear, liquid, mixed, and glaciated Morrison columns at FP32 floors."""
    import cupy as cp
    from woof.core.rrtmgp import hydrometeor_paths
    from woof.verify.npref import np_rrtmgp_hydrometeor_paths

    plev = np.broadcast_to(
        [100000.0, 85000.0, 65000.0, 40000.0], (4, 4)).copy()
    play = np.sqrt(plev[:, :-1] * plev[:, 1:])
    tlay = np.array([
        [289.0, 278.0, 260.0],  # clear
        [291.0, 281.0, 268.0],  # liquid
        [284.0, 267.0, 248.0],  # mixed
        [269.0, 250.0, 230.0],  # glaciated
    ])
    qc = np.array([
        [0.0, 0.0, 0.0],
        [8.0e-4, 1.1e-3, 2.0e-4],
        [3.0e-4, 4.0e-4, 0.0],
        [0.0, 0.0, 0.0],
    ])
    qr = np.array([
        [0.0, 0.0, 0.0],
        [1.0e-4, 3.0e-4, 5.0e-5],
        [2.0e-4, 5.0e-5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    qi = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0e-4, 3.0e-4, 6.0e-4],
        [2.0e-4, 5.0e-4, 9.0e-4],
    ])
    qs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [5.0e-5, 2.0e-4, 4.0e-4],
        [1.0e-4, 4.0e-4, 1.2e-3],
    ])
    nc = np.where(qc > 0.0, np.array([8.0e7, 5.0e7, 2.0e7]), 0.0)
    nr = np.where(qr > 0.0, np.array([3.0e5, 1.0e5, 4.0e4]), 0.0)
    ni = np.where(qi > 0.0, np.array([8.0e5, 4.0e5, 2.0e5]), 0.0)
    ns = np.where(qs > 0.0, np.array([3.0e4, 2.0e4, 8.0e3]), 0.0)
    kwargs = dict(microphysics="morrison", play=play, tlay=tlay,
                  nc=nc, nr=nr, ni=ni, ns=ns)
    ref = np_rrtmgp_hydrometeor_paths(plev, qc, qr, qi, qs, **kwargs)
    got = hydrometeor_paths(
        cp.asarray(plev), cp.asarray(qc), cp.asarray(qr), cp.asarray(qi),
        cp.asarray(qs), **{name: cp.asarray(value) if name not in
                          {"microphysics"} else value
                          for name, value in kwargs.items()})
    fp32 = np.finfo(np.float32).eps
    for name in ("clwp", "ciwp", "reliq", "dgice"):
        actual = cp.asnumpy(getattr(got, name))
        expected = getattr(ref, name)
        tolerance = 8.0 * fp32 * np.maximum(np.abs(expected), 1.0)
        assert np.all(np.abs(actual - expected) <= tolerance), name
    assert np.all(ref.clwp[0] == 0.0) and np.all(ref.ciwp[0] == 0.0)
    assert np.all(ref.ciwp[1] == 0.0)
    assert np.all(ref.clwp[3] == 0.0)

    kessler_ref = np_rrtmgp_hydrometeor_paths(
        plev, qc, qr, qi, qs, microphysics="kessler")
    kessler = hydrometeor_paths(
        cp.asarray(plev), cp.asarray(qc), cp.asarray(qr), cp.asarray(qi),
        cp.asarray(qs), microphysics="kessler")
    for name in ("clwp", "ciwp", "reliq", "dgice"):
        actual = cp.asnumpy(getattr(kessler, name))
        expected = getattr(kessler_ref, name)
        tolerance = 8.0 * fp32 * np.maximum(np.abs(expected), 1.0)
        assert np.all(np.abs(actual - expected) <= tolerance), name


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_driver_flux_mapping_matches_float64_mirror_regimes():
    """Radiation-slot outputs for clear/liquid/mixed/glaciated columns."""
    import cupy as cp
    from woof.core.rrtmgp import _fluxes_to_radiation
    from woof.verify.npref import np_rrtmgp_fluxes_to_radiation

    ny, nx, nlay = 2, 2, 3
    plev = np.broadcast_to(
        [100000.0, 85000.0, 65000.0, 40000.0], (ny * nx, nlay + 1)).copy()
    exner = (np.sqrt(plev[:, :-1] * plev[:, 1:]) / 1.0e5) ** (287 / 1004.5)
    lev = np.arange(nlay + 1, dtype=np.float64)[None, :]
    regime = np.arange(ny * nx, dtype=np.float64)[:, None]
    lw_dn = 330.0 - 52.0 * lev + 6.0 * regime
    lw_up = 390.0 - 31.0 * lev + 4.0 * regime
    sw_dn = 760.0 - 180.0 * lev - 90.0 * regime
    sw_up = 120.0 - 18.0 * lev + 14.0 * regime
    ref = np_rrtmgp_fluxes_to_radiation(
        lw_up, lw_dn, sw_up, sw_dn, plev, exner, ny=ny, nx=nx)
    got = _fluxes_to_radiation(
        cp.asarray(lw_up), cp.asarray(lw_dn), cp.asarray(sw_up),
        cp.asarray(sw_dn), cp.asarray(plev), cp.asarray(exner), ny=ny, nx=nx)
    fp32 = np.finfo(np.float32).eps
    for name in ("rthratenlw", "rthratensw", "swdown", "glw"):
        actual = cp.asnumpy(getattr(got, name))
        expected = getattr(ref, name)
        tolerance = 8.0 * fp32 * np.maximum(np.abs(expected), 1.0)
        assert np.all(np.abs(actual - expected) <= tolerance), name
    # OLR is the TOP level's upward longwave off the same bottom-to-top
    # level stack whose level 0 gives GLW just above -- a selection, so it
    # is exact rather than toleranced.  The float64 mirror predates the
    # field and does not carry it, which is why this is asserted against
    # the input directly.
    np.testing.assert_array_equal(
        cp.asnumpy(got.olr),
        lw_up[:, -1].reshape(ny, nx).astype(np.float32))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_cloud_optics_combine_with_gas_by_band():
    import cupy as cp
    from woof.core.rrtmgp import (CloudOpticsResult, GasOpticsResult,
                                   add_cloud_optics, load_gas_tables)

    sw = load_gas_tables("sw")
    shape = (1, 2, sw.ngpt)
    gas = GasOpticsResult(cp.ones(shape, cp.float32),
                          cp.full(shape, 0.2, cp.float32),
                          cp.zeros(shape, cp.float32))
    cloud = CloudOpticsResult(
        cp.full((1, 2, sw.nband), 2.0, cp.float32),
        cp.full((1, 2, sw.nband), 0.5, cp.float32),
        cp.full((1, 2, sw.nband), 0.8, cp.float32))
    total = add_cloud_optics(sw, gas, cloud)
    np.testing.assert_allclose(cp.asnumpy(total.tau), 3.0, rtol=0, atol=0)
    np.testing.assert_allclose(cp.asnumpy(total.ssa), 0.4,
                               rtol=2e-7, atol=0)
    np.testing.assert_allclose(cp.asnumpy(total.g), 2.0 / 3.0,
                               rtol=2e-7, atol=0)

    lw = load_gas_tables("lw")
    gas_lw = GasOpticsResult(cp.ones((1, 2, lw.ngpt), cp.float32))
    cloud_lw = CloudOpticsResult(
        cp.full((1, 2, lw.nband), 2.0, cp.float32),
        cp.full((1, 2, lw.nband), 0.25, cp.float32),
        cp.zeros((1, 2, lw.nband), cp.float32))
    total_lw = add_cloud_optics(lw, gas_lw, cloud_lw)
    np.testing.assert_allclose(cp.asnumpy(total_lw.tau), 2.5,
                               rtol=0, atol=0)


def test_rrtmgp_lw_fused_finalizer_pins_legacy_fp32_rounding():
    """The fused LW expression retains each former CuPy kernel boundary."""
    source = (Path(__file__).parents[1] / "woof" / "core" / "kernels" /
              "rrtmgp_cloud.cu").read_text(encoding="utf-8")
    body = source.split("void rrtmgp_finalize_cloud_lw(", 1)[1].split(
        "void rrtmgp_finalize_cloud_sw(", 1)[0]

    assert "tc = __fmul_rn(tc, (float)cloud_mask[idx]);" in body
    assert "__fsub_rn(1.0f, ssa_cloud[band_idx])" in body
    assert "__fmul_rn(tc, one_minus_ssa)" in body
    assert "__fadd_rn(tau_gas[idx], cloud_absorption)" in body


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["lw", "sw"])
@pytest.mark.parametrize("cloud_case", ["masked", "unmasked", "clear_edge"])
def test_rrtmgp_fused_cloud_finalization_is_bit_identical(kind, cloud_case):
    import cupy as cp
    from woof.core.rrtmgp import (
        CloudOpticsResult, GasOpticsResult, _finalize_cloud_optics,
        add_cloud_optics, delta_scale, load_gas_tables,
    )

    tables = load_gas_tables(kind)
    # Odd cell counts exercise the final partial launch block.
    shape = (3, 5, tables.ngpt)
    band_shape = (3, 5, tables.nband)
    tau_gas = cp.linspace(0.001, 1.7, np.prod(shape), dtype=cp.float32) \
        .reshape(shape)
    ssa_gas = (cp.linspace(0.01, 0.65, np.prod(shape), dtype=cp.float32)
               .reshape(shape) if kind == "sw" else None)
    gas = GasOpticsResult(tau_gas, ssa=ssa_gas, g=None)
    legacy_gas = GasOpticsResult(
        tau_gas, ssa=ssa_gas,
        g=cp.zeros_like(tau_gas) if kind == "sw" else None)
    if cloud_case == "clear_edge":
        tau_gas.reshape(-1)[0] = cp.nan
        if ssa_gas is not None:
            ssa_gas.reshape(-1)[1] = cp.inf
        cloud = CloudOpticsResult(
            cp.zeros(band_shape, cp.float32),
            cp.full(band_shape, 0.5, cp.float32),
            cp.full(band_shape, 0.25, cp.float32))
    else:
        cloud = CloudOpticsResult(
            cp.linspace(0.0, 2.0, np.prod(band_shape), dtype=cp.float32)
            .reshape(band_shape),
            cp.linspace(0.05, 0.95, np.prod(band_shape), dtype=cp.float32)
            .reshape(band_shape),
            cp.linspace(0.0, 0.88, np.prod(band_shape), dtype=cp.float32)
            .reshape(band_shape))
    mask = ((cp.arange(np.prod(shape)).reshape(shape) % 3) != 0
            if cloud_case == "masked" else None)

    legacy = add_cloud_optics(tables, legacy_gas, cloud, cloud_mask=mask)
    fused = _finalize_cloud_optics(tables, gas, cloud, cloud_mask=mask)
    if kind == "lw":
        expected = (legacy.tau,)
        actual = (fused.tau,)
    else:
        expected = delta_scale(legacy.tau, legacy.ssa, legacy.g)
        actual = (fused.tau, fused.ssa, fused.g)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(want))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_column_driver_1974_flux_ranges_and_zenith_dependence():
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 30, 1, 2
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 210.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)
    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": cp.zeros(shape, cp.float32),
        "qi": cp.zeros(shape, cp.float32),
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
    }
    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 18), cp.asarray([[40.0, 40.0]]),
        cp.asarray([[-100.0, -160.0]]),
        trace_gas_overrides={"co2": 330.0e-6})
    state = SimpleNamespace(elapsed_seconds=0.0, qc=atmosphere["qc"],
                            qr=cp.zeros(shape, cp.float32))
    result = radiation(atmosphere=atmosphere, fields=fields, state=state,
                       cfg=SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0,
                                           radt_minutes=12.0))
    sw = cp.asnumpy(result.swdown)
    glw = cp.asnumpy(result.glw)
    assert np.all((sw >= 0.0) & (sw <= 1200.0))
    assert np.all((glw >= 150.0) & (glw <= 450.0))
    assert sw[0, 0] > sw[0, 1]
    assert result.rthratenlw.shape == shape
    assert result.rthratensw.shape == shape
    assert bool(cp.isfinite(result.rthratenlw).all())
    assert bool(cp.isfinite(result.rthratensw).all())
    assert radiation.trace_vmr["co2"] == pytest.approx(330.0e-6)


def test_solar_constant_matches_wrf_radconst():
    """SOLCON = 1370*ECCFAC (module_radiation_driver.F:3504-3509).

    1369.704 W/m2 at julian 92.75 (Apr 3 1974 18Z) was derived
    independently by the astro audit directly from the Fortran; the
    perihelion/aphelion ordering pins the eccentricity phase.
    """
    from datetime import datetime
    from woof.core.rrtmgp import RRTMGPRadiation

    solcon = RRTMGPRadiation._solar_constant(datetime(1974, 4, 3, 18))
    assert solcon == pytest.approx(1369.704, abs=2.0e-3)
    january = RRTMGPRadiation._solar_constant(datetime(1974, 1, 3, 12))
    july = RRTMGPRadiation._solar_constant(datetime(1974, 7, 4, 12))
    assert january > 1370.0 > july


def _wrf_v461_coszen_oracle(valid_time, latitude_deg, longitude_deg,
                             hour_offset_seconds=0.0, generation="wrf_461"):
    """Literal CPU oracle for WRF's radconst/calc_coszen path.

    ``generation`` picks the equation-of-time phase and sin(2da) coefficient
    (woof.core.solar.SOLAR_GENERATIONS): WRF 4.6.1's julian - 1 and 0.04089,
    or WRF 4.8's julian - 0.5 and 0.040849, the default since 7c936f94a0a.
    """
    from woof.core.solar import SOLAR_GENERATIONS
    phase = SOLAR_GENERATIONS[generation]["phase_days"]
    sin_two = SOLAR_GENERATIONS[generation]["sin_two_coefficient"]
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0
            + valid_time.microsecond / 3.6e9)
    julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = np.pi / 180.0
    dpd = 360.0 / 365.0
    if julian >= 80.0:
        solar_longitude = dpd * (julian - 80.0)
    else:
        solar_longitude = dpd * (julian + 285.0)
    declination = np.arcsin(
        np.sin(23.5 * degrad) * np.sin(solar_longitude * degrad))
    da = 2.0 * np.pi * (julian - phase) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - sin_two * np.sin(2.0 * da))
    local_time = (hour + hour_offset_seconds / 3600.0
                  + equation / 60.0 + np.asarray(longitude_deg) / 15.0)
    hour_angle = 15.0 * (local_time - 12.0) * degrad
    latitude = np.asarray(latitude_deg) * degrad
    mu = (np.sin(latitude) * np.sin(declination)
          + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
    return np.clip(mu, -1.0, 1.0)


@pytest.mark.parametrize(
    ("valid_time", "offset_seconds"),
    [
        ("1974-01-03T00:00:00", 0.0),       # radconst's julian < 80 branch
        ("1974-04-03T18:12:00", 360.0),    # real74 call plus radt midpoint
        ("1974-12-31T23:59:59", 0.5),      # fixed-365 year boundary phase
    ],
)
def test_cosine_zenith_matches_wrf_v461_cpu_oracle(
        monkeypatch, valid_time, offset_seconds):
    """Pin WRF v4.6.1 radconst/calc_coszen without requiring a GPU.

    ``solar_generation = "wrf_461"`` is the numpy transcription this test
    pins.  The default generation (wrf_480, 7c936f94a0a) is a device
    kernel graded word for word against both releases' Fortran in
    tests/test_solar_generation_gpu.py; it has no CPU form to pin here.
    """
    import sys
    from types import SimpleNamespace
    from datetime import datetime
    from woof.core.rrtmgp import RRTMGPRadiation

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        clip=np.clip, cos=np.cos, deg2rad=np.deg2rad, sin=np.sin))
    latitude = np.array([[40.0, -12.5], [67.25, -78.0]], np.float32)
    longitude = np.array([[-100.0, 14.75], [179.5, -45.0]], np.float32)
    radiation = object.__new__(RRTMGPRadiation)
    radiation.latitude_deg = latitude
    radiation.longitude_deg = longitude
    instant = datetime.fromisoformat(valid_time)

    got = radiation._cosine_zenith(
        instant, hour_offset_seconds=offset_seconds, generation="wrf_461")
    expected = _wrf_v461_coszen_oracle(
        instant, latitude, longitude, offset_seconds)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2.0e-7)


def test_cosine_zenith_keeps_wrf_fixed365_leap_phase_and_clamps(monkeypatch):
    """WRF uses 365 in leap years and clamps roundoff to physical bounds."""
    import sys
    from types import SimpleNamespace
    from datetime import datetime
    from woof.core.rrtmgp import RRTMGPRadiation

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        clip=np.clip, cos=np.cos, deg2rad=np.deg2rad, sin=np.sin))
    radiation = object.__new__(RRTMGPRadiation)
    radiation.latitude_deg = np.array([[35.0]], np.float32)
    radiation.longitude_deg = np.array([[-97.0]], np.float32)
    # The numpy (wrf_461) transcription; see the oracle test above.
    common = dict(hour_offset_seconds=0.0, generation="wrf_461")
    nonleap_day61 = radiation._cosine_zenith(
        datetime(1975, 3, 2, 6), **common)
    leap_day61 = radiation._cosine_zenith(
        datetime(1976, 3, 1, 6), **common)
    np.testing.assert_array_equal(leap_day61, nonleap_day61)

    instant = datetime(1974, 1, 15, 12)
    julian = 14.5
    degrad = np.pi / 180.0
    declination = np.arcsin(
        np.sin(23.5 * degrad)
        * np.sin((360.0 / 365.0) * (julian + 285.0) * degrad))
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
    radiation.latitude_deg = np.array(
        [[np.rad2deg(declination)]], np.float32)
    radiation.longitude_deg = np.array([[-equation / 4.0]], np.float32)
    mu = radiation._cosine_zenith(instant, generation="wrf_461")
    assert mu[0, 0] == 1.0
    assert np.all((-1.0 <= mu) & (mu <= 1.0))


@pytest.mark.gpu
@requires_gpu
def test_rrtmgp_zenith_midpoint_and_solcon_scaling():
    """R9 wiring: zenith at the interval CENTER, SW scaled to SOLCON.

    WRF evaluates calc_coszen at xtime + radt*0.5 inside the Solar_step
    block (module_radiation_driver.F:1206-1208) and scales every SW band
    by scon/rrsw_scon (module_ra_rrtmg_sw.F:9867-9871, 10872).  The SW
    RTE is linear in the incident source, so doubling the declared solar
    constant must exactly double SWDOWN.
    """
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 30, 1, 2
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 210.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)
    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": cp.zeros(shape, cp.float32),
        "qi": cp.zeros(shape, cp.float32),
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
    }
    cfg = SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0,
                          radt_minutes=12.0)
    state = SimpleNamespace(elapsed_seconds=720.0, qc=atmosphere["qc"],
                            qr=cp.zeros(shape, cp.float32))
    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 18), cp.asarray([[40.0, 40.0]]),
        cp.asarray([[-100.0, -160.0]]),
        trace_gas_overrides={"co2": 330.0e-6})
    seen = {}
    zenith = radiation._cosine_zenith

    def capture(valid_time, **kwargs):
        seen["valid_time"] = valid_time
        seen["offset"] = kwargs.get("hour_offset_seconds", 0.0)
        return zenith(valid_time, **kwargs)

    radiation._cosine_zenith = capture
    base = radiation(atmosphere=atmosphere, fields=fields, state=state,
                     cfg=cfg)
    # WRF midpoints ONLY the hour angle: the zenith call keeps the
    # call-time instant (18:12 for elapsed 720 s) and carries half the
    # 12-min interval as the xtime offset (module_radiation_driver.F:
    # 3514-3541 -- declin/EOT from call-time julian).
    assert seen["valid_time"] == datetime(1974, 4, 3, 18, 12)
    assert seen["offset"] == 360.0
    # Independent CPU transcription of WRF radconst/calc_coszen at the
    # first grid point (40N, 100W), in the default generation (WRF 4.8's
    # phase and coefficient, 7c936f94a0a).  Declination/EOT use CALL time;
    # the hour angle alone receives the six-minute radiation midpoint.
    expected_mu = _wrf_v461_coszen_oracle(
        datetime(1974, 4, 3, 18, 12), 40.0, -100.0, 360.0,
        generation="wrf_480")
    got_mu = float(cp.asnumpy(zenith(
        datetime(1974, 4, 3, 18, 12), hour_offset_seconds=360.0))[0, 0])
    assert got_mu == pytest.approx(expected_mu, abs=5.0e-6)
    total = float(np.sum(np.asarray(
        radiation.sw_tables.solar_source, dtype=np.float64)))
    solcon = RRTMGPRadiation._solar_constant(
        datetime(1974, 4, 3) + timedelta(hours=18, minutes=12))
    # The default generation keeps SOLCON on the card: the call scales its
    # solar source by _solar_constant_device (a float32 0-d device array,
    # capture-safe since 2.9.0 round 2), never by the host _solar_constant,
    # so that is the knob to double.
    radiation._solar_constant_device = (
        lambda valid_time: cp.asarray(2.0 * solcon, dtype=cp.float32))
    doubled = radiation(atmosphere=atmosphere, fields=fields, state=state,
                        cfg=cfg)
    swdown_base = cp.asnumpy(base.swdown)
    swdown_doubled = cp.asnumpy(doubled.swdown)
    assert swdown_base.min() > 0.0
    np.testing.assert_allclose(swdown_doubled, 2.0 * swdown_base,
                               rtol=1.0e-6)
    # The default declared TSI differs from the table total by the
    # eccentricity-adjusted solar constant, not by 1.0.
    assert solcon / total == pytest.approx(1369.704 / 1360.8577,
                                           rel=1.0e-4)


REAL74_BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
requires_real74 = pytest.mark.skipif(
    not (REAL74_BUNDLE / "geo_em" / "geo_em.d01.nc").is_file(),
    reason="WRF_1974_MP55 reference bundle not present")


@pytest.mark.gpu
@requires_gpu
@requires_real74
def test_rrtmgp_real74_12z_full_field_surface_flux_gates():
    from dataclasses import replace
    import cupy as cp
    from woof.verify.cases.real74_d01 import (
        phase3_config, prepare_phase3_case)

    cfg = replace(phase3_config(), ra_physics=4, run_seconds=0.0)
    prepared = prepare_phase3_case(cfg)
    state = prepared.initial_result.state
    driver = state.physics
    driver.compute(state, cfg)
    swdown = cp.asnumpy(driver.fields["swdown"])
    glw = cp.asnumpy(driver.fields["glw"])
    assert np.all(np.isfinite(swdown))
    assert np.all(np.isfinite(glw))
    assert np.all((swdown >= 0.0) & (swdown <= 1200.0))
    assert np.all((glw >= 150.0) & (glw <= 450.0))
    # The driver midpoints the HOUR ANGLE only (module_radiation_driver.F:
    # 1206-1208 xtime + radt*0.5; :3514-3541 declin/EOT at call time);
    # compare on the identical convention so terminator cells agree.
    mu = cp.asnumpy(driver.radiation_callable._cosine_zenith(
        driver.radiation_callable.start_time,
        hour_offset_seconds=0.5 * driver.radt_seconds))
    assert np.all(swdown[mu <= 0.0] == 0.0)
    daylight = mu > 0.05
    assert np.corrcoef(mu[daylight], swdown[daylight])[0, 1] > 0.95


# ---------------------------------------------------------------------------
# (f) WRF icloud=1 cloud fraction, McICA fraction weighting, rain exclusion
#
# Reference authority: WRF v4.6.1 phys/module_radiation_driver.F:3761-3986
# (cal_cldfra1, selected by the Registry default icloud=1 at
# Registry.EM_COMMON:2498) and the RRTMG McICA cloud plumbing in
# phys/module_ra_rrtmg_sw.F / module_ra_rrtmg_lw.F.
# ---------------------------------------------------------------------------

def test_cal_cldfra1_xu_randall_hand_fixtures():
    """Hand-computed cal_cldfra1 fixtures (module_radiation_driver.F:3858-3981).

    Expected values are evaluated by hand from the Fortran formula with
    ALPHA0=100, GAMMA=0.49, QCLDMIN=1e-12, PEXP=0.25, RHGRID=1.0,
    SVP1=0.61078, SVP2=17.2693882, SVPI2=21.8745584, SVP3=35.86,
    SVPI3=7.66, ep_2=287/461.6 (lines 3806-3816).
    """
    from woof.verify.npref import np_cal_cldfra1

    # Morrison-style moisture set (F_QC=F_QI=F_QS=T -> lines 3870-3877):
    # QCLD = QI+QC+QS, weight = (QI+QS)/QCLD.
    qc = np.array([[0.0, 5.0e-4, 2.0e-4]])
    qi = np.array([[0.0, 2.0e-4, 1.0e-4]])
    qs = np.array([[0.0, 1.0e-4, 5.0e-5]])
    # Saturated layer: qv = 1.05*qvs_weight; partial layer: qv = 0.85*qvs.
    qv = np.array([[4.0e-3, 0.004495526138768293, 0.004688258453295113]])
    t = np.array([[280.0, 270.0, 275.0]])
    p = np.array([[90000.0, 70000.0, 80000.0]])
    got = np_cal_cldfra1(qv, qc, qi, qs, t, p,
                         f_qc=True, f_qi=True, f_qs=True)
    # Clear (QCLD < QCLDMIN -> line 3951-3956), saturated (RHUM >= RHGRID ->
    # lines 3957-3963), and the Xu-Randall partial branch (lines 3964-3979):
    # (0.85)**0.25 * (1 - exp(-100*3.5e-4/(0.15*qvs)**0.49)) = 0.6510817...
    np.testing.assert_allclose(
        got, [[0.0, 1.0, 0.6510817482579461]], rtol=1e-12, atol=0.0)

    # Kessler-style moisture set (F_QC only -> lines 3891-3899): weight from
    # the 273.15 K threshold; QCLD = QC.
    qc_k = np.array([[2.0e-12, 5.0e-3, 4.0e-4, 4.0e-4]])
    zero = np.zeros_like(qc_k)
    qv_k = np.array([[0.004872019265725163, 0.009734294492918877,
                      0.0077952308251602615, 0.0020540114451528643]])
    t_k = np.array([[285.0, 285.0, 285.0, 263.0]])
    p_k = np.array([[90000.0, 90000.0, 90000.0, 70000.0]])
    got_k = np_cal_cldfra1(qv_k, qc_k, zero, zero, t_k, p_k,
                           f_qc=True, f_qi=False, f_qs=False)
    # Column 0: raw fraction < 0.01 truncates to zero (line 3979).
    # Column 1: ARG clamps at -6.9 (line 3972):
    #   0.999**0.25 * (1 - exp(-6.9)) = 0.9987423728071186.
    # Column 2: warm cloud, weight=0 -> 0.5420351983104458.
    # Column 3: cold cloud, weight=1 (qvs = qvsi) -> 0.8886662075047305.
    np.testing.assert_allclose(
        got_k,
        [[0.0, 0.9987423728071186, 0.5420351983104458, 0.8886662075047305]],
        rtol=1e-12, atol=0.0)


@pytest.mark.gpu
@requires_gpu
def test_cal_cldfra1_cuda_matches_float64_mirror_regimes():
    """Clear, partial-liquid, saturated-mixed, and glaciated columns."""
    import cupy as cp
    from woof.core.rrtmgp import cal_cldfra1
    from woof.verify.npref import np_cal_cldfra1

    p = np.broadcast_to([95000.0, 75000.0, 52000.0], (4, 3)).copy()
    t = np.array([
        [289.0, 278.0, 260.0],   # clear
        [291.0, 281.0, 268.0],   # partial liquid
        [284.0, 267.0, 248.0],   # saturated mixed
        [269.0, 250.0, 230.0],   # glaciated
    ])
    qc = np.array([
        [0.0, 0.0, 0.0],
        [8.0e-4, 1.1e-3, 2.0e-4],
        [3.0e-4, 4.0e-4, 0.0],
        [0.0, 0.0, 0.0],
    ])
    qi = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0e-4, 3.0e-4, 6.0e-4],
        [2.0e-4, 5.0e-4, 9.0e-4],
    ])
    qs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [5.0e-5, 2.0e-4, 4.0e-4],
        [1.0e-4, 4.0e-4, 1.2e-3],
    ])
    # Sub-saturated everywhere except the saturated mixed row; kept away
    # from the RHUM>=1 and CLDFRA<0.01 branch thresholds so FP32/FP64
    # branch selection cannot disagree.
    rh = np.array([
        [0.45, 0.50, 0.40],
        [0.80, 0.85, 0.70],
        [1.10, 1.08, 1.12],
        [0.75, 0.80, 0.85],
    ])
    ep2 = 287.0 / 461.6
    esw = 1000.0 * 0.61078 * np.exp(17.2693882 * (t - 273.15) / (t - 35.86))
    esi = 1000.0 * 0.61078 * np.exp(21.8745584 * (t - 273.15) / (t - 7.66))
    qvsw = ep2 * esw / (p - esw)
    qvsi = ep2 * esi / (p - esi)
    qcld = qc + qi + qs
    weight = np.where(qcld > 0.0, (qi + qs) / np.maximum(qcld, 1e-30), 0.0)
    qv = rh * ((1.0 - weight) * qvsw + weight * qvsi)

    ref = np_cal_cldfra1(qv, qc, qi, qs, t, p,
                         f_qc=True, f_qi=True, f_qs=True)
    got = cp.asnumpy(cal_cldfra1(
        cp.asarray(qv, dtype=cp.float32), cp.asarray(qc, dtype=cp.float32),
        cp.asarray(qi, dtype=cp.float32), cp.asarray(qs, dtype=cp.float32),
        cp.asarray(t, dtype=cp.float32), cp.asarray(p, dtype=cp.float32),
        f_qc=True, f_qi=True, f_qs=True))
    assert np.all(np.abs(got - ref) <= 64.0 * np.finfo(np.float32).eps
                  * np.maximum(np.abs(ref), 1.0))
    assert np.all(ref[0] == 0.0)
    assert np.all(ref[2] == 1.0)
    assert np.all((ref[1] > 0.01) & (ref[1] < 1.0))
    assert np.all((ref[3] > 0.01) & (ref[3] < 1.0))

    # Kessler moisture set on the liquid row.
    ref_k = np_cal_cldfra1(qv, qc, np.zeros_like(qc), np.zeros_like(qc),
                           t, p, f_qc=True, f_qi=False, f_qs=False)
    got_k = cp.asnumpy(cal_cldfra1(
        cp.asarray(qv, dtype=cp.float32), cp.asarray(qc, dtype=cp.float32),
        cp.zeros((4, 3), dtype=cp.float32), cp.zeros((4, 3), dtype=cp.float32),
        cp.asarray(t, dtype=cp.float32), cp.asarray(p, dtype=cp.float32),
        f_qc=True, f_qi=False, f_qs=False))
    assert np.all(np.abs(got_k - ref_k) <= 64.0 * np.finfo(np.float32).eps
                  * np.maximum(np.abs(ref_k), 1.0))


@pytest.mark.gpu
@requires_gpu
def test_mcica_maxran_masks_match_float64_mirror_and_overlap():
    """kissvec + maximum-random subcolumn masks are bit-identical to the mirror.

    Transcription pins: module_ra_rrtmg_sw.F:1727-1744 (pmid seeding),
    2008-2040 (kissvec), 1778-1813 (icld=2 maximum-random), 1941-1977
    (subcolumn cloud decision); WRF drives the generators with irng=0 and
    permuteseed=1 (SW, lines 11220-11222) / 150 (LW, lines 12687-12689 of
    module_ra_rrtmg_lw.F).
    """
    import cupy as cp
    from woof.core.rrtmgp import mcica_cloud_masks
    from woof.verify.npref import np_mcica_maxran_masks

    ngpt = 224
    nlay = 20
    play = np.geomspace(98763.4321, 11234.567, nlay)[None, :].repeat(3, 0)
    play = (play + np.array([[0.0], [17.3], [41.9]])).astype(np.float32)
    cldfra = np.zeros((3, nlay), dtype=np.float32)
    cldfra[1, 5:9] = 1.0                      # contiguous overcast block
    cldfra[2, 4] = 0.3
    cldfra[2, 9] = 0.5
    cldfra[2, 14] = 0.7
    cldfra[2, 15] = 0.7                       # adjacent equal-fraction pair

    got = cp.asnumpy(mcica_cloud_masks(
        cp.asarray(play), cp.asarray(cldfra), ngpt, permuteseed=1))
    ref = np_mcica_maxran_masks(play, cldfra, ngpt, permuteseed=1)
    np.testing.assert_array_equal(got, ref)

    # Clear column stays clear; overcast block is cloudy in every subcolumn.
    assert not got[0].any()
    assert got[1, 5:9].all()
    assert not got[1, :5].any() and not got[1, 9:].any()
    # Binomial statistics: the subcolumn mean approximates the fraction.
    for lay, frac in ((4, 0.3), (9, 0.5), (14, 0.7)):
        assert abs(got[2, lay].mean() - frac) < 0.15
    # Maximum overlap within a contiguous cloudy block: with equal adjacent
    # fractions the cloudy subcolumn set must be identical layer-to-layer
    # (module_ra_rrtmg_sw.F:1803-1813), which random overlap would break.
    np.testing.assert_array_equal(got[2, 14], got[2, 15])

    # The LW generator consumes a different permuteseed; masks must differ.
    other = cp.asnumpy(mcica_cloud_masks(
        cp.asarray(play), cp.asarray(cldfra), ngpt, permuteseed=150))
    assert (other[2, 9] != got[2, 9]).any()


def test_mcica_jump_operators_fold_the_levels_exactly():
    """One folded operator per subcolumn == walking that subcolumn's levels.

    The kernel used to apply up to ``bit_length(ngpt-1)`` level operators,
    one per set bit of ``g``.  It now applies a single operator the host
    folded, which is only legal because every level is a power of one map
    and each of the four algebras composes exactly: affine mod 2**32, GF(2)
    matrix product, and modular multiplication of the two MWC residues.
    This is that claim, checked against the walk it replaced -- on the host,
    so it holds without a GPU and fails loudly if either side drifts.
    """
    from woof.core.rrtmgp import (
        _MCICA_MWC, _mcica_gf2_apply, _mcica_jump_tables,
        _mcica_xorshift_step_matrix)

    def level_operators(nlay, njump):
        a, c = 69069, 1327217885
        A, C = 1, 0
        for _ in range(nlay):
            A, C = (a * A) & 0xFFFFFFFF, (a * C + c) & 0xFFFFFFFF
        affine = []
        for _ in range(njump):
            affine.append((A, C))
            A, C = (A * A) & 0xFFFFFFFF, (A * C + C) & 0xFFFFFFFF
        step = _mcica_xorshift_step_matrix()
        M = [1 << i for i in range(32)]
        for _ in range(nlay):
            M = [_mcica_gf2_apply(step, col) for col in M]
        matrices = []
        for _ in range(njump):
            matrices.append(M)
            M = [_mcica_gf2_apply(M, col) for col in M]
        mwc = []
        for _, modulus in _MCICA_MWC:
            m = pow(pow(65536, -1, modulus), nlay, modulus)
            level = []
            for _ in range(njump):
                level.append(m)
                m = (m * m) % modulus
            mwc.append(level)
        return affine, matrices, mwc

    p3, p4 = _MCICA_MWC[0][1], _MCICA_MWC[1][1]
    rng = np.random.default_rng(20260819)
    for nlay, ngpt in ((99, 256), (75, 224), (5, 16)):
        njump = max(1, int(ngpt - 1).bit_length())
        affine, matrices, mwc = level_operators(nlay, njump)
        _, f1, f2, f3, f4 = _mcica_jump_tables(nlay, ngpt)
        f1 = np.asarray(getattr(f1, "get", lambda: f1)()).reshape(ngpt, 2)
        f2 = np.asarray(getattr(f2, "get", lambda: f2)()).reshape(32, ngpt).T
        f3 = np.asarray(getattr(f3, "get", lambda: f3)())
        f4 = np.asarray(getattr(f4, "get", lambda: f4)())
        for g in range(ngpt):
            s1, s2 = (int(x) for x in rng.integers(1, 2 ** 32 - 1, 2))
            s3 = int(rng.integers(2, p3))
            s4 = int(rng.integers(2, p4))
            walk1, walk2, walk3, walk4 = s1, s2, s3 % p3, s4 % p4
            rest, j = g, 0
            while rest:
                if rest & 1:
                    A, C = affine[j]
                    walk1 = (A * walk1 + C) & 0xFFFFFFFF
                    walk2 = _mcica_gf2_apply(matrices[j], walk2)
                    walk3 = (walk3 * mwc[0][j]) % p3
                    walk4 = (walk4 * mwc[1][j]) % p4
                rest >>= 1
                j += 1
            folded = (
                (int(f1[g, 0]) * s1 + int(f1[g, 1])) & 0xFFFFFFFF,
                _mcica_gf2_apply([int(v) for v in f2[g]], s2),
                (s3 % p3) * int(f3[g]) % p3,
                (s4 % p4) * int(f4[g]) % p4,
            )
            assert folded == (walk1, walk2, walk3, walk4), (
                f"subcolumn {g} of {ngpt} (nlay={nlay}) lands elsewhere")


@pytest.mark.gpu
@requires_gpu
def test_mcica_fraction_weighting_produces_intermediate_fluxes():
    """A half-fraction cloud layer must sit strictly between clear/overcast."""
    from datetime import datetime
    import cupy as cp
    from woof.core.rrtmgp import (
        RRTMGPRadiation, _expand_band_to_gpoint, _interface_temperatures,
        add_cloud_optics, cloud_optics, delta_scale, gas_optics,
        load_cloud_tables, load_gas_tables, mcica_cloud_masks,
        planck_sources, lw_rte, sw_rte)

    nz = 30
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 210.0, nz)
    play = cp.asarray(np.broadcast_to(play_col, (3, nz)), dtype=cp.float32)
    plev = cp.asarray(np.broadcast_to(plev_col, (3, nz + 1)),
                      dtype=cp.float32)
    tlay = cp.asarray(np.broadcast_to(t_col, (3, nz)), dtype=cp.float32)
    qv = cp.asarray(np.broadcast_to(
        np.geomspace(8.0e-3, 1.0e-6, nz), (3, nz)), dtype=cp.float32)
    tlev = _interface_temperatures(play, plev, tlay)
    tsfc = cp.full((3,), 288.0, dtype=cp.float32)

    cldfra = cp.zeros((3, nz), dtype=cp.float32)
    cldfra[1, 12] = 0.5
    cldfra[2, 12] = 1.0
    clwp = cp.zeros((3, nz), dtype=cp.float32)
    clwp[1:, 12] = 60.0                      # in-cloud liquid water, g m-2
    zero = cp.zeros((3, nz), dtype=cp.float32)
    reliq = cp.full((3, nz), 10.0, dtype=cp.float32)
    dgice = cp.full((3, nz), 50.0, dtype=cp.float32)

    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 18), cp.asarray([[40.0]]), cp.asarray([[-100.0]]),
        trace_gas_overrides={"co2": 330.0e-6})

    lw = load_gas_tables("lw")
    vmr = radiation._gas_vmr(lw, play, qv)
    gas = gas_optics(lw, play, plev, tlay, vmr)
    cloud = cloud_optics(load_cloud_tables("lw"), clwp, zero, reliq, dgice)
    mask = mcica_cloud_masks(play, cldfra, lw.ngpt, permuteseed=150)
    optics = add_cloud_optics(lw, gas, cloud, cloud_mask=mask)
    sources = planck_sources(lw, play, plev, tlay, tlev, tsfc, vmr)
    emis = _expand_band_to_gpoint(
        cp.full((3, lw.nband), 0.96, dtype=cp.float32), lw)
    glw = cp.asnumpy(lw_rte(optics.tau, sources.lay_source,
                            sources.lev_source, sources.sfc_source, emis,
                            top_at_1=False).flux_dn[:, 0])
    assert glw[0] < glw[1] < glw[2]
    assert glw[1] - glw[0] > 0.1 * (glw[2] - glw[0])
    assert glw[2] - glw[1] > 0.1 * (glw[2] - glw[0])

    sw = load_gas_tables("sw")
    vmr_sw = radiation._gas_vmr(sw, play, qv)
    gas_sw = gas_optics(sw, play, plev, tlay, vmr_sw)
    cloud_sw = cloud_optics(load_cloud_tables("sw"), clwp, zero, reliq, dgice)
    mask_sw = mcica_cloud_masks(play, cldfra, sw.ngpt, permuteseed=1)
    optics_sw = add_cloud_optics(sw, gas_sw, cloud_sw, cloud_mask=mask_sw)
    tau, ssa, asym = delta_scale(optics_sw.tau, optics_sw.ssa, optics_sw.g)
    mu = cp.full((3,), 0.6, dtype=cp.float32)
    albedo = cp.ascontiguousarray(
        cp.full((3, sw.ngpt), 0.18, dtype=cp.float32))
    inc = cp.ascontiguousarray(cp.broadcast_to(
        cp.asarray(sw.solar_source, dtype=cp.float32)[None, :],
        (3, sw.ngpt)))
    swdn = cp.asnumpy(sw_rte(tau, ssa, asym, mu, albedo, albedo, inc,
                             top_at_1=False).flux_dn[:, 0])
    assert swdn[0] > swdn[1] > swdn[2]
    assert swdn[0] - swdn[1] > 0.1 * (swdn[0] - swdn[2])
    assert swdn[1] - swdn[2] > 0.1 * (swdn[0] - swdn[2])


@pytest.mark.gpu
@requires_gpu
def test_rainy_but_cloud_free_column_is_radiatively_clear():
    """QRAIN must not feed radiation: WRF builds the liquid path from QC only.

    Oracle: module_ra_rrtmg_sw.F:11029-11034 and module_ra_rrtmg_lw.F:
    12488-12493 (gliqwp = qc1d * pdel*100/g*1000; rain absent), and
    cal_cldfra1 counts no rain in QCLD (module_radiation_driver.F:
    3902-3918, "Rain is not part of cloud").
    """
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 30, 1, 2
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 210.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)
    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": cp.zeros(shape, cp.float32),
        "qi": cp.zeros(shape, cp.float32),
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
    }
    qr = cp.zeros(shape, cp.float32)
    qr[2:8, 0, 0] = 1.5e-3                 # rain shaft, no cloud condensate
    radiation = RRTMGPRadiation(
        datetime(1974, 4, 3, 18), cp.asarray([[40.0, 40.0]]),
        cp.asarray([[-100.0, -100.0]]),
        trace_gas_overrides={"co2": 330.0e-6})
    state = SimpleNamespace(elapsed_seconds=0.0, qc=atmosphere["qc"], qr=qr)
    result = radiation(atmosphere=atmosphere, fields=fields, state=state,
                       cfg=SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0,
                                           radt_minutes=12.0))
    swdown = cp.asnumpy(result.swdown)
    glw = cp.asnumpy(result.glw)
    lw_heat = cp.asnumpy(result.rthratenlw)
    sw_heat = cp.asnumpy(result.rthratensw)
    # Identical thermodynamic columns: the rainy column must match the
    # clear column exactly.
    np.testing.assert_array_equal(swdown[0, 0], swdown[0, 1])
    np.testing.assert_array_equal(glw[0, 0], glw[0, 1])
    np.testing.assert_array_equal(lw_heat[:, 0, 0], lw_heat[:, 0, 1])
    np.testing.assert_array_equal(sw_heat[:, 0, 0], sw_heat[:, 0, 1])


@pytest.mark.gpu
@requires_gpu
def test_hydrometeor_paths_incloud_scaling_matches_wrf_division():
    """Grid-box paths divide by max(0.01, CLDFRA) for in-cloud optics.

    Oracle: module_ra_rrtmg_sw.F:11032-11033 and module_ra_rrtmg_lw.F:
    12491-12492.
    """
    import cupy as cp
    from woof.core.rrtmgp import hydrometeor_paths
    from woof.verify.npref import np_rrtmgp_hydrometeor_paths

    plev = cp.asarray([[100000.0, 90000.0, 80000.0]], dtype=cp.float32)
    qc = cp.asarray([[1.0e-3, 4.0e-4]], dtype=cp.float32)
    qi = cp.asarray([[0.0, 2.0e-4]], dtype=cp.float32)
    qs = cp.asarray([[0.0, 1.0e-4]], dtype=cp.float32)
    zero = cp.zeros_like(qc)
    cldfra = cp.asarray([[0.4, 0.0]], dtype=cp.float32)
    got = hydrometeor_paths(plev, qc, zero, qi, qs,
                            microphysics="kessler", cldfra=cldfra)
    grid_liq = 10000.0 / 9.80665 * 1000.0 * np.array([1.0e-3, 4.0e-4])
    grid_ice = 10000.0 / 9.80665 * 1000.0 * np.array([0.0, 3.0e-4])
    np.testing.assert_allclose(
        cp.asnumpy(got.clwp)[0], grid_liq / np.array([0.4, 0.01]), rtol=2e-6)
    np.testing.assert_allclose(
        cp.asnumpy(got.ciwp)[0], grid_ice / np.array([0.4, 0.01]), rtol=2e-6)
    ref = np_rrtmgp_hydrometeor_paths(
        cp.asnumpy(plev), cp.asnumpy(qc), None, cp.asnumpy(qi),
        cp.asnumpy(qs), microphysics="kessler", cldfra=cp.asnumpy(cldfra))
    fp32 = np.finfo(np.float32).eps
    for name in ("clwp", "ciwp"):
        actual = cp.asnumpy(getattr(got, name))
        expected = getattr(ref, name)
        assert np.all(np.abs(actual - expected)
                      <= 8.0 * fp32 * np.maximum(np.abs(expected), 1.0)), name


# ---------------------------------------------------------------------------
# The column chunk is a throughput knob, not a physics knob
# ---------------------------------------------------------------------------

def _chunk_invariance_case():
    """A cloudy, horizontally heterogeneous state whose McICA masks matter.

    Heterogeneous surface pressure is essential: the subcolumn generator
    seeds KISS from the fractional Pa of each column's bottom four layer
    pressures (woof/core/kernels/rrtmgp_mcica.cu:36-45), so a uniform
    column stack would give every column the same mask and could not detect
    a chunk-dependent seed.
    """
    from datetime import datetime
    from types import SimpleNamespace

    import cupy as cp

    nz, ny, nx = 49, 8, 16
    rng = np.random.default_rng(19740403)
    p_top, p_sfc = 10000.0, 100000.0
    eta = np.linspace(1.0, 0.0, nz + 1)
    plev_col = p_top + (p_sfc - p_top) * eta
    psfc = p_sfc + 1500.0 * rng.standard_normal((ny, nx))
    scale = (psfc - p_top) / (p_sfc - p_top)
    plev = p_top + (plev_col[:, None, None] - p_top) * scale[None, :, :]
    play = 0.5 * (plev[:-1] + plev[1:])
    t = np.clip(288.0 - 6.5e-3 * (8000.0 * np.log(p_sfc / play))
                + 1.5 * rng.standard_normal(play.shape), 190.0, 320.0)
    exner = (play / 100000.0) ** (287.0 / 1004.5)
    qv = np.clip(np.geomspace(1.2e-2, 2.0e-6, nz)[:, None, None]
                 * (1.0 + 0.25 * rng.standard_normal(play.shape)),
                 1.0e-8, 4.0e-2)
    patch = rng.random((ny, nx))
    qc = np.zeros(play.shape, np.float64)
    qi = np.zeros(play.shape, np.float64)
    for k in range(4, 15):
        qc[k] = np.where(patch > 0.45, 6.0e-4, 0.0)
    for k in range(24, 39):
        qi[k] = np.where(patch < 0.65, 1.2e-4, 0.0)
    A = (lambda a: cp.asarray(np.ascontiguousarray(a), dtype=cp.float32))
    atmosphere = {
        "pressure": A(play), "p_interface": A(plev), "temperature": A(t),
        "theta": A(t / exner), "exner": A(exner), "qv": A(qv),
        "qc": A(qc), "qi": A(qi),
    }
    fields = {"tsk": A(288.0 + 6.0 * rng.standard_normal((ny, nx))),
              "albedo": A(0.15 + 0.1 * rng.random((ny, nx))),
              "emiss": A(0.94 + 0.05 * rng.random((ny, nx)))}
    state = SimpleNamespace(elapsed_seconds=720.0, p_top=p_top,
                            qc=atmosphere["qc"],
                            qr=cp.zeros_like(atmosphere["qc"]),
                            qi=atmosphere["qi"],
                            qs=cp.zeros_like(atmosphere["qi"]))
    cfg = SimpleNamespace(mp_physics=1, dt=6.0, radt=12.0, radt_minutes=12.0)
    lat = A(35.0 + 0.027 * np.arange(ny)[:, None] * np.ones(nx)[None, :])
    lon = A(-90.0 + 0.027 * np.ones(ny)[:, None] * np.arange(nx)[None, :])
    return (datetime(1974, 4, 3, 18), lat, lon, atmosphere, fields, state, cfg)


def _chunk_digests(chunks):
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    start, lat, lon, atmosphere, fields, state, cfg = _chunk_invariance_case()
    out = {}
    for chunk in chunks:
        radiation = RRTMGPRadiation(
            start, lat, lon, column_chunk=chunk,
            trace_gas_overrides={"co2": 330.0e-6})
        result = radiation(atmosphere=atmosphere, fields=fields, state=state,
                           cfg=cfg)
        out[chunk] = tuple(
            hashlib.sha256(np.ascontiguousarray(
                cp.asnumpy(getattr(result, name))).tobytes()).hexdigest()
            for name in ("rthratenlw", "rthratensw", "swdown", "glw"))
    return out


@pytest.mark.gpu
@requires_gpu
def test_radiation_is_byte_identical_across_column_chunks():
    """``column_chunk`` may buy throughput; it may not move the forecast.

    ``configs/real74_4dom.toml`` runs 6250 rather than the library default
    3125 because that is 33% faster per call on the d01 grid (2.86 s -> 1.91 s
    measured, 250x200x49, three runs) for 978.2 MiB more shared workspace.
    That trade is only legitimate while the answer does not depend on the
    chunking, which is what this asserts -- including two chunk sizes that do
    not divide the column count, so the ragged tail is covered.
    """

    digests = _chunk_digests((128, 64, 32, 50, 3125))
    reference = digests[128]
    for chunk, got in digests.items():
        assert got == reference, (
            f"column_chunk={chunk} changed the radiation answer; the chunk "
            "loop must slice columns and nothing else"
        )


@pytest.mark.gpu
@requires_gpu
def test_the_chunk_invariance_gate_fails_on_a_chunk_dependent_seed(
        monkeypatch):
    """The failing form of the gate above.

    A gate that has never been observed to fail is not evidence, so this
    reintroduces the defect it exists to catch: a McICA permutation seed that
    advances with the chunk index instead of coming from the column's own
    pressures.  WRF's own ``mcica_subcol_gen`` takes ``permuteseed`` as a
    scheme constant for exactly this reason.
    """

    from woof.core import rrtmgp as rrtmgp_module

    real = rrtmgp_module.mcica_cloud_masks
    calls = {"n": 0}

    def chunk_dependent(play, cldfra, ngpt, permuteseed, **kwargs):
        calls["n"] += 1
        return real(play, cldfra, ngpt, permuteseed + calls["n"], **kwargs)

    monkeypatch.setattr(rrtmgp_module, "mcica_cloud_masks", chunk_dependent)
    digests = _chunk_digests((128, 64))
    assert calls["n"] == 2 + 4          # 1+1 chunk pair, then 2+2
    assert digests[128] != digests[64], (
        "the negative control did not fire: either the masks stopped "
        "reaching the fluxes or this state has no cloud left"
    )


# ---------------------------------------------------------------------------
# Microphysics -> cloud-optics coupling.
#
# woof/core/rrtmgp.py used to resolve the scheme with
# ``{6: "wsm6", 8: "thompson", 10: "morrison", 18: "nssl"}.get(
#     mp_physics, "kessler")``.
# mp_physics=28 (THOMPSONAERO) therefore landed on "kessler" on the DEFAULT
# radiation engine, which does three things at once: hydrometeor_paths takes
# the constant 10 um / 50 um branch, the scheme's effc/effi/effs are never
# read out of state, and cal_cldfra1 is called with f_qi = f_qs = False so an
# overcast ice cloud produces CLDFRA 0 and radiates as clear sky.  The tests
# below pin each of those three consequences shut, plus the fail-closed
# default that let it happen silently.
# ---------------------------------------------------------------------------

# The CPU-only coupling declarations run on every platform in their own
# module; importing this shared selector set does not open a device.
from test_rrtmgp_coupling import _RTE_RRTMGP_COUPLED_MP_PHYSICS


@pytest.mark.gpu
@requires_gpu
def test_mp28_ice_cloud_is_radiatively_visible_and_matches_mp8():
    """End-to-end through the shipped adapter, on an ice-and-snow column.

    Three columns of identical thermodynamics; column 0 is clear, columns 1
    and 2 carry the same cloud ice and snow.  The run is driven twice with
    the SAME state and the same effective radii, once as mp_physics=8 and
    once as mp_physics=28.

    * mp=28 must equal mp=8 bit for bit -- WRF gives the two packages the
      same radiative coupling (see the citations on
      ``test_thompsonaero_mp28_resolves_to_the_thompson_cloud_optics_
      coupling``).
    * under mp=28 the cloudy columns must not radiate like the clear one.
    * mp_physics=1 on the SAME state is the negative control that shows
      what the defect was: Kessler's resolution calls ``cal_cldfra1`` with
      f_qi = f_qs = False, an ice-only column has QCLD = QC = 0 < QCLDMIN
      everywhere, CLDFRA is 0, every McICA subcolumn is clear, and the
      cloudy columns come back BIT-IDENTICAL to the clear one.

    Measured on this tree: SWDOWN 810.4371 W/m2 in the cloudy columns under
    mp=28 against 829.9672 W/m2 clear -- a 19.53 W/m2 shortwave signal that
    the pre-fix mp=28 threw away entirely.  Before the scheme table was
    fixed the first two assertions failed: mp=28 was byte-identical to
    mp_physics=1 and differed from mp_physics=8.
    """
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp
    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 30, 1, 3
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 210.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)

    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)

    qi = cp.zeros(shape, cp.float32)
    qs = cp.zeros(shape, cp.float32)
    qi[14:20, 0, 1:] = 3.0e-4
    qs[14:20, 0, 1:] = 5.0e-4
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": cp.zeros(shape, cp.float32),
        "qi": qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
        "xland": cp.ones((ny, nx), cp.float32),
    }
    # The mp=8/mp=28 state contract: micron effective radii written every
    # step by launch_effective_radius / launch_aerosol_effective_radius.
    state = SimpleNamespace(
        elapsed_seconds=0.0, qc=atmosphere["qc"], qi=qi, qs=qs,
        qr=cp.zeros(shape, cp.float32),
        effc=cp.full(shape, 2.49, cp.float32),
        effi=cp.full(shape, 60.0, cp.float32),
        effs=cp.full(shape, 300.0, cp.float32))

    def run(mp_physics):
        radiation = RRTMGPRadiation(
            datetime(1974, 4, 3, 18), cp.asarray([[40.0] * nx]),
            cp.asarray([[-100.0] * nx]),
            trace_gas_overrides={"co2": 330.0e-6})
        result = radiation(
            atmosphere=atmosphere, fields=fields, state=state,
            cfg=SimpleNamespace(mp_physics=mp_physics, dt=60.0, radt=12.0,
                                radt_minutes=12.0))
        return {name: cp.asnumpy(getattr(result, name))
                for name in ("rthratenlw", "rthratensw", "swdown", "glw")}

    aero = run(28)
    classic = run(8)
    kessler = run(1)

    for name, value in aero.items():
        np.testing.assert_array_equal(value, classic[name], err_msg=(
            f"mp_physics=28 and mp_physics=8 disagree on {name}; WRF gives "
            "thompson and thompsonaero the same radiative coupling"))

    # Column 0 carries no condensate in any run and is the control.
    assert aero["swdown"][0, 0] == kessler["swdown"][0, 0], (
        "the clear control column moved; the two runs are not comparable")
    # The cloudy columns must actually be cloudy under mp=28.
    shading = aero["swdown"][0, 0] - aero["swdown"][0, 1:]
    assert (shading > 1.0).all(), (
        "an ice-and-snow column is radiatively indistinguishable from the "
        f"clear column under mp_physics=28: dSWDOWN={shading}")
    assert (np.abs(aero["rthratensw"][14:20, 0, 1:]
                   - aero["rthratensw"][14:20, 0, :1]) > 0.0).all(), (
        "the ice cloud produced no shortwave heating signature")
    # The negative control: under Kessler's resolution -- which is what
    # mp_physics=28 silently got before this was fixed -- the very same ice
    # and snow are invisible, bit for bit.
    for name, value in kessler.items():
        cloudy = value[..., 1:]
        clear = np.broadcast_to(value[..., :1], cloudy.shape)
        np.testing.assert_array_equal(cloudy, clear, err_msg=(
            f"{name}: the Kessler negative control stopped being a control "
            "-- it now sees the ice cloud, so the assertion above proves "
            "nothing"))


@pytest.mark.gpu
@requires_gpu
def test_cal_cldfra1_ice_flags_decide_whether_an_ice_cloud_exists():
    """The mechanism behind the adapter test above, measured directly.

    ``cal_cldfra1`` with ``f_qi = f_qs = False`` (what mp=28 got while it
    resolved to Kessler) takes WRF's ``F_QC .and. .not. F_QI .and. .not.
    F_QS`` arm (module_radiation_driver.F:3891-3899), where QCLD = QC.  On a
    column with no cloud water that is below QCLDMIN everywhere, so CLDFRA
    is 0 no matter how much ice and snow the column carries.
    """
    import cupy as cp
    from woof.core.rrtmgp import cal_cldfra1

    qv = cp.asarray([[3.0e-4, 8.0e-5]], dtype=np.float32)
    qc = cp.zeros((1, 2), dtype=np.float32)
    qi = cp.asarray([[3.0e-4, 2.0e-4]], dtype=np.float32)
    qs = cp.asarray([[5.0e-4, 4.0e-4]], dtype=np.float32)
    tlay = cp.asarray([[258.0, 240.0]], dtype=np.float32)
    play = cp.asarray([[60000.0, 40000.0]], dtype=np.float32)

    with_ice = cp.asnumpy(cal_cldfra1(
        qv, qc, qi, qs, tlay, play, f_qc=True, f_qi=True, f_qs=True))
    without_ice = cp.asnumpy(cal_cldfra1(
        qv, qc, qi, qs, tlay, play, f_qc=True, f_qi=False, f_qs=False))
    assert (with_ice > 0.5).all(), with_ice
    assert (without_ice == 0.0).all(), without_ice


@pytest.mark.gpu
@requires_gpu
def test_mp28_effective_radii_reach_cloud_optics_instead_of_constants():
    """The second consequence: the scheme's radii, not 10 um / 50 um.

    ``hydrometeor_paths``'s Kessler branch returns constant reliq = 10 um
    and dgice = 50 um (the two ``cp.full_like`` values), discarding
    effc/effi/effs entirely.  On the ice-and-snow column below the Thompson
    branch instead produces the mass-weighted ice diameter clipped to 180 um
    -- a 3.6x difference in the quantity cloud optics interpolates on.
    """
    import cupy as cp
    from woof.core.rrtmgp import cloud_optics_scheme, hydrometeor_paths

    plev = cp.asarray([[70000.0, 50000.0, 30000.0]], dtype=np.float32)
    qc = cp.zeros((1, 2), dtype=np.float32)
    qi = cp.asarray([[3.0e-4, 2.0e-4]], dtype=np.float32)
    qs = cp.asarray([[5.0e-4, 4.0e-4]], dtype=np.float32)
    effc = cp.full((1, 2), 2.49, dtype=np.float32)
    effi = cp.asarray([[60.0, 90.0]], dtype=np.float32)
    effs = cp.asarray([[300.0, 400.0]], dtype=np.float32)

    scheme = cloud_optics_scheme(28)
    got = hydrometeor_paths(plev, qc, None, qi, qs, microphysics=scheme,
                            effc=effc, effi=effi, effs=effs)
    kess = hydrometeor_paths(plev, qc, None, qi, qs, microphysics="kessler")
    assert np.allclose(cp.asnumpy(got.dgice), 180.0)
    assert np.allclose(cp.asnumpy(kess.dgice), 50.0)
    # And the ice water path is unchanged by the branch -- only the size is,
    # so the difference above is a pure optics change, not a mass change.
    np.testing.assert_array_equal(
        cp.asnumpy(got.ciwp), cp.asnumpy(kess.ciwp))


@pytest.mark.gpu
@requires_gpu
def test_mp28_ice_cloud_is_visible_through_the_production_physics_driver():
    """The same defect, through the runtime an operator actually reaches.

    ``initialize_physics`` resolves ``ra_physics=4`` to
    :class:`RRTMGPRadiation` (woof/core/physics.py), and
    ``PhysicsDriver.compute`` calls it at the WRF radiation cadence.  The
    default template ships that radiation engine, and the tree route allows
    a per-domain microphysics override to mp_physics=28, so this is the
    path a user gets -- not a synthetic adapter call.

    MEASURED on this 6x2x16 domain, surface values, with an identical
    seeded ice-and-snow cloud in every "cloudy" run:

        run                                        SWDOWN      GLW
        mp_physics=28, before the scheme-table fix  871.5204  350.0330
        mp_physics=28, after                        399.1432  372.6432
        mp_physics=8   (unaffected either way)      399.1432  372.6432
        mp_physics=28, no cloud at all              871.5332  350.0275

    i.e. an mp=28 run put 472.38 W/m2 too much shortwave on the ground
    (2.18x the correct value) and 22.61 W/m2 too little downwelling
    longwave, while the identical mp=8 run was right.  Compare the last
    row: the pre-fix cloudy mp=28 answer sat 0.013 W/m2 from the fully
    clear-sky one, so the ice cloud was not merely mis-sized -- it was
    99.997% invisible.
    """
    from datetime import datetime
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    def run(mp_physics, *, cloudy):
        cfg = RunConfig(
            nx=6, ny=2, nz=16, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
            run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=mp_physics, sf_sfclay_physics=1,
            sf_surface_physics=2, bl_pbl_physics=1, ra_physics=4, radt=12.0)
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(
            coord, lambda z: 298.0 + 0.004 * np.asarray(z, np.float64),
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        state = init_moist_balanced(
            cfg, coord, base,
            lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
        state.u[...] = cp.float32(6.0)
        state.v[...] = cp.float32(0.5)
        if cloudy:
            # A cold ice-and-snow cloud with the scheme's own radii, exactly
            # what launch_effective_radius / launch_aerosol_effective_radius
            # leave in state after a microphysics step.
            state.qi[10:14, :, :] = cp.float32(3.0e-4)
            state.qs[10:14, :, :] = cp.float32(5.0e-4)
            state.effi[10:14, :, :] = cp.float32(60.0)
            state.effs[10:14, :, :] = cp.float32(300.0)
        landmask = np.ones((cfg.ny, cfg.nx))
        landmask[:, -1] = 0.0
        tsk = np.full((cfg.ny, cfg.nx), 299.0)
        tsk[:, landmask[0] == 0.0] = 296.0
        soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
        soil_m = np.full((4, cfg.ny, cfg.nx), 0.31)
        soil_m[:, landmask == 0.0] = 1.0
        driver = initialize_physics(
            state, cfg, landmask=landmask, tsk=tsk,
            soil_temperature=soil_t, soil_moisture=soil_m,
            liquid_moisture=soil_m,
            ivgtyp=np.where(landmask, 10, 17),
            isltyp=np.where(landmask, 6, 14), vegfra=55.0, tmn=286.0,
            swdown=450.0, glw=310.0, pblh=700.0,
            radiation_start_time=datetime(1974, 4, 3, 18),
            radiation_latitude=np.full((cfg.ny, cfg.nx), 40.0),
            radiation_longitude=np.full((cfg.ny, cfg.nx), -100.0))
        from woof.core.rrtmgp import RRTMGPRadiation
        assert isinstance(driver.radiation_callable, RRTMGPRadiation), (
            "ra_physics=4 no longer resolves to RTE+RRTMGP; this test is "
            "measuring a different engine")
        driver.compute(state, cfg)
        return {name: cp.asnumpy(driver.fields[name]).copy()
                for name in ("swdown", "glw")}

    aero = run(28, cloudy=True)
    classic = run(8, cloudy=True)
    clear = run(28, cloudy=False)

    for name, value in aero.items():
        np.testing.assert_array_equal(value, classic[name], err_msg=(
            f"{name}: mp_physics=28 and mp_physics=8 disagree on the "
            "production radiation path"))
    # The ice cloud must shade the surface.  Before the fix mp=28's SWDOWN
    # was the clear-sky answer to within 0.4 W/m2.
    shading = clear["swdown"] - aero["swdown"]
    assert (shading > 100.0).all(), (
        "the seeded ice cloud changes surface shortwave by less than "
        f"100 W/m2 under mp_physics=28: dSWDOWN={shading.ravel()}")
    assert (aero["glw"] > clear["glw"]).all(), (
        "the seeded ice cloud added no downwelling longwave under "
        "mp_physics=28")


@pytest.mark.gpu
@requires_gpu
def test_preflight_prices_radius_columns_for_exactly_the_schemes_that_use_them():
    """The allocation rail and the adapter must budget the same thing.

    ``woof/core/preflight.py::rrtmgp_column_shapes`` prices
    ``columns/effc``, ``columns/effi`` and ``columns/effs`` for every scheme
    whose radii the RTE+RRTMGP adapter reads, and its own comment records
    that mp=28 was priced there BEFORE the adapter routed it -- deliberately
    over-pricing, because an over-priced rail refuses a run that would have
    fit while an under-priced one lets a run breach the budget.  With the
    scheme table fixed the two must now agree exactly.

    "Consumes" is MEASURED, not read off the same table preflight is being
    checked against: the adapter is driven twice on one ice-and-snow column
    with two different ``effi`` values, and a scheme consumes its radii if
    and only if the fluxes move.  The probe deliberately keeps the merged
    ice+snow diameter inside the adapter's [10, 180] um clip on both legs --
    a snow radius large enough to saturate that clip would hide the effect
    of ``effi`` entirely and make the probe report False for every scheme.
    """
    from datetime import datetime
    from types import SimpleNamespace
    import cupy as cp

    from woof.config import RunConfig
    from woof.core.preflight import rrtmgp_column_shapes
    from woof.core.rrtmgp import RRTMGPRadiation

    nz, ny, nx = 20, 1, 1
    plev_col = np.geomspace(100000.0, 1.1, nz + 1)
    play_col = np.sqrt(plev_col[:-1] * plev_col[1:])
    t_col = np.linspace(290.0, 215.0, nz)
    exner_col = (play_col / 100000.0) ** (287.0 / 1004.0)
    shape = (nz, ny, nx)

    def expand(x):
        return cp.asarray(
            np.broadcast_to(x[:, None, None], shape).copy(), dtype=cp.float32)

    qi = cp.zeros(shape, cp.float32)
    qs = cp.zeros(shape, cp.float32)
    qc = cp.zeros(shape, cp.float32)
    qi[10:14] = 3.0e-4
    qs[10:14] = 5.0e-4
    qc[4:8] = 4.0e-4
    atmosphere = {
        "pressure": expand(play_col),
        "p_interface": cp.asarray(np.broadcast_to(
            plev_col[:, None, None], (nz + 1, ny, nx)).copy(),
            dtype=cp.float32),
        "temperature": expand(t_col),
        "theta": expand(t_col / exner_col),
        "exner": expand(exner_col),
        "qv": expand(np.geomspace(8.0e-3, 1.0e-6, nz)),
        "qc": qc,
        "qi": qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 288.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.18, cp.float32),
        "emiss": cp.full((ny, nx), 0.96, cp.float32),
        "xland": cp.ones((ny, nx), cp.float32),
    }

    def flux(mp_physics, effi_um):
        # One state carrying every field any scheme's coupling asks for, so
        # the only thing that varies between the two calls is effi.
        state = SimpleNamespace(
            # Registry.EM_COMMON:3017: WSM3 has combined qc/qr only.
            elapsed_seconds=0.0,
            qc=qc + qi if mp_physics == 3 else qc,
            qi=cp.zeros_like(qi) if mp_physics == 3 else qi,
            qs=cp.zeros_like(qs) if mp_physics == 3 else qs,
            qr=qs if mp_physics == 3 else cp.zeros(shape, cp.float32),
            nc=cp.full(shape, 1.0e8, cp.float32),
            nr=cp.full(shape, 1.0e3, cp.float32),
            ni=cp.full(shape, 1.0e5, cp.float32),
            ns=cp.full(shape, 1.0e3, cp.float32),
            effc=cp.full(shape, 8.0, cp.float32),
            effr=cp.full(shape, 100.0, cp.float32),
            effi=cp.full(shape, np.float32(effi_um), cp.float32),
            effs=cp.full(shape, 20.0, cp.float32),
            physics=SimpleNamespace(microphysics_updates=1))
        radiation = RRTMGPRadiation(
            datetime(1974, 4, 3, 18), cp.asarray([[40.0]]),
            cp.asarray([[-100.0]]),
            trace_gas_overrides={"co2": 330.0e-6})
        result = radiation(
            atmosphere=atmosphere, fields=fields, state=state,
            cfg=SimpleNamespace(mp_physics=mp_physics, dt=60.0, radt=12.0,
                                radt_minutes=12.0))
        return np.concatenate([
            np.ravel(cp.asnumpy(getattr(result, name)))
            for name in ("rthratenlw", "rthratensw", "swdown", "glw")])

    for mp_physics in _RTE_RRTMGP_COUPLED_MP_PHYSICS:
        cfg = RunConfig(
            nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
            run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=mp_physics, ra_physics=4, radt=12.0)
        priced = {"columns/effc", "columns/effi",
                  "columns/effs"} <= set(rrtmgp_column_shapes(cfg))
        consumes = not np.array_equal(flux(mp_physics, 5.0),
                                      flux(mp_physics, 40.0))
        assert priced == consumes, (
            f"mp_physics={mp_physics}: preflight prices effc/effi/effs "
            f"columns={priced}, but changing effi from 5 to 40 um moved "
            f"the radiation answer={consumes}")


@requires_gpu
@pytest.mark.parametrize("ncol", [7, 8])
def test_rte_buffered_fold_geometries_match_partials(ncol, monkeypatch):
    """Packed and single-column folds preserve every output word."""
    import cupy as cp
    import woof.core.rrtmgp as r

    rng = np.random.default_rng(910 + ncol)
    nlay = 9
    for kind in ("lw", "sw"):
        tables = r.load_gas_tables(kind)
        ngpt = tables.ngpt
        shape = (ncol, nlay, ngpt)
        tau = cp.asarray(rng.uniform(0.01, 3, shape), dtype=r.DTYPE)
        ssa = cp.asarray(rng.uniform(0.01, 0.9, shape), dtype=r.DTYPE)
        cloud_shape = (ncol, nlay, tables.nband)
        cloud = r.CloudOpticsResult(
            tau=cp.asarray(rng.uniform(0, 2, cloud_shape), dtype=r.DTYPE),
            ssa=cp.asarray(rng.uniform(0, 0.9, cloud_shape), dtype=r.DTYPE),
            g=cp.asarray(rng.uniform(0, 0.8, cloud_shape), dtype=r.DTYPE))
        surface = cp.asarray(rng.uniform(0.7, 0.95, (ncol, ngpt)),
                             dtype=r.DTYPE)
        inc = cp.asarray(rng.uniform(1, 30, (ncol, ngpt)), dtype=r.DTYPE)
        if kind == "lw":
            edge = np.linspace(101325, 100, nlay + 1)
            play = cp.asarray(np.tile((edge[:-1] + edge[1:]) * 0.5,
                                      (ncol, 1)), dtype=r.DTYPE)
            plev = cp.asarray(np.tile(edge, (ncol, 1)), dtype=r.DTYPE)
            tlay = cp.asarray(rng.uniform(200, 300, (ncol, nlay)), dtype=r.DTYPE)
            tlev = cp.asarray(rng.uniform(200, 300, (ncol, nlay + 1)),
                              dtype=r.DTYPE)
            tsfc = cp.asarray(rng.uniform(270, 310, ncol), dtype=r.DTYPE)
            vmr = cp.full((ncol, nlay, tables.ngas + 1), 1e-5, dtype=r.DTYPE)
            meta = r._interpolation_metadata(tables, play, tlay, validate=False)
            src = r._planck_sources(tables, play, plev, tlay, tlev, tsfc,
                                    vmr, metadata=meta, validate=False, out=None)
            planck = r.PlanckInputs(tables=tables, play=play, plev=plev,
                                   tlay=tlay, tlev=tlev, tsfc=tsfc, vmr=vmr,
                                   metadata=meta)
        else:
            mu = cp.asarray(rng.uniform(0.1, 0.95, (ncol, nlay)), dtype=r.DTYPE)
        mask = cp.asarray(rng.integers(0, 2, shape), dtype=cp.bool_)
        for cloud_mode in ("off", "unmasked", "masked"):
            fused = None if cloud_mode == "off" else r.FusedCloudOptics(
                tables=tables, cloud=cloud,
                mask=mask if cloud_mode == "masked" else None)
            for top in (False, True):
                for derive in ((False, True) if kind == "lw" else (False,)):
                    def run(tile, columns):
                        monkeypatch.setattr(r, "_rte_gpt_tile",
                                            lambda *args, **kwargs: tile)
                        if kind == "lw":
                            return r._lw_rte(
                                tau, None if derive else src.lay_source,
                                None if derive else src.lev_source,
                                None if derive else src.sfc_source, surface,
                                inc, top_at_1=top, out=None, incident_out=None,
                                fused_cloud=fused, planck=planck if derive else None,
                                _fold_columns=columns)
                        return r._sw_rte(
                            tau, ssa, cp.zeros_like(tau), mu, surface, surface,
                            inc, top_at_1=top, out=None, fused_cloud=fused,
                            _fold_columns=columns)

                    monkeypatch.delenv("WOOF_RTE_COLUMNS_PER_BLOCK", raising=False)
                    monkeypatch.setenv("WOOF_RTE_FOLD", "0")
                    reference = run(16, 1)
                    monkeypatch.setenv("WOOF_RTE_FOLD", "1")
                    fields = ("flux_up", "flux_dn") + (
                        ("flux_dir",) if kind == "sw" else ())
                    for tile in ((16, 32, 64, 128) if kind == "lw" else (16, 32)):
                        for columns in (1, 2, 4):
                            monkeypatch.setenv("WOOF_RTE_COLUMNS_PER_BLOCK", str(columns))
                            result = run(tile, 1)
                            for field in fields:
                                np.testing.assert_array_equal(
                                    cp.asnumpy(getattr(reference, field)).view(np.uint32),
                                    cp.asnumpy(getattr(result, field)).view(np.uint32),
                                    err_msg=str((kind, cloud_mode, top, derive,
                                                 tile, columns, field)))


def test_fused_finalize_matches_the_kernel_it_replaces():
    """The solver's in-register finalize equals finalize-kernel-then-solver.

    The workspace driver no longer launches rrtmgp_finalize_cloud_lw/_sw:
    the solvers combine gas optics, band cloud optics and the McICA mask in
    registers (``FusedCloudOptics``), replicating that kernel's per-op
    rounding exactly.  Flux-level bitwise equality over both bands, with
    and without a mask, is the claim -- any single-ulp difference in the
    combined tau propagates through expf and shows up across millions of
    elements.
    """
    import cupy as cp

    import woof.core.rrtmgp as rrtmgp

    if not hasattr(cp, "RawModule"):
        pytest.skip("device-only path")
    rng = np.random.default_rng(21)
    D = rrtmgp.DTYPE
    for kind, nlay in (("lw", 30), ("sw", 22)):
        tables = rrtmgp.load_gas_tables(kind)
        ngpt, nband = tables.ngpt, tables.nband
        ncol = 96
        shp = (ncol, nlay, ngpt)
        gas_tau = cp.asarray(rng.uniform(0, 4, shp), dtype=D)
        gas_ssa = cp.asarray(rng.uniform(0, 1, shp), dtype=D)
        bshp = (ncol, nlay, nband)
        cld = rrtmgp.CloudOpticsResult(
            tau=cp.asarray(rng.uniform(0, 5, bshp), dtype=D),
            ssa=cp.asarray(rng.uniform(0, 1, bshp), dtype=D),
            g=cp.asarray(rng.uniform(0, 0.9, bshp), dtype=D))
        for mask in (cp.asarray(rng.integers(0, 2, shp), dtype=cp.bool_),
                     None):
            fused = rrtmgp.FusedCloudOptics(
                tables=tables, cloud=cld, mask=mask)
            if kind == "lw":
                gas = rrtmgp.GasOpticsResult(tau=gas_tau, col_dry=None)
                opt = rrtmgp._finalize_cloud_optics(
                    tables, gas, cld, cloud_mask=mask)
                lay = cp.asarray(rng.uniform(0, 20, shp), dtype=D)
                lev = cp.asarray(
                    rng.uniform(0, 20, (ncol, nlay + 1, ngpt)), dtype=D)
                sfc = cp.asarray(rng.uniform(0, 20, (ncol, ngpt)), dtype=D)
                emis = cp.asarray(
                    rng.uniform(0.7, 1, (ncol, ngpt)), dtype=D)
                a = rrtmgp._lw_rte(opt.tau, lay, lev, sfc, emis,
                                   top_at_1=False, out=None,
                                   incident_out=None)
                b = rrtmgp._lw_rte(gas_tau, lay, lev, sfc, emis,
                                   top_at_1=False, out=None,
                                   incident_out=None, fused_cloud=fused)
                pairs = ((a.flux_up, b.flux_up), (a.flux_dn, b.flux_dn))
            else:
                gas = rrtmgp.GasOpticsResult(
                    tau=gas_tau, ssa=gas_ssa, col_dry=None)
                opt = rrtmgp._finalize_cloud_optics(
                    tables, gas, cld, cloud_mask=mask)
                mu0 = cp.asarray(rng.uniform(0.05, 1, (ncol,)), dtype=D)
                alb = cp.asarray(rng.uniform(0, 1, (ncol, ngpt)), dtype=D)
                inc = cp.asarray(
                    rng.uniform(1, 100, (ncol, ngpt)), dtype=D)
                a = rrtmgp._sw_rte(opt.tau, opt.ssa, opt.g, mu0, alb, alb,
                                   inc, top_at_1=False, out=None)
                b = rrtmgp._sw_rte(gas_tau, gas_ssa, None, mu0, alb, alb,
                                   inc, top_at_1=False, out=None,
                                   fused_cloud=fused)
                pairs = ((a.flux_up, b.flux_up), (a.flux_dn, b.flux_dn),
                         (a.flux_dir, b.flux_dir))
            for x, y in pairs:
                assert bool(cp.all(x.view(cp.uint32) == y.view(cp.uint32))), (
                    kind, mask is not None)


def test_in_solver_planck_matches_the_kernel():
    """The LW solver's derived Planck sources equal the standalone kernel's.

    The workspace driver passes ``PlanckInputs`` and the solver evaluates
    lay/lev/sfc_source in registers through the same device words the
    standalone kernel calls (rrtmgp_planck_common.cuh, included by both
    units, both compiled without contraction).  This test holds the two
    paths together bitwise at the flux level, the tripwire for a future
    edit that gives one of them its own arithmetic again.

    That shared header is the GPUWM_WRF_EXACT=1 composition
    (woof.core.kernels: a default process builds rrtmgp_rte from
    rrtmgp_planck_common_default.cuh, 2.9.1's pinned intrinsics, beside
    rrtmgp_gas's verbatim helpers, and compiles rrtmgp_gas with
    contraction), so the test runs itself in a strict process when it is
    not already in one.
    """
    import os
    import subprocess
    import sys

    import cupy as cp

    import woof.core.rrtmgp as rrtmgp

    if not hasattr(cp, "RawModule"):
        pytest.skip("device-only path")
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        env = dict(os.environ, GPUWM_WRF_EXACT="1")
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             f"{__file__}::test_in_solver_planck_matches_the_kernel"],
            env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
        return
    D = rrtmgp.DTYPE
    tables = rrtmgp.load_gas_tables("lw")
    ngpt = tables.ngpt
    rng = np.random.default_rng(23)
    for ncol, nlay in ((96, 30), (7, 12)):
        nlev = nlay + 1
        edge = np.linspace(101325.0, 2.0, nlev)
        play = cp.asarray(np.tile(0.5 * (edge[:-1] + edge[1:]), (ncol, 1)),
                          dtype=D)
        plev = cp.asarray(np.tile(edge, (ncol, 1)), dtype=D)
        tlay = cp.asarray(rng.uniform(190, 320, (ncol, nlay)), dtype=D)
        tlev = cp.asarray(rng.uniform(190, 320, (ncol, nlev)), dtype=D)
        tsfc = cp.asarray(rng.uniform(250, 320, (ncol,)), dtype=D)
        qv = cp.asarray(rng.uniform(1e-6, 2e-2, (ncol, nlay)), dtype=D)
        adapter = object.__new__(rrtmgp.RRTMGPRadiation)
        pressure = np.array([0.5, 1, 5, 20, 100, 300, 700, 1013.0]) * 100.0
        ozone = np.array([2e-6, 4e-6, 8e-6, 6e-6, 1e-6, 8e-8, 5e-8, 3e-8])
        order = np.argsort(pressure)
        adapter._ozone_logp = cp.asarray(np.log(pressure[order]), dtype=D)
        adapter._ozone_vmr = cp.asarray(ozone[order], dtype=D)
        adapter.trace_vmr = {"co2": 4.2e-4}
        vmr = adapter._gas_vmr(tables, play, qv, validate=False)
        meta = rrtmgp._interpolation_metadata(tables, play, tlay,
                                              validate=False)
        tau = cp.asarray(rng.uniform(0, 4, (ncol, nlay, ngpt)), dtype=D)
        emis = cp.asarray(rng.uniform(0.7, 1.0, (ncol, ngpt)), dtype=D)
        src = rrtmgp._planck_sources(tables, play, plev, tlay, tlev, tsfc,
                                     vmr, metadata=meta, validate=False,
                                     out=None)
        planck = rrtmgp.PlanckInputs(
            tables=tables, play=play, plev=plev, tlay=tlay, tlev=tlev,
            tsfc=tsfc, vmr=vmr, metadata=meta)
        for top_at_1 in (False, True):
            a = rrtmgp._lw_rte(tau, src.lay_source, src.lev_source,
                               src.sfc_source, emis, top_at_1=top_at_1,
                               out=None, incident_out=None)
            b = rrtmgp._lw_rte(tau, None, None, None, emis,
                               top_at_1=top_at_1, out=None,
                               incident_out=None, planck=planck)
            for n in ("flux_up", "flux_dn"):
                x, y = getattr(a, n), getattr(b, n)
                assert bool(cp.all(x.view(cp.uint32) == y.view(cp.uint32))), (
                    ncol, nlay, top_at_1, n)
