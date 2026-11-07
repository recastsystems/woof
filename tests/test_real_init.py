"""Phase 3 Task 8: WRF real-data vertical interpolation and initialization."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig
from woof.core import constants as c
from woof.core.grid import make_vertical_coord
from woof.ingest.real import (
    HRRR_ANALYZED_HYDROMETEORS,
    DECLARED_ANALYZED_HYDROMETEORS,
    HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE,
    HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS,
    _WRF_QV_MIN_VALUE,
    _cap_stratospheric_qv,
    _floor_flag_sh_surface_mixing_ratio,
    _mixing_ratio_to_relative_humidity,
    _saturation_mixing_ratio,
    _specific_humidity_to_mixing_ratio,
    _wrf_flag_sh_surface_specific_humidity,
    hydrostatic_residual,
    initialize_real,
    surface_pressure_from_surface,
)
from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.soil import (
    preprocess_noah_soil as _preprocess_noah_soil)


# These fixtures hand the soil router the raw SST/SKINTEMP pair on purpose:
# what they pin is WRF's OWN per-cell water-skin fallback
# (module_initialize_real.F:2844-2866), which is exactly what
# `water_temperature_policy = "wrf_compat"` names.  The router refuses that
# pair when nobody declares a decision, so the declaration is made once here
# instead of at every call site below, and the tests keep asserting the
# historical numbers they were written for.
def preprocess_noah_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_noah_soil(fields, **kwargs)

from woof.ingest.vert import interpolate_logp_gpu
from woof.verify.npref import np_vertical_interpolate_logp


BUNDLE = Path(os.environ.get("WOOF_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MET_EM = BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc"
requires_bundle = pytest.mark.skipif(
    not MET_EM.is_file(), reason="WRF_1974_MP55 reference bundle not present"
)


def _soil_type(fields):
    return np.where(np.asarray(fields["LANDSEA"]) >= 0.5, 6, 14)


def test_explicit_wrf_eta_levels_are_preserved():
    eta = np.array([1.0, 0.92, 0.73, 0.48, 0.21, 0.0], dtype=np.float64)
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    np.testing.assert_array_equal(coord.znw, eta)
    assert np.all(coord.dnw < 0.0)


def test_initialize_real_rejects_declared_and_catalog_orography_conflict():
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    fields = {
        "TT": np.full((2, 1, 1), 270.0),
        "RH": np.full((2, 1, 1), 50.0),
        "GHT": np.array([[[5500.0]], [[100.0]]]),
        "UU": np.zeros((2, 1, 2)),
        "VV": np.zeros((2, 2, 1)),
        "PSFC": np.full((1, 1), 100000.0),
        "T2": np.full((1, 1), 280.0),
        "D2": np.full((1, 1), 275.0),
        "U10": np.zeros((1, 2)),
        "V10": np.zeros((2, 1)),
        "SOURCE_OROGRAPHY": np.full((1, 1), 123.0, dtype=np.float64),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(1999, 5, 3, 12), levels_hpa=levels,
        fields=fields)
    eta = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    coord = make_vertical_coord(2, eta_levels=eta)
    cfg = RunConfig(
        nx=1, ny=1, nz=2, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=30.0, run_seconds=30.0, moist=True)
    with pytest.raises(ValueError) as caught:
        initialize_real(
            snapshot, cfg, coord, np.zeros((1, 1)),
            source_orography=np.full((1, 1), 456.0))
    message = str(caught.value)
    assert "declared source_orography" in message
    assert "forcing catalog SOURCE_OROGRAPHY" in message


def test_pressure_level_surface_humidity_requires_exactly_d2_or_rh2():
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    fields = {
        "TT": np.full((2, 1, 1), 270.0),
        "RH": np.full((2, 1, 1), 50.0),
        "GHT": np.array([[[5500.0]], [[100.0]]]),
        "UU": np.zeros((2, 1, 2)),
        "VV": np.zeros((2, 2, 1)),
        "PSFC": np.full((1, 1), 100000.0),
        "T2": np.full((1, 1), 280.0),
        "U10": np.zeros((1, 2)),
        "V10": np.zeros((2, 1)),
    }
    cfg = RunConfig(
        nx=1, ny=1, nz=2, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=30.0, run_seconds=30.0, moist=True)
    terrain = np.zeros((1, 1))

    for extras in ({}, {"D2": np.full((1, 1), 275.0),
                          "RH2": np.full((1, 1), 60.0)}):
        snapshot = HorizontalSnapshot(
            valid_time=datetime(1999, 5, 3, 12), levels_hpa=levels,
            fields={**fields, **extras})
        with pytest.raises(KeyError, match="exactly one of D2 or RH2"):
            initialize_real(
                snapshot, cfg, make_vertical_coord(2, eta_levels=[1.0, 0.5, 0.0]),
                terrain, source_orography=terrain)


def test_logp_mirror_interpolation_and_wrf_extrapolation():
    source_p = np.array([100000.0, 80000.0, 50000.0, 20000.0])[:, None, None]
    scalar = 4.0 + 2.5 * np.log(source_p)
    target_p = np.array([110000.0, 90000.0, 65000.0, 20000.0])[:, None, None]

    got = np_vertical_interpolate_logp(
        scalar, source_p, target_p, below="constant", above="error"
    )
    assert got[0, 0, 0] == scalar[0, 0, 0]
    np.testing.assert_allclose(got[1:3, 0, 0],
                               4.0 + 2.5 * np.log(target_p[1:3, 0, 0]),
                               rtol=0.0, atol=2.0e-14)
    assert got[-1, 0, 0] == scalar[-1, 0, 0]

    theta = np.full_like(source_p, 300.0)
    standard = np_vertical_interpolate_logp(
        theta, source_p, target_p[:1], below="temperature", above="error"
    )[0, 0, 0]
    p1, pt = source_p[0, 0, 0], target_p[0, 0, 0]
    t1 = 300.0 * (p1 / c.P0) ** c.RCP
    dp = pt - p1
    pavg = 0.5 * (pt + p1)
    dhdp = 11880.516 * 0.1902632 * (pavg / 100.0) ** (0.1902632 - 1.0)
    expected = (t1 + dhdp * (dp / 100.0) * 0.0065) * (c.P0 / pt) ** c.RCP
    assert standard == pytest.approx(expected, rel=0.0, abs=2.0e-12)


def test_logp_mirror_above_source_top_is_fatal():
    source_p = np.array([100000.0, 70000.0, 20000.0])[:, None, None]
    field = np.array([290.0, 270.0, 220.0])[:, None, None]
    target_p = np.array([10000.0])[:, None, None]

    with pytest.raises(ValueError, match="above source top"):
        np_vertical_interpolate_logp(
            field, source_p, target_p, below="constant", above="error"
        )


@requires_gpu
@pytest.mark.gpu
def test_logp_gpu_kernel_matches_float64_mirror():
    import cupy as cp

    rng = np.random.default_rng(23)
    nsrc, nz, ny, nx = 9, 7, 4, 6
    source_1d = np.geomspace(102000.0, 7000.0, nsrc)
    source_p = source_1d[:, None, None] * (
        1.0 + rng.normal(0.0, 0.002, (nsrc, ny, nx)))
    field = rng.normal(280.0, 7.0, (nsrc, ny, nx))
    target_p = np.geomspace(108000.0, 7500.0, nz)[:, None, None] * np.ones(
        (1, ny, nx))
    ref = np_vertical_interpolate_logp(
        field, source_p, target_p, below="temperature", above="error"
    )
    got = interpolate_logp_gpu(
        cp.asarray(field, cp.float32), cp.asarray(source_p, cp.float32),
        cp.asarray(target_p, cp.float32), below="temperature", above="error"
    )
    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=3.0e-5, atol=2.0e-3)

    with pytest.raises(ValueError, match="above source top"):
        interpolate_logp_gpu(
            cp.asarray(field, cp.float32), cp.asarray(source_p, cp.float32),
            cp.full((1, ny, nx), 5000.0, cp.float32),
            below="temperature", above="error",
        )


def test_qv_construction_uses_wrf_floor_and_invalid_guard():
    temperature = np.array([250.0, 300.0, 300.0, 0.0])
    pressure = np.array([100000.0, 100000.0, 1000.0, 100000.0])
    rh = np.array([0.0, 50.0, 100.0, 100.0])

    got = _saturation_mixing_ratio(temperature, pressure, rh)
    np.testing.assert_array_equal(got[[0, 2, 3]], 1.0e-6)

    es_hpa = 0.5 * (10.0 * c.SVP1) * np.exp(
        c.SVP2 * (temperature[1] - c.SVPT0)
        / (temperature[1] - c.SVP3)
    )
    # WRF v4.6.1 rh_to_mxrat1's own EPS = 0.622 (module_initialize_real.F:7366),
    # not module ep_2 -- the ingest lane review's parity residual.
    expected = 0.622 * es_hpa / (pressure[1] / 100.0 - es_hpa)
    assert got[1] == pytest.approx(expected, rel=0.0, abs=1.0e-15)
    assert got[1] > 1.0e-6


def test_hrrr_specific_humidity_uses_wrf_dry_air_conversion_without_floor():
    specific = np.array([0.0, 1.0e-7, 0.001, 0.02])
    got = _specific_humidity_to_mixing_ratio(specific)
    np.testing.assert_array_equal(got, specific / (1.0 - specific))
    assert got[0] == 0.0
    with pytest.raises(ValueError, match=r"finite in \[0, 1\)"):
        _specific_humidity_to_mixing_ratio([0.0, 1.0])
    with pytest.raises(ValueError, match=r"finite in \[0, 1\)"):
        _specific_humidity_to_mixing_ratio([np.nan])

    undershoot = np.array([-3.6e-5, 0.0, 0.01])
    np.testing.assert_allclose(
        _specific_humidity_to_mixing_ratio(
            undershoot, allow_wps_undershoot=True),
        undershoot / (1.0 - undershoot), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match=r"finite in \[-0.028126, 1\)"):
        _specific_humidity_to_mixing_ratio(
            [-0.028127], allow_wps_undershoot=True)


def test_hrrr_relative_humidity_is_inverse_of_wrf_mixing_ratio_relation():
    temperature = np.array([220.0, 260.0, 290.0, 305.0])
    pressure = np.array([15000.0, 50000.0, 85000.0, 100000.0])
    expected_rh = np.array([3.0, 35.0, 78.0, 102.0])
    qv = _saturation_mixing_ratio(temperature, pressure, expected_rh)
    # Avoid the 1e-6 floor in the inverse identity's deliberately dry point.
    active = qv > 1.0e-6
    got = _mixing_ratio_to_relative_humidity(
        temperature[active], pressure[active], qv[active])
    np.testing.assert_allclose(
        got, np.clip(expected_rh[active], 0.0, 100.0),
        rtol=0.0, atol=2.0e-13)
    with pytest.raises(ValueError, match="mixing ratio non-negative"):
        _mixing_ratio_to_relative_humidity(280.0, 90000.0, -1.0e-6)
    mapped_specific_humidity = np.array([-3.6e-5, 0.0, 0.01])
    mapped_qv = _specific_humidity_to_mixing_ratio(
        mapped_specific_humidity, allow_wps_undershoot=True)
    mapped_rh = _mixing_ratio_to_relative_humidity(
        np.full(3, 280.0), np.full(3, 90000.0), mapped_qv,
        allow_wps_undershoot=True)
    assert mapped_rh[0] < 0.0
    assert mapped_rh[1] == 0.0
    assert mapped_rh[2] > 0.0


def test_humidity_elementwise_workers_are_byte_identical():
    """Setup-only row/level workers cannot alter per-element arithmetic."""
    rng = np.random.default_rng(7416)
    shape = (11, 7, 9)
    temperature = rng.uniform(215.0, 310.0, shape)
    pressure = rng.uniform(5000.0, 102000.0, shape)
    specific = rng.uniform(0.0, 0.025, shape)
    specific[0, 0, 0] = -3.6e-5

    expected_qv = _specific_humidity_to_mixing_ratio(
        specific, allow_wps_undershoot=True, column_workers=1)
    expected_rh = _mixing_ratio_to_relative_humidity(
        temperature, pressure, expected_qv,
        allow_wps_undershoot=True, column_workers=1)
    expected_cap = _cap_stratospheric_qv(
        expected_qv, pressure, column_workers=1)
    for workers in (2, 3, 8, 32):
        qv = _specific_humidity_to_mixing_ratio(
            specific, allow_wps_undershoot=True,
            column_workers=workers)
        rh = _mixing_ratio_to_relative_humidity(
            temperature, pressure, qv, allow_wps_undershoot=True,
            column_workers=workers)
        cap = _cap_stratospheric_qv(
            qv, pressure, column_workers=workers)
        np.testing.assert_array_equal(qv, expected_qv)
        np.testing.assert_array_equal(rh, expected_rh)
        np.testing.assert_array_equal(cap, expected_cap)


def test_thermodynamic_elementwise_workers_are_byte_identical():
    from woof.ingest.real import (
        _moist_specific_volume,
        _potential_temperature_from_temperature,
        _temperature_from_potential_temperature,
    )

    rng = np.random.default_rng(7434)
    shape = (13, 7, 11)
    temperature = rng.uniform(205.0, 315.0, shape)
    pressure = rng.uniform(9000.0, 103000.0, shape)
    qv = rng.uniform(1.0e-6, 0.025, shape)
    expected_theta = _potential_temperature_from_temperature(
        temperature, pressure, column_workers=1)
    expected_temperature = _temperature_from_potential_temperature(
        expected_theta, pressure, column_workers=1)
    expected_alpha = _moist_specific_volume(
        expected_theta, qv, pressure, column_workers=1)
    for workers in (2, 3, 8, 32):
        theta = _potential_temperature_from_temperature(
            temperature, pressure, column_workers=workers)
        diagnosed_temperature = _temperature_from_potential_temperature(
            theta, pressure, column_workers=workers)
        alpha = _moist_specific_volume(
            theta, qv, pressure, column_workers=workers)
        np.testing.assert_array_equal(theta, expected_theta)
        np.testing.assert_array_equal(
            diagnosed_temperature, expected_temperature)
        np.testing.assert_array_equal(alpha, expected_alpha)


def test_flag_sh_surface_fallback_matches_wrf_whole_domain_decision():
    pressure = np.broadcast_to(
        np.array([95000.0, 70000.0, 30000.0])[:, None, None],
        (3, 2, 3)).copy()
    spfh = np.arange(18, dtype=np.float64).reshape(3, 2, 3) * 1.0e-4
    q2 = np.full((2, 3), 0.006)
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(q2, spfh, pressure), q2)

    q2[0, 0] = -1.0e-5
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(q2, spfh, pressure), spfh[0])
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            q2, spfh, pressure, force_fallback=False), q2)
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            np.full_like(q2, 0.006), spfh, pressure, force_fallback=True),
        spfh[0])
    # Reversed source-level order selects the opposite endpoint but preserves
    # the same nearest-to-surface physical level.
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            q2, spfh[::-1], pressure[::-1]), spfh[0])


def test_noah_soil_layer_interpolation_and_surface_consistency():
    land = np.array([[1.0, 0.0], [1.0, 0.0]])
    fields = {
        "LANDSEA": land,
        "SKINTEMP": np.array([[290.0, 280.0], [292.0, 281.0]]),
        "SST": np.array([[0.0, 285.0], [0.0, 286.0]]),
        "ST000007": np.full((2, 2), 289.0),
        "ST007028": np.full((2, 2), 287.0),
        "ST028100": np.full((2, 2), 284.0),
        "ST100289": np.full((2, 2), 281.0),
        "TMN": np.full((2, 2), 279.0),
        "SM000007": np.full((2, 2), 0.12),
        "SM007028": np.full((2, 2), 0.18),
        "SM028100": np.full((2, 2), 0.24),
        "SM100289": np.full((2, 2), 0.30),
        "SNOW_EC": np.array([[0.012, 0.0], [0.001, 0.0]]),
    }
    soil = preprocess_noah_soil(fields, soil_type=_soil_type(fields))
    np.testing.assert_array_equal(soil.landmask, land)
    np.testing.assert_array_equal(soil.xice, 0.0)
    np.testing.assert_allclose(soil.tsk[:, 1], [285.0, 286.0])
    np.testing.assert_array_equal(soil.soil_moisture[:, :, 1], 1.0)
    np.testing.assert_allclose(soil.snow_water[:, 0], [12.0, 1.0])
    np.testing.assert_allclose(soil.snow_depth[:, 0], [0.06, 0.005])
    assert soil.soil_temperature.shape == (4, 2, 2)
    assert np.all((soil.soil_temperature >= 170.0)
                  & (soil.soil_temperature <= 400.0))
    assert np.all((soil.soil_moisture >= 0.0) & (soil.soil_moisture <= 1.0))

    fields_without_tmn = dict(fields)
    fields_without_tmn.pop("TMN")
    with pytest.raises(KeyError, match="TMN"):
        preprocess_noah_soil(
            fields_without_tmn, soil_type=_soil_type(fields_without_tmn))


def test_hrrr_soil_depth_nodes_are_interpolated_to_noah_midpoints():
    depths = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60,
                       1.0, 1.6, 3.0])
    fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), 289.0),
        "TMN": np.full((1, 1), 281.0),
        "SOILT": (280.0 + 2.0 * depths)[:, None, None],
        "SOILW": (0.10 + 0.05 * depths)[:, None, None],
    }
    soil = preprocess_noah_soil(fields, soil_type=np.full((1, 1), 6))
    target = np.array([0.05, 0.25, 0.70, 1.50])
    np.testing.assert_allclose(
        soil.soil_temperature[:, 0, 0], 280.0 + 2.0 * target,
        rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        soil.soil_moisture[:, 0, 0], 0.10 + 0.05 * target,
        rtol=0.0, atol=1.0e-13)
    assert soil.deep_soil_temperature[0, 0] == 281.0

    with pytest.raises(KeyError, match="SOILT and SOILW together"):
        preprocess_noah_soil(
            {key: value for key, value in fields.items() if key != "SOILW"},
            soil_type=np.full((1, 1), 6))


def test_gfs_exact_noah_layers_are_copied_without_era5_interpolation():
    fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), 290.0),
        "TMN": np.full((1, 1), 280.0),
    }
    temperature_names = (
        "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200")
    moisture_names = (
        "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200")
    for layer, name in enumerate(temperature_names):
        fields[name] = np.full((1, 1), 289.0 - layer)
    for layer, name in enumerate(moisture_names):
        fields[name] = np.full((1, 1), 0.10 + 0.05 * layer)

    soil = preprocess_noah_soil(fields, soil_type=np.full((1, 1), 6))
    np.testing.assert_array_equal(
        soil.soil_temperature[:, 0, 0], [289.0, 288.0, 287.0, 286.0])
    np.testing.assert_allclose(
        soil.soil_moisture[:, 0, 0], [0.10, 0.15, 0.20, 0.25],
        rtol=0.0, atol=1.0e-15)

    with pytest.raises(KeyError, match="all four temperature and moisture"):
        preprocess_noah_soil(
            {key: value for key, value in fields.items()
             if key != "GFS_ST100200"},
            soil_type=np.full((1, 1), 6))

    with pytest.raises(ValueError, match="cannot be mixed"):
        preprocess_noah_soil(
            {**fields,
             "SOILT": np.full((9, 1, 1), 285.0),
             "SOILW": np.full((9, 1, 1), 0.2)},
            soil_type=np.full((1, 1), 6))


def test_noah_exports_wrf_repaired_deep_soil_temperature():
    fields = {
        "LANDSEA": np.array([[1.0, 0.0]]),
        "SKINTEMP": np.array([[289.0, 285.0]]),
        "SST": np.array([[0.0, 286.0]]),
        "TMN": np.array([[100.0, 500.0]]),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full((1, 2), 285.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full((1, 2), 0.2)

    soil = preprocess_noah_soil(
        fields, soil_type=np.array([[6, 14]]))
    np.testing.assert_array_equal(
        soil.deep_soil_temperature, [[289.0, 286.0]])


def test_noah_snow_state_follows_wrf_real_presence_invariants():
    """module_initialize_real.F:517-543, including all flag pairings."""
    base = {
        "LANDSEA": np.ones((1, 2)),
        "SKINTEMP": np.full((1, 2), 270.0),
        "ST000007": np.full((1, 2), 269.0),
        "ST007028": np.full((1, 2), 270.0),
        "ST028100": np.full((1, 2), 271.0),
        "ST100289": np.full((1, 2), 272.0),
        "TMN": np.full((1, 2), 273.0),
        "SM000007": np.full((1, 2), 0.25),
        "SM007028": np.full((1, 2), 0.25),
        "SM028100": np.full((1, 2), 0.25),
        "SM100289": np.full((1, 2), 0.25),
    }

    soil_type = _soil_type(base)
    neither = preprocess_noah_soil(base, soil_type=soil_type)
    np.testing.assert_array_equal(neither.snow_water, 0.0)
    np.testing.assert_array_equal(neither.snow_depth, 0.0)

    depth_only = preprocess_noah_soil(
        {**base, "SNOWH": [[0.35, 0.10]]}, soil_type=soil_type)
    np.testing.assert_allclose(depth_only.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(depth_only.snow_depth, [[0.35, 0.10]])

    swe_only = preprocess_noah_soil(
        {**base, "SNOW": [[70.0, 20.0]]}, soil_type=soil_type)
    np.testing.assert_allclose(swe_only.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(swe_only.snow_depth, [[0.35, 0.10]])

    both = preprocess_noah_soil({
        **base, "SNOW": [[70.0, 20.0]], "SNOWH": [[0.50, 0.25]],
    }, soil_type=soil_type)
    np.testing.assert_allclose(both.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(both.snow_depth, [[0.50, 0.25]])


@requires_bundle
def test_met_em_soil_oracle_is_remapped_at_noah_midpoints():
    import netCDF4

    names = ("LANDSEA", "SKINTEMP", "SST", "ST000007", "ST007028",
             "ST028100", "ST100289", "SM000007", "SM007028",
             "SM028100", "SM100289", "SNOW", "SOILTEMP", "HGT_M")
    with netCDF4.Dataset(MET_EM) as ds:
        fields = {name: np.asarray(ds.variables[name][0], dtype=np.float64)
                  for name in names}
    # SNOW is already kg m-2 in met_em; the public API accepts that spelling.
    land = fields["LANDSEA"] >= 0.5
    fields["TMN"] = np.where(
        land, fields["SOILTEMP"] - 0.0065 * fields["HGT_M"],
        fields["SOILTEMP"],
    )
    soil = preprocess_noah_soil(fields, soil_type=_soil_type(fields))
    target = np.array([0.05, 0.25, 0.70, 1.50])
    # WRF nodes are the integer-cm layer midpoints (char2int2: (0+7)/2=3,
    # (7+28)/2=17, (28+100)/2=64, (100+289)/2=194 cm), TSK at 0, TMN at 3 m.
    source = np.array([0.0, 0.03, 0.17, 0.64, 1.94, 3.0])
    valid_tmn = ((fields["TMN"] >= 170.0) & (fields["TMN"] <= 400.0)
                 & np.isfinite(fields["TMN"]))
    wrf_tmn = np.where(land & valid_tmn, fields["TMN"], soil.tsk)
    temp_nodes = np.stack([
        soil.tsk, fields["ST000007"], fields["ST007028"],
        fields["ST028100"], fields["ST100289"], wrf_tmn,
    ])
    moisture_nodes = np.stack([
        fields["SM000007"], fields["SM000007"], fields["SM007028"],
        fields["SM028100"], fields["SM100289"], fields["SM100289"],
    ])
    expected_t = np.stack([
        temp_nodes[k] + (temp_nodes[k + 1] - temp_nodes[k])
        * ((z - source[k]) / (source[k + 1] - source[k]))
        for z in target for k in [np.searchsorted(source, z) - 1]
    ])
    expected_m = np.stack([
        moisture_nodes[k] + (moisture_nodes[k + 1] - moisture_nodes[k])
        * ((z - source[k]) / (source[k + 1] - source[k]))
        for z in target for k in [np.searchsorted(source, z) - 1]
    ])
    expected_t[:, ~land] = soil.tsk[~land]
    expected_m[:, ~land] = 1.0
    t_rmse = np.sqrt(np.mean((soil.soil_temperature - expected_t) ** 2))
    m_rmse = np.sqrt(np.mean((soil.soil_moisture - expected_m) ** 2))
    assert t_rmse < 1.0e-12
    assert m_rmse < 1.0e-12


def test_stratospheric_qv_cap_pins_wrf_thresholds():
    """rh_to_mxrat1 caps (module_initialize_real.F:7490-7506).

    Both comparisons are strict: p < qv_max_p_safe (10000 Pa) and
    qv > qv_max_flag (1e-5) force qv_max_value (3e-6)
    (Registry.EM_COMMON:2306-2308).
    """
    from woof.ingest.real import _cap_stratospheric_qv

    pressure = np.array([9999.9, 10000.0, 5000.0, 5000.0, 5000.0])
    qv = np.array([2.0e-5, 2.0e-5, 1.0e-5, 1.00001e-5, 5.0e-6])
    got = _cap_stratospheric_qv(qv, pressure)
    np.testing.assert_array_equal(
        got, [3.0e-6, 2.0e-5, 1.0e-5, 3.0e-6, 5.0e-6])


def _stratospheric_snapshot(cp, ny, nx):
    """Synthetic snapshot with levels above 100 hPa carrying wet qv."""
    from woof.ingest.horiz import HorizontalSnapshot

    levels = np.array([50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 850.0,
                       1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = np.broadcast_to(
        215.0 + 72.0 * (pressure / 100000.0) ** 0.20, shape).copy()
    height = np.broadcast_to(
        -7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 50.0 * (pressure / 100000.0), shape).copy()
    u = np.broadcast_to(12.0 + 0.8 * np.log(100000.0 / pressure),
                        (levels.size, ny, nx + 1)).copy()
    v = np.broadcast_to(-3.0 + 0.4 * np.log(100000.0 / pressure),
                        (levels.size, ny + 1, nx)).copy()
    fields = {
        "TT": temperature, "GHT": height, "RH": rh, "UU": u, "VV": v,
        "PSFC": np.full((ny, nx), 96000.0),
        "T2": np.full((ny, nx), 286.0), "D2": np.full((ny, nx), 279.0),
        "U10": np.full((ny, nx + 1), 11.0),
        "V10": np.full((ny + 1, nx), -2.0),
    }
    return HorizontalSnapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels,
        fields={name: cp.asarray(value, cp.float32)
                for name, value in fields.items()},
    )


def test_vectorized_moisture_integral_is_byte_identical_to_scalar_oracle():
    from woof.ingest.real import (
        _integrate_moisture, _integrate_moisture_scalar_reference)

    rng = np.random.default_rng(7405)
    nlev, ny, nx = 9, 5, 7
    pressure_column = np.array(
        [100000.0, 92500.0, 85000.0, 70000.0, 50000.0,
         30000.0, 20000.0, 10000.0, 5000.0])
    pressure = np.broadcast_to(
        pressure_column[:, None, None], (nlev, ny, nx)).copy()
    pressure += rng.uniform(-40.0, 40.0, pressure.shape)
    temperature = rng.uniform(210.0, 305.0, pressure.shape)
    qv = rng.uniform(1.0e-6, 0.025, pressure.shape)
    base_height = np.array(
        [80.0, 700.0, 1450.0, 3000.0, 5600.0,
         8900.0, 11100.0, 14500.0, 19000.0])
    height = np.broadcast_to(
        base_height[:, None, None], pressure.shape).copy()
    height += rng.uniform(-25.0, 25.0, pressure.shape)
    # Exercise WRF's non-increasing-height branch in selected upper columns.
    height[6, 1, 2] = height[5, 1, 2] - 5.0
    height[4, 3, 5] = height[3, 3, 5]
    psfc = rng.uniform(62000.0, 101000.0, (ny, nx))
    tsfc = rng.uniform(265.0, 310.0, (ny, nx))
    qsfc = rng.uniform(1.0e-5, 0.025, (ny, nx))
    surface_height = rng.uniform(0.0, 2400.0, (ny, nx))

    expected = _integrate_moisture_scalar_reference(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height)
    for workers in (1, 2, 3, 8):
        actual = _integrate_moisture(
            qv, pressure, temperature, height, psfc, tsfc, qsfc,
            surface_height, column_workers=workers)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)


@pytest.mark.parametrize("hypsometric_opt", [1, 2])
def test_base_and_rebalance_workers_are_byte_identical(hypsometric_opt):
    from woof.ingest.real import (
        _make_real_base, _rebalance_moist_pressure)

    rng = np.random.default_rng(7427 + hypsometric_opt)
    ny, nx, nz = 11, 13, 9
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2)
    terrain = rng.uniform(0.0, 2400.0, (ny, nx))
    expected_base = _make_real_base(
        coord, terrain, 10000.0, 290.0,
        hypsometric_opt=hypsometric_opt, column_workers=1)
    dry_mass = expected_base.mub + rng.uniform(-500.0, 500.0, (ny, nx))
    qv = rng.uniform(1.0e-6, 0.025, (nz, ny, nx))
    expected_pressure = _rebalance_moist_pressure(
        expected_base.pb, qv, dry_mass, expected_base, coord,
        column_workers=1)

    for workers in (2, 3, 8, 32):
        base = _make_real_base(
            coord, terrain, 10000.0, 290.0,
            hypsometric_opt=hypsometric_opt, column_workers=workers)
        for name in ("mub", "pb", "alb", "thb", "phb", "terrain_z"):
            np.testing.assert_array_equal(
                getattr(base, name), getattr(expected_base, name))
        pressure = _rebalance_moist_pressure(
            np.full_like(expected_base.pb, -999.0), qv, dry_mass, base,
            coord, column_workers=workers)
        np.testing.assert_array_equal(pressure, expected_pressure)


@requires_gpu
@pytest.mark.gpu
def test_moisture_integral_uses_unadjusted_psfc_with_capped_source_qv():
    """integ_moist parity (module_initialize_real.F:1457, 7022, 1116).

    WRF integrates moisture with the ORIGINAL met surface pressure
    (integ_moist's psfc = p_gc level 1) and qv_gc already carrying the
    rh_to_mxrat1 stratospheric caps; only p_dts (:1482) pairs the
    sfcprs2-adjusted psfc with the resulting intq.  The float64 oracle
    below reproduces exactly that pairing.
    """
    import cupy as cp
    from woof.ingest.real import (_cap_stratospheric_qv,
                                   _integrate_moisture, initialize_real)

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    snapshot = _stratospheric_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]
    result = initialize_real(snapshot, cfg, coord, terrain,
                             source_orography=source_orography,
                             p_top=5000.0, sfcp_to_sfcp=True)

    fields = {name: np.asarray(cp.asnumpy(value), dtype=np.float64)
              for name, value in snapshot.fields.items()}
    pressure = np.broadcast_to(
        snapshot.levels_hpa[:, None, None] * 100.0,
        fields["TT"].shape).copy()
    source_qv = _saturation_mixing_ratio(fields["TT"], pressure,
                                         fields["RH"])
    assert float(source_qv[0].min()) > 1.0e-5  # the 50 hPa cap must bite
    source_qv = _cap_stratospheric_qv(source_qv, pressure)
    np.testing.assert_array_equal(source_qv[0], 3.0e-6)
    expected_pd, expected_intq, _ = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"], fields["PSFC"],
        fields["T2"], result.surface_qv, source_orography)
    np.testing.assert_array_equal(
        result.integrated_moisture_pressure, expected_intq)
    np.testing.assert_array_equal(
        result.dry_mass, result.surface_pressure - expected_intq - 5000.0)
    # The adjusted-psfc pairing the audit flagged is measurably different.
    _, adjusted_intq, _ = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"],
        result.surface_pressure, fields["T2"], result.surface_qv,
        source_orography)
    assert float(np.max(np.abs(adjusted_intq - expected_intq))) > 0.0
    # WRF calls rh_to_mxrat1 after eta interpolation and again after its
    # final hydrostatic pressure diagnosis.  The uploaded target state must
    # therefore retain the strict stratospheric cap, not just the source
    # pressure-level column used by integ_moist.
    final_qv = cp.asnumpy(result.state.qv)
    upper = result.total_pressure < 10000.0
    assert np.any(upper)
    assert np.any(final_qv[upper] == np.float32(3.0e-6))
    assert np.all(final_qv[upper] <= np.float32(1.0e-5))


def _synthetic_horizontal_snapshot(cp, ny, nx):
    from woof.ingest.horiz import HorizontalSnapshot

    levels = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = 215.0 + 72.0 * (pressure / 100000.0) ** 0.20
    temperature = np.broadcast_to(temperature, shape).copy()
    height = np.broadcast_to(-7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 50.0 * (pressure / 100000.0), shape).copy()
    u = np.broadcast_to(12.0 + 0.8 * np.log(100000.0 / pressure),
                        (levels.size, ny, nx + 1)).copy()
    v = np.broadcast_to(-3.0 + 0.4 * np.log(100000.0 / pressure),
                        (levels.size, ny + 1, nx)).copy()
    fields = {
        "TT": temperature, "GHT": height, "RH": rh, "UU": u, "VV": v,
        "PSFC": np.full((ny, nx), 96000.0),
        "T2": np.full((ny, nx), 286.0), "D2": np.full((ny, nx), 279.0),
        "U10": np.full((ny, nx + 1), 11.0),
        "V10": np.full((ny + 1, nx), -2.0),
    }
    return HorizontalSnapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels,
        fields={name: cp.asarray(value, cp.float32) for name, value in fields.items()},
    )


def test_cpu_real_state_and_lbc_materialization_do_not_call_cuda(monkeypatch):
    """The public CPU preprocessing path must work with no CUDA device.

    This pins the production failure found by the genuine GFS d01-d06 gate:
    CPU interpolation previously ended by allocating ``DomainState`` and LBC
    storage through CuPy, raising ``cudaErrorNoDevice`` before WRF export.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.ingest.lateral_bc import (
        attach_lateral_boundaries,
        build_state_lateral_boundaries,
    )
    from woof.verify.npref import np_wrf_real_vert_interp

    class Plan:
        def __init__(self, source, surface, target):
            self.source = np.asarray(source, dtype=np.float32)
            self.surface = np.asarray(surface, dtype=np.float32)
            self.target = np.asarray(target, dtype=np.float32)

        def apply(self, field, surface_value, **options):
            options.pop("values_are_finite", None)
            return np.asarray(np_wrf_real_vert_interp(
                field, surface_value, self.source, self.surface,
                self.target, **options), dtype=np.float32)

    class Backend:
        name = "cpu-test"
        array_module = np

        @staticmethod
        def float32(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def prepare_wrf_vertical(source, surface, target):
            return Plan(source, surface, target)

        @staticmethod
        def regular_plan(*_args, **_kwargs):  # pragma: no cover - contract
            raise AssertionError("unused")

        masked_nearest = regular_plan
        rotate_earth_to_grid = regular_plan
        era5_rh_to_water = regular_plan

        @staticmethod
        def receipt():
            return {"backend": "cpu-test"}

    ny, nx = 12, 13
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=10800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0, specified=True)
    snapshot = _synthetic_horizontal_snapshot(np, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 50.0 * np.sin(np.arange(nx))[None, :]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU preprocessing attempted a CuPy allocation")

    monkeypatch.setattr(cp, "zeros", forbidden)
    monkeypatch.setattr(cp, "ones", forbidden)
    monkeypatch.setattr(cp, "asarray", forbidden)

    def initialize():
        # The synthetic source column tops at 100 hPa exactly, so the
        # model top is pinned there; the DEFAULT (50 hPa) is pinned by
        # tests/test_ptop_default.py and would sit above this source.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0,
            preprocess_backend=Backend(), state_backend="cpu")

    first = initialize()
    second = initialize()
    for result in (first, second):
        assert isinstance(result.state.u, np.ndarray)
        update_diagnostics(result.state, cfg.hypsometric_opt)
        assert np.isfinite(result.state.p).all()
    times = (snapshot.valid_time,
             snapshot.valid_time + timedelta(hours=3))
    boundaries = build_state_lateral_boundaries(
        [first.state, second.state], times,
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(first.state, boundaries)
    assert first.state.lateral_boundaries is boundaries
    assert isinstance(first.state._scratch["lbc_forcing_tables"], np.ndarray)


@requires_gpu
@pytest.mark.gpu
def test_hrrr_defaults_to_rh_vertical_path_unless_use_sh_qv_is_explicit():
    import cupy as cp

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0, mp_physics=6)
    base_snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    fields = dict(base_snapshot.fields)
    levels = base_snapshot.levels_hpa
    fields["PRES"] = cp.asarray(
        np.broadcast_to(levels[:, None, None] * 100.0,
                        (levels.size, ny, nx)), cp.float32)
    fields["SPFH"] = cp.zeros((levels.size, ny, nx), cp.float32)
    fields["Q2"] = cp.zeros((ny, nx), cp.float32)
    # FLAG_SH must make a separately supplied RH field unnecessary: real.exe
    # overwrites rh_gc from the horizontally mapped SPECHUMD/TT/PRES values.
    fields.pop("RH")
    for name in ("QC", "QR", "QI", "QS", "QG"):
        fields[name] = cp.full(
            (levels.size, ny, nx), 2.5e-4, cp.float32)
    fields.pop("D2")
    snapshot = HorizontalSnapshot(
        valid_time=base_snapshot.valid_time, levels_hpa=levels, fields=fields)
    # Byte-equal source/target terrain exercises WRF's defined sfcprs2
    # no-op (exp(0) = 1); case-level wrfinput gates own the provenance
    # tripwire for regressed source-orography data.
    source_orography = np.full((ny, nx), 300.0)
    terrain = source_orography.copy()

    # p_top pinned to the synthetic column's own 100 hPa top; the flag
    # under test is use_sh_qv, not the model top.
    default_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography, p_top=10000.0)
    direct_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography, p_top=10000.0,
        use_sh_qv=True)
    default_qv = cp.asnumpy(default_result.state.qv)
    direct_qv = cp.asnumpy(direct_result.state.qv)
    np.testing.assert_array_equal(
        default_qv, np.full(default_qv.shape, np.float32(1.0e-6)))
    np.testing.assert_array_equal(
        direct_qv, np.zeros(direct_qv.shape, dtype=np.float32))
    for state in (default_result.state, direct_result.state):
        for name in ("qc", "qr", "qi", "qs", "qg"):
            value = cp.asnumpy(getattr(state, name))
            assert np.isfinite(value).all()
            assert float(value.min()) >= 0.0
            assert float(value.max()) <= 2.5e-4 * (1.0 + 2.0e-7)
            assert np.any(value > 0.0)

    bad_fields = dict(fields)
    bad_fields["QG"] = cp.full(
        (levels.size, ny, nx), -1.0e-7, cp.float32)
    bad_snapshot = HorizontalSnapshot(
        valid_time=base_snapshot.valid_time, levels_hpa=levels,
        fields=bad_fields)
    with pytest.raises(ValueError, match="non-finite or negative"):
        initialize_real(
            bad_snapshot, cfg,
            make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                                eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0)


@requires_gpu
@pytest.mark.gpu
def test_real_init_builds_nonnegative_balanced_fp32_domain_state():
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]
    result = initialize_real(snapshot, cfg, coord, terrain,
                             source_orography=source_orography,
                             p_top=10000.0, sfcp_to_sfcp=True)
    state = result.state

    assert state.thp.dtype == cp.float32 and state.qv.dtype == cp.float32
    assert bool(cp.isfinite(state.thp).all()) and bool(cp.isfinite(state.php).all())
    assert float(state.qv.min()) >= 1.0e-6
    residual = hydrostatic_residual(result)
    assert residual.shape == (cfg.ny, cfg.nx)
    assert residual.max() < 2.0e-2

    expected_psfc = surface_pressure_from_surface(
        np.full((ny, nx), 96000.0), source_orography, terrain,
        np.full((ny, nx), 286.0), result.surface_qv,
    )
    np.testing.assert_allclose(result.surface_pressure, expected_psfc,
                               rtol=0.0, atol=2.0e-9)
    update_diagnostics(state)
    np.testing.assert_allclose(cp.asnumpy(state.p), result.total_pressure,
                               rtol=4.0e-4, atol=4.0)

    baseline = residual[0, 0]
    state.php[1, 0, 0] += cp.float32(1.0)
    assert hydrostatic_residual(result)[0, 0] > baseline + 0.5


@requires_gpu
@pytest.mark.gpu
def test_real_init_parallel_cpu_backend_is_worker_stable_and_matches_cuda():
    import cupy as cp

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=1800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0)
    snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]

    def initialize(**options):
        # p_top pinned to the synthetic column's own 100 hPa top.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0,
            **options)

    cuda = initialize(preprocess_backend="cuda")
    try:
        cpu_serial = initialize(
            preprocess_backend="cpu", preprocess_workers=1)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")
    cpu_parallel = initialize(
        preprocess_backend="cpu", preprocess_workers=8)

    setup_names = (
        "surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
        "total_pressure", "total_geopotential", "total_specific_volume",
        "integrated_moisture_pressure",
    )
    state_names = ("mup", "thp", "php", "qv", "u", "v", "w")
    for name in setup_names:
        np.testing.assert_array_equal(
            getattr(cpu_parallel, name), getattr(cpu_serial, name))
    for name in state_names:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(cpu_parallel.state, name)),
            cp.asnumpy(getattr(cpu_serial.state, name)))

    for name in ("mup", "thp", "qv", "u", "v", "w"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(cpu_parallel.state, name)),
            cp.asnumpy(getattr(cuda.state, name)),
            rtol=3.0e-5, atol=5.0e-3)
    np.testing.assert_allclose(
        cp.asnumpy(cpu_parallel.state.php), cp.asnumpy(cuda.state.php),
        rtol=3.0e-5, atol=2.0e-2)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mp_physics", (6, 8))
def test_real_init_cpu_backend_handles_hrrr_hydrometeor_inventory(
        mp_physics, monkeypatch, tmp_path):
    import cupy as cp

    if mp_physics == 8:
        monkeypatch.setenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", "1")
        monkeypatch.setenv("WOOF_THOMPSON_TABLE_ROOT", str(tmp_path))

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=1800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0, mp_physics=mp_physics,
        moist_cq=mp_physics == 8, top_lid=mp_physics != 8)
    base = _synthetic_horizontal_snapshot(cp, ny, nx)
    fields = dict(base.fields)
    fields["PRES"] = cp.asarray(
        np.broadcast_to(
            base.levels_hpa[:, None, None] * 100.0,
            (base.levels_hpa.size, ny, nx)), dtype=cp.float32)
    fields["SPFH"] = cp.zeros(
        (base.levels_hpa.size, ny, nx), dtype=cp.float32)
    fields["Q2"] = cp.zeros((ny, nx), dtype=cp.float32)
    fields.pop("RH")
    fields.pop("D2")
    for index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        fields[name] = cp.full(
            (base.levels_hpa.size, ny, nx), index * 2.5e-5,
            dtype=cp.float32)
    snapshot = HorizontalSnapshot(
        valid_time=base.valid_time, levels_hpa=base.levels_hpa,
        fields=fields)
    terrain = np.full((ny, nx), 300.0)
    # Byte-equal source/target terrain is WRF's defined sfcprs2 no-op.
    source_orography = terrain

    def initialize(backend):
        # p_top pinned to the synthetic column's own 100 hPa top.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, use_sh_qv=True,
            p_top=10000.0,
            preprocess_backend=backend,
            preprocess_workers=8 if backend == "cpu" else None)

    cuda = initialize("cuda")
    try:
        cpu = initialize("cpu")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")
    for name in ("qc", "qr", "qi", "qs", "qg"):
        actual = cp.asnumpy(getattr(cpu.state, name))
        expected = cp.asnumpy(getattr(cuda.state, name))
        assert np.isfinite(actual).all()
        assert np.all(actual >= 0.0)
        np.testing.assert_allclose(
            actual, expected, rtol=3.0e-5, atol=5.0e-8)
    if mp_physics == 8:
        # Since 66ca6e205 an mp=8 cold start seeds analysed rain and ice
        # with real.exe's make_RainNumber / make_IceNumber
        # (module_initialize_real.F:4840-4852) instead of starting both
        # numbers at zero.  Every cell here carries qr and qi and no
        # analysed number, so every cell is seeded, and the two
        # preparation backends must seed the same numbers.
        for name in ("ni", "nr"):
            actual = cp.asnumpy(getattr(cpu.state, name))
            expected = cp.asnumpy(getattr(cuda.state, name))
            assert np.isfinite(actual).all() and np.all(actual > 0.0), name
            np.testing.assert_allclose(
                actual, expected, rtol=3.0e-5, atol=0.0, err_msg=name)


def test_real74_d01_30min_gate_admits_complete_no_pbl_operator():
    from woof.config import validate_run_config
    from woof.verify.cases.real74_d01 import config, phase3_config

    legacy = validate_run_config(config(run_seconds=1800.0))
    assert legacy.km_opt == 4
    assert legacy.bl_pbl_physics == 0

    supported = validate_run_config(phase3_config(run_seconds=1800.0))
    assert supported.km_opt == 4
    assert supported.bl_pbl_physics == 1


# ---------------------------------------------------------------------------
# mp_physics=28 (Thompson aerosol-aware) real-data ingest.
#
# Before this lane existed, woof/ingest/real.py named mp_physics 1, 6, 8, 10
# and 18 at three separate sites and 28 at none of them.  The consequence was
# not an error: an mp=28 run from a cloudy HRRR analysis silently produced a
# condensate-free initial state, because the five analyzed species were never
# declared required (:1106), never shape-checked (:1150) and never vertically
# interpolated (:1390).  Each of these tests fails against that tree.
#
# WRF authority, v4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28:
#   Registry/Registry.EM_COMMON:3036   thompsonaero package membership
#   Registry/Registry.EM_COMMON:3024   thompson, for the moist-list identity
#   dyn_em/module_initialize_real.F:2332-2345   aer_init_opt=0, 3-D aerosol
#   dyn_em/module_initialize_real.F:4501-4510   aer_init_opt=0, 2-D emission
#   dyn_em/module_initialize_real.F:2735-2736   the FATAL ArWen deviates from
#   phys/module_mp_thompson.F:493,:531          thompson_init's MAXVAL tests
# ---------------------------------------------------------------------------


# Both fixtures use the same independent NumPy preprocessing ABI.
from test_real_init_dry_slot import (
    _ReferencePreprocessBackend, _ReferenceVerticalPlan)


def _SYNTHETIC_IF_28(mp_physics):
    """mp=28 tests about numbers or the ingest name the synthetic aerosol profile.

    Since 2.9.7 'auto' needs the 225 MB WIF dataset or refuses by name (the
    suite keeps its download offline), so a test that is not about the aerosol
    source names the profile it reads, as a user would.
    """
    return {"mp28_aerosol_source": "synthetic"} if int(mp_physics) == 28 else {}


def _analyzed_hrrr_real_init(
        mp_physics, *, state_backend="cpu", terrain_m=0.0,
        drop=(), reshape=None, wif_grid_latlon=None, wif_valid_date=None,
        analyzed_species=None, horizontal_operators=None, init_kwargs=None,
        preprocess_backend=None, shape=(2, 3), cloud_water=None,
        landmask=1.0, extra_fields=None, **config_overrides):
    """One decoded-native-HRRR real initialization, never a fabricated state.

    ``drop`` removes analyzed species from the decoded snapshot and
    ``reshape`` truncates one of them, so the required-field and shape gates
    can be exercised for real rather than asserted about.  ``init_kwargs``
    reach :func:`initialize_real` unchanged.  ``cloud_water`` replaces the
    analyzed QC with one value per source level (kg/kg), and ``landmask``
    is the target LANDMASK every production door passes (a scalar fills
    the grid; None passes none).  ``extra_fields`` are added to the
    decoded snapshot as they are, for analysed number fields.
    """

    ny, nx = shape
    nz = 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    height += terrain_m
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    row_index = np.arange(ny, dtype=np.float32)[None, :, None]
    column_index = np.arange(nx, dtype=np.float32)[None, None, :]
    analyzed = {}
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        value = np.asarray(
            species_index * 1.0e-6
            * (1.0 + level_index + 0.25 * row_index
               + 0.125 * column_index), dtype=np.float32)
        # One exact zero so a nonzero-mask fingerprint is a real discriminator.
        value[0, 0, 0] = np.float32(0.0)
        analyzed[name] = value
    if cloud_water is not None:
        analyzed["QC"] = np.ascontiguousarray(np.broadcast_to(
            np.asarray(cloud_water, dtype=np.float32)[:, None, None],
            (levels.size, ny, nx)))
    for name in drop:
        analyzed.pop(name)
    if reshape is not None:
        analyzed[reshape] = analyzed[reshape][:, :, :-1].copy()
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
        **analyzed,
    }
    if extra_fields is not None:
        fields.update({name: np.broadcast_to(
            np.asarray(value, dtype=np.float32),
            (ny, nx) if name.endswith("_SFC") else (levels.size, ny, nx)
        ).copy() for name, value in extra_fields.items()})
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields,
        horizontal_operators=horizontal_operators)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0,
        dt=30.0, run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True,
        terrain_opt=1, mp_physics=mp_physics, **config_overrides)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.full((ny, nx), float(terrain_m), dtype=np.float64)
    init_kwargs = dict(init_kwargs or {})
    if landmask is not None:
        init_kwargs.setdefault("landmask", np.broadcast_to(
            np.asarray(landmask, dtype=np.float32), (ny, nx)))
    result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0, use_sh_qv=True,
        preprocess_backend=(_ReferencePreprocessBackend()
                            if preprocess_backend is None
                            else preprocess_backend),
        state_backend=state_backend,
        wif_grid_latlon=wif_grid_latlon, wif_valid_date=wif_valid_date,
        analyzed_species=analyzed_species, **init_kwargs)
    return result, cfg


def _host_array(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


@pytest.mark.parametrize("mp_physics", (28, 38))
def test_mp28_real_ingest_retains_every_analyzed_hydrometeor(mp_physics,
                                                             monkeypatch):
    """mp=28's Registry moist list is character for character mp=8's.

    Registry.EM_COMMON:3024 gives ``thompson`` moist:qv,qc,qr,qi,qs,qg and
    :3036 gives ``thompsonaero`` the same six; the aerosol-aware scheme adds
    only ``scalar:`` members.  So all five decoded HRRR mass species must
    survive to the state, exactly as for mp=8.  Against the tree that omitted
    28 from the three mp tuples this state was identically zero and nothing
    raised.
    """
    if mp_physics == 38:
        # initialize_real asks the machine floor (validate_run_preparation)
        # first, and under WRF's write_thompson_mp38table=false that floor
        # refuses where the 674 MB native hail table is not staged, so this
        # case passed only on machines that had staged it.  Ingest never
        # reads that table; the floor and the table policy have their own
        # tests (tests/test_thompson_hail_port.py).
        import woof.ingest.real as real_ingest
        monkeypatch.setattr(real_ingest, "validate_run_preparation",
                            lambda cfg: None)
    result, _ = _analyzed_hrrr_real_init(mp_physics, mp28_aerosol_source="synthetic")
    state = result.state

    evidence = result.hydrometeor_initialization
    if mp_physics == 38:
        # Unlike 28, native real.exe leaves all five GH number/volume
        # moments at their allocation zeros, even where mass is present.
        for name in ("nc", "nr", "ni", "ng", "qvolg"):
            assert not _host_array(getattr(state, name)).view(np.uint32).any(), name
    # v2 since the 1.4.1 merge: the HRRR vertical-disposition work on the
    # release line moved the correspondence schema while the port was off
    # the line, and woof/ingest/real.py now emits v2 for every scheme.  The
    # mp=28 claim this test makes -- that a 28 run retains every analyzed
    # hydrometeor -- is unchanged; only the schema stamp moved.
    assert evidence["schema"] == "gpuwm-real-hydrometeor-correspondence-v2"
    assert evidence["mp_physics"] == mp_physics
    assert set(evidence["retained_correspondence"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert evidence["discarded_source_species"] == {}
    for source_name, state_name in sorted(
            evidence["retained_correspondence"].items()):
        live = _host_array(getattr(state, state_name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert np.isfinite(live).all()
        assert live.min() >= 0.0
        assert np.count_nonzero(live) > 0, (
            f"{source_name}->{state_name} carried no analyzed mass")
        assert live.max() <= float(
            evidence["decoded_source_species"][source_name]["maximum"]
        ) * (1.0 + 4.0e-7)

    # An mp=8 initialization of the same snapshot is the control: the two
    # must agree bit for bit on the mass species, because the mass path is
    # the same code and the only difference is which scalars exist.
    control, _ = _analyzed_hrrr_real_init(8)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        np.testing.assert_array_equal(
            _host_array(getattr(state, name)),
            _host_array(getattr(control.state, name)))


def test_the_moist_pressure_recurrence_loads_total_water():
    """WRF's initial ``p'`` recurrence carries every moist species.

    ``module_initialize_real.F:3913-3916`` accumulates
    ``qtot = sum(moist(i,kk,j,im), im = PARAM_FIRST_SCALAR, num_3d_m)`` --
    ``num_3d_m = num_moist`` (:1856) -- and the hydrometeor ``vert_interp``
    calls that fill those slots (:1862-1982) all run before the recurrence
    at :3908, so the analyzed condensate is loaded there, not just vapour.
    Both ``qvf2 = 1./(1.+qtot)`` and ``qvf1 = qtot*qvf2`` are built from it,
    and the downward leg (:3931-3935) repeats the sum on the half-level
    average.

    The oracle is the recurrence itself replayed on this initialization's
    own state: it reads only ``qtot``, ``mu``, the base state and the
    vertical coordinate (``pressure_guess`` supplies shape and the finite
    check), so feeding it the ingested qv+qc+qr+qi+qs+qg must reproduce
    ``total_pressure`` exactly.  Feeding it vapour alone -- what a
    vapour-only ingest produces -- must NOT, and the second assertion pins
    that the two really are distinguishable here.
    """
    from woof.ingest.real import _make_real_base, _rebalance_moist_pressure

    result, cfg = _analyzed_hrrr_real_init(6)
    coord = make_vertical_coord(
        cfg.nz, hybrid_opt=2, etac=0.2,
        eta_levels=np.linspace(1.0, 0.0, cfg.nz + 1))
    base = _make_real_base(
        coord, np.zeros((cfg.ny, cfg.nx)), 10000.0, cfg.base_temp,
        hypsometric_opt=cfg.hypsometric_opt)
    vapour = _host_array(result.state.qv).astype(np.float64)
    # WRF's own accumulation order: the moist package's qc,qr,qi,qs,qg
    # after qv, summed left to right.
    condensate = None
    for name in ("qc", "qr", "qi", "qs", "qg"):
        species = _host_array(getattr(result.state, name)).astype(np.float64)
        assert np.count_nonzero(species) > 0, name
        condensate = species if condensate is None else condensate + species
    dry_mass = np.asarray(result.dry_mass, dtype=np.float64)
    pressure = np.asarray(result.total_pressure, dtype=np.float64)

    total_water = _rebalance_moist_pressure(
        pressure.copy(), vapour + condensate, dry_mass, base, coord)
    vapour_only = _rebalance_moist_pressure(
        pressure.copy(), vapour, dry_mass, base, coord)
    np.testing.assert_array_equal(pressure, total_water)
    # The condensate is worth several Pa on this column, so the equality
    # above is a real discriminator and not two ways of writing qv.
    assert np.abs(total_water - vapour_only).max() > 1.0


def test_mp28_real_ingest_requires_the_analyzed_inventory_by_name():
    """The :1106 required-field gate, exercised rather than asserted."""
    with pytest.raises(KeyError, match=r"missing real-data field\(s\).*QI"):
        _analyzed_hrrr_real_init(28, drop=("QI",))


def test_mp28_real_ingest_shape_checks_the_analyzed_inventory():
    """The :1150 mass-shape gate.

    Without 28 in that tuple a truncated QG reached the vertical plan and
    failed later, inside the interpolation, with a message about neither the
    field nor the scheme.
    """
    with pytest.raises(ValueError, match="mass-field shapes do not match"):
        _analyzed_hrrr_real_init(28, reshape="QG")


def test_the_analyzed_inventory_tuple_and_its_moist_packages_agree():
    """The admission tuple and the retention table answer one question.

    ``HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS`` decides whether the decoder
    SUPPLIES the analyzed inventory for an id;
    ``HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE`` decides which of it that
    id's Registry package can HOLD.  Those are the same question asked
    twice, so the ids must be identical -- an id in the tuple with no
    package row would raise a bare ``KeyError`` mid-initialization, and a
    package row for an id outside the tuple is a retention rule nothing
    reads.  The tuple stays a literal so the repo-wide admission census can
    see the site; this is what keeps the literal accurate.
    """
    assert (tuple(sorted(HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE))
            == tuple(sorted(HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS)))
    assert (len(set(HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS))
            == len(HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS))
    for mp_physics, package in sorted(
            HRRR_ANALYZED_HYDROMETEOR_MOIST_PACKAGE.items()):
        retained = package["retained"]
        assert retained, (
            f"mp={mp_physics} would be admitted and retain nothing")
        # Retention is a SUBSEQUENCE of the decoded inventory, in the decoded
        # order: the receipt's partition and the interpolation order both
        # depend on it, and a reordered or invented name would silently
        # change which field a decoded species lands in.
        assert tuple(
            name for name in DECLARED_ANALYZED_HYDROMETEORS if name in retained
        ) == tuple(retained), f"mp={mp_physics} retention is not decoded order"
        # A named scheme cites its own capability row; it has no WRF
        # Registry package.  FAST_SBM's package line lives in WRF's
        # registry.sbm (:318).
        # UDM (mp=27) cites its package line in the WRF 4.7.1 Registry by version.
        assert package["registry_citation"].startswith(
            ("Registry/Registry.EM_COMMON:", "woof.microphysics_schemes:",
             "Registry/registry.sbm:", "WRF 4.7.1 Registry/Registry.EM_COMMON:"))


def test_mp50_real_ingest_retains_the_condensate_p3_can_hold():
    """P3 one-category is a first-class real-data selector.

    Registry.EM_COMMON:3038 gives ``p3_1category`` moist:qv,qc,qr,qi --
    strictly RICHER than Kessler's moist:qv,qc,qr at :3015, which this
    module has always admitted -- so by the tuple's own stated rule mp=50
    belongs in it, and real.exe agrees: module_initialize_real.F:1859-1977
    interpolates a decoded species exactly when its P_Q* is in the active
    ``num_moist`` package, which for P3 is QC, QR and QI.

    MEASURED on the tree that omitted 50: this same decoded snapshot
    produced qc/qr/qi identically zero with ``hydrometeor_initialization``
    empty and nothing raised -- a condensate-free initial state from a
    cloudy analysis, verbatim the defect the tuple exists to prevent.
    """
    result, _ = _analyzed_hrrr_real_init(50)
    state = result.state

    evidence = result.hydrometeor_initialization
    assert evidence["schema"] == "gpuwm-real-hydrometeor-correspondence-v2"
    assert evidence["mp_physics"] == 50
    assert set(evidence["retained_correspondence"]) == {"QC", "QR", "QI"}
    for source_name, state_name in sorted(
            evidence["retained_correspondence"].items()):
        live = _host_array(getattr(state, state_name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert np.isfinite(live).all()
        assert live.min() >= 0.0
        assert np.count_nonzero(live) > 0, (
            f"{source_name}->{state_name} carried no analyzed mass")
        assert live.max() <= float(
            evidence["decoded_source_species"][source_name]["maximum"]
        ) * (1.0 + 4.0e-7)

    # mp=10 is the control: the mass path is one piece of code and the only
    # difference is which species the package holds, so the three P3 keeps
    # must be bit for bit what a six-species scheme gets from the same
    # snapshot.  A P3 arm that quietly interpolated differently would pass
    # every assertion above and fail this one.
    control, _ = _analyzed_hrrr_real_init(10)
    for name in ("qc", "qr", "qi"):
        np.testing.assert_array_equal(
            _host_array(getattr(state, name)),
            _host_array(getattr(control.state, name)))

    # P3's Registry scalars are source-absent, so they arrive at exact FP32
    # zero and the scheme owns their first update -- it floors nitot at
    # nsmall before any mean size is taken (module_mp_p3.F:2572-2573) and
    # zeroes an unsupported rime pair in calc_bulkRhoRime (:6799-6813).
    for name in ("ni", "nr", "qir", "qib"):
        live = _host_array(getattr(state, name))
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} is not exact FP32 zero")


def test_mp50_real_ingest_discards_snow_and_graupel_by_name():
    """P3 has no qs and no qg, and the receipt says so rather than omitting.

    An mp=50 state allocates no snow and no graupel field at all
    (woof/core/state.py:464-497), because P3 carries ONE ice category and
    predicts rime mass fraction and rime density instead of splitting the
    frozen mass.  So two of the five decoded species cannot be kept -- and
    a decoded species that was deliberately dropped must stay
    distinguishable from one that was accidentally lost.
    """
    result, _ = _analyzed_hrrr_real_init(50)
    evidence = result.hydrometeor_initialization

    assert set(evidence["decoded_source_species"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert set(evidence["discarded_source_species"]) == {"QS", "QG"}
    for name, policy in sorted(evidence["discarded_source_species"].items()):
        assert policy["policy"] == (
            "discard-source-species-absent-from-active-moist-package")
        assert policy["wrf_commit"] == (
            "d66e442fccc04111067e29274c9f9eaccc3cef28")
        # The citation names P3's OWN package line, not Kessler's, because
        # which package did the discarding is the content of the claim.
        assert policy["registry_citation"] == (
            "Registry/Registry.EM_COMMON:3038")
        assert policy["source"]["nonzero_count"] > 0, (
            f"{name} was discarded from an empty source and proves nothing")
    assert getattr(result.state, "qs", None) is None
    assert getattr(result.state, "qg", None) is None
    # The partition is total and disjoint, which is what every downstream
    # receipt validator checks before it will accept the preparation.
    assert (set(evidence["retained_correspondence"])
            | set(evidence["discarded_source_species"])
            == set(evidence["decoded_source_species"]))
    assert not (set(evidence["retained_correspondence"])
                & set(evidence["discarded_source_species"]))


def test_mp50_real_ingest_requires_the_analyzed_inventory_by_name():
    """The required-field gate, exercised for P3 rather than asserted.

    QI is a species P3 keeps, so its absence has to be named at the
    required-field check.  Without 50 in the tuple the field was never
    declared required and the run proceeded to a condensate-free state.
    """
    with pytest.raises(KeyError, match=r"missing real-data field\(s\).*QI"):
        _analyzed_hrrr_real_init(50, drop=("QI",))


def test_mp50_real_ingest_requires_even_the_species_it_discards():
    """QS is required, then discarded -- the Kessler rule, applied to P3.

    real.exe requires the decoded inventory and only then drops what the
    active package lacks, so a truncated analysis is a broken analysis for
    mp=50 exactly as it is for every other admitted id.  Silently accepting
    a missing QS because P3 has no snow would make an incomplete decode
    indistinguishable from a complete one.
    """
    with pytest.raises(KeyError, match=r"missing real-data field\(s\).*QS"):
        _analyzed_hrrr_real_init(50, drop=("QS",))


def test_mp50_real_ingest_shape_checks_the_analyzed_inventory():
    """The mass-shape gate, for a species P3 discards.

    Without 50 in the tuple a truncated QG reached neither the shape check
    nor the interpolation for mp=50: nothing looked at it at all.
    """
    with pytest.raises(ValueError, match="mass-field shapes do not match"):
        _analyzed_hrrr_real_init(50, reshape="QG")


def test_mp9_real_ingest_retains_every_analyzed_hydrometeor():
    """Milbrandt-Yau is a first-class real-data selector.

    Registry.EM_COMMON:3025 gives ``milbrandt2mom``
    moist:qv,qc,qr,qi,qs,qg,qh -- the first admitted package WIDER than
    the decoded inventory -- so all five decoded HRRR mass species must
    survive to the state and the receipt must discard nothing.  Before 9
    joined the tuple, this same snapshot produced qc/qr/qi/qs/qg
    identically zero with ``hydrometeor_initialization`` empty and nothing
    raised: the mp=28 omission failure, back a third time
    (``test_mp9_reverted_tuple_reproduces_the_condensate_free_start``
    below keeps that measurement).
    """
    result, _ = _analyzed_hrrr_real_init(9)
    state = result.state

    evidence = result.hydrometeor_initialization
    assert evidence["schema"] == "gpuwm-real-hydrometeor-correspondence-v2"
    assert evidence["mp_physics"] == 9
    assert set(evidence["retained_correspondence"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert evidence["discarded_source_species"] == {}
    for source_name, state_name in sorted(
            evidence["retained_correspondence"].items()):
        live = _host_array(getattr(state, state_name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert np.isfinite(live).all()
        assert live.min() >= 0.0
        assert np.count_nonzero(live) > 0, (
            f"{source_name}->{state_name} carried no analyzed mass")
        assert live.max() <= float(
            evidence["decoded_source_species"][source_name]["maximum"]
        ) * (1.0 + 4.0e-7)

    # mp=10 is the control: the mass path is one piece of code and the only
    # difference is which scalars sit beside it, so the five species must
    # be bit for bit what Morrison gets from the same snapshot.
    control, _ = _analyzed_hrrr_real_init(10)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        np.testing.assert_array_equal(
            _host_array(getattr(state, name)),
            _host_array(getattr(control.state, name)))


def test_mp9_real_ingest_leaves_hail_and_the_moments_at_exact_zero():
    """The package members with no decoded source begin at exact FP32 zero.

    The native HRRR decoder reads exactly QC/QI/QR/QS/QG
    (woof/ingest/hrrr.py:47) -- no hail mass anywhere -- and real.exe's
    own QH arm (dyn_em/module_initialize_real.F:1979-1997) runs only under
    ``flag_qh``, which a source that supplies no QH never raises, so stock
    real.exe leaves QH at zero for this source too.  The six number
    moments (scalar:qnc,qnr,qni,qns,qng,qnh, Registry.EM_COMMON:3025) are
    source-absent exactly like mp=8/10/16/18/28's scalars.  The scheme is
    defined at that zero: its moment-consistency pass prescribes each
    number from its own mass wherever Qx>epsQ and Nx<epsN
    (phys/module_mp_milbrandt2mom.F:1444-1530), so exact zero is the value
    that hands the scheme its own prescribed size distributions on the
    first call.
    """
    result, _ = _analyzed_hrrr_real_init(9)
    state = result.state
    for name in ("qh", "nc", "nr", "ni", "ns", "ng", "nh"):
        live = _host_array(getattr(state, name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} is not exact FP32 zero")


def test_mp9_real_ingest_requires_the_analyzed_inventory_by_name():
    """The required-field gate, exercised for mp=9 rather than asserted.

    Without 9 in the tuple no species was ever declared required and the
    run proceeded to a condensate-free state; with it, a truncated decode
    must be named at the required-field check.
    """
    with pytest.raises(KeyError, match=r"missing real-data field\(s\).*QI"):
        _analyzed_hrrr_real_init(9, drop=("QI",))


def test_mp9_real_ingest_shape_checks_the_analyzed_inventory():
    """The mass-shape gate, for a species Milbrandt-Yau retains.

    Without 9 in the tuple a truncated QG reached neither the shape check
    nor the interpolation for mp=9: nothing looked at it at all.
    """
    with pytest.raises(ValueError, match="mass-field shapes do not match"):
        _analyzed_hrrr_real_init(9, reshape="QG")


def test_mp9_reverted_tuple_reproduces_the_condensate_free_start(monkeypatch):
    """The defect this lane fixed, kept runnable.

    With 9 reverted out of the admission tuple, the identical cloudy
    snapshot initializes an mp=9 forecast with every mass species
    identically zero, an empty receipt, and nothing raised -- the exact
    silent failure the ONE-tuple design exists to prevent, reproduced
    through the tuple rather than asserted from memory.  If this test ever
    fails, admission stopped flowing through
    HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS and the census-visible literal is
    no longer the single point of truth.
    """
    from woof.ingest import real as real_module

    monkeypatch.setattr(
        real_module, "HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS",
        tuple(value for value in real_module.HRRR_ANALYZED_HYDROMETEOR_MP_PHYSICS
              if value != 9))
    result, _ = _analyzed_hrrr_real_init(9)
    assert result.hydrometeor_initialization == {}
    for name in ("qc", "qr", "qi", "qs", "qg"):
        live = _host_array(getattr(result.state, name))
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} unexpectedly carried analyzed mass")


def test_mp28_real_ingest_leaves_the_aerosols_for_the_init_hook():
    """The aerosol fields must arrive EXACTLY zero, and say so.

    Zero here is not "unset": ``thompson_init`` decides whether to install
    its synthetic CCN/IN profile by testing MAXVAL(nwfa) < eps
    (phys/module_mp_thompson.F:493) and MAXVAL(nifa) < eps (:531).  Any
    nonzero placeholder written by the ingest would flip those tests and
    permanently suppress the profile, leaving the aerosol-aware physics inert
    with no error anywhere.  WRF's own initializer writes exactly 0.0 for
    both 3-D fields (dyn_em/module_initialize_real.F:2332-2345) and both 2-D
    emissions (:4501-4510) under aer_init_opt=0.
    """
    result, cfg = _analyzed_hrrr_real_init(28, mp28_aerosol_source="synthetic")
    state = result.state
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        live = _host_array(getattr(state, name))
        assert live.dtype == np.float32
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} is not exact FP32 zero")

    receipt = result.aerosol_initialization
    assert receipt["policy"] == (
        "aer-init-opt-0-zero-then-thompson-init-synthetic-profile")
    assert receipt["registry_citation"] == (
        "Registry/Registry.EM_COMMON:3036")
    assert receipt["real_citation"] == (
        "dyn_em/module_initialize_real.F:2332-2345")
    assert receipt["wrf_real_refuses_this_configuration"] == (
        "dyn_em/module_initialize_real.F:2735-2736")
    assert receipt["deferred_to"] == (
        "woof.core.physics.initialize_physics -> "
        "woof.core.microphysics.microphysics_init")
    assert receipt["awaiting_profile_fill"] is True
    assert receipt["aer_init_opt"] == 0 and receipt["wif_input_opt"] == 0
    assert set(receipt["not_initialized_here"]) == {
        "nwfa", "nifa", "nwfa2d", "nifa2d"}
    # The number moments are not in this receipt: the cold-start closure
    # sets them from the analyzed mass, and their receipt is
    # hydrometeor_initialization["cold_start_moment_closure"].
    fingerprints = receipt["source_absent_state_fields"]
    assert set(fingerprints) == {"nwfa", "nifa", "nwfa2d", "nifa2d"}
    assert all(item["nonzero_count"] == 0 for item in fingerprints.values())

    # Every other scheme carries an empty aerosol receipt, so the field can
    # never be read as "this run has aerosol provenance" when it does not.
    control, _ = _analyzed_hrrr_real_init(8)
    assert control.aerosol_initialization == {}
    assert cfg.mp_physics == 28


def test_mp28_real_ingest_refuses_an_aerosol_source_it_cannot_read():
    """WIF selectors fail closed inside the ingest, not only in the config.

    ``initialize_real`` is reachable with a RunConfig that never went through
    ``validate_run_config``; accepting wif_input_opt=1 there would promise a
    metgrid WIF stream this module cannot read and then hand the microphysics
    an all-zero aerosol field as if it were the requested climatology.
    """
    # RE-BASELINED (development branch).  Two of the four rows moved and two
    # did not, and the split is the whole point: the CLIMATOLOGY selectors
    # are now implemented, so refusing them would be refusing the default;
    # the FIRST-GUESS and qnbca selectors are still unimplemented
    # capabilities, so they still refuse by name.
    for overrides, expect in (
            ({"wif_input_opt": 2}, "wif_input_opt"),
            ({"aer_init_opt": 2}, "aer_init_opt")):
        with pytest.raises(NotImplementedError) as caught:
            _analyzed_hrrr_real_init(28, **overrides)
        message = str(caught.value)
        assert expect in message
    # Half a climatology selection is refused as a mixed selection, not as
    # an unimplemented one -- ArWen will not guess which half was meant.
    for overrides in ({"wif_input_opt": 1}, {"aer_init_opt": 1}):
        with pytest.raises(NotImplementedError) as caught:
            _analyzed_hrrr_real_init(28, **overrides)
        assert "MIXED aerosol-source selection" in str(caught.value)

    # Not a blanket refusal: the same selectors are inert under mp=8 and the
    # ingest must not start policing another scheme's namelist.
    result, _ = _analyzed_hrrr_real_init(8)
    assert result.aerosol_initialization == {}


def test_mp28_real_ingest_does_not_call_the_profile_fill_itself():
    """The ingest is not allowed to be the caller of ``microphysics_init``.

    Proven structurally rather than by comment: the module source contains no
    call, and the state it returns is empty of aerosol.  The fill belongs to
    ``woof.core.physics.initialize_physics``, which BOTH production
    real-data front doors reach with the state this function returns
    (``woof/ingest/hrrr_physics.py``), so a call here would be the second
    one.
    """
    import inspect
    import re

    from woof.ingest import hrrr_physics
    import woof.ingest.real as real_module

    source = inspect.getsource(real_module)
    assert not re.search(r"\bmicrophysics_init\s*\(", source), (
        "woof/ingest/real.py must not call microphysics_init")
    assert not re.search(r"\bthompson_aerosol_init_fill\s*\(", source)

    # The named successor really is on this path.
    physics_source = inspect.getsource(hrrr_physics)
    assert "initialize_physics(" in physics_source


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_then_microphysics_init_fills_exactly_once():
    """End to end: real ingest -> physics init -> WRF's synthetic profile.

    This is the ownership proof the ingest lane owes.  The ingest leaves the
    aerosol at exact zero; the FIRST ``microphysics_init`` performs both
    fills and reports ``{'ccn': True, 'in': True}``; a SECOND call reports
    ``{'ccn': False, 'in': False}`` and changes not one bit, because
    ``thompson_init``'s own MAXVAL guard (module_mp_thompson.F:493/:531) now
    sees a populated field.  So the fill happens exactly once on this path,
    and an ingest that had populated the aerosol itself would be preserved
    rather than overwritten.

    The values are checked against WRF's own parameters rather than merely
    for being nonzero: naCCN1=50.0E6 and naCCN0=300.0E6 (:96-97) bound nwfa
    to [5.0e7, 3.5e8], naIN1=0.5E6 and naIN0=1.5E6 (:94-95) bound nifa to
    [5.0e5, 2.0e6], and both profiles decrease monotonically upward from a
    terrain below the 1000 m ``h_01`` breakpoint (:500-506).
    """
    import cupy as cp

    from woof.core.microphysics import microphysics_init

    result, cfg = _analyzed_hrrr_real_init(
        28, state_backend="cuda", terrain_m=250.0)
    state = result.state
    assert isinstance(state.nwfa, cp.ndarray)
    for name in ("nwfa", "nifa", "nwfa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0

    first = microphysics_init(state, cfg)
    assert first == {"thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    nwfa2d = cp.asnumpy(state.nwfa2d)

    assert nwfa.dtype == np.float32 and nwfa.shape == (8, 2, 3)
    assert nifa.dtype == np.float32 and nifa.shape == (8, 2, 3)
    assert nwfa2d.dtype == np.float32 and nwfa2d.shape == (2, 3)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert 0.5e6 <= nifa.min() and nifa.max() <= 2.0e6
    assert nwfa2d.min() > 0.0 and np.isfinite(nwfa2d).all()
    # nifa2d is never assigned anywhere in module_mp_thompson.F; it is not
    # even a thompson_init dummy argument.  Zero is WRF's behaviour.
    assert int(cp.asnumpy(state.nifa2d).view(np.uint32).max()) == 0
    for column in ((0, 0), (1, 2)):
        profile = nwfa[:, column[0], column[1]]
        assert np.all(np.diff(profile) <= 0.0)
        assert np.all(np.diff(nifa[:, column[0], column[1]]) <= 0.0)

    second = microphysics_init(state, cfg)
    assert second == {"thompson_aerosol_profile": {"ccn": False, "in": False}}
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), nwfa)
    np.testing.assert_array_equal(cp.asnumpy(state.nifa), nifa)
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa2d), nwfa2d)


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_through_initialize_physics_fills_exactly_once():
    """The whole production chain, not the hook in isolation.

    ``initialize_real`` -> ``woof.core.physics.initialize_physics`` is the
    path every real-data front door takes
    (``woof/ingest/hrrr_physics.py::initialize_prepared_physics`` calls
    ``initialize_physics`` on ``result.state``, and
    ``::initialize_hrrr_physics`` delegates to it), and it is the reason this
    module must not perform the fill itself.  WRF's own structure is the same
    one: ``phys/module_physics_init.F:1635`` calls ``mp_init`` as the last
    physics initializer, and ``mp_init``'s THOMPSONAERO arm calls
    ``thompson_init``; ``dyn_em/module_initialize_real.F`` never does.

    ANSWER TO THE OWNERSHIP QUESTION, measured: the physics init path is
    responsible.  The ingest leaves exact zero, the first driver's
    ``microphysics_init_receipt`` reports both fills ran, and a second
    ``initialize_physics`` on the same state reports neither ran -- because
    ``thompson_init``'s own MAXVAL presence tests now see a populated field.
    So a duplicate call is idempotent rather than destructive, and an ingest
    that DID carry aerosol would survive the physics init untouched; the
    reason the ingest still must not call it is structural (WRF's split, and
    the receipt this function publishes), not a race.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics

    result, cfg = _analyzed_hrrr_real_init(
        28, state_backend="cuda", terrain_m=250.0)
    state = result.state
    assert result.aerosol_initialization["awaiting_profile_fill"] is True
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0

    driver = initialize_physics(state, cfg)
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    nwfa2d = cp.asnumpy(state.nwfa2d)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert 0.5e6 <= nifa.min() and nifa.max() <= 2.0e6
    assert nwfa2d.min() > 0.0

    second = initialize_physics(state, cfg)
    assert second.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": False, "in": False}}
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), nwfa)
    np.testing.assert_array_equal(cp.asnumpy(state.nifa), nifa)
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa2d), nwfa2d)

    # The mass species the ingest DID initialize are untouched by the fill.
    for name in ("qc", "qr", "qi", "qs", "qg"):
        live = cp.asnumpy(getattr(state, name))
        assert np.count_nonzero(live) > 0
        assert np.isfinite(live).all()


@pytest.fixture
def wif_dataset():
    """WRF's monthly WIF aerosol dataset, or skip.

    WOOF does not redistribute the 225 MB file, so the default-path test
    can only run where a copy exists.  It is located the same way the
    PRODUCT locates it -- through the resolver's own search order -- so the
    fixture cannot pass on a path the shipping code would not have found.
    """
    from woof.ingest.wif_climatology import resolve_wif_climatology
    try:
        resolution = resolve_wif_climatology()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if not resolution.resolved:
        pytest.skip(str(resolution.fallback_reason))
    return resolution.path


def test_mp28_real_ingest_receipt_announces_the_synthetic_fallback(
        monkeypatch, tmp_path):
    """A run on the synthetic profile SAYS SO, by name, in its receipt.

    RE-BASELINED (development branch).  This test used to assert that the
    ingest receipt agreed with the published DEVIATION -- that WOOF had no
    QNWFA/QNIFA ingest lane and that real.exe FATALs the configuration.
    Both clauses are now false of the default, so pinning them would pin the
    defect.  What must not drift apart is the pair that is still real: a run
    that reached the synthetic profile, and the sentence that tells its
    reader what that means.  Reconstructing the old assertion: every clause
    of the retired deviation is asserted below, on the branch it is still
    true of, plus the two facts the old pin could not express -- that this
    branch is now a FALLBACK, and that it is reached rather than chosen.
    """
    from woof.config import MP28_AEROSOL_SYNTHETIC_FALLBACK
    from woof.ingest.real import WRF_REAL_MP28_AEROSOL_SOURCE_POLICY
    from woof.ingest import wif_climatology

    # Guarantee the fallback: no override, and a working directory that
    # holds no dataset.  (Do not merely unset the env -- the resolver's last
    # candidate is the cwd, which is exactly the point of the search order.)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_PATH_ENV, raising=False)
    monkeypatch.delenv(wif_climatology.WIF_CLIMATOLOGY_ROOT_ENV, raising=False)
    # THE STAGED ROOT is a rung too (merge: static-dataset-door +
    # wif-default).  `woof fetch-tables --wif` installs the dataset into
    # $WOOF_WIF_DATA_ROOT, defaulting to ~/.woof/wif, and
    # resolve_wif_climatology searches it ahead of the cwd.  Unsetting the
    # two file/root overrides is therefore no longer enough to guarantee
    # "nothing resolves": on a machine that has actually staged the dataset
    # -- the reference host has -- this test would otherwise resolve the
    # real 225 MB copy and pass, or fail, for a reason it is not about.
    # Point the staged root at an empty directory instead.
    monkeypatch.setenv("WOOF_WIF_DATA_ROOT", str(tmp_path / "no-staged-wif"))
    monkeypatch.chdir(tmp_path)

    # 2.9.7: 'auto' no longer REACHES the synthetic profile.  With no copy
    # on the machine and the pinned download failing (the suite keeps it
    # offline), the run is refused by name with the deliberate way out
    # (a real-data mp=28 run, 2026-10-23 12:58Z, ran synthetic
    # aerosol with only a warning).
    with pytest.raises(wif_climatology.MissingWifClimatologyDataset) as refused:
        _analyzed_hrrr_real_init(28)
    said = str(refused.value)
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in said
    assert "pinned download" in said
    assert "mp28_aerosol_source='synthetic'" in said

    # The profile taken BY NAME still says so in the receipt.
    result, _ = _analyzed_hrrr_real_init(28, mp28_aerosol_source="synthetic")
    receipt = result.aerosol_initialization

    assert receipt["aerosol_source"] == "thompson_init-synthetic-profile"
    assert receipt["synthetic_fallback_in_use"] is True
    assert receipt["synthetic_fallback_requested"] is True
    assert receipt["aerosol_source_statement"] == (
        MP28_AEROSOL_SYNTHETIC_FALLBACK)
    # The retired deviation's surviving clauses, on the branch they describe.
    assert "thompson_init" in receipt["aerosol_source_statement"]
    assert "not be reported as one" in receipt["aerosol_source_statement"]
    # Unchanged: the fields are still left at exact zero for thompson_init.
    assert "module_mp_thompson.F" in receipt["microphysics_citation"]
    assert receipt["awaiting_profile_fill"] is True
    assert receipt["not_initialized_here"] == (
        WRF_REAL_MP28_AEROSOL_SOURCE_POLICY["not_initialized_here"])


def test_mp28_real_ingest_defaults_to_the_wif_climatology(
        monkeypatch, wif_dataset):
    """THE DEFAULT.  No flag, no selector: the data is simply used.

    The gate this lane exists for.  ``mp28_aerosol_source`` is left at its
    default and the two namelist selectors at WRF's Registry defaults; the
    only thing that differs from the fallback test above is that the dataset
    is reachable.  If this ever needs an opt-in to pass, the remedy has
    become a workaround.
    """
    monkeypatch.setenv(
        "WOOF_WIF_CLIMATOLOGY", str(wif_dataset))

    # The grid geometry a GLOBAL dataset needs.  Passed the way the mapped
    # front door passes it (woof/mapped_direct.py) -- from the model grid
    # and the frame's valid time -- rather than through any test-only hook,
    # so what this asserts is what a user's run does.
    lat2d = np.array([[39.4, 39.4, 39.4], [39.6, 39.6, 39.6]])
    lon2d = np.array([[-98.7, -98.5, -98.3], [-98.7, -98.5, -98.3]])
    result, _ = _analyzed_hrrr_real_init(
        28, wif_grid_latlon=(lat2d, lon2d),
        wif_valid_date="2026-08-25T18:00:00")
    receipt = result.aerosol_initialization

    assert receipt["aerosol_source"] == "wif-climatology"
    assert receipt.get("synthetic_fallback_in_use") is None
    assert receipt["awaiting_profile_fill"] is False
    assert receipt["dataset"]["resolved"] is True
    assert receipt["dataset"]["sha256"]
    # STALE PIN, CORRECTED IN THE MERGE.  development branch wrote "...-v1"
    # here, but woof/ingest/wif_climatology.py has stamped "...-v2" since
    # 474e0e9a0 ("the WIF climatology data path moves onto the project's Rust"),
    # an ANCESTOR of that commit -- so this assertion was already false of
    # the code when it was written.  Nothing caught it because the
    # ``wif_dataset`` fixture SKIPPED: it resolves the dataset the way the
    # product does, and before the staged root became a rung of that search
    # its only rungs were two unset env vars and a cwd that never holds a
    # 225 MB file.  tests/test_wif_climatology_equivalence.py:156 has
    # pinned v2 the whole time; this line is brought into agreement with
    # the code and with it.
    assert receipt["wif_climatology"]["schema"] == (
        "wrf-v4.7.1-wif-climatology-ingest-v2")


def _pressure_level_real_init(mp_physics, *, init_kwargs=None, shape=(2, 3),
                              **config_overrides):
    """The other production lane: pressure-level TT/RH forcing (ERA5, GFS).

    No analyzed hydrometeors exist on this lane for ANY scheme, which is
    exactly why the mp=28 aerosol policy cannot live inside the native-HRRR
    ``if hydrometeors:`` branch -- a user arriving with ERA5 must still get
    the exact-zero aerosol state thompson_init's presence test needs.
    ``init_kwargs`` reach :func:`initialize_real` unchanged.
    """
    ny, nx = shape
    nz = 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    fields = {
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "RH": np.full(pressure.shape, 60.0, dtype=np.float32),
        "D2": np.full((ny, nx), 283.0, dtype=np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0,
        dt=30.0, run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True,
        terrain_opt=1, mp_physics=mp_physics, **config_overrides)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.zeros((ny, nx), dtype=np.float64)
    return initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="cpu", **(init_kwargs or {})), cfg


def test_mp28_pressure_level_lane_also_publishes_the_aerosol_policy():
    """ERA5/GFS forcing reaches mp=28 too, and gets the same policy."""
    result, _ = _pressure_level_real_init(28, mp28_aerosol_source="synthetic")
    state = result.state

    # No analyzed condensate exists on this lane; qc/qr are explicitly zeroed
    # and every other species is at its allocation zero.  That is the same
    # for mp=8 and is not an aerosol question.
    for name in ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni"):
        live = _host_array(getattr(state, name))
        assert int(live.view(np.uint32).max()) == 0, name
    assert result.hydrometeor_initialization == {}

    receipt = result.aerosol_initialization
    assert receipt["policy"] == (
        "aer-init-opt-0-zero-then-thompson-init-synthetic-profile")
    assert receipt["awaiting_profile_fill"] is True
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(_host_array(getattr(state, name)).view(
            np.uint32).max()) == 0

    control, _ = _pressure_level_real_init(8)
    assert control.aerosol_initialization == {}


def test_mp28_pressure_level_lane_refuses_an_unreadable_aerosol_source():
    with pytest.raises(NotImplementedError, match="wif_input_opt=1"):
        _pressure_level_real_init(28, wif_input_opt=1)


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_runs_on_the_production_cuda_preprocessing():
    """The shipped backend, not only the NumPy reference.

    Everything above selects an independent CPU reference implementation of
    the preprocessing ABI so the scheme-selection logic can be tested without
    a GPU.  This one runs the production CUDA vertical interpolation and the
    device state, then the real physics init, so the mp=28 real-data lane is
    proven on the code path a user actually gets.
    """
    import cupy as cp

    from woof.core.physics import initialize_physics
    from woof.ingest.preprocess_backend import resolve_preprocess_backend

    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = 250.0 + np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
    }
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        fields[name] = np.asarray(
            species_index * 1.0e-6 * (1.0 + level_index)
            * np.ones((1, ny, nx), dtype=np.float32), dtype=np.float32)
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0, dt=30.0,
        run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        mp_physics=28)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.full((ny, nx), 250.0, dtype=np.float64)

    backend = resolve_preprocess_backend("cuda")
    assert backend.receipt()["backend"] != "cpu-reference-test"
    result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0, use_sh_qv=True,
        preprocess_backend="cuda", state_backend="cuda",
        landmask=np.ones((ny, nx), dtype=np.float32))
    state = result.state

    for name in ("qc", "qr", "qi", "qs", "qg"):
        live = cp.asnumpy(getattr(state, name))
        assert np.isfinite(live).all() and live.min() >= 0.0
        assert np.count_nonzero(live) == live.size
    # The aerosols stay at real.exe's zero until the profile fill; the
    # three number moments are closed over the analyzed mass (every cell
    # carries mass here, so every cell is written) through the scheme's
    # own entry block, on the CUDA preprocessing backend as on the CPU one.
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0
    for name in ("nc", "nr", "ni"):
        live = cp.asnumpy(getattr(state, name))
        assert np.isfinite(live).all() and live.min() > 0.0
    closure = result.hydrometeor_initialization["cold_start_moment_closure"]
    assert closure["repaired_cells_total"] == 3 * nz * ny * nx
    assert set(closure["written_state_fields"]) == {"nc", "nr", "ni"}
    assert result.aerosol_initialization["awaiting_profile_fill"] is True

    driver = initialize_physics(state, cfg)
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert cp.asnumpy(state.nwfa2d).min() > 0.0


def test_mp28_real_ingest_state_is_shaped_typed_and_physical():
    """Task-level acceptance: shapes, dtypes and ranges of the mp=28 state.

    The scheme is only reachable from real initial conditions if the state it
    receives is a real one.  Every field mp=28 adds over mp=8 is checked for
    shape, dtype and value, and the shared thermodynamic state is graded by
    the same discrete moist-hydrostatic residual mp=8 is graded by -- and
    required to be IDENTICAL to it, because the aerosol scheme changes no
    part of the mass/thermodynamic setup.
    """
    result, cfg = _analyzed_hrrr_real_init(28, mp28_aerosol_source="synthetic")
    control, _ = _analyzed_hrrr_real_init(8)
    state = result.state
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                 "nwfa", "nifa"):
        live = _host_array(getattr(state, name))
        assert live.shape == (nz, ny, nx), name
        assert live.dtype == np.float32, name
        assert np.isfinite(live).all(), name
        assert live.min() >= 0.0, name
    for name in ("nwfa2d", "nifa2d"):
        live = _host_array(getattr(state, name))
        assert live.shape == (ny, nx), name
        assert live.dtype == np.float32, name

    # WRF's effective-radius cold start, shared with mp=8
    # (module_model_constants.F RE_QC_BG/RE_QI_BG/RE_QS_BG).
    assert float(_host_array(state.effc).max()) == pytest.approx(2.49)
    assert float(_host_array(state.effi).max()) == pytest.approx(4.99)
    assert float(_host_array(state.effs).max()) == pytest.approx(9.99)

    # Vapour is a real analysis, not a placeholder.
    qv = _host_array(state.qv)
    assert 0.0 < qv.min() and qv.max() < 0.05

    residual = hydrostatic_residual(result)
    control_residual = hydrostatic_residual(control)
    assert np.isfinite(residual).all()
    np.testing.assert_array_equal(residual, control_residual)


#: What the refusing d02 actually held, and what its healthy neighbours
#: held, from the gpuwm 1.8.4 nested HRRR tree at 39.0,-103.0 on cycle
#: 2026-08-08T09.  216x272 cells; exactly two below zero.
_SAN_JUAN_SURFACE_QV = (-1.785645637e-05, -1.478747851e-05)
_SAN_JUAN_SURFACE_QV_MAX = 0.0176926
#: The same tree's ROOT, which passed: its minimum is small but positive,
#: so the floor must leave it alone.  And the flat-terrain Oklahoma d02
#: from the release's completing cell, three orders of magnitude clear.
_COLORADO_ROOT_SURFACE_QV_MIN = 3.043969e-05
_OKLAHOMA_CHILD_SURFACE_QV_MIN = 0.00871458


def test_flag_sh_surface_qv_floor_lifts_the_refusing_colorado_cells():
    """The two real negative cells become WRF's qv_min_value, nothing else.

    These are the values that produced "prepared near-surface surface_qv
    is outside the physical range 0.0..0.2" on the 1.8.4 nested-HRRR tree
    over eastern Colorado.  The floor is WRF's own ``qv_min_value``
    (Registry default 1e-6, module_initialize_real.F:7499-7503) -- the
    same constant :func:`_saturation_mixing_ratio` already applies to the
    RH lane's surface value -- so after it the guard's 0.0 bound cannot be
    crossed by this lane.
    """
    surface_qv = np.full((4, 4), 0.004)
    surface_qv[1, 2] = _SAN_JUAN_SURFACE_QV[0]
    surface_qv[2, 1] = _SAN_JUAN_SURFACE_QV[1]
    surface_qv[0, 0] = _SAN_JUAN_SURFACE_QV_MAX
    psfc = np.full((4, 4), 68_000.0)

    floored, receipt = _floor_flag_sh_surface_mixing_ratio(surface_qv, psfc)

    assert floored.min() == pytest.approx(_WRF_QV_MIN_VALUE)
    assert floored[1, 2] == _WRF_QV_MIN_VALUE
    assert floored[2, 1] == _WRF_QV_MIN_VALUE
    # Every healthy cell is byte-identical, including the domain maximum.
    untouched = np.ones((4, 4), dtype=bool)
    untouched[1, 2] = untouched[2, 1] = False
    np.testing.assert_array_equal(floored[untouched], surface_qv[untouched])
    assert receipt["floored_cells"] == 2
    assert receipt["negative_cells"] == 2
    assert receipt["min_pre_floor"] == pytest.approx(
        min(_SAN_JUAN_SURFACE_QV))
    assert receipt["qv_min_value"] == _WRF_QV_MIN_VALUE


@pytest.mark.parametrize(
    "minimum",
    [_COLORADO_ROOT_SURFACE_QV_MIN, _OKLAHOMA_CHILD_SURFACE_QV_MIN])
def test_flag_sh_surface_qv_floor_is_a_no_op_above_wrf_qv_min(minimum):
    """Shapes that already passed keep every bit and report nothing.

    The Colorado ROOT on the very cycle whose child refused, and the
    Oklahoma child that completed.  Both sit above WRF's floor, so the
    fix cannot move a single existing artifact -- which is the whole
    reason it is safe to apply on a release line.
    """
    surface_qv = np.linspace(minimum, 0.02, 12).reshape(3, 4)
    psfc = np.full((3, 4), 84_000.0)

    floored, receipt = _floor_flag_sh_surface_mixing_ratio(surface_qv, psfc)

    np.testing.assert_array_equal(floored, surface_qv)
    assert receipt == {}


def test_flag_sh_surface_qv_floor_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        _floor_flag_sh_surface_mixing_ratio(
            np.zeros((2, 3)), np.zeros((3, 2)))


def test_rh_lane_surface_qv_already_carries_the_same_floor():
    """The divergence this fix closed, stated as a test.

    GFS/ERA5 build surface_qv through :func:`_saturation_mixing_ratio`,
    which floors at WRF's qv_min_value inline (WRF v4.6.1 rh_to_mxrat1:7402)
    -- so that lane could never present a negative to the prepared
    near-surface guard, and a nested GFS run over the same eastern
    Colorado placement completed while the HRRR one refused.  The FLAG_SH
    lane had no floor at all; now both share this constant.
    """
    bone_dry = _saturation_mixing_ratio(
        np.full((2, 2), 250.0), np.full((2, 2), 68_000.0),
        np.zeros((2, 2)))

    assert np.all(bone_dry == _WRF_QV_MIN_VALUE)


def test_explicit_five_analyzed_species_preserves_native_state_bytes():
    legacy, _ = _analyzed_hrrr_real_init(6)
    declared, _ = _analyzed_hrrr_real_init(6, analyzed_species=HRRR_ANALYZED_HYDROMETEORS)
    for name in ("mup", "thp", "php", "qv", "qc", "qr", "qi", "qs", "qg", "u", "v", "w"):
        np.testing.assert_array_equal(_host_array(getattr(legacy.state, name)),
                                      _host_array(getattr(declared.state, name)))
    for name in ("total_pressure", "total_specific_volume", "surface_qv"):
        np.testing.assert_array_equal(getattr(legacy,name),getattr(declared,name))


@pytest.mark.parametrize("selected", [(), ("QC",), ("QC", "QR")])
def test_declared_analyzed_inventory_retains_only_file_supplied_mass(selected):
    absent=tuple(name for name in HRRR_ANALYZED_HYDROMETEORS if name not in selected)
    result, _ = _analyzed_hrrr_real_init(6, drop=absent, analyzed_species=selected)
    receipt=result.hydrometeor_initialization
    assert receipt["declared_analyzed_species"]==list(selected)
    assert set(receipt["source_absent_state_fields"])=={name.lower() for name in absent}
    for name in absent:
        assert np.count_nonzero(_host_array(getattr(result.state,name.lower())))==0
        assert receipt["source_absent_state_fields"][name.lower()]["nonzero_count"]==0
    for name in selected:
        assert np.count_nonzero(_host_array(getattr(result.state,name.lower())))>0
    assert np.isfinite(result.total_pressure).all()


def test_declared_analyzed_field_cannot_be_absent():
    with pytest.raises(KeyError,match="QC"):
        _analyzed_hrrr_real_init(6,drop=("QC",),analyzed_species=("QC",))


def test_the_hydrometeor_receipt_names_the_horizontal_owner_and_the_zero_w():
    """The operator each analyzed species took, and W's zero, are READ.

    The regular-source pass publishes the owner of every output field on
    the snapshot; the initializer copies the five hydrometeors' owners
    into the receipt so a parabolic entry (the overshoot the
    non-negativity check refuses) is visible after the fact.  The native
    decoder publishes no owners and says so.  Vertical velocity is zero on
    every route and the receipt states the policy instead of leaving it
    to be inferred from an absence.
    """
    from woof.ingest.real import WRF_REAL_VERTICAL_VELOCITY_POLICY

    owners = {name: "bilinear" for name in ("QC", "QR", "QI", "QS", "QG")}
    owners["TT"] = "parabolic"
    result, _ = _analyzed_hrrr_real_init(8, horizontal_operators=owners)
    receipt = result.hydrometeor_initialization
    assert receipt["horizontal_operator"] == {
        name: "bilinear" for name in ("QC", "QR", "QI", "QS", "QG")}
    assert receipt["vertical_velocity"] == WRF_REAL_VERTICAL_VELOCITY_POLICY
    assert receipt["vertical_velocity"]["policy"] == "exact-fp32-zero"
    assert int(_host_array(result.state.w).view(np.uint32).max()) == 0

    native, _ = _analyzed_hrrr_real_init(8)
    assert native.hydrometeor_initialization["horizontal_operator"] == {
        name: "unrecorded" for name in ("QC", "QR", "QI", "QS", "QG")}


def test_mp28_cold_start_closes_the_number_moments_over_the_imported_mass():
    """Mass in, numbers consistent with it: the scheme's own entry block.

    real.exe seeds the aerosol-aware scheme's three number moments where
    the mass is present (make_DropletNumber, make_RainNumber,
    make_IceNumber) and Thompson's entry block rediagnoses them on the
    first call.  The start used to leave them at zero, so the state
    carried mass with no number in every cloudy cell (orphan cells), and
    anything that reads the state there -- the between-step reflectivity
    operator, a t=0 analysis, a picture of the initial frame -- read a
    rain number at the scheme's R2 floor and diagnosed a reflectivity
    burst.  The cold start now seeds them and runs the same entry block
    once, from
    woof.core.thompson_entry (the authority woof.da.moments
    .repair_moments applies), on the density the initializer formed,
    and says so in hydrometeor_initialization.
    """
    from woof.da.moments import moment_consistency_report, scheme_moments
    from woof.core.thompson_entry import R1

    result, _ = _analyzed_hrrr_real_init(28, mp28_aerosol_source="synthetic")
    state = result.state
    view = {name: _host_array(getattr(state, name))
            for name in ("qc", "qr", "qi", "nc", "nr", "ni")}
    view["alt"] = np.asarray(result.total_specific_volume, dtype=np.float32)
    report = moment_consistency_report(view, mp_physics=28)
    assert report["offending_cells_total"] == 0, report
    assert report["nonfinite_cells_total"] == 0
    for pair in scheme_moments(28).pairs:
        mass, number = view[pair.mass], view[pair.number]
        assert number.dtype == np.float32
        assert np.isfinite(number).all() and (number >= 0.0).all()
        assert (number[mass > R1] > 0.0).all(), pair
        assert (number[mass <= R1] == 0.0).all(), pair

    closure = result.hydrometeor_initialization["cold_start_moment_closure"]
    assert closure["repaired"] is True
    assert closure["repaired_cells_total"] == sum(
        int(np.count_nonzero(view[pair.mass] > R1))
        for pair in scheme_moments(28).pairs)
    assert closure["authority"] == scheme_moments(28).repair_authority
    assert closure["mp_physics"] == 28
    assert set(closure["written_state_fields"]) == {"nc", "nr", "ni"}


#: A stand-in HRRR pressure-level cloud column, one value per source level
#: (100..1000 hPa), 0.01 to 0.3 g/kg: a typical cloud-water range.
_CLOUDY_COLUMN_KG_KG = (1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 2.0e-4, 5.0e-5)


@pytest.mark.parametrize("landmask, diameter_um", [(1.0, 8.2), (0.0, 14.9)])
def test_mp28_cold_start_droplet_number_is_real_exe_make_droplet_number(
        landmask, diameter_um):
    """Analysed cloud water starts as cloud, not drizzle.

    real.exe sets the droplet number where cloud water has none
    (WRF v4.7.1 module_initialize_real.F:4829-4838) from
    make_DropletNumber (:9119-9158).  With no aerosol yet (thompson_init
    fills it later) the surface sizes the drops: nu_c = 4 at 11 um over
    land, nu_c = 12 at 17 um over water, mean volume diameters of 8.2 and
    14.9 um.  The start used to hand the entry block a zero number, which
    floored it at 2 m^-3 and returned 0.02 to 0.7 drops per cm3 near
    89 um: drops that rained out and froze in the first minutes.
    """
    from woof.core.thompson_entry import (
        droplet_mean_diameter_m, make_droplet_number)

    result, _ = _analyzed_hrrr_real_init(
        28, cloud_water=_CLOUDY_COLUMN_KG_KG, landmask=landmask, mp28_aerosol_source="synthetic")
    state = result.state
    qc = _host_array(state.qc)
    nc = _host_array(state.nc)
    alt = np.asarray(result.total_specific_volume, dtype=np.float32)
    rho = (np.float32(1.0) / alt).astype(np.float32)
    cloudy = qc > 0.0
    assert cloudy.sum() > 0
    xland = np.float32(1.0 if landmask >= 0.5 else 2.0)
    wrf = (make_droplet_number((qc * rho)[cloudy], np.float32(0.0), xland)
           / rho[cloudy]).astype(np.float32)
    # The entry block's rediagnosis returns the same population unless a
    # size clamp fires, and none does between 1 and 100 um.
    np.testing.assert_allclose(nc[cloudy], wrf, rtol=2.0e-6)
    per_cm3 = nc[cloudy] * rho[cloudy] * 1.0e-6
    diameter = droplet_mean_diameter_m(
        qc[cloudy] * rho[cloudy], nc[cloudy] * rho[cloudy]) * 1.0e6
    np.testing.assert_allclose(diameter, diameter_um, atol=0.1)
    assert per_cm3.min() > 1.0, per_cm3.min()
    seed = result.hydrometeor_initialization[
        "cold_start_moment_closure"]["droplet_number_seed"]
    assert seed["seeded_cells"] == int(cloudy.sum())
    branch = "land_branch_cells" if landmask >= 0.5 else "water_branch_cells"
    assert seed[branch] == int(cloudy.sum())
    assert seed["aerosol_branch_cells"] == 0


def test_mp28_cold_start_without_a_landmask_refuses_to_guess_the_drop_size():
    with pytest.raises(ValueError, match="LANDMASK"):
        _analyzed_hrrr_real_init(
            28, cloud_water=_CLOUDY_COLUMN_KG_KG, landmask=None, mp28_aerosol_source="synthetic")


def _real_exe_seed_then_entry(species, mass, alt, temperature):
    """real.exe's make_RainNumber/make_IceNumber seed, then the entry block.

    The seed where the mass is present (module_initialize_real.F:4840-4852,
    WRF v4.7.1) in REAL, per volume at rho = 1./alt; then Thompson's entry
    block over the cells with mass above R1, at the closure's density.
    """
    from woof.core.thompson_entry import (
        R1, make_ice_number, make_rain_number, np_thompson_entry_numbers)

    function = make_rain_number if species == "rain" else make_ice_number
    rho = (np.float32(1.0) / alt).astype(np.float32)
    seed = np.zeros_like(mass)
    present = mass > 0.0
    seed[present] = (function((mass * rho)[present], temperature[present])
                     / rho[present]).astype(np.float32)
    closed = np_thompson_entry_numbers(
        species, mass, seed, 1.0 / alt.astype(np.float64))
    return np.where(mass > R1, closed.astype(np.float32), seed)


def _state_temperature(result):
    state = result.state
    theta = (_host_array(state.thb).astype(np.float64)
             + _host_array(state.thp).astype(np.float64))
    pressure = np.asarray(result.total_pressure, dtype=np.float64)
    return (theta * (pressure / c.P0) ** c.RCP).astype(np.float32)


@pytest.mark.parametrize("mp_physics", [8, 28])
def test_cold_start_rain_and_ice_numbers_are_real_exe_make_numbers(
        mp_physics):
    """Analysed rain and ice start at real.exe's sizes, on mp=8 and mp=28.

    real.exe gives rain and ice mass without a number make_RainNumber and
    make_IceNumber (WRF v4.7.1 module_initialize_real.F:4840-4852,
    :9163-9194, :9044-9114): a Marshall-Palmer intercept that rises to
    8e8 at or below -2 C, so supercooled rain starts as drizzle, and a
    crystal size read from temperature.  The start used to leave mp=8 at
    zero and hand mp=28's entry block a zero number, which made every
    rain drop 1 mm and every crystal 5 um, capped at 999e3 per m3.
    """
    from woof.core.thompson_entry import rain_median_volume_diameter_m

    result, _ = _analyzed_hrrr_real_init(mp_physics, **_SYNTHETIC_IF_28(mp_physics))
    state = result.state
    alt = np.asarray(result.total_specific_volume, dtype=np.float32)
    rho = (np.float32(1.0) / alt).astype(np.float32)
    temperature = _state_temperature(result)
    closure = result.hydrometeor_initialization["cold_start_moment_closure"]
    assert closure["mp_physics"] == mp_physics
    for species, mass_name, number_name in (
            ("rain", "qr", "nr"), ("ice", "qi", "ni")):
        mass = _host_array(getattr(state, mass_name))
        number = _host_array(getattr(state, number_name))
        present = mass > 0.0
        assert present.sum() > 0
        expected = _real_exe_seed_then_entry(species, mass, alt, temperature)
        np.testing.assert_allclose(number[present], expected[present],
                                   rtol=1.0e-4)
        assert (number[~present] == 0.0).all()
        seed = closure[f"{species}_number_seed"]
        assert seed["seeded_cells"] == int(present.sum())
    qr = _host_array(state.qr)
    nr = _host_array(state.nr)
    mvd = rain_median_volume_diameter_m(qr * rho, nr * rho)
    cold = (qr > 0.0) & (temperature <= np.float32(271.15))
    warm = (qr > 0.0) & (temperature >= np.float32(273.15))
    assert cold.sum() > 0 and warm.sum() > 0
    # Supercooled rain starts as drizzle, well under the 1 mm the entry
    # block gives a zero number, and smaller than any warm drop.
    assert mvd[cold].max() < 0.35e-3, mvd[cold].max()
    assert mvd[cold].max() < mvd[warm].min(), (mvd[cold], mvd[warm])
    assert closure["rain_number_seed"]["supercooled_cells"] == int(cold.sum())
    if mp_physics == 8:
        # mp=8 has no droplet number: real.exe's P_QNC test is false.
        assert set(closure["written_state_fields"]) == {"nr", "ni"}
        assert closure["droplet_number_seed"] is None
        assert [entry["species"] for entry in closure["species"]] == [
            "rain", "ice"]


def test_analysed_zero_numbers_are_seeded_like_absent_ones():
    """An installed analysed NR/NI of zero is filled, as real.exe fills it.

    real.exe seeds wherever the number it holds is <= 0, after the
    analysed numbers are in (module_initialize_real.F:4840-4852).  The
    closure used to run before the install, so an analysed zero
    overwrote the seed and left rain and ice mass with no number.
    """
    extra = {"QNR": 0.0, "QNR_SFC": 0.0, "QNI": 0.0, "QNI_SFC": 0.0}
    analysed, _ = _analyzed_hrrr_real_init(
        28, extra_fields=extra,
        init_kwargs={"analyzed_number_fields": ("QNI", "QNR")}, mp28_aerosol_source="synthetic")
    absent, _ = _analyzed_hrrr_real_init(28, mp28_aerosol_source="synthetic")
    assert "number_moments" in analysed.hydrometeor_initialization
    for number_name, mass_name in (("nr", "qr"), ("ni", "qi")):
        got = _host_array(getattr(analysed.state, number_name))
        mass = _host_array(getattr(analysed.state, mass_name))
        assert (got[mass > 0.0] > 0.0).all(), number_name
        np.testing.assert_array_equal(
            got.view(np.uint32),
            _host_array(getattr(absent.state, number_name)).view(np.uint32))


def test_the_seed_temperature_is_built_only_when_a_cell_is_seeded():
    """No rain or ice cell to seed, no temperature built; else built once.

    The temperature real.exe sizes rain and ice by is a full 3-D pass
    over the domain, and a start whose rain and ice numbers are all
    analysed above zero, or whose rain and ice mass is zero, seeds no
    cell and has no use for it.
    """
    from types import SimpleNamespace

    from woof.ingest.real import _thompson_cold_start_moment_closure

    shape = (3, 2, 2)
    alt = np.full(shape, 0.9, dtype=np.float32)
    calls = []

    def temperature():
        calls.append(1)
        return np.full(shape, 268.0, dtype=np.float32)

    def state(mass, number):
        return SimpleNamespace(
            qc=np.zeros(shape, np.float32), nc=np.zeros(shape, np.float32),
            qr=np.full(shape, mass, np.float32),
            nr=np.full(shape, number, np.float32),
            qi=np.full(shape, mass, np.float32),
            ni=np.full(shape, number, np.float32))

    for unseeded in (state(0.0, 0.0), state(2.0e-5, 1234.5)):
        receipt = _thompson_cold_start_moment_closure(
            unseeded, np, SimpleNamespace(mp_physics=8), alt,
            temperature=temperature)
        assert receipt["rain_number_seed"]["seeded_cells"] == 0
        assert receipt["ice_number_seed"]["seeded_cells"] == 0
    assert calls == []

    seeded = state(2.0e-5, 0.0)
    receipt = _thompson_cold_start_moment_closure(
        seeded, np, SimpleNamespace(mp_physics=8), alt,
        temperature=temperature)
    assert calls == [1]
    assert receipt["rain_number_seed"]["seeded_cells"] == seeded.qr.size
    assert receipt["ice_number_seed"]["seeded_cells"] == seeded.qi.size
    assert (seeded.nr > 0.0).all() and (seeded.ni > 0.0).all()


def test_closure_keeps_an_analysed_number_above_zero_and_seeds_the_rest():
    """Only numbers at or below zero are written; every other bit is kept."""
    from types import SimpleNamespace

    from woof.ingest.real import _thompson_cold_start_moment_closure

    shape = (3, 2, 2)
    mass = np.full(shape, 2.0e-5, dtype=np.float32)
    mass[0, 0, 0] = 0.0
    installed = np.full(shape, 1234.5, dtype=np.float32)
    installed[1] = 0.0
    installed[2, 1, 1] = -3.0
    state = SimpleNamespace(
        qc=np.zeros(shape, np.float32), nc=np.zeros(shape, np.float32),
        qr=mass.copy(), nr=installed.copy(),
        qi=mass.copy(), ni=installed.copy())
    alt = np.full(shape, 0.9, dtype=np.float32)
    temperature = np.full(shape, 268.0, dtype=np.float32)
    receipt = _thompson_cold_start_moment_closure(
        state, np, SimpleNamespace(mp_physics=8), alt,
        temperature=temperature)
    keep = installed > 0.0
    for species, name in (("rain", "nr"), ("ice", "ni")):
        got = getattr(state, name)
        np.testing.assert_array_equal(got[keep].view(np.uint32),
                                      installed[keep].view(np.uint32))
        expected = _real_exe_seed_then_entry(species, mass, alt, temperature)
        np.testing.assert_array_equal(got[~keep].view(np.uint32),
                                      expected[~keep].view(np.uint32))
        assert receipt[f"{species}_number_seed"]["seeded_cells"] == int(
            np.count_nonzero(~keep & (mass > 0.0)))


def test_ntu_real_ingest_retains_analyzed_mass_and_source_absent_moments():
    result,_=_analyzed_hrrr_real_init(56)
    control,_=_analyzed_hrrr_real_init(10)
    assert set(result.hydrometeor_initialization['retained_correspondence'])==set(HRRR_ANALYZED_HYDROMETEORS)
    assert result.hydrometeor_initialization['discarded_source_species']=={}
    for name in ('qc','qr','qi','qs','qg'):
        got=_host_array(getattr(result.state,name))
        assert np.count_nonzero(got)>0
        assert got.tobytes()==_host_array(getattr(control.state,name)).tobytes()
    from woof.core.ntu import NTU_SCALAR_ATTRS
    for name in ('qh',)+NTU_SCALAR_ATTRS:
        assert not _host_array(getattr(result.state,name)).view('u4').any(),name
    with pytest.raises(KeyError,match=r'missing real-data field\(s\).*QI'):
        _analyzed_hrrr_real_init(56,drop=('QI',))


def test_ntu_real_ingest_retains_declared_hail_mass():
    result,_=_analyzed_hrrr_real_init(56,analyzed_species=DECLARED_ANALYZED_HYDROMETEORS,
                                    extra_fields={'QH':np.float32(3e-5)})
    assert result.hydrometeor_initialization['retained_correspondence']['QH']=='qh'
    assert np.count_nonzero(_host_array(result.state.qh))>0
