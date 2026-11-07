"""RUC vegetation, albedo and sea-ice options as operational HRRR sets them.

CPU-only.  Pins the host half of ``usemonalb``, ``rdlai2d`` and
``fractional_seaice`` for the RUC land surface against the HRRR v4.1.21
WRF fork (``module_physics_init.F`` landuse_init,
``module_initialize_real.F`` monthly_interp_to_date,
``module_surface_driver.F:1365-1368``), and holds the default-off identity:
with every switch at its default, ``initialize_landuse`` and the RUC
runtime parameters resolve exactly what they resolved before the switches
existed.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.landuse import (
    initialize_landuse,
    monthly_background_albedo,
    monthly_snow_albedo,
    prescribed_monthly_field,
    surface_leaf_area,
    surface_snow_albedo,
    usemonalb_landuse_inputs,
)
from woof.core.ruc_runtime import (
    RucRuntimeParameters,
    XICE_THRESHOLD,
    XICE_THRESHOLD_FRACTIONAL,
)
from woof.static.build import monthly_interp_to_date

_MODIS = dict(mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21,
              isice=15)


def _ruc_config(**overrides) -> RunConfig:
    base = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                     sf_sfclay_physics=1, sf_surface_physics=3,
                     num_soil_layers=9, bl_pbl_physics=1)
    return dataclasses.replace(base, **overrides)


def _landuse_inputs():
    lu = np.array([[1, 12, 17, 15], [10, 7, 17, 16]], dtype=np.int32)
    soil = np.array([[6, 6, 14, 16], [6, 6, 14, 6]], dtype=np.int32)
    landmask = np.array([[1, 1, 0, 1], [1, 1, 0, 1]], dtype=np.float32)
    snow = np.zeros((2, 4), np.float32)
    xice = np.zeros((2, 4), np.float32)
    return lu, soil, landmask, snow, xice


# --- configuration -----------------------------------------------------------

def test_usemonalb_and_rdlai2d_are_admitted_for_ruc_and_noah():
    for lsm in (2, 3):
        cfg = _ruc_config(sf_surface_physics=lsm, usemonalb=True,
                          rdlai2d=True,
                          **({} if lsm == 3 else {"num_soil_layers": 4}))
        validate_run_config(cfg)


def test_usemonalb_and_rdlai2d_are_refused_where_no_branch_reads_them():
    cfg = _ruc_config(sf_surface_physics=4, usemonalb=True,
                      num_soil_layers=4)
    with pytest.raises(ValueError, match="Noah-MP and the slab carry no"):
        validate_run_config(cfg)


def test_opt_thcnd_keeps_its_noah_only_refusal():
    with pytest.raises(ValueError, match="opt_thcnd is a Noah LSM option"):
        validate_run_config(_ruc_config(opt_thcnd=2))


def test_fractional_seaice_is_zero_or_one_and_requires_a_supported_surface():
    validate_run_config(_ruc_config(fractional_seaice=1))
    with pytest.raises(ValueError, match="fractional_seaice=2"):
        validate_run_config(_ruc_config(fractional_seaice=2))
    noah = _ruc_config(sf_surface_physics=2, num_soil_layers=4,
                       fractional_seaice=1)
    validate_run_config(noah)
    assert RunConfig.__dataclass_fields__["fractional_seaice"].default == 0


def test_case_data_routes_take_the_fractional_land_use_branch_from_config():
    # module_physics_init.F:1463-1465: landuse_init reads the threshold the
    # surface driver runs.  The default answers False, the value the
    # case-data routes (woof/runtime.py) passed before the key existed.
    from woof.core.landuse import ruc_fractional_seaice
    assert ruc_fractional_seaice(_ruc_config()) is False
    assert ruc_fractional_seaice(_ruc_config(fractional_seaice=1)) is True
    assert ruc_fractional_seaice(object()) is False


def test_checkpoints_carry_fractional_seaice_only_when_set():
    # A header written at 0 is the header a build without the field wrote
    # (restart._drop_default_off_run_keys); resume treats an absent key as
    # 0 both ways and still refuses a 0 <-> 1 change, which changes the
    # surface the RUC seam integrates.
    from woof.io import restart
    off, on = _ruc_config(), _ruc_config(fractional_seaice=1)
    echo_off = restart.configuration_echo(off)
    echo_on = restart.configuration_echo(on)
    assert "fractional_seaice" not in echo_off
    assert echo_on["fractional_seaice"] == 1
    assert "fractional_seaice" not in restart._configuration_digest_values(
        echo_off)
    restart._require_config_match(dict(echo_off), off, "old.npz")
    with pytest.raises(restart.RestartMismatchError, match="fractional_seaice"):
        restart._require_config_match(dict(echo_off), on, "old.npz")
    with pytest.raises(restart.RestartMismatchError, match="fractional_seaice"):
        restart._require_config_match(dict(echo_on), off, "new.npz")
    restart._require_config_match(dict(echo_on), on, "new.npz")


# --- the RUC runtime's one threshold ----------------------------------------

def test_ruc_parameters_resolve_wrfs_two_thresholds():
    off = RucRuntimeParameters()
    on = RucRuntimeParameters(fractional_seaice=1, rdlai2d=True)
    assert off.xice_threshold == XICE_THRESHOLD == 0.5
    assert on.xice_threshold == XICE_THRESHOLD_FRACTIONAL == 0.02
    assert off.rdlai2d is False and on.rdlai2d is True
    # At the defaults the identity is the one a header written before the
    # switches carried; set, each key binds.
    identity = off.restart_identity()
    assert identity["xice_threshold"] == 0.5
    assert "fractional_seaice" not in identity
    assert "rdlai2d" not in identity
    identity = on.restart_identity()
    assert identity["xice_threshold"] == 0.02
    assert identity["fractional_seaice"] == 1
    assert identity["rdlai2d"] is True
    with pytest.raises(ValueError, match="fractional_seaice"):
        RucRuntimeParameters(fractional_seaice=2)
    with pytest.raises(TypeError, match="rdlai2d"):
        RucRuntimeParameters(rdlai2d=1)


def test_seaice_albedo_override_follows_the_threshold():
    from woof.core.ruc_runtime import _ruc_seaice_albedo_override
    xice = np.array([0.0, 0.1, 0.3, 0.6, 1.0], np.float32)
    albbck = np.full(5, 0.2, np.float32)
    out_half, ice_half = _ruc_seaice_albedo_override(
        albbck, xice, 0.65, arrays=np)
    out_frac, ice_frac = _ruc_seaice_albedo_override(
        albbck, xice, 0.65, arrays=np, xice_threshold=0.02)
    np.testing.assert_array_equal(ice_half, [False, False, False, True, True])
    np.testing.assert_array_equal(ice_frac, [False, True, True, True, True])
    np.testing.assert_array_equal(
        out_half, np.array([0.2, 0.2, 0.2, 0.65, 0.65], np.float32))
    np.testing.assert_array_equal(
        out_frac, np.array([0.2, 0.65, 0.65, 0.65, 0.65], np.float32))


# --- landuse_init under usemonalb ---------------------------------------------

def _monthly(shape, base):
    months = np.arange(1, 13, dtype=np.float64)[:, None, None]
    return base + months * np.ones((1, *shape))


def test_default_landuse_is_unchanged_by_the_new_keywords():
    lu, soil, landmask, snow, xice = _landuse_inputs()
    before = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=datetime(2026, 10, 2, 21), cen_lat=38.5, **_MODIS)
    after = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=datetime(2026, 10, 2, 21), cen_lat=38.5, **_MODIS,
        usemonalb=False)
    for field in dataclasses.fields(before):
        np.testing.assert_array_equal(getattr(before, field.name),
                                      getattr(after, field.name))
    # LANDUSE.TBL summer row for the MODIS classes, as before.
    np.testing.assert_allclose(before.albbck[0, :2], [0.12, 0.17], atol=1e-7)


def test_usemonalb_leaves_real_exes_monthly_albbck_in_place():
    lu, soil, landmask, snow, xice = _landuse_inputs()
    valid = datetime(2026, 10, 2, 21)
    albedo12m = _monthly(lu.shape, 10.0)            # percent, 11..22
    snoalb = np.full(lu.shape, 70.0)                # percent
    static = {"ALBEDO12M": albedo12m, "SNOALB": snoalb, "LANDMASK": landmask}
    cfg = _ruc_config(usemonalb=True)
    kwargs = usemonalb_landuse_inputs(cfg, static, valid)
    result = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=valid, cen_lat=38.5, **_MODIS, **kwargs)
    expected = monthly_background_albedo(albedo12m, landmask, valid)
    # Land takes the interpolated monthly value, water 0.08 (:1235-1238).
    np.testing.assert_array_equal(result.albbck, expected)
    assert np.all(result.albbck[landmask < 0.5] == np.float32(0.08))
    land = landmask > 0.5
    interp = monthly_interp_to_date(albedo12m, valid) / 100.0
    np.testing.assert_allclose(result.albbck[land], interp[land], rtol=1e-6)
    # No snow: ALBEDO = ALBBCK.
    np.testing.assert_array_equal(result.albedo, result.albbck)
    # Everything the switch does not touch is the table's, as before.
    off = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=valid, cen_lat=38.5, **_MODIS)
    for name in ("embck", "emiss", "z0", "znt", "mavail", "ivgtyp",
                 "isltyp", "landmask", "xland", "lakemask", "snowc"):
        np.testing.assert_array_equal(getattr(result, name),
                                      getattr(off, name))


def test_usemonalb_snow_takes_snoalb_and_sea_ice_keeps_the_table_row():
    lu, soil, landmask, snow, xice = _landuse_inputs()
    valid = datetime(2026, 1, 20, 12)
    snow = snow.copy()
    snow[0, 0] = 25.0                               # SNOW >= 10 -> SNOWC = 1
    xice = xice.copy()
    xice[1, 2] = 0.7                                # a water cell with ice
    albedo12m = _monthly(lu.shape, 10.0)
    snoalb = np.full(lu.shape, 70.0)
    static = {"ALBEDO12M": albedo12m, "SNOALB": snoalb, "LANDMASK": landmask}
    kwargs = usemonalb_landuse_inputs(_ruc_config(usemonalb=True), static,
                                      valid)
    on = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=valid, cen_lat=38.5, **_MODIS, fractional_seaice=True,
        **kwargs)
    off = initialize_landuse(
        lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
        valid_time=valid, cen_lat=38.5, **_MODIS, fractional_seaice=True)
    # :1614-1615 snow-covered cell: ALBEDO = SNOALB (fraction).
    assert on.snowc[0, 0] == 1.0
    assert on.albedo[0, 0] == np.float32(0.70)
    # :1635 sea ice: ALBBCK is the table's ice row under both settings and
    # the fractional blend with 0.08 open water is unchanged.
    assert on.ivgtyp[1, 2] == 15
    assert on.albbck[1, 2] == off.albbck[1, 2]
    assert on.albedo[1, 2] == off.albedo[1, 2]
    assert on.albedo[1, 2] == pytest.approx(0.7 * off.albbck[1, 2]
                                            + 0.3 * 0.08, abs=1e-6)


def test_usemonalb_without_the_monthly_field_is_refused_by_name():
    lu, soil, landmask, snow, xice = _landuse_inputs()
    with pytest.raises(ValueError, match="usemonalb=True needs"):
        initialize_landuse(
            lu, soil_type=soil, landmask=landmask, snow=snow, xice=xice,
            valid_time=datetime(2026, 10, 2), cen_lat=38.5, **_MODIS,
            usemonalb=True)
    with pytest.raises(ValueError, match="lacks \\['ALBEDO12M'"):
        usemonalb_landuse_inputs(_ruc_config(usemonalb=True),
                                 {"SNOALB": np.zeros((2, 4)),
                                  "LANDMASK": landmask}, datetime(2026, 10, 2))
    assert usemonalb_landuse_inputs(_ruc_config(), {}, datetime(2026, 10, 2)) == {}


def test_monthly_snow_albedo_is_percent_to_fraction_with_water_008():
    landmask = np.array([[1.0, 0.0]])
    np.testing.assert_array_equal(
        monthly_snow_albedo(np.array([[84.0, 55.0]]), landmask),
        np.array([[0.84, 0.08]], np.float32))


# --- monthly_interp_to_date against the fork's integer-day arithmetic ---------

def _wrf_monthly_interp(field_in, date: datetime):
    """``module_initialize_real.F:7545-7611`` (HRRR v4.1.21 fork), literal.

    middle(l) = julyr*1000 + julday of the 15th; middle(0) = middle(1) - 31;
    middle(13) = middle(12) + 31; the bracket is
    ``middle(l) < target <= middle(l+1)``; integer-day weights.
    """
    year = date.year
    middle = [0] * 14
    for month in range(1, 13):
        middle[month] = year * 1000 + datetime(year, month, 15).timetuple().tm_yday
    middle[0] = middle[1] - 31
    middle[13] = middle[12] + 31
    target = year * 1000 + date.timetuple().tm_yday
    for l in range(0, 13):
        if middle[l] < target <= middle[l + 1]:
            month1, month2 = (12, 1) if l in (0, 12) else (l, l + 1)
            return ((field_in[month2 - 1] * (target - middle[l])
                     + field_in[month1 - 1] * (middle[l + 1] - target))
                    / (middle[l + 1] - middle[l]))
    raise AssertionError(date)


@pytest.mark.parametrize("date", [
    datetime(2026, 1, 1), datetime(2026, 1, 15), datetime(2026, 2, 14),
    datetime(2026, 3, 15, 23, 59), datetime(2026, 4, 30), datetime(2026, 5, 16),
    datetime(2026, 6, 15), datetime(2026, 7, 4), datetime(2026, 8, 31),
    datetime(2026, 9, 1), datetime(2026, 10, 2, 21), datetime(2026, 11, 30),
    datetime(2026, 12, 31), datetime(2024, 2, 29), datetime(2024, 12, 16),
])
def test_monthly_interp_to_date_matches_the_fork_exactly(date):
    rng = np.random.default_rng(7)
    monthly = rng.uniform(0.0, 60.0, size=(12, 3, 2)).astype(np.float32)
    expected = _wrf_monthly_interp(monthly, date)
    np.testing.assert_array_equal(prescribed_monthly_field(monthly, date),
                                  expected)


def test_prescribed_surface_fields_round_each_real_operation():
    rng = np.random.default_rng(19)
    monthly = rng.uniform(0.0, 60.0, size=(12, 7, 5))
    date = datetime(2026, 10, 2, 21)
    real_input = monthly.astype(np.float32)
    expected = _wrf_monthly_interp(real_input, date)
    legacy = monthly_interp_to_date(monthly, date)
    assert np.any(legacy.astype(np.float32).view(np.uint32)
                  != expected.view(np.uint32))
    np.testing.assert_array_equal(
        surface_leaf_area(_ruc_config(rdlai2d=True), monthly, date), expected)
    np.testing.assert_array_equal(
        surface_leaf_area(_ruc_config(), monthly, date), legacy)
    np.testing.assert_array_equal(
        surface_leaf_area(_ruc_config(sf_surface_physics=2, rdlai2d=True),
                          monthly, date), legacy)
    mask = np.ones((7, 5), np.float32)
    np.testing.assert_array_equal(
        monthly_background_albedo(monthly, mask, date),
        expected / np.float32(100.0))


def test_other_surface_models_keep_their_previous_albedo_seed():
    cfg = _ruc_config(sf_surface_physics=2, usemonalb=True, rdlai2d=True,
                      num_soil_layers=4)
    assert usemonalb_landuse_inputs(cfg, {}, datetime(2026, 10, 2)) == {}


def test_ruc_monthly_snow_albedo_uses_real_division_and_the_water_value():
    from woof.core.noah import noah_initial_snow_albedo

    static = {"SNOALB": np.array([[69.182983, 44.921673]], np.float64),
              "LANDMASK": np.array([[1., 0.]], np.float32),
              "LU_INDEX": np.array([[1, 17]], np.int32)}
    on = _ruc_config(usemonalb=True)
    np.testing.assert_array_equal(
        surface_snow_albedo(on, static, None),
        monthly_snow_albedo(static["SNOALB"], static["LANDMASK"]))
    off = _ruc_config()
    np.testing.assert_array_equal(
        surface_snow_albedo(off, static, None),
        noah_initial_snow_albedo(static["SNOALB"], static["LU_INDEX"], None,
                                rdmaxalb=off.rdmaxalb))


def test_prescribed_monthly_fields_match_the_compiled_fork():
    fixture = (Path(__file__).parents[1] / "woof/data/ruc/oracle"
               / "monthly_interp.json")
    oracle = json.loads(fixture.read_text())
    monthly = np.asarray(oracle["monthly_bits"], np.uint32).view(np.float32)
    output = np.asarray(oracle["output_bits"], np.uint32)
    assert monthly.shape == (12, 7, 5)
    assert len(oracle["dates"]) == 15
    for case, date in enumerate(oracle["dates"]):
        actual = prescribed_monthly_field(
            monthly.astype(np.float64), datetime.fromisoformat(date))
        np.testing.assert_array_equal(actual.view(np.uint32), output[case])


# --- the offline-child road ---------------------------------------------------

def _child_call(monkeypatch, cfg, statics):
    """Run offline_child_run._initialize_child_physics up to the driver
    allocation and return the keywords land use and the RUC mosaic got."""
    import sys
    from types import SimpleNamespace

    from woof import offline_child_run
    from woof.core import landuse, physics
    from woof.ingest import ruc_mosaic

    seen = {}

    def landuse_init(*args, **kwargs):
        seen["landuse"] = kwargs
        return None

    def mosaic_inputs(cfg, static, **kwargs):
        seen["mosaic"] = kwargs
        return {}

    monkeypatch.setattr(landuse, "initialize_landuse", landuse_init)
    monkeypatch.setattr(ruc_mosaic, "ruc_mosaic_physics_inputs", mosaic_inputs)
    monkeypatch.setattr(physics, "initialize_physics",
                        lambda *args, **kwargs: SimpleNamespace(fields={}))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    fields = {name: np.ones((2, 2)) for name in
              ("XLAT", "XLONG", "LU_INDEX", "ISLTYP", "LANDMASK", "SNOW",
               "TSK", "VEGFRA", "TMN")}
    fields.update(TSLB=np.ones((9, 2, 2)), SMOIS=np.ones((9, 2, 2)))
    surface = SimpleNamespace(fields=fields, identity={
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17, "ISLAKE": 21,
        "ISICE": 15, "ISOILWATER": 14})
    initial = SimpleNamespace(fields=fields, receipt={"p_top": 5000.0})
    offline_child_run._initialize_child_physics(
        None, cfg, initial, surface, datetime(2026, 1, 20, 12),
        **({} if statics is None else {"terrain_drag_static": statics}))
    return seen


def test_offline_child_land_use_follows_fractional_seaice(monkeypatch):
    # module_physics_init.F:1463-1465: landuse_init and the RUC mosaic read
    # the threshold the seam runs; the default passes False, as before.
    off = _child_call(monkeypatch, _ruc_config(), None)
    assert off["landuse"]["fractional_seaice"] is False
    assert off["mosaic"]["fractional_seaice"] is False
    assert "usemonalb" not in off["landuse"]
    on = _child_call(monkeypatch, _ruc_config(fractional_seaice=1), None)
    assert on["landuse"]["fractional_seaice"] is True
    assert on["mosaic"]["fractional_seaice"] is True


def test_offline_child_usemonalb_takes_the_childs_own_static_albedo(
        monkeypatch):
    from woof.offline_child import OfflineChildContractError

    landmask = np.array([[1.0, 1.0], [0.0, 1.0]])
    statics = {"ALBEDO12M": _monthly((2, 2), 10.0),
               "SNOALB": np.full((2, 2), 70.0), "LANDMASK": landmask}
    seen = _child_call(monkeypatch, _ruc_config(usemonalb=True), statics)
    kwargs = seen["landuse"]
    assert kwargs["usemonalb"] is True
    np.testing.assert_array_equal(
        kwargs["albbck_monthly"],
        monthly_background_albedo(statics["ALBEDO12M"], landmask,
                                  datetime(2026, 1, 20, 12)))
    np.testing.assert_array_equal(
        kwargs["snoalb"], monthly_snow_albedo(statics["SNOALB"], landmask))
    # A child-grid file or the parent-derived surface carries no monthly
    # albedo: refused by name instead of running the table albedo.
    with pytest.raises(OfflineChildContractError, match="usemonalb=true on a "
                                                        "downscaled child"):
        _child_call(monkeypatch, _ruc_config(usemonalb=True), None)


def test_a_case_catalogue_entry_may_set_the_three_hrrr_surface_switches():
    # woof/case_catalog.py refused fractional_seaice with the registry's
    # stale "not implemented" reason while RunConfig carried and the RUC
    # seam read it; the registry row is now implemented, so the catalogue
    # takes it and still type-checks it against the declared enum.
    from woof.case_catalog import CatalogError, validate_native_overrides
    validate_native_overrides({"shared": {
        "fractional_seaice": 1, "usemonalb": True, "rdlai2d": True}})
    with pytest.raises(CatalogError, match="fractional_seaice"):
        validate_native_overrides({"shared": {"fractional_seaice": 2}})


def test_monthly_surface_template_selects_the_switches_without_overrides():
    from woof.domain_wizard import experiment_from_text, render_config
    from woof.physics_compat import single_domain_runtime_switches
    from woof.physics_registry import physics_registry

    profile = "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1"
    settings = single_domain_runtime_switches(profile)
    assert settings["usemonalb"] is True
    assert settings["rdlai2d"] is True
    assert settings["fractional_seaice"] == 1
    assert settings["ra_rrtmg_variant"] == "rrtmg_legacy"
    text = render_config(
        name="monthly-surface", start_time=datetime(2026, 10, 2, 21), hours=1,
        projection={"map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -97.5,
                    "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5},
        dims=[(96, 64)], ratios=(), fetch_hints={"source": "hrrr-prs"},
        case_data=None, profile=profile)
    exp = experiment_from_text(text, source="monthly-surface.toml")
    cfg = exp.domains[0].run
    assert cfg.usemonalb and cfg.rdlai2d and cfg.fractional_seaice == 1
    validate_run_config(cfg)
    templates = physics_registry()["templates"]
    for key, value in templates.items():
        if key not in (profile,
                "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1",
                "wsm6-ysu-mm5-ssib-rrtm-dudhia-v1"):
            assert "fractional_seaice" not in value.get("parameters", {})
