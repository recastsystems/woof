"""Eclipse door, GPU geometry and totality recovery regressions."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import numpy as np
import pytest
from woof.eclipse_windows import besselian_row_at, EclipseGeometry, EclipseRadiation


def test_utc_date_selection_and_endpoints():
    for hour in (15, 18, 21):
        assert besselian_row_at(datetime(2024, 4, 8, hour))[0] is not None
    assert besselian_row_at(datetime(2024, 4, 8, 14, 59, 59))[0] is None
    assert besselian_row_at(datetime(2024, 4, 9, 18))[0] is None
    assert besselian_row_at(datetime(2024, 4, 8, 11, tzinfo=timezone(timedelta(hours=-7))))[1] == 18


@pytest.mark.gpu
def test_full_corridor_shadow_sequence_and_nightside():
    import cupy as cp
    lat, lon = np.meshgrid(np.linspace(25, 48, 192), np.linspace(-110, -73, 256), indexing="ij")
    geometry = EclipseGeometry(lat, lon)
    maxima = []
    for minute in range(15*60, 21*60+1, 5):
        field = geometry.at(datetime(2024, 4, 8)+timedelta(minutes=minute))
        assert bool(cp.all(cp.isfinite(field)))
        assert float(field.min()) >= 0 and float(field.max()) <= 1
        maxima.append(float(field.max()))
    assert max(maxima) == 1
    assert maxima[0] == 0 and maxima[-1] == 0
    geometry.at(datetime(2024, 4, 9, 18))
    assert not bool(cp.any(geometry.obscuration))
    back = EclipseGeometry(-lat, lon+180)
    assert not bool(cp.any(back.at(datetime(2024, 4, 8, 18, 45))))


@pytest.mark.gpu
def test_totality_recovery_scales_heating_and_all_sw_fluxes(monkeypatch):
    import cupy as cp
    import woof.core.physics as physics
    monkeypatch.setattr(physics, "couple_column_tendencies", lambda *a, **k: SimpleNamespace(rtheta=k["rtheta"].copy()))
    radiation = SimpleNamespace(start_time=datetime(2024,4,8,12),
                                latitude_deg=np.zeros((4,5)), longitude_deg=np.zeros((4,5)))
    eclipse = EclipseRadiation(radiation)
    driver = SimpleNamespace(fields={n:cp.full((4,5), 800, cp.float32) for n in
        ("swdown","gsw","swddir","swddif","swddni","swupt","swdntc","lwupb")},
        rthratensw=cp.full((3,4,5), 0.01, cp.float32),
        rthratenlw=cp.full((3,4,5), -0.02, cp.float32),
        rthraten=cp.full((3,4,5), -0.01, cp.float32),
        radiation_tendencies=SimpleNamespace(rtheta=cp.full((3,4,5), -0.01, cp.float32)), swint=None)
    eclipse.capture(driver)
    state=SimpleNamespace(elapsed_seconds=0)
    for obscuration in (0.5,1.0,0.25,0.0):
        eclipse.geometry.at=lambda valid, **kwargs: cp.full((4,5), obscuration, cp.float32)
        eclipse.apply(driver,state,SimpleNamespace(radt=0,radt_minutes=0))
        for name in ("swdown","gsw","swddir","swddif","swddni","swupt","swdntc"):
            cp.testing.assert_array_equal(driver.fields[name], cp.full((4,5),800*(1-obscuration),cp.float32))
        cp.testing.assert_allclose(driver.rthratensw, 0.01*(1-obscuration))
        cp.testing.assert_array_equal(driver.fields["lwupb"],800)
        cp.testing.assert_allclose(driver.rthraten, -0.02+0.01*(1-obscuration))


def test_explicit_off_and_non_eclipse_runs_allocate_no_overlay():
    cfg=SimpleNamespace(ra_sw_eclipse=0, ra_physics=4, ra_lw_physics=-1,ra_sw_physics=-1, run_seconds=43200)
    radiation=SimpleNamespace(start_time=datetime(2024,4,8,12))
    assert EclipseRadiation.for_run(cfg,radiation) is None
    cfg.ra_sw_eclipse=1
    radiation.start_time=datetime(2024,4,9,12)
    assert EclipseRadiation.for_run(cfg,radiation) is None

@pytest.mark.gpu
def test_cuda_against_unmodified_wrf461_geometry():
    """73,000 public-domain WRF oracle values over the full six-hour window.

    tools/evidence: original phys/module_ra_eclipse.F at WRF v4.6.1,
    solar_eclipse with radt=0, julian=98, gmt=0, year=2024. The fixture
    is source-rounded FP32. Real native-run bitwise captures are replayed
    separately by tools/cloud_aerosol_wrf471_oracle/replay_eclipse.py.
    """
    from pathlib import Path
    import cupy as cp
    rows = np.loadtxt(Path(__file__).parent/'data/eclipse_wrf461_reference.csv',
                      delimiter=',',skiprows=1)
    geometry = EclipseGeometry(rows[:1000,1],rows[:1000,2])
    for block in rows.reshape(73,1000,5):
        actual = geometry.at(datetime(2024,4,8)+timedelta(minutes=float(block[0,0])))
        cp.testing.assert_allclose(actual,cp.asarray(block[:,3]),atol=4e-5,rtol=0)
        cp.testing.assert_array_equal(geometry.mask,cp.asarray(block[:,4],dtype=cp.int32))

@pytest.mark.gpu
def test_undimmed_checkpoint_cache_and_interpolated_fluxes(monkeypatch):
    import cupy as cp
    import woof.core.physics as physics
    monkeypatch.setattr(physics,'couple_column_tendencies',lambda *a,**k:SimpleNamespace(rtheta=k['rtheta'].copy()))
    radiation=SimpleNamespace(start_time=datetime(2024,4,8,12),latitude_deg=np.zeros((4,5)),longitude_deg=np.zeros((4,5)))
    driver=SimpleNamespace(fields={'swdown':cp.full((4,5),800,cp.float32)},
        rthratensw=cp.full((3,4,5),.01,cp.float32),rthratenlw=cp.full((3,4,5),-.02,cp.float32),
        rthraten=cp.full((3,4,5),-.01,cp.float32),
        radiation_tendencies=SimpleNamespace(rtheta=cp.full((3,4,5),-.01,cp.float32)),swint=object())
    original=EclipseRadiation(radiation)
    original.bind(driver)
    assert not original.ready()
    original.capture(driver)
    original.geometry.at=lambda valid, **kwargs:cp.ones((4,5),cp.float32)
    original.apply(driver,SimpleNamespace(elapsed_seconds=0),SimpleNamespace(radt=0,radt_minutes=0))
    # Restored storage remains undimmed even when its published flux is zero.
    resumed=EclipseRadiation(radiation)
    resumed.bind(driver)
    assert resumed.ready()
    cp.testing.assert_array_equal(resumed.base_fields['swdown'],800)
    # SWINT freshly evaluates undimmed sunlight before each overlay call.
    for undimmed in (900,880):
        driver.fields['swdown'].fill(undimmed)
        resumed.geometry.at=lambda valid, **kwargs:cp.full((4,5),.5,cp.float32)
        resumed.apply(driver,SimpleNamespace(elapsed_seconds=60),SimpleNamespace(radt=0,radt_minutes=0))
        cp.testing.assert_array_equal(driver.fields['swdown'],undimmed/2)
        cp.testing.assert_allclose(driver.rthratensw,.005)


def test_eclipse_health_inventory_has_explicit_integer_mask_bounds():
    import numpy as np
    from woof.core.health import gpu_integer_policy, rule_for_field

    assert gpu_integer_policy("surface.ecmask", np.int32) is not None
    mask = rule_for_field("surface.ecmask")
    assert (mask.lower, mask.upper) == (0.0, 3.0)
    obsc = rule_for_field("surface.ecobsc")
    assert (obsc.lower, obsc.upper) == (0.0, 1.0)



def test_off_comparison_reuses_prepared_state_with_recorded_runtime_override(tmp_path, monkeypatch):
    from woof.experiment import load_experiment
    from woof.prepared_single_domain_forecast import preflight_prepared_forecast
    from test_prepared_single_domain_forecast import (
        _prepared_fixture, _bind_synthetic_preflight_geometry, _sha256)
    fixture = _prepared_fixture(tmp_path, "gfs")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    arguments = dict(source=fixture.source, prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        run_seconds=fixture.run_seconds,
        history_interval_seconds=load_experiment(fixture.experiment).root.history_interval_s)
    original = preflight_prepared_forecast(**arguments)
    changed = preflight_prepared_forecast(**arguments, runtime_run_overrides={"ra_sw_eclipse": 0})
    assert changed.cache_identity == original.cache_identity
    assert changed.file_sha256 == original.file_sha256
    assert changed.experiment.root.run.ra_sw_eclipse == 0
    row = next(r for r in changed.execution_plan["physics_overrides"]
               if r["field"] == "domain_config.run.ra_sw_eclipse")
    assert row["prepared"] == 1 and row["executed"] == 0
    assert row["prepared_state_changed"] is False


def test_analytic_callback_without_solar_metadata_remains_attachable():
    cfg=SimpleNamespace(ra_sw_eclipse=1,ra_physics=4,ra_lw_physics=-1,
                        ra_sw_physics=-1,run_seconds=43200)
    assert EclipseRadiation.for_run(cfg,lambda **kwargs: None) is None
    with pytest.raises(ValueError,match="requires start_time"):
        EclipseRadiation.for_run(cfg,SimpleNamespace(latitude_deg=0))


@pytest.mark.gpu
def test_zero_shadow_preserves_held_carrier_rounding_exactly():
    import cupy as cp
    radiation=SimpleNamespace(start_time=datetime(2024,4,8,12),
        latitude_deg=np.full((192,256),36.5),longitude_deg=np.full((192,256),-91.5))
    shape=(49,192,256)
    sw=cp.full(shape,.000043217,cp.float32)
    lw=cp.full(shape,-.000072345,cp.float32)
    # A held carrier has already passed through mass coupling and uncoupling.
    # Deliberately retain a one-word residual that LW+SW would overwrite.
    held=cp.nextafter(lw+sw,cp.full(shape,cp.inf,cp.float32))
    driver=SimpleNamespace(fields={'swdown':cp.full((192,256),800,cp.float32)},
        rthratensw=sw,rthratenlw=lw,rthraten=held.copy(),
        _wrf_rthraten=held.copy(),_wrf_rthraten_coupled=False,swint=None)
    overlay=EclipseRadiation(radiation)
    overlay.capture(driver)
    for elapsed in range(0,3600,60):
        overlay.apply(driver,SimpleNamespace(elapsed_seconds=elapsed),SimpleNamespace(radt=0,radt_minutes=0))
        cp.testing.assert_array_equal(driver._wrf_rthraten,held)
        cp.testing.assert_array_equal(driver.rthraten,held)
        cp.testing.assert_array_equal(driver.rthratensw,sw)
        cp.testing.assert_array_equal(driver.fields['swdown'],800)
