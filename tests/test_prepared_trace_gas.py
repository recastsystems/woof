"""Declared trace gases remain pinned inputs of every prepared physics setup."""
from datetime import datetime
from types import SimpleNamespace
import hashlib
import sys

import numpy as np
import pytest

from woof.case_data import trace_gas_overrides_from_config
from woof.branch import emit_experiment_toml


def _config(tmp_path, co2=0.000731):
    table = dict(forcing="not-fetched.grib", vtable="Vtable", wps_namelist="namelist.wps",
                 geog_root="geog", sfcp_to_sfcp=True, output_title="gas control")
    if co2 is not None:
        table["co2_vmr"] = co2
    path = tmp_path / "case.toml"
    path.write_text(emit_experiment_toml({"case_data": table}), encoding="utf-8")
    return path


@pytest.mark.parametrize("co2", [None, 0.000731, 0.005])
def test_gas_setting_consumes_exact_hash_bound_case_authority(tmp_path, co2):
    path = _config(tmp_path, co2)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert trace_gas_overrides_from_config(path, expected_sha256=digest) == (
        None if co2 is None else {"co2": co2})
    path.write_text(path.read_text() + "# changed authority\n")
    with pytest.raises(ValueError, match="digest differs"):
        trace_gas_overrides_from_config(path, expected_sha256=digest)


def test_gas_setting_uses_captured_authority_after_original_changes(tmp_path, monkeypatch):
    from woof.config_authority import authority_environment
    path = _config(tmp_path)
    payload = path.read_bytes()
    snapshot = tmp_path / "captured.toml"
    snapshot.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    for key, value in authority_environment(source=path, payload_path=snapshot, sha256=digest).items():
        monkeypatch.setenv(key, value)
    path.write_text("changed original, no longer TOML")
    assert trace_gas_overrides_from_config(path, expected_sha256=digest) == {"co2": .000731}
    snapshot.write_bytes(payload + b"# tampered\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        trace_gas_overrides_from_config(path, expected_sha256=digest)


@pytest.mark.parametrize("value", [True, -0.0004, 0.])
def test_gas_setting_keeps_existing_case_owner_validation(tmp_path, value):
    with pytest.raises(ValueError, match="co2_vmr"):
        trace_gas_overrides_from_config(_config(tmp_path, value))


@pytest.fixture
def numpy_as_cupy(monkeypatch):
    """NumPy stands in for ``cupy`` for one test, and leaves no module bound to it.

    The initializer imports cupy inside its body, so only that import has to
    see the stand-in.  A module imported for the first time while the stand-in
    is installed keeps NumPy as its ``cp`` for the rest of the process: the
    breakage this prevents is woof.core.sase imported that way, after which
    test_sase.py on the same worker failed on ``numpy.asnumpy``.  So the
    modules the initializer reaches are imported before the stand-in, and any
    module still first imported under it is forgotten afterwards, together with
    its attribute on its package, so the next import builds it on real cupy.
    """

    from woof.ingest import hrrr_physics  # noqa: F401 - imported before the stand-in
    import woof.core.diagnostics  # noqa: F401
    import woof.core.landuse  # noqa: F401
    import woof.core.physics  # noqa: F401
    import woof.static.build  # noqa: F401

    before = set(sys.modules)
    monkeypatch.setitem(sys.modules, "cupy", np)
    yield
    for name in sorted(set(sys.modules) - before, reverse=True):
        module = sys.modules.pop(name)
        package, _, attribute = name.rpartition(".")
        if package and getattr(sys.modules.get(package), attribute, None) is module:
            delattr(sys.modules[package], attribute)


@pytest.mark.parametrize("co2", [None, {"co2": .000731}])
@pytest.mark.parametrize("simulation_start", [None, datetime(2026, 7, 19, 21)])
def test_prepared_initializer_passes_gas_vertical_chunk_and_parent_to_factory(
        monkeypatch, numpy_as_cupy, co2, simulation_start):
    # Exercise the entire prepared initializer through its real input validation.
    # Replace only GPU execution and the already-tested radiation factory with
    # recorders; this control makes no claim about radiative transfer numbers.
    from woof.ingest import hrrr_physics
    import woof.core.diagnostics as diagnostics
    import woof.core.landuse as landuse
    import woof.core.physics as physics
    monkeypatch.setattr(diagnostics, "update_diagnostics", lambda *a: None)
    landuse_dates = []
    def initialize_landuse(*args, **kwargs):
        landuse_dates.append(kwargs["valid_time"])
        return object()
    monkeypatch.setattr(landuse, "initialize_landuse", initialize_landuse)
    monkeypatch.setattr(hrrr_physics, "noah_initial_snow_albedo", lambda *a, **k: .8)
    shape = (3, 3)
    one = np.ones(shape, dtype=np.float32)
    fields = {name: one.copy() for name in hrrr_physics._CANONICAL_SURFACE_FIELDS}
    fields.update(TSK=280*one, TMN=280*one, TSLB=np.full((4,*shape),280.),
                  SMOIS=np.full((4,*shape),.2), SH2O=np.full((4,*shape),.2),
                  SEAICE=0*one, SNOW=0*one, SNOWH=0*one)
    surface = SimpleNamespace(fields=fields)
    state = object()
    result = SimpleNamespace(state=state, surface_pressure=99000*one, surface_qv=.01*one)
    met = SimpleNamespace(fields=dict(LANDSEA=one, T2=280*one, SKINTEMP=280*one,
                        U10=np.ones((3,4)), V10=np.ones((4,3))))
    static = dict(LU_INDEX=7*one, SCT_DOM=6*one, LANDMASK=one,
                  GREENFRAC=np.full((12,*shape),.5), LAI12M=np.full((12,*shape),2.),
                  SNOALB=80*one)
    cfg = SimpleNamespace(nx=3, ny=3, sf_surface_physics=2, num_soil_layers=4,
                          hypsometric_opt=2, rdmaxalb=True)
    grid = SimpleNamespace(ref_lat=38., latlon_mass=lambda: (38*one,-97*one))
    attrs = dict(MMINLU="MODIFIED_IGBP_MODIS_NOAH",ISWATER=17,ISLAKE=21,ISICE=15)
    parent = object()
    radiation = object()
    factory_calls, driver_calls = [], []
    def factory(*args, **kwargs):
        factory_calls.append((args,kwargs))
        return radiation
    monkeypatch.setitem(sys.modules, "woof.core.radiation_composition",
                        SimpleNamespace(make_radiation=factory))
    def initialize(*args, **kwargs):
        driver_calls.append((args,kwargs))
        return SimpleNamespace(noah_params=None, fields={
            key: np.zeros(shape) for key in
            ("snoalb","lai","shdmin","shdmax","psfc","t2","q2","th2","u10","v10")})
    monkeypatch.setattr(physics, "initialize_physics", initialize)
    when = datetime(2026,7,20)
    hrrr_physics.initialize_prepared_physics(result,cfg,met,surface,static,attrs,grid,when,
        p_top=7300., column_chunk=23, trace_gas_overrides=co2, ozone_parent=parent,
        simulation_start_time=simulation_start)
    args, kwargs = factory_calls[0]
    assert args[0] is cfg and args[1] == (simulation_start or when)
    assert landuse_dates == [when]
    assert driver_calls[0][1]["radiation_start_time"] == (simulation_start or when)
    np.testing.assert_array_equal(args[2], 38*one)
    np.testing.assert_array_equal(args[3], -97*one)
    assert kwargs == dict(p_top=7300., column_chunk=23,
                          trace_gas_overrides=co2, ozone_parent=parent)
    assert driver_calls[0][1]["radiation"] is radiation


@pytest.mark.parametrize("value", [None, {"co2": .000731}])
def test_native_cache_gas_policy_is_bound_to_its_preparation_proof(value):
    from test_hrrr_prepared_background import _identity_pair, runner
    identity, proof = _identity_pair()
    identity["trace_gas_overrides"] = value
    proof["trace_gas_overrides"] = value
    assert runner._validate_hrrr_source_identity(identity, proof) is identity
    proof["trace_gas_overrides"] = {"co2": .000532}
    with pytest.raises(ValueError, match="trace-gas settings differ"):
        runner._validate_hrrr_source_identity(identity, proof)


from test_rrtmg_legacy_wiring import profile, env  # independent real column deck


@pytest.mark.gpu
def test_prepared_gas_reaches_real_radiative_outputs_and_restart_identity(env, monkeypatch):
    """Numerical radiation witness, not a dynamical forecast.

    The independent real-column deck already supplies atmospheric diagnostics;
    recorders replace unrelated land/state setup. The prepared initializer,
    gas factory, CUDA radiation solver, tile clone and restart identity are real.
    """
    import cupy as cp
    from woof.config import RunConfig
    from woof.ingest import hrrr_physics
    from woof.core import diagnostics, landuse, physics
    from woof.core.rrtmgp import RRTMGPRadiation
    from woof.core.radiation_composition import adapter_restart_identity
    from woof.core.streaming import _tile_scheme
    from test_rrtmg_legacy_wiring import START

    monkeypatch.setattr(diagnostics, "update_diagnostics", lambda *a: None)
    monkeypatch.setattr(landuse, "initialize_landuse", lambda *a, **k: object())
    monkeypatch.setattr(hrrr_physics, "noah_initial_snow_albedo", lambda *a, **k: .8)
    # The SW reference deck's radii are solver operands, whereas this driver
    # contract consumes diagnosed state radii in microns (as in its own tests).
    state = SimpleNamespace(**(vars(env.state) | {
        name: cp.full_like(env.state.qv, value)
        for name, value in (("effc",10.),("effi",30.),("effs",50.))}))
    shape = (env.ny, env.nx)
    one = np.ones(shape, np.float32)
    cfg = RunConfig(nx=env.nx,ny=env.ny,nz=env.nz,dx=3000.,dy=3000.,ztop=20000.,
        dt=10.,run_seconds=120.,moist=True,mp_physics=8,sf_surface_physics=2,
        ra_lw_physics=4,ra_sw_physics=4,ra_rrtmg_variant="rte-rrtmgp")
    surface = {name: one.copy() for name in hrrr_physics._CANONICAL_SURFACE_FIELDS}
    surface.update(TSK=280*one,TMN=280*one,TSLB=np.full((4,*shape),280.),
        SMOIS=np.full((4,*shape),.2),SH2O=np.full((4,*shape),.2),
        SEAICE=0*one,SNOW=0*one,SNOWH=0*one)
    result = SimpleNamespace(state=state,surface_pressure=99000*one,surface_qv=.01*one)
    met = SimpleNamespace(fields=dict(LANDSEA=one,T2=280*one,SKINTEMP=280*one,
        U10=np.ones((env.ny,env.nx+1)),V10=np.ones((env.ny+1,env.nx))))
    static = dict(LU_INDEX=7*one,SCT_DOM=6*one,LANDMASK=one,
        GREENFRAC=np.full((12,*shape),.5),LAI12M=np.full((12,*shape),2.),SNOALB=80*one)
    lat, lon = env.lat.reshape(shape), env.lon.reshape(shape)
    grid = SimpleNamespace(ref_lat=38.,latlon_mass=lambda:(lat,lon))
    attrs = dict(MMINLU="MODIFIED_IGBP_MODIS_NOAH",ISWATER=17,ISLAKE=21,ISICE=15)
    def attach(*args, radiation, **kwargs):
        return SimpleNamespace(radiation_callable=radiation,noah_params=None,
            fields={name:cp.zeros(shape,cp.float32) for name in
            ("snoalb","lai","shdmin","shdmax","psfc","t2","q2","th2","u10","v10")})
    monkeypatch.setattr(physics,"initialize_physics",attach)
    calls = dict(atmosphere=env.atmosphere, fields=env.fields, state=state, cfg=cfg)
    engines, outputs = [], []
    for gases in (None, {"co2": .000731}):
        driver = hrrr_physics.initialize_prepared_physics(result,cfg,met,
            SimpleNamespace(fields=surface),static,attrs,grid,START,
            p_top=env.p_top,column_chunk=16,trace_gas_overrides=gases)
        engine = driver.radiation_callable
        direct = RRTMGPRadiation(START,lat,lon,column_chunk=16,trace_gas_overrides=gases)
        measured, reference = engine(**calls), direct(**calls)
        for name in ("rthratenlw","rthratensw","glw","swdown","gsw","olr"):
            np.testing.assert_array_equal(cp.asnumpy(getattr(measured,name)),
                                          cp.asnumpy(getattr(reference,name)))
        engines.append(engine)
        outputs.append(measured)
    assert engines[1].trace_vmr["co2"] == .000731
    maximum_glw_change = float(cp.max(cp.abs(outputs[1].glw-outputs[0].glw)))
    assert maximum_glw_change > .01
    print(f"prepared CO2 witness: max GLW change {maximum_glw_change:.9g} W m-2")
    assert adapter_restart_identity(engines[0]) != adapter_restart_identity(engines[1])
    twin = _tile_scheme(engines[1],lat,lon)
    assert twin.trace_gas_overrides == {"co2": .000731}
    assert adapter_restart_identity(twin) == adapter_restart_identity(engines[1])
    repeated = twin(**calls)
    np.testing.assert_array_equal(cp.asnumpy(repeated.glw),cp.asnumpy(outputs[1].glw))


@pytest.mark.parametrize("kind", ["metem", "wrfinput"])
@pytest.mark.parametrize("nested", [False, True])
def test_external_input_provider_carries_pinned_gas_and_declared_column_cap(tmp_path, monkeypatch, kind, nested):
    from woof import runtime, wrfinput_forecast, metem_forecast
    from woof.ingest import wrfinput, prepared_cache, hrrr_physics
    from woof.core import radiation_composition
    from woof.ingest import lateral_bc
    path = _config(tmp_path)
    from dataclasses import replace
    from test_cam_ozone import _tree
    exp = _tree()
    exp = replace(exp, domains=exp.domains if nested else (exp.root,), column_chunk=23,
                  vertical=replace(exp.vertical, p_top=7300.))
    inputs = SimpleNamespace(experiment_config=path,experiment=exp,boundaries=object(),
        authority_sha256={"experiment_config":hashlib.sha256(path.read_bytes()).hexdigest()})
    domain = exp.domains[-1] if nested else exp.root
    state, radiation = object(), object()
    when = datetime(2001,6,15)
    raw = dict(XLAT=np.full((2,3),38.),XLONG=np.full((2,3),-97.))
    grid = SimpleNamespace(latlon_mass=lambda: (raw["XLAT"],raw["XLONG"]))
    restored = SimpleNamespace(raw=raw,initial_result=object(),met=object(),surface=object(),
        global_attributes={"CEN_LAT":37.5})
    bundle = SimpleNamespace(restored=restored,landuse=object(),cache=object(),
        cache_identity={},static_fields={},fractional_seaice=False,isoilwater=19,
        geog_selection=SimpleNamespace(landuse_global_attrs=lambda:{}))
    calls, factories = [], []
    def initialize(*args, **kwargs):
        calls.append((args,kwargs))
        return object()
    def factory(*args, **kwargs):
        factories.append((args,kwargs))
        return radiation
    monkeypatch.setattr(runtime,"declared_constant_glw",lambda exp:None)
    monkeypatch.setattr(prepared_cache,"restore_prepared_cache",lambda *a,**k:restored)
    monkeypatch.setattr(hrrr_physics,"initialize_prepared_physics",initialize)
    monkeypatch.setattr(wrfinput,"initialize_wrfinput_physics",initialize)
    monkeypatch.setattr(wrfinput,"restore_domain_state",lambda *a,**k:state)
    monkeypatch.setattr(wrfinput_forecast,"wrf_initial_result",lambda *a:object())
    monkeypatch.setattr(lateral_bc,"attach_lateral_boundaries",lambda *a:None)
    monkeypatch.setattr(radiation_composition,"make_radiation",factory)
    provider = (metem_forecast.MetemInitialization(inputs) if kind=="metem"
                else wrfinput_forecast.WrfInitialization(inputs))
    initialized = provider.restore_domain(domain,grid,bundle,start_time=when,
        scratch_arena=None,dycore_state_workspace=None)
    initialized.initialize_physics()
    args, kwargs = calls[0]
    selected = kwargs if kind=="metem" else factories[0][1]
    assert selected["trace_gas_overrides"] == {"co2":.000731}
    assert selected["column_chunk"] == 23
    assert selected["p_top"] == 7300.
    if kind=="metem":
        assert args[0] is restored.initial_result and args[2] is restored.met
        assert args[3] is restored.surface
        assert kwargs["fractional_seaice"] is False and kwargs["isoilwater"] == 19
    else:
        assert args[0] is state and args[1] is restored
        assert selected["center_lat"] == 37.5
        assert kwargs["radiation"] is radiation
        assert factories[0][0][0] is domain.run and factories[0][0][1] == when
        assert factories[0][0][2] is raw["XLAT"] and factories[0][0][3] is raw["XLONG"]

    if nested:
        from woof.core.cam_ozone import DriverOzoneProvider
        assert isinstance(selected["ozone_parent"], DriverOzoneProvider)
        cam = kwargs["cam_ozone"]
        assert cam.mode == "parent-interpolated" and cam.column_chunk == 23
        assert cam.start_time == exp.start_time
        np.testing.assert_array_equal(cam.latitude_deg, raw["XLAT"])
    else:
        assert "cam_ozone" not in kwargs
