"""Independent LW/SW selections retain their engines, carriers and ownership."""
from dataclasses import replace
from datetime import datetime
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.radiation_composition import ComposedRadiation, make_radiation

PAIRS = [(lw, sw, variant) for lw, sw in product((0, 1, 4, 90), repeat=2)
         for variant in (("rte-rrtmgp", "rrtmg_legacy") if 4 in (lw, sw) else ("rte-rrtmgp",))]


def _cfg(lw, sw, variant="rte-rrtmgp"):
    return RunConfig(nx=12, ny=12, nz=12, dx=3000., dy=3000., ztop=12000.,
        dt=10., run_seconds=120., ra_lw_physics=lw, ra_sw_physics=sw,
        ra_rrtmg_variant=variant)


@pytest.mark.parametrize("lw,sw,variant", PAIRS)
def test_individually_implemented_spectra_do_not_require_a_paired_preset(lw, sw, variant):
    from woof.physics_compat import validate_resolved_physics_vertical_levels
    cfg = validate_run_config(_cfg(lw, sw, variant))
    receipt = validate_resolved_physics_vertical_levels(cfg, p_top=5000.)
    radiation = [item for item in receipt["checks"] if "RRTMG" in item["component"]]
    assert len(radiation) == int(lw == 4) + int(sw == 4)


def test_unselected_longwave_cap_cannot_refuse_legacy_shortwave():
    from woof.physics_compat import validate_resolved_physics_vertical_levels
    cfg = replace(_cfg(0, 4, "rrtmg_legacy"), nz=129)
    receipt = validate_resolved_physics_vertical_levels(cfg, p_top=5000.)
    assert all("longwave" not in item["component"] for item in receipt["checks"])


def test_mixed_variants_keep_modern_workspace_and_legacy_call_peak(monkeypatch):
    from woof.core import preflight as pf
    from woof.core.model import uses_modern_rrtmgp_workspace, SharedRRTMGPChunkWorkspace
    from woof.experiment import experiment_from_run_config
    from woof.core import rrtmg_legacy
    modern = experiment_from_run_config(_cfg(4, 4), datetime(2000, 1, 1))
    child = replace(modern.root, grid_id=2, parent_id=1,
        run=replace(modern.root.run, ra_rrtmg_variant="rrtmg_legacy"))
    mixed = replace(modern, domains=(modern.root, child))
    # A deterministic engine cost isolates the mixed-domain lifecycle arithmetic;
    # the real engine cost is independently covered by legacy pricing tests.
    monkeypatch.setattr(rrtmg_legacy, "legacy_radiation_vram_bytes", lambda **kw: 1234567)
    estimate = pf.estimate_experiment(mixed, column_chunk=16)
    workspace = SharedRRTMGPChunkWorkspace(nz=12, column_chunk=16,
        p_top=mixed.vertical.p_top, _array_module=np)
    assert uses_modern_rrtmgp_workspace(mixed)
    assert estimate.workspace_bytes == workspace.nbytes
    assert estimate.legacy_call_peak_by_domain == (0, 1234567)
    assert estimate.transient_peak_bytes == max(
        estimate.domains[0].transient_bytes, estimate.domains[1].transient_bytes + 1234567)
    assert estimate.uses_legacy_radiation
    assert estimate.k_tables_bytes > 0


@pytest.mark.parametrize("swint", [0, 1])
def test_composition_merges_each_owning_spectrum(monkeypatch, swint):
    import sys
    from woof.core.physics import RadiationResult
    monkeypatch.setitem(sys.modules, "cupy", np)
    shape = (2, 2, 3)
    lw = RadiationResult(np.full(shape, 11, np.float32), np.full(shape, -99, np.float32),
        np.full(shape[1:], -99, np.float32), np.full(shape[1:], 22, np.float32),
        olr=np.full(shape[1:], 33, np.float32))
    sw = RadiationResult(np.full(shape, -88, np.float32), np.full(shape, 44, np.float32),
        np.full(shape[1:], 55, np.float32), np.full(shape[1:], -88, np.float32),
        gsw=np.full(shape[1:], 66, np.float32), coszen=np.full(shape[1:], .7, np.float32),
        swddir=np.full(shape[1:], 40, np.float32),
        swddif=np.full(shape[1:], 15, np.float32))
    class Leaf:
        publishes_olr = True
        def __init__(self, result): self.result = result
        def __call__(self, **kwargs): return self.result
    adapter = ComposedRadiation(datetime(2000,1,1), np.zeros(shape[1:]), np.zeros(shape[1:]),
        longwave_adapter=Leaf(lw), shortwave_adapter=Leaf(sw))
    result = adapter(atmosphere={"pressure": np.zeros(shape)}, fields={},
                     state=SimpleNamespace(), cfg=replace(_cfg(4,1), swint_opt=swint))
    for name in ("rthratenlw", "glw", "olr"):
        assert getattr(result, name) is getattr(lw, name)
    for name in ("rthratensw", "swdown", "gsw", "coszen"):
        assert getattr(result, name) is getattr(sw, name)
    assert result.swddir is sw.swddir
    assert result.swddif is sw.swddif


# Reuse the independent real-column fixture deck, with mixed day/night geography.
from test_rrtmg_legacy_wiring import profile, env


def _wrf_driver_sw(lw, sw):
    """WRF 4.6.1 module_radiation_driver.F:2870, as the legacy engine applies it.

    The driver recovers RTHRATSW from the rounded total after the SW routine
    adds its heating to the LW tendency, so a paired legacy run's SW words are
    fl(fl(LW + SW) - LW).  A spectrum the composition runs on its own carries
    no LW, so its SW words are the solver's SW before that round trip.
    """
    lw = np.asarray(lw, dtype=np.float32)
    total = (lw + np.asarray(sw, dtype=np.float32)).astype(np.float32)
    return (total - lw).astype(np.float32)


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["rte-rrtmgp", "rrtmg_legacy"])
@pytest.mark.parametrize("swint", [0, 1])
def test_selected_spectrum_equals_paired_engine_and_never_calls_inactive_solver(env, monkeypatch, variant, swint):
    import cupy as cp
    from woof.core import rrtmgp, rrtmg_lw, rrtmg_sw
    from test_rrtmg_legacy_wiring import START
    cfg = SimpleNamespace(**(vars(env.cfg) | dict(ra_physics=0, ra_lw_physics=4,
        ra_sw_physics=4, ra_rrtmg_variant=variant, o3input=2, use_mp_re=1,
        swrad_scat=1., dt=10., wrf_rrtmg_compatibility="none", swint_opt=swint)))
    lat, lon = env.lat.reshape(env.ny,env.nx), env.lon.reshape(env.ny,env.nx)
    fields = dict(env.fields, glw=cp.full((env.ny,env.nx), 300., cp.float32))
    # The default physics driver allocates native split history consumers.
    for name in ("swddir", "swddif", "swddni"):
        fields[name] = cp.zeros((env.ny, env.nx), cp.float32)
    atmosphere = dict(env.atmosphere, theta=env.atmosphere["temperature"] / env.atmosphere["exner"])
    state = env.state
    if variant == "rte-rrtmgp":
        # The legacy fixture has zero radii in clear cells; the modern
        # array contract requires valid micron radii throughout. Both
        # modern calls receive this identical state.
        state = SimpleNamespace(**(vars(state) | {
            name: cp.full_like(state.qv, radius)
            for name, radius in (("effc", 10.), ("effi", 30.), ("effs", 50.))}))
    arguments = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    paired = make_radiation(cfg, START, lat, lon, p_top=env.p_top, column_chunk=16)
    reference = paired(**arguments)
    def forbidden(*args, **kwargs):
        raise AssertionError("unselected spectrum solver was called")
    for active in ("lw", "sw"):
        selected = SimpleNamespace(**(vars(cfg) | dict(ra_lw_physics=4 if active=="lw" else 0,
                                                       ra_sw_physics=4 if active=="sw" else 0)))
        adapter = make_radiation(selected, START, lat, lon, p_top=env.p_top, column_chunk=16)
        with monkeypatch.context() as patch:
            if variant == "rrtmg_legacy":
                owner, name = ((rrtmg_sw.CudaSW, "rrtmg_sw_batched") if active=="lw"
                               else (rrtmg_lw, "gpu_rrtmg_lw_batched"))
                patch.setattr(owner, name, forbidden)
            else:
                for name in (("sw_rte", "_sw_rte") if active=="lw" else ("lw_rte", "_lw_rte")):
                    patch.setattr(rrtmgp, name, forbidden)
            actual = adapter(**(arguments | {"cfg": selected}))
        names = ("rthratenlw", "glw", "olr") if active=="lw" else ("rthratensw", "swdown", "gsw", "coszen")
        for name in names:
            got = cp.asnumpy(getattr(actual,name))
            if name == "rthratensw" and variant == "rrtmg_legacy":
                # The paired run applied WRF's round trip with its LW.
                got = _wrf_driver_sw(cp.asnumpy(reference.rthratenlw), got)
            np.testing.assert_array_equal(got, cp.asnumpy(getattr(reference,name)), err_msg=name)
        if active == "sw":
            for name in ("swddir", "swddif"):
                np.testing.assert_array_equal(cp.asnumpy(getattr(actual, name)),
                                              cp.asnumpy(getattr(reference, name)))
        absent = actual.rthratensw if active=="lw" else actual.rthratenlw
        assert bool(cp.all(absent == 0))
        if active == "sw":
            assert actual.glw is fields["glw"]
            assert actual.olr is None


def test_shared_workspace_does_not_reconfigure_legacy_engine():
    from woof.core.radiation_composition import attach_modern_workspace
    from woof.core.rrtmgp import RRTMGPRadiation
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    modern = object.__new__(RRTMGPRadiation)
    modern.column_chunk = 17
    legacy = object.__new__(RRTMGLegacyRadiation)
    legacy.column_chunk = None
    workspace = SimpleNamespace(column_chunk=31)
    attach_modern_workspace(legacy, workspace)
    assert legacy.column_chunk is None
    assert not hasattr(legacy, "chunk_workspace")
    mixed = ComposedRadiation(datetime(2000,1,1), None, None,
        longwave_adapter=legacy, shortwave_adapter=modern)
    attach_modern_workspace(mixed, workspace)
    assert modern.column_chunk == 31 and modern.chunk_workspace is workspace
    assert legacy.column_chunk is None


def test_nested_legacy_ozone_is_carried_and_unknown_leaf_arrays_refuse(monkeypatch):
    from woof.io import restart
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    from test_restart import _cfg as restart_cfg, _shim_driver_state
    cfg = restart_cfg()
    state, driver = _shim_driver_state(cfg, monkeypatch)
    leaf = object.__new__(RRTMGLegacyRadiation)
    ozone = np.arange(state.p.size, dtype=np.float32).reshape(state.p.shape)
    leaf._o33d_grid = ozone
    driver.radiation_callable = ComposedRadiation(datetime(2000,1,1), None, None,
                                                longwave_adapter=leaf)
    manifest = restart._driver_manifest(driver)
    assert manifest["radiation/o33d_grid"] is ozone
    restored = ozone + np.float32(1)
    driver.radiation_callable._o33d_grid = restored
    assert leaf._o33d_grid is restored
    leaf.unpriced_history = np.zeros((2,3), np.float32)
    with pytest.raises(restart.RestartManifestError, match="unpriced_history"):
        restart._driver_manifest(driver)


@pytest.mark.parametrize("lw,sw", [(4,0),(0,4),(4,1),(1,4)])
def test_modern_mixed_microphysics_admits_every_coupled_two_moment_scheme(lw,sw):
    # This used to pin the mp=9 refusal on every pair with a modern arm.
    # Every implemented microphysics selector has an RTE+RRTMGP
    # cloud-optics row now -- Milbrandt-Yau's is its own two-moment radii
    # -- so a modern arm beside an off, Dudhia or RRTM arm validates for
    # mp=9 exactly as it does for Morrison, at plan review.
    for mp_physics in (9, 10):
        validate_run_config(replace(_cfg(lw,sw), mp_physics=mp_physics, moist=True))


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["rte-rrtmgp", "rrtmg_legacy"])
def test_mixed_engines_match_independent_spectra_and_tile_identity(env, variant):
    import cupy as cp
    from test_rrtmg_legacy_wiring import START
    from woof.core.streaming import _tile_scheme
    from woof.core.radiation_composition import radiation_adapters
    from woof.io.restart import (_radiation_setup_identity, _callable_state_check,
        RADIATION_CALLABLE_ARRAYS, RADIATION_CALLABLE_CONTAINERS)
    cfg = SimpleNamespace(**(vars(env.cfg) | dict(ra_physics=0, ra_lw_physics=4,
        ra_sw_physics=4, ra_rrtmg_variant=variant, o3input=2, use_mp_re=1,
        swrad_scat=1., dt=10., wrf_rrtmg_compatibility="none")))
    lat, lon = env.lat.reshape(env.ny,env.nx), env.lon.reshape(env.ny,env.nx)
    state = SimpleNamespace(**(vars(env.state) | {
        name: cp.full_like(env.state.qv, radius)
        for name, radius in (("effc", 10.), ("effi", 30.), ("effs", 50.))}))
    fields = dict(env.fields, glw=cp.full((env.ny,env.nx), 300., cp.float32))
    atmosphere = dict(env.atmosphere, theta=env.atmosphere["temperature"] / env.atmosphere["exner"])
    arguments = dict(atmosphere=atmosphere, fields=fields, state=state)
    references = {}
    for selector in (1, 4):
        selected = SimpleNamespace(**(vars(cfg) | dict(ra_lw_physics=selector, ra_sw_physics=selector)))
        engine = make_radiation(selected, START, lat, lon, p_top=env.p_top, column_chunk=16)
        references[selector] = engine(**arguments, cfg=selected)
    for lw, sw in ((4,1),(1,4)):
        selected = SimpleNamespace(**(vars(cfg) | dict(ra_lw_physics=lw, ra_sw_physics=sw)))
        adapter = make_radiation(selected, START, lat, lon, p_top=env.p_top, column_chunk=16)
        result = adapter(**arguments, cfg=selected)
        for names, selector in ((("rthratenlw", "glw", "olr"),lw),
                                (("rthratensw", "swdown", "gsw", "coszen"),sw)):
            for name in names:
                got = cp.asnumpy(getattr(result,name))
                if name == "rthratensw" and variant == "rrtmg_legacy" and selector == 4:
                    # The paired legacy reference applied WRF's round trip with its LW.
                    got = _wrf_driver_sw(cp.asnumpy(references[4].rthratenlw), got)
                np.testing.assert_array_equal(got,
                    cp.asnumpy(getattr(references[selector],name)), err_msg=f"{lw}/{sw}:{name}")
        for leaf in radiation_adapters(adapter):
            _callable_state_check(leaf, RADIATION_CALLABLE_ARRAYS,
                                 RADIATION_CALLABLE_CONTAINERS, "radiation component")
        identity = _radiation_setup_identity(SimpleNamespace(radiation_callable=adapter), selected)
        assert identity["scheme_ids"] == {"lw":lw,"sw":sw}
        assert identity["algorithm"] == "independent-radiation-spectra-v1"
        tile_lat = cp.full((2,3), 42., cp.float32)
        tile_lon = cp.full((2,3), -88., cp.float32)
        twin = _tile_scheme(adapter, tile_lat, tile_lon)
        assert twin is not adapter
        for old, fresh in zip(radiation_adapters(adapter), radiation_adapters(twin)):
            assert fresh is not old
            np.testing.assert_array_equal(cp.asnumpy(fresh.latitude_deg),cp.asnumpy(tile_lat))
            if isinstance(getattr(old, "longwave", None), bool):
                assert (fresh.longwave, fresh.shortwave) == (old.longwave, old.shortwave)
        changed = _radiation_setup_identity(SimpleNamespace(radiation_callable=twin), selected)
        assert identity != changed


def test_legacy_shortwave_only_prices_no_inactive_longwave(monkeypatch):
    from woof.core import rrtmg_legacy as legacy
    from woof.core.preflight import estimate_experiment
    from woof.experiment import experiment_from_run_config
    def forbidden(*args, **kwargs):
        raise AssertionError("inactive longwave pricing attempted")
    monkeypatch.setattr(legacy, "_lw_coeffs", forbidden)
    monkeypatch.setattr(legacy._lw, "lw_batched_vram_bytes", forbidden)
    cfg = replace(_cfg(0,4,"rrtmg_legacy"), nz=129)
    estimate = estimate_experiment(experiment_from_run_config(cfg, datetime(2000,1,1)))
    assert estimate.workspace_bytes > 0


@pytest.mark.parametrize("lw,sw,variant", [(1,1,"rte-rrtmgp"),(1,4,"rte-rrtmgp"),
                                          (4,1,"rrtmg_legacy")])
def test_explicit_co2_is_consumed_by_each_implemented_absorption_pair(lw,sw,variant):
    from woof.core.radiation_composition import trace_gas_override_status
    assert trace_gas_override_status(_cfg(lw,sw,variant), {"co2":4e-4}) == "applied"


@pytest.mark.parametrize("lw,sw", [(0,0),(0,1),(90,0),(0,90),(90,1),(90,90)])
def test_declared_gas_is_inactive_when_selected_spectra_do_not_use_it(lw,sw):
    from woof.core.radiation_composition import trace_gas_override_status
    declared = {"co2":4e-4}
    assert trace_gas_override_status(_cfg(lw,sw), declared) == "inactive"
    assert declared == {"co2":4e-4}
    if (lw,sw) == (0,0):
        assert make_radiation(_cfg(lw,sw), datetime(2000,1,1), None, None,
                              trace_gas_overrides=declared) is None


def test_modern_absorption_uses_the_override_without_requiring_both_spectra():
    from woof.core.radiation_composition import trace_gas_override_status
    for pair in ((4,4),(4,1),(0,4),(4,0),(90,4)):
        assert trace_gas_override_status(_cfg(*pair), {"co2":4e-4}) == "applied"


@pytest.mark.gpu
@pytest.mark.parametrize("inactive_options",[False,True])
def test_classic_pair_geography_and_dycore_cross_reused_tile_buffers(inactive_options):
    import cupy as cp
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest
    from woof.core import streaming
    from woof.core.physics import initialize_physics
    from woof.core.dycore import step
    cfg = replace(_cfg(1,1), nx=48, ny=40, dt=1., radt=.05, terrain_opt=1)
    reference_cfg = cfg
    if inactive_options:
        cfg = replace(cfg,ra_rrtmg_variant="rrtmg_legacy",o3input=0,use_mp_re=0)
    geo = harness.make_geography(cfg)
    def build(selected):
        state, _ = harness.make_physics_state(selected, 123, geography=geo)
        radiation = make_radiation(selected, datetime(2001,6,15,12), geo.lat, geo.lon,
                                   column_chunk=31)
        initialize_physics(state,selected,radiation=radiation,
            radiation_start_time=datetime(2001,6,15,12),
            radiation_latitude=geo.lat,radiation_longitude=geo.lon)
        return state
    resident, tiled = build(reference_cfg), build(cfg)
    run = streaming.make_stepper(tiled,cfg,
        streaming.StreamingOptions(mode="on",tile_nx=24,tile_ny=20,nbuffers=2,store="host"),
        build=streaming.prepared_domain_builder(
            SimpleNamespace(state=tiled,cfg=SimpleNamespace(run=cfg),parent=None),
            check_geography=True))
    assert len(run.tiled_run.specs) == 4
    for _ in range(4):
        step(resident,reference_cfg)
        run(tiled,cfg)
        for key,value in carrier_manifest(resident).items():
            host=cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value)
            np.testing.assert_array_equal(run.store[key],host,err_msg=key)


@pytest.mark.parametrize("lw,sw", list(product((0,1,4,90),repeat=2)))
@pytest.mark.parametrize("variant", ["rte-rrtmgp","rrtmg_legacy"])
def test_scalar_radiation_options_validate_only_selected_engines(lw,sw,variant):
    cfg = replace(_cfg(lw,sw,variant),o3input=0,use_mp_re=0)
    if 4 in (lw,sw) and variant == "rte-rrtmgp":
        # One message per knob: each names the operation this arm
        # substitutes, so a caller who moved one switch is not told the
        # other is a problem too.
        with pytest.raises(ValueError,match=r"o3input=0 is not implemented"):
            validate_run_config(cfg)
        with pytest.raises(ValueError,match=r"use_mp_re=0 is not implemented"):
            validate_run_config(replace(_cfg(lw,sw,variant),use_mp_re=0))
    else:
        actual = validate_run_config(cfg)
        assert (actual.o3input,actual.use_mp_re,actual.ra_rrtmg_variant) == (0,0,variant)
    # Inactive values retain their type/range contract too.
    with pytest.raises(ValueError,match="o3input"):
        validate_run_config(replace(cfg,o3input=1))
    with pytest.raises(ValueError,match="use_mp_re"):
        validate_run_config(replace(cfg,use_mp_re=2))
