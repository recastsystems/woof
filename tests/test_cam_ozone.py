"""Root CAM ozone is a nested dependency, not a radiation preset."""
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.cam_ozone import CamOzoneState, cam_ozone_domain_ids
from woof.experiment import experiment_from_run_config


def _tree(root_pair=(0, 0), middle=False):
    run = RunConfig(nx=28, ny=28, nz=12, dx=3000., dy=3000., ztop=16000.,
                    dt=3., run_seconds=30., ra_lw_physics=root_pair[0],
                    ra_sw_physics=root_pair[1])
    exp = experiment_from_run_config(run, datetime(2000, 6, 1, 12))
    child = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3,
                    i_parent_start=8, j_parent_start=8,
                    run=replace(run, nx=16, ny=16, dx=1000., dy=1000., dt=1.,
                                ra_lw_physics=4, ra_sw_physics=4,
                                ra_rrtmg_variant="rrtmg_legacy", o3input=2))
    if middle:
        grandchild = replace(child, grid_id=3, parent_id=2)
        child = replace(child, run=replace(child.run, ra_rrtmg_variant="rte-rrtmgp"))
        return replace(exp, domains=(exp.root, child, grandchild))
    return replace(exp, domains=(exp.root, child))


def test_parent_only_host_ozone_force_uses_bounded_canonical_source(monkeypatch):
    import sys
    from woof.core import cam_ozone, nest_interp, streaming
    # Import what transfer_parent_ozone imports lazily BEFORE the stub goes in, as a full suite run has:
    # woof.core.state binds cupy at import, and a stub has no float32 (red whenever this file ran alone).
    import woof.core.nest  # noqa: F401
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        ndarray=np.ndarray, asarray=np.asarray, ascontiguousarray=np.ascontiguousarray))
    exp = _tree()
    canonical = np.full((12, 28, 28), 7., np.float32)
    parent = SimpleNamespace(physics=SimpleNamespace(
        o3rad=np.full_like(canonical, -99.), call_counts={'cam_ozone': 4}))
    owner = SimpleNamespace(store={cam_ozone.CARRIER_KEY: canonical}, _geography={},
        template_state=parent, decision=SimpleNamespace(tile_ny=7, tile_nx=8),
        scalars={'call_counts': {'cam_ozone': 4}})
    parent._streamed_domain = owner
    target = np.zeros((12, 16, 16), np.float32)
    driver = SimpleNamespace(o3rad=target, call_counts={'cam_ozone': 0},
                             cam_ozone=SimpleNamespace(mode=cam_ozone.ROUTING_PARENT_INTERPOLATED))
    child = SimpleNamespace(physics=driver)
    node = SimpleNamespace(state=child, cfg=exp.domains[1], parent=SimpleNamespace(state=parent))
    reg = nest_interp.register_nest(nri=3, nrj=3, i_parent_start=8, j_parent_start=8,
        child_nx=16, child_ny=16, parent_nx=28, parent_ny=28, wrapper='interp')
    seen = []
    def interpolate(source, cropped):
        assert np.all(source == 7.)
        assert source.shape[-2] < 28 and source.shape[-1] < 28
        assert cropped.nyc <= 7 and cropped.nxc <= 8
        seen.append((cropped.nyc, cropped.nxc))
        return np.full((12, cropped.nyc, cropped.nxc), 7., np.float32)
    monkeypatch.setattr(nest_interp, 'sint', interpolate)
    monkeypatch.setattr(streaming, 'refresh_from_store',
                        lambda *args, **kwargs: pytest.fail('host parent used resident refresh'))
    moved = cam_ozone.transfer_parent_ozone(node, reg)
    assert len(seen) == 6 and moved > 0
    assert np.all(target == 7.) and np.all(parent.physics.o3rad == -99.)
    assert driver.call_counts['cam_ozone'] == 1
    assert owner.scalars['call_counts']['cam_ozone'] == 4


@pytest.mark.parametrize("pair", [(0, 0), (1, 1), (4, 4), (90, 1)])
def test_only_ancestors_of_actual_nested_cam_consumer_are_carried(pair):
    exp = _tree(pair, middle=True)
    assert cam_ozone_domain_ids(exp) == frozenset({1, 2, 3})
    unrelated = replace(exp.domains[-1], grid_id=4, parent_id=1,
                        run=replace(exp.domains[-1].run, o3input=0))
    assert cam_ozone_domain_ids(replace(exp, domains=exp.domains+(unrelated,))) == {1, 2, 3}
    assert cam_ozone_domain_ids(replace(exp, domains=(exp.root,))) == set()
    assert cam_ozone_domain_ids(replace(exp, domains=(exp.root, unrelated))) == set()


def test_auxiliary_memory_is_derived_and_absent_without_nested_consumer():
    from woof.core.preflight import physics_array_shapes, estimate_domain
    exp = _tree()
    assert physics_array_shapes(exp.root.run) == {}
    shapes = physics_array_shapes(exp.root.run, cam_ozone=True)
    assert shapes["radiation/o33d_grid"] == (12, 28, 28)
    estimate = estimate_domain(exp.root, cam_ozone=True)
    items = {item.name: item for item in estimate.items}
    assert items["radiation/o33d_grid"].nbytes == 12 * 28 * 28 * 4
    assert "atmosphere/pressure" in items
    # An active ordinary driver gains exactly one 3-D device carrier.
    cfg = replace(exp.root.run, mp_physics=10, moist=True)
    original = physics_array_shapes(cfg)
    changed = physics_array_shapes(cfg, cam_ozone=True)
    assert {key: value for key, value in changed.items() if key not in original} == {
        "radiation/o33d_grid": (12, 28, 28)}
    assert all(changed[key] == value for key, value in original.items())


def test_shared_root_producer_matches_existing_cam_chain_and_rebinds_latitude():
    from woof.ingest.wrf_ozone import o33d_profile
    lat = np.array([[-50., -15.], [30., 68.]], np.float32)
    lon = np.full_like(lat, 12.)
    pressure = np.broadcast_to(np.geomspace(98000., 50., 12).astype(np.float32)
                               [:, None, None], (12, 2, 2)).copy()
    start = datetime(2000, 12, 30, 23, 59, 59)
    owner = CamOzoneState(start, lat, lon, "root-climatology")
    for elapsed in (0., 1., 86401.):
        now = start + timedelta(seconds=elapsed)
        hour = now.hour + now.minute/60. + now.second/3600.
        columns = pressure.transpose(1, 2, 0).reshape(-1, 12).copy()
        expected = o33d_profile(now.timetuple().tm_yday,
                               np.float32(now.timetuple().tm_yday-1 + hour/24.),
                               lat.reshape(-1), columns)
        actual = owner.evaluate(pressure, elapsed)
        np.testing.assert_array_equal(actual.transpose(1,2,0).reshape(-1,12), expected)
    old = owner.evaluate(pressure, 0.).copy()
    owner.latitude_deg[...] *= -1
    fresh = CamOzoneState(start, owner.latitude_deg, lon, "root-climatology")
    np.testing.assert_array_equal(owner.evaluate(pressure, 0.), fresh.evaluate(pressure, 0.))
    assert not np.array_equal(old, fresh.evaluate(pressure, 0.))


@pytest.mark.parametrize("mode", ["legacy-root", "parent-interpolated"])
def test_a_retained_consumer_cannot_fall_back_to_local_climatology(mode):
    owner = CamOzoneState(datetime(2000,1,1), np.zeros((2,2)), np.zeros((2,2)), mode)
    with pytest.raises(ValueError, match="only the root"):
        owner.evaluate(np.full((3,2,2), 50000., np.float32), 0.)


@pytest.mark.gpu
@pytest.mark.parametrize("legacy", [False, True])
def test_root_cam_uses_existing_cadence_and_retains_legacy_bytes(legacy):
    import cupy as cp
    from tilestream import harness
    from woof.core.cam_ozone import attach_cam_ozone
    from woof.core.physics import _prepare_atmosphere
    from woof.core.radiation_composition import legacy_radiation_adapter
    cfg = replace(_tree().root.run, nx=12, ny=12, moist=True, mp_physics=10,
                  ra_lw_physics=4 if legacy else 0,
                  ra_sw_physics=4 if legacy else 0,
                  ra_rrtmg_variant="rrtmg_legacy", radt=.2, dt=3.)
    state, driver = harness.make_physics_state(cfg, 104)
    reference, control = harness.make_physics_state(cfg, 104)
    geo = harness.make_geography(cfg)
    mode = "legacy-root" if legacy else "root-climatology"
    owner = CamOzoneState(datetime(2011,4,27,18), geo.lat, geo.lon, mode)
    attach_cam_ozone(state, cfg, owner)
    held = None
    for elapsed, expected_calls in ((0., 1), (3., 1), (6., 1), (9., 1), (12., 2)):
        state.elapsed_seconds = reference.elapsed_seconds = elapsed
        driver.compute(state, cfg)
        if legacy:
            control.compute(reference, cfg)
            expected = legacy_radiation_adapter(control.radiation_callable)._o33d_grid
            np.testing.assert_array_equal(cp.asnumpy(driver.o3rad), expected)
            np.testing.assert_array_equal(cp.asnumpy(driver.rthratenlw), cp.asnumpy(control.rthratenlw))
            np.testing.assert_array_equal(cp.asnumpy(driver.rthratensw), cp.asnumpy(control.rthratensw))
        assert driver.call_counts["cam_ozone"] == expected_calls
        if expected_calls == 1 and held is not None:
            np.testing.assert_array_equal(cp.asnumpy(driver.o3rad), held)
        held = cp.asnumpy(driver.o3rad)
    assert np.isfinite(held).all() and np.all(held > 0)
    assert driver.carriers.record("o3rad").last_update_model_time == 12.


@pytest.mark.gpu
def test_force_uses_store_ozone_and_missing_producer_is_observable():
    import cupy as cp
    from tilestream import harness
    from woof.core.cam_ozone import attach_cam_ozone, transfer_parent_ozone, CARRIER_KEY
    from woof.core.nest_interp import register_nest, sint
    from woof.core.rrtmg_legacy import ParentOzoneProvider
    from woof.core.streaming import publish_store
    exp = _tree()
    # Radiation-free nodes still own the auxiliary driver they need.
    parent, _ = harness.make_physics_state(exp.root.run, 104)
    child_cfg = replace(exp.domains[1].run, ra_lw_physics=0, ra_sw_physics=0)
    child, _ = harness.make_physics_state(child_cfg, 105)
    for state, cfg, mode in ((parent, exp.root.run, "root-climatology"),
                              (child, child_cfg, "parent-interpolated")):
        geo = harness.make_geography(cfg)
        attach_cam_ozone(state, cfg, CamOzoneState(exp.start_time, geo.lat, geo.lon, mode))
    node = SimpleNamespace(state=child, cfg=exp.domains[1], parent=SimpleNamespace(state=parent))
    reg = register_nest(nri=3, nrj=3, i_parent_start=8, j_parent_start=8,
                        child_nx=16, child_ny=16, parent_nx=28, parent_ny=28,
                        stagger="", wrapper="interp")
    with pytest.raises(RuntimeError, match="first ozone producer"):
        transfer_parent_ozone(node, reg)
    parent.physics.compute(parent, exp.root.run)
    canonical = cp.asnumpy(parent.physics.o3rad)
    old = ParentOzoneProvider(SimpleNamespace(_o33d_grid=canonical), reg)()
    transfer_parent_ozone(node, reg)
    np.testing.assert_array_equal(cp.asnumpy(child.physics.o3rad).transpose(1,2,0).reshape(-1,12), old)
    # A real published host-store shape, with stale resident values on BOTH ends.
    from woof.core import streaming
    setattr(parent, streaming._STORE_ATTR, {CARRIER_KEY: canonical.copy()})
    setattr(child, streaming._STORE_ATTR, {CARRIER_KEY: np.full(child.p.shape, -99., np.float32)})
    parent.physics.o3rad.fill(-7.)
    child.physics.o3rad.fill(-8.)
    assert transfer_parent_ozone(node, reg) > 0
    np.testing.assert_array_equal(getattr(child, streaming._STORE_ATTR)[CARRIER_KEY]
                                 .transpose(1,2,0).reshape(-1,12), old)
    wrong = sint(cp.full_like(parent.physics.o3rad, -7.), reg)
    assert not np.array_equal(cp.asnumpy(wrong), cp.asnumpy(child.physics.o3rad))


@pytest.mark.gpu
def test_auxiliary_ozone_is_one_checkpointed_stream_carrier(tmp_path):
    import cupy as cp
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest
    from woof.core.cam_ozone import attach_cam_ozone, CARRIER_KEY
    from woof.io import restart
    cfg = replace(_tree().root.run, nx=12, ny=12)
    def build():
        state, _ = harness.make_physics_state(cfg, 112)
        geo = harness.make_geography(cfg)
        attach_cam_ozone(state, cfg, CamOzoneState(datetime(2000,6,1,12),
                                                 geo.lat, geo.lon, "root-climatology"))
        return state
    state = build()
    state.physics.compute(state, cfg)
    inventory = carrier_manifest(state)
    assert inventory[CARRIER_KEY] is state.physics.o3rad
    assert [key for key in inventory if "o3rad" in key or "o33d" in key] == [CARRIER_KEY]
    path = tmp_path / "cam.npz"
    restart.write_restart(path, state, cfg)
    restored = build()
    restart.restore_restart(path, restored, cfg)
    np.testing.assert_array_equal(cp.asnumpy(restored.physics.o3rad), cp.asnumpy(state.physics.o3rad))
    assert restored.physics.call_counts == state.physics.call_counts
    assert restored.physics.carriers.state() == state.physics.carriers.state()
    from test_restart import _rewrite_restart_archive
    missing = _rewrite_restart_archive(path, tmp_path / "missing-held.npz",
        lambda payload, header: payload.pop(CARRIER_KEY))
    before = cp.asnumpy(restored.physics.o3rad)
    with pytest.raises(restart.RestartMismatchError, match="retained CAM ozone"):
        restart.restore_restart(missing, restored, cfg)
    np.testing.assert_array_equal(cp.asnumpy(restored.physics.o3rad), before)
    # Resume between cadence ticks: pressure changes cannot overwrite held ozone.
    restored.elapsed_seconds = state.elapsed_seconds = 3.
    restored.physics.compute(restored, cfg)
    state.physics.compute(state, cfg)
    np.testing.assert_array_equal(cp.asnumpy(restored.physics.o3rad), cp.asnumpy(state.physics.o3rad))


@pytest.mark.gpu
def test_root_cam_reused_tiles_match_resident_and_keep_producer_clock():
    import cupy as cp
    from woof.core import streaming
    from woof.core.dycore import step
    from woof.core.cam_ozone import attach_cam_ozone, CARRIER_KEY
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest
    cfg = replace(_tree().root.run, nx=48, ny=40, dt=1., radt=.05, terrain_opt=1)
    def build():
        geo = harness.make_geography(cfg)
        state, _ = harness.make_physics_state(cfg, 117, geography=geo)
        attach_cam_ozone(state, cfg, CamOzoneState(datetime(2000,6,1,12),
                                                 geo.lat, geo.lon, "root-climatology"))
        return state
    reference, tiled = build(), build()
    node = SimpleNamespace(state=tiled, cfg=SimpleNamespace(run=cfg), parent=None)
    options = streaming.StreamingOptions(mode="on", tile_nx=24, tile_ny=20,
                                         nbuffers=2, store="host")
    run = streaming.make_stepper(tiled, cfg, options,
                build=streaming.prepared_domain_builder(node, check_geography=True))
    assert len(run.tiled_run.specs) == 4
    for _ in range(4):
        step(reference, cfg)
        run(tiled, cfg)
    store = run.store
    actual = carrier_manifest(reference)
    for key in actual:
        expected = actual[key]
        host = cp.asnumpy(expected) if isinstance(expected, cp.ndarray) else np.asarray(expected)
        np.testing.assert_array_equal(store[key], host, err_msg=key)
    assert CARRIER_KEY in store
    assert run.scalars["call_counts"]["cam_ozone"] == 2
    assert run.scalars["carriers"]["o3rad"]["last_update_model_time"] == 3.


def test_auxiliary_cam_drives_the_existing_adaptive_observer():
    from woof.core.adaptive_clock import AdaptiveClockDriver
    from woof.core.radiation_carriers import CarrierContract
    controller = object.__new__(AdaptiveClockDriver)
    controller._radiation_actual = {}
    controller._radiation_seen = {}
    carriers = CarrierContract()
    physics = SimpleNamespace(carriers=carriers, stepra=4, radt_seconds=12.)
    # A clock with no resolved radiation calendar (radt_ticks None), the
    # case the time-based policy observed here serves.  Every DomainClock
    # carries a spec, and development branch (e292dbcf5) lets a
    # resolved calendar's exact tick deadline decide before that policy.
    node = SimpleNamespace(state=SimpleNamespace(physics=physics),
                           clock=SimpleNamespace(
                               elapsed_seconds=10.,
                               spec=SimpleNamespace(radt_ticks=None)),
                           cfg=SimpleNamespace(run=SimpleNamespace(dt=2.)))
    carriers.declare("o3rad", source="cam_ozone", model_time=0.)
    assert controller._observe_radiation(1, node) is None
    controller._drive_radiation_on_time(1, node)
    assert physics.radiation_due_override is True
    carriers.declare("o3rad", source="cam_ozone", model_time=10.)
    assert controller._observe_radiation(1, node) == 10.
    node.clock.elapsed_seconds = 11.
    node.cfg.run.dt = 1.
    controller._drive_radiation_on_time(1, node)
    assert physics.radiation_due_override is False


@pytest.mark.parametrize("cap", [1, 3, 4096])
def test_cam_column_cap_changes_workspace_without_changing_values(cap):
    latitude = np.arange(35, 47, dtype=np.float32).reshape(3,4)
    pressure = np.broadcast_to(np.geomspace(98000., 50., 12).astype(np.float32)
                                [:,None,None], (12,3,4)).copy()
    args = (datetime(2000,5,1,12), latitude, np.zeros_like(latitude), "root-climatology")
    expected = CamOzoneState(*args, column_chunk=12).evaluate(pressure, 3600.)
    actual = CamOzoneState(*args, column_chunk=cap).evaluate(pressure, 3600.)
    np.testing.assert_array_equal(actual, expected)


def test_shared_stream_context_prices_only_required_cam_domains():
    from woof.core import streaming
    from woof.core.cam_ozone import memory_increment_per_cell
    from tilestream import autoplan
    exp = _tree()
    unrelated = replace(exp.domains[1], grid_id=3,
                        run=replace(exp.domains[1].run, o3input=0))
    exp = replace(exp, domains=exp.domains+(unrelated,),
                  tiles=streaming.StreamingOptions(mode="on",tile_nx=8,tile_ny=8))
    for dc in exp.domains:
        options = streaming.options_for_domain(dc, exp.tiles)
        assert options.radiation_context.cam_ozone is (dc.grid_id in (1,2))
        assert "radiation_context" not in options.to_mapping()
        base = autoplan.footprint_for(dc.run)
        priced = streaming.radiation_footprint(dc.run, options)
        delta = memory_increment_per_cell(dc.run) if dc.grid_id in (1,2) else (0.,0.)
        assert priced.bytes_per_cell == base.bytes_per_cell + delta[0]
        assert priced.store_bytes_per_cell == base.store_bytes_per_cell + delta[1]
    assert memory_increment_per_cell(exp.domains[1].run) == (4.,4.)


@pytest.mark.parametrize("nx,ny", [(1,1),(48,40),(100,3)])
def test_auxiliary_driver_cell_bound_covers_real_allocation_shapes(nx, ny):
    from math import prod
    from woof.core.cam_ozone import memory_increment_per_cell
    from woof.core.preflight import physics_array_shapes, atmosphere_transient_shapes
    cfg = replace(_tree().root.run, nx=nx, ny=ny)
    device, carried = memory_increment_per_cell(cfg)
    persistent = sum(prod(shape)*4 for shape in physics_array_shapes(cfg,cam_ozone=True).values())
    transient = sum(prod(shape)*4 for shape in atmosphere_transient_shapes(cfg,cam_ozone=True).values())
    assert carried * nx*ny*cfg.nz >= persistent
    assert device * nx*ny*cfg.nz >= persistent+transient


@pytest.mark.parametrize("cap", [True, 0, -1, 2.5, float("nan")])
def test_cam_chunk_rejects_nonpositive_or_noninteger_count(cap):
    with pytest.raises(ValueError, match="positive integer"):
        CamOzoneState(datetime(2000,1,1), np.zeros((2,2)),
                      np.zeros((2,2)), "root-climatology", column_chunk=cap)


def test_monthly_workspace_respects_declared_column_cap(monkeypatch):
    from woof.ingest import wrf_ozone
    actual = wrf_ozone.interp_ozone_to_latitudes
    chunks = []
    def measured(latitude, *args, **kwargs):
        chunks.append(len(latitude))
        return actual(latitude, *args, **kwargs)
    monkeypatch.setattr(wrf_ozone, "interp_ozone_to_latitudes", measured)
    owner = CamOzoneState(datetime(2000,1,1), np.zeros((3,4)),
                          np.zeros((3,4)), "root-climatology", column_chunk=5)
    result = owner.evaluate(np.full((12,3,4), 50000., np.float32), 0.)
    assert result.shape == (12,3,4) and chunks == [5,5,2]


@pytest.mark.gpu
@pytest.mark.parametrize("root_pair", [(0,0),(4,4)])
def test_nested_legacy_consumer_matches_original_parent_provider_on_reused_tiles(root_pair):
    import cupy as cp
    from tilestream import harness
    from tilestream.physics_inventory import carrier_manifest
    from woof.core import streaming
    from woof.core.physics import initialize_physics
    from woof.core.cam_ozone import cam_ozone_setup, transfer_parent_ozone, ozone_parent_for
    from woof.core.radiation_composition import make_radiation
    from woof.core.nest_interp import register_nest
    from woof.core.rrtmg_legacy import ParentOzoneProvider
    cfg = replace(_tree(root_pair).root.run, nx=48, ny=40, nz=12,
                  terrain_opt=1, moist=True, mp_physics=8, radt=.2)
    exp = experiment_from_run_config(cfg, datetime(2001,6,15,12))
    child_cfg = replace(cfg, nx=48, ny=40, dx=1000., dy=1000., dt=1., radt=.1,
                        ra_lw_physics=4, ra_sw_physics=1, ra_rrtmg_variant="rrtmg_legacy")
    child_dc = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3,
                        i_parent_start=8, j_parent_start=8, run=child_cfg)
    exp = replace(exp, domains=(exp.root,child_dc))
    def build(dc, owned):
        geo = harness.make_geography(dc.run)
        state, _ = harness.make_physics_state(dc.run, 123, geography=geo)
        grid = SimpleNamespace(latlon_mass=lambda: (geo.lat,geo.lon))
        cam = cam_ozone_setup(exp=exp,dc=dc,grid=grid) if owned else None
        radiation = make_radiation(dc.run,exp.start_time,geo.lat,geo.lon,
                                   column_chunk=31,ozone_parent=ozone_parent_for(cam))
        initialize_physics(state,dc.run,radiation=radiation,
            radiation_start_time=exp.start_time,radiation_latitude=geo.lat,
            radiation_longitude=geo.lon,cam_ozone=cam)
        return state
    parent = build(exp.root,True)
    resident, tiled, reference = (build(child_dc,owned) for owned in (True,True,False))
    registration = register_nest(nri=3,nrj=3,i_parent_start=8,j_parent_start=8,
        child_nx=48,child_ny=40,parent_nx=48,parent_ny=40,stagger="",wrapper="interp")
    parent_node = SimpleNamespace(state=parent)
    nodes = [SimpleNamespace(state=state,cfg=child_dc,parent=parent_node)
             for state in (resident,tiled)]
    options = streaming.StreamingOptions(mode="on",tile_nx=24,tile_ny=20,
                                         nbuffers=2,store="host")
    run = streaming.make_stepper(tiled,child_cfg,options,
        # This differential isolates atmospheric/radiation transport; the
        # public tree witness exercises the rolling-boundary clock itself.
        build=streaming.prepared_domain_builder(
            SimpleNamespace(state=tiled,cfg=child_dc,parent=None),
            check_geography=True))
    assert len(run.tiled_run.specs) == 4
    from woof.core.dycore import step
    for parent_time in (0.,3.,6.,9.,12.):
        parent.elapsed_seconds = parent_time
        parent.physics.compute(parent,cfg)
        for node in nodes:
            transfer_parent_ozone(node,registration)
        reference.physics.radiation_callable.longwave_adapter._ozone_provider = ParentOzoneProvider(
            SimpleNamespace(_o33d_grid=cp.asnumpy(parent.physics.o3rad)),registration)
        for _ in range(3):
            step(reference,child_cfg)
            step(resident,child_cfg)
            run(tiled,child_cfg)
        for name in ("rthratenlw","rthratensw"):
            np.testing.assert_array_equal(cp.asnumpy(getattr(resident.physics,name)),
                                          cp.asnumpy(getattr(reference.physics,name)))
        for key,value in carrier_manifest(resident).items():
            host = cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value)
            np.testing.assert_array_equal(run.store[key],host,err_msg=key)
    assert parent.physics.call_counts["cam_ozone"] == 2
    assert resident.physics.call_counts["radiation"] == 3


@pytest.mark.parametrize("root_pair", [(0,0),(4,4)])
def test_public_document_resolves_and_roundtrips_mixed_domain_radiation(tmp_path, root_pair):
    import tomllib
    from test_experiment import _write
    from woof.experiment import load_experiment, build_experiment
    from woof.experiment_document import render_experiment_document
    from woof.config import radiation_scheme_ids
    path = _write(tmp_path, shared='moist = true\nmp_physics = 6',
        d01=f'ra_lw_physics = {root_pair[0]}\nra_sw_physics = {root_pair[1]}\nra_rrtmg_variant = "rte-rrtmgp"',
        d02='ra_lw_physics = 4\nra_sw_physics = 1\nra_rrtmg_variant = "rrtmg_legacy"\no3input = 2\nuse_mp_re = 1\nswrad_scat = 1.0')
    exp = load_experiment(path)
    assert radiation_scheme_ids(exp.root.run) == root_pair
    assert radiation_scheme_ids(exp.domains[1].run) == (4,1)
    assert exp.root.run.ra_rrtmg_variant == "rte-rrtmgp"
    assert exp.domains[1].run.ra_rrtmg_variant == "rrtmg_legacy"
    assert cam_ozone_domain_ids(exp) == {1,2}
    raw = tomllib.loads(path.read_text())
    restored = build_experiment(tomllib.loads(render_experiment_document(raw)), source="roundtrip")
    assert [domain.run for domain in restored.domains] == [domain.run for domain in exp.domains]
    # Omitting domain overrides continues to inherit the one shared choice.
    raw['shared'].update(ra_lw_physics=4,ra_sw_physics=4,ra_rrtmg_variant="rrtmg_legacy")
    for row in raw['domain']:
        for name in ('ra_lw_physics','ra_sw_physics','ra_rrtmg_variant'):
            row.pop(name)
    ordinary = build_experiment(raw, source="shared-default")
    assert all(radiation_scheme_ids(d.run)==(4,4) and d.run.ra_rrtmg_variant=="rrtmg_legacy"
               for d in ordinary.domains)


def test_derived_cam_context_never_enters_public_config_or_cache_identity():
    import json
    from woof.core import streaming
    from woof.experiment import domain_config_document, experiment_config_document
    from woof.ingest.prepared_cache import prepared_domain_config_identity
    from woof.prepared_domain_tree_forecast import _without_forecast_stop
    from woof.runplan import _config_snapshot
    from woof.core.model import restart_identity_payload
    options = streaming.StreamingOptions(mode="on",tile_nx=8,tile_ny=8,
        vram_budget_bytes=3*2**30,host_budget_bytes=4*2**30)
    exp = _tree()
    exp = replace(exp,tiles=options,domains=(exp.root,replace(exp.domains[1],tiles=options)))
    assert exp.domains[1].tiles.radiation_context.cam_ozone_domains == frozenset({1,2})
    documents = [domain_config_document(exp.domains[1]), experiment_config_document(exp),
        prepared_domain_config_identity(exp.domains[1]), _without_forecast_stop(exp),
        _config_snapshot(exp,None), restart_identity_payload(exp)]
    for document in documents:
        assert "radiation_context" not in json.dumps(document,default=str)
    wire = prepared_domain_config_identity(exp.domains[1])["tiles"]
    assert wire == options.to_mapping()
    assert wire["vram_budget_bytes"] == 3*2**30 and wire["host_budget_bytes"] == 4*2**30
    # Altering a derived estimate alone cannot alter any of these documents.
    changed = replace(exp.domains[1], tiles=replace(exp.domains[1].tiles,
        radiation_context=streaming.RadiationMemoryContext(1,1.,True,frozenset({99}))))
    assert prepared_domain_config_identity(changed) == prepared_domain_config_identity(exp.domains[1])
    public_change = replace(exp.domains[1], tiles=replace(exp.domains[1].tiles,vram_budget_bytes=7*2**29))
    assert prepared_domain_config_identity(public_change) != prepared_domain_config_identity(exp.domains[1])
    from woof.ingest.prepared_cache import _json_copy
    with pytest.raises(TypeError, match="cannot contain a set"):
        _json_copy({"unclaimed_field":frozenset({1,2})})


def test_prepared_meteorology_reuses_across_public_tile_roads_only():
    from woof.core.streaming import StreamingOptions
    from woof.ingest.prepared_cache import (prepared_domain_config_identity,
        effective_prepared_domain_config, compare_prepared_domain_config)
    exp = _tree()
    resident = exp.domains[1]
    streamed = replace(resident, tiles=StreamingOptions(mode="on",tile_nx=8,tile_ny=8,
                                                       vram_budget_bytes=3*2**30))
    a, b = (prepared_domain_config_identity(dc) for dc in (resident,streamed))
    assert a != b  # the human-readable resolved document still records the choice
    assert compare_prepared_domain_config(effective_prepared_domain_config(a),
        effective_prepared_domain_config(b))[1] == []
    changed = prepared_domain_config_identity(replace(streamed,run=replace(streamed.run,dx=4000.)))
    assert compare_prepared_domain_config(effective_prepared_domain_config(a),
        effective_prepared_domain_config(changed))[1] == ['run.dx']
