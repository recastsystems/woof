"""Legacy all-member spectra and initialized real-source physics word gates."""

from dataclasses import replace
from fractions import Fraction
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.fixture(autouse=True)
def card_free_of_earlier_tests():
    """Give each real-input test the card a fresh process would see.

    Admission prices the card by its free bytes. A battery leg runs every
    file in one process, and blocks an earlier test released stay in CuPy's
    pool, where the driver still counts them as used. Measured on the 16 GB
    release card with the ensemble device files in one process: the
    four-member real forecast found 1.44 GB free against the 1.84 GB it
    needs and was refused, and the production door's memory gate refused a
    3.03 GiB tree at one and at four members. Each passes in a process of
    its own. Unreferenced blocks go back to the driver before and after
    each test; nothing still referenced is touched.
    """
    import gc
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    yield
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.parametrize("members", [1, 4, 10])
@pytest.mark.parametrize("ozone", [0, 2])
def test_legacy_native_spectra_share_geometry_and_match_independent_columns(members, ozone):
    import cupy as cp
    import test_rrtmg_legacy_wiring as native
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    from woof.ensemble.batch_physics_init import prepare_legacy_member_radiation
    ny, nx = 3, 5
    cells = ny * nx
    bundle = native._bundle(native.profile.__wrapped__(), members * cells, seed=20261022)
    latitude = np.linspace(20, 45, cells).astype(np.float32)
    longitude = np.linspace(-170, 170, cells).astype(np.float32)
    env = native._env_from(bundle, np.arange(members * cells), members * ny, nx,
                           np.tile(latitude, members), np.tile(longitude, members))
    env.cfg.o3input = ozone
    shared_land = env.fields["xland"][:ny].copy()
    # Static fields derive from the same geography/source in each member.
    bundle["xland"] = np.tile(shared_land.get().reshape(-1), members)
    env.fields["xland"] = shared_land if members > 1 else shared_land
    original = RRTMGLegacyRadiation(native.START, latitude.reshape(ny, nx), longitude.reshape(ny, nx),
                                  p_top=env.p_top, o3input=ozone, column_chunk=members * 5)
    bound = prepare_legacy_member_radiation(original, members=members, available_bytes=1 << 26,
                                           shared_surface_fields=("xland",) if members > 1 else ())
    inputs_before = {name: value.get().tobytes() for name, value in env.atmosphere.items()}
    outputs = native._call(bound, env)
    assert {name: value.get().tobytes() for name, value in env.atmosphere.items()} == inputs_before
    for member in range(members):
        scalar = native._env_from(bundle, np.arange(member * cells, (member + 1) * cells), ny, nx,
                                  latitude, longitude)
        scalar.cfg.o3input = ozone
        reference = native._call(RRTMGLegacyRadiation(native.START, latitude.reshape(ny, nx),
                                      longitude.reshape(ny, nx), p_top=env.p_top, o3input=ozone,
                                      column_chunk=5), scalar)
        for name in ("rthratenlw", "rthratensw", "swdown", "glw", "gsw", "coszen", "olr"):
            value = getattr(outputs, name)
            actual = value[:, member * ny:(member + 1) * ny] if value.ndim == 3 else value[member * ny:(member + 1) * ny]
            assert actual.get().tobytes() == getattr(reference, name).get().tobytes(), (member, name, ozone)
    if members > 1:
        assert bound.latitude_deg.shape == (ny, nx)
        assert bound.latitude_deg is original.latitude_deg
        assert bound._C is original._C
        assert bound._sw_tables is original._sw_tables
        assert bound._ensemble_radiation_receipt["geometry_shared_once"]


@pytest.fixture(scope="module")
def prepared_real(tmp_path_factory):
    directory = os.environ.get("WOOF_TEST_WRF_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set WOOF_TEST_WRF_REAL_DIRECTORY to the retained real-input directory")
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.wrfinput_forecast import prepare_wrf_run
    run = resolve_wrfinput_run(Path(directory))
    return prepare_wrf_run(run, tmp_path_factory.mktemp("member-physics-source"), run_seconds=24)


@pytest.mark.parametrize("members", [4, 10])
@pytest.mark.parametrize("bootstrap_route", ["wrf_wrapper", "ordinary_bootstrap"])
def test_real_input_shared_bootstrap_whole_physics_driver_matches_each_member(prepared_real, members, bootstrap_route):
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    from woof.core.cam_ozone import cam_ozone_setup
    from woof.core.radiation_composition import make_radiation
    from woof.ingest.wrfinput import restore_domain_state, initialize_wrfinput_physics
    from woof.ensemble.batch_state import BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES
    from woof.ensemble.batch_physics_init import (
        initialize_wrfinput_member_physics, initialize_member_physics_from_bootstrap,
    )
    from woof.core.microphysics import apply
    inputs = prepared_real
    domain, bundle, grid = inputs.experiment.root, inputs.domains[0], inputs.grids[0]
    cfg = domain.run
    assert (cfg.mp_physics, cfg.sf_sfclay_physics, cfg.sf_surface_physics, cfg.bl_pbl_physics) in (
        (8, 1, 2, 1), (55, 91, 2, 1))
    scalar_states, scalar_drivers = [], []
    for unused in range(members):
        state = restore_domain_state(bundle.restored, cfg)
        cam = cam_ozone_setup(exp=inputs.experiment, dc=domain, grid=grid)
        radiation = make_radiation(cfg, inputs.experiment.start_time,
                                   bundle.restored.raw["XLAT"], bundle.restored.raw["XLONG"],
                                   p_top=state.p_top)
        driver = initialize_wrfinput_physics(state, bundle.restored, cfg, radiation=radiation,
                    radiation_start_time=inputs.experiment.start_time,
                    radiation_latitude=bundle.restored.raw["XLAT"], radiation_longitude=bundle.restored.raw["XLONG"],
                    landuse=bundle.landuse, cam_ozone=cam)
        scalar_states.append(state)
        scalar_drivers.append(driver)
    reference_state = scalar_states[0]
    names = state_array_shapes(cfg)
    arrays = {name: getattr(reference_state, name).get() for name in names}
    controls = {"physics", "lateral_boundaries", "_scratch", "_scratch_arena", "_host_setup_state", "_phb_host"}
    scalars = {name: value for name, value in vars(reference_state).items() if name not in names and name not in controls}
    rational = Fraction(str(cfg.dt))
    clock = dict(ticks=0, step_ticks=rational.numerator, tick_den=rational.denominator,
                 run_ticks=round(cfg.run_seconds * rational.denominator), step_count=0,
                 dt_fp32=np.float32(cfg.dt), dtbc_fp32=np.float32(0))
    prepared = PreparedHostMember(cfg, arrays, scalars, clock, phb_host=reference_state._phb_host)
    batch = BatchedDomainState.from_prepared((prepared,) * members, array_module=cp,
                 available_bytes=25 << 30, shared_fields=tuple(SHARED_STATE_CANDIDATES & names.keys()))
    cam = cam_ozone_setup(exp=inputs.experiment, dc=domain, grid=grid)
    if bootstrap_route == "wrf_wrapper":
        adapter = initialize_wrfinput_member_physics(batch, bundle.restored,
                     start_time=inputs.experiment.start_time, landuse=bundle.landuse,
                     cam_ozone=cam, available_bytes=20 << 30)
    else:
        before = cp.get_default_memory_pool().used_bytes()
        bootstrap_state = restore_domain_state(bundle.restored, cfg)
        radiation = make_radiation(cfg, inputs.experiment.start_time,
            bundle.restored.raw["XLAT"], bundle.restored.raw["XLONG"], p_top=bootstrap_state.p_top)
        bootstrap_driver = initialize_wrfinput_physics(bootstrap_state, bundle.restored, cfg,
            radiation=radiation, radiation_start_time=inputs.experiment.start_time,
            radiation_latitude=bundle.restored.raw["XLAT"], radiation_longitude=bundle.restored.raw["XLONG"],
            landuse=bundle.landuse, cam_ozone=cam)
        increment = cp.get_default_memory_pool().used_bytes() - before
        adapter = initialize_member_physics_from_bootstrap(batch, bootstrap_state, bootstrap_driver,
            available_bytes=20 << 30, bootstrap_pool_live_increment_bytes=increment)
        assert bootstrap_state.physics is None and bootstrap_driver.state is None
        assert adapter.receipt["bootstrap_pool_live_increment_bytes"] == increment
        del bootstrap_driver, bootstrap_state
    assert adapter.receipt["scalar_member_drivers_retained"] == 0
    for name in adapter.receipt["shared_surface_fields"]:
        assert adapter.driver.fields[name].shape == (cfg.ny, cfg.nx)
    assert all(row["ownership"] == "shared" and row["shape"] == (cfg.ny, cfg.nx)
               for row in adapter.receipt["shared_surface_allocations"])
    for member, state in enumerate(scalar_states):
        increment = np.float32(member) * np.float32(0.03125)
        state.thp += increment
        batch.member_view("thp", member)[:] += increment
    for step in range(2):
        batch.elapsed_seconds = float(step * cfg.dt)
        held = adapter.compute()
        for member, (state, driver) in enumerate(zip(scalar_states, scalar_drivers)):
            state.elapsed_seconds = float(step * cfg.dt)
            reference = driver.compute(state, cfg)
            for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi", "rqs"):
                expected = getattr(reference, name)
                actual = getattr(held, name)
                if expected is None:
                    assert actual is None
                    continue
                height = cfg.ny + (name == "rv")
                assert actual[:, member * height:(member + 1) * height].get().tobytes() == expected.get().tobytes(), (step, member, name)
            for name, expected in driver.fields.items():
                actual = adapter.driver.fields[name]
                if name in adapter.receipt["shared_surface_fields"]:
                    view = actual
                else:
                    height = expected.shape[-2]
                    view = actual[:, member * height:(member + 1) * height] if actual.ndim == 3 else actual[member * height:(member + 1) * height]
                assert view.get().tobytes() == expected.get().tobytes(), (step, member, name)
        result = adapter.apply_microphysics()
        for member, (state, driver) in enumerate(zip(scalar_states, scalar_drivers)):
            expected = apply(state, cfg, cfg.dt)
            driver.accept_microphysics(expected, dt=cfg.dt)
            for name in adapter.receipt["model_member_fields"]:
                assert batch.member_view(name, member).get().tobytes() == getattr(state, name).get().tobytes(), (step, member, name)
            assert result.rainnc[member * cfg.ny:(member + 1) * cfg.ny].get().tobytes() == expected.rainnc.get().tobytes()
    assert adapter.driver.call_counts == scalar_drivers[0].call_counts
