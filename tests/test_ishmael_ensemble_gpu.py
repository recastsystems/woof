"""All-member native columns must preserve each ordinary member's words."""
import numpy as np
import pytest
from types import SimpleNamespace
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def test_two_habit_members_share_terrain_and_preserve_native_rings():
    import cupy as cp
    from test_mp_spec_zone_ring import _square_moist_state
    from woof.core import microphysics
    from woof.core.physics_inventory import ring_guard_row
    from woof.ensemble.batch_physics import (ColumnBuffers, ColumnField,
        PackedThompsonState, prepare_registered_column_batch, registered_column_scratch_fields)
    ny, nx, nz, members = 8, 9, 12, 2
    states = []
    patch = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) * np.float32(0.5)
    for member in range(members):
        state, cfg = _square_moist_state(ny=ny, nx=nx, nz=nz, mp=55, dt=12, specified=True)
        state.thp += np.float32(member * 0.01)
        state.phb = cp.asarray(cp.asnumpy(state.phb)[:, None, None] + patch[None] * np.float32(9.81))
        state.thb = cp.asarray(cp.asnumpy(state.thb)[:, None, None] + patch[None] * np.float32(0.001))
        for habit, names in enumerate((("qi", "ni", "qvoli", "qaoli"),
                                      ("qi2", "ni2", "qvoli2", "qaoli2"),
                                      ("qi3", "ni3", "qvoli3", "qaoli3")), 1):
            for name, value in zip(names, (2.3e-6, 1e5, 9e-10, 3e-10), strict=True):
                getattr(state, name)[nz // 2:] = np.float32(value * habit * (member + 1))
        state.physics = SimpleNamespace(refl_10cm=None)
        states.append(state)
    names = tuple(sorted(set(ring_guard_row(55)["state_fields"]) | {"p", "php", "h_diabatic"}))
    bank = ColumnBuffers(tuple(ColumnField(n, getattr(states[0], n).shape) for n in names),
                         members=members, available_bytes=1 << 27)
    arrays = {n: bank.pack(n, cp.stack([getattr(s, n) for s in states])) for n in names}
    arrays.update(thb=states[0].thb, phb=states[0].phb)
    fields = registered_column_scratch_fields(cfg, nz=nz, ny=ny, nx=nx, reflectivity=True)
    scratch = ColumnBuffers(fields, members=members, available_bytes=1 << 27)
    packed = PackedThompsonState(arrays, {f.name: scratch.packed(f.name) for f in fields})
    packed.physics = SimpleNamespace(refl_10cm=None)
    bound = prepare_registered_column_batch(packed, cfg, members=members, ny=ny, nx=nx,
                                           dt=12, available_bytes=1 << 27)
    for step in range(2):
        result = bound(refl_10cm_due=True)
        for member, state in enumerate(states):
            ordinary = microphysics.apply(state, cfg, 12, refl_10cm_due=True)
            for name in names:
                assert arrays[name][:, member * ny:(member + 1) * ny].get().tobytes() == getattr(state, name).get().tobytes(), (step, member, name)
            for name in ("rainnc", "rainncv", "snownc", "snowncv", "sr"):
                assert getattr(result, name)[member * ny:(member + 1) * ny].get().tobytes() == getattr(ordinary, name).get().tobytes(), (step, member, name)
            assert packed.physics.refl_10cm[:, member * ny:(member + 1) * ny].get().tobytes() == state.physics.refl_10cm.get().tobytes()
            state.physics.refl_10cm = None
        packed.physics.refl_10cm = None


def test_analysis_radar_uses_native_psd_without_advancing_habit_moments():
    import cupy as cp
    from woof.core import ishmael
    from test_mp_spec_zone_ring import _square_moist_state
    from woof.physics_registry import consumer_row_for_selector
    state,cfg=_square_moist_state(ny=8,nx=9,nz=12,mp=55,dt=12,specified=False)
    row=consumer_row_for_selector("microphysics","native_column",55)
    before={n:getattr(state,n).get().tobytes() for n in row["transport"]}
    radar=ishmael.reflectivity(state)
    assert bool(cp.all(cp.isfinite(radar)))
    assert bool(cp.all(radar>=-35))
    assert {n:getattr(state,n).get().tobytes() for n in row["transport"]}==before
    for n in row["transport"]:
        getattr(state,n).fill(0)
    radar=ishmael.reflectivity(state)
    assert bool(cp.all(radar==-35))


def test_real_input_whole_member_physics_bootstrap(tmp_path):
    import os
    from pathlib import Path
    directory = os.environ.get("WOOF_TEST_ISHMAEL_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set WOOF_TEST_ISHMAEL_REAL_DIRECTORY to the real mp55/cu0 input")
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.wrfinput_forecast import prepare_wrf_run
    from test_ensemble_batch_physics_init_gpu import (
        test_real_input_shared_bootstrap_whole_physics_driver_matches_each_member,
    )
    inputs = prepare_wrf_run(resolve_wrfinput_run(Path(directory)), tmp_path / "input", run_seconds=120)
    test_real_input_shared_bootstrap_whole_physics_driver_matches_each_member(inputs, 2, "wrf_wrapper")


def test_real_input_complete_member_graph_and_rust_history(tmp_path):
    """Two ordinary references run serially, then one resident member batch.

    Keep hashes rather than three resident model workspaces on shared GPUs.
    Downloads and synchronization belong solely to this proof observer.
    """
    import gc
    import hashlib
    import os
    from dataclasses import replace
    from datetime import timedelta
    from pathlib import Path
    import cupy as cp
    directory = os.environ.get("WOOF_TEST_ISHMAEL_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set WOOF_TEST_ISHMAEL_REAL_DIRECTORY to the real mp55/cu0 input")
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.wrfinput_forecast import prepare_wrf_run
    from woof.prepared_domain_tree_forecast import _with_terrain_acoustics
    from woof.core import dycore
    from woof.core.clock import resolve_clock
    from woof.core.device_inventory import state_array_shapes
    from woof.core.state import refresh_model_time
    from woof.core.refl import consume_refl_10cm
    from woof.ingest.wrfinput import restore_domain_state
    from woof.ingest.lateral_bc import attach_lateral_boundaries, bind_lateral_boundary_clock
    from woof.ensemble.batch_forecast import PreparedRealEnsemble
    from woof.ensemble.batch_perturbation import initialize_member_winds
    from woof.ensemble.batch_dycore import member_domain_view
    from woof.ensemble.native_forecast import borrow_physics_scratch
    from woof.io.wrfout import WrfoutWriter, state_frame
    from test_ensemble_batch_real_forecast_gpu import (
        _driver_arrays, _member_array, _physics_readout, _scalar_physics,
    )
    inputs = _with_terrain_acoustics(prepare_wrf_run(
        resolve_wrfinput_run(Path(directory)), tmp_path / "input", run_seconds=120))
    cfg = inputs.experiment.root.run
    domain = replace(inputs.experiment.root, history_interval_s=cfg.dt)
    inputs = replace(inputs, experiment=replace(inputs.experiment, domains=(domain,)))
    assert cfg.mp_physics == 55 and cfg.specified
    names = state_array_shapes(cfg)
    seeds = (2026100200, 2026100201)
    digest = lambda value: hashlib.sha256(value.get().tobytes()).hexdigest()

    def history(state, refl, step, member):
        path = tmp_path / f"history-{step}-{member}.nc"
        frame = state_frame(state, include_diagnostic_pressure=True)
        frame["REFL_10CM"] = cp.asnumpy(refl)
        valid = inputs.experiment.start_time + timedelta(seconds=step * cfg.dt)
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                         global_attrs={"START_DATE": inputs.experiment.start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                                       "DT": np.float32(cfg.dt)}, field_schema=frame, engine="rust") as writer:
            writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), frame)
        result = hashlib.sha256(path.read_bytes()).hexdigest()
        path.unlink()
        return result

    def ordinary(member):
        state = restore_domain_state(inputs.domains[0].restored, cfg)
        clock = resolve_clock(inputs.experiment, lbc_interval_s=float(inputs.boundary_interval_seconds)).domain_clock(1)
        attach_lateral_boundaries(state, inputs.boundaries)
        bind_lateral_boundary_clock(state, clock)
        driver = _scalar_physics(state, inputs)
        initialize_member_winds(state=state, cfg=cfg, member_indices=(member,), seeds=(seeds[member],),
                                phase="after_physics_before_step")
        result = []
        for step in range(3):
            if step:
                if clock.lbc_reset_due():
                    clock.mark_force()
                clock.prepare_step()
                refresh_model_time(state, clock, kernel_launch=True)
                dycore.step(state, cfg, acoustic=True, refl_10cm_due=True)
                clock.advance()
                refresh_model_time(state, clock)
            cp.cuda.get_current_stream().synchronize()
            arrays = _driver_arrays(driver)
            result.append(({name: digest(getattr(state, name)) for name in names},
                           {name: (value.shape, value.ndim, digest(value)) for name, value in arrays.items()},
                           dict(driver.call_counts), driver.microphysics_updates,
                           history(state, consume_refl_10cm(state), step, member) if step else None))
        return result

    expected = []
    for member in range(2):
        expected.append(ordinary(member))
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    runtime = PreparedRealEnsemble(inputs, members=2, member_seeds=seeds,
                                   member_initializer=initialize_member_winds)
    for step in range(3):
        if step:
            runtime.step()
        cp.cuda.get_current_stream().synchronize()
        arrays = _driver_arrays(runtime.physics.driver)
        refl = runtime.consume_output_diagnostics() if step else None
        for member in range(2):
            state_words, physics_words, counts, updates, history_words = expected[member][step]
            for name in names:
                assert digest(runtime.batch.member_view(name, member)) == state_words[name], (step, member, name)
            assert arrays.keys() == physics_words.keys()
            for name, (shape, ndim, words) in physics_words.items():
                actual = _member_array(arrays[name], SimpleNamespace(shape=shape, ndim=ndim), member, 2)
                assert digest(actual) == words, (step, member, "physics", name)
            assert runtime.physics.driver.call_counts == counts
            assert runtime.physics.driver.microphysics_updates == updates
            if step:
                view = member_domain_view(runtime.batch, member)
                view.physics = _physics_readout(runtime.physics.driver, member=member, members=2, ny=cfg.ny, nx=cfg.nx)
                borrow_physics_scratch(view, runtime.physics.driver, member=member, members=2, ny=cfg.ny, nx=cfg.nx)
                assert history(view, refl[:, member*cfg.ny:(member+1)*cfg.ny], step, member) == history_words
