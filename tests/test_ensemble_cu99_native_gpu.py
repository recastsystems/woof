"""CU99 private carriers through the production packed default-suite seam."""
from fractions import Fraction
from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _driver(edrates, member, *, full_base=False, specified=False):
    import cupy as cp
    from test_physics_driver import _full_state
    state, cfg, driver = _full_state(nx=8, ny=6, nz=16, dt=12., mp_physics=8,
        cu_physics=99, kf_edrates=edrates, cudt_minutes=.4, bldt=0., radt=0.,
        ra_physics=4, ra_lw_physics=4, ra_sw_physics=4,
        ra_rrtmg_variant="rrtmg_legacy", hailcast_opt=0,
        radiation_start_time=datetime(2026, 10, 16, 9),
        radiation_latitude=np.full((6, 8), 30., np.float32),
        radiation_longitude=np.full((6, 8), -98., np.float32),
        specified=specified, spec_zone=1)
    state.physics = driver
    if full_base:
        from dataclasses import replace
        cfg = replace(cfg, terrain_opt=1)
        for name in ("thb", "pb", "alb", "phb", "dphb_resid"):
            value = getattr(state, name)
            setattr(state, name, cp.broadcast_to(value[:, None, None],
                (value.shape[0], cfg.ny, cfg.nx)).copy())
        state._phb_host = state.phb.get()
    if state.mub2d is None:
        state.mub2d = cp.full((cfg.ny, cfg.nx), state.mub, cp.float32)
    state.thp += cp.float32(member * .03125)
    driver.cu_nca[:, ::2] = cp.float32(36. + member * 12.)
    driver.cu_pratec[:, ::2] = cp.float32(.001 * (member + 1))
    driver.cu_raincv[:, ::2] = cp.float32(.012 * (member + 1))
    driver.cumulus_callable.w0avg.fill(cp.float32(.01 * (member + 1)))
    for number, array in enumerate(driver.cu_rates.values(), 1):
        array[:, :, ::2] = cp.float32(number * (member + 1) * 1.e-8)
    if edrates:
        for number, name in enumerate(("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"), 1):
            getattr(driver.cumulus_callable, name).fill(cp.float32(number + member))
    driver.recouple_cumulus_tendencies(state, cfg)
    return state, cfg, driver


def _prepared(state, cfg, *, extras=()):
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_state import PreparedHostMember
    names = state_array_shapes(cfg)
    arrays = {name: getattr(state, name).get() for name in names}
    arrays.update({spec.name: np.zeros(spec.shape, spec.dtype) for spec in extras})
    controls = {"physics", "lateral_boundaries", "_scratch", "_scratch_arena", "_host_setup_state", "_phb_host"}
    scalars = {name: value for name, value in vars(state).items()
               if name not in names and name not in controls}
    dt = Fraction(str(cfg.dt))
    clock = dict(ticks=0, step_ticks=dt.numerator, tick_den=dt.denominator,
        run_ticks=120 * dt.denominator, step_count=0,
        dt_fp32=np.float32(cfg.dt), dtbc_fp32=np.float32(0))
    return PreparedHostMember(cfg, arrays, scalars, clock, phb_host=state._phb_host,
        scratch={name: value.get() for name, value in state._scratch.items()})


@pytest.mark.parametrize("edrates", [0, 1])
@pytest.mark.parametrize("specified", [False, True])
def test_packed_default_suite_cu99_matches_private_member_words(edrates, specified):
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    from woof.ensemble.batch_physics_init import (bootstrap_physics_snapshot,
        initialize_member_physics_from_bootstrap)
    from woof.ensemble.batch_physics_init import _bootstrap_array_inventory
    from woof.ensemble.native_forecast import member_output_view
    from types import SimpleNamespace
    pairs = [_driver(edrates, member, specified=specified) for member in range(2)]
    source, cfg, bootstrap = _driver(edrates, 0, specified=specified)
    snapshots = tuple(bootstrap_physics_snapshot(c, d) for s, c, d in pairs)
    assert all("cumulus_callable" not in item.owner_digests for item in snapshots)
    batch = BatchedDomainState.from_prepared(tuple(_prepared(s, c) for s, c, d in pairs),
        available_bytes=2 << 30, array_module=cp,
        shared_fields=tuple(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    adapter = initialize_member_physics_from_bootstrap(batch, source, bootstrap,
        available_bytes=2 << 30, member_bootstraps=snapshots)
    assert adapter.driver.cumulus_callable is not bootstrap.cumulus_callable
    assert adapter.driver.cumulus_callable._driver_ref() is adapter.driver
    assert adapter.driver.cumulus_callable._history_state is adapter.state
    assert len(adapter.driver._ensemble_persistent_couplers) == 3
    for number in range(6):
        batch.elapsed_seconds = float(number * cfg.dt)
        got = adapter.compute()
        for member, (state, member_cfg, driver) in enumerate(pairs):
            state.elapsed_seconds = float(number * member_cfg.dt)
            expected = driver.compute(state, member_cfg)
            for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi", "rqs"):
                want, have = getattr(expected, name), getattr(got, name)
                if want is None:
                    assert have is None
                else:
                    height = want.shape[-2]
                    view = have[:, member * height:(member + 1) * height]
                    assert view.get().tobytes() == want.get().tobytes(), (number, member, name)
            # The public native pack declines restart requests to ordinary
            # members. Compare its entire member-owned scientific inventory
            # directly, without pretending its prepared launchers are an
            # ordinary serializable PhysicsDriver.
            expected_manifest = {path: value for _, value, path, _ in
                _bootstrap_array_inventory(member_cfg, driver, cp.ndarray)[0]}
            for name, want in expected_manifest.items():
                have = adapter.driver
                for part in name.split("/")[1:]:
                    have = have[part] if isinstance(have, dict) else getattr(have, part)
                if have.shape != want.shape:
                    height = want.shape[-2]
                    have = (have[:, member * height:(member + 1) * height] if have.ndim == 3
                            else have[member * height:(member + 1) * height])
                assert have.get().tobytes() == want.get().tobytes(), (number, member, name)
            view = member_output_view(SimpleNamespace(batch=batch, physics=adapter), member)
            expected_fields = driver.output_fields()
            actual_fields = view.physics.output_fields()
            assert actual_fields.keys() == expected_fields.keys()
            for name, want in expected_fields.items():
                assert actual_fields[name].get().tobytes() == want.get().tobytes(), (number, member, name)
            for name in state._scratch:
                if name.startswith("cu_"):
                    assert view._scratch[name].get().tobytes() == state._scratch[name].get().tobytes()
            assert adapter.driver.call_counts == driver.call_counts
        adapter.apply_microphysics()
        from woof.core.microphysics import apply
        for member, (state, member_cfg, driver) in enumerate(pairs):
            driver.accept_microphysics(apply(state, member_cfg, member_cfg.dt), dt=member_cfg.dt)
            for name in adapter.receipt["model_member_fields"]:
                assert batch.member_view(name, member).get().tobytes() == getattr(state, name).get().tobytes(), (number, member, name)


@pytest.mark.parametrize("full_base", [False, True])
def test_cu99_packed_moist_rk_and_rust_history_match_separate_members(tmp_path, full_base):
    """Real RK plumbing, retained CU cadence and Rust writer inventory."""
    import cupy as cp
    from types import SimpleNamespace
    from woof.core import dycore
    from woof.core.device_inventory import state_array_shapes
    from woof.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    from woof.ensemble.batch_physics_init import (bootstrap_physics_snapshot,
        initialize_member_physics_from_bootstrap)
    from woof.ensemble.batch_moist_dycore import (workspace_specs,
        required_scratch_slots, prepare_moist_step)
    from woof.ensemble.native_forecast import member_output_view
    from woof.io.wrfout import WrfoutWriter, state_frame
    pairs = [_driver(1, member, full_base=full_base) for member in range(2)]
    source, cfg, bootstrap = _driver(1, 0, full_base=full_base)
    snapshots = tuple(bootstrap_physics_snapshot(c, d) for s, c, d in pairs)
    extras = workspace_specs(cfg)
    batch = BatchedDomainState.from_prepared(
        tuple(_prepared(s, c, extras=extras) for s, c, d in pairs),
        available_bytes=2 << 30, array_module=cp, extra_specs=extras,
        scratch_slots=required_scratch_slots(cfg),
        shared_fields=tuple(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    adapter = initialize_member_physics_from_bootstrap(batch, source, bootstrap,
        available_bytes=2 << 30, member_bootstraps=snapshots)
    advance = prepare_moist_step(batch, physics_adapter=adapter)
    owners = SimpleNamespace(batch=batch, physics=adapter)
    def write(path, state, valid):
        frame = state_frame(state, include_diagnostic_pressure=True)
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                dx=cfg.dx, dy=cfg.dy, global_attrs={"DT": np.float32(cfg.dt)},
                field_schema=frame, engine="rust") as writer:
            writer.write_frame(valid, frame)
    for step in range(4):
        advance()
        for member, (state, member_cfg, driver) in enumerate(pairs):
            dycore.step(state, member_cfg, acoustic=True)
            for name in state_array_shapes(cfg):
                assert batch.member_view(name, member).get().tobytes() == getattr(state, name).get().tobytes(), (step, member, name)
            assert state.elapsed_seconds == batch.elapsed_seconds
            packed_path = tmp_path / f"packed-{step}-{member}.nc"
            single_path = tmp_path / f"single-{step}-{member}.nc"
            valid = f"2026-10-16_09:00:{(step + 1) * 12:02d}"
            write(packed_path, member_output_view(owners, member), valid)
            write(single_path, state, valid)
            assert packed_path.read_bytes() == single_path.read_bytes(), (step, member, "history")
        assert batch.clock["step_count"] == step + 1
