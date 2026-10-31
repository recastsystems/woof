"""Short CUDA checks of translation, ground stress, history and restart."""
from datetime import datetime

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def build_case(*, wind=(0., 0.), frame=(0., 0.), bubble=True, drag=False, km_opt=1):
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation
    from woof.core.diagnostics import update_diagnostics

    values = dict(nx=64, ny=64, nz=40, dx=1000., dy=1000., ztop=10000.,
                  dt=1., run_seconds=3., km_opt=km_opt,
                  sf_sfclay_physics=1 if drag else 0,
                  bl_pbl_physics=1 if drag else 0, isfflx=0)
    # Also usable against a runtime that predates the moving-frame fields.
    if "umove" in RunConfig.__dataclass_fields__:
        values.update(umove=frame[0], vmove=frame[1])
    else:
        assert frame == (0., 0.)
    cfg = RunConfig(**values)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    y = (np.arange(cfg.ny) + .5) * cfg.dy - .5 * cfg.ny * cfg.dy

    def perturbation(x, z):
        radius = np.sqrt((x[None, None, :] / 4000.) ** 2
                         + (y[None, :, None] / 4000.) ** 2
                         + ((z[:, None, None] - 2500.) / 2000.) ** 2)
        return np.where((radius < 1.) & bubble, 2. * np.cos(.5 * np.pi * radius) ** 2, 0.)

    state = init_theta_perturbation(cfg, coord, base, perturbation)
    state.u[...] = np.float32(wind[0])
    state.v[...] = np.float32(wind[1])
    update_diagnostics(state)
    if drag:
        from woof.core.physics import initialize_physics
        initialize_physics(state, cfg, tsk=300., pblh=500.)
    return cfg, state


def advance(state, cfg, count=3):
    from woof.core.dycore import step
    for index in range(count):
        step(state, cfg)
        state.elapsed_seconds = (index + 1) * cfg.dt


def words(array):
    import cupy as cp
    return cp.asnumpy(array).view(np.uint32)


def test_both_axes_are_exactly_galilean_with_ground_history():
    from woof.io.wrfout import state_frame
    from woof.moving_frame import initialize_moving_frame
    a_cfg, a = build_case(wind=(12., -8.), frame=(12., -8.))
    b_cfg, b = build_case()
    initialize_moving_frame(a, a_cfg)
    advance(a, a_cfg)
    advance(b, b_cfg)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        np.testing.assert_array_equal(words(getattr(a, name)), words(getattr(b, name)))
    fa, fb = state_frame(a), state_frame(b)
    for name in fa:
        offset = {"U": 12., "V": -8.}.get(name, 0.)
        expected = fb[name] + np.float32(offset) if offset else fb[name]
        np.testing.assert_array_equal(fa[name].view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("frame", [(0., 0.), (4., -2.), (8., -4.)])
def test_uniform_flow_stays_uniform(frame):
    from woof.io.wrfout import state_frame
    cfg, state = build_case(wind=(8., -4.), frame=frame, bubble=False)
    advance(state, cfg)
    output = state_frame(state)
    np.testing.assert_array_equal(output["U"], np.full_like(output["U"], 8.))
    np.testing.assert_array_equal(output["V"], np.full_like(output["V"], -4.))


def test_first_step_stress_and_pbl_momentum_use_ground_wind():
    a_cfg, a = build_case(wind=(12., -8.), frame=(12., -8.), bubble=False, drag=True)
    c_cfg, c = build_case(wind=(12., -8.), bubble=False, drag=True)
    ta, tc = a.physics.compute(a, a_cfg), c.physics.compute(c, c_cfg)
    for name in ("ust", "wspd", "u10", "v10", "hfx", "qfx"):
        np.testing.assert_array_equal(words(a.physics.fields[name]), words(c.physics.fields[name]))
    assert np.all(words(a.physics.fields["ust"]) != 0)
    for name in ("ru", "rv"):
        np.testing.assert_array_equal(words(getattr(ta, name)), words(getattr(tc, name)))


def test_fast_physics_preparation_also_uses_ground_winds(monkeypatch):
    from woof.core import physics
    monkeypatch.setattr(physics, "WRF_EXACT", False)
    _, moving = build_case(wind=(12., -8.), frame=(12., -8.), bubble=False)
    _, fixed = build_case(wind=(12., -8.), bubble=False)
    a = physics._prepare_atmosphere(moving)
    b = physics._prepare_atmosphere(fixed)
    for key in ("u", "v"):
        np.testing.assert_array_equal(words(a[key]), words(b[key]))


@pytest.mark.parametrize("isfflx", [0, 1, 2])
def test_les_wall_stress_uses_ground_wind_in_each_surface_arm(isfflx):
    from dataclasses import replace
    import cupy as cp
    from woof.core.dycore import launch_wrf_smag2d_vertical
    results = []
    for frame in ((12., -8.), (0., 0.)):
        cfg, state = build_case(wind=(12., -8.), frame=frame, bubble=False, drag=True)
        state.physics.compute(state, cfg)
        cfg = replace(cfg, bl_pbl_physics=0, isfflx=isfflx, tke_drag_coefficient=.002)
        ru, rv = cp.zeros_like(state.u), cp.zeros_like(state.v)
        launch_wrf_smag2d_vertical(state, cfg, cp.zeros_like(state.thp),
            ru=ru, rv=rv, rw=None, rth=cp.zeros_like(state.thp), rqv=None,
            time_t=False, interior=False)
        results.append((words(ru), words(rv)))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
        assert np.any(a != 0)


def test_les_surface_tke_production_uses_ground_wind():
    from dataclasses import replace
    import cupy as cp
    from woof.core.dycore import launch_wrf_tke_km
    from woof.moving_frame import initialize_moving_frame
    results = []
    for frame in ((12., -8.), (0., 0.)):
        cfg, state = build_case(wind=(12., -8.), frame=frame, bubble=False, km_opt=2)
        state.u += (cp.arange(cfg.nz, dtype=cp.float32) * .5)[:, None, None]
        state.v += (cp.arange(cfg.nz, dtype=cp.float32) * .25)[:, None, None]
        initialize_moving_frame(state, cfg)
        cfg = replace(cfg, tke_drag_coefficient=.002)
        state.tke[...] = 1.
        launch_wrf_tke_km(state, cfg, cp.zeros_like(state.thp), cp.zeros_like(state.thp),
                           time_t=False)
        results.append(words(state.existing_scratch("smag_rtke")))
    np.testing.assert_array_equal(*results)


def test_restart_restores_relative_winds_without_second_subtraction(tmp_path):
    from woof.io.restart import write_restart, restore_restart
    cfg, state = build_case(wind=(12., -8.), frame=(12., -8.))
    advance(state, cfg, 1)
    path = tmp_path / "restart.npz"
    write_restart(path, state, cfg)
    _, restored = build_case(wind=(12., -8.), frame=(12., -8.))
    restore_restart(path, restored, cfg)
    advance(state, cfg, 1)
    advance(restored, cfg, 1)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        np.testing.assert_array_equal(words(getattr(state, name)), words(getattr(restored, name)))


def test_async_history_winds_and_per_output_offsets(tmp_path):
    import netCDF4
    from woof.io.wrfout import AsyncDomainWrfoutWriter, state_frame
    cfg, state = build_case(wind=(12., -8.), frame=(12., -8.))
    advance(state, cfg, 1)
    expected = state_frame(state, include_diagnostic_pressure=True)
    writer = AsyncDomainWrfoutWriter(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                                    dx=cfg.dx, dy=cfg.dy, title="moving frame", global_attrs={})
    path = tmp_path / "history.nc"
    try:
        writer.submit(path, datetime(2000, 1, 1, 0, 0, 1), state)
    finally:
        writer.close()
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        assert ds.UMOVE == 12. and ds.VMOVE == -8.
        assert ds.DOMAIN_X_OFFSET_M == 12. and ds.DOMAIN_Y_OFFSET_M == -8.
        for name in ("U", "V"):
            np.testing.assert_array_equal(ds[name][0].view(np.uint32), expected[name].view(np.uint32))
