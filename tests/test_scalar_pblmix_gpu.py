"""GPU scalar_pblmix must preserve WRF's column rates and chunk boundaries."""
from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.mynn_scalar_mix_gpu import scalar_pblmix_columns_cuda

DATA = Path(__file__).parent / "data" / "scalar_pblmix_wrf461.npz"


@requires_gpu
@pytest.mark.parametrize("chunk", [1, 7, 32])
def test_scalar_pblmix_cuda_matches_wrf_fortran_and_chunking(chunk):
    with np.load(DATA) as data:
        arrays = {key: np.ascontiguousarray(data[key][:, :4]).reshape(32, 50)
                  for key in ("qn", "dz", "rho", "exch_h", "dt", "tendency")}
    all_rates = []
    for start in range(0, 32, chunk):
        stop = min(start + chunk, 32)
        inputs = [cp.asarray(arrays[key][start:stop])
                  for key in ("qn", "dz", "rho", "exch_h")]
        scratch = cp.empty((stop - start, 251), dtype=cp.float32, order="F")
        solved, rate = scalar_pblmix_columns_cuda(
            *inputs, cp.asarray(arrays["dt"][start:stop, 0]), scratch=scratch)
        np.testing.assert_array_equal(cp.asnumpy(solved[:, -1]),
                                      arrays["qn"][start:stop, -1])
        all_rates.append(cp.asnumpy(rate))
    np.testing.assert_array_equal(np.concatenate(all_rates).view(np.uint32),
                                  arrays["tendency"].view(np.uint32))


@requires_gpu
@pytest.mark.parametrize("column_chunk", [None, 7])
def test_scalar_pblmix_reaches_coupled_mynn(monkeypatch, column_chunk):
    from woof.config import RunConfig, validate_run_config
    from woof.core.dycore import step, stability_report
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    from woof.core.mynn_scalar_mix import scalar_pblmix_column
    import woof.core.mynn_scalar_mix_gpu as scalar_module
    import woof.core.mynn_pbl_runtime as runtime_module

    if column_chunk is not None:
        monkeypatch.setattr(runtime_module, "resolve_mynn_column_chunk",
                            lambda nz: column_chunk)

    cfg = RunConfig(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=28, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0, scalar_pblmix=1, bl_mynn_mixlength=2)
    validate_run_config(cfg)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)
    tsk = np.full((cfg.ny, cfg.nx), 301.0)
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    driver = initialize_physics(
        state, cfg, landmask=np.ones_like(tsk), tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=10, isltyp=6,
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    # Nonuniform aerosol profiles make a disconnected/zero tendency fail.
    profile = cp.linspace(1.0, 0.1, cfg.nz, dtype=cp.float32)[:, None, None]
    state.nwfa[...] *= profile
    state.nifa[...] *= profile
    captured = []
    original = scalar_module.scalar_pblmix_columns_cuda

    def capture(qn, dz, rho, exch_h, delt, **kwargs):
        result = original(qn, dz, rho, exch_h, delt, **kwargs)
        captured.append((*(cp.asnumpy(a).copy() for a in (qn, dz, rho, exch_h)),
                         np.float32(delt), cp.asnumpy(result[1])))
        return result

    monkeypatch.setattr(scalar_module, "scalar_pblmix_columns_cuda", capture)
    for _ in range(3):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]
    assert len(captured) == (12 if column_chunk is None else 84)
    if column_chunk == 7:
        # 48 columns make six 7-column calls and one 6-column final call.
        assert {entry[0].shape[0] for entry in captured} == {6, 7}
    for qn, dz, rho, kh, dt, rate in captured:
        for column in {0, qn.shape[0] // 2, qn.shape[0] - 1}:
            _, reference = scalar_pblmix_column(
                qn[column], dz[column], rho[column], kh[column], dt)
            np.testing.assert_array_equal(rate[column].view(np.uint32),
                                          reference.view(np.uint32))
    extra = driver.pbl_tendencies.extra_scalars
    assert set(extra) == {"nc", "ni", "nwfa", "nifa"}
    for name in ("nwfa", "nifa"):
        assert bool(cp.isfinite(extra[name]).all())
        assert bool(cp.any(extra[name] != 0))
    # WRF does not PBL-mix the precipitating rain-number scalar.
    assert "nr" not in extra


def _coupled_scalar_pblmix_run(monkeypatch, pbl, surface, steps=3):
    """Three coupled steps under mp=28 with scalar_pblmix=1; returns the
    driver and every diff4d call the run made."""
    from woof.config import RunConfig, validate_run_config
    from woof.core.dycore import step, stability_report
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    import woof.core.mynn_scalar_mix_gpu as scalar_module

    cfg = RunConfig(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=28, sf_sfclay_physics=surface, sf_surface_physics=2,
        bl_pbl_physics=pbl, bldt=0.0, scalar_pblmix=1)
    validate_run_config(cfg)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)
    tsk = np.full((cfg.ny, cfg.nx), 301.0)
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    driver = initialize_physics(
        state, cfg, landmask=np.ones_like(tsk), tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=10, isltyp=6,
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    profile = cp.linspace(1.0, 0.1, cfg.nz, dtype=cp.float32)[:, None, None]
    state.nwfa[...] *= profile
    state.nifa[...] *= profile
    captured = []
    original = scalar_module.scalar_pblmix_columns_cuda

    def capture(qn, dz, rho, exch_h, delt, **kwargs):
        result = original(qn, dz, rho, exch_h, delt, **kwargs)
        captured.append((*(cp.asnumpy(a).copy() for a in (qn, dz, rho, exch_h)),
                         np.float32(delt), cp.asnumpy(result[1])))
        return result

    monkeypatch.setattr(scalar_module, "scalar_pblmix_columns_cuda", capture)
    for _ in range(steps):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]
    return driver, captured


@requires_gpu
@pytest.mark.parametrize("pbl,surface", [(1, 91), (2, 2), (9, 91), (11, 91),
                                      (12, 1), (12, 91)])
def test_scalar_pblmix_reaches_every_declared_exchange_producer(monkeypatch, pbl, surface):
    # What breaks without this: mp=28 with scalar_pblmix=1 (WRF's own
    # check_a_mundo default for aerosol IC/BCs) was refused under MYJ,
    # Shin-Hong and UW although WRF runs diff4d on any scheme's EXCH_H.
    from woof.core.mynn_scalar_mix import scalar_pblmix_column

    driver, captured = _coupled_scalar_pblmix_run(monkeypatch, pbl, surface)
    # Four species per step, one chunk of all 48 columns each.
    assert len(captured) == 12
    assert {entry[0].shape[0] for entry in captured} == {48}
    assert any(np.any(entry[3][:, 1:] > 0) for entry in captured), "the scheme's EXCH_H never mixed"
    for qn, dz, rho, kh, dt, rate in captured:
        for column in {0, qn.shape[0] // 2, qn.shape[0] - 1}:
            _, reference = scalar_pblmix_column(
                qn[column], dz[column], rho[column], kh[column], dt)
            np.testing.assert_array_equal(rate[column].view(np.uint32),
                                          reference.view(np.uint32))
    extra = driver.pbl_tendencies.extra_scalars
    assert {"nc", "ni", "nwfa", "nifa"} <= set(extra)
    for name in ("nwfa", "nifa"):
        assert bool(cp.isfinite(extra[name]).all())
        assert bool(cp.any(extra[name] != 0))
    assert "nr" not in extra


@requires_gpu
def test_scalar_pblmix_without_a_pbl_scheme_mixes_nothing(monkeypatch):
    # WRF does not call its PBL driver with bl_pbl_physics=0, so the
    # request is a no-op there and must be one here, not a refusal.
    driver, captured = _coupled_scalar_pblmix_run(monkeypatch, 0, 91)
    assert captured == []
    assert not (getattr(driver.pbl_tendencies, "extra_scalars", None) or {})
