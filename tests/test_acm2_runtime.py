"""Coupled ACM2 state and checkpoint checks on an idealized sounding.

These checks do not replace real native-WRF capture or forecast proof.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.config import RunConfig


def _build(pbl=7):
    cp = pytest.importorskip("cupy")
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = RunConfig(nx=8, ny=6, nz=40, dx=3000.0, dy=3000.0,
                    ztop=16000.0, dt=12.0, run_seconds=0.0,
                    time_step_sound=4, moist=True, mp_physics=1,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=pbl, bldt=0.0)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def vapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=3.2)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, vapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)
    land = np.ones((cfg.ny, cfg.nx), np.float64)
    land[:, -2:] = 0.0
    tsk = np.where(land, 301.0, 297.0)
    soil_t = np.stack([tsk - d for d in (0.5, 1.0, 1.5, 2.0)])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    soil_m[:, land == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=land, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(land, 10, 17), isltyp=np.where(land, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    return state, cfg, driver


@pytest.mark.gpu
def test_acm2_coupled_restart_preserves_carriers_and_next_step(tmp_path):
    cp = pytest.importorskip("cupy")
    from woof.core.dycore import step, stability_report
    from woof.io.restart import restore_restart, write_restart
    from woof.io.wrfout import state_frame

    state, cfg, driver = _build()
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_acm2"
    for _ in range(3):
        step(state, cfg)
    assert not stability_report(state, cfg)["nan"]
    assert not bool(cp.all(driver.fields["pblh"] == cp.float32(500.0)))
    assert float(cp.max(driver.fields["exch_h"])) > 0.0
    frame = state_frame(state)
    for name in ("PBLH", "EXCH_H", "EXCH_M", "KPBL", "RMOL", "REGIME"):
        assert name in frame, name
        assert np.isfinite(np.asarray(frame[name])).all(), name

    path = tmp_path / "acm2.npz"
    write_restart(path, state, cfg)
    fresh, fresh_cfg, restored = _build()
    restore_restart(path, fresh, fresh_cfg)
    carried = ("pblh", "kpbl", "regime", "rmol", "exch_h", "exch_m")

    def equal(a, b, label):
        ah, bh = cp.asnumpy(a), cp.asnumpy(b)
        assert np.array_equal(ah.view(np.uint32), bh.view(np.uint32)), label

    for name in carried:
        assert restored.fields[name] is not driver.fields[name]
        equal(restored.fields[name], driver.fields[name], name)
    step(state, cfg)
    step(fresh, fresh_cfg)
    for name in carried:
        equal(restored.fields[name], driver.fields[name], f"next/{name}")
    for name in ("u", "v", "w", "thp", "qv", "qc", "qr"):
        equal(getattr(fresh, name), getattr(state, name), f"next/{name}")
