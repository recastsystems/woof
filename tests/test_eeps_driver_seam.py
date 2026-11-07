"""EEPS (bl_pbl_physics=16) through the physics driver and the dycore.

On the card: ``initialize_physics`` seeds PEK_PBL/PEP_PBL at EEPSINIT's
minpek/minpep, the state carries the transported PEK_ADV/PEP_ADV, the
first PBL call copies the one into the other (WRF's itimestep == 1 arm),
the dycore advects the pair between calls, and a checkpoint resumes on
exactly those words.  The column itself is graded against WRF in
tests/test_eeps_wrf461_parity.py.
"""

from __future__ import annotations

import pytest

from woof.config import EEPS_PBL_SCHEME, RunConfig


def _driver_cfg(**overrides) -> RunConfig:
    base = dict(nx=8, ny=5, nz=24, dx=3000.0, dy=3000.0, dt=6.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                sf_sfclay_physics=1, bl_pbl_physics=EEPS_PBL_SCHEME,
                sf_surface_physics=2, num_soil_layers=4, bldt=0.0,
                radt=0.0, cu_physics=0, km_opt=4, c_s=0.25)
    base.update(overrides)
    return RunConfig(**base)


def _driver_state(cfg):
    import cupy as cp
    import numpy as np

    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = cp.asarray(np.full(tuple(state.u.shape), 6.0),
                              dtype=state.u.dtype)
    if state.qv is not None:
        state.qv[...] = 0.008
    return state


def _skin(cfg):
    """A skin temperature that warms eastward, so the surface heat flux,
    the TKE it drives and therefore PEK_ADV vary along the wind."""
    import numpy as np

    return np.broadcast_to(np.linspace(300.0, 312.0, cfg.nx),
                           (cfg.ny, cfg.nx)).copy()


@pytest.mark.gpu
def test_the_driver_seeds_runs_and_couples_the_scheme():
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.core.eeps_pbl import EEPS_MINPEK, EEPS_MINPEP
    from woof.core.moist import moist_species
    from woof.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)

    cfg = _driver_cfg()
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=_skin(cfg),
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
    f = driver.fields
    assert f["pek_pbl"].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert np.all(cp.asnumpy(f["pek_pbl"]) == EEPS_MINPEK)
    assert np.all(cp.asnumpy(f["pep_pbl"]) == EEPS_MINPEP)
    # WRF's 'i0': the advected pair is zero until the first PBL call.
    assert not np.any(cp.asnumpy(state.pek_adv))
    assert {"pek_adv", "pep_adv"} <= set(moist_species(state))

    run_steps(state, cfg, 10)
    assert driver.call_counts["ysu"] == 10          # the PBL-slot counter
    for name in ("pek_pbl", "pep_pbl", "pblh", "exch_h", "exch_m"):
        assert np.all(np.isfinite(cp.asnumpy(f[name]))), name
    pek = cp.asnumpy(f["pek_pbl"])
    assert pek.max() > 10 * EEPS_MINPEK               # the closure is live
    assert float(cp.asnumpy(f["pblh"]).max()) > 0.0
    assert float(cp.asnumpy(f["exch_h"]).max()) > 0.0
    assert np.all(np.isfinite(cp.asnumpy(state.thp)))
    # The pair is finite and positive after transport.
    adv = cp.asnumpy(state.pek_adv)
    assert np.all(np.isfinite(adv)) and adv.min() >= 0.0


@pytest.mark.gpu
def test_the_dycore_transports_the_pair_between_pbl_calls():
    """With the PBL due every second step, the step after a call advects
    PEK_ADV away from the copy the call wrote."""
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)

    cfg = _driver_cfg(bldt=0.2)                    # 12 s = every 2 steps
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=_skin(cfg),
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
    run_steps(state, cfg, 4)
    written = cp.asnumpy(driver.fields["pek_pbl"])
    carried = cp.asnumpy(state.pek_adv)
    # The PBL wrote PEK_ADV = PEK_PBL on its last call; the dycore has
    # moved PEK_ADV since, and PEK_PBL stays the call's own copy.
    assert not np.array_equal(written.view(np.uint32),
                              carried.view(np.uint32))
    assert np.all(np.isfinite(carried))


def _eeps_restart_state(cp):
    """tests/test_restart.py's small full-physics state under EEPS, with a
    sheared wind and a warm skin so the closure is turbulent at the
    checkpoint."""
    import numpy as np

    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, bl_pbl_physics=EEPS_PBL_SCHEME)
    profile = np.linspace(2.0, 14.0, cfg.nz, dtype=np.float32)
    state.u[...] = cp.asarray(profile)[:, None, None]
    driver.fields["tsk"][...] = cp.float32(306.0)
    driver.fields["tslb"][...] = cp.float32(303.0)
    return state, cfg, driver


@pytest.mark.gpu
def test_an_eeps_run_restarts_bit_identically(tmp_path):
    """20 steps + checkpoint + 20 steps == 40 steps, bit for bit, with the
    transported PEK_ADV/PEP_ADV and PEK_PBL/PEP_PBL in the checkpoint."""
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.core.eeps_pbl import EEPS_MINPEK
    from woof.io import restart
    from test_restart import _assert_restart_equal

    state_a, cfg_a, _ = _eeps_restart_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _eeps_restart_state(cp)
    run_steps(state_b, cfg_b, 20)
    assert float(cp.asnumpy(state_b.pek_adv).max()) > 10 * EEPS_MINPEK
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _eeps_restart_state(cp)
    restart.restore_restart(mid, state_c, cfg_c)
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)
    _assert_restart_equal(resumed, reference)

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for key in ("state/pek_adv", "state/pep_adv", "fields/pek_pbl",
                "fields/pep_pbl"):
        assert key in keys, key
    assert np.isfinite(cp.asnumpy(state_c.thp)).all()
