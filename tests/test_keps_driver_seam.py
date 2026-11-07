"""The K-epsilon PBL (bl_pbl_physics=17) through the physics driver.

On the card: ``initialize_physics`` allocates the scheme's fields, the
dycore's step calls ``_run_keps``, the tendencies are coupled, and a
forecast that checkpoints mid-run resumes bit for bit.
"""

from __future__ import annotations

import pytest

from woof.config import KEPS_PBL_SCHEME, RunConfig


def _driver_cfg(**overrides) -> RunConfig:
    base = dict(nx=6, ny=5, nz=24, dx=3000.0, dy=3000.0, dt=6.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                sf_sfclay_physics=1, bl_pbl_physics=KEPS_PBL_SCHEME,
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
    profile = np.linspace(3.0, 15.0, cfg.nz, dtype=np.float32)
    state.u[...] = cp.asarray(profile)[:, None, None]
    if state.qv is not None:
        state.qv[...] = 0.008
    return state


@pytest.mark.gpu
def test_the_driver_runs_the_scheme_and_couples_it():
    """Allocated, reset on steps 1-2, then prognostic, finite and coupled."""
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)
    from woof.core.physics_inventory import KEPS_DIAGNOSTICS_3D, KEPS_FIELDS

    cfg = _driver_cfg()
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=310.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
    for name in (*KEPS_FIELDS, *KEPS_DIAGNOSTICS_3D):
        assert driver.fields[name].shape == (cfg.nz, cfg.ny, cfg.nx), name
    run_steps(state, cfg, 2)
    # module_bl_keps.F:250-258: the first two calls set the floors.
    assert float(cp.max(driver.fields["keps_tke"])) == pytest.approx(1.0e-4)
    assert float(cp.max(driver.fields["keps_pr"])) == 1.0
    assert float(cp.max(cp.abs(driver.fields["exch_m"]))) == 0.0
    run_steps(state, cfg, 10)
    for name in ("keps_tke", "keps_diss", "keps_tpe", "keps_pr", "pblh",
                 "exch_h", "exch_m", "wt_tur", "wu_tur", "wv_tur"):
        values = cp.asnumpy(driver.fields[name])
        assert np.all(np.isfinite(values)), name
    # A warm surface under shear: the scheme leaves its floors and mixes.
    assert float(cp.max(driver.fields["keps_tke"])) > 1.0e-4
    assert float(cp.max(driver.fields["exch_m"])) >= 0.09
    assert float(cp.max(driver.fields["pblh"])) > 0.0
    # WQ_TUR is WRF's never-assigned INTENT(OUT): it stays the zero it is.
    assert float(cp.max(cp.abs(driver.fields["wq_tur"]))) == 0.0
    assert np.all(np.isfinite(cp.asnumpy(state.thp)))


def _keps_restart_state(cp):
    """tests/test_restart.py's small full-physics state under K-epsilon,
    with shear and a warm surface so the scheme is active at the
    checkpoint rather than sitting on its floors."""
    import numpy as np

    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, bl_pbl_physics=KEPS_PBL_SCHEME)
    profile = np.linspace(2.0, 14.0, cfg.nz, dtype=np.float32)
    state.u[...] = cp.asarray(profile)[:, None, None]
    driver.fields["tsk"][...] = cp.float32(306.0)
    driver.fields["tslb"][...] = cp.float32(303.0)
    return state, cfg, driver


@pytest.mark.gpu
def test_a_keps_run_restarts_bit_identically(tmp_path):
    """20 steps + checkpoint + 20 steps == 40 steps, bit for bit.

    The scheme carries TKE, dissipation and TPE from call to call, and its
    first two calls are a reset keyed on itimestep: a resume must continue
    on the stored words, not reset (the checkpoint is at step 20, far past
    the reset), and must reproduce the advected copies and fluxes too.
    """
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal

    state_a, cfg_a, _ = _keps_restart_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _keps_restart_state(cp)
    run_steps(state_b, cfg_b, 20)
    assert float(cp.max(driver_b.fields["keps_tke"])) > 1.0e-4
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _keps_restart_state(cp)
    restart.restore_restart(mid, state_c, cfg_c)
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)
    _assert_restart_equal(resumed, reference)

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for name in ("keps_tke", "keps_diss", "keps_tpe", "keps_pr",
                 "keps_tke_adv", "keps_diss_adv", "keps_tpe_adv",
                 "wu_tur", "wv_tur", "wt_tur", "wq_tur"):
        assert f"fields/{name}" in keys, name
    assert np.isfinite(cp.asnumpy(state_c.thp)).all()


@pytest.mark.gpu
def test_scalar_package_is_transported_without_ice():
    import cupy as cp
    import numpy as np
    from woof.core.dycore import run_steps
    from woof.core.physics import initialize_physics, DECLARED_CONSTANT_GLW_WM2
    from woof.core.moist import extra_moist_species
    cfg = _driver_cfg(mp_physics=1)
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=310.0,
        glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
    run_steps(state, cfg, 2)
    names = ("keps_tke_adv", "keps_diss_adv", "keps_tpe_adv")
    assert set(names).issubset(extra_moist_species(state))
    profile = np.broadcast_to(np.linspace(0.01, 0.1, cfg.nx, dtype=np.float32),
                              (cfg.nz, cfg.ny, cfg.nx)).copy()
    for name in names:
        assert driver.fields[name] is getattr(state, name)
        getattr(state, name)[...] = cp.asarray(profile)
    before = {name: cp.asnumpy(getattr(state, name)) for name in names}
    run_steps(state, cfg, 3)
    for name in names:
        got = cp.asnumpy(getattr(state, name))
        assert np.isfinite(got).all(), name
        assert not np.array_equal(got, before[name]), name


@pytest.mark.gpu
def test_keps_receives_the_landed_bep_surface_blend():
    import cupy as cp
    import numpy as np
    from woof.core.dycore import run_steps
    from test_urban_default_off_identity import _small_driver
    state, cfg, driver = _small_driver(bl_pbl_physics=17,
        sf_sfclay_physics=1, sf_urban_physics=2)
    assert driver.urban is not None and driver.urban.option == 2
    run_steps(state, cfg, 4)
    city = cp.asnumpy(driver.fields["utype_urb2d"]) > 0
    assert city.any() and (~city).any()
    assert driver.urban.pbl_terms is not None
    for name in ("hfx", "qfx", "ust", "pblh", "keps_tke", "keps_diss",
                 "keps_tpe", "exch_m"):
        assert np.isfinite(cp.asnumpy(driver.fields[name])).all(), name
    assert (cp.asnumpy(driver.fields["sh_urb2d"])[city] != 0).any()


@pytest.mark.gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_keps_nested_dry_scalar_package(feedback):
    """Both clocks run KEPS and exchange its Registry scalar package."""
    from dataclasses import replace
    from datetime import datetime
    import cupy as cp
    import numpy as np
    from woof.experiment import (DomainConfig, ExperimentConfig,
                                  ProjectionConfig, VerticalConfig)
    from woof.core.model import execute_experiment
    from woof.core.nest import NestCoupler
    from woof.core.nest_fields import nest_field_kinds
    from woof.core.physics import initialize_physics, DECLARED_CONSTANT_GLW_WM2
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    cfg = _driver_cfg(nx=40, ny=40, nz=24, moist=False, mp_physics=0,
                      run_seconds=24.0)
    child_cfg = replace(cfg, nx=30, ny=30, dx=1000.0, dy=1000.0,
                        dt=2.0, grid_id=2, nested=True)
    root = DomainConfig(grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=24.0, run=cfg, time_step=6)
    child = DomainConfig(grid_id=2, parent_id=1, i_parent_start=10,
        j_parent_start=10, parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=24.0, run=child_cfg)
    exp = ExperimentConfig(name="keps-nest", start_time=datetime(1974, 4, 3, 12),
        run_seconds=24.0, feedback=feedback,
        vertical=VerticalConfig(tuple(np.linspace(1., 0., cfg.nz + 1)),
                                5000., 2, .2),
        projection=ProjectionConfig("lambert", 35., -97., 30., 60., -97.),
        restart_interval_s=0., domains=(root, child))
    def prepare(state, dc, grid, start_time):
        from woof.core.diagnostics import update_diagnostics
        update_diagnostics(state, dc.run.hypsometric_opt)
        return initialize_physics(state, dc.run, landmask=1., tsk=310.,
            glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.)

    model = assemble_idealized_tree(exp, _driver_state(cfg),
        coupler_factory=lambda node: NestCoupler(node, feedback=feedback),
        domain_preparer=prepare)
    execute_experiment(model)
    for node in model.walk_parent_first():
        names = ("keps_tke_adv", "keps_diss_adv", "keps_tpe_adv")
        assert set(names) <= set(nest_field_kinds(node.cfg.run))
        assert node.state.elapsed_seconds == 24.0
        for name in (*names, "keps_tke", "keps_diss", "keps_tpe"):
            value = node.state.physics.fields[name]
            assert np.isfinite(cp.asnumpy(value)).all(), (feedback, name)
    coupler = model.nodes_by_grid_id[2].coupler
    assert coupler.force_count == 4
    assert bool(coupler.feedback_count) == bool(feedback)


@pytest.mark.gpu
def test_keps_ensemble_member_fields_are_independent():
    import cupy as cp
    import numpy as np
    from woof.core.physics import initialize_physics, DECLARED_CONSTANT_GLW_WM2
    from woof.core.physics_inventory import KEPS_FIELDS
    from woof.core.dycore import run_steps
    cfg = _driver_cfg()
    a, b = _driver_state(cfg), _driver_state(cfg)
    drivers = [initialize_physics(s, cfg, landmask=1., tsk=310.,
               glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.) for s in (a, b)]
    before = {n: cp.asnumpy(drivers[1].fields[n]) for n in KEPS_FIELDS}
    run_steps(a, cfg, 4)
    for n in KEPS_FIELDS:
        assert drivers[0].fields[n].data.ptr != drivers[1].fields[n].data.ptr
        assert np.array_equal(cp.asnumpy(drivers[1].fields[n]), before[n])
    assert float(cp.max(drivers[0].fields['keps_tke'])) > 1.e-4
    assert b.elapsed_seconds == 0.
