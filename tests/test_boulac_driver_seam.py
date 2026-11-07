"""Real BouLac driver and byte-identical restart continuation on a GPU."""
from __future__ import annotations
import pytest

def _state(cp, *, bldt=0.0):
    import numpy as np
    from test_restart import _physics_state
    state, cfg, driver = _physics_state(cp, bl_pbl_physics=8, nz=24, ztop=2500., dt=2., bldt=bldt)
    state.u[...] = cp.asarray(np.linspace(2., 42., cfg.nz, dtype=np.float32))[:, None, None]
    driver.fields['tsk'][...] = cp.float32(306)
    driver.fields['tslb'][...] = cp.float32(303)
    return state, cfg, driver

@pytest.mark.gpu
def test_boulac_driver_runs_and_publishes_carried_state():
    import cupy as cp
    import numpy as np
    from woof.core.dycore import run_steps
    from woof.core.physics_inventory import BOULAC_STATE_3D
    from woof.io.history_layout import live_state_history_fields
    state, cfg, driver = _state(cp)
    assert float(driver.fields['tke_pbl'][0, 0, 0]) == pytest.approx(1.e-4)
    run_steps(state, cfg, 10)
    assert driver.call_counts['ysu'] > 0
    for name in BOULAC_STATE_3D + ('exch_h', 'exch_m', 'pblh'):
        assert np.isfinite(cp.asnumpy(driver.fields[name])).all(), name
    assert float(driver.fields['tke_pbl'].max()) > 1.e-4
    assert 'TKE_PBL' in live_state_history_fields(state)
    assert driver.pbl_tendencies is not None

@pytest.mark.gpu
@pytest.mark.parametrize("bldt", [0.0, 0.1])
def test_boulac_restart_continuation_is_byte_identical(tmp_path, bldt):
    import cupy as cp
    import numpy as np
    from woof.core.dycore import run_steps
    from woof.io import restart
    from woof.core.physics_inventory import BOULAC_STATE_3D
    from test_restart import _assert_restart_equal
    a, ca, _ = _state(cp, bldt=bldt)
    run_steps(a, ca, 40)
    reference = restart.write_restart(tmp_path / 'reference.npz', a, ca)
    b, cb, db = _state(cp, bldt=bldt)
    run_steps(b, cb, 20)
    assert float(db.fields['tke_pbl'].max()) > 1.e-4
    assert float(cp.abs(db.fields['wu_tur']).max()) > 0.
    if bldt:
        # Step 20 lies between calls at steps 19 and 22. The restart must
        # retain the last uncoupled rates, not only the TKE column.
        assert 0 < db.call_counts['ysu'] < 20
        assert db.pbl_raw_rates
        assert float(cp.abs(db.pbl_raw_rates['dtheta']).max()) > 0.
    mid = restart.write_restart(tmp_path / 'mid.npz', b, cb)
    c, cc, dc = _state(cp, bldt=bldt)
    restart.restore_restart(mid, c, cc)
    for name in BOULAC_STATE_3D + ('exch_h', 'exch_m'):
        assert cp.asnumpy(db.fields[name]).tobytes() == cp.asnumpy(dc.fields[name]).tobytes(), name
    run_steps(c, cc, 20)
    resumed = restart.write_restart(tmp_path / 'resumed.npz', c, cc)
    _assert_restart_equal(resumed, reference)
    keys = set(restart.read_restart_header(reference)['array_manifest'])
    assert {f'fields/{n}' for n in ('tke_pbl', 'el_pbl', 'exch_h', 'exch_m')} <= keys
    assert np.isfinite(cp.asnumpy(c.thp)).all()


def _bep_state(cp):
    from datetime import datetime
    import numpy as np
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    cfg = RunConfig(nx=6, ny=4, nz=24, dx=2000., dy=2000.,
                    ztop=2500., dt=2., run_seconds=0., time_step_sound=4,
                    moist=True, mp_physics=10, sf_sfclay_physics=1,
                    sf_surface_physics=2, bl_pbl_physics=8, sf_urban_physics=2,
                    ra_physics=90, cu_physics=0, radt_minutes=1., bldt=0.)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.+.004*np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .011*np.exp(-np.asarray(z)/2400.))
    state.u[...] = cp.asarray(np.linspace(2., 42., cfg.nz, dtype=np.float32))[:,None,None]
    driver = initialize_physics(
        state, cfg, landmask=1., tsk=306., soil_temperature=303.,
        soil_moisture=.31, liquid_moisture=.31, ivgtyp=13, isltyp=6,
        vegfra=0., tmn=286., swdown=450., glw=310.,
        frc_urb2d=.6, radiation_start_time=datetime(2024, 5, 21, 21),
        radiation_latitude=np.full((cfg.ny,cfg.nx),40.),
        radiation_longitude=np.full((cfg.ny,cfg.nx),-95.))
    return state, cfg, driver


@pytest.mark.gpu
def test_boulac_bep_sources_and_restart_are_byte_identical(tmp_path):
    import cupy as cp
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal
    a, ca, da = _bep_state(cp)
    run_steps(a, ca, 20)
    terms = da.urban.pbl_terms
    assert terms is not None
    assert float(cp.abs(terms['a_u_bep']).max()) > 0.
    assert float(cp.abs(terms['b_e_bep']).max()) > 0.
    reference = restart.write_restart(tmp_path/'reference-bep.npz', a, ca)
    b, cb, db = _bep_state(cp)
    run_steps(b, cb, 10)
    assert float(db.fields['tke_pbl'].max()) > 1.e-4
    mid = restart.write_restart(tmp_path/'mid-bep.npz', b, cb)
    c, cc, dc = _bep_state(cp)
    restart.restore_restart(mid, c, cc)
    assert cp.asnumpy(db.fields['tke_pbl']).tobytes() == cp.asnumpy(dc.fields['tke_pbl']).tobytes()
    run_steps(c, cc, 10)
    resumed = restart.write_restart(tmp_path/'resumed-bep.npz', c, cc)
    _assert_restart_equal(resumed, reference)


@pytest.mark.gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_boulac_nested_domains_advance_private_tke(monkeypatch, feedback):
    import cupy as cp
    from woof.core.model import execute_experiment
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor
    monkeypatch.setattr(executor, "I_PARENT_START", 25)
    monkeypatch.setattr(executor, "J_PARENT_START", 25)
    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    parent = moving.parent_cfg(nx=72, ny=72, nz=24, dt=6., run_seconds=18.,
                               bl_pbl_physics=8, cu_physics=0,
                               nwp_diagnostics=0)
    child = moving.child_cfg(parent, nx=48, ny=48)
    boundaries = moving.domain_boundaries(parent, seconds=21600.)
    state, geo = moving.build_parent(parent, boundaries=boundaries, warmup=0)
    exp = executor.experiment(parent, child, steps=3)
    model = executor.assemble(exp, state, geo.grid, feedback=feedback)
    if feedback:
        executor.initial_feedback(model)
    result = execute_experiment(model, validate_state=True,
                                pool_trim_per_period=False)
    assert result.steps == 12
    assert model.node(2).coupler.force_count == 3
    assert model.node(2).coupler.feedback_count == (4 if feedback else 0)
    tke = []
    for node in model.walk_parent_first():
        field = node.state.physics.fields["tke_pbl"]
        assert bool(cp.all(cp.isfinite(field)))
        assert float(field.max()) > 1.e-4
        assert node.state.physics.call_counts["ysu"] > 0
        tke.append(field)
    assert tke[0].data.ptr != tke[1].data.ptr


@pytest.mark.gpu
def test_boulac_member_columns_match_independent_calls():
    import cupy as cp
    import numpy as np
    from woof.core.boulac import launch_boulac
    from woof.verify.boulac_oracle import load_grid
    inp, _ = load_grid("rural40")
    nz, ncol = inp["nz"], inp["ncol"]
    columns = {key: cp.asarray(inp[source].reshape(nz, 1, ncol))
               for key, source in (("dz", "dz8w"), ("u", "u"), ("v", "v"),
                                   ("th", "th"), ("qv", "qv"), ("qc", "qc"),
                                   ("rho", "rho"))}
    surface = {key: cp.asarray(inp[key].reshape(1, ncol))
               for key in ("hfx", "qfx", "ust")}
    seeds = [cp.asarray(inp["tke"].reshape(nz, 1, ncol)) * factor
             for factor in (cp.float32(1), cp.float32(2))]
    packed = cp.concatenate(seeds, axis=1)
    actual = launch_boulac(
        {key: cp.concatenate([value, value], axis=1) for key, value in columns.items()},
        {key: cp.concatenate([value, value], axis=0) for key, value in surface.items()},
        packed, dt=float(inp["dt"]), cp_air=float(inp["cp"]), g=float(inp["g"]))
    for member, seed in enumerate(seeds):
        expected = launch_boulac(columns, surface, seed, dt=float(inp["dt"]),
                                 cp_air=float(inp["cp"]), g=float(inp["g"]))
        for key, value in expected.items():
            view = actual[key][:, member:member + 1] if value.ndim == 3 else actual[key][member:member + 1]
            assert cp.asnumpy(view).tobytes() == cp.asnumpy(value).tobytes(), (member, key)
        assert np.array_equal(cp.asnumpy(packed[:, member:member + 1]).view(np.uint32),
                              cp.asnumpy(seed).view(np.uint32))


@pytest.mark.gpu
def test_boulac_ensemble_executor_keeps_original_member_results():
    import cupy as cp
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from woof.ensemble.suite_capabilities import plan_suite
    members = [_state(cp), _state(cp)]
    references = [_state(cp), _state(cp)]
    for states in (members, references):
        states[1][0].u *= cp.float32(1.5)
        states[1][0].elapsed_seconds = 10.
    executor = MemberPhysicsExecutor([
        MemberPhysicsBinding(state, cfg, driver, index)
        for index, (state, cfg, driver) in enumerate(members)])
    actual = executor.compute()
    for index, (state, cfg, driver) in enumerate(references):
        expected = driver.compute(state, cfg)
        plan = plan_suite(cfg, members=2)
        assert next(row for row in plan.components if row.component == 'pbl').option_id == 'boulac'
        for name in ('ru', 'rv', 'rtheta', 'rqv', 'rqc'):
            assert cp.asnumpy(getattr(actual[index], name)).tobytes() == cp.asnumpy(getattr(expected, name)).tobytes(), (index, name)
        for name in ('tke_pbl', 'el_pbl', 'exch_h', 'exch_m', 'pblh'):
            assert cp.asnumpy(members[index][2].fields[name]).tobytes() == cp.asnumpy(driver.fields[name]).tobytes(), (index, name)
    assert executor.call_counts['compute'] == [1, 1]
    assert executor.receipt()['clock_policy'] == 'independent_original_member_clocks'
