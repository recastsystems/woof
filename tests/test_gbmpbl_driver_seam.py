"""The GBM moist TKE PBL (bl_pbl_physics=12) through the physics driver.

On the card: ``initialize_physics`` allocates the scheme's fields at
gbmpblinit's cold start, the dycore's step calls ``_run_gbmpbl``, the
tendencies are coupled, and a restart resumes on the carried TKE_PBL bit
for bit.  Split from tests/test_gbmpbl_door.py because the helpers here
import cupy, which marks the whole module ``gpu`` (tests/conftest.py).
"""

from __future__ import annotations

import pytest

from woof.config import GBM_PBL_SCHEME
from woof.core.physics_inventory import GBMPBL_MASS_OUTPUTS


def _gbm_state(cp, **overrides):
    """tests/test_restart.py's small full-physics state under GBM.

    Morrison (carries ice), analytic radiation every minute, KF, the
    revised MM5 surface layer and Noah; a sheared wind and a warm skin so
    the column is turbulent at the checkpoint rather than quiescent.
    """
    import numpy as np

    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, bl_pbl_physics=GBM_PBL_SCHEME,
                                        **overrides)
    profile = np.linspace(2.0, 14.0, cfg.nz, dtype=np.float32)
    state.u[...] = cp.asarray(profile)[:, None, None]
    # An unstable lower layer, capped above it, exercises carried TKE.
    theta = np.concatenate((np.linspace(304.0, 298.0, 4),
                            np.linspace(301.0, 330.0, cfg.nz - 4))).astype(np.float32)
    state.thp[...] += cp.asarray(theta)[:, None, None] - state.total_theta()
    driver.fields["tsk"][...] = cp.float32(306.0)
    driver.fields["tslb"][...] = cp.float32(303.0)
    return state, cfg, driver


@pytest.mark.gpu
def test_the_driver_runs_the_scheme_and_couples_it():
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps

    state, cfg, driver = _gbm_state(cp)
    for name in ("tke_pbl", "el_pbl", "exch_tke"):
        assert driver.fields[name].shape == (cfg.nz, cfg.ny, cfg.nx)
    assert float(driver.fields["tke_pbl"][3, 1, 1]) == np.float32(0.001)
    run_steps(state, cfg, 10)
    assert driver.call_counts["ysu"] == 10        # the PBL-slot counter
    for name in ("tke_pbl", "el_pbl", "exch_tke", "exch_h", "exch_m",
                 "pblh", "ust"):
        assert np.all(np.isfinite(cp.asnumpy(driver.fields[name]))), name
    # TKE left the cold start, the diffusivities are live, KPBL is a level
    tke = cp.asnumpy(driver.fields["tke_pbl"])
    assert tke.max() > 0.01
    assert float(cp.asnumpy(driver.fields["exch_h"]).max()) > 0.0
    kpbl = cp.asnumpy(driver.fields["kpbl"])
    assert kpbl.min() >= 1 and kpbl.max() <= cfg.nz
    from woof.io.history_layout import live_state_history_fields
    assert live_state_history_fields(state)["REGIME"] is driver.fields["regime"]
    assert np.all(np.isfinite(cp.asnumpy(state.thp)))


@pytest.mark.gpu
@pytest.mark.parametrize("bldt", [0.0, 0.5])
def test_a_gbm_run_restarts_bit_identically(tmp_path, bldt):
    """20 steps + checkpoint + 20 steps == 40 steps, bit for bit.

    TKE_PBL is the scheme's prognostic state: read at the top of every
    call, rewritten with the first GBM2D pass's TKE.  The 20-step boundary
    is off the radiation calendar, so the held RTHRATEN crosses it as
    stored state too.
    """
    import cupy as cp
    import numpy as np

    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _assert_restart_equal

    state_a, cfg_a, _ = _gbm_state(cp, bldt=bldt)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _gbm_state(cp, bldt=bldt)
    run_steps(state_b, cfg_b, 20)
    assert driver_b.call_counts["ysu"] == (20 if bldt == 0.0 else 7)
    assert float(cp.abs(driver_b.fields["tke_pbl"]).max()) > 0.01
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _gbm_state(cp, bldt=bldt)
    restart.restore_restart(mid, state_c, cfg_c)
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)
    _assert_restart_equal(resumed, reference)
    # Restoration itself preserves every serialized array word, before advancing.
    state_d, cfg_d, _ = _gbm_state(cp, bldt=bldt)
    restart.restore_restart(mid, state_d, cfg_d)
    roundtrip = restart.write_restart(tmp_path / "roundtrip.npz", state_d, cfg_d)
    _assert_restart_equal(roundtrip, mid)

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for name in ("tke_pbl", "el_pbl", "exch_tke", "exch_h", "exch_m"):
        assert f"fields/{name}" in keys, name
    assert np.isfinite(cp.asnumpy(state_c.thp)).all()


@pytest.mark.gpu
def test_the_unit_compiles_to_the_frame_it_is_priced_at():
    import cupy as cp

    from woof.core import gbmpbl
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    fn = gbmpbl._module(cp.cuda.Device().id).get_function("gbmpbl_column")
    local = int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn.ptr))
    priced = CHAINED_TRANSLATION_UNIT_FRAMES["gbmpbl_composed"]
    assert local <= priced.max_local_size_bytes, local


@pytest.mark.gpu
@pytest.mark.parametrize("name", (*GBMPBL_MASS_OUTPUTS, "pblh", "ust"))
def test_nonfinite_output_is_recorded_without_a_step_host_read(monkeypatch, name):
    import cupy as cp
    from woof.core import gbmpbl, health_ledger

    outputs = {name: cp.ones((2, 2, 2), dtype=cp.float32)
               for name in gbmpbl.GBMPBL_COLUMN_OUTPUTS}
    outputs.update(pblh=cp.ones((2, 2), dtype=cp.float32),
                   ust=cp.ones((2, 2), dtype=cp.float32))
    # Last element catches both the 3-D scan and the smaller surface tail.
    outputs[name].reshape(-1)[-1] = cp.nan
    ledger = health_ledger.HealthLedger()
    def forbid_read(*args, **kwargs):
        raise AssertionError("GBM output validation read the device before the drain")
    with health_ledger.deferring(ledger):
        with monkeypatch.context() as patch:
            patch.setattr(cp, "asnumpy", forbid_read)
            assert gbmpbl.invalid_gbmpbl_output(outputs) is None
            assert ledger.records == 1
            assert ledger.drains == 0
    with pytest.raises(FloatingPointError, match=f"GBM PBL returned non-finite {name}"):
        ledger.drain()
    assert ledger.drains == 1


@pytest.mark.gpu
def test_preallocated_output_carriers_keep_every_output_word():
    import cupy as cp
    import numpy as np
    from woof.core import gbmpbl
    from woof.verify.gbmpbl_oracle import ORACLE_DIR, read_cases

    case = read_cases(ORACLE_DIR / "gbm-cases.bin")[0]
    shape = (case["nz"], 1, case["ncol"])
    def dev(name):
        return cp.asarray(np.ascontiguousarray(case[name].reshape(shape)))
    columns = {name: dev(name) for name in
               ("u", "v", "qv", "qc", "qi", "p", "dz8w", "rthraten")}
    columns.update(t=dev("th") * dev("pi"), exner=dev("pi"))
    surface = {name: cp.asarray(case[name].reshape(shape[1:]))
               for name in gbmpbl.GBMPBL_SURFACE_INPUTS}
    first_tke, second_tke = dev("tke"), dev("tke")
    reference = gbmpbl.gbmpbl_step(columns, surface, first_tke,
                                   dt=case["dt"], flqi=True)
    names = (*gbmpbl.GBMPBL_COLUMN_OUTPUTS, *gbmpbl.GBMPBL_SURFACE_OUTPUTS)
    carriers = {name: cp.empty_like(reference[name]) for name in names}
    actual = gbmpbl.gbmpbl_step(columns, surface, second_tke,
                                dt=case["dt"], flqi=True, out=carriers)
    for name in names:
        assert actual[name] is carriers[name]
        cp.testing.assert_array_equal(reference[name].view(cp.uint32),
                                      actual[name].view(cp.uint32))
    cp.testing.assert_array_equal(first_tke.view(cp.uint32), second_tke.view(cp.uint32))


@pytest.mark.gpu
def test_gbm_member_execution_preserves_each_independent_member():
    from test_ensemble_physics_execution_gpu import (
        test_member_local_real_driver_and_microphysics_match_each_independent_call)
    test_member_local_real_driver_and_microphysics_match_each_independent_call(
        members=2, mp=6, pbl=12, sfclay=1, cumulus=0)


@pytest.mark.gpu
@pytest.mark.parametrize('feedback', [0, 1])
def test_gbm_advances_both_domains_with_local_carried_tke(feedback):
    import cupy as cp
    import numpy as np
    from dataclasses import replace
    from types import MappingProxyType, SimpleNamespace
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.model import (DomainNode, ExperimentState, FeedbackScratch,
                                  ModelRuntimeStatus, execute_experiment)
    from woof.core.nest import NestCoupler
    from test_feedback import _feedback_experiment
    from test_restart import _physics_state

    exp = _feedback_experiment(6.0, feedback=feedback)
    configs, states = [], []
    for dc in exp.domains:
        child = dc.parent_id != 0
        state, cfg, driver = _physics_state(cp, nx=dc.run.nx, ny=dc.run.ny,
            dx=1000.0 if child else 3000.0, dy=1000.0 if child else 3000.0,
            dt=1.0 if child else 3.0, run_seconds=6.0, grid_id=dc.grid_id,
            nested=child, specified=not child, bl_pbl_physics=12,
            mp_physics=6, cu_physics=0)
        # Different starting turbulence catches accidental cross-domain aliasing.
        driver.fields['tke_pbl'][...] = cp.float32(0.002 if child else 0.003)
        state._nest_restart_classification = 'REBUILT'
        states.append(state)
        configs.append(replace(dc, run=cfg, history_interval_s=6.0,
                               time_step=None if child else 3))
    exp = replace(exp, domains=tuple(configs), restart_interval_s=0.0,
                  vertical=replace(exp.vertical, eta_levels=()))
    clock = resolve_clock(exp); clocks = clock.clocks()
    root = DomainNode(cfg=configs[0], grid=SimpleNamespace(), state=states[0],
                      clock=clocks[1], parent=None, children=[], coupler=None)
    child = DomainNode(cfg=configs[1], grid=SimpleNamespace(), state=states[1],
                       clock=clocks[2], parent=root, children=[], coupler=None)
    root.children.append(child)
    child.coupler = NestCoupler(child, feedback=feedback)
    # Strict force arithmetic couples a specified parent and its nest, as
    # the stock real.exe route does. Supply the parent's actual boundary
    # tables and clock instead of disabling that production requirement.
    from woof.ingest.lateral_bc import (attach_lateral_boundaries,
        bind_lateral_boundary_clock, build_lateral_boundaries,
        domain_boundary_snapshot)
    first = domain_boundary_snapshot(root.state)
    later = {name: value.copy() for name, value in first.items()}
    later['qv'] *= np.float32(1.001)
    boundaries = build_lateral_boundaries([first, later], [0.0, 6.0],
        spec_bdy_width=root.cfg.run.spec_bdy_width,
        spec_zone=root.cfg.run.spec_zone, relax_zone=root.cfg.run.relax_zone)
    attach_lateral_boundaries(root.state, boundaries)
    bind_lateral_boundary_clock(root.state, root.clock)
    if feedback:
        child.coupler.feedback_prepare(child, FeedbackScratch())
        child.coupler.feedback_commit(child)
        child.coupler.feedback_finalize(child)
    model = ExperimentState(root=root, nodes_by_grid_id=MappingProxyType({1:root,2:child}),
        schedule=build_schedule(exp, clock), memory_ledger=None,
        experiment_fingerprint='gbm-nested-runtime-v1')
    model._scratch_arena = None; model._dycore_state_workspace = None
    model._runtime_status = ModelRuntimeStatus(); model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None; model._last_checkpoint = None
    execute_experiment(model, validate_state=False, pool_trim_per_period=False)
    assert root.state.physics.call_counts['ysu'] == 2
    assert child.state.physics.call_counts['ysu'] == 6
    assert root.state.physics.fields['tke_pbl'].data.ptr != child.state.physics.fields['tke_pbl'].data.ptr
    for node in (root, child):
        assert np.isfinite(cp.asnumpy(node.state.physics.fields['tke_pbl'])).all()
        assert np.isfinite(cp.asnumpy(node.state.thp)).all()
    # The initial restriction plus the two completed parent periods.
    assert child.coupler.feedback_count == (3 if feedback else 0)
