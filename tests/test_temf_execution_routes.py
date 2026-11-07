"""TEMF reaches the shared forecast door and both nest feedback modes."""
from dataclasses import replace

import pytest


def test_go_wrfinput_preserves_shared_forecast_arguments(monkeypatch, tmp_path):
    from woof import cli, capabilities, provenance_gate
    seen = []
    monkeypatch.setattr(provenance_gate, 'announce', lambda *args: None)
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *args: None)
    monkeypatch.setattr(cli, '_dispatch', lambda args: seen.append(args) or 0)
    assert cli.main(['go', '--wrfinput', str(tmp_path), '--outdir', str(tmp_path/'out'),
                     '--run-seconds', '10800', '--benchmark', '--checkpoint-at', '3600']) == 0
    args, = seen
    assert args.command == 'go' and args.wrfinput == tmp_path
    assert args.run_seconds == 10800 and args.benchmark
    assert args.checkpoint_at is not None
    with pytest.raises(SystemExit):
        cli.main(['go', '--wrfinput', str(tmp_path), '--unknown-temf-flag'])


@pytest.mark.gpu
@pytest.mark.parametrize('feedback', [0, 1])
def test_temf_nested_execution_keeps_domain_local_carriers(feedback):
    import cupy as cp
    from test_physics_driver import _full_state
    from test_da_nested_forecast_gpu import _experiment
    from woof.experiment import DomainConfig
    from woof.core.nest import NestCoupler
    from woof.core.model import execute_experiment, FeedbackScratch
    from woof.core.physics import initialize_physics
    from woof.core.physics_inventory import TEMF_PBL_STATE_3D, TEMF_PBL_STATE_2D
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree

    state, cfg, driver = _full_state(nx=33, ny=33, nz=24, dt=6.,
        dx=3000., dy=3000., ztop=15000., run_seconds=12.,
        bl_pbl_physics=10, sf_sfclay_physics=10, sf_surface_physics=0,
        ra_physics=0, mp_physics=0, cu_physics=0, bldt=0., km_opt=4,
        specified=True, spec_bdy_width=5, spec_zone=1, relax_zone=4)
    from woof.ingest.lateral_bc import build_state_lateral_boundaries, attach_lateral_boundaries
    attach_lateral_boundaries(state, build_state_lateral_boundaries(
        [state, state], [0., 3600.], spec_bdy_width=5, spec_zone=1, relax_zone=4))
    exp = _experiment(cfg)
    root = replace(exp.root, time_step=6, history_interval_s=12.)
    child_cfg = replace(cfg, nx=27, ny=27, dx=1000., dy=1000., dt=2.,
                        nested=True, grid_id=2, specified=False)
    child = DomainConfig(grid_id=2, parent_id=1, i_parent_start=12,
        j_parent_start=12, parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=12., run=child_cfg)
    exp = replace(exp, run_seconds=12., domains=(root, child), feedback=feedback)
    def prepare(s, dc, grid, start):
        if getattr(s, 'physics', None) is None:
            initialize_physics(s, dc.run, landmask=1., tsk=299., glw=310., swdown=450.)
    model = assemble_idealized_tree(exp, state, domain_preparer=prepare,
        coupler_factory=lambda node: NestCoupler(node, feedback=feedback))
    if feedback:
        node = model.node(2)
        node.coupler.feedback_prepare(node, FeedbackScratch())
        node.coupler.feedback_commit(node)
        node.coupler.feedback_finalize(node)
    execute_experiment(model, experiment=exp)
    a, b = model.node(1), model.node(2)
    assert a.clock.step_count == 2 and b.clock.step_count == 6
    for node, count in ((a, 2), (b, 6)):
        assert node.state.physics.call_counts['ysu'] == count
        for name in TEMF_PBL_STATE_3D + TEMF_PBL_STATE_2D:
            value = node.state.physics.fields[name]
            assert bool(cp.isfinite(value).all()), name
            assert value.data.ptr != (b if node is a else a).state.physics.fields[name].data.ptr
