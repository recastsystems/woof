from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu

@requires_gpu
@pytest.mark.gpu
def test_mp13_two_member_acoustic_mass_loading_words():
    import cupy as cp
    from woof.core.acoustic import prepare_moist_cq as scalar_cq
    from woof.ensemble.batch_acoustic import prepare_moist_cq as batch_cq
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from test_ensemble_batch_nesting_gpu import _same
    state, references, cfg, _ = _pack_physical(2, moist=True, mp_physics=13,
        mapped=True, nz=12, ny=7, nx=11)
    for member, reference in enumerate(references):
        for name, value in (('qi', .0001), ('qs', .0002)):
            getattr(reference, name).fill(np.float32(value * (member + 1)))
            state.member_view(name, member)[:] = getattr(reference, name)
    observed = batch_cq(state, cfg)
    assert observed[-1]
    for member, reference in enumerate(references):
        expected = scalar_cq(reference, cfg)
        for name, value in zip(('acoustic_cqu', 'acoustic_cqv', 'acoustic_cqw'), expected[:3], strict=True):
            _same(cp, state.scratch_member_view(name, member), value)

@requires_gpu
@pytest.mark.gpu
def test_mp13_ensemble_original_executor_matches_independent_members():
    import cupy as cp
    from woof.core.microphysics import apply
    from woof.ensemble.batch_dycore import member_domain_view
    from woof.ensemble.physics_execution import MemberPhysicsBinding, MemberPhysicsExecutor
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from test_ensemble_batch_nesting_gpu import _same
    batch, references, cfg, _ = _pack_physical(2, moist=True, mp_physics=13,
        mapped=True, nz=12, ny=7, nx=11)
    states = [member_domain_view(batch, member) for member in range(2)]
    for member, (state, reference) in enumerate(zip(states, references, strict=True)):
        for name, value in (('qc', .0001), ('qi', .00005), ('qs', .0002)):
            for target in (state, reference):
                getattr(target, name).fill(np.float32(value * (member + 1)))
    executor = MemberPhysicsExecutor([MemberPhysicsBinding(s, cfg, None, m)
                                     for m, s in enumerate(states)])
    results = executor.apply_microphysics()
    for state, reference, observed in zip(states, references, results, strict=True):
        expected = apply(reference, cfg, cfg.dt)
        for name in ('qv', 'qc', 'qr', 'qi', 'qs', 'rimi', 'thp', 'ru_tend', 'rv_tend'):
            if getattr(state, name, None) is not None:
                _same(cp, getattr(state, name), getattr(reference, name))
        _same(cp, observed.rainnc, expected.rainnc)
        _same(cp, observed.rainncv, expected.rainncv)
    assert executor.call_counts['microphysics'] == [1, 1]

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('feedback', [0, 1])
def test_mp13_two_member_nest_force_feedback_words(feedback):
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    from test_ensemble_batch_nesting_gpu import _fixture, _same
    parent, child, nodes, edge = _fixture(2, ratio=3, mapped=True,
        smooth=0, mp_physics=13, feedback=feedback, mu_halo=True)
    # Nonzero member-specific diagnostic values catch accidental transport.
    for member, (pnode, cnode, _) in enumerate(nodes):
        for batch, node in ((parent, pnode), (child, cnode)):
            node.state.rimi.fill(np.float32(.25 + member * .125))
            batch.member_view('rimi', member)[:] = node.state.rimi
    pc = tuple(p.clock for p, _, _ in nodes)
    cc = tuple(c.clock for _, c, _ in nodes)
    tables = edge.force(parent_clocks=pc, child_clocks=cc)
    assert 'rimi' not in tables
    assert {'qi', 'qs'} <= tables.keys()
    for member, (_, cnode, _) in enumerate(nodes):
        cnode.coupler.force(cnode)
        for field, sides in tables.items():
            for side, values in sides.items():
                for actual, expected in zip(values, cnode.coupler._last_tables[field][side], strict=True):
                    _same(cp, actual[member], expected)
    for p, c in zip(pc, cc, strict=True):
        c.ticks = p.ticks
    edge.feedback_prepare(parent_clocks=pc, child_clocks=cc)
    edge.feedback_commit()
    edge.feedback_finalize()
    for _, cnode, scratch in nodes:
        cnode.coupler.feedback_prepare(cnode, scratch)
        cnode.coupler.feedback_commit(cnode)
        cnode.coupler.feedback_finalize(cnode)
    for member, (pnode, cnode, _) in enumerate(nodes):
        for batch, node in ((parent, pnode), (child, cnode)):
            for name in state_array_shapes(batch.cfg):
                _same(cp, batch.member_view(name, member), getattr(node.state, name))
            assert bool(cp.all(node.state.rimi == np.float32(.25 + member * .125)))

def test_offline_parent_reads_every_mp13_transported_field():
    from woof.offline_child import _transported_source_fields, _scheme_wrf_to_state
    assert _transported_source_fields(13) == ('qv','qc','qr','qi','qs')
    assert _scheme_wrf_to_state(13)['RIMI'] == 'rimi'

def test_mp13_edges_retire_the_missing_closure_guard_and_append_field_code():
    from woof.core import microphysics_transition as mt
    assert 13 in mt.PORTED_MP_PHYSICS
    assert 13 not in mt.UNVALIDATED_MIXED_EDGE_SELECTORS
    assert mt._EDGE_FIELD_CODES['rimi'] == 27
    assert mt._EDGE_FIELD_CODES['nifa'] == 26
    cfg=lambda mp: SimpleNamespace(mp_physics=mp, moist=True, moist_cq=True, nest_microphysics_transition='same-scheme-only')
    for mp in mt.PORTED_MP_PHYSICS:
        if mp == 13: continue
        entry=mt.resolve_microphysics_transition(cfg(mp),cfg(13))
        exit_=mt.resolve_microphysics_transition(cfg(13),cfg(mp))
        assert entry.mixed and exit_.mixed
        assert any(r['target_field']=='rimi' and r['reason']=='wrf_sbu_ylin_cloud_snow_riming_diagnosis' for r in entry.species_actions())

@requires_gpu
@pytest.mark.gpu
def test_mp13_edge_riming_matches_every_wrf_oracle_word_and_couples_once():
    import cupy as cp
    from woof.core import microphysics_transition as mt
    from tools.sbu_ylin_wrf461_oracle.validate_sbu_ylin_oracle import read_cases, read_outputs
    base=Path(__file__).resolve().parents[1]/'woof/data/sbu_ylin/oracle'
    cases=read_cases(base/'sbu-ylin-cases.txt')
    # The actual state carries inverse density. The WRF reference uses the
    # density represented by that float32 value, with zero tolerance.
    lines=(base/'sbu-ylin-edge-rimi.txt').read_text().splitlines()
    assert tuple(map(int,lines[0].split())) == (cases['ncol'],cases['nz'])
    expected=np.array([[int(word,16) for word in line.split()] for line in lines[1:]],np.uint32).view(np.float32).T[:,None,:]
    def volume(field):
        return cp.asarray(np.ascontiguousarray(cases['levels'][:,:,field].T[:,None,:]))
    shape=expected.shape
    state=SimpleNamespace(qv=volume(1),qc=volume(2),qr=volume(3),qi=volume(4),qs=volume(5),qg=cp.zeros(shape,np.float32),
        alt=cp.reciprocal(volume(6)),mub2d=cp.full(shape[1:],3,np.float32),mup=cp.full(shape[1:],2,np.float32),
        c1h=cp.ones(shape[0],np.float32),c2h=cp.ones(shape[0],np.float32))
    cfg=lambda mp: SimpleNamespace(mp_physics=mp,moist=True,moist_cq=True,nest_microphysics_transition='same-scheme-only')
    contract=mt.resolve_microphysics_transition(cfg(6),cfg(13))
    out=cp.empty(shape,np.float32)
    mt.launch_microphysics_edge_field(contract,state,'rimi',out=out,coupled=False)
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),expected.copy().view(np.uint32))
    physical=out.copy()
    mt.launch_microphysics_edge_field(contract,state,'rimi',out=out,coupled=True)
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),cp.asnumpy(physical*np.float32(6)).view(np.uint32))
    # A warm-only source has no snow. Its placeholder must not be interpreted as snow.
    del state.qs
    contract=mt.resolve_microphysics_transition(cfg(1),cfg(13))
    mt.launch_microphysics_edge_field(contract,state,'rimi',out=out,coupled=False)
    assert cp.count_nonzero(out).item() == 0
    # P3 has no QS. Diagnose the target snow from its rime pair first.
    state.qi=cp.full(shape,0.002,np.float32)
    state.qir=cp.full(shape,0.001,np.float32)
    state.qib=cp.full(shape,0.001/200,np.float32)
    state.nr=cp.zeros(shape,np.float32)
    state.ni=cp.zeros(shape,np.float32)
    contract=mt.resolve_microphysics_transition(cfg(50),cfg(13))
    snow=cp.empty_like(out)
    mt.launch_microphysics_edge_field(contract,state,'qs',out=snow,coupled=False)
    mt.launch_microphysics_edge_field(contract,state,'rimi',out=out,coupled=False)
    from_p3=out.copy()
    state.qs=snow
    contract=mt.resolve_microphysics_transition(cfg(6),cfg(13))
    mt.launch_microphysics_edge_field(contract,state,'rimi',out=out,coupled=False)
    np.testing.assert_array_equal(cp.asnumpy(from_p3).view(np.uint32),cp.asnumpy(out).view(np.uint32))


def test_go_native_input_uses_the_complete_forecast_owner(monkeypatch, tmp_path):
    from woof.cli import build_parser
    from woof.go_cli import go_main, GoRefusal
    import woof.wrfinput_forecast as wf
    import woof.wrfinput_door as door
    from datetime import datetime
    monkeypatch.setattr(door, 'resolve_wrfinput_run', lambda _: SimpleNamespace(
        experiment=SimpleNamespace(start_time=datetime(2026, 10, 16))))
    seen = []
    monkeypatch.setattr(wf, 'run_wrf_forecast', lambda *a, **kw: seen.append((a, kw)) or 0)
    parser = build_parser(go_only=True)
    args = parser.parse_args(['go', '--wrfinput', str(tmp_path), '--outdir', str(tmp_path/'out'),
                             '--run-seconds', '10800', '--products', 't2', '--allow-shared-gpu',
                             '--run-stamp', 'off'])
    assert go_main(args) == 0
    assert seen[0][0] == (tmp_path, tmp_path/'out')
    assert seen[0][1] == dict(run_seconds=10800., render_products='t2',
                            exclusive_gpu=True, allow_shared_gpu=True, restart=None, benchmark=False)
    args.source = 'gfs'
    with pytest.raises(GoRefusal, match='source'):
        go_main(args)


@requires_gpu
@pytest.mark.gpu
def test_mp13_terrain_member_force_matches_scalar_with_acoustic_halo():
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    from test_ensemble_batch_nesting_gpu import _fixture, _same
    parent, child, nodes, edge = _fixture(2, ratio=3, mapped=True,
        smooth=0, mp_physics=13, feedback=0, terrain=True, mu_halo=True)
    tables = edge.force(parent_clocks=tuple(p.clock for p, _, _ in nodes),
                        child_clocks=tuple(c.clock for _, c, _ in nodes))
    for member, (pnode, cnode, _) in enumerate(nodes):
        cnode.coupler.force(cnode)
        for field, sides in tables.items():
            for side, values in sides.items():
                for actual, expected in zip(values, cnode.coupler._last_tables[field][side], strict=True):
                    _same(cp, actual[member], expected)
        for bank, reference in ((parent, pnode.state), (child, cnode.state)):
            for name in state_array_shapes(bank.cfg):
                _same(cp, bank.member_view(name, member), getattr(reference, name))
