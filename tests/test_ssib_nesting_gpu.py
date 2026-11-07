"""Compare bounded SSiB feedback with the existing native-oracled operators."""
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.mark.gpu
@pytest.mark.parametrize('ratio', [3,4])
@pytest.mark.parametrize('canonical', [False,True])
def test_ssib_feedback_matches_wrf_operators_and_keeps_private_state(ratio, canonical):
    import cupy as cp
    from woof.core.nest_interp import register_nest, copy_fcn, copy_fcnm
    from woof.core.ssib_nesting import PICK_FIELDS, MEAN_FIELDS, feedback
    reg=register_nest(nri=ratio,nrj=ratio,i_parent_start=4,j_parent_start=4,
                      child_nx=32,child_ny=32,parent_nx=20,parent_ny=20,
                      wrapper='bdy')
    pf,cf,expected={},{},{}
    for n,name in enumerate(PICK_FIELDS+MEAN_FIELDS+('ssib_ect',)):
        shape=(3,20,20) if name in ('smois','tslb') else (20,20)
        cshape=(*shape[:-2],32,32)
        pf[name]=cp.full(shape,-n-10,cp.float32)
        cf[name]=cp.arange(np.prod(cshape),dtype=cp.float32).reshape(cshape)/37+n
        expected[name]=pf[name].copy()
        if name!='ssib_ect':
            (copy_fcnm if name in PICK_FIELDS else copy_fcn)(
                expected[name],cf[name],reg,spec_zone=1)
    parent=SimpleNamespace(physics=SimpleNamespace(fields=pf))
    child=SimpleNamespace(physics=SimpleNamespace(fields=cf))
    if canonical:
        parent._streamed_domain=SimpleNamespace(
            store={f'fields/{n}':a.get() for n,a in pf.items()})
        child._streamed_domain=SimpleNamespace(
            store={f'fields/{n}':a.get() for n,a in cf.items()})
    feedback(parent,child,reg,spec_zone=1,chunk_shape=(9,11))
    actual=({n:a for n,a in pf.items()} if not canonical else
            {n:parent._streamed_domain.store[f'fields/{n}'] for n in pf})
    for n,a in actual.items():
        np.testing.assert_array_equal(a.get() if hasattr(a,'get') else a,
                                      expected[n].get(),err_msg=n)


def test_disabled_feedback_never_reads_surface_fields():
    from woof.core.nest import NestCoupler
    child=SimpleNamespace()
    coupler=object.__new__(NestCoupler)
    coupler.child_node=child
    coupler.feedback=0
    coupler.feedback_commit(child)


@pytest.mark.gpu
@pytest.mark.parametrize('feedback_mode', [0, 1])
def test_production_coupler_commits_native_surface_feedback(feedback_mode):
    import cupy as cp
    from dataclasses import replace
    from test_nest_coupler import _DeviceState, _nodes
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.core.ssib_nesting import feedback, PICK_FIELDS, MEAN_FIELDS
    parent, child = _nodes()
    for node in (parent, child):
        node.cfg.run = replace(node.cfg.run, sf_surface_physics=8)
        node.state = _DeviceState(node.state, cp)
        # This flat-terrain fixture has a shared vertical base profile.
        # Strict mode checks its dimensionality before terrain feedback.
        node.state.phb = cp.arange(3, dtype=cp.float32)
        fields = {}
        for k, name in enumerate(PICK_FIELDS+MEAN_FIELDS+('ssib_ect',)):
            shape = (node.cfg.run.ny, node.cfg.run.nx)
            if name in ('smois', 'tslb'):
                shape = (3, *shape)
            fields[name] = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)+k
        node.state.physics = SimpleNamespace(fields=fields)
    child.clock.ticks = parent.clock.ticks
    coupler = NestCoupler(child, feedback=feedback_mode, smooth_option=0)
    expected = {n:a.copy() for n,a in parent.state.physics.fields.items()}
    if feedback_mode:
        target = SimpleNamespace(physics=SimpleNamespace(fields=expected))
        feedback(target, child.state, coupler.registrations['m'], spec_zone=1)
    coupler.feedback_prepare(child, FeedbackScratch())
    coupler.feedback_commit(child)
    for n, want in expected.items():
        np.testing.assert_array_equal(parent.state.physics.fields[n].get(),
                                      want.get(), err_msg=n)
