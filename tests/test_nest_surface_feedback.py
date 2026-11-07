"""Masked feedback keeps native pickups, integers and bounded carrier owners."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_interp import (
    copy_fcni, copy_fcnm, feedback_parent_bounds, feedback_pick_window,
    register_nest)
from woof.verify.npref import np_copy_fcni, np_copy_fcnm


@pytest.mark.gpu
@pytest.mark.parametrize('ratio', [2, 3, 4, 5])
@pytest.mark.parametrize('stagger', ['', 'x', 'y'])
@pytest.mark.parametrize('dtype', [np.float32, np.int32, np.int64])
def test_windowed_masked_feedback_matches_full_native_pick(ratio, stagger, dtype):
    import cupy as cp
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=4,
                        j_parent_start=4, child_nx=6*ratio,
                        child_ny=6*ratio, parent_nx=16, parent_ny=16,
                        stagger=stagger)
    # Multiple leading axes exercise mosaic/soil carriers. Large INTEGER
    # values detect an accidental trip through REAL32.
    pshape = (2, 3, reg.nyp, reg.nxp)
    cshape = (2, 3, reg.nyc, reg.nxc)
    p = np.arange(np.prod(pshape), dtype=dtype).reshape(pshape)
    c = np.arange(np.prod(cshape), dtype=dtype).reshape(cshape)
    if dtype != np.float32:
        p += 2**25
        c += 2**25 + (2**40 if dtype == np.int64 else 0)
    else:
        p *= np.float32(.001)
        c *= np.float32(.007)
    mirror = np_copy_fcnm if dtype == np.float32 else np_copy_fcni
    expected = mirror(p.reshape((-1, *p.shape[-2:])),
                      c.reshape((-1, *c.shape[-2:])),
                      i_parent_start=4, j_parent_start=4, nri=ratio, nrj=ratio,
                      xstag=bool(reg.xstag), ystag=bool(reg.ystag)).astype(dtype).reshape(pshape)
    full, bounded = cp.asarray(p), cp.asarray(p)
    child = cp.asarray(c)
    kernel = copy_fcnm if dtype == np.float32 else copy_fcni
    kernel(full, child, reg)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for j in range(jlo, jhi+1, 2):
        for i in range(ilo, ihi+1, 2):
            win = (slice(j, min(j+2, jhi+1)), slice(i, min(i+2, ihi+1)))
            donor = feedback_pick_window(reg, win)
            packed = cp.ascontiguousarray(child[(...,)+donor])
            result = cp.empty(p.shape[:-2] + (win[0].stop-j, win[1].stop-i), dtype)
            kernel(result, packed, reg, parent_window=win, child_window=donor)
            bounded[(...,)+win] = result
    np.testing.assert_array_equal(cp.asnumpy(full), expected)
    np.testing.assert_array_equal(cp.asnumpy(bounded), expected)


@pytest.mark.gpu
@pytest.mark.parametrize('bounded', [False, True])
def test_coupler_feedback_updates_canonical_soil_and_integer_carriers(bounded):
    import cupy as cp
    from test_nest_coupler import _DeviceState, _nodes
    from test_nest_bounded_operands import _host_facade
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.core.nest_operands import NestWindowSource

    parent, child = _nodes()
    for index, node in enumerate((parent, child)):
        ny, nx = node.state.mup.shape
        fields = {'smois': np.full((4, ny, nx), .2+index*.3, np.float32),
                  'ivgtyp': np.full((ny, nx), 2**25+index+1, np.int32)}
        node.state = _host_facade(node.state) if bounded else _DeviceState(node.state, cp)
        if bounded:
            node.state._streamed_domain.store.update({'fields/'+k: v for k, v in fields.items()})
            slots = {}
            def scratch(shape, slot, dtype=None, slots=slots):
                if slot not in slots:
                    slots[slot] = cp.empty(shape, dtype or np.float32)
                return slots[slot]
            node.state.scratch = scratch
        else:
            node.state.physics = SimpleNamespace(fields={k: cp.asarray(v) for k, v in fields.items()})
    child.clock.ticks = parent.clock.ticks
    coupler = NestCoupler(child, feedback=1, smooth_option=0)
    coupler.feedback_prepare(child, FeedbackScratch())
    coupler.feedback_commit(child)
    reg = coupler.registrations['m']
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg)
    for name, value, original in [('smois', .5, .2), ('ivgtyp', 2**25+2, 2**25+1)]:
        actual = cp.asnumpy(NestWindowSource(parent.state).array('fields/'+name))
        expected = np.full_like(actual, original)
        expected[..., jlo:jhi+1, ilo:ihi+1] = value
        np.testing.assert_array_equal(actual, expected)
        assert name in coupler.surface_feedback_fields
    assert coupler.transition_receipt()['surface_feedback_fields'] == list(coupler.surface_feedback_fields)


@pytest.mark.gpu
@pytest.mark.parametrize('streamed_endpoint', ['parent', 'child', 'both'])
@pytest.mark.parametrize('smooth_option', [0, 1, 2])
def test_strict_base_feedback_matches_resident_for_canonical_endpoints(
        monkeypatch, streamed_endpoint, smooth_option):
    import cupy as cp
    from woof import wrf_exact
    from woof.core.state import DomainState
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.core.nest_operands import NestWindowSource
    from test_nest_coupler import _DeviceState, _nodes
    from test_nest_bounded_operands import _host_facade

    monkeypatch.setattr(wrf_exact, 'ENABLED', True)
    outputs = []
    for bounded in (False, True):
        parent, child = _nodes()
        rng = np.random.default_rng(3179)
        for index, node in enumerate((parent, child)):
            host = node.state
            nz, ny, nx = host.thp.shape
            host.phb = (np.arange(nz+1, dtype=np.float32)[:, None, None]*30000
                        + rng.random((nz+1, ny, nx), dtype=np.float32)*30)
            for name in ('pb', 'alb', 'al', 'p', 'alt', 'p_perturbation', 'dphb_resid'):
                setattr(host, name, rng.random((nz, ny, nx), dtype=np.float32))
            streams = bounded and (streamed_endpoint == 'both' or
                                   streamed_endpoint == ('parent' if index == 0 else 'child'))
            node.state = _host_facade(host) if streams else _DeviceState(host, cp)
            if streams:
                # Strict lazily allocated pressure belongs to the canonical store.
                owner = node.state._streamed_domain
                assert 'state/p_perturbation' in owner.store
                assert 'setup/p_perturbation' not in owner._geography
                slots = {}
                def scratch(shape, slot, dtype=None, slots=slots):
                    if slot not in slots:
                        slots[slot] = cp.empty(shape, dtype or np.float32)
                    return slots[slot]
                node.state.scratch = scratch
            else:
                node.state.set_base_geopotential = lambda phb, state=node.state: DomainState.set_base_geopotential(state, phb)
        child.clock.ticks = parent.clock.ticks
        coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
        coupler.feedback_prepare(child, FeedbackScratch())
        coupler.feedback_commit(child)
        source = NestWindowSource(parent.state)
        outputs.append({name: cp.asnumpy(source.array(name)) for name in
                        ('mub2d', 'pb', 'alb', 'phb', 'dphb_resid', 'p', 'alt')})
        assert all(isinstance(value, np.ndarray) for value in parent.state._setup_feedback_base.values())
        assert parent.state._feedback_base_children == {'2': [4, 4]}
        before = {key: value.copy() for key, value in outputs[-1].items()}
        coupler.feedback_prepare(child, FeedbackScratch())
        coupler.feedback_commit(child)
        for name, expected in before.items():
            np.testing.assert_array_equal(cp.asnumpy(source.array(name)), expected)
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name].view(np.uint32),
                                      outputs[1][name].view(np.uint32), err_msg=name)
