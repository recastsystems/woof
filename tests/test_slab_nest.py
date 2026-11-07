"""Slab feedback preserves WRF's masked pickup and unmapped parent cells."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


@pytest.mark.parametrize("feedback,scheme,expected", [(0, 1, 0), (1, 0, 0), (1, 1, 1)])
def test_coupler_restricts_surface_only_for_two_way_slab(monkeypatch, feedback, scheme, expected):
    from dataclasses import replace
    from test_nest_coupler import _nodes
    from woof.core import nest, slab_nest
    import woof.wrf_exact as exact

    parent, child = _nodes()
    for node in (parent, child):
        node.cfg.run = replace(node.cfg.run, sf_surface_physics=scheme,
                               num_soil_layers=5 if scheme else 4)
    coupler = nest.NestCoupler(child, feedback=feedback)
    coupler._prepared_feedback = {"kinds": ("mu",), "ticks": child.clock.ticks}
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    monkeypatch.setattr(nest, "copy_fcn", lambda *a, **k: None)
    monkeypatch.setattr(exact, "ENABLED", False)
    calls = []
    monkeypatch.setattr(slab_nest, "feedback_slab_surface",
                        lambda *args, **kw: calls.append((args, kw)))
    coupler.feedback_commit(child)
    assert len(calls) == expected
    if expected:
        args, kwargs = calls[0]
        assert args[:2] == (parent.state, child.state)
        assert args[2] is coupler.registrations["m"]
        assert kwargs == {"spec_zone": 1, "chunk_shape": None}


@requires_gpu
@pytest.mark.parametrize("ratio", [2, 3, 5])
@pytest.mark.parametrize("stored", [False, True])
def test_surface_feedback_matches_native_primitives(ratio, stored):
    import cupy as cp
    from woof.core.nest_interp import register_nest, copy_fcn, copy_fcnm
    from woof.core.slab_nest import feedback_slab_surface

    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=4,
                        j_parent_start=4, child_nx=ratio * 6,
                        child_ny=ratio * 6, parent_nx=16, parent_ny=16)
    parents, children, expected = {}, {}, {}
    for n, name in enumerate(("tsk", "tslb", "achfx", "aclhf")):
        prefix = (5,) if name == "tslb" else ()
        children[name] = cp.arange(np.prod(prefix + (reg.nyc, reg.nxc)),
                                   dtype=cp.float32).reshape(prefix + (reg.nyc, reg.nxc)) + n
        parents[name] = cp.full(prefix + (reg.nyp, reg.nxp), -999., cp.float32)
        expected[name] = parents[name].copy()
        primitive = copy_fcnm if name in ("tsk", "tslb") else copy_fcn
        primitive(expected[name], children[name], reg, spec_zone=1)

    def state(fields):
        if not stored:
            return SimpleNamespace(physics=SimpleNamespace(fields=fields))
        # The resident template is deliberately poisoned. Only the
        # canonical store's fields may be used after a tile sweep.
        return SimpleNamespace(
            physics=SimpleNamespace(fields={k: cp.full_like(v, -777.) for k, v in fields.items()}),
            _streamed_domain=SimpleNamespace(
                store={f"fields/{k}": cp.asnumpy(v) for k, v in fields.items()},
                template_state=None, _geography={}))

    parent, child = state(parents), state(children)
    feedback_slab_surface(parent, child, reg, chunk_shape=(ratio * 2, ratio * 2))
    for name, want in expected.items():
        got = (parent._streamed_domain.store[f"fields/{name}"]
               if stored else cp.asnumpy(parent.physics.fields[name]))
        np.testing.assert_array_equal(got.view(np.uint32), cp.asnumpy(want).view(np.uint32))
