"""Selected surface carriers follow WRF's real/integer feedback operators."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import json

import numpy as np
import pytest
from conftest import requires_gpu


def test_feedback_inventory_follows_native_registry_flags():
    from woof.core.noahmp_nest import feedback_fields
    rows = json.loads((Path(__file__).resolve().parents[1] /
        "tools/noahmp_options_wrf461_oracle/native_feedback_fields.json").read_text())
    native = {row[0] for row in rows}
    common = {"tslb", "smois", "sh2o", "snow", "snowh", "canwat", "tsk",
              "snowc", "acsnow", "acsnom"}
    assert len(feedback_fields()) == len(set(feedback_fields())) == 56
    assert set(feedback_fields()) <= native | common
    assert {"taussxy", "pgsxy", "lai", "rs"}.isdisjoint(feedback_fields())


@requires_gpu
@pytest.mark.parametrize("ratio", [2, 3, 5])
@pytest.mark.parametrize("stored", [False, "canonical", "published"])
def test_bounded_feedback_matches_native_operator_for_every_carrier(ratio, stored, monkeypatch):
    import cupy as cp
    from woof.core.noahmp_nest import feedback_fields, feedback_selected_noahmp
    from woof.core.nest_interp import register_nest, copy_fcnm, copy_fcni
    from woof.core.noahmp_runtime import NOAHMP_STATE_SNOW_3D, NOAHMP_STATE_SNOWSOIL_3D
    cfg = SimpleNamespace(sf_surface_physics=4, opt_btr=2, opt_rsf=2)
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=3,
        j_parent_start=3, child_nx=ratio*6, child_ny=ratio*6,
        parent_nx=14, parent_ny=14, stagger="", wrapper="bdy")
    pfields, cfields, expected = {}, {}, {}
    for index, name in enumerate(feedback_fields()):
        levels = ((3,) if name in NOAHMP_STATE_SNOW_3D else
                  (7,) if name in NOAHMP_STATE_SNOWSOIL_3D else
                  (4,) if name in ("tslb", "smois", "sh2o") else ())
        dtype = cp.int32 if name == "isnowxy" else cp.float32
        p = cp.full((*levels,14,14), -17-index, dtype=dtype)
        c = cp.arange(int(np.prod((*levels,reg.nyc,reg.nxc))), dtype=dtype).reshape(*levels,reg.nyc,reg.nxc)
        if name != "isnowxy":
            # NaN payloads and signed zero must travel as words, too.
            c.view(cp.uint32).reshape(-1)[::7] = np.uint32(0x80000000)
            c.view(cp.uint32).reshape(-1)[::13] = np.uint32(0x7fc00123)
        want = p.copy()
        (copy_fcni if name == "isnowxy" else copy_fcnm)(want,c,reg,spec_zone=1)
        expected[name] = want.get().tobytes()
        pfields[name], cfields[name] = p, c
    parent = SimpleNamespace(cfg=SimpleNamespace(run=cfg),
        state=SimpleNamespace(physics=SimpleNamespace(fields=pfields)))
    child = SimpleNamespace(cfg=SimpleNamespace(run=cfg), parent=parent,
        state=SimpleNamespace(physics=SimpleNamespace(fields=cfields)))
    if stored:
        for state, fields in ((parent.state,pfields),(child.state,cfields)):
            store = {f"fields/{name}": value.get() for name,value in fields.items()}
            if stored == "canonical":
                state._streamed_domain = SimpleNamespace(store=store,_geography={},template_state=state)
            else:
                from woof.core.streaming import _STORE_ATTR
                setattr(state, _STORE_ATTR, store)
            for value in fields.values():
                value[...] = -999  # Canonical store must win over this stale slab.
    else:
        monkeypatch.setattr(cp,"asnumpy",lambda *a,**k: pytest.fail("resident feedback read the host"))
    receipt = feedback_selected_noahmp(child,reg,spec_zone=1,chunk_shape=(8,9))
    for name,want in expected.items():
        from woof.core.streaming import domain_store
        store = (parent.state._streamed_domain.store if stored == "canonical"
                 else domain_store(parent.state))
        actual = (store[f"fields/{name}"].tobytes()
                  if stored else pfields[name].get().tobytes())
        assert actual == want, (ratio,stored,name)
    assert bool(receipt["sync_bytes"]) == bool(stored)


@requires_gpu
@pytest.mark.parametrize("feedback", [0, 1])
def test_live_coupler_routes_selected_surface_feedback(feedback):
    import cupy as cp
    from test_feedback import _domain, _seed_state
    from woof.core.state import DomainState
    from woof.core.nest import NestCoupler
    from woof.core.model import FeedbackScratch
    from woof.core.noahmp_nest import feedback_fields

    nodes=[]
    for grid_id,parent_id,nx in ((1,0,14),(2,1,18)):
        cfg=_domain(grid_id,parent_id,nx=nx,ny=nx,ratio=3)
        cfg=replace(cfg,run=replace(cfg.run,sf_surface_physics=4,opt_btr=2,opt_rsf=2))
        state=DomainState(cfg.run)
        _seed_state(state,child=bool(parent_id))
        state.physics=SimpleNamespace(fields={name:cp.full((nx,nx),
            -2 if name=='isnowxy' else 280+parent_id,
            dtype=cp.int32 if name=='isnowxy' else cp.float32)
            for name in feedback_fields()})
        nodes.append(SimpleNamespace(cfg=cfg,state=state,clock=SimpleNamespace(ticks=0)))
    parent,child=nodes;child.parent=parent
    coupler=NestCoupler(child,feedback=feedback)
    coupler.feedback_prepare(child,FeedbackScratch())
    coupler.feedback_commit(child)
    assert float(parent.state.physics.fields['tvxy'][4,4]) == (281 if feedback else 280)
    assert float(parent.state.physics.fields['tvxy'][0,0]) == 280


@requires_gpu
@pytest.mark.parametrize("stored", [False, True])
def test_wetland_feedback_copies_store_words_and_keeps_domain_static_maps(stored):
    import cupy as cp
    from woof.core.noahmp_nest import feedback_fields, feedback_selected_noahmp
    from woof.core.nest_interp import register_nest, copy_fcnm
    from woof.core.nest_operands import NestWindowSource

    cfg = SimpleNamespace(sf_surface_physics=4, noahmp_wetland=1)
    reg = register_nest(nri=3, nrj=3, i_parent_start=3, j_parent_start=3,
                        child_nx=18, child_ny=18, parent_nx=14, parent_ny=14,
                        stagger="", wrapper="bdy")
    carried = ("wetstorexy", "wetfsatxy", "wetlhxy")
    static = ("wetfsatmx", "wetcap")
    nodes = []
    for width, initial in ((14, -17), (18, 23)):
        fields = {name: cp.full((width, width), initial,
                               cp.int32 if name == "isnowxy" else cp.float32)
                  for name in (*feedback_fields(), *carried, *static)}
        state = SimpleNamespace(physics=SimpleNamespace(fields=fields))
        nodes.append(SimpleNamespace(cfg=SimpleNamespace(run=cfg), state=state))
    parent, child = nodes
    child.parent = parent
    expected = {}
    for name in carried:
        donor = child.state.physics.fields[name]
        donor.view(cp.uint32).reshape(-1)[::7] = np.uint32(0x80000000)
        donor.view(cp.uint32).reshape(-1)[::13] = np.uint32(0x7fc00123)
        want = parent.state.physics.fields[name].copy()
        copy_fcnm(want, donor, reg, spec_zone=1)
        expected[name] = want.get().tobytes()
    if stored:
        for node in nodes:
            fields = node.state.physics.fields
            node.state._streamed_domain = SimpleNamespace(
                store={f"fields/{name}": value.get() for name, value in fields.items()},
                _geography={}, template_state=node.state)
            for value in fields.values():
                value.fill(-999)
    before = {name: cp.asnumpy(NestWindowSource(parent.state).array(f"fields/{name}")).tobytes()
              for name in static}
    receipt = feedback_selected_noahmp(child, reg, spec_zone=1, chunk_shape=(8, 9))
    for name, want in {**expected, **before}.items():
        actual = NestWindowSource(parent.state).array(f"fields/{name}")
        assert cp.asnumpy(actual).tobytes() == want, name
    assert set(carried) <= set(receipt["fields"])
    assert set(static).isdisjoint(receipt["fields"])
    assert bool(receipt["sync_bytes"]) == stored


@requires_gpu
@pytest.mark.parametrize("parent_mode,child_mode", [(1, 0), (0, 1)])
def test_mixed_wetland_feedback_refuses_before_any_carrier_write(parent_mode, child_mode):
    from woof.core.noahmp_nest import feedback_selected_noahmp

    def node(mode):
        return SimpleNamespace(
            cfg=SimpleNamespace(run=SimpleNamespace(sf_surface_physics=4, noahmp_wetland=mode)),
            state=SimpleNamespace(physics=SimpleNamespace(fields={})))

    child = node(child_mode)
    child.parent = node(parent_mode)
    with pytest.raises(ValueError, match="noahmp_wetland must match"):
        feedback_selected_noahmp(child, None, spec_zone=1, chunk_shape=(8, 9))
