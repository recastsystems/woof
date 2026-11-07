"""All-member FORCE and feedback match each original resident edge word."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _same(cp, actual, expected):
    assert cp.asnumpy(actual).tobytes() == cp.asnumpy(expected).tobytes()


def _fixture(members, *, ratio, mapped, smooth, mp_physics=8, feedback=1, terrain=False, mu_halo=False):
    import cupy as cp
    from woof.core.model import FeedbackScratch
    from woof.core.nest import NestCoupler
    from woof.core.nest_interp import register_nest
    from woof.core.preflight import nest_field_kinds
    from woof.ensemble.batch_nesting import PreparedMemberNestEdge
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from test_nest_coupler import _clock

    parent, parents, pcfg, _ = _pack_physical(members, moist=True, mp_physics=mp_physics,
        mapped=mapped, terrain=terrain, mu_halo=mu_halo, boundary="specified", nz=6, ny=37, nx=39)
    child, children, ccfg, _ = _pack_physical(members, moist=True, mp_physics=mp_physics,
        mapped=mapped, terrain=terrain, mu_halo=mu_halo, boundary="nested", nz=6, ny=27, nx=30)
    pcfg = replace(pcfg, grid_id=1, dt=float(ratio * 3))
    ccfg = replace(ccfg, grid_id=2, dt=3.)
    parent.cfg, child.cfg = pcfg, ccfg
    pd = SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
        i_parent_start=1, j_parent_start=1, run=pcfg)
    cd = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=ratio,
        i_parent_start=5, j_parent_start=5, run=ccfg)
    nodes = []
    for member, (pstate, cstate) in enumerate(zip(parents, children, strict=True)):
        # A changed endpoint in every member catches parent-zero broadcasts.
        pstate.u += np.float32((member + 1) * .0625)
        parent.member_view("u", member)[:] = pstate.u
        pclock = _clock(1, 0, step_ticks=ratio * 3, dt=ratio * 3, advanced=True)
        cclock = _clock(2, 1, step_ticks=3, dt=3)
        pnode = SimpleNamespace(cfg=pd, state=pstate, clock=pclock)
        cnode = SimpleNamespace(cfg=cd, state=cstate, clock=cclock, parent=pnode)
        cnode.coupler = NestCoupler(cnode, feedback=feedback, smooth_option=smooth)
        nodes.append((pnode, cnode, FeedbackScratch()))
    regs = {stagger: register_nest(nri=ratio, nrj=ratio,
        i_parent_start=5, j_parent_start=5, child_nx=ccfg.nx, child_ny=ccfg.ny,
        parent_nx=pcfg.nx, parent_ny=pcfg.ny, stagger="" if stagger == "m" else stagger,
        wrapper="bdy") for stagger in ("m", "x", "y")}
    edge = PreparedMemberNestEdge(parent, child, registrations=regs,
        fields=nest_field_kinds(ccfg), parent_dt_fp32=np.float32(ratio * 3),
        parent_interval_ticks=ratio * 3, parent_tick_den=1, available_bytes=1 << 30,
        feedback=feedback, smooth_option=smooth, array_module=cp,
        member_ids=tuple(100 + member for member in range(members)))
    return parent, child, nodes, edge


@pytest.mark.parametrize("members", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [2, 3])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("smooth", [0, 1, 2])
def test_complete_member_force_and_feedback_equal_original_edges(members, ratio, mapped, smooth):
    import cupy as cp
    from woof.core.device_inventory import state_array_shapes
    parent, child, nodes, edge = _fixture(members, ratio=ratio, mapped=mapped, smooth=smooth)
    pclock = tuple(p.clock for p, _, _ in nodes)
    cclock = tuple(c.clock for _, c, _ in nodes)
    for period in range(2):
        tables = edge.force(parent_clocks=pclock, child_clocks=cclock)
        for pnode, cnode, _ in nodes:
            cnode.coupler.force(cnode)
        cp.cuda.get_current_stream().synchronize()
        for member, (pnode, cnode, _) in enumerate(nodes):
            for field, sides in tables.items():
                for side, values in sides.items():
                    for observed, expected in zip(values, cnode.coupler._last_tables[field][side], strict=True):
                        _same(cp, observed[member], expected)
            for batch, scalar in ((parent, pnode.state), (child, cnode.state)):
                for name in state_array_shapes(batch.cfg):
                    _same(cp, batch.member_view(name, member), getattr(scalar, name))
            assert cnode.clock.dtbc_fp32.tobytes() == np.float32(0).tobytes()
        # Stand in for the child's completed substeps with an exact word
        # copy into its clock. Coupling tests do not claim a dycore forecast.
        for p, c in zip(pclock, cclock, strict=True):
            c.ticks = p.ticks
        edge.feedback_prepare(parent_clocks=pclock, child_clocks=cclock)
        edge.feedback_commit()
        edge.feedback_finalize()
        for _, cnode, scratch in nodes:
            cnode.coupler.feedback_prepare(cnode, scratch)
            cnode.coupler.feedback_commit(cnode)
            cnode.coupler.feedback_finalize(cnode)
        cp.cuda.get_current_stream().synchronize()
        for member, (pnode, cnode, _) in enumerate(nodes):
            for batch, scalar in ((parent, pnode.state), (child, cnode.state)):
                for name in state_array_shapes(batch.cfg):
                    _same(cp, batch.member_view(name, member), getattr(scalar, name))
        if period == 0:
            for member, (pnode, cnode, _) in enumerate(nodes):
                # The next FORCE must consume a fresh own parent endpoint.
                pnode.state.u += np.float32((member + 1) * .03125)
                parent.member_view("u", member)[:] = pnode.state.u
                pnode.clock.advance()
    receipt = edge.receipt()
    assert receipt["force_count"] == receipt["feedback_count"] == 2
    assert receipt["numerical_member_loops"] == 0
    assert receipt["forecast_graph_admission_changed"] is False


def test_member_scratch_backings_have_dense_native_slabs_including_mass():
    import cupy as cp
    _, _, _, edge = _fixture(4, ratio=3, mapped=True, smooth=2)
    for views in (edge._parent_views, edge._child_views):
        for value in views.values():
            assert value.flags.c_contiguous
            assert value.strides[0] == value[0].nbytes
            assert any(value is backing for backing in edge.storage.arrays.values())
        assert views["mu"].shape[1] == 1
        assert views["w"] is views["ph"]
        assert views["t"] is views["qv"]
        assert views["u"] is not views["v"]


def test_force_clock_refusal_precedes_any_state_or_table_change():
    import cupy as cp
    from woof.ensemble.batch_state import BatchStateUnsupported
    _, _, nodes, edge = _fixture(4, ratio=3, mapped=True, smooth=2)
    for value in edge.storage.arrays.values():
        if value.dtype == np.float32 and value.ndim > 1:
            value.fill(np.float32(-7))
    before = {name: cp.asnumpy(value).tobytes() for name, value in edge.storage.arrays.items()}
    nodes[2][0].clock.ticks += 1
    with pytest.raises(BatchStateUnsupported, match="parent lead or step"):
        edge.force(parent_clocks=[p.clock for p, _, _ in nodes],
                   child_clocks=[c.clock for _, c, _ in nodes])
    assert {name: cp.asnumpy(value).tobytes() for name, value in edge.storage.arrays.items()} == before
    assert not edge.valid and edge.force_count == 0


@pytest.mark.parametrize("members", [2, 4, 8])
def test_rebound_adaptive_intervals_match_standalone_force_feedback_without_new_banks(members):
    import cupy as cp
    from woof.ensemble.batch_state import BatchStateUnsupported
    from woof.core.device_inventory import state_array_shapes
    parent, child, nodes, edge = _fixture(members, ratio=3, mapped=True, smooth=2)
    pc, cc = tuple(p.clock for p, _, _ in nodes), tuple(c.clock for _, c, _ in nodes)
    pointers = {name: int(value.data.ptr) for name, value in edge.storage.arrays.items()}
    coupling = tuple(row[:2] for row in edge._forces)
    for period, (parent_dt, child_dt) in enumerate(((9, 3), (12, 4), (6, 2), (15, 5))):
        for p, c in zip(pc, cc, strict=True):
            p.adaptive_state, c.adaptive_state = {"started": True}, {"started": True}
            if period:
                p.step_ticks, p.dt_fp32 = parent_dt, np.float32(parent_dt)
                c.step_ticks, c.dt_fp32 = child_dt, np.float32(child_dt)
                p.advance()
        if period:
            before = {name: value.get().tobytes() for name, value in edge.storage.arrays.items()}
            with pytest.raises(BatchStateUnsupported, match="bound original interval"):
                edge.force(parent_clocks=pc, child_clocks=cc)
            assert {name: value.get().tobytes() for name, value in edge.storage.arrays.items()} == before
        assert edge.rebind_parent_interval(parent_clocks=pc, child_clocks=cc) is bool(period)
        assert {name: int(value.data.ptr) for name, value in edge.storage.arrays.items()} == pointers
        assert tuple(row[:2] for row in edge._forces) == coupling
        tables = edge.force(parent_clocks=pc, child_clocks=cc)
        for _, cnode, _ in nodes:
            cnode.coupler.force(cnode)
        cp.cuda.get_current_stream().synchronize()
        for member, (pnode, cnode, _) in enumerate(nodes):
            for field, sides in tables.items():
                for side, values in sides.items():
                    for actual, expected in zip(values, cnode.coupler._last_tables[field][side], strict=True):
                        _same(cp, actual[member], expected)
            for batch, state in ((parent, pnode.state), (child, cnode.state)):
                for name in state_array_shapes(batch.cfg):
                    _same(cp, batch.member_view(name, member), getattr(state, name))
        for p, c in zip(pc, cc, strict=True):
            c.ticks = p.ticks
        edge.feedback_prepare(parent_clocks=pc, child_clocks=cc)
        edge.feedback_commit()
        edge.feedback_finalize()
        for _, cnode, scratch in nodes:
            cnode.coupler.feedback_prepare(cnode, scratch)
            cnode.coupler.feedback_commit(cnode)
            cnode.coupler.feedback_finalize(cnode)
        cp.cuda.get_current_stream().synchronize()
        for member, (pnode, cnode, _) in enumerate(nodes):
            for batch, state in ((parent, pnode.state), (child, cnode.state)):
                for name in state_array_shapes(batch.cfg):
                    _same(cp, batch.member_view(name, member), getattr(state, name))
            assert cnode.clock.dtbc_fp32.tobytes() == np.float32(0).tobytes()
            if period < 3:
                pnode.state.u += np.float32((member + 1) * .03125)
                parent.member_view("u", member)[:] = pnode.state.u
    assert edge.receipt()["interval_rebind_count"] == 3
    assert edge.force_count == edge.feedback_count == 4


@pytest.mark.parametrize("mismatch", ["endpoint_lattice", "member_lattice", "adaptive_dt_image"])
def test_force_refuses_clock_lattice_and_fp32_interval_mismatches_before_launch(mismatch):
    import cupy as cp
    from woof.ensemble.batch_state import BatchStateUnsupported
    _, _, nodes, edge = _fixture(4, ratio=3, mapped=True, smooth=0)
    pc, cc = [p.clock for p, _, _ in nodes], [c.clock for _, c, _ in nodes]
    if mismatch == "endpoint_lattice":
        cc[1].tick_den = 2
    elif mismatch == "member_lattice":
        pc[1].tick_den = cc[1].tick_den = 2
    else:
        for p in pc:
            p.dt_fp32 = np.float32(4.5)
            p.adaptive_state = {"started": True}
    before = {name: cp.asnumpy(value).tobytes() for name, value in edge.storage.arrays.items()}
    with pytest.raises(BatchStateUnsupported, match="clock lattice|parent lead or step|FP32 step"):
        edge.force(parent_clocks=pc, child_clocks=cc)
    assert {name: cp.asnumpy(value).tobytes() for name, value in edge.storage.arrays.items()} == before
    assert edge.force_count == 0 and not edge.valid


def test_feedback_transaction_requires_commit_and_rejects_rebound_state():
    import cupy as cp
    from woof.ensemble.batch_state import BatchStateUnsupported
    parent, _, nodes, edge = _fixture(4, ratio=3, mapped=False, smooth=1)
    pc = [p.clock for p, _, _ in nodes]
    cc = [c.clock for _, c, _ in nodes]
    for p, c in zip(pc, cc, strict=True):
        c.ticks = p.ticks
    edge.feedback_prepare(parent_clocks=pc, child_clocks=cc)
    with pytest.raises(RuntimeError, match="must commit"):
        edge.feedback_finalize()
    parent.storage.arrays["u"] = cp.copy(parent.storage.arrays["u"])
    with pytest.raises(BatchStateUnsupported, match="backing changed"):
        edge.feedback_commit()


def test_force_rejects_shared_side_outputs_before_launch():
    import cupy as cp
    from woof.ensemble.batch_nesting import prepare_member_bdy_interp1
    _, _, _, edge = _fixture(4, ratio=3, mapped=True, smooth=0)
    out = dict(edge.tables["u"])
    out["east"] = out["west"]
    before = {side: tuple(cp.asnumpy(value).tobytes() for value in values) for side, values in out.items()}
    with pytest.raises(ValueError, match="overlap"):
        prepare_member_bdy_interp1(edge._parent_views["u"], edge._child_views["u"],
            edge.registrations["x"], members=4, parent_dt_fp32=edge.parent_dt_fp32,
            parent_interval_ticks=edge.parent_interval_ticks, out=out,
            geometry=edge.geometry["x"], spec_zone=edge.child.cfg.spec_zone,
            relax_zone=edge.child.cfg.relax_zone, spec_bdy_width=edge.child.cfg.spec_bdy_width)
    assert {side: tuple(cp.asnumpy(value).tobytes() for value in values) for side, values in out.items()} == before


@pytest.mark.parametrize("members", [2, 4, 8])
def test_member_coupling_operator_microbenchmark_and_word_gate(members):
    import cupy as cp
    import json
    import time
    from woof.core.device_inventory import state_array_shapes
    parent, _, nodes, edge = _fixture(members, ratio=3, mapped=True, smooth=2)
    loops = 8
    pc, cc = [p.clock for p, _, _ in nodes], [c.clock for _, c, _ in nodes]
    # Warm every stock and indexed launch before either timed arm. Both
    # live parents receive exactly this same initial transaction.
    edge.force(parent_clocks=pc, child_clocks=cc)
    for _, cnode, _ in nodes:
        cnode.coupler.force(cnode)
    for p, c in zip(pc, cc, strict=True):
        c.ticks = p.ticks
    edge.feedback_prepare(parent_clocks=pc, child_clocks=cc)
    edge.feedback_commit()
    edge.feedback_finalize()
    for pnode, cnode, scratch in nodes:
        cnode.coupler.feedback_prepare(cnode, scratch)
        cnode.coupler.feedback_commit(cnode)
        cnode.coupler.feedback_finalize(cnode)
        pnode.clock.advance()
    cp.cuda.get_current_stream().synchronize()
    first_parent_ticks, first_child_ticks = pc[0].ticks, cc[0].ticks
    # Each arm starts at identical clocks and fields and executes the same
    # edge transactions. These walls measure coupling operators only.
    started = time.perf_counter()
    for iteration in range(loops):
        for pnode, cnode, scratch in nodes:
            cnode.coupler.force(cnode)
            cnode.clock.ticks = pnode.clock.ticks
            cnode.coupler.feedback_prepare(cnode, scratch)
            cnode.coupler.feedback_commit(cnode)
            cnode.coupler.feedback_finalize(cnode)
            if iteration + 1 < loops:
                pnode.clock.advance()
    cp.cuda.get_current_stream().synchronize()
    ordinary = time.perf_counter() - started
    from test_nest_coupler import _clock
    packed_pc = [_clock(1, 0, step_ticks=9, dt=9, advanced=True) for _ in range(members)]
    packed_cc = [_clock(2, 1, step_ticks=3, dt=3) for _ in range(members)]
    for p, c in zip(packed_pc, packed_cc, strict=True):
        p.ticks, c.ticks = first_parent_ticks, first_child_ticks
    started = time.perf_counter()
    for iteration in range(loops):
        edge.force(parent_clocks=packed_pc, child_clocks=packed_cc)
        for p, c in zip(packed_pc, packed_cc, strict=True):
            c.ticks = p.ticks
        edge.feedback_prepare(parent_clocks=packed_pc, child_clocks=packed_cc)
        edge.feedback_commit()
        edge.feedback_finalize()
        if iteration + 1 < loops:
            for p in packed_pc:
                p.advance()
    cp.cuda.get_current_stream().synchronize()
    batched = time.perf_counter() - started
    for member, (pnode, _, _) in enumerate(nodes):
        for name in state_array_shapes(parent.cfg):
            _same(cp, parent.member_view(name, member), getattr(pnode.state, name))
    print(json.dumps({"contract": "member-nest-operator-microbenchmark-v1", "members": members,
        "transactions": loops, "ordinary_seconds": ordinary, "member_indexed_seconds": batched,
        "ratio": ordinary / batched, "scope": "coupling operators only; synthetic resident fields",
        "forecast_speedup_claimed": False, "parent_fields_byte_identical": True}))
