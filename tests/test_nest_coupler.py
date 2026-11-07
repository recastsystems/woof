"""CPU contracts for Phase-5 Task 13 nest forcing (GPU parity is marked)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import RunConfig
from woof.core import preflight as pf
from woof.core.clock import DomainClock, DomainTicks
from woof.core.dycore import _boundary_forced
from woof.core.nest import NestCoupler
from woof.core.microphysics_transition import MP8_TO_MP18_POLICY
from woof.core.state import adopt_wrf_theta_carrier
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.ingest.lateral_bc import (apply_specified_relaxation,
                                     attach_nest_boundaries,
                                     couple_nest_field, _weights)
from woof.verify.npref import (np_couple_nest_field, np_nest_force,
                                np_spec_bdyupdate,
                                np_specified_relaxation,
                                np_w_relaxation_current)


def _run(nx, ny, *, nested, grid_id):
    return RunConfig(nx=nx, ny=ny, nz=2, dx=1000.0, dy=1000.0,
                     ztop=10000.0, dt=3.0, run_seconds=9.0,
                     nested=nested, specified=not nested, grid_id=grid_id,
                     spec_bdy_width=5, spec_zone=1, relax_zone=4)


def _clock(grid_id, parent_id, *, step_ticks, dt, advanced=False):
    spec = DomainTicks(
        grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
        step_ticks=step_ticks, dt_fp32=np.float32(dt), history_ticks=100,
        restart_ticks=None, radt_ticks=None, stepra=None, cudt_ticks=None,
        stepcu=None, bldt_ticks=None, stepbl=None)
    clock = DomainClock(spec, tick_den=1, run_ticks=1000)
    if advanced:
        clock.advance()
    return clock


class _State:
    def __init__(self, run, *, moisture_fields=(
            "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng")):
        nz, ny, nx = run.nz, run.ny, run.nx
        self.mub2d = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) / 9 + 5
        self.mup = np.full((ny, nx), np.float32(0.25))
        self.u = np.full((nz, ny, nx + 1), np.float32(2.0))
        self.v = np.full((nz, ny + 1, nx), np.float32(-1.5))
        self.w = np.full((nz + 1, ny, nx), np.float32(0.75))
        self.thp = np.full((nz, ny, nx), np.float32(1.25))
        self.php = np.full((nz + 1, ny, nx), np.float32(3.0))
        self.thb = np.array([300.0, 302.0], dtype=np.float32)
        self.c1h = np.array([0.8, 0.6], dtype=np.float32)
        self.c2h = np.array([1.0, 2.0], dtype=np.float32)
        self.c1f = np.array([1.0, 0.7, 0.4], dtype=np.float32)
        self.c2f = np.array([0.0, 1.5, 3.0], dtype=np.float32)
        self.msft = np.full((ny, nx), np.float32(1.25))
        self.msfu = np.full((ny, nx + 1), np.float32(1.5))
        self.msfv = np.full((ny + 1, nx), np.float32(1.75))
        # DomainState.set_map_geometry owns both representations. The strict
        # relaxation kernel consumes the inverse, including for non-v calls.
        self.msfv_inv = np.float32(1.0) / self.msfv
        for index, name in enumerate(moisture_fields, start=1):
            setattr(self, name, np.full(
                (nz, ny, nx), np.float32(index / 100.0)))
        self.has_msf = True
        # Broadcast base profile, like an unmaterialized production base.
        self.phb = np.zeros((nz + 1,), np.float32)
        self._scratch = {}
        self.lateral_boundaries = None
        # Match both production input doors: strict kernels consume WRF t_2
        # directly, whereas default kernels rebase thp against thb themselves.
        adopt_wrf_theta_carrier(self)

    def scratch(self, shape, slot, dtype=None):
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = np.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


class _DeviceState:
    def __init__(self, host, cp):
        self._cp = cp
        self._scratch = {}
        for name, value in vars(host).items():
            if name == "_scratch":
                continue
            setattr(self, name, cp.asarray(value) if isinstance(
                value, np.ndarray) else value)

    def scratch(self, shape, slot, dtype=None):
        dtype = self._cp.dtype(self._cp.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result

    def existing_scratch(self, slot):
        # A newly loaded state has no acoustic mass halo yet. Its absence is
        # distinct from allocating a zero halo, just as in DomainState.
        return self._scratch.get(slot)


def _nodes(*, parent_shape=(16, 16), child_start=(4, 4), state_factory=_State):
    prun = _run(*parent_shape, nested=False, grid_id=1)
    # Large enough that the child RK arena backing covers the full parent;
    # the production chain has the same refinement relation.
    crun = _run(30, 30, nested=True, grid_id=2)
    pcfg = SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                           i_parent_start=1, j_parent_start=1, run=prun)
    ccfg = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                           i_parent_start=child_start[0], j_parent_start=child_start[1],
                           run=crun)
    parent = SimpleNamespace(
        cfg=pcfg, state=state_factory(prun),
        clock=_clock(1, 0, step_ticks=3, dt=9.0, advanced=True))
    child_clock = _clock(2, 1, step_ticks=1, dt=3.0)
    child_clock.prepare_step()
    child = SimpleNamespace(cfg=ccfg, state=state_factory(crun), parent=parent,
                            clock=child_clock)
    return parent, child


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("start", [4, 31])
def test_parent_coupling_window_contains_all_sint_donors(ratio, start):
    from woof.core.nest import parent_footprint_window
    from woof.core.streaming import window_slices
    from woof.core.nest_interp import register_nest

    cfg = SimpleNamespace(parent_grid_ratio=ratio,
                          i_parent_start=start, j_parent_start=start + 1,
                          run=_run(37, 43, nested=True, grid_id=2))
    for stagger in ("", "x", "y"):
        reg = register_nest(
            nri=ratio, nrj=ratio, i_parent_start=start,
            j_parent_start=start + 1, child_nx=37, child_ny=43,
            parent_nx=96, parent_ny=96, stagger=stagger, wrapper="bdy")
        _, y, x = window_slices((reg.nyp, reg.nxp),
                                parent_footprint_window(cfg))
        assert x.start <= int(reg.ci.min()) - 2
        assert int(reg.ci.max()) + 2 < x.stop
        assert y.start <= int(reg.cj.min()) - 2
        assert int(reg.cj.max()) + 2 < y.stop


def test_f16_manifest_equals_landed_registration_inventory_and_dtypes():
    parent, child = _nodes()
    coupler = NestCoupler(child)
    for stagger, reg in coupler.registrations.items():
        for name in ("ci", "ip", "cj", "jp", "xig", "xjg"):
            slot = f"nest_sint_{name}_{stagger}"
            host = getattr(reg, name)
            assert coupler.slot_shapes[slot] == host.shape
            assert coupler.slot_dtypes[slot] == host.dtype.name
    assert not any("donor" in name for name in coupler.slot_shapes)
    assert coupler.slot_shapes["nest_parent_field"] == (3 * 16 * 16,)
    assert coupler.slot_shapes["nest_child_field"] == (3 * 30 * 30,)
    # F16 says the persistent frames match what bdy_interp1 writes, not a
    # potentially wider configured maximum.
    wide = pf.nest_slot_shapes(child.cfg, 7, parent.cfg)
    assert wide["nest_u_bxs"] == (2, 30, 5)


def test_f16_parent_field_lifetime_is_audited_and_resident_neutral():
    parent, child = _nodes()
    domains = (parent.cfg, child.cfg)
    shapes = pf.shared_scratch_arena_shapes(domains)
    aliases = pf.shared_scratch_arena_aliases(domains)
    assert pf.scratch_slot_lifetime("nest_parent_field").kind == \
        "write_before_read"
    assert aliases == {"nest_parent_field": "rk_ru",
                       "nest_child_field": "rk_ww",
                       "lbc_nested_relax": "acoustic_a"}
    assert np.prod(shapes["nest_parent_field"]) <= np.prod(shapes["rk_ru"])
    assert np.prod(shapes["nest_child_field"]) <= np.prod(shapes["rk_ww"])
    assert aliases["nest_parent_field"] != aliases["nest_child_field"]
    assert pf.shared_scratch_arena_bytes(domains) == sum(
        4 * int(np.prod(shape)) for name, shape in shapes.items()
        if name not in aliases)


def test_skinny_child_routes_full_fields_to_matching_dead_rk_stagger(
    monkeypatch,
):
    """A valid nx<nz child must not assume W is the largest RK field."""
    import woof.core.nest as nest_mod

    class SkinnyState:
        def __init__(self):
            nz, ny, nx = 80, 60, 60
            self.u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
            self.v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
            self.w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
            self.thp = np.zeros((nz, ny, nx), dtype=np.float32)
            self.php = np.zeros((nz + 1, ny, nx), dtype=np.float32)
            self.mup = np.zeros((ny, nx), dtype=np.float32)
            self._scratch = {}
            self.requests = []

        def scratch(self, shape, slot, dtype=None):
            shape = tuple(int(n) for n in shape)
            dtype = np.dtype(np.float32 if dtype is None else dtype)
            self.requests.append((slot, shape))
            if slot not in self._scratch:
                self._scratch[slot] = np.zeros(shape, dtype=dtype)
            result = self._scratch[slot]
            assert result.shape == shape and result.dtype == dtype
            return result

        def existing_scratch(self, slot):
            return self._scratch.get(slot)

    state = SkinnyState()
    child = SimpleNamespace(state=state)
    coupler = object.__new__(NestCoupler)
    coupler.child_node = child
    fields = ("u", "v", "w", "t", "ph", "mu")
    field_capacity = max(
        state.u.size, state.v.size, state.w.size, state.thp.size,
        state.php.size, state.mup.size)
    coupler.slot_shapes = {"nest_child_field": (field_capacity,)}
    coupler.slot_dtypes = {"nest_child_field": "float32"}

    def fake_couple(_state, kind, out):
        out[...] = np.float32(fields.index(kind) + 1)

    monkeypatch.setattr(nest_mod, "couple_nest_field", fake_couple)
    expected_shapes = {
        "u": state.u.shape, "v": state.v.shape, "w": state.w.shape,
        "t": state.thp.shape, "ph": state.php.shape,
        "mu": (1, *state.mup.shape),
    }

    assert state.u.size > state.w.size
    assert state.v.size > state.w.size
    for kind in fields:
        result = coupler._coupled_child_field(kind)
        assert result.shape == expected_shapes[kind]
        assert np.shares_memory(result, state._scratch["nest_child_field"])
        assert np.all(result == np.float32(fields.index(kind) + 1))

    assert state.requests == [
        ("nest_child_field", (field_capacity,)),
        ("nest_child_field", (field_capacity,)),
        ("nest_child_field", (field_capacity,)),
        ("nest_child_field", (field_capacity,)),
        ("nest_child_field", (field_capacity,)),
        ("nest_child_field", (field_capacity,)),
    ]

    parent_run = RunConfig(
        nx=120, ny=100, nz=80, dx=12000.0, dy=12000.0,
        ztop=20000.0, dt=60.0, run_seconds=120.0,
        nested=False, specified=True, grid_id=1,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    child_run = RunConfig(
        nx=60, ny=60, nz=80, dx=4000.0, dy=4000.0,
        ztop=20000.0, dt=20.0, run_seconds=120.0,
        nested=True, specified=False, grid_id=2,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    parent_cfg = SimpleNamespace(
        grid_id=1, parent_id=0, parent_grid_ratio=1, run=parent_run)
    child_cfg = SimpleNamespace(
        grid_id=2, parent_id=1, parent_grid_ratio=3, run=child_run)
    aliases = pf.shared_scratch_arena_aliases((parent_cfg, child_cfg))
    assert aliases["nest_parent_field"] == "rk_ww"
    assert aliases["nest_child_field"] == "rk_ru"
    assert aliases["nest_parent_field"] != aliases["nest_child_field"]
    shapes = pf.shared_scratch_arena_shapes((parent_cfg, child_cfg))
    assert shapes["lbc_nested_relax"] == (80 * 60 * 61,)
    # The shared tree's parent acoustic backing is large enough here; the
    # runtime must request the full logical capacity rather than a W-shaped
    # child view of that larger physical allocation.
    assert np.prod(shapes["lbc_nested_relax"]) <= np.prod(
        shapes["acoustic_a"])
    assert aliases["lbc_nested_relax"] == "acoustic_a"


def test_bind_geometry_uses_table_names_not_allocation_order():
    _, child = _nodes()
    coupler = NestCoupler(child)

    class _ReverseRegistration:
        def __init__(self, stagger):
            self.stagger = stagger

        def device_tables(self, alloc):
            tables = {}
            for marker, name in enumerate(
                    reversed(("ci", "ip", "cj", "jp", "xig", "xjg")),
                    start=1):
                slot = f"nest_sint_{name}_{self.stagger}"
                table = alloc(name, coupler.slot_shapes[slot],
                              np.dtype(coupler.slot_dtypes[slot]))
                table[...] = marker
                tables[name] = table
            return tables

    coupler.registrations = {
        stagger: _ReverseRegistration(stagger)
        for stagger in ("m", "x", "y")}
    coupler._bind_geometry()
    for stagger in ("m", "x", "y"):
        for marker, name in enumerate(
                reversed(("ci", "ip", "cj", "jp", "xig", "xjg")),
                start=1):
            slot = f"nest_sint_{name}_{stagger}"
            assert np.all(child.state._scratch[slot] == marker)


def test_w_full_level_coupling_uses_hand_derived_wrf_real_tree():
    _, child = _nodes()
    state = child.state
    got = np_couple_nest_field(state, "w", dtype=np.float32)
    # couple_or_uncouple_em.F:123,279 at k=1,j=0,i=13:
    # mub=6.4444447, mup=.25, c1f=.7, c2f=1.5, w=.75, msft=1.25.
    # Literal WRF REAL stores give hybrid 0x40c5f4a0, map-divided weight
    # 0x409e5d4d, and donor 3.711667 = 0x406d8bf4.  Multiplying before
    # the division instead lands ...bf3.
    actual = got[1, 0, 13]
    assert actual == np.float32(3.711667)
    assert actual.view(np.uint32) == np.uint32(0x406D8BF4)
    assert actual.view(np.uint32) != np.uint32(0x406D8BF3)


@pytest.mark.parametrize("field", ["u", "v"])
def test_staggered_low_and_high_physical_faces_pin_distinct_wrf_real_trees(
    field,
):
    """WRF duplicates the low mass halo and repairs only the high face."""
    base = np.float32(693.0506)
    perturbation = np.float32(6.3163424)
    state = SimpleNamespace(
        mub2d=np.full((2, 2), base, dtype=np.float32),
        mup=np.full((2, 2), perturbation, dtype=np.float32),
        c1h=np.asarray([1.0], dtype=np.float32),
        c2h=np.asarray([0.0], dtype=np.float32),
        c1f=np.asarray([1.0, 1.0], dtype=np.float32),
        c2f=np.asarray([0.0, 0.0], dtype=np.float32),
        thb=np.asarray([300.0], dtype=np.float32),
        u=np.ones((1, 2, 3), dtype=np.float32),
        v=np.ones((1, 3, 2), dtype=np.float32),
        msfu=np.ones((2, 3), dtype=np.float32),
        msfv=np.ones((3, 2), dtype=np.float32),
        msft=np.ones((2, 2), dtype=np.float32),
        has_msf=True,
    )

    # couple_or_uncouple_em.F:140-145: fl(0.5 *
    # fl(fl(fl(B+B)+Q)+Q)); :149-166 repairs only the high face to fl(B+Q).
    low_sum = np.float32(base + base)
    low_sum = np.float32(low_sum + perturbation)
    low_sum = np.float32(low_sum + perturbation)
    low_expected = np.float32(np.float32(0.5) * low_sum)
    high_expected = np.float32(base + perturbation)
    assert low_expected.view(np.uint32) == np.uint32(0x442ED77B)
    assert high_expected.view(np.uint32) == np.uint32(0x442ED77C)

    coupled = np_couple_nest_field(state, field, dtype=np.float32)
    if field == "u":
        low_actual = coupled[0, 0, 0]
        high_actual = coupled[0, 0, -1]
    else:
        low_actual = coupled[0, 0, 0]
        high_actual = coupled[0, -1, 0]
    assert low_actual.view(np.uint32) == low_expected.view(np.uint32)
    assert high_actual.view(np.uint32) == high_expected.view(np.uint32)
    assert low_actual.view(np.uint32) != high_actual.view(np.uint32)


def test_w_davies_current_mirror_is_map_factor_free_regression():
    _, child = _nodes()
    state = child.state
    current = np_w_relaxation_current(state, dtype=np.float32)
    donor = np_couple_nest_field(state, "w", dtype=np.float32)
    actual = current[1, 0, 13]
    # module_bc_em.F:327-333/1740: mass_weight has no msfty division.
    assert actual == np.float32(4.639583)
    assert actual.view(np.uint32) == np.uint32(0x40947777)
    assert actual != donor[1, 0, 13]


def test_nested_spec_bdyupdate_w_updates_only_the_frame():
    field = np.arange(2 * 7 * 8, dtype=np.float64).reshape(2, 7, 8)
    tendency = np.linspace(-2.0, 3.0, field.size).reshape(field.shape)
    got = np_spec_bdyupdate(field, tendency, 0.25, spec_zone=1)
    expected = field.copy()
    expected[:, 0, :] += 0.25 * tendency[:, 0, :]
    expected[:, -1, :] += 0.25 * tendency[:, -1, :]
    expected[:, 1:-1, 0] += 0.25 * tendency[:, 1:-1, 0]
    expected[:, 1:-1, -1] += 0.25 * tendency[:, 1:-1, -1]
    np.testing.assert_array_equal(got, expected)


def test_np_nest_force_builds_all_dry_tables_with_wrf_w_value_pin():
    parent, child = _nodes()
    coupler = NestCoupler(child)
    fields = ("u", "v", "w", "t", "ph", "mu")
    tables = np_nest_force(
        parent.state, child.state, coupler.registrations,
        field_names=fields, parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5,
        dtype=np.float32)
    assert set(tables) == set(fields)
    assert all(set(tables[name]) == {"west", "east", "south", "north"}
               for name in fields)
    # Child west VALUE is nfld verbatim.  At k=1,j=0,i=0 the independently
    # hand-derived WRF stored-weight coupling is 3.1050003 = 0x4046b853.
    actual = np.float32(tables["w"]["west"][0][1, 0, 0])
    assert actual == np.float32(3.1050003)
    assert actual.view(np.uint32) == np.uint32(0x4046B853)


def test_nested_davies_callers_force_zero_spec_exp(monkeypatch):
    import dataclasses

    from woof.ingest import lateral_bc as lbc

    _, child = _nodes()
    state = child.state
    cfg = dataclasses.replace(child.cfg.run, spec_exp=0.75)
    assert cfg.nested and cfg.spec_exp != 0.0
    fields = {}
    for name, target in {
            "u": state.u, "v": state.v, "w": state.w,
            "theta": state.thp, "phi": state.php,
            "mu": state.mup[None], "qv": state.qv}.items():
        nz, ny, nx = target.shape

        def side(shape):
            return SimpleNamespace(
                value=np.zeros(shape, np.float32),
                tendency=np.zeros(shape, np.float32))

        width = 5
        boundary = SimpleNamespace(
            west=side((nz, ny, width)), east=side((nz, ny, width)),
            south=side((nz, width, nx)), north=side((nz, width, nx)))
        fields[name] = boundary
    state.lateral_boundaries = object()
    state._lateral_boundary_device = SimpleNamespace(
        rolling=True, valid=True, clock=child.clock,
        intervals=(SimpleNamespace(fields=fields),))
    for name, target in (("ru_t", state.u), ("rv_t", state.v),
                         ("rw_t", state.w), ("rth_t", state.thp),
                         ("rph_t", state.php), ("rmu_t", state.mup)):
        setattr(state, name, np.zeros_like(target))
    for name in ("u", "v", "w", "thp", "php"):
        setattr(state, name + "0", getattr(state, name).copy())
    state.mup0 = state.mup.copy()

    weight_calls = []
    apply_calls = []

    def fake_weights(*args, **kwargs):
        weight_calls.append((args, kwargs))
        return np.zeros(5, np.float32), np.zeros(5, np.float32)

    def fake_apply(*args, **kwargs):
        apply_calls.append(kwargs)

    monkeypatch.setattr(lbc, "_resident_weights", fake_weights)
    monkeypatch.setattr(lbc, "apply_specified_relaxation", fake_apply)
    lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=0)
    lbc.apply_state_scalar_lateral_boundary(
        state, cfg, "qv", np.zeros_like(state.qv))

    assert len(weight_calls) == 2
    assert all(call[0][5] == 0.0 for call in weight_calls)
    assert all(call[1]["wrf_real"] is True for call in weight_calls)
    assert apply_calls and all(call["spec_exp"] == 0.0
                               for call in apply_calls)


def test_force_uses_node_parent_clock_and_preserves_parent(monkeypatch):
    import woof.core.nest as nest_mod

    parent, child = _nodes()
    coupler = NestCoupler(child)
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    monkeypatch.setattr(
        nest_mod, "couple_nest_field",
        lambda state, kind, out, **kwargs: out.__setitem__(
            Ellipsis, np_couple_nest_field(state, kind, dtype=np.float32)))
    calls = []

    def fake_bdy(parent_field, child_field, reg, **kwargs):
        calls.append((parent_field.shape, child_field.shape,
                      kwargs["parent_dt_fp32"],
                      kwargs["parent_interval_ticks"]))
        for value, tendency in kwargs["out"].values():
            value[...] = np.float32(1.0)
            tendency[...] = np.float32(2.0)
        return kwargs["out"]

    attached = {}
    monkeypatch.setattr(nest_mod, "bdy_interp1", fake_bdy)
    monkeypatch.setattr(
        nest_mod, "attach_nest_boundaries",
        lambda state, fields, **kwargs: attached.update(fields=fields,
                                                         kwargs=kwargs))
    before = {name: getattr(parent.state, name).copy()
              for name in ("mup", "u", "v", "w", "thp", "php")}
    if WRF_EXACT:
        before = _np_initial_force_round_trip(parent.state)
    assert isinstance(child.clock, DomainClock)
    coupler.force(child)

    assert len(calls) == 6
    assert all(dt == np.float32(9.0) and ticks == 3
               for _, _, dt, ticks in calls)
    assert set(attached["fields"]) == {"u", "v", "w", "theta", "phi", "mu"}
    assert child.clock.dtbc_fp32.view(np.uint32) == np.uint32(0)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(parent.state, name), original)
    assert dict(coupler.transition_receipt()) == {
        "policy_id": "same-scheme-only",
        "source_mp_physics": 0,
        "target_mp_physics": 0,
        "mixed": False,
        "stock_wrf_equivalent": True,
        "source_domain": 1,
        "target_domain": 2,
        "surface_feedback_fields": [],
        "requested_policy": "same-scheme-only",
        "effective_policy": "same-scheme-only",
        "observation_scope": "current_process_since_build_or_restore",
        "process_start_parent_ticks": 0,
        "process_force_count": 1,
        "parent_interval_ticks": 3,
        "final_parent_ticks": 3,
        "expected_cumulative_force_count": 1,
        # The delayed-child work made the expected count relative to when the
        # child actually starts, and publishes that offset so the two numbers
        # can be read together.  This child starts with its parent, so 0.
        "domain_start_ticks": 0,
        "current_process_coverage_complete": True,
        "first_parent_ticks": 3,
        "last_parent_ticks": 3,
        "first_parent_step": 1,
        "last_parent_step": 1,
        "parent_step_count": 1,
        "force_count_matches_parent_steps": True,
    }


def test_attach_nest_boundaries_keeps_mutable_device_views():
    _, child = _nodes()
    state = child.state
    width = 5
    fields = {}
    names = {"u": state.u, "v": state.v, "w": state.w,
             "theta": state.thp, "phi": state.php, "mu": state.mup[None]}
    for name, target in names.items():
        nz, ny, nx = target.shape
        fields[name] = {
            "west": (np.zeros((nz, ny, width), np.float32),
                     np.zeros((nz, ny, width), np.float32)),
            "east": (np.zeros((nz, ny, width), np.float32),
                     np.zeros((nz, ny, width), np.float32)),
            "south": (np.zeros((nz, width, nx), np.float32),
                      np.zeros((nz, width, nx), np.float32)),
            "north": (np.zeros((nz, width, nx), np.float32),
                      np.zeros((nz, width, nx), np.float32)),
        }
    attach_nest_boundaries(state, fields, clock=child.clock,
                           spec_bdy_width=7, spec_zone=1, relax_zone=4)
    resident = state._lateral_boundary_device
    assert resident.rolling and resident.clock is child.clock
    assert state.lateral_boundaries.spec_bdy_width == 5
    assert resident.intervals[0].fields["w"].west.value is \
        fields["w"]["west"][0]


def test_boundary_forced_is_bit_inert_for_legacy_and_true_for_child():
    legacy = _run(9, 9, nested=False, grid_id=1)
    child = _run(9, 9, nested=True, grid_id=2)
    assert _boundary_forced(legacy) is True  # root is specified
    assert _boundary_forced(child) is True
    periodic = RunConfig(nx=9, ny=9, nz=2, dx=1.0, dy=1.0,
                         ztop=1000.0, dt=1.0, run_seconds=1.0)
    assert _boundary_forced(periodic) is False


@requires_gpu
@pytest.mark.gpu
def test_gpu_couple_nest_field_all_kinds_matches_independent_wrf_mirror():
    import cupy as cp

    _, child = _nodes()
    host = child.state
    device = _DeviceState(host, cp)
    fields = ("u", "v", "w", "t", "ph", "mu",
              "qv", "qc", "qr", "qi", "qs", "qg",
              "nr", "ni", "ns", "ng")
    for name in fields:
        target_name = {"t": "thp", "ph": "php"}.get(name, name)
        shape = ((1, *host.mup.shape) if name == "mu"
                 else getattr(host, target_name).shape)
        out = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(device, name, out=out)
        expected = np_couple_nest_field(host, name, dtype=np.float32)
        np.testing.assert_array_equal(cp.asnumpy(out), expected,
                                      err_msg=name)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("width", [1, 5, 32])
@pytest.mark.parametrize("has_msf", [True, False])
def test_gpu_frame_coupling_preserves_words_and_untouched_interior(width, has_msf):
    import cupy as cp

    parent, _ = _nodes()
    host = parent.state
    host.has_msf = has_msf
    rng = np.random.default_rng(20260930)
    for value in vars(host).values():
        if isinstance(value, np.ndarray):
            value[...] += rng.uniform(-0.1, 0.1, value.shape).astype(np.float32)
    host.thb = np.broadcast_to(host.thb[:, None, None], host.thp.shape).copy()
    device = _DeviceState(host, cp)
    for kind in ("u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
                 "qi", "qs", "qg", "nr", "ni", "ns", "ng"):
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        shape = ((1, *host.mup.shape) if kind == "mu"
                 else getattr(host, attr).shape)
        full = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(device, kind, out=full)
        bounded = cp.full(shape, np.float32(-123.25), dtype=cp.float32)
        couple_nest_field(device, kind, out=bounded, frame_width=width)
        j, i = np.indices(shape[-2:])
        mask = np.minimum(np.minimum(j, shape[-2] - 1 - j),
                          np.minimum(i, shape[-1] - 1 - i)) < width
        expected = np.full(shape, np.float32(-123.25), dtype=np.float32)
        expected[:, mask] = cp.asnumpy(full)[:, mask]
        np.testing.assert_array_equal(
            cp.asnumpy(bounded).view(np.uint32), expected.view(np.uint32),
            err_msg=kind)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("window", [
    (3, 9, 4, 11), (-8, 5, -8, 6), (10, 24, 10, 24),
    (-8, 24, -8, 24), (30, 40, 30, 40)])
@pytest.mark.parametrize("has_msf,thb_3d", [(True, False), (False, True)])
def test_gpu_window_coupling_preserves_words_and_untouched_exterior(
        window, has_msf, thb_3d):
    import cupy as cp
    from woof.core.streaming import window_slices

    parent, _ = _nodes()
    host = parent.state
    rng = np.random.default_rng(20260930)
    for name, value in vars(host).items():
        if isinstance(value, np.ndarray):
            value[...] += rng.uniform(-0.1, 0.1, value.shape).astype(np.float32)
    host.has_msf = has_msf
    if thb_3d:
        host.thb = np.broadcast_to(
            host.thb[:, None, None], host.thp.shape).copy()
        host.thb += rng.uniform(-1, 1, host.thp.shape).astype(np.float32)
    device = _DeviceState(host, cp)
    for kind in ("u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
                 "qi", "qs", "qg", "nr", "ni", "ns", "ng"):
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        shape = ((1, *host.mup.shape) if kind == "mu"
                 else getattr(host, attr).shape)
        full = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(device, kind, out=full)
        bounded = cp.full(shape, np.float32(-123.25), dtype=cp.float32)
        couple_nest_field(device, kind, out=bounded, window=window)
        expected = np.full(shape, np.float32(-123.25), dtype=np.float32)
        sl = window_slices(shape, window)
        expected[sl] = cp.asnumpy(full)[sl]
        np.testing.assert_array_equal(
            cp.asnumpy(bounded).view(np.uint32), expected.view(np.uint32),
            err_msg=kind)


@requires_gpu
@pytest.mark.gpu
def test_gpu_force_tables_match_np_nest_force_real_emulation():
    import cupy as cp
    from woof.wrf_exact import ENABLED as wrf_exact

    host_parent, host_child = _nodes()
    if wrf_exact:
        # Strict nesting carries native t_2 = theta - 300. The general
        # fixture also exercises a varying thb in fast mode; that is a
        # different state convention, not a native strict input.
        host_parent.state.thb.fill(np.float32(300.0))
        host_child.state.thb.fill(np.float32(300.0))
    parent = SimpleNamespace(
        cfg=host_parent.cfg, state=_DeviceState(host_parent.state, cp),
        clock=host_parent.clock)
    child = SimpleNamespace(
        cfg=host_child.cfg, state=_DeviceState(host_child.state, cp),
        parent=parent, clock=host_child.clock)
    coupler = NestCoupler(child)
    fields = ("u", "v", "w", "t", "ph", "mu")
    expected = np_nest_force(
        host_parent.state, host_child.state, coupler.registrations,
        field_names=fields, parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5,
        dtype=np.float32)
    before = {name: cp.asnumpy(getattr(parent.state, name)).copy()
              for name in ("mup", "u", "v", "w", "thp", "php")}
    if WRF_EXACT:
        before = _np_initial_force_round_trip(host_parent.state)
    coupler.force(child)

    application_name = {"t": "theta", "ph": "phi"}
    for name in fields:
        got_field = coupler._last_tables[application_name.get(name, name)]
        for side in ("west", "east", "south", "north"):
            for got, want in zip(got_field[side], expected[name][side]):
                np.testing.assert_array_equal(
                    cp.asnumpy(got), np.float32(want),
                    err_msg=f"{name}/{side}")
    for name, original in before.items():
        # Native strict FORCE intentionally leaves couple/uncouple FP32
        # rounding on prognostics. Parent byte identity is a fast-mode
        # contract; the rolling-table comparison above applies to both.
        if wrf_exact and name != "mup":
            continue
        np.testing.assert_array_equal(cp.asnumpy(getattr(parent.state, name)),
                                      original)
    assert child.clock.dtbc_launch_fp32.view(np.uint32) == np.uint32(0)


def _np_initial_force_round_trip(state):
    """Independent REAL coupling/uncoupling of the fixture's initial state.

    WRF v4.6.1 dyn_em/couple_or_uncouple_em.F:119-166,193-245,279-345.
    Strict forcing performs both passes in place. An assertion that its
    input words never change would reject WRF's own rounded arithmetic.
    This initial state has no previous acoustic mass halo.
    WRF's public-domain notice is in licenses/LICENSE-WRF-public-domain.txt.
    """
    assert state.has_msf
    assert np.all(state.thb == np.float32(300))
    f32 = np.float32
    half = ((state.c1h[:, None, None] * state.mub2d[None])
            + state.c2h[:, None, None])
    perturbation = state.c1h[:, None, None] * state.mup[None]
    full = (((state.c1f[:, None, None] * state.mub2d[None])
             + state.c2f[:, None, None])
            + (state.c1f[:, None, None] * state.mup[None]))
    mass = half + perturbation
    inverse_u = np.empty_like(state.u)
    total = ((half[:, :, 1:] + half[:, :, :-1])
             + perturbation[:, :, 1:]) + perturbation[:, :, :-1]
    inverse_u[:, :, 1:-1] = (f32(2) * state.msfu[None, :, 1:-1]) / total
    low = ((half[:, :, 0] + half[:, :, 0])
           + perturbation[:, :, 0]) + perturbation[:, :, 0]
    inverse_u[:, :, 0] = (f32(2) * state.msfu[None, :, 0]) / low
    inverse_u[:, :, -1] = state.msfu[None, :, -1] / mass[:, :, -1]
    inverse_v = np.empty_like(state.v)
    total = ((half[:, 1:, :] + half[:, :-1, :])
             + perturbation[:, 1:, :]) + perturbation[:, :-1, :]
    inverse_v[:, 1:-1, :] = (f32(2) * state.msfv[None, 1:-1, :]) / total
    low = ((half[:, 0, :] + half[:, 0, :])
           + perturbation[:, 0, :]) + perturbation[:, 0, :]
    inverse_v[:, 0, :] = (f32(2) * state.msfv[None, 0, :]) / low
    inverse_v[:, -1, :] = state.msfv[None, -1, :] / mass[:, -1, :]
    return {
        "mup": state.mup.copy(),
        "u": np_couple_nest_field(state, "u") * inverse_u,
        "v": np_couple_nest_field(state, "v") * inverse_v,
        "w": np_couple_nest_field(state, "w") * (state.msft[None] / full),
        "thp": (state.thp * mass) * (f32(1) / mass),
        "php": np_couple_nest_field(state, "ph") * (f32(1) / full),
    }


@requires_gpu
@pytest.mark.gpu
def test_gpu_mp8_to_mp18_force_canonicalizes_every_missing_parent_field():
    """The admitted mixed edge reaches all MP18 boundary endpoints."""
    import cupy as cp

    host_parent, host_child = _nodes()
    host_parent.cfg.run = replace(
        host_parent.cfg.run, mp_physics=8, moist=True, moist_cq=True)
    host_child.cfg.run = replace(
        host_child.cfg.run, mp_physics=18, moist=True, moist_cq=True,
        nest_microphysics_transition=MP8_TO_MP18_POLICY)
    parent_shape = host_parent.state.qv.shape
    host_parent.state.alt = np.full(
        parent_shape, np.float32(1.0 / 1.05), dtype=np.float32)
    child_shape = host_child.state.qv.shape
    for name in (
            "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh",
            "qnn", "qvolg", "qvolh"):
        value = np.zeros(child_shape, dtype=np.float32)
        if name == "qnn":
            value.fill(np.float32(408163264.0))
        setattr(host_child.state, name, value)

    parent = SimpleNamespace(
        cfg=host_parent.cfg, state=_DeviceState(host_parent.state, cp),
        clock=host_parent.clock)
    child = SimpleNamespace(
        cfg=host_child.cfg, state=_DeviceState(host_child.state, cp),
        parent=parent, clock=host_child.clock)
    coupler = NestCoupler(child)
    coupler.force(child)

    assert set(coupler._last_tables) >= {
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
        "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
    }
    receipt = dict(coupler.transition_receipt())
    assert receipt["policy_id"] == MP8_TO_MP18_POLICY
    assert receipt["source_mp_physics"] == 8
    assert receipt["target_mp_physics"] == 18
    assert receipt["process_start_parent_ticks"] == 0
    assert receipt["process_force_count"] == 1
    assert receipt["expected_cumulative_force_count"] == 1
    assert receipt["current_process_coverage_complete"] is True
    assert receipt["first_parent_ticks"] == 3
    assert receipt["last_parent_ticks"] == 3
    processing = receipt["per_species_processing_counts"]
    assert len(processing) == sum(
        receipt["species_action_counts"].values())
    assert all(item["initialization_count"] == 0 for item in processing)
    assert all(item["lateral_forcing_count"] == 1 for item in processing)
    assert all(item["total_edge_processing_count"] == 1
               for item in processing)


@requires_gpu
@pytest.mark.gpu
def test_gpu_nested_w_relaxation_matches_map_factor_free_wrf_mirror():
    import cupy as cp

    _, child = _nodes()
    host = child.state
    device = _DeviceState(host, cp)
    nz, ny, nx = host.w.shape
    width = 5

    def boundary(array_module):
        def side(shape):
            return SimpleNamespace(
                value=array_module.zeros(shape, dtype=array_module.float32),
                tendency=array_module.zeros(shape, dtype=array_module.float32))
        return SimpleNamespace(
            west=side((nz, ny, width)), east=side((nz, ny, width)),
            south=side((nz, width, nx)), north=side((nz, width, nx)))

    host_boundary = boundary(np)
    device_boundary = boundary(cp)
    tendency = cp.zeros_like(device.w)
    fcx, gcx = _weights(
        width, 1, 4, child.clock.spec.dt_fp32, 0.0, wrf_real=True)
    apply_specified_relaxation(
        device.w, tendency, device_boundary, dtbc=np.float32(0.0),
        dt=child.clock.spec.dt_fp32, spec_zone=1, relax_zone=4,
        spec_exp=0.0, apply_relax=True, state=device, field_name="w",
        weights=(cp.asarray(fcx), cp.asarray(gcx)), clear_specified=True,
        source_field=device.w)

    current = np_w_relaxation_current(host, dtype=np.float32)
    expected = np_specified_relaxation(
        current, np.zeros_like(current), host_boundary, dtbc=0.0,
        dt=float(child.clock.spec.dt_fp32), spec_zone=1, relax_zone=4,
        spec_exp=0.0, apply_relax=True)
    wrong = np_specified_relaxation(
        np_couple_nest_field(host, "w", dtype=np.float32),
        np.zeros_like(current), host_boundary, dtbc=0.0,
        dt=float(child.clock.spec.dt_fp32), spec_zone=1, relax_zone=4,
        spec_exp=0.0, apply_relax=True)
    got = cp.asnumpy(tendency)
    np.testing.assert_allclose(got, expected, rtol=2.0e-5, atol=2.0e-6)
    point = (1, 1, nx // 2)
    assert abs(got[point] - expected[point]) < abs(got[point] - wrong[point])


# --------------------------------------------------------------------------
# A COUPLED DOMAIN THAT IS STREAMED: node.state is not the domain
# --------------------------------------------------------------------------

def _publish(state, arrays):
    """Give ``state`` a streaming store, the way ``streaming.attach`` does."""
    from woof.core import streaming

    setattr(state, streaming._STORE_ATTR,
            {f"state/{name}": value for name, value in arrays.items()})


def test_force_composes_both_store_corridors(monkeypatch):
    """The coupler reads both live stores without an edge-shape policy."""
    import woof.core.nest as nest_mod

    parent, child = _nodes()
    coupler = NestCoupler(child)
    for node in (parent, child):
        _publish(node.state, {name: value.copy() for name, value in
                             vars(node.state).items() if isinstance(value, np.ndarray)})
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    monkeypatch.setattr(nest_mod, "bdy_interp1", lambda *a, **k: k["out"])
    monkeypatch.setattr(nest_mod, "attach_nest_boundaries", lambda *a, **k: None)
    monkeypatch.setattr(nest_mod, "couple_nest_field", lambda state, kind, out, **kwargs: out)
    seen = []
    original = nest_mod._sync_in
    def record(state, attrs, window=None):
        seen.append((state, window))
        return original(state, attrs, window)
    monkeypatch.setattr(nest_mod, "_sync_in", record)
    coupler.force(child)
    assert coupler.force_count == 1
    assert any(state is parent.state and window is not None for state, window in seen)
    assert any(state is child.state and window is not None for state, window in seen)


@pytest.mark.parametrize("smooth_option", [0, 1, 2])
@pytest.mark.parametrize("start", [False, True])
def test_feedback_finalize_commits_only_recomputed_diagnostics(
        monkeypatch, smooth_option, start):
    """A windowed diagnosis must never overwrite live off-window store cells."""
    import woof.core.diagnostics as diagnostics
    from woof.core import streaming

    parent, child = _nodes()
    coupler = NestCoupler(child, feedback=1, smooth_option=smooth_option)
    coupler._prepared_feedback = {}
    live = {}
    for index, name in enumerate(("p", "al", "alt")):
        setattr(parent.state, name, np.full_like(parent.state.thp, -99.))
        live[name] = np.arange(parent.state.thp.size, dtype=np.float32).reshape(
            parent.state.thp.shape) + index
    _publish(parent.state, live)
    expected = {name: value.copy() for name, value in live.items()}
    seen = []
    def diagnose(state, option, *, window):
        seen.append(window)
        j0, i0, nj, ni = window
        sl = (Ellipsis, slice(j0, j0 + nj), slice(i0, i0 + ni))
        for name in expected:
            getattr(state, name)[sl] = 42.
            expected[name][sl] = 42.
    monkeypatch.setattr(diagnostics, "update_diagnostics", diagnose)
    coupler.feedback_finalize(child, start=start)
    if WRF_EXACT and not start:
        # WRF's per-step update_after_feedback_em exchanges halos only.
        # Diagnosis at nest start remains required in both arithmetic modes.
        assert seen == []
    else:
        assert len(seen) == 1
        assert seen[0][2] < parent.cfg.run.ny and seen[0][3] < parent.cfg.run.nx
    for name in expected:
        np.testing.assert_array_equal(live[name], expected[name])
    assert coupler._prepared_feedback is None


def test_force_reads_the_parent_STORE_and_not_the_frozen_state(monkeypatch):
    """The defect, as a unit: a streamed parent's state is at attach time.

    The store here carries a mass field that differs from the one on the
    state, exactly as it does after one sweep.  ``force`` must couple the
    STORE's numbers.  With the store unpublished -- the code path before
    this existed -- it couples the state's, and that is asserted too, so the
    test fails if the fix stops mattering rather than passing vacuously.
    """
    import woof.core.nest as nest_mod

    parent, child = _nodes()
    coupler = NestCoupler(child)
    monkeypatch.setattr(coupler, "_bind_geometry", lambda: None)
    monkeypatch.setattr(nest_mod, "bdy_interp1",
                        lambda *a, **k: k["out"])
    monkeypatch.setattr(nest_mod, "attach_nest_boundaries",
                        lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(
        nest_mod, "couple_nest_field",
        lambda state, kind, out, **kwargs: (seen.append((kind, float(state.mup[0, 0]))),
                                  out)[1])

    swept = parent.state.mup.copy() + np.float32(7.0)
    _publish(parent.state, {"mup": swept})
    # The REAL seam, no stand-in.  ``refresh_from_store`` used to be
    # CuPy-only (``ndarray.set``), which is why a fake stood here; the
    # windowed corridor gave it a host-mirror branch, so the seam under
    # test is the seam that ships -- window arithmetic, clamp rule and the
    # resident no-op included.  The probe cell (0, 0) is INSIDE the
    # clamped footprint window of this fixture's geometry (child origin 3,
    # span 10, halo 8 on a 16^2 parent covers the whole parent), so a
    # windowed pull must still deliver the store's number there; the
    # windowed-vs-stale distinction has its own GPU gate
    # (tilestream/test_nest_executor.py, halo-starved control).

    coupler.force(child)
    parent_mu = [mu for kind, mu in seen if kind == "mu"]
    assert parent_mu, "no parent field was coupled"
    assert parent_mu[0] == pytest.approx(float(swept[0, 0]))
    assert coupler.force_count == 1
    assert coupler.force_sync_bytes > 0, (
        "the FORCE corridor moved no bytes for a published store; the "
        "receipt is dead and the windowed pull cannot be audited")


def test_resident_domains_never_consult_a_store():
    """``domain_store`` is ``None`` for every state that never streamed."""
    from woof.core.streaming import (commit_to_store, domain_store,
                                      refresh_from_store)

    _parent, child = _nodes()
    assert domain_store(child.state) is None
    assert refresh_from_store(child.state, ("mup", "u")) == 0
    assert commit_to_store(child.state, ("mup", "u")) == 0
