"""DA lane 7: the increment as an RK-stage tendency, its hydrostatic php,
and the MASPT recorder.  CPU arithmetic here; the dycore tests are in
tests/test_da_iau_forcing_gpu.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.da import iau
from woof.da.maspt import MasptRecorder


def test_box_density_integrates_to_one_over_any_step_partition():
    knots = iau.box_density(100.0, 1800.0)
    rng = np.random.default_rng(1)
    edges = np.concatenate([[0.0], np.sort(rng.uniform(0, 2500, 57)),
                            [2500.0]])
    weights = [iau.integrate_density(knots, a, b)
               for a, b in zip(edges[:-1], edges[1:])]
    assert abs(sum(weights) - 1.0) < 1e-12
    assert iau.integrate_density(knots, 0.0, 100.0) == 0.0
    assert abs(iau.integrate_density(knots, 100.0, 1000.0) - 0.5) < 1e-12


def test_interpolation_densities_are_a_partition_of_one():
    dens = iau.interpolation_densities([-1800.0, -900.0, 0.0],
                                       -1800.0, 1800.0)
    shares = [iau.integrate_density(k, -np.inf, np.inf) for k in dens]
    assert abs(sum(shares) - 1.0) < 1e-12
    # the last slot holds beyond its time: it owns the whole second half
    assert shares[-1] > 0.5
    for t in np.linspace(-1800, 1800, 41)[:-1]:
        total = sum(iau.integrate_density(k, t, t + 90.0) for k in dens)
        assert abs(total - 90.0 / 3600.0) < 1e-12


def test_interpolation_densities_refuse_slots_off_the_window():
    with pytest.raises(iau.IauError):
        iau.interpolation_densities([0.0, 4000.0], 0.0, 3600.0)
    with pytest.raises(iau.IauError):
        iau.interpolation_densities([10.0, 5.0], 0.0, 3600.0)


def test_resolve_iau_mode():
    assert iau.resolve_iau_mode(None, None) == "oneshot"
    assert iau.resolve_iau_mode(None, 900.0) == "split"
    assert iau.resolve_iau_mode("3d", 1800.0) == "3d"
    with pytest.raises(iau.IauError):
        iau.resolve_iau_mode("3d", None)
    with pytest.raises(iau.IauError):
        iau.resolve_iau_mode("oneshot", 900.0)


def _stub_state(nz=3, ny=4, nx=5):
    z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
    return SimpleNamespace(
        thp=z(nz, ny, nx), u=z(nz, ny, nx + 1), v=z(nz, ny + 1, nx),
        w=z(nz + 1, ny, nx), php=z(nz + 1, ny, nx), qv=None,
        mup=np.full((ny, nx), 100.0, np.float32),
        mub2d=np.full((ny, nx), 9.0e4, np.float32),
        c1h=np.ones(nz, np.float32), c2h=np.zeros(nz, np.float32),
        c1f=np.ones(nz + 1, np.float32), c2f=np.zeros(nz + 1, np.float32),
        has_msf=False, elapsed_seconds=0.0,
        rth_t=z(nz, ny, nx), ru_t=z(nz, ny, nx + 1), rv_t=z(nz, ny + 1, nx),
        rph_t=z(nz + 1, ny, nx), rw_t=z(nz + 1, ny, nx))


def test_fold_couples_by_column_mass_and_sums_to_the_increment():
    state = _stub_state()
    dth = np.ones_like(state.thp)
    forcing = iau.IauForcing(
        state, [iau.IauNode({"thp": dth}, iau.box_density(0.0, 300.0))])
    iau.attach(state, forcing)
    cfg = SimpleNamespace(dt=45.0, specified=False, nested=False)
    added = np.zeros_like(state.thp, dtype=np.float64)
    t = 0.0
    while t < 400.0:
        state.elapsed_seconds = t
        state.rth_t[...] = 0.0
        tend = iau.fold_step_tendencies(state, cfg, None)
        if tend is not None:
            tend.add_to_slow(state)
            # uncouple and integrate as the RK final stage does
            added += cfg.dt * state.rth_t / (9.0e4 + 100.0)
        t += cfg.dt
    assert abs(forcing.applied_fraction - 1.0) < 1e-12
    np.testing.assert_allclose(added, 1.0, rtol=2e-6)
    receipt = forcing.receipt()
    assert receipt["steps_forced"] == 7       # 0..315 overlaps [0, 300]


def test_fold_outside_the_window_returns_the_physics_object_itself():
    state = _stub_state()
    forcing = iau.IauForcing(
        state, [iau.IauNode({"thp": np.ones_like(state.thp)},
                            iau.box_density(600.0, 300.0))])
    sentinel = object()
    state.elapsed_seconds = 0.0
    assert forcing.fold(state, SimpleNamespace(dt=60.0), sentinel) is sentinel


def test_specified_ring_is_excluded_like_the_physics_couplers():
    state = _stub_state()
    forcing = iau.IauForcing(state, [iau.IauNode(
        {"thp": np.ones_like(state.thp), "u": np.ones_like(state.u)},
        iau.box_density(0.0, 60.0))])
    tend = forcing.fold(state, SimpleNamespace(dt=60.0, specified=True,
                                               nested=False), None)
    rth = tend.slow["rth_t"]
    assert (rth[:, 0, :] == 0).all() and (rth[:, :, -1] == 0).all()
    assert (rth[:, 1:-1, 1:-1] > 0).all()
    ru = tend.slow["ru_t"]
    assert (ru[:, :, 0] == 0).all() and (ru[:, :, -1] == 0).all()


def test_a_boundary_zone_left_to_the_boundary_is_masked_on_every_field():
    state = _stub_state(nz=2, ny=8, nx=9)
    ones = {name: np.ones_like(getattr(state, name)) for name in ("thp", "u", "v", "w", "php")}
    cfg = SimpleNamespace(dt=60.0, specified=True, nested=False)
    ring = iau.IauForcing(state, [iau.IauNode(ones, iau.box_density(0.0, 60.0))]).fold(state, cfg, None)
    zone = iau.IauForcing(state, [iau.IauNode(ones, iau.box_density(0.0, 60.0))],
                          boundary_rows=3).fold(state, cfg, None)
    # the default leaves the specified ring and leaves php and w alone (the physics rule)
    assert (ring.slow["rth_t"][:, 1, 1] > 0).all() and (ring.slow["rph_t"][:, 0, 0] > 0).all()
    for slot in ("rth_t", "ru_t", "rv_t", "rph_t", "rw_t"):
        field = zone.slow[slot]
        assert not field[..., :3, :].any() and not field[..., -3:, :].any(), slot
        assert not field[..., :, :3].any() and not field[..., :, -3:].any(), slot
        assert (field[..., 3:-3, 3:-3] > 0).all(), slot
    with pytest.raises(iau.IauError):
        iau.IauForcing(state, [iau.IauNode(ones, iau.box_density(0.0, 60.0))], boundary_rows=0)


def test_shares_must_sum_to_one():
    state = _stub_state()
    half = ((0.0, 0.5 / 60.0), (60.0, 0.5 / 60.0))
    with pytest.raises(iau.IauError):
        iau.IauForcing(state, [iau.IauNode({"thp": np.ones_like(state.thp)},
                                           half)])


def test_a_field_the_step_never_reaches_is_refused():
    state = _stub_state()
    state.mystery = np.zeros_like(state.thp)
    with pytest.raises(iau.IauError):
        iau.IauForcing(state, [iau.IauNode(
            {"mystery": np.ones_like(state.thp)},
            iau.box_density(0.0, 60.0))])


def test_step_tendencies_sum_scalars_with_the_physics_without_writing_it():
    base_qv = np.full((2, 2, 2), 3.0, np.float32)
    base = SimpleNamespace(scalar_for=lambda n: base_qv if n == "qv" else None,
                           add_to_slow=lambda s: None)
    own = np.full((2, 2, 2), 2.0, np.float32)
    tend = iau.IauStepTendencies(base, {}, {"qv": own, "qr": own})
    np.testing.assert_array_equal(tend.scalar_for("qv"), 5.0)
    np.testing.assert_array_equal(base_qv, 3.0)
    assert tend.scalar_for("qr") is own
    assert tend.scalar_for("qc") is None


def test_maspt_recorder_measures_the_lowest_level_tendency():
    nz, ny, nx = 2, 12, 12
    state = SimpleNamespace(p=np.full((nz, ny, nx), 1.0e5),
                            mup=np.zeros((ny, nx)),
                            thp=np.zeros((nz, ny, nx)),
                            thb=np.full(nz, 300.0), elapsed_seconds=0.0)
    rec = MasptRecorder(start_seconds=0.0, horizon_seconds=600.0, margin=2)

    def step(s, run, **kw):
        s.p[0] += 100.0          # 1 hPa a step
        s.p[0, 0, 0] += 1e6      # boundary noise, outside the interior
        s.elapsed_seconds += run.dt

    stepped = rec.stepper(step)
    run = SimpleNamespace(dt=60.0)
    for _ in range(12):
        stepped(state, run)
    receipt = rec.receipt()
    assert len(receipt["series"]["maspt"]) == 10   # horizon is 600 s
    assert abs(receipt["maspt_first_30min"] - 60.0) < 1e-9   # hPa/h
    assert receipt["insertion_jump"] is None


def test_maspt_recorder_measures_the_insertion_jump():
    nz, ny, nx = 2, 8, 8
    state = SimpleNamespace(p=np.full((nz, ny, nx), 1.0e5),
                            mup=np.zeros((ny, nx)),
                            thp=np.zeros((nz, ny, nx)),
                            thb=np.full(nz, 300.0), elapsed_seconds=0.0)
    rec = MasptRecorder(start_seconds=0.0, horizon_seconds=600.0)
    rec.before_insertion(state)
    state.thp[0] += 2.0                       # the analysis warms 2 K
    state.p[0] += 50.0

    def step(s, run, **kw):
        s.elapsed_seconds += run.dt

    rec.stepper(step)(state, SimpleNamespace(dt=60.0))
    jump = rec.receipt()["insertion_jump"]
    from woof.core import constants as c
    expected = 302.0 * (100050.0 / c.P0) ** c.RCP - 300.0
    assert abs(jump["lowest_level_t_max_abs_k"] - expected) < 1e-9
    assert abs(jump["lowest_level_p_mean_abs_hpa"] - 0.5) < 1e-12


# The 4D leg runner these tests drove was the local data assimilation cycle
# driver's, removed in gpuwm 3.0.3.


def test_node_groups_share_one_each():
    state = _stub_state()
    one = np.ones_like(state.thp)
    forcing = iau.IauForcing(state, [
        iau.IauNode({"thp": one}, iau.box_density(0.0, 600.0)),
        iau.IauNode({"thp": one}, iau.box_density(300.0, 60.0),
                    group="hydrometeors")])
    assert set(forcing.groups) == {"analysis", "hydrometeors"}


# -- the tiled road: the forcing served compute window by compute window -------------------------
#
# The breakage: a streamed domain steps tile buffers, never the DomainState the door attached the
# forcing to, so a DA leg that fell to [tiles] integrated its whole window with nothing forced (G10
# 0.0, camp-us-20230710-r n5q15 cycle 10:00 mem003, 2026-10-27).  A window's fold must give the
# resident words on every point it owns, and the lease must book each domain step once.

def _random_domain(nz=3, ny=23, nx=29, seed=7):
    rng = np.random.default_rng(seed)
    r = lambda *s: rng.uniform(0.5, 1.5, s).astype(np.float32)  # noqa: E731
    z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
    return SimpleNamespace(
        thp=r(nz, ny, nx), u=r(nz, ny, nx + 1), v=r(nz, ny + 1, nx),
        w=r(nz + 1, ny, nx), php=r(nz + 1, ny, nx), qv=None,
        mup=(100.0 * r(ny, nx)), mub2d=(9.0e4 * r(ny, nx)),
        c1h=r(nz), c2h=r(nz), c1f=r(nz + 1), c2f=r(nz + 1),
        msfu=r(ny, nx + 1), msfv=r(ny + 1, nx), msft=r(ny, nx), has_msf=True,
        elapsed_seconds=0.0,
        rth_t=z(nz, ny, nx), ru_t=z(nz, ny, nx + 1), rv_t=z(nz, ny + 1, nx),
        rph_t=z(nz + 1, ny, nx), rw_t=z(nz + 1, ny, nx))


_STAGGER = {"u": "u", "ru_t": "u", "msfu": "u", "v": "v", "rv_t": "v", "msfv": "v"}


def _window_state(domain, spec):
    out = SimpleNamespace(has_msf=domain.has_msf, qv=None, elapsed_seconds=domain.elapsed_seconds)
    for name, value in vars(domain).items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            setattr(out, name, iau.window_of(np, value, spec, _STAGGER.get(name, "mass")).copy())
        elif isinstance(value, np.ndarray):
            setattr(out, name, value)
    return out


def _owned(array, spec, variant, local=False):
    """The window's owned interior (mass columns, and every face between them) in domain or local
    coordinates: the points the scatter writes, whose face neighbours are always in the window."""
    ey, ex = (1 if variant == "v" else 0), (1 if variant == "u" else 0)
    y0, x0 = (spec.cj0, spec.ci0) if local else (0, 0)
    return array[..., spec.j0 - y0:spec.j1 + ey - y0, spec.i0 - x0:spec.i1 + ex - x0]


@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("flags", [dict(specified=True), dict(open_x=True, open_y=True), {}])
def test_a_tiled_sweep_forces_the_resident_words_on_every_owned_point(rows, flags):
    from tilestream.spec import plan_tiles

    domain = _random_domain()
    rng = np.random.default_rng(11)
    names = ("thp", "u", "v", "w", "php")
    inc = {name: rng.normal(size=getattr(domain, name).shape).astype(np.float32) for name in names}
    knots = iau.box_density(0.0, 600.0)
    cfg = SimpleNamespace(**{"dt": 30.0, "ny": 23, "nx": 29, "specified": False, "nested": False, **flags})
    resident = iau.IauForcing(domain, [iau.IauNode(dict(inc), knots)], boundary_rows=rows)
    streamed = iau.IauForcing(domain, [iau.IauNode(dict(inc), knots)], boundary_rows=rows)
    specs = plan_tiles(29, 23, 8, 7, halo=4, periodic=False)
    lease = iau.IauSweepLease(cfg, forcing=streamed)
    for step in range(22):                           # past the window's end at 600 s
        domain.elapsed_seconds = 30.0 * step
        full = resident.fold(domain, cfg, None)
        lease.begin(windows=len(specs))
        for k, spec in enumerate(specs):
            window = _window_state(domain, spec)
            lease.bind_window(window, spec, k)
            got = window._iau_forcing.fold(window, cfg, None)
            if full is None:
                assert got is None
                continue
            for slot, value in full.slow.items():
                variant = _STAGGER.get(slot, "mass")
                np.testing.assert_array_equal(_owned(got.slow[slot], spec, variant, local=True),
                                              _owned(value, spec, variant), err_msg=f"{slot} tile {k}")
        lease.finish()
        assert streamed.steps_forced == resident.steps_forced
        assert [n.applied for n in streamed.nodes] == [n.applied for n in resident.nodes]
    assert streamed.applied_fraction == resident.applied_fraction
    assert abs(streamed.applied_fraction - 1.0) < 1e-12


def test_a_sweep_that_forced_part_of_the_domain_is_never_booked():
    from tilestream.spec import plan_tiles

    domain = _random_domain()
    cfg = SimpleNamespace(dt=30.0, ny=23, nx=29, specified=True, nested=False)
    forcing = iau.IauForcing(domain, [iau.IauNode({"thp": np.ones_like(domain.thp)},
                                                  iau.box_density(0.0, 600.0))])
    specs = plan_tiles(29, 23, 8, 7, halo=4, periodic=False)
    lease = iau.IauSweepLease(cfg, forcing=forcing)
    lease.begin(windows=len(specs))
    window = _window_state(domain, specs[0])
    lease.bind_window(window, specs[0], 0)
    window._iau_forcing.fold(window, cfg, None)
    with pytest.raises(iau.IauError):
        lease.finish()
    assert forcing.applied_fraction == 0.0 and forcing.steps_forced == 0


def test_the_lease_follows_the_state_and_is_released_with_it():
    domain = _random_domain()
    run = SimpleNamespace(cfg=SimpleNamespace(ny=23, nx=29))
    iau.attach_iau_sweep_lease(run, domain)
    assert getattr(run, "_iau_sweep_lease", None) is None
    forcing = iau.IauForcing(domain, [iau.IauNode({"thp": np.ones_like(domain.thp)},
                                                  iau.box_density(0.0, 600.0))])
    iau.attach(domain, forcing)
    iau.attach_iau_sweep_lease(run, domain)
    lease = run._iau_sweep_lease
    assert lease.forcing is forcing
    iau.attach_iau_sweep_lease(run, domain)
    assert run._iau_sweep_lease is lease
    iau.detach(domain)
    iau.attach_iau_sweep_lease(run, domain)
    assert run._iau_sweep_lease is None
