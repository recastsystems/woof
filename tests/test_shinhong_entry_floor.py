"""Shin-Hong's ENTRY TKE contract: the driver enforces WRF's floor.

The composed-suite fatality ``FloatingPointError: Shin-Hong returned
non-finite tke tendency`` (task #206, and a field report against the
pre-2.2.1 line) has exactly one producing input class: entry TKE at or
below zero, or non-finite, at the chain's entry.  Proven two ways:

* the pinned hazard (tests/test_shinhong_composed_tke_guard.py): zero
  entry TKE sends ``mixlen``'s ``el0 = alph*szq*0.5/sq`` to 0/0 (``sq``
  is an integral of ``sqrt(q2)``), the NaN walks el -> disel -> prodq2's
  q2, and the published (tke, el) pair goes non-finite while every
  tendency stays finite -- the exact crash signature;
* MEASURED (work/probe_shinhong_big_ensemble.py, 2026-08-16): 24,000
  adversarial columns spanning stable/neutral/convective/weak-inversion
  regimes, degenerate surface inputs (hfx = 0, ust = 0, calm, subnormal
  fluxes) and evolved TKE profiles, all with entry TKE >= the floor:
  ZERO tke-confined non-finites.  WRF's own floors (epsgm/epsgh/epsru,
  the requ threshold, the QNSE overwrite, prodq2's amax1) close every
  internal lane; only the entry is unguarded.

WRF guarantees the entry invariant with ``shinhonginit`` (TKE filled at
epsq2l/2) plus the scheme's own amax1 floors; woof's cold start and
writeback maintain it too.  But ``_run_shinhong`` used to trust
``state.e_sgs`` unconditionally, so any OTHER writer -- a hand-carried
restart, an analysis state, an external tool, a bit-flip on a no-ECC
card, an injected experiment state -- could feed the chain the one input
class that reproduces the crash (pre-2.2.1: fatal at the first PBL call;
since 2.2.1: a garbage chain step healed after the fact by the passenger
repair).  The contract under test: the driver re-asserts WRF's
shinhonginit floor AT THE SEAM, before the launch, loudly once --
upstream of the guard, which stays.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.shinhong import SHINHONG_TKE_FLOOR

F = np.float32

TENDENCY_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                    "rqcblten", "rqiblten", "exch_h")


# ---------------------------------------------------------------------------
# The input class, pinned through the float32 WRF authority: the hazard
# is entry TKE <= 0 or non-finite, and NOT the merely-sub-floor band.
# ---------------------------------------------------------------------------

def _convective_column(kte=49):
    """The task #206 convective column (test_shinhong_composed_tke_guard)."""
    z = np.cumsum(np.full(kte, 250.0, np.float64))
    dz8w = np.full(kte, 250.0, F)
    psfc = F(97000.0)
    p = (psfc * np.exp(-z / 8000.0)).astype(F)
    p2di = np.empty(kte + 1, F)
    p2di[0] = psfc
    p2di[1:] = (psfc * np.exp(-(z + 125.0) / 8000.0)).astype(F)
    pi2d = (p / F(100000.0)) ** F(0.2854)
    t = (302.0 - 0.0075 * z).astype(F)
    args = (np.full(kte, 8.0, F), np.full(kte, 3.0, F), t,
            (0.014 * np.exp(-z / 3000.0)).astype(F),
            np.zeros(kte, F), np.zeros(kte, F), p, pi2d, p2di, dz8w)
    surface = dict(psfc=psfc, znt=F(0.1), ust=F(0.45), hfx=F(250.0),
                   qfx=F(1.2e-4), wspd=F(8.5), br=F(-0.35), psim=F(1.2),
                   psih=F(1.5), xland=F(1.0), corf=F(1.0e-4),
                   u10=F(7.0), v10=F(2.5), dt=F(72.0), dx=F(12000.0),
                   dy=F(12000.0), tke_diag=1)
    return args, surface


@pytest.mark.parametrize("value", [0.0, -0.25, np.nan],
                         ids=["zero", "negative", "nan"])
def test_entry_at_or_below_zero_or_nan_is_the_hazard_class(value):
    """Each class member reproduces the crash signature: non-finite
    confined to the passenger pair, every tendency finite beside it."""
    from woof.verify.shinhong_ref import np_shinhong_column

    args, surface = _convective_column()
    out = np_shinhong_column(*args, np.full(49, value, F), **surface)
    assert not np.isfinite(out["tke"]).all(), (
        "the hazard member vanished; re-derive the class before touching "
        "the driver contract")
    for name in TENDENCY_OUTPUTS:
        assert np.isfinite(out[name]).all(), name
    assert np.isfinite(out["hpbl"]) and np.isfinite(out["wstar"])


def test_sub_floor_positive_entry_is_out_of_domain_but_not_the_hazard():
    """(0, floor) entries stay finite -- the chain's own amax1 floors
    absorb them.  They are still outside WRF's shinhonginit domain, so
    the driver heals them too, but the CLASS boundary of the crash is
    <= 0 / non-finite and this pin keeps that boundary accurate."""
    from woof.verify.shinhong_ref import np_shinhong_column

    args, surface = _convective_column()
    out = np_shinhong_column(
        *args, np.full(49, F(0.5) * SHINHONG_TKE_FLOOR, F), **surface)
    for name in TENDENCY_OUTPUTS + ("tke", "el"):
        assert np.isfinite(out[name]).all(), name


# ---------------------------------------------------------------------------
# The driver contract: the REAL _run_shinhong body heals the entry at
# the seam (launch/validation stubbed, the disposition-test idiom).
# ---------------------------------------------------------------------------

def _seam_stubbed_driver(monkeypatch, e_sgs):
    import types

    from woof.core import physics as ph

    kte, ny, nx = 4, 2, 3
    shape = (kte, ny, nx)

    def fake_out():
        names = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h",
                 "tke", "el")
        out = {name: np.full(shape, 0.25, F) for name in names}
        out.update(hpbl=np.full((ny, nx), 700.0, F),
                   kpbl=np.full((ny, nx), 2, np.int32),
                   wstar=np.full((ny, nx), 1.0, F),
                   delta=np.full((ny, nx), 100.0, F))
        return out

    class FakeState:
        f = np.full((ny, nx), 1.0e-4, F)

        def scratch(self, shape, name):
            class _V:
                def view(self, dtype):
                    return np.zeros(shape, dtype=np.uint32)
            return _V()

    FakeState.e_sgs = e_sgs

    class FakeTendencies:
        def materialize(self, components):
            pass

    fields = {name: np.full((ny, nx), 0.5, F)
              for name in ("znt", "ust", "hfx", "qfx", "wspd", "br",
                           "fm", "fh", "xland", "u10", "v10", "pblh",
                           "kpbl")}
    fields["exch_h"] = np.full(shape, 0.5, F)
    driver = types.SimpleNamespace(
        state=FakeState(),
        fields=fields,
        terrain_drag=None,
        bldt_seconds=72.0,
        _shinhong_passenger_advisory=False,
        _shinhong_entry_advisory=False,
        pbl_tendencies=FakeTendencies(),
        last_ysu=None,
        gf_rthblten=np.zeros(shape, F),
        gf_rqvblten=np.zeros(shape, F),
    )
    # THE REAL SEAM, BOUND TO THE FAKE.  _run_shinhong ends its due call
    # at PhysicsDriver._couple_pbl_slot, which mass-couples the raw rates
    # AND retains RTHBLTEN/RQVBLTEN for the cumulus scheme -- one call,
    # so that a PBL scheme cannot be wired up correctly and still starve
    # Grell-Freitas.  A stub here would let this suite go green while the
    # seam it stands in for was gone, which is exactly what a
    # SimpleNamespace predating the method already did once.
    driver._couple_pbl_slot = types.MethodType(
        ph.PhysicsDriver._couple_pbl_slot, driver)
    # golden/exact: every PBL arm runs WRF's post-PBL scalar_pblmix after
    # its coupling; the real coupler, at scalar_pblmix = 0, does nothing.
    driver._couple_scalar_pblmix = types.MethodType(
        ph.PhysicsDriver._couple_scalar_pblmix, driver)
    # tracer_opt/madwrf_opt: the scalar PBL-mix coupler reads both since the
    # tracer and MAD-WRF ports (RunConfig defaults, 0).
    cfg = types.SimpleNamespace(dx=12000.0, dy=12000.0, bl_pbl_physics=11,
                                scalar_pblmix=0, tracer_opt=0, madwrf_opt=0,
                                tracer_pblmix=1)
    atmosphere = {name: np.full(shape, 0.25, F)
                  for name in ("u", "v", "theta", "qv", "qc", "qi",
                               "pressure", "exner", "dz")}
    atmosphere["p_interface"] = np.full((kte + 1, ny, nx), 900e2, F)

    captured = {}

    def fake_launch(*args, **kwargs):
        # Identity AND a value snapshot: the real body writes out["tke"]
        # back into the same array afterwards, so what the launch SAW is
        # only observable through a copy taken here.
        captured["tke"] = args[10]
        captured["tke_values"] = np.array(args[10], copy=True)
        out = fake_out()
        captured["out"] = out
        return out

    monkeypatch.setattr(ph, "launch_shinhong", fake_launch)
    monkeypatch.setattr(ph, "invalid_shinhong_outputs",
                        lambda out, status: ())
    monkeypatch.setattr(
        ph, "couple_ysu_tendencies", lambda state, cfg, out: FakeTendencies())
    monkeypatch.setattr(
        ph, "_composed_optional_tendency_components", lambda cfg: ())
    monkeypatch.setattr(
        ph, "_pbl_optional_tendency_components", lambda cfg: ())
    monkeypatch.setattr(
        ph, "physics_reuses_pbl_composition", lambda cfg: False)
    return ph, driver, atmosphere, cfg, captured


def test_driver_heals_degenerate_entry_before_launch(monkeypatch, capsys):
    """Zero/negative/NaN/sub-floor entries reach the launch AT the WRF
    floor, healed in place, healthy values untouched, said once."""
    kte, ny, nx = 4, 2, 3
    e = np.full((kte, ny, nx), 0.25, F)
    e[0, 0, 0] = F(0.0)                       # the injected-cliff member
    e[1, 0, 1] = F(-3.0)                      # negative
    e[2, 1, 2] = np.nan                       # non-finite
    e[3, 1, 0] = F(0.5) * SHINHONG_TKE_FLOOR  # sub-floor positive
    ph, driver, atmosphere, cfg, captured = _seam_stubbed_driver(
        monkeypatch, e)

    ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)

    assert captured["tke"] is driver.state.e_sgs, (
        "the heal must be in place on state.e_sgs -- the restart stream "
        "and the D1 instrument read the same array the launch consumed")
    sent = captured["tke_values"]
    assert np.isfinite(sent).all()
    assert (sent >= F(SHINHONG_TKE_FLOOR)).all()
    assert sent[0, 0, 0] == F(SHINHONG_TKE_FLOOR)
    assert sent[1, 0, 1] == F(SHINHONG_TKE_FLOOR)
    assert sent[2, 1, 2] == F(SHINHONG_TKE_FLOOR)
    assert sent[3, 1, 0] == F(SHINHONG_TKE_FLOOR)
    # every healthy value is bit-untouched
    assert (sent == F(0.25)).sum() == sent.size - 4
    assert driver._shinhong_entry_advisory is True
    # The due call really went through the coupling/retention seam: the
    # PRE-coupling theta and qv rates are what a Grell-Freitas call at
    # the top of the next step reads.
    np.testing.assert_array_equal(driver.gf_rthblten, captured["out"]["dtheta"])
    np.testing.assert_array_equal(driver.gf_rqvblten, captured["out"]["dqv"])
    err = capsys.readouterr().err
    assert "entry" in err and "4" in err and "shinhonginit" in err

    # Second call, now-healthy state (the writeback filled it with the
    # fake launch's finite tke): silent, values passed through unchanged.
    before = driver.state.e_sgs.copy()
    ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)
    assert capsys.readouterr().err == ""
    np.testing.assert_array_equal(captured["tke_values"], before)


def test_driver_healthy_entry_is_untouched_and_silent(monkeypatch, capsys):
    """The floor is the identity on every legal state: no copy, no write,
    no advisory -- a healthy run's trajectory is bit-identical."""
    kte, ny, nx = 4, 2, 3
    e = np.full((kte, ny, nx), F(SHINHONG_TKE_FLOOR), F)
    e[2] = F(3.75)
    original = e.copy()
    ph, driver, atmosphere, cfg, captured = _seam_stubbed_driver(
        monkeypatch, e)

    ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)

    assert captured["tke"] is e
    np.testing.assert_array_equal(captured["tke_values"], original)
    assert driver._shinhong_entry_advisory is False
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# End to end on the real engine: the injected #206 cliff, healed at the
# seam -- the chain never computes the NaN step the passenger repair
# used to clean up after.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_injected_cliff_micro_run_completes_without_needing_the_repair():
    """e_sgs = 0 at the first PBL call (the exact #206 field-crash
    injection, fatal on every pre-2.2.1 build): the run completes, the
    carrier is floored and MOVED, the entry advisory fired, and the
    passenger repair was never needed -- the heal is upstream of it.
    """
    import cupy as cp

    from woof.config import RunConfig, validate_run_config
    from woof.core.dycore import run_steps
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.core.state import init_at_rest

    cfg = validate_run_config(RunConfig(
        nx=8, ny=8, nz=16, dx=500.0, dy=500.0, ztop=2000.0,
        dt=3.0, run_seconds=60.0, time_step_sound=4,
        moist=True, mp_physics=0,
        bl_pbl_physics=11,
        sf_sfclay_physics=91,
        sf_surface_physics=0,
        km_opt=4, c_s=0.25,
        bldt=0.0, radt=0.0, cu_physics=0))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = cp.float32(2.0)
    initialize_physics(state, cfg, landmask=1.0, tsk=305.0)

    state.e_sgs[...] = cp.float32(0.0)        # the injected cliff

    run_steps(state, cfg, 2)

    after = cp.asnumpy(state.e_sgs)
    assert np.isfinite(after).all()
    assert (after >= F(SHINHONG_TKE_FLOOR)).all()
    assert float(after.max()) > float(SHINHONG_TKE_FLOOR)
    driver = state.physics
    assert driver._shinhong_entry_advisory is True
    assert driver._shinhong_passenger_advisory is False, (
        "the heal must preempt the chain's NaN entirely; reaching the "
        "output repair means the entry contract did not hold")
