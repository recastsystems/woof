"""Shin-Hong's passenger TKE chain must not kill a run (task #206).

The composed thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1 suite -- a
user-selectable shipped profile -- died on a bare 12 km GFS case with
``FloatingPointError: Shin-Hong returned non-finite tke tendency``,
while an unmodified WRF v4.6.1 ran the identical initial state to
completion.  The diagnosis, reproduced CPU-side through the float32 WRF
authority below: the SGS TKE diagnostic chain (``mixlen``/``prodq2``/
``vdifq``), which woof always computes and WRF's default never does,
contains WRF's own unguarded divisions by quantities that legitimately
reach zero, and can go non-finite in the passenger pair (tke, el) while
every tendency beside it stays finite.  The old driver policy validated
the passenger with the same fatality as the tendencies, and would have
fed the NaN back into ``state.e_sgs`` had it continued.  The failure is
edge-of-cliff: identical prepared artifacts crashed on one sm_120
box and completed 7/7 frames on another whose driver JIT differs in the
last ULP, which is exactly why the guard must not depend on any one
build's luck.

The policy under test: non-finite confined to (tke, el) is repaired to
the chain's own floor (WRF's shinhonginit epsq2l/2) with one loud
advisory; non-finite in ANYTHING a consumer reads stays fatal,
first-invalid ordering preserved.  All CPU-hermetic: the repair
dispatches on the array module, and the mechanism pin runs through
``woof.verify.shinhong_ref``.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.core.shinhong import (
    SHINHONG_PASSENGER_OUTPUTS,
    SHINHONG_TKE_FLOOR,
    repair_shinhong_passenger_outputs,
)
from woof.verify.shinhong_ref import EPSQ2L, np_shinhong_column

F = np.float32


# ---------------------------------------------------------------------------
# The mechanism, pinned through the float32 WRF authority.
# ---------------------------------------------------------------------------

def _convective_column(kte=49):
    """A plausible daytime convective column (unstable, moist, westerly)."""
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


TENDENCY_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                    "rqcblten", "rqiblten", "exch_h")


def test_the_passenger_chain_can_go_non_finite_while_tendencies_hold():
    """The class exists: entry TKE at the 0/0 cliff produces non-finite
    tke through WRF's own transcribed arithmetic, with every tendency
    finite beside it.  This is the CPU reproduction of the composed-run
    crash signature (first-invalid == tke, everything before it clean).
    """
    args, surface = _convective_column()
    out = np_shinhong_column(*args, np.zeros(49, F), **surface)
    assert not np.isfinite(out["tke"]).all(), (
        "the hazard lane vanished: either the chain grew a guard "
        "(update this pin AND the driver policy note) or the authority "
        "changed")
    for name in TENDENCY_OUTPUTS:
        assert np.isfinite(out[name]).all(), (
            f"{name} must stay finite while the passenger chain fails; "
            "the repair policy is only sound under this property")
    assert np.isfinite(out["hpbl"]) and np.isfinite(out["wstar"])


def test_the_guarded_cold_start_stays_finite():
    """The WRF cold-start floor (epsq2l/2) keeps the same column finite
    everywhere -- the control that the instrument discriminates."""
    args, surface = _convective_column()
    out = np_shinhong_column(
        *args, np.full(49, SHINHONG_TKE_FLOOR, F), **surface)
    for name in TENDENCY_OUTPUTS + ("tke", "el"):
        assert np.isfinite(out[name]).all(), name


def test_the_floor_is_wrfs_own_cold_start_value():
    """core/shinhong.py may not depend on the verification tree, so the
    floor is a literal there; this is the gate that keeps the pair equal
    (the state.py cold start carries the same literal, gated in
    tests/test_shinhong_runtime.py)."""
    assert F(SHINHONG_TKE_FLOOR) == EPSQ2L / F(2.0)


# ---------------------------------------------------------------------------
# The repair policy, on NumPy arrays (no device).
# ---------------------------------------------------------------------------

def test_repair_replaces_only_non_finite_passenger_values():
    tke = np.array([[0.2, np.nan], [np.inf, 0.31]], F)
    el = np.array([[np.nan, 5.0], [7.5, 12.0]], F)
    out = {"tke": tke.copy(), "el": el.copy()}
    counts = repair_shinhong_passenger_outputs(out)
    assert counts == {"tke": 2, "el": 1}
    assert np.isfinite(out["tke"]).all() and np.isfinite(out["el"]).all()
    assert out["tke"][0, 0] == F(0.2) and out["tke"][1, 1] == F(0.31)
    assert out["tke"][0, 1] == F(SHINHONG_TKE_FLOOR)
    assert out["tke"][1, 0] == F(SHINHONG_TKE_FLOOR)
    assert out["el"][0, 0] == F(0.0) and out["el"][0, 1] == F(5.0)
    assert out["tke"].dtype == F and out["el"].dtype == F


def test_repair_is_the_identity_on_finite_passengers():
    tke = np.array([0.2, 0.31], F)
    el = np.array([5.0, 7.5], F)
    out = {"tke": tke, "el": el}
    assert repair_shinhong_passenger_outputs(out) == {}
    assert out["tke"] is tke and out["el"] is el


# ---------------------------------------------------------------------------
# The driver's disposition, pinned to the failing composition.
# ---------------------------------------------------------------------------

def test_passenger_set_is_exactly_the_unconsumed_pair():
    """The policy's soundness rests on nothing reading tke/el except the
    scheme's own feedback and the instrument; anything else the driver
    consumes must stay in the fatal set."""
    assert SHINHONG_PASSENGER_OUTPUTS == ("tke", "el")


def test_driver_disposition_repairs_passengers_and_keeps_tendencies_fatal(
        monkeypatch):
    """Drive the REAL ``PhysicsDriver._run_shinhong`` body on NumPy
    arrays: launch and validation are stubbed at the seam (their device
    contracts have their own suites), so what runs here is exactly the
    disposition code that decided the composed run's fate.
    """
    import types

    from woof.core import physics as ph

    kte, ny, nx = 4, 2, 3
    shape = (kte, ny, nx)

    def fake_out(bad):
        names = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h",
                 "tke", "el")
        out = {name: np.full(shape, 0.25, F) for name in names}
        out.update(hpbl=np.full((ny, nx), 700.0, F),
                   kpbl=np.full((ny, nx), 2, np.int32),
                   wstar=np.full((ny, nx), 1.0, F),
                   delta=np.full((ny, nx), 100.0, F))
        for name in bad:
            out[name][..., 0] = np.nan
        return out

    class FakeState:
        f = np.full((ny, nx), 1.0e-4, F)
        e_sgs = np.full(shape, 0.005, F)

        def scratch(self, shape, name):
            class _V:
                def view(self, dtype):
                    return np.zeros(shape, dtype=np.uint32)
            return _V()

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
        pbl_tendencies=FakeTendencies(),
        last_ysu=None,
        gf_rthblten=np.zeros(shape, F),
        gf_rqvblten=np.zeros(shape, F),
    )
    # THE REAL SEAM, BOUND TO THE FAKE: a due call ends at
    # PhysicsDriver._couple_pbl_slot, which couples the raw rates AND
    # retains RTHBLTEN/RQVBLTEN for the cumulus scheme.  Stubbing it
    # would let this disposition suite pass over a driver that no longer
    # feeds Grell-Freitas anything.
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

    planned = {}

    def fake_launch(*args, **kwargs):
        return fake_out(planned["bad"])

    def fake_invalid(out, status):
        names = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h",
                 "tke", "el", "hpbl", "kpbl", "wstar", "delta")
        return tuple(n for n in names if n in planned["bad"])

    monkeypatch.setattr(ph, "launch_shinhong", fake_launch)
    monkeypatch.setattr(ph, "invalid_shinhong_outputs", fake_invalid)
    monkeypatch.setattr(
        ph, "couple_ysu_tendencies", lambda state, cfg, out: FakeTendencies())
    monkeypatch.setattr(
        ph, "_composed_optional_tendency_components", lambda cfg: ())
    monkeypatch.setattr(
        ph, "_pbl_optional_tendency_components", lambda cfg: ())
    monkeypatch.setattr(
        ph, "physics_reuses_pbl_composition", lambda cfg: False)

    # Passenger-only failure: the run continues, e_sgs stays finite, and
    # the advisory latch flips exactly once.
    planned["bad"] = ("tke",)
    ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)
    assert np.isfinite(driver.state.e_sgs).all()
    assert driver.state.e_sgs[0, 0, 0] == F(SHINHONG_TKE_FLOOR)
    assert driver._shinhong_passenger_advisory is True
    # A repaired passenger does not cost the cumulus scheme its forcing:
    # the run continued, so the retention seam ran and the lanes carry
    # the finite rates this step computed.
    assert driver.gf_rthblten is not None
    assert np.isfinite(driver.gf_rthblten).all()
    assert driver.gf_rqvblten is not None
    assert np.isfinite(driver.gf_rqvblten).all()

    # A tendency failure stays fatal with the historical message shape,
    # even when a passenger is also bad.
    planned["bad"] = ("dtheta", "tke")
    with pytest.raises(FloatingPointError, match="non-finite dtheta"):
        ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)

    # A consumed 2-D output (hpbl) is fatal too: passenger means the
    # pair, not "everything after exch_h".
    planned["bad"] = ("tke", "hpbl")
    with pytest.raises(FloatingPointError, match="non-finite hpbl"):
        ph.PhysicsDriver._run_shinhong(driver, atmosphere, cfg)
