"""DA lane 7 on the card: the increment forced inside the RK step lands
whole, and the insertion shock is what the hydrostatic rebalance and IAU
remove.

The column is the nested-forecast ratchet's balanced parent over a hill
(tests/test_da_nested_forecast_gpu.py), physics off, so what moves is the
dynamics' answer to the insertion alone.  The increment is a 1.5 K warm
blob at about 3 km, the shape a radar-heating or warm-bubble analysis
leaves.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.gpu]

DT = 15.0
STEPS = 40                 # 600 s
WINDOW = 300.0             # 3D-IAU over the first 20 steps


def _run_cfg():
    from idealised_parent_fixture import _parent_run

    run = _parent_run()
    zeros = {name: 0 for name in ("mp_physics", "ra_lw_physics",
                                  "ra_sw_physics", "bl_pbl_physics",
                                  "sf_sfclay_physics",
                                  "sf_surface_physics", "cu_physics")
             if name in run.__dataclass_fields__}
    return dataclasses.replace(run, **zeros)


def _state(run):
    import cupy as cp

    from idealised_parent_fixture import _build_parent_state

    state, _coord = _build_parent_state(run)
    state.thp[...] = 0.0
    state.u[...] = 0.0
    state.v[...] = 0.0
    if state.qv is not None:
        state.qv[...] = cp.asarray(0.004, dtype=state.qv.dtype)
    from woof.core.diagnostics import update_diagnostics
    update_diagnostics(state, run.hypsometric_opt)
    return state


def _blob(state):
    import cupy as cp

    nz, ny, nx = state.thp.shape
    k = np.arange(nz)[:, None, None]
    j = np.arange(ny)[None, :, None]
    i = np.arange(nx)[None, None, :]
    r2 = (((i - nx / 2) / 4.0) ** 2 + ((j - ny / 2) / 4.0) ** 2
          + ((k - nz // 5) / 4.0) ** 2)
    blob = np.where(r2 < 1.0, 1.5 * np.cos(0.5 * np.pi * np.sqrt(r2)) ** 2,
                    0.0)
    return cp.asarray(blob, dtype=state.thp.dtype)


def _coupled_theta(state):
    """Sum of (c1h*mu + c2h) * dnw * theta: the integral flux-form
    transport conserves on a periodic domain."""
    import cupy as cp

    mu = state.mub2d + state.mup
    chm = state.c1h[:, None, None] * mu[None] + state.c2h[:, None, None]
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    theta = (thb + state.thp).astype(cp.float64)
    return float((chm.astype(cp.float64)
                  * (-state.dnw.astype(cp.float64))[:, None, None]
                  * theta).sum())


def _integrate(state, run, steps=STEPS):
    """Step and keep each step's w and lowest-level pressure."""
    import cupy as cp

    from woof.core.dycore import step

    trace = []
    for _ in range(steps):
        step(state, run)
        trace.append((state.w.copy(), state.p[0].astype(cp.float64)))
    cp.cuda.Stream.null.synchronize()
    return trace


def _arm(kind):
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.da import hydrostatic as hydro
    from woof.da import iau

    run = _run_cfg()
    state = _state(run)
    blob = _blob(state)
    expected = None
    if kind != "control":
        mu = state.mub2d + state.mup
        chm = state.c1h[:, None, None] * mu[None] + state.c2h[:, None, None]
        expected = float((chm.astype(cp.float64)
                          * (-state.dnw.astype(cp.float64))[:, None, None]
                          * blob.astype(cp.float64)).sum())
    forcing = None
    if kind in ("oneshot", "oneshot+hydro"):
        before = hydro.capture_column(state)
        state.thp[...] += blob
        update_diagnostics(state, run.hypsometric_opt)
        if kind == "oneshot+hydro":
            hydro.rebalance_after_insertion(
                state, before, hypsometric_opt=run.hypsometric_opt)
    elif kind in ("3d", "3d+hydro"):
        spread = {"thp": blob}
        if kind == "3d+hydro":
            dphp, _ = hydro.geopotential_increment(
                state, spread, hypsometric_opt=run.hypsometric_opt)
            spread["php"] = dphp
        forcing = iau.IauForcing(
            state, [iau.IauNode(spread, iau.box_density(
                float(state.elapsed_seconds), WINDOW))])
        iau.attach(state, forcing)
    theta0 = _coupled_theta(state)
    p_start = state.p[0].astype(cp.float64)
    trace = _integrate(state, run)
    if forcing is not None:
        iau.detach(state)
    return {"state": state, "trace": trace, "p_start": p_start,
            "expected": expected,
            "theta": _coupled_theta(state), "theta0": theta0,
            "forcing": forcing}


@pytest.fixture(scope="module")
def arms():
    return {kind: _arm(kind) for kind in ("control", "oneshot",
                                          "oneshot+hydro", "3d",
                                          "3d+hydro")}


def test_the_forced_increment_lands_whole(arms):
    control = arms["control"]["theta"]
    for kind in ("oneshot", "3d", "3d+hydro"):
        arm = arms[kind]
        added = arm["theta"] - control
        print(kind, "added", added, "expected", arm["expected"])
        assert abs(added - arm["expected"]) < 0.01 * abs(arm["expected"])
    receipt = arms["3d"]["forcing"].receipt()
    assert abs(receipt["applied_fraction"] - 1.0) < 1e-12
    assert receipt["steps_forced"] == int(WINDOW / DT)


def _anomalies(arm, control):
    """Per step: max |w - w_control| and the domain-mean |tendency of the
    lowest-level pressure anomaly| (Pa/s), the MASPT of the insertion."""
    import cupy as cp

    w_anom, p_tend = [], []
    prev = arm["p_start"] - control["p_start"]
    for (w, p0), (wc, p0c) in zip(arm["trace"], control["trace"]):
        w_anom.append(float(cp.abs(w - wc).max()))
        now = p0 - p0c
        p_tend.append(float(cp.abs(now - prev).mean()) / DT)
        prev = now
    return np.asarray(w_anom), np.asarray(p_tend)


def test_iau_and_the_rebalance_remove_the_insertion_burst(arms):
    control = arms["control"]
    table = {}
    for kind in ("oneshot", "oneshot+hydro", "3d", "3d+hydro"):
        w_anom, p_tend = _anomalies(arms[kind], control)
        table[kind] = {"w_first_min": w_anom[:4].max(),
                       "w_all": w_anom.max(),
                       "dpdt_first_min": p_tend[:4].mean(),
                       "dpdt_all": p_tend.mean()}
    for kind, row in table.items():
        print(kind, {k: round(float(v), 5) for k, v in row.items()})
    shock = table["oneshot"]
    # A warm blob rises by its buoyancy at once, balanced or not, so the
    # rebalance alone moves the first minute's w a little; the acoustic
    # ring it removes is in the surface pressure tendency.
    assert table["oneshot+hydro"]["w_first_min"] < shock["w_first_min"]
    assert table["oneshot+hydro"]["dpdt_first_min"] < 0.5 * shock[
        "dpdt_first_min"]
    assert table["3d"]["w_first_min"] < 0.5 * shock["w_first_min"]
    assert table["3d"]["dpdt_first_min"] < 0.5 * shock["dpdt_first_min"]
    assert table["3d+hydro"]["dpdt_first_min"] <= table["3d"][
        "dpdt_first_min"]


def test_the_rebalance_is_zero_off_the_increment_and_keeps_pressure():
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.da import hydrostatic as hydro

    run = _run_cfg()
    state = _state(run)
    blob = _blob(state)
    p0 = state.p.copy()
    dphp, receipt = hydro.geopotential_increment(
        state, {"thp": blob}, hypsometric_opt=run.hypsometric_opt)
    untouched = (blob == 0).all(axis=0)
    assert float(cp.abs(dphp[:, untouched]).max()) == 0.0
    assert receipt["max_abs_dp_hyd_pa"] == 0.0     # theta only: no loading
    # plain insertion: the EOS reads the warm blob at a fixed thickness
    state.thp[...] += blob
    update_diagnostics(state, run.hypsometric_opt)
    plain = float(cp.abs(state.p - p0).max())
    state.php[...] = (state.php.astype(cp.float64) + dphp).astype(
        state.php.dtype)
    update_diagnostics(state, run.hypsometric_opt)
    balanced = float(cp.abs(state.p - p0).max())
    print("max |dp| plain", plain, "Pa; rebalanced", balanced, "Pa")
    assert plain > 100.0
    assert balanced < 0.02 * plain


def test_an_unforced_step_is_byte_identical():
    """No forcing, and a forcing whose window has not begun, step the same
    bytes as the dycore without the hook: the IAU slot changes nothing
    outside the window it is given."""
    import cupy as cp

    from woof.da import iau
    from woof.ensemble.state_sha import live_state_sha256

    run = _run_cfg()
    plain = _state(run)
    _integrate(plain, run, steps=6)
    late = _state(run)
    forcing = iau.IauForcing(late, [iau.IauNode(
        {"thp": _blob(late)}, iau.box_density(10_000.0, 300.0))])
    iau.attach(late, forcing)
    _integrate(late, run, steps=6)
    iau.detach(late)
    cp.cuda.Stream.null.synchronize()
    assert forcing.steps_forced == 0
    assert live_state_sha256(plain) == live_state_sha256(late)
