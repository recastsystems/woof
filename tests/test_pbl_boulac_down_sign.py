"""Correct-mode ML1, second key: the BouLac downward length sign in WRF's own
BouLac PBL (bl_pbl_physics = 8, module_bl_boulac.F) and the k-epsilon PBL
(17, module_bl_keps.F boulac_length).

Both schemes carry the loop MYNN copied (tests/test_mynn_boulac_down_sign.py
has the derivation): where a sinking parcel stops in a layer of exactly
constant theta, the linear branch divides by ``beta*(theta(izz)-theta(iz))``
where the bbb -> 0 limit of the quadratic branch is
``beta*(theta(iz)-theta(izz))``; the last step is negative and the downward
length short by twice it (WRF v4.6.1 module_bl_boulac.F:606; wrf-model/WRF
PR 2431 fixes both modules for release-v4.8.1).

Breakage prevented: a default (correct-mode) BouLac or k-epsilon run that
keeps the short downward length, a receipt that lists ML1 as applied while
the defect still runs (or as uncorrected once it does not), and an exact run
that no longer reproduces WRF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof import correct_mode
from woof.config import RunConfig

F = np.float32
KERNELS = Path(correct_mode.__file__).parent / "core" / "kernels"


@pytest.mark.parametrize("unit,fixed,wrf", [
    ("boulac.cu", "FSUB(thi, th[izz])", "FSUB(th[izz], thi)"),
    ("keps.cu", "FSUB(thi, W(KS_TH, izz))", "FSUB(W(KS_TH, izz), thi)")])
def test_each_kernel_holds_both_signs_behind_the_define(unit, fixed, wrf):
    """Exact mode compiles WRF's line; ML1 compiles the limit's sign; and the
    define reaches the downward loop only (the upward one has no defect)."""
    text = (KERNELS / unit).read_text()
    assert text.count("#if defined(BOULAC_DOWN_SIGN)") == 1
    at = text.index("#if defined(BOULAC_DOWN_SIGN)")
    guarded = text[at:text.index("#endif", at)]
    on, off = guarded.split("#else")
    assert fixed in on and wrf not in on
    assert wrf in off and fixed not in off
    assert "zdo_sup" in guarded and "zup_inf" not in guarded


def test_the_second_key_is_ml1_scoped_to_boulac_and_keps():
    assert correct_mode.FINDING_BY_KEY["pbl_boulac_down_sign"] == "ML1"
    assert correct_mode.CORRECTIONS_BY_FINDING["ML1"].keys == (
        "mynn_boulac_down_sign", "pbl_boulac_down_sign")
    assert RunConfig.__dataclass_fields__["pbl_boulac_down_sign"].default \
        is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.resolve("exact").settings["pbl_boulac_down_sign"] is False
    assert correct_mode.resolve("correct").settings["pbl_boulac_down_sign"] is True
    assert correct_mode.resolve("correct", ["ML1"]).settings["pbl_boulac_down_sign"] is False
    for scheme, read in ((8, True), (17, True), (5, False), (1, False), (0, False)):
        assert correct_mode.key_in_scope("pbl_boulac_down_sign",
                                         {"bl_pbl_physics": scheme}) is read, scheme
    # The MYNN key does not reach BouLac or k-epsilon, nor this key MYNN.
    assert not correct_mode.key_in_scope("mynn_boulac_down_sign", {"bl_pbl_physics": 8})


def test_a_one_key_arm_reads_as_itself():
    """An arm that turns off only the BouLac/k-epsilon key keeps ML1 applied
    on MYNN and not on a BouLac domain.  As for a one-key arm of WB5, the
    receipt's ``keys`` say which keys were on; ``uncorrected`` lists only
    findings that no key in scope could correct, so it stays empty."""
    base = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0, ztop=10000.0,
                dt=1.0, run_seconds=10.0, moist=True, mp_physics=0)
    settings = {**correct_mode.resolve("correct").settings, "pbl_boulac_down_sign": False}
    boulac = RunConfig(**base, bl_pbl_physics=8, **settings)
    mynn = RunConfig(**base, bl_pbl_physics=5, **settings)
    receipt = correct_mode.run_receipt({1: boulac, 2: mynn})["domains"]
    assert "ML1" not in receipt["1"]["applied_findings"]
    assert receipt["1"]["uncorrected"] == []
    assert "pbl_boulac_down_sign" not in receipt["1"]["keys"]
    assert "pbl_boulac_down_sign" not in receipt["1"]["in_effect"]["keys"]
    assert "ML1" in receipt["2"]["applied_findings"] and receipt["2"]["uncorrected"] == []
    assert receipt["2"]["keys"]["mynn_boulac_down_sign"] is True
    assert "pbl_boulac_down_sign" not in receipt["2"]["keys"]


def _columns(ny=4, nx=16, nz=30, seed=2431, constant=True):
    """Columns of 100 m layers, stable at 2 to 8 K/km, each with (when
    ``constant``) three levels of exactly equal theta at a random height, and
    TKE that can sink parcels into it.  Without ``constant`` no two adjacent
    levels share a theta word, so the linear branch is never reached."""
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    dz = np.full(shape, F(100.0), dtype=F)
    lapse = rng.uniform(0.002, 0.008, shape).astype(F)
    th = (F(295.0) + np.cumsum(lapse * dz, axis=0, dtype=F)).astype(F)
    if constant:
        k0 = rng.integers(3, nz - 10, (ny, nx))
        for j in range(ny):
            for i in range(nx):
                k = int(k0[j, i])
                th[k:k + 3, j, i] = th[k, j, i]
                th[k + 3:, j, i] = (th[k + 3:, j, i] - (th[k + 3, j, i] - th[k, j, i])
                                    + F(0.3)).astype(F)
    else:
        assert np.all(th[1:] != th[:-1])
    p = (F(100000.0) * np.exp(-np.cumsum(dz, axis=0, dtype=np.float64) / 8000.0)).astype(F)
    rho = (p / (F(287.04) * th * (p / F(100000.0)) ** F(0.2857))).astype(F)
    return {
        "dz": dz, "th": th, "rho": rho, "p": p,
        "u": rng.uniform(-10.0, 10.0, shape).astype(F),
        "v": rng.uniform(-10.0, 10.0, shape).astype(F),
        "qv": np.full(shape, F(0.006), dtype=F), "qc": np.zeros(shape, F),
        "tke": rng.uniform(0.05, 3.0, shape).astype(F),
        "hfx": rng.uniform(-20.0, 120.0, (ny, nx)).astype(F),
        "qfx": rng.uniform(0.0, 5.0e-5, (ny, nx)).astype(F),
        "ust": rng.uniform(0.05, 0.6, (ny, nx)).astype(F),
    }


def _boulac(values, down_sign):
    import cupy as cp
    from woof.core.boulac import launch_boulac
    d = {k: cp.asarray(np.ascontiguousarray(v)) for k, v in values.items()}
    tke = d["tke"].copy()
    columns = {k: d[k] for k in ("dz", "u", "v", "th", "rho", "qv", "qc")}
    out = launch_boulac(columns, {k: d[k] for k in ("hfx", "qfx", "ust")}, tke,
                        dt=6.0, cp_air=1004.5, g=9.81, down_sign=down_sign)
    names = ("el_pbl", "exch_h", "rthblten", "rublten", "pblh")
    return {**{n: cp.asnumpy(out[n]) for n in names}, "tke": cp.asnumpy(tke)}


def _keps(values, down_sign):
    import cupy as cp
    from woof.core.keps import keps_step
    from woof.core.physics_inventory import KEPS_STATE_3D
    nz, ny, nx = values["th"].shape
    d = {k: cp.asarray(np.ascontiguousarray(v)) for k, v in values.items()}
    pi = (d["p"] / F(100000.0)) ** F(0.2857)
    p8w = cp.concatenate([d["p"][:1] + F(500.0), d["p"]], axis=0)
    p8w = cp.ascontiguousarray(p8w)
    inputs = {"u": d["u"], "v": d["v"], "th": d["th"], "qv": d["qv"], "qc": d["qc"],
              "rho": d["rho"], "dz8w": d["dz"], "pi": cp.ascontiguousarray(pi.astype(cp.float32)),
              "rthraten": cp.zeros((nz, ny, nx), cp.float32), "p8w": p8w,
              "ust": d["ust"], "hfx": d["hfx"], "qfx": d["qfx"],
              "br": cp.full((ny, nx), F(0.05), cp.float32),
              "znt": cp.full((ny, nx), F(0.1), cp.float32),
              "psim": cp.zeros((ny, nx), cp.float32), "psih": cp.zeros((ny, nx), cp.float32)}
    state = {name: cp.full((nz, ny, nx), F(1.0e-4), cp.float32) for name in KEPS_STATE_3D}
    state["tke_pbl"] = d["tke"].copy()
    # The k-epsilon scheme reads its BouLac length only in the dissipation
    # repair of thetaphi_alphabeta (module_bl_keps.F:915-917), where TKE is
    # more than 1e4 times a dissipation of at most 1e-6; a dissipation this
    # small puts every column there.
    state["diss_pbl"] = cp.full((nz, ny, nx), F(1.0e-9), cp.float32)
    state["pr_pbl"] = cp.ones((nz, ny, nx), cp.float32)
    state["pblh"] = cp.full((ny, nx), F(800.0), cp.float32)
    out = keps_step(inputs, state, dt=6.0, itimestep=3, down_sign=down_sign)
    names = ("exch_h", "exch_m", "rthblten", "rublten")
    return {**{n: cp.asnumpy(out[n]) for n in names},
            "tke": cp.asnumpy(state["tke_pbl"]), "diss": cp.asnumpy(state["diss_pbl"])}


def _changed_columns(a, b):
    changed = None
    for name in a:
        x, y = a[name], b[name]
        diff = np.any((x != y).reshape(x.shape[0], -1) if x.ndim == 3
                      else (x != y).reshape(1, -1), axis=0)
        changed = diff if changed is None else changed | diff
    return changed


@requires_gpu
@pytest.mark.parametrize("run", [_boulac, _keps], ids=["boulac-8", "keps-17"])
def test_the_define_moves_columns_with_a_constant_theta_layer_only(run):
    """On columns with an exactly constant layer, the built kernel with the
    define differs from the one without in some columns; on columns where no
    two adjacent levels share a theta word, the two builds are bit-identical
    (the linear branch is never reached, so the define is inert)."""
    hit = _columns()
    old, new = run(hit, False), run(hit, True)
    for name in old:
        assert np.all(np.isfinite(new[name])), name
    changed = _changed_columns(old, new)
    assert int(changed.sum()) >= 1
    control = _columns(constant=False)
    a, b = run(control, False), run(control, True)
    for name in a:
        assert a[name].tobytes() == b[name].tobytes(), name


@requires_gpu
def test_boulac_length_grows_where_the_sign_moves_it():
    """The fixed last step is +|tl| where WRF's is -|tl|, so wherever the
    define changes BouLac's mixing length it makes it longer, never shorter."""
    hit = _columns()
    old, new = _boulac(hit, False), _boulac(hit, True)
    moved = old["el_pbl"] != new["el_pbl"]
    assert int(moved.sum()) >= 1
    assert np.all(new["el_pbl"][moved] > old["el_pbl"][moved])


@requires_gpu
def test_the_driver_hands_the_key_to_both_launchers(monkeypatch):
    """The physics driver passes ``down_sign`` only when the key is on, so a
    default run compiles the define and an exact run calls each launcher
    exactly as before the key existed."""
    import woof.core.boulac as boulac
    import woof.core.keps as keps
    from woof.core.dycore import run_steps
    from test_boulac_driver_seam import _state as boulac_state
    from test_keps_driver_seam import _driver_cfg, _driver_state
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, initialize_physics
    seen = []
    real_boulac, real_keps = boulac.launch_boulac, keps.keps_step
    monkeypatch.setattr(boulac, "launch_boulac", lambda *a, **k: seen.append(
        ("boulac", k.get("down_sign"))) or real_boulac(*a, **k))
    monkeypatch.setattr(keps, "keps_step", lambda *a, **k: seen.append(
        ("keps", k.get("down_sign"))) or real_keps(*a, **k))
    import cupy as cp
    expected_default = True if correct_mode.CORRECT_KEYS_DEFAULT else None
    state, cfg, _ = boulac_state(cp)
    assert cfg.pbl_boulac_down_sign is correct_mode.CORRECT_KEYS_DEFAULT
    run_steps(state, cfg, 2)
    assert seen and set(seen) == {("boulac", expected_default)}
    for exact in (False, True):
        seen.clear()
        overrides = correct_mode.resolve("exact").settings if exact else {}
        kcfg = _driver_cfg(**overrides)
        kstate = _driver_state(kcfg)
        initialize_physics(kstate, kcfg, landmask=1.0, tsk=310.0,
                           glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
        run_steps(kstate, kcfg, 3)
        assert seen and set(seen) == {("keps", None if exact else expected_default)}, exact
