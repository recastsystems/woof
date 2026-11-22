# SPDX-License-Identifier: Apache-2.0
"""The three hrrrv5 closures touch only the columns they correct, on every step (woof.hex.mpasphys.rrfs.closures).

A review of the three closures (20 Nov 2026) found that a hook could
  * clip or poison masked columns (water, sea ice, the boundary ring): the skin clamp and the saturation cap ran on
    every column, and a masked column's invalid predictor reached the fluxes through ``x * 0``;
  * overwrite what other processes wrote to masked columns between two land steps (the transient skin and surface
    humidity were put back over the whole array, so over water MYNN-SFC's new surface humidity was replaced by the
    previous step's);
  * apply a correction cached at the last evaluation (every 5th step) to a column that froze since.
These tests run each hook on host arrays (numpy for cupy; torch on the CPU for the nets) with one eligible land
column turning to ice, one water column with an invalid predictor and a supersaturated surface, one sea-ice column
and one boundary column with a skin below the clamp.  They also show what two further questions of that review
mean here: the closure reads the live soil field at every evaluation (it holds no soil handle across steps and writes
no soil), and every cached correction is re-applied to the live state on each step with its conserved totals intact.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs import closures
from woof.hex.mpasphys.rrfs.closures import autophysics_pbl as ap
from woof.hex.mpasphys.rrfs.closures.common import LV_J_PER_KG, MAXSMC, REFSMC, WLTSMC

FIELDS = (("diag_physics", "hfx"), ("diag_physics", "lh"), ("diag_physics", "qfx"), ("sfc_input", "skintemp"),
          ("diag_physics", "qsfc"))
LAND_TO_ICE, WATER, ICE, BOUNDARY = 0, 1, 2, 3
MASKED = [WATER, ICE, BOUNDARY]
V2 = (0.18, 0.52, 0.10, 0.40, 0.09, -16.0, -0.22, -0.45)


class _HostCupy:
    """numpy standing in for cupy, with the two cuda calls the hooks make."""

    cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0),
                           get_current_stream=lambda: SimpleNamespace(ptr=0))

    def __getattr__(self, name):
        return getattr(np, name)


class _Provider:
    """One hrrrv5 provider's resident pools on the host: 200 columns, four of them special (see the module doc)."""

    def __init__(self, n=200, seed=5, day=True):
        rng = np.random.default_rng(seed)
        f = lambda lo, hi, *shape: rng.uniform(lo, hi, shape or (n,)).astype(np.float32)
        self.cp = _HostCupy()
        self.n = n
        self.session = SimpleNamespace(layout=SimpleNamespace(
            dimensions={"nCellsSolve": n}, configs={"config_start_time": "2024-05-21_15:00:00"}))
        xland = np.ones(n, np.float32)
        xland[WATER] = 2.0
        xice = np.zeros(n, np.float32)
        xice[ICE] = 1.0
        bdy = np.zeros(n, np.int32)
        bdy[BOUNDARY] = 1
        skin = f(285, 312)
        skin[ICE], skin[BOUNDARY] = 250.0, 170.0                  # 170 K: below the hook's 180 K clamp
        qsfc = f(0.006, 0.016)
        qsfc[WATER], qsfc[ICE] = 0.030, 0.008                     # above saturation at their skins
        u10 = f(-8, 8)
        u10[WATER] = np.nan                                       # a masked column's invalid predictor
        self.pools = {
            ("mesh", "latCell"): f(0.70, 0.78), ("mesh", "lonCell"): f(-1.68, -1.58),
            ("mesh", "areaCell"): np.full(n, 9.0e6, np.float32), ("mesh", "bdyMaskCell"): bdy,
            ("sfc_input", "xland"): xland, ("sfc_input", "xice"): xice,
            ("sfc_input", "ivgtyp"): rng.integers(1, 15, n).astype(np.int32),
            ("sfc_input", "isltyp"): rng.integers(1, 13, n).astype(np.int32),
            ("sfc_input", "smois"): f(0.12, 0.42, 9, n), ("sfc_input", "tslb"): f(285, 300, 9, n),
            ("sfc_input", "sh2o"): f(0.12, 0.42, 9, n),
            ("sfc_input", "skintemp"): skin, ("sfc_input", "vegfra"): f(20, 90),
            ("diag_physics", "hfx"): f(20, 320), ("diag_physics", "lh"): f(40, 420), ("diag_physics", "qfx"): f(1e-5, 2e-4),
            ("diag_physics", "qsfc"): qsfc, ("diag_physics", "gsw"): f(300, 850) if day else np.zeros(n, np.float32),
            ("diag_physics", "glw"): f(320, 400), ("diag_physics", "u10"): u10, ("diag_physics", "v10"): f(-8, 8),
            ("diag_physics", "chs"): f(0.01, 0.05), ("diag_physics", "cqs"): f(0.01, 0.05),
            ("diag", "surface_pressure"): f(95000, 99000),
            ("diag_physics", "kzh"): f(0.5, 40, 50, n), ("diag_physics", "kzm"): f(0.5, 40, 50, n),
            ("tend_physics", "rthblten"): np.zeros((50, n), np.float32),
            ("tend_physics", "rqvblten"): np.zeros((50, n), np.float32),
            ("tend_physics", "rublten"): np.zeros((50, n), np.float32),
            ("tend_physics", "rvblten"): np.zeros((50, n), np.float32)}
        t = (300.0 + np.cumsum(f(-0.4, 0.1, 50, 1, n), axis=0)).astype(np.float32)
        self.prep = {"t_p": t, "qv_p": f(0.006, 0.014, 50, 1, n), "pres2_hyd_p": f(70000, 98000, 50, 1, n),
                     "rho_p": f(0.95, 1.15, 50, 1, n), "pi_p": f(0.97, 1.0, 50, 1, n), "u_p": f(-10, 10, 50, 1, n),
                     "v_p": f(-10, 10, 50, 1, n), "dz_p": f(30, 120, 50, 1, n)}
        self.reads: list = []

    def field(self, pool, name):
        arr = self.pools[pool, name]           # KeyError for a field this provider lacks: the hooks' defaults
        self.reads.append(((pool, name), arr))
        return arr

    def prepared(self, name):
        return self.prep[name]

    def snapshot(self):
        return {key: self.pools[key].copy() for key in FIELDS}


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _changed(a, b, cells):
    """{field: [cells whose word changed]} between two snapshots."""
    out = {}
    for key in FIELDS:
        diff = [int(c) for c in cells if _bits(a[key])[c] != _bits(b[key])[c]]
        if diff:
            out[key[1]] = diff
    return out


def _call(step, dt=18.0):
    return SimpleNamespace(time_seconds=step * dt, dt=dt)


def _ready(hook):
    """The torch hooks on the CPU: what _ensure does at the first step, with the host as the device."""
    import torch
    hook._device = torch.device("cpu")
    hook._soil = tuple(torch.tensor(t, dtype=torch.float32) for t in (MAXSMC, REFSMC, WLTSMC))
    hook._load(hook._device)
    hook._ensure = lambda _device_id: None
    return hook


def _hook(kind):
    table = closures.load_table()
    if kind == "autophysics_pbl":
        return ap.AutoPhysicsPblHook(params=V2, night_gate="both", gain=1.0, stride=5, tsk_couple="diag",
                                     tsk_eq_weight=0.5)
    pytest.importorskip("torch")
    if kind == "surface_flux_net":
        from woof.hex.mpasphys.rrfs.closures.surface_flux_net import SurfaceFluxNetHook
        path = closures.check_file("sfn_r4.pt", table["files"]["sfn_r4.pt"])
        return _ready(SurfaceFluxNetHook(checkpoint=path, scale=0.5, stride=5, tsk_couple="diag", tsk_eq_weight=1.0))
    from woof.hex.mpasphys.rrfs.closures.neural_pbl import NeuralPblHook
    path = closures.check_file("neural_pbl_v1.pt", table["files"]["neural_pbl_v1.pt"])
    return _ready(NeuralPblHook(checkpoint=path, stride=5, dry_gate=kind.endswith("v2"), tsk_couple="diag",
                                tsk_eq_weight=1.0))


KINDS = ["surface_flux_net", "neural_pbl_v1", "neural_pbl_v2", "autophysics_pbl"]


# ------------------------------------------------------------------ 1. masked columns, the first call
@pytest.mark.parametrize("kind", KINDS)
def test_masked_columns_are_untouched_by_a_land_step(kind):
    provider = _Provider()
    hook = _hook(kind)
    before = provider.snapshot()
    hook.before_land_step(provider)
    hook.after_land_step(provider, _call(0))
    after = provider.snapshot()
    assert _changed(before, after, MASKED) == {}          # water, sea ice, boundary: every word as RUC left it
    land = [c for c in range(provider.n) if c not in MASKED]
    assert len(_changed(before, after, land).get("hfx", [])) > 100   # the hook acts on land


# ------------------------------------------------------------------ 2. what others write between land steps
@pytest.mark.parametrize("kind", KINDS)
def test_updates_between_land_steps_survive_the_skin_restore(kind):
    provider = _Provider()
    hook = _hook(kind)
    before = provider.snapshot()
    hook.after_land_step(provider, _call(0))
    # Between two land steps the surface layer writes water's surface humidity, the sea-ice step its skin, the
    # boundary update the ring's skin; the hook must put back only the land columns it shifted.
    skin, qsfc = provider.pools["sfc_input", "skintemp"], provider.pools["diag_physics", "qsfc"]
    skin[MASKED] = [291.25, 251.5, 179.75]
    qsfc[MASKED] = [0.01425, 0.00375, 0.00175]
    between = provider.snapshot()
    hook.before_land_step(provider)
    restored = provider.snapshot()
    assert _changed(between, restored, MASKED) == {}
    land = [c for c in range(provider.n) if c not in MASKED]
    back = _changed(before, restored, land)
    assert "skintemp" not in back and "qsfc" not in back        # land's skin and humidity: RUC's own again


# ------------------------------------------------------------------ 3. a correction cached on a column that froze since
@pytest.mark.parametrize("kind", KINDS)
def test_a_column_that_freezes_mid_stride_gets_no_cached_correction(kind):
    provider = _Provider()
    hook = _hook(kind)
    hook.after_land_step(provider, _call(0))                  # call 1 evaluates; calls 2-5 reuse it
    hook.before_land_step(provider)
    provider.pools["sfc_input", "xice"][LAND_TO_ICE] = 1.0    # the land column is sea ice now
    before = provider.snapshot()
    hook.after_land_step(provider, _call(1))
    after = provider.snapshot()
    assert (hook.calls, hook.evals) == (2, 1)                 # a cached step
    assert _changed(before, after, [LAND_TO_ICE] + MASKED) == {}


# ------------------------------------------------------------------ 3b. the boundary-layer subnet's columns
def _pbl_slot(provider):
    return SimpleNamespace(provider=provider, nc=provider.n, metadata={"dt_pbl": 18.0},
                           pools={key: provider.pools[key] for key in provider.pools
                                  if key[0] == "tend_physics" or key[1] in ("kzh", "kzm")})


def _tendencies(provider):
    return {name: provider.pools["tend_physics", name].copy() for name in ("rthblten", "rqvblten", "rublten", "rvblten")}


@pytest.mark.parametrize("kind", ["neural_pbl_v1", "neural_pbl_v2"])
def test_the_boundary_layer_subnet_leaves_masked_and_newly_frozen_columns_alone(kind):
    torch = pytest.importorskip("torch")
    provider = _Provider(day=False)
    provider.prep["u_p"][:, 0, WATER] = np.nan                # the masked column's invalid predictor
    hook = _hook(kind)
    slot = _pbl_slot(provider)
    t0 = _tendencies(provider)
    hook.after_pbl_step(slot, _call(0))
    t1 = _tendencies(provider)
    for name in t0:
        assert np.array_equal(_bits(t0[name])[:, MASKED], _bits(t1[name])[:, MASKED]), name
    assert np.abs(t1["rthblten"] - t0["rthblten"]).max() > 0  # it acts on land
    provider.pools["sfc_input", "xice"][LAND_TO_ICE] = 1.0
    hook.after_pbl_step(slot, _call(1))                       # cached residual diffusivities
    t2 = _tendencies(provider)
    assert (hook.pbl_calls, hook.pbl_evals) == (2, 1)
    for name in t0:
        assert np.array_equal(_bits(t1[name])[:, LAND_TO_ICE], _bits(t2[name])[:, LAND_TO_ICE]), name
    del torch


# ------------------------------------------------------------------ 4. the closure's soil field, every step
class _ReleasingProvider(_Provider):
    """The facade's release between steps: each read of the soil and surface pools hands out the live array, and the
    arrays handed out at the last step are poisoned once the step ends (a held handle reads NaN)."""

    LIVE = (("sfc_input", "smois"), ("sfc_input", "tslb"), ("sfc_input", "sh2o"))

    def end_step(self, rng):
        for key, arr in self.reads:
            if key in self.LIVE:
                arr[...] = np.nan
        self.reads.clear()
        for key in self.LIVE:
            lo, hi = (0.12, 0.42) if key[1] != "tslb" else (285.0, 300.0)
            self.pools[key] = rng.uniform(lo, hi, self.pools[key].shape).astype(np.float32)


def test_the_closure_reads_the_live_soil_field_at_every_evaluation_and_writes_no_soil():
    rng = np.random.default_rng(17)
    provider = _ReleasingProvider()
    provider.pools["diag_physics", "u10"][WATER] = 0.0
    hook = _hook("autophysics_pbl")
    for step in range(11):                                    # evaluations at calls 1, 6 and 11
        hook.before_land_step(provider)
        soil = {key: provider.pools[key].copy() for key in provider.LIVE}
        skin = provider.pools["sfc_input", "skintemp"].copy()
        hfx0, lh0 = provider.pools["diag_physics", "hfx"].copy(), provider.pools["diag_physics", "lh"].copy()
        hook.after_land_step(provider, _call(step))
        for key in provider.LIVE:                             # no soil write, ever
            assert np.array_equal(_bits(provider.pools[key]), _bits(soil[key])), key
        d_hfx = provider.pools["diag_physics", "hfx"] - hfx0
        assert np.all(np.isfinite(d_hfx))
        if step in (0, 5, 10):                                # this step's evaluation used this step's soil
            st = hook._get_static(provider, provider.n)
            smois = soil["sfc_input", "smois"]
            elig = ((st["xland"] < 1.5) & (provider.pools["sfc_input", "xice"] < 0.01)
                    & (st["interior"] > 0.5)).astype(np.float32)
            want = ap.deltas(np, land_mask=elig, gsw=provider.pools["diag_physics", "gsw"], hfx=hfx0, lh=lh0,
                             sat0=smois[0] / st["maxsmc"],
                             sat_sub=np.float32(0.5) * (smois[1] + smois[2]) / st["maxsmc"],
                             t0=provider.prep["t_p"][0, 0], qv0=provider.prep["qv_p"][0, 0],
                             pres0=provider.prep["pres2_hyd_p"][0, 0], tsk=skin,
                             u10=provider.pools["diag_physics", "u10"], v10=provider.pools["diag_physics", "v10"],
                             params=V2, gain=1.0)[0]
            land = elig > 0
            assert np.allclose(d_hfx[land], want[land], rtol=1e-5, atol=1e-3)
        provider.end_step(rng)
    assert (hook.calls, hook.evals) == (11, 3)


# ------------------------------------------------------------------ 5. budgets, every step, cached or not
def test_the_closure_conserves_by_day_and_keeps_water_and_latent_heat_together_on_every_step():
    provider = _Provider(day=True)
    provider.pools["diag_physics", "u10"][WATER] = 0.0
    hook = _hook("autophysics_pbl")
    rng = np.random.default_rng(23)
    land = [c for c in range(provider.n) if c not in MASKED]
    for step in range(12):
        hook.before_land_step(provider)
        for key in (("diag_physics", "hfx"), ("diag_physics", "lh")):   # RUC's fluxes move from step to step
            provider.pools[key][:] = (provider.pools[key] * rng.uniform(0.9, 1.1, provider.n)).astype(np.float32)
        h0, l0, q0 = (provider.pools["diag_physics", k].astype(np.float64) for k in ("hfx", "lh", "qfx"))
        hook.after_land_step(provider, _call(step))
        dh = provider.pools["diag_physics", "hfx"].astype(np.float64) - h0
        dl = provider.pools["diag_physics", "lh"].astype(np.float64) - l0
        dq = provider.pools["diag_physics", "qfx"].astype(np.float64) - q0
        assert np.abs(dh[land]).max() > 1.0
        assert np.abs(dh + dl)[land].max() < 1e-3                       # dH + dLH = 0 in full sun, float32 adds
        assert np.allclose(dq[land] * LV_J_PER_KG, dl[land], rtol=1e-3, atol=2e-3)


def test_the_surface_nets_reapply_one_correction_through_a_stride():
    pytest.importorskip("torch")
    for kind in ("surface_flux_net", "neural_pbl_v1"):
        provider = _Provider()
        provider.pools["diag_physics", "u10"][WATER] = 0.0
        hook = _hook(kind)
        land = [c for c in range(provider.n) if c not in MASKED]
        seen = []
        for step in range(5):
            hook.before_land_step(provider)
            h0, l0, q0 = (provider.pools["diag_physics", k].astype(np.float64) for k in ("hfx", "lh", "qfx"))
            hook.after_land_step(provider, _call(step))
            dh = provider.pools["diag_physics", "hfx"].astype(np.float64) - h0
            dl = provider.pools["diag_physics", "lh"].astype(np.float64) - l0
            dq = provider.pools["diag_physics", "qfx"].astype(np.float64) - q0
            assert np.allclose(dq[land] * LV_J_PER_KG, dl[land], rtol=1e-3, atol=2e-3)
            seen.append((dh[land], dl[land]))
        assert (hook.calls, hook.evals) == (5, 1)
        for dh, dl in seen[1:]:                                        # the same shift, re-applied each step
            assert np.allclose(dh, seen[0][0], atol=1e-3) and np.allclose(dl, seen[0][1], atol=1e-3)


def test_the_boundary_layer_redistribution_is_recomputed_from_the_live_columns_on_cached_steps():
    torch = pytest.importorskip("torch")
    from woof.hex.mpasphys.rrfs.closures.neural_pbl import conservative_tendencies
    provider = _Provider(day=False)
    hook = _hook("neural_pbl_v1")
    slot = _pbl_slot(provider)
    hook.after_pbl_step(slot, _call(0))
    cached = hook._cached_pbl[id(provider)]
    rng = np.random.default_rng(29)
    for step in range(1, 5):                                           # the state moves; the residuals are cached
        for name in ("t_p", "u_p", "v_p", "qv_p"):
            provider.prep[name] *= rng.uniform(0.995, 1.005, provider.prep[name].shape).astype(np.float32)
        before = _tendencies(provider)
        hook.after_pbl_step(slot, _call(step))
        added = {k: _tendencies(provider)[k].astype(np.float64) - before[k] for k in before}
        L = 10
        cols = lambda name: torch.from_numpy(provider.prep[name][:L, 0].T.copy())
        theta = cols("t_p") / torch.clamp(cols("pi_p"), min=0.2)
        want = conservative_tendencies(*cached, theta, cols("qv_p"), cols("u_p"), cols("v_p"), cols("rho_p"),
                                       cols("dz_p"), torch.from_numpy(provider.pools["diag_physics", "kzh"][:L].T.copy()),
                                       torch.from_numpy(provider.pools["diag_physics", "kzm"][:L].T.copy()), 18.0,
                                       k_floor_pos=hook._k_floor_pos, cfl_max=hook._cfl_max)
        mass = np.clip(provider.prep["rho_p"][:L, 0] * provider.prep["dz_p"][:L, 0], 1.0, None).astype(np.float64)
        for name, w in zip(("rthblten", "rqvblten", "rublten", "rvblten"), want):
            got = added[name][:L]
            assert np.allclose(got, w.numpy().T, rtol=1e-4, atol=1e-9), name   # the live state's redistribution
            col = np.abs((mass * got).sum(axis=0))
            scale = np.abs(mass * got).sum(axis=0) + 1e-30
            assert float((col / scale).max()) < 1e-4, name                   # every column's total unchanged
    assert (hook.pbl_calls, hook.pbl_evals) == (5, 1)
