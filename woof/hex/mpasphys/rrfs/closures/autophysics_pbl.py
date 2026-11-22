# SPDX-License-Identifier: Apache-2.0
"""The 8-parameter symbolic surface closure (AutoPhysics-PBL): no weights and no PyTorch, cupy on the card.

After RUC's land step, on interior unfrozen land:

1. Daytime Bowen law (energy-conserving: dH + dLH == 0 in full sun)::

       dB = f_sun * clip(HFX + LH, 0, 550) * clip(p0 (rh0 - p1) + p2 (seff - p3) - p4 max(T0 - 20 C, 0) / 15, -0.3, 0.3)

   capped at the latent flux and at 65 W/m2; ``seff = 0.4 sat0 + 0.6 (sat1 + sat2) / 2`` (soil saturation).
2. Nocturnal regime-gated laws, ``f_night = 1 - f_sun``, ``f_sun = clip(gsw / 120, 0, 1)``::

       phi_U       = sig((T0 - 20 C) / 3) + sig((5 C - T0) / 3)       (warm and freezing nights)
       phi_weak    = sig((1.5 - (T0 - Tsk)) / 1.2)                   (weak inversions)
       phi_night   = clip(phi_U (0.35 + 0.65 phi_weak), 0, 1.15)
       dE_sh       = f_night phi_night clip(p5 (0.4 + 0.6 phi_calm), -28, 12)
       dE_lh       = f_night clip(p6 max(rh0 - 0.5, 0) 50, -25, 25)
       dTsk        = f_night phi_night clip(p7 (0.3 + 0.7 phi_calm), -1.5, 1.5)

   with ``phi_calm = 1 / (1 + |V10|^2 / 12.25)``.  The ``warm`` night gate (set v2_warm_night) keeps only the warm-night
   branch of phi_U, so the nocturnal cooling acts on warm nights only.
3. The skin temperature and surface humidity take the equilibrium shift of the flux change (weight 0.5) plus dTsk
   for the surface diagnostics and MYNN only ('diag' coupling: undone before the next land step).

The law and the parameter sets are the project's own (v2 and v3_bowen fitted through coupled 30 km runs against
station observations); the sets ship in autophysics_pbl_sets.v1.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from .common import CP_J_PER_KG_K, EPS_Q, LV_J_PER_KG, MAXSMC, SkinBackup

NIGHT_GATES = ("both", "warm")


def load_parameters(path: Path, name: str) -> dict:
    sets = json.loads(Path(path).read_text(encoding="utf-8"))["sets"]
    if name not in sets:
        raise ValueError(f"parameter set {name!r} is not in {path} ({sorted(sets)})")
    row = sets[name]
    params = tuple(float(x) for x in row["params"])
    if len(params) != 8:
        raise ValueError(f"parameter set {name!r} has {len(params)} numbers, the closure takes 8")
    gate = row.get("night_gate", "both")
    if gate not in NIGHT_GATES:
        raise ValueError(f"parameter set {name!r} night_gate {gate!r} is not one of {NIGHT_GATES}")
    return {"params": params, "night_gate": gate}


def sigmoid(xp, x):
    return xp.float32(1.0) / (xp.float32(1.0) + xp.exp(-xp.clip(x, xp.float32(-18.0), xp.float32(18.0))))


def qsat(xp, T, P):
    tc = xp.clip(T - xp.float32(273.15), xp.float32(-60.0), xp.float32(60.0))
    es = xp.float32(611.2) * xp.exp(xp.float32(17.67) * tc / (tc + xp.float32(243.5)))
    denom = xp.maximum(P - xp.float32(1.0 - EPS_Q) * es, xp.float32(1000.0))
    return xp.clip(xp.float32(EPS_Q) * es / denom, xp.float32(1e-6), xp.float32(0.06))


def deltas(xp, *, land_mask, gsw, hfx, lh, sat0, sat_sub, t0, qv0, pres0, tsk, u10, v10, params, gain,
           night_gate="both"):
    """The closure's flux and skin shifts (d_hfx, d_lh, d_qfx, dT_sk_res), every input float32 on one backend.

    ``xp`` is cupy in a run and numpy in the tests; the operations are the same either way."""
    f32 = xp.float32
    seff = f32(0.4) * sat0 + f32(0.6) * sat_sub
    qsat0 = qsat(xp, t0, pres0)
    rh0 = xp.clip(qv0 / qsat0, f32(0.0), f32(1.25))
    t0c = t0 - f32(273.15)
    wspd2 = u10 * u10 + v10 * v10
    p0, p1, p2, p3, p4, p5, p6, p7 = (f32(v) for v in params)
    g_sc = f32(gain)
    f_sun = xp.clip(gsw / f32(120.0), f32(0.0), f32(1.0))
    f_night = f32(1.0) - f_sun
    e_avail = xp.clip(hfx + lh, f32(0.0), f32(550.0))
    frac_b = xp.clip(p0 * (rh0 - p1) + p2 * (seff - p3) - p4 * xp.maximum(t0c - f32(20.0), f32(0.0)) / f32(15.0),
                     f32(-0.30), f32(0.30))
    dB_raw = f_sun * e_avail * frac_b
    dB = xp.clip(xp.minimum(dB_raw, xp.maximum(lh, f32(0.0))), f32(-65.0), f32(65.0))
    phi_calm = f32(1.0) / (f32(1.0) + wspd2 / f32(12.25))
    if night_gate == "warm":
        phi_U = sigmoid(xp, (t0c - f32(20.0)) / f32(3.0))
    else:
        phi_U = sigmoid(xp, (t0c - f32(20.0)) / f32(3.0)) + sigmoid(xp, (f32(5.0) - t0c) / f32(3.0))
    phi_weak_inv = sigmoid(xp, (f32(1.5) - (t0 - tsk)) / f32(1.2))
    phi_night_g = xp.clip(phi_U * (f32(0.35) + f32(0.65) * phi_weak_inv), f32(0.0), f32(1.15))
    dE_sh = f_night * phi_night_g * xp.clip(p5 * (f32(0.4) + f32(0.6) * phi_calm), f32(-28.0), f32(12.0))
    dE_lh = f_night * xp.clip(p6 * xp.maximum(rh0 - f32(0.50), f32(0.0)) * f32(50.0), f32(-25.0), f32(25.0))
    dT_sk_res = f_night * phi_night_g * xp.clip(p7 * (f32(0.3) + f32(0.7) * phi_calm), f32(-1.5), f32(1.5))
    mask_g = land_mask * g_sc
    d_hfx = (dB + dE_sh) * mask_g
    d_lh = (-dB + dE_lh) * mask_g
    d_qfx = d_lh / f32(LV_J_PER_KG)
    return d_hfx, d_lh, d_qfx, dT_sk_res * mask_g


class AutoPhysicsPblHook:
    """The closure on RUC's land slot."""

    slots = ("land",)

    def __init__(self, *, params, night_gate="both", gain=1.0, stride=1, tsk_couple="diag", tsk_eq_weight=0.50):
        if len(params) != 8:
            raise ValueError(f"the closure takes 8 parameters, got {len(params)}")
        if night_gate not in NIGHT_GATES:
            raise ValueError(f"night_gate {night_gate!r} is not one of {NIGHT_GATES}")
        self.params = tuple(float(x) for x in params)
        self.night_gate = night_gate
        self.gain = float(gain)
        self.stride = max(1, int(stride))
        self.tsk_couple = str(tsk_couple)
        self.tsk_eq_weight = float(tsk_eq_weight)
        self._static: Dict[int, Dict[str, Any]] = {}
        self._skin = SkinBackup()
        self._cached: Dict[int, Tuple[Any, ...]] = {}
        self.calls = self.evals = 0
        self.host_ms = 0.0
        self.label = ""

    def counters(self) -> dict:
        return {"calls": self.calls, "evaluations": self.evals, "stride": self.stride, "params": list(self.params),
                "night_gate": self.night_gate, "host_ms_on_step_path": round(self.host_ms, 3),
                "host_ms_per_call": round(self.host_ms / max(1, self.calls), 4),
                "parallelism": "GPU (cupy elementwise on the resident pools; no torch)"}

    def _get_static(self, provider: Any, n: int) -> Dict[str, Any]:
        cached = self._static.get(id(provider))
        if cached is not None and cached["n"] == n:
            return cached
        cp = provider.cp

        def _f(pool, name):
            return cp.asarray(provider.field(pool, name))[..., :n]

        xland = _f("sfc_input", "xland").astype(cp.float32)
        isltyp = cp.clip(_f("sfc_input", "isltyp").astype(cp.int32), 0, 16)
        maxsmc = cp.maximum(cp.asarray(MAXSMC, dtype=cp.float32)[isltyp], cp.float32(0.15))
        try:
            interior = (cp.asarray(provider.field("mesh", "bdyMaskCell"))[:n] == 0).astype(cp.float32)
        except Exception:
            interior = cp.ones((n,), dtype=cp.float32)
        cached = {"n": n, "xland": xland, "maxsmc": maxsmc, "interior": interior}
        self._static[id(provider)] = cached
        return cached

    def before_land_step(self, provider: Any) -> None:
        if self.tsk_couple == "diag" and self.gain != 0.0:
            self._skin.restore(provider)

    def after_land_step(self, provider: Any, call: Any) -> None:
        self.calls += 1
        if self.gain == 0.0:
            return
        t0_wall = time.perf_counter()
        cp = provider.cp
        f32 = cp.float32
        n = int(provider.session.layout.dimensions["nCellsSolve"])
        st = self._get_static(provider, n)

        def _f(pool, name):
            arr = provider.field(pool, name)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] != n:
                arr = arr.T
            return arr[..., :n]

        def _p0(name):
            v = provider.prepared(name)
            t = v.reshape(tuple(d for d in v.shape[:-1] if d != 1) + (v.shape[-1],))
            if t.ndim == 2 and t.shape[0] == n and t.shape[1] != n:
                t = t.T
            return t[0, :n].astype(f32)

        pid = id(provider)
        need_eval = self.stride <= 1 or (self.calls - 1) % self.stride == 0 or pid not in self._cached
        hfx_arr = _f("diag_physics", "hfx")
        lh_arr = _f("diag_physics", "lh")
        qfx_arr = _f("diag_physics", "qfx")
        tsk_arr = provider.field("sfc_input", "skintemp")[:n]
        qsfc_arr = provider.field("diag_physics", "qsfc")[:n]
        # Interior unfrozen land at THIS step, cached correction or not; the only columns written (see below).
        xice = _f("sfc_input", "xice").astype(f32)
        eligible = (st["xland"] < f32(1.5)) & (xice < f32(0.01)) & (st["interior"] > f32(0.5))
        land_mask = eligible.astype(f32)
        if need_eval:
            self.evals += 1
            smois = _f("sfc_input", "smois").astype(f32)
            d_hfx, d_lh, d_qfx, dT_sk_res = deltas(
                cp, land_mask=land_mask, gsw=_f("diag_physics", "gsw").astype(f32), hfx=hfx_arr.astype(f32),
                lh=lh_arr.astype(f32), sat0=smois[0] / st["maxsmc"],
                sat_sub=f32(0.5) * (smois[1] + smois[2]) / st["maxsmc"], t0=_p0("t_p"), qv0=_p0("qv_p"),
                pres0=_p0("pres2_hyd_p"), tsk=tsk_arr.astype(f32), u10=_f("diag_physics", "u10").astype(f32),
                v10=_f("diag_physics", "v10").astype(f32), params=self.params, gain=self.gain,
                night_gate=self.night_gate)
            d_hfx, d_lh, d_qfx, dT_sk_res = (cp.where(eligible, d, f32(0.0)) for d in (d_hfx, d_lh, d_qfx, dT_sk_res))
            if self.stride > 1:
                self._cached[pid] = (d_hfx, d_lh, d_qfx, dT_sk_res)
        else:
            d_hfx, d_lh, d_qfx, dT_sk_res = self._cached[pid]
        # THE BREAKAGE THIS PREVENTS (the three closures, 20 Nov 2026): the shifts were added and the skin clamp and saturation cap applied on every
        # column, so a masked column's invalid input reached its fluxes (NaN * 0), a boundary skin below 180 K or a
        # supersaturated water surface was clipped, and a correction cached at the last evaluation landed on a column
        # that froze since.  On eligible columns the arithmetic is unchanged, word for word.
        hfx_arr[:] = cp.where(eligible, hfx_arr + d_hfx, hfx_arr).astype(hfx_arr.dtype)
        lh_arr[:] = cp.where(eligible, lh_arr + d_lh, lh_arr).astype(lh_arr.dtype)
        qfx_arr[:] = cp.where(eligible, qfx_arr + d_qfx, qfx_arr).astype(qfx_arr.dtype)
        if self.tsk_couple == "diag":
            self._skin.save(provider, tsk_arr, qsfc_arr, eligible)
            tsk = tsk_arr.astype(f32)
            chs = _f("diag_physics", "chs").astype(f32)
            rho0 = _p0("rho_p")
            psfc = _f("diag", "surface_pressure").astype(f32)
            denom_h = cp.maximum(rho0 * f32(CP_J_PER_KG_K) * cp.maximum(chs, f32(0.008)), f32(8.0))
            d_tsk_eq = cp.clip(d_hfx / denom_h, f32(-2.5), f32(2.5))
            d_tsk_tot = cp.clip(f32(self.tsk_eq_weight) * d_tsk_eq + dT_sk_res, f32(-3.0), f32(3.0)) * land_mask
            tsk_new = cp.clip(tsk + d_tsk_tot, f32(180.0), f32(360.0))
            tsk_arr[:] = cp.where(eligible, tsk_new, tsk).astype(tsk_arr.dtype)
            cqs = _f("diag_physics", "cqs").astype(f32)
            denom_q = cp.maximum(rho0 * f32(LV_J_PER_KG) * cp.maximum(cqs, f32(0.008)), f32(20000.0))
            d_qsfc_eq = cp.clip(f32(self.tsk_eq_weight) * d_lh / denom_q, f32(-0.0025), f32(0.0025)) * land_mask
            qsat_sk = qsat(cp, tsk_new, psfc)
            qsfc_new = cp.clip(qsfc_arr.astype(f32) + d_qsfc_eq, f32(1e-6), qsat_sk)
            qsfc_arr[:] = cp.where(eligible, qsfc_new, qsfc_arr).astype(qsfc_arr.dtype)
        self.host_ms += (time.perf_counter() - t0_wall) * 1000.0
