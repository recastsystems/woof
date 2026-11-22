# SPDX-License-Identifier: Apache-2.0
"""The surface-flux net (checkpoint r4, applied at half scale every fifth land step): a per-column network after RUC's
land step that shifts sensible and latent heat flux by an energy-conserving Bowen shift plus two bounded net residuals,
and the skin temperature and surface humidity the surface diagnostics and MYNN-EDMF read.

Inputs are 49 local land-surface and lowest-layer features (radiation, RUC fluxes, soil water and temperature, the four
lowest levels, the diurnal angle); the checkpoint masks the calendar and latitude.  Outputs are tanh-bounded by the
checkpoint's caps.  The network (SurfaceFluxNet) runs in PyTorch on the
card, reading and writing the resident pools in place through DLPack (no copies off the card, no host syncs).
"""
from __future__ import annotations

import contextlib
import math
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from .common import CP_J_PER_KG_K, EPS_Q, LV_J_PER_KG, MAXSMC, REFSMC, WLTSMC, SkinBackup, flux_deltas, start_utc

N_CONT_FEATURES = 49
DEFAULT_CAPS = {"bowen": 80.0, "nete_sh": 30.0, "nete_lh": 30.0, "tsk_res": 2.0, "qsfc_rel": 0.15}


def build_surface_flux_net(*, variant="bowen_nete", hidden=64, n_blocks=2, dropout=0.10, pure_local_physics=True,
                           night_cap_factor=1.0, caps=None, tsk_eq_weight=1.0, device=None):
    import torch
    import torch.nn as nn

    cap_cfg = dict(DEFAULT_CAPS)
    if caps:
        cap_cfg.update({k: float(v) for k, v in caps.items()})

    class ResMLPBlock(nn.Module):
        def __init__(self, dim, p_drop):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.act = nn.SiLU()
            self.drop = nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity()
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))

    class SurfaceFluxNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = str(variant)
            self.hidden = int(hidden)
            self.n_blocks = int(n_blocks)
            self.night_cap_factor = float(night_cap_factor)
            self.tsk_eq_weight = float(tsk_eq_weight)
            self.veg_emb = nn.Embedding(22, 6)
            self.soil_emb = nn.Embedding(18, 6)
            fmask = torch.ones(N_CONT_FEATURES, dtype=torch.float32)
            if pure_local_physics:
                fmask[45] = 0.0
                fmask[46] = 0.0
                fmask[47] = 0.0
            self.register_buffer("feat_mask", fmask)
            self.in_norm = nn.LayerNorm(N_CONT_FEATURES)
            self.stem = nn.Sequential(nn.Linear(N_CONT_FEATURES + 12, hidden), nn.SiLU())
            self.blocks = nn.ModuleList([ResMLPBlock(hidden, float(dropout)) for _ in range(n_blocks)])
            self.out_norm = nn.LayerNorm(hidden)
            self.out_act = nn.SiLU()
            self.head = nn.Linear(hidden, 1 if self.variant == "bowen_only" else 5)
            for name in ("bowen", "nete_sh", "nete_lh", "tsk_res", "qsfc_rel"):
                self.register_buffer(f"cap_{name}", torch.tensor(cap_cfg[name], dtype=torch.float32))

        def forward(self, cont_feats, ivgtyp, isltyp):
            x_c = self.in_norm(torch.clamp(cont_feats, -10.0, 10.0) * self.feat_mask)
            v_e = self.veg_emb(torch.clamp(ivgtyp, 0, 21))
            s_e = self.soil_emb(torch.clamp(isltyp, 0, 17))
            h = self.stem(torch.cat([x_c, v_e, s_e], dim=-1))
            for blk in self.blocks:
                h = blk(h)
            raw = self.head(self.out_act(self.out_norm(h)))
            if self.night_cap_factor < 1.0:
                gsw_pos = torch.clamp(cont_feats[:, 0] * 500.0, min=0.0)
                sg = self.night_cap_factor + (1.0 - self.night_cap_factor) * torch.clamp(gsw_pos / 80.0, 0.0, 1.0)
            else:
                sg = 1.0
            d_bowen = self.cap_bowen * torch.tanh(raw[:, 0]) * sg
            if self.variant == "bowen_only":
                z = torch.zeros_like(d_bowen)
                return d_bowen, z, z, z, z
            d_esh = self.cap_nete_sh * torch.tanh(raw[:, 1]) * sg
            d_elh = self.cap_nete_lh * torch.tanh(raw[:, 2]) * sg
            d_tsk = self.cap_tsk_res * torch.tanh(raw[:, 3]) * sg
            d_qrel = self.cap_qsfc_rel * torch.tanh(raw[:, 4]) * sg
            return d_bowen, d_esh, d_elh, d_tsk, d_qrel

    model = SurfaceFluxNet()
    return model.to(device) if device is not None else model


def load_checkpoint(path, device="cuda"):
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model = build_surface_flux_net(
        variant=ckpt.get("variant", "bowen_nete"), hidden=int(ckpt.get("hidden", 128)),
        n_blocks=int(ckpt.get("n_blocks", 4)), dropout=float(ckpt.get("dropout", 0.0)),
        pure_local_physics=bool(ckpt.get("pure_local_physics", False)),
        night_cap_factor=float(ckpt.get("night_cap_factor", 1.0)), caps=ckpt.get("caps", DEFAULT_CAPS),
        tsk_eq_weight=float(ckpt.get("tsk_eq_weight", 1.0)), device=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt


def qsat_torch(T, P):
    """Tetens saturation specific humidity qsat(T [K], P [Pa])."""
    import torch
    tc = torch.clamp(T - 273.15, -60.0, 60.0)
    es = 611.2 * torch.exp(17.67 * tc / (tc + 243.5))
    denom = torch.clamp(P - (1.0 - EPS_Q) * es, min=1000.0)
    return torch.clamp(EPS_Q * es / denom, min=1e-6, max=0.06)


def eligible_now(st, xice):
    """Interior unfrozen land at this step (bool): the only columns a closure writes.  Read every step, cached
    correction or not.  THE BREAKAGE THIS PREVENTS: a correction cached at the last evaluation landed on a column
    that froze since."""
    return (st["xland"] < 1.5) & (xice < 0.01) & (st["interior"] > 0.5)


def add_where(arr, eligible, delta):
    """``arr += delta`` on eligible columns only; every other word stays as it was, whatever ``delta`` holds there (a
    masked column's invalid predictor gives NaN, and NaN * 0 is NaN)."""
    import torch
    arr.copy_(torch.where(eligible, arr + delta, arr))


def skin_shift(provider, n, eligible, d_hfx, d_tsk_res, d_qrel, tsk_eq_weight, _f, _p):
    """The 'diag' skin coupling: the Monin-Obukhov equilibrium skin shift for the sensible-flux shift plus the net's
    residual, and the surface humidity's relative shift capped at the shifted skin's saturation.

    Written on ``eligible`` columns only: the 180-360 K clamp and the saturation cap used to run on every column, so
    a boundary column's skin or a water column's supersaturated surface humidity was clipped by a hook that does not
    act there (the arithmetic on eligible columns is unchanged, word for word)."""
    import torch
    land_mask = eligible.float()
    tsk_t = torch.from_dlpack(provider.field("sfc_input", "skintemp")[:n])
    qsfc_t = torch.from_dlpack(provider.field("diag_physics", "qsfc")[:n])
    chs_t = _f("diag_physics", "chs").float()
    rho0_t = _p("rho_p")[0].float()
    psfc_t = _f("diag", "surface_pressure").float()
    # HFX = rho cp chs (tsk - t1)  =>  d_tsk_eq = d_HFX / (rho cp chs)
    denom_h = torch.clamp(rho0_t * CP_J_PER_KG_K * torch.clamp(chs_t, min=0.008), min=8.0)
    d_tsk_eq = torch.clamp(d_hfx / denom_h, -2.5, 2.5)
    d_tsk_tot = torch.clamp(tsk_eq_weight * d_tsk_eq + d_tsk_res, -3.0, 3.0) * land_mask
    tsk_new = torch.clamp(tsk_t + d_tsk_tot, 180.0, 360.0)
    tsk_t.copy_(torch.where(eligible, tsk_new, tsk_t))
    qsat_sk = qsat_torch(tsk_new, psfc_t)
    d_qsfc_tot = torch.clamp(qsfc_t * d_qrel, -0.005, 0.005) * land_mask
    qsfc_new = torch.clamp(qsfc_t + d_qsfc_tot, min=1e-6)
    qsfc_new = torch.minimum(qsfc_new, qsat_sk)
    qsfc_t.copy_(torch.where(eligible, qsfc_new, qsfc_t))


def engine_stream(cp):
    """Run torch's kernels on the stream the engine's cupy kernels are on.

    THE BREAKAGE THIS PREVENTS: on a non-default cupy stream, torch's default-stream kernels would read the pools
    before the engine's writes to them finished (and the engine would read before torch's adds).  On the default
    stream (the physics step's, today) this is no context at all, so the order is the overlay's."""
    ptr = cp.cuda.get_current_stream().ptr
    if not ptr:
        return contextlib.nullcontext()
    import torch
    return torch.cuda.stream(torch.cuda.ExternalStream(ptr))


class TorchHookBase:
    """Lazy torch on the card, the static per-mesh tensors and the step counters."""

    slots: Tuple[str, ...] = ("land",)

    def __init__(self, checkpoint, *, stride, tsk_couple, tsk_eq_weight):
        self.checkpoint = str(checkpoint)
        self.stride = max(1, int(stride))
        self.tsk_couple = str(tsk_couple)
        self.tsk_eq_weight = float(tsk_eq_weight)
        self.model = None
        self._device = None
        self._soil = None
        self._static: Dict[int, Dict[str, Any]] = {}
        self._skin = SkinBackup()
        self.calls = self.evals = 0
        self.host_ms = 0.0
        self.label = ""

    def counters(self) -> dict:
        return {"calls": self.calls, "evaluations": self.evals, "stride": self.stride,
                "host_ms_on_step_path": round(self.host_ms, 3),
                "host_ms_per_call": round(self.host_ms / max(1, self.calls), 4),
                "parallelism": "GPU (PyTorch on the run's card, DLPack views of the resident pools)"}

    def _load(self, device):
        raise NotImplementedError

    def _ensure(self, cp_device_id: int) -> None:
        if self.model is not None:
            return
        import torch  # here, at the first step: the engine's kernel pack is already loaded
        self._device = torch.device(f"cuda:{cp_device_id}")
        self._soil = tuple(torch.tensor(t, device=self._device, dtype=torch.float32) for t in (MAXSMC, REFSMC, WLTSMC))
        self._load(self._device)

    def _get_static(self, provider: Any, n: int):
        import torch
        cached = self._static.get(id(provider))
        if cached is not None and cached["n"] == n:
            return cached

        def _f(pool, name):
            return torch.from_dlpack(provider.field(pool, name))[..., :n]

        lat = _f("mesh", "latCell").float()
        lon = _f("mesh", "lonCell").float()
        area = _f("mesh", "areaCell").float()
        dx_m = torch.sqrt(torch.clamp(area, min=1.0))
        log_dx = torch.log(torch.clamp(dx_m, min=500.0) / 3000.0) / 2.5
        try:
            interior = (torch.from_dlpack(provider.field("mesh", "bdyMaskCell"))[:n] == 0).float()
        except Exception:
            interior = torch.ones(n, device=lat.device, dtype=torch.float32)
        xland = _f("sfc_input", "xland").float()
        ivgtyp = torch.from_dlpack(provider.field("sfc_input", "ivgtyp"))[:n].long()
        isltyp = torch.from_dlpack(provider.field("sfc_input", "isltyp"))[:n].long()
        maxsmc_t, refsmc_t, wltsmc_t = self._soil
        isl = torch.clamp(isltyp, 0, 16)
        refsmc, wltsmc = refsmc_t[isl], wltsmc_t[isl]
        cached = {"n": n, "lat": lat, "lon": lon, "sin_lat": torch.sin(lat), "cos_lat": torch.cos(lat),
                  "log_dx": log_dx, "interior": interior, "xland": xland, "ivgtyp": ivgtyp, "isltyp": isltyp,
                  "maxsmc": maxsmc_t[isl], "wltsmc": wltsmc, "aw_denom": torch.clamp(refsmc - wltsmc, min=0.05),
                  "start_utc": start_utc(provider)}
        self._static[id(provider)] = cached
        return cached

    def _diurnal(self, st, t_sec, n):
        import torch
        utc_now = st["start_utc"] + timedelta(seconds=t_sec)
        doy = float(utc_now.timetuple().tm_yday)
        utc_hr = utc_now.hour + utc_now.minute / 60.0 + utc_now.second / 3600.0
        lst_rad = (utc_hr / 24.0) * (2.0 * math.pi) + st["lon"]
        sin_lst = torch.sin(lst_rad)
        cos_lst = torch.cos(lst_rad)
        decl = 0.409 * math.sin(2.0 * math.pi * (doy - 81.0) / 365.0)
        hour_ang = lst_rad - math.pi
        cos_sza = torch.clamp(st["sin_lat"] * math.sin(decl) + st["cos_lat"] * math.cos(decl) * torch.cos(hour_ang),
                              -1.0, 1.0)
        sin_doy = torch.full((n,), math.sin(2.0 * math.pi * doy / 365.25), device=self._device, dtype=torch.float32)
        cos_doy = torch.full((n,), math.cos(2.0 * math.pi * doy / 365.25), device=self._device, dtype=torch.float32)
        return sin_lst, cos_lst, cos_sza, sin_doy, cos_doy

    def _sfc_features(self, provider, call, st, n):
        """The 49 surface features [n, 49] on the card (the training code's channel order)."""
        import torch

        def _f(pool, name, default=None):
            try:
                arr = provider.field(pool, name)
            except Exception:
                if default is None:
                    raise
                return torch.full((n,), float(default), device=self._device, dtype=torch.float32)
            t = torch.from_dlpack(arr).float()
            if t.ndim == 2 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 2 and t.shape[0] == n and t.shape[1] != n:
                t = t.T
            return t[..., :n]

        def _p(name):
            v = provider.prepared(name)
            t = torch.from_dlpack(v.reshape(tuple(d for d in v.shape[:-1] if d != 1) + (v.shape[-1],))).float()
            if t.ndim == 2 and t.shape[0] == n and t.shape[1] != n:
                t = t.T
            return t[..., :n]

        gsw = _f("diag_physics", "gsw", 0.0)
        glw = _f("diag_physics", "glw", 300.0)
        alb = _f("diag_physics", "sfc_albedo", 0.2)
        grdflx = _f("diag_physics", "grdflx", 0.0)
        hfx = _f("diag_physics", "hfx", 0.0)
        lh = _f("diag_physics", "lh", 0.0)
        etot = hfx + lh
        bfrac = hfx / (hfx.abs() + lh.abs() + 20.0)
        smois = _f("sfc_input", "smois")
        tslb = _f("sfc_input", "tslb")
        tsk = _f("sfc_input", "skintemp")
        qsfc = _f("diag_physics", "qsfc")
        canwat = _f("diag_physics", "canwat", 0.0)
        mavail = _f("diag_physics", "mavail", 0.5)
        sm0, sm1, sm2, sm3 = smois[0], smois[1], smois[2], smois[3]
        sat0 = sm0 / torch.clamp(st["maxsmc"], min=0.1)
        sat1 = sm1 / torch.clamp(st["maxsmc"], min=0.1)
        aw0 = (sm0 - st["wltsmc"]) / st["aw_denom"]
        dts0 = (tslb[0] - tsk) / 10.0
        dts1 = (tslb[1] - tsk) / 10.0
        dts2 = (tslb[2] - tsk) / 10.0
        dts3 = (tslb[3] - tsk) / 10.0
        vegfra = _f("sfc_input", "vegfra", 50.0) / 100.0
        lai = _f("diag_physics", "lai", 1.5) / 5.0
        snowc = _f("sfc_input", "snowc", 0.0)
        rainncv = _f("diag_physics", "rainncv", 0.0)
        rain_ind = torch.log1p(torch.clamp(rainncv, min=0.0) * 100.0)
        psfc = _f("diag", "surface_pressure", 101325.0)
        t_p = _p("t_p")
        qv_p = _p("qv_p")
        pres_p = _p("pres2_hyd_p")
        t0, t1, t2, t3 = t_p[0], t_p[1], t_p[2], t_p[3]
        qv0, qv1, qv2 = qv_p[0], qv_p[1], qv_p[2]
        qsat0 = qsat_torch(t0, pres_p[0])
        rh0 = torch.clamp(qv0 / qsat0, 0.0, 1.5)
        vpd0 = torch.clamp(qsat0 - qv0, min=0.0) * 100.0
        u10 = _f("diag_physics", "u10", 0.0)
        v10 = _f("diag_physics", "v10", 0.0)
        wspd10 = torch.sqrt(u10 * u10 + v10 * v10 + 1e-6) / 10.0
        chs = torch.log1p(torch.clamp(_f("diag_physics", "chs", 0.01), min=0.0) * 100.0)
        cqs = torch.log1p(torch.clamp(_f("diag_physics", "cqs", 0.01), min=0.0) * 100.0)
        sin_lst, cos_lst, cos_sza, sin_doy, cos_doy = self._diurnal(st, float(getattr(call, "time_seconds", 0.0)), n)
        return torch.stack([
            gsw / 500.0, glw / 400.0, alb, grdflx / 100.0, hfx / 200.0, lh / 200.0, etot / 300.0, bfrac,
            sm0, sm1, sm2, sm3, sat0, sat1, aw0, dts0, dts1, dts2, dts3, canwat * 10.0, mavail,
            (tsk - 288.0) / 15.0, qsfc * 100.0, vegfra, lai, snowc, rain_ind, psfc / 1.0e5 - 1.0,
            (t0 - 288.0) / 15.0, (t1 - t0) / 5.0, (t2 - t0) / 10.0, (t3 - t0) / 15.0,
            qv0 * 100.0, (qv1 - qv0) * 100.0, (qv2 - qv0) * 100.0, rh0, vpd0,
            (tsk - t0) / 5.0, (qsfc - qv0) * 100.0, wspd10, chs, cqs,
            sin_lst, cos_lst, cos_sza, sin_doy, cos_doy, st["lat"], st["log_dx"],
        ], dim=-1)

    def before_land_step(self, provider: Any) -> None:
        if self.tsk_couple == "diag":
            with engine_stream(provider.cp):
                self._skin.restore(provider)

    def after_land_step(self, provider: Any, call: Any) -> None:
        with engine_stream(provider.cp):
            self._after_land_step(provider, call)

    def after_pbl_step(self, slot: Any, call: Any) -> None:
        with engine_stream(slot.provider.cp):
            self._after_pbl_step(slot, call)

    def _after_pbl_step(self, slot: Any, call: Any) -> None:  # pragma: no cover - a land-only hook is never called here
        raise NotImplementedError


class SurfaceFluxNetHook(TorchHookBase):
    """The surface-flux net on RUC's land slot."""

    slots = ("land",)

    def __init__(self, checkpoint, *, scale, stride, tsk_couple, tsk_eq_weight):
        super().__init__(checkpoint, stride=stride, tsk_couple=tsk_couple, tsk_eq_weight=tsk_eq_weight)
        self.scale = float(scale)
        self._cached: Dict[int, Tuple[Any, ...]] = {}

    def _load(self, device):
        self.model, ckpt = load_checkpoint(self.checkpoint, device=device)
        if "tsk_eq_weight" in ckpt:
            self.tsk_eq_weight = float(ckpt["tsk_eq_weight"])

    def _after_land_step(self, provider: Any, call: Any) -> None:
        self.calls += 1
        if self.scale == 0.0:
            return
        t0 = time.perf_counter()
        cp = provider.cp
        self._ensure(int(cp.cuda.Device().id))
        import torch
        n = int(provider.session.layout.dimensions["nCellsSolve"])
        st = self._get_static(provider, n)

        def _f(pool, name):
            return torch.from_dlpack(provider.field(pool, name))[..., :n]

        def _p(name):
            v = provider.prepared(name)
            return torch.from_dlpack(v.reshape(tuple(d for d in v.shape[:-1] if d != 1) + (v.shape[-1],)))[..., :n]

        eligible = eligible_now(st, _f("sfc_input", "xice").float())
        pid = id(provider)
        need_eval = self.stride <= 1 or (self.calls - 1) % self.stride == 0 or pid not in self._cached
        with torch.no_grad():
            if need_eval:
                self.evals += 1
                feats = self._sfc_features(provider, call, st, n)
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = self.model(feats, st["ivgtyp"], st["isltyp"])
                if self.scale != 1.0:
                    s = self.scale
                    d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = (d_bowen * s, d_esh * s, d_elh * s, d_tsk_res * s,
                                                                d_qrel * s)
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = (torch.where(eligible, d, 0.0) for d in
                                                            (d_bowen, d_esh, d_elh, d_tsk_res, d_qrel))
                if self.stride > 1:
                    self._cached[pid] = (d_bowen, d_esh, d_elh, d_tsk_res, d_qrel)
            else:
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = self._cached[pid]
            d_hfx, d_lh = flux_deltas(d_bowen, d_esh, d_elh)
            d_qfx = d_lh / LV_J_PER_KG
            add_where(_f("diag_physics", "hfx"), eligible, d_hfx)
            add_where(_f("diag_physics", "lh"), eligible, d_lh)
            add_where(_f("diag_physics", "qfx"), eligible, d_qfx)
            if self.tsk_couple in ("diag", "persist"):
                if self.tsk_couple == "diag":
                    self._skin.save(provider, provider.field("sfc_input", "skintemp")[:n],
                                    provider.field("diag_physics", "qsfc")[:n], eligible)
                skin_shift(provider, n, eligible, d_hfx, d_tsk_res, d_qrel, self.tsk_eq_weight, _f, _p)
        self.host_ms += (time.perf_counter() - t0) * 1000.0
