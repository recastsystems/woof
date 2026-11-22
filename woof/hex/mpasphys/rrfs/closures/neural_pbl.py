# SPDX-License-Identifier: Apache-2.0
"""The neural boundary layer (v1; v2 is v1 with the dry-air gate): two subnets of one checkpoint.

* After RUC's land step, a surface subnet (the surface-flux net's inputs and outputs) shifts the land fluxes and the
  transient skin, as surface_flux_net does.
* After MYNN-EDMF, a nocturnal subnet predicts bounded log-residuals of the eddy diffusivities (heat, moisture,
  momentum) at the nine interfaces of the lowest ten levels; the residual diffusivity becomes an extra vertical
  diffusive flux with zero flux at the bottom and at the tenth level's top, so its tendencies leave every column's
  mass-weighted theta, qv, u and v sums unchanged (``conservative_tendencies``).

The dry-air gate (v2) damps only the drying side of the surface moisture shift in cool or dry air and the cooling side
in calm inversions, on both subnets.  Over snow it ratchets moisture upward (seen on an
unseen day, 2024-01-16): it stays a sub-switch, off unless named.

The networks (NeuralPBLFlux) run in PyTorch on the card.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from .common import G_ACCEL, LV_J_PER_KG, flux_deltas
from .surface_flux_net import DEFAULT_CAPS, TorchHookBase, add_where, eligible_now, qsat_torch, skin_shift

N_SFC_CONT = 49
N_PBL_LEVELS = 10
N_PBL_INTERFACES = N_PBL_LEVELS - 1
N_PBL_CONT = 54
DEFAULT_PBL_CFG = {"logk_cap": 0.70, "k_floor_pos": 0.50, "cfl_max": 0.08, "day_pbl_scale": 0.25}


def build_neural_pbl_net(*, sfc_hidden=64, sfc_blocks=2, pbl_hidden=64, pbl_blocks=2, dropout=0.10,
                         pure_local_physics=True, night_cap_factor=0.25, sfc_caps=None, pbl_cfg=None,
                         tsk_eq_weight=1.0, device=None):
    import torch
    import torch.nn as nn

    cap_cfg = dict(DEFAULT_CAPS)
    if sfc_caps:
        cap_cfg.update({k: float(v) for k, v in sfc_caps.items()})
    pcfg = dict(DEFAULT_PBL_CFG)
    if pbl_cfg:
        pcfg.update({k: float(v) for k, v in pbl_cfg.items()})

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

    class SurfaceFluxSubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.night_cap_factor = float(night_cap_factor)
            self.veg_emb = nn.Embedding(22, 6)
            self.soil_emb = nn.Embedding(18, 6)
            fmask = torch.ones(N_SFC_CONT, dtype=torch.float32)
            if pure_local_physics:
                fmask[45] = 0.0
                fmask[46] = 0.0
                fmask[47] = 0.0
            self.register_buffer("feat_mask", fmask)
            self.in_norm = nn.LayerNorm(N_SFC_CONT)
            self.stem = nn.Sequential(nn.Linear(N_SFC_CONT + 12, sfc_hidden), nn.SiLU())
            self.blocks = nn.ModuleList([ResMLPBlock(sfc_hidden, dropout) for _ in range(sfc_blocks)])
            self.out_norm = nn.LayerNorm(sfc_hidden)
            self.out_act = nn.SiLU()
            self.head = nn.Linear(sfc_hidden, 5)
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
            return (self.cap_bowen * torch.tanh(raw[:, 0]) * sg, self.cap_nete_sh * torch.tanh(raw[:, 1]) * sg,
                    self.cap_nete_lh * torch.tanh(raw[:, 2]) * sg, self.cap_tsk_res * torch.tanh(raw[:, 3]) * sg,
                    self.cap_qsfc_rel * torch.tanh(raw[:, 4]) * sg)

    class NocturnalPBLSubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.veg_emb = nn.Embedding(22, 6)
            self.soil_emb = nn.Embedding(18, 6)
            self.in_norm = nn.LayerNorm(N_PBL_CONT)
            self.stem = nn.Sequential(nn.Linear(N_PBL_CONT + 12, pbl_hidden), nn.SiLU())
            self.blocks = nn.ModuleList([ResMLPBlock(pbl_hidden, dropout) for _ in range(pbl_blocks)])
            self.out_norm = nn.LayerNorm(pbl_hidden)
            self.out_act = nn.SiLU()
            self.head = nn.Linear(pbl_hidden, 3 * N_PBL_INTERFACES)
            self.register_buffer("logk_cap", torch.tensor(pcfg["logk_cap"], dtype=torch.float32))
            self.register_buffer("day_pbl_scale", torch.tensor(pcfg["day_pbl_scale"], dtype=torch.float32))

        def forward(self, pbl_feats, ivgtyp, isltyp):
            x_c = self.in_norm(torch.clamp(pbl_feats, -10.0, 10.0))
            v_e = self.veg_emb(torch.clamp(ivgtyp, 0, 21))
            s_e = self.soil_emb(torch.clamp(isltyp, 0, 17))
            h = self.stem(torch.cat([x_c, v_e, s_e], dim=-1))
            for blk in self.blocks:
                h = blk(h)
            raw = self.head(self.out_act(self.out_norm(h)))
            # 1 at night (gsw = 0), tapering to day_pbl_scale in strong sun; pbl_feats[:, 40] is gsw / 500
            gsw_pos = torch.clamp(pbl_feats[:, 40] * 500.0, min=0.0)
            night_w = 1.0 - (1.0 - self.day_pbl_scale) * torch.clamp(gsw_pos / 120.0, 0.0, 1.0)
            gate = (self.logk_cap * night_w).unsqueeze(-1)
            return (gate * torch.tanh(raw[:, 0:9]), gate * torch.tanh(raw[:, 9:18]), gate * torch.tanh(raw[:, 18:27]))

    class NeuralPBLFluxNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.sfc = SurfaceFluxSubNet()
            self.pbl = NocturnalPBLSubNet()

    model = NeuralPBLFluxNet()
    return model.to(device) if device is not None else model


def load_checkpoint(path, device="cuda"):
    import torch
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model = build_neural_pbl_net(
        sfc_hidden=int(ckpt.get("sfc_hidden", 64)), sfc_blocks=int(ckpt.get("sfc_blocks", 2)),
        pbl_hidden=int(ckpt.get("pbl_hidden", 64)), pbl_blocks=int(ckpt.get("pbl_blocks", 2)),
        dropout=float(ckpt.get("dropout", 0.0)), pure_local_physics=bool(ckpt.get("pure_local_physics", True)),
        night_cap_factor=float(ckpt.get("night_cap_factor", 0.25)), sfc_caps=ckpt.get("sfc_caps", DEFAULT_CAPS),
        pbl_cfg=ckpt.get("pbl_cfg", DEFAULT_PBL_CFG), tsk_eq_weight=float(ckpt.get("tsk_eq_weight", 1.0)),
        device=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt


def conservative_tendencies(d_log_kh, d_log_kq, d_log_km, theta10, qv10, u10, v10, rho10, dz10, kzh10, kzm10,
                            dt_pbl: float, *, k_floor_pos: float = 0.50, cfl_max: float = 0.08):
    """Column-conserving vertical diffusive redistribution over the lowest ten levels.

    Inputs [N, 9] log-residuals and [N, 10] columns; returns [N, 10] tendencies of theta, qv, u, v whose
    mass-weighted column sums (rho dz, clamped at 1 as below) are zero to float32 rounding: the interface fluxes vanish
    at the bottom and at the tenth level's top, and each interior flux leaves one level and enters the next."""
    import torch

    dz_int = torch.clamp(0.5 * (dz10[:, :-1] + dz10[:, 1:]), min=5.0)
    rho_int = torch.clamp(0.5 * (rho10[:, :-1] + rho10[:, 1:]), min=0.2)
    kh_mynn = torch.clamp(0.5 * (kzh10[:, :-1] + kzh10[:, 1:]), min=0.0, max=250.0)
    km_mynn = torch.clamp(0.5 * (kzm10[:, :-1] + kzm10[:, 1:]), min=0.0, max=250.0)
    cfl_cap = cfl_max * (dz_int * dz_int) / max(float(dt_pbl), 1.0)
    dt_over_dz2 = max(float(dt_pbl), 1.0) / (dz_int * dz_int)

    def _delta_k(d_log_k, k_mynn):
        # MYNN's implicit solve saturates its effective flux near dz^2/(2 dt): k_eff = k / (1 + 2 k dt / dz^2).
        k_eff = k_mynn / (1.0 + 2.0 * k_mynn * dt_over_dz2)
        k_base = torch.where(d_log_k >= 0.0, torch.clamp(k_eff, min=k_floor_pos), k_eff)
        return torch.clamp(k_base * torch.expm1(d_log_k), -cfl_cap, cfl_cap)

    dk_h = _delta_k(d_log_kh, kh_mynn)
    dk_q = _delta_k(d_log_kq, kh_mynn)
    dk_m = _delta_k(d_log_km, km_mynn)
    f_th = -rho_int * dk_h * (theta10[:, 1:] - theta10[:, :-1]) / dz_int
    f_qv = -rho_int * dk_q * (qv10[:, 1:] - qv10[:, :-1]) / dz_int
    f_u = -rho_int * dk_m * (u10[:, 1:] - u10[:, :-1]) / dz_int
    f_v = -rho_int * dk_m * (v10[:, 1:] - v10[:, :-1]) / dz_int
    mass_col = torch.clamp(rho10 * dz10, min=1.0)
    inv_mass = 1.0 / mass_col

    def _div(flux_9):
        out = torch.zeros_like(mass_col)
        out[:, :-1] -= flux_9
        out[:, 1:] += flux_9
        return out * inv_mass

    return _div(f_th), _div(f_qv), _div(f_u), _div(f_v)


class NeuralPblHook(TorchHookBase):
    """The neural boundary layer on RUC's land slot and MYNN-EDMF's slot."""

    slots = ("land", "pbl")

    def __init__(self, checkpoint, *, stride, dry_gate, tsk_couple, tsk_eq_weight):
        super().__init__(checkpoint, stride=stride, tsk_couple=tsk_couple, tsk_eq_weight=tsk_eq_weight)
        self.dry_gate = bool(dry_gate)
        self.pbl_calls = self.pbl_evals = 0
        self._cached_sfc: Dict[int, Tuple[Any, ...]] = {}
        self._cached_pbl: Dict[int, Tuple[Any, ...]] = {}
        self._k_floor_pos = DEFAULT_PBL_CFG["k_floor_pos"]
        self._cfl_max = DEFAULT_PBL_CFG["cfl_max"]

    def counters(self) -> dict:
        record = super().counters()
        record.update(pbl_calls=self.pbl_calls, pbl_evaluations=self.pbl_evals, dry_gate=self.dry_gate)
        return record

    def _load(self, device):
        self.model, ckpt = load_checkpoint(self.checkpoint, device=device)
        if "tsk_eq_weight" in ckpt:
            self.tsk_eq_weight = float(ckpt["tsk_eq_weight"])
        pcfg = ckpt.get("pbl_cfg") or {}
        self._k_floor_pos = float(pcfg.get("k_floor_pos", DEFAULT_PBL_CFG["k_floor_pos"]))
        self._cfl_max = float(pcfg.get("cfl_max", DEFAULT_PBL_CFG["cfl_max"]))

    # ------------------------------------------------------------------ after RUC's land step
    def _after_land_step(self, provider: Any, call: Any) -> None:
        self.calls += 1
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
        need_eval = self.stride <= 1 or (self.calls - 1) % self.stride == 0 or pid not in self._cached_sfc
        with torch.no_grad():
            if need_eval:
                self.evals += 1
                feats = self._sfc_features(provider, call, st, n)
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = self.model.sfc(feats, st["ivgtyp"], st["isltyp"])
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = (torch.where(eligible, d, 0.0) for d in
                                                            (d_bowen, d_esh, d_elh, d_tsk_res, d_qrel))
                if self.stride > 1:
                    self._cached_sfc[pid] = (d_bowen, d_esh, d_elh, d_tsk_res, d_qrel)
            else:
                d_bowen, d_esh, d_elh, d_tsk_res, d_qrel = self._cached_sfc[pid]
            d_hfx, d_lh = flux_deltas(d_bowen, d_esh, d_elh)
            if self.dry_gate:
                # v2: the Clausius-Clapeyron dry/cool-air moisture gate (damps only drying shifts in cool or dry air,
                # where dTd/dq is largest) and the calm-inversion guard (damps only cooling shifts in calm
                # already-inverted air).
                t0_g = _p("t_p")[0].float()
                qv0_g = _p("qv_p")[0].float()
                gsw_g = _f("diag_physics", "gsw").float()
                tsk_g = _f("sfc_input", "skintemp").float()
                u10_g = _f("diag_physics", "u10").float()
                v10_g = _f("diag_physics", "v10").float()
                w10_g = torch.sqrt(u10_g * u10_g + v10_g * v10_g + 1e-6)
                g_q = torch.clamp((qv0_g - 0.0040) / 0.0045, 0.12, 1.0)
                g_t = torch.clamp((t0_g - 278.0) / 8.0, 0.15, 1.0)
                dry_g = torch.minimum(g_q, g_t)
                d_lh = torch.where(d_lh < 0.0, d_lh * dry_g, d_lh)
                d_qrel = torch.where(d_qrel < 0.0, d_qrel * dry_g, d_qrel)
                is_calm_inv = ((gsw_g < 25.0) & (tsk_g < t0_g - 0.5)).float()
                cool_g = 1.0 - is_calm_inv * (1.0 - torch.clamp((w10_g - 2.0) / 4.5, 0.12, 1.0))
                d_hfx = torch.where(d_hfx < 0.0, d_hfx * cool_g, d_hfx)
                d_tsk_res = torch.where(d_tsk_res < 0.0, d_tsk_res * cool_g, d_tsk_res)
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

    # ------------------------------------------------------------------ after MYNN-EDMF
    def _after_pbl_step(self, slot: Any, call: Any) -> None:
        self.pbl_calls += 1
        t0 = time.perf_counter()
        provider = slot.provider
        cp = provider.cp
        self._ensure(int(cp.cuda.Device().id))
        import torch
        n = int(slot.nc)
        st = self._get_static(provider, n)

        def _f(pool, name):
            t = torch.from_dlpack(provider.field(pool, name))
            if t.ndim == 2 and t.shape[0] > t.shape[1] and t.shape[1] <= 128:
                t = t.T
            return t[..., :n]

        def _p(name):
            v = provider.prepared(name)
            t = torch.from_dlpack(v).reshape(tuple(d for d in v.shape[:-1] if d != 1) + (v.shape[-1],))
            if t.ndim == 2 and t.shape[0] > t.shape[1] and t.shape[1] <= 128:
                t = t.T
            return t[..., :n]

        eligible = eligible_now(st, _f("sfc_input", "xice").float())
        lm = eligible.unsqueeze(-1)
        dt_pbl = float(slot.metadata.get("dt_pbl", getattr(call, "dt", 180.0)))
        pid = id(provider)
        need_eval = self.stride <= 1 or (self.pbl_calls - 1) % self.stride == 0 or pid not in self._cached_pbl
        L = N_PBL_LEVELS
        t10 = _p("t_p")[:L].float().T
        pi10 = torch.clamp(_p("pi_p")[:L].float().T, min=0.2)
        theta10 = t10 / pi10
        qv10 = _p("qv_p")[:L].float().T
        u10 = _p("u_p")[:L].float().T
        v10 = _p("v_p")[:L].float().T
        rho10 = _p("rho_p")[:L].float().T
        dz10 = _p("dz_p")[:L].float().T
        kzh10 = torch.from_dlpack(slot.pools[("diag_physics", "kzh")])[:L, :n].float().T
        kzm10 = torch.from_dlpack(slot.pools[("diag_physics", "kzm")])[:L, :n].float().T
        with torch.no_grad():
            if need_eval:
                self.pbl_evals += 1
                feats = self._pbl_features(provider, call, st, n, t10, theta10, qv10, u10, v10, dz10, kzh10)
                d_log_kh, d_log_kq, d_log_km = self.model.pbl(feats, st["ivgtyp"], st["isltyp"])
                d_log_kh, d_log_kq, d_log_km = (torch.where(lm, d, 0.0) for d in (d_log_kh, d_log_kq, d_log_km))
                if self.stride > 1:
                    self._cached_pbl[pid] = (d_log_kh, d_log_kq, d_log_km)
            else:
                d_log_kh, d_log_kq, d_log_km = self._cached_pbl[pid]
            if self.dry_gate:
                # v2: the calm-inversion guard on diffusivity suppression (damps only d_log_kh < 0).
                gsw_p = _f("diag_physics", "gsw").float()
                w0_p = torch.sqrt(u10[:, 0] * u10[:, 0] + v10[:, 0] * v10[:, 0] + 1e-6)
                inv_calm = ((gsw_p < 25.0) & (theta10[:, 1] > theta10[:, 0] + 0.3)).float()
                cool_p = (1.0 - inv_calm * (1.0 - torch.clamp((w0_p - 2.0) / 4.5, 0.12, 1.0))).unsqueeze(-1)
                d_log_kh = torch.where(d_log_kh < 0.0, d_log_kh * cool_p, d_log_kh)
            d_rth, d_rqv, d_ru, d_rv = conservative_tendencies(
                d_log_kh, d_log_kq, d_log_km, theta10, qv10, u10, v10, rho10, dz10, kzh10, kzm10, dt_pbl,
                k_floor_pos=self._k_floor_pos, cfl_max=self._cfl_max)
            # This step's columns only (a cached residual on a column that froze since, or a masked column's invalid
            # predictor, never reaches MYNN's tendencies); the redistribution itself is the live state's, every step.
            for name, tend in (("rthblten", d_rth), ("rqvblten", d_rqv), ("rublten", d_ru), ("rvblten", d_rv)):
                add_where(torch.from_dlpack(slot.pools[("tend_physics", name)])[:L, :n], eligible, tend.T)
        self.host_ms += (time.perf_counter() - t0) * 1000.0

    def _pbl_features(self, provider, call, st, n, t10, theta10, qv10, u10, v10, dz10, kzh10):
        """The 54 column features [N, 54] (no calendar, no latitude)."""
        import torch

        def _f(pool, name, default=0.0):
            try:
                arr = provider.field(pool, name)
            except Exception:
                return torch.full((n,), float(default), device=self._device, dtype=torch.float32)
            t = torch.from_dlpack(arr).float()
            if t.ndim == 2 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 2 and t.shape[0] == n and t.shape[1] != n:
                t = t.T
            return t[..., :n]

        d_theta = (theta10[:, 1:] - theta10[:, 0:1]) / 5.0
        d_qv = (qv10[:, 1:] - qv10[:, 0:1]) * 200.0
        wspd = torch.sqrt(u10 * u10 + v10 * v10 + 1e-4) / 15.0
        log_kzh = torch.log1p(torch.clamp(kzh10, min=0.0, max=250.0)) / 5.0
        dz01 = torch.clamp(0.5 * (dz10[:, 0] + dz10[:, 1]), min=5.0)
        du01_sq = (u10[:, 1] - u10[:, 0]) ** 2 + (v10[:, 1] - v10[:, 0]) ** 2 + 0.25
        ri_01 = torch.clamp((G_ACCEL / torch.clamp(theta10[:, 0], min=200.0)) * (theta10[:, 1] - theta10[:, 0])
                            * dz01 / du01_sq, -5.0, 10.0)
        dz03 = torch.clamp(dz10[:, 0] + dz10[:, 1] + dz10[:, 2], min=15.0)
        du03_sq = (u10[:, 3] - u10[:, 0]) ** 2 + (v10[:, 3] - v10[:, 0]) ** 2 + 0.5
        ri_03 = torch.clamp((G_ACCEL / torch.clamp(theta10[:, 0], min=200.0)) * (theta10[:, 3] - theta10[:, 0])
                            * dz03 / du03_sq, -5.0, 10.0)
        gsw = _f("diag_physics", "gsw", 0.0) / 500.0
        glw = _f("diag_physics", "glw", 300.0) / 400.0
        hfx = _f("diag_physics", "hfx", 0.0) / 100.0
        lh = _f("diag_physics", "lh", 0.0) / 100.0
        hpbl = _f("diag_physics", "hpbl", 500.0) / 1000.0
        ust = _f("diag_physics", "ust", 0.2)
        tsk = _f("sfc_input", "skintemp", 288.0)
        dtsk0 = (theta10[:, 0] - tsk) / 5.0
        smois0 = _f("sfc_input", "smois")[0]
        psfc = _f("diag", "surface_pressure", 101325.0)
        qsat0 = qsat_torch(t10[:, 0], psfc)
        rh0 = torch.clamp(qv10[:, 0] / qsat0, 0.0, 1.5)
        t0_norm = (t10[:, 0] - 288.0) / 15.0
        sin_lst, cos_lst, cos_sza, _, _ = self._diurnal(st, float(getattr(call, "time_seconds", 0.0)), n)
        scalars = torch.stack([ri_01, ri_03, gsw, glw, hfx, lh, hpbl, ust, dtsk0, smois0, rh0, t0_norm,
                               cos_sza, sin_lst, cos_lst, st["log_dx"]], dim=-1)
        return torch.cat([d_theta, d_qv, wspd, log_kzh, scalars], dim=-1)
