# SPDX-License-Identifier: Apache-2.0
"""A learned column correction through the external-tendency slot (woof.hex.ad.external_tendency). OFF by default.

Word: ``WOOF_HEX_LEARNED_CORRECTION=/path/checkpoint.pt``. Unset (the default), this module imports nothing and
installs nothing: the forecast is the plain model bit for bit. Set, the forecast door installs one in-process
callback on the slot; each model step the network reads the step-start state and returns the four physical rates
of the slot (theta K/s dry, qv kg/kg/s, u and v m/s2 at cells). The slot's own kernel adds them before the edge
projection and the coupling, and skips zero words, so a zero network is the plain model bit for bit.

Optional words: ``GPUWM_HEX_LEARNED_CORRECTION_ARCH`` (PhysConvNetV2, the default; ColumnNet), and
``GPUWM_HEX_LEARNED_CORRECTION_SCALE`` (default 1).

The networks and the 17 input features are the project's research code (November 2026), moved onto the slot from
a wrapper around driver_rrfs.resident_get_tend. Two differences from that wrapper, both on purpose: the cell winds are the step-start
reconstructed winds the slot gives (``u_zonal``, ``u_meridional``) instead of its own reconstruction from the edge
wind, and rho_zz is read at the same time level as theta_m (the wrapper read rho_zz at time level 2). One card only:
the slot's wind adds on several cards are not yet proven to match one card, so more than one rank refuses.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

WORD = "WOOF_HEX_LEARNED_CORRECTION"
WORD_ARCH = "GPUWM_HEX_LEARNED_CORRECTION_ARCH"
WORD_SCALE = "GPUWM_HEX_LEARNED_CORRECTION_SCALE"

RD, RV, CP = 287.0, 461.6, 1004.5
GAMMA = CP / (CP - RD)
RV_OVER_RD = RV / RD
P0 = 100000.0
#: 10 K/day, 5 g/kg/day, 15 m/s/day, 15 m/s/day: the networks' tanh caps (the checkpoints carry them as buffers)
CAPS = (10.0 / 86400.0, 5e-3 / 86400.0, 15.0 / 86400.0, 15.0 / 86400.0)


def _models():
    import torch
    import torch.nn as nn

    class ColumnNet(nn.Module):
        def __init__(self, c_in=17, n_lev=59, hidden=128):
            super().__init__()
            self.register_buffer("caps", torch.tensor(CAPS, dtype=torch.float32).view(1, 4, 1))
            self.col_proj = nn.Linear(c_in * n_lev, 32)
            self.col_norm = nn.LayerNorm(32)
            self.lev_emb = nn.Parameter(torch.zeros(1, n_lev, 8))
            in_dim = c_in + 32 + 8
            self.fc1, self.ln1 = nn.Linear(in_dim, hidden), nn.LayerNorm(hidden)
            self.fc2, self.ln2 = nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
            self.fc3, self.ln3 = nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, 4)
            self.act = nn.SiLU()

        def forward(self, feats, aux=None):
            nc, c, nz = feats.shape
            ctx = self.act(self.col_norm(self.col_proj(feats.reshape(nc, c * nz)))).unsqueeze(1).expand(nc, nz, -1)
            h = torch.cat([feats.permute(0, 2, 1), ctx, self.lev_emb.expand(nc, -1, -1)], dim=-1)
            h1 = self.act(self.ln1(self.fc1(h)))
            h2 = h1 + self.act(self.ln2(self.fc2(h1)))
            h3 = h2 + self.act(self.ln3(self.fc3(h2)))
            return self.caps * torch.tanh(self.head(h3).permute(0, 2, 1))

    class _LevelNorm(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.ln = nn.LayerNorm(channels)

        def forward(self, x):
            return self.ln(x.transpose(1, 2)).transpose(1, 2)

    class _Block(nn.Module):
        def __init__(self, channels, ctx_dim, kernel_size=3, dilation=1):
            super().__init__()
            pad = (kernel_size - 1) * dilation // 2
            self.norm1 = _LevelNorm(channels)
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            self.norm2 = _LevelNorm(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
            self.film = nn.Linear(ctx_dim, channels * 2)
            self.act = nn.SiLU()

        def forward(self, x, ctx, mask):
            h = self.act(self.norm1(x)) * mask
            h = self.conv1(h)
            gamma, beta = self.film(ctx).unsqueeze(-1).chunk(2, dim=1)
            h = self.act(self.norm2(h) * (1.0 + 0.5 * torch.tanh(gamma)) + beta) * mask
            return (x + self.conv2(h)) * mask

    class PhysConvNetV2(nn.Module):
        def __init__(self, c_in=17, n_lev=59, channels=64):
            super().__init__()
            self.n_lev = n_lev
            self.register_buffer("caps", torch.tensor(CAPS, dtype=torch.float32).view(1, 4, 1))
            self.lev_emb = nn.Parameter(torch.zeros(1, 8, n_lev))
            self.ctx_net = nn.Sequential(nn.Linear(10, 32), nn.LayerNorm(32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
            in_ch, c1 = c_in + 8, channels // 2
            self.stem1 = nn.Conv1d(in_ch, c1, kernel_size=1)
            self.stem3 = nn.Conv1d(in_ch, channels - c1, kernel_size=3, padding=1)
            self.blocks = nn.ModuleList([_Block(channels, 32, 3, d) for d in (1, 2, 2, 1)])
            self.out_norm = _LevelNorm(channels)
            self.out_act = nn.SiLU()
            self.head = nn.Conv1d(channels, 4, kernel_size=1)

        def forward(self, feats, aux):
            nc = feats.shape[0]
            mask = aux["trop_mask"].T.unsqueeze(1)                    # (ncol, 1, nz)
            f = torch.clamp(feats, -6.0, 6.0) * mask
            ctx = self.ctx_net(torch.cat([f[:, 12:17, 0], f[:, 0:2, 0], f[:, 3:4, 0], f[:, 5:7, 0]], dim=-1))
            x = torch.cat([f, self.lev_emb.expand(nc, -1, -1)], dim=1) * mask
            h = torch.cat([self.stem1(x), self.stem3(x)], dim=1) * mask
            for block in self.blocks:
                h = block(h, ctx, mask)
            return self.caps * torch.tanh(self.head(self.out_act(self.out_norm(h)))) * mask

    return {"ColumnNet": ColumnNet, "PhysConvNetV2": PhysConvNetV2}


class LearnedCorrection:
    """The slot callback: callback(k, state) -> {theta, qv, u, v} as (nz, ncol) float32 CUDA tensors."""

    def __init__(self, checkpoint: str | Path, init_path: str | Path, *, arch: str = "PhysConvNetV2",
                 scale: float = 1.0, dt_seconds: float | None = None):
        # Absolute now: the MPAS-physics route enters the run's native working folder (session_v842) before the
        # first step reads them. THE BREAKAGE THIS FIXES (0.6.2, 21 Nov 2026): a door given a
        # relative --init, as the manual's commands are, stopped at the first step with "No such file or directory".
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.init_path = Path(init_path).expanduser().resolve()
        self.dt_seconds = None if dt_seconds is None else float(dt_seconds)
        self.arch, self.scale = arch, float(scale)
        self.ready = False
        self.calls = 0

    def _build(self, state: Any) -> None:
        # torch is imported here, at the first step, after the engine has loaded its own kernel pack (an
        # import-order finding: torch first changes the NVRTC the engine compiles with).
        import numpy as np
        import torch
        from netCDF4 import Dataset

        self.torch = torch
        with Dataset(str(self.init_path), "r") as ds:
            get = lambda name: np.asarray(ds.variables[name][:])
            lat, lon = get("latCell").astype(np.float64), get("lonCell").astype(np.float64)
            zgrid, zz = get("zgrid").astype(np.float32), get("zz").astype(np.float32)
            ter, landmask = get("ter").astype(np.float32), get("landmask").astype(np.float32)
            area = get("areaCell").astype(np.float64)
            xtime = b"".join(ds.variables["xtime"][0]).decode().strip() if "xtime" in ds.variables else None
        n_cells = lat.shape[0]
        if int(state.ncol) != n_cells:
            raise RuntimeError(f"{WORD}: the slot has {state.ncol} columns and the init {n_cells}: the learned "
                               "correction runs on one card only: its wind adds on several cards are not yet proven to match one card")
        if xtime is None:
            raise RuntimeError(f"{WORD}: {self.init_path} has no xtime, so the correction has no clock")
        self.start = datetime.strptime(xtime[:19], "%Y-%m-%d_%H:%M:%S")
        self.dt = float(state.dt) if state.dt else self.dt_seconds
        if not self.dt:
            raise RuntimeError(f"{WORD}: the step length is unknown (pass --dt): the correction needs its clock")
        dev = torch.device("cuda", torch.cuda.current_device())
        t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        z_mid = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
        self.zz, self.dz = t(zz.T), t((zgrid[:, 1:] - zgrid[:, :-1]).T)
        self.z_agl = t((z_mid - zgrid[:, :1]).T)
        self.lat, self.lon = t(lat), t(lon)
        self.ter, self.landmask = t(ter), t(landmask)
        self.log_dx = torch.log(torch.clamp(t(np.sqrt(area)) / 10000.0, min=1e-3))
        nz = zz.shape[1]
        model = _models()[self.arch](c_in=17, n_lev=nz).to(dev)
        sd = torch.load(str(self.checkpoint), map_location=dev, weights_only=True)
        model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
        self.model = model.eval()
        self.ready = True

    def _features(self, theta_m, qv, rho_zz, u_cell, v_cell, utc):
        torch = self.torch
        nz, nc = theta_m.shape
        qv = torch.clamp(qv, min=0.0)
        theta = theta_m / (1.0 + RV_OVER_RD * qv)
        rho_dry = torch.clamp(rho_zz * self.zz, min=1e-6)
        p = P0 * torch.pow(torch.clamp(RD * rho_dry * theta_m / P0, min=1e-6), GAMMA)
        temp = theta * torch.pow(torch.clamp(p / P0, min=1e-6), RD / CP)
        tc = torch.clamp(temp - 273.15, -90.0, 60.0)
        es = 611.2 * torch.exp(17.67 * tc / (tc + 243.5))
        rh = torch.clamp(qv / torch.clamp(0.622 * es / torch.clamp(p - es, min=100.0), min=1e-7), 0.0, 1.5)
        dz_mid = torch.clamp(0.5 * (self.dz[:-1] + self.dz[1:]), min=1.0)

        def ddz(f):
            g = (f[1:] - f[:-1]) / dz_mid
            out = torch.empty_like(f)
            out[0], out[-1], out[1:-1] = g[0], g[-1], 0.5 * (g[:-1] + g[1:])
            return out

        dth, du, dv = ddz(theta), ddz(u_cell), ddz(v_cell)
        gm = du * du + dv * dv
        doy = utc.timetuple().tm_yday
        hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
        decl = 0.4091 * math.sin(2.0 * math.pi * (doy - 81) / 365.0)
        csza = torch.clamp(torch.sin(self.lat) * math.sin(decl)
                           + torch.cos(self.lat) * math.cos(decl) * torch.cos((hour - 12.0) * math.pi / 12.0 + self.lon),
                           -1.0, 1.0)
        b = lambda x: x.unsqueeze(0).expand(nz, nc)
        feats = torch.stack([
            torch.clamp((theta - 300.0) / 40.0, -5.0, 5.0), torch.clamp(qv * 100.0, 0.0, 5.0),
            torch.clamp((temp - 260.0) / 40.0, -5.0, 5.0), rh,
            torch.clamp(torch.log(torch.clamp(p / P0, 1e-4, 2.0)), -5.0, 1.0),
            torch.clamp(u_cell / 25.0, -5.0, 5.0), torch.clamp(v_cell / 25.0, -5.0, 5.0),
            torch.clamp(dth * 100.0, -5.0, 5.0), torch.clamp(du * 100.0, -5.0, 5.0), torch.clamp(dv * 100.0, -5.0, 5.0),
            torch.clamp(torch.log10(gm + 1e-8) / 4.0 + 1.0, -2.0, 2.0), torch.clamp(self.z_agl / 15000.0, 0.0, 3.0),
            b(csza), b(self.lat / (math.pi * 0.5)), b(torch.clamp(self.ter / 2000.0, 0.0, 4.0)), b(self.landmask),
            b(torch.clamp(self.log_dx / 3.0, -2.0, 2.0)),
        ], dim=0).permute(2, 0, 1).contiguous()                      # (ncol, 17, nz)
        trop = torch.clamp((p - 10000.0) / 5000.0, 0.0, 1.0)          # 1 below 150 hPa, 0 above 100 hPa
        return feats, {"trop_mask": trop}

    def __call__(self, k: int, state: Any) -> dict[str, Any]:
        if not self.ready:
            self._build(state)
        torch = self.torch
        view = lambda name: torch.from_dlpack(state[name])
        utc = self.start + timedelta(seconds=k * self.dt)
        with torch.no_grad():
            feats, aux = self._features(view("theta_m"), view("qv"), view("rho_zz"),
                                        view("u_zonal"), view("u_meridional"), utc)
            out = self.model(feats, aux)                              # (ncol, 4, nz)
            mask = aux["trop_mask"]
            if "bdy_mask" in state:
                mask = mask * (view("bdy_mask") == 0).to(mask.dtype).unsqueeze(0)
            if self.scale != 1.0:
                out = out * self.scale
            rates = [(out[:, i, :].T * mask).contiguous().to(torch.float32) for i in range(4)]
        self.calls += 1
        if k in (0, 1) or k % 36 == 0:
            print(f"LEARNED_CORRECTION step {k}: max |rate| theta {float(rates[0].abs().max()):.3e} K/s, "
                  f"qv {float(rates[1].abs().max()):.3e}, u {float(rates[2].abs().max()):.3e}, "
                  f"v {float(rates[3].abs().max()):.3e}", flush=True)
        return dict(zip(("theta", "qv", "u", "v"), rates))


def step_seconds(selected: Any) -> float | None:
    """A forecast door's ``--dt`` as seconds, or None when the route picks its own step (unset, ``auto``, ``row``)."""
    if selected is None or str(selected).strip().lower() in ("", "auto", "row"):
        return None
    return float(selected)


def bind_step(seconds: Any) -> LearnedCorrection | None:
    """Give the installed correction the step the forecast route resolved, unless the door named one.

    THE BREAKAGE THIS FIXES (0.6.2, 20 Nov 2026): on the MPAS-physics route (hrrrv5, the default route) the
    slot's state carries no step length and the door's ``--dt`` is unset or ``auto``, so every run with
    WOOF_HEX_LEARNED_CORRECTION set stopped at its first step ("the step length is unknown (pass --dt)"), and
    ``--dt auto`` stopped at install (``float('auto')``).  The route calls this once its config_dt is known."""
    from . import external_tendency
    held = external_tendency.installed()
    callback = getattr(held, "callback", None)
    if not isinstance(callback, LearnedCorrection):
        return None
    if callback.dt_seconds is None:
        callback.dt_seconds = float(seconds)
    return callback


def install_from_env(init_path: str | Path, dt_seconds: float | None = None) -> LearnedCorrection | None:
    """The forecast routes' one call (the MPAS suite route, forecast_v842.run_forecast, and the WOOF-ArWen/x4 route,
    run_cuda_v841_forecast.main). Word unset: None, and nothing is imported or installed (the plain model)."""
    checkpoint = os.environ.get(WORD)
    if not checkpoint:
        return None
    from . import external_tendency
    if external_tendency.installed() is not None:
        return None     # this process already has a callback on the slot (one install per forecast process)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"{WORD}={checkpoint} is not a file")
    arch = os.environ.get(WORD_ARCH, "PhysConvNetV2")
    if arch not in ("PhysConvNetV2", "ColumnNet"):
        raise ValueError(f"{WORD_ARCH}={arch}: not PhysConvNetV2 or ColumnNet")
    callback = LearnedCorrection(checkpoint, init_path, arch=arch, scale=float(os.environ.get(WORD_SCALE, "1")),
                                 dt_seconds=step_seconds(dt_seconds))
    external_tendency.install(callback)
    print(f"LEARNED_CORRECTION installed: {arch} {checkpoint} scale {callback.scale}", flush=True)
    return callback
