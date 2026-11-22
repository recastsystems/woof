# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.target_loss: the forecast-error loss a learned correction is trained on, and its gradient.

A correction is trained through the hex model and its adjoint: the model runs a window, the state at the window's end
is compared with a target, and the adjoint carries dJ/dx_N back. This module is the J and the dJ/dx_N.

STATE SPACE (``StateLoss``). The model state at the window end against an analysis valid then, on the same mesh and
levels (the target is the front door's own init made from the analysis on the same cull, so no interpolation of the
model is needed; ``tools`` below build it):

    J = sum_f omega_f sum_s c_s sum_{k,i} W_f[k,i] ((S_s m (x_f - y_f))[k,i] / sigma_f[k])^2   /  sum_f omega_f

* fields f: ``theta`` (dry potential temperature, K, from the state's theta_m and qv), ``qv`` (kg/kg), ``uzonal`` and
  ``umerid`` (m/s at cells, MPAS's own reconstruction of the edge-normal ``u`` with ``coeffs_reconstruct``), ``plow``
  (pressure on the lowest model level, Pa: pressure_base + pressure_p);
* W_f[k,i] = a_i w[k,i] / sum a: cell area times the layer's share of the column's mass (dp from the target's pressure),
  zero outside the judged region m (interior cells, the analysis's own core: HRRR's grid less its blend taper) and above
  ``top_m``; sum W = 1 per field, so J = 1 is an error of one sigma everywhere;
* sigma_f[k]: the analysis-error size per field and level (``SIGMA``, a starting table, see there);
* S_s: the multi-scale option, the difference smoothed over a length L_s by explicit diffusion on the mesh's own cells
  (``MeshOps.smooth``; its transpose ``MeshOps.smooth_t`` exactly), weight c_s; the default is one scale, L = 0.

Every array is level-major ``(nz, ncol)`` (``u``: ``(nz, nEdges)``), the layout of the plan's allocations and of the
torch bridge's contract. The arithmetic runs on the card when the arrays are CuPy (``xp`` follows the input); numpy
arrays run the same code on the host (tests, the finite-difference check). ``grad_prognostic`` hands dJ/dx_N for the
plan's own state fields (theta_m, qv, u, pressure_p), and ``objective_terms`` writes it as the window runner's
``--objective`` terms over a plan's allocations (``woof.hex.ad_doors.FieldMap``).

OBSERVATION SPACE (``StationLoss``): 2 m temperature, 2 m dewpoint and 10 m wind at stations by interpolation from the
model's LOWEST LEVEL (not the surface layer's own 2 m / 10 m diagnostics: those are not plan state fields at the window
end, so no derivative of them is handed out): the three nearest cells by inverse distance, temperature moved from the
level's height to the station's by 6.5 K/km, dewpoint from qv and the level's pressure, wind as the level's.

THE CHECKS (``python -m woof.hex.ad.target_loss check``): the finite-difference check of dJ/dx (central differences
along random directions, float64), the dot tests of the smoothing and the reconstruction against their transposes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RD, RV, CP, P0, G = 287.0, 461.6, 1004.5, 1.0e5, 9.80616    # MPAS-A's constants (mpas_constants.F)
RVORD = RV / RD
CV = CP - RD
GAMMA = CP / CV                                                   # p = p0 (Rd rho theta_m / p0)^(cp/cv)
FIELDS = ("theta", "qv", "uzonal", "umerid", "plow")

#: Analysis-error sizes. A STARTING TABLE, not measured from the analyses: 1 K in potential temperature; 15 % of the
#: level's mean target vapour (at least 1e-5 kg/kg); wind 2 m/s at the ground rising linearly with (1 - p/p0) to
#: 3.5 m/s near 100 hPa; 100 Pa in lowest-level pressure. Replace with the analysis's own error estimates when known.
SIGMA = {"theta": 1.0, "qv_frac": 0.15, "qv_floor": 1.0e-5, "wind_low": 2.0, "wind_high": 3.5, "plow": 100.0}

#: HRRR's grid (NCEP's CONUS Lambert conformal, the same numbers as init-sources.v1.json's rrfs coverage window).
HRRR_GRID = {"truelat": 38.5, "stand_lon": -97.5, "lat1": 21.138123, "lon1": -122.719528, "dx": 3000.0,
             "nx": 1799, "ny": 1059, "radius": 6371229.0}


def array_module(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except ImportError:
        return np


def _scatter(xp, index, weights, size):
    """sum of weights into index (both flat); the transpose of a gather, with no atomics order dependence on the host."""
    return xp.bincount(index.ravel(), weights=weights.ravel(), minlength=size)[:size]


class MeshOps:
    """The mesh operators the loss needs, on the card when ``xp`` is CuPy: reconstruction to cell winds and its
    transpose, the cell Laplacian and the diffusion smoother and its transpose."""

    def __init__(self, static: dict, xp=np):
        self.xp = xp
        nc, ne = int(static["nCells"]), int(static["nEdges"])
        self.ncol, self.nedges = nc, ne
        f = lambda k, dt=np.float64: xp.asarray(np.asarray(static[k], dtype=dt))
        self.area = f("areaCell")
        coe = np.asarray(static["cellsOnEdge"], dtype=np.int64) - 1          # (ne, 2), -1 or >= nc: off the mesh
        ok = (coe[:, 0] >= 0) & (coe[:, 0] < nc) & (coe[:, 1] >= 0) & (coe[:, 1] < nc)
        self.e_ok = np.flatnonzero(ok)
        self.c1, self.c2 = xp.asarray(coe[ok, 0]), xp.asarray(coe[ok, 1])
        ce = (np.asarray(static["dvEdge"], np.float64) / np.asarray(static["dcEdge"], np.float64))[ok]
        self.ce = xp.asarray(ce)
        rate = np.zeros(nc)
        np.add.at(rate, coe[ok, 0], ce)
        np.add.at(rate, coe[ok, 1], ce)
        lam = float((rate / np.asarray(static["areaCell"], np.float64)).max())
        self.kappa = 0.5 / lam                                                  # m^2 per step, monotone: kappa*lam <= 1
        self.kdiag = xp.asarray(self.kappa / np.asarray(static["areaCell"], np.float64))
        # reconstruction: u_zonal[i] = sum_j (coeffs_reconstruct[i, j, :] . east_i) u[edgesOnCell[i, j]]
        eoc = np.asarray(static["edgesOnCell"], dtype=np.int64) - 1
        neo = np.asarray(static["nEdgesOnCell"], dtype=np.int64)
        cr = np.asarray(static["coeffs_reconstruct"], dtype=np.float64)        # (nc, maxEdges, 3)
        lat, lon = np.asarray(static["latCell"], np.float64), np.asarray(static["lonCell"], np.float64)
        east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
        north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], -1)
        valid = (np.arange(eoc.shape[1])[None, :] < neo[:, None]) & (eoc >= 0) & (eoc < ne)
        self.r_idx = xp.asarray(np.where(valid, eoc, 0))
        self.r_z = xp.asarray(np.where(valid, np.einsum("ijc,ic->ij", cr, east), 0.0))
        self.r_m = xp.asarray(np.where(valid, np.einsum("ijc,ic->ij", cr, north), 0.0))

    # -- reconstruction ---------------------------------------------------------------------------------------------
    def reconstruct(self, u):
        g = u[:, self.r_idx]                                                    # (nz, nc, maxE)
        return (g * self.r_z[None]).sum(-1), (g * self.r_m[None]).sum(-1)

    def reconstruct_t(self, guz, gum):
        xp = self.xp
        nz = guz.shape[0]
        contrib = guz[:, :, None] * self.r_z[None] + gum[:, :, None] * self.r_m[None]
        index = self.r_idx[None, :, :] + (xp.arange(nz) * self.nedges)[:, None, None]
        index = xp.broadcast_to(index, contrib.shape)
        return _scatter(xp, index, contrib, nz * self.nedges).reshape(nz, self.nedges)

    # -- Laplacian and smoothing ------------------------------------------------------------------------------------
    def laplacian(self, x):
        """(L x)_i = sum over interior edges e of i: (dv/dc)_e (x_other - x_i); L is symmetric."""
        xp = self.xp
        nz = x.shape[0]
        flux = self.ce[None, :] * (x[:, self.c2] - x[:, self.c1])
        off = (xp.arange(nz) * self.ncol)[:, None]
        n = nz * self.ncol
        out = _scatter(xp, self.c1[None, :] + off, flux, n) - _scatter(xp, self.c2[None, :] + off, flux, n)
        return out.reshape(nz, self.ncol)

    def steps_for(self, length_m: float) -> int:
        """explicit diffusion steps whose Gaussian width (std) is length_m: sigma^2 = 2 n kappa."""
        return int(round(length_m ** 2 / (2.0 * self.kappa))) if length_m > 0 else 0

    def smooth(self, x, length_m: float):
        for _ in range(self.steps_for(length_m)):
            x = x + self.kdiag[None, :] * self.laplacian(x)
        return x

    def smooth_t(self, g, length_m: float):
        for _ in range(self.steps_for(length_m)):
            g = g + self.laplacian(self.kdiag[None, :] * g)
        return g


def read_mesh(path) -> dict:
    """The mesh arrays the loss needs, from an init.nc of the mesh (a static.nc lacks the reconstruction weights)."""
    import netCDF4
    keys = ("areaCell", "cellsOnEdge", "dvEdge", "dcEdge", "edgesOnCell", "nEdgesOnCell", "coeffs_reconstruct",
            "latCell", "lonCell", "bdyMaskCell")
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        out = {k: np.asarray(d.variables[k][:]) for k in keys if k in d.variables}
        out["nCells"], out["nEdges"] = len(d.dimensions["nCells"]), len(d.dimensions["nEdges"])
    missing = [k for k in keys if k not in out and k != "bdyMaskCell"]
    if missing:
        raise SystemExit(f"DEFECT: woof.hex.ad.target_loss: {path} lacks {', '.join(missing)} (pass an init.nc of the mesh)")
    if not np.any(out["coeffs_reconstruct"]):
        # THE BREAKAGE: the cull's static.nc carries coeffs_reconstruct as zeros (the init computes them), so every cell
        # wind read 0 and the reconstruction's dot test divided by zero (30 km cull, 19 Nov). The init's are the model's.
        raise SystemExit(f"DEFECT: woof.hex.ad.target_loss: {path} has coeffs_reconstruct all zero (a static file; the "
                         "init computes them): pass an init.nc of the same mesh as --static")
    return out


def read_state(path, time_index: int = -1) -> dict:
    """theta, qv, u, pressure (nz, n) float64 and zgrid from an MPAS init, history or restart file (level-major)."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        v = d.variables

        def get(name):
            a = np.asarray(v[name][:])
            if "Time" in v[name].dimensions:
                a = a[time_index]
            return a.astype(np.float64)
        qv = get("qv")
        if "theta" in v:
            theta = get("theta")
        else:
            theta = get("theta_m") / (1.0 + RVORD * qv)
        if "pressure" in v:
            p = get("pressure")
        else:
            p = get("pressure_base") + get("pressure_p")
        out = {"theta": theta.T.copy(), "qv": qv.T.copy(), "u": get("u").T.copy(), "pressure": p.T.copy()}
        if "zgrid" in v:
            out["zgrid"] = get("zgrid").T.copy()
        if "pressure_base" in v:
            out["pressure_base"] = get("pressure_base").T.copy()
        if "rho" in v:
            out["rho"] = get("rho").T.copy()
        out["xtime"] = "".join(c.decode() if isinstance(c, bytes) else str(c)
                               for c in np.asarray(v["xtime"][time_index]).ravel()).strip() if "xtime" in v else ""
    return out


def hrrr_core_mask(lat_rad, lon_rad, inset_m: float = 300e3, grid: dict = HRRR_GRID):
    """True where a cell lies inside HRRR's grid by at least inset_m (the hrrr-gfs blend's 300 km taper): there the
    blended analysis is HRRR's alone. Spherical Lambert conformal, one standard parallel (HRRR's 38.5/38.5)."""
    phi0 = math.radians(grid["truelat"])
    n = math.sin(phi0)
    R = grid["radius"]
    F = math.cos(phi0) * math.tan(math.pi / 4 + phi0 / 2) ** n / n
    lam0 = math.radians(grid["stand_lon"])

    def xy(lat, lon):
        rho = R * F / np.tan(np.pi / 4 + lat / 2) ** n
        th = n * (((lon - lam0) + np.pi) % (2 * np.pi) - np.pi)
        return rho * np.sin(th), -rho * np.cos(th)
    x0, y0 = xy(math.radians(grid["lat1"]), math.radians(grid["lon1"]))
    x, y = xy(np.asarray(lat_rad), np.asarray(lon_rad))
    x, y = x - x0, y - y0
    w, h = (grid["nx"] - 1) * grid["dx"], (grid["ny"] - 1) * grid["dx"]
    return (x >= inset_m) & (x <= w - inset_m) & (y >= inset_m) & (y <= h - inset_m)


class Target:
    """y_f, sigma_f and W_f for one window end, built once from the analysis (or two analyses and a time weight)."""

    def __init__(self, fields: dict, weights: dict, sigma: dict, meta: dict):
        self.y, self.w, self.sigma, self.meta = fields, weights, sigma, meta

    @classmethod
    def build(cls, mesh: MeshOps, static: dict, analyses: list, alphas: list, *, top_m: float = 15000.0,
              core: str = "hrrr", inset_m: float = 300e3):
        xp = np
        ops = mesh if mesh.xp is np else MeshOps(static, np)
        y = {}
        zg = None
        for a, path in zip(alphas, analyses):
            s = read_state(path)
            uz, um = ops.reconstruct(s["u"])
            part = {"theta": s["theta"], "qv": s["qv"], "uzonal": uz, "umerid": um, "plow": s["pressure"][:1],
                    "pressure": s["pressure"]}
            for k, v in part.items():
                y[k] = y.get(k, 0.0) + a * v
            zg = s.get("zgrid", zg)
        p = y.pop("pressure")
        nz, nc = p.shape
        ph = np.empty((nz + 1, nc))
        ph[1:nz] = 0.5 * (p[:-1] + p[1:])
        ph[0] = p[0] + 0.5 * (p[0] - p[1])
        ph[nz] = np.maximum(p[-1] - 0.5 * (p[-2] - p[-1]), 0.0)
        dp = ph[:-1] - ph[1:]
        wcol = dp / dp.sum(0, keepdims=True)
        m = np.ones(nc, bool)
        if "bdyMaskCell" in static:
            m &= np.asarray(static["bdyMaskCell"]) == 0
        if core == "hrrr":
            m &= hrrr_core_mask(static["latCell"], static["lonCell"], inset_m)
        lev = np.ones((nz, nc), bool)
        if zg is not None and top_m:
            lev = 0.5 * (zg[:-1] + zg[1:]) <= top_m
        area = np.asarray(static["areaCell"], np.float64) * m
        w3 = area[None, :] * wcol * lev
        w3 /= w3.sum()
        w2 = (area / area.sum())[None, :]
        pm = (p * w3).sum(1) / np.maximum(w3.sum(1), 1e-300)                    # level-mean pressure, judged region
        qm = (y["qv"] * w3).sum(1) / np.maximum(w3.sum(1), 1e-300)
        sig_w = SIGMA["wind_low"] + (SIGMA["wind_high"] - SIGMA["wind_low"]) * np.clip(1.0 - pm / P0, 0.0, 0.9) / 0.9
        sigma = {"theta": np.full(nz, SIGMA["theta"]), "qv": np.maximum(SIGMA["qv_frac"] * qm, SIGMA["qv_floor"]),
                 "uzonal": sig_w, "umerid": sig_w.copy(), "plow": np.full(1, SIGMA["plow"])}
        weights = {f: (w2 if f == "plow" else w3) for f in FIELDS}
        meta = {"analyses": [str(a) for a in analyses], "alphas": [float(a) for a in alphas], "judged_cells": int(m.sum()),
                "cells": int(nc), "levels": int(nz), "levels_judged": int(lev.any(1).sum()), "top_m": top_m,
                "core": core, "inset_m": inset_m, "sigma_table": SIGMA}
        return cls({k: v.astype(np.float32) for k, v in y.items()}, weights, sigma, meta)

    def save(self, path):
        arrays = {}
        for f in FIELDS:
            arrays[f"y_{f}"], arrays[f"w_{f}"], arrays[f"sigma_{f}"] = self.y[f], self.w[f].astype(np.float32), self.sigma[f]
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path, xp=np):
        z = np.load(path)
        y = {f: xp.asarray(z[f"y_{f}"]) for f in FIELDS}
        w = {f: xp.asarray(z[f"w_{f}"].astype(np.float64)) for f in FIELDS}
        s = {f: xp.asarray(z[f"sigma_{f}"]) for f in FIELDS}
        return cls(y, w, s, json.loads(str(z["meta"])))

    def to(self, xp):
        return Target({k: xp.asarray(v) for k, v in self.y.items()}, {k: xp.asarray(v) for k, v in self.w.items()},
                      {k: xp.asarray(v) for k, v in self.sigma.items()}, self.meta)


class StateLoss:
    """J(x_N) in state space and dJ/dx_N; see the module text. ``scales``: [(length_m, weight), ..]; the default one
    scale at length 0 is the plain weighted squared error."""

    def __init__(self, mesh: MeshOps, target: Target, scales=((0.0, 1.0),), field_weights=None):
        self.mesh, self.t = mesh, target
        self.scales = [(float(l), float(c)) for l, c in scales]
        self.omega = {f: 1.0 for f in FIELDS}
        self.omega.update(field_weights or {})
        self.mask = (target.w["theta"].sum(0) > 0) * 1.0                        # judged cells, (ncol,)

    # x: {"theta", "qv", "uzonal", "umerid", "plow"} (nz, n) arrays
    def value_and_grad(self, x: dict, want_grad: bool = True):
        xp = array_module(x["theta"])
        mesh, t = self.mesh, self.t
        total, parts, grad = 0.0, {}, {}
        wsum = sum(self.omega[f] for f in FIELDS)
        for f in FIELDS:
            om = self.omega[f] / wsum
            if om == 0.0:
                continue
            d = (x[f].astype(xp.float64) - t.y[f].astype(xp.float64)) * self.mask[None, :]
            sig2 = (t.sigma[f].astype(xp.float64) ** 2)[:, None]
            jf, g = 0.0, xp.zeros_like(d) if want_grad else None
            for length, c in self.scales:
                ds = d if length == 0 else mesh.smooth(d, length)
                r = t.w[f] * ds / sig2
                jf += c * float((r * ds).sum())
                if want_grad:
                    gs = 2.0 * c * r
                    g = g + (gs if length == 0 else mesh.smooth_t(gs, length))
            parts[f] = jf
            total += om * jf
            if want_grad:
                grad[f] = om * g * self.mask[None, :]
        return total, parts, grad

    # -- the plan's own state fields ----------------------------------------------------------------------------------
    def from_prognostic(self, theta_m, qv, u, pressure_p, pressure_base_low):
        uz, um = self.mesh.reconstruct(u)
        return {"theta": theta_m / (1.0 + RVORD * qv), "qv": qv, "uzonal": uz, "umerid": um,
                "plow": pressure_base_low[None, :] + pressure_p[:1]}

    def grad_prognostic(self, theta_m, qv, grad: dict):
        """dJ/d(theta_m, qv, u, pressure_p) from dJ/d(theta, qv, uzonal, umerid, plow) (level-major, like the state)."""
        xp = array_module(theta_m)
        a = 1.0 + RVORD * qv.astype(xp.float64)
        gth = grad.get("theta", 0.0)
        out = {"theta_m": gth / a, "qv": grad.get("qv", 0.0) - gth * RVORD * theta_m / a ** 2}
        out["u"] = self.mesh.reconstruct_t(grad["uzonal"], grad["umerid"])
        gp = xp.zeros_like(theta_m, dtype=xp.float64)
        gp[0] = grad["plow"][0]
        out["pressure_p"] = gp
        return out


    # -- the reverse entry's named fields (the bridge reverse: hexad_field "theta", "rho", "u", species "qv") -------------------
    def from_entry(self, theta, qv, rho, u):
        """x from the reverse entry's fields: dry theta, qv, dry density rho, edge-normal u (level-major). The lowest
        level's pressure is MPAS's equation of state p = p0 (Rd rho theta (1 + Rv/Rd qv) / p0)^(cp/cv)."""
        uz, um = self.mesh.reconstruct(u)
        p1 = P0 * (RD * rho[:1] * theta[:1] * (1.0 + RVORD * qv[:1]) / P0) ** GAMMA
        return {"theta": theta, "qv": qv, "uzonal": uz, "umerid": um, "plow": p1}

    def grad_entry(self, theta, qv, rho, grad: dict):
        """dJ/d(theta, qv, rho, u) for the reverse entry (seed at the window end), level-major like the plan."""
        xp = array_module(theta)
        p1 = P0 * (RD * rho[:1] * theta[:1] * (1.0 + RVORD * qv[:1]) / P0) ** GAMMA
        gp = grad["plow"] * GAMMA * p1                                          # dJ/dln p, (1, ncol)
        g_th = xp.array(grad["theta"], dtype=xp.float64, copy=True)
        g_qv = xp.array(grad["qv"], dtype=xp.float64, copy=True)
        g_rho = xp.zeros(theta.shape, dtype=xp.float64)
        g_th[:1] += gp / theta[:1]
        g_qv[:1] += gp * RVORD / (1.0 + RVORD * qv[:1])
        g_rho[:1] = gp / rho[:1]
        return {"theta": g_th, "qv": g_qv, "rho": g_rho, "u": self.mesh.reconstruct_t(grad["uzonal"], grad["umerid"])}


def objective_terms(grad_prog: dict, table: dict, fmap, row=None, name="target loss gradient") -> dict:
    """The runner's --objective document for y = dJ/dx_N over a plan's window-end allocations (FieldMap.resolve as
    the door's metric_objective uses it). Words with a zero gradient are left out."""
    terms = []
    for field, g in grad_prog.items():
        f = fmap.resolve(table, field, row, window_end=True)
        g = np.asarray(g.get() if hasattr(g, "get") else g)
        nlev, ncol = f["shape"]
        if g.shape != (nlev, ncol):
            g = g[:nlev, :ncol]
        base, stride = f["offset"] // 4, f.get("read_shape", f["shape"])[-1]
        k, c = np.nonzero(g)
        terms.extend([[f["alloc"], int(base + kk * stride + cc), float(g[kk, cc])] for kk, cc in zip(k, c)])
    return {"name": name, "terms": terms}


def torch_loss(loss: StateLoss):
    """A torch.autograd.Function over StateLoss (torch is the caller's dependency, imported here only). Arrays cross
    by DLPack, no copies: J = f(theta, qv, uzonal, umerid, plow) as 0-d tensor; backward hands dJ/dx."""
    import torch
    import cupy

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            x = {f: cupy.from_dlpack(t.detach()) for f, t in zip(FIELDS, xs)}
            j, _, g = loss.value_and_grad(x)
            ctx.g = [torch.from_dlpack(g[f].astype(cupy.float32)) for f in FIELDS]
            return torch.tensor(j, device=xs[0].device, dtype=torch.float64)

        @staticmethod
        def backward(ctx, gj):
            return tuple(gj.to(g.dtype) * g for g in ctx.g)
    return _F.apply


# ---------------------------------------------------------------------------------------------------------------------
# observation space: stations, from the lowest model level
# ---------------------------------------------------------------------------------------------------------------------
class StationLoss:
    """2 m T, 2 m Td and 10 m wind at stations from the lowest level (see the module text). H is a fixed sparse map
    (3 nearest cells, inverse distance), so the operator is linear in theta, qv, uz, um except T and Td's use of the
    level's pressure (taken from the trajectory: a constant of the operator)."""

    LAPSE = 0.0065

    def __init__(self, static: dict, z_level1, ter, p_level1, obs: list, sigma=(1.5, 2.0, 2.5)):
        from math import radians
        lat, lon = np.asarray(static["latCell"]), np.asarray(static["lonCell"])
        xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
        self.obs = []
        idx, wts = [], []
        for o in obs:
            la, lo = radians(o["lat"]), radians(o["lon"])
            p = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
            d = np.arccos(np.clip(xyz @ p, -1, 1)) * 6371e3
            nn = np.argsort(d)[:3]
            if d[nn[0]] > 30e3 or (("bdyMaskCell" in static) and np.asarray(static["bdyMaskCell"])[nn].any()):
                continue
            w = 1.0 / np.maximum(d[nn], 1.0)
            idx.append(nn)
            wts.append(w / w.sum())
            self.obs.append(o)
        self.idx, self.wts = np.array(idx, int).reshape(-1, 3), np.array(wts).reshape(-1, 3)
        h = lambda f: (f[self.idx] * self.wts).sum(1)
        self.dz = np.array([o.get("elev", np.nan) for o in self.obs]) + 2.0 - h(np.asarray(z_level1))
        self.dz = np.where(np.isfinite(self.dz), self.dz, 0.0)
        self.p1 = h(np.asarray(p_level1))
        self.exner = (self.p1 / P0) ** (RD / CP)
        self.sigma = sigma

    def h(self, f):
        return (f[self.idx] * self.wts).sum(1)

    def ht(self, g, n):
        out = np.zeros(n)
        np.add.at(out, self.idx.ravel(), (g[:, None] * self.wts).ravel())
        return out

    def predict(self, theta1, qv1, uz1, um1):
        t2 = self.h(theta1) * self.exner - self.LAPSE * self.dz
        q = np.maximum(self.h(qv1), 1e-8)
        e = q * self.p1 / (0.622 + q) / 100.0                                    # hPa
        lne = np.log(e / 6.112)
        td = 273.15 + 243.5 * lne / (17.67 - lne)                                # Bolton (1980)
        return t2, td, self.h(uz1), self.h(um1), q, e, lne

    def value_and_grad(self, theta1, qv1, uz1, um1):
        n = theta1.shape[0]
        t2, td, u, v, q, e, lne = self.predict(theta1, qv1, uz1, um1)
        ob = lambda k: np.array([o.get(k, np.nan) for o in self.obs], float)
        st, sd, sw = self.sigma
        parts, grads = {}, {}
        dt, dd = t2 - ob("t2m"), td - ob("td2m")
        du, dv = u - ob("u10"), v - ob("v10")
        m = lambda d: np.isfinite(d)
        J = 0.0
        jt = np.where(m(dt), dt, 0.0) / st ** 2
        jd = np.where(m(dd), dd, 0.0) / sd ** 2
        ju, jv = np.where(m(du), du, 0.0) / sw ** 2, np.where(m(dv), dv, 0.0) / sw ** 2
        cnt = max(1, int(m(dt).sum() + m(dd).sum() + m(du).sum() + m(dv).sum()))
        J = float((jt * np.nan_to_num(dt)).sum() + (jd * np.nan_to_num(dd)).sum() + (ju * np.nan_to_num(du)).sum()
                  + (jv * np.nan_to_num(dv)).sum()) / cnt
        parts = {"t2m_rms": float(np.sqrt(np.nanmean(dt ** 2))), "td2m_rms": float(np.sqrt(np.nanmean(dd ** 2))),
                 "wind_rms": float(np.sqrt(np.nanmean(du ** 2 + dv ** 2))), "count": cnt}
        # dTd/dq through e(q): dTd/dlne = 243.5*17.67/(17.67-lne)^2; dlne/dq = 0.622/(q (0.622+q))
        dtd_dq = 243.5 * 17.67 / (17.67 - lne) ** 2 * 0.622 / (q * (0.622 + q))
        dtd_dq = np.where(self.h(qv1) > 1e-8, dtd_dq, 0.0)
        grads["theta"] = self.ht(2 * jt * self.exner / cnt, n)
        grads["qv"] = self.ht(2 * jd * dtd_dq / cnt, n)
        grads["uzonal"] = self.ht(2 * ju / cnt, n)
        grads["umerid"] = self.ht(2 * jv / cnt, n)
        return J, parts, grads

    @staticmethod
    def as_state_grad(grads: dict, nz: int, xp=np) -> dict:
        """The station gradient (lowest level) as a StateLoss-shaped gradient (nz, ncol) per field, plow zero, so
        StateLoss.grad_entry / grad_prognostic carry it to the reverse entry's or the plan's fields."""
        out = {}
        for f in ("theta", "qv", "uzonal", "umerid"):
            g = xp.zeros((nz, grads[f].shape[0]))
            g[0] = xp.asarray(grads[f])
            out[f] = g
        out["plow"] = xp.zeros((1, grads["theta"].shape[0]))
        return out


# ---------------------------------------------------------------------------------------------------------------------
# checks and command line
# ---------------------------------------------------------------------------------------------------------------------
def _dot(a, b):
    return float(sum(float((a[k].astype(np.float64) * b[k].astype(np.float64)).sum()) for k in a))


def gradient_check(loss: StateLoss, x: dict, *, seed=1, hs=(1e-1, 1e-2, 1e-3, 1e-4), prognostic=None,
                   entry=None) -> dict:
    """Central finite differences of J along random directions (each field scaled by its sigma) against <dJ/dx, dx>,
    float64 arithmetic. For the quadratic J the two agree to rounding at every h; for the prognostic chain (theta from
    theta_m and qv) the error falls as h^2."""
    xp = array_module(x["theta"])
    rs = np.random.default_rng(seed)
    out = {"state": [], "prognostic": [], "entry": [], "transpose": {}}
    x64 = {k: v.astype(xp.float64) for k, v in x.items()}
    j0, _, g = loss.value_and_grad(x64)
    dx = {f: xp.asarray(rs.standard_normal(x[f].shape)) * loss.t.sigma[f].astype(xp.float64)[:, None] for f in FIELDS}
    gd = sum(float((g[f] * dx[f]).sum()) for f in FIELDS)
    for h in hs:
        jp = loss.value_and_grad({f: x64[f] + h * dx[f] for f in FIELDS}, want_grad=False)[0]
        jm = loss.value_and_grad({f: x64[f] - h * dx[f] for f in FIELDS}, want_grad=False)[0]
        fd = (jp - jm) / (2 * h)
        out["state"].append({"h": h, "fd": fd, "grad_dot_dx": gd, "rel": abs(fd - gd) / max(abs(gd), 1e-300)})
    if prognostic is not None:
        thm, qv, u, pp, pbl = (prognostic[k].astype(xp.float64) for k in ("theta_m", "qv", "u", "pressure_p",
                                                                          "pressure_base_low"))
        J = lambda a, b, c, d: loss.value_and_grad(loss.from_prognostic(a, b, c, d, pbl), want_grad=False)[0]
        _, _, gs = loss.value_and_grad(loss.from_prognostic(thm, qv, u, pp, pbl))
        gp = loss.grad_prognostic(thm, qv, gs)
        d_thm = xp.asarray(rs.standard_normal(thm.shape)) * 1.0
        d_qv = xp.asarray(rs.standard_normal(qv.shape)) * 1e-4
        d_u = xp.asarray(rs.standard_normal(u.shape)) * 2.0
        d_pp = xp.asarray(rs.standard_normal(pp.shape)) * 100.0
        gd = float((gp["theta_m"] * d_thm).sum() + (gp["qv"] * d_qv).sum() + (gp["u"] * d_u).sum()
                   + (gp["pressure_p"] * d_pp).sum())
        for h in hs:
            fd = (J(thm + h * d_thm, qv + h * d_qv, u + h * d_u, pp + h * d_pp)
                  - J(thm - h * d_thm, qv - h * d_qv, u - h * d_u, pp - h * d_pp)) / (2 * h)
            out["prognostic"].append({"h": h, "fd": fd, "grad_dot_dx": gd, "rel": abs(fd - gd) / max(abs(gd), 1e-300)})
    if entry is not None:
        th, qv, rho, u = (entry[k].astype(xp.float64) for k in ("theta", "qv", "rho", "u"))
        J = lambda a, b, c, d: loss.value_and_grad(loss.from_entry(a, b, c, d), want_grad=False)[0]
        _, _, gs = loss.value_and_grad(loss.from_entry(th, qv, rho, u))
        ge = loss.grad_entry(th, qv, rho, gs)
        d = {"theta": xp.asarray(rs.standard_normal(th.shape)), "qv": xp.asarray(rs.standard_normal(qv.shape)) * 1e-4,
             "rho": xp.asarray(rs.standard_normal(rho.shape)) * 1e-3, "u": xp.asarray(rs.standard_normal(u.shape)) * 2.0}
        gd = sum(float((ge[k] * d[k]).sum()) for k in d)
        for h in hs:
            fd = (J(th + h * d["theta"], qv + h * d["qv"], rho + h * d["rho"], u + h * d["u"])
                  - J(th - h * d["theta"], qv - h * d["qv"], rho - h * d["rho"], u - h * d["u"])) / (2 * h)
            out["entry"].append({"h": h, "fd": fd, "grad_dot_dx": gd, "rel": abs(fd - gd) / max(abs(gd), 1e-300)})
    m = loss.mesh
    a = xp.asarray(rs.standard_normal((3, m.ncol)))
    b = xp.asarray(rs.standard_normal((3, m.ncol)))
    for length in sorted({l for l, _ in loss.scales if l > 0} | {150e3}):
        l1, l2 = float((m.smooth(a, length) * b).sum()), float((a * m.smooth_t(b, length)).sum())
        out["transpose"][f"smooth_{int(length / 1e3)}km"] = {"steps": m.steps_for(length), "Sx_y": l1, "x_STy": l2,
                                                           "rel": abs(l1 - l2) / abs(l1)}
    ue = xp.asarray(rs.standard_normal((3, m.nedges)))
    uz, um = m.reconstruct(ue)
    l1 = float((uz * a).sum() + (um * b).sum())
    l2 = float((ue * m.reconstruct_t(a, b)).sum())
    out["transpose"]["reconstruct"] = {"Ru_ab": l1, "u_RTab": l2, "rel": abs(l1 - l2) / abs(l1)}
    out["J0"] = j0
    return out


def state_from_file(loss: StateLoss, path, time_index=-1) -> dict:
    xp = loss.mesh.xp
    s = read_state(path, time_index)
    u = xp.asarray(s["u"])
    uz, um = loss.mesh.reconstruct(u)
    return {"theta": xp.asarray(s["theta"]), "qv": xp.asarray(s["qv"]), "uzonal": uz, "umerid": um,
            "plow": xp.asarray(s["pressure"][:1])}


def read_stations(root, valid: str, window_min: float = 10.0) -> list:
    """METAR rows (station,valid,lon,lat,elevation,tmpf,dwpf,drct,sknt,..) under root (every *.csv with that header)
    within window_min of valid (UTC 'YYYY-MM-DDTHH:MM'): per station the nearest row that carries a temperature."""
    import csv
    t0 = dt_parse(valid)
    best = {}
    for f in sorted(Path(root).rglob("*.csv")):
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            if not r.fieldnames or "tmpf" not in r.fieldnames or "valid" not in r.fieldnames:
                continue
            for row in r:
                try:
                    dtm = abs((dt_parse(row["valid"].replace(" ", "T")) - t0).total_seconds()) / 60.0
                except ValueError:
                    continue
                if dtm > window_min or not row.get("tmpf"):
                    continue
                if row["station"] in best and best[row["station"]][0] <= dtm:
                    continue
                num = lambda k: float(row[k]) if row.get(k) not in (None, "", "M") else float("nan")
                tk = (num("tmpf") - 32.0) * 5.0 / 9.0 + 273.15
                td = (num("dwpf") - 32.0) * 5.0 / 9.0 + 273.15
                spd, dr = num("sknt") * 0.514444, math.radians(num("drct")) if row.get("drct") else float("nan")
                best[row["station"]] = (dtm, {"id": row["station"], "lat": num("lat"), "lon": num("lon"),
                                              "elev": num("elevation"), "t2m": tk, "td2m": td,
                                              "u10": -spd * math.sin(dr), "v10": -spd * math.cos(dr)})
    return [v[1] for v in best.values()]


def dt_parse(text: str):
    import datetime as _dt
    return _dt.datetime.strptime(text[:16], "%Y-%m-%dT%H:%M")


def baseline(manifest: dict, xp, obs_roots: dict | None = None) -> list:
    """J of the plain model (the forecast frame at the window end) and of persistence (the start state) on every
    window, against the window's target file; the station loss beside it where a day has stations."""
    meshes = {}
    rows = []
    for w in manifest["windows"]:
        key = w.get("row", "q30")
        if key not in meshes:
            st0 = read_mesh(manifest["rows"][key]["static"])
            meshes[key] = (st0, MeshOps(st0, xp))
        st, ops = meshes[key]
        tg = Target.load(w["target"]["npz"], xp)
        loss = StateLoss(ops, tg)
        row = {"id": w["id"], "row": key, "day": w["day"], "split": w["split"], "steps": w["steps"]}
        for label, path in (("plain", w["plain_model_frame"]), ("persistence", w["start_state"])):
            if not Path(path).is_file():
                row[label] = None
                continue
            x = state_from_file(loss, path)
            j, parts, _ = loss.value_and_grad(x, want_grad=False)
            row[label] = {"J": j, "parts": parts}
            root = (obs_roots or {}).get(w["day"])
            if root:
                s = read_state(path)
                obs = read_stations(root, w["end"][:16])
                zmid = 0.5 * (s["zgrid"][0] + s["zgrid"][1]) if "zgrid" in s else None
                if obs and zmid is not None:
                    sl = StationLoss(st, zmid, s["zgrid"][0], s["pressure"][0], obs)
                    uz, um = MeshOps(st, np).reconstruct(s["u"][:1])
                    jo, po, _ = sl.value_and_grad(s["theta"][0], s["qv"][0], uz[0], um[0])
                    row[label]["obs"] = {"J": jo, **po}
        rows.append(row)
    return rows


def _xp(name):
    if name == "numpy":
        return np
    import cupy
    return cupy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m woof.hex.ad.target_loss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("target", help="build one window's target file from its analysis file(s)")
    t.add_argument("--static", required=True)
    t.add_argument("--analysis", action="append", required=True, help="analysis init.nc on the same cull (1 or 2)")
    t.add_argument("--alpha", type=float, action="append", help="time weights, one per analysis (default 1)")
    t.add_argument("--out", required=True)
    t.add_argument("--core", default="hrrr", choices=("hrrr", "all"))
    t.add_argument("--top-m", type=float, default=15000.0)
    s = sub.add_parser("score", help="J of a state file (init / history frame) against a target file")
    s.add_argument("--static", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--time-index", type=int, default=-1)
    s.add_argument("--scales", default="0:1", help="LENGTH_KM:WEIGHT,.. (multi-scale)")
    s.add_argument("--xp", default="cupy", choices=("cupy", "numpy"))
    c = sub.add_parser("check", help="the finite-difference check of the gradient and the transpose dot tests")
    c.add_argument("--static", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--state", required=True)
    c.add_argument("--scales", default="0:1,150:0.5")
    c.add_argument("--xp", default="cupy", choices=("cupy", "numpy"))
    c.add_argument("--out")
    b = sub.add_parser("baseline", help="plain model and persistence J on every window of a manifest")
    b.add_argument("--manifest", required=True)
    b.add_argument("--obs", action="append", default=[], help="DAY=DIR of METAR csv files (station loss)")
    b.add_argument("--xp", default="cupy", choices=("cupy", "numpy"))
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "baseline":
        m = json.loads(Path(a.manifest).read_text())
        rows = baseline(m, _xp(a.xp), dict(o.split("=", 1) for o in a.obs))
        Path(a.out).write_text(json.dumps(rows, indent=1))
        for r in rows:
            f = lambda k: ("%.4f" % r[k]["J"]) if r.get(k) else "-"
            print(r["id"], r["split"], "plain", f("plain"), "persistence", f("persistence"))
        return 0
    if a.cmd == "target":
        st = read_mesh(a.static)
        ops = MeshOps(st, np)
        alphas = a.alpha or [1.0] * len(a.analysis)
        tg = Target.build(ops, st, a.analysis, alphas, top_m=a.top_m, core=a.core)
        tg.save(a.out)
        print(json.dumps(tg.meta))
        return 0
    xp = _xp(a.xp)
    st = read_mesh(a.static)
    ops = MeshOps(st, xp)
    tg = Target.load(a.target, xp)
    scales = [(float(p.split(":")[0]) * 1e3, float(p.split(":")[1])) for p in a.scales.split(",")]
    loss = StateLoss(ops, tg, scales)
    x = state_from_file(loss, a.state, getattr(a, "time_index", -1))
    if a.cmd == "score":
        j, parts, _ = loss.value_and_grad(x, want_grad=False)
        print(json.dumps({"J": j, "parts": parts, "state": a.state, "target": a.target, "scales": scales}))
        return 0
    s = read_state(a.state)
    pb = s.get("pressure_base")
    prog = None
    if pb is not None:
        prog = {"theta_m": xp.asarray(s["theta"] * (1 + RVORD * s["qv"])), "qv": xp.asarray(s["qv"]),
                "u": xp.asarray(s["u"]), "pressure_p": xp.asarray(s["pressure"] - pb),
                "pressure_base_low": xp.asarray(pb[0])}
    ent = None
    if "rho" in s:
        ent = {"theta": xp.asarray(s["theta"]), "qv": xp.asarray(s["qv"]), "rho": xp.asarray(s["rho"]),
               "u": xp.asarray(s["u"])}
        p1 = P0 * (RD * s["rho"][0] * s["theta"][0] * (1 + RVORD * s["qv"][0]) / P0) ** GAMMA
        eos = {"p_eos_minus_p_file_max_Pa": float(np.abs(p1 - s["pressure"][0]).max()),
               "p_eos_minus_p_file_mean_Pa": float((p1 - s["pressure"][0]).mean())}
    res = gradient_check(loss, x, prognostic=prog, entry=ent)
    if ent is not None:
        res["equation_of_state"] = eos
    res.update({"state_file": a.state, "target_file": a.target, "scales": scales, "xp": a.xp})
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
