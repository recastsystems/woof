"""Write the BouLac column-oracle input grids (bl_pbl_physics = 8).

Each grid is one call shape of WRF's ``boulac`` entry: one level count, one
time step, one ``flag_bep`` / ``idiff`` setting, ``ncol`` columns.  The
Fortran driver (``run_bl_boulac.F90``) reads a grid, runs ``nsteps``
consecutive steps with TKE (and, under BEP, DLG_BEP) carried by the scheme
itself, applies each step's tendencies to the state in float32
(``x = x + dt*xten``), and records every step's inputs and outputs.

The cases are chosen to reach branches, not to look meteorological:

* regime families: dry convective (deep and shallow), stable nocturnal with
  a low-level jet, neutral windy, strong directional and speed shear,
  high terrain (thin air, coarser dz), cloud-topped (qc > 0), developed TKE;
* ``pbl_height`` never finding a level above thv(1) + 0.5 (``pblh = 0``,
  ``wstar = 0**(1/3)``) with a positive heat flux;
* exactly repeated potential temperatures (the ``bbb == 0`` arms of
  ``dissip_bougeault``, both ``th(izz) == th(iz)`` and not);
* calm lowest level with ``ust > 0`` (WRF's rural drag ``-ust**2/dz/|V|``
  is -inf there and the level-1 wind stays exactly 0);
* the ``exch >= 0.1`` floor, the ``tke >= temin`` floor, ``hfx <= 0``;
* BEP urban columns at frc_urb2d 0, 0.25, 0.6 and 1.0 with nonzero
  implicit and explicit source terms, building length scales (``dl_u > 0``,
  the harmonic-mean arm of ``length_bougeault``) and reduced
  ``sf_bep``/``vl_bep``;
* a stretched 61-level grid at dt = 10 s and a 1-step calm probe whose
  ``ust = 0`` column is WRF's 0/0 (the declared divergence).

Usage: python3 make_cases.py OUTDIR
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F = np.float32
CP = F(1004.5)   # module_model_constants.F cp = 7.*r_d/2. (r_d = 287.)
G = F(9.81)      # module_model_constants.F g

MAGIC = 0xB0B1A8


def stretched_dz(nz: int, dz0: float, ratio: float, dzmax: float) -> np.ndarray:
    dz = np.empty(nz, dtype=np.float64)
    for k in range(nz):
        dz[k] = min(dz0 * ratio ** k, dzmax)
    return dz


def faces(dz: np.ndarray) -> np.ndarray:
    z = np.zeros(dz.size + 1)
    z[1:] = np.cumsum(dz)
    return z


class Grid:
    def __init__(self, name, nz, dt, nsteps, flag_bep=0, idiff=0):
        self.name = name
        self.nz = nz
        self.dt = F(dt)
        self.nsteps = nsteps
        self.flag_bep = flag_bep
        self.idiff = idiff
        self.cols = []

    def add(self, label, **col):
        col["label"] = label
        self.cols.append(col)

    def write(self, outdir: Path) -> dict:
        nz, nc = self.nz, len(self.cols)
        names3 = ("dz8w", "u", "v", "th", "qv", "qc", "rho", "tke")
        names2 = ("hfx", "qfx", "ust", "frc_urb2d", "pblh")
        bep3 = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
                "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
                "dlg_bep", "dl_u_bep", "vl_bep")
        arrays = {}
        for name in names3 + bep3:
            arrays[name] = np.zeros((nz, nc), dtype=F)
        arrays["sf_bep"] = np.ones((nz + 1, nc), dtype=F)
        for name in names2:
            arrays[name] = np.zeros(nc, dtype=F)
        for name in ("vl_bep",):
            arrays[name][...] = F(1.0)
        for i, col in enumerate(self.cols):
            for name in names3 + bep3:
                if name in col:
                    arrays[name][:, i] = np.asarray(col[name], dtype=F)
            if "sf_bep" in col:
                arrays["sf_bep"][:, i] = np.asarray(col["sf_bep"], dtype=F)
            for name in names2:
                if name in col:
                    arrays[name][i] = F(col[name])
        path = outdir / f"{self.name}.in.bin"
        with open(path, "wb") as fh:
            fh.write(struct.pack("<6i", MAGIC, nz, nc, self.nsteps,
                                 self.flag_bep, self.idiff))
            fh.write(struct.pack("<3f", float(self.dt), float(CP), float(G)))
            # Fortran column-major (ncol fastest, then level): the on-disk
            # order is (level, column) row-major, i.e. arrays[name] as is.
            for name in names3:
                fh.write(np.ascontiguousarray(arrays[name]).tobytes())
            for name in names2:
                fh.write(np.ascontiguousarray(arrays[name]).tobytes())
            for name in bep3:
                fh.write(np.ascontiguousarray(arrays[name]).tobytes())
            fh.write(np.ascontiguousarray(arrays["sf_bep"]).tobytes())
        return {"grid": self.name, "nz": nz, "ncol": nc,
                "nsteps": self.nsteps, "flag_bep": self.flag_bep,
                "idiff": self.idiff, "dt": float(self.dt),
                "labels": [c["label"] for c in self.cols]}


def theta_profile(zc, kind, rng_shift=0.0):
    """Potential temperature at mass points for a regime family."""
    th = np.empty_like(zc)
    if kind == "convective":
        zi = 1400.0 + 300.0 * rng_shift
        for k, z in enumerate(zc):
            if z < 80.0:
                th[k] = 301.2 - 1.2 * z / 80.0
            elif z < zi:
                th[k] = 300.0 + 0.0002 * (z - 80.0)
            elif z < zi + 150.0:
                th[k] = 300.3 + 5.0 * (z - zi) / 150.0
            else:
                th[k] = 305.3 + 0.004 * (z - zi - 150.0)
    elif kind == "shallow_convective":
        zi = 450.0 + 100.0 * rng_shift
        for k, z in enumerate(zc):
            if z < zi:
                th[k] = 295.5 - 0.6 * np.exp(-z / 40.0)
            else:
                th[k] = 295.5 + 3.0 + 0.005 * (z - zi)
    elif kind == "stable":
        for k, z in enumerate(zc):
            if z < 300.0:
                th[k] = 285.0 + (0.012 + 0.004 * rng_shift) * z
            else:
                th[k] = 285.0 + 3.6 + 1.2 * rng_shift + 0.003 * (z - 300.0)
    elif kind == "neutral":
        for k, z in enumerate(zc):
            th[k] = 293.0 if z < 1000.0 else 293.0 + 0.003 * (z - 1000.0)
    elif kind == "weakly_stable":
        th = 290.0 + 0.0015 * zc + 0.003 * np.maximum(zc - 2000.0, 0.0)
    elif kind == "isentropic":
        th = np.full_like(zc, 300.0)
    elif kind == "steps":
        # Exactly repeated values in pairs: th(izz+1) == th(izz) for half the
        # layer pairs (bbb == 0) while th(izz) != th(iz).
        th = 292.0 + 0.5 * np.floor(zc / (2.0 * 60.0))
        th = th + 0.004 * np.maximum(zc - 1500.0, 0.0)
    else:
        raise ValueError(kind)
    return th


def wind_profile(zc, kind, speed, direction_deg=240.0):
    u = np.empty_like(zc)
    v = np.empty_like(zc)
    d = np.deg2rad(direction_deg)
    for k, z in enumerate(zc):
        if kind == "log":
            s = speed * np.log(z / 0.1 + 1.0) / np.log(1000.0 / 0.1 + 1.0)
            s = min(s, speed * 1.2)
            u[k], v[k] = -s * np.sin(d), -s * np.cos(d)
        elif kind == "jet":
            s = speed * np.exp(-((z - 300.0) / 200.0) ** 2) + 3.0 * (z / 3000.0)
            u[k], v[k] = s, 0.3 * s
        elif kind == "shear":
            s = 30.0 * np.tanh((z - 400.0) / 250.0)
            u[k], v[k] = s, -0.5 * s + 4.0 * np.sin(z / 300.0)
        elif kind == "turning":
            ang = d + np.deg2rad(60.0) * min(z / 1500.0, 1.0)
            s = speed * min(1.0, 0.4 + z / 800.0)
            u[k], v[k] = -s * np.sin(ang), -s * np.cos(ang)
        else:
            raise ValueError(kind)
    return u, v


def moisture(zc, kind, cloud=False):
    if kind == "moist":
        qv = 0.016 * np.exp(-zc / 2500.0)
    else:
        qv = 0.006 * np.exp(-zc / 3000.0)
    qc = np.zeros_like(zc)
    if cloud:
        qc = np.where((zc > 900.0) & (zc < 1500.0),
                      4.0e-4 * np.sin(np.pi * (zc - 900.0) / 600.0), 0.0)
    return qv, qc


def density(zc, scale=1.0):
    return 1.2 * scale * np.exp(-zc / 8500.0)


def tke_profile(zc, kind, zi=1200.0):
    if kind == "cold":
        return np.full_like(zc, 1.0e-4)
    if kind == "developed":
        return np.maximum(2.5 * (1.0 - zc / zi) * np.exp(-zc / (0.6 * zi)),
                          1.0e-4)
    if kind == "stable_developed":
        return np.maximum(0.3 * np.exp(-zc / 120.0), 1.0e-4)
    raise ValueError(kind)


def rural_grid(name, nz, dt, nsteps, dz0, ratio, dzmax):
    g = Grid(name, nz, dt, nsteps)
    dz = stretched_dz(nz, dz0, ratio, dzmax)
    zc = faces(dz)[:-1] + 0.5 * dz
    variants = range(4)
    for v in variants:
        sh = v / 3.0
        # 1 deep convective, cold and developed TKE
        th = theta_profile(zc, "convective", sh)
        u, w = wind_profile(zc, "log", 4.0 + 2.0 * v)
        qv, qc = moisture(zc, "dry")
        g.add(f"convective-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "developed" if v % 2 else "cold"),
              hfx=150.0 + 80.0 * v, qfx=8.0e-5, ust=0.3 + 0.05 * v)
        # 2 shallow convective, moist
        th = theta_profile(zc, "shallow_convective", sh)
        u, w = wind_profile(zc, "turning", 6.0 + v)
        qv, qc = moisture(zc, "moist")
        g.add(f"shallow-convective-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv,
              qc=qc, rho=density(zc), tke=tke_profile(zc, "developed", 500.0),
              hfx=60.0 + 20.0 * v, qfx=1.5e-4, ust=0.25)
        # 3 stable nocturnal with a low-level jet
        th = theta_profile(zc, "stable", sh)
        u, w = wind_profile(zc, "jet", 10.0 + 2.0 * v)
        qv, qc = moisture(zc, "dry")
        g.add(f"stable-jet-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(
                  zc, "stable_developed" if v % 2 else "cold"),
              hfx=-10.0 - 12.0 * v, qfx=-5.0e-6 * v, ust=0.05 + 0.05 * v)
        # 4 neutral windy
        th = theta_profile(zc, "neutral")
        u, w = wind_profile(zc, "log", 15.0 + 3.0 * v, 200.0 + 20.0 * v)
        qv, qc = moisture(zc, "dry")
        g.add(f"neutral-windy-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "developed", 900.0),
              hfx=0.0 if v < 2 else 5.0, qfx=0.0, ust=0.6 + 0.12 * v)
        # 5 strong shear, weakly stable
        th = theta_profile(zc, "weakly_stable")
        u, w = wind_profile(zc, "shear", 0.0)
        u = u * (0.6 + 0.2 * v)
        qv, qc = moisture(zc, "dry")
        g.add(f"strong-shear-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "cold"),
              hfx=-5.0 + 10.0 * v, qfx=1.0e-5, ust=0.4)
        # 6 high terrain: thin air, coarser layers, convective
        dzh = dz * 1.15
        zch = faces(dzh)[:-1] + 0.5 * dzh
        th = theta_profile(zch, "convective", sh) + 12.0
        u, w = wind_profile(zch, "turning", 8.0 + 2.0 * v)
        qv, qc = moisture(zch, "dry")
        g.add(f"high-terrain-{v}", dz8w=dzh, u=u, v=w, th=th, qv=qv * 0.5,
              qc=qc, rho=density(zch, 0.62), tke=tke_profile(zch, "developed"),
              hfx=250.0 + 40.0 * v, qfx=4.0e-5, ust=0.35)
        # 7 cloud-topped moist boundary layer
        th = theta_profile(zc, "convective", 0.0)
        u, w = wind_profile(zc, "log", 7.0)
        qv, qc = moisture(zc, "moist", cloud=True)
        g.add(f"cloud-topped-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "developed"),
              hfx=40.0 + 30.0 * v, qfx=1.2e-4, ust=0.3)
        # 8 isentropic column: pbl_height finds nothing (pblh = 0), bbb == 0
        th = theta_profile(zc, "isentropic")
        u, w = wind_profile(zc, "log", 5.0 + v)
        qv = np.full_like(zc, 0.004)
        qc = np.zeros_like(zc)
        g.add(f"isentropic-pblh0-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv,
              qc=qc, rho=density(zc), tke=tke_profile(
                  zc, "developed" if v % 2 else "cold"),
              hfx=30.0 * v, qfx=0.0, ust=0.2 + 0.1 * v)
        # 9 exactly repeated theta pairs
        th = theta_profile(zc, "steps")
        u, w = wind_profile(zc, "turning", 9.0)
        qv, qc = moisture(zc, "dry")
        g.add(f"repeated-theta-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "developed", 1500.0),
              hfx=20.0 * (v - 1), qfx=2.0e-5, ust=0.3)
        # 10 calm lowest level, ust > 0: -inf drag at level 1
        th = theta_profile(zc, "stable", sh)
        u, w = wind_profile(zc, "jet", 6.0)
        u[0] = 0.0
        w[0] = 0.0
        qv, qc = moisture(zc, "dry")
        g.add(f"calm-level1-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "cold"),
              hfx=-5.0, qfx=0.0, ust=0.02 + 0.01 * v)
        # 11 very weak everything: exch floor, tke floor
        th = theta_profile(zc, "stable", 1.0) + 0.01 * zc
        u = np.full_like(zc, 0.5)
        w = np.full_like(zc, -0.25)
        qv, qc = moisture(zc, "dry")
        g.add(f"floors-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=np.full_like(zc, 1.0e-4),
              hfx=-1.0, qfx=0.0, ust=0.01 + 0.01 * v)
        # 12 deep convective with strong developed TKE filling the column
        th = theta_profile(zc, "convective", 2.0 + v)
        u, w = wind_profile(zc, "turning", 12.0)
        qv, qc = moisture(zc, "moist", cloud=v >= 2)
        g.add(f"deep-developed-{v}", dz8w=dz, u=u, v=w, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "developed", 3500.0) * 2.0,
              hfx=400.0 + 50.0 * v, qfx=2.0e-4, ust=0.5)
    return g


def bep_grid(name, nz, dt, nsteps):
    g = Grid(name, nz, dt, nsteps, flag_bep=1)
    dz = stretched_dz(nz, 10.0, 1.10, 500.0)
    z = faces(dz)
    zc = z[:-1] + 0.5 * dz
    for frc in (0.0, 0.25, 0.6, 1.0):
        for kind in range(6):
            th = theta_profile(zc, ("convective", "stable", "neutral",
                                    "shallow_convective", "weakly_stable",
                                    "convective")[kind], kind / 5.0)
            u, w = wind_profile(zc, ("log", "jet", "log", "turning", "shear",
                                     "turning")[kind], 6.0 + kind)
            qv, qc = moisture(zc, "moist" if kind % 2 else "dry",
                              cloud=kind == 5)
            rho = density(zc)
            hfx = (180.0, -20.0, 0.0, 90.0, 10.0, 260.0)[kind]
            ust = (0.4, 0.12, 0.7, 0.3, 0.45, 0.5)[kind]
            qfx = (6.0e-5, -2.0e-6, 0.0, 1.2e-4, 2.0e-5, 1.5e-4)[kind]
            # Building canopy over the lowest ~10 levels (z < ~150 m).
            canopy = zc < (60.0 + 20.0 * kind)
            nb = int(canopy.sum())
            speed1 = float(np.hypot(u[0], w[0]))
            a_u = np.zeros(nz)
            a_v = np.zeros(nz)
            b_u = np.zeros(nz)
            b_v = np.zeros(nz)
            a_t = np.zeros(nz)
            b_t = np.zeros(nz)
            a_q = np.zeros(nz)
            b_q = np.zeros(nz)
            a_e = np.zeros(nz)
            b_e = np.zeros(nz)
            dlg = np.zeros(nz)
            dl_u = np.zeros(nz)
            sf = np.ones(nz + 1)
            vl = np.ones(nz)
            if frc > 0.0:
                # BEP's weighted source terms (urban_bep_couple): drag on the
                # canopy levels, a wall/roof heat source, a TKE wake source,
                # fractions of the horizontal surface and of the volume.
                for k in range(nb):
                    depth = 1.0 - k / max(nb, 1)
                    a_u[k] = -frc * (0.02 + 0.01 * kind) * depth
                    a_v[k] = a_u[k]
                    b_u[k] = frc * 1.0e-4 * (kind - 2.5) * depth
                    b_v[k] = -0.5 * b_u[k]
                    a_t[k] = -frc * 1.0e-5 * depth
                    b_t[k] = frc * (2.0e-3 + 5.0e-4 * kind) * depth
                    b_q[k] = frc * 1.0e-7 * depth
                    b_e[k] = frc * (0.01 + 0.004 * kind) * depth
                    dlg[k] = 0.5 * zc[k] + 2.0 + kind
                    dl_u[k] = frc * (8.0 + 4.0 * kind) * depth
                    sf[k] = 1.0 - frc * 0.5 * depth
                    vl[k] = 1.0 - frc * 0.45 * depth
                # BEP's kte+1 face is unobstructed (module_sf_bep.F:573).
                sf[nz] = 1.0
            # The coupling folds the rural surface flux and drag into level 1
            # on every column once flag_bep is on (urban_bep_couple).
            a_u[0] += -(1.0 - frc) * ust * ust / dz[0] / max(speed1, 0.1)
            a_v[0] = a_u[0]
            b_t[0] += (1.0 - frc) * hfx / dz[0] / rho[0] / float(CP)
            b_q[0] += (1.0 - frc) * qfx / dz[0] / rho[0]
            tke = tke_profile(zc, "developed" if kind % 2 == 0 else "cold")
            g.add(f"bep-frc{frc:g}-k{kind}", dz8w=dz, u=u, v=w, th=th, qv=qv,
                  qc=qc, rho=rho, tke=tke, hfx=hfx, qfx=qfx, ust=ust,
                  frc_urb2d=frc, a_u_bep=a_u, a_v_bep=a_v, a_t_bep=a_t,
                  a_q_bep=a_q, a_e_bep=a_e, b_u_bep=b_u, b_v_bep=b_v,
                  b_t_bep=b_t, b_q_bep=b_q, b_e_bep=b_e, dlg_bep=dlg,
                  dl_u_bep=dl_u, vl_bep=vl, sf_bep=sf)
    return g


def calm_probe_grid(name, nz):
    """One step, ust = 0 on a calm lowest level: WRF's 0/0 (declared)."""
    g = Grid(name, nz, 30.0, 1)
    dz = stretched_dz(nz, 50.0, 1.09, 600.0)
    zc = faces(dz)[:-1] + 0.5 * dz
    th = theta_profile(zc, "stable", 0.5)
    u, w = wind_profile(zc, "jet", 6.0)
    qv, qc = moisture(zc, "dry")
    for label, ust, calm in (("calm-ust0", 0.0, True),
                             ("windy-ust0", 0.0, False),
                             ("calm-ust-small", 1.0e-3, True)):
        uu, ww = u.copy(), w.copy()
        if calm:
            uu[0] = 0.0
            ww[0] = 0.0
        g.add(label, dz8w=dz, u=uu, v=ww, th=th, qv=qv, qc=qc,
              rho=density(zc), tke=tke_profile(zc, "cold"),
              hfx=-2.0, qfx=0.0, ust=ust)
    return g


def idiff_grid(name, nz):
    """idiff = 1: TKE, length scale and exchange coefficients only."""
    base = rural_grid(name, nz, 30.0, 3, 50.0, 1.09, 600.0)
    g = Grid(name, nz, 30.0, 3, idiff=1)
    g.cols = base.cols[:12]
    return g


def main(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    grids = [
        rural_grid("rural40", 40, 30.0, 10, 50.0, 1.09, 600.0),
        rural_grid("rural61", 61, 10.0, 6, 20.0, 1.06, 450.0),
        bep_grid("bep40", 40, 20.0, 10),
        calm_probe_grid("calmprobe40", 40),
        idiff_grid("idiff40", 40),
    ]
    index = [g.write(out) for g in grids]
    (out / "cases-index.json").write_text(json.dumps(index, indent=1) + "\n")
    total = sum(i["ncol"] for i in index)
    print(f"wrote {len(index)} grids, {total} columns, to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
