"""Grell-Devenyi (cu_physics = 93) oracle cases: writer and reader.

The cases are synthetic soundings built from a handful of regime templates
(deep tropical, ocean, shallow, capped, dry, moist-neutral, strong shear,
night, elevated terrain, cold, subsiding, strongly forced), each perturbed
with a fixed seed so every member is a different column.  Nothing here
computes an expected value: the WRF program run_gd.F90 reads the files this
module writes and every output word comes from the scheme itself.

Three fixtures:

* ``columns``  -- 120 columns, one row, periodic in both directions (so the
  driver's 4-column edge band admits every column), one call;
* ``edge``     -- a 12 x 11 non-periodic domain, one call: the edge band
  (module_cu_gd.F:181-194, 310-327) is what this one grades;
* ``cadence``  -- 16 columns over 6 steps with stepcu = 3, the inputs
  drifting every step: the call-or-hold cadence, PRATEC persistence and
  RAINC accumulation, and the APR_* accumulators across calls.

Every fixture runs with F_QI true and false (the warm-rain arm: no QI in the
moist package, so the 258 K split is off and all condensate is RQCCUTEN).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

F4 = np.float32
LEVEL_INPUTS = ("u", "v", "w", "t", "q", "p", "pi", "rho", "dz8w", "p8w",
                "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
SURFACE_INPUTS = ("ht", "xland", "gsw")
ACCUMULATORS = ("htop", "hbot", "apr_gr", "apr_w", "apr_mc", "apr_st",
                "apr_as", "apr_capma", "apr_capme", "apr_capmi")
LEVEL_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "gdc",
                 "gdc2")
SURFACE_OUTPUTS = ("raincv", "pratec", "rainc", "htop", "hbot", "apr_gr",
                   "apr_w", "apr_mc", "apr_st", "apr_as", "apr_capma",
                   "apr_capme", "apr_capmi", "mass_flux")
ENSDIM = 144

#: WRF module_model_constants as the cumulus driver hands them to GRELLDRV.
XLV = F4(2.5e6)
R_D = F4(287.0)
CP = F4(F4(7.0) * R_D / F4(2.0))
G = F4(9.81)
R_V = F4(461.6)

REGIMES = ("deep_land", "deep_ocean", "shallow", "capped", "dry",
           "moist_neutral", "strong_shear", "night_land", "elevated",
           "cold", "subsiding", "strong_forcing", "ocean_capped",
           "night_ocean", "marginal")


@dataclass
class Fixture:
    name: str
    nx: int
    ny: int
    nz: int
    nsteps: int
    stepcu: int
    periodic_x: bool
    periodic_y: bool
    dt: float
    dx: float
    regimes: list = field(default_factory=list)       # per column, j-major
    steps: list = field(default_factory=list)         # per step: dict of arrays
    accum: dict = field(default_factory=dict)


def _eta(nz: int):
    full = 1.0 - (np.arange(nz + 1) / nz) ** 1.25
    full[0], full[-1] = 1.0, 0.0
    return full


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _column(regime: str, rng, nz: int):
    """One column's inputs on levels 1..nz (float64; cast at write time)."""
    ptop = 5000.0
    ht = 0.0
    ps = rng.uniform(98500.0, 101500.0)
    t0 = rng.uniform(296.0, 304.0)
    lapse = rng.uniform(6.2, 7.0) * 1e-3
    rh_sfc, rh_top = rng.uniform(0.65, 0.85), rng.uniform(0.3, 0.55)
    u0, v0 = rng.uniform(-5, 5), rng.uniform(-5, 5)
    shear = rng.uniform(5, 15)
    w0 = rng.uniform(0.02, 0.25)
    xland, gsw = 1.0, rng.uniform(300.0, 800.0)
    th_adv = rng.uniform(-4e-5, -1e-5)      # upper-level advective cooling
    qv_adv = rng.uniform(0.5e-8, 4e-8)      # low-level moistening
    bl_heat = rng.uniform(1e-4, 6e-4)
    cap = 0.0
    cap_base = 0.0                          # metres above the BL top
    if regime == "deep_ocean":
        xland = 2.0
        t0 = rng.uniform(299.0, 303.0)
        rh_sfc = rng.uniform(0.8, 0.9)
        bl_heat = rng.uniform(0.0, 1e-4)
    elif regime == "shallow":
        # a strong inversion 1.5-3 km above the mixed layer: tops stop low
        rh_top = rng.uniform(0.05, 0.2)
        w0 = rng.uniform(0.0, 0.05)
        cap, cap_base = rng.uniform(4.0, 9.0), rng.uniform(1500.0, 3000.0)
    elif regime == "capped":
        cap = rng.uniform(3.0, 7.0)
    elif regime == "ocean_capped":
        xland = 2.0
        cap = rng.uniform(0.5, 3.0)
        rh_sfc = rng.uniform(0.75, 0.9)
    elif regime == "night_ocean":
        xland, gsw = 2.0, 0.0
        rh_sfc = rng.uniform(0.75, 0.9)
        bl_heat = rng.uniform(-1e-4, 1e-4)
    elif regime == "marginal":
        lapse = rng.uniform(4.8, 5.6) * 1e-3
        rh_sfc = rng.uniform(0.6, 0.72)
        th_adv = rng.uniform(0.0, 3e-5)
        qv_adv = rng.uniform(-2e-8, 0.5e-8)
        w0 = rng.uniform(-0.05, 0.05)
    elif regime == "dry":
        rh_sfc, rh_top = rng.uniform(0.08, 0.25), rng.uniform(0.03, 0.1)
    elif regime == "moist_neutral":
        lapse = rng.uniform(5.2, 5.8) * 1e-3
        rh_sfc, rh_top = rng.uniform(0.92, 0.98), rng.uniform(0.85, 0.95)
    elif regime == "strong_shear":
        shear = rng.uniform(30.0, 55.0)
    elif regime == "night_land":
        gsw = 0.0
        bl_heat = -rng.uniform(1e-4, 4e-4)
    elif regime == "elevated":
        ht = rng.uniform(1200.0, 2200.0)
        ps = 101325.0 * np.exp(-ht / 8000.0) + rng.uniform(-500, 500)
        t0 = rng.uniform(290.0, 298.0)
    elif regime == "cold":
        t0 = rng.uniform(268.0, 278.0)
        rh_sfc = rng.uniform(0.8, 0.95)
    elif regime == "subsiding":
        w0 = -rng.uniform(0.02, 0.2)
        th_adv = rng.uniform(0.5e-5, 2e-5)
    elif regime == "strong_forcing":
        th_adv = rng.uniform(-1.5e-4, -6e-5)
        qv_adv = rng.uniform(5e-8, 1.2e-7)
        w0 = rng.uniform(0.2, 0.6)

    eta = _eta(nz)
    p8w = ptop + eta * (ps - ptop)                    # faces 1..nz+1
    p = 0.5 * (p8w[:-1] + p8w[1:])                    # mass levels
    t = np.empty(nz)
    q = np.empty(nz)
    z = np.empty(nz)
    zprev, pprev, tvprev = ht, ps, t0
    ztrop = rng.uniform(11500.0, 14500.0)
    # A well-mixed boundary layer (dry adiabatic, constant q, so RH rises
    # toward its top the way a real afternoon mixed layer's does) under a
    # free troposphere of the regime's lapse rate and RH.
    hbl = rng.uniform(500.0, 1500.0)
    qbl = rh_sfc * _qsat(t0, ps)
    tbl = t0 - 9.8e-3 * hbl
    rh_free = min(rh_sfc + 0.1, 0.95)
    for k in range(nz):
        # three passes of the hypsometric step settle T(z) and z together
        zk = zprev
        for _ in range(3):
            zrel = zk - ht
            if zrel < hbl:
                tk = t0 - 9.8e-3 * zrel
                qk = min(qbl, _qsat(tk, p[k]))
            else:
                tk = tbl - lapse * (min(zrel, ztrop - ht) - hbl)
                tk = max(tk, 205.0)
                above = zrel - hbl - cap_base
                if cap and 0.0 <= above < 1200.0:
                    tk += cap * (1.0 - above / 1200.0)
                frac = min(max((zrel - hbl) / (10000.0 - hbl), 0.0), 1.0)
                rh = rh_free + (rh_top - rh_free) * frac
                if cap and above >= 0.0:
                    rh *= 0.35
                qk = rh * _qsat(tk, p[k])
            tv = tk * (1.0 + 0.608 * qk)
            zk = zprev + 287.0 * 0.5 * (tv + tvprev) / 9.81 * np.log(pprev / p[k])
        t[k], q[k], z[k] = tk, max(qk, 1e-9), zk
        zprev, pprev, tvprev = zk, p[k], tv
    zrel = z - ht
    u = u0 + shear * zrel / 10000.0 + rng.normal(0, 0.5, nz)
    v = v0 + 0.3 * shear * zrel / 10000.0 + rng.normal(0, 0.5, nz)
    ztop = z[-1] - ht
    wf = w0 * np.sin(np.pi * np.clip(zrel / min(ztop, 16000.0), 0, 1))
    wf[0] = 0.0
    rho = p / (287.0 * t * (1.0 + 0.608 * q))
    pi = (p / 1.0e5) ** (287.0 / 1004.5)
    dz8w = np.diff(np.concatenate(([ht], 0.5 * (z[:-1] + z[1:]), [z[-1] + 500.0])))
    upper = (zrel > 3000.0) & (zrel < 12000.0)
    lower = zrel < 1500.0
    rthften = np.where(upper, th_adv, 0.2 * th_adv) + rng.normal(0, 2e-6, nz)
    rqvften = np.where(lower, qv_adv, 0.0) + rng.normal(0, 2e-9, nz)
    rthraten = np.full(nz, -1.5 / 86400.0) + rng.normal(0, 2e-6, nz)
    rthblten = np.where(zrel < 800.0, bl_heat, 0.0)
    rqvblten = np.where(zrel < 800.0, abs(bl_heat) * 3e-4, 0.0)
    return dict(u=u, v=v, w=wf, t=t, q=q, p=p, pi=pi, rho=rho, dz8w=dz8w,
                p8w=p8w[:-1], rthften=rthften, rqvften=rqvften,
                rthraten=rthraten, rthblten=rthblten, rqvblten=rqvblten,
                ht=ht, xland=xland, gsw=gsw)


def _assemble(columns, nx, ny, nz):
    out = {}
    for name in LEVEL_INPUTS:
        a = np.empty((ny, nz, nx))
        for c, col in enumerate(columns):
            a[c // nx, :, c % nx] = col[name]
        out[name] = a.astype(F4)
    for name in SURFACE_INPUTS:
        a = np.empty((ny, nx))
        for c, col in enumerate(columns):
            a[c // nx, c % nx] = col[name]
        out[name] = a.astype(F4)
    return out


def build_fixture(name: str, *, nz: int = 40, seed: int = 93) -> Fixture:
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    if name == "columns":
        per = 8
        nx, ny, nsteps, stepcu, px, py, dt = len(REGIMES) * per, 1, 1, 1, True, True, 60.0
        regimes = [r for r in REGIMES for _ in range(per)]
    elif name == "edge":
        nx, ny, nsteps, stepcu, px, py, dt = 12, 11, 1, 1, False, False, 60.0
        deep = ("deep_land", "deep_ocean", "strong_forcing", "moist_neutral")
        regimes = [deep[c % len(deep)] for c in range(nx * ny)]
    elif name == "cadence":
        nx, ny, nsteps, stepcu, px, py, dt = 16, 1, 6, 3, True, True, 30.0
        deep = ("deep_land", "deep_ocean", "strong_forcing", "moist_neutral")
        regimes = [deep[c % len(deep)] for c in range(nx * ny)]
    else:
        raise ValueError(name)
    columns = [_column(r, rng, nz) for r in regimes]
    fx = Fixture(name, nx, ny, nz, nsteps, stepcu, px, py, dt, 12000.0,
                 regimes)
    base = _assemble(columns, nx, ny, nz)
    for s in range(nsteps):
        step = {k: v.copy() for k, v in base.items()}
        if s:
            # the state drifts between calls: warming low levels, cooling
            # aloft, a moisture pulse -- enough that each due call differs
            step["t"] = (base["t"] + F4(0.15 * s) * np.where(
                np.arange(nz)[None, :, None] < 8, 1, -1)).astype(F4)
            step["q"] = (base["q"] * F4(1.0 + 0.01 * s)).astype(F4)
            step["rthften"] = (base["rthften"] * F4(1.0 + 0.2 * s)).astype(F4)
        fx.steps.append(step)
    acc = {n: np.zeros((ny, nx), F4) for n in ACCUMULATORS}
    acc["hbot"][...] = F4(nz + 1)
    fx.accum = acc
    return fx


def write_input(fx: Fixture, path: Path, f_qi: bool) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8i", fx.nx, fx.ny, fx.nz, fx.nsteps, fx.stepcu,
                             int(fx.periodic_x), int(fx.periodic_y),
                             int(f_qi)))
        fh.write(np.array([fx.dt, fx.dx, XLV, CP, G, R_V], "<f4").tobytes())
        for s, step in enumerate(fx.steps):
            for name in LEVEL_INPUTS:
                fh.write(np.ascontiguousarray(step[name], "<f4").tobytes())
            for name in SURFACE_INPUTS:
                fh.write(np.ascontiguousarray(step[name], "<f4").tobytes())
            if s == 0:
                for name in ACCUMULATORS:
                    fh.write(np.ascontiguousarray(fx.accum[name], "<f4").tobytes())


def read_output(path: Path, nx: int, ny: int, nz: int, nsteps: int) -> list:
    raw = Path(path).read_bytes()
    off = 0
    steps = []

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a

    for _ in range(nsteps):
        rec = {"due": int(take("<i4", 1)[0])}
        for name in LEVEL_OUTPUTS:
            rec[name] = take("<f4", nx * nz * ny).reshape(ny, nz, nx)
        sfc = take("<f4", nx * ny * len(SURFACE_OUTPUTS))
        for n, name in enumerate(SURFACE_OUTPUTS):
            rec[name] = sfc[n * nx * ny:(n + 1) * nx * ny].reshape(ny, nx)
        rec["ktop_deep"] = take("<i4", nx * ny).reshape(ny, nx)
        ens = take("<f4", 2 * nx * ny * ENSDIM)
        # Fortran (ims:ime, jms:jme, 1:ensdim): i fastest, then j, then n.
        rec["xf_ens"] = ens[:nx * ny * ENSDIM].reshape(ENSDIM, ny, nx)
        rec["pr_ens"] = ens[nx * ny * ENSDIM:].reshape(ENSDIM, ny, nx)
        steps.append(rec)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return steps


def read_input(path: Path) -> dict:
    """Parse a run_gd.F90 input file back into header, steps, accumulators."""
    raw = Path(path).read_bytes()
    hdr = struct.unpack_from("<8i", raw, 0)
    nx, ny, nz, nsteps, stepcu, px, py, fqi = hdr
    con = np.frombuffer(raw, "<f4", 6, 32)
    off = 32 + 24
    n3, n2 = nx * nz * ny, nx * ny
    steps = []
    acc = None
    for s in range(nsteps):
        step = {}
        for name in LEVEL_INPUTS:
            step[name] = np.frombuffer(raw, "<f4", n3, off).reshape(ny, nz, nx)
            off += 4 * n3
        for name in SURFACE_INPUTS:
            step[name] = np.frombuffer(raw, "<f4", n2, off).reshape(ny, nx)
            off += 4 * n2
        if s == 0:
            acc = {}
            for name in ACCUMULATORS:
                acc[name] = np.frombuffer(raw, "<f4", n2, off).reshape(ny, nx)
                off += 4 * n2
        steps.append(step)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return dict(nx=nx, ny=ny, nz=nz, nsteps=nsteps, stepcu=stepcu,
                periodic_x=bool(px), periodic_y=bool(py), f_qi=bool(fqi),
                dt=F4(con[0]), dx=F4(con[1]), steps=steps, accum=acc)


#: gd.cu's input level-field order, as file field names.
KERNEL_LEVEL_INPUTS = ("u", "v", "w", "t", "q", "p", "pi", "rho", "rthften",
                       "rqvften", "rthraten", "rthblten", "rqvblten")


def emulate_driver(inp: dict, run_columns) -> list:
    """run_gd.F90's driver around a column kernel, in run_gd's output layout.

    ``run_columns(lvin, scin, iin)`` takes gd.cu's packed inputs ((ncol, 13,
    nz), (ncol, 15) float32, (ncol, 2) int32) and returns NumPy
    ``(lev (ncol,6,nz), sca (ncol,13), isc (ncol,5), ens (ncol,288))``.  The
    WRF-array persistence is the same as gd_host_harness.cpp's: outside the
    j band nothing is written but RAINCV/PRATEC/ktop_deep reset; RAINC
    accumulates PRATEC*DT every step.
    """
    nx, ny, nz = inp["nx"], inp["ny"], inp["nz"]
    nc = nx * ny
    dt = inp["dt"]
    band = band_flags(nx, ny, inp["periodic_x"], inp["periodic_y"]).reshape(nc)
    tend = np.zeros((6, ny, nz, nx), F4)
    raincv = np.zeros(nc, F4)
    pratec = np.zeros(nc, F4)
    rainc = np.zeros(nc, F4)
    acc = inp["accum"]
    htop = acc["htop"].reshape(nc).copy()
    hbot = acc["hbot"].reshape(nc).copy()
    apr = np.stack([acc[n].reshape(nc) for n in ACCUMULATORS[2:]]).copy()
    ktop = np.zeros(nc, np.int32)
    xf = np.zeros((ENSDIM, nc), F4)
    pr = np.zeros((ENSDIM, nc), F4)
    out = []
    for s, step in enumerate(inp["steps"], start=1):
        due = int(s == 1 or inp["stepcu"] == 1 or s % inp["stepcu"] == 0)
        if due:
            lvin = np.stack([step[n].transpose(0, 2, 1).reshape(nc, nz)
                             for n in KERNEL_LEVEL_INPUTS], axis=1).astype(F4)
            scin = np.empty((nc, 15), F4)
            scin[:, 0] = step["p8w"][:, 0, :].reshape(nc)
            scin[:, 1] = step["ht"].reshape(nc)
            scin[:, 2] = step["xland"].reshape(nc)
            scin[:, 3] = step["gsw"].reshape(nc)
            scin[:, 4] = dt
            scin[:, 5] = htop
            scin[:, 6] = hbot
            scin[:, 7:15] = apr.T
            iin = np.empty((nc, 2), np.int32)
            iin[:, 0] = band
            iin[:, 1] = int(inp["f_qi"])
            lev, sca, isc, ens = run_columns(lvin, scin, iin)
            cc = np.nonzero(band >= 1)[0]
            # advanced indices on axes 1 and 3 move to the front: (K, 6, nz)
            tend[:, cc // nx, :, cc % nx] = lev[cc]
            htop, hbot = sca[:, 2].copy(), sca[:, 3].copy()
            raincv, pratec = sca[:, 0].copy(), sca[:, 1].copy()
            apr = sca[:, 4:12].T.copy()
            ktop = isc[:, 0].copy()
            xf = ens[:, :ENSDIM].T.copy()
            pr = ens[:, ENSDIM:].T.copy()
        rainc = (rainc + pratec * dt).astype(F4)
        rec = {"due": due}
        for f, name in enumerate(LEVEL_OUTPUTS):
            rec[name] = tend[f].copy()
        sfc = dict(raincv=raincv, pratec=pratec, rainc=rainc, htop=htop,
                   hbot=hbot, mass_flux=np.zeros(nc, F4))
        for n, name in enumerate(ACCUMULATORS[2:]):
            sfc[name] = apr[n]
        for name in SURFACE_OUTPUTS:
            rec[name] = np.asarray(sfc[name], F4).reshape(ny, nx).copy()
        rec["ktop_deep"] = ktop.reshape(ny, nx).copy()
        rec["xf_ens"] = xf.reshape(ENSDIM, ny, nx).copy()
        rec["pr_ens"] = pr.reshape(ENSDIM, ny, nx).copy()
        out.append(rec)
    return out


def band_flags(nx: int, ny: int, periodic_x: bool, periodic_y: bool):
    """GRELLDRV's write window per mass column, 0-based (ny, nx) int32.

    2 = inside both bands (the scheme writes rain and tendencies), 1 = inside
    the j band but outside the i band (ktop_deep written, cuten forced 0),
    0 = outside the j band (GRELLDRV writes nothing there).  The bands are
    ibeg = ids+4 .. iend = ide-5 unless periodic (module_cu_gd.F:181-194);
    with ide = nx+1 that is 0-based columns 4 .. nx-5.
    """
    i = np.arange(nx)
    j = np.arange(ny)
    iin = np.ones(nx, bool) if periodic_x else (i >= 4) & (i <= nx - 5)
    jin = np.ones(ny, bool) if periodic_y else (j >= 4) & (j <= ny - 5)
    flags = np.where(jin[:, None], np.where(iin[None, :], 2, 1), 0)
    return flags.astype(np.int32)


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("columns", "edge", "cadence"):
        fx = build_fixture(name)
        for fqi in (1, 0):
            write_input(fx, out / f"gd-{name}-fqi{fqi}.in", bool(fqi))
        print(name, fx.nx, fx.ny, fx.nz, fx.nsteps)
