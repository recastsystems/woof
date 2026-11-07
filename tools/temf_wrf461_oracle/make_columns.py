"""Write the TEMF oracle's column cases (tools/temf_wrf461_oracle).

    python3 make_columns.py <OUT_DIR>

Three vertical grids (35, 45 and 61 mass levels, different first-layer
depths and stretching) each carry the same twelve regime families, six
columns per family (72 columns per grid, 216 in all), run for 30 steps
at a grid-specific dt:

    0 convective land, dry          6 sea, stable
    1 convective land, moist        7 very shallow mixed layer
    2 cloud-topped (stratocumulus)  8 deep convective (h above 4 km)
    3 stable night                  9 calm: zero and uniform winds
    4 neutral, windy               10 high terrain
    5 sea, convective              11 edge cases: MAVAIL = 0, a dry
                                      column, southern hemisphere, snow

The columns are built in float64 (hydrostatic interfaces, a mixed layer
under an inversion, a log wind profile with veer), then rounded once to
float32; WRF's own phy_prep arithmetic forms the mass heights and mass
pressures in float32.  Both the oracle and WOOF read the same words, so
physical consistency matters only for realism.  The skin temperature is
given per step (a heating ramp by day, a cooling ramp by night).

The layout is the one run_temf.F90 documents.  ``cases-<grid>.json`` names
every column's family and parameters.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
MAGIC = 1413827910
G, R_D, R_V = 9.81, 287.0, 461.6
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
EP2 = R_D / R_V
OMEGA = 7.2921e-5

GRIDS = {
    # name: (nz, first interface depth m, stretch, max depth m, dt s)
    "g35": (35, 56.0, 1.13, 1100.0, 120.0),
    "g45": (45, 38.0, 1.085, 650.0, 60.0),
    "g61": (61, 22.0, 1.065, 420.0, 20.0),
}
NSTEPS = 30
FAMILIES = ("convective-land-dry", "convective-land-moist", "cloud-topped",
            "stable-night", "neutral-windy", "sea-convective", "sea-stable",
            "very-shallow", "deep-convective", "calm", "high-terrain",
            "edge-cases")
PER_FAMILY = 6


def interfaces(nz, dz0, stretch, dzmax):
    dz = np.empty(nz)
    d = dz0
    for k in range(nz):
        dz[k] = min(d, dzmax)
        d *= stretch
    return np.concatenate(([0.0], np.cumsum(dz)))


def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return EP2 * es / (p - es)


def column(rng, fam, var, zw_agl, dt):
    """One column's float64 description."""
    nz = len(zw_agl) - 1
    zm_agl = 0.5 * (zw_agl[:-1] + zw_agl[1:])
    c = dict(ht=0.0, xland=1.0, mavail=0.3, znt=0.1, lat=40.0,
             theta_s=295.0, zi=1000.0, jump=3.0, gamma=4.5e-3,
             ml_lapse=0.0, q_ml=8e-3, rh_top=0.6, ug=8.0, vg=1.0,
             veer=0.3, dtsk=6.0, tsk_rate=0.0, hfx0=150.0, qfx0=5e-5,
             hpbl0=0.0, cloud=0.0, calm=False, dry=False)
    u01 = lambda a, b: float(rng.uniform(a, b))
    if fam == 0:
        c.update(zi=u01(1000, 2600), dtsk=u01(6, 12), mavail=u01(0.15, 0.35),
                 q_ml=u01(3e-3, 7e-3), ml_lapse=-u01(0, 1.5e-3),
                 tsk_rate=u01(1e-3, 3e-3), hfx0=u01(150, 400),
                 znt=u01(0.05, 0.5), ug=u01(3, 10))
    elif fam == 1:
        c.update(zi=u01(1200, 2000), dtsk=u01(4, 9), mavail=u01(0.4, 0.8),
                 q_ml=u01(12e-3, 17e-3), rh_top=u01(0.93, 1.0),
                 ml_lapse=-u01(0, 1e-3), tsk_rate=u01(1e-3, 2.5e-3),
                 hfx0=u01(120, 300), theta_s=u01(298, 303),
                 znt=u01(0.05, 0.8), ug=u01(2, 9))
    elif fam == 2:
        c.update(zi=u01(500, 1200), dtsk=u01(0.5, 2.5),
                 xland=2.0 if var % 2 == 0 else 1.0,
                 mavail=1.0 if var % 2 == 0 else 0.7,
                 znt=2e-4 if var % 2 == 0 else 0.2, q_ml=u01(8e-3, 11e-3),
                 rh_top=1.0, cloud=u01(2e-4, 6e-4), jump=u01(6, 10),
                 theta_s=u01(286, 292), hfx0=u01(5, 40), ug=u01(5, 10))
    elif fam == 3:
        c.update(zi=u01(60, 250), dtsk=-u01(2, 8), mavail=u01(0.2, 0.6),
                 jump=u01(1, 4), gamma=u01(5e-3, 12e-3),
                 tsk_rate=-u01(2e-4, 1e-3), hfx0=-u01(5, 60),
                 qfx0=u01(0, 1e-5), ug=u01(1.5, 7), hpbl0=u01(100, 1500),
                 znt=u01(0.02, 0.5), ml_lapse=u01(2e-3, 2e-2))
    elif fam == 4:
        c.update(zi=u01(600, 1500), dtsk=u01(-0.3, 0.6), ug=u01(12, 25),
                 vg=u01(-5, 5), mavail=u01(0.2, 1.0), hfx0=u01(-10, 20),
                 znt=u01(0.01, 1.0), hpbl0=u01(300, 1200))
    elif fam == 5:
        c.update(xland=2.0, mavail=1.0, znt=u01(1e-4, 4e-4),
                 zi=u01(400, 1100), dtsk=u01(1, 3.5), q_ml=u01(10e-3, 16e-3),
                 rh_top=u01(0.8, 0.98), hfx0=u01(10, 60),
                 theta_s=u01(293, 301), ug=u01(4, 14))
    elif fam == 6:
        c.update(xland=2.0, mavail=1.0, znt=u01(1e-4, 3e-4),
                 zi=u01(100, 400), dtsk=-u01(1, 4), q_ml=u01(6e-3, 10e-3),
                 hfx0=-u01(2, 25), ug=u01(5, 15), hpbl0=u01(0, 600),
                 ml_lapse=u01(1e-3, 6e-3))
    elif fam == 7:
        c.update(zi=u01(70, 220), dtsk=u01(1.5, 5), jump=u01(6, 12),
                 gamma=u01(6e-3, 10e-3), hfx0=u01(30, 120),
                 mavail=u01(0.2, 0.5), ug=u01(2, 7), znt=u01(0.02, 0.3))
    elif fam == 8:
        c.update(ht=u01(800, 1600), zi=u01(3600, 5200), dtsk=u01(10, 18),
                 mavail=u01(0.05, 0.2), q_ml=u01(2e-3, 6e-3),
                 ml_lapse=-u01(0, 5e-4), jump=u01(1, 3),
                 tsk_rate=u01(1e-3, 3e-3), hfx0=u01(350, 600),
                 theta_s=u01(305, 315), ug=u01(2, 10), znt=u01(0.05, 0.3))
    elif fam == 9:
        c.update(calm=True, zi=u01(300, 1200), dtsk=(u01(2, 6) if var < 3
                                                    else -u01(1, 4)),
                 hfx0=(u01(20, 100) if var < 3 else -u01(2, 20)),
                 ug=0.0, vg=0.0)
    elif fam == 10:
        c.update(ht=u01(2000, 3600), zi=u01(800, 2200), dtsk=u01(3, 10),
                 theta_s=u01(300, 318), q_ml=u01(2e-3, 6e-3),
                 mavail=u01(0.1, 0.4), hfx0=u01(80, 300),
                 tsk_rate=u01(5e-4, 2e-3), znt=u01(0.1, 1.0), ug=u01(4, 15))
    else:
        if var == 0:      # MAVAIL = 0: the CHS 0/0 (declared fix F3)
            c.update(mavail=0.0, dtsk=u01(6, 10), q_ml=2e-3, hfx0=200.0)
        elif var == 1:    # a dry column
            c.update(dry=True, mavail=0.0, dtsk=u01(3, 8), hfx0=100.0)
        elif var == 2:    # southern hemisphere, convective
            c.update(lat=-35.0, dtsk=u01(4, 8), hfx0=150.0)
        elif var == 3:    # snow-covered, cold, stable
            c.update(theta_s=262.0, dtsk=-u01(3, 6), znt=1e-3,
                     mavail=0.95, q_ml=1.2e-3, hfx0=-20.0, hpbl0=200.0)
        elif var == 4:    # tropics, near-zero Coriolis, convective
            c.update(lat=2.0, theta_s=300.0, q_ml=17e-3, rh_top=0.97,
                     dtsk=u01(2, 4), hfx0=60.0, mavail=0.9)
        else:             # strong wind over rough terrain at night
            c.update(dtsk=-u01(1, 3), ug=u01(15, 22), znt=1.5,
                     hfx0=-40.0, hpbl0=800.0)
    # Profiles.
    zi = c["zi"]
    theta = np.where(
        zm_agl <= zi, c["theta_s"] + c["ml_lapse"] * zm_agl,
        c["theta_s"] + c["ml_lapse"] * zi + c["jump"]
        + c["gamma"] * (zm_agl - zi))
    theta = theta + rng.normal(0.0, 0.03, nz) * (zm_agl < zi)
    psfc = 101325.0 * np.exp(-c["ht"] / 8000.0) + u01(-800, 800)
    # Hydrostatic interfaces with a first-guess moisture profile.
    qv = np.where(zm_agl <= zi, c["q_ml"],
                  c["q_ml"] * 0.4 * np.exp(-(zm_agl - zi) / 2500.0))
    if c["dry"]:
        qv[:] = 0.0
    pw = np.empty(nz + 1)
    pw[0] = psfc
    # A tropopause: temperature never falls below t_trop, so the column's
    # top keeps the 200-220 K of a real model top rather than the 170 K a
    # constant theta lapse reaches at 20 km.
    t_trop = u01(208.0, 220.0)
    for _ in range(3):
        for k in range(nz):
            pm = pw[k] * np.exp(-0.5 * (zw_agl[k + 1] - zw_agl[k]) / 8000.0)
            t = theta[k] * (pm / 1e5) ** RCP
            if t < t_trop:
                theta[k] = t_trop / (pm / 1e5) ** RCP
                t = t_trop
            tv = t * (1.0 + 0.608 * qv[k])
            pw[k + 1] = pw[k] * np.exp(-G * (zw_agl[k + 1] - zw_agl[k])
                                       / (R_D * tv))
    pmass = 0.5 * (pw[:-1] + pw[1:])
    tmass = theta * (pmass / 1e5) ** RCP
    qs = qsat(tmass, pmass)
    if not c["dry"]:
        # The mixed layer moistens toward its top to rh_top there.
        frac = np.clip(zm_agl / zi, 0.0, 1.0)
        rh_ml = 0.55 + (c["rh_top"] - 0.55) * frac ** 2
        qv = np.where(zm_agl <= zi, np.minimum(c["q_ml"], rh_ml * qs),
                      np.minimum(qv, 0.5 * qs))
    qc = np.zeros(nz)
    if c["cloud"] > 0.0:
        layer = (zm_agl > zi - 250.0) & (zm_agl <= zi)
        qc[layer] = c["cloud"] * (zm_agl[layer] - (zi - 250.0)) / 250.0
        qv[layer] = qs[layer]
    qi = np.zeros(nz)
    if fam == 10 and var % 2 == 1:
        qi[(zm_agl > zi) & (zm_agl < zi + 1500.0)] = 2e-5
    # Wind.
    if c["calm"]:
        u = np.full(nz, 3.0 if var % 2 else 0.0)
        v = np.full(nz, -1.0 if var % 2 else 0.0)
        u[:2] = 0.0
        v[:2] = 0.0
    else:
        z0 = max(c["znt"], 1e-4)
        shape = np.minimum(1.0, np.log(zm_agl / z0) / np.log(800.0 / z0))
        ang = c["veer"] * (1.0 - shape)
        u = shape * (c["ug"] * np.cos(ang) - c["vg"] * np.sin(ang))
        v = shape * (c["ug"] * np.sin(ang) + c["vg"] * np.cos(ang))
        u = u + rng.normal(0.0, 0.05, nz)
        v = v + rng.normal(0.0, 0.05, nz)
    tv = tmass * (1.0 + 0.608 * qv - qc)
    rho = pmass / (R_D * tv)
    t1 = tmass[0]
    tsk0 = t1 + c["dtsk"]
    tsk = tsk0 + c["tsk_rate"] * np.arange(NSTEPS) * dt
    fcor = 2.0 * OMEGA * np.sin(np.radians(c["lat"]))
    return dict(params=c, psfc=psfc, pw=pw, theta=theta, qv=qv, qc=qc,
                qi=qi, u=u, v=v, rho=rho, tsk=tsk, fcor=fcor)


def write_grid(out: Path, name: str, seed: int) -> dict:
    nz, dz0, stretch, dzmax, dt = GRIDS[name]
    zw_agl = interfaces(nz, dz0, stretch, dzmax)
    rng = np.random.default_rng(seed)
    cols, manifest = [], []
    for fam in range(len(FAMILIES)):
        for var in range(PER_FAMILY):
            col = column(rng, fam, var, zw_agl, dt)
            cols.append(col)
            manifest.append({"index": len(manifest), "family": FAMILIES[fam],
                             "variant": var,
                             "params": {k: (float(v) if not isinstance(v, bool)
                                            else v)
                                        for k, v in col["params"].items()}})
    ncol = len(cols)

    def f32(rows):
        return np.asarray(rows, dtype=np.float64).astype(F32)

    ht = f32([c["params"]["ht"] for c in cols])
    # WRF phy_prep in float32: z_at_w = (phb+ph)/g is given here as the
    # terrain plus the interface height; z = 0.5*(z_at_w(k)+z_at_w(k+1)).
    zw = (ht[None, :].astype(np.float64) + zw_agl[:, None]).astype(F32)
    z = np.empty((nz + 1, ncol), dtype=F32)
    z[:nz] = F32(0.5) * (zw[:-1] + zw[1:])
    z[nz] = zw[nz]
    p8w = f32([c["pw"] for c in cols]).T.copy()
    pmass = (F32(0.5) * (p8w[:-1] + p8w[1:])).astype(F32)
    pii = (pmass.astype(np.float64) / 1e5) ** RCP
    rec = {
        "ht": ht,
        "xland": f32([c["params"]["xland"] for c in cols]),
        "mavail": f32([c["params"]["mavail"] for c in cols]),
        "znt": f32([c["params"]["znt"] for c in cols]),
        "fcor": f32([c["fcor"] for c in cols]),
        "psfc": p8w[0].copy(),
        "hfx0": f32([c["params"]["hfx0"] for c in cols]),
        "qfx0": f32([c["params"]["qfx0"] for c in cols]),
        "hpbl0": f32([c["params"]["hpbl0"] for c in cols]),
    }
    tsk = f32([c["tsk"] for c in cols]).T.copy()          # (nsteps, ncol)
    prof = {n: f32([c[n] for c in cols]).T.copy()
            for n in ("u", "v", "theta", "qv", "qc", "qi", "rho")}
    path = out / f"cases-{name}.bin"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8i", MAGIC, 1, ncol, nz, NSTEPS, 0, 0, 0))
        fh.write(struct.pack("<f", dt))
        for key in ("ht", "xland", "mavail", "znt", "fcor", "psfc", "hfx0",
                    "qfx0", "hpbl0"):
            fh.write(rec[key].astype("<f4").tobytes())
        fh.write(tsk.astype("<f4").tobytes())
        for arr in (prof["u"], prof["v"], prof["theta"], prof["qv"],
                    prof["qc"], prof["qi"], pmass, pii.astype(F32),
                    prof["rho"]):
            fh.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        fh.write(z.astype("<f4").tobytes())
        fh.write(p8w.astype("<f4").tobytes())
    meta = {"grid": name, "nz": nz, "ncol": ncol, "nsteps": NSTEPS,
            "dt": dt, "first_interface_m": dz0, "stretch": stretch,
            "max_layer_m": dzmax, "seed": seed, "columns": manifest}
    (out / f"cases-{name}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for seed, name in enumerate(GRIDS, start=20261008):
        meta = write_grid(out, name, seed)
        print(f"{name}: {meta['ncol']} columns, {meta['nz']} levels, "
              f"{meta['nsteps']} steps, dt {meta['dt']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
