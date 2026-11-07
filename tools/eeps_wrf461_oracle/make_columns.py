"""Column inputs for the EEPS (bl_pbl_physics=16) WRF v4.6.1 oracle.

Writes one ``inputs.bin`` that ``run_bl_eeps`` reads: a header, then every
3-D field as (nz, ncol) float32 in WRF's bottom-up order (Fortran
``field(i, k, 1)`` with i fastest), then the 2-D surface fields, then the
starting PEK/PEP/PEK_ADV/PEP_ADV (read only when the run starts after its
first step, the restart-shaped arm).

Eight regimes, nine columns each, deterministic (no RNG state crosses a
regime).  The regimes are the brief's list: convective daytime land, stable
night, neutral windy ocean, cloud-topped (stratocumulus under a strong
inversion with cloud-top radiative cooling, the top-down source's arm),
very shallow PBL, PBL near the model top (a low-topped column with the
inversion inside the top levels the PBLH search reaches), and mixed-phase
cloud with ice below the freezing level (the melting arm).  The eighth is
a deep moist tropical column whose saturated layers take the moist
Brunt-Vaisala arm.  Every column starts from EEPSINIT on the first step of
a cold run.
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

F = np.float32
RD, CP, G = 287.0, 1004.5, 9.81
EP1 = 461.6 / 287.0 - 1.0

FIELDS_3D = ("u", "v", "t", "qv", "qc", "qi", "qr", "qs", "qg", "p", "pi",
             "rho", "rthraten", "dz8w")
FIELDS_2D = ("psfc", "qsfc", "tsk", "ust", "rmol", "wspd", "xland", "hfx",
             "qfx")
REGIMES = ("convective_land", "stable_night", "neutral_windy_ocean",
           "cloud_topped", "very_shallow", "near_model_top", "mixed_phase",
           "moist_tropical")
PER_REGIME = 9


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _column(regime: str, m: int, nz: int):
    """One column, float64 internally, as WRF bottom-up arrays."""
    f = m / (PER_REGIME - 1)                      # 0..1 within the regime
    top = 4200.0 if regime == "near_model_top" else 18000.0
    # Stretched layer thicknesses whose sum is the model top.
    eta = np.linspace(0.0, 1.0, nz)
    w = 1.0 + 9.0 * eta ** 1.6
    if regime == "very_shallow":
        w = 1.0 + 14.0 * eta ** 1.3
    dz = top * w / w.sum()
    zi = np.concatenate(([0.0], np.cumsum(dz)))
    z = 0.5 * (zi[1:] + zi[:-1])
    psfc = 101000.0 - 1500.0 * f
    xland = 1.0
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    qr = np.zeros(nz)
    qs = np.zeros(nz)
    qg = np.zeros(nz)
    rthraten = -2.0e-5 * np.ones(nz)              # clear-sky cooling, K/s
    rh = 0.6 - 0.4 * np.clip(z / 12000.0, 0.0, 1.0)
    ugeo = 6.0 + 6.0 * f
    if regime == "convective_land":
        zml = 900.0 + 1400.0 * f
        th = np.where(z < zml, 300.0 + 0.0005 * z, 300.0 + 0.0005 * zml
                      + 1.5 + 0.005 * (z - zml))
        hfx, qfx, ust, wspd = 150.0 + 300.0 * f, 1.0e-4 + 2.0e-4 * f, \
            0.3 + 0.3 * f, 3.0 + 5.0 * f
        rmol = -(0.02 + 0.08 * f)
        tsk_off = 4.0 + 6.0 * f
        rh = np.where(z < zml, 0.45, 0.3)
    elif regime == "stable_night":
        th = 285.0 + 10.0 * (1.0 - np.exp(-z / (150.0 + 200.0 * f))) \
            + 0.004 * z
        hfx, qfx, ust, wspd = -(10.0 + 50.0 * f), 1.0e-6 * (1 - f), \
            0.08 + 0.17 * f, 1.0 + 4.0 * f
        rmol = 0.01 + 0.09 * (1.0 - f)
        tsk_off = -(2.0 + 3.0 * f)
        ugeo = 4.0 + 8.0 * f
    elif regime == "neutral_windy_ocean":
        zml = 800.0 + 600.0 * f
        th = np.where(z < zml, 290.0, 290.0 + 0.004 * (z - zml))
        hfx, qfx, ust, wspd = 20.0 * f, 5.0e-5 + 1.0e-4 * f, \
            0.5 + 0.4 * f, 15.0 + 10.0 * f
        rmol = (-1.0e-3 + 2.0e-3 * f)
        tsk_off = 0.5
        xland = 2.0
        ugeo = 16.0 + 10.0 * f
        rh = np.where(z < zml, 0.8, 0.5)
    elif regime == "cloud_topped":
        zinv = 800.0 + 400.0 * f
        th = np.where(z < zinv, 288.0, 288.0 + 8.0 + 4.0 * f
                      + 0.005 * (z - zinv))
        hfx, qfx, ust, wspd = 10.0 + 20.0 * f, 2.0e-5 + 4.0e-5 * f, \
            0.25 + 0.15 * f, 6.0 + 4.0 * f
        rmol = -(0.002 + 0.01 * f)
        tsk_off = 0.8
        xland = 2.0 if m % 3 else 1.0
        rh = np.where(z < zinv, 0.97, 0.25)
        cloud = (z > 0.55 * zinv) & (z < zinv)
        qc = np.where(cloud, (0.2e-3 + 0.3e-3 * f)
                      * (z - 0.55 * zinv) / (0.45 * zinv), 0.0)
        # Longwave cooling peaks in the cloud's top layer.
        ktop = int(np.nonzero(cloud)[0].max()) if cloud.any() else 0
        if m % 4 != 3:      # every fourth column keeps clear-sky cooling
            rthraten[ktop] = -(1.0e-3 + 1.5e-3 * f)
            if ktop >= 1:
                rthraten[ktop - 1] = -3.0e-4
    elif regime == "very_shallow":
        th = 270.0 + (12.0 + 6.0 * f) * (1.0 - np.exp(-z / 40.0)) \
            + 0.003 * z
        hfx, qfx, ust, wspd = -(5.0 + 25.0 * f), 0.0, 0.03 + 0.07 * f, \
            0.5 + 1.5 * f
        rmol = 0.2 + 0.6 * f
        tsk_off = -(4.0 + 4.0 * f)
        ugeo = 2.0 + 3.0 * f
        rh = 0.7 * np.ones(nz)
        if m % 2:
            # Thin ice fog under the inversion.
            qi = np.where(z < 60.0, 2.0e-6, 0.0)
    elif regime == "near_model_top":
        zml = top * (0.6 + 0.35 * f)
        th = np.where(z < zml, 302.0, 302.0 + 3.0 + 0.008 * (z - zml))
        hfx, qfx, ust, wspd = 250.0 + 200.0 * f, 1.5e-4, 0.4 + 0.3 * f, \
            4.0 + 4.0 * f
        rmol = -(0.03 + 0.05 * f)
        tsk_off = 6.0
        rh = np.where(z < zml, 0.35, 0.2)
    elif regime == "mixed_phase":
        th = 278.0 + 0.0045 * z
        hfx, qfx, ust, wspd = 40.0 * f - 10.0, 3.0e-5, 0.2 + 0.2 * f, \
            4.0 + 6.0 * f
        rmol = -0.01 if hfx > 0 else 0.02
        tsk_off = 0.5
        rh = np.where((z > 600.0) & (z < 5000.0), 0.98, 0.5)
        layer = (z > 600.0) & (z < 5000.0)
        qc = np.where(layer & (z < 3500.0), 1.0e-4 * (1.0 + f), 0.0)
        qi = np.where(layer & (z > 1500.0), 5.0e-5 * (1.0 + f), 0.0)
        # Ice below the freezing level: the melting arm (T >= t0).
        qi = np.where(z < 900.0, 2.0e-5 * (1.0 + f) * (m % 2), qi)
        qs = np.where(layer, 1.0e-4, 0.0)
        qg = np.where(layer & (z > 2000.0), 2.0e-5, 0.0)
        qr = np.where(z < 2500.0, 5.0e-5 * f, 0.0)
        if m % 3 == 0:
            qc = np.where(layer & (z > 1500.0) & (z < 2500.0), 1.0e-12, qc)
    else:  # moist_tropical
        th = 299.0 + 0.003 * z + 3.0e-7 * z * z
        hfx, qfx, ust, wspd = 20.0 + 60.0 * f, 1.0e-4 + 1.0e-4 * f, \
            0.3 + 0.2 * f, 5.0 + 5.0 * f
        rmol = -(0.005 + 0.02 * f)
        tsk_off = 1.0
        xland = 2.0
        rh = np.where(z < 6000.0, 1.0, 0.6)
        qc = np.where((z > 500.0) & (z < 4500.0), 3.0e-4 * (0.5 + f), 0.0)
        qr = np.where(z < 4500.0, 2.0e-4 * f, 0.0)
    # Hydrostatic pressure with the virtual temperature of the profile.
    p = np.empty(nz)
    pk = psfc
    qv = np.zeros(nz)
    pint = psfc
    for k in range(nz):
        pmid = pint * np.exp(-G * (z[k] - (zi[k])) / (RD * th[k]
                                                      * (pint / 1.0e5)
                                                      ** (RD / CP)))
        tk = th[k] * (pmid / 1.0e5) ** (RD / CP)
        qv[k] = rh[k] * _qsat(tk, pmid)
        tv = tk * (1.0 + 0.608 * qv[k])
        p[k] = pmid
        pint = pint * np.exp(-G * dz[k] / (RD * tv))
    del pk
    pi = (p / 1.0e5) ** (RD / CP)
    t = th * pi
    rho = p / (RD * t * (1.0 + 0.608 * qv))
    spd = ugeo * np.clip(np.log1p(z / 0.1) / np.log1p(1500.0 / 0.1), 0.05,
                         1.0)
    turn = np.deg2rad(30.0 * (1.0 - np.clip(z / 1500.0, 0.0, 1.0)))
    ang = np.deg2rad(40.0 * m)
    u = spd * np.cos(ang + turn)
    v = spd * np.sin(ang + turn)
    if regime == "neutral_windy_ocean":
        # A low-level jet: the shear term is large below the jet core.
        u = u + 4.0 * np.exp(-((z - 400.0) / 200.0) ** 2)
    tsk = t[0] + tsk_off
    qsfc = _qsat(tsk, psfc) * (1.0 if xland == 2.0 else 0.7)
    return dict(u=u, v=v, t=t, qv=qv, qc=qc, qi=qi, qr=qr, qs=qs, qg=qg,
                p=p, pi=pi, rho=rho, rthraten=rthraten, dz8w=dz,
                psfc=psfc, qsfc=qsfc, tsk=tsk, ust=ust, rmol=rmol,
                wspd=wspd, xland=xland, hfx=hfx, qfx=qfx)


def build(nz: int):
    cols = [_column(r, m, nz) for r in REGIMES for m in range(PER_REGIME)]
    ncol = len(cols)
    out3 = {n: np.stack([c[n] for c in cols], axis=1).astype(F)
            for n in FIELDS_3D}
    out2 = {n: np.array([c[n] for c in cols]).astype(F) for n in FIELDS_2D}
    regime = np.repeat(np.arange(len(REGIMES), dtype=np.int32), PER_REGIME)
    return ncol, out3, out2, regime


def restart_state(nz: int, ncol: int):
    """Non-trivial carried TKE/epsilon for the restart-shaped arm."""
    k = np.arange(nz, dtype=np.float64)[:, None]
    i = np.arange(ncol, dtype=np.float64)[None, :]
    pek = (0.05 + 1.5 * np.exp(-k / 6.0) * (1.0 + 0.5 * np.sin(i)))
    pep = (1.0e-5 + 4.0e-3 * np.exp(-k / 5.0) * (1.0 + 0.3 * np.cos(i)))
    return pek.astype(F), pep.astype(F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--nz", type=int, default=44)
    ap.add_argument("--nsteps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--first-step", type=int, default=1)
    ap.add_argument("--no-qs-qg", action="store_true",
                    help="call EEPS without the optional QS3D/QG3D")
    args = ap.parse_args()
    ncol, f3, f2, regime = build(args.nz)
    pek, pep = restart_state(args.nz, ncol)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<5i", ncol, args.nz, args.nsteps,
                             args.first_step, 0 if args.no_qs_qg else 1))
        fh.write(struct.pack("<f", args.dt))
        for name in FIELDS_3D:
            fh.write(np.ascontiguousarray(f3[name]).tobytes())
        for name in FIELDS_2D:
            fh.write(f2[name].tobytes())
        # PEK_PBL/PEP_PBL differ from the advected pair on purpose: with
        # bl_eeps_tkeadv=1 the column must read only PEK_ADV/PEP_ADV.
        for arr in (pek, pep, (pek * F(0.75)).astype(F),
                    (pep * F(1.25)).astype(F)):
            fh.write(np.ascontiguousarray(arr).tobytes())
        fh.write(regime.tobytes())
    print(f"wrote {args.out}: ncol={ncol} nz={args.nz} "
          f"nsteps={args.nsteps} dt={args.dt} first={args.first_step}")


if __name__ == "__main__":
    main()
