"""Write the CU_OSAS column-oracle input files (tools/osas_wrf461_oracle).

Also ``cases-<grid>-default.bin``: the same calls with XKT2 drawn by the
default arm's counter-based generator, for the default-arm grade.

Usage::

    python make_cases.py OUT_DIR

Writes ``cases-k44.bin`` (44 mass levels, p_top 50 hPa) and ``cases-k29.bin``
(29 levels, p_top 100 hPa) in the stream layout run_osas.F90 documents, plus
``cases-k44-single.bin``: the k44 columns again, one column per row (im = 1),
which the layout check compares against the 12 x 8 tile word for word
(OSASCNV's TOTFLG early returns and SHALCV's shared K1..K2 window are the only
places columns of a row meet).

The columns are analytic soundings, not a forecast sample, so the mix is
chosen rather than drawn: deep tropical, midlatitude severe (the Weisman and
Klemp 1982 profile), shallow-only (moist boundary layer under a dry
inversion), capped, dry, near-saturated, each over land (XLAND = 1) and sea
(XLAND = 2), at three surface pressures, with ascending, still and
descending cloud-base motion (the scheme's ACRTFCT and DTCONV read it), a
range of shear, and with or without resolved cloud water and ice.  XKT2 is
drawn per column from a fixed seed, with 0 and the largest double below 1
included.  Each file holds six calls: three chained at DT = 60 s, STEPCU = 1
(the second and third run on the state the previous call left), two chained
at DT = 30 s, STEPCU = 3, and one at DT = 20 s, STEPCU = 5.  Everything here
is float32 except XKT2, as WRF holds it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
KAPPA = RD / CP

ETA44 = np.array([
    1.000, 0.993, 0.983, 0.970, 0.954, 0.934, 0.909, 0.880, 0.845, 0.807,
    0.765, 0.719, 0.672, 0.622, 0.571, 0.520, 0.468, 0.420, 0.376, 0.335,
    0.298, 0.263, 0.231, 0.202, 0.175, 0.150, 0.127, 0.106, 0.088, 0.073,
    0.060, 0.050, 0.042, 0.035, 0.029, 0.024, 0.019, 0.015, 0.011, 0.008,
    0.005, 0.003, 0.0015, 0.0005, 0.000])
ETA29 = np.array([
    1.000, 0.990, 0.975, 0.955, 0.930, 0.900, 0.860, 0.815, 0.765, 0.710,
    0.650, 0.590, 0.530, 0.470, 0.410, 0.355, 0.305, 0.260, 0.220, 0.185,
    0.150, 0.120, 0.095, 0.072, 0.052, 0.035, 0.021, 0.010, 0.003, 0.000])

KINDS = ("tropical", "severe", "shallow", "capped", "dry", "saturated")


def esat(t):
    """Bolton (1980) liquid saturation vapour pressure, Pa."""
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def profile(kind, z, dtk, rh_scale):
    """Temperature (K) and relative humidity at heights z (m) for one kind.

    The severe kind is defined in potential temperature, so it returns
    theta and a flag; the integrator turns it into temperature.
    """
    if kind == "severe":
        ztr, thtr = 12000.0, 343.0
        th = np.where(z <= ztr, 300.0 + dtk + (thtr - 300.0) * (z / ztr) ** 1.25,
                      thtr + dtk + 0.012 * (z - ztr) * 1.0)
        rh = np.where(z <= ztr, 1.0 - 0.75 * (z / ztr) ** 1.25, 0.25)
        return th, np.clip(rh * rh_scale, 0.02, 1.0), True
    if kind == "tropical":
        t = 300.0 + dtk - 6.0e-3 * np.minimum(z, 16000.0)
        t = np.where(z > 16000.0, t + 1.5e-3 * (z - 16000.0), t)
        rh = np.interp(z, [0, 2000, 8000, 11000, 16000, 30000],
                       [0.85, 0.80, 0.65, 0.40, 0.20, 0.05])
    elif kind == "shallow":
        t = 298.0 + dtk - 9.5e-3 * np.minimum(z, 600.0) \
            - 6.0e-3 * np.clip(z - 600.0, 0, 900.0) \
            + 4.0 * np.clip((z - 1500.0) / 300.0, 0, 1) \
            - 6.5e-3 * np.clip(z - 1800.0, 0, 13000.0)
        t = np.where(z > 14800.0, t[0] * 0 + t, t)
        rh = np.interp(z, [0, 1400, 1700, 2500, 30000], [0.9, 0.95, 0.2, 0.15, 0.05])
    elif kind == "capped":
        t = 303.0 + dtk - 9.8e-3 * np.minimum(z, 1500.0) \
            + 6.0 * np.clip((z - 1500.0) / 500.0, 0, 1) \
            - 7.5e-3 * np.clip(z - 2000.0, 0, 12000.0)
        rh = np.interp(z, [0, 1400, 2000, 6000, 12000, 30000],
                       [0.75, 0.85, 0.45, 0.5, 0.3, 0.05])
    elif kind == "dry":
        t = 305.0 + dtk - 9.8e-3 * np.minimum(z, 3000.0) \
            - 7.0e-3 * np.clip(z - 3000.0, 0, 11000.0)
        rh = np.interp(z, [0, 3000, 30000], [0.18, 0.12, 0.05])
    elif kind == "saturated":
        t = 299.0 + dtk - 5.2e-3 * np.minimum(z, 15000.0) \
            + 1.0e-3 * np.clip(z - 15000.0, 0, 15000.0)
        rh = np.interp(z, [0, 9000, 13000, 30000], [0.97, 0.95, 0.6, 0.1])
    else:
        raise ValueError(kind)
    t = np.maximum(t, 195.0)
    return t, np.clip(rh * rh_scale, 0.02, 1.0), False


def column(kind, eta, ptop, psfc, dtk, rh_scale, w0, shear, cloud, rng):
    """One column on the eta grid: every WRF CU_OSAS input."""
    nz = eta.size - 1
    zf = np.arange(0.0, 32000.0, 5.0)
    a, rh, is_theta = profile(kind, zf, dtk, rh_scale)
    # Hydrostatic integration upward from psfc with virtual temperature.
    p = np.empty_like(zf)
    t = np.empty_like(zf)
    p[0] = psfc
    for n in range(zf.size):
        tn = a[n] * (p[n] / 1.0e5) ** KAPPA if is_theta else a[n]
        tn = max(tn, 195.0)
        if is_theta and zf[n] > 12000.0:
            tn = max(213.0 + (zf[n] - 12000.0) * 1.0e-3, 195.0)
        t[n] = tn
        e = rh[n] * esat(tn)
        qv = 0.622 * e / max(p[n] - e, 1.0)
        if kind == "severe":
            qv = min(qv, 0.014)
        tv = tn * (1.0 + 0.608 * qv)
        if n + 1 < zf.size:
            p[n + 1] = p[n] * np.exp(-G * (zf[n + 1] - zf[n]) / (RD * tv))
    pint = ptop + eta * (psfc - ptop)
    pmass = 0.5 * (pint[:-1] + pint[1:])
    lp = np.log(p[::-1])
    zint = np.interp(np.log(pint), lp, zf[::-1])
    zmass = np.interp(np.log(pmass), lp, zf[::-1])
    tm = np.interp(np.log(pmass), lp, t[::-1])
    rhm = np.interp(np.log(pmass), lp, rh[::-1])
    e = rhm * esat(tm)
    qv = 0.622 * e / (pmass - e)
    if kind == "severe":
        qv = np.minimum(qv, 0.014)
    qv = np.maximum(qv, 1.0e-7)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    if cloud:
        lay = (pmass < 85000.0) & (pmass > 60000.0) & (rhm > 0.8)
        qc[lay] = 1.0e-4 * rng.uniform(0.2, 1.5)
        ice = (pmass < 40000.0) & (pmass > 15000.0)
        qi[ice] = 2.0e-5 * rng.uniform(0.2, 2.0)
    dz8w = np.diff(zint)
    rho = pmass / (RD * tm * (1.0 + 0.608 * qv))
    pii = (pmass / 1.0e5) ** KAPPA
    hdepth = 10000.0
    w = np.where(zint < hdepth, w0 * np.sin(np.pi * zint / hdepth), 0.0)
    w[0] = 0.0
    u = 3.0 + shear * np.tanh(zmass / 3000.0)
    v = 2.0 - 0.3 * shear * np.sin(zmass / 4000.0)
    return dict(t=tm, qv=qv, qc=qc, qi=qi, u=u, v=v, p=pmass, pi=pii, rho=rho,
                dz8w=dz8w, w=w, p8w=pint)


def build_columns(eta, ptop, seed):
    rng = np.random.default_rng(seed)
    cols = []
    meta = []
    for kind in KINDS:
        for n in range(16):
            xland = 1.0 if n % 2 == 0 else 2.0
            psfc = (101300.0, 99000.0, 96000.0, 88000.0)[n % 4] if xland == 1.0 \
                else (101500.0, 100800.0, 101000.0, 100200.0)[n % 4]
            dtk = (-3.0, 0.0, 2.0, 4.0)[(n // 2) % 4]
            rh_scale = (0.85, 1.0, 1.08, 0.95)[(n // 4) % 4]
            w0 = (-0.3, 0.0, 0.6, 2.5)[(n + n // 4) % 4]
            shear = (5.0, 15.0, 30.0, 0.0)[(n // 3) % 4]
            cloud = n % 3 == 1
            c = column(kind, eta, ptop, psfc, dtk, rh_scale, w0, shear, cloud, rng)
            c["xland"] = xland
            cols.append(c)
            meta.append((kind, n, xland, psfc, dtk, rh_scale, w0, shear, cloud))
    return cols, meta


def pack(cols, im, jm, kx, key, levels):
    """Fortran (im, levels, jm) order as a C-ordered (jm, levels, im) array."""
    arr = np.empty((jm, levels, im), dtype=np.float32)
    for c, col in enumerate(cols):
        j, i = divmod(c, im)
        arr[j, :, i] = np.asarray(col[key], dtype=np.float32)
    return arr


def write_file(path, cols, im, jm, kx, calls, seed, default_arm=False):
    rng = np.random.default_rng(seed)
    ncol = im * jm
    assert len(cols) == ncol
    with open(path, "wb") as f:
        np.array([im, jm, kx, len(calls)], dtype="<i4").tofile(f)
        for n, (dt, stepcu, itimestep, chain, perturb) in enumerate(calls):
            np.array([dt], dtype="<f4").tofile(f)
            np.array([stepcu, itimestep, chain], dtype="<i4").tofile(f)
            use = cols
            if perturb:
                # Drawn in column space, so the single-column layout receives
                # exactly the same perturbation as the tiled one.
                prng = np.random.default_rng(seed + 1000 + n)
                dtp = prng.normal(0.0, 0.4, (ncol, kx))
                dqp = prng.uniform(0.95, 1.05, (ncol, kx))
                use = []
                for c, col in enumerate(cols):
                    col = dict(col)
                    col["t"] = (np.asarray(col["t"], np.float32) + dtp[c]).astype(np.float32)
                    col["qv"] = (np.asarray(col["qv"], np.float32) * dqp[c]).astype(np.float32)
                    use.append(col)
            fields = {key: pack(use, im, jm, kx, key, kx)
                      for key in ("t", "qv", "qc", "qi", "u", "v", "p", "pi", "rho", "dz8w")}
            for key in ("t", "qv", "qc", "qi", "u", "v", "p", "pi", "rho", "dz8w"):
                fields[key].astype("<f4").tofile(f)
            pack(cols, im, jm, kx, "w", kx + 1).astype("<f4").tofile(f)
            pack(cols, im, jm, kx, "p8w", kx + 1).astype("<f4").tofile(f)
            xland = np.empty((jm, im), dtype=np.float32)
            for c, col in enumerate(cols):
                j, i = divmod(c, im)
                xland[j, i] = col["xland"]
            xland.astype("<f4").tofile(f)
            draw = rng.random(ncol)
            if default_arm:
                # The default arm's own draw (column_cumulus's counter-based
                # generator keyed on the step and the column), so the oracle
                # grades the default arm's arithmetic on the draw it makes.
                from woof.core.column_cumulus_r8 import random_input_host
                draw = random_input_host(ncol, step=itimestep)
            elif n == 0:
                draw[0] = 0.0
                draw[1] = np.nextafter(1.0, 0.0)
            xkt2 = np.empty((jm, im), dtype=np.float64)
            for c in range(ncol):
                j, i = divmod(c, im)
                xkt2[j, i] = draw[c]
            xkt2.astype("<f8").tofile(f)


CALLS = (
    # dt, stepcu, itimestep, chain, perturb
    (60.0, 1, 1, 0, False),
    (60.0, 1, 2, 1, False),
    (60.0, 1, 3, 1, False),
    (30.0, 3, 3, 0, True),
    (30.0, 3, 6, 1, False),
    (20.0, 5, 5, 0, True),
)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, eta, ptop, seed, im, jm in (("k44", ETA44, 5000.0, 4401, 12, 8),
                                         ("k29", ETA29, 10000.0, 2901, 12, 8)):
        cols, meta = build_columns(eta, ptop, seed)
        kx = eta.size - 1
        write_file(out / f"cases-{name}.bin", cols, im, jm, kx, CALLS, seed)
        write_file(out / f"cases-{name}-default.bin", cols, im, jm, kx, CALLS,
                   seed, default_arm=True)
        if name == "k44":
            write_file(out / f"cases-{name}-single.bin", cols, 1, im * jm, kx,
                       CALLS, seed)
        with open(out / f"cases-{name}.meta.csv", "w", encoding="utf-8") as f:
            f.write("column,kind,variant,xland,psfc,dtk,rh_scale,w0,shear,cloud\n")
            for c, m in enumerate(meta):
                f.write(",".join([str(c)] + [str(x) for x in m]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
