"""Write the MRF column-oracle case file (``mrf-cases.bin``).

Every case is a full column on the same vertical count (NZ levels) with its
own spacing, terrain height, surface type, time step and ice flag, built to
reach a named branch of WRF v4.6.1 ``module_bl_mrf.F`` rather than to look
like any particular day.  The surface-layer inputs (BR, PSIM, PSIH, GZ1OZ0,
WSPD, ...) are made consistent with the column the way an MM5 similarity
layer would make them, then a handful of probes overwrite single words to
sit exactly on a comparison the scheme makes.

Output layout (little-endian stream, read by run_bl_mrf.F90):
  int32 magic, ncase, nz, nstep
  per case: float32 dt, int32 flag_qi, float32 xland, psfc, znt, ust, psim,
            psih, hfx, qfx, tsk, gz1oz0, wspd, br, zol, regime, hol, pbl
            then nz float32 each of u, v, t, qv, qc, qi, p, pi, dz8w, z
            (bottom-up, k = 1 is the lowest mass level)

The case table is written beside it as ``mrf-cases.json`` (name, family,
the branch it exists for), which the fixture packer copies into the
packaged receipts.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC = 0x4D524644          # 'MRFD'
NZ = 40
NSTEP = 4
F32 = np.float32

G = 9.81
RD = 287.0
CP = 7.0 * RD / 2.0
RCP = RD / CP
KARMAN = 0.4


def _profile(nz, dz0, stretch, dzmax):
    dz = np.minimum(dz0 * stretch ** np.arange(nz), dzmax)
    zw = np.concatenate(([0.0], np.cumsum(dz)))
    return zw


def build_case(name, family, why, *, dz0=50.0, stretch=1.08, dzmax=900.0,
               ht=200.0, psfc=97000.0, theta_s=300.0, h=1000.0,
               gamma=5.0, jump=1.5, superad=0.0, stable=0.0, hs=150.0,
               q_ml=8e-3, ug=8.0, vg=2.0, turn=0.25, z0=0.1, xland=1.0,
               hfx=150.0, qfx=8e-5, dtsk=None, dt=60.0, flag_qi=1,
               cloud=None, ice=None, uniform_wind=False, calm_low=False,
               br=None, wspd_floor=0.1, qc_words=None, rng=None):
    rng = rng or np.random.default_rng(zlib.crc32(name.encode()))
    zw = _profile(NZ, dz0, stretch, dzmax)
    zm = 0.5 * (zw[:-1] + zw[1:])
    # potential temperature
    theta = np.empty(NZ)
    for k, z in enumerate(zm):
        if stable > 0.0:
            theta[k] = theta_s + stable * (1.0 - math.exp(-z / hs)) \
                + gamma * z / 1000.0
        elif z < h:
            theta[k] = theta_s - superad * math.exp(-z / 40.0)
        else:
            theta[k] = theta_s + jump + gamma * (z - h) / 1000.0
    theta += rng.normal(0.0, 0.02, NZ)
    qv = np.where(zm < h, q_ml, q_ml * 0.6 * np.exp(-(zm - h) / 2500.0))
    qv = np.maximum(qv * (1.0 + rng.normal(0.0, 0.01, NZ)), 1e-7)
    qc = np.zeros(NZ)
    qi = np.zeros(NZ)
    if cloud is not None:
        lo, hi, amt = cloud
        sel = (zm >= lo) & (zm <= hi)
        qc[sel] = amt * (0.6 + 0.4 * rng.random(sel.sum()))
    if qc_words is not None:
        for k, word in qc_words.items():
            qc[k] = word
            qi[k] = 0.0
    if ice is not None:
        lo, hi, amt = ice
        sel = (zm >= lo) & (zm <= hi)
        qi[sel] = amt * (0.6 + 0.4 * rng.random(sel.sum()))
    # hydrostatic pressure on interfaces, then mid levels
    pw = np.empty(NZ + 1)
    pw[0] = psfc
    for k in range(NZ):
        tv = theta[k] * (pw[k] / 1e5) ** RCP * (1.0 + 0.608 * qv[k])
        pw[k + 1] = pw[k] * math.exp(-G * (zw[k + 1] - zw[k]) / (RD * tv))
    p = 0.5 * (pw[:-1] + pw[1:])
    pi = (p / 1e5) ** RCP
    t = theta * pi
    # wind
    if uniform_wind:
        u = np.full(NZ, ug)
        v = np.full(NZ, vg)
    else:
        hb = max(h, 200.0)
        fac = np.minimum(1.0, np.log((zm + z0) / z0) / math.log((hb + z0) / z0))
        ang = turn * (1.0 - fac)
        u = ug * fac * np.cos(ang) - vg * fac * np.sin(ang)
        v = ug * fac * np.sin(ang) + vg * fac * np.cos(ang)
        u += rng.normal(0.0, 0.05, NZ)
        v += rng.normal(0.0, 0.05, NZ)
    if calm_low:
        u[0] = 0.0
        v[0] = 0.0
    # float32 geometry exactly as phy_prep forms it from z_at_w
    zw32 = (ht + zw).astype(F32)
    z32 = (F32(0.5) * (zw32[:-1] + zw32[1:])).astype(F32)
    dz32 = (zw32[1:] - zw32[:-1]).astype(F32)
    # surface layer, MM5-similarity consistent
    za = 0.5 * (zw[1] - zw[0])
    wspd_raw = math.hypot(u[0], v[0])
    th1 = theta[0]
    if dtsk is None:
        dtsk = hfx / 60.0
    thsk = th1 + dtsk
    tsk = thsk * (psfc / 1e5) ** RCP
    vconv = 0.0
    if thsk > th1:
        vconv = (G / th1 * max(hfx, 0.0) / (1.2 * CP) * max(h, 100.0)) ** (1 / 3)
    wspd = max(math.sqrt(wspd_raw ** 2 + vconv ** 2), wspd_floor)
    qsk = q_ml * 1.1
    dthv = th1 * (1 + 0.608 * qv[0]) - thsk * (1 + 0.608 * qsk)
    if br is None:
        br = G / th1 * za * dthv / wspd ** 2
    gz1oz0 = math.log((za + z0) / z0)
    if br > 0:
        brc = min(br, 0.2)
        zol = brc * gz1oz0 / (1.1 - 5.0 * brc)
        psim = max(-5.0 * zol, -10.0)
        psih = psim
        regime = 1.0 if br >= 0.2 else 2.0
    elif br == 0:
        zol = 0.0
        psim = psih = 0.0
        regime = 3.0
    else:
        zol = max(br * gz1oz0, -9.9999)
        x = (1.0 - 16.0 * zol) ** 0.25
        psim = 2 * math.log((1 + x) / 2) + math.log((1 + x * x) / 2) \
            - 2 * math.atan(x) + math.pi / 2
        psih = 2 * math.log((1 + x * x) / 2)
        psim = min(psim, 0.9 * gz1oz0)
        psih = min(psih, 0.9 * gz1oz0)
        regime = 4.0
    ust = KARMAN * wspd / (gz1oz0 - psim)
    rec = dict(name=name, family=family, why=why, dt=dt, flag_qi=flag_qi,
               xland=xland, psfc=psfc, znt=z0, ust=ust, psim=psim, psih=psih,
               hfx=hfx, qfx=qfx, tsk=tsk, gz1oz0=gz1oz0, wspd=wspd, br=br,
               zol=zol, regime=regime, hol=0.0, pbl=0.0)
    cols = dict(u=u, v=v, t=t, qv=qv, qc=qc, qi=qi, p=p, pi=pi)
    return rec, cols, dz32, z32


def cases():
    out = []

    def add(*a, **k):
        out.append(build_case(*a, **k))

    # A. convective daytime land
    for i, (hfx, h, ug, q) in enumerate([
            (60, 400, 3, 6e-3), (120, 700, 5, 8e-3), (200, 1100, 8, 10e-3),
            (300, 1600, 4, 12e-3), (400, 2200, 10, 9e-3), (450, 2600, 2, 5e-3),
            (250, 900, 15, 14e-3), (180, 1300, 6, 3e-3), (350, 1800, 12, 11e-3),
            (90, 500, 1.5, 7e-3), (220, 1500, 7, 16e-3), (150, 1000, 20, 8e-3)]):
        add(f"conv-land-{i:02d}", "convective-land",
            "SFCFLG unstable arm, thermal-enhanced PBL, countergradient terms",
            h=h, ug=ug, vg=ug * 0.3, hfx=hfx, q_ml=q, qfx=hfx / 2.5e6 * 1.2,
            superad=1.5, gamma=4.0 + i % 4, jump=0.5 + 0.3 * (i % 5),
            dt=[60.0, 30.0, 90.0][i % 3], flag_qi=i % 2, z0=0.05 + 0.1 * (i % 4))
    # B. stable night land
    for i, (hfx, stab, ug) in enumerate([
            (-5, 2, 3), (-15, 4, 5), (-30, 6, 8), (-50, 8, 2), (-60, 10, 12),
            (-10, 3, 1), (-25, 5, 15), (-40, 7, 4), (-20, 12, 6), (-8, 1, 10)]):
        add(f"stable-land-{i:02d}", "stable-night",
            "BR > 0: no countergradient, Ri-based free-atmosphere K above "
            "a shallow PBL",
            stable=stab, hs=100 + 30 * i, gamma=3.0, hfx=hfx, qfx=1e-6,
            dtsk=-(2.0 + stab), ug=ug, vg=1.0, q_ml=5e-3, h=50.0,
            dt=[60.0, 120.0][i % 2], flag_qi=(i + 1) % 2, z0=0.1)
    # C. neutral windy
    for i, ug in enumerate([15, 20, 25, 18, 22, 30]):
        add(f"neutral-wind-{i:02d}", "neutral-windy",
            "near-zero BR, mechanically driven PBL",
            h=800 + 100 * i, superad=0.0, gamma=3.0, jump=0.2, hfx=5.0,
            qfx=2e-5, dtsk=0.05, ug=ug, vg=-0.2 * ug, q_ml=6e-3, dt=60.0,
            flag_qi=1, z0=0.3)
    # D. cloud-topped
    for i, (h, base, top, amt, ice) in enumerate([
            (800, 700, 1000, 3e-4, None), (1000, 900, 1300, 5e-4, None),
            (600, 500, 800, 2e-4, (700, 900, 5e-5)),
            (1200, 1100, 1500, 4e-4, (1300, 1700, 1e-4)),
            (900, 800, 1400, 6e-4, (1000, 1400, 2e-4)),
            (700, 600, 1600, 1e-4, None),
            (1500, 1400, 2200, 3e-4, (1800, 2600, 1.5e-4)),
            (500, 450, 700, 8e-4, None),
            (1100, 1000, 1300, 2.5e-4, (1100, 1300, 3e-4)),
            (900, 2500, 4000, 2e-4, (3000, 5000, 2e-4))]):
        add(f"cloud-top-{i:02d}", "cloud-topped",
            "IMVDIF moist Ri: qc+qi above 0.01e-3 on adjacent levels",
            h=h, cloud=(base, top, amt), ice=ice, hfx=[30, 80, -10, 120, 60,
                                                       200, 40, -20, 90, 150][i],
            q_ml=10e-3, gamma=6.0, jump=2.0, ug=6.0 + i, vg=2.0,
            dt=60.0, flag_qi=0 if i in (3, 6) else 1, z0=0.1,
            stable=4.0 if i in (2, 7) else 0.0, hs=200.0)
    # E. very shallow PBL
    for i, (dz0, stab, hfx) in enumerate([
            (20, 15, -40), (30, 20, -80), (15, 25, -30), (40, 0, 20),
            (25, 0, 5), (10, 30, -60)]):
        add(f"shallow-{i:02d}", "very-shallow",
            "PBL inside the first layers: BRDN >= BRCR, KPBL = KL",
            dz0=dz0, stretch=1.1, stable=stab, hs=30.0, gamma=8.0,
            h=60.0 if stab == 0 else 40.0, jump=4.0, hfx=hfx, qfx=1e-6,
            dtsk=-(3.0 + stab / 5) if hfx < 0 else 0.3, ug=2.0 + i, vg=0.5,
            q_ml=4e-3, dt=30.0, flag_qi=1, z0=0.05)
    # F. PBL near model top (short columns, deep thermals)
    for i, (dz0, hfx) in enumerate([(25, 400), (30, 350), (35, 450),
                                    (20, 300), (40, 500), (28, 420)]):
        add(f"near-top-{i:02d}", "near-model-top",
            "thermal reaches K = 1: KPBL <= 1 clears PBLFLG",
            dz0=dz0, stretch=1.0, dzmax=dz0, h=3000.0, gamma=0.5, jump=0.0,
            superad=3.0, hfx=hfx, qfx=1.5e-4, ug=2.0, vg=1.0, q_ml=12e-3,
            dt=60.0, flag_qi=i % 2, z0=0.2)
    # G. water points
    for i, (hfx, dtsk, ug, h, stab) in enumerate([
            (80, 3.0, 10, 600, 0), (150, 6.0, 15, 900, 0), (20, 0.5, 6, 500, 0),
            (-10, -2.0, 8, 100, 3), (-30, -4.0, 12, 80, 6), (5, 0.1, 20, 700, 0),
            (250, 10.0, 18, 1300, 0), (40, 1.0, 1.0, 400, 0),
            (0, 0.0, 9, 600, 0), (-5, -1.0, 3, 60, 2)]):
        add(f"water-{i:02d}", "water",
            "XLAND >= 1.5: UST without the 0.1 floor, HGAMQ = 0",
            xland=2.0, z0=2e-4, hfx=hfx, qfx=max(hfx, 10) / 2.5e6,
            dtsk=dtsk, ug=ug, vg=0.3 * ug, h=h, stable=stab, hs=80.0,
            superad=0.3 if hfx > 0 else 0.0, gamma=5.0, jump=1.0,
            q_ml=12e-3, ht=0.0, psfc=101300.0, dt=60.0, flag_qi=i % 2)
    # H. edge probes
    add("probe-calm-low", "probe", "UX(KL)=VX(KL)=0: WSPD1 = 1e-9",
        calm_low=True, hfx=50.0, h=500.0, ug=4.0, dt=60.0)
    add("probe-uniform-wind", "probe", "zero shear: SS = 1e-9 floor",
        uniform_wind=True, ug=7.0, vg=-3.0, hfx=100.0, h=800.0, dt=60.0)
    add("probe-land-ust-floor", "probe", "land UST below 0.1 floored",
        ug=0.3, vg=0.1, hfx=-2.0, dtsk=-0.2, stable=1.0, h=50.0, z0=0.01,
        wspd_floor=0.1, dt=60.0)
    add("probe-gamcrt-clamp", "probe",
        "HGAMT/VPERT reach GAMCRT; XKZMAX clamp", hfx=900.0, ug=0.5, vg=0.0,
        h=3000.0, superad=4.0, q_ml=18e-3, qfx=6e-4, dt=60.0, dz0=80.0)
    add("probe-prmax", "probe", "very stable Ri: PRNUM clipped to PRMAX",
        stable=25.0, hs=400.0, gamma=10.0, ug=3.0, hfx=-20.0, dtsk=-6.0,
        h=40.0, dt=120.0)
    add("probe-dt-short", "probe", "dt = 10 s", hfx=200.0, h=1200.0,
        dt=10.0)
    add("probe-dt-long", "probe", "dt = 180 s", hfx=200.0, h=1200.0,
        dt=180.0)
    add("probe-qc-threshold", "probe",
        "qc+qi exactly at the 0.01e-3 test and one ULP above",
        h=1000.0, hfx=60.0, dt=60.0,
        qc_words={24: F32(0.01e-3), 25: F32(0.01e-3),
                  29: np.nextafter(F32(0.01e-3), F32(1.0)),
                  30: np.nextafter(F32(0.01e-3), F32(1.0))})
    # BR word probes on the SFCFLG compare (BR .GT. 0.0)
    for nm, word in (("br-pos-zero", 0.0), ("br-neg-zero", -0.0),
                     ("br-pos-subnormal", 1.4e-45),
                     ("br-neg-subnormal", -1.4e-45),
                     ("br-min-normal", 1.1754944e-38)):
        add(f"probe-{nm}", "probe-br",
            "the BR .GT. 0.0 compare on a signed-zero/subnormal word",
            hfx=30.0, h=600.0, ug=10.0, br=word, dt=60.0)
    return out


def write(dirpath: Path) -> None:
    built = cases()
    ncase = len(built)
    with open(dirpath / "mrf-cases.bin", "wb") as fh:
        fh.write(struct.pack("<4i", MAGIC, ncase, NZ, NSTEP))
        for rec, cols, dz32, z32 in built:
            scal = [rec[k] for k in ("xland", "psfc", "znt", "ust", "psim",
                                     "psih", "hfx", "qfx", "tsk", "gz1oz0",
                                     "wspd", "br", "zol", "regime", "hol",
                                     "pbl")]
            fh.write(np.asarray([rec["dt"]], "<f4").tobytes())
            fh.write(struct.pack("<i", int(rec["flag_qi"])))
            fh.write(np.asarray(scal, "<f4").tobytes())
            for key in ("u", "v", "t", "qv", "qc", "qi", "p", "pi"):
                fh.write(np.asarray(cols[key], "<f4").tobytes())
            fh.write(np.asarray(dz32, "<f4").tobytes())
            fh.write(np.asarray(z32, "<f4").tobytes())
    table = [{k: rec[k] for k in ("name", "family", "why", "dt", "flag_qi",
                                   "xland")} for rec, *_ in built]
    (dirpath / "mrf-cases.json").write_text(
        json.dumps({"nz": NZ, "nstep": NSTEP, "cases": table}, indent=1)
        + "\n", encoding="utf-8")
    print(f"wrote {ncase} cases x {NSTEP} steps on {NZ} levels")


if __name__ == "__main__":
    write(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
