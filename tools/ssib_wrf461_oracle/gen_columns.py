"""Generate the SSiB oracle column set: columns.bin (see layout.py).

The set is chosen to reach every branch the brief names: each of the 13
SSiB vegetation types by day and by night, bare soil, sea ice (full,
fractional, newly formed), snow accumulation into the four-layer pack and
melt back out of it, frozen soil, wet and dry canopies, crops at three
latitudes, the USGS and MODIS category maps, both SW radiation arms
(RRTMG's four components, ra_sw_physics=4; the cloud-cover split,
ra_sw_physics=1), a first model level lower than the canopy, and open
water that SSIB must leave untouched.

Everything is a deterministic function of the column recipe; numpy's
seeded generator only jitters values inside physically sane ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from layout import FORCING, STATE, STATIC

F32 = np.float32
NZ = 5


def _sun(lat_deg, julian, hour):
    dec = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (julian - 80.0) / 365.0)
    lat = np.deg2rad(lat_deg)
    h = np.deg2rad(15.0 * (hour - 12.0))
    c = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h)
    return np.maximum(c, 0.0)


def column(recipe, nstep, dt, rng):
    """Return (static, state0, forcing[nstep]) for one recipe dict."""
    r = dict(recipe)
    lat = r.get("lat", 38.0)
    j0 = r.get("julian0", 172.0)
    hour0 = r.get("hour0", 6.0)
    st = {n: 0.0 for n in STATE}
    st.update(smois1=r.get("smois", 0.28), smois2=r.get("smois", 0.28) + 0.01,
              smois3=r.get("smois", 0.28) + 0.02,
              tslb1=r.get("tc", 293.0), tslb2=r.get("tg", 292.0),
              tslb3=r.get("td", 289.0), snow=r.get("snow", 0.0),
              canwat=r.get("canwat", 0.0), tsk=r.get("tg", 292.0),
              t2=r.get("tc", 293.0), albedo=0.2, ust=0.3, znt=0.1,
              gz1oz0=6.0, isnow=1.0, snowden=0.0)
    static = {"xland": r.get("xland", 1.0), "ivgtyp": r.get("ivgtyp", 7),
              "xlat": lat, "mminlu": r.get("mminlu", 1),
              "ra_sw": r.get("ra_sw", 4), "dtbl": dt}
    frc = np.zeros((nstep, len(FORCING) + NZ), F32)
    fi = {n: i for i, n in enumerate(FORCING)}
    t_base = r.get("t1", 295.0)
    t_amp = r.get("t_amp", 4.0)
    for s in range(nstep):
        hours = hour0 + s * dt / 3600.0
        jul = j0 + hours / 24.0
        mu = float(_sun(lat, jul, hours % 24.0)) if r.get("sun", True) else 0.0
        cld = r.get("cloud", 0.2) + 0.1 * np.sin(s / 7.0)
        cld = float(min(max(cld, 0.0), 1.0))
        warm = r.get("warm_after", None)
        t1 = t_base + t_amp * np.sin(2 * np.pi * (hours - 9.0) / 24.0)
        if warm is not None and s >= warm:
            t1 += r.get("warm_by", 12.0) * min(1.0, (s - warm) / 20.0)
        p1 = r.get("p1", 99000.0)
        es = 611.2 * np.exp(17.67 * (t1 - 273.15) / (t1 - 29.65))
        qsat = 0.622 * es / (p1 - es)
        qv1 = r.get("rh", 0.6) * qsat
        sw = 1050.0 * mu * (1.0 - 0.65 * cld)
        dif = 0.15 + 0.6 * cld
        vals = {
            "rainncv": 0.0, "raincv": 0.0,
            "glw": 280.0 + 80.0 * cld + 1.5 * (t1 - 280.0),
            "coszen": mu if mu > 0 else -0.2,
            "swdown": sw,
            "swvisdir": 0.48 * sw * (1 - dif), "swvisdif": 0.48 * sw * dif,
            "swnirdir": 0.52 * sw * (1 - dif), "swnirdif": 0.52 * sw * dif,
            "dz8w1": r.get("dz", 58.0), "u1": r.get("u", 4.0) + 0.3 * np.sin(s / 5.0),
            "v1": r.get("v", -2.0), "qv1": qv1, "t1": t1, "p1": p1,
            "psfc": p1 + 600.0, "hfx_sea": 12.0, "qfx_sea": 2.0e-5,
            "lh_sea": 50.0, "tsk_sea": 271.4, "xice": r.get("xice", 0.0),
            "xice_save": r.get("xice_save", r.get("xice", 0.0)),
            "julian": jul - 1.0, "t2_sfclay": t1 + 0.6,
        }
        rain = r.get("rain", None)
        if rain is not None:
            lo, hi, amount, conv = rain
            if lo <= s < hi:
                vals["rainncv"] = amount * (1 - conv)
                vals["raincv"] = amount * conv
        for n, v in vals.items():
            frc[s, fi[n]] = v
        prof = np.array([0.0, cld * 0.6, cld, cld * 0.8, 1.2 * cld])
        frc[s, len(FORCING):] = prof
    return static, st, frc


def recipes():
    out = []
    # A. Every SSiB vegetation type, day start and night start, both SW arms.
    for t in range(1, 14):
        for night, hour0 in ((False, 7.0), (True, 19.0)):
            out.append(dict(name=f"veg{t}-{'night' if night else 'day'}",
                            ivgtyp=t, mminlu=1, hour0=hour0,
                            ra_sw=4 if (t + night) % 2 == 0 else 1,
                            smois=0.22 + 0.01 * t, lat=36.0 + t,
                            tc=294.0, tg=293.0, td=290.0))
    # B. Wet canopy: warm rain, convective and stratiform.
    for t, conv in ((1, 0.8), (4, 0.2), (7, 0.5), (2, 1.0)):
        out.append(dict(name=f"wetcanopy{t}", ivgtyp=t, mminlu=1, hour0=10.0,
                        rain=(5, 40, 1.5, conv), cloud=0.8, rh=0.9,
                        canwat=0.3))
    # C. Snow accumulation into the four-layer pack, then melt.
    for t, sw in ((1, 4), (7, 4), (11, 1), (12, 4), (6, 1), (9, 4)):
        out.append(dict(name=f"snowcycle{t}", ivgtyp=t, mminlu=1, hour0=8.0,
                        julian0=95.0, lat=40.0, t1=266.0, t_amp=2.0,
                        tc=268.0, tg=268.5, td=272.0, smois=0.25,
                        rain=(2, 40, 1.6, 0.3), cloud=0.9, rh=0.85,
                        warm_after=60, warm_by=24.0, ra_sw=sw))
    # D. Deep initial snow: cold (stays layered) and warm (melts out).
    for t, warm in ((1, False), (7, False), (11, False),
                    (1, True), (7, True), (10, True)):
        out.append(dict(name=f"deepsnow{t}-{'warm' if warm else 'cold'}",
                        ivgtyp=t, mminlu=1, hour0=9.0, julian0=80.0,
                        lat=50.0, snow=140.0 if not warm else 45.0,
                        t1=262.0 if not warm else 279.0,
                        tc=263.0 if not warm else 271.0,
                        tg=264.0 if not warm else 271.5, td=270.0,
                        warm_after=None if not warm else 10,
                        warm_by=10.0, cloud=0.1))
    # E. Frozen soil, no snow.
    for t in (7, 11, 3):
        out.append(dict(name=f"frozensoil{t}", ivgtyp=t, mminlu=1, hour0=11.0,
                        julian0=20.0, lat=55.0, t1=258.0, t_amp=3.0,
                        tc=259.0, tg=260.0, td=268.0, smois=0.3, rh=0.7))
    # F. Sea ice.
    for name, xice, xsave, hour0, snow in (
            ("ice-full-day", 1.0, 1.0, 10.0, 0.0),
            ("ice-full-night", 1.0, 1.0, 22.0, 0.0),
            ("ice-full-snow", 1.0, 1.0, 12.0, 20.0),
            ("ice-half", 0.5, 0.5, 13.0, 0.0),
            ("ice-frac60", 0.6, 0.6, 2.0, 5.0),
            ("ice-new", 0.9, 0.0, 12.0, 0.0),
            ("ice-new-small", 0.3, 0.01, 15.0, 0.0),
            ("ice-sliver", 0.03, 0.03, 12.0, 0.0),
            ("ice-snowfall", 1.0, 1.0, 9.0, 0.0)):
        out.append(dict(name=name, xland=1.0, xice=xice, xice_save=xsave,
                        hour0=hour0, julian0=100.0, lat=72.0, t1=258.0,
                        t_amp=2.0, tc=255.0, tg=256.0, td=265.0, snow=snow,
                        rh=0.8, ra_sw=4 if xice != 0.5 else 1,
                        rain=((3, 30, 0.8, 0.0) if name == "ice-snowfall"
                              else None)))
    # G. USGS categories (16 = water is never a land column).
    for cat in (2, 5, 7, 10, 11, 14, 19, 24):
        out.append(dict(name=f"usgs{cat}", ivgtyp=cat, mminlu=2, hour0=9.0,
                        lat=40.0, smois=0.27))
    # H. MODIS categories, and the modified-IGBP name.
    for cat, mm in ((1, 3), (5, 3), (10, 3), (12, 3), (15, 3), (16, 3),
                    (8, 4)):
        out.append(dict(name=f"modis{cat}-{mm}", ivgtyp=cat, mminlu=mm,
                        hour0=9.0, lat=42.0, smois=0.27,
                        tc=275.0 if cat == 15 else 294.0,
                        tg=274.0 if cat == 15 else 293.0))
    # I. Crops (type 12 runs CROPS) at three latitudes and phenology dates.
    for lat, j0 in ((40.0, 150.0), (-35.0, 300.0), (60.0, 200.0)):
        out.append(dict(name=f"crop-lat{int(lat)}", ivgtyp=12, mminlu=1,
                        hour0=8.0, lat=lat, julian0=j0, smois=0.3))
    # J. First model level below the canopy top (RASIT5 moves D).
    for t in (1, 2):
        out.append(dict(name=f"lowlevel{t}", ivgtyp=t, mminlu=1, hour0=12.0,
                        dz=16.0))
    # K. Dry soil, hot sun.
    for t in (8, 11):
        out.append(dict(name=f"drysoil{t}", ivgtyp=t, mminlu=1, hour0=9.0,
                        smois=0.04, t1=306.0, rh=0.15, tc=308.0, tg=310.0,
                        cloud=0.0, lat=30.0))
    # L. Open water: SSIB never runs, nothing may change but the blend.
    for k in range(2):
        out.append(dict(name=f"water{k}", xland=2.0, ivgtyp=16, mminlu=2,
                        hour0=12.0))
    return out


def build(path: Path, nstep: int, dt: float, seed: int = 20261008):
    rng = np.random.default_rng(seed)
    rs = recipes()
    statics, states, forcings = [], [], []
    for r in rs:
        s, st, f = column(r, nstep, dt, rng)
        statics.append([s[n] for n in STATIC])
        states.append([st[n] for n in STATE])
        forcings.append(f)
    ncol = len(rs)
    with open(path, "wb") as fh:
        np.array([ncol, nstep, NZ], "<i4").tofile(fh)
        np.asarray(statics, "<f4").tofile(fh)
        np.asarray(states, "<f4").tofile(fh)
        frc = np.stack(forcings, axis=1).astype("<f4")   # (nstep, ncol, nf)
        frc.tofile(fh)
    meta = {"ncol": ncol, "nstep": nstep, "nz": NZ, "dt": dt,
            "names": [r["name"] for r in rs]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def read_columns(path: Path):
    raw = path.read_bytes()
    ncol, nstep, nz = np.frombuffer(raw[:12], "<i4")
    off = 12
    def take(n):
        nonlocal off
        a = np.frombuffer(raw, "<f4", n, off)
        off += 4 * n
        return a
    static = take(ncol * len(STATIC)).reshape(ncol, len(STATIC))
    state = take(ncol * len(STATE)).reshape(ncol, len(STATE))
    nf = len(FORCING) + nz
    forcing = take(nstep * ncol * nf).reshape(nstep, ncol, nf)
    return int(ncol), int(nstep), int(nz), static, state, forcing


def read_out(path: Path, ncol: int, nstep: int):
    return np.fromfile(path, "<f4").reshape(nstep, ncol, len(STATE))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--nstep", type=int, default=288)
    ap.add_argument("--dt", type=float, default=300.0)
    a = ap.parse_args()
    m = build(a.out, a.nstep, a.dt)
    print(f"{m['ncol']} columns x {m['nstep']} steps -> {a.out}")
