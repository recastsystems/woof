#!/usr/bin/env python3
"""Write the SBU-YLin oracle's column cases as float32 hex words.

Every case is a full 40-level column built from a deterministic, seeded
recipe, so the file is reproducible on any host with numpy.  The families
cover each branch of clphy1d_ylin: deep convection, stratiform melting,
cold snow, mixed-phase riming (Ri > 0), warm rain only, dry air,
supersaturation, and the edge set (exact zeros, signed zeros, values on the
1e-8 / 1e-10 / 1e-6 thresholds, small negatives the scheme clips, below
-40 C homogeneous freezing, heavy rain that forces many sedimentation
sub-steps, and long time steps).

The column is the WRF call's argument list after moist_physics_prep_em:
full theta, vapour and the five condensates in kg/kg, dry density, Exner
pii, full pressure, mass-level height above sea level, layer depth dz8w,
terrain height, the time step and the incoming RAINNC.

Layout (see run_sbu_ylin.F90): ``ncol nz`` then per column ``dt ht rainnc``
and nz lines of ``th qv ql qr qi qs rho pii p z dz8w``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NZ = 40
F32 = np.float32
RD = 287.04
CP = 1004.0
RCP = RD / CP
FIELDS = ("th", "qv", "ql", "qr", "qi", "qs", "rho", "pii", "p", "z", "dz8w")


def _hex(x) -> str:
    return "%08X" % int(np.asarray(x, dtype=F32).view(np.uint32))


def _qsat_liquid(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _qsat_ice(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    es = 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _column(rng: np.random.Generator, *, tsfc: float, ht: float,
            lapse: float = 6.5e-3, ztrop: float = 12000.0,
            psfc: float = 1.0e5) -> dict[str, np.ndarray]:
    """A hydrostatic column in float64, before hydrometeors are added."""
    dz = np.empty(NZ)
    dz[0] = 50.0 + 30.0 * rng.random()
    for k in range(1, NZ):
        dz[k] = min(dz[k - 1] * 1.12, 900.0)
    zw = ht + np.concatenate(([0.0], np.cumsum(dz)))
    zm = 0.5 * (zw[:-1] + zw[1:])
    height = zm - ht
    t = np.where(height < ztrop, tsfc - lapse * height,
                 tsfc - lapse * ztrop)
    p = np.empty(NZ)
    p_w = psfc * np.exp(-9.81 * (zw[0] - ht) / (RD * tsfc))
    for k in range(NZ):
        p[k] = p_w * np.exp(-9.81 * (zm[k] - zw[k]) / (RD * t[k]))
        p_w = p[k] * np.exp(-9.81 * (zw[k + 1] - zm[k]) / (RD * t[k]))
    pii = (p / 1.0e5) ** RCP
    return {"t": t, "p": p, "pii": pii, "z": zm, "dz8w": dz,
            "height": height}


def _finish(col: dict, rh: np.ndarray, hyd: dict[str, np.ndarray]) -> dict:
    t, p = col["t"], col["p"]
    qv = rh * _qsat_liquid(t, p)
    rho = p / (RD * t * (1.0 + 0.61 * qv))
    out = {
        "th": t / col["pii"], "qv": qv,
        "ql": hyd.get("ql", np.zeros(NZ)), "qr": hyd.get("qr", np.zeros(NZ)),
        "qi": hyd.get("qi", np.zeros(NZ)), "qs": hyd.get("qs", np.zeros(NZ)),
        "rho": rho, "pii": col["pii"], "p": p, "z": col["z"],
        "dz8w": col["dz8w"],
    }
    return {k: np.asarray(v, dtype=F32) for k, v in out.items()}


def _bump(height, lo, hi, peak, rng):
    """A smooth hydrometeor layer between lo and hi metres AGL."""
    centre = 0.5 * (lo + hi)
    width = max(0.5 * (hi - lo), 1.0)
    shape = np.clip(1.0 - ((height - centre) / width) ** 2, 0.0, None)
    return peak * shape * (0.7 + 0.6 * rng.random(height.shape))


def deep_convective(rng):
    col = _column(rng, tsfc=298.0 + 6.0 * rng.random(), ht=200.0 * rng.random())
    h = col["height"]
    rh = np.clip(0.95 - h / 30000.0, 0.3, 1.0) + 0.05 * rng.random(NZ)
    hyd = {"ql": _bump(h, 800, 8000, 2.5e-3 * rng.random() + 5e-4, rng),
           "qr": _bump(h, 0, 6500, 4e-3 * rng.random() + 1e-3, rng),
           "qi": _bump(h, 8000, 14000, 8e-4 * rng.random() + 1e-4, rng),
           "qs": _bump(h, 5000, 12000, 4e-3 * rng.random() + 5e-4, rng)}
    return col, rh, hyd


def stratiform(rng):
    col = _column(rng, tsfc=280.0 + 8.0 * rng.random(), ht=300.0 * rng.random())
    h = col["height"]
    rh = np.clip(0.92 + 0.08 * rng.random(NZ), 0.0, 1.02)
    zm = (col["t"][0] - 273.15) / 6.5e-3
    hyd = {"qs": _bump(h, zm - 200, 9000, 1e-3 * rng.random() + 2e-4, rng),
           "qr": _bump(h, 0, zm + 300, 8e-4 * rng.random() + 1e-4, rng),
           "ql": _bump(h, zm - 500, zm + 1500, 2e-4 * rng.random(), rng),
           "qi": _bump(h, 6000, 10000, 1e-4 * rng.random(), rng)}
    return col, rh, hyd


def cold_snow(rng):
    col = _column(rng, tsfc=255.0 + 12.0 * rng.random(), ht=1500.0 * rng.random())
    h = col["height"]
    qsw = _qsat_liquid(col["t"], col["p"])
    qsi = _qsat_ice(col["t"], col["p"])
    rh = (qsi / qsw) * (0.92 + 0.14 * rng.random(NZ))
    hyd = {"qs": _bump(h, 0, 6000, 1e-3 * rng.random() + 1e-4, rng),
           "qi": _bump(h, 1500, 8000, 2.5e-4 * rng.random() + 1e-5, rng)}
    return col, rh, hyd


def mixed_phase_riming(rng):
    col = _column(rng, tsfc=274.0 + 6.0 * rng.random(), ht=500.0 * rng.random())
    h = col["height"]
    rh = np.clip(0.97 + 0.04 * rng.random(NZ), 0.0, 1.01)
    zlo = (col["t"][0] - 268.0) / 6.5e-3
    zhi = (col["t"][0] - 258.0) / 6.5e-3
    hyd = {"ql": _bump(h, zlo, zhi, 1e-3 * rng.random() + 2e-4, rng),
           "qs": _bump(h, zlo - 800, zhi + 2000, 2e-3 * rng.random() + 5e-4, rng),
           "qi": _bump(h, zhi, zhi + 3000, 2e-4 * rng.random(), rng),
           "qr": _bump(h, 0, zlo, 3e-4 * rng.random(), rng)}
    return col, rh, hyd


def warm_rain(rng):
    col = _column(rng, tsfc=300.0 + 4.0 * rng.random(), ht=100.0 * rng.random(),
                  lapse=5.0e-3)
    h = col["height"]
    rh = np.clip(0.88 + 0.12 * rng.random(NZ), 0.0, 1.0)
    hyd = {"ql": _bump(h, 500, 4000, 1.5e-3 * rng.random() + 2e-4, rng),
           "qr": _bump(h, 0, 3500, 2e-3 * rng.random() + 1e-4, rng)}
    return col, rh, hyd


def dry(rng):
    col = _column(rng, tsfc=285.0 + 20.0 * rng.random(), ht=800.0 * rng.random())
    rh = 0.08 + 0.4 * rng.random(NZ)
    return col, rh, {}


def supersaturated(rng):
    col = _column(rng, tsfc=270.0 + 25.0 * rng.random(), ht=400.0 * rng.random())
    h = col["height"]
    rh = 0.9 + 0.35 * rng.random(NZ)
    hyd = {}
    if rng.random() < 0.5:
        hyd["ql"] = _bump(h, 1000, 3000, 1e-4 * rng.random(), rng)
    return col, rh, hyd


def melting_virga(rng):
    """Snow falling through the melting level into dry warm air.

    The one family where melting-snow evaporation (Psmltevp) has work to
    do: T > 0 C, snow present and relative humidity under 90 %.
    """
    col = _column(rng, tsfc=281.0 + 8.0 * rng.random(), ht=600.0 * rng.random())
    h = col["height"]
    zm = (col["t"][0] - 273.15) / 6.5e-3
    rh = np.where(h < zm + 500.0, 0.35 + 0.4 * rng.random(NZ),
                  0.95 + 0.05 * rng.random(NZ))
    hyd = {"qs": _bump(h, zm - 900, 7000, 1.5e-3 * rng.random() + 3e-4, rng),
           "qi": _bump(h, 4000, 8000, 1e-4 * rng.random(), rng)}
    return col, rh, hyd


def edge(rng, which: int):
    """The edge set: each ``which`` exercises one boundary of the scheme."""
    col = _column(rng, tsfc=260.0 + 35.0 * rng.random(), ht=300.0 * rng.random())
    h = col["height"]
    rh = np.clip(0.6 + 0.4 * rng.random(NZ), 0.0, 1.0)
    z = np.zeros(NZ)
    hyd = {"ql": z.copy(), "qr": z.copy(), "qi": z.copy(), "qs": z.copy()}
    if which == 0:          # every hydrometeor exactly zero, moist air
        pass
    elif which == 1:        # every hydrometeor exactly zero, very dry air
        rh = np.full(NZ, 1e-6)
    elif which == 2:        # values straddling the 1e-8 fall threshold
        for name in hyd:
            hyd[name] = np.where(np.arange(NZ) % 2 == 0, 1.0e-8, 1.00001e-8)
    elif which == 3:        # 1e-10 Bergeron and 1e-6 autoconversion edges
        hyd["ql"] = np.where(np.arange(NZ) % 3 == 0, 1.0e-10,
                             np.where(np.arange(NZ) % 3 == 1, 1.0e-6, 1.1e-6))
        hyd["qi"] = np.full(NZ, 2.0e-9)
    elif which == 4:        # small negatives the scheme clips at entry
        for name in hyd:
            hyd[name] = -1.0e-12 * (1 + rng.random(NZ))
    elif which == 5:        # heavy rain: many sedimentation sub-steps
        hyd["qr"] = _bump(h, 0, 8000, 1.2e-2, rng)
        hyd["qs"] = _bump(h, 3000, 9000, 8e-3, rng)
        hyd["qi"] = _bump(h, 6000, 12000, 3e-3, rng)
    elif which == 6:        # below -40 C: homogeneous freezing of cloud water
        col = _column(rng, tsfc=235.0 + 5.0 * rng.random(), ht=0.0)
        h = col["height"]
        rh = np.full(NZ, 0.9)
        hyd["ql"] = _bump(h, 0, 4000, 5e-4, rng)
        hyd["qi"] = _bump(h, 2000, 6000, 1e-4, rng)
    elif which == 7:        # warm level with cloud ice: melting (Pimlt)
        col = _column(rng, tsfc=290.0, ht=0.0)
        h = col["height"]
        rh = np.full(NZ, 1.0)
        hyd["qi"] = _bump(h, 0, 2000, 4e-4, rng)
        hyd["ql"] = _bump(h, 0, 2000, 4e-4, rng)
    elif which == 8:        # tiny subnormal and minimal-normal words
        tiny = np.float32(np.finfo(np.float32).tiny)
        hyd["ql"] = np.full(NZ, float(tiny))
        hyd["qs"] = np.full(NZ, 1.0e-40)
        hyd["qr"] = np.full(NZ, 1.0e-39)
    elif which == 9:        # qi above the aggregation threshold, cold
        col = _column(rng, tsfc=250.0, ht=0.0)
        h = col["height"]
        rh = np.full(NZ, 0.95)
        hyd["qi"] = _bump(h, 0, 9000, 3e-3, rng)
    return col, rh, hyd


FAMILIES = (
    ("deep_convective", deep_convective, 10),
    ("stratiform", stratiform, 9),
    ("cold_snow", cold_snow, 9),
    ("mixed_phase_riming", mixed_phase_riming, 9),
    ("warm_rain", warm_rain, 8),
    ("dry", dry, 6),
    ("supersaturated", supersaturated, 7),
    ("melting_virga", melting_virga, 6),
)
DTS = (20.0, 10.0, 30.0, 6.0, 60.0, 18.0, 45.0, 3.0)
LONG_DTS = (120.0, 300.0, 600.0)


def build_cases(seed: int = 20261008):
    rng = np.random.default_rng(seed)
    cases = []
    for family, recipe, count in FAMILIES:
        for n in range(count):
            col, rh, hyd = recipe(rng)
            fields = _finish(col, rh, hyd)
            ht = F32(col["z"][0] - 0.5 * col["dz8w"][0])
            dt = F32(DTS[n % len(DTS)])
            cases.append((f"{family}-{n}", dt, ht, F32(rng.random() * 30.0),
                          fields))
    for which in range(10):
        for variant in range(2):
            col, rh, hyd = edge(rng, which)
            fields = _finish(col, rh, hyd)
            if which == 4 and variant == 1:
                for name in ("ql", "qr", "qi", "qs"):
                    fields[name][::2] = np.float32(-0.0)
            ht = F32(col["z"][0] - 0.5 * col["dz8w"][0])
            dt = F32(LONG_DTS[(which + variant) % 3] if variant else
                     DTS[which % len(DTS)])
            cases.append((f"edge{which}-{variant}", dt, ht, F32(0.0), fields))
    # Long-dt copies of physically active families.
    for n, (family, recipe, _) in enumerate(FAMILIES[:5]):
        col, rh, hyd = recipe(rng)
        fields = _finish(col, rh, hyd)
        ht = F32(col["z"][0] - 0.5 * col["dz8w"][0])
        cases.append((f"longdt-{family}", F32(LONG_DTS[n % 3]), ht,
                      F32(5.0), fields))
    return cases


def write_cases(path: Path, cases) -> None:
    lines = [f"{len(cases)} {NZ}"]
    for _, dt, ht, rainnc, fields in cases:
        lines.append(" ".join(_hex(v) for v in (dt, ht, rainnc)))
        for k in range(NZ):
            lines.append(" ".join(_hex(fields[name][k]) for name in FIELDS))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--names", type=Path,
                        help="also write one case name per line")
    args = parser.parse_args()
    cases = build_cases()
    write_cases(args.output, cases)
    if args.names:
        args.names.write_text("\n".join(c[0] for c in cases) + "\n",
                              encoding="ascii")
    print(f"{len(cases)} columns x {NZ} levels -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
