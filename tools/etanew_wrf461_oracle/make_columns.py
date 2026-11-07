#!/usr/bin/env python3
"""Write the Eta Ferrier (WRF mp_physics=95, ETAMPNEW) column-oracle inputs.

The inputs are built deterministically from closed-form soundings (no random
number generator, so the bits do not depend on the NumPy version) and are
written twice:

* ``columns_inputs.csv`` -- the canonical input set.  Every float is its
  IEEE-754 binary32 bit pattern in hex, so the GPU port is fed exactly the
  bits the Fortran saw.
* ``columns.in`` -- the same values in the fixed format ``run_etanew.F90``
  reads.

Usage: ``python3 make_columns.py OUT_DIR``.

ETAMP_NEW in ARW reads only the advected total condensate ``qt`` and the
three carried fractions (``f_ice_phy``, ``f_rain_phy``, ``f_rimef_phy``);
the moist ``qc``/``qr``/``qs`` it receives are overwritten from them on
exit (module_mp_etanew.F:268-297).  A column is therefore specified as
cloud water, rain and ice amounts, from which ``qt`` and the fractions are
formed the way ETAMP_NEW itself forms them, plus edge columns whose
fractions are deliberately inconsistent.

Regimes (8 columns each, 10 groups = 80 columns):

``convective``      deep moist convection, supersaturated updraft, cloud
                    water, rain, rimed ice aloft
``stratiform``      deep nimbostratus, ice above a melting layer, rain below
``snow``            whole column below freezing, ice supersaturated,
                    light supercooled water
``rimed_mixed``     mixed-phase rimed ice at -3 to -10 C with heavy cloud
                    water (rime splintering band, large rime factors)
``warm_rain``       tropical warm cloud and rain, no ice anywhere
``dry``             desert and stable dry columns with stray condensate
``near_saturated``  air within 0.1 % of water / ice saturation
``freezing_rain``   warm nose aloft over a sub-freezing surface layer
``cold_anvil``      tropopause 190-205 K, ice and liquid colder than -40 C
``edge``            thresholds and limits (see ``_edge``)

The time step cycles over 6, 20, 36, 72 and 150 s (the last reaches the
large-time-step exponential arms of deposition and rain evaporation).
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

NZ = 50
P_TOP = 5000.0
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
G = 9.81
T0C = 273.15
NSTEPS = 10
DTS = (6.0, 20.0, 36.0, 72.0, 150.0)
DX = (12000.0, 3000.0, 1000.0, 4000.0)

FIELDS = ("p", "pi", "th", "qv", "qt", "dz8w", "rho",
          "f_ice", "f_rain", "f_rimef")


def f32hex(x: float) -> str:
    return struct.pack(">f", np.float32(x)).hex().upper()


def esat_w(t: np.ndarray) -> np.ndarray:
    tc = t - T0C
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def esat_i(t: np.ndarray) -> np.ndarray:
    tc = t - T0C
    return 611.2 * np.exp(21.8745584 * tc / (t - 7.66))


def qsat(t: np.ndarray, p: np.ndarray, ice: bool = False) -> np.ndarray:
    e = esat_i(t) if ice else esat_w(t)
    e = np.minimum(e, 0.5 * p)
    return 0.622 * e / (p - e)


def eta_levels(psfc: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(NZ + 1) / NZ
    eta = 1.0 - (0.35 * n + 0.65 * n ** 1.6)
    eta[-1] = 0.0
    pint = P_TOP + eta * (psfc - P_TOP)
    pm = 0.5 * (pint[:-1] + pint[1:])
    return pint, pm


def bump(p: np.ndarray, pc: float, half: float) -> np.ndarray:
    x = (p - pc) / half
    return np.where(np.abs(x) < 1.0, np.cos(0.5 * math.pi * x) ** 2, 0.0)


class Column:
    def __init__(self, regime: str, idx: int, psfc: float):
        self.regime = regime
        self.idx = idx
        self.dt = DTS[idx % len(DTS)]
        self.dx = DX[(idx // 5) % 4]
        self.rainnc = 0.0 if idx % 3 else 1.25 * (idx % 7)
        self.pint, self.p = eta_levels(psfc)
        self.t = np.zeros(NZ)
        self.qv = np.zeros(NZ)
        self.qc = np.zeros(NZ)
        self.qr = np.zeros(NZ)
        self.qs = np.zeros(NZ)
        self.f_rimef = np.ones(NZ)
        self.raw = None          # explicit (qt, f_ice, f_rain) override

    def temperature(self, tsfc, lapse, ttrop, inversion=0.0, inv_depth=0.0,
                    nose=0.0, nose_p=0.0):
        z = 7400.0 * np.log(self.pint[0] / self.p)
        t = tsfc - lapse * z
        t = np.maximum(t, ttrop)
        if inversion:
            frac = np.clip(1.0 - z / max(inv_depth, 1.0), 0.0, 1.0)
            t = t - inversion * frac
            t = np.maximum(t, tsfc - inversion)
        if nose:
            t = t + nose * bump(self.p, nose_p, 9000.0)
        self.t = t

    def humidity(self, rh_low, rh_high, p_break, ice_above=False):
        rh = np.where(self.p > p_break, rh_low, rh_high)
        q = qsat(self.t, self.p, ice=False)
        if ice_above:
            qi_s = qsat(self.t, self.p, ice=True)
            q = np.where(self.t < T0C, qi_s, q)
        self.qv = np.maximum(rh * q, 1.0e-7)

    def arrays(self) -> dict[str, np.ndarray]:
        tv = self.t * (1.0 + 0.608 * self.qv)
        rho = self.p / (R_D * tv)
        dz = R_D * tv / G * np.log(self.pint[:-1] / self.pint[1:])
        pi = (self.p / 1.0e5) ** RCP
        th = self.t / pi
        if self.raw is not None:
            qt, f_ice, f_rain = self.raw
        else:
            qt = self.qc + self.qr + self.qs
            with np.errstate(divide="ignore", invalid="ignore"):
                f_ice = np.where(qt > 0, self.qs / np.where(qt > 0, qt, 1), 0)
                liq = self.qc + self.qr
                f_rain = np.where(liq > 0, self.qr / np.where(liq > 0, liq, 1),
                                  0)
        return {"p": self.p, "pi": pi, "th": th, "qv": self.qv, "qt": qt,
                "dz8w": dz, "rho": rho, "f_ice": f_ice, "f_rain": f_rain,
                "f_rimef": self.f_rimef}


def _convective(i, j):
    c = Column("convective", i, 100800.0 - 400.0 * j)
    c.temperature(299.5 + 0.9 * j, 0.0068 + 0.0002 * j, 196.0 + j)
    c.humidity(0.93 + 0.01 * j, 0.75, 45000.0)
    up = bump(c.p, 70000.0, 25000.0)
    c.qv = c.qv * (1.0 + (0.01 + 0.005 * j) * up)
    warm = c.t > T0C
    c.qc = (0.4e-3 + 0.35e-3 * j) * bump(c.p, 75000.0, 20000.0)
    c.qr = np.where(warm, (0.5e-3 + 0.9e-3 * j) * bump(c.p, 85000.0, 20000.0),
                    0.0)
    c.qs = np.where(~warm, (0.1e-3 + 0.35e-3 * j)
                    * bump(c.p, 40000.0, 18000.0), 0.0)
    c.f_rimef = np.where(c.qs > 0, 1.0 + 2.5 * j, 1.0)
    return c


def _stratiform(i, j):
    c = Column("stratiform", i, 100500.0 - 500.0 * j)
    c.temperature(283.0 + 1.2 * j, 0.0062, 205.0)
    c.humidity(0.97, 0.98, 40000.0)
    warm = c.t > T0C
    deck = bump(c.p, 60000.0, 35000.0)
    c.qs = np.where(~warm, (0.08e-3 + 0.06e-3 * j) * deck, 0.0)
    c.qr = np.where(warm, (0.05e-3 + 0.08e-3 * j) * bump(c.p, 90000.0, 15000.0),
                    0.0)
    c.qc = (0.02e-3 * j) * bump(c.p, 70000.0, 10000.0)
    # a melting layer straddling 0 C carries rimed ice into warm air
    melt = np.abs(c.t - T0C) < 3.0
    c.qs = np.where(melt, c.qs + 0.1e-3, c.qs)
    c.f_rimef = np.where(c.qs > 0, 1.0 + 0.4 * j, 1.0)
    return c


def _snow(i, j):
    c = Column("snow", i, 99000.0 - 1500.0 * j)
    c.temperature(268.5 - 1.6 * j, 0.0062, 205.0)
    c.humidity(0.97, 0.7, 50000.0, ice_above=True)
    c.qv = c.qv * (1.0 + (0.02 + 0.01 * j) * bump(c.p, 75000.0, 20000.0))
    c.qs = (0.05e-3 + 0.15e-3 * j) * bump(c.p, 78000.0, 25000.0)
    c.qc = (0.02e-3 * j) * bump(c.p, 85000.0, 8000.0)
    c.f_rimef = np.where(c.qs > 0, 1.0 + 0.6 * j, 1.0)
    return c


def _rimed_mixed(i, j):
    c = Column("rimed_mixed", i, 100000.0 - 800.0 * j)
    # -3 to -10 C across 700-500 hPa
    c.temperature(281.0 - 0.8 * j, 0.0066, 210.0)
    c.humidity(0.99, 0.99, 30000.0)
    c.qv = c.qv * (1.0 + 0.004 * j * bump(c.p, 62000.0, 15000.0))
    band = bump(c.p, 62000.0, 15000.0)
    c.qc = (0.5e-3 + 0.4e-3 * j) * band
    c.qs = (0.2e-3 + 0.25e-3 * j) * bump(c.p, 55000.0, 20000.0)
    c.qr = 0.3e-3 * bump(c.p, 90000.0, 8000.0)
    k = np.arange(NZ)
    # rime factors 1.1**n spanning the VEL_RF table, up to and past 1.1**40
    c.f_rimef = np.where(c.qs > 0, 1.1 ** ((k + 5 * j) % 44), 1.0)
    return c


def _warm_rain(i, j):
    c = Column("warm_rain", i, 101000.0 + 100.0 * j)
    c.temperature(301.0 + 0.5 * j, 0.0045, 280.0)
    c.humidity(0.95 + 0.005 * j, 0.9, 70000.0)
    cloud = bump(c.p, 85000.0, 12000.0)
    c.qv = np.where(cloud > 0, qsat(c.t, c.p) * (1.0 + 0.002 * j * cloud),
                    c.qv)
    c.qc = (0.3e-3 + 0.3e-3 * j) * cloud
    # rain rates from drizzle to downpour so every drop-size arm is used
    c.qr = (10.0 ** (-6.0 + 0.45 * j)) * bump(c.p, 92000.0, 12000.0)
    return c


def _dry(i, j):
    c = Column("dry", i, 95000.0 + 800.0 * j)
    c.temperature(305.0 - 4.0 * j, 0.0090, 210.0, inversion=3.0 * (j % 3),
                  inv_depth=300.0)
    c.humidity(0.05 + 0.03 * j, 0.02, 60000.0)
    k = np.arange(NZ)
    # stray condensate that evaporates or sublimates in one step
    c.qc = np.where(k % 9 == 4, 2.0e-5 * (j + 1), 0.0)
    c.qr = np.where(k % 11 == 3, 1.0e-4 * (j + 1), 0.0)
    c.qs = np.where(k % 7 == 5, 5.0e-5 * (j + 1), 0.0)
    c.f_rimef = np.where(c.qs > 0, 2.0, 1.0)
    return c


def _near_saturated(i, j):
    c = Column("near_saturated", i, 100000.0)
    c.temperature(278.0 + 2.0 * j, 0.0065, 205.0)
    k = np.arange(NZ)
    ice = (j % 2 == 1)
    q = np.where(c.t < T0C, qsat(c.t, c.p, ice=ice), qsat(c.t, c.p))
    # +/- 0.05 %, 0.1 % (RHLIMIT) and 0.2 % either side of saturation
    off = np.array([-2e-3, -1e-3, -5e-4, 0.0, 5e-4, 1e-3, 2e-3])
    c.qv = q * (1.0 + off[(k + j) % 7])
    c.qc = np.where(k % 3 == 0, 1.0e-6 * (j + 1), 0.0)
    c.qs = np.where((k % 4 == 1) & (c.t < T0C), 2.0e-6 * (j + 1), 0.0)
    return c


def _freezing_rain(i, j):
    c = Column("freezing_rain", i, 100500.0 - 300.0 * j)
    c.temperature(270.5 - 0.5 * j, 0.0050, 205.0, inversion=6.0,
                  inv_depth=700.0, nose=7.0 + 0.6 * j, nose_p=82000.0)
    c.humidity(0.99, 0.85, 60000.0)
    warm = c.t > T0C
    c.qr = (0.3e-3 + 0.2e-3 * j) * (bump(c.p, 85000.0, 12000.0)
                                    + bump(c.p, 97000.0, 5000.0))
    c.qs = np.where(~warm, (0.1e-3 + 0.05e-3 * j)
                    * bump(c.p, 65000.0, 12000.0), 0.0)
    c.qc = 0.05e-3 * bump(c.p, 90000.0, 6000.0)
    c.f_rimef = np.where(c.qs > 0, 1.0 + 1.5 * j, 1.0)
    return c


def _cold_anvil(i, j):
    c = Column("cold_anvil", i, 100000.0)
    c.temperature(303.0, 0.0076 + 0.0001 * j, 190.0 + 2.0 * j)
    c.humidity(0.9, 0.95, 70000.0, ice_above=True)
    c.qv = c.qv * (1.0 + 0.05 * bump(c.p, 22000.0, 8000.0))
    very_cold = c.t < T0C - 40.0
    c.qc = np.where(very_cold, (0.05e-3 + 0.05e-3 * j)
                    * bump(c.p, 25000.0, 10000.0), 0.0)
    c.qs = (0.2e-3 + 0.4e-3 * j) * bump(c.p, 22000.0, 12000.0)
    c.f_rimef = np.where(c.qs > 0, 1.0 + 5.0 * j, 1.0)
    return c


def _edge(i, j):
    c = Column("edge", i, 100000.0)
    c.temperature(285.0, 0.0065, 205.0)
    c.humidity(0.5, 0.3, 70000.0)
    k = np.arange(NZ)
    if j == 0:      # bone dry: vapour at and below EPSQ, no condensate
        c.qv = np.where(k % 2 == 0, 1.0e-12, 1.0e-13)
    elif j == 1:    # condensate either side of EPSQ (1e-12), EPSQ1, CLIMIT
        vals = np.array([0.0, 1.0e-12, 1.001e-12, 1.0011e-12, 2.0e-12,
                         1.0e-11, 1.0e-10])
        qt = vals[k % 7]
        c.raw = (qt, (k % 3) / 2.0, ((k + 1) % 3) / 2.0)
    elif j == 2:    # ice in warm air (melting) with very large rime factors
        c.qs = 0.8e-3 * bump(c.p, 90000.0, 12000.0)
        c.f_rimef = np.where(c.qs > 0, 30.0 + 0.5 * (k % 40), 1.0)
        # two levels of 30 g/kg unrimed ice above 0 C: the large-ice number
        # passes NLImax at the largest mean size, so the second pass raises
        # the rime factor past RFmax and the fall-speed lookup takes the
        # VEL_RF(:, Nrime) end of the table (module_mp_etanew.F:1115, 1162)
        heavy = (k == 2) | (k == 3)
        c.qs = np.where(heavy, 30.0e-3, c.qs)
        c.f_rimef = np.where(heavy, 1.0, c.f_rimef)
    elif j == 3:    # temperatures exactly 0 C and T_ICE (-40 C), saturated
        c.t = np.where(k % 3 == 0, T0C, np.where(k % 3 == 1, T0C - 40.0,
                                                 c.t))
        c.qc = np.full(NZ, 0.1e-3)
        c.qs = np.where(k % 2 == 0, 0.05e-3, 0.0)
        c.qv = qsat(c.t, c.p)
    elif j == 4:    # very heavy rain and cloud water: rain rates and rain
        # contents past the 1 mm mean-drop end of the tables
        # (module_mp_etanew.F:1492, 1509-1510)
        c.qr = 40.0e-3 * bump(c.p, 80000.0, 20000.0)
        c.qc = 5.0e-3 * bump(c.p, 70000.0, 20000.0)
        c.qv = qsat(c.t, c.p)
    elif j == 5:    # 20 % water supersaturation near the ground
        c.qv = np.where(k < 8, 1.2 * qsat(c.t, c.p), c.qv)
        c.qc = np.where(k < 4, 1.0e-4, 0.0)
    elif j == 6:    # 181 K aloft (just inside the 180 K table floor)
        c.temperature(240.0, 0.0090, 181.0)
        c.humidity(0.8, 0.9, 80000.0, ice_above=True)
        c.qs = np.where(c.p < 30000.0, 1.0e-6, 0.0)
        c.qc = np.where(c.p < 20000.0, 1.0e-7, 0.0)
    else:           # fractions outside [0, 1], rime factor below 1
        c.qs = 0.3e-3 * bump(c.p, 50000.0, 15000.0)
        c.qc = 0.3e-3 * bump(c.p, 80000.0, 15000.0)
        c.qr = 0.2e-3 * bump(c.p, 90000.0, 10000.0)
        qt = c.qc + c.qr + c.qs
        f_ice = np.where(k % 4 == 0, 1.3, np.where(k % 4 == 1, -0.2, 0.37))
        f_rain = np.where(k % 5 == 0, 1.5, np.where(k % 5 == 1, -0.1, 0.91))
        c.raw = (qt, f_ice, f_rain)
        c.f_rimef = np.where(qt > 0, 0.5, 0.0)
    return c


GROUPS = (_convective, _stratiform, _snow, _rimed_mixed, _warm_rain, _dry,
          _near_saturated, _freezing_rain, _cold_anvil, _edge)


def build() -> list[Column]:
    cols = []
    for maker in GROUPS:
        for j in range(8):
            cols.append(maker(len(cols), j))
    return cols


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = build()
    with open(out / "columns_inputs.csv", "w", newline="\n") as fc, \
            open(out / "columns.in", "w", newline="\n") as fi:
        fc.write("# Eta Ferrier (WRF 4.6.1 mp_physics=95) column-oracle "
                 "inputs; floats are binary32 bit patterns (hex)\n")
        fc.write("# column rows: C,col,regime,nsteps,dt,dx,rainnc\n")
        fc.write("# level rows (k=1 is the surface): L,col,k,"
                 + ",".join(FIELDS) + "\n")
        fi.write(f"{len(cols):6d} {NZ:6d}\n")
        for c in cols:
            a = c.arrays()
            fc.write(f"C,{c.idx},{c.regime},{NSTEPS},{f32hex(c.dt)},"
                     f"{f32hex(c.dx)},{f32hex(c.rainnc)}\n")
            fi.write(f"{NSTEPS:6d} {f32hex(c.dt)} {f32hex(c.dx)} "
                     f"{f32hex(c.rainnc)}\n")
            for k in range(NZ):
                words = [f32hex(a[f][k]) for f in FIELDS]
                fc.write(f"L,{c.idx},{k + 1}," + ",".join(words) + "\n")
                fi.write(" ".join(words) + "\n")
    print(f"wrote {len(cols)} columns x {NZ} levels to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_columns.py OUT_DIR")
    main(Path(sys.argv[1]))
