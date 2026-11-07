"""Regimes and variant modes for the WRF v4.6.1 NSSL variant column oracle.

Every case is a slab of independent columns (one column per regime) that
both engines advance from the SAME float32 inputs: the official Fortran
``nssl_2mom_driver`` (``variant_column.F90``) and GPUWM's production NSSL
coordinator (``run_gpu.py``).  Inputs are written as 9-significant-digit
decimals, which round-trip float32 exactly through either reader.

Regimes (one column each):

* ``convective`` -- deep updraft with cloud, rain, ice, snow, graupel and
  hail, supersaturated core, warm base to a cold anvil;
* ``stratiform`` -- weak ascent, snow aloft melting through a bright band
  into rain;
* ``hail``       -- large low-concentration graupel and hail riming dense
  supercooled cloud in a strong updraft (graupel-to-hail conversion range),
  hail melting below the freezing level;
* ``cold``       -- an all-ice column below 250 K with ice nucleation;
* ``warm``       -- warm-rain only (no ice phase anywhere);
* ``edges``      -- threshold and near-zero moments, a dry layer, tiny
  numbers, and mixing ratios straddling every qxmin gate.

Modes name the resolved WRF selector set plus the coefficient options; the
three-moment and one-moment modes carry only the moments their Registry
packages allocate (absent moments are zero on input, as WRF's unallocated
scalars are never read by the driver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

F32 = np.float32
RD = 287.04
CP = 1004.0
NZ = 40
DZ = 250.0

REGIMES = ("convective", "stratiform", "hail", "cold", "warm", "edges")

#: Field order of the oracle input/output (Registry names).
STATE_FIELDS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh", "qzr", "qzg", "qzh",
)
ENV_FIELDS = ("pressure", "exner", "rho", "dz", "w")


@dataclass(frozen=True)
class Mode:
    """One resolved NSSL variant plus coefficient options."""

    label: str
    ipconc: int            # 0 one-moment, 5 two-moment, 8 three-moment
    hail_on: int           # module hail switch (Registry 1 or 2 -> 1)
    density_on: int        # Registry nssl_density_on (0, 1, 2)
    ccn_on: int
    icdx: int = 6
    icdxhl: int = 6
    alphah: float = 0.0
    alphahl: float = 1.0
    cccn: float = 0.5e9
    cnoh: float = 4.0e5
    cnohl: float = 4.0e4
    cnor: float = 8.0e5
    cnos: float = 3.0e6
    rho_qh: float = 500.0
    rho_qhl: float = 900.0
    rho_qs: float = 100.0
    # GPUWM selector spelling of the same mode
    selectors: dict = field(default_factory=dict)

    @property
    def two_moment(self) -> bool:
        return self.ipconc >= 5


MODES = (
    # default lane (control: the existing specialized path must match too)
    Mode("mp18-default", 5, 1, 2, 1),
    # (A) coefficient test endpoints. WRF does not check these bounds.
    Mode("alphah-0", 5, 1, 2, 1, alphah=0.0,
         selectors={"nssl_alphah": 0.0}),
    Mode("alphah-15", 5, 1, 2, 1, alphah=15.0,
         selectors={"nssl_alphah": 15.0}),
    Mode("alphah-2p5", 5, 1, 2, 1, alphah=2.5,
         selectors={"nssl_alphah": 2.5}),
    Mode("alphahl-0", 5, 1, 2, 1, alphahl=0.0,
         selectors={"nssl_alphahl": 0.0}),
    Mode("alphahl-15", 5, 1, 2, 1, alphahl=15.0,
         selectors={"nssl_alphahl": 15.0}),
    Mode("icdx-0", 5, 1, 2, 1, icdx=0, selectors={"nssl_icdx": 0}),
    Mode("icdx-3", 5, 1, 2, 1, icdx=3, icdxhl=3,
         selectors={"nssl_icdx": 3, "nssl_icdxhl": 3}),
    Mode("icdx-1", 5, 1, 2, 1, icdx=1, icdxhl=1,
         selectors={"nssl_icdx": 1, "nssl_icdxhl": 1}),
    Mode("icdx-4-5", 5, 1, 2, 1, icdx=4, icdxhl=5,
         selectors={"nssl_icdx": 4, "nssl_icdxhl": 5}),
    Mode("icdxhl-0", 5, 1, 2, 1, icdxhl=0, selectors={"nssl_icdxhl": 0}),
    Mode("cccn-low", 5, 1, 2, 1, cccn=1.0e7,
         selectors={"nssl_cccn": 1.0e7}),
    Mode("cccn-high", 5, 1, 2, 1, cccn=5.0e10,
         selectors={"nssl_cccn": 5.0e10}),
    # (B) fixed density
    Mode("density-0", 5, 1, 0, 1, selectors={"nssl_density_on": 0}),
    Mode("density-0-nohail", 5, 0, 0, 1,
         selectors={"nssl_density_on": 0, "nssl_hail_on": 0}),
    # (C) one-moment (deprecated mp=19 and mp=21 remaps)
    Mode("mp19-one-moment-hail", 0, 1, 1, 1,
         selectors={"nssl_2moment_on": 0, "nssl_hail_on": 2,
                    "nssl_density_on": 1}),
    Mode("mp21-one-moment", 0, 0, 0, 1,
         selectors={"nssl_2moment_on": 0, "nssl_hail_on": 0,
                    "nssl_density_on": 0}),
    # (D) three-moment
    Mode("three-moment", 8, 1, 2, 1, selectors={"nssl_3moment": 1}),
    Mode("three-moment-nohail", 8, 0, 1, 1,
         selectors={"nssl_3moment": 1, "nssl_hail_on": 0}),
)
MODE_BY_LABEL = {mode.label: mode for mode in MODES}


def _qvs(temperature, pressure):
    # WRF Soong-Ogura form (module tabqvs): 380/p * exp(17.27*(T-273.15)/(T-35.86))
    return 380.0 / pressure * np.exp(
        17.2693882 * (temperature - 273.15) / (temperature - 35.86))


def _column(regime: str, nz: int = NZ):
    z = (np.arange(nz) + 0.5) * DZ
    w = np.zeros(nz)
    out = {name: np.zeros(nz) for name in STATE_FIELDS}

    def blob(center, width, peak):
        return peak * np.exp(-((z - center) / width) ** 2)

    if regime == "cold":
        t_sfc, lapse = 250.0, 6.0e-3
    elif regime == "warm":
        t_sfc, lapse = 303.0, 1.0e-3
    elif regime == "stratiform":
        t_sfc, lapse = 284.0, 6.0e-3
    else:
        t_sfc, lapse = 300.0, 6.8e-3
    temperature = np.maximum(t_sfc - lapse * z, 205.0)
    pressure = 100000.0 * np.exp(-z / 8000.0)
    exner = (pressure / 1.0e5) ** (RD / CP)
    rh = np.full(nz, 0.85)

    if regime == "convective":
        w = 22.0 * np.sin(np.pi * np.clip(z / 11000.0, 0, 1)) + 0.5
        rh[(z > 1000) & (z < 9000)] = 1.012
        out["qc"] = blob(4000, 2200, 2.2e-3)
        out["qr"] = blob(2200, 1800, 3.0e-3)
        out["qi"] = blob(9000, 1500, 2.5e-4)
        out["qs"] = blob(8000, 1800, 9.0e-4)
        out["qg"] = blob(6000, 1800, 4.0e-3)
        out["qh"] = blob(5000, 1500, 3.0e-3)
    elif regime == "stratiform":
        w = 0.25 + 0.1 * np.sin(z / 1500.0)
        rh[z > 2000] = 1.002
        rh[z <= 2000] = 0.92
        out["qs"] = blob(4500, 1700, 1.6e-3)
        out["qr"] = blob(1200, 900, 8.0e-4)
        out["qi"] = blob(7000, 1500, 6.0e-5)
        out["qc"] = blob(2600, 500, 2.0e-4)
        out["qg"] = blob(3000, 600, 2.0e-4)
    elif regime == "hail":
        w = 30.0 * np.sin(np.pi * np.clip(z / 10000.0, 0, 1)) - 3.0
        rh[(z > 1500) & (z < 8000)] = 1.006
        out["qc"] = blob(5500, 1500, 4.5e-3)
        out["qr"] = blob(2000, 1500, 2.0e-3)
        out["qg"] = blob(6000, 1500, 7.0e-3)
        out["qh"] = blob(4000, 2200, 6.0e-3)
        out["qi"] = blob(8500, 1000, 1.5e-4)
        out["qs"] = blob(7500, 1000, 3.0e-4)
    elif regime == "cold":
        w = 1.2 * np.sin(np.pi * np.clip(z / 6000.0, 0, 1))
        rh[:] = 1.0
        out["qi"] = blob(3000, 1500, 2.0e-4)
        out["qs"] = blob(2000, 1500, 5.0e-4)
        out["qg"] = blob(1500, 700, 1.0e-4)
    elif regime == "warm":
        w = 4.0 * np.sin(np.pi * np.clip(z / 5000.0, 0, 1))
        rh[(z > 600) & (z < 4000)] = 1.004
        out["qc"] = blob(2000, 1000, 1.5e-3)
        out["qr"] = blob(1000, 900, 1.2e-3)
    elif regime == "edges":
        w = 2.0 * np.cos(z / 900.0)
        rh = 0.6 + 0.5 * np.abs(np.sin(z / 700.0))
        ladder = np.array([0.0, 1.0e-14, 1.1e-13, 1.0e-12, 3.0e-12, 1.0e-9,
                           1.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 1.0e-5, 1.0e-4,
                           3.0e-4, 1.0e-3])
        for i, name in enumerate(("qc", "qr", "qi", "qs", "qg", "qh")):
            out[name] = np.roll(np.resize(ladder, nz), 2 * i)

    qvs = _qvs(temperature, pressure)
    out["qv"] = rh * qvs
    out["theta"] = temperature / exner
    rho = pressure / (RD * temperature * (1.0 + 0.61 * out["qv"]))

    # Number, volume and sixth moments consistent with assumed mean sizes
    # (Registry units: numbers #/kg, volumes m3/kg, Z m6/m3).
    def number(q, density, diameter):
        return q / (density * math.pi / 6.0 * diameter ** 3)

    out["qndrop"] = number(out["qc"], 1000.0, 16.0e-6)
    out["qnr"] = number(out["qr"], 1000.0, 0.9e-3)
    out["qni"] = out["qi"] / 2.0e-11
    out["qns"] = number(out["qs"], 100.0, 1.2e-3)
    graupel_density = 400.0 + 200.0 * np.sin(z / 1700.0)
    hail_density = 800.0 + 80.0 * np.cos(z / 1300.0)
    dg = 2.5e-3 if regime != "hail" else 6.0e-3
    dh = 7.0e-3 if regime != "hail" else 1.6e-2
    out["qng"] = number(out["qg"], graupel_density, dg)
    out["qnh"] = number(out["qh"], hail_density, dh)
    out["qvolg"] = out["qg"] / graupel_density
    out["qvolh"] = out["qh"] / hail_density
    out["qnn"] = np.full(nz, 0.5e9 / 1.225) * (0.4 + 0.6 * np.exp(-z / 3000.0))
    if regime == "edges":
        # number moments straddling cxmin and the per-category minimums
        out["qnr"] = np.where(np.arange(nz) % 5 == 0, 0.0, out["qnr"])
        out["qng"] = np.where(np.arange(nz) % 7 == 0, 0.0, out["qng"])
        out["qndrop"] = np.where(np.arange(nz) % 3 == 0, 1.0e-9, out["qndrop"])

    def sixth(q, n_per_kg, density, alpha):
        g = ((6 + alpha) * (5 + alpha) * (4 + alpha)) / (
            (3 + alpha) * (2 + alpha) * (1 + alpha))
        n = n_per_kg * rho
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = g * (6.0 * rho * q / (math.pi * density)) ** 2 / n
        return np.where(n > 0, zz, 0.0)

    out["qzr"] = sixth(out["qr"], out["qnr"], 1000.0, 1.5)
    out["qzg"] = sixth(out["qg"], out["qng"], graupel_density, 2.0)
    out["qzh"] = sixth(out["qh"], out["qnh"], hail_density, 3.0)

    env = {
        "pressure": pressure, "exner": exner, "rho": rho,
        "dz": np.full(nz, DZ), "w": w,
    }
    return ({k: v.astype(F32) for k, v in out.items()},
            {k: v.astype(F32) for k, v in env.items()})


def apply_mode(state: dict, mode: Mode) -> dict:
    """Zero the moments the mode's Registry packages never allocate."""
    s = {k: v.copy() for k, v in state.items()}
    if not mode.two_moment:
        for name in ("qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn"):
            s[name][:] = 0.0
    if not mode.ccn_on:
        s["qnn"][:] = 0.0
    if not mode.hail_on:
        for name in ("qh", "qnh", "qvolh", "qzh"):
            s[name][:] = 0.0
    density_g = (mode.hail_on == 1) if mode.ipconc == 0 else (
        mode.density_on >= 1)
    density_h = mode.ipconc != 0 and mode.density_on == 2 and mode.hail_on
    if not density_g:
        s["qvolg"][:] = 0.0
    if not density_h:
        s["qvolh"][:] = 0.0
    if mode.ipconc < 6:
        for name in ("qzr", "qzg", "qzh"):
            s[name][:] = 0.0
    return s


def build_case(mode: Mode, regimes=REGIMES, nz: int = NZ):
    """Return (state, env) arrays shaped (nx, nz) for the slab."""
    states, envs = [], []
    for regime in regimes:
        state, env = _column(regime, nz)
        states.append(apply_mode(state, mode))
        envs.append(env)
    state = {k: np.stack([s[k] for s in states]) for k in STATE_FIELDS}
    env = {k: np.stack([e[k] for e in envs]) for k in ENV_FIELDS}
    return state, env


def write_oracle_input(path, mode: Mode, state, env, nsteps: int,
                       itimestep0: int, dt: float) -> None:
    nx, nz = state["theta"].shape
    with open(path, "w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{nx} {nz} {nsteps} {itimestep0} {float(F32(dt)):.9g}\n")
        stream.write(f"{mode.ipconc} {mode.hail_on} {mode.density_on} "
                     f"{mode.ccn_on} {mode.icdx} {mode.icdxhl}\n")
        stream.write(" ".join(f"{F32(v):.9g}" for v in (
            mode.alphah, mode.alphahl, mode.cccn, mode.cnoh, mode.cnohl,
            mode.cnor, mode.cnos, mode.rho_qh, mode.rho_qhl,
            mode.rho_qs)) + "\n")
        for ix in range(nx):
            for k in range(nz):
                values = [state[n][ix, k] for n in STATE_FIELDS]
                values += [env[n][ix, k] for n in ENV_FIELDS]
                stream.write(" ".join(f"{F32(v):.9g}" for v in values))
                stream.write("\n")
