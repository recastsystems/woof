"""Column fixture for the WDM7 WRF v4.6.1 oracle.

usage: python3 fixture.py OUTDIR

WDM7 is WDM6 plus a hail category, so the fixture is the WDM6 oracle's 84
columns (tools/wdm6_wrf461_oracle/fixture.py, imported, not copied) with no
initial hail -- their graupel still makes hail through phaut and the rain
accretion of graupel -- followed by hail regimes of WDM7's own: supercells
over land and ocean, hail falling through a melting layer (complete and
near-complete melting), hail riming warm and supercooled cloud, hail
collecting ice, snow, graupel and rain, hail sublimating into dry cold air
and evaporating into dry warm air, graupel converting to hail, hail alone,
hail at its thresholds, subnormal and negative hail, very heavy hail, a
cold anvil, and random mixtures.  Seeded, so the files are reproducible.

Writes one input file per configuration (``CONFIGS``) on 50 levels, one
(``DEEP_CONFIG``) on 72 levels (WOOF's second kernel tier), and
``columns.txt`` naming every column.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


io7 = _load("wdm7_oracle_io", _HERE / "oracle_io.py")
f6 = _load("wdm6_oracle_fixture",
           _HERE.parent / "wdm6_wrf461_oracle" / "fixture.py")

NZ, DEEP_NZ = f6.NZ, f6.DEEP_NZ
CCN0 = f6.CCN0
EP2 = f6.EP2

# name: (delt, nsteps, itimestep0)
CONFIGS = {
    "dt20": (20.0, 1, 2),
    "dt20_coldstart": (20.0, 1, 1),
    "dt60_x10": (60.0, 10, 2),
    "dt300": (300.0, 1, 2),
    "dt150_x3": (150.0, 3, 2),
    "dt10_x30": (10.0, 30, 2),
}
DEEP_CONFIG = ("deep72_dt30_x2", (30.0, 2, 2))


def _qh(c):
    if not hasattr(c, "qh"):
        c.qh = np.zeros(f6.NZ_ACTIVE)
    return c.qh


def _esat_i(t):
    return 610.78 * np.exp(21.875 * (t - 273.15) / (t - 7.66))


def hail_columns(rng):
    ZM = f6.ZM
    blob = f6._blob
    cols = []
    # A. supercells: hail aloft falling through the melting level.
    for i, (scale, peak, xland) in enumerate(
            ((1.0, 2.0e-3, 1.0), (2.0, 5.0e-3, 1.0), (3.0, 8.0e-3, 1.0),
             (5.0, 1.5e-2, 1.0), (1.5, 3.0e-3, 2.0), (3.0, 6.0e-3, 2.0))):
        c = f6.convective(f"hail_supercell_{i}", rng, xland, scale,
                          ccn=10 ** rng.uniform(8.0, 9.3),
                          tsfc=303.0 - 2.0 * i)
        _qh(c)[:] = blob(5500.0 + 300.0 * i, 2200.0, peak)
        cols.append(c)
    # B. hail melting into rain: near-complete and complete melting.
    for i, peak in enumerate((1.0e-6, 2.0e-5, 1.0e-4, 6.0e-4)):
        tsfc = 281.0 + 3.0 * i
        c = f6.Column(f"hail_melting_{i}", 99000.0, tsfc,
                      f6.rh_profile(0.92, 0.98, 0.5))
        frz = (tsfc - 273.15) / 6.5e-3
        _qh(c)[:] = np.where(ZM > frz - 600.0, blob(frz + 400.0, 900.0, peak),
                             0.0)
        c.qr[:] = np.where(ZM < frz + 200.0, (0.1 + 0.2 * i) * 1e-3, 0.0)
        c.nr[:] = np.where(c.qr > 0, 10 ** (3.0 + 0.3 * i), 0.0)
        c.qc[:] = blob(frz + 300.0, 500.0, 2.0e-4)
        c.nc[:] = np.where(c.qc > 1e-8, 1.5e8, 0.0)
        cols.append(c)
    # C. hail riming warm cloud (T > 0 everywhere): phacw against qc.
    for i, (qc, qh) in enumerate(((5.0e-4, 8.0e-3), (2.0e-3, 2.0e-3))):
        c = f6.Column(f"hail_warm_riming_{i}", 101000.0, 305.0, 1.0,
                      lapse=4.0e-3)
        c.qc[:] = np.where(ZM < 5000.0, qc, 0.0)
        c.nc[:] = np.where(c.qc > 0, 2.0e8, 0.0)
        _qh(c)[:] = np.where((ZM > 1000.0) & (ZM < 4000.0), qh, 0.0)
        cols.append(c)
    # D. hail collecting every cold species below freezing.
    for i in range(3):
        c = f6.Column(f"hail_cold_collection_{i}", 85000.0, 268.0 - 4.0 * i,
                      f6.rh_profile(1.0, 0.98, 0.6), lapse=6.0e-3)
        c.qc[:] = blob(2000.0, 900.0, 6.0e-4)
        c.nc[:] = np.where(c.qc > 1e-8, 1.0e8, 0.0)
        c.qi[:] = blob(5000.0, 1500.0, 2.0e-4)
        c.qs[:] = blob(3500.0, 1500.0, 8.0e-4)
        c.qg[:] = blob(3000.0, 1200.0, 1.5e-3)
        c.qr[:] = blob(1500.0, 800.0, (0.5 + i) * 1.0e-4)
        c.nr[:] = np.where(c.qr > 1e-9, 10 ** (3.0 + i), 0.0)
        _qh(c)[:] = blob(3000.0, 1500.0, (2.0 + 3.0 * i) * 1.0e-3)
        cols.append(c)
    # E. hail sublimating (cold, dry) and evaporating (warm, dry).
    for i in range(2):
        c = f6.Column(f"hail_sublimating_{i}", 70000.0, 262.0 - 5.0 * i,
                      0.3 + 0.2 * i)
        _qh(c)[:] = blob(2500.0, 1500.0, 3.0e-3)
        cols.append(c)
    for i in range(2):
        c = f6.Column(f"hail_evaporating_{i}", 98000.0, 308.0 - 3.0 * i,
                      f6.rh_profile(0.25 + 0.15 * i, 0.5, 0.3),
                      lapse=8.5e-3)
        _qh(c)[:] = np.where(ZM < 5000.0, (1.0 + 2.0 * i) * 1.0e-3, 0.0)
        cols.append(c)
    # F. graupel converting to hail (qg > qs0) and rain accreting graupel.
    for i in range(3):
        c = f6.Column(f"graupel_to_hail_{i}", 90000.0, 272.0 - 3.0 * i,
                      f6.rh_profile(0.98, 0.95, 0.5))
        c.qg[:] = blob(2500.0, 1500.0, (1.0 + 1.5 * i) * 1.0e-3)
        c.qr[:] = blob(1500.0, 1000.0, (0.5 + 1.0 * i) * 1.0e-4)
        c.nr[:] = np.where(c.qr > 1e-9, 2.0e3, 0.0)
        cols.append(c)
    # G. hail alone; hail at its thresholds; subnormal, negative, heavy.
    c = f6.Column("hail_only", 100000.0, 290.0, 0.7)
    _qh(c)[:] = blob(4000.0, 2000.0, 4.0e-3)
    cols.append(c)
    c = f6.Column("hail_exact_thresholds", 95000.0, 280.0, 0.9)
    _qh(c)[:] = 1.0e-9
    _qh(c)[18:26] = 1.0e-4
    c.qg[:] = 6.0e-4                    # qs0: phaut's threshold
    c.qr[20:24] = 1.0e-4                # delta2/delta3 switch values
    c.qs[20:24] = 1.0e-4
    c.nr[20:24] = 1.0e3
    cols.append(c)
    c = f6.Column("hail_subnormal", 100000.0, 285.0, 0.7)
    _qh(c)[:] = np.float32(1.0e-40)
    _qh(c)[10] = np.float32(1.4e-45)
    cols.append(c)
    c = f6.convective("hail_negative_from_advection", rng, 1.0, 0.8, 2e8)
    _qh(c)[:] = blob(6000.0, 2000.0, 3.0e-3)
    _qh(c)[::4] = -2.0e-6
    cols.append(c)
    c = f6.convective("hail_very_heavy", rng, 1.0, 4.0, 4e8)
    _qh(c)[:] = blob(5000.0, 2500.0, 3.0e-2)
    cols.append(c)
    c = f6.Column("hail_cold_anvil", 100000.0, 300.0,
                  f6.rh_profile(0.8, 0.9, 0.95), tstrat=200.0, ztrop=13000.0)
    c.qc[:] = np.where(c.t < 233.0, blob(11500.0, 1200.0, 2e-4), 0.0)
    c.nc[:] = np.where(c.qc > 1e-9, 5e7, 0.0)
    c.qi[:] = blob(12000.0, 1500.0, 3e-4)
    _qh(c)[:] = blob(10000.0, 2000.0, 1.0e-3)
    cols.append(c)
    # H. random mixtures with hail.
    for i in range(8):
        c = f6.Column(f"hail_random_{i}", rng.uniform(70000.0, 103000.0),
                      rng.uniform(262.0, 308.0),
                      f6.rh_profile(rng.uniform(0.3, 1.0),
                                    rng.uniform(0.3, 1.02),
                                    rng.uniform(0.1, 0.9)),
                      xland=float(rng.choice([1.0, 2.0])))
        for field, lo, hi in (("qc", -6, -2.7), ("qr", -7, -2.3),
                              ("qi", -7, -3.5), ("qs", -7, -2.8),
                              ("qg", -7, -2.3), ("qh", -7, -1.8)):
            if field == "qh" or rng.random() < 0.8:
                arr = _qh(c) if field == "qh" else getattr(c, field)
                arr[:] = blob(rng.uniform(500, 12000), rng.uniform(300, 3000),
                              10 ** rng.uniform(lo, hi))
        c.nc[:] = np.where(c.qc > 1e-10, 10 ** rng.uniform(6, 9.5), 0.0)
        c.nr[:] = np.where(c.qr > 1e-10, 10 ** rng.uniform(0, 6), 0.0)
        c.nn[:] = 10 ** rng.uniform(7.5, 10.5)
        cols.append(c)
    return cols


def build_columns(seed=20261008):
    cols = f6.build_columns()
    rng = np.random.default_rng(seed)
    cols += hail_columns(rng)
    for c in cols:              # every column carries qh, zero by default
        _qh(c)
    return cols


def assemble(cols):
    levels, surface = f6.assemble(cols)
    levels["qh"] = np.stack([np.asarray(c.qh, np.float32) for c in cols],
                            axis=1)
    ncol = len(cols)
    surface["hail"] = np.linspace(0.0, 2.0, ncol).astype(np.float32)
    return levels, surface


def main(argv):
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    f6.use_grid(NZ)
    cols = build_columns()
    levels, surface = assemble(cols)
    ncol = len(cols)
    for name, (delt, nsteps, it0) in CONFIGS.items():
        lv = {k: v.copy() for k, v in levels.items()}
        if it0 == 1:
            # Cold start: WRF overwrites nn with ccn0 at itimestep 1; WOOF
            # fills nn with ccn0 at allocation.  Both start from that field.
            lv["nn"][...] = np.float32(CCN0)
        io7.write_input(out / f"{name}.in",
                        io7.Case(ncol, NZ, nsteps, it0, np.float32(delt),
                                 np.float32(CCN0), lv, surface))
    f6.use_grid(DEEP_NZ)
    deep_cols = build_columns()
    deep_levels, deep_surface = assemble(deep_cols)
    assert [c.name for c in deep_cols] == [c.name for c in cols]
    name, (delt, nsteps, it0) = DEEP_CONFIG
    io7.write_input(out / f"{name}.in",
                    io7.Case(ncol, DEEP_NZ, nsteps, it0, np.float32(delt),
                             np.float32(CCN0), deep_levels, deep_surface))
    f6.use_grid(NZ)
    with open(out / "columns.txt", "w", encoding="ascii") as f:
        for i, c in enumerate(cols):
            f.write(f"{i} {c.name} xland={c.xland}\n")
    print(f"{ncol} columns x {NZ} levels, {len(CONFIGS)} configurations, "
          f"plus {DEEP_CONFIG[0]} on {DEEP_NZ} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
