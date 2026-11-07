#!/usr/bin/env python3
"""Write the Ferrier-Aligo advected-rime (WRF mp_physics=15) oracle inputs.

mp_physics=15 (``fer_mp_hires_advect``) runs the same column physics as
mp_physics=5 but carries the rime factor as the advected scalar
``QRIMEF = rime factor * qi`` instead of the ``F_ICE``/``F_RAIN``/``F_RIMEF``
state arrays.  The column set is therefore built from the two existing
mp=5 sets, converted to the advected package's state, plus one new regime
aimed at the arithmetic only this package has (``F_RIMEF = QRIMEF/QI`` on
entry and ``QRIMEF = QI*F_RIMEF`` on exit):

* the lane's 72 columns (``../make_columns.py``: nine regimes);
* the refute set's 65 columns (``../refute/make_columns_refute.py``:
  sixteen regimes aimed at the code the lane columns never ran);
* 16 ``rimef_bounds`` columns: QRIMEF at and beyond its bounds -- zero
  (the real.exe cold start), exactly QI, one ULP below QI, small and large
  negatives (advection undershoot), rime factors at and far above the
  RFmax/RFmx caps, QI exactly at and one ULP above EPSQ, orphan rime where
  there is no ice, a subnormal QRIMEF, a negative zero, ice in warm air with
  large rime, and a 60-step cold run.

Converted columns take ``QRIMEF = float32(qi) * float32(f_rimef)`` in
binary32, the same product FER_HIRES_ADVECT writes back, so a converted
column starts from the state an mp=15 run would carry for the mp=5 column's
ice.  The F_ICE and F_RAIN inputs of the mp=5 sets are dropped: the advected
package has no such state.

Written twice, as the mp=5 harness does: ``columns_inputs.csv`` (canonical,
binary32 hex) and ``columns.in`` (the fixed format
``run_ferrier_advect.F90`` reads).  No random numbers.

usage: make_columns_advect.py OUT_DIR
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = _load("_ferrier_make_columns", _ROOT / "make_columns.py")
rc = _load("_ferrier_make_columns_refute",
           _ROOT / "refute" / "make_columns_refute.py")

NZ = mc.NZ
T0C = mc.T0C
EPSQ = np.float32(1.0e-12)          # module_mp_fer_hires.F:168
FIELDS = ("p", "pi", "th", "qv", "qc", "qr", "qi", "qrimef", "dz8w", "rho")


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).astype(np.float32)


def advect_arrays(col) -> dict[str, np.ndarray]:
    """The mp=15 state of an mp=5 column: f_rimef folded into QRIMEF."""
    a = col.arrays()
    out = {name: f32(a[name]) for name in FIELDS if name != "qrimef"}
    override = getattr(col, "qrimef", None)
    if override is not None:
        out["qrimef"] = f32(override)
    else:
        out["qrimef"] = out["qi"] * f32(a["f_rimef"])   # binary32 product
    return out


class RimeCol(rc.Col):
    """A refute-style column whose QRIMEF is set directly (binary32)."""

    qrimef = None


def _rimef_bounds(start: int) -> list[RimeCol]:
    cols: list[RimeCol] = []
    k = np.arange(NZ)

    def base(j: int, *, dt: float, gid: int, nsteps: int = 10) -> RimeCol:
        c = RimeCol("rimef_bounds", start + j, 100500.0 - 200.0 * j, dt=dt,
                    gid=gid, nsteps=nsteps)
        c.dx = (12000.0, 3000.0, 1000.0, 4000.0)[j % 4]
        c.rainnc = 0.0 if j % 3 else 0.5 * j
        c.temperature(297.0 - 0.5 * j, 0.0068, 200.0)
        c.humidity(0.94, 0.8, 50000.0, ice_above=True)
        up = mc.bump(c.p, 65000.0, 25000.0)
        c.qv = c.qv * (1.0 + 0.02 * up)
        warm = c.t > T0C
        c.qc = (0.6e-3) * mc.bump(c.p, 72000.0, 20000.0)
        c.qr = np.where(warm, 1.0e-3 * mc.bump(c.p, 88000.0, 15000.0), 0.0)
        c.qi = np.where(~warm, 0.8e-3 * mc.bump(c.p, 45000.0, 22000.0), 0.0)
        return c

    def qi32(c) -> np.ndarray:
        return f32(c.qi)

    dts = (6.0, 20.0, 36.0, 72.0)
    for j in range(16):
        dt = dts[j % 4]
        gid = 1 + j % 3
        if j == 15:
            c = base(j, dt=36.0, gid=2, nsteps=60)
        else:
            c = base(j, dt=dt, gid=gid)
        q = qi32(c)
        if j == 0:      # real.exe cold start: QRIMEF = 0, F_RIMEF = 0
            c.qrimef = np.zeros(NZ)
        elif j == 1:    # exactly qi: F_RIMEF = 1 exactly
            c.qrimef = q
        elif j == 2:    # one ULP below qi: F_RIMEF just below 1
            c.qrimef = np.where(q > 0, np.nextafter(q, np.float32(0)), q)
        elif j == 3:    # small negative undershoot left by advection
            c.qrimef = np.where(q > 0, -1.0e-9, 0.0)
        elif j == 4:    # rime factor -1
            c.qrimef = -q
        elif j == 5:    # rime factor at and around the RFmax cap
            rf = np.where(k % 3 == 0, 45.0, np.where(k % 3 == 1, 45.26, 46.0))
            c.qrimef = q * f32(rf)
        elif j == 6:    # rime factor 1000
            c.qrimef = q * np.float32(1000.0)
        elif j == 7:    # rime factor 1e6
            c.qrimef = q * np.float32(1.0e6)
        elif j == 8:    # qi exactly EPSQ (no-ice arm) with nonzero QRIMEF
            c.qi = np.where(k % 2 == 0, EPSQ, c.qi)
            c.qrimef = np.where(k % 2 == 0, 5.0e-12, f32(c.qi) * 3.0)
        elif j == 9:    # qi one ULP above EPSQ, rime factor 50
            up1 = np.nextafter(EPSQ, np.float32(1))
            c.qi = np.where(k % 2 == 0, up1, c.qi)
            c.qrimef = f32(c.qi) * np.float32(50.0)
        elif j == 10:   # qi one ULP above EPSQ, QRIMEF 1e-6 (F ~ 1e6)
            up1 = np.nextafter(EPSQ, np.float32(1))
            c.qi = np.where(k < 30, up1, c.qi)
            c.qrimef = np.where(k < 30, 1.0e-6, f32(c.qi) * 2.0)
        elif j == 11:   # orphan rime: no ice, zero ice, negative ice
            c.qi = np.where(k % 3 == 0, 0.0,
                            np.where(k % 3 == 1, -1.0e-8, c.qi))
            c.qrimef = np.where(k % 3 == 2, f32(c.qi) * 4.0, 1.0e-5)
        elif j == 12:   # subnormal QRIMEF over ordinary ice
            c.qi = np.where(c.qi > 0, np.maximum(c.qi, 1.0e-4), c.qi)
            c.qrimef = np.where(f32(c.qi) > 0, np.float32(1.0e-40), 0.0)
        elif j == 13:   # negative zero QRIMEF over ice
            c.qrimef = np.copysign(np.zeros(NZ), np.where(q > 0, -1.0, 1.0))
        elif j == 14:   # ice in warm air, large and mixed rime factors
            c.qi = 0.9e-3 * mc.bump(c.p, 88000.0, 14000.0) + c.qi
            rf = np.array((0.0, 1.0, 3.0, 20.0, -0.1, 60.0))[k % 6]
            c.qrimef = f32(c.qi) * f32(rf)
        else:           # cold snow, rime factors 1..10, 60 steps
            c.temperature(266.0, 0.0062, 205.0)
            c.humidity(0.98, 0.7, 50000.0, ice_above=True)
            c.qv = c.qv * (1.0 + 0.03 * mc.bump(c.p, 75000.0, 20000.0))
            c.qr = np.zeros(NZ)
            c.qc = 0.05e-3 * mc.bump(c.p, 85000.0, 8000.0)
            c.qi = 0.4e-3 * mc.bump(c.p, 78000.0, 25000.0)
            rf = 1.0 + (k % 10)
            c.qrimef = f32(c.qi) * f32(rf)
        cols.append(c)
    return cols


def build() -> list:
    cols = []
    for maker in mc.GROUPS:
        for j in range(8):
            cols.append(maker(len(cols), j))
    for c in rc.build():
        c.idx = len(cols)
        cols.append(c)
    cols.extend(_rimef_bounds(len(cols)))
    for i, c in enumerate(cols):
        c.idx = i
    return cols


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = build()
    f32hex = mc.f32hex
    with open(out / "columns_inputs.csv", "w", newline="\n") as fc, \
            open(out / "columns.in", "w", newline="\n") as fi:
        fc.write("# Ferrier-Aligo advected rime (WRF 4.6.1 mp_physics=15) "
                 "column-oracle inputs; floats are binary32 bit patterns "
                 "(hex)\n")
        fc.write("# column rows: C,col,regime,gid,nsteps,dt,dx,rainnc\n")
        fc.write("# level rows (k=1 is the surface): L,col,k,"
                 + ",".join(FIELDS) + "\n")
        fi.write(f"{len(cols):6d} {NZ:6d}\n")
        for c in cols:
            a = advect_arrays(c)
            nsteps = getattr(c, "nsteps", mc.NSTEPS)
            fc.write(f"C,{c.idx},{c.regime},{c.gid},{nsteps},"
                     f"{f32hex(c.dt)},{f32hex(c.dx)},{f32hex(c.rainnc)}\n")
            fi.write(f"{c.gid:6d} {nsteps:6d} {f32hex(c.dt)} "
                     f"{f32hex(c.dx)} {f32hex(c.rainnc)}\n")
            for kk in range(NZ):
                words = [_hex(a[f][kk]) for f in FIELDS]
                fc.write(f"L,{c.idx},{kk + 1}," + ",".join(words) + "\n")
                fi.write(" ".join(words) + "\n")
    print(f"wrote {len(cols)} columns x {NZ} levels to {out}")


def _hex(x: np.float32) -> str:
    """Bit pattern of an already-binary32 value (keeps -0.0 and subnormals)."""
    return f"{int(np.float32(x).view(np.uint32)):08X}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
