"""Write the WRF v4.6.1 TEMF modules with WOOF's declared fixes.

    python3 apply_fixes.py <WRF_SOURCE_ROOT> <OUT_DIR> [F1,F2,...]

With no third argument every fix is applied (the oracle's "fixed" arm);
a comma-separated subset builds the single-fix attribution arms.

The fixed arm of the oracle compiles these copies; the stock arm compiles
the byte-unmodified files.  Each fix is an exact text substitution that
must match exactly once, so an edited or different WRF file is refused
rather than half-patched.  The same six fixes are the port's default
(woof/core/kernels/temf.cu, "DECLARED DIVERGENCES"):

F1  module_bl_temf.F:386.  TVAL, the updraft taper weight, is declared
    INTEGER, so ``tval = 1. - (zm - 0.9 ht)/(0.1 ht)`` truncates every
    weight inside the taper band to 0 and the band becomes a step at
    0.9 ht.  It is declared REAL.  Bits move only at levels strictly
    inside (0.9 ht, ht); a weight of exactly 0 or 1 gives the same words.

F2  module_bl_temf.F:1334.  The bottom row of SOLVE_IMPLICIT_TEMF divides
    its diagonal term by DZM(KBOT+1) and its off-diagonal term by
    DZM(KBOT), so a uniform profile under zero surface flux does not stay
    uniform and the column total is not conserved.  The diagonal takes
    DZM(KBOT).  The u, v, thetal and qt solves pass DZM(1) = DZM(2)
    (module_bl_temf.F:515), so their words do not move; the TE solve
    passes DZT(2:KTE) as its DZM and is where the fix acts.

F3  module_sf_temfsfclay.F:351-352.  EXCH_TEMF and CHS are FLQC/MAVAIL with
    FLQC already multiplied by MAVAIL, so MAVAIL = 0 gives 0/0 = NaN in
    CHS, which the land surface model then reads.  Where MAVAIL is zero
    they are formed without the MAVAIL factor; every other column keeps
    WRF's words.

The last three are index slips against the scheme's own convention that
TEMF level k (k >= 2) is WRF mass level k-1 (module_bl_temf.F:478-502,
"NOTE this is where the indexing gets changed from WRF to TEMF basis"),
the convention the author's own correction at :923-924 restores for RST:

F4  module_bl_temf.F:490.  THETAL(k) is formed from THX, QVX, QCX at WRF
    level k-1 but from TX at level k, the temperature of the level above,
    so a cloudy level's liquid-water potential temperature does not belong
    to its own temperature and the inverse conversion (THLQT2THQVQC, at
    level k-1) returns a different theta and qc: a spurious tendency in
    every cloudy level.  TX(k-1).

F5  module_bl_temf.F:724, :734.  The updraft temperature TUPD(k) and the
    moist updraft THETAV take PI2D(k), the Exner function of the level
    above the updraft level, while its saturation mixing ratio takes
    P2D(k-1): the parcel is cooled by one level and saturates early, so the
    LCL and the cloud-top search sit low.  PI2D(k-1).

F6  module_bl_temf.F:513.  DZT(kt), the spacing of mass levels kt and kt+1
    (the depth of turbulence layer kt), is formed as Z2D(kt+1) - Z2D(kt),
    which is the spacing of TEMF levels kt+1 and kt+2: every gradient
    divides by the next layer up, so on a stretched grid every gradient is
    under-estimated by the stretch ratio, and the top level needed the
    :516 patch because the formula reads past the column.  DZT(1)
    (:506) is right.  Z2D(kt) - Z2D(kt-1); the :516 top-level copy stays.
"""
from __future__ import annotations

import sys
from pathlib import Path

FIXES = {
    "phys/module_bl_temf.F": [
        ("F4",
         "            ((cp + rt(i,k)*cpv) * tx(i,k)))\n",
         "            ((cp + rt(i,k)*cpv) * tx(i,k-1)))  ! WOOF F4\n"),
        ("F6",
         "         dzt(i,kt) = z2d(i,kt+1) - z2d(i,kt)\n",
         "         dzt(i,kt) = z2d(i,kt) - z2d(i,kt-1)  ! WOOF F6\n"),
        ("F5",
         "               TUPD(i,k) = thup_temfx(i,k) * pi2d(i,k)   \n",
         "               TUPD(i,k) = thup_temfx(i,k) * pi2d(i,k-1)  ! WOOF F5\n"),
        ("F5",
         "                  thetavUPDmoist(i,k) = (thup_temfx(i,k) + "
         "((xlv/cp)*qlUPD(i,k)/pi2d(i,k))) * &\n",
         "                  thetavUPDmoist(i,k) = (thup_temfx(i,k) + "
         "((xlv/cp)*qlUPD(i,k)/pi2d(i,k-1))) * &  ! WOOF F5\n"),
        ("F1",
         "   integer, dimension( its:ite) ::  tval\n",
         "   real, dimension( its:ite)    ::  tval   ! WOOF F1: was INTEGER\n"),
        ("F2",
         "   BU(kbot) = -1.0/dt - Khlf(kbot+1)/(dzm(kbot+1)*dzt(kbot+1))\n",
         "   BU(kbot) = -1.0/dt - Khlf(kbot+1)/(dzm(kbot)*dzt(kbot+1))  ! WOOF F2\n"),
    ],
    "phys/module_sf_temfsfclay.F": [
        ("F3",
         "      exch_temfx(i)  = flqc(i) / mavail(i)\n"
         "      chs(i) = flqc(i) / rho(i) / mavail(i) \n",
         "      if (mavail(i) /= 0.) then   ! WOOF F3\n"
         "         exch_temfx(i)  = flqc(i) / mavail(i)\n"
         "         chs(i) = flqc(i) / rho(i) / mavail(i) \n"
         "      else\n"
         "         exch_temfx(i) = rho(i) * fth(i)/fth0 * wm(i) * leps(i) / PrT0 / log(zm(i)/z0t(i)) / zt(i)\n"
         "         chs(i) = exch_temfx(i) / rho(i)\n"
         "      end if\n"),
    ],
}


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    wanted = (set(sys.argv[3].split(",")) if len(sys.argv) > 3
              else {tag for edits in FIXES.values() for tag, _, _ in edits})
    out.mkdir(parents=True, exist_ok=True)
    for rel, edits in FIXES.items():
        text = (root / rel).read_text(encoding="ascii")
        for tag, old, new in edits:
            if tag not in wanted:
                continue
            count = text.count(old)
            if count != 1:
                print(f"{tag}: expected one match in {rel}, found {count}",
                      file=sys.stderr)
                return 1
            text = text.replace(old, new)
        (out / Path(rel).name).write_text(text, encoding="ascii")
        done = sorted({tag for tag, _, _ in edits if tag in wanted})
        print(f"wrote {out / Path(rel).name} ({','.join(done) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
