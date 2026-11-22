#!/usr/bin/env python3
"""Write an INSTRUMENTED copy of WRF's module_mp_thompson.F.

WHY
---
``wp08-freeze``'s surviving ``nr`` residual at 0-based level 0 was
attributed to a units disagreement in the sedimentation presence gate:
``thompson.cu:438`` opens rain fall-speed calculation on ``qr > 1.0e-12f``,
a MIXING RATIO, while ``module_mp_thompson.F:3616`` opens on
``rr(k) .gt. R1``, a MASS CONCENTRATION.

That attribution was then published as FALSIFIED, on the reasoning that
WRF's TAU+1 test at :3236 fails at that level too, so WRF floors ``rr`` to
``R1`` and closes its own gate.  The reasoning reads ``qrten`` at the END of
the step and assumes it is the value :3236 tested.  It is not: the rain
evaporation block at :3501-3573 runs in between, SUBTRACTS from ``qrten``,
and then rebuilds ``rr(k) = MAX(R1, (qr1d(k) + DT*qrten(k))*rho(k))`` at
:3568 without re-testing anything.  Only a measurement inside the routine
settles it, and no committed column fixture can expose that point: the
fixtures are entry and exit columns and this is the middle of
``mp_thompson``.

This script produces the copy that does expose it, DETERMINISTICALLY: it
matches its two anchors by exact text, refuses to run if either is not found
exactly once, and adds ONLY a declaration and WRITE statements.  No physics
line is touched and nothing is read that WRF has not already computed.

WHAT IT ADDS
------------
1. One local declaration in ``mp_thompson``, ``aa_k``.
2. One block immediately after the rain fall-speed / sub-step loop closes
   at :3640-3642, i.e. at the moment rain sedimentation is about to run.
   Per level it appends a ``SEDIN`` row holding ``rr``, ``nr``, ``rho``,
   ``rhof``, ``vtrk``, ``vtnrk``, the entry ``qr1d``/``nr1d``, the surviving
   ``qrten``/``nrten`` and ``L_qr``; then one ``SEDSCAL`` row holding
   ``nstep``, ``ksed1(1)`` and ``onstep(1)``.

   ``L_qr`` is the decisive field.  It is written in exactly two places
   before this point -- :1887/:1904 at entry classification and :3239/:3254
   in the TAU+1 block -- so its value here IS the branch :3236 took.

FIDELITY PROOF
--------------
Build the instrumented source with ``build_aero.sh``, which re-checks every
committed fixture byte for byte at the end.  Measured 2026-08-02 on the
lane node: all 44 aerosol CSVs and both ``wp08-freeze`` files reproduced
exactly, so the instrumentation is inert.

WHAT IT MEASURED
----------------
For ``wp08-freeze`` at 0-based level 1 (Fortran k=2), the level where the
two gates disagree:

    L_qr   = T                          <- :3236 took the TRUE branch
    rr     = 1.17481534483293570E-12    <- above R1, so :3616 OPENS
    nr     = 1.58222410827875137E-02
    vtrk   = 5.02253711223602295E-01    vtnrk = 3.16547304391860962E-01
    qr1d   = 0                          qrten = 8.52096171399807645E-14
    nstep  = 1                          ksed1(1) = 6

and at 0-based level 2, whose velocities WOOF-ArWen inherits instead:

    vtrk   = 2.59040689468383789E+00    vtnrk = 1.68873035907745361E+00

``(qr1d + qrten*DT) = 8.521e-13`` is below R1 at the END of the step, which
is what the falsification read; ``rr`` is 1.175e-12, which is what :3616
actually tests.  WRF gives that level a real fall speed and WOOF-ArWen gives it
the level above's, 5.3x faster in number.

USAGE
-----
    python3 instrument_sedimentation_entry.py PRISTINE.F OUTPUT.F
"""

from __future__ import annotations

import sys
from pathlib import Path


#: The integer declaration line in ``mp_thompson``.  The counter goes after
#: it rather than reusing ``k``, which the surrounding code still owns.
_DECL_ANCHOR = ("      INTEGER:: i, k, k2, n, nn, nstep, k_0, kbot, IT, "
                "iexfrq, k_melting\n")
_DECL_ADD = "      INTEGER:: aa_k\n"

#: The three lines that close the rain fall-speed / sub-step-count loop.
#: Everything the dump reads is final at this point and nothing between
#: here and the sedimentation loop changes any of it.
_SED_ANCHOR = """      if (ksed1(1) .eq. kte) ksed1(1) = kte-1
      if (nstep .gt. 0) onstep(1) = 1./REAL(nstep)
      endif
"""

_SED_ADD = _SED_ANCHOR + """
      do aa_k = kts, kte
         write(*,'(A,I4,10(A,ES24.17),A,L2)') 'SEDIN,', aa_k, &
            ',', rr(aa_k), ',', nr(aa_k), ',', rho(aa_k), ',', rhof(aa_k), &
            ',', vtrk(aa_k), ',', vtnrk(aa_k), ',', qr1d(aa_k), &
            ',', nr1d(aa_k), ',', qrten(aa_k), ',', nrten(aa_k), &
            ',', L_qr(aa_k)
      enddo
      write(*,'(A,I6,A,I6,A,ES24.17)') 'SEDSCAL,', nstep, ',', ksed1(1), &
         ',', onstep(1)
"""


def instrument(text: str) -> str:
    for anchor in (_DECL_ANCHOR, _SED_ANCHOR):
        found = text.count(anchor)
        if found != 1:
            raise SystemExit(
                f"anchor found {found} times, expected exactly 1: "
                f"{anchor.splitlines()[0]!r}")
    text = text.replace(_DECL_ANCHOR, _DECL_ANCHOR + _DECL_ADD)
    return text.replace(_SED_ANCHOR, _SED_ADD)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.splitlines()[-2].strip(), file=sys.stderr)
        return 2
    source, destination = Path(argv[1]), Path(argv[2])
    destination.write_text(instrument(source.read_text()))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
