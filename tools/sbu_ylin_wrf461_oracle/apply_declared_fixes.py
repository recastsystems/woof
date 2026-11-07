#!/usr/bin/env python3
"""Write the declared-divergence copy of WRF's module_mp_sbu_ylin.F.

WOOF does not carry a WRF defect bit for bit: it fixes it and declares the
divergence.  This script applies exactly the declared fixes, each as one
exact-text replacement that must match once, to a COPY of the pinned
source.  The oracle is built twice, from the byte-unmodified file and from
this copy, so the port can be held to 0 ULP against the fixed scheme while
the pristine comparison shows where, and only where, the fixes move WRF's
answer.

D1  Psmltevp (evaporation of melting snow) drops the thermodynamic factor
    ``abr``.  The scheme computes abr two lines above and never uses it, so
    ``tmp2`` is a sum of positive terms, ``amin1(0.0, tmp2)`` is always 0
    and the process is dead code.  Purdue Lin, the code this scheme was
    built from, multiplies by it (module_mp_lin.F ``tmp2=abr*xnos*(...)``).
D2  Psaci and Psacw (ice and cloud water collected by snow) omit the
    fall-speed coefficient ``av_s``.  The collection rate is the integral of
    A(D) V(D) N(D) dD = aa_s av_s N0_s Gamma(ba+bv+1) / lambda^(ba+bv+1);
    without av_s the coefficient has units m^(bv_s-1) instead of s^-1, so
    ``exp(-tmp1*dtb)`` is dimensionally wrong.  Purdue Lin carries av_s in
    the same product.  Both the T < 0 and the T > 0 copy are fixed.
D3  The closing ``qv < qvmin`` repair zeroes cloud water and ice BEFORE it
    adds them back into vapour, so the add is a no-op and the condensate is
    destroyed.  The fix adds first.  Unreachable from any state the scheme
    itself produces (every vapour write above it is floored at qvmin), so
    it moves no measured word; it is fixed so the code says what it means.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIXES = (
    ("D1",
     "            tmp1=av_s(k)*sqrho(k)*olambdas(k)**(5+bv_s(k)+2*mu_s)/visc(k)\n"
     "            tmp2= N0_s(k)*( vf1s*olambdas(k)*olambdas(k)+ &\n",
     "            tmp1=av_s(k)*sqrho(k)*olambdas(k)**(5+bv_s(k)+2*mu_s)/visc(k)\n"
     "            tmp2= abr*N0_s(k)*( vf1s*olambdas(k)*olambdas(k)+ &\n"),
    ("D2-cold",
     "          save1 = aa_s(k)*sqrho(k)*N0_s(k)* &\n",
     "          save1 = aa_s(k)*av_s(k)*sqrho(k)*N0_s(k)* &\n"),
    ("D2-warm",
     "            save1 =aa_s(k)*sqrho(k)*N0_s(k)* &\n",
     "            save1 =aa_s(k)*av_s(k)*sqrho(k)*N0_s(k)* &\n"),
    ("D3",
     "            qlz(k)=0.0\n"
     "            qiz(k)=0.0\n"
     "            qvz(k)=amax1( qvmin,qvz(k)+qlz(k)+qiz(k) )\n",
     "            qvz(k)=amax1( qvmin,qvz(k)+qlz(k)+qiz(k) )\n"
     "            qlz(k)=0.0\n"
     "            qiz(k)=0.0\n"),
)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: apply_declared_fixes.py PRISTINE.F OUT.F", file=sys.stderr)
        return 2
    text = Path(argv[1]).read_text(encoding="ascii")
    for tag, old, new in FIXES:
        count = text.count(old)
        if count != 1:
            print(f"{tag}: expected exactly one match, found {count}",
                  file=sys.stderr)
            return 3
        text = text.replace(old, new)
    Path(argv[2]).write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
