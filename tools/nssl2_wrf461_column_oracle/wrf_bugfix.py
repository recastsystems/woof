"""Make the "WRF with its documented defects corrected" reference source.

  wrf_bugfix.py IN.f90 OUT.f90

WOOF does not copy four WRF v4.6.1 defects in nssl_2mom_gs (each listed in
README.md with its line in module_mp_nssl_2mom.F and the evidence that it
is a defect).  Grading WOOF against stock WRF therefore cannot be bitwise
at the cells those defects touch.  This tool applies exactly the four
corrections to WRF's public or preprocessed source, nothing else, so the oracle
can grade WOOF bitwise against WRF-as-intended and, separately, measure
what each defect changes in stock WRF.

  1 raindn  :16522  raindn(lh) = Min(Max(rimc3, rimdn(lh)), rimc4) clamps
                    the CLOUD rime density into the RAIN rime density it
                    has just computed; corrected to clamp raindn.
  2 qiacr   :16806  the rain-ice mass kernel uses xdia(mgs,lh,3), the
                    graupel diameter, beside dab1lh(li,lr); corrected to
                    the rain diameter xdia(mgs,lr,3).
  3 else    :16877  "ELSE ! single-moment rain" closes the outer iacr/eri/
                    temperature test instead of the ipconc >= 3 test, so
                    the single-moment rate runs for two-moment rain when
                    that outer test fails; corrected to pair with
                    ipconc >= 3.
  4 rfrzfac :17761  the rain-freezing over-depletion limiter multiplies by
                    qrfrz*dtp/qx (> 1); corrected to its reciprocal.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")


def find_one(pred, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    if len(hits) != 1:
        sys.exit(f"{what}: expected one anchor, found {len(hits)}")
    return hits[0]


i = find_one(lambda l: l.strip() == "raindn(mgs,lh) = Min( Max( rimc3, rimdn(mgs,lh) ), rimc4 )", "raindn")
lines[i] = lines[i].replace("rimdn(mgs,lh)", "raindn(mgs,lh)")

i = find_one(lambda l: "dab1lh(mgs,li,lr)*xdia(mgs,lh,3)*xdia(mgs,li,3)" in l, "qiacr")
lines[i] = lines[i].replace("dab1lh(mgs,li,lr)*xdia(mgs,lh,3)", "dab1lh(mgs,li,lr)*xdia(mgs,lr,3)")

i = find_one(lambda l: "((0.25/gf4)*pi)*eri(mgs)*cx(mgs,li)*qx(mgs,lr)" in l, "else")
# WRF's preprocessed .f90 has no comments; its public .F retains
# "ELSE ! single-moment rain". Looking for a bare ELSE in the latter
# silently selected an unrelated earlier size branch and moved its ENDIF.
# Compare Fortran statements in both inputs, and require the native branch
# context before moving any control flow.
def statement(index):
    return lines[index].split("!", 1)[0].strip().upper()


e = max(j for j in range(i) if statement(j) == "ELSE")
a = max(j for j in range(e) if statement(j) == "ENDIF")
c = min(j for j in range(i, len(lines)) if statement(j) == "ENDIF")
if any(statement(j) for j in range(a + 1, e)):
    sys.exit("else: unexpected statements between the ENDIF and the ELSE")
if not any("IPCONC" in statement(j) and ".GE. 3" in statement(j)
           for j in range(max(0, a - 100), a)):
    sys.exit("else: missing native ipconc >= 3 branch before the ENDIF")
lines.insert(c + 1, "      ENDIF")
del lines[a]

i = find_one(lambda l: l.strip() == "fac = ( qrfrz(mgs) )*dtp/qx(mgs,lr)", "rfrzfac")
lines[i] = lines[i].replace("fac = ( qrfrz(mgs) )*dtp/qx(mgs,lr)", "fac = qx(mgs,lr)/(( qrfrz(mgs) )*dtp)")

open(dst, "w").write("\n".join(lines))
print("corrected", dst)
