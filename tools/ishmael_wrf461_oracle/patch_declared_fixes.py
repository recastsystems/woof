"""Apply WOOF's declared ISHMAEL defect fixes to a copy of the WRF source.

WOOF never runs bit-exact to a WRF defect: each fix below is applied in
the port (woof/core/kernels/ishmael.cu, default and strict arms alike)
and declared.  The oracle builds the pristine module AND this patched copy,
so the port is graded word for word against WRF's own arithmetic with only
the declared lines changed, and the pristine comparison measures exactly
what each fix moves.

Every fix is anchored on its line number AND its exact text, and the
script refuses a source that does not match, so it can never patch a
different WRF.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

#: id -> (1-based line, exact pristine text, replacement text, why)
DECLARED_FIXES = {
    "D1-ice-rain-depletion-sum": (
        1105,
        "                   sink = tmpsum*i_rhoair(k)",
        "                   sink = tmpsumr*i_rhoair(k)",
        "The rain over-depletion guard for ice-rain riming divides by the "
        "ice-CLOUD riming sum (tmpsum) instead of the ice-RAIN riming sum "
        "(tmpsumr) it was written to bound (module_mp_jensen_ishmael.F:1104-"
        "1113, the copy of the cloud guard at :1094-1102).  WRF therefore "
        "scales rain riming by qr/(cloud riming) whenever cloud riming "
        "exceeds the available rain and never limits rain riming by itself.",
    ),
}

ANCHORS = {
    "D1-ice-rain-depletion-sum": (
        1104, "  !.. Do not let ice-rain collection overdeplete rainwater"),
}


def apply(text: str) -> str:
    lines = text.split("\n")
    for fix_id, (lineno, old, new, _why) in DECLARED_FIXES.items():
        a_line, a_text = ANCHORS[fix_id]
        if lines[a_line - 1].rstrip() != a_text.rstrip():
            raise SystemExit(f"{fix_id}: anchor line {a_line} does not match")
        if lines[lineno - 1].rstrip() != old.rstrip():
            raise SystemExit(f"{fix_id}: line {lineno} does not match the "
                             "pinned WRF text")
        lines[lineno - 1] = new
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    raw = args.source.read_bytes()
    print("pristine sha256", hashlib.sha256(raw).hexdigest())
    args.out.write_text(apply(raw.decode("ascii")), encoding="ascii")
    for fix_id, (_l, _o, _n, why) in DECLARED_FIXES.items():
        print(f"{fix_id}: {why}")


if __name__ == "__main__":
    main()
