"""Compare two TEMF oracle output files field by field.

    python3 compare_arms.py <cases.bin> <ref.bin> <other.bin> [--step N]

Prints, per field, the mismatched words, the maximum ULP distance and the
first mismatch (step, level, column), and the first step at which any
field differs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.verify.temf_oracle import (ALL_FIELDS, compare, read_cases,  # noqa: E402
                                      read_outputs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("ref")
    ap.add_argument("other")
    ap.add_argument("--step", type=int, default=None,
                    help="compare one step only (0-based)")
    args = ap.parse_args()
    cases = read_cases(args.cases)
    ref = read_outputs(args.ref, cases)
    got = read_outputs(args.other, cases)
    if args.step is not None:
        ref = {k: v[args.step:args.step + 1] for k, v in ref.items()}
        got = {k: v[args.step:args.step + 1] for k, v in got.items()}
    report = compare(ref, got, ALL_FIELDS)
    total = sum(r["mismatched"] for r in report.values())
    for name, r in report.items():
        if r["mismatched"]:
            print(f"{name:12s} {r['mismatched']:8d}/{r['words']:8d} "
                  f"max {r['max_ulp']:>11d} ULP  first {r['first']}")
    first_step = None
    for s in range(next(iter(ref.values())).shape[0]):
        one = compare({k: v[s] for k, v in ref.items()},
                      {k: v[s] for k, v in got.items()}, ALL_FIELDS)
        if any(r["mismatched"] for r in one.values()):
            first_step = s
            break
    print(f"TOTAL mismatched words {total}; first differing step "
          f"{first_step}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
