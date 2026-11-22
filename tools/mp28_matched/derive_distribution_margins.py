"""Derive the third gate's margins from the CONTROL's committed rows only.

``docs/public/wrf-comparison/mp28-distribution-gate.md`` states its two
amplification bounds relative to the WRF-against-itself control measured on
the same run set, with fixed margins M (median) and P (p95).  Those margins
must be justified by measurement noise and by nothing else -- in
particular, by no number WOOF-ArWen ever produced.  This script is that
justification, runnable by anyone:

* it reads ONLY the control arm (``G1_control``) of the committed
  second-gate receipt -- WRF build B against WRF build A, two builds of
  unmodified source one optimization flag apart;
* it field-block bootstraps those 140 rows (resample the 14 fields with
  replacement, keeping each field's 10 step-rows together, because steps
  within a field are serially correlated and rows are not iid);
* it forms the ratio of two independent bootstrap replicates of each
  statistic -- the spread two equally benign arms would show against each
  other if both were draws from the control's own process -- and reports
  the 99th percentile of that ratio.

The declared margins are these p99 values rounded UP to one decimal.  The
seed is fixed so the derivation is a computation, not a sample.

Usage:
    python derive_distribution_margins.py [--receipt PATH] [--seed N]
                                          [--replicates N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = (REPO_ROOT / "docs" / "public" / "receipts"
                   / "mp28-shortwindow-gate" / "shortwindow-gate.json")

#: Fixed in the declaration.  Changing either changes the derivation and
#: therefore the document that quotes it.
SEED = 20260802
REPLICATES = 20000


def order_statistics(values: np.ndarray) -> tuple[float, float]:
    """(median, p95) exactly as the declaration defines them.

    Ascending sort; median is the mean of the two middle order statistics
    for even n (the middle one for odd n); p95 is the nearest-rank order
    statistic at rank ceil(0.95 n), 1-based.  +inf participates and sorts
    above every finite value.
    """
    ordered = np.sort(values)
    n = ordered.size
    if n % 2 == 0:
        median = 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    else:
        median = ordered[n // 2]
    p95 = ordered[int(np.ceil(0.95 * n)) - 1]
    return float(median), float(p95)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    rows = receipt["G1_control"]["rows"]
    fields = sorted({row["field"] for row in rows})
    by_field = {field: np.array([row["ratio"] for row in rows
                                 if row["field"] == field])
                for field in fields}

    all_ratios = np.array([row["ratio"] for row in rows])
    point_median, point_p95 = order_statistics(all_ratios)
    print(f"control rows: {all_ratios.size} over {len(fields)} fields")
    print(f"control point statistics: median {point_median:.4f}, "
          f"p95 {point_p95:.4f}")

    rng = np.random.default_rng(args.seed)
    medians = np.empty(args.replicates)
    p95s = np.empty(args.replicates)
    for i in range(args.replicates):
        pick = rng.choice(fields, size=len(fields), replace=True)
        sample = np.concatenate([by_field[field] for field in pick])
        medians[i], p95s[i] = order_statistics(sample)

    pairs_a = rng.integers(0, args.replicates, args.replicates)
    pairs_b = rng.integers(0, args.replicates, args.replicates)
    median_ratio = medians[pairs_a] / medians[pairs_b]
    p95_ratio = p95s[pairs_a] / p95s[pairs_b]

    print(f"bootstrap ({args.replicates} field-block replicates, "
          f"seed {args.seed}):")
    print(f"  arm/arm median-statistic ratio: "
          f"p95 {np.percentile(median_ratio, 95):.3f}, "
          f"p99 {np.percentile(median_ratio, 99):.3f}")
    print(f"  arm/arm p95-statistic ratio:    "
          f"p95 {np.percentile(p95_ratio, 95):.3f}, "
          f"p99 {np.percentile(p95_ratio, 99):.3f}")
    print("declared margins are the p99 values rounded up to one decimal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
