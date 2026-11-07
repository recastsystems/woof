"""Print the QNSE surface layer's distance from its WRF v4.6.1 oracle.

    python tools/qnsesfc_wrf461_oracle/compare_qnsesfc.py [--fixture NPZ]
        [--tables-only] [--scheme 4|2] [--json OUT.json]

Grades the host tables (CPU) and the CUDA kernel (GPU) word for word, every
field, replaying WRF's pre-call state and free-running.  Exits 0 only when
every word matches.  ``--scheme 2`` runs the Eta (MYJ) kernel on QNSE's
columns instead: the measurement that the grade tells the schemes apart.
Run it with GPUWM_WRF_EXACT=1 for the strict build and without it for the
default build (woof's import installs the strict compiler hook when the
variable is set).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from woof.verify import qnsesfc_oracle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=str(qnsesfc_oracle.FIXTURE))
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--scheme", type=int, default=4, choices=(2, 4))
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    fixture = qnsesfc_oracle.load(args.fixture)
    tables = qnsesfc_oracle.compare_tables(fixture)
    report = (None if args.tables_only
              else qnsesfc_oracle.measure(fixture,
                                          sf_sfclay_physics=args.scheme))
    mode = ("strict GPUWM_WRF_EXACT=1"
            if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default")
    header = (f"QNSE surface layer vs WRF v4.6.1 QNSESFC ({mode}; "
              f"kernel row sf_sfclay_physics={args.scheme}; "
              f"fixture {args.fixture})")
    print(qnsesfc_oracle.format_report(tables, report, header), end="")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"mode": mode, "scheme": args.scheme, "tables": tables,
                       "report": report}, out, indent=1, sort_keys=True)
    bad_tables = any(stats["mismatch"] for stats in tables.values())
    bad_fields = report is not None and any(
        qnsesfc_oracle.failing_fields(report).values())
    return 1 if (bad_tables or bad_fields) else 0


if __name__ == "__main__":
    sys.exit(main())
