"""Outcome census of a KSAS oracle run: which exits the fixture reaches.

    python summarize.py CASES_DIR ORACLE_BIN

Prints, per call group, how many columns rained, how many changed the
column at all (a column can be modified and still return no rain: the
`rain <= 0` restore at module_cu_ksas.F:2220-2229 then puts it back), and
the cloud base/top levels seen; then the same counts per sounding category.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_cases  # noqa: E402


def census(cases_dir: Path, oracle_bin: Path) -> dict:
    groups = gen_cases.read(cases_dir / "cases.bin")
    outs = gen_cases.read_outputs(oracle_bin, groups)
    meta = json.loads((cases_dir / "cases.json").read_text())["groups"]
    rows = []
    per_cat = collections.defaultdict(collections.Counter)
    for g, o, m in zip(groups, outs, meta):
        r = o["raincv"]
        moved = np.abs(o["rthcuten"]).max(axis=0) > 0
        rows.append({
            "dx": float(g["dx"]), "dt": float(g["dt"]), "stepcu": g["stepcu"],
            "mp_physics": g["mp_physics"], "nz": g["nz"], "ncol": g["ncol"],
            "raining": int((r > 0).sum()), "moved": int(moved.sum()),
            "max_raincv_mm": float(r.max()),
            "hbot": sorted({int(v) for v in o["hbot"][r > 0]}),
            "htop": sorted({int(v) for v in o["htop"][r > 0]}),
        })
        for cat, rr, mv in zip(m["categories"], r, moved):
            per_cat[cat]["columns"] += 1
            per_cat[cat]["raining"] += int(rr > 0)
            per_cat[cat]["moved"] += int(mv)
    return {"groups": rows,
            "categories": {k: dict(v) for k, v in sorted(per_cat.items())}}


def main() -> None:
    result = census(Path(sys.argv[1]), Path(sys.argv[2]))
    for row in result["groups"]:
        print(f"dx {row['dx']:7.0f}  dt {row['dt']:5.1f}  stepcu {row['stepcu']}"
              f"  mp {row['mp_physics']:2d}  nz {row['nz']}:  raining "
              f"{row['raining']:2d}/{row['ncol']}  moved {row['moved']:2d}"
              f"  max RAINCV {row['max_raincv_mm']:.3f} mm"
              f"  HBOT {row['hbot']}  HTOP {row['htop']}")
    for cat, c in result["categories"].items():
        print(f"{cat:15s} columns {c['columns']:3d}  raining {c['raining']:3d}"
              f"  moved {c['moved']:3d}")


if __name__ == "__main__":
    main()
