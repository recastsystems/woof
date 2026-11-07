#!/usr/bin/env python3
"""Bitwise comparison of two Eta Ferrier column-oracle result sets.

usage: compare_etanew.py REF_DIR TEST_DIR [INPUTS_CSV] [--json OUT] [--subset]

Each directory holds ``levels.csv`` and ``surface.csv`` (and optionally
``init.csv``) in the format ``run_etanew.F90`` writes (floats as binary32
hex).  A port is graded by writing the same files from the same
``columns_inputs.csv`` and running this script against the Fortran
reference.

Prints, per output field: rows compared, rows that differ in any bit, max
ULP distance, max absolute and max relative difference, and the worst
column with its regime.  Exit status 0 only if every field is bitwise
equal.  ``--json`` also writes the per-field table.  ``--subset`` grades
only the rows REF_DIR holds (the committed fixtures keep steps 1, 2 and
10 of a ten-step run; a port writes every step).  ``tables.csv`` (the
SHA-256 of each saturation table) is compared line for line.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    if head[:2] == ["col", "name"]:
        nkey = 3
        fields = head[nkey:]
        data = {(int(r[0]), r[1], int(r[2])):
                np.array([int(x, 16) for x in r[nkey:]], dtype=np.uint32)
                for r in body}
        return fields, data
    nkey = 3 if head[2] == "k" else 2
    fields = head[nkey:]
    data = {tuple(int(x) for x in r[:nkey]):
            np.array([int(x, 16) for x in r[nkey:]], dtype=np.uint32)
            for r in body}
    return fields, data


def _ordered(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.int64)
    return np.where(b & 0x80000000, 0x80000000 - b, b)


def _regimes(inputs: Path | None) -> dict[int, str]:
    if inputs is None or not inputs.exists():
        return {}
    out = {}
    with open(inputs) as f:
        for line in f:
            if line.startswith("C,"):
                parts = line.split(",")
                out[int(parts[1])] = parts[2]
    return out


def compare(ref: Path, test: Path, inputs: Path | None = None,
            names=("levels.csv", "surface.csv", "init.csv"),
            subset: bool = False):
    regimes = _regimes(inputs)
    exact = True
    table = []
    for required in ("levels.csv", "surface.csv", "init.csv"):
        if not (ref / required).is_file() or not (test / required).is_file():
            print(f"{required}: required oracle output missing")
            return False, table

    if (ref / "tables.csv").exists() and not (test / "tables.csv").exists():
        print("tables.csv: required saturation table digests missing")
        return False, table
    if (ref / "tables.csv").exists() and (test / "tables.csv").exists():
        same = ((ref / "tables.csv").read_text()
                == (test / "tables.csv").read_text())
        exact &= same
        print(f"tables.csv   saturation tables {'equal' if same else 'DIFFER'}")
    for name in names:
        if not (ref / name).exists() or not (test / name).exists():
            if (ref / name).exists() != (test / name).exists():
                print(f"{name}: present in only one result set")
                if name != "init.csv":
                    exact = False
            continue
        fields, a = _load(ref / name)
        fields_b, b = _load(test / name)
        if subset and fields == fields_b and set(a) <= set(b):
            b = {key: b[key] for key in a}
        if fields != fields_b or a.keys() != b.keys():
            print(f"{name}: layouts differ")
            return False, table
        keys = sorted(a)
        if not keys:
            print(f"{name}: no output rows")
            return False, table
        A = np.stack([a[k] for k in keys])
        B = np.stack([b[k] for k in keys])
        fa = A.view(np.float32).astype(np.float64)
        fb = B.view(np.float32).astype(np.float64)
        ulp = np.abs(_ordered(A) - _ordered(B))
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(fa - fb) / np.maximum(np.abs(fa), np.abs(fb))
        rel = np.where(np.isfinite(rel), rel, 0.0)
        for j, fld in enumerate(fields):
            diff = A[:, j] != B[:, j]
            n = int(diff.sum())
            exact &= n == 0
            mu = int(ulp[:, j].max())
            mad = float(np.nanmax(np.abs(fa[:, j] - fb[:, j])))
            mrel = float(rel[:, j].max())
            worst = keys[int(ulp[:, j].argmax())]
            tag = regimes.get(worst[0], "?") if n else ""
            table.append({"file": name, "field": fld, "rows": len(keys),
                          "differ": n, "max_ulp": mu, "max_abs": mad,
                          "max_rel": mrel,
                          "worst": list(worst) if n else None,
                          "worst_regime": tag or None})
            print(f"{name:12s} {fld:16s} rows={len(keys):6d} differ={n:6d} "
                  f"max_ulp={mu:12d} max_abs={mad:.3e} max_rel={mrel:.3e}"
                  + (f" worst={worst} ({tag})" if n else ""))
    print("RESULT", "BITWISE EQUAL" if exact else "NOT EQUAL")
    return exact, table


def main(argv):
    out_json = None
    subset = "--subset" in argv
    argv = [a for a in argv if a != "--subset"]
    if "--json" in argv:
        i = argv.index("--json")
        out_json = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    inp = Path(argv[3]) if len(argv) == 4 else \
        Path(argv[1]).resolve().parent / "columns_inputs.csv"
    ok, table = compare(Path(argv[1]), Path(argv[2]), inp, subset=subset)
    if out_json is not None:
        out_json.write_text(json.dumps({"bitwise_equal": ok,
                                        "fields": table}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
