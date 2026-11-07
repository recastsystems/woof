"""Count, per field, the words the stock WRF build and the fixed build differ.

    python3 compare_variants.py FIXTURE_DIR

The fixed build is WRF v4.6.1 with fixes.patch applied.  This report is the
measured size of the declared divergence: which output fields the two fixes
move, in how many columns, and by how much at most.  Inputs are compared too
(they must agree at step 1, and every later step's inputs follow from the
previous step's outputs through the harness update).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load(stem: Path) -> dict[str, np.ndarray]:
    raw = stem.with_suffix(".bin").read_bytes()
    out = {}
    for line in stem.with_suffix(".manifest").read_text().splitlines():
        name, dtype, count, offset = line.split()
        dt = {"f8": "<f8", "f4": "<f4", "i4": "<i4"}[dtype]
        out[name] = np.frombuffer(raw, dtype=dt, count=int(count),
                                  offset=int(offset)).copy()
    return out


def main(fixdir: str) -> int:
    d = Path(fixdir)
    total = 0
    for man in sorted(d.glob("zm-*.manifest")):
        if man.stem.endswith("-stock"):
            continue
        fixed = load(d / man.stem)
        stock = load(d / f"{man.stem}-stock")
        ncol = int(fixed["const/ncol"][0])
        rows = []
        for name, a in fixed.items():
            b = stock[name]
            if a.dtype.kind == "f":
                diff = a.view(np.uint32 if a.itemsize == 4 else np.uint64) != \
                    b.view(np.uint32 if b.itemsize == 4 else np.uint64)
            else:
                diff = a != b
            n = int(diff.sum())
            if n:
                cols = set(np.nonzero(diff)[0] % ncol) if a.size % ncol == 0 else set()
                with np.errstate(invalid="ignore"):
                    mx = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
                rows.append((name, n, len(cols), mx))
                total += n
        print(f"# {man.stem}: {len(rows)} arrays differ between stock and fixed")
        for name, n, nc, mx in rows:
            print(f"{man.stem} {name} words={n} columns={nc} max_abs={mx:.6g}")
    print(f"# total differing words: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
