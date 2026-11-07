"""Word-by-word comparison of two K-epsilon oracle fixture files.

    python tools/keps_wrf461_oracle/compare_sets.py A.bin B.bin [--outputs-only]

Each ``.bin`` sits beside its ``.manifest`` (written by run_bl_keps.F90).
Prints, per field present in both, the number of differing float32 words
and the largest ULP distance (NaN words compare by bit pattern).  Asserts
nothing: build.sh and the parity tests are the gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_manifest(path: Path) -> dict:
    fields = {}
    meta = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit():
            name, nlev, ncol, offset = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            fields[name] = (nlev, ncol, offset)
        elif parts:
            meta[parts[0]] = parts[1:]
    return {"fields": fields, "meta": meta}


def load(bin_path: Path) -> tuple[dict, dict]:
    manifest = read_manifest(bin_path.with_suffix(".manifest"))
    words = np.fromfile(bin_path, dtype="<f4")
    out = {}
    for name, (nlev, ncol, offset) in manifest["fields"].items():
        count = max(nlev, 1) * ncol
        block = words[offset:offset + count]
        out[name] = block.reshape(nlev, ncol) if nlev else block
    return out, manifest["meta"]


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def main(argv: list[str]) -> int:
    a, _ = load(Path(argv[1]))
    b, _ = load(Path(argv[2]))
    only_outputs = "--outputs-only" in argv
    total = 0
    for name in a:
        if name not in b:
            continue
        if only_outputs and "@" not in name:
            continue
        x, y = a[name], b[name]
        differ = x.view(np.uint32) != y.view(np.uint32)
        n = int(differ.sum())
        total += n
        if n:
            ulp = ulp_distance(x[differ], y[differ])
            print(f"{name:24s} {n:6d} words differ, max ULP {int(ulp.max())}")
    print(f"TOTAL differing words: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
