"""Summaries of a UW shallow cumulus oracle build: which columns convect,
and where two builds of the same case differ (field, count, max ULP).

  python summarize.py BUILD_DIR
"""
import sys
from pathlib import Path
import numpy as np

DT = {"f8": np.float64, "f4": np.float32, "i4": np.int32}


def load(stem):
    raw = Path(str(stem) + ".bin").read_bytes()
    out = {}
    for line in Path(str(stem) + ".manifest").read_text().splitlines():
        name, dt, n, off = line.split()
        out[name] = np.frombuffer(raw, DT[dt], int(n), int(off))
    return out


def ulp(a, b):
    a = np.asarray(a); b = np.asarray(b)
    it = np.int64 if a.dtype == np.float64 else np.int32
    ia = a.view(it).astype(np.int64); ib = b.view(it).astype(np.int64)
    m = np.int64(0x7FFFFFFFFFFFFFFF) if a.dtype == np.float64 else np.int64(0x7FFFFFFF)
    ia = np.where(ia < 0, -(ia & m), ia)
    ib = np.where(ib < 0, -(ib & m), ib)
    return np.abs(ia - ib)


def main(build):
    build = Path(build)
    for man in sorted((build / "ref").glob("*.manifest")):
        stem = man.with_suffix("")
        ref = load(stem)
        if "const/ncol" in ref:
            ncol = int(ref["const/ncol"][0]); ns = int(ref["const/nsteps"][0])
            act = [int((ref[f"s{s}/cush"] != -1.0).sum()) for s in range(1, ns + 1)]
            umf = [int((ref[f"s{s}/cmfmc2"].reshape(ncol, -1) != 0).any(1).sum())
                   for s in range(1, ns + 1)]
            print(f"{stem.name}: ncol={ncol} steps={ns} cush!=-1 per step {act}"
                  f" columns with mass flux per step {umf}")
        else:
            ncol = int(ref["in/ncol"][0])
            act = int((ref["out/cush"] != -1.0).sum())
            umf = int((ref["out/umf"].reshape(ncol, -1) != 0).any(1).sum())
            print(f"{stem.name}: ncol={ncol} cush!=-1 {act} with mass flux {umf}")
        for v in ("strict", "o0", "snan"):
            other = load(build / v / stem.name)
            bad = []
            for k, a in ref.items():
                b = other[k]
                if a.dtype == np.int32:
                    d = int((a != b).sum()); mu = 0
                else:
                    u = ulp(a, b); d = int((u > 0).sum()); mu = int(u.max(initial=0))
                if d:
                    bad.append(f"{k}:{d}@{mu}")
            if bad:
                print(f"   {v}: " + " ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))


if __name__ == "__main__":
    main(sys.argv[1])


def same_values(stem_a, stem_b):
    """(value_mismatches, signed_zero_only) between two fixtures: equal
    values (+0 and -0 count as equal; NaN equals NaN) per field."""
    a, b = load(stem_a), load(stem_b)
    bad, sz = 0, 0
    for k, x in a.items():
        y = b[k]
        if x.dtype == np.int32:
            bad += int((x != y).sum())
            continue
        eq = (x == y) | (np.isnan(x) & np.isnan(y))
        bad += int((~eq).sum())
        sz += int((eq & (np.signbit(x) != np.signbit(y))).sum())
    return bad, sz
