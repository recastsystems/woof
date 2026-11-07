"""Read ISHMAEL oracle/port step files and compare them word for word.

A step file (written by run_ishmael.F90, and by the CUDA harness in the
same layout) is::

    int32 magic, ncol, nz, nsteps; float32 dt
    per step: float32 (nz, ncol) x 35 in OUTPUT_FIELDS order,
              float32 (ncol,)   x 4  in SURFACE_FIELDS order

``compare`` reports, per field and step, the words that differ in their bit
pattern, how many of those are signed-zero-only, and the largest total-order
FP32 ULP distance (woof.core.fp32_ulp, the repository's one metric).
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

MAGIC_OUT = 1229146959
OUTPUT_FIELDS = (
    "th", "qv", "qc", "qr", "nr",
    "qi1", "ni1", "ai1", "ci1", "qi2", "ni2", "ai2", "ci2",
    "qi3", "ni3", "ai3", "ci3",
    "effc", "effi", "dbz",
    "vmi1", "di1", "rhopo1", "phii1", "vmi2", "di2", "rhopo2", "phii2",
    "vmi3", "di3", "rhopo3", "phii3",
    "itype1", "itype2", "itype3",
)
SURFACE_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv")


def read_steps(path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path}: truncated step header")
    magic, ncol, nz, nsteps = np.frombuffer(raw[:16], dtype="<i4")
    if magic != MAGIC_OUT:
        raise ValueError(f"{path}: not an ISHMAEL step file")
    dt = float(np.frombuffer(raw[16:20], dtype="<f4")[0])
    ncol, nz, nsteps = int(ncol), int(nz), int(nsteps)
    if min(ncol, nz, nsteps) < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{path}: invalid step dimensions or timestep")
    expected_bytes = 20 + 4 * nsteps * ncol * (nz * len(OUTPUT_FIELDS) + len(SURFACE_FIELDS))
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: step history has {len(raw)} bytes, expected {expected_bytes}")
    off = 20
    steps = []
    for _ in range(nsteps):
        step = {}
        for name in OUTPUT_FIELDS:
            step[name] = np.frombuffer(raw, dtype="<f4", count=nz * ncol,
                                       offset=off).reshape(nz, ncol)
            off += 4 * nz * ncol
        for name in SURFACE_FIELDS:
            step[name] = np.frombuffer(raw, dtype="<f4", count=ncol, offset=off)
            off += 4 * ncol
        steps.append(step)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return {"ncol": ncol, "nz": nz, "nsteps": nsteps, "dt": dt, "steps": steps}


def write_steps(path, steps, *, dt: float) -> None:
    """Write ``steps`` (a list of field dicts) in the oracle's layout."""
    nz, ncol = steps[0]["th"].shape
    with open(path, "wb") as fh:
        fh.write(np.array([MAGIC_OUT, ncol, nz, len(steps)], "<i4").tobytes())
        fh.write(np.array([dt], "<f4").tobytes())
        for step in steps:
            for name in OUTPUT_FIELDS + SURFACE_FIELDS:
                fh.write(np.ascontiguousarray(step[name], "<f4").tobytes())


def _ulp(got, want):
    try:
        from woof.core.fp32_ulp import fp32_ulp_distance
    except ImportError:                                  # standalone use
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from woof.core.fp32_ulp import fp32_ulp_distance
    return fp32_ulp_distance(got, want)


def compare(got: dict, want: dict) -> dict:
    """Per (step, field): differing words, signed-zero-only words, max ULP."""
    if (got["ncol"], got["nz"]) != (want["ncol"], want["nz"]):
        raise ValueError("step files describe different column batches")
    if got["nsteps"] != want["nsteps"]:
        raise ValueError("step files describe different step histories")
    if np.float32(got["dt"]).view(np.uint32) != np.float32(want["dt"]).view(np.uint32):
        raise ValueError("step files describe different timesteps (dt)")
    nsteps = got["nsteps"]
    if nsteps < 1 or len(got["steps"]) != nsteps or len(want["steps"]) != nsteps:
        raise ValueError("step history is empty or incomplete")
    rows = []
    hashes = []
    compared_words = 0
    nonfinite_words = 0
    for s in range(nsteps):
        for name in OUTPUT_FIELDS + SURFACE_FIELDS:
            a = np.asarray(got["steps"][s][name], np.float32)
            b = np.asarray(want["steps"][s][name], np.float32)
            if a.shape != b.shape:
                raise ValueError(f"step {s+1} field {name}: mismatched array shapes")
            expected = ((got["nz"], got["ncol"]) if name in OUTPUT_FIELDS
                        else (got["ncol"],))
            if a.shape != expected:
                raise ValueError(f"step {s+1} field {name}: invalid geometry {a.shape}")
            compared_words += a.size
            nonfinite_words += int((~np.isfinite(a) | ~np.isfinite(b)).sum())
            hashes.append({"step": s+1, "field": name,
                           "got_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                           "want_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                           "got_nonfinite": int((~np.isfinite(a)).sum()),
                           "want_nonfinite": int((~np.isfinite(b)).sum())})
            bits_a, bits_b = a.view(np.uint32), b.view(np.uint32)
            diff = bits_a != bits_b
            ndiff = int(diff.sum())
            if not ndiff:
                continue
            zero_only = int((diff & (a == 0) & (b == 0)).sum())
            ulp = _ulp(a, b)
            cols = np.unique(np.nonzero(diff)[-1])
            probes = []
            for index in np.flatnonzero(diff)[:16]:
                idx = np.unravel_index(index, a.shape)
                probes.append({"index": [int(v) for v in idx],
                               "got_word": int(bits_a[idx]), "want_word": int(bits_b[idx]),
                               "got": float(a[idx]), "want": float(b[idx])})
            rows.append({"step": s + 1, "field": name, "words": ndiff,
                         "signed_zero_only": zero_only,
                         "max_ulp": int(np.max(ulp)),
                         "columns": [int(c) for c in cols[:12]], "probes": probes})
    differing_words = int(sum(r["words"] for r in rows))
    return {"nsteps": nsteps, "ncol": got["ncol"], "nz": got["nz"],
            "dt": got["dt"], "field_hashes": hashes, "differences": rows,
            "total_words": differing_words, "compared_words": compared_words,
            "nonfinite_words": nonfinite_words,
            "passed": differing_words == 0 and nonfinite_words == 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("got", type=Path)
    ap.add_argument("want", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    report = compare(read_steps(args.got), read_steps(args.want))
    for row in report["differences"][:200]:
        print(f"step {row['step']:3d} {row['field']:8s} words {row['words']:6d} "
              f"(+-0 only {row['signed_zero_only']}) max_ulp {row['max_ulp']} "
              f"cols {row['columns']}")
    print(f"total differing words: {report['total_words']} over "
          f"{report['nsteps']} steps")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
