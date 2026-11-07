"""Grade a kernel run against the WRF oracle, field by field, in ULP.

    python3 validate_kernel.py ORACLE.out KERNEL.out FIXTURE [--quiet]

Both files are run_gd.F90's layout (gd_host_harness.cpp and the CUDA test
write the same one).  Prints, per field, the count of differing words and
the worst ULP distance; exits 1 if any word differs.  +0 and -0 are the same
value and count as 0 ULP; nothing else is forgiven.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_cases  # noqa: E402

FIELDS = (gd_cases.LEVEL_OUTPUTS + gd_cases.SURFACE_OUTPUTS
          + ("ktop_deep", "xf_ens", "pr_ens"))


def ordered(a: np.ndarray) -> np.ndarray:
    """float32 -> int64 on a line where adjacent floats differ by 1."""
    i = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype.kind != "f":
        return np.abs(a.astype(np.int64) - b.astype(np.int64))
    d = np.abs(ordered(a) - ordered(b))
    nan = np.isnan(a) | np.isnan(b)
    return np.where(nan & ~(np.isnan(a) & np.isnan(b)), np.int64(2**40), d)


def compare(oracle_steps, kernel_steps):
    """{field: (differing words, max ulp)} over all steps."""
    if len(oracle_steps) != len(kernel_steps):
        raise ValueError("oracle and kernel step counts differ")
    out = {}
    for a, b in zip(oracle_steps, kernel_steps):
        if a["due"] != b["due"]:
            out["due"] = (1, 1)
        for name in FIELDS:
            if a[name].shape != b[name].shape:
                raise ValueError(f"{name}: oracle and kernel shapes differ")
            d = ulp(a[name], b[name])
            n, m = out.get(name, (0, 0))
            out[name] = (n + int(np.count_nonzero(d)), max(m, int(d.max(initial=0))))
    return out


def main(argv) -> int:
    oracle, kernel, fixture = Path(argv[0]), Path(argv[1]), argv[2]
    fx = gd_cases.build_fixture(fixture)
    a = gd_cases.read_output(oracle, fx.nx, fx.ny, fx.nz, fx.nsteps)
    b = gd_cases.read_output(kernel, fx.nx, fx.ny, fx.nz, fx.nsteps)
    res = compare(a, b)
    bad = {k: v for k, v in res.items() if v[0]}
    if "--quiet" not in argv:
        for k, (n, m) in res.items():
            print(f"  {k:>10}: {n:6d} words differ, max {m} ULP")
    print(f"{kernel.name}: {'0 ULP on every word' if not bad else 'MISMATCH ' + str(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
