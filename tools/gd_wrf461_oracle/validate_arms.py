"""Cross-check the oracle arms build.sh produced, word for word.

    python3 tools/gd_wrf461_oracle/validate_arms.py BUILD_DIR

Required: stock-strict == stock-scalar == every tree-* arm, and
fixed-strict == fixed-scalar, on every output word of every fixture.
Reported (not required to be zero): where the fixed arm differs from the
stock arm -- the closure-count fix's footprint, which must be confined to
water columns (xland > 1.5) because the patch only touches the over-water
branch of cup_forcing_ens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gd_cases  # noqa: E402

FIELDS = (gd_cases.LEVEL_OUTPUTS + gd_cases.SURFACE_OUTPUTS
          + ("ktop_deep", "xf_ens", "pr_ens"))


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype.kind == "f" else np.int32)


def _diff(a_steps, b_steps):
    bad = {}
    for s, (a, b) in enumerate(zip(a_steps, b_steps)):
        if a["due"] != b["due"]:
            bad[f"step{s}:due"] = 1
        for name in FIELDS:
            n = int(np.count_nonzero(_bits(a[name]) != _bits(b[name])))
            if n:
                bad[f"step{s}:{name}"] = n
    return bad


def main(build: Path) -> int:
    failures = 0
    for name in ("columns", "edge", "cadence"):
        fx = gd_cases.build_fixture(name)
        for fqi in (1, 0):
            stem = f"gd-{name}-fqi{fqi}"
            outs = {p.name.split(".")[1]: gd_cases.read_output(
                p, fx.nx, fx.ny, fx.nz, fx.nsteps)
                for p in sorted((build / "out").glob(f"{stem}.*.out"))}
            ref = outs["stock-scalar"]
            for arm, steps in outs.items():
                if arm.startswith("fixed"):
                    continue
                bad = _diff(ref, steps)
                failures += bool(bad)
                print(f"{stem} {arm:>14} vs stock-scalar: "
                      f"{'IDENTICAL' if not bad else bad}")
            bad = _diff(outs["fixed-scalar"], outs["fixed-strict"])
            failures += bool(bad)
            print(f"{stem}   fixed-strict vs fixed-scalar: "
                  f"{'IDENTICAL' if not bad else bad}")
            last = ref[-1]
            fixed = outs["fixed-scalar"][-1]
            xland = fx.steps[0]["xland"]
            moved = np.zeros(xland.shape, bool)
            for f in gd_cases.LEVEL_OUTPUTS:
                moved |= np.any(_bits(last[f]) != _bits(fixed[f]), axis=1)
            moved |= _bits(last["pratec"]) != _bits(fixed["pratec"])
            land_moved = int(np.count_nonzero(moved & (xland < 1.5)))
            failures += land_moved > 0
            raining = int(np.count_nonzero(last["pratec"] > 0))
            print(f"{stem}   raining columns (stock) {raining}/{xland.size}; "
                  f"fixed-vs-stock moved columns: water "
                  f"{int(np.count_nonzero(moved & (xland > 1.5)))}, "
                  f"land {land_moved} (must be 0)")
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
