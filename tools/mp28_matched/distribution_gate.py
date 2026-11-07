"""Evaluate the pre-registered distribution gate for mp_physics = 28.

THIS tool implements the gate declared in
``docs/public/wrf-comparison/mp28-distribution-gate.md`` before any of the
runs it reads existed, exactly as ``shortwindow_gate.py`` implemented the
second gate's declaration.  Nothing here selects a statistic, a margin or
a screen; they were fixed in that document's design commit (`0d69a648`)
and owner-approved before any build started.  Any change to the constants
below without a matching change to that document above its MEASUREMENTS
line (which is forbidden after the fact) is a defect.

The amplification condition is relative to the control on the same run
set, on two order statistics of the ratio rows:

    D1   median_tested <= 1.7 x median_control
    D2   p95_tested    <= 2.1 x p95_control

with the margins derived from the second gate's committed control rows
alone by ``derive_distribution_margins.py`` (seed 20260802, p99 of the
arm/arm bootstrap ratio, rounded up to one decimal).

Usage:
    python distribution_gate.py --runs DIR --node2-runs DIR --out FILE
                                [--dt 60]

Expected layout under --runs: the second gate's, unchanged (banked
scripts produce it).  --node2-runs needs only the WRF quartet
(sw-wrf-mp{08,28}, sw-wrf-novec-mp{08,28}); it feeds the cross-node
replication screen and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shortwindow import dual_run_screen, wrf_frames          # noqa: E402
from shortwindow_gate import (                                # noqa: E402
    GATE_FIELDS, G0_TOL, MP28_ONLY, N_STEPS, NIFA_CEIL, NIFA_FLOOR,
    NWFA_CEIL, NWFA_FLOOR, PUBLISHED_ONLY, arwen_frame, divergence_rows,
)

DECLARED_IN = "docs/public/wrf-comparison/mp28-distribution-gate.md"
DESIGN_COMMIT = "0d69a648"

#: D1/D2 margins, transcribed from the declaration's section 3.
M_MEDIAN = 1.7
P_P95 = 2.1

#: Cross-node replication screen, declaration section 4: test-machine-2's control
#: arm statistics must sit within 2 % relative of test-machine-1's, or the gate is
#: VOID (the noise floor exceeds what the margin derivation assumed).
CROSS_NODE_REL = 0.02

#: 14 gate fields x steps 1..10.  A shortfall in cells PRESENT (before the
#: declared both-zero drop) voids the gate.
EXPECTED_CELLS = len(GATE_FIELDS) * N_STEPS


def order_statistics(values: np.ndarray) -> tuple[float, float]:
    """(median, p95) exactly as the declaration defines them.

    Identical to ``derive_distribution_margins.order_statistics`` -- the
    declaration names that function as the reference implementation.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n % 2 == 0:
        median = 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    else:
        median = ordered[n // 2]
    p95 = ordered[int(np.ceil(0.95 * n)) - 1]
    return float(median), float(p95)


def ratio_rows(rows8: list, rows28: list, *, steps) -> dict:
    """The declared ratio row set for one arm.

    The prior gates' convention, by reference: a both-zero row drops out;
    ``d_mp08 = 0`` with ``d_mp28 > 0`` is +inf and PARTICIPATES in the
    order statistics.  ``cells_present`` counts every (field, step) pair
    that carried both divergences before the zero-drop; the declaration
    voids the gate if it is short of 140.
    """
    by8 = {r["step"]: r for r in rows8}
    by28 = {r["step"]: r for r in rows28}
    rows, dropped_both_zero, cells_present = [], 0, 0
    for s in steps:
        a, b = by8.get(s), by28.get(s)
        if a is None or b is None:
            continue
        for f in GATE_FIELDS:
            d8, d28 = a.get(f), b.get(f)
            if d8 is None or d28 is None:
                continue
            cells_present += 1
            if d8 == 0.0 and d28 == 0.0:
                dropped_both_zero += 1
                continue
            ratio = float("inf") if d8 == 0.0 else d28 / d8
            rows.append({"field": f, "step": s, "d_mp08": d8,
                         "d_mp28": d28, "ratio": ratio})
    ratios = np.array([r["ratio"] for r in rows], dtype=np.float64)
    median, p95 = (order_statistics(ratios) if ratios.size
                   else (float("nan"), float("nan")))
    finite = ratios[np.isfinite(ratios)]
    return {
        "rows": rows,
        "cells_present": cells_present,
        "dropped_both_zero": dropped_both_zero,
        "n_rows": len(rows),
        "n_infinite": int(np.sum(~np.isfinite(ratios))),
        "median": median,
        "p95": p95,
        "max_finite": (float(finite.max()) if finite.size else None),
    }


def wrf_divergences(runs: Path, tested_dir: str, reference_dir: str,
                    mp: int, dt: float) -> list:
    """M8 rows for a WRF-vs-WRF pair (the control arm, either node)."""
    m = f"{mp:02d}"
    fields = GATE_FIELDS + PUBLISHED_ONLY + (MP28_ONLY if mp == 28 else ())
    ref = sorted((runs / f"{reference_dir}-mp{m}").glob("wrfout_d01_*"))
    tst = sorted((runs / f"{tested_dir}-mp{m}").glob("wrfout_d01_*"))
    if not ref or not tst:
        return []
    ref_rows = list(wrf_frames(ref[0], fields))
    tst_rows = list(wrf_frames(tst[0], fields))
    return divergence_rows(
        lambda i: tst_rows[i] if i < len(tst_rows) else None,
        ref_rows, fields, dt)


def arwen_divergences(runs: Path, mp: int, dt: float) -> list:
    m = f"{mp:02d}"
    fields = GATE_FIELDS + PUBLISHED_ONLY + (MP28_ONLY if mp == 28 else ())
    ref = sorted((runs / f"sw-wrf-mp{m}").glob("wrfout_d01_*"))
    if not ref:
        return []
    ref_rows = list(wrf_frames(ref[0], fields))
    run = runs / f"sw-arwen-mp{m}"
    return divergence_rows(lambda i: arwen_frame(run, i, dt),
                           ref_rows, fields, dt)


def control_statistics(runs: Path, dt: float, steps) -> dict:
    rows8 = wrf_divergences(runs, "sw-wrf-novec", "sw-wrf", 8, dt)
    rows28 = wrf_divergences(runs, "sw-wrf-novec", "sw-wrf", 28, dt)
    return ratio_rows(rows8, rows28, steps=steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path,
                    help="test-machine-1's run tree plus the local WOOF runs")
    ap.add_argument("--node2-runs", required=True, type=Path,
                    help="test-machine-2's independently built WRF quartet")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dt", type=float, default=60.0)
    args = ap.parse_args()
    R = args.runs
    steps = range(1, N_STEPS + 1)

    report = {
        "declared_in": DECLARED_IN,
        "design_commit": DESIGN_COMMIT,
        "margins": {"median": M_MEDIAN, "p95": P_P95,
                    "source": "tools/mp28_matched/"
                              "derive_distribution_margins.py seed 20260802"},
        "cross_node_rel": CROSS_NODE_REL,
        "expected_cells_per_arm": EXPECTED_CELLS,
        "gate_fields": list(GATE_FIELDS),
        "start_s": 1800.0, "frame_dt_s": args.dt,
    }

    # --- G2: dual-run byte screen ------------------------------------------
    report["dual_run"] = {f"mp{mp:02d}": dual_run_screen(R, mp)
                          for mp in (8, 28)}
    g2 = all(v.get("status") == "ok" and v.get("identical")
             for v in report["dual_run"].values())

    # --- divergences, all four row sets ------------------------------------
    tested8 = arwen_divergences(R, 8, args.dt)
    tested28 = arwen_divergences(R, 28, args.dt)
    report["tested_frames"] = {"08": tested8, "28": tested28}

    # --- G0: installation screen at step 0, tested slot --------------------
    g0_rows, g0_ok = [], True
    for m, rows in (("08", tested8), ("28", tested28)):
        row0 = next((r for r in rows if r["step"] == 0), None)
        if row0 is None:
            g0_ok = False
            g0_rows.append({"scheme": f"mp{m}", "missing": True})
            continue
        for f in GATE_FIELDS:
            d = row0.get(f)
            if d is None:
                continue
            ok = d <= G0_TOL
            g0_ok &= ok
            if d != 0.0 or not ok:
                g0_rows.append({"scheme": f"mp{m}", "field": f, "d": d,
                                "pass": bool(ok)})
    report["G0"] = {"tol": G0_TOL, "nonzero_or_failing": g0_rows,
                    "pass": bool(g0_ok)}

    # --- G3: finite everywhere; aerosol inside the clamp band --------------
    nonfinite, bound_viol = [], []
    for mp in (8, 28):
        run = R / f"sw-arwen-mp{mp:02d}"
        for p in sorted(run.glob("frame_t*.npz")):
            z = np.load(p)
            for f in z.files:
                if not np.isfinite(z[f]).all():
                    nonfinite.append({"run": run.name, "frame": p.name,
                                      "field": f})
            if mp == 28:
                for f, lo, hi in (("QNWFA", NWFA_FLOOR, NWFA_CEIL),
                                  ("QNIFA", NIFA_FLOOR, NIFA_CEIL)):
                    lo_v, hi_v = float(z[f].min()), float(z[f].max())
                    if lo_v < lo or hi_v > hi:
                        bound_viol.append({"frame": p.name, "field": f,
                                           "min": lo_v, "max": hi_v})
    report["G3"] = {"nonfinite": nonfinite, "bound_violations": bound_viol,
                    "pass": bool(not nonfinite and not bound_viol)}

    # --- the two arms -------------------------------------------------------
    tested = ratio_rows(tested8, tested28, steps=steps)
    control = control_statistics(R, args.dt, steps)
    node2_control = control_statistics(args.node2_runs, args.dt, steps)
    report["tested_arm"] = tested
    report["control_arm"] = control
    report["node2_control_arm"] = node2_control

    # --- control validity (declaration section 3) ---------------------------
    control_valid = (math.isfinite(control["median"])
                     and control["median"] > 0.0
                     and math.isfinite(control["p95"]))
    report["control_valid"] = bool(control_valid)

    # --- cross-node replication screen (section 4) ---------------------------
    if (node2_control["n_rows"] and control["n_rows"]
            and control_valid and math.isfinite(node2_control["median"])
            and node2_control["median"] > 0.0
            and math.isfinite(node2_control["p95"])):
        med_rel = abs(node2_control["median"] / control["median"] - 1.0)
        p95_rel = abs(node2_control["p95"] / control["p95"] - 1.0)
        cross_node = {"median_rel": med_rel, "p95_rel": p95_rel,
                      "bound": CROSS_NODE_REL,
                      "pass": bool(med_rel <= CROSS_NODE_REL
                                   and p95_rel <= CROSS_NODE_REL)}
    else:
        cross_node = {"median_rel": None, "p95_rel": None,
                      "bound": CROSS_NODE_REL, "pass": False,
                      "reason": "test-machine-2 control arm missing or degenerate"}
    report["cross_node"] = cross_node

    # --- D1 / D2 -------------------------------------------------------------
    d1 = bool(control_valid
              and tested["median"] <= M_MEDIAN * control["median"])
    d2 = bool(control_valid and tested["p95"] <= P_P95 * control["p95"])
    report["D1"] = {"tested_median": tested["median"],
                    "control_median": control["median"],
                    "bound": (M_MEDIAN * control["median"]
                              if control_valid else None),
                    "pass": d1}
    report["D2"] = {"tested_p95": tested["p95"],
                    "control_p95": control["p95"],
                    "bound": (P_P95 * control["p95"]
                              if control_valid else None),
                    "pass": d2}

    # --- verdict, exactly as declared (section 7) ----------------------------
    void_reasons = []
    if not report["G0"]["pass"]:
        void_reasons.append("G0")
    if not g2:
        void_reasons.append("G2")
    if tested["cells_present"] != EXPECTED_CELLS:
        void_reasons.append(
            f"tested row-set shortfall {tested['cells_present']}"
            f"/{EXPECTED_CELLS}")
    if control["cells_present"] != EXPECTED_CELLS:
        void_reasons.append(
            f"control row-set shortfall {control['cells_present']}"
            f"/{EXPECTED_CELLS}")
    if not control_valid:
        void_reasons.append("control degenerate")
    if not cross_node["pass"]:
        void_reasons.append("cross-node replication")

    if void_reasons:
        outcome = "void"
    elif d1 and d2 and report["G3"]["pass"]:
        outcome = "certify"
    else:
        outcome = "hold"
    report["verdict"] = {
        "G0": report["G0"]["pass"], "G2": bool(g2),
        "G3": report["G3"]["pass"],
        "control_valid": bool(control_valid),
        "cross_node": cross_node["pass"],
        "D1": d1, "D2": d2,
        "void_reasons": void_reasons,
        "outcome": outcome,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({
        "outcome": outcome,
        "tested": {"median": tested["median"], "p95": tested["p95"],
                   "n": tested["n_rows"], "inf": tested["n_infinite"]},
        "control": {"median": control["median"], "p95": control["p95"],
                    "n": control["n_rows"], "inf": control["n_infinite"]},
        "node2_control": {"median": node2_control["median"],
                          "p95": node2_control["p95"]},
        "D1": report["D1"], "D2": report["D2"],
        "cross_node": cross_node,
        "dual_run_identical": {k: v.get("identical")
                               for k, v in report["dual_run"].items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
