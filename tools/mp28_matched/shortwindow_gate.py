"""Evaluate the pre-registered short-window gate for mp_physics = 28.

The successor of ``shortwindow.py``, which was a post-hoc supplement and
says so in its own receipt.  THIS tool implements the gate declared in
``docs/public/wrf-comparison/mp28-shortwindow-gate.md`` BEFORE any of the runs
it reads existed.  Nothing here selects a statistic or a threshold; they
were fixed in that document's design commit, and any change to the
constants below without a matching change to that document (above its
MEASUREMENTS line, which is forbidden after the fact) is a defect.

Differences from the post-hoc pass, all declared in advance:

* ``U``, ``V``, ``PH``, ``MU`` are in the gate set: the frame writer dumps
  them now, closing the instrumentation gap the first pass named;
* ``RAINNC`` is published but excluded from the gate in BOTH slots (the
  tested model's accumulator restarts at zero mid-storm, so the statistic
  compares two different quantities);
* a row with ``d_mp08 = 0`` and ``d_mp28 > 0`` FAILS as an infinite ratio
  instead of being silently skipped;
* the WRF-against-itself control runs over the SAME window and its G1
  verdict is binding: a control failure makes the outcome INCONCLUSIVE.

Usage:
    python shortwindow_gate.py --runs DIR --out FILE [--dt 60]

Expected layout under --runs (names declared in the design):
    sw-wrf-mp08/wrfout_d01_*          build A, continuous 0->2400 s
    sw-wrf-mp28/wrfout_d01_*
    sw-wrf-novec-mp08/wrfout_d01_*    build B, the control
    sw-wrf-novec-mp28/wrfout_d01_*
    sw-arwen-mp08/frame_t*.npz        WOOF-ArWen, from build A's 1800 s frame
    sw-arwen-mp08-b/                  its duplicate (dual-run screen)
    sw-arwen-mp28/  sw-arwen-mp28-b/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shortwindow import dual_run_screen, rms_diff, wrf_frames  # noqa: E402

#: The 14-field gate comparison set, exactly as declared.
GATE_FIELDS = ("U", "V", "W", "PH", "MU", "T",
               "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
               "QNRAIN", "QNICE")

#: Published rows the verdict never reads.
PUBLISHED_ONLY = ("RAINNC",)
MP28_ONLY = ("QNCLOUD", "QNWFA", "QNIFA")

#: G0: both sides of step 0 are the same file read back.
G0_TOL = 1.0e-8

#: G1: the first document's V3 constant, in the regime its control
#: showed it can discriminate in.
G1_RATIO = 3.0

#: G3: WRF's terminal clamp band, the constants the first pass's V4 used.
NWFA_FLOOR, NWFA_CEIL = 11.1e6, 9999.0e6
NIFA_FLOOR, NIFA_CEIL = 5.0e3, 9999.0e6

#: 50 steps x 12 s at one frame per 5 steps -> 11 frames, steps 0..10.
N_STEPS = 10


def arwen_frame(run: Path, step: int, frame_dt: float):
    p = run / f"frame_t{int(round(step * frame_dt))}.npz"
    return np.load(p) if p.exists() else None


def divergence_rows(tested_frames, wrf_rows, fields, frame_dt: float):
    """One row per compared frame: {field: ||A - W||2 / ||W||2}."""
    rows = []
    for i, wf in enumerate(wrf_rows):
        A = tested_frames(i)
        if A is None:
            continue
        row = {"step": i, "time_s": 1800.0 + i * frame_dt}
        for f in fields:
            a = A[f] if f in getattr(A, "files", A) else None
            w = wf.get(f)
            if a is not None and w is not None:
                row[f] = rms_diff(a, w)
        rows.append(row)
    return rows


def g1_rows(rows8, rows28, *, steps):
    """The declared amplification condition over the gate set."""
    out, ok_all = [], True
    by8 = {r["step"]: r for r in rows8}
    by28 = {r["step"]: r for r in rows28}
    for s in steps:
        a, b = by8.get(s), by28.get(s)
        if a is None or b is None:
            continue
        for f in GATE_FIELDS:
            d8, d28 = a.get(f), b.get(f)
            if d8 is None or d28 is None:
                continue
            if d8 == 0.0 and d28 == 0.0:
                continue                    # no information either way
            if d8 == 0.0:
                ratio, ok = float("inf"), False   # declared: this FAILS
            else:
                ratio = d28 / d8
                ok = ratio <= G1_RATIO
            ok_all &= ok
            out.append({"field": f, "step": s, "d_mp08": d8, "d_mp28": d28,
                        "ratio": ratio, "pass": bool(ok)})
    finite = [r["ratio"] for r in out if np.isfinite(r["ratio"])]
    return {"rows": out, "pass": bool(ok_all),
            "worst": (max(finite) if finite else None),
            "n_rows": len(out),
            "n_fail": sum(1 for r in out if not r["pass"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dt", type=float, default=60.0)
    args = ap.parse_args()
    R = args.runs

    report = {
        "declared_in": "docs/public/wrf-comparison/mp28-shortwindow-gate.md",
        "start_s": 1800.0, "frame_dt_s": args.dt,
        "gate_fields": list(GATE_FIELDS),
        "published_only_fields": list(PUBLISHED_ONLY + MP28_ONLY),
        "g1_ratio": G1_RATIO, "g0_tol": G0_TOL,
    }

    # --- G2: dual-run byte screen (shared machinery with the first pass) --
    report["dual_run"] = {f"mp{mp:02d}": dual_run_screen(R, mp)
                          for mp in (8, 28)}
    g2 = all(v.get("status") == "ok" and v.get("identical")
             for v in report["dual_run"].values())

    # --- divergences, tested (ArWen) and control (build B) slots ----------
    slots = {}
    for slot, mk_tested in (
        ("tested", lambda mp:
            (lambda i, run=R / f"sw-arwen-mp{mp:02d}":
                arwen_frame(run, i, args.dt))),
        ("control", lambda mp:
            (lambda i, frames=[None]:
                _control_frame(R, mp, i, frames))),
    ):
        slots[slot] = {}
        for mp in (8, 28):
            m = f"{mp:02d}"
            wout = sorted((R / f"sw-wrf-mp{m}").glob("wrfout_d01_*"))
            if not wout:
                slots[slot][m] = {"status": "missing-reference"}
                continue
            fields = GATE_FIELDS + PUBLISHED_ONLY + (
                MP28_ONLY if mp == 28 else ())
            wrf_rows = list(wrf_frames(wout[0], fields))
            rows = divergence_rows(mk_tested(mp), wrf_rows, fields, args.dt)
            slots[slot][m] = {"status": "ok", "frames": rows}
    report["tested"] = slots["tested"]
    report["control"] = slots["control"]

    # --- G0: installation screen at step 0, tested slot only --------------
    g0_rows, g0_ok = [], True
    for m in ("08", "28"):
        frames = report["tested"].get(m, {}).get("frames") or []
        row0 = next((r for r in frames if r["step"] == 0), None)
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

    # --- G1, both slots ----------------------------------------------------
    steps = range(1, N_STEPS + 1)
    report["G1"] = g1_rows(
        report["tested"].get("08", {}).get("frames") or [],
        report["tested"].get("28", {}).get("frames") or [], steps=steps)
    report["G1_control"] = g1_rows(
        report["control"].get("08", {}).get("frames") or [],
        report["control"].get("28", {}).get("frames") or [], steps=steps)

    # --- G3: finite everywhere; aerosol inside the clamp band -------------
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

    # --- verdict, exactly as declared --------------------------------------
    control_diagnostic = report["G1_control"]["pass"]
    if not g2 or report["G1_control"]["n_rows"] == 0 \
            or report["G1"]["n_rows"] == 0:
        outcome = "inconclusive"
    elif not control_diagnostic:
        outcome = "inconclusive"
    elif report["G0"]["pass"] and report["G1"]["pass"] \
            and report["G3"]["pass"]:
        outcome = "pass"
    else:
        outcome = "fail"
    report["verdict"] = {
        "G0": report["G0"]["pass"], "G1": report["G1"]["pass"],
        "G2": bool(g2), "G3": report["G3"]["pass"],
        "control_diagnostic": bool(control_diagnostic),
        "outcome": outcome,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"outcome": outcome,
                      "G1_worst": report["G1"]["worst"],
                      "G1_fail_rows": report["G1"]["n_fail"],
                      "G1_control_worst": report["G1_control"]["worst"],
                      "G1_control_fail_rows": report["G1_control"]["n_fail"],
                      "dual_run_identical": {
                          k: v.get("identical")
                          for k, v in report["dual_run"].items()}},
                     indent=1))
    return 0


def _control_frame(R: Path, mp: int, i: int, cache: list):
    """Build B's i-th history frame, loaded once and indexed per call."""
    if cache[0] is None:
        wout = sorted(
            (R / f"sw-wrf-novec-mp{mp:02d}").glob("wrfout_d01_*"))
        fields = GATE_FIELDS + PUBLISHED_ONLY + (
            MP28_ONLY if mp == 28 else ())
        cache[0] = list(wrf_frames(wout[0], fields)) if wout else []
    rows = cache[0]
    return rows[i] if i < len(rows) else None


if __name__ == "__main__":
    raise SystemExit(main())
