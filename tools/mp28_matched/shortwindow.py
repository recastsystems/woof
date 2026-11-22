"""Short-window paired trajectory test, 1800 s -> 1920 s.

ADDED AFTER the 7200 s runs, and labelled as such wherever it is reported.
Its motive is in the measurement, not in the answer: over the long run the
two models' trajectories decorrelate in this unsheared convective case, and
once they have, every metric compares two different realisations of the same
statistical process.  A short window from a MATURE state is the only regime
in which "matched trajectory" means what it says -- every microphysical
process is active, and 10 steps is too few for chaos to amplify.

The statistic is M8, already declared: normalised RMS field difference
``||A - W||2 / ||W||2``.  The verdict condition applied is V3, also already
declared: the mp=28 disagreement must not exceed 3x the mp=8 disagreement.

Usage:
    python shortwindow.py --runs DIR --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import netCDF4

FIELDS_MP8 = ("U", "V", "W", "T", "PH", "MU", "QVAPOR", "QCLOUD", "QRAIN",
              "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE", "RAINNC")
FIELDS_MP28 = FIELDS_MP8 + ("QNCLOUD", "QNWFA", "QNIFA")


def rms_diff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = float(np.sqrt((b * b).mean()))
    num = float(np.sqrt(((a - b) ** 2).mean()))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_run_screen(runs: Path, mp: int) -> dict:
    """Byte-compare the duplicated WOOF-ArWen short-window run against its twin.

    The 5090 has no ECC, so a single GPU run is not a reproducible result.
    ``compare.py`` applies this screen to the 7200 s runs and writes it into
    its receipt; this file used to claim the screen for the short window
    without performing it, which is exactly the gap the screen exists to
    close.  Every frame file present in BOTH directories is hashed.
    """
    m = f"{mp:02d}"
    a, b = runs / f"sw-arwen-mp{m}", runs / f"sw-arwen-mp{m}-b"
    if not a.is_dir() or not b.is_dir():
        return {"status": "missing", "a": str(a), "b": str(b)}
    names = sorted({p.name for p in a.glob("frame_t*.npz")}
                   & {p.name for p in b.glob("frame_t*.npz")})
    differing = [n for n in names if _sha256(a / n) != _sha256(b / n)]
    return {"status": "ok", "a": str(a), "b": str(b),
            "frames_compared": len(names), "identical": not differing,
            "differing": differing}


def wrf_frames(path: Path, fields):
    with netCDF4.Dataset(path) as ds:
        n = ds.variables["Times"].shape[0]
        for i in range(n):
            yield {f: np.asarray(ds.variables[f][i], np.float64)
                   for f in fields if f in ds.variables}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dt", type=float, default=60.0,
                    help="seconds between compared frames "
                         "(WRF history interval == WOOF-ArWen "
                         "frame-steps * timestep)")
    args = ap.parse_args()

    report = {"note": "post-hoc supplement; statistic M8 and condition V3 "
                      "are the pre-declared ones",
              "start_s": 1800.0, "frame_dt_s": args.dt,
              "dual_run": {f"mp{mp:02d}": dual_run_screen(args.runs, mp)
                           for mp in (8, 28)},
              "runs": {}}
    for mp in (8, 28):
        m = f"{mp:02d}"
        fields = FIELDS_MP28 if mp == 28 else FIELDS_MP8
        wout = sorted((args.runs / f"sw-wrf-mp{m}").glob("wrfout_d01_*"))
        aout = args.runs / f"sw-arwen-mp{m}"
        if not wout or not (aout / "series.json").exists():
            report["runs"][m] = {"status": "missing"}
            continue
        rows = []
        wframes = list(wrf_frames(wout[0], fields))
        for i, wf in enumerate(wframes):
            p = aout / f"frame_t{int(round(i * args.dt))}.npz"
            if not p.exists():
                continue
            A = np.load(p)
            row = {"step": i, "time_s": 1800.0 + i * args.dt}
            for f in fields:
                if f in A.files and f in wf:
                    row[f] = rms_diff(A[f], wf[f])
            rows.append(row)
        report["runs"][m] = {"status": "ok", "frames": rows}

    # V3 on M8 over the short window.
    v3, v3_pass = [], True
    r8 = report["runs"].get("08", {}).get("frames") or []
    r28 = report["runs"].get("28", {}).get("frames") or []
    for a, b in zip(r8, r28):
        for f in FIELDS_MP8:
            d8, d28 = a.get(f), b.get(f)
            if d8 is None or d28 is None or d8 <= 0.0:
                continue
            ratio = d28 / d8
            ok = ratio <= 3.0
            v3_pass &= ok
            v3.append({"field": f, "step": a["step"], "d_mp08": d8,
                       "d_mp28": d28, "ratio": ratio, "pass": bool(ok)})
    report["V3_short_window"] = {"rows": v3, "pass": bool(v3_pass),
                                 "worst": max((r["ratio"] for r in v3),
                                              default=None)}
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"V3_short_window_pass": report["V3_short_window"]["pass"],
                      "worst_ratio": report["V3_short_window"]["worst"],
                      "dual_run_identical": {
                          k: v.get("identical")
                          for k, v in report["dual_run"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
