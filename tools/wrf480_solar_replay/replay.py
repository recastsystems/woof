"""WRF 4.8.0 solar replay: WOOF-ArWen's device radconst/calc_coszen against the
referee build's own Fortran routines, word for word.

  python -m tools.wrf480_solar_replay.replay inputs DIR
  native-480 DIR/<grid>.in DIR/<grid>.native480        (build_native.sh)
  python -m tools.wrf480_solar_replay.replay gpu DIR --generation wrf_480
  python -m tools.wrf480_solar_replay.replay compare DIR native480 gpu-wrf_480-solar

Native side: the calendar scalars WRF's frame forms
(woof.core.wrf_radiation_clock.wrf_calendar_scalars), then the referee
build's own radconst + calc_coszen object code (radiation call at
xtime+radt*0.5, and the swint_opt call at xtime).  WOOF-ArWen side: the calendar
the production radiation adapters form (woof.core.swint.calendar_scalars,
the rrtmg_legacy statements), then woof.core.solar.solar_geometry_device,
the kernel every default consumer runs.  A word difference therefore covers
the calendar and the routine.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

F = np.float32
LABELS = ("declin", "solcon", "coszen", "hrang", "coszen_loc", "hrang_loc")


def grids():
    # A071 shape: 151 x 151 at about 3 km around the iowa-convective centre.
    j, i = np.mgrid[0:151, 0:151]
    iowa = (F(41.0) + (j - 75).astype(np.float32) * F(0.027),
            F(-93.5) + (i - 75).astype(np.float32) * F(0.036))
    # Every latitude band and hour angle: global 2 degree.
    j, i = np.mgrid[0:91, 0:180]
    world = ((-90.0 + 2.0 * j).astype(np.float32),
             (-179.0 + 2.0 * i).astype(np.float32))
    return {"iowa3km": iowa, "world2deg": world}


def calendars():
    cal = {"iowa3km": [], "world2deg": []}
    # A071: 2024-05-25 18Z, dt 12 s, radt 3 min, one hour.
    s = datetime(2024, 5, 25, 18)
    cal["iowa3km"] += [("A071", s, 180.0 * k, 3.0) for k in range(21)]
    # HRRR recipe (A195 shape): 2024-05-21 21Z, dt 15 s, radt 1 min, one hour.
    s = datetime(2024, 5, 21, 21)
    cal["iowa3km"] += [("A195", s, 60.0 * k, 1.0) for k in range(61)]
    # A whole leap year from one start, 6-hourly plus 12 s, radt 15.
    s = datetime(2024, 1, 1, 0)
    cal["world2deg"] += [("year2024", s, 21600.0 * k + 12.0, 15.0)
                         for k in range(366 * 4)]
    # Days whose JULIAN word sits at a binade edge, every 7 minutes, one-day
    # runs started at 00Z that day.
    for d in (0, 1, 3, 7, 15, 31, 63, 127, 255, 364):
        s = datetime(2023, 1, 1) + timedelta(days=d)
        cal["world2deg"] += [(f"edge-d{d}", s, 420.0 * k, 10.0)
                             for k in range(206)]
    # A 30-minute start hour across a new year, three days.
    s = datetime(2023, 12, 30, 18, 30)
    cal["world2deg"] += [("newyear-1830", s, 900.0 * k + 36.0, 15.0)
                         for k in range(4 * 24 * 3)]
    return cal


def cmd_inputs(out: Path):
    from woof.core.wrf_radiation_clock import wrf_calendar_scalars
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    cals = calendars()
    for name, (lat, lon) in grids().items():
        rows, native = [], []
        for case, start, elapsed, radt in cals[name]:
            julian, xtime, gmt = wrf_calendar_scalars(start, elapsed)
            native.append((julian, xtime, gmt, F(radt)))
            rows.append({"case": case, "start": start.isoformat(),
                         "elapsed": elapsed, "radt": radt})
        with open(out / f"{name}.in", "wb") as f:
            np.asarray([lat.size, len(native)], np.int32).tofile(f)
            np.ascontiguousarray(lat, np.float32).tofile(f)
            np.ascontiguousarray(lon, np.float32).tofile(f)
            np.asarray(native, np.float32).tofile(f)
        meta[name] = {"shape": list(lat.shape), "calls": rows}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def _read_grid(path: Path):
    with open(path, "rb") as f:
        n, ncal = (int(v) for v in np.fromfile(f, np.int32, 2))
        lat = np.fromfile(f, np.float32, n)
        lon = np.fromfile(f, np.float32, n)
        cal = np.fromfile(f, np.float32, 4 * ncal).reshape(ncal, 4)
    return n, ncal, lat, lon, cal


def cmd_gpu(out: Path, generation: str, path: str):
    import time
    import cupy as cp
    from woof.core.swint import calendar_scalars
    from woof.core.solar import solar_geometry_device
    meta = json.loads((out / "meta.json").read_text())
    timing = {}
    for name in meta:
        n, ncal, lat, lon, _cal = _read_grid(out / f"{name}.in")
        lat_d, lon_d = cp.asarray(lat), cp.asarray(lon)
        res = np.empty((ncal, 2 + 4 * n), np.float32)
        for k, row in enumerate(meta[name]["calls"]):
            start = datetime.fromisoformat(row["start"])
            julian, xtime, gmt = calendar_scalars(start, row["elapsed"],
                                                  generation=generation)
            mid = F(xtime + F(row["radt"]) * F(0.5))
            if path == "geometry_at":  # dudhia/rrtmgp/urban/noahmp/gsfcsw entry
                from woof.core.solar import geometry_at
                valid = start + timedelta(seconds=row["elapsed"])
                ds, cz, ha = geometry_at(valid, lat_d, lon_d, generation=generation,
                                         hour_offset_seconds=30.0 * row["radt"],
                                         start_time=start, elapsed_seconds=row["elapsed"])
                _d2, czl, hal = geometry_at(valid, lat_d, lon_d, generation=generation,
                                            start_time=start, elapsed_seconds=row["elapsed"])
                ds = cp.asnumpy(ds)
            elif path == "solar":
                ds, cz, ha = solar_geometry_device(julian, mid, gmt, lat_d, lon_d,
                                                   generation=generation)
                _d2, czl, hal = solar_geometry_device(julian, xtime, gmt, lat_d, lon_d,
                                                      generation=generation)
                ds = cp.asnumpy(ds)
            else:  # the 4.6.1 rrtmg_legacy kernel the strict wrf_461 adapter runs
                from woof.core.rrtmg_legacy import solar_geometry_device as legacy
                cz, ha, declin, solcon = legacy(julian, mid, gmt, lat_d, lon_d)
                czl, hal, _d, _s = legacy(julian, xtime, gmt, lat_d, lon_d)
                ds = np.asarray([declin, solcon], np.float32)
            res[k, :2] = ds
            for m, a in enumerate((cz, ha, czl, hal)):
                res[k, 2 + m * n:2 + (m + 1) * n] = cp.asnumpy(a).reshape(-1)
        res.tofile(out / f"{name}.gpu-{generation}-{path}")
        # Kernel time: one radiation-call geometry, warm, synchronised.
        for _ in range(5):
            solar_geometry_device(F(145.75), F(1.5), F(18.0), lat_d, lon_d,
                                  generation=generation)
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        for _ in range(500):
            solar_geometry_device(F(145.75), F(1.5), F(18.0), lat_d, lon_d,
                                  generation=generation)
        cp.cuda.Device().synchronize()
        timing[name] = (time.perf_counter() - t) / 500 * 1e3
    (out / f"timing-{generation}-{path}.json").write_text(json.dumps(timing))
    print(json.dumps(timing))


def cmd_compare(out: Path, native: str, gpu: str):
    meta = json.loads((out / "meta.json").read_text())
    report = {"native": native, "gpu": gpu, "grids": {}}
    total = diff = 0
    for name in meta:
        n, ncal, *_ = _read_grid(out / f"{name}.in")
        a = np.fromfile(out / f"{name}.{native}", np.float32).reshape(ncal, 2 + 4 * n)
        b = np.fromfile(out / f"{name}.{gpu}", np.float32).reshape(ncal, 2 + 4 * n)
        ne = a.view(np.uint32) != b.view(np.uint32)
        spans = ((0, 1), (1, 2)) + tuple((2 + m * n, 2 + (m + 1) * n) for m in range(4))
        per = {lab: {"words": int(ncal * (hi - lo)),
                     "different": int(ne[:, lo:hi].sum())}
               for lab, (lo, hi) in zip(LABELS, spans)}
        cases = {}
        for k in np.nonzero(ne.any(axis=1))[0]:
            c = meta[name]["calls"][int(k)]["case"]
            cases[c] = cases.get(c, 0) + 1
        words = int(ne.size)
        different = int(ne.sum())
        total += words
        diff += different
        report["grids"][name] = {"calls": ncal, "points": n, "fields": per,
                                 "words": words, "different": different,
                                 "calls_with_differences_by_case": cases}
    report["words"], report["different"] = total, diff
    text = json.dumps(report, indent=1)
    (out / f"compare-{native}-vs-{gpu}.json").write_text(text)
    print(text)
    return 0 if diff == 0 else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inputs")
    s.add_argument("dir", type=Path)
    s = sub.add_parser("gpu")
    s.add_argument("dir", type=Path)
    s.add_argument("--generation", default="wrf_480")
    s.add_argument("--path", default="solar", choices=("solar", "geometry_at", "legacy"))
    s = sub.add_parser("compare")
    s.add_argument("dir", type=Path)
    s.add_argument("native")
    s.add_argument("gpu")
    a = p.parse_args(argv)
    if a.cmd == "inputs":
        return cmd_inputs(a.dir)
    if a.cmd == "gpu":
        return cmd_gpu(a.dir, a.generation, a.path)
    return cmd_compare(a.dir, a.native, a.gpu)


if __name__ == "__main__":
    sys.exit(main())
