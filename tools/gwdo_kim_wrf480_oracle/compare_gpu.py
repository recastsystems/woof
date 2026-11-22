#!/usr/bin/env python3
"""Grade the gwd_opt = 1 GPU kernels against the native referees' words.

    python compare_gpu.py INPUT_DIR OUT_DIR [--write-fixture DIR [--every N]]
                          [--json OUT]

INPUT_DIR holds make_inputs.py's sets, OUT_DIR build.sh's outputs.  For each
set it runs the production launcher (woof.core.terrain_drag.launch_gwdo)
twice: ``gwdo_version="wrf_480"`` over every case against the WRF v4.8.0
referee's words, and ``"wrf_471"`` (the frozen kernel) over the distinct
(dx, dt) cases against the WRF v4.7.1 referee's words.  It prints one JSON
document with, per field, the words compared and the words that are not
bit-identical, and times the two kernels on the largest set.  Set
GPUWM_WRF_EXACT=1 for the strict build.  ``--write-fixture`` writes the
.npz pairs the pytest gate reads (tests/data/oracles/gwdo_kim480/), every
N-th column of the real sets (``--every``, default 97) plus up to 400
of the columns WRF dragged.  Exit status 0 only
when every word of every set is bit-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof.verify import gwdo_kim480_oracle as go  # noqa: E402


def _subset(inputs, wrf, every, moved_cap=400):
    """Every ``every``-th column plus up to ``moved_cap`` of the columns
    WRF dragged (evenly spaced), so a sparse-drag frame keeps its active
    columns in the committed fixture."""
    ncol = inputs["u"].shape[1]
    moved = np.flatnonzero((wrf["rublten"].view(np.uint32)
                            != inputs["rublten0"][None].view(np.uint32))
                           .any(axis=(0, 1)))
    if moved.size > moved_cap:
        moved = moved[np.linspace(0, moved.size - 1, moved_cap).astype(int)]
    keep = np.union1d(np.arange(0, ncol, every), moved)
    sub_in = {}
    for k, v in inputs.items():
        v = np.asarray(v)
        if k in ("dx", "dt", "dx_factor", "if_nonhyd", "source",
                 "edge_notes"):
            sub_in[k] = v
        elif k == "column_ij":
            sub_in[k] = v[:, keep]
        elif v.ndim >= 1 and v.shape[-1] == inputs["u"].shape[1]:
            sub_in[k] = v[..., keep]
        else:
            sub_in[k] = v
    sub_wrf = {k: np.ascontiguousarray(v[..., keep]) for k, v in wrf.items()}
    return sub_in, sub_wrf


def _time(inputs, version, repeat=20):
    """Median ms per launch of one case over the whole set (one card)."""
    import cupy as cp
    from woof.core.terrain_drag import launch_gwdo

    nz, ncol = inputs["u"].shape
    lev = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32)  # noqa
                               .reshape(a.shape[0], 1, ncol))
    row = lambda a: cp.asarray(np.ascontiguousarray(a, np.float32)  # noqa
                               .reshape(1, ncol))
    atm = {"u": lev(inputs["u"]), "v": lev(inputs["v"]),
           "temperature": lev(inputs["t"]), "qv": lev(inputs["qv"]),
           "pressure": lev(inputs["p"]), "exner": lev(inputs["pi"]),
           "p_interface": lev(inputs["p_interface"]), "z": lev(inputs["z"]),
           "dz": lev(inputs["dz"])}
    oa = [row(inputs["oa"][m]) for m in range(4)]
    ol = [row(inputs["ol"][m]) for m in range(4)]
    stat = dict(var=row(inputs["var"]), con=row(inputs["oc1"]),
                sina=row(inputs["sina"]), cosa=row(inputs["cosa"]))
    du = lev(inputs["rublten0"])
    dv = lev(inputs["rvblten0"])
    extra = {}
    if version == "wrf_480":
        extra = dict(omax=row(inputs["elvmax"]), dx_factor=2.0,
                     if_nonhyd=True)
    times = []
    for n in range(repeat + 3):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        launch_gwdo(atm, du, dv, oa=oa, ol=ol, dx=3000.0, dt=20.0,
                    gwdo_version=version, **stat, **extra)
        end.record()
        end.synchronize()
        if n >= 3:
            times.append(cp.cuda.get_elapsed_time(start, end))
    return float(np.median(times))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--write-fixture", type=Path)
    ap.add_argument("--every", type=int, default=97)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {"strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
              "device": dev["name"].decode(),
              "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
              "sets": {}}
    exact = True
    words = {"wrf_480": [0, 0], "wrf_471": [0, 0]}
    largest = None
    for f in sorted(args.input_dir.glob("gwdo-kim480-inputs-*.npz")):
        name = f.stem.replace("gwdo-kim480-inputs-", "")
        with np.load(f) as d:
            inputs = {k: d[k] for k in d.files}
        binned = go.read_inputs_bin(f.with_suffix(".bin"))
        for k, v in binned.items():
            if np.asarray(inputs[k]).tobytes() != v.tobytes():
                raise SystemExit(f"{name}: npz and bin disagree on {k}")
        nz, ncol = inputs["u"].shape
        row = {"columns": ncol, "levels": nz}
        for version, ref, ncase in (
                ("wrf_480", f"gwdo-kim480-wrf480-{name}.bin",
                 len(inputs["dx"])),
                ("wrf_471", f"gwdo-kim471-wrf471-{name}.bin",
                 len(go.distinct_dx_dt(inputs)[0]))):
            wrf = go.read_outputs_bin(args.out_dir / ref, nz, ncol, ncase)
            got = go.run_gpu(inputs, gwdo_version=version)
            table = go.compare(got, wrf, inputs["regime"])
            n_words = sum(r["words"] for r in table.values())
            n_diff = sum(r["not_bit_identical"] for r in table.values())
            words[version][0] += n_words
            words[version][1] += n_diff
            exact &= n_diff == 0
            row[version] = {
                "words": n_words, "not_bit_identical": n_diff,
                "fields": {k: {kk: vv for kk, vv in r.items()
                               if kk != "columns_missed_by_regime"}
                           for k, r in table.items()},
                "active_columns_per_case": [
                    int(x) for x in ((wrf["rublten"].view(np.uint32)
                                      != inputs["rublten0"][None]
                                      .view(np.uint32)).any(axis=1)
                                     .sum(axis=1))],
                "blocking_columns_per_case": go.blocking_columns(inputs,
                                                                 wrf),
            }
            if args.write_fixture and version == "wrf_480":
                args.write_fixture.mkdir(parents=True, exist_ok=True)
                fin, fwrf = inputs, wrf
                if name.startswith("real-"):
                    fin, fwrf = _subset(inputs, wrf, args.every)
                np.savez_compressed(args.write_fixture
                                    / f"gwdo-kim480-inputs-{name}.npz", **fin)
                np.savez_compressed(args.write_fixture
                                    / f"gwdo-kim480-wrf480-{name}.npz", **fwrf)
        report["sets"][name] = row
        if largest is None or ncol > largest[1]["u"].shape[1]:
            largest = (name, inputs)
    if largest is not None:
        name, inputs = largest
        report["timing_ms_per_launch"] = {
            "set": name, "columns": int(inputs["u"].shape[1]),
            "levels": int(inputs["u"].shape[0]),
            "wrf_471": _time(inputs, "wrf_471"),
            "wrf_480": _time(inputs, "wrf_480")}
    report["words"] = {k: {"compared": v[0], "not_bit_identical": v[1]}
                       for k, v in words.items()}
    report["bitwise_everywhere"] = bool(exact)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
