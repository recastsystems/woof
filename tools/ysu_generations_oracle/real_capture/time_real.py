"""Time the YSU column kernel per generation on recorded real columns.

    python -m tools.ysu_generations_oracle.real_capture.time_real CAPTURE_IN.bin CAPTURE_OUT.bin [REPEATS]

A separate timing run (never inside a reported forecast): the real columns of
one captured step are launched as one (nz, 1, ncol) slab, the way
run_port_wide groups them, REPEATS times per generation after one warm
launch, timed by CUDA events around the whole batch.  Prints one JSON line
with the median per-launch milliseconds of each generation.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def main(argv):
    import cupy as cp

    from woof.core.ysu import launch_ysu
    from woof.verify.ysu_wide_oracle import load_wide_oracle
    from woof.wrf_exact import ENABLED

    inp, out = Path(argv[0]), Path(argv[1])
    repeats = int(argv[2]) if len(argv) > 2 else 50
    with tempfile.TemporaryDirectory() as d:
        shutil.copyfile(inp, Path(d) / "ysu-wide-inputs.bin")
        shutil.copyfile(out, Path(d) / "ysu-wide-wrf.bin")
        cols = load_wide_oracle(Path(d))
    nz, n = cols[0].nz, len(cols)
    assert all(c.nz == nz and c.topo == 0 for c in cols)
    dt = float(cols[0].scalars["dt"])
    topdown = cols[0].topdown

    def lev(name):
        return cp.asarray(np.ascontiguousarray(np.stack([c.levels[name] for c in cols], axis=1).reshape(nz, 1, n)))

    def sfc(name):
        return cp.asarray(np.asarray([c.scalars[name] for c in cols], np.float32).reshape(1, n))

    args = (lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"), lev("qi"), lev("p"),
            cp.asarray(np.ascontiguousarray(np.stack([c.p_interface for c in cols], axis=1).reshape(nz + 1, 1, n))),
            lev("pi"), lev("dz"), lev("rthraten"))
    kw = {k: sfc(k) for k in ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim", "psih",
                              "xland", "u10", "v10")}
    result = {"columns": n, "nz": nz, "strict_arithmetic": bool(ENABLED), "repeats": repeats,
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
    for version in ("wrf_461", "wrf_480", "wrf_461", "wrf_480"):
        launch_ysu(*args, dt=dt, ysu_topdown_pblmix=topdown, ysu_version=version, **kw)
        cp.cuda.Device().synchronize()
        times = []
        for _ in range(repeats):
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record()
            launch_ysu(*args, dt=dt, ysu_topdown_pblmix=topdown, ysu_version=version, **kw)
            b.record()
            b.synchronize()
            times.append(cp.cuda.get_elapsed_time(a, b))
        result[f"{version}_ms_median"] = float(np.median(times))
    result["ratio_480_over_461"] = result["wrf_480_ms_median"] / result["wrf_461_ms_median"]
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
