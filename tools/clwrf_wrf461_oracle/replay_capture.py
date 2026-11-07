"""Strict CUDA replay of actual native CLWRF call captures."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.core.clwrf import HISTORY, INPUTS, launch


def replay(path):
    import cupy as cp
    raw = path.read_bytes()
    if raw[:8] != b"CLWRF001":
        raise ValueError(f"{path}: unrecognized native capture")
    nx, ny, before_count, interval, restart = struct.unpack_from("<5i", raw, 8)
    dt, xtime, seconds = struct.unpack_from("<3f", raw, 28)
    count = nx * ny
    if not 0 < nx <= 16 or not 0 < ny <= 16 or len(raw) != 44 + (len(INPUTS) + 2*len(HISTORY))*count*4:
        raise ValueError(f"{path}: invalid native capture dimensions or length")
    arrays = np.frombuffer(raw, dtype="<f4", count=(len(INPUTS) + 2*len(HISTORY))*count, offset=40).reshape(-1, ny, nx)
    inputs = {name: cp.asarray(value) for name, value in zip(INPUTS, arrays[:len(INPUTS)])}
    before = arrays[len(INPUTS):len(INPUTS) + len(HISTORY)]
    expected = arrays[len(INPUTS) + len(HISTORY):]
    output = {name: cp.asarray(value) for name, value in zip(HISTORY, before)}
    counter = cp.full((ny, nx), before_count, dtype=cp.int32)
    launch(inputs, output, counter, dt=dt, xtime=xtime, seconds=seconds, interval=interval)
    actual = np.stack([cp.asnumpy(output[name]) for name in HISTORY])
    differing = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    expected_count = struct.unpack_from("<i", raw, len(raw)-4)[0]
    counters = cp.asnumpy(counter)
    count_diff = int(np.any(counters != expected_count))
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [ny, nx], "dt": dt, "xtime": xtime, "seconds": seconds,
        "interval": interval, "restart": bool(restart), "before_nsteps": before_count,
        "native_nsteps": expected_count, "words": len(HISTORY)*count + 1,
        "differing_words": differing + count_diff,
        "input_summary": {name: {"min": float(value.min()), "max": float(value.max()),
            "nonzero": int(np.count_nonzero(value))} for name, value in zip(INPUTS, arrays)},
        "nonfinite_native_output_words": int(np.count_nonzero(~np.isfinite(expected)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        parser.error("set GPUWM_WRF_EXACT=1 for a bitwise native replay")
    rows = [replay(path) for path in args.captures]
    from woof.certify.kernel_manifest import kernel_manifest
    receipt = {"kind": "real strict native WRF CLWRF call captures", "calls": rows,
        "native_source": "WRF v4.6.1 phys/module_diag_cl.F",
        "source_sha256": "2ec1e964a9ae94244d513fac38e0ce9a33212b21ca232fcc01de890afd152a54",
        "kernel_manifest": kernel_manifest(),
        "words": sum(row["words"] for row in rows),
        "differing_words": sum(row["differing_words"] for row in rows)}
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    raise SystemExit(bool(receipt["differing_words"]))
