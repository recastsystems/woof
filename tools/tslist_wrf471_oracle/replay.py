"""Replay a capture from instrument.py through the production kernel loader.

Run in separate processes for GPUWM_WRF_EXACT=0 and =1. Proof input must
come from the instrumented native executable. Shape checks alone cannot
authenticate provenance; retain the native source/build/artifact receipt
binding this stream's hash. GPU synchronization here is proof-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel, module_source, module_options
from woof.wrf_exact import ENABLED


def read_capture(path):
    data = path.read_bytes()
    # The native configuration uses -fconvert=big-endian for WRF's I/O.
    dims = np.frombuffer(data[:24], dtype=">i4").astype(np.int32)
    nx, ny, nz, levels, theta_m, unstagger = map(int, dims)
    if (nx, ny) != (2, 2) or not (2 <= nz <= 1000 and 1 <= levels <= nz):
        raise ValueError(f"bad native capture dimensions: {dims}")
    values = np.frombuffer(data[24:], dtype=">f4").astype(np.float32)
    cursor = 0

    def take(count):
        nonlocal cursor
        if cursor + count > values.size:
            raise ValueError("truncated native capture")
        value = values[cursor:cursor + count]
        cursor += count
        return value

    hour = take(1)[0]
    nxy = nx * ny
    u, v = take(nz * ny * (nx + 1)), take(nz * (ny + 1) * nx)
    w, phb, ph = (take((nz + 1) * nxy) for _ in range(3))
    theta, pressure = take(nz * nxy), take(nz * nxy)
    cosa, sina = take(nxy), take(nxy)
    fnm, fnp = take(nz), take(nz)
    surface = [take(nxy) for _ in range(13)]
    moisture = [take(nz * nxy) for _ in range(6)]
    # WRF allocates these carriers through e_vert even when max_ts_level is
    # smaller. The native statements initialize only 1:max_ts_level. The
    # capture deliberately preserves the allocation; never grade its tail.
    tail = values.size - cursor - 15
    if tail % 7:
        raise ValueError("native profile allocation is not seven equal arrays")
    allocated_levels = tail // 7
    if allocated_levels not in (levels, nz + 1):
        raise ValueError("unexpected native profile allocation extent")
    expected = np.concatenate([take(15), *(take(allocated_levels)[:levels] for _ in range(7))])
    if cursor != values.size:
        raise ValueError("native capture has unexpected trailing words")
    return (data, (nx, ny, nz, levels, theta_m, unstagger), hour,
            (u, v, w, phb, ph, theta, pressure, cosa, sina, fnm, fnp),
            surface, moisture, expected, allocated_levels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data, dims, hour, carriers, surface, moisture, expected, allocated_levels = read_capture(args.capture)
    nx, ny, nz, levels, theta_m, unstagger = dims
    carriers = [cp.asarray(a) for a in carriers]
    surface = [cp.asarray(a) for a in surface]
    moisture = [cp.asarray(a) for a in moisture]
    surface_ptrs = cp.asarray([a.data.ptr for a in surface], dtype=cp.uint64)
    moisture_ptrs = cp.asarray([a.data.ptr for a in moisture], dtype=cp.uint64)
    ij = cp.asarray([0, 0], dtype=cp.int32)
    output = cp.empty(expected.size, dtype=cp.float32)
    kernel = get_kernel("tslist", "tslist_gather")
    arguments = (ij, surface_ptrs, moisture_ptrs, *carriers, output,
                 *(np.int32(i) for i in (nx, ny, nz, 1, levels, 0, unstagger, 0, theta_m)),
                 hour, np.uint64(0), np.int32(0), np.uint64(0), np.int32(1))
    kernel((1,), (128,), arguments, shared_mem=4 * (nz + 1))
    actual = cp.asnumpy(output)
    different = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    # Timing is one bounded batch of this replay, never a proof campaign.
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    for _ in range(100):
        kernel((1,), (128,), arguments, shared_mem=4 * (nz + 1))
    end.record()
    end.synchronize()
    receipt = {
        "schema": "arwen.tslist.native-replay.v1",
        "strict": ENABLED, "input_sha256": hashlib.sha256(data).hexdigest(),
        "source_sha256": hashlib.sha256(module_source("tslist").encode()).hexdigest(),
        "compiler_options": module_options("tslist"),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "levels": levels, "native_levels": nz, "unstagger": bool(unstagger),
        "native_allocated_profile_levels": allocated_levels,
        "uncomputed_allocation_tail_words_excluded": 7 * (allocated_levels - levels),
        "words": int(expected.size), "different_words": int(different.size),
        "differences": [{"index": int(i), "native": f"{expected.view(np.uint32)[i]:08x}",
                         "gpu": f"{actual.view(np.uint32)[i]:08x}"} for i in different[:32]],
        "kernel_ms": cp.cuda.get_elapsed_time(begin, end) / 100,
        "timing_launches": 100,
        "capture": str(args.capture),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if ENABLED and different.size:
        raise SystemExit("strict tslist replay differs from native")


if __name__ == "__main__":
    main()
