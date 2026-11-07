"""One bounded integration check of the CUDA/Rust output primitive.

This is not a forecast and does not stand in for the required woof go proof.
The first sample is compared byte-for-byte with every native text file. Later
samples deliberately reuse the same captured fields, exercise both ring slots,
and prove checkpoint rollback without silently overwriting live D2H buffers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import cupy as cp
import numpy as np

from woof.core.tslist_sampler import TslistSampler
from woof.tslist_bridge import RustTslistWriter
from replay import read_capture


def bits(value):
    return f"{struct.unpack('I', struct.pack('f', value))[0]:08x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    data, dims, hour, carriers, surface, moisture, expected, _ = read_capture(args.case / "tslist-real-input.bin")
    nx, ny, nz, levels, theta_m, unstagger = dims
    shapes = [(nz, ny, nx+1), (nz, ny+1, nx), *[(nz+1, ny, nx)]*3,
              *[(nz, ny, nx)]*2, *[(ny, nx)]*2, (nz,), (nz,)]
    u, v, w, phb, ph, theta, pressure, cosa, sina, fnm, fnp = [
        cp.asarray(a.reshape(s)) for a, s in zip(carriers, shapes, strict=True)]
    surface = [cp.asarray(a.reshape(ny, nx)) for a in surface]
    moisture = [cp.asarray(a.reshape(nz, ny, nx)) for a in moisture]
    plan = "\t".join(["1", "35", "30", bits(34.809), bits(-81.719), bits(176.9),
                      "v1", bits(35), bits(30), "proof", "Native real input"]) + "\n"

    def open_sampler(saved=None):
        writer = RustTslistWriter(args.output, plan, geographic=False, domain=1,
            levels=levels, start="2000-01-24_12:00:00", saved=saved)
        return TslistSampler(ij=np.array([[0, 0]], np.int32), surface=surface, moisture=moisture,
            u=u, v=v, w=w, phb=phb, ph=ph, theta=theta, pressure=pressure,
            cosa=cosa, sina=sina, fnm=fnm, fnp=fnp, levels=levels, buffer_size=3,
            unstagger=bool(unstagger), theta_m=bool(theta_m), writer=writer)

    sampler = open_sampler()
    started = time.perf_counter()
    for sample in range(8):
        sampler.sample(np.float32(hour + np.float32(sample) / np.float32(20)))
    saved = sampler.checkpoint()
    for sample in range(8, 11):
        sampler.sample(np.float32(hour + np.float32(sample) / np.float32(20)))
    sampler.close()
    continued = {p.name: p.read_bytes() for p in args.output.iterdir()}
    sampler = open_sampler(saved)
    for sample in range(8, 11):
        sampler.sample(np.float32(hour + np.float32(sample) / np.float32(20)))
    sampler.close()
    rows = []
    for ending in ("TS", "UU", "VV", "WW", "PH", "TH", "QV", "PR"):
        name = f"proof.d01.{ending}"
        actual = (args.output / name).read_bytes()
        if actual != continued[name]:
            raise AssertionError(f"checkpoint continuation changed {name}")
        expected_lines = (args.case / name).read_bytes().splitlines(keepends=True)
        actual_lines = actual.splitlines(keepends=True)
        if actual_lines[:2] != expected_lines[:2]:
            raise AssertionError(f"native first record differs: {name}: {actual_lines[:2]!r} vs {expected_lines[:2]!r}")
        if len(actual_lines) != 12:
            raise AssertionError(f"wrong sample count in {name}")
        rows.append({"file": name, "bytes": len(actual), "rows": len(actual_lines)-1,
            "sha256": hashlib.sha256(actual).hexdigest(), "native_first_record_equal": True})
    receipt = {"schema": "arwen.tslist.async-rust-proof.v1", "samples": 11,
        "buffer_size": 3, "ring_slots": 2, "checkpoint_sample": 8,
        "checkpoint_rollback_byte_equal": True, "files": rows,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "wall_seconds_including_resume_validation": time.perf_counter()-started,
        "forecast": False}
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
