#!/usr/bin/env python3
"""Decode the bounded native ZLD capture after a witnessed WRF run.

The native routine writes its inputs directly. This tool only changes array
layout for the replay API and binds source, compiler, run and binary receipts.
Do not use a captured NPZ as evidence without the referenced run receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run_receipt.read_text())
    if not run.get("native_run_completed") or not run.get("strict_numerics"):
        raise ValueError("a completed strict native run receipt is required")
    raw = args.capture.read_bytes()
    header = np.frombuffer(raw, dtype="<i4", count=11)
    magic, nx, ny, nz, nl = map(int, header[:5])
    if magic != 47101 or not (0 < nx <= 4 and 0 < ny <= 4 and nz >= 2 and nl > 0):
        raise ValueError("invalid bounded WRF 4.7.1 capture header")
    offset = 44
    data = {}
    shapes = {
        "u": (nx+1,nz,ny), "v": (nx,nz,ny+1), "w": (nx,nz+1,ny),
        "t": (nx,nz,ny), "qv": (nx,nz,ny), "ph": (nx,nz+1,ny),
        "phb": (nx,nz+1,ny), "pp": (nx,nz,ny), "pb": (nx,nz,ny),
        "p": (nx,nz,ny), "pw": (nx,nz+1,ny), "terrain": (nx,ny),
        "z_levels": (nl,),
    }
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        a = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).reshape(shape, order="F")
        offset += count*4
        data[name] = np.ascontiguousarray(a.transpose(1,2,0) if len(shape)==3
                                        else a.T if len(shape)==2 else a)
    if offset != len(raw):
        raise ValueError("native capture has unexpected trailing bytes")
    # These are replay targets, not new meteorological inputs. Pressure
    # is evaluated by the unchanged PLD reference on the same native fields.
    data["p_levels"] = np.array([100000.,92500.,85000.,70000.,50000.,30000.,20000.], dtype="f4")
    data["provenance"] = json.dumps({
        "real_native_capture": True, "kind": "raw inputs at native WRF ZLD entry",
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "native_bounds": header[5:].tolist(), "run_receipt": run,
        "run_receipt_sha256": sha(args.run_receipt),
        "decoder_sha256": sha(Path(__file__)),
        "pressure_targets": "explicit replay targets applied to captured native inputs",
    }, sort_keys=True)
    np.savez(args.output, **data)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": sha(args.output), "dimensions": [nz,ny,nx]}))


if __name__ == "__main__":
    main()
