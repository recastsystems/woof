"""Strict word replay of complete GBMPBL bindings captured in native WRF.

The stream is emitted by instrument_native.py in a big-endian WRF build.
No synthetic inputs or numerical tolerances are used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import cupy as cp
import numpy as np

from woof.core import gbmpbl


def replay(path: Path) -> dict:
    data = path.read_bytes()
    nx, ny, nz, dt = struct.unpack_from(">iiif", data)
    if min(nx, ny, nz) < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid capture header: {(nx, ny, nz, dt)}")
    position = 16

    def take(column: bool, *, integer: bool = False) -> np.ndarray:
        nonlocal position
        count = nx * ny * (nz if column else 1)
        values = np.frombuffer(data, dtype=">i4" if integer else ">f4",
                               count=count, offset=position)
        position += count * 4
        if column:
            values = values.reshape((nx, nz, ny), order="F").transpose(1, 2, 0)
        else:
            values = values.reshape((nx, ny), order="F").T
        return np.ascontiguousarray(values, dtype=np.int32 if integer else np.float32)

    columns = {name: take(True)
               for name in (*gbmpbl.GBMPBL_COLUMN_INPUTS, "tke_pbl")}
    surface = {name: take(False) for name in gbmpbl.GBMPBL_SURFACE_INPUTS}
    native = {name: take(True)
              for name in (*gbmpbl.GBMPBL_COLUMN_OUTPUTS, "tke_pbl")}
    native.update({name: take(False, integer=name == "kpbl")
                   for name in ("pblh", "ust", "kpbl")})
    if position != len(data):
        raise ValueError(f"capture size mismatch: consumed {position}, have {len(data)}")

    tke = cp.asarray(columns.pop("tke_pbl"))
    output = gbmpbl.gbmpbl_step(
        {name: cp.asarray(value) for name, value in columns.items()},
        {name: cp.asarray(value) for name, value in surface.items()},
        tke, dt=dt, flqi=True)
    output["tke_pbl"] = tke
    fields = {}
    for name, reference in native.items():
        actual = cp.asnumpy(output[name])
        different = actual.view(np.uint32) != reference.view(np.uint32)
        fields[name] = {
            "words": reference.size,
            "different_words": int(different.sum()),
            "gpu_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "native_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
            "native_nonfinite": int((~np.isfinite(reference)).sum()),
            "max_absolute_difference": float(np.nanmax(
                abs(actual.astype(np.float64) - reference))),
        }
        if different.any():
            index = tuple(np.argwhere(different)[0])
            fields[name]["first_difference"] = {
                "index": list(map(int, index)),
                "gpu": float(actual[index]), "native": float(reference[index])}
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "shape": [nz, ny, nx], "dt": dt, "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("GPUWM_NUMERICS") != "strict":
        parser.error("set GPUWM_NUMERICS=strict before starting the replay")
    calls = [replay(path) for path in args.captures]
    receipt = {
        "schema": "gpuwm.gbmpbl-native-replay.v1",
        "module_source_sha256": hashlib.sha256(gbmpbl.module_source().encode()).hexdigest(),
        "numerics": "strict",
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "calls": calls,
        "total_words": sum(field["words"] for call in calls for field in call["fields"].values()),
        "different_words": sum(field["different_words"] for call in calls for field in call["fields"].values()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "calls"},
                     indent=2), flush=True)
    return int(bool(receipt["different_words"]))


if __name__ == "__main__":
    raise SystemExit(main())
