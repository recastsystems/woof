"""Bitwise PX module replay of calls captured inside a real WRF forecast.

The stream contains the native driver's constants, inputs and all INOUT
words immediately before and after PXSFCLAY1D. It has no NetCDF dependency.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from woof.core.physics_inventory import PX_SFCLAY_INPUTS, PX_SFCLAY_STATE

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data/pxsfclay/native"
CONSTANTS = ("cp", "g", "rovcp", "r", "xlv", "dx", "svp1", "svp2", "svp3",
             "svpt0", "ep1", "ep2", "karman")
MAGIC = 1347965776


def read_capture(path):
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    magic, columns, step, row = struct.unpack_from("<4i", raw)
    if magic != MAGIC or columns <= 0 or step <= 0:
        raise ValueError(f"invalid native PX capture header: {path}")
    if len(raw) != 16 + 4 * (len(CONSTANTS) + 60 * columns):
        raise ValueError(f"invalid native PX capture length: {path}")
    words = np.frombuffer(raw, dtype="<f4", offset=16)
    return ({"file": Path(path).name, "columns": columns,
             "itimestep": step, "j": row,
             "raw_sha256": hashlib.sha256(raw).hexdigest()},
            dict(zip(CONSTANTS, words[:13])), words[13:].reshape(60, columns))


def fixture_manifest(directory=FIXTURE_DIR):
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"native PX fixture changed: {name}")
    return manifest


def replay_capture(path):
    import cupy as cp
    from woof.core.pxsfclay import launch_pxsfclay, wrf_constants
    from woof.wrf_exact import ENABLED

    if not ENABLED:
        raise ValueError("native bitwise replay requires GPUWM_WRF_EXACT=1")
    record, constants, arrays = read_capture(path)
    for key, value in wrf_constants().items():
        if value.view("u4") != constants[key].view("u4"):
            raise ValueError(f"native PX physical constant differs: {key}")
    inputs = {key: cp.asarray(arrays[i].copy())
              for i, key in enumerate(PX_SFCLAY_INPUTS)}
    state = {key: cp.asarray(arrays[12 + i].copy())
             for i, key in enumerate(PX_SFCLAY_STATE)}
    launch_pxsfclay(inputs, state, dx=float(constants["dx"]),
                   itimestep=record["itimestep"])
    fields = {}
    for i, key in enumerate(PX_SFCLAY_STATE):
        actual = cp.asnumpy(state[key]).view("u4")
        expected = arrays[36 + i].view("u4")
        fields[key] = {"words": record["columns"],
                       "differing": int(np.count_nonzero(actual != expected))}
    return {**record, "fields": fields}


def replay(directory=FIXTURE_DIR):
    from woof.wrf_exact import referee_glibc
    directory = Path(directory)
    manifest = fixture_manifest(directory)
    version = ".".join(map(str, referee_glibc()))
    if version != manifest["referee_glibc"]:
        raise ValueError(f"native capture requires GPUWM_WRF_GLIBC={manifest['referee_glibc']}")
    records = [replay_capture(directory / name)
               for name in sorted(manifest["sha256"]) if name.endswith(".bin.gz")]
    fields = [field for record in records for field in record["fields"].values()]
    return {"schema": "gpuwm-pxsfclay-native-replay-v1", "strict": True,
            "referee_glibc": version, "captures": records,
            "words": sum(field["words"] for field in fields),
            "differing": sum(field["differing"] for field in fields)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.directory)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "captures"}))
    return int(report["words"] == 0 or report["differing"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
