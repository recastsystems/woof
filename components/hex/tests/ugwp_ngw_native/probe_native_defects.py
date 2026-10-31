"""Preliminary standalone defect measurements; real forecast witnesses still required.

Build strict and --bounds-check work directories with build_oracle.py first.
Run under the lane CPU cpuset. This tool neither excludes words nor declares
native undefined behavior: pool/NATIVE-DEFECT-POLICY.md requires the real strict
executable, both hosts and several rank counts before those decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def probe(work: Path, kind: str) -> None:
    import numpy as np

    sys.argv = ["compare_standalone.py", "--work", str(work), "--receipt", str(work / "probe-valid.json")]
    values = runpy.run_path(str(Path(__file__).with_name("compare_standalone.py")))
    inputs = [array.copy() for array in values["inputs"]]
    output = values["arrays"]()
    day = 80.0
    if kind == "launch":
        inputs[10].fill(25000.0)
    elif kind == "after_last_day":
        day = 367.0
    elif kind == "before_first_day":
        day = -1.0
    pointer = values["ptr"]
    values["native"](values["nl"], values["nc"], 1.e-25, .00025, day,
                     *[pointer(a) for a in inputs[:6]], 7, 7,
                     *[pointer(a) for a in inputs[6:]], *[pointer(a) for a in output[:9]],
                     pointer(output[9][0]), pointer(output[9][1]),
                     pointer(output[10][0]), pointer(output[10][1]))
    print("PROBE_RESULT:" + json.dumps({"kind": kind, "outputs": {
        name: {"sha256": hashlib.sha256(array.tobytes()).hexdigest(),
               "nonfinite_words": int(np.sum(~np.isfinite(array)))}
        for name, array in zip(values["names"], output)}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--bounds-work", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--probe", choices=("launch", "after_last_day", "before_first_day"))
    args = parser.parse_args()
    if args.probe:
        probe(args.work.resolve(), args.probe)
    else:
        if args.bounds_work is None or args.receipt is None:
            parser.error("parent mode requires --bounds-work and --receipt")
        rows = []
        for build, work in (("strict", args.work), ("bounds", args.bounds_work)):
            for kind in ("launch", "after_last_day", "before_first_day"):
                for repeat in range(2):
                    cmd = [sys.executable, __file__, "--work", str(work.resolve()), "--probe", kind]
                    try:
                        result = subprocess.run(cmd, text=True, capture_output=True, timeout=5)
                        row = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                    except subprocess.TimeoutExpired:
                        row = {"exit_code": "timeout"}
                    rows.append({"build": build, "kind": kind, "repeat": repeat, **row})
        args.receipt.write_text(json.dumps({
            "schema": "p15.native-defect-probe/v1", "acceptance": False, "real_native_forecast": False,
            "status": "Standalone only; real strict executable host/rank witnesses remain required",
            "binary_hashes": {key: hashlib.sha256((work / "native/liboracle.so").read_bytes()).hexdigest()
                              for key, work in (("strict", args.work), ("bounds", args.bounds_work))},
            "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runs": rows,
        }, indent=2) + "\n")
