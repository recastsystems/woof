"""Compare every TMN state word to the unmodified WRF 4.6.1 subroutine.

Run on a queued GPU with --oracle libtmn.so --out receipt.json.
Raw arrays remain on the box. They are not a delivery artifact.
"""
import argparse
import ctypes as C
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = ("tmn", "tlag", "tyr", "tyra", "tdly", "nday", "nyear")


def initial(ncol=64, lagday=150):
    tmn = np.linspace(220., 315., ncol, dtype=np.float32).reshape(1, ncol)
    return dict(tmn=tmn.copy(), tyr=tmn.copy(), tyra=np.zeros_like(tmn),
                tdly=np.zeros_like(tmn), tlag=np.repeat(tmn[None], lagday, axis=0),
                nday=np.zeros(1, np.float32), nyear=np.ones(1, np.int32))


def oracle_function(path):
    library = C.CDLL(str(Path(path).resolve()))
    func = library.column_tmn
    func.argtypes = [C.c_int, C.c_int] + [C.c_void_p] * 8 + [C.c_float, C.c_float, C.c_int]
    return func


def oracle_step(func, fields, tsk, julian, dt, year):
    func(tsk.size, fields["tlag"].shape[0], tsk.ctypes.data,
         *[fields[k].ctypes.data for k in ("tmn", "tlag", "tyr", "tyra", "tdly", "nday", "nyear")],
         np.float32(julian), np.float32(dt), year)


def differences(a, b):
    words_a, words_b = a.view(np.uint32), b.view(np.uint32)
    count = int(np.count_nonzero(words_a != words_b))
    # The oracle corpus has only nonnegative values. Ordered words equal ULP distance.
    return count, int(np.max(np.abs(words_a.astype(np.int64) - words_b.astype(np.int64)))), float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def validate(oracle, steps=370 * 24, mutate=False):
    import cupy as cp
    from woof.core.tmn_update import launch
    from woof.core.kernels import module_source, module_options
    host = initial()
    device = {k: cp.asarray(v) for k, v in host.items()}
    func = oracle_function(oracle)
    maxima = {k: dict(different_words=0, max_ulp=0, max_abs=0.) for k in FIELDS}
    checkpoints = {}
    resumed = {}
    checked = 0
    start = datetime(2023, 8, 1)
    module = None
    if mutate:
        src = module_source("tmn_update")
        assert "0.6f" in src
        src = src.replace("0.6f", "0.600000083446502685546875f")
        module = cp.RawModule(code=src, options=module_options("tmn_update"))
    for step in range(steps):
        valid = start + timedelta(hours=step)
        # WRF surface_driver julian_in is one based, from domain_clock_get.
        julian = np.float32(valid.timetuple().tm_yday + valid.hour / 24.)
        # Alternate land/water columns and polar/tropical temperatures.
        # TMNUPDATE itself has no land mask, so all columns update.
        tsk = (np.linspace(210., 325., 64, dtype=np.float32)
               + np.float32((step % 24 - 12) * .125)).reshape(1, 64)
        oracle_step(func, host, tsk, julian, 3600., valid.year)
        gpu_tsk = cp.asarray(tsk)
        launch(device, gpu_tsk, julian=julian, dt=3600., year=valid.year, module=module)
        for saved in resumed.values():
            launch(saved, gpu_tsk, julian=julian, dt=3600., year=valid.year)
        if step in (13, 151 * 24 + 13):
            # Persist all seven fields and restore them into fresh allocations.
            snapshots = {k: cp.asnumpy(v) for k, v in device.items()}
            checkpoints[str(step)] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in snapshots.items()}
            resumed[str(step)] = {k: cp.asarray(v) for k, v in snapshots.items()}
        checked += 1
        for k in FIELDS:
            result = cp.asnumpy(device[k])
            count, ulp, absolute = differences(host[k], result)
            row = maxima[k]
            row["different_words"] += count
            row["max_ulp"] = max(row["max_ulp"], ulp)
            row["max_abs"] = max(row["max_abs"], absolute)
        if mutate and any(row["different_words"] for row in maxima.values()):
            return dict(mutation_caught=True, first_check=step, fields=maxima)
    for key, saved in resumed.items():
        checkpoints[key]["resumed_equal"] = all(cp.asnumpy(saved[k]).tobytes() == cp.asnumpy(device[k]).tobytes() for k in FIELDS)
    hashes = {k: hashlib.sha256(cp.asnumpy(v).tobytes()).hexdigest() for k, v in device.items()}
    return dict(columns=64, lagday=150, steps=steps, dt=3600., checks=checked,
                years=[2023, 2024], land_water="alternating labels; routine updates all columns",
                fields=maxima, restarts=checkpoints, full_field_sha256=hashes,
                passed=all(row["different_words"] == 0 for row in maxima.values())
                and all(row["resumed_equal"] for row in checkpoints.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--steps", type=int, default=370*24)
    args = parser.parse_args()
    result = validate(args.oracle, args.steps, args.mutate)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", result.get("mutation_caught", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
