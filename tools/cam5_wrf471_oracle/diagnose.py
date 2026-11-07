#!/usr/bin/env python3
"""One stock-module diagnostic per CAM scheme. This is not a real-input gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def compare(got: dict, expected: dict) -> dict:
    if got.keys() != expected.keys():
        raise ValueError("output field coverage differs")
    fields = {}
    for name, ref in expected.items():
        a, b = np.asarray(got[name]), np.asarray(ref)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"output ABI differs for {name}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"nonfinite output for {name}")
        x, y = a.ravel().view(np.uint32), b.ravel().view(np.uint32)
        fields[name] = {"words32": int(x.size), "differ": int(np.count_nonzero(x != y))}
    return {"fields": fields,
            "words32": sum(v["words32"] for v in fields.values()),
            "differ": sum(v["differ"] for v in fields.values())}


def timed(call, cp):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    wall = time.perf_counter()
    start.record()
    value = call()
    end.record()
    end.synchronize()
    return value, {"cuda_ms": float(cp.cuda.get_elapsed_time(start, end)),
                   "host_ms": (time.perf_counter() - wall) * 1000.0,
                   "samples": 1, "includes_launcher_status_readback": True,
                   "comparison_scheme": None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cu7", type=Path, required=True, help="stock native output stem")
    p.add_argument("--shcu2", type=Path, required=True, help="stock native output stem")
    p.add_argument("--mp11", type=Path, required=True, help="legacy synthetic deck directory")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import cupy as cp
    from woof.core import camzm, cammgmp, uwshcu
    from woof.verify import camzm_oracle as zm
    from woof.verify import uwshcu_oracle as sh
    from tools.cammgmp_wrf461_oracle import compare_gpu as mg
    from tools.cammgmp_wrf461_oracle.oracle_io import read_deck, read_outputs, read_helpers

    # Compile and upload constants before timing any of the launches.
    for module in (camzm, uwshcu):
        module._module()
    cammgmp._module()
    rows = {}
    fx = zm.load(a.cu7)
    tile = zm.tile(fx)
    s = zm.step_arrays(fx, 1)
    inputs = {n: cp.asarray(np.ascontiguousarray(s[n][:, None, :]))
              for n in zm.MASS_INPUTS + zm.FULL_INPUTS}
    inputs.update({n: cp.asarray(np.ascontiguousarray(s[n][None, :]))
                   for n in zm.SURFACE_INPUTS + zm.INT_SURFACE_INPUTS})
    out, timing = timed(lambda: camzm.camzm_step(
        inputs, dt=tile["dt"], dx=tile["dx"], cudt=tile["cudt"],
        itimestep=s["itimestep"], bl_pbl_physics=tile["bl_pbl_physics"],
        sf_sfclay_physics=tile["sf_sfclay_physics"], diagnostics=True), cp)
    got = {n: cp.asnumpy(v)[:, 0, :] if v.ndim == 3 else cp.asnumpy(v)[0]
           for n, v in out.items()}
    rows["cu7"] = compare(got, s["out"]) | {"timing": timing, "step": 1}

    fx = sh.load(a.shcu2)
    s = sh.step_arrays(fx, 1)
    mass = lambda n: cp.asarray(np.ascontiguousarray(s[n].T)[:, None, :])
    sfc = lambda n: cp.asarray(np.ascontiguousarray(s[n])[None, :])
    inputs = {n: mass(n) for n in ("u", "v", "t", "qv", "qc", "qi", "qnc", "qni",
                                  "p", "z", "pi")}
    inputs.update({"dlf_in": mass("dlf_in"), "cmfmc_in": mass("cmfmc_in")})
    inputs.update({n: mass(n) for n in ("p8w", "z_at_w", "tke_pbl")})
    inputs.update({n: sfc(n) for n in ("pblh", "ht", "cutop_in", "cubot_in", "rliq_in")})
    carried = {"cush": sfc("cush_in")}
    out, timing = timed(lambda: uwshcu.uwshcu_step(
        inputs, carried, dt=s["dt"], has_qnc=s["has_qnc"], has_qni=s["has_qni"]), cp)
    out["cush"] = carried["cush"]
    names = sh.STEP_MASS_OUTPUTS + sh.STEP_FULL_OUTPUTS + sh.STEP_SURFACE_OUTPUTS
    got = {n: cp.asnumpy(out[n])[:, 0, :].T if out[n].ndim == 3
           else cp.asnumpy(out[n])[0] for n in names}
    rows["shcu2"] = compare(got, {n: s[n] for n in names}) | {"timing": timing, "step": 1}

    deck = read_deck(a.mp11 / "cases.bin")[:1]
    # The legacy deck contains generated soundings. Its stock file is used
    # explicitly; neither a corrected nor an owned-math file is a native gate.
    ref = read_outputs(a.mp11 / "pristine-out.bin")[:1]
    out, timing = timed(lambda: mg.run_group(deck[0]), cp)
    report = mg.grade(ref, [out])
    rows["mp11"] = {"fields": report, "words32": sum(v["words"] for v in report.values()),
                    "differ": sum(v["differ"] for v in report.values()), "timing": timing,
                    "legacy_source_version": "4.6.1", "group": 0}
    helpers = read_helpers(a.mp11 / "helpers.bin")
    x = cp.asarray(helpers["erf_x"])
    y = cp.empty_like(x)
    status = cp.zeros(1, dtype=cp.int32)
    cammgmp._module().get_function("cammgmp_helpers")(
        ((x.size + 127) // 128,), (128,), (x, y, np.int32(x.size), np.int32(3), status))
    rows["mp11"]["native_erf"] = compare({"erf": cp.asnumpy(y)}, {"erf": helpers["erf"]})
    rows["mp11"]["native_erf"]["status"] = int(status.get()[0])
    from woof.core.kernels import module_source, module_options
    receipt = {"kind": "synthetic-column-diagnostic", "real_native_capture": False,
               "whole_forecast": False, "fold_ready": False,
               "strict_numerics": True, "cupy_version": cp.__version__, "schemes": rows,
               "modules": {n: {"source_sha256": hashlib.sha256(module_source(n).encode()).hexdigest(),
                                "options": module_options(n)}
                           for n in ("camzm", "uwshcu", "cammgmp")},
               "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(str(a.cu7) + ".bin"), Path(str(a.shcu2) + ".bin"),
                                    a.mp11 / "cases.bin", a.mp11 / "pristine-out.bin",
                                    a.mp11 / "helpers.bin")}}
    a.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({n: {k: v[k] for k in ("words32", "differ", "timing")}
                      for n, v in rows.items()}, indent=2))
    raise SystemExit(1 if any(v["differ"] for v in rows.values())
                     or rows["mp11"]["native_erf"]["differ"]
                     or rows["mp11"]["native_erf"]["status"] else 0)


if __name__ == "__main__":
    main()
