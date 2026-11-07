#!/usr/bin/env python3
"""Compile and execute the new leaves with the default fast numerics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replay_levels import analytic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core import level_diagnostics as levels, nwp_windows, nwp_hail
    from woof.core.kernels import module_source, module_options
    if wrf_exact.ENABLED:
        raise ValueError("this check requires default fast numerics")
    a = {k: cp.asarray(v) for k, v in analytic().items()}
    nz, ny, nx = a["t"].shape
    inputs = levels.LevelInputs(*(a[k] for k in ("u", "v", "t", "qv", "ph", "phb", "pp", "pb")))
    p = cp.empty((8, a["p_levels"].size, ny, nx), dtype="f4")
    z = cp.empty((9, a["z_levels"].size, ny, nx), dtype="f4")
    levels.pressure_levels(inputs, a["p"], a["pw"], a["p_levels"], p)
    levels.height_levels(inputs, a["terrain"], a["z_levels"], z)
    qg, rho = cp.full((nz, ny, nx), .003, dtype="f4"), cp.ones((nz, ny, nx), dtype="f4")
    surface = cp.zeros((ny, nx), dtype="f4")
    out = cp.zeros((12, ny, nx), dtype="f4")
    nwp_windows.update(a["w"], a["pw"], cp.linspace(1., 0., nz+1, dtype="f4"), qg, rho,
                       a["ph"], a["phb"], surface, surface, rho, out,
                       dt=20., xtime=1./3., history_minutes=1, step=1)
    nwp_hail.update(qg, rho, out[7], out[8], mp_physics=6)
    outputs = {name: cp.asnumpy(value) for name, value in (("pld", p), ("zld", z), ("nwp", out))}
    receipt = dict(qualification="compile and finite-output smoke only; no native parity or speed claim",
                   strict=wrf_exact.ENABLED,
                   gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                   results={name: dict(words=int(value.size), finite=bool(np.isfinite(value).all()))
                            for name, value in outputs.items()},
                   modules={name: dict(source_sha256=hashlib.sha256(module_source(name).encode()).hexdigest(),
                                       options=list(wrf_exact.effective_options(module_options(name))))
                            for name in ("level_diagnostics", "nwp_windows", "nwp_hail")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt["results"]))
    if not all(r["finite"] for r in receipt["results"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
