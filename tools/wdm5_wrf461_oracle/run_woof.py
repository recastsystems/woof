"""Run WOOF's WDM5 GPU kernel on an oracle input file.

usage: python3 run_woof.py INPUT OUTPUT [INPUT OUTPUT ...]

Drives the production launcher ``woof.core.wdm6.launch_wdm5`` (the call
the WDM family adapter ``woof.core.wdm6.apply`` makes every microphysics
step at mp_physics=14) and the production reflectivity launcher
``woof.core.refl.launch_refl10cm_wdm5`` on exactly the fields the WRF
oracle reads, as (nz, ny=1, nx=ncol) device arrays, and writes the same
layout the Fortran driver writes.  Step s feeds step s+1, as consecutive WRF
calls do.

The arithmetic follows the process environment: GPUWM_WRF_EXACT=1 selects
the strict build, unset the default build.  Effective radii are written as
WOOF stores them, in microns; compare.py scales WRF's metres by the one
float32 multiply WRF's RRTMG applies (re*1.E6) before comparing words.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import read_input, write_output  # noqa: E402


def run(inp: Path, out: Path) -> dict:
    import cupy as cp
    from woof.wrf_exact import ENABLED, install
    install()
    from woof.core.wdm6 import launch_wdm5
    from woof.core.refl import launch_refl10cm_wdm5

    case = read_input(inp)
    nz, ncol = case.nz, case.ncol
    shape = (nz, 1, ncol)

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a, np.float32).reshape(shape))

    lv = case.levels
    theta, qv, qc, qr = dev(lv["th"]), dev(lv["qv"]), dev(lv["qc"]), dev(lv["qr"])
    qi, qs = dev(lv["qi"]), dev(lv["qs"])
    nn, nc, nr = dev(lv["nn"]), dev(lv["nc"]), dev(lv["nr"])
    rho, pii, p, dz = dev(lv["den"]), dev(lv["pii"]), dev(lv["p"]), dev(lv["delz"])
    s2 = (1, ncol)
    sf = case.surface
    rainnc = cp.asarray(sf["rain"].reshape(s2).copy())
    snownc = cp.asarray(sf["snow"].reshape(s2).copy())
    rainncv = cp.zeros(s2, cp.float32)
    snowncv = cp.zeros(s2, cp.float32)
    sr = cp.zeros(s2, cp.float32)
    effc = cp.zeros(shape, cp.float32)
    effi = cp.zeros(shape, cp.float32)
    effs = cp.zeros(shape, cp.float32)
    refl = cp.zeros(shape, cp.float32)
    t = cp.zeros(shape, cp.float32)

    steps = []
    for _ in range(case.nsteps):
        launch_wdm5(theta, qv, qc, qr, qi, qs, nn, nc, nr, rho, pii, p, dz,
                    rainnc, rainncv, snownc, snowncv, sr, float(case.delt),
                    effc=effc, effi=effi, effs=effs)
        # The WRF driver's t1d = th*pii (module_mp_wdm5.F:274), as the WDM
        # family adapter forms refl_t.
        t[...] = theta * pii
        launch_refl10cm_wdm5(qv, qr, nr, qs, t, p, refl)
        cp.cuda.runtime.deviceSynchronize()

        def host(a):
            return cp.asnumpy(a).reshape(nz, ncol)

        levels = {"th": host(theta), "qv": host(qv), "qc": host(qc),
                  "qr": host(qr), "qi": host(qi), "qs": host(qs),
                  "nn": host(nn), "nc": host(nc), "nr": host(nr),
                  "refl_10cm": host(refl), "re_cloud": host(effc),
                  "re_ice": host(effi), "re_snow": host(effs)}
        surface = {"rain": cp.asnumpy(rainnc).ravel(),
                   "rainncv": cp.asnumpy(rainncv).ravel(),
                   "snow": cp.asnumpy(snownc).ravel(),
                   "snowncv": cp.asnumpy(snowncv).ravel(),
                   "sr": cp.asnumpy(sr).ravel()}
        steps.append((levels, surface))
    write_output(out, steps)
    dev_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return {"input": str(inp), "output": str(out), "strict": bool(ENABLED),
            "device": dev_props["name"].decode(),
            "cc": f"{dev_props['major']}.{dev_props['minor']}",
            "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
            "cupy": cp.__version__}


def main(argv):
    if len(argv) < 3 or len(argv) % 2 != 1:
        print(__doc__, file=sys.stderr)
        return 2
    receipts = [run(Path(argv[i]), Path(argv[i + 1]))
                for i in range(1, len(argv), 2)]
    print(json.dumps(receipts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
