"""Run WOOF's WDM7 GPU kernel on an oracle input file.

usage: python3 run_woof.py INPUT OUTPUT [INPUT OUTPUT ...]

Drives the production launcher ``woof.core.wdm6.launch_wdm7`` (the call
``woof.core.wdm6.apply`` makes every microphysics step at mp_physics = 26)
and the production reflectivity launcher ``woof.core.refl.
launch_refl10cm_wdm6`` at ``hail_opt = 0`` (refl10cm_wdm7 is refl10cm_wdm6
byte for byte, with WDM6's hail_opt = 0 radar constants) on exactly the
fields the WRF oracle reads, as (nz, ny=1, nx=ncol) device arrays, and
writes run_wdm7.F90's layout.  Step s feeds step s+1, as consecutive WRF
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
# The WDM6 fixture can already have imported its oracle_io in this process.
# Bind WDM7's layout by its path so qh and the hail accumulators cannot be
# silently read with WDM6's shorter record.
import importlib.util
_io_spec = importlib.util.spec_from_file_location(
    "wdm7_run_oracle_io", Path(__file__).resolve().parent / "oracle_io.py")
_io = importlib.util.module_from_spec(_io_spec)
sys.modules[_io_spec.name] = _io
_io_spec.loader.exec_module(_io)
read_input, write_output = _io.read_input, _io.write_output


def run(inp: Path, out: Path) -> dict:
    import cupy as cp
    from woof.wrf_exact import ENABLED, install
    install()
    from woof.core.wdm6 import launch_wdm7
    from woof.core.refl import launch_refl10cm_wdm6

    case = read_input(inp)
    nz, ncol = case.nz, case.ncol
    shape = (nz, 1, ncol)

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a, np.float32).reshape(shape))

    lv = case.levels
    theta, qv, qc, qr = dev(lv["th"]), dev(lv["qv"]), dev(lv["qc"]), dev(lv["qr"])
    qi, qs, qg, qh = dev(lv["qi"]), dev(lv["qs"]), dev(lv["qg"]), dev(lv["qh"])
    nn, nc, nr = dev(lv["nn"]), dev(lv["nc"]), dev(lv["nr"])
    rho, pii, p, dz = dev(lv["den"]), dev(lv["pii"]), dev(lv["p"]), dev(lv["delz"])
    s2 = (1, ncol)
    sf = case.surface
    xland = cp.asarray(sf["xland"].reshape(s2))
    rainnc = cp.asarray(sf["rain"].reshape(s2).copy())
    snownc = cp.asarray(sf["snow"].reshape(s2).copy())
    graupelnc = cp.asarray(sf["graupel"].reshape(s2).copy())
    hailnc = cp.asarray(sf["hail"].reshape(s2).copy())
    rainncv = cp.zeros(s2, cp.float32)
    snowncv = cp.zeros(s2, cp.float32)
    graupelncv = cp.zeros(s2, cp.float32)
    hailncv = cp.zeros(s2, cp.float32)
    sr = cp.zeros(s2, cp.float32)
    effc = cp.zeros(shape, cp.float32)
    effi = cp.zeros(shape, cp.float32)
    effs = cp.zeros(shape, cp.float32)
    refl = cp.zeros(shape, cp.float32)
    t = cp.zeros(shape, cp.float32)

    steps = []
    for _ in range(case.nsteps):
        launch_wdm7(theta, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr, rho, pii,
                    p, dz, xland, rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, hailnc, hailncv, sr,
                    float(case.delt), effc=effc, effi=effi, effs=effs)
        # The WRF driver's t1d = th*pii (module_mp_wdm7.F:566), as
        # woof.core.wdm6.apply forms refl_t.
        t[...] = theta * pii
        launch_refl10cm_wdm6(qv, qr, nr, qs, qg, t, p, refl, hail_opt=0, mp_physics=26)
        cp.cuda.runtime.deviceSynchronize()

        def host(a):
            return cp.asnumpy(a).reshape(nz, ncol)

        levels = {"th": host(theta), "qv": host(qv), "qc": host(qc),
                  "qr": host(qr), "qi": host(qi), "qs": host(qs),
                  "qg": host(qg), "qh": host(qh), "nn": host(nn),
                  "nc": host(nc), "nr": host(nr), "refl_10cm": host(refl)}
        for name, arr in (("re_cloud", effc), ("re_ice", effi),
                          ("re_snow", effs)):
            levels[name] = host(arr)
        surface = {"rain": cp.asnumpy(rainnc).ravel(),
                   "rainncv": cp.asnumpy(rainncv).ravel(),
                   "snow": cp.asnumpy(snownc).ravel(),
                   "snowncv": cp.asnumpy(snowncv).ravel(),
                   "graupel": cp.asnumpy(graupelnc).ravel(),
                   "graupelncv": cp.asnumpy(graupelncv).ravel(),
                   "hail": cp.asnumpy(hailnc).ravel(),
                   "hailncv": cp.asnumpy(hailncv).ravel(),
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
