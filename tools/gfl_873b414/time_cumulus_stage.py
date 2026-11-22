"""The whole cumulus stage, GFL (WRF v4.8.0) against GFDRV (WRF 4.6.1/4.7.1), through the engine adapters.

usage: time_cumulus_stage.py PRE [REPORT.json] [CALLS] [OUTPUTS.npz]
One captured native call's columns become the engine's [nz, ny, nx] state, the
forcing lanes the driver hands both adapters, and the surface fields. Each
adapter call is the whole stage the physics driver times as cumulus: packing
from the WRF layout, the kernel, and the outputs back on the grid
(rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten, raincv). Calls
alternate GFL / GFDRV, synchronized around each call (a timing run only).
OUTPUTS.npz keeps the GFL adapter's words so two trees can be compared word
for word (the lane-placement adapter against the column-placement one).
"""
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def main():
    import cupy as cp
    root = Path.cwd()
    sys.path.insert(0, str(root / 'tools/gfl_873b414'))
    sys.path.insert(0, str(root / 'tests'))
    from replay_gfl import read_pre
    from _run_config_fake import run_config_fake
    from woof.core import gf, gfl
    d, a, shape = read_pre(sys.argv[1])
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    nz = d['kte']
    ny, nx = j1 - j0, i1 - i0

    def vol(name, k=nz):
        return cp.ascontiguousarray(cp.asarray(a[name][j0:j1, :k, i0:i1]).transpose(1, 0, 2))

    def srf(name, dtype=cp.float32):
        return cp.ascontiguousarray(cp.asarray(a[name][j0:j1, i0:i1]).astype(dtype))

    volume = vol('p')
    state = NS(p=volume, ht=srf('ht'), w=vol('w', nz + 1), qi=None)
    atmosphere = {'u': vol('u'), 'v': vol('v'), 'temperature': vol('t'), 'qv': vol('q'), 'pressure': volume,
                  'exner': vol('pi'), 'rho': vol('rho'), 'dz': vol('dz8w'), 'p_interface': vol('p8w', nz + 1)}
    fields = {'hfx': srf('hfx'), 'qfx': srf('qfx'), 'xland': srf('xland'), 'gsw': srf('gsw'),
              'kpbl': srf('kpbl', cp.int32)}
    driver = NS(rthratenlw=vol('rthraten'), rthratensw=cp.zeros_like(volume), gf_rthdynten=vol('rthften'),
                gf_rqvdynten=vol('rqvften'), gf_rthblten=vol('rthblten'), gf_rqvblten=vol('rqvblten'))
    import os
    if os.environ.get('GFL_STAGE_MASK'):
        # The driver's active-column list (GF's gf_active_column): every third column held out.
        driver.gf_active_column = (cp.arange(ny * nx) % 3 != 0).reshape(ny, nx)
    cfg = run_config_fake(dt=float(d['dt']), dx=float(d['dxcell']), ishallow=int(d['ishallow_g3']),
                          clos_choice=int(d['ichoice_deep']), lightning_option=0, cu_rad_feedback=False)
    new, old = gfl.GrellFreitasLi(), gf.GrellFreitas()
    new.bind_driver(driver); old.bind_driver(driver)

    def call(adapter):
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        result = adapter(atmosphere=atmosphere, fields=dict(fields), state=state, cfg=cfg)
        cp.cuda.Device().synchronize()
        return time.perf_counter() - t, result

    _, first = call(new); call(old)  # compile, size scratch
    tn, to = [], []
    for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 9):
        tn.append(call(new)[0]); to.append(call(old)[0])
    out = {'pre': sys.argv[1], 'columns': ny * nx, 'nz': nz,
           'device': cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode(),
           'gfl_stage_ms': [round(x * 1e3, 3) for x in tn], 'gfdrv_stage_ms': [round(x * 1e3, 3) for x in to],
           'gfl_stage_median_ms': round(statistics.median(tn) * 1e3, 3),
           'gfdrv_stage_median_ms': round(statistics.median(to) * 1e3, 3)}
    out['ratio_gfl_over_gfdrv'] = round(out['gfl_stage_median_ms'] / out['gfdrv_stage_median_ms'], 3)
    import os
    out['numerics'] = os.environ.get('GPUWM_NUMERICS', 'fast'); out['wrf_exact'] = os.environ.get('GPUWM_WRF_EXACT', '0')
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
    if len(sys.argv) > 4:
        words = {name: cp.asnumpy(getattr(first, name)) for name in ('rthcuten', 'rqvcuten', 'rqccuten', 'rainc')
                 if getattr(first, name, None) is not None}
        if first.rqicuten is not None:
            words['rqicuten'] = cp.asnumpy(first.rqicuten)
        for name, value in (first.history_rates or {}).items():
            words['history_' + name] = cp.asnumpy(value)
        np.savez(sys.argv[4], **words)


if __name__ == '__main__':
    main()
