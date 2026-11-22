"""Kernel time of WRF v4.8.0 GFL against today's cu_physics=3 kernel (GFDRV, gf.cu) on the same columns.

GFL is timed two ways: ``gfl`` is launch_columns on [ncol, levels] fields (it moves
them into the lane placement and back, two fused copies), ``gfl_lane`` is
launch_lane_columns on fields already in the lane placement, as the forecast
adapter packs them (the like-for-like row beside GFDRV, whose fields are packed
before its timer too).

usage: time_gf_vs_gfl.py PRE [REPORT.json]
Both schemes get the captured columns of one native call (same inputs, same
card, same process, alternating), fast numerics unless the environment says
otherwise. Reports the median of N warm calls each, synchronized around each
call (a timing run only; no product path syncs this way).
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def main():
    import cupy as cp
    sys.path.insert(0, str(Path(__file__).parent))
    from replay_gfl import read_pre, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT, LEVEL_INOUT
    from woof.core import gfl_873b414 as gfl
    from woof.core import gf as gfmod
    pre = sys.argv[1]
    d, a, shape = read_pre(pre)
    ni, nk, nj = shape
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    nz = d['kte']; ncol = (i1 - i0) * (j1 - j0)

    # Repacking runs on the GPU; the host only reads the file.
    def lev(x, k=nk):
        return cp.ascontiguousarray(cp.asarray(x)[j0:j1, :k, i0:i1].transpose(0, 2, 1).reshape(ncol, k))

    def srf(x):
        return cp.ascontiguousarray(cp.asarray(x)[j0:j1, i0:i1].reshape(ncol))

    # ---- GFL (WRF v4.8.0) fields
    base = {n: lev(a[n]) for n in LEVEL_IN + ('rqvften', 'rthften', 'rthblten', 'rqvblten', 'rthraten')}
    base.update({n: srf(a[n]) for n in SURF_IN + ('kpbl',)})
    for n, typ in (('itimestep', cp.int32), ('ichoice_deep', cp.int32), ('ishallow_g3', cp.int32),
                   ('dt', cp.float32), ('dxcell', cp.float32), ('xlv', cp.float32), ('cp', cp.float32),
                   ('g', cp.float32), ('r_v', cp.float32)):
        base[n] = cp.asarray([d[n]], dtype=typ)
    inout0 = {n: lev(a[n]) for n in ('gdc', 'gdc2') + LEVEL_INOUT}
    inout0.update({n: srf(a[n]) for n in SURF_INOUT + INT_INOUT})

    vertical = (d['kms'], d['kme'], d['kds'], d['kde'], d['kts'], d['kte'])

    def run_gfl():
        f = dict(base)
        f.update({k: v.copy() for k, v in inout0.items()})
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        gfl.launch_columns(f, ncol=ncol, nz=nz, vertical=vertical)
        cp.cuda.Device().synchronize(); return time.perf_counter() - t

    lane = hasattr(gfl, 'launch_lane_columns')
    if lane:
        levels = d['kme'] - d['kms'] + 1
        lane_base = dict(base)
        for n0, v0 in list(base.items()) + list(inout0.items()):
            if v0.ndim == 2:
                padded = cp.zeros((gfl.lane_shape(ncol, levels)[0] * gfl.LANES, levels), dtype=v0.dtype)
                padded[:ncol] = v0
                lane_base[n0] = cp.ascontiguousarray(
                    padded.reshape(-1, gfl.LANES, levels).transpose(0, 2, 1))
            elif n0 in inout0:
                lane_base[n0] = v0

    def run_gfl_lane():
        f = dict(lane_base)
        f.update({k: lane_base[k].copy() for k in inout0})
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        gfl.launch_lane_columns(f, ncol=ncol, nz=nz, vertical=vertical)
        cp.cuda.Device().synchronize(); return time.perf_counter() - t

    # ---- GFDRV (gf.cu), packed exactly as woof/core/gf.py packs it
    lvin = cp.empty((ncol, len(gfmod._IN_LEV), nz), dtype=cp.float32)
    names = {'u': 'u', 'v': 'v', 'w': 'w', 't': 't', 'qv': 'q', 'p': 'p', 'pi': 'pi', 'rho': 'rho', 'dz8w': 'dz8w',
             'p8w': 'p8w', 'rthften': 'rthften', 'rqvften': 'rqvften', 'rthraten': 'rthraten',
             'rthblten': 'rthblten', 'rqvblten': 'rqvblten'}
    for slot, name in enumerate(gfmod._IN_LEV):
        lvin[:, slot, :] = lev(a[names[name]])[:, :nz]
    scin = cp.zeros((ncol, len(gfmod._IN_SCA)), dtype=cp.float32)
    scin[:, 0] = srf(a['ht']); scin[:, 1] = srf(a['hfx']); scin[:, 2] = srf(a['qfx']); scin[:, 3] = srf(a['xland'])
    scin[:, 4] = d['dt']; scin[:, 5] = d['dxcell']
    iin = cp.empty((ncol, 3), dtype=cp.int32)
    iin[:, 0] = srf(a['kpbl']); iin[:, 1] = d['ishallow_g3']; iin[:, 2] = d['ichoice_deep']
    levo = cp.empty((ncol, len(gfmod._OUT_LEV), nz), dtype=cp.float32)
    sca = cp.empty((ncol, len(gfmod._OUT_SCA)), dtype=cp.float32)
    isc = cp.empty((ncol, len(gfmod._OUT_ISCA)), dtype=cp.int32)
    fn = gfmod._gf_module(nz).get_function('gf_gfdrv_stage')
    tile = gfmod.gf_tile_columns(fn, ncol)
    ws = cp.empty(gfmod.gf_workspace_floats(nz, tile), dtype=cp.float32)

    def run_gf():
        cp.cuda.Device().synchronize(); t = time.perf_counter()
        for lo in range(0, ncol, tile):
            hi = min(lo + tile, ncol)
            fn((((hi - lo) + gfmod.GF_BLOCK - 1) // gfmod.GF_BLOCK,), (gfmod.GF_BLOCK,),
               (lvin[lo:hi], scin[lo:hi], iin[lo:hi], levo[lo:hi], sca[lo:hi], isc[lo:hi], ws,
                np.int32(0), np.int32(hi - lo), np.int32(nz)))
        cp.cuda.Device().synchronize(); return time.perf_counter() - t

    run_gfl(); run_gf()  # compile, size scratch
    if lane:
        run_gfl_lane()
    tg, tl, tn = [], [], []
    for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 7):
        tl.append(run_gfl()); tg.append(run_gf())
        if lane:
            tn.append(run_gfl_lane())
    out = {'pre': pre, 'columns': ncol, 'nz': nz, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
           'gfl_873b414_ms': [round(x * 1e3, 3) for x in tl], 'gf_wrf461_ms': [round(x * 1e3, 3) for x in tg],
           'gfl_median_ms': round(statistics.median(tl) * 1e3, 3), 'gf_median_ms': round(statistics.median(tg) * 1e3, 3)}
    out['ratio_gfl_over_gf'] = round(out['gfl_median_ms'] / out['gf_median_ms'], 3)
    if lane:
        out['gfl_lane_ms'] = [round(x * 1e3, 3) for x in tn]
        out['gfl_lane_median_ms'] = round(statistics.median(tn) * 1e3, 3)
        out['ratio_gfl_lane_over_gf'] = round(out['gfl_lane_median_ms'] / out['gf_median_ms'], 3)
    import os
    out['numerics'] = os.environ.get('GPUWM_NUMERICS', 'fast'); out['wrf_exact'] = os.environ.get('GPUWM_WRF_EXACT', '0')
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
