"""GFL lane kernel: registers, local memory and time for each GFL_MIN_BLOCKS (launch bounds), one card.

usage: probe_blocks.py PRE REPORT.json compile|time MIN_BLOCKS...
compile: build each variant (run one process per variant in parallel; the disk cache keeps them).
time: the kernel-only time of each variant on fields already in the lane placement, alternating
with GFDRV, 7 calls each, plus each variant's output words against variant 1 (all must be 0 different).
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _attr(function, which):
    """CU_FUNC_ATTRIBUTE: 4 = registers per thread, 3 = local bytes per thread."""
    import cupy as cp
    if hasattr(function, 'attributes'):
        return function.attributes['num_regs' if which == 4 else 'local_size_bytes']
    return cp.cuda.driver.funcGetAttribute(which, function.ptr)


def variant(gfl, device, blocks):
    from woof.core.kernels import compile_no_ftz
    import cupy as cp
    with cp.cuda.Device(device):
        gfl._spill_heap()
        source = f'#define GFL_MIN_BLOCKS {blocks}\n' + gfl.cuda_source()
        return compile_no_ftz(source, module_key=f'{gfl.MODULE_KEY}.mb{blocks}', filename=f'gfl_lanes_mb{blocks}.cu',
                              device=device)


def main():
    import cupy as cp
    sys.path.insert(0, str(Path.cwd() / 'tools/gfl_873b414'))
    from woof.core import gfl_873b414 as gfl
    mode, blocks = sys.argv[3], [int(x) for x in sys.argv[4:]]
    device = cp.cuda.runtime.getDevice()
    if mode == 'compile':
        for b in blocks:
            variant(gfl, device, b)
        return
    from replay_gfl import read_pre, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT, LEVEL_INOUT
    from woof.core import gf as gfmod
    d, a, shape = read_pre(sys.argv[1])
    ni, nk, nj = shape
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    nz = d['kte']; ncol = (i1 - i0) * (j1 - j0)
    vertical = (d['kms'], d['kme'], d['kds'], d['kde'], d['kts'], d['kte'])

    def lane(x):
        v = cp.asarray(x)[j0:j1, :, i0:i1].transpose(0, 2, 1).reshape(ncol, nk)
        padded = cp.zeros((gfl.lane_shape(ncol, nk)[0] * gfl.LANES, nk), dtype=v.dtype)
        padded[:ncol] = v
        return cp.ascontiguousarray(padded.reshape(-1, gfl.LANES, nk).transpose(0, 2, 1))

    def srf(x):
        return cp.ascontiguousarray(cp.asarray(x)[j0:j1, i0:i1].reshape(ncol))

    base = {n: lane(a[n]) for n in LEVEL_IN + ('rqvften', 'rthften', 'rthblten', 'rqvblten', 'rthraten')}
    base.update({n: srf(a[n]) for n in SURF_IN + ('kpbl',)})
    for n, typ in (('itimestep', cp.int32), ('ichoice_deep', cp.int32), ('ishallow_g3', cp.int32),
                   ('dt', cp.float32), ('dxcell', cp.float32), ('xlv', cp.float32), ('cp', cp.float32),
                   ('g', cp.float32), ('r_v', cp.float32)):
        base[n] = cp.asarray([d[n]], dtype=typ)
    inout0 = {n: lane(a[n]) for n in ('gdc', 'gdc2') + LEVEL_INOUT}
    inout0.update({n: srf(a[n]) for n in SURF_INOUT + INT_INOUT})

    # GFDRV, packed as time_gf_vs_gfl packs it
    def lev(x):
        return cp.ascontiguousarray(cp.asarray(x)[j0:j1, :nz, i0:i1].transpose(0, 2, 1).reshape(ncol, nz))
    lvin = cp.empty((ncol, len(gfmod._IN_LEV), nz), dtype=cp.float32)
    names = {'q': 'q', 'qv': 'q'}
    for slot, name in enumerate(gfmod._IN_LEV):
        lvin[:, slot, :] = lev(a[names.get(name, name)])
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

    report = {'columns': ncol, 'nz': nz, 'device': cp.cuda.runtime.getDeviceProperties(device)['name'].decode(),
              'gfdrv_regs': _attr(fn, 4), 'gfdrv_local_bytes': _attr(fn, 3), 'variants': {}}
    reference = None
    for b in blocks:
        module = variant(gfl, device, b)
        kernel = module.get_function('gfl_lane_columns')
        gfl._module.cache_clear()
        gfl._SCRATCH.clear()
        original = gfl._module
        gfl._module = lambda dev, m=module: m
        resident = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.ptr, gfl.LANES, 0)

        def run_gfl():
            f = dict(base)
            f.update({k: v.copy() for k, v in inout0.items()})
            cp.cuda.Device().synchronize(); t = time.perf_counter()
            info = gfl.launch_lane_columns(f, ncol=ncol, nz=nz, vertical=vertical)
            cp.cuda.Device().synchronize(); return time.perf_counter() - t, f, info

        _, f, info = run_gfl(); run_gf()
        words = cp.concatenate([f[k].view(cp.uint32).ravel() for k in sorted(inout0)])
        if reference is None:
            reference = words
        tl, tg = [], []
        for _ in range(7):
            tl.append(run_gfl()[0]); tg.append(run_gf())
        report['variants'][b] = {'regs': _attr(kernel, 4), 'local_bytes': _attr(kernel, 3),
                                 'resident_blocks_per_sm': resident, 'blocks': info['blocks'],
                                 'gfl_lane_median_ms': round(statistics.median(tl) * 1e3, 3),
                                 'gfdrv_median_ms': round(statistics.median(tg) * 1e3, 3),
                                 'different_from_first_variant': int(cp.count_nonzero(words != reference)),
                                 'highwater_max': int(info['scratch_highwater'].max())}
        report['variants'][b]['ratio'] = round(report['variants'][b]['gfl_lane_median_ms']
                                               / report['variants'][b]['gfdrv_median_ms'], 3)
        gfl._module = original
        print(b, report['variants'][b], flush=True)
    Path(sys.argv[2]).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
