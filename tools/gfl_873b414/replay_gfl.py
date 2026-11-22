"""Module replay: a captured WRF v4.8.0 GFL call through woof.core.gfl_873b414 on the GPU.

usage: replay_gfl.py PRE POST [REPORT.json]
PRE/POST are the capture wrapper's files (tools/gfl_873b414/gfl_capture_wrapper.F90):
PRE holds every argument word of one native cu_grell_freitas_li call, POST every
inout word after it. The replay starts each inout array from its PRE words, runs
the tile's columns on the GPU, and compares EVERY word of every inout array
(the whole kms:kme / ims:ime memory extent, held words included) with POST.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

LEVEL_IN = ('u', 'v', 'w', 't', 'q', 'rho', 'p', 'pi', 'p8w', 'dz8w')
SURF_IN = ('ht', 'xland', 'hfx', 'qfx', 'gsw')
SURF_INOUT = ('htop', 'hbot', 'xmb_shallow', 'raincv', 'pratec')
INT_INOUT = ('k22_shallow', 'kbcon_shallow', 'ktop_shallow')
LEVEL_INOUT = ('rthcuten', 'rqvcuten', 'rqccuten', 'rqicuten', 'rucuten', 'rvcuten')
DIMS = ('ims', 'ime', 'jms', 'jme', 'kms', 'kme', 'ids', 'ide', 'jds', 'jde', 'kds', 'kde',
        'its', 'ite', 'jts', 'jte', 'kts', 'kte')


class Reader:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        self.at = 0

    def take(self, dtype, count):
        n = np.dtype(dtype).itemsize * count
        out = np.frombuffer(self.raw, dtype=dtype, count=count, offset=self.at).copy()
        self.at += n
        return out

    def done(self):
        if self.at != len(self.raw):
            raise ValueError(f'{len(self.raw) - self.at} unread bytes')


def read_pre(path):
    r = Reader(path)
    head = r.take('<i4', 21)
    d = dict(zip(DIMS + ('itimestep', 'ichoice_deep', 'ishallow_g3'), map(int, head)))
    flags = r.take('<i4', 4)
    if tuple(flags) != (1, 1, 1, 1):
        raise ValueError(f'replay expects rthften, rqvften, gdc, gdc2 present; flags {tuple(flags)}')
    d['dt'], d['dxcell'], d['xlv'], d['cp'], d['g'], d['r_v'] = r.take('<f4', 6)
    ni, nk, nj = d['ime'] - d['ims'] + 1, d['kme'] - d['kms'] + 1, d['jme'] - d['jms'] + 1
    vol = lambda: r.take('<f4', ni * nk * nj).reshape(nj, nk, ni)
    sur = lambda dt='<f4': r.take(dt, ni * nj).reshape(nj, ni)
    a = {}
    for n in LEVEL_IN: a[n] = vol()
    a['rqvften'] = vol(); a['rthften'] = vol()
    for n in ('rthblten', 'rqvblten', 'rthraten'): a[n] = vol()
    for n in SURF_IN: a[n] = sur()
    a['kpbl'] = sur('<i4')
    read_inout(r, a, vol, sur)
    r.done()
    return d, a, (ni, nk, nj)


def read_inout(r, a, vol, sur):
    for n in SURF_INOUT: a[n] = sur()
    for n in INT_INOUT: a[n] = sur('<i4')
    a['gdc'] = vol(); a['gdc2'] = vol()
    for n in LEVEL_INOUT: a[n] = vol()


def read_post(path, shape):
    ni, nk, nj = shape
    r = Reader(path)
    a = {}
    read_inout(r, a, lambda: r.take('<f4', ni * nk * nj).reshape(nj, nk, ni),
               lambda dt='<f4': r.take(dt, ni * nj).reshape(nj, ni))
    r.done()
    return a


def main():
    import cupy as cp
    from woof.core import gfl_873b414 as gfl
    pre, post = sys.argv[1], sys.argv[2]
    report = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    d, a, shape = read_pre(pre)
    ni, nk, nj = shape
    want = read_post(post, shape)
    # The tile's columns: WRF's driver loops i=its..ite (init) and its..itf (scheme), j=jts..jte.
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    nz = d['kte']
    ncol = (i1 - i0) * (j1 - j0)
    if d['ide'] - 1 < d['ite'] or d['jde'] - 1 < d['jte']:
        raise ValueError('tile reaches the staggered edge; the column entry assumes ite <= ide-1')
    vertical = (d['kms'], d['kme'], d['kds'], d['kde'], d['kts'], d['kte'])

    # Repacking and comparing run on the GPU: the host only reads the files.
    dev = {n: cp.asarray(v) for n, v in a.items()}

    def lev(x):
        x = dev[x] if isinstance(x, str) else cp.asarray(x)
        return cp.ascontiguousarray(x[j0:j1, :, i0:i1].transpose(0, 2, 1).reshape(ncol, nk))

    def srf(x):
        x = dev[x] if isinstance(x, str) else cp.asarray(x)
        return cp.ascontiguousarray(x[j0:j1, i0:i1].reshape(ncol))

    f = {}
    for n in LEVEL_IN + ('rqvften', 'rthften', 'rthblten', 'rqvblten', 'rthraten', 'gdc', 'gdc2') + LEVEL_INOUT:
        f[n] = lev(n)
    for n in SURF_IN + SURF_INOUT + ('kpbl',) + INT_INOUT:
        f[n] = srf(n)
    for n, typ in (('itimestep', cp.int32), ('ichoice_deep', cp.int32), ('ishallow_g3', cp.int32),
                   ('dt', cp.float32), ('dxcell', cp.float32), ('xlv', cp.float32), ('cp', cp.float32),
                   ('g', cp.float32), ('r_v', cp.float32)):
        f[n] = cp.asarray([d[n]], dtype=typ)
    gfl._module(cp.cuda.runtime.getDevice())  # compile outside the timed call
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    info = gfl.launch_columns(f, ncol=ncol, nz=nz, vertical=vertical)
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    info2 = None
    # Second, timed-warm call on a copy of the same inputs (scratch already sized).
    g = {k: v.copy() for k, v in f.items()}
    for n in LEVEL_INOUT + ('gdc', 'gdc2'): g[n] = lev(n)
    for n in SURF_INOUT + INT_INOUT: g[n] = srf(n)
    cp.cuda.Device().synchronize(); t1 = time.perf_counter()
    gfl.launch_columns(g, ncol=ncol, nz=nz, vertical=vertical)
    cp.cuda.Device().synchronize()
    warm = time.perf_counter() - t1

    got = {}
    for n in ('gdc', 'gdc2') + LEVEL_INOUT:
        x = dev[n].copy()
        x[j0:j1, :, i0:i1] = f[n].reshape(j1 - j0, i1 - i0, nk).transpose(0, 2, 1)
        got[n] = x
    for n in SURF_INOUT + INT_INOUT:
        x = dev[n].copy()
        x[j0:j1, i0:i1] = f[n].reshape(j1 - j0, i1 - i0)
        got[n] = x
    rows, total, bad = {}, 0, 0
    changed = 0
    for n in got:
        y = cp.asarray(want[n]).view(cp.uint32).ravel()
        x = got[n].view(cp.uint32).ravel()
        diff = int(cp.count_nonzero(x != y))
        changed += int(cp.count_nonzero(y != dev[n].view(cp.uint32).ravel()))
        rows[n] = {'words': int(x.size), 'different': diff}
        total += x.size; bad += diff
    active = int(np.count_nonzero(want['raincv'][j0:j1, i0:i1] > 0))
    out = {'pre': str(pre), 'post': str(post), 'pre_sha256': hashlib.sha256(Path(pre).read_bytes()).hexdigest(),
           'post_sha256': hashlib.sha256(Path(post).read_bytes()).hexdigest(),
           'itimestep': d['itimestep'], 'columns': ncol, 'nz': nz, 'dx': float(d['dxcell']),
           'raining_columns_native': active, 'words_changed_by_native_call': changed,
           'words_compared': int(total), 'words_different': int(bad), 'fields': rows,
           'diagnostics_nonzero_columns': int(cp.count_nonzero(info['diagnostics'])),
           'scratch_highwater_max_bytes': int(info['scratch_highwater'].max()),
           'first_call_s': wall, 'warm_call_s': warm, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    print(json.dumps({k: v for k, v in out.items() if k != 'fields'}, indent=1))
    for n, r in rows.items():
        if r['different']: print('DIFF', n, r)
    if report: report.write_text(json.dumps(out, indent=1) + '\n')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
