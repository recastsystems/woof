"""Reduce a full-tile GFL capture to a small committed oracle of selected columns.

usage: make_oracle.py PRE POST_GLIBC POST_CRGAMMA OUTDIR [N]
Picks N columns (default 96) from the tile: half where the native call rained
(deep or shallow), a quarter where only the shallow scheme fired (ktop_shallow>0
and no rain), the rest quiet, all chosen by a fixed stride. Writes a PRE file
for a 1-row tile of those columns (WRF bounds kept in the vertical, ims=its=1,
ime=ite=N, ide=N+1) and the two POST files in the same layout. Columns are
independent in GFL (one thread per column in the CUDA port; the full-tile
replay checks the same claim on every column of the domain).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_gfl import (read_pre, read_post, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT,
                        LEVEL_INOUT)


def main():
    pre, post_glibc, post_cr, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 96
    d, a, shape = read_pre(pre)
    g = read_post(post_glibc, shape)
    c = read_post(post_cr, shape)
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    rain = g['raincv'][j0:j1, i0:i1].ravel()
    shallow = g['ktop_shallow'][j0:j1, i0:i1].ravel()
    differs = np.zeros(rain.size, bool)
    for k in g:
        x = g[k].view(np.uint32); y = c[k].view(np.uint32)
        if x.ndim == 3:
            differs |= (x[j0:j1, :, i0:i1] != y[j0:j1, :, i0:i1]).any(axis=1).ravel()
        else:
            differs |= (x[j0:j1, i0:i1] != y[j0:j1, i0:i1]).ravel()
    groups = [np.flatnonzero((rain > 0) & differs), np.flatnonzero((rain > 0) & ~differs),
              np.flatnonzero((rain == 0) & (shallow > 0)), np.flatnonzero((rain == 0) & (shallow == 0))]
    want = [n // 4, n // 4, n // 4, n - 3 * (n // 4)]
    pick = []
    for idx, k in zip(groups, want):
        if idx.size:
            pick.extend(idx[np.linspace(0, idx.size - 1, min(k, idx.size)).astype(int)].tolist())
    pick = np.array(sorted(set(pick)))
    m = pick.size
    nx = i1 - i0
    jj, ii = pick // nx + j0, pick % nx + i0
    nk = shape[1]
    hdr = dict(d)
    hdr.update(ims=1, ime=m, jms=1, jme=1, ids=1, ide=m + 1, jds=1, jde=2, its=1, ite=m, jts=1, jte=1)
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)

    def vol(x):  # [nj,nk,ni] -> [1,nk,m]
        return np.ascontiguousarray(x[jj, :, ii].T[None])

    def sur(x):
        return np.ascontiguousarray(x[jj, ii][None])

    with open(outd / 'pre.bin', 'wb') as f:
        np.array([hdr[k] for k in ('ims', 'ime', 'jms', 'jme', 'kms', 'kme', 'ids', 'ide', 'jds', 'jde', 'kds',
                                   'kde', 'its', 'ite', 'jts', 'jte', 'kts', 'kte', 'itimestep', 'ichoice_deep',
                                   'ishallow_g3')], '<i4').tofile(f)
        np.array([1, 1, 1, 1], '<i4').tofile(f)
        np.array([d['dt'], d['dxcell'], d['xlv'], d['cp'], d['g'], d['r_v']], '<f4').tofile(f)
        for k in LEVEL_IN + ('rqvften', 'rthften', 'rthblten', 'rqvblten', 'rthraten'): vol(a[k]).tofile(f)
        for k in SURF_IN: sur(a[k]).tofile(f)
        sur(a['kpbl']).tofile(f)
        write_inout(f, a, vol, sur)
    for name, src in (('post_native_glibc239.bin', g), ('post_native_crgamma.bin', c)):
        with open(outd / name, 'wb') as f:
            write_inout(f, src, vol, sur)
    (outd / 'columns.json').write_text(json.dumps({
        'source_pre': f'captures/{pre.name}', 'columns': m, 'nz': int(d['kte']), 'itimestep': d['itimestep'],
        'i': (ii + d['ims']).tolist(), 'j': (jj + d['jms']).tolist(),
        'raining': int((rain[pick] > 0).sum()), 'gamma_sensitive': int(differs[pick].sum()),
        'shallow_only': int(((rain[pick] == 0) & (shallow[pick] > 0)).sum())}, indent=1) + '\n')
    print(m, 'columns written to', outd)


def write_inout(f, src, vol, sur):
    for k in SURF_INOUT: sur(src[k]).tofile(f)
    for k in INT_INOUT: sur(src[k]).tofile(f)
    vol(src['gdc']).tofile(f); vol(src['gdc2']).tofile(f)
    for k in LEVEL_INOUT: vol(src[k]).tofile(f)


if __name__ == '__main__':
    main()
