"""Scratch highwater of the GFL columns against the level count: the captured columns cut to kte levels.

usage: hw_probe.py PRE REPORT.json KTE...
The bottom kte+1 storage levels of every captured column run with kts=1, kte (the
words are not a replay; only each column's scratch need is read). Reports the
largest need per level count beside the launcher's share for that count.
"""
import json
import sys
from pathlib import Path


def main():
    import cupy as cp
    sys.path.insert(0, str(Path.cwd() / 'tools/gfl_873b414'))
    from replay_gfl import read_pre, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT, LEVEL_INOUT
    from woof.core import gfl_873b414 as gfl
    d, a, shape = read_pre(sys.argv[1])
    ni, nk, nj = shape
    i0, i1 = d['its'] - d['ims'], d['ite'] - d['ims'] + 1
    j0, j1 = d['jts'] - d['jms'], d['jte'] - d['jms'] + 1
    ncol = (i1 - i0) * (j1 - j0)
    out = {'columns': ncol, 'rows': []}
    for kte in [int(x) for x in sys.argv[3:]]:
        k = kte + 1

        def lev(x):
            return cp.ascontiguousarray(cp.asarray(x)[j0:j1, :k, i0:i1].transpose(0, 2, 1).reshape(ncol, k))

        def srf(x):
            return cp.ascontiguousarray(cp.asarray(x)[j0:j1, i0:i1].reshape(ncol))

        f = {n: lev(a[n]) for n in LEVEL_IN + ('rqvften', 'rthften', 'rthblten', 'rqvblten', 'rthraten', 'gdc', 'gdc2')
             + LEVEL_INOUT}
        f.update({n: srf(a[n]) for n in SURF_IN + SURF_INOUT + ('kpbl',) + INT_INOUT})
        for n, typ in (('itimestep', cp.int32), ('ichoice_deep', cp.int32), ('ishallow_g3', cp.int32),
                       ('dt', cp.float32), ('dxcell', cp.float32), ('xlv', cp.float32), ('cp', cp.float32),
                       ('g', cp.float32), ('r_v', cp.float32)):
            f[n] = cp.asarray([d[n]], dtype=typ)
        info = gfl.launch_columns(f, ncol=ncol, nz=kte, vertical=(1, k, 1, k, 1, kte))
        hw = info['scratch_highwater']
        row = {'kte': kte, 'storage_levels': k, 'highwater_max': int(hw.max()), 'share': int(info['share_bytes']),
               'columns_over_share': int(cp.count_nonzero(hw > info['share_bytes'])),
               'raining_columns': int(cp.count_nonzero(f['raincv'] > 0))}
        print(row, flush=True)
        out['rows'].append(row)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
