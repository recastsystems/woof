"""Every word of every variable of every history file in two WOOF-ArWen output folders.

usage: compare_wrfout.py DIR_A DIR_B [REPORT.json]
Files are paired by name (wrfout_d01_*); a file or variable present on one side
only is reported. Floats compare as raw bits (NaN payloads and -0 included).
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def words(v):
    v.set_auto_maskandscale(False)
    a = np.asarray(v[...])
    if a.dtype.kind == 'f':
        return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint64)
    return a


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    fa = {p.name: p for p in a.rglob('wrfout_d0*') if p.is_file() and p.suffix != '.json'}
    fb = {p.name: p for p in b.rglob('wrfout_d0*') if p.is_file() and p.suffix != '.json'}
    out = {'a': str(a), 'b': str(b), 'files': [], 'only_a': sorted(set(fa) - set(fb)),
           'only_b': sorted(set(fb) - set(fa)), 'words': 0, 'different': 0}
    for name in sorted(set(fa) & set(fb)):
        with Dataset(fa[name]) as x, Dataset(fb[name]) as y:
            row = {'file': name, 'words': 0, 'different': 0, 'vars_only_a': sorted(set(x.variables) - set(y.variables)),
                   'vars_only_b': sorted(set(y.variables) - set(x.variables)), 'differing': {}}
            for k in sorted(set(x.variables) & set(y.variables)):
                p, q = words(x.variables[k]), words(y.variables[k])
                if p.shape != q.shape:
                    row['differing'][k] = 'shape'; continue
                d = int(np.count_nonzero(p != q))
                row['words'] += int(p.size); row['different'] += d
                if d: row['differing'][k] = d
            out['files'].append(row); out['words'] += row['words']; out['different'] += row['different']
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=1))
    for r in out['files']:
        print(r['file'], r['words'], r['different'], dict(list(r['differing'].items())[:12]), r['vars_only_a'][:5], r['vars_only_b'][:5])
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
