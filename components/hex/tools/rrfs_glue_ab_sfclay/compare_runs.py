"""Fail on any history/diag file, variable, shape, dtype or byte mismatch."""
import json
from pathlib import Path
import sys
import netCDF4
import numpy as np

left, right = map(Path, sys.argv[1:3])
a = {p.name: p for p in left.glob('*.nc')}
b = {p.name: p for p in right.glob('*.nc')}
assert a.keys() == b.keys(), (list(a), list(b))
assert a, 'No output files'
result = dict(files={}, compared_words=0, compared_text_bytes=0, different=0)
for name in sorted(a):
    with netCDF4.Dataset(a[name]) as x, netCDF4.Dataset(b[name]) as y:
        assert x.variables.keys() == y.variables.keys(), name
        assert {k: len(v) for k, v in x.dimensions.items()} == {k: len(v) for k, v in y.dimensions.items()}, name
        row = dict(words=0, text_bytes=0, different=0, variables=len(x.variables))
        for field in x.variables:
            vx, vy = x.variables[field], y.variables[field]
            vx.set_auto_maskandscale(False)
            vy.set_auto_maskandscale(False)
            xx, yy = np.asarray(vx[:]), np.asarray(vy[:])
            assert xx.shape == yy.shape and xx.dtype == yy.dtype, (name, field)
            xx, yy = np.ascontiguousarray(xx), np.ascontiguousarray(yy)
            if xx.dtype.kind in 'SU':
                count = xx.nbytes
                row['text_bytes'] += count
                result['compared_text_bytes'] += count
                different = int(np.count_nonzero(xx.view(np.uint8) != yy.view(np.uint8)))
            else:
                row['words'] += xx.size
                result['compared_words'] += xx.size
                different = int(np.count_nonzero(np.any(
                    xx.view(np.uint8).reshape(-1, xx.itemsize) !=
                    yy.view(np.uint8).reshape(-1, yy.itemsize), axis=1)))
            row['different'] += different
            result['different'] += different
        result['files'][name] = row
print(json.dumps(result, indent=2))
assert result['different'] == 0
