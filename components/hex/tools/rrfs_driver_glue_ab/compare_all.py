"""Compare every history/diag/restart variable, including field/file sets and text bytes."""
import json
import sys
from pathlib import Path
import netCDF4
import numpy as np

left, right, output = map(Path, sys.argv[1:])
files = lambda root: {p.name: p for pattern in ('history.*.nc', 'diag.*.nc', 'restart.*.nc') for p in root.glob(pattern)}
a, b = files(left), files(right)
receipt = {'left': str(left), 'right': str(right), 'only_left': sorted(a.keys()-b.keys()),
           'only_right': sorted(b.keys()-a.keys()), 'files': {}, 'words': 0, 'different': 0,
           'text_bytes': 0, 'text_different': 0, 'schema_differences': []}
for name in sorted(a.keys() & b.keys()):
    row = {'words': 0, 'different': 0, 'text_bytes': 0, 'text_different': 0, 'differing_fields': {}}
    with netCDF4.Dataset(a[name]) as x, netCDF4.Dataset(b[name]) as y:
        if set(x.variables) != set(y.variables):
            receipt['schema_differences'].append([name, 'variable sets', sorted(set(x.variables)^set(y.variables))])
        for field in sorted(set(x.variables) & set(y.variables)):
            xv, yv = x[field], y[field]
            xv.set_auto_maskandscale(False); yv.set_auto_maskandscale(False)
            if xv.dimensions != yv.dimensions:
                receipt['schema_differences'].append([name, field, 'dimensions'])
            xx, yy = np.asarray(xv[:]), np.asarray(yv[:])
            if xx.shape != yy.shape or xx.dtype != yy.dtype:
                receipt['schema_differences'].append([name, field, 'shape or dtype'])
                continue
            if xx.dtype.kind in 'SU':
                wx, wy = np.ascontiguousarray(xx).view(np.uint8), np.ascontiguousarray(yy).view(np.uint8)
                n = int(np.count_nonzero(wx != wy))
                row['text_bytes'] += wx.size; row['text_different'] += n
            else:
                wx, wy = np.ascontiguousarray(xx).view('V'+str(xx.dtype.itemsize)), np.ascontiguousarray(yy).view('V'+str(yy.dtype.itemsize))
                n = int(np.count_nonzero(wx != wy))
                row['words'] += xx.size; row['different'] += n
            if n:
                row['differing_fields'][field] = n
    receipt['files'][name] = row
    for key in ('words', 'different', 'text_bytes', 'text_different'):
        receipt[key] += int(row[key])
receipt['passed'] = bool(a) and not any(receipt[k] for k in ('only_left','only_right','schema_differences','different','text_different'))
output.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
raise SystemExit(0 if receipt['passed'] else 1)
