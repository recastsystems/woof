#!/usr/bin/env python3
"""Compare native NetCDF variable storage words, including NaN payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import netCDF4
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('first', type=Path)
p.add_argument('second', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--hours', nargs='*', type=int)
a = p.parse_args()
def sha(file):
    with file.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
rows = []
for left in sorted(a.first.glob('*.nc')):
    if left.name.split('.')[0] not in ('history', 'diag', 'restart', 'tracer'):
        continue
    if a.hours is not None and int(left.name.split('_')[1].split('.')[0]) - 12 not in a.hours:
        continue
    right = a.second / left.name
    if not right.exists():
        rows.append(dict(file=left.name, missing='second'))
        continue
    row = dict(file=left.name, first_sha256=sha(left), second_sha256=sha(right), words=0, different_words=0, fields={})
    with netCDF4.Dataset(left) as l, netCDF4.Dataset(right) as r:
        l.set_auto_maskandscale(False); r.set_auto_maskandscale(False)
        for key in sorted(set(l.variables) | set(r.variables)):
            if key not in l.variables or key not in r.variables:
                row['fields'][key] = dict(missing='first' if key not in l.variables else 'second')
                continue
            x, y = np.asarray(l[key][:]), np.asarray(r[key][:])
            if x.shape != y.shape or x.dtype != y.dtype:
                row['fields'][key] = dict(layout=[str(x.dtype), list(x.shape), str(y.dtype), list(y.shape)])
                continue
            kind = f'u{x.dtype.itemsize}' if x.dtype.itemsize in (1, 2, 4, 8) else 'u1'
            wx, wy = np.ascontiguousarray(x).view(kind), np.ascontiguousarray(y).view(kind)
            count = int(np.count_nonzero(wx != wy))
            row['words'] += wx.size
            row['different_words'] += count
            if count:
                row['fields'][key] = dict(words=wx.size, different_words=count)
    rows.append(row)
result = dict(first=str(a.first.resolve()), second=str(a.second.resolve()), files=rows,
              words=sum(r.get('words',0) for r in rows), different_words=sum(r.get('different_words',0) for r in rows),
              complete=all('missing' not in r and all('missing' not in v and 'layout' not in v for v in r.get('fields',{}).values()) for r in rows))
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'files'}))
