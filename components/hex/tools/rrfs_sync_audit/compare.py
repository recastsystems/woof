"""Compare every output variable, refusing missing files, variables or changed layouts."""
import json
from pathlib import Path
import sys
import netCDF4
import numpy as np


def compare(left, right):
    names = [{p.name for p in root.glob('*.nc')} for root in (left, right)]
    if not names[0] or names[0] != names[1]:
        raise AssertionError(f'output files differ or empty: {names}')
    files = {}
    for name in sorted(names[0]):
        fields, words, different, text_bytes, text_different = {}, 0, 0, 0, 0
        with netCDF4.Dataset(left / name) as a, netCDF4.Dataset(right / name) as b:
            if set(a.variables) != set(b.variables):
                raise AssertionError(f'{name}: variable sets differ')
            for field in a.variables:
                av, bv = a.variables[field], b.variables[field]
                av.set_auto_maskandscale(False)
                bv.set_auto_maskandscale(False)
                x, y = np.asarray(av[:]), np.asarray(bv[:])
                if x.dtype != y.dtype or x.shape != y.shape or av.dimensions != bv.dimensions:
                    raise AssertionError(f'{name}:{field}: dtype/shape/dimensions differ')
                if x.dtype.hasobject:
                    raise AssertionError(f'{name}:{field}: unsupported object dtype')
                xb, yb = np.ascontiguousarray(x).view(np.uint8), np.ascontiguousarray(y).view(np.uint8)
                nbytes = int(np.count_nonzero(xb != yb))
                if x.dtype.kind in 'SU':
                    text_bytes += x.nbytes
                    text_different += nbytes
                else:
                    nwords = int(np.any((xb != yb).reshape(-1, x.dtype.itemsize), axis=1).sum())
                    words += x.size
                    different += nwords
                if nbytes:
                    fields[field] = nbytes
        files[name] = dict(words=words, different=different, text_bytes=text_bytes,
                           text_different=text_different, changed_fields_bytes=fields)
    return dict(files=files, words=sum(v['words'] for v in files.values()),
                different=sum(v['different'] for v in files.values()),
                text_bytes=sum(v['text_bytes'] for v in files.values()),
                text_different=sum(v['text_different'] for v in files.values()))


if __name__ == '__main__':
    result = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
    if result['different'] or result['text_different']:
        raise SystemExit(1)
