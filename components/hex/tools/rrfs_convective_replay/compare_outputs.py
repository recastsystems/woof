"""Compare every history/diag variable, refusing missing fields or changed layouts."""
import json
from pathlib import Path
import sys
import numpy as np
import netCDF4

base, tip, out = map(Path, sys.argv[1:4])
files = lambda root: {p.name for pattern in ('history.*.nc', 'diag.*.nc') for p in root.glob(pattern)}
assert files(base) == files(tip) and files(base), 'output file set differs or is empty'
receipt = {}
for name in sorted(files(base)):
    with netCDF4.Dataset(base/name) as a, netCDF4.Dataset(tip/name) as b:
        assert set(a.variables) == set(b.variables), name
        words = different = text_bytes = text_different = 0
        for field in a.variables:
            x, y = a[field], b[field]
            assert x.dtype == y.dtype and x.shape == y.shape, (name, field)
            x.set_auto_maskandscale(False)
            y.set_auto_maskandscale(False)
            xv, yv = np.asarray(x[:]), np.asarray(y[:])
            equal = (xv.view(np.dtype(f'u{xv.itemsize}')) == yv.view(np.dtype(f'u{yv.itemsize}'))) if xv.dtype.kind not in 'SU' else (xv == yv)
            n = int(np.count_nonzero(~equal))
            if xv.dtype.kind in 'SU':
                text_bytes += xv.nbytes
                text_different += n
            else:
                words += xv.size
                different += n
        receipt[name] = dict(variables=len(a.variables), words=int(words), different=different,
                             text_bytes=text_bytes, text_different=text_different)
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
assert all(r['different'] == r['text_different'] == 0 for r in receipt.values())
