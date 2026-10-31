#!/usr/bin/env python3
"""Word for word: every native word of the history/diag/restart files a hex run also wrote, as bit patterns.

The N-rank proof of the RRFS route compares hex on N same-host ranks with native RRFS-MPAS on N MPI ranks
(native is not bitwise across rank counts, so N is fixed and stated). Native first, so every native word is
checked; variables only hex writes (mesh and invariant extras in its restart files) are not compared.

usage: rank_compare.py NATIVE_RUN HEX_RUN RECEIPT.json
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np


def compare(native: Path, hex_run: Path):
    report, total, differ, files = {}, 0, 0, []
    for name in sorted(p.name for p in native.glob('*.nc') if p.name.split('.')[0] in ('history', 'diag', 'restart')):
        if not (hex_run / name).is_file():
            continue
        files.append(name)
        rows = {}
        with netCDF4.Dataset(native / name) as a, netCDF4.Dataset(hex_run / name) as b:
            for var in a.variables:
                if var not in b.variables:
                    rows[var] = 'absent in hex'
                    continue
                va, vb = a.variables[var], b.variables[var]
                va.set_auto_maskandscale(False)
                vb.set_auto_maskandscale(False)
                x, y = np.asarray(va[:]), np.asarray(vb[:])
                if x.dtype.kind in 'SUc':
                    if x.tobytes() != y.tobytes():
                        rows[var] = 'text differs'
                    continue
                if x.shape != y.shape:
                    rows[var] = f'shape {x.shape} vs {y.shape}'
                    continue
                bits = np.int32 if x.itemsize == 4 else np.int64
                ne = x.view(bits) != y.view(bits) if x.dtype.kind == 'f' else x != y
                count = int(ne.sum())
                total += x.size
                differ += count
                if count:
                    largest = (float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))[ne]))
                               if x.dtype.kind in 'fi' else -1)
                    rows[var] = f'{count}/{x.size} max|d|={largest:.4g}'
        report[name] = rows
    return dict(native=str(native), hex=str(hex_run), files=files, total=total, differ=differ, report=report)


if __name__ == '__main__':
    native, hex_run, receipt = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    result = compare(native, hex_run)
    receipt.write_text(json.dumps(result, indent=1) + '\n')
    print(f"{native.name} vs {hex_run.name}: {len(result['files'])} files, "
          f"{result['differ']} of {result['total']} words differ")
    for name, rows in result['report'].items():
        if rows:
            print(' ', name, '->', '; '.join(f'{k}: {v}' for k, v in list(rows.items())[:12]))
