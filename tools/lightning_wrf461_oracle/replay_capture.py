"""Replay a native driver's real LPI capture, outside the step loop.

The stream layout is emitted by capture_wrf461.py. Only option 3 is
captured: its calculation is column local, so
the selected native patch can be replayed without a domain reduction.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np


def read_capture(path):
    raw = path.read_bytes()
    offset = 0

    def read(dtype, shape):
        nonlocal offset
        dtype = np.dtype(dtype)
        count = int(np.prod(shape))
        value = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy().reshape(shape)
        offset += count * dtype.itemsize
        return value

    nx, ny, nz, option, method, cell = map(int, read('<i4', (6,)))
    if option != 3 or min(nx, ny, nz) < 1:
        raise ValueError('This native capture format supports only valid option-3 column patches')
    dt, dx, dy, interval, factor, num, den, adjust = map(float, read('<f4', (8,)))
    seconds = float(read('<f8', (1,))[0])
    fields = {name: read('<i4' if name == 'top' else '<f4', (ny, nx))
              for name in ('lat', 'lon', 'land', 'ht', 'top')}
    for name in ('t', 'z', 'w', 'refl', 'dz', 'theta', 'pi', 'qv', 'qc', 'qr', 'qi', 'qs', 'qg'):
        fields[name] = np.ascontiguousarray(read('<f4', (ny, nz, nx)).transpose(1, 0, 2))
    for name in ('inputnum', 'inputden'):
        fields[name] = read('<f4', (12, ny, nx))
    expected = read('<f4', (ny, nx))
    clock = float(read('<f4', (1,))[0])
    if offset != len(raw):
        raise ValueError(f'Native capture length mismatch: decoded {offset}, file has {len(raw)}')
    return fields, expected, dict(shape=[nz, ny, nx], option=option, method=method,
        cellcount_method=cell, dt=dt, dx=dx, dy=dy, interval=interval, factor=factor,
        num=num, den=den, adjust=adjust, seconds=seconds, native_post_activation=clock,
        capture_bytes=len(raw), capture_sha256=hashlib.sha256(raw).hexdigest())


def replay(path, out):
    if os.environ.get('GPUWM_WRF_EXACT') != '1':
        raise ValueError('The real native replay requires GPUWM_WRF_EXACT=1')
    import cupy as cp
    from woof.core.lightning import launch_lightning
    from woof.certify.kernel_manifest import kernel_manifest
    fields, expected, receipt = read_capture(path)
    output = {name: cp.zeros(expected.shape, dtype=cp.float32) for name in
              ('ic_flashrate', 'cg_flashrate', 'ic_flashcount', 'cg_flashcount', 'lpi')}
    launch_lightning({name: cp.asarray(value) for name, value in fields.items()}, output,
        option=3, method=receipt['method'], cellcount_method=receipt['cellcount_method'],
        dx=receipt['dx'], dy=receipt['dy'], interval=receipt['interval'],
        factor=receipt['factor'], num=receipt['num'], den=receipt['den'], adjust=receipt['adjust'])
    actual = cp.asnumpy(output['lpi'])
    a, b = actual.view(np.uint32), expected.view(np.uint32)
    delta = np.abs(a.astype(np.int64) - b.astype(np.int64))
    receipt.update(kind='real strict native WRF forecast capture; LPI output only',
        native_source='WRF v4.6.1 phys/module_ltng_lpi.F', words=int(a.size),
        differing_words=int(np.count_nonzero(a != b)), max_ulp=int(delta.max(initial=0)),
        native_nonzero=int(np.count_nonzero(expected)), cuda_nonzero=int(np.count_nonzero(actual)),
        native_min=float(expected.min()), native_max=float(expected.max()),
        cuda_min=float(actual.min()), cuda_max=float(actual.max()),
        input_summary={name: {'minimum': float(fields[name].min()),
            'maximum': float(fields[name].max()), 'nonzero': int(np.count_nonzero(fields[name]))}
            for name in ('w', 'qc', 'qr', 'qi', 'qs', 'qg', 'theta', 'pi')},
        kernel_manifest=kernel_manifest())
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ('kernel_manifest', 'input_summary')}))
    return int(receipt['differing_words'] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return replay(args.capture, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
