#!/usr/bin/env python3
"""Run the real one-card hrrrv5 CONUS case on an already allocated AMD host.

Use an installed engine/hex environment and the retained prepared data:
python hex/tree/tools/amd_conus_rerun.py --data /path/hrrrv5-conus --out /path/run
This tool neither obtains a machine nor changes the physics or allocator.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=.1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--start-time', default='2026-11-01_00:00:00')
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if args.hours < .05:
        parser.error('--hours must cover at least the first three-minute history')
    start = datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
    required = ('grid.nc', 'static.nc', 'init.nc', 'invariant.nc',
                'conus3km.ugwp_oro_data.nc', 'lbc')
    for name in required:
        if not (data / name).exists():
            parser.error(f'missing prepared input: {data / name}')
    from netCDF4 import Dataset
    with Dataset(data / 'init.nc') as dataset:
        dimensions = {name: len(dataset.dimensions[name])
                      for name in ('nCells', 'nVertLevels', 'nSoilLevels')}
    if dimensions != dict(nCells=2341818, nVertLevels=59, nSoilLevels=9):
        parser.error(f'prepared input is not the native hrrrv5 CONUS shape: {dimensions}')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['WOOF_BACKEND'] = 'amd'
    os.environ['GPUWM_NUMERICS'] = 'fast'
    os.environ['WOOF_RUST_BACKEND'] = 'hip'
    os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(min(4, int(os.environ.get(name, '4'))))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    import woof
    import cupy as cp
    from woof.hip_backend import install, device_info
    install()
    if not cp.cuda.runtime.is_hip:
        raise RuntimeError('this command requires physical AMD with ROCm CuPy')
    cp.cuda.Device(args.device).use()
    free, total = cp.cuda.runtime.memGetInfo()
    # Prior physical sample, minus the bounded arena saving, plus 4 GiB.
    # This is a provisional rerun budget, not a measured capacity claim.
    required_free = 205658734592 - (5798341368 - 162267136) + 4 * 1024**3
    if free < required_free:
        raise RuntimeError(f'the provisional rerun budget needs {required_free / 1024**3:.3f} GiB free; '
                           f'available {free / 1024**3:.3f} GiB / total {total / 1024**3:.3f} GiB')
    command = ['forecast', '--backend', 'amd', '--mesh', 'conus3km',
               '--grid', str(data / 'grid.nc'), '--static', str(data / 'static.nc'),
               '--init', str(data / 'init.nc'), '--invariant', str(data / 'invariant.nc'),
               '--lbc-dir', str(data / 'lbc'), '--orographic-statistics',
               str(data / 'conus3km.ugwp_oro_data.nc'),
               '--init-source', start.strftime('GFS-%Y-%m-%d-%HZ'),
               '--start-time', args.start_time, '--hours', str(args.hours), '--dt', '18',
               '--history-every-minutes', '3', '--history-from-minutes', '3',
               '--out', str(out / 'forecast'), '--scratch', str(out / 'scratch'),
               '--no-products', '--numerics', 'fast', '--physics-suite', 'hrrrv5',
               '--physics-default-backend', 'hex-cuda']
    receipt = dict(complete=False, passed=False, device=device_info(),
                   free_before_bytes=free, total_device_bytes=total,
                   provisional_required_free_bytes=required_free,
                   data=str(data), argv=[sys.executable, '-m', 'woof.hex.cli', *command],
                   configuration='native hrrrv5, CONUS 2341818 x 59, nine soil levels',
                   dimensions=dimensions,
                   capacity_status='provisional admission; full forecast fit is the result being measured',
                   measurement='pool is process allocator; card-used includes other processes',
                   began_utc=time.strftime('%FT%TZ', time.gmtime()))
    path = out / 'receipt.json'
    path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)
    done = threading.Event()
    peaks = dict(pool_used_bytes=0, pool_reserved_bytes=0, card_used_bytes=0, rss_bytes=0)

    def sample():
        cp.cuda.Device(args.device).use()
        with (out / 'memory.jsonl').open('a', buffering=1) as handle:
            while not done.is_set():
                try:
                    available, capacity = cp.cuda.runtime.memGetInfo()
                    status = dict(line.split(':', 1) for line in
                                  Path('/proc/self/status').read_text().splitlines())
                    row = dict(utc=time.strftime('%FT%TZ', time.gmtime()),
                               pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
                               pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
                               card_used_bytes=capacity - available,
                               rss_bytes=int(status['VmRSS'].split()[0]) * 1024)
                    for key in peaks:
                        peaks[key] = max(peaks[key], row[key])
                    handle.write(json.dumps(row) + '\n')
                except Exception as error:
                    handle.write(json.dumps(dict(monitor_error=repr(error))) + '\n')
                    return
                done.wait(.5)

    monitor = threading.Thread(target=sample, name='amd-memory', daemon=True)
    monitor.start()
    begin = time.monotonic()
    try:
        from woof.hex import forecast_door
        # This is the requested capacity measurement, not an ordinary forecast
        # admission. The shipped fitted CUDA row has no measured AMD basis and
        # prices this shape above 600 GiB. Use the explicit prior-sample budget
        # checked above; record it without presenting it as a fitted AMD row.
        def capacity_measurement(arguments, config=None):
            available, _ = cp.cuda.runtime.memGetInfo()
            if available < required_free:
                raise RuntimeError(f'capacity measurement needs {required_free} free bytes; '
                                   f'available now: {available}')
            record = dict(backend='amd', suite='hrrrv5',
                          admitted=True, status='capacity measurement; not a qualified fit',
                          basis='prior physical sample minus measured arena saving plus 4 GiB',
                          required_free_bytes=required_free, free_bytes=available,
                          default_fitted_cuda_admission_used=False)
            (out / 'device-memory-admission.json').write_text(json.dumps(record, indent=2))
            arguments.native_memory_admission = record
            return record
        forecast_door.admit_native_device = capacity_measurement
        from woof.hex.mpasphys import memory_admission
        memory_admission.admit = capacity_measurement
        from woof.hex.cli import main as forecast
        code = forecast(command)
        cp.cuda.get_current_stream().synchronize()
        receipt.update(complete=True, passed=code in (None, 0), exit_code=code)
        return 0 if receipt['passed'] else 1
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        done.set()
        monitor.join(timeout=2)
        receipt.update(seconds=time.monotonic() - begin, sampled_peaks=peaks,
                       host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        path.write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
