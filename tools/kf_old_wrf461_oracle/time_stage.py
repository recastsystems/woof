"""Time one paired CU99/CU1 column batch using captured native inputs.

This measures the device column stage, not forecast wall time. Transfers,
compilation and the reset of each running mean occur outside timed regions.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from .replay_native import INPUTS, read_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--nx', type=int, default=300)
    parser.add_argument('--ny', type=int, default=300)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--ktau', type=int, default=1,
                        help='native time-step capture to repeat (default: first call)')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    import cupy as cp
    from woof.core.kf import (advance_w0avg, kf_old_species_for_microphysics,
                               kf_phase_mode_for_microphysics, launch_kf,
                               launch_kf_old, w0avg_factors)
    from woof.wrf_exact import ENABLED
    from woof.numerics import mode
    from woof.core.kernels import module_source
    import hashlib

    gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

    captures = [read_capture(p) for p in sorted(
        args.captures.glob(f'kf-{args.ktau:08d}-*.bin'))]
    if not captures:
        raise ValueError(f'need native KTAU={args.ktau} captures for this paired timing')
    first = captures[0]
    for row in captures:
        assert all(row[key] == first[key] for key in
                   ('nz', 'dt', 'dx', 'stepcu', 'adaptive', 'cudt_minutes'))
    columns = args.nx * args.ny
    selection = np.arange(columns) % len(captures)
    device = {}
    for name in INPUTS:
        host = np.stack([row['inputs'][name] for row in captures], axis=1)
        expanded = np.ascontiguousarray(host[:, selection].reshape(-1, args.ny, args.nx))
        device[name] = cp.asarray(expanded)
    original_mean = device['w0avg'].copy()
    nca = cp.asarray(np.asarray([row['nca'] for row in captures], np.float32)[selection]
                     .reshape(args.ny, args.nx))
    factors = w0avg_factors(dt=first['dt'], cudt_minutes=first['cudt_minutes'],
                            stepcu=first['stepcu'], adaptive=first['adaptive'])

    def old():
        return launch_kf_old(device['u'], device['v'], device['w'], device['t'],
            device['qv'], device['p'], device['rho'], device['dz8w'], device['pi'],
            device['w0avg'], nca, dx=first['dx'], dt=first['dt'],
            cudt_minutes=first['cudt_minutes'], stepcu=first['stepcu'],
            adaptive=first['adaptive'], species=kf_old_species_for_microphysics(6),
            kf_edrates=first['kf_edrates'])

    def current():
        advance_w0avg(device['w0avg'], device['w'], factors=factors)
        return launch_kf(device['u'], device['v'], device['t'], device['qv'],
            device['p'], device['rho'], device['pi'], device['dz8w'], device['w0avg'],
            dx=first['dx'], dt=first['dt'], cudt_minutes=first['cudt_minutes'],
            phase_mode=kf_phase_mode_for_microphysics(6), f_qr=True)

    launches = ((99, old), (1, current))
    results = {str(scheme): dict(samples=[]) for scheme, _ in launches}
    for scheme, launch in launches:
        device['w0avg'][...] = original_mean
        output = launch()
        cp.cuda.get_current_stream().synchronize()
        del output
    # Alternate the arms so one shared-card scheduling burst cannot be
    # mistaken for a change in the relative column physics cost.
    for _ in range(args.samples):
        for scheme, launch in launches:
            device['w0avg'][...] = original_mean
            cp.cuda.get_current_stream().synchronize()
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            started = time.perf_counter()
            begin.record()
            output = launch()
            end.record()
            end.synchronize()
            results[str(scheme)]['samples'].append(dict(
                device_ms=cp.cuda.get_elapsed_time(begin, end),
                wall_ms=(time.perf_counter()-started)*1000))
            del output
    for result in results.values():
        samples = result['samples']
        result.update(
            median_device_ms=float(np.median([row['device_ms'] for row in samples])),
            median_wall_ms=float(np.median([row['wall_ms'] for row in samples])))
    receipt = dict(schema='gpuwm-cu99-paired-stage-time-v1', strict=ENABLED,
        numerics=mode(), gpu=gpu['name'].decode(),
        free_bytes_before=free_bytes, total_bytes=total_bytes,
        source_sha256={str(scheme): hashlib.sha256(module_source(name).encode()).hexdigest()
                       for scheme,name in ((99,'kf_old'), (1,'kf'))},
        shape=[first['nz'], args.ny, args.nx], distinct_native_columns=len(captures),
        ktau=args.ktau,
        native_raining_columns=sum(bool(row['outputs']['raincv'][0] > 0)
                                   for row in captures),
        captured_inputs=[dict(name=row['name'], sha256=row['sha256']) for row in captures],
        mp_physics=6, kf_edrates=first['kf_edrates'], schemes=results,
        ratio=results['99']['median_device_ms']/results['1']['median_device_ms'],
        scope=f'{args.samples} warmed interleaved column-stage calls per scheme; repeated KTAU={args.ktau} native columns; includes W0AVG and launcher work, excludes driver composition, health readback, IO and transfers')
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'captured_inputs'}))


if __name__ == '__main__':
    main()
