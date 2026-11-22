"""MYNN-EDMF stage time, v6.1.1 against the pinned c6d08709, same card, same real inputs, interleaved (CUDA events).

Fast numerics (the default runs; strict is the proof mode). Each repetition launches the pinned module, then the
v6.1.1 module, on fresh device copies of the same captured 44c0849 call; the receipt has every sample and the
median ratio. Lane width is each module's own launch rule unless WOOF_HEX_MYNNEDMF_LANES forces one.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]


def main(a):
    import numpy as np
    from woof.hex import numerics
    numerics.set_mode(a.numerics, source='mynn-v61-timing')
    import cupy as cp
    from types import SimpleNamespace
    from edmf_replay import records, load, calls, interface_pint, v6_arguments
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c6d08709 as pinned
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_v611 as v611
    codec = records()
    device = SimpleNamespace(cuda=cp.cuda)
    backends = {
        'c6d08709': pinned.MynnEdmfCuda(compiled_image=pinned.load_cuda_packet(
            json.loads(a.pinned_packet.read_text())['packet'], device)),
        'v611': v611.MynnEdmfCuda(compiled_image=v611.load_cuda_packet(
            json.loads(a.packet.read_text())['packet'], device)),
    }
    rows = []
    for index, call in enumerate(calls(a.capture)):
        pre = load(a.capture, call['prefix'], 'pre', list(pinned.ARGUMENTS), codec)
        hosts = {
            'c6d08709': {k: (np.array(v, order='F') if v is not None and pinned.ARGUMENTS[k]['dimensions']
                             else (None if v is None else v.reshape(-1)[0].item()))
                         for k, v in pre.items() if k not in ('errmsg', 'errflg')},
            'v611': v6_arguments(pinned, pre, interface_pint(a.interface, index, codec), v611),
        }
        if a.tile > 1:
            # Each real column repeated a.tile times along i (the tiled 3 km-sized shape, as benchmark_shape.py tiles).
            for name, host in hosts.items():
                small = int(host['ime']) - int(host['ims']) + 1
                for k, v in list(host.items()):
                    if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == small:
                        host[k] = np.asfortranarray(np.repeat(v, a.tile, axis=0))
                extra = small * (a.tile - 1)
                host.update(ime=int(host['ime']) + extra, ite=int(host['ite']) + extra, ide=int(host['ide']) + extra)
        samples = {name: [] for name in backends}
        widths = {}
        for repetition in range(a.warmup + a.repeats):
            for name, backend in backends.items():
                values = {k: (cp.asarray(v, order='F') if isinstance(v, np.ndarray) else v)
                          for k, v in hosts[name].items()}
                cp.cuda.Stream.null.synchronize()
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                backend.run(values)
                stop.record()
                stop.synchronize()
                if repetition >= a.warmup:
                    samples[name].append(cp.cuda.get_elapsed_time(start, stop))
                widths[name] = backend.last_lane_width
                del values
        medians = {name: statistics.median(s) for name, s in samples.items()}
        rows.append(dict(call=call['call'], columns=int(hosts['v611']['ite']) - int(hosts['v611']['its']) + 1,
                         medians_ms=medians, ratio=medians['v611'] / medians['c6d08709'],
                         lane_width=widths, samples_ms=samples))
        print(json.dumps({k: rows[-1][k] for k in ('call', 'medians_ms', 'ratio', 'lane_width')}), flush=True)
    receipt = dict(schema='mynn-v61-stage-timing-v1', numerics=a.numerics, tile=a.tile, rows=rows,
                   median_ratio=statistics.median(r['ratio'] for r in rows),
                   lane_width_forced=os.environ.get('WOOF_HEX_MYNNEDMF_LANES'),
                   device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                   method='CUDA events around MynnEdmfCuda.run (argument packing + arena zero + kernel), interleaved')
    a.receipt.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--pinned-packet', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--interface', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--numerics', default='fast')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--tile', type=int, default=1, help='repeat every captured column N times along i')
    main(p.parse_args())
