"""Replay ETARA calls captured at the argument boundary of a real WRF run.

Use GPUWM_WRF_EXACT=1. The instrument_real.py hook writes big-endian streams
from the strict native build. Every packed output word is checked, including
unchanged inputs, cloud counters and the saved solar clock. No tolerance or
undefined-value exemption is applied here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_capture(path):
    raw = path.read_bytes()
    words = np.frombuffer(raw, dtype='>u4').astype(np.uint32)
    h = words[:8].view(np.int32)
    if tuple(h[[0, 3, 4, 5]]) != (990461, 1177, 27, 123192):
        raise ValueError(f'{path}: invalid native capture header {h.tolist()}')
    nc, nz = int(h[1]), int(h[2])
    if not 1 <= nz <= 81 or not 1 <= nc <= 1024:
        raise ValueError(f'{path}: invalid column or layer count')
    pos = 8

    def take(n):
        nonlocal pos
        v = words[pos:pos+n].copy()
        if v.size != n:
            raise ValueError(f'{path}: truncated capture')
        pos += n
        return v

    table = take(123192)
    before_clock = take(4)
    before_f = take(nc * 1177).reshape(nc, 1177)
    before_i = take(nc * 27).reshape(nc, 27)
    after_clock = take(4)
    after_f = take(nc * 1177).reshape(nc, 1177)
    after_i = take(nc * 27).reshape(nc, 27)
    if pos != words.size:
        raise ValueError(f'{path}: trailing words: {words.size-pos}')
    return h, table, before_clock, before_f, before_i, after_clock, after_f, after_i


def compare(actual, expected):
    actual = np.asarray(actual, dtype=np.uint32)
    expected = np.broadcast_to(expected, actual.shape)
    bad = np.argwhere(actual != expected)
    return {'words': int(actual.size), 'differing_words': int(len(bad)),
            'first': [{'index': p.tolist(), 'gpu': f'{int(actual[tuple(p)]):08x}',
                       'native': f'{int(expected[tuple(p)]):08x}'} for p in bad[:12]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('GPUWM_WRF_EXACT') != '1':
        p.error('real native bitwise replay requires GPUWM_WRF_EXACT=1')
    import cupy as cp
    from woof.core.kernels import get_kernel, module_options
    layout = json.loads((ROOT/'woof/core/gfdleta_column_layout.json').read_text())
    results = []
    captures = []
    kernel = get_kernel('gfdleta_column', 'gd_radiation_call')
    for path in a.captures:
        h, table, bc, bf, bi, ac, af, ai = read_capture(path)
        captures.append((path, h, table, bf, bi, af, ai))
        nc = int(h[1])
        t = cp.asarray(table)
        f = cp.asarray(bf).view(cp.float32)
        ints = cp.asarray(bi).view(cp.int32)
        clocks = cp.asarray(np.tile(bc, (nc, 1))).view(cp.float32)
        scratch = cp.empty((nc, layout['scratch_bytes']), cp.uint8)
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        kernel(((nc+31)//32,), (32,), (t, f, ints, scratch, clocks, np.int32(nc)))
        end.record()
        end.synchronize()
        got = cp.asnumpy(f).view(np.uint32)
        checks = {'floats': compare(got, af), 'integers': compare(cp.asnumpy(ints).view(np.uint32), ai),
                  'clock': compare(cp.asnumpy(clocks).view(np.uint32), ac),
                  'table_unchanged': compare(cp.asnumpy(t), table)}
        fields = {}
        for d in layout['fields']:
            if d['kind'] == 'real':
                fields[d['name']] = compare(got[:, d['offset']:d['offset']+d['count']],
                                           af[:, d['offset']:d['offset']+d['count']])
        descriptors = {d['name']: d for d in layout['fields']}
        def values(name, words):
            d = descriptors[name]
            dtype = np.float32 if d['kind'] == 'real' else np.int32
            return words[:, d['offset']].copy().view(dtype).tolist()
        results.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'header': h.tolist(), 'input_kind': 'real-native-wrf-run',
                        'coverage': {'gfdl_lw': values('gfdl_lw', bi),
                                     'gfdl_sw': values('gfdl_sw', bi),
                                     'cosine_zenith_after': values('czmean', af),
                                     'shortwave_down_after': values('rswin', af),
                                     'julian': values('julian', bf),
                                     'gmt': values('gmt', bf),
                                     'xtime': values('xtime', bf)},
                        'cuda_elapsed_ms': cp.cuda.get_elapsed_time(start, end),
                        'checks': checks, 'fields': fields})
    # Replay the production pooled entry point from the same live LW entry
    # to the live SW exit. Fewer slots than columns also checks scratch reuse.
    # Pair by native call number and directory, never by supplied file order.
    production = []
    pooled = get_kernel('gfdleta_column', 'gd_radiation_columns_pooled')
    for first in captures:
        path, h, table, bf, bi, _, _ = first
        if int(h[7]) != 1:
            continue
        matches = [last for last in captures
                   if last[0].parent == path.parent and int(last[1][7]) == 2]
        if len(matches) != 1:
            p.error(f'{path}: production replay needs exactly one native SW exit in the same directory')
        last = matches[0]
        if not np.array_equal(h[1:7], last[1][1:7]) or not np.array_equal(table, last[2]):
            p.error(f'{path}: native LW/SW captures have different grids or tables')
        nc = int(h[1])
        slots = min(3, nc)
        t = cp.asarray(table)
        f = cp.asarray(bf).view(cp.float32)
        ints = cp.asarray(bi).view(cp.int32)
        scratch = cp.empty((slots, layout['scratch_bytes']), cp.uint8)
        pooled(((slots+31)//32,), (32,),
               (t, f, ints, scratch, np.int32(nc), np.int32(slots), np.int32(1), np.int32(1)))
        production.append({'native_lw_entry': str(path), 'native_sw_exit': str(last[0]),
                           'columns': nc, 'scratch_slots': slots,
                           'checks': {'floats': compare(cp.asnumpy(f).view(np.uint32), last[5]),
                                      'integers': compare(cp.asnumpy(ints).view(np.uint32), last[6]),
                                      'table_unchanged': compare(cp.asnumpy(t), table)}})
    comparisons = results + production
    result = {'pass': all(c['differing_words'] == 0 for r in comparisons for c in r['checks'].values()),
              'strict': True, 'module_options': module_options('gfdleta_column'),
              'native_pin': 'd66e442fccc04111067e29274c9f9eaccc3cef28', 'captures': results,
              'production_pooled': production,
              'code_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted((ROOT/'woof/core/kernels').glob('gfdleta*'))
                              if path.is_file()}}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'pass': result['pass'], 'captures': len(results),
                      'production_pairs': len(production),
                      'differing_words': sum(c['differing_words'] for r in comparisons for c in r['checks'].values())}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
