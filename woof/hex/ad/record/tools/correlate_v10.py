#!/usr/bin/env python3
"""Correlate Python-visible argument types with raw driver launch arguments.

A match is made only when all recorded argument bytes and geometry agree, and
when the Python observation precedes the driver call. Missing Python coverage
or a matched pointer is not proof of complete command capture.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import collections
import gzip
import json


def rows(path):
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)
    except EOFError:
        # A supervisor interruption can leave the last deflate member open.
        pass


def compatible(observer, driver):
    args, params = observer['arguments'], driver.get('parameters', [])
    if len(args) != len(params):
        return False
    for arg, param in zip(args, params):
        raw = bytes.fromhex(param.get('bytes_hex', ''))
        if arg['kind'] == 'device':
            if len(raw) != 8 or int.from_bytes(raw, 'little') != arg['base'] + arg['offset']:
                return False
        elif arg['kind'] == 'scalar':
            if raw != bytes.fromhex(arg['bytes_hex']):
                return False
        else:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', required=True)
    parser.add_argument('--observer', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    visible = list(rows(args.observer))
    visible.sort(key=lambda row: row['clock_ns'])
    counters = collections.Counter()
    pending = collections.defaultdict(collections.deque)
    pos = 0
    matches = []
    for row in rows(args.driver):
        if row['type'] != 'LaunchKernel':
            continue
        counters['driver_launches'] += 1
        while pos < len(visible) and visible[pos]['clock_ns'] <= row['clock_ns']:
            obs = visible[pos]
            pending[obs['kernel']].append(obs)
            pos += 1
        name = bytes.fromhex(row['name_hex']).decode('utf-8', 'replace')
        candidates = pending[name]
        grid, block = row['grid'], row['block']
        stream = int(row['stream'], 16) if row['stream'].startswith('0x') else 0
        for i in range(len(candidates) - 1, -1, -1):
            obs = candidates[i]
            if row['clock_ns'] - obs['clock_ns'] > 1_000_000_000:
                break
            if (obs['grid'] + [1] * (3 - len(obs['grid'])) != grid or
                obs['block'] + [1] * (3 - len(obs['block'])) != block or
                obs['stream'] != stream):
                continue
            if compatible(obs, row):
                matches.append({'driver_sequence': row['sequence'], 'observer_ordinal': obs['ordinal']})
                del candidates[i]
                counters['byte_exact_matches'] += 1
                for arg in obs['arguments']:
                    counters['typed_' + arg['kind'] + '_arguments'] += 1
                break
        else:
            counters['driver_without_matching_python_types'] += 1
    result = {'schema': 'ad.binder.launch-correlation.partial/v1', 'complete': False,
              'counters': dict(sorted(counters.items())),
              'python_launch_rows': len(visible), 'matched_pairs': matches,
              'limitations': ['only Python-visible calls with exact matching argument bytes can be typed',
                              'a matched argument does not prove buffer lifetime or access mode']}
    with gzip.open(args.out, 'wt', encoding='utf-8', compresslevel=1) as out:
        json.dump(result, out, separators=(',', ':'), sort_keys=True)
    print(json.dumps({k:v for k,v in result.items() if k != 'matched_pairs'}, sort_keys=True))


if __name__ == '__main__':
    main()
