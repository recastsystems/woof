#!/usr/bin/env python3
"""Associate observed SGEAM host calls with same-run launches and device roots.

This check never makes a replay plan: observed scalars and addresses are not
replay inputs, and physical roots do not establish logical ownership.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import collections
import json
from pathlib import Path

from correlate_v10 import rows
from make_partial_plan import AllocationLookup


def audit(call_rows, driver_rows, slot_rows):
    pending, calls = {}, []
    errors = collections.Counter()
    for row in call_rows:
        key = tuple(row.get(name) for name in ('pid', 'tid', 'id'))
        if row.get('schema') != 'ad.binder.phys-cublas-geam/v1':
            errors['schema'] += 1
        elif row.get('event') == 'enter':
            if key in pending:
                errors['duplicate_entry'] += 1
            else:
                pending[key] = row
        elif row.get('event') == 'exit':
            entry = pending.pop(key, None)
            if not entry or row.get('status') != 0 or row.get('clock_ns', 0) < entry['clock_ns']:
                errors['bad_exit'] += 1
            else:
                calls.append({'entry': entry, 'exit': row, 'launches': []})
        else:
            errors['event'] += 1
    errors['unclosed'] += len(pending)
    calls.sort(key=lambda row: row['entry']['clock_ns'])
    roots, live, commands = [], {}, []
    for row in driver_rows:
        kind, clock = row.get('type'), row.get('clock_ns')
        if kind in ('MemAlloc', 'MemAllocAsync') and row.get('status') == 0:
            base = row['a']
            if base in live:
                errors['duplicate_live_root'] += 1
                continue
            root = {'id': len(roots) + 1, 'recorded_base': base, 'bytes': row['count'],
                    'allocated_clock_ns': clock, 'freed_clock_ns': None}
            live[base] = root
            roots.append(root)
        elif kind in ('MemFree', 'MemFreeAsync') and row.get('status') == 0 and row.get('a'):
            root = live.pop(row['a'], None)
            if root is None:
                errors['unknown_freed_root'] += 1
            else:
                root['freed_clock_ns'] = clock
        if kind in {'LaunchKernel', 'Memcpy', 'MemcpyAsync', 'MemcpyHtoD', 'MemcpyHtoDAsync',
                    'MemcpyDtoH', 'MemcpyDtoHAsync', 'MemcpyDtoD', 'MemcpyDtoDAsync',
                    'MemsetD8', 'MemsetD8Async', 'MemsetD16', 'MemsetD16Async',
                    'MemsetD32', 'MemsetD32Async', 'MemAlloc', 'MemAllocAsync',
                    'MemFree', 'MemFreeAsync', 'EventRecord', 'StreamWaitEvent',
                    'EventSynchronize', 'StreamSynchronize', 'CtxSynchronize'}:
            commands.append(row)
    pointer = AllocationLookup(roots).pointer
    slots = list(slot_rows)
    for call in calls:
        entry, exit = call['entry'], call['exit']
        begin, end = entry['clock_ns'], exit['clock_ns']
        same_slot = [slot for slot in slots if slot.get('thread_id') == entry['tid']
                     and slot.get('begin_ns', end + 1) <= begin and slot.get('end_ns', 0) >= end]
        if len(same_slot) != 1:
            errors['slot_association'] += 1
        else:
            call['slot'] = same_slot[0]['operation']
            call['step'] = same_slot[0]['step']
        relevant = [row for row in commands if begin <= row.get('clock_ns', -1) <= end]
        if any(row.get('thread_id') != entry['tid'] for row in relevant):
            errors['overlapping_other_thread'] += 1
        launches = [row for row in relevant if row['type'] == 'LaunchKernel']
        # cuBLAS splits a transpose whose n exceeds the grid's 65,535 limit into launches of at most 65,535 columns
        # each (n = 103,950 on the 96-to-24 km zoom globe: 65,535 then 38,415, A and C advanced by the columns done).
        # THE BREAKAGE THIS PREVENTS: every one-card recording of a mesh above 65,535 columns was refused at sealing
        # ("launch_count"; zoom globe 1 h plan, 2026-11-18), though the seal already re-issues the whole SGEAM
        # (seal_phys_stage2: the first launch carries the call, the others are dropped).
        if not launches:
            errors['launch_count'] += 1
            continue
        names, done = [], 0
        try:
            for launch in launches:
                name = bytes.fromhex(launch['name_hex']).decode('utf-8')
                names.append(name)
                pars = [bytes.fromhex(arg['bytes_hex']) for arg in launch['parameters']]
                scalar = pars[0]
                ints = [int.from_bytes(scalar[i:i + 4], 'little') for i in (0, 8, 12, 16, 20)]
                addresses = [int.from_bytes(pars[i], 'little') for i in (1, 2)]
                part = ints[2]
                if not (name.startswith('_Z36transpose_readWrite_alignment_kernel')
                        and len(scalar) == 48 and len(pars[1]) == len(pars[2]) == 8 and 0 < part
                        and ints[:2] + ints[3:] == [entry['alpha_bits'], entry['m'], entry['lda'], entry['ldc']]
                        and addresses == [entry['a'] + 4 * done, entry['c'] + 4 * done * entry['ldc']]):
                    errors['launch_arguments'] += 1
                done += part
            if done != entry['n']:
                errors['launch_arguments'] += 1
        except (KeyError, ValueError, IndexError, UnicodeDecodeError):
            errors['launch_decode'] += 1
        call['launches'] = [{'sequence': launch.get('sequence'), 'name': name}
                            for launch, name in zip(launches, names + [None] * (len(launches) - len(names)))]
        if not (entry.get('pointer_mode') == 0 and entry.get('mode_status') == 0
                and entry.get('stream_status') == 0 and entry.get('stream') == 0
                and entry.get('host_scalars') is True and entry.get('transa') == 1
                and entry.get('transb') == 1 and entry.get('alpha_bits') == 0x3f800000
                and entry.get('beta_bits') == 0 and entry.get('a') == entry.get('b')):
            errors['unsupported_observed_call'] += 1
            continue
        m, n, lda, ldb, ldc = (entry[key] for key in ('m', 'n', 'lda', 'ldb', 'ldc'))
        if m <= 0 or n <= 0 or lda < n or ldb < n or ldc < m:
            errors['invalid_matrix_extent'] += 1
            continue
        extents = {'a': ((m - 1) * lda + n) * 4, 'b': ((m - 1) * ldb + n) * 4,
                   'c': ((n - 1) * ldc + m) * 4}
        call['bindings'] = {}
        for name, size in extents.items():
            try:
                binding = pointer(entry[name], begin, size)
            except ValueError:
                binding = None
            if binding is None:
                errors['unmapped_' + name] += 1
            call['bindings'][name] = {'root': binding, 'bytes': size}
    return {'schema': 'ad.binder.phys-cublas-geam-audit/v1', 'complete': False,
            'call_count': len(calls), 'calls': calls,
            'errors': {k: v for k, v in sorted(errors.items()) if v},
            'limitations': ['Only same-run observed calls are represented.',
                            'Physical roots and address matches do not prove logical ownership.',
                            'Live scalar derivation, dataflow, ordering and replay equality remain unproved.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calls', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--slots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.calls.open(encoding='utf-8') as source:
        observed = [json.loads(line) for line in source if line.endswith('\n')]
    result = audit(observed, rows(args.driver), rows(args.slots))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'calls': result['call_count'], 'errors': result['errors'],
                      'locations': [{'step': call.get('step'), 'slot': call.get('slot'),
                                     'sequence': (call.get('launches') or [{}])[0].get('sequence')}
                                    for call in result['calls']]}, sort_keys=True))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
