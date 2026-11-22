#!/usr/bin/env python3
"""Join same-process native host calls to driver events by thread and clock.

This is an observation audit, never a replay plan or a census. In particular,
recorded addresses, dimensions and results are not authorized replay inputs.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

from correlate_v10 import rows


def audit(library_rows, driver_rows, slot_rows):
    waiting = {}
    calls = []
    failures = collections.Counter()
    for row in library_rows:
        key = (row.get('pid'), row.get('tid'), row.get('id'))
        if row.get('schema') != 'ad.binder.phys-library-call/v1':
            failures['bad_library_schema'] += 1
        elif row.get('event') == 'enter':
            if key in waiting:
                failures['duplicate_entry'] += 1
            else:
                waiting[key] = row
        elif row.get('event') == 'exit':
            entry = waiting.pop(key, None)
            if entry is None or entry.get('kind') != row.get('kind'):
                failures['unmatched_exit'] += 1
            elif not row.get('success') or row.get('clock_ns', 0) < entry.get('clock_ns', 0):
                failures['failed_or_backwards_call'] += 1
            else:
                calls.append({'entry': entry, 'exit': row, 'driver_types': collections.Counter(),
                              'launches': []})
        else:
            failures['unknown_library_event'] += 1
    failures['unclosed_entries'] += len(waiting)
    calls.sort(key=lambda call: call['entry']['clock_ns'])
    by_thread = collections.defaultdict(list)
    for call in calls:
        by_thread[call['entry']['tid']].append(call)
    starts = {}
    for tid, thread_calls in by_thread.items():
        starts[tid] = [call['entry']['clock_ns'] for call in thread_calls]
        if any(a['exit']['clock_ns'] > b['entry']['clock_ns']
               for a, b in zip(thread_calls, thread_calls[1:])):
            failures['overlapping_library_calls_on_thread'] += 1
    module_images = {}
    functions = {}
    for row in driver_rows:
        kind = row.get('type')
        if kind == 'ModuleImage' and row.get('status') == 0:
            module_images[row['module']] = row.get('sha256')
        elif kind == 'ModuleFunction':
            functions[row['function']] = (module_images.get(row['module']), row['module'])
        elif kind == 'ModuleUnload' and row.get('status') == 0:
            module_images.pop(row.get('a'), None)
            functions = {key: value for key, value in functions.items() if value[1] != row.get('a')}
        tid, timestamp = row.get('thread_id'), row.get('clock_ns')
        if tid not in by_thread or not isinstance(timestamp, int):
            continue
        index = bisect.bisect_right(starts[tid], timestamp) - 1
        if index < 0:
            continue
        call = by_thread[tid][index]
        if timestamp > call['exit']['clock_ns']:
            continue
        call['driver_types'][str(kind)] += 1
        if kind == 'LaunchKernel':
            function = row.get('function', '0')
            try:
                image = functions.get(int(function, 16), (None, None))[0]
                name = bytes.fromhex(row.get('name_hex', '')).decode('utf-8')
            except (TypeError, ValueError, UnicodeDecodeError):
                failures['invalid_launch_identity'] += 1
                image, name = None, ''
            call['launches'].append({'sequence': row.get('sequence'), 'name': name,
                                     'image_sha256': image})
    slots = list(slot_rows)
    for call in calls:
        entry, end = call['entry'], call['exit']['clock_ns']
        matched = [slot for slot in slots if slot.get('thread_id') == entry['tid']
                   and slot.get('begin_ns', end + 1) <= entry['clock_ns']
                   and slot.get('end_ns', 0) >= end]
        if len(matched) == 1:
            call['slot'] = matched[0].get('operation')
            call['step'] = matched[0].get('step')
        elif len(matched) > 1:
            failures['overlapping_host_slots'] += 1
            call['slot'], call['step'] = None, None
        else:
            call['slot'], call['step'] = None, None
    # Query and execution must agree on all live inputs. A query may not
    # submit kernels. Missing pairs are kept visible rather than inferred.
    pending = {}
    pair_fields = ('kind', 'pid', 'tid', 'input', 'output', 'count', 'segments',
                   'segment_size', 'stream', 'op', 'dtype')
    for call in calls:
        entry = call['entry']
        if not str(entry.get('kind', '')).startswith('cub_'):
            continue
        key = tuple(entry.get(field) for field in pair_fields)
        if entry.get('workspace') == 0:
            if call['launches']:
                failures['query_submitted_kernel'] += 1
            pending.setdefault(key, []).append(call)
        else:
            previous = pending.get(key, [])
            if len(previous) != 1:
                failures['unpaired_cub_execution'] += 1
            else:
                query = previous.pop()
                if query['exit'].get('workspace_bytes') != entry.get('workspace_bytes'):
                    failures['workspace_size_changed'] += 1
    failures['unpaired_cub_query'] += sum(map(len, pending.values()))
    counts = collections.Counter()
    signatures = collections.Counter()
    for call in calls:
        entry = call['entry']
        mode = 'query' if entry.get('workspace') == 0 and entry['kind'].startswith('cub_') else 'execute'
        counts[(call['step'], call['slot'], entry['kind'], mode)] += 1
        signatures[(entry['kind'], entry.get('op'), entry.get('dtype'), mode)] += 1
        call['driver_types'] = dict(sorted(call['driver_types'].items()))
    return {'schema': 'ad.binder.phys-library-audit/v1', 'complete': False,
            'call_count': len(calls), 'driver_launches_in_calls': sum(len(x['launches']) for x in calls),
            'counts': [{'step': a, 'slot': b, 'kind': c, 'mode': d, 'calls': n}
                       for (a, b, c, d), n in sorted(counts.items(), key=lambda x: str(x[0]))],
            'signatures': [{'kind': a, 'op': b, 'dtype': c, 'mode': d, 'calls': n}
                           for (a, b, c, d), n in sorted(signatures.items(), key=lambda x: str(x[0]))],
            'failures': {k: v for k, v in sorted(failures.items()) if v}, 'calls': calls,
            'limitations': ['Only calls intercepted in this ordinary process are represented.',
                            'Windows associate events, not their semantic inputs or outputs.',
                            'Every replay binding and host frame rule requires independent proof.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--slots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.library.open(encoding='utf-8') as source:
        library = [json.loads(line) for line in source if line.endswith('\n')]
    result = audit(library, rows(args.driver), rows(args.slots))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('call_count', 'driver_launches_in_calls',
                                                    'counts', 'signatures', 'failures')}, sort_keys=True))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
