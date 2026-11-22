#!/usr/bin/env python3
"""Find same-recording allocation bindings for observed native host calls.

The output is diagnostic: source expressions, dynamic control flow, arena
lifetimes, and the single-stream ordering must still be proven before a call
can replace recorded CUDA commands in a replay plan.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import bisect
import collections
import json
from pathlib import Path

from correlate_v10 import rows
from make_partial_plan import AllocationLookup


def audit(catalog, library, driver):
    if catalog.get('complete') is not False or library.get('failures'):
        raise ValueError('expected an error-free, explicitly partial observation')
    pointer = AllocationLookup(catalog['allocations']).pointer
    timed = []
    commands = {'LaunchKernel', 'Memcpy', 'MemcpyAsync', 'MemcpyHtoD', 'MemcpyHtoDAsync',
                'MemcpyDtoH', 'MemcpyDtoHAsync', 'MemcpyDtoD', 'MemcpyDtoDAsync',
                'MemsetD8', 'MemsetD8Async', 'MemsetD16', 'MemsetD16Async',
                'MemsetD32', 'MemsetD32Async', 'MemAlloc', 'MemAllocAsync',
                'MemFree', 'MemFreeAsync', 'EventRecord', 'StreamWaitEvent',
                'EventSynchronize', 'StreamSynchronize', 'CtxSynchronize'}
    for row in driver:
        timestamp = row.get('clock_ns')
        if row.get('type') in commands and isinstance(timestamp, int):
            timed.append((timestamp, row['sequence'], row.get('thread_id'), row['type']))
    timed.sort()
    times = [row[0] for row in timed]
    candidates = []
    failures = collections.Counter()
    for call in library['calls']:
        entry, end = call['entry'], call['exit']
        if call.get('step') is None:
            continue
        kind = entry.get('kind')
        if kind.startswith('cub_') and not entry.get('workspace'):
            continue  # query was checked against its corresponding execution
        clock = entry['clock_ns']
        reasons = []
        if entry.get('stream') != 0 or entry.get('dtype') not in (4, 6):
            reasons.append('unsupported stream or dtype')
        word = {4: 4, 6: 8}.get(entry.get('dtype'), 0)
        bindings = {}
        if kind == 'cub_reduce':
            count = entry['count']
            extents = {'input': count * word, 'output': word,
                       'workspace': entry['workspace_bytes']}
        elif kind == 'cub_segmented_reduce':
            count = entry['segments'] * entry['segment_size']
            extents = {'input': count * word, 'output': entry['segments'] * word,
                       'workspace': entry['workspace_bytes']}
        elif kind == 'thrust_argsort':
            shape = entry['shape']
            if len(shape) != 1 or entry.get('keys') != 0:
                reasons.append('unsupported thrust shape or keys')
                count = 0
            else:
                count = shape[0]
            extents = {'input': count * word, 'indices': count * 8}
        else:
            failures['unsupported_call_kind'] += 1
            continue
        if not isinstance(count, int) or count <= 0 or any(size <= 0 for size in extents.values()):
            reasons.append('invalid extent')
        else:
            for name, size in extents.items():
                try:
                    found = pointer(entry[name], clock, size)
                except (KeyError, ValueError):
                    found = None
                if found is None:
                    reasons.append('unmapped ' + name)
                else:
                    bindings[name] = {'target': found, 'bytes': size}
        lo = bisect.bisect_left(times, clock)
        hi = bisect.bisect_right(times, end['clock_ns'])
        inside = timed[lo:hi]
        own = [item for item in inside if item[2] == entry['tid']]
        other = [item for item in inside if item[2] != entry['tid']]
        launches = [item[1] for item in own if item[3] == 'LaunchKernel']
        recorded = [launch['sequence'] for launch in call['launches']]
        if launches != recorded or not launches:
            reasons.append('launch interval mismatch')
        # multirank (L20): in the 2-rank recording each of the 8 thrust argsorts of radiation_lw (rank-local layer
        # arrays) encloses one MemcpyDtoDAsync that no one-rank recording had. It is issued inside the library call,
        # like its launches, so the native replacement (the same thrust entry) issues it again.
        allowed = {'LaunchKernel'} if kind.startswith('cub_') else {'LaunchKernel', 'MemsetD8Async', 'MemcpyDtoDAsync'}
        if any(item[3] not in allowed for item in own):
            reasons.append('unsupported enclosed command')
        if other:
            reasons.append('another thread issued a command during the host call')
        failures.update(reasons)
        candidates.append({'step': call['step'], 'slot': call.get('slot'), 'kind': kind,
                           'call_id': entry['id'], 'bindings': bindings,
                           'launch_sequences': launches,
                           'enclosed_commands': [{'sequence': seq, 'type': typ}
                                                 for _, seq, _, typ in own],
                           'binding_errors': reasons,
                           'live_input_derivation_proven': False,
                           'workspace_lifetime_proven': False,
                           'single_stream_replacement_proven': False})
    counts = collections.Counter(candidate['kind'] for candidate in candidates)
    return {'schema': 'ad.binder.phys-library-binding-audit/v1', 'complete': False,
            'candidates': candidates, 'counts': dict(sorted(counts.items())),
            'binding_errors': dict(sorted(failures.items())),
            'limitations': ['Address matching is not ownership or dataflow proof.',
                            'No captured address or result is a replay input.',
                            'Thrust allocator workspace and all live input rules remain unproved.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--driver', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.catalog.read_text()), json.loads(args.library.read_text()),
                   rows(args.driver))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'counts': result['counts'], 'binding_errors': result['binding_errors'],
                      'native_replay_proven': False}, sort_keys=True))


if __name__ == '__main__':
    main()
