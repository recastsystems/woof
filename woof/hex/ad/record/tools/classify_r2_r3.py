#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R2/R3 classification of an hrrrv5 partial plan (COMMON-AD 103.2, 103.3).

Nothing here seals a plan.

* Every plan operation is placed in the host slot whose [begin_ns, end_ns] interval contains
  its driver record clock (driver trace clock and host-slot clock are the same clock domain
  in the recorder). Operations outside every slot are `outside` (dycore, transport, glue).
* R3: operations in the held slots (radiation_sw, radiation_lw, cloudiness,
  o3climatology_from_MPAS, land_surface, and their allocate_/deallocate_ bookkeeping slots)
  leave the milestone-1 plan; a held upload at the slot end replaces them.
* R2: each scalar launch word of a live kernel (relocated pointer words masked under 61.2)
  is `constant_both` only if it is byte-identical at every aligned launch of every recorded
  step in BOTH recordings. Alignment: same block, same index among the live kernels of that
  block, same function. A block whose live kernel sequence differs between the recordings
  aligns nothing (`single_recording`).
"""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path

HELD = {
    'radiation_sw': 'radiation', 'radiation_lw': 'radiation', 'cloudiness': 'radiation',
    'o3climatology_from_MPAS': 'radiation',
    'allocate_radiation_sw': 'radiation', 'allocate_radiation_lw': 'radiation',
    'allocate_cloudiness': 'radiation', 'deallocate_radiation_sw': 'radiation',
    'deallocate_radiation_lw': 'radiation', 'deallocate_cloudiness': 'radiation',
    'land_surface': 'land', 'allocate_lsm': 'land', 'deallocate_lsm': 'land',
}
# P1 milestone 2 (G2, 2026-11-06): BINDER_PHYS_MILESTONE=2 records and seals radiation and land live, so no slot is
# held and nothing leaves the plan. THE BREAKAGE THIS PREVENTS: with the recorder holds off, the milestone-1 table
# would still strip every radiation and land operation and replace it with held uploads that no longer exist.
import os as _os
MILESTONE = _os.environ.get('BINDER_PHYS_MILESTONE', '1')
if MILESTONE == '2':
    HELD = {}
elif MILESTONE != '1':
    raise SystemExit(f'BINDER_PHYS_MILESTONE must be 1 or 2, not {MILESTONE!r}')
SKIP_TYPES = {'WrappedCallBegin', 'WrappedCallEnd', 'CUPTIIntervalBegin', 'CUPTIIntervalEnd'}


def load(p):
    return json.load(gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p))


def jsonl(p):
    with gzip.open(p, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def masked(arg):
    raw = arg.get('bytes_hex')
    if raw is None or arg.get('kind') != 'patched':
        return raw
    data = bytearray.fromhex(raw)
    for patch in arg.get('pointers', []):
        data[patch['byte_offset']:patch['byte_offset'] + 8] = bytes(8)
    return data.hex()


class Recording:
    def __init__(self, plan, driver, slots):
        self.plan = load(plan)
        want = {op['recorded_sequence'] for ops in self.plan['blocks'].values()
                for op in ops if 'recorded_sequence' in op}
        self.clock = {}
        for r in jsonl(driver):
            s = r.get('sequence')
            if s in want and r.get('type') not in SKIP_TYPES:
                self.clock.setdefault(s, r['clock_ns'])
        self.slots = []
        for s in jsonl(slots):
            end = s['end_ns'] if s.get('end_ns') is not None else 1 << 62
            self.slots.append((s['begin_ns'], end, s['operation'], s['step']))
        self.slots.sort()
        self.unclocked = 0

    def slot_of(self, op):
        seq = op.get('recorded_sequence')
        if seq is None or seq not in self.clock:
            self.unclocked += 1
            return 'unclocked'
        c = self.clock[seq]
        hits = [s for s in self.slots if s[0] <= c <= s[1]]
        if not hits:
            return 'outside'
        return min(hits, key=lambda s: s[1] - s[0])[2]

    def annotate(self):
        return {n: [(self.slot_of(op), op) for op in ops] for n, ops in self.plan['blocks'].items()}


def live_kernels(blocks):
    return {n: [op for slot, op in ops if op['kind'] == 'kernel' and slot not in HELD]
            for n, ops in blocks.items()}


def main():
    ap = argparse.ArgumentParser()
    for tag in ('a', 'b'):
        ap.add_argument(f'--plan-{tag}', required=True)
        ap.add_argument(f'--driver-{tag}', required=True)
        ap.add_argument(f'--slots-{tag}', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ra = Recording(a.plan_a, a.driver_a, a.slots_a)
    rb = Recording(a.plan_b, a.driver_b, a.slots_b)
    ba, bb = ra.annotate(), rb.annotate()
    rep = {'schema': 'ad.binder.phys-r2-r3/v1', 'complete': False,
           'recordings': {'a': a.plan_a, 'b': a.plan_b}}
    rep['unclocked_ops'] = {'a': ra.unclocked, 'b': rb.unclocked}

    per_slot = collections.defaultdict(collections.Counter)
    for ops in ba.values():
        for slot, op in ops:
            per_slot[slot][op['kind']] += 1
    rep['ops_by_slot_a'] = {s: dict(c) for s, c in sorted(per_slot.items())}

    held_targets = collections.defaultdict(set)
    for name, ops in ba.items():
        for slot, op in ops:
            if slot not in HELD:
                continue
            k = op['kind']
            if k in ('upload', 'clear', 'device_copy'):
                d = op.get('destination') or op.get('target')
                if d:
                    held_targets[(name, slot)].add(d['allocation'])
            elif k == 'kernel':
                for arg in op['arguments']:
                    if arg.get('kind') == 'device':
                        held_targets[(name, slot)].add(arg['target']['allocation'])
    rep['held_slot_device_allocations'] = {f'{n}:{s}': len(v) for (n, s), v in sorted(held_targets.items())}

    ka, kb = live_kernels(ba), live_kernels(bb)
    mismatch = sorted(n for n in ka if [k['function'] for k in ka[n]] != [k['function'] for k in kb.get(n, [])])
    rep['blocks_live_sequence_mismatch'] = mismatch
    groups = collections.defaultdict(list)
    for n in ka:
        groups[tuple(k['function'] for k in ka[n])].append(n)
    counts = collections.Counter()
    per_fn = collections.defaultdict(collections.Counter)
    needs = []
    table_a, table_b = [], []
    for names in groups.values():
        n0 = names[0]
        for i, k0 in enumerate(ka[n0]):
            for j, arg in enumerate(k0['arguments']):
                if arg.get('kind') not in ('bytes', 'patched'):
                    continue
                wa = {masked(ka[n][i]['arguments'][j]) for n in names}
                if any(n in mismatch for n in names):
                    v, wb = 'single_recording', set()
                else:
                    wb = {masked(kb[n][i]['arguments'][j]) for n in names}
                    if len(wa) > 1 or len(wb) > 1:
                        v = 'varies_within'
                    elif wa != wb:
                        v = 'differs_between'
                    else:
                        v = 'constant_both'
                key = f"{','.join(names)}|{i}|{k0['function']}|{j}"
                table_a.append(f'{key}|{sorted(wa)}')
                table_b.append(f'{key}|{sorted(wb)}')
                for n in names:
                    counts[v] += 1
                    per_fn[k0['function']][v] += 1
                    if v != 'constant_both':
                        needs.append({'block': n, 'kernel_index': i, 'function': k0['function'],
                                      'argument': j, 'verdict': v,
                                      'a': sorted(x for x in wa if x)[:4], 'b': sorted(x for x in wb if x)[:4]})
    rep['r2_counts'] = dict(counts)
    rep['r2_word_table_sha256'] = {'a': hashlib.sha256('\n'.join(table_a).encode()).hexdigest(),
                                   'b': hashlib.sha256('\n'.join(table_b).encode()).hexdigest()}
    rep['r2_constant_by_function'] = {f: c['constant_both'] for f, c in sorted(per_fn.items())}
    rep['r2_needs_live_rule_by_function'] = {f: {k: v for k, v in c.items() if k != 'constant_both'}
                                             for f, c in sorted(per_fn.items()) if set(c) - {'constant_both'}}
    rep['r2_needs_live_rule'] = needs

    d2h, alloc, up, unres = [], [], [], []
    for name, ops in ba.items():
        for slot, op in ops:
            if slot in HELD:
                continue
            k = op['kind']
            if k == 'device_to_host':
                d2h.append([name, slot, op.get('recorded_sequence'), op.get('bytes')])
            elif k == 'allocate':
                alloc.append([name, slot, op.get('id'), op.get('bytes')])
            elif k == 'upload' and not op.get('data'):
                up.append([name, slot, op.get('recorded_sequence'), op.get('destination'), op.get('bytes')])
            elif k == 'kernel':
                for j, arg in enumerate(op['arguments']):
                    if arg.get('kind') not in ('device', 'bytes', 'patched', 'module_global'):
                        unres.append([name, slot, op['function'], j, arg.get('kind')])
    rep['live_residual'] = {'device_to_host': len(d2h), 'allocate': len(alloc),
                            'upload_without_bytes': len(up), 'unresolved_kernel_arguments': len(unres)}
    rep['live_device_to_host_by_slot'] = dict(collections.Counter(f'{x[0]}:{x[1]}' for x in d2h))
    rep['live_allocate_by_slot'] = dict(collections.Counter(f'{x[0]}:{x[1]}' for x in alloc))
    rep['live_upload_without_bytes_by_slot'] = dict(collections.Counter(f'{x[0]}:{x[1]}' for x in up))
    rep['live_unresolved_by_kind'] = dict(collections.Counter(f'{x[1]}:{x[2]}:{x[4]}' for x in unres))
    rep['live_lists'] = {'device_to_host': d2h, 'allocate': alloc, 'upload_without_bytes': up, 'unresolved': unres}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=list))
    for k in ('unclocked_ops', 'r2_counts', 'live_residual', 'blocks_live_sequence_mismatch',
              'r2_needs_live_rule_by_function', 'live_device_to_host_by_slot', 'live_allocate_by_slot',
              'live_upload_without_bytes_by_slot', 'live_unresolved_by_kind'):
        print(k, json.dumps(rep[k], default=list))


if __name__ == '__main__':
    main()
