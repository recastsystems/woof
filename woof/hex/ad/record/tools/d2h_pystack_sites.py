#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Site every live device-to-host copy of an hrrrv5 partial plan from the tracer's own
PyStack records (BINDER_D2H_STACKS=1, driver_trace_cupti_held.c). 

The tracer writes a PyStack record immediately before each device-to-host copy it wraps
(BEFORE_MemcpyDtoH*, and the generic MemcpyAsync DtoH branch), on the copying thread.
For each live (non-held, R3) plan device_to_host op, the matching PyStack record is the one
with the largest sequence below the op's recorded_sequence on the same thread with the
same byte count. Nothing here seals a plan.
Usage: d2h_pystack_sites.py PLAN DRIVER SLOTS OUT.json
"""
import collections
import json
import sys

from classify_r2_r3 import HELD, Recording, jsonl

plan, driver, slots, out = sys.argv[1:5]
rec = Recording(plan, driver, slots)
want = {}
for name, ops in rec.annotate().items():
    for slot, op in ops:
        if op['kind'] == 'device_to_host' and slot not in HELD:
            want[op['recorded_sequence']] = (name, slot, op.get('bytes'))
stacks = collections.defaultdict(list)  # thread -> [(seq, bytes, frames)]
thread_of = {}
for r in jsonl(driver):
    t = r.get('type')
    if t == 'PyStack':
        stacks[r.get('thread_id')].append((r['sequence'], r.get('bytes'), r.get('frames', '')))
    s = r.get('sequence')
    if s in want and s not in thread_of and t not in ('WrappedCallBegin', 'WrappedCallEnd'):
        thread_of[s] = r.get('thread_id')
groups = collections.defaultdict(list)
unsited = []
for seq, (name, slot, nbytes) in sorted(want.items()):
    cand = [x for x in stacks.get(thread_of.get(seq), []) if x[0] < seq and (nbytes is None or x[1] == nbytes)]
    if not cand:
        unsited.append({'sequence': seq, 'block': name, 'slot': slot, 'bytes': nbytes})
        continue
    frames = max(cand)[2]
    hex_frames = [f for f in frames.split(';') if f.startswith('woof/hex/')] or frames.split(';')
    # innermost hexcore frame that is not the generic container/array helper
    site = next((f for f in hex_frames if 'cuda_backend/containers.py' not in f), hex_frames[0])
    groups[(site, slot)].append({'sequence': seq, 'block': name, 'bytes': nbytes, 'frames': frames})
rows = []
for (site, slot), v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    rows.append({'site': site, 'slot': slot, 'count': len(v),
                 'blocks': dict(collections.Counter(x['block'] for x in v)),
                 'bytes': sorted({x['bytes'] for x in v}),
                 'sequences': [x['sequence'] for x in v],
                 'stacks': sorted({x['frames'] for x in v})[:4]})
rep = {'schema': 'ad.binder.phys-d2h-pystack-sites/v1', 'complete': False,
       'live_device_to_host': len(want), 'sited': len(want) - len(unsited),
       'unsited': unsited, 'sites': rows}
json.dump(rep, open(out, 'w'), indent=1)
print('live', len(want), 'sited', len(want) - len(unsited), 'unsited', len(unsited))
for r in rows:
    print(r['count'], r['site'], r['slot'], r['bytes'], r['blocks'])
