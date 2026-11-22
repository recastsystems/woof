#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 (ad/phys): witness for the TEMPO excluded residue from two ordinary forwards of the same case and build.

For every recorded boundary, compares the TEMPO scheduling words (cost, order, hist: tempo-sched/step-N.json from
record_observer.py) of forward A and forward B entry by entry, and lists the per-warp scratch arrays (state, work,
PBL work) both forwards hold. The words are timing-based (clock64 cycle counts and an atomic scatter), so two runs
of the same forecast give different values: that is what this file records. The second half of the witness, that
nothing outside these arrays depends on them, is the replay's compare at every step boundary (runner events
excluded_residue: masked_equal true while unmasked_equal is false).
Usage: residue_witness.py OBSERVER_A OBSERVER_B OUT.json
"""
import hashlib
import json
import sys
from pathlib import Path


def rows(observer, step):
    data = json.load(open(Path(observer) / 'tempo-sched' / f'step-{step}.json'))
    return {(json.dumps(r['key']), r['name']): r for r in data['rows']}


def witness(observer_a, observer_b, out):
    steps = sorted(int(p.stem.split('-')[1]) for p in (Path(observer_a) / 'tempo-sched').glob('step-*.json'))
    per_step, totals = [], {}
    for step in steps:
        a, b = rows(observer_a, step), rows(observer_b, step)
        if set(a) != set(b):
            sys.exit(f'refusing: forwards hold different TEMPO entries at step {step}')
        entries = []
        for key in sorted(a):
            ra, rb = a[key], b[key]
            if ra['bytes'] != rb['bytes']:
                sys.exit(f'refusing: TEMPO {key} has two sizes at step {step}')
            row = {'key': key[0], 'array': key[1], 'bytes': ra['bytes']}
            if 'hex' in ra:
                xa, xb = bytes.fromhex(ra['hex']), bytes.fromhex(rb['hex'])
                row['differing_bytes'] = sum(1 for p, q in zip(xa, xb) if p != q)
                t = totals.setdefault(key[1], [0, 0])
                t[0] += row['differing_bytes']
                t[1] += ra['bytes']
            entries.append(row)
        per_step.append({'step': step, 'entries': entries})
    summary = (f'{len(steps)} boundaries; differing bytes between the two forwards: ' +
               ', '.join(f'{name} {d} of {n}' for name, (d, n) in sorted(totals.items())))
    doc = {'schema': 'ad.binder.tempo-residue-witness/v1', 'forward_a': str(observer_a), 'forward_b': str(observer_b),
           'summary': summary, 'steps': per_step}
    text = json.dumps(doc, indent=1)
    Path(out).write_text(text)
    return {'path': Path(out).name, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'summary': summary}


if __name__ == '__main__':
    print(json.dumps(witness(*sys.argv[1:4])))
