"""Fail-closed full-field gate, with compact WRF column reference receipts.

Hashes cover each field of each computed frame, including signed zero and
NaN payloads. Equal hashes therefore imply zero differing binary32 words.
The declared default and strict tolerances are both zero ULP. The reference
receipt can travel in the source bundle without transferring column dumps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def field_hashes(index, case, directory):
    ncol, nk, nsteps = case['ncol'], case['nk'], case['nsteps']
    path = Path(directory) / (case['name'] + '.ref')
    raw = path.read_bytes()
    fields = [(n, nk * ncol) for n in index['out3d']]
    fields += [(n, ncol) for n in index['out2d']]
    per = sum(size for _, size in fields)
    if len(raw) != 4 * per * (nsteps + 1):
        raise ValueError(f'{path}: {len(raw)} bytes, expected {4 * per * (nsteps + 1)}')
    words = np.frombuffer(raw, dtype='<u4').reshape(nsteps + 1, per)
    result = {}
    offset = 0
    for name, size in fields:
        result[name] = [hashlib.sha256(words[step, offset:offset+size].tobytes()).hexdigest()
                        for step in range(1, nsteps+1)]
        offset += size
    return {'file_sha256': hashlib.sha256(raw).hexdigest(), 'fields': result,
            'computed_words': per * nsteps}


def verify(index, reference, directory):
    failures, checked = [], 0
    rows = {}
    cases = [c for c in index['cases'] if c.get('in_claim', True)]
    if not cases:
        failures.append({'cause': 'no admitted cases'})
    for case in cases:
        name = case['name']
        if name not in reference['cases']:
            failures.append({'case': name, 'cause': 'missing reference'})
            continue
        expected = reference['cases'][name]
        fixture = Path(directory).parent  # fixture digest is independently checked by runner
        if case['sha256'] != expected['input_sha256']:
            failures.append({'case': name, 'cause': 'input digest changed'})
            continue
        try:
            got = field_hashes(index, case, directory)
        except (OSError, ValueError) as exc:
            failures.append({'case': name, 'cause': str(exc)})
            continue
        checked += got['computed_words']
        rows[name] = got
        for field, hashes in got['fields'].items():
            want = expected['fields'].get(field, [])
            for step, digest in enumerate(hashes, 1):
                if step > len(want) or digest != want[step-1]:
                    failures.append({'case': name, 'field': field, 'step': step,
                                     'expected': want[step-1] if step <= len(want) else None,
                                     'actual': digest})
        if set(got['fields']) != set(expected['fields']):
            failures.append({'case': name, 'cause': 'field inventory changed'})
    return {'pass': bool(checked) and not failures, 'tolerance_ulp': 0,
            'computed_words': checked, 'case_count': len(cases),
            'columns': max((c['ncol'] for c in cases), default=0),
            'failures': failures, 'cases': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['reference', 'verify'])
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('outputs', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    index = json.loads((args.fixtures / 'index.json').read_text())
    for case in index['cases']:
        digest = hashlib.sha256((args.fixtures / (case['name']+'.bin')).read_bytes()).hexdigest()
        if digest != case['sha256']:
            raise ValueError(f"fixture {case['name']} digest changed")
    if args.mode == 'reference':
        result = {'schema': 'p3-full-field-reference-v1', 'variant': index['variant'],
                  'wrf_source_sha256': '716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6',
                  'wrf_commit': 'd66e442fccc04111067e29274c9f9eaccc3cef28',
                  'fortran_arm': 'CR', 'cases': {}}
        for case in index['cases']:
            if not case.get('in_claim', True): continue
            row = field_hashes(index, case, args.outputs)
            row.update(input_sha256=case['sha256'], ncol=case['ncol'], nk=case['nk'],
                       nsteps=case['nsteps'], dt=case['dt'])
            result['cases'][case['name']] = row
        args.reference.write_text(json.dumps(result, indent=1)+'\n')
        return 0
    reference = json.loads(args.reference.read_text())
    if reference['variant'] != index['variant']:
        raise ValueError('reference variant differs from fixture variant')
    result = verify(index, reference, args.outputs)
    if args.json: args.json.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases', 'failures')}, sort_keys=True))
    print('differing field frames:', len(result['failures']))
    for row in result['failures'][:10]: print(json.dumps(row))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
