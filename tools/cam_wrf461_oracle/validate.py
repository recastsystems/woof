"""Compare every returned column field; no missing or extra output is accepted."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load(root):
    root = Path(root)
    payload = (root / 'columns.bin').read_bytes()
    fields = {}
    with (root / 'columns.tsv').open() as stream:
        for row in csv.DictReader(stream):
            key = (row['spectrum'], int(row['column']), row['field'])
            if key in fields:
                raise ValueError(f'duplicate field {key}')
            start, count = int(row['offset_bytes']), int(row['count'])
            if start < 0 or count < 1 or start + 8*count > len(payload):
                raise ValueError(f'out-of-bounds field {key}')
            fields[key] = np.frombuffer(payload, '<f8', count, start).copy()
    for spectrum in ('lw','sw'):
        columns = {key[1] for key in fields if key[0] == spectrum and key[2].startswith('output:')}
        if columns != set(range(1,65)):
            raise ValueError(f'{spectrum}: expected columns 1..64, found {sorted(columns)}')
    return fields


def ordered(values):
    bits = values.view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where(bits & sign, ~bits, bits | sign)


def compare(reference, candidate, *, strict=True, atol=1e-10, rtol=1e-8):
    if reference.keys() != candidate.keys():
        raise ValueError('field inventory differs')
    rows=[]
    for key, expected in reference.items():
        actual = candidate[key]
        if actual.shape != expected.shape:
            raise ValueError(f'{key}: shape mismatch')
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            raise ValueError(f'{key}: nonfinite output')
        left, right = ordered(expected), ordered(actual)
        ulp = np.maximum(left,right) - np.minimum(left,right)
        different = expected.view(np.uint64) != actual.view(np.uint64)
        allowed = np.abs(actual-expected) <= atol + rtol*np.abs(expected)
        rows.append({'spectrum': key[0], 'column': key[1], 'field': key[2],
                     'count': len(expected), 'changed_words': int(different.sum()),
                     'max_ulp': int(ulp.max()), 'max_abs': float(np.abs(actual-expected).max()),
                     'pass': bool(not different.any() if strict else allowed.all())})
    return {'pass': all(row['pass'] for row in rows), 'strict': strict,
            'atol': 0.0 if strict else atol, 'rtol': 0.0 if strict else rtol,
            'fields': len(rows), 'words': sum(row['count'] for row in rows),
            'mismatches': [row for row in rows if not row['pass']]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--default',action='store_true')
    parser.add_argument('--mutation-controls',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    reference=load(args.reference)
    result=compare(reference, load(args.candidate), strict=not args.default)
    if args.mutation_controls:
        controls=[]
        for spectrum,field in (('lw','qrl'),('lw','qrlcs'),('lw','flns'),('lw','flnt'),('sw','qrs'),('sw','qrscs'),('sw','fsns'),('sw','fsntoa')):
            key=(spectrum,1,'output:'+field)
            candidate={k:v.copy() for k,v in reference.items()}
            candidate[key].view(np.uint64)[0]^=np.uint64(1)
            grade=compare(reference,candidate)
            controls.append({'spectrum':spectrum,'field':field,'detected':not grade['pass'],
                             'max_ulp':grade['mismatches'][0]['max_ulp']})
        result['mutation_controls']=controls
        result['pass']=result['pass'] and all(x['detected'] for x in controls)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':
    main()
