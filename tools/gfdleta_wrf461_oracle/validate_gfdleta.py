"""Compare the float32 words emitted by the stock WRF column oracle.

This is validation code. It performs no production field processing.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

SURFACE_FIELDS = frozenset('GLW GSW SWDOWN SWDOWNC RLWTOA RSWTOA CFRACL CFRACM CFRACH'.split())
LEVEL_FIELDS = frozenset('RTHRATENLW RTHRATENSW'.split())
TABLE_FIELDS = frozenset('STEMP GTEMP CO251 CDT51 C2D51 CO258 CDT58 C2D58 CO2M51 CDTM51 C2DM51 CO2M58 CDTM58 C2DM58 CO231 CDT31 C2D31 CO238 CDT38 C2D38 CO271 CDT71 C2D71 CO278 CDT78 C2D78 EM1 EM1WDE EM3 TABLE1 TABLE2 TABLE3 SOURCE DSRCE XDUO3N XDO3N2 XDO3N3 XDO3N4 PRGFDL AXSD SOLARD TABLE_AUX'.split())

def read_words(path):
    result = {}
    with Path(path).open(newline='', encoding='ascii') as stream:
        reader = csv.DictReader(stream)
        schema = reader.fieldnames
        if schema not in [['column','level','field','bits'], ['field','index','bits'], ['field','column','index','bits']]:
            raise ValueError(f'{path}: unexpected CSV schema {schema!r}')
        keys = [x for x in schema if x != 'bits']
        for number, row in enumerate(reader, 2):
            if None in row or any(row[x] is None for x in schema):
                raise ValueError(f'{path}:{number}: malformed row')
            key = tuple(row[x] for x in keys)
            if key in result:
                raise ValueError(f'{path}:{number}: duplicate key {key}')
            word = row['bits']
            if len(word) != 8 or any(x not in '0123456789ABCDEFabcdef' for x in word):
                raise ValueError(f'{path}:{number}: not an eight-digit float32 word')
            bits = int(word, 16)
            if bits & 0x7f800000 == 0x7f800000:
                raise ValueError(f'{path}:{number}: nonfinite word {word}')
            result[key] = (row['field'],bits)
    if not result:
        raise ValueError(f'{path}: empty fixture')
    return result

def _float(bits):
    return struct.unpack('>f', struct.pack('>I',bits))[0]

def _rank(bits):
    return int(monotone_fp32_key(np.array([bits], dtype=np.uint32).view(np.float32))[0])

def compare_words(reference, candidate):
    missing = sorted(set(reference)-set(candidate))
    extra = sorted(set(candidate)-set(reference))
    fields = {}
    first = None
    for key in sorted(set(reference) & set(candidate)):
        field, want = reference[key]
        got = candidate[key][1]
        stat = fields.setdefault(field, {'count':0,'word_misses':0,'max_ulp':0,'max_abs':0.0})
        stat['count'] += 1
        stat['word_misses'] += got != want
        stat['max_ulp'] = max(stat['max_ulp'],abs(_rank(got)-_rank(want)))
        stat['max_abs'] = max(stat['max_abs'],abs(_float(got)-_float(want)))
        if first is None and got != want:
            first = {'key':key,'reference':f'{want:08X}','candidate':f'{got:08X}'}
    return {'pass':not missing and not extra and first is None,
            'reference_count':len(reference),'candidate_count':len(candidate),
            'missing':missing[:20],'missing_count':len(missing),
            'extra':extra[:20],'extra_count':len(extra),
            'first_miss':first,'fields':fields}

def check_coverage(directory):
    directory=Path(directory)
    columns=0
    for n in (12,24,42,60):
        for arm in (0,1):
            stem=f'n{n}-a{arm}'
            rows=read_words(directory/f'{stem}-fields.csv')
            for field in SURFACE_FIELDS:
                keys={(int(k[0]),int(k[1])) for k,v in rows.items() if v[0]==field}
                if keys!={(c,0) for c in range(1,9)}:
                    raise ValueError(f'{stem}: incomplete surface field {field}')
            for field in LEVEL_FIELDS:
                keys={(int(k[0]),int(k[1])) for k,v in rows.items() if v[0]==field}
                if keys!={(c,k) for c in range(1,9) for k in range(1,n+1)}:
                    raise ValueError(f'{stem}: incomplete level field {field}')
            for c,field in [(2,'CFRACL'),(3,'CFRACM'),(4,'CFRACH')]:
                if _float(rows[(str(c),'0',field)][1])<=0:
                    raise ValueError(f'{stem}: missing intended cloud at column {c}')
            for field in ['SWDOWN','GSW','SWDOWNC']:
                if _float(rows[('7','0',field)][1]) != 0.:
                    raise ValueError(f'{stem}: night {field} not zero')
            if not (0 < _float(rows[('8','0','CZMEAN')][1]) < .2):
                raise ValueError(f'{stem}: low-sun case not exercised')
            tables=read_words(directory/f'{stem}-tables.csv')
            if {v[0] for v in tables.values()} != TABLE_FIELDS:
                raise ValueError(f'{stem}: incomplete initialization tables')
            for field in ['CO251','CDT51','C2D51','CO258','CDT58','C2D58']:
                if sum(v[0]==field for v in tables.values())!=(n+1)**2:
                    raise ValueError(f'{stem}: wrong {field} shape')
            columns+=8
    return {'pass':True,'columns':columns,'grids':[12,24,42,60],'co2tf':[0,1]}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        result=compare_words(read_words(args.reference),read_words(args.candidate))
        result['reference_sha256']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
        result['candidate_sha256']=hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    except (OSError,ValueError) as exc:
        result={'pass':False,'error':str(exc)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pass':result['pass'],'first_miss':result.get('first_miss'),'error':result.get('error')}))
    return 0 if result['pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
