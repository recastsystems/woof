"""Compare complete FLG word decks. A WRF repeat calibrates this check only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

CORE = {**dict.fromkeys(('SW_DOWN','SW_UP','LW_DOWN','LW_UP','NET_DOWN','NET_UP','DIRECT','DIFFUSE'),25),
        **dict.fromkeys(('SW_HEATING','LW_HEATING','HEATING'),24),
        **dict.fromkeys(('GLW','GSW','SWDOWN','OLR'),1)}
DRIVER = {**dict.fromkeys(('RTHRATEN','RTHRATENSW','RTHRATENLW'),24),
          **dict.fromkeys(('GLW','GSW','SWDOWN','RLWTOA'),1)}

def load_words(path):
    words={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['case','field','index','bits']:
            raise ValueError(f'{path}: incorrect header')
        for row in reader:
            key=(int(row['case']),row['field'],int(row['index']))
            if key in words:
                raise ValueError(f'{path}: duplicate {key}')
            if len(row['bits'])!=8:
                raise ValueError(f'{path}: incorrect float32 word {key}')
            value=int(row['bits'],16)
            if not 0<=value<2**32 or not math.isfinite(as_float(value)):
                raise ValueError(f'{path}: nonfinite or invalid word {key}')
            words[key]=value
    return words

def as_float(word):
    return struct.unpack('<f',struct.pack('<I',word))[0]

def ordered(word):
    return (~word & 0xffffffff) if word & 0x80000000 else word | 0x80000000

def validate_reference(root):
    with (root/'cases.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    expected=[(ice,scenario,aerosol,gas) for ice in range(8) for scenario in range(8)
              for aerosol in range(2) for gas in range(2)]
    actual=[tuple(int(r[k]) for k in ('ice','scenario','aerosol','new_gases')) for r in rows]
    if actual!=expected or [int(r['case']) for r in rows]!=list(range(1,257)):
        raise ValueError('reference must contain all 256 declared columns in order')
    for filename,fields,cases in [('columns.csv',CORE,range(1,257)),
                                 ('driver.csv',DRIVER,range(1,257,4))]:
        required={(c,field,k) for c in cases for field,count in fields.items() for k in range(1,count+1)}
        if set(load_words(root/filename))!=required:
            raise ValueError(f'{filename}: incomplete reference field or column coverage')

def compare(reference,candidate,*,tolerances=None):
    reference=Path(reference); candidate=Path(candidate)
    validate_reference(reference)
    result={'status':'pass','fields':{},'strict':tolerances is None,'files':{}}
    for name in ('cases.csv','inputs.csv'):
        a=(reference/name).read_bytes(); b=(candidate/name).read_bytes()
        if a!=b:
            raise ValueError(f'{name}: candidate does not use the reference inputs')
    for filename in ('columns.csv','driver.csv'):
        ref=load_words(reference/filename); got=load_words(candidate/filename)
        if set(ref)!=set(got):
            raise ValueError(f'{filename}: missing or unexpected candidate keys')
        result['files'][filename]={side:hashlib.sha256((root/filename).read_bytes()).hexdigest()
                                  for side,root in [('reference',reference),('candidate',candidate)]}
        for key,a in ref.items():
            b=got[key]; field=filename+':'+key[1]
            row=result['fields'].setdefault(field,{'count':0,'bit_misses':0,'max_ulp':0,'max_abs':0.0})
            row['count']+=1
            row['max_ulp']=max(row['max_ulp'],abs(ordered(a)-ordered(b)))
            delta=abs(as_float(a)-as_float(b)); row['max_abs']=max(row['max_abs'],delta)
            if a!=b:
                row['bit_misses']+=1
                row.setdefault('first_miss',{'case':key[0],'index':key[2],
                                            'reference':f'{a:08x}','candidate':f'{b:08x}'})
                limit=0 if tolerances is None else tolerances[field]
                if tolerances is None or delta>limit:
                    result['status']='fail'
    if tolerances is not None:
        if set(tolerances)!=set(result['fields']) or any(not math.isfinite(v) or v<0 for v in tolerances.values()):
            raise ValueError('default tolerances must name every field with finite nonnegative limits')
        result['tolerances']=tolerances
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--tolerances',type=Path)
    args=parser.parse_args()
    try:
        limits=None if args.tolerances is None else json.loads(args.tolerances.read_text())
        result=compare(args.reference,args.candidate,tolerances=limits)
    except (OSError,ValueError,KeyError) as error:
        result={'status':'error','error':str(error)}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'error':result.get('error')}))
    return 0 if result['status']=='pass' else 1

if __name__=='__main__':
    raise SystemExit(main())
