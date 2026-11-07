"""Diagnostic/validation packer for stock WRF column receipts."""
import csv,json,ctypes,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[2]
LAYOUT=json.loads((ROOT/'woof/core/gfdleta_column_layout.json').read_text())

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def pack(build,stem,n):
    build=Path(build)
    fields={r['name'].upper():r for r in LAYOUT['fields']}
    f=np.zeros((8,LAYOUT['float_count']),np.float32);ints=np.zeros((8,LAYOUT['int_count']),np.int32)
    for r in read(build/(stem+'-input.csv')):
        desc=fields[r['field']];c=int(r['column'])-1;idx=max(0,int(r['index'])-1)
        bits=int(r['bits'],16)
        if desc['kind']=='real':f.view(np.uint32)[c,desc['offset']+idx]=bits
        else:ints[c,desc['offset']+idx]=np.asarray(bits,np.uint32).view(np.int32)
    specs=LAYOUT['tables'];values={}
    for r in read(build/(stem+'-context.csv')):values.setdefault(r['field'],[]).append(int(r['bits'],16))
    arrays=[];offsets={};start=0
    for name,(kind,shape) in specs.items():
        dims=[int(x) for x in shape.split(',')] if shape else [1]
        count=math.prod(dims);a=np.zeros(count,np.uint32)
        src=np.array(values[name],np.uint32)
        if kind=='INTEGER':src=src.view(np.float32).astype(np.int32).view(np.uint32)
        if len(dims)==2 and dims==[82,82]:
            for j in range(n+1):a[j*82:j*82+n+1]=src[j*(n+1):(j+1)*(n+1)]
        else:a[:len(src)]=src
        arrays.append(a);offsets[name]=start;start+=count
    table=np.concatenate(arrays)
    return f,ints,table,offsets

OUTPUTS={'RTHRATEN':'THRATEN','RTHRATENLW':'THRATENLW','RTHRATENSW':'THRATENSW','CLDFRA':'CLDFRA','GLW':'GLW','GSW':'GSW','SWDOWN':'RSWIN','SWDOWNC':'RSWINC','RLWTOA':'RLWTOA','RSWTOA':'RSWTOA','CFRACL':'CFRACL','CFRACM':'CFRACM','CFRACH':'CFRACH','CZMEAN':'CZMEAN','ACFRST':'ACFRST','ACFRCV':'ACFRCV','HTOPR':'HTOPR','HBOTR':'HBOTR','CUPPT':'CUPPT'}

def compare(build,stem,f):
    fields={r['name'].upper():r for r in LAYOUT['fields']};stats={}
    for row in read(Path(build)/(stem+'-fields.csv')):
        field=row['field'];d=fields[OUTPUTS[field]];c=int(row['column'])-1;k=max(0,int(row['level'])-1)
        got=int(f.view(np.uint32)[c,d['offset']+k]);want=int(row['bits'],16)
        stat=stats.setdefault(field,{'count':0,'misses':0,'first':None})
        stat['count']+=1
        if got!=want:
            stat['misses']+=1
            if stat['first'] is None:stat['first']={'column':c+1,'level':int(row['level']),'got':f'{got:08X}','want':f'{want:08X}'}
    return {'pass':all(v['misses']==0 for v in stats.values()),'fields':stats}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('stem');p.add_argument('n',type=int);p.add_argument('output',type=Path);a=p.parse_args()
    f,ints,table,_=pack(a.build,a.stem,a.n)
    lib=ctypes.CDLL(str(a.build.parent/'build/host-column.so'))
    fn=lib.gd_host_columns;fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3
    fn(table.ctypes.data,f.ctypes.data,ints.ctypes.data,8,LAYOUT['float_count'],LAYOUT['int_count'])
    result=compare(a.build,a.stem,f);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
