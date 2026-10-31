import sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from woof.hex.mpasphys.rrfs import mp_nssl2m as m
from woof.hex.mpasphys.rrfs.nssl_phases import _block
source=m.MODULE_SOURCE;start=source.index('struct NsslState {');o,e=_block(source,start)
fields={}
for line in source[o+1:e].splitlines():
 match=re.match(r'^(bool|int|float|double) (\w+)(?:\s*=.*|\{\})?;',line)
 if match:fields[match[2]]=match[1]
funcs={}
for match in re.finditer(r'^__device__[^\n]+?\b(f_\w+)\(',source,re.M):
 start=match.start();o,e=_block(source,start);sig=source[start:o];body=source[o+1:e]
 local=set(re.findall(r'\b(?:bool|int|float|double|NsslText) (\w+)',sig))|set(re.findall(r'^\s*(?:bool|int|float|double|NsslText) (\w+)\b',body,re.M))
 local|=set(re.findall(r'Nssl(?:Array|Optional)<[^>]+> (\w+)',sig+'\n'+body))
 writes=[]
 for a in re.finditer(r'(?<![.\w])([a-z]\w*)\s*(?:=(?!=)|\+=|-=|\*=|/=|\+\+|--)',body):
  name=a[1]
  if name in fields and name not in local:writes.append(dict(name=name,line=source[:o+1+a.start()].count('\n')+1,expression=body[a.start():body.index('\n',a.start())]))
 funcs[match[1]]=dict(writes=writes,calls=sorted(set(re.findall(r'\b(f_\w+)\(',body))))
seen=set()
def visit(name):
 if name in seen:return
 seen.add(name)
 for child in funcs.get(name,{}).get('calls',[]):visit(child)
visit('f_nssl_2mom_driver')
rows={name:funcs[name]['writes'] for name in sorted(seen) if name in funcs and funcs[name]['writes']}
names=sorted({r['name'] for rows_ in rows.values() for r in rows_})
out=dict(scalar_fields=len(fields),reachable_functions=sorted(seen),mutated_scalars=names,writes_by_function=rows)
from woof.hex.mpasphys.rrfs.nssl_writeback import MUTATED
assert tuple(names)==MUTATED
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(scalar_fields=len(fields),mutated_scalars=names,writes_by_function={n:len(v) for n,v in rows.items()}),indent=2))
