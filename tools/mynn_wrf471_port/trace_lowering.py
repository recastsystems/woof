#!/usr/bin/env python3
"""Debug the F77-to-C step using entry/exit snapshots of real calls."""
import argparse
import json
import re
from pathlib import Path
from lower import fixed,read_units,schema,split

p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('source',type=Path)
a=p.parse_args();units,_=read_units(a.source/'phys/module_bl_mynnedmf.F')
fields={u['name']:schema(u) for u in units}
src=(a.work/'model.f').read_text().splitlines()
# Unfold only this generator's fixed-format continuation convention.
statements=[]
for line in src:
    if len(line)>5 and line[5]=='&':statements[-1]+=line[6:]
    else:statements.append(line[6:] if line[:5].strip()=='' else line.strip())
out=[];name=None;active=False
for s in statements:
    m=re.match(r'(?:subroutine|real function) (\w+)\(',s)
    if m:name=m[1];active=False
    # Numerical body begins after the DATA and declarations. The generator
    # always emits all literals as DATA, followed by executable statements.
    is_decl=(s.startswith(('implicit ','integer ','real ','double precision ','logical ','parameter ','data ','subroutine ')) or re.match(r'real function ',s))
    if name in fields and not active and not is_decl:
        out+=['integer trace_count','save trace_count','data trace_count /0/',
              'character*160 trace_path','trace_count=trace_count+1']
        active=True
        out+=['write(trace_path,\'(a,i6.6,a)\') "trace/'+name+'-",trace_count,".entry"',
              'open(99,file=trace_path,access="stream",form="unformatted",convert="little_endian")']
        out+=['write(99) '+n for n,f in fields[name].items() if f['type']!='character']
        out+=['close(99)']
    if name in fields and active and s in ('return','end'):
        out+=['write(trace_path,\'(a,i6.6,a)\') "trace/'+name+'-",trace_count,".exit"',
              'open(99,file=trace_path,access="stream",form="unformatted",convert="little_endian")']
        out+=['write(99) '+n for n,f in fields[name].items() if f['type']!='character']
        out+=['close(99)']
    out.append(s)
    if s=='end':name=None
(a.work/'trace.f').write_text('\n'.join(fixed(s) for s in out)+'\n')

cpp=(a.work/'host.cpp').read_text()
header='''
#include <cstdio>
#include <vector>
#include <string>
#include <map>
struct Trace {
 std::vector<void*> ptr; std::vector<int> count; std::string path;
 Trace(const char*name,std::vector<void*>p,std::vector<int>n):ptr(p),count(n){
  static std::map<std::string,int> serial; char s[200];
  sprintf(s,"trace/%s-%06d",name,++serial[name]);path=s;write(".entry");}
 void write(const char*end){FILE*f=fopen((path+end).c_str(),"wb");
  for(int j=0;j<ptr.size();++j)fwrite(ptr[j],4,count[j],f);fclose(f);}
 ~Trace(){write(".exit");}
};
'''
for name,ff in fields.items():
    args=[n for n,f in ff.items() if f['type']!='character']
    pattern=r'((?:int|integer|real|doublereal)\s+'+name+r'_+\(([^{};]*)\)\s*\{)'
    match=re.search(pattern,cpp,re.S)
    if not match:raise ValueError(name)
    actual=[re.findall(r'\b\w+\b',arg)[-1] for arg in match[2].split(',')]
    names=dict(zip(args,actual));assert len(args)==len(actual),(name,args,actual)
    def dimcount(f):
        if not f['dims']:return '1'
        counts=[]
        for d in split(f['dims'] or ''):
            lo,hi=d.split(':') if ':' in d else ('1',d)
            expr='(('+hi+')-('+lo+')+1)'
            expr=re.sub(r'\b[a-z]\w*\b',lambda m:'(*'+names[m[0]]+')' if m[0] in names else m[0],expr)
            counts.append(expr)
        return '*'.join(counts) or '1'
    trace='Trace trace("'+name+'",{'+','.join(actual)+'},{'+','.join(dimcount(ff[n]) for n in args)+'});'
    cpp,n=re.subn(pattern,lambda m:m[1]+'\n'+trace,cpp,count=1,flags=re.S)
    if n!=1:raise ValueError(name)
(a.work/'trace.cpp').write_text(header+cpp)
(a.work/'trace-fields.json').write_text(json.dumps(fields,indent=2)+'\n')
