"""Bound both storage banks over all native call stacks and column heights."""
import collections,json,re,sys
from pathlib import Path
from woof.hex.mpasphys.rrfs import mp_nssl2m as m
calls=collections.defaultdict(set);alloc=[];fn=None
for line in m.MODULE_SOURCE.splitlines():
    if line.startswith('__device__'):
        match=re.search(r'\bf_\w+',line);fn=match[0] if match else None
        continue
    if fn:calls[fn].update(re.findall(r'\b(f_\w+)\(',line))
    if fn and 'nssl_array<' in line and '(arena' in line:
        typ,lo,hi=re.search(r'nssl_array<(\w+),\s*\d+>\(arena, \{(.*?)\}, \{(.*?)\}\)',line).groups()
        alloc.append((fn,typ,lo.split(','),hi.split(',')))
def live(fn,sizes,path=()):
    assert fn not in path
    return sizes[fn]+max((live(c,sizes,path+(fn,)) for c in calls[fn]),default=0)
roots=tuple(calls);rows=[]
for collection in (False,True):
    bounds=[]
    for nz in (1,2,59,128,500,501,10000):
        ctx=dict(nx=1,ny=1,nz=nz,its=1,ite=1,kts=1,kte=nz,na=30,nxtra=1,nproc=1,nor=3,norz=3,ng1=1,ngs=1 if collection else min(500,nz),ngscnt=1 if collection else min(500,nz),ng=12,ndiam=10,lc=3,lr=4,li=5,ls=6,lh=7,lv=2,lhab=9,lqmx=30)
        sizes=[collections.Counter(),collections.Counter()]
        for fn,typ,lo,hi in alloc:
            count=1
            for a,b in zip(lo,hi):count*=eval(b,{'__builtins__':{}},ctx)-eval(a,{'__builtins__':{}},ctx)+1
            bank=int(typ=='double');sizes[bank][fn]+=7+count*(8 if bank else 4)
        selected=('f_nssl_2mom_gs',) if collection else roots
        peak=[max(live(fn,b) for fn in selected) for b in sizes]
        stride=65536+(16 if collection else 10240)*nz
        capacity=[stride*3//4,stride-stride*3//4]
        assert all(x<y for x,y in zip(peak,capacity)),(collection,nz,peak,capacity)
        bounds.append(sizes);rows.append(dict(collection=collection,nz=nz,peak=peak,capacity=capacity))
    # Each extent is affine before the 500-point cap; above it the cap only
    # removes nonnegative growth. Independently bound intercept and slope.
    for bank in (0,1):
        slope=collections.Counter({fn:bounds[1][bank][fn]-bounds[0][bank][fn] for fn in roots})
        intercept=collections.Counter({fn:bounds[0][bank][fn]-slope[fn] for fn in roots})
        assert all(v>=0 for v in slope.values())
        ratio=.25 if bank else .75
        assert max(live(fn,slope) for fn in selected)<=(16 if collection else 10240)*ratio
        assert max(live(fn,intercept) for fn in selected)<=65536*ratio
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
