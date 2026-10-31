"""Audit the conservative arena bound against every generated local allocation.

The pinned driver enters with nx=ny=1, nor=norz=0. Callees may use a
three-level halo. Init bounds lhab by 9 and na by lqmx=30; nxtra=1.
The 128/500-element chunks are capped by the column's active nx*nz points.
No routine recurses, and NsslFrame restores its entry offset on return.
The largest root-to-leaf sum bounds every live stack, including all local
arrays in each frame and eight-byte cells for every interleaved element.
"""
import argparse
import ast
import collections
import json
import re
from pathlib import Path

from woof.hex.mpasphys.rrfs import mp_nssl2m as m


def degree(node):
    if isinstance(node, ast.Constant):return 0
    if isinstance(node, ast.Name):return int(node.id in {'nz','kte','ngs','ngscnt'})
    if isinstance(node, ast.UnaryOp):return degree(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):return degree(node.left)+degree(node.right)
        assert isinstance(node.op, (ast.Add,ast.Sub))
        return max(degree(node.left),degree(node.right))
    raise AssertionError(ast.dump(node))


def audit():
    assert m.SOURCE_SHA256 == '06bd80040c47a9613cb4a528f961d947404363b10043402040f8a1298e42edd4'
    lowered=m.cuda_source(shared_math_source='')
    assert lowered.count('const int ngs = nssl_min(128, nx*nz);') == 1
    assert lowered.count('const int ngs = nssl_min(500, nx*nz);') == 2
    alloc=[];calls=collections.defaultdict(set);frames=set();fn=None
    for line in m.MODULE_SOURCE.splitlines():
        if line.startswith('__device__'):
            match=re.search(r'\bf_\w+',line);fn=match[0] if match else None
            continue
        if fn:
            if 'NsslFrame nssl_frame(arena);' in line:frames.add(fn)
            calls[fn].update(re.findall(r'\b(f_\w+)\(',line))
        if fn and 'nssl_array<' in line and '(arena' in line:
            typ,lo,hi=re.search(r'nssl_array<(\w+),\s*\d+>\(arena, \{(.*?)\}, \{(.*?)\}\)',line).groups()
            lo,hi=lo.split(','),hi.split(',')
            assert sum(degree(ast.parse(f'({b})-({a})+1',mode='eval').body) for a,b in zip(lo,hi)) <= 1
            alloc.append((fn,typ,lo,hi))
    def visit(fn,path):
        assert fn not in path,('recursive arena frame',path,fn)
        for child in calls[fn]:visit(child,path+(fn,))
    for fn in tuple(calls):visit(fn,())
    assert {fn for fn,_,_,_ in alloc} <= frames
    def live(fn, sizes):
        return sizes[fn] + max((live(c,sizes) for c in calls[fn]),default=0)
    rows=[];all_sizes=[]
    for nz in (1,2,59,128,500,501,1000,10000):
        ctx=dict(nx=1,ny=1,nz=nz,its=1,ite=1,kts=1,kte=nz,na=30,nxtra=1,nproc=1,
                 nor=3,norz=3,ng1=1,ngs=min(500,nz),ngscnt=min(500,nz),ng=12,
                 ndiam=10,lc=3,lr=4,li=5,ls=6,lh=7,lv=2,lhab=9,lqmx=30)
        by=collections.Counter()
        for fn,typ,lo,hi in alloc:
            count=1
            for a,b in zip(lo,hi):
                count*=eval(b,{'__builtins__':{}},ctx)-eval(a,{'__builtins__':{}},ctx)+1
            by[fn]+=7+count*8
        total=max(live(fn,by) for fn in tuple(calls))
        stride=65536+10240*nz
        assert total < stride
        all_sizes.append(by)
        rows.append(dict(nz=nz,max_live_stack=total,stride=stride,by_routine=by))
    # Below the500-point cap each frame's bound is affine. Bound the
    # intercept and slope independently over every path. Above the cap,
    # the only remaining nz terms are nonnegative column/halo extents.
    slopes=collections.Counter({fn:all_sizes[1][fn]-all_sizes[0][fn] for fn in calls})
    intercepts=collections.Counter({fn:all_sizes[0][fn]-slopes[fn] for fn in calls})
    assert all(v>=0 for v in slopes.values())
    assert max(live(fn,slopes) for fn in tuple(calls)) <= 10240
    assert max(live(fn,intercepts) for fn in tuple(calls)) <= 65536
    return dict(declarations=len(alloc),acyclic_call_graph=True,rows=rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    result=audit()
    if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
