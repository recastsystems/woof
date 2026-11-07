"""Keep each native ordered plume fold in registers at its own level."""
import re
from pathlib import Path

def fold_locals(s):
    marker='__device__ M471_CALL /* Subroutine */ int dmp_mf__('
    start=s.index(marker,s.index(marker)+len(marker))
    end=s.index('} /* dmp_mf__ */',start)
    body=s[start:end]
    pattern=re.compile(r'for \(ip = 1; ip <= 8; \+\+ip\) \{\s*i__1 = \*kte - 1;\s*#pragma unroll 1\s*for \(k = \*kts; k <= i__1; \+\+k\) \{')
    matches=list(pattern.finditer(body))
    counts=[]
    for m in reversed(matches):
        at=m.end();depth=1;pos=at
        while depth:
            if body[pos]=='{':depth+=1
            if body[pos]=='}':depth-=1
            pos+=1
        inner=body[at:pos-1]
        outer_end=body.index('}',pos)+1
        assert re.sub(r'/\*.*?\*/|\s+','',body[pos:outer_end],flags=re.S)=='}'
        # Every carrier has one native ordered += at this level. qv is
        # derived from the newly accumulated total and condensate in order.
        targets=list(dict.fromkeys(re.findall(r'\b((?:s_aw\w+|edmf_\w+)\[k(?: \+ 1)?\])\s*(?:\+=|=)',inner)))
        assert targets,inner
        before=[];after=[]
        for i,target in enumerate(targets):
            name='m471_fold'+str(i)
            before.append('real '+name+'='+target+';')
            after.append(target+'='+name+';')
            inner=inner.replace(target,name)
        replacement=('for(k=*kts;k<=*kte-1;++k){\n'+'\n'.join(before)+
                     '\n#pragma unroll 1\nfor(ip=1;ip<=8;++ip){\n'+inner+
                     '\n}\n'+'\n'.join(after)+'\n}')
        body=body[:m.start()]+replacement+body[outer_end:]
        counts.append(len(targets))
    assert len(matches)==10,(len(matches),counts)
    return s[:start]+body+s[end:]

if __name__=='__main__':
    import sys
    Path(sys.argv[2]).write_text(fold_locals(Path(sys.argv[1]).read_text()))
