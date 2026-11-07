"""Skip private plume carriers whose only native consumer is disabled."""
from pathlib import Path
import re

def guard_private(s):
    marker='__device__ M471_CALL /* Subroutine */ int dmp_mf__('
    start=s.index(marker,s.index(marker)+len(marker))
    end=s.index('} /* dmp_mf__ */',start)
    body=s[start:end]
    options={'upqke':'tke_opt__', **{k:'scalar_opt__' for k in ('upqnc','upqni','upqnwfa','upqnifa','upqnbca')}}
    options.update(qken='tke_opt__',**{k:'scalar_opt__' for k in ('qncn','qnin','qnwfan','qnifan','qnbcan')})
    counts={}
    for name,option in options.items():
        lhs=re.escape(name)+(r'\[[^;]*?\]' if name.startswith('up') else '')
        pattern=r'(?m)^(\s*)('+lhs+r'\s*=[^;]+;)'
        body,n=re.subn(pattern,lambda m:m[1]+'if (*'+option+' > 0) { '+m[2]+' }',body)
        assert n==(3 if name.startswith('up') else 1),(name,n)
        counts[name]=n
    return s[:start]+body+s[end:],counts

if __name__=='__main__':
    import sys,json
    s,c=guard_private(Path(sys.argv[1]).read_text());Path(sys.argv[2]).write_text(s);print(json.dumps(c))
