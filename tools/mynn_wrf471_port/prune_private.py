"""Remove stores to native local arrays with no active consumer.

The generated work arena escapes the kernel, so the CUDA compiler cannot
remove these dead local stores as native Fortran can. Native DMP's envm
carriers are read only in its literal FALSE branch. upqv and exneri have
no readers. This transformation keeps public output stores intact.
"""
from pathlib import Path
import re
from guard_private import guard_private

def prune_private(s):
    s,guards=guard_private(s)
    marker='__device__ M471_CALL /* Subroutine */ int dmp_mf__('
    start=s.index(marker,s.index(marker)+len(marker))
    end=s.index('} /* dmp_mf__ */',start)
    body=s[start:end]
    counts={}
    for name in ('envm_thl__','envm_sqv__','envm_sqc__','envm_u__','envm_v__','upqv','exneri'):
        pattern=r'(?m)^(\s*)('+re.escape(name)+r'\[[^;]*?\]\s*(?:\+=|=)[^;]+;)'
        body,n=re.subn(pattern,lambda m:m[1]+'if (FALSE_) { '+m[2]+' }',body)
        assert n==(2 if name.startswith('envm_') else 1),(name,n)
        counts[name]=n
    return s[:start]+body+s[end:],counts

if __name__=='__main__':
    import sys,json
    s,c=prune_private(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(s);print(json.dumps(c))
