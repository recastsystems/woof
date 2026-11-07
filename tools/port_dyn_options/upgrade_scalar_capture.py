"""Repair the native tap so scalar proof uses a nonzero real field."""
from pathlib import Path
import sys,json,hashlib
root=Path(sys.argv[1]);p=root/'dyn_em/module_advect_em.F';source=p.read_text();oldhash=hashlib.sha256(source.encode()).hexdigest()
for name in ('scalar','scalar_pd'):
 start=source.index('SUBROUTINE advect_'+name+' ');end=source.index('END SUBROUTINE advect_'+name,start);body=source[start:end]
 body=body.replace('   LOGICAL :: dyn_capture','   LOGICAL, SAVE :: dyn_captured_nonzero=.false.\n   LOGICAL :: dyn_capture',1)
 body=body.replace('   dyn_capture=dyn_capture_count==4 .AND. dyn_status==0 .AND. LEN_TRIM(dyn_dir)>0', '''   dyn_capture=.NOT.dyn_captured_nonzero .AND. dyn_capture_count>=4 &
     .AND. dyn_status==0 .AND. LEN_TRIM(dyn_dir)>0 &
     .AND. MAXVAL(ABS(field(its:ite,kts:MIN(kte,kde-1),jts:jte)))>0.0
   IF(dyn_capture) dyn_captured_nonzero=.true.''',1)
 assert 'IF(dyn_capture) dyn_captured_nonzero=.true.' in body
 source=source[:start]+body+source[end:]
p.write_text(source);(root/'dyn-nonzero-instrumentation.json').write_text(json.dumps(dict(before_sha256=oldhash,after_sha256=hashlib.sha256(source.encode()).hexdigest(),scalar_call='First nonzero tile after call 3',arithmetic_changes=False),indent=2)+'\n')
