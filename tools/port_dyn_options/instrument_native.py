"""Capture real native advection calls, without changing their arithmetic."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
path = root / 'dyn_em/module_advect_em.F'
source = path.read_text()
original = hashlib.sha256(source.encode()).hexdigest()
rows = [('u', 'u', 2), ('v', 'v', 3), ('w', 'w', 4),
        ('scalar', 'field', 1), ('scalar_pd', 'field', 5),
        ('weno_u', 'u', 7), ('weno_v', 'v', 8), ('weno_w', 'w', 9)]
for kind, field, mode in rows:
    start = source.index('SUBROUTINE advect_' + kind + ' ')
    end = source.index('END SUBROUTINE advect_' + kind, start)
    body = source[start:end]
    decl = '''   INTEGER, SAVE :: dyn_capture_count=0
   INTEGER :: dyn_unit,dyn_status,dyn_flags
   CHARACTER(LEN=1024) :: dyn_dir,dyn_path
   LOGICAL :: dyn_capture
'''
    body = body.replace('   LOGICAL :: degrade_xs', decl + '   LOGICAL :: degrade_xs', 1)
    assert 'dyn_capture_count' in body, kind
    at = body.index('    IF (config_flags%polar)' if mode == 5 else '   specified = .false.')
    flags = ['periodic_x', 'periodic_y', 'specified', 'nested', 'open_xs',
             'open_xe', 'open_ys', 'open_ye', 'symmetric_xs', 'symmetric_xe',
             'symmetric_ys', 'symmetric_ye', 'polar']
    setflags = '\n'.join(f'     IF(config_flags%{key}) dyn_flags=IBSET(dyn_flags,{i})'
                         for i, key in enumerate(flags))
    spacing = 'rdzu' if kind.endswith('w') else 'rdzw'
    dt = 'dt' if mode == 5 else '0.0'
    masses = 'mut,mub,mu_old' if mode == 5 else 'mut,mut,mut'
    tenddec = 'MERGE(1,0,tenddec)' if mode == 5 else '0'
    tap = f'''   dyn_capture_count=dyn_capture_count+1
   CALL get_environment_variable('WOOF_DYN_CAPTURE',dyn_dir,STATUS=dyn_status)
   dyn_capture=dyn_capture_count==4 .AND. dyn_status==0 .AND. LEN_TRIM(dyn_dir)>0
   IF (dyn_capture) THEN
     WRITE(dyn_path,'(A,A,I0,A,I0,A)') TRIM(dyn_dir),'/{kind}-',its,'-',jts,'.bin'
     OPEN(NEWUNIT=dyn_unit,FILE=TRIM(dyn_path),ACCESS='STREAM',FORM='UNFORMATTED', &
       STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
     dyn_flags=0
{setflags}
     WRITE(dyn_unit) 1,{mode},ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte,time_step,config_flags%h_mom_adv_order, &
       config_flags%v_mom_adv_order,config_flags%h_sca_adv_order, &
       config_flags%v_sca_adv_order,dyn_flags,{tenddec}
     WRITE(dyn_unit) rdx,rdy,{dt}
     WRITE(dyn_unit) {field},{field}_old,tendency,ru,rv,rom,rom
     WRITE(dyn_unit) {masses},msfux,msfuy,msfvx,msfvy,msftx,msfty
     WRITE(dyn_unit) c1,c2,fzm,fzp,{spacing},{spacing}
   END IF
'''
    body = body[:at] + tap + body[at:]
    outputs = ',h_tendency,z_tendency' if mode == 5 else ''
    body += f'''\n   IF (dyn_capture) THEN
     WRITE(dyn_unit) tendency{outputs}
     CLOSE(dyn_unit)
   END IF
'''
    source = source[:start] + body + source[end:]
path.write_text(source)
(root / 'dyn-capture-instrumentation.json').write_text(json.dumps({
    'native_source_sha256': original,
    'instrumented_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'call': 4, 'arithmetic_changes': False,
    'protocol': 'tools/advect_wrf471_oracle/run_advect.F90 header and inputs; appended native outputs'
}, indent=2) + '\n')
