"""Capture native stress output operators without changing arithmetic."""
from pathlib import Path
import hashlib
import re
import json
import sys

root = Path(sys.argv[1])
path = root / 'dyn_em/module_diffusion_em.F'
source = path.read_text()
original = hashlib.sha256(source.encode()).hexdigest()
for mode, name in enumerate(('11_22_33', '12_21', '13_31', '23_32')):
    begin = re.search(r'SUBROUTINE cal_titau_' + name + r'\s*\(', source).start()
    end = source.index('END SUBROUTINE cal_titau_' + name, begin)
    body = source[begin:end]
    decl = '''    INTEGER, SAVE :: stress_count=0
    INTEGER :: stress_unit,stress_status
    CHARACTER(LEN=1024) :: stress_dir,stress_path
    LOGICAL :: stress_capture
'''
    marker = '! End declarations.'
    pos = body.index(marker)
    body = body[:pos] + decl + body[pos:]
    marker = '    j_end   = j_end   + je_ext'
    pos = body.index(marker) + len(marker)
    weights = 'fnm,fnp' if mode in (2, 3) else 'rho(ims,kms:kme,jms),rho(ims,kms:kme,jms)'
    capture = f'''
    stress_count=stress_count+1
    CALL get_environment_variable('WOOF_STRESS_CAPTURE',stress_dir,STATUS=stress_status)
    stress_capture=stress_count<=4 .AND. config_flags%m_opt==1 .AND. &
      stress_status==0 .AND. LEN_TRIM(stress_dir)>0
    IF (stress_capture) THEN
      WRITE(stress_path,'(A,A,I0,A,I0,A,I0,A)') TRIM(stress_dir), &
        '/stress-{mode}-',stress_count,'-',its,'-',jts,'.bin'
      OPEN(NEWUNIT=stress_unit,FILE=TRIM(stress_path),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
      WRITE(stress_unit) 1,{mode},ims,ime,jms,jme,kms,kme, &
        i_start,i_end,j_start,j_end,kts,ktf
      WRITE(stress_unit) xkx,rho,defor,mtau
      WRITE(stress_unit) {weights}
    END IF
'''
    body = body[:pos] + capture + body[pos:]
    body += '''
    IF (stress_capture) THEN
      WRITE(stress_unit) mtau
      CLOSE(stress_unit)
    END IF
'''
    source = source[:begin] + body + source[end:]
path.write_text(source)
(root / 'stress-capture-instrumentation.json').write_text(json.dumps({
    'native_source_sha256': original,
    'instrumented_sha256': hashlib.sha256(source.encode()).hexdigest(),
    'calls': [1, 2, 3, 4], 'arithmetic_changes': False,
}, indent=2) + '\n')
