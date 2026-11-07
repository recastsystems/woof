"""Add sparse stream captures around ACMPBL without changing its arithmetic."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('wrf_source', type=Path)
parser.add_argument('--patch', type=Path, required=True)
args = parser.parse_args()
path = args.wrf_source / 'phys/module_bl_acm.F'
before = path.read_text()
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'f18685f7ea091f567c3f37e983f26357dcd7e9ec2eea9a6e70635c506a66d60c'
decl = '''
! Lane evidence only. Sparse real native input/output capture.
      INTEGER :: lane_unit, lane_i, lane_j, lane_ncol
      LOGICAL :: lane_capture
      CHARACTER(LEN=256) :: lane_file
'''

def columns(expression):
    return f'''      DO lane_j = jts,jte
        IF (MOD(lane_j-1,17) /= 0) CYCLE
        DO lane_i = its,ite
          IF (MOD(lane_i-1,17) /= 0) CYCLE
          WRITE(lane_unit) {expression}
        ENDDO
      ENDDO
'''

def fields(names, dimensions):
    indices = '(lane_i,kts:kte,lane_j)' if dimensions == 3 else '(lane_i,lane_j)'
    return ''.join(columns(name + indices) for name in names)

def header(phase):
    return f'''      WRITE(lane_file,'("acm2-",i0,"-",i0,"-",i0,"-{phase}.bin")') XTIME,its,jts
      OPEN(NEWUNIT=lane_unit,FILE=TRIM(lane_file),STATUS='NEW', &
           ACCESS='STREAM',FORM='UNFORMATTED',ACTION='WRITE')
      WRITE(lane_unit) 1094929714,lane_ncol,kte-kts+1,XTIME,its,ite,jts,jte,kts,kte
      WRITE(lane_unit) DTPBL,EP1,G,ROVCP,RD,CPD
'''

begin = '''      lane_capture = (XTIME == 1 .OR. XTIME == 301 .OR. XTIME == 601)
      lane_ncol = 0
      DO lane_j = jts,jte
        IF (MOD(lane_j-1,17) /= 0) CYCLE
        DO lane_i = its,ite
          IF (MOD(lane_i-1,17) /= 0) CYCLE
          lane_ncol = lane_ncol + 1
        ENDDO
      ENDDO
      lane_capture = lane_capture .AND. lane_ncol > 0
      IF (lane_capture) THEN
'''
begin += header('in') + columns('lane_i,lane_j')
begin += fields(['U3D','V3D','PP3D','DZ8W','TH3D','T3D','QV3D','QC3D','QI3D','RR3D'], 3)
begin += fields(['UST','HFX','QFX','TSK','PSFC','WSPD','PSIM','GZ1OZ0','MUT','PBLH','REGIME','RMOL'], 2)
begin += fields(['EXCH_H','EXCH_M','RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN'], 3)
begin += '''      CLOSE(lane_unit)
      ENDIF
'''
end = '      IF (lane_capture) THEN\n' + header('out')
end += fields(['RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN','EXCH_H','EXCH_M'], 3)
end += fields(['PBLH','REGIME','RMOL','KPBL2D','UST'], 2)
end += '''      CLOSE(lane_unit)
      ENDIF
'''
after = before.replace('   RDT = 1.0 / DTPBL', decl + '\n' + begin + '\n   RDT = 1.0 / DTPBL', 1)
after = after.replace('   END SUBROUTINE ACMPBL', end + '\n   END SUBROUTINE ACMPBL', 1)
assert after != before
path.write_text(after)
diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
    fromfile='phys/module_bl_acm.F', tofile='phys/module_bl_acm.F'))
args.patch.write_text(diff)
print(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'captured_steps': [1,301,601], 'sampling_stride': 17,
                  'diff_bytes': len(diff.encode())}))
