#!/usr/bin/env python3
"""IO-only capture surrounding the unchanged pinned native bucket block."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
NAMES=('rainc','rainnc',*('ac'+side+suffix for side in ('sw','lw') for suffix in ('upt','uptc','dnt','dntc','upb','upbc','dnb','dnbc')))
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_misc.F');raw=(HERE/'upstream/module_diag_misc.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];s=raw.decode()
 decl='   INTEGER :: idump';start='   IF(bucket_mm .gt. 0. .AND. MOD(NINT(CURR_SECS2),3600) .EQ. 0)THEN';end='! Compute precipitation accumulation in a given time window: prec_acc_dt'
 for marker in (decl,start,end):assert s.count(marker)==1
 s=s.replace(decl,decl+'''
   CHARACTER(LEN=4096) :: gpuwm_bucket_dir,gpuwm_bucket_file
   INTEGER :: gpuwm_bucket_status,gpuwm_bucket_unit,gpuwm_bucket_i,gpuwm_bucket_j
   INTEGER :: gpuwm_bucket_i0,gpuwm_bucket_i1,gpuwm_bucket_j0,gpuwm_bucket_j1
   REAL :: gpuwm_bucket_max
   LOGICAL :: gpuwm_bucket_capture
''')
 cap='''   CALL GET_ENVIRONMENT_VARIABLE('WOOF_BUCKET_CAPTURE_DIR',gpuwm_bucket_dir,STATUS=gpuwm_bucket_status)
   gpuwm_bucket_capture=gpuwm_bucket_status==0 .AND. LEN_TRIM(gpuwm_bucket_dir)>0
   gpuwm_bucket_capture=gpuwm_bucket_capture .AND. bucket_mm>0.
   gpuwm_bucket_capture=gpuwm_bucket_capture .AND. (itimestep==1 .OR. &
        MOD(NINT(curr_secs2),3600)==0 .OR. MOD(NINT(curr_secs2+dt),3600)==0)
   IF(gpuwm_bucket_capture)THEN
      gpuwm_bucket_i0=ips
      gpuwm_bucket_j0=jps
      gpuwm_bucket_max=-1.
      DO gpuwm_bucket_j=jps,MIN(jpe,jde-1)
      DO gpuwm_bucket_i=ips,MIN(ipe,ide-1)
         IF(rainnc(gpuwm_bucket_i,gpuwm_bucket_j)>gpuwm_bucket_max)THEN
            gpuwm_bucket_max=rainnc(gpuwm_bucket_i,gpuwm_bucket_j)
            gpuwm_bucket_i0=gpuwm_bucket_i
            gpuwm_bucket_j0=gpuwm_bucket_j
         ENDIF
      ENDDO
      ENDDO
      gpuwm_bucket_i0=MAX(ips,MIN(gpuwm_bucket_i0,MIN(ipe,ide-1)-3))
      gpuwm_bucket_j0=MAX(jps,MIN(gpuwm_bucket_j0,MIN(jpe,jde-1)-3))
      gpuwm_bucket_i1=MIN(gpuwm_bucket_i0+3,ipe,ide-1)
      gpuwm_bucket_j1=MIN(gpuwm_bucket_j0+3,jpe,jde-1)
      WRITE(gpuwm_bucket_file,'(A,"/bucket-",I6.6,"-",I6.6,"-",I6.6,".bin")') &
           TRIM(gpuwm_bucket_dir),ips,jps,itimestep
      OPEN(NEWUNIT=gpuwm_bucket_unit,FILE=TRIM(gpuwm_bucket_file),ACCESS='STREAM', &
           FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_bucket_unit) 47107,gpuwm_bucket_i1-gpuwm_bucket_i0+1,gpuwm_bucket_j1-gpuwm_bucket_j0+1,itimestep
      WRITE(gpuwm_bucket_unit) curr_secs2,xtime,bucket_mm,bucket_J
'''
 def writes():
  return ''.join(f'      WRITE(gpuwm_bucket_unit) {prefix}{name}(gpuwm_bucket_i0:gpuwm_bucket_i1,gpuwm_bucket_j0:gpuwm_bucket_j1)\n' for prefix in ('','i_') for name in NAMES)
 cap+=writes()+'   ENDIF\n';out='   IF(gpuwm_bucket_capture)THEN\n'+writes()+'      CLOSE(gpuwm_bucket_unit)\n   ENDIF\n'
 s=s.replace(start,cap+start).replace(end,out+end);target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='IO around unchanged native bucket arithmetic; bounded wettest-rain 4x4 neighborhood per MPI patch'),indent=2)+'\n')
if __name__=='__main__':main()
