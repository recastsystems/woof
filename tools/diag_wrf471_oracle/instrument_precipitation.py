#!/usr/bin/env python3
"""Bounded IO-only capture around the unchanged native precipitation block."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
FIELDS=('raincv','rainncv','snowncv','t2','prec_acc_c','prec_acc_nc','snow_acc_nc')
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((HERE/'sources.json').read_text());row=next(r for r in manifest['sources'] if r['path']=='phys/module_diag_misc.F')
 raw=(HERE/'upstream/module_diag_misc.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 s=raw.decode();decl='   INTEGER :: idump';marker='! Compute precipitation accumulation in a given time window: prec_acc_dt';end='   if (diag_print .eq. 0 ) return'
 for x in (decl,marker,end):assert s.count(x)==1,x
 s=s.replace(decl,decl+'''
   CHARACTER(LEN=4096) :: gpuwm_prec_dir,gpuwm_prec_file
   INTEGER :: gpuwm_prec_status,gpuwm_prec_unit,gpuwm_prec_i,gpuwm_prec_j
   INTEGER :: gpuwm_prec_i0,gpuwm_prec_i1,gpuwm_prec_j0,gpuwm_prec_j1
   REAL :: gpuwm_prec_max
   LOGICAL :: gpuwm_prec_capture
''')
 cap='''   CALL GET_ENVIRONMENT_VARIABLE('WOOF_PRECIP_CAPTURE_DIR',gpuwm_prec_dir,STATUS=gpuwm_prec_status)
   gpuwm_prec_capture=gpuwm_prec_status==0 .AND. LEN_TRIM(gpuwm_prec_dir)>0
   gpuwm_prec_capture=gpuwm_prec_capture .AND. prec_acc_dt>0.
   gpuwm_prec_capture=gpuwm_prec_capture .AND. (itimestep==1 .OR. &
        MOD(NINT(curr_secs2),3600)==0 .OR. MOD(NINT(curr_secs2+dt),3600)==0)
   IF(gpuwm_prec_capture)THEN
      gpuwm_prec_i0=ips
      gpuwm_prec_j0=jps
      gpuwm_prec_max=-1.
      DO gpuwm_prec_j=jps,MIN(jpe,jde-1)
      DO gpuwm_prec_i=ips,MIN(ipe,ide-1)
         IF(rainncv(gpuwm_prec_i,gpuwm_prec_j)>gpuwm_prec_max)THEN
            gpuwm_prec_max=rainncv(gpuwm_prec_i,gpuwm_prec_j)
            gpuwm_prec_i0=gpuwm_prec_i
            gpuwm_prec_j0=gpuwm_prec_j
         ENDIF
      ENDDO
      ENDDO
      gpuwm_prec_i0=MAX(ips,MIN(gpuwm_prec_i0,MIN(ipe,ide-1)-3))
      gpuwm_prec_j0=MAX(jps,MIN(gpuwm_prec_j0,MIN(jpe,jde-1)-3))
      gpuwm_prec_i1=MIN(gpuwm_prec_i0+3,ipe,ide-1)
      gpuwm_prec_j1=MIN(gpuwm_prec_j0+3,jpe,jde-1)
      WRITE(gpuwm_prec_file,'(A,"/precip-",I6.6,"-",I6.6,"-",I6.6,".bin")') &
           TRIM(gpuwm_prec_dir),ips,jps,itimestep
      OPEN(NEWUNIT=gpuwm_prec_unit,FILE=TRIM(gpuwm_prec_file),ACCESS='STREAM', &
           FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_prec_unit) 47106,gpuwm_prec_i1-gpuwm_prec_i0+1,gpuwm_prec_j1-gpuwm_prec_j0+1,itimestep
      WRITE(gpuwm_prec_unit) curr_secs2,prec_acc_dt
'''
 for name in FIELDS:cap+=f'      WRITE(gpuwm_prec_unit) {name}(gpuwm_prec_i0:gpuwm_prec_i1,gpuwm_prec_j0:gpuwm_prec_j1)\n'
 cap+='   ENDIF\n'
 out='   IF(gpuwm_prec_capture)THEN\n'
 for name in FIELDS[-3:]:out+=f'      WRITE(gpuwm_prec_unit) {name}(gpuwm_prec_i0:gpuwm_prec_i1,gpuwm_prec_j0:gpuwm_prec_j1)\n'
 out+='      CLOSE(gpuwm_prec_unit)\n   ENDIF\n'
 s=s.replace(marker,cap+marker).replace(end,out+end)
 target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=manifest['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='unchanged native arithmetic; at most 4x4 wettest-precipitation columns per MPI patch on first step and hourly boundary calls'),indent=2)+'\n')
if __name__=='__main__':main()
