#!/usr/bin/env python3
"""Bounded IO around unchanged native accumulated-tendency arithmetic."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
ACC=('athmp','aqvmp','athcu','aqvcu','aucu','avcu','athsh','aqvsh','aush','avsh','athbl','aqvbl','aubl','avbl','athralw','athrasw')
RATES=('h_diabatic','qv_diabatic','rthcuten','rqvcuten','rucuten','rvcuten','rthshten','rqvshten','rushten','rvshten','rthblten','rqvblten','rublten','rvblten','rthratenlw','rthratensw')
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_misc.F');raw=(HERE/'upstream/module_diag_misc.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];s=raw.decode()
 decl='   INTEGER :: idump';start='   IF ( acc_phy_tend .gt. 0 ) THEN';end='! Handle accumulations with buckets to prevent round-off truncation in long runs'
 for marker in (decl,start,end):assert s.count(marker)==1
 s=s.replace(decl,decl+'''
   CHARACTER(LEN=4096) :: gpuwm_tend_dir,gpuwm_tend_file
   INTEGER :: gpuwm_tend_status,gpuwm_tend_unit,gpuwm_tend_i0,gpuwm_tend_i1
   INTEGER :: gpuwm_tend_j0,gpuwm_tend_j1,gpuwm_tend_k0,gpuwm_tend_k1
   LOGICAL :: gpuwm_tend_capture
''')
 cap='''   CALL GET_ENVIRONMENT_VARIABLE('WOOF_TEND_CAPTURE_DIR',gpuwm_tend_dir,STATUS=gpuwm_tend_status)
   gpuwm_tend_capture=gpuwm_tend_status==0 .AND. LEN_TRIM(gpuwm_tend_dir)>0
   gpuwm_tend_capture=gpuwm_tend_capture .AND. acc_phy_tend==1
   gpuwm_tend_capture=gpuwm_tend_capture .AND. (itimestep<=2 .OR. &
        MOD(NINT(curr_secs2),3600)==0 .OR. MOD(NINT(curr_secs2+dt),3600)==0)
   IF(gpuwm_tend_capture)THEN
      gpuwm_tend_i0=MAX(ips,MIN(ips+4,MIN(ipe,ide-1)-3))
      gpuwm_tend_j0=MAX(jps,MIN(jps+4,MIN(jpe,jde-1)-3))
      gpuwm_tend_i1=MIN(gpuwm_tend_i0+3,ipe,ide-1)
      gpuwm_tend_j1=MIN(gpuwm_tend_j0+3,jpe,jde-1)
      gpuwm_tend_k0=kps
      gpuwm_tend_k1=kpe
      WRITE(gpuwm_tend_file,'(A,"/tend-",I6.6,"-",I6.6,"-",I6.6,".bin")') &
           TRIM(gpuwm_tend_dir),ips,jps,itimestep
      OPEN(NEWUNIT=gpuwm_tend_unit,FILE=TRIM(gpuwm_tend_file),ACCESS='STREAM', &
           FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_tend_unit) 47109,gpuwm_tend_i1-gpuwm_tend_i0+1, &
           gpuwm_tend_j1-gpuwm_tend_j0+1,gpuwm_tend_k1-gpuwm_tend_k0+1, &
           itimestep,cu_used,shcu_used,history_interval
      WRITE(gpuwm_tend_unit) curr_secs2,dt
'''
 def writes(names):
  return ''.join(f'      WRITE(gpuwm_tend_unit) {name}(gpuwm_tend_i0:gpuwm_tend_i1, &\n           gpuwm_tend_k0:gpuwm_tend_k1,gpuwm_tend_j0:gpuwm_tend_j1)\n' for name in names)
 cap+=writes(RATES)+writes(ACC)+'   ENDIF\n'
 out='   IF(gpuwm_tend_capture)THEN\n'+writes(ACC)+'      CLOSE(gpuwm_tend_unit)\n   ENDIF\n'
 s=s.replace(start,cap+start).replace(end,out+end);target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='IO surrounding untouched native sixteen-rate accumulation and reset; bounded interior 4x4 columns, all levels, each MPI patch'),indent=2)+'\n')
if __name__=='__main__':main()
