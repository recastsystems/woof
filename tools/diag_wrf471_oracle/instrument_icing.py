#!/usr/bin/env python3
"""Bounded IO inside unchanged native AFWA icing; no arithmetic edits."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
VOLUMES=('t_phy','z','dz8w','rho','qrain','qcloud','ncloud')
PLANES=('fzlev','icing_lg','icing_sm','qicing_lg_max','qicing_sm_max','icingtop','icingbot')
RAW=('qicing_lg','qicing_sm')
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_afwa.F');raw=(HERE/'upstream/module_diag_afwa.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 s=raw.decode();start=s.index('  SUBROUTINE icing_diagnostics (');end=s.index('  END SUBROUTINE icing_diagnostics',start);body=s[start:end]
 decl='''    INTEGER,SAVE :: gpuwm_icing_count=0
    INTEGER :: gpuwm_icing_unit,gpuwm_icing_status,gpuwm_icing_i,gpuwm_icing_j
    INTEGER :: gpuwm_icing_ib,gpuwm_icing_jb,gpuwm_icing_ie,gpuwm_icing_je,gpuwm_icing_k
    REAL :: gpuwm_icing_best,gpuwm_icing_sum
    LOGICAL :: gpuwm_icing_capture
    CHARACTER(LEN=2048) :: gpuwm_icing_dir,gpuwm_icing_path
'''
 cap='''    CALL GET_ENVIRONMENT_VARIABLE('WOOF_ICING_CAPTURE_DIR',gpuwm_icing_dir,STATUS=gpuwm_icing_status)
    gpuwm_icing_capture=gpuwm_icing_status==0.AND.LEN_TRIM(gpuwm_icing_dir)>0.AND.gpuwm_icing_count<3
    IF(gpuwm_icing_capture)THEN
      gpuwm_icing_count=gpuwm_icing_count+1
      gpuwm_icing_best=-1.
      gpuwm_icing_ib=ips
      gpuwm_icing_jb=jps
      DO gpuwm_icing_j=jps,MIN(jpe,jde-1)
       DO gpuwm_icing_i=ips,MIN(ipe,ide-1)
        gpuwm_icing_sum=0.
        DO gpuwm_icing_k=kps,MIN(kpe,kde-1)
         IF(t_phy(gpuwm_icing_i,gpuwm_icing_k,gpuwm_icing_j)<273.15) &
          gpuwm_icing_sum=gpuwm_icing_sum+MAX(qcloud(gpuwm_icing_i,gpuwm_icing_k,gpuwm_icing_j),0.)
        END DO
        IF(gpuwm_icing_sum>gpuwm_icing_best)THEN
         gpuwm_icing_best=gpuwm_icing_sum
         gpuwm_icing_ib=gpuwm_icing_i
         gpuwm_icing_jb=gpuwm_icing_j
        END IF
       END DO
      END DO
      gpuwm_icing_ib=MAX(ips,MIN(gpuwm_icing_ib,MIN(ipe,ide-1)-3))
      gpuwm_icing_jb=MAX(jps,MIN(gpuwm_icing_jb,MIN(jpe,jde-1)-3))
      gpuwm_icing_ie=MIN(gpuwm_icing_ib+3,ipe,ide-1)
      gpuwm_icing_je=MIN(gpuwm_icing_jb+3,jpe,jde-1)
      WRITE(gpuwm_icing_path,'(A,"/icing-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
       TRIM(gpuwm_icing_dir),ips,jps,gpuwm_icing_count
      OPEN(NEWUNIT=gpuwm_icing_unit,FILE=TRIM(gpuwm_icing_path),ACCESS='STREAM', &
       FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_icing_unit) 47110,gpuwm_icing_ie-gpuwm_icing_ib+1, &
       gpuwm_icing_je-gpuwm_icing_jb+1,kpe-kps+1,MIN(kpe,kde-1)-kps+1,icing_opt,gpuwm_icing_count,kpe
'''
 for name in VOLUMES:cap+=f'      WRITE(gpuwm_icing_unit) {name}(gpuwm_icing_ib:gpuwm_icing_ie,kps:kpe,gpuwm_icing_jb:gpuwm_icing_je)\n'
 cap+='    END IF\n'
 out='    IF(gpuwm_icing_capture)THEN\n'
 for name in PLANES:out+=f'      WRITE(gpuwm_icing_unit) {name}(gpuwm_icing_ib:gpuwm_icing_ie,gpuwm_icing_jb:gpuwm_icing_je)\n'
 for name in RAW:out+=f'      WRITE(gpuwm_icing_unit) {name}(gpuwm_icing_ib:gpuwm_icing_ie,kps:kpe,gpuwm_icing_jb:gpuwm_icing_je)\n'
 out+='      CLOSE(gpuwm_icing_unit)\n    END IF\n'
 token='    ! Initializations';assert body.count(token)==1
 body=body.replace(token,decl+'\n'+cap+'\n'+token);s=s[:start]+body+out+s[end:];target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='IO only; first three actual icing calls, actual strongest supercooled cloud neighborhood, 4x4 columns, full native patch depth'),indent=2)+'\n')
if __name__=='__main__':main()
