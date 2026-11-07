#!/usr/bin/env python3
"""Bounded real-entry IO around unchanged native AFWA buoyancy calls."""
from pathlib import Path
import argparse,json,hashlib
HERE=Path(__file__).resolve().parent
FIELDS=('afwa_cape','afwa_cin','afwa_zlfc','afwa_plfc','afwa_lidx','afwa_cape_mu','afwa_cin_mu')
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_afwa.F');raw=(HERE/'upstream/module_diag_afwa.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];s=raw.decode()
 decl='''    INTEGER,SAVE :: gpuwm_buoy_count=0
    INTEGER :: gpuwm_buoy_unit,gpuwm_buoy_status,gpuwm_buoy_ie,gpuwm_buoy_je
    LOGICAL :: gpuwm_buoy_capture
    CHARACTER(LEN=2048) :: gpuwm_buoy_dir,gpuwm_buoy_path
'''
 token='    ! Chirp the routine name for debugging purposes';assert s.count(token)==1;s=s.replace(token,decl+'\n'+token)
 token='      IF ( config_flags % afwa_buoy_opt == 1 ) THEN';assert s.count(token)==1
 cap='''
          CALL GET_ENVIRONMENT_VARIABLE('WOOF_BUOY_CAPTURE_DIR',gpuwm_buoy_dir,STATUS=gpuwm_buoy_status)
          gpuwm_buoy_capture=gpuwm_buoy_status==0.AND.LEN_TRIM(gpuwm_buoy_dir)>0.AND.gpuwm_buoy_count<3
          IF(gpuwm_buoy_capture)THEN
            gpuwm_buoy_count=gpuwm_buoy_count+1
            gpuwm_buoy_ie=MIN(ips+3,ipe,ide-1)
            gpuwm_buoy_je=MIN(jps+3,jpe,jde-1)
            WRITE(gpuwm_buoy_path,'(A,"/buoy-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
             TRIM(gpuwm_buoy_dir),ips,jps,gpuwm_buoy_count
            OPEN(NEWUNIT=gpuwm_buoy_unit,FILE=TRIM(gpuwm_buoy_path),ACCESS='STREAM', &
             FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
            WRITE(gpuwm_buoy_unit) 47112,gpuwm_buoy_ie-ips+1,gpuwm_buoy_je-jps+1,k_end-k_start+1
'''
 for name in ('grid%t_phy','ptot','qvapr','grid%z','rh'):cap+=f'            WRITE(gpuwm_buoy_unit) {name}(ips:gpuwm_buoy_ie,kms:kms+k_end-k_start,jps:gpuwm_buoy_je)\n'
 cap+='            WRITE(gpuwm_buoy_unit) grid%ht(ips:gpuwm_buoy_ie,jps:gpuwm_buoy_je)\n          END IF\n'
 s=s.replace(token,token+cap)
 token='      ENDIF  ! afwa_buoy_opt == 1';assert s.count(token)==1
 out='          IF(gpuwm_buoy_capture)THEN\n'
 for name in FIELDS:out+=f'            WRITE(gpuwm_buoy_unit) grid%{name}(ips:gpuwm_buoy_ie,jps:gpuwm_buoy_je)\n'
 out+='            CLOSE(gpuwm_buoy_unit)\n          END IF\n';s=s.replace(token,out+token)
 target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='IO only; first three output calls, 4x4 actual interior columns, actual k_end-k_start+1 depth, actual RH and all seven driver outputs'),indent=2)+'\n')
if __name__=='__main__':main()
