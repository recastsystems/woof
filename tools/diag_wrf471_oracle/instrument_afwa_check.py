#!/usr/bin/env python3
"""Capture real held inputs and native check outcomes; no changed decisions."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_afwa.F')
 raw=(HERE/'upstream/module_diag_afwa.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 s=raw.decode();where='    ! Chirp the routine name for debugging purposes'
 decl='''    INTEGER, SAVE :: gpuwm_check_count=0
    INTEGER :: gpuwm_check_unit,gpuwm_check_env,gpuwm_check_ie,gpuwm_check_je
    INTEGER :: gpuwm_check_i,gpuwm_check_j,gpuwm_check_k,gpuwm_check_result
    LOGICAL :: gpuwm_check_capture
    CHARACTER(LEN=2048) :: gpuwm_check_dir,gpuwm_check_file

'''
 # The original fatal-check loop executes unchanged before this IO-only
 # block. A completed capture therefore witnesses that native accepted it.
 cap='''    CALL GET_ENVIRONMENT_VARIABLE('WOOF_AFWA_CHECK_CAPTURE_DIR',gpuwm_check_dir,STATUS=gpuwm_check_env)
    gpuwm_check_capture = gpuwm_check_env==0 .AND. LEN_TRIM(gpuwm_check_dir)>0
    gpuwm_check_capture = gpuwm_check_capture .AND. (gpuwm_check_count==0 .OR. &
        (is_output_timestep .AND. gpuwm_check_count<4))
    IF(gpuwm_check_capture)THEN
      gpuwm_check_count=gpuwm_check_count+1
      gpuwm_check_ie=MIN(its+3,ite,ide-1)
      gpuwm_check_je=MIN(jts+3,jte,jde-1)
      WRITE(gpuwm_check_file,'(A,"/afwa-check-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
        TRIM(gpuwm_check_dir),its,jts,gpuwm_check_count
      OPEN(NEWUNIT=gpuwm_check_unit,FILE=TRIM(gpuwm_check_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_check_unit)47108,gpuwm_check_ie-its+1,gpuwm_check_je-jts+1, &
        k_end-k_start+1,grid%itimestep,MERGE(1,0,is_output_timestep), &
        MERGE(1,0,is_first_timestep),config_flags%afwa_bad_data_check
      WRITE(gpuwm_check_unit)u_phy(its:gpuwm_check_ie,k_start:k_end,jts:gpuwm_check_je)
      WRITE(gpuwm_check_unit)v_phy(its:gpuwm_check_ie,k_start:k_end,jts:gpuwm_check_je)
      WRITE(gpuwm_check_unit)th_phy(its:gpuwm_check_ie,k_start:k_end,jts:gpuwm_check_je)
      gpuwm_check_result=0
      IF(config_flags%afwa_bad_data_check>0 .AND. is_output_timestep .AND. .NOT.is_first_timestep)THEN
        DO gpuwm_check_i=its,gpuwm_check_ie
          DO gpuwm_check_k=k_start,k_end
            DO gpuwm_check_j=jts,gpuwm_check_je
              IF(u_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)>300. .OR. &
                 u_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)<-300. .OR. &
                 v_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)>300. .OR. &
                 v_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)<-300. .OR. &
                 th_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)>9999. .OR. &
                 th_phy(gpuwm_check_i,gpuwm_check_k,gpuwm_check_j)<99.)gpuwm_check_result=1
            END DO
          END DO
        END DO
      END IF
      WRITE(gpuwm_check_unit)gpuwm_check_result
      CLOSE(gpuwm_check_unit)
    END IF

'''
 token='    ! 3-D arrays for moisture variables'
 assert s.count(where)==s.count(token)==1
 s=s.replace(where,decl+where).replace(token,cap+token)
 target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(source=str(target),source_pin=m['wrf'],original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),method='unchanged native fatal-check loop, then bounded held U/V/theta input and native predicate outcome IO; first plus three output calls, 4x4 full-depth columns'),indent=2)+'\n')
if __name__=='__main__':main()
