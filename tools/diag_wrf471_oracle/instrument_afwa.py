#!/usr/bin/env python3
"""Bounded real native AFWA inputs/results; IO only, unchanged arithmetic."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
DECL='''    ! Bounded input/result capture; native expressions remain unchanged.
    INTEGER, SAVE :: gpuwm_afwa_count = 0
    INTEGER :: gpuwm_afwa_unit, gpuwm_afwa_status, gpuwm_afwa_ie, gpuwm_afwa_je
    LOGICAL :: gpuwm_afwa_capture
    CHARACTER(LEN=2048) :: gpuwm_afwa_directory, gpuwm_afwa_filename

'''
CAP='''    CALL GET_ENVIRONMENT_VARIABLE('WOOF_AFWA_CAPTURE_DIR',gpuwm_afwa_directory,STATUS=gpuwm_afwa_status)
    gpuwm_afwa_capture = gpuwm_afwa_status == 0 .AND. LEN_TRIM(gpuwm_afwa_directory) > 0
    gpuwm_afwa_capture = gpuwm_afwa_capture .AND. (gpuwm_afwa_count == 0 .OR. &
         (is_output_timestep .AND. gpuwm_afwa_count < 4))
    IF (gpuwm_afwa_capture) THEN
      gpuwm_afwa_count = gpuwm_afwa_count + 1
      gpuwm_afwa_ie = MIN(its+3,ite,ide-1)
      gpuwm_afwa_je = MIN(jts+3,jte,jde-1)
      WRITE(gpuwm_afwa_filename,'(A,"/afwa-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
           TRIM(gpuwm_afwa_directory),its,jts,gpuwm_afwa_count
      OPEN(NEWUNIT=gpuwm_afwa_unit,FILE=TRIM(gpuwm_afwa_filename),ACCESS='STREAM', &
           FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_afwa_unit) 47104,gpuwm_afwa_ie-its+1,gpuwm_afwa_je-jts+1,kme-kms+1,k_end-k_start+1
      WRITE(gpuwm_afwa_unit) grid%itimestep,MERGE(1,0,is_output_timestep),MERGE(1,0,is_first_timestep)
'''
for n in ('qvapr','qcloud','dz8w','rho','grid%t_phy','grid%z'):
 CAP+=f'      WRITE(gpuwm_afwa_unit) {n}(its:gpuwm_afwa_ie,kms:kme,jts:gpuwm_afwa_je)\n'
for n in ('grid%ht','grid%psfc','grid%t2','grid%q2','grid%u10','grid%v10','grid%raincv','grid%rainncv','grid%rainc','grid%rainnc'):
 CAP+=f'      WRITE(gpuwm_afwa_unit) {n}(its:gpuwm_afwa_ie,jts:gpuwm_afwa_je)\n'
CAP+='    END IF\n\n'
OUT='    IF (gpuwm_afwa_capture) THEN\n'
for n in ('afwa_precip','afwa_totprecip','afwa_pwat','afwa_mslp','afwa_heatidx','afwa_wchill','afwa_fits'):
 OUT+=f'      WRITE(gpuwm_afwa_unit) grid%{n}(its:gpuwm_afwa_ie,jts:gpuwm_afwa_je)\n'
OUT+='      CLOSE(gpuwm_afwa_unit)\n    END IF\n\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text()); row=next(x for x in m['sources'] if x['path']=='phys/module_diag_afwa.F')
 target=a.checkout/row['path'];raw=target.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'],'outside native source pin'
 s=raw.decode();decl='    ! Chirp the routine name for debugging purposes';cap='    ! Time-step precipitation (convective + nonconvective)';end='  END SUBROUTINE afwa_diagnostics_driver'
 for x in (decl,cap,end):assert s.count(x)==1,x
 s=s.replace(decl,DECL+decl).replace(cap,CAP+cap).replace(end,OUT+end);target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='input/result IO only; first call and up to three native output calls; 4x4 columns, full native memory depth'),indent=2)+'\n')
if __name__=='__main__':main()
