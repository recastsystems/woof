#!/usr/bin/env python3
"""IO-only extension: real AFWA ptype, radar, VIL and visibility captures."""
from pathlib import Path
import argparse,hashlib,json
from instrument_afwa import DECL,HERE
VOLUMES=('qvapr','qcloud','dz8w','rho','grid%t_phy','grid%z','qrain','qsnow','qgrpl','u_phy','v_phy','ptot')
PLANES=('grid%ht','grid%psfc','grid%t2','grid%q2','grid%u10','grid%v10','grid%raincv','grid%rainncv','grid%rainc','grid%rainnc','grid%swdown')
PTYPE=('afwa_rain','afwa_snow','afwa_ice','afwa_fzra','afwa_snowfall')
OUTPUTS=('afwa_precip','afwa_totprecip','afwa_pwat','afwa_mslp','afwa_heatidx','afwa_wchill','afwa_fits',*PTYPE,'vil','radarvil','refd','refd_com','echotop','afwa_vis','afwa_vis_dust','afwa_vis_alpha')
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--select-condensate-max',action='store_true');a=p.parse_args()
 m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_afwa.F');target=a.checkout/row['path']
 raw=(HERE/'upstream/module_diag_afwa.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 cap='''    CALL GET_ENVIRONMENT_VARIABLE('WOOF_AFWA_CAPTURE_DIR',gpuwm_afwa_directory,STATUS=gpuwm_afwa_status)
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
      WRITE(gpuwm_afwa_unit) 47105,gpuwm_afwa_ie-its+1,gpuwm_afwa_je-jts+1,kme-kms+1,k_end-k_start+1
      WRITE(gpuwm_afwa_unit) grid%itimestep,MERGE(1,0,is_output_timestep),MERGE(1,0,is_first_timestep)
'''
 for n in VOLUMES:cap+=f'      WRITE(gpuwm_afwa_unit) {n}(its:gpuwm_afwa_ie,kms:kme,jts:gpuwm_afwa_je)\n'
 for n in (*PLANES,*(f'grid%{k}' for k in PTYPE)):cap+=f'      WRITE(gpuwm_afwa_unit) {n}(its:gpuwm_afwa_ie,jts:gpuwm_afwa_je)\n'
 cap+='    END IF\n\n';out='    IF (gpuwm_afwa_capture) THEN\n'
 for n in OUTPUTS:out+=f'      WRITE(gpuwm_afwa_unit) grid%{n}(its:gpuwm_afwa_ie,jts:gpuwm_afwa_je)\n'
 out+='      CLOSE(gpuwm_afwa_unit)\n    END IF\n\n'
 if a.select_condensate_max:
  selection='''      gpuwm_afwa_best = -1.
      gpuwm_afwa_ib = its
      gpuwm_afwa_jb = jts
      DO gpuwm_afwa_jj=jts,MIN(jte,jde-1)
        DO gpuwm_afwa_ii=its,MIN(ite,ide-1)
          gpuwm_afwa_sum=0.
          DO gpuwm_afwa_kk=k_start,k_end
            gpuwm_afwa_sum=gpuwm_afwa_sum+ABS(qrain(gpuwm_afwa_ii,gpuwm_afwa_kk,gpuwm_afwa_jj)) &
                 +ABS(qsnow(gpuwm_afwa_ii,gpuwm_afwa_kk,gpuwm_afwa_jj)) &
                 +ABS(qgrpl(gpuwm_afwa_ii,gpuwm_afwa_kk,gpuwm_afwa_jj))
          END DO
          IF(gpuwm_afwa_sum>gpuwm_afwa_best)THEN
            gpuwm_afwa_best=gpuwm_afwa_sum
            gpuwm_afwa_ib=gpuwm_afwa_ii
            gpuwm_afwa_jb=gpuwm_afwa_jj
          END IF
        END DO
      END DO
      gpuwm_afwa_ib=MAX(its,MIN(gpuwm_afwa_ib,MIN(ite,ide-1)-3))
      gpuwm_afwa_jb=MAX(jts,MIN(gpuwm_afwa_jb,MIN(jte,jde-1)-3))
'''
  cap=cap.replace('      gpuwm_afwa_ie = MIN(its+3,ite,ide-1)',selection+'      gpuwm_afwa_ie = MIN(gpuwm_afwa_ib+3,ite,ide-1)').replace('gpuwm_afwa_je = MIN(jts+3,jte,jde-1)','gpuwm_afwa_je = MIN(gpuwm_afwa_jb+3,jte,jde-1)')
  cap=cap.replace('gpuwm_afwa_ie-its+1,gpuwm_afwa_je-jts+1','gpuwm_afwa_ie-gpuwm_afwa_ib+1,gpuwm_afwa_je-gpuwm_afwa_jb+1')
  cap=cap.replace('its:gpuwm_afwa_ie','gpuwm_afwa_ib:gpuwm_afwa_ie').replace('jts:gpuwm_afwa_je','gpuwm_afwa_jb:gpuwm_afwa_je')
  out=out.replace('its:gpuwm_afwa_ie','gpuwm_afwa_ib:gpuwm_afwa_ie').replace('jts:gpuwm_afwa_je','gpuwm_afwa_jb:gpuwm_afwa_je')
  out+='''    IF(gpuwm_afwa_capture)THEN
      OPEN(NEWUNIT=gpuwm_afwa_unit,FILE=TRIM(gpuwm_afwa_filename)//'.coordinates',STATUS='NEW')
      WRITE(gpuwm_afwa_unit,*)gpuwm_afwa_ib,gpuwm_afwa_jb,gpuwm_afwa_ie,gpuwm_afwa_je,gpuwm_afwa_best
      CLOSE(gpuwm_afwa_unit)
    END IF
'''
 declaration=DECL
 if a.select_condensate_max:declaration=declaration.replace('    LOGICAL ::', '    INTEGER :: gpuwm_afwa_ib,gpuwm_afwa_jb,gpuwm_afwa_ii,gpuwm_afwa_jj,gpuwm_afwa_kk\n    REAL :: gpuwm_afwa_best,gpuwm_afwa_sum\n    LOGICAL ::')
 s=raw.decode();decl='    ! Chirp the routine name for debugging purposes';where='    ! Time-step precipitation (convective + nonconvective)';end='  END SUBROUTINE afwa_diagnostics_driver'
 for token in (decl,where,end):assert s.count(token)==1
 s=s.replace(decl,declaration+decl).replace(where,cap+where).replace(end,out+end);target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection='maximum absolute QR+QS+QG column per patch' if a.select_condensate_max else 'first columns per patch',kind='unchanged native expressions; bounded input/result IO only, 4x4 columns, first and three history calls'),indent=2)+'\n')
if __name__=='__main__':main()
