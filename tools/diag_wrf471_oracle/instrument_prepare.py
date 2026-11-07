#!/usr/bin/env python3
"""Capture native phy_prep inputs/outputs; bounded IO additions only."""
import argparse, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 target=a.checkout/'dyn_em/module_big_step_utilities_em.F';original=target.read_bytes()
 manifest=json.loads((HERE/'sources.json').read_text());row=next(r for r in manifest['sources'] if r['path']=='dyn_em/module_big_step_utilities_em.F')
 if hashlib.sha256(original).hexdigest()!=row['sha256']:raise ValueError('native source is outside pin')
 text=original.decode();start=text.index('   SUBROUTINE phy_prep (');end=text.index('END SUBROUTINE phy_prep',start);body=text[start:end]
 decl='''   INTEGER, SAVE :: gpuwm_capture_count = 0
   INTEGER :: gpuwm_unit, gpuwm_status, gpuwm_is, gpuwm_ie, gpuwm_js, gpuwm_je
   CHARACTER(LEN=2048) :: gpuwm_directory, gpuwm_filename
   LOGICAL :: gpuwm_capture_active
'''
 body=body.replace('   INTEGER :: n\n','   INTEGER :: n\n'+decl,1)
 capture='''    gpuwm_capture_active = .FALSE.
    CALL GET_ENVIRONMENT_VARIABLE('WOOF_DIAG_CAPTURE_DIR',gpuwm_directory,STATUS=gpuwm_status)
    IF (gpuwm_status == 0 .AND. LEN_TRIM(gpuwm_directory)>0 .AND. gpuwm_capture_count<2) THEN
       gpuwm_capture_active = .TRUE.
       gpuwm_capture_count = gpuwm_capture_count+1
       gpuwm_is=i_start
       gpuwm_ie=MIN(i_start+3,i_end)
       gpuwm_js=j_start
       gpuwm_je=MIN(j_start+3,j_end)
       WRITE(gpuwm_filename,'(A,"/prep-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
          TRIM(gpuwm_directory),gpuwm_is,gpuwm_js,gpuwm_capture_count
       OPEN(NEWUNIT=gpuwm_unit,FILE=TRIM(gpuwm_filename),ACCESS='STREAM', &
          FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
       WRITE(gpuwm_unit) 47103,gpuwm_ie-gpuwm_is+1,gpuwm_je-gpuwm_js+1,k_end-k_start+1
'''
 mass='gpuwm_is:gpuwm_ie,k_start:k_end,gpuwm_js:gpuwm_je';full='gpuwm_is:gpuwm_ie,k_start:kte,gpuwm_js:gpuwm_je'
 for field in ('p','pb','alt'):capture+=f'       WRITE(gpuwm_unit) {field}({mass})\n'
 capture+=f'       WRITE(gpuwm_unit) moist({mass},P_QV)\n'
 for field in ('ph','phb'):capture+=f'       WRITE(gpuwm_unit) {field}({full})\n'
 capture+='       WRITE(gpuwm_unit) fzm(k_start:kte),fzp(k_start:kte)\n    END IF\n'
 anchor='!  compute thermodynamics and velocities at pressure points (or half levels)'
 body=body.replace(anchor,capture+'\n'+anchor,1)
 body+='    IF (gpuwm_capture_active) THEN\n'+f'       WRITE(gpuwm_unit) rho({mass})\n       WRITE(gpuwm_unit) p8w({full})\n'+'       CLOSE(gpuwm_unit)\n    END IF\n\n'
 changed=(text[:start]+body+text[end:]).encode();target.write_bytes(changed)
 import difflib
 delta=list(difflib.unified_diff(text.splitlines(),changed.decode().splitlines()))
 removed=[s for s in delta if s.startswith('-') and not s.startswith('---')]
 if removed:raise ValueError('instrumentation changed existing expressions')
 a.receipt.write_text(json.dumps({'source':str(target),'original_sha256':row['sha256'],'instrumented_sha256':hashlib.sha256(changed).hexdigest(),'added_lines':len(changed.decode().splitlines())-len(text.splitlines()),'removed_lines':0,'limit':'first two calls per MPI patch, first 4x4 columns, all levels','wrf':manifest['wrf']},indent=2)+'\n')
if __name__=='__main__':main()
