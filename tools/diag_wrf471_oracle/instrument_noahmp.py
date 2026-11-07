#!/usr/bin/env python3
"""Capture actual Noah-MP diagnostics without changing native arithmetic."""
from pathlib import Path
import argparse,hashlib,json,re
HERE=Path(__file__).resolve().parent

def layout():
 s=(HERE/'upstream/module_diag_misc.F').read_text();sub=s[s.index('   subroutine noahmp_output_calc('):s.index('   end subroutine noahmp_output_calc')]
 header=sub[:sub.index('!----------------------------------------------------------------------')];names=re.findall(r'[a-z][a-z0-9_]*',header.replace('&','').split('(',1)[1].rsplit(')',1)[0]);acc=names[names.index('acsagv'):names.index('isnow')]+names[names.index('acqlat'):names.index('sag')];return tuple(names[names.index('sag'):]),tuple(acc)
INPUTS,AMOUNTS=layout()
def main():
 p=argparse.ArgumentParser();p.add_argument('checkout',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();m=json.loads((HERE/'sources.json').read_text());row=next(x for x in m['sources'] if x['path']=='phys/module_diag_misc.F');raw=(HERE/'upstream/module_diag_misc.F').read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];s=raw.decode();decl='   real :: kjconvert';start='   if (noahmp_acc_dt .gt. 0.) then';end='   end subroutine noahmp_output_calc'
 for k in (decl,start,end):assert s.count(k)==1
 s=s.replace(decl,decl+'''
   CHARACTER(LEN=4096) :: gpuwm_noah_dir,gpuwm_noah_file
   INTEGER :: gpuwm_noah_status,gpuwm_noah_unit,gpuwm_noah_i0,gpuwm_noah_i1
   INTEGER :: gpuwm_noah_j0,gpuwm_noah_j1
   LOGICAL :: gpuwm_noah_capture
''')
 cap='''   CALL GET_ENVIRONMENT_VARIABLE('WOOF_NOAHMP_CAPTURE_DIR',gpuwm_noah_dir,STATUS=gpuwm_noah_status)
   gpuwm_noah_capture=gpuwm_noah_status==0 .AND. LEN_TRIM(gpuwm_noah_dir)>0
   gpuwm_noah_capture=gpuwm_noah_capture .AND. noahmp_acc_dt>0.
   gpuwm_noah_capture=gpuwm_noah_capture .AND. (itimestep<=2 .OR. &
        MOD(NINT(curr_secs2),3600)==0 .OR. MOD(NINT(curr_secs2+dt),3600)==0)
   IF(gpuwm_noah_capture)THEN
      gpuwm_noah_i0=i_start(1)
      gpuwm_noah_j0=j_start(1)
      gpuwm_noah_i1=MIN(gpuwm_noah_i0+3,i_end(1))
      gpuwm_noah_j1=MIN(gpuwm_noah_j0+3,j_end(1))
      WRITE(gpuwm_noah_file,'(A,"/noahmp-",I6.6,"-",I6.6,"-",I6.6,".bin")') &
           TRIM(gpuwm_noah_dir),gpuwm_noah_i0,gpuwm_noah_j0,itimestep
      OPEN(NEWUNIT=gpuwm_noah_unit,FILE=TRIM(gpuwm_noah_file),ACCESS='STREAM', &
           FORM='UNFORMATTED',STATUS='NEW',CONVERT='LITTLE_ENDIAN')
      WRITE(gpuwm_noah_unit) 47111,gpuwm_noah_i1-gpuwm_noah_i0+1, &
           gpuwm_noah_j1-gpuwm_noah_j0+1,itimestep,opt_run,stepwtd,54,58
      WRITE(gpuwm_noah_unit) curr_secs2,dt,noahmp_acc_dt
'''
 def writes(names):return ''.join(f'      WRITE(gpuwm_noah_unit) {k}(gpuwm_noah_i0:gpuwm_noah_i1, &\n           gpuwm_noah_j0:gpuwm_noah_j1)\n' for k in names)
 cap+=writes(INPUTS)+writes(('isnow',))+writes(AMOUNTS)+'   ENDIF\n';out='   IF(gpuwm_noah_capture)THEN\n'+writes(AMOUNTS)+'      CLOSE(gpuwm_noah_unit)\n   ENDIF\n';s=s.replace(start,cap+start).replace(end,out+end);target=a.checkout/row['path'];target.write_text(s)
 a.receipt.write_text(json.dumps(dict(wrf=m['wrf'],source=str(target),original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumenter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind='IO-only around unchanged Noah-MP 58-amount arithmetic/reset; actual 54 flux/forcing inputs and integer snow index, bounded 4x4 per MPI patch'),indent=2)+'\n')
if __name__=='__main__':main()
