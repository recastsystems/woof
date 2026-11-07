"""Add opt-in real-run KFCPS input/output capture to our native source copy."""
from pathlib import Path
import hashlib
import json
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path, help='private proof directory holding WRF-4.7.1')
root = parser.parse_args().root.resolve()
p = root / 'WRF-4.7.1/phys/module_cu_kf.F'
original = p.read_bytes()
assert hashlib.sha256(original).hexdigest() == '8042c90f7b7f5b14b36acd809f4f310b8fc3f75101039169e602e7df3b5d2f61'
text = original.decode()
decl = '''
! CU99 proof capture, opt-in, in this private native source only.
   CHARACTER(LEN=512) :: capdir, capfile
   INTEGER :: capunit, capstat
   LOGICAL :: capon, capexists
'''
before = '''
   CALL GET_ENVIRONMENT_VARIABLE('WOOF_KF_CAPTURE_DIR', capdir, STATUS=capstat)
   capon = capstat == 0 .AND. LEN_TRIM(capdir) > 0
   capon = capon .AND. (ktau == 1 .OR. MOD(ktau,240) == 0)
   IF (capon) THEN
      DO j=jts,jte
         DO i=its,ite
            IF (MOD(i,37) /= 0 .OR. MOD(j,43) /= 0) CYCLE
            IF (nca(i,j) > 0.5*dt) CYCLE
            WRITE(capfile,'(A,"/kf-",I8.8,"-",I5.5,"-",I5.5,".bin")') TRIM(capdir), ktau,i,j
            OPEN(NEWUNIT=capunit,FILE=TRIM(capfile),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE')
            WRITE(capunit) INT(Z'4B463939'),1,kte-kts+1,ktau,i,j,stepcu, &
                 MERGE(1,0,adapt_step_flag),MERGE(1,0,warm_rain), &
                 MERGE(1,0,f_qi),MERGE(1,0,f_qs),MERGE(1,0,f_qr),kf_edrates
            WRITE(capunit) dt,dx,cudt,nca(i,j)
            WRITE(capunit) u(i,kts:kte,j),v(i,kts:kte,j),w(i,kts:kte+1,j), &
                 t(i,kts:kte,j),th(i,kts:kte,j),qv(i,kts:kte,j), &
                 dz8w(i,kts:kte,j),pcps(i,kts:kte,j),rho(i,kts:kte,j), &
                 pi(i,kts:kte,j),w0avg(i,kts:kte,j)
            CLOSE(capunit)
         END DO
      END DO
   END IF
'''
after = '''
   IF (capon) THEN
      DO j=jts,jte
         DO i=its,ite
            IF (MOD(i,37) /= 0 .OR. MOD(j,43) /= 0) CYCLE
            WRITE(capfile,'(A,"/kf-",I8.8,"-",I5.5,"-",I5.5,".bin")') TRIM(capdir), ktau,i,j
            INQUIRE(FILE=TRIM(capfile),EXIST=capexists)
            IF (.NOT. capexists) CYCLE
            OPEN(NEWUNIT=capunit,FILE=TRIM(capfile),ACCESS='STREAM', &
                 FORM='UNFORMATTED',STATUS='OLD',POSITION='APPEND')
            WRITE(capunit) MERGE(1,0,cu_act_flag(i,j))
            WRITE(capunit) raincv(i,j),pratec(i,j),nca(i,j),timec_kf(i,j)
            WRITE(capunit) rthcuten(i,kts:kte,j),rqvcuten(i,kts:kte,j), &
                 rqccuten(i,kts:kte,j),rqrcuten(i,kts:kte,j), &
                 rqicuten(i,kts:kte,j),rqscuten(i,kts:kte,j), &
                 w0avg(i,kts:kte,j),udr_kf(i,kts:kte,j),ddr_kf(i,kts:kte,j), &
                 uer_kf(i,kts:kte,j),der_kf(i,kts:kte,j)
            CLOSE(capunit)
         END DO
      END DO
   END IF
'''
text = text.replace('   REAL    :: W0AVGfctr, W0fctr, W0den', '   REAL    :: W0AVGfctr, W0fctr, W0den\n' + decl, 1)
text = text.replace('   DXSQ=DX*DX', before + '\n   DXSQ=DX*DX', 1)
assert 'END SUBROUTINE KFCPS' in text
text = text.replace('END SUBROUTINE KFCPS', after + '\nEND SUBROUTINE KFCPS', 1)
assert 'INTEGER :: capunit' in text
p.write_text(text)
(root/'capture-instrumentation.json').write_text(json.dumps({
    'source':str(p),'original_sha256':hashlib.sha256(original).hexdigest(),
    'instrumented_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'enable_env':'WOOF_KF_CAPTURE_DIR','selection':'eligible NCA, i%37=0, j%43=0, KTAU=1 or KTAU%240=0',
    'scope':'All arithmetic unchanged; entry/exit binary I/O only; no scientific input modifications.'
},indent=2)+'\n')
print('Instrumented private native KFCPS')
