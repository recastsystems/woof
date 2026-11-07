from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
needle='      INTEGER ::  J\n'
assert s.count(needle)==1
s=s.replace(needle,needle+"""      INTEGER :: misc_unit
      CHARACTER(LEN=512) :: misc_dir,misc_file
      LOGICAL :: misc_capture
""")
surfaces=('hfx','lh','qfx','tsk','flhc','exch_temf','flqc','ust','chs2','cqs2','chs','cpm','qgh','qsfc','u10','v10','th2','t2','q2')
controls=('hfx_force','lh_force','tsk_force','hfx_force_tend','lh_force_tend','tsk_force_tend')
def arrays(names):
 return ''.join(f'      WRITE(misc_unit) {n}(its:ite,jts:jte)\n' for n in names)
pre="""      CALL GET_ENVIRONMENT_VARIABLE('MISC_SF89_CAPTURE_DIR',misc_dir)
      misc_capture=LEN_TRIM(misc_dir)>0 .AND. &
         (itimestep==1 .OR. itimestep==2 .OR. itimestep==30 .OR. itimestep==100)
      IF (misc_capture) THEN
      WRITE(misc_file,'(A,A,I4.4,A,I6.6,A,I6.6,A)') &
        TRIM(misc_dir),'/sf89-s',itimestep,'-i',its,'-j',jts,'.input'
      OPEN(NEWUNIT=misc_unit,FILE=TRIM(misc_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
      WRITE(misc_unit) its,ite,jts,jte,itimestep
      WRITE(misc_unit) dt,cp,xlv
"""+''.join(f'      WRITE(misc_unit) {n}\n' for n in controls)+''.join(f'      WRITE(misc_unit) {n}(its:ite,kts,jts:jte)\n' for n in ('th3d','pi3d','rho'))+arrays(('mavail',)+surfaces)+"""      CLOSE(misc_unit)
      END IF
"""
needle='   ! Update forcing fluxes to the current timestep'
assert s.count(needle)==1;s=s.replace(needle,pre+needle)
post="""      IF (misc_capture) THEN
      WRITE(misc_file,'(A,A,I4.4,A,I6.6,A,I6.6,A)') &
        TRIM(misc_dir),'/sf89-s',itimestep,'-i',its,'-j',jts,'.output'
      OPEN(NEWUNIT=misc_unit,FILE=TRIM(misc_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
"""+''.join(f'      WRITE(misc_unit) {n}\n' for n in controls)+arrays(surfaces)+"""      CLOSE(misc_unit)
      END IF
"""
needle='   END SUBROUTINE idealscmsfclay\n'
assert s.count(needle)==1;s=s.replace(needle,post+needle);p.write_text(s)
