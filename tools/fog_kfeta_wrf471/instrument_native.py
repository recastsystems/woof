from pathlib import Path
import sys
root=Path(sys.argv[1])
# Capture changes only: all numerical expressions remain native.
for kind in ('sf','bl'):
    p=root/f'phys/module_{kind}_fogdes.F'
    s=p.read_text()
    anchor='   INTEGER :: i,j\n' if kind=='sf' else '   INTEGER :: i,j,k,grav_settling2\n'
    decl='''   INTEGER :: cap_unit, cap_calls=0, cap_status
   LOGICAL :: cap_done=.false., cap_now
   CHARACTER(512) :: cap_dir
   CHARACTER(32) :: cap_rank
'''
    assert anchor in s
    s=s.replace(anchor,anchor+decl,1)
    exec_anchor="   IF     (mminlu" if kind=='sf' else '  grav_settling2 ='
    start=s.index(exec_anchor)
    extra='''   cap_calls=cap_calls+1
   CALL get_environment_variable('WOOF_CAPTURE_DIR',cap_dir,status=cap_status)
   cap_now = .false.
   IF (cap_status==0 .AND. .NOT. cap_done .AND. cap_calls>=6) THEN
      IF (MAXVAL(qc_curr(its:ite,kts:kte,jts:jte))>1.e-8 .OR. cap_calls>=30) THEN
         cap_now=.true.
         CALL get_environment_variable('OMPI_COMM_WORLD_RANK',cap_rank)
'''
    extra+=f"         OPEN(newunit=cap_unit,file=TRIM(cap_dir)//'/{kind}.'//TRIM(cap_rank)//'.bin', &\n              access='stream',form='unformatted',status='replace')\n"
    extra+='         WRITE(cap_unit) ite-its+1, kte-kts+1, jte-jts+1, grav_settling\n'
    if kind=='sf':
        extra+='''         WRITE(cap_unit) dtbl,nlcat
         WRITE(cap_unit) qc_curr(its:ite,kts,jts:jte),rho(its:ite,kts,jts:jte),dz8w(its:ite,kts,jts:jte)
         WRITE(cap_unit) ivgtyp(its:ite,jts:jte),lai(its:ite,jts:jte),wspd(its:ite,jts:jte)
         WRITE(cap_unit) fgdp(its:ite,jts:jte)
'''
    else:
        extra+='''         WRITE(cap_unit) qc_curr(its:ite,kts:kte,jts:jte),dz8w(its:ite,kts:kte,jts:jte)
         WRITE(cap_unit) vdfg(its:ite,jts:jte),dqc(its:ite,kts:kte,jts:jte)
'''
    extra+='      ENDIF\n   ENDIF\n'
    s=s[:start]+extra+s[start:]
    end=f'  END SUBROUTINE {kind}_fogdes'
    tail='   IF (cap_now) THEN\n'
    if kind=='sf':
        tail+='      WRITE(cap_unit) vdfg(its:ite,jts:jte),fgdp(its:ite,jts:jte),dfgdp(its:ite,jts:jte)\n'
    else:
        tail+='      WRITE(cap_unit) dqc(its:ite,kts:kte,jts:jte)\n'
    tail+='      CLOSE(cap_unit)\n      cap_done=.true.\n   ENDIF\n'
    s=s.replace(end,tail+end,1)
    p.write_text(s)
# Whole KF_eta_CPS patch including real RQVFTEN and W0AVG after native update.
p=root/'phys/module_cu_kfeta.F'
s=p.read_text()
anchor='   REAL    :: W0AVGfctr, W0fctr, W0den\n'
decl='''   INTEGER :: cap_unit,cap_status
   LOGICAL :: cap_done=.false.,cap_now
   CHARACTER(512) :: cap_dir
   CHARACTER(32) :: cap_rank
'''
assert anchor in s
s=s.replace(anchor,anchor+decl,1)
anchor='! New trigger function\n     IF (trigger.eq.2) THEN'
extra='''   CALL get_environment_variable('WOOF_CAPTURE_DIR',cap_dir,status=cap_status)
   cap_now=.false.
   IF (cap_status==0 .AND. .NOT. cap_done .AND. KTAU>=31) THEN
      cap_now=.true.
      CALL get_environment_variable('OMPI_COMM_WORLD_RANK',cap_rank)
      OPEN(newunit=cap_unit,file=TRIM(cap_dir)//'/kf.'//TRIM(cap_rank)//'.bin', &
           access='stream',form='unformatted',status='replace')
      WRITE(cap_unit) ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte,ids,ide,jds,jde,trigger
      WRITE(cap_unit) dt,dx,cudt,stepcu
      WRITE(cap_unit) T,QV,Pcps,rho,pi,dz8w,U,V,W0AVG,RQVFTEN
      WRITE(cap_unit) NCA,RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RQRCUTEN,RQSCUTEN
   ENDIF
'''
assert anchor in s
s=s.replace(anchor,extra+anchor,1)
anchor='   END SUBROUTINE KF_eta_CPS'
tail='''   IF (cap_now) THEN
      WRITE(cap_unit) RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN,RQRCUTEN,RQSCUTEN
      WRITE(cap_unit) RAINCV,PRATEC,NCA,CUTOP,CUBOT,SHALL
      IF (trigger==2) WRITE(cap_unit) tpart_h,tpart_v
      CLOSE(cap_unit)
      cap_done=.true.
   ENDIF
'''
s=s.replace(anchor,tail+anchor,1)
p.write_text(s)
