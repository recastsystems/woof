"""Capture first nonzero RK2 scalar updates and print native stage lengths."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1])
p=root/'dyn_em/module_em.F';source=p.read_text();old=hashlib.sha256(source.encode()).hexdigest()
for name,mode in [('rk_update_scalar',1),('rk_update_scalar_pd',2)]:
    begin=source.index('SUBROUTINE '+name+'(')
    end=source.index('END SUBROUTINE '+name,begin)
    body=source[begin:end]
    declarations='''   LOGICAL, SAVE :: dyn_seen(3)=.false.
   LOGICAL :: dyn_tap
   INTEGER :: dyn_unit,dyn_status,dyn_flags
   CHARACTER(LEN=1024) :: dyn_dir,dyn_path
'''
    body=body.replace('   INTEGER :: i,j,k,im',declarations+'   INTEGER :: i,j,k,im',1)
    field='scalar_2' if mode==1 else 'scalar'
    at=body.index('      i_start = its')
    inputs='scalar_1,scalar_2,sc_tend,advect_tend,msftx,msfty' if mode==1 else 'scalar,sc_tend'
    outputs='scalar_1,scalar_2,sc_tend' if mode==1 else 'scalar,sc_tend'
    tap=f'''   CALL get_environment_variable('WOOF_RK_CAPTURE',dyn_dir,STATUS=dyn_status)
   dyn_tap=.false.
   IF(config_flags%rk_ord==2.AND.rk_step>=1.AND.rk_step<=3) THEN
     dyn_tap=.NOT.dyn_seen(rk_step).AND.dyn_status==0.AND.LEN_TRIM(dyn_dir)>0 &
       .AND.MAXVAL(ABS({field}(its:MIN(ite,ide-1),kts:MIN(kte,kde-1),jts:MIN(jte,jde-1),scs:sce)))>0.
   ENDIF
   IF(dyn_tap) THEN
     dyn_seen(rk_step)=.true.
     WRITE(dyn_path,'(A,A,I0,A,I0,A,I0,A)') TRIM(dyn_dir),'/{name}-',rk_step,'-',its,'-',jts,'.bin'
     OPEN(NEWUNIT=dyn_unit,FILE=TRIM(dyn_path),ACCESS='STREAM',FORM='UNFORMATTED', &
       STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
     dyn_flags=0
     IF(config_flags%specified) dyn_flags=IBSET(dyn_flags,0)
     IF(config_flags%nested) dyn_flags=IBSET(dyn_flags,1)
     IF(config_flags%periodic_x) dyn_flags=IBSET(dyn_flags,2)
     WRITE(dyn_unit) 1,{mode},ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte,scs,sce,rk_step,config_flags%rk_ord,spec_zone,dyn_flags
     WRITE(dyn_unit) dt
     WRITE(dyn_unit) {inputs}
     WRITE(dyn_unit) mu_old,mu_new,mu_base,c1,c2
   ENDIF
'''
    body=body[:at]+tap+body[at:]
    body+=f'\n   IF(dyn_tap) THEN\n     WRITE(dyn_unit) {outputs}\n     CLOSE(dyn_unit)\n   ENDIF\n'
    source=source[:begin]+body+source[end:]
p.write_text(source)
q=root/'dyn_em/solve_em.F';s=q.read_text();oldsolve=hashlib.sha256(s.encode()).hexdigest()
at=s.index('   Runge_Kutta_loop:')
where=s.index('     ENDIF',s.index("' unknown solver",at))+len('     ENDIF')
s=s[:where]+'''
     IF(rk_order==2) THEN
       WRITE(wrf_err_message,*) 'DYN_RK2_STAGE ',rk_step,rk_order,dt_rk,dts_rk,number_of_small_timesteps
       CALL wrf_message(wrf_err_message)
     ENDIF
'''+s[where:]
q.write_text(s)
(root/'rk2-instrumentation.json').write_text(json.dumps(dict(module_em_before=old,module_em_after=hashlib.sha256(source.encode()).hexdigest(),solve_em_before=oldsolve,solve_em_after=hashlib.sha256(s.encode()).hexdigest(),arithmetic_changes=False,protocol='26 int32 header, dt float32, native Fortran arrays and outputs; first nonzero per RK stage'),indent=2)+'\n')
