from pathlib import Path
import numpy as np, subprocess, hashlib,json
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--wrf',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
root=args.work;root.mkdir(parents=True,exist_ok=True);out=args.out;out.mkdir(parents=True,exist_ok=True)
wrf=args.wrf/'phys'; source=(wrf/'module_cumulus_driver.F').read_text(); sub=source[source.index('SUBROUTINE aer_time_int_cu'):source.index('END SUBROUTINE aer_p_int_cu')+len('END SUBROUTINE aer_p_int_cu')]
rng=np.random.default_rng(73);ncol=64;nz=8
monthly=(rng.random((10,12,30,ncol))*10000).astype('f4');pm=np.broadcast_to(np.linspace(100,90000,30,dtype='f4')[None,:,None],(12,30,ncol)).copy();p=np.broadcast_to(np.array([100000,80000,50000,30000,10000,3000,100,25],dtype='f4')[:,None],(nz,ncol)).copy()
fields={n:(rng.random((nz,ncol))*scale).astype('f4') for n,scale in [('qc',.001),('qi',.001),('qs',.001),('cloud',1),('deep',.6),('shallow',.4),('cu_qc',.002),('cu_qi',.002),('cu_qs',.002),('cu_ec',20),('cu_ei',100),('cu_es',500),('ec',20),('ei',100),('es',500)]}
for name in ('qc','qi','qs','cu_qc','cu_qi','cu_qs'):fields[name][:,0:4]=0
fields['cu_qc'][:,1]=.001;fields['qc'][:,2]=.001
np.concatenate([monthly.ravel(),pm.ravel(),p.ravel()]+[v.ravel() for v in fields.values()]).astype('f4').tofile(root/'handoff-input.bin')
rad=(wrf/'module_radiation_driver.F').read_text().splitlines();feedback='\n'.join(rad[1336:1391]);# exact source body through radius update
feedback=feedback.replace('cldfra_dp','deep').replace('cldfra_sh','shallow').replace('CLDFRA','cloud').replace('cldfra_cu','cf').replace('qc_cu_weight','wc').replace('qi_cu_weight','wi').replace('qs_cu_weight','ws').replace('qc_cu','cu_qc').replace('qi_cu','cu_qi').replace('qs_cu','cu_qs').replace('EFCS','cu_ec').replace('EFIS','cu_ei').replace('EFSS','cu_es').replace('EFCG','ec').replace('EFIG','ei').replace('EFSG','es').replace('re_cloud','rc').replace('re_ice','ri').replace('re_snow','rs')
feedback=feedback.replace('IF (aercu_opt.gt.0.0) THEN','').replace('has_reqc = 1','').replace('has_reqi = 1','').replace('has_reqs = 1','')
# inspect source endpoints, make terminal loops explicit independently of crop.
feedback=feedback[:feedback.index('has_reqc')] if 'has_reqc' in feedback else feedback
# Keep original core body between first DO and completed radius conversions.
feedback=feedback[:feedback.index('                rs(i,k,j)  = rs(i,k,j) * 1.E-6')+len('                rs(i,k,j)  = rs(i,k,j) * 1.E-6')]
feedback+='\n qs(i,k,j)=qs(i,k,j)+cu_qs(i,k,j)*cf(i,k,j)\n enddo\n enddo\n enddo\n'
names=list(fields); decl=','.join(names+['cf','wc','wi','ws','rc','ri','rs'])
program=f'''program handoffs
implicit none
integer,parameter:: nc=64,nz=8
real:: monthly(nc,30,1,12,10),pm(nc,30,1,12,1),p(nc,nz,1),time(nc,30,1,10),pt(nc,30,1,1),result(nc,nz,1,11)
real,dimension(nc,nz,1):: {decl}
real:: days(8)=[0.,15.,44.5,74.,180.25,349.,364.75,365.]
integer:: i,j,k,t,its,ite,jts,jte,kts,kte
its=1;ite=nc;jts=1;jte=1;kts=1;kte=nz
open(10,file='{root}/handoff-input.bin',access='stream',form='unformatted')
read(10)monthly,pm,p,{','.join(names)}
close(10)
open(11,file='{root}/handoff-output.bin',access='stream',form='unformatted',status='replace')
do t=1,8
call aer_time_int_cu(1,days(t),monthly,time,30,12,10,1,nc,1,1,1,nz,1,nc,1,1,1,nz,1,nc,1,1,1,nz)
call aer_time_int_cu(1,days(t),pm,pt,30,12,1,1,nc,1,1,1,nz,1,nc,1,1,1,nz,1,nc,1,1,1,nz)
call aer_p_int_cu(p,pt(:,:,:,1),30,time,result,10,p,1,nc,1,1,1,nz,1,nc,1,1,1,nz,1,nc,1,1,1,nz)
write(11)result(:,:,:,1:10)
enddo
{feedback}
write(11)qc,qi,qs,cloud,rc,ri,rs
close(11)
end program
{sub}
subroutine wrf_error_fatal(text)
character(*)::text
print *,text
stop 1
end subroutine
'''
(root/'handoff-oracle.f90').write_text(program)
subprocess.run(['gfortran','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-ffree-line-length-none',str(root/'handoff-oracle.f90'),'-o',str(root/'handoff-oracle')],check=True)
subprocess.run([str(root/'handoff-oracle')],check=True)
words=np.fromfile(root/'handoff-output.bin','f4');aerosol=words[:8*10*nz*ncol].reshape(8,10,nz,ncol);rad=words[8*10*nz*ncol:].reshape(7,nz,ncol)
np.savez_compressed(out/'handoffs.npz',monthly=monthly,pressure_monthly=pm,pressure=p,days=np.array([0,15,44.5,74,180.25,349,364.75,365],dtype='f4'),aerosol=aerosol,radiation=rad,**fields)
(out/'manifest.json').write_text(json.dumps({'source_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28','columns':64,'months':8,'model_levels':nz,'source_sha256':{name:hashlib.sha256((wrf/name).read_bytes()).hexdigest() for name in ['module_cumulus_driver.F','module_radiation_driver.F']},'generated_fortran_sha256':hashlib.sha256(program.encode()).hexdigest(),'compile':['gfortran','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-ffree-line-length-none'],'fixture_sha256':hashlib.sha256((out/'handoffs.npz').read_bytes()).hexdigest()},indent=2)+'\n')
print('HANDOFF_ORACLE',words.size,'words', (out/'handoffs.npz').stat().st_size)
