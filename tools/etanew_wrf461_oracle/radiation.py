"""Build the unchanged WRF 4.6.1 cal_cldfra1 routine for older Eta fixtures."""
from pathlib import Path
import subprocess
import sys
source, build, out = map(Path,sys.argv[1:])
text=(source/'phys/module_radiation_driver.F').read_text()
a=text.index('   SUBROUTINE cal_cldfra1(')
b=text.index('   END SUBROUTINE cal_cldfra1',a)+len('   END SUBROUTINE cal_cldfra1')
program='''program oracle
implicit none
integer,parameter::n=12
integer::i
integer::flag(n,1,1)
real::qv(n,1,1),qc(n,1,1),qi(n,1,1),qs(n,1,1),t(n,1,1),p(n,1,1),ice(n,1,1),rain(n,1,1),cloud(n,1,1)
qv(:,1,1)=[.0003,.0003,.0003,.0003,.003,.003,.0,.0008,.0008,.0008,.0001,.0001]
qc(:,1,1)=[0.,0.,0.,0.,.0001,0.,0.,.00002,.00002,.00002,.0,.0]
qs(:,1,1)=[.00002,.00002,.00002,.00002,.0,.0001,0.,.00005,.00005,.00005,1.e-13,.000001]
ice(:,1,1)=[0.,.25,.75,1.,0.,1.,.5,0.,.5,1.,1.,.8]
t(:,1,1)=[250.,250.,250.,250.,280.,260.,260.,260.,260.,260.,240.,240.]
p=40000.; qi=0.; rain=0.
call cal_cldfra1(cloud,qv,qc,qi,qs,.true.,.true.,.false.,.true.,t,p,ice,rain,95,flag, &
1,n,1,1,1,1,1,n,1,1,1,1,1,n,1,1,1,1)
write(*,'(A)') 'qv,qc,qs,t,p,f_ice,cldfra'
do i=1,n
write(*,'(7(Z8.8,:,","))') transfer(qv(i,1,1),0),transfer(qc(i,1,1),0), &
transfer(qs(i,1,1),0),transfer(t(i,1,1),0),transfer(p(i,1,1),0), &
transfer(ice(i,1,1),0),transfer(cloud(i,1,1),0)
enddo
contains
'''+text[a:b]+'\nend program oracle\n'
out.mkdir(parents=True,exist_ok=True)
(out/'oracle.f90').write_text(program)
subprocess.run(['gfortran','-O0','-ffp-contract=off','-ffree-line-length-none','-I'+str(build/'frame'),str(out/'oracle.f90'),'-o',str(out/'oracle')],check=True)
result=subprocess.check_output([str(out/'oracle')])
(out/'radiation.csv').write_bytes(result)
print(result.decode())
