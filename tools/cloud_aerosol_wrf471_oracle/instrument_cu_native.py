"""Capture real native CU radiation merges without replacing physics."""
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1])/'phys/module_radiation_driver.F'
s=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa'
def capture(mode,phase):
 return f'''          IF (icloud_cu.eq.{mode}) THEN
            DO j=jts,jte
            DO i=its,ite
              CALL lane_cu_capture({mode},{phase},kte-kts+1, &
                qc(i,kts:kte,j),qi(i,kts:kte,j),qc_cu(i,kts:kte,j),qi_cu(i,kts:kte,j), &
                cldfra_dp(i,kts:kte,j),cldfra_sh(i,kts:kte,j),cldfra(i,kts:kte,j))
            ENDDO
            ENDDO
          ENDIF
'''
needle='          IF ( F_QC .AND. icloud_cu .EQ. 1 ) THEN'
assert needle in s;s=s.replace(needle,capture(1,0)+needle,1)
needle='#if (EM_CORE == 1)\n! temporarily modify hydrometeors (for P3'
assert needle in s;s=s.replace(needle,capture(1,1)+'\n'+needle,1)
needle='             CALL wrf_debug (1, \'use kf cldfra\')'
assert needle in s;s=s.replace(needle,needle+'\n'+capture(2,0),1)
needle='             IF (aercu_opt.gt.0.0) THEN'
assert needle in s;s=s.replace(needle,capture(2,1)+needle,1)
helper='''
subroutine lane_cu_capture(mode,phase,n,qc,qi,qccu,qicu,deep,shallow,cf)
use iso_c_binding, only: c_int
implicit none
integer,intent(in)::mode,phase,n
real,intent(in)::qc(n),qi(n),qccu(n),qicu(n),deep(n),shallow(n),cf(n)
integer,save::kept(0:1,2)=0
integer::u,status
real::zero(n)
character(1024)::folder,filename
interface
 function lane_getpid() bind(C,name='getpid') result(pid)
 import c_int
 integer(c_int)::pid
 end function
end interface
call get_environment_variable('WOOF_CU_CAPTURE',folder,status=status)
if(status/=0.or.len_trim(folder)==0) return
if(kept(phase,mode)>=128) return
if(kept(phase,mode)>=4.and.all(qccu==0.).and.all(qicu==0.).and.all(deep==0.).and.all(shallow==0.)) return
kept(phase,mode)=kept(phase,mode)+1
write(filename,'(a,a,i0,a,i0,a,i0,a)') trim(folder),'/cu-',mode,'-',lane_getpid(),'-',kept(phase,mode),'.bin'
zero=0.
if(phase==0) then
 open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='new',convert='little_endian')
 write(u) mode,n,qc,qi,qccu,qicu,deep,shallow
 if(mode==1) then
   write(u) zero
 else
   write(u) cf
 endif
else
 open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='old',position='append', &
  convert='little_endian')
 write(u) qc,qi
 if(mode==1) then
   write(u) zero
 else
   write(u) cf
 endif
endif
close(u)
end subroutine
'''
s=s.replace('END MODULE module_radiation_driver',helper+'\nEND MODULE module_radiation_driver')
p.write_text(s)
