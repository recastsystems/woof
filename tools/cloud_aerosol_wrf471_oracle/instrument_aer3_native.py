from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1])/'phys/module_radiation_driver.F'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa', 'use a pristine owned WRF driver copy'
aerosol=p.with_name('module_ra_aerosol.F')
assert hashlib.sha256(aerosol.read_bytes()).hexdigest()=='57c7e7e50a2be0a6ab169fa376da077b2b4251bf9c20fa6a06446934379c8870'
s=p.read_text()
needle='                                    tauaer_sw,ssaaer_sw,asyaer_sw, taod5503d)'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
         DO j=jts,jte
         DO i=its,ite
           CALL lane_aer3_capture(kte-kts+1,p(i,kts:kte+1,j),t(i,kts:kte,j),qv(i,kts:kte,j), &
             dz8w(i,kts:kte,j),qnwfa(i,kts:kte,j),qnifa(i,kts:kte,j),taod5503d(i,kts:kte,j), &
             tauaer_sw(i,kts:kte,j,:),asyaer_sw(i,kts:kte,j,:),ssaaer_sw(i,kts:kte,j,:), &
             aerssa2d(i,j),aerasy2d(i,j),taod5502d(i,j),angexp2d(i,j))
         ENDDO
         ENDDO
''',1)
helper='''
subroutine lane_aer3_capture(n,p,t,qv,dz,nwfa,nifa,aod,tau,asy,ssa,s,g,a,ang)
use iso_c_binding,only:c_int
implicit none
integer,intent(in)::n
real,intent(in)::p(n+1),t(n),qv(n),dz(n),nwfa(n),nifa(n),aod(n),tau(n,14),asy(n,14),ssa(n,14),s,g,a,ang
integer,save::seen=0,kept=0
integer::u,status
character(1024)::folder,filename
interface
 function lane_getpid() bind(C,name='getpid') result(pid)
 import c_int
 integer(c_int)::pid
 end function
end interface
seen=seen+1
if(mod(seen,64)/=0.or.kept>=128) return
call get_environment_variable('WOOF_AER3_CAPTURE',folder,status=status)
if(status/=0.or.len_trim(folder)==0) return
kept=kept+1
write(filename,'(a,a,i0,a,i0,a)') trim(folder),'/aer3-',lane_getpid(),'-',kept,'.bin'
open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='new',convert='little_endian')
write(u) n,p,t,qv,dz,nwfa,nifa,aod,tau,asy,ssa,s,g,a,ang
close(u)
end subroutine
'''
s=s.replace('END MODULE module_radiation_driver',helper+'\nEND MODULE module_radiation_driver');p.write_text(s)
