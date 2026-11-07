"""Add bounded binary captures to an owned copy of public WRF 4.7.1.

No physics statements are replaced. The original input and result words are
written beside the native run, at most 64 sampled columns per process.
"""
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1])
pins = {
    "module_radiation_driver.F": "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa",
    "module_ra_farms.F": "1b9e060d71dd7218c39e5a5724a7886307507a7cb85d7db33cd3ed52c0034a60",
}
for name, digest in pins.items():
    path = root / "phys" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name

path = root / "phys/module_radiation_driver.F"
text = path.read_text()
start = text.index("      SUBROUTINE cal_cldfra3(")
end = text.index("      END SUBROUTINE cal_cldfra3", start)
routine = text[start:end]
routine = routine.replace("      character*512 dbg_msg", "      integer :: lane_capture_unit\n      character*512 dbg_msg", 1)
routine = routine.replace("!..Initialize cloud fraction, compute RH, and rho-air.", """      call lane_cloud_before(kte-kts+1,qv,qc,qi,qs,dz,p,t, &
     & xland,gridkm,modify_qvapor,max_relh,lane_capture_unit)

!..Initialize cloud fraction, compute RH, and rho-air.""", 1)
routine += """      if (lane_capture_unit /= -1) then
         write(lane_capture_unit) cldfra,qv,qc,qi
         close(lane_capture_unit)
      endif
"""
text = text[:start] + routine + text[end:]
text += """
subroutine lane_cloud_before(n,qv,qc,qi,qs,dz,p,t,land,grid,modify,maxrh,u)
use iso_c_binding, only: c_int
implicit none
integer,intent(in)::n
real,intent(in)::qv(n),qc(n),qi(n),qs(n),dz(n),p(n),t(n),land,grid,maxrh
logical,intent(in)::modify
integer,intent(out)::u
integer,save::seen=0,kept=0
integer::status
character(1024)::folder,filename
interface
 function lane_getpid() bind(C,name='getpid') result(pid)
 import c_int
 integer(c_int)::pid
 end function
end interface
u=-1
seen=seen+1
if (mod(seen,64)/=0.or.kept>=64) return
call get_environment_variable('WOOF_CAPTURE_DIR',folder,status=status)
if (status/=0.or.len_trim(folder)==0) return
kept=kept+1
write(filename,'(a,a,i0,a,i0,a)') trim(folder),'/cloud-',lane_getpid(),'-',kept,'.bin'
open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='new',convert='little_endian')
write(u) n,qv,qc,qi,qs,dz,p,t,land,grid,real(merge(1,0,modify)),maxrh
end subroutine
"""
path.write_text(text)

path = root / "phys/module_ra_farms.F"
text = path.read_text()
needle = "        end do i_loop"
assert needle in text
text = text.replace(needle, """           call lane_farms_capture(kte-kts+1,has_reqc,has_reqi,has_reqs, &
             p8w(i,kts:kte,j),rho(i,kts:kte,j),dz8w(i,kts:kte,j), &
             qv(i,kts:kte,j),qi(i,kts:kte,j),qs(i,kts:kte,j),qc(i,kts:kte,j), &
             re_cloud(i,kts:kte,j),re_ice(i,kts:kte,j),re_snow(i,kts:kte,j), &
             cldfra(i,kts:kte,j),albedo(i,j),aerssa2d(i,j),aerasy2d(i,j), &
             aod5502d(i,j),angexp2d(i,j),coszen_loc(i,j),julian, &
             swdown2(i,j),swddir2(i,j),swddni2(i,j),swddif2(i,j),swdownc2(i,j),swddnic2(i,j))
""" + needle, 1)
text += """
subroutine lane_farms_capture(n,hc,hi,hs,p,rho,dz,qv,qi,qs,qc,rc,ri,rs,cf, &
 alb,ssa,asy,aod,ang,cz,julian,sw,dir,dni,dif,swc,dnic)
use iso_c_binding, only: c_int
implicit none
integer,intent(in)::n,hc,hi,hs
real,intent(in)::p(n),rho(n),dz(n),qv(n),qi(n),qs(n),qc(n),rc(n),ri(n),rs(n),cf(n)
real,intent(in)::alb,ssa,asy,aod,ang,cz,julian,sw,dir,dni,dif,swc,dnic
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
if(mod(seen,64)/=0.or.kept>=64) return
call get_environment_variable('WOOF_CAPTURE_DIR',folder,status=status)
if(status/=0.or.len_trim(folder)==0) return
kept=kept+1
write(filename,'(a,a,i0,a,i0,a)') trim(folder),'/farms-',lane_getpid(),'-',kept,'.bin'
open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='new',convert='little_endian')
write(u) n,hc,hi,hs,p,rho,dz,qv,qi,qs,qc,rc,ri,rs,cf,alb,ssa,asy,aod,ang,cz,julian,sw,dir,dni,dif,swc,dnic
close(u)
end subroutine
"""
path.write_text(text)
