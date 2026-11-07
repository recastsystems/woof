from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1])/'phys/module_ra_eclipse.F'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='c18f6bebf1f8090f1dd36a95c3513824de4f522f3f35c83a96cc987eaf618381', 'use a pristine owned WRF eclipse copy'
s=p.read_text();needle='end subroutine';assert s.lower().count(needle)==3
idx=s.lower().index(needle);s=s[:idx]+'''call lane_eclipse_capture(ims,ime,jms,jme,its,ite,jts,jte,t1,degrad, &
 X,Y,d,l1,l2,mu,t0,tmin,tmax,tanf1,tanf2,Dt,xlat,xlon,Obscur,mask,elat_track,elon_track)
'''+s[idx:]
helper='''
subroutine lane_eclipse_capture(ims,ime,jms,jme,its,ite,jts,jte,hour,degrad, &
 X,Y,d,l1,l2,mu,t0,tmin,tmax,tanf1,tanf2,Dt,lat,lon,obsc,mask,elat,elon)
use iso_c_binding,only:c_int
implicit none
integer,intent(in)::ims,ime,jms,jme,its,ite,jts,jte,mask(ims:ime,jms:jme)
real,intent(in)::hour,degrad,X(4),Y(4),d(4),l1(4),l2(4),mu(4),t0,tmin,tmax,tanf1,tanf2,Dt, &
lat(ims:ime,jms:jme),lon(ims:ime,jms:jme),obsc(ims:ime,jms:jme),elat,elon
integer,save::count=0
integer::u,status,i,j,n
character(1024)::folder,filename
interface
 function lane_getpid() bind(C,name='getpid') result(pid)
 import c_int
 integer(c_int)::pid
 end function
end interface
call get_environment_variable('WOOF_ECLIPSE_CAPTURE',folder,status=status)
if(status/=0.or.len_trim(folder)==0.or.count>=64)return
count=count+1
write(filename,'(a,a,i0,a,i0,a)') trim(folder),'/eclipse-',lane_getpid(),'-',count,'.bin'
n=(ite-its+1)*(jte-jts+1)
open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='new',convert='little_endian')
write(u)n,hour,degrad,t0,tmin,tmax,X,Y,d,l1,l2,mu,tanf1,tanf2,Dt,elat,elon
write(u)((lat(i,j),i=its,ite),j=jts,jte),((lon(i,j),i=its,ite),j=jts,jte), &
 ((obsc(i,j),i=its,ite),j=jts,jte),((mask(i,j),i=its,ite),j=jts,jte)
close(u)
end subroutine
'''
s=s.replace('end module module_ra_eclipse',helper+'\nend module module_ra_eclipse');p.write_text(s)
