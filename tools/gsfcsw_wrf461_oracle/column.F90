! Independent oracle: byte-unmodified WRF module, one call per sounding.
subroutine gsfc_column(nlay,warm,fqi,fqn,julday,solcon,latitude,inp,out) bind(C)
use iso_c_binding
use module_ra_gsfcsw, only: gsfcswrad,gsfc_swinit
implicit none
integer(c_int),value :: nlay,warm,fqi,fqn,julday
real(c_float),value :: solcon,latitude
real(c_float),intent(in) :: inp(9*(nlay+1)+2)
real(c_float),intent(out) :: out(3*(nlay+1)+2)
real :: t(1,nlay+1,1),p(1,nlay+1,1),p8(1,nlay+1,1),ex(1,nlay+1,1)
real :: qv(1,nlay+1,1),qc(1,nlay+1,1),qi(1,nlay+1,1),cf(1,nlay+1,1),qn(1,nlay+1,1)
real :: zero(1,nlay+1,1),heat(1,nlay+1,1),tc(1,nlay+1,1),ti(1,nlay+1,1)
real :: alb(1,1),mu(1,1),gsw(1,1),toa(1,1),obscur(1,1)
integer :: n
n=nlay+1
t(1,:,1)=inp(1:n); p(1,:,1)=inp(n+1:2*n); p8(1,:,1)=inp(2*n+1:3*n)
ex(1,:,1)=inp(3*n+1:4*n); qv(1,:,1)=inp(4*n+1:5*n); qc(1,:,1)=inp(5*n+1:6*n)
qi(1,:,1)=inp(6*n+1:7*n); cf(1,:,1)=inp(7*n+1:8*n); qn(1,:,1)=inp(8*n+1:9*n)
mu(1,1)=inp(9*n+1); alb(1,1)=inp(9*n+2)
zero=0.; heat=0.; tc=0.; ti=0.; obscur=0.; gsw=0.; toa=0.
call gsfc_swinit(latitude,.true.)
call gsfcswrad(heat,gsw,zero,zero,alb,t,qv,qc,zero,qi,zero,zero,qn,p,p8,ex,cf,toa, &
 1004.5,9.81,julday,solcon,ti,tc,warm/=0, &
 f_qv=.true.,f_qc=.true.,f_qr=.false.,f_qi=fqi/=0,f_qs=.false.,f_qg=.false.,f_qndrop=fqn/=0, &
 coszen=mu,obscur=obscur, &
 ids=1,ide=2,jds=1,jde=2,kds=1,kde=n,ims=1,ime=1,jms=1,jme=1,kms=1,kme=n, &
 its=1,ite=1,jts=1,jte=1,kts=1,kte=nlay)
out(1:n)=heat(1,:,1); out(n+1:2*n)=ti(1,:,1); out(2*n+1:3*n)=tc(1,:,1)
out(3*n+1)=gsw(1,1); out(3*n+2)=toa(1,1)
end subroutine
