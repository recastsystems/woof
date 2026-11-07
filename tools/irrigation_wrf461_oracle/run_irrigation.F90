program run_irrigation
use module_irrigation
implicit none
integer :: option,c,step,ph,freq,sh,nh,sj,ej,rnd,i,j,u,on
real :: frac,amount,jul,xt,gmt,dt,rain,channel,qr,rho,dz
real :: f(1:1,1:1),r(1:1,1:1,1:1),z(1:1,1:1,1:1),q(1:1,1:1,1:1)
integer :: p(1:1,1:1)
open(newunit=u,file='oracle.txt',status='replace')
write(u,'(a)') 'option case step i j ph freq sh nh sj ej rnd frac amount jul xt gmt dt rho dz rain channel qr'
do option=1,3
 do c=0,191
  ph=mod(c,3); freq=1+2*mod(c/3,2); sh=mod(c/6,2)*22; nh=4
  sj=100; ej=200
  if(mod(c/12,2)==1)then
   sj=300; ej=50
  endif
  rnd=mod(c/24,3); i=1; j=1
  frac=1.
  select case(mod(c/72,3))
   case(0); frac=0.
   case(1); frac=0.05
  end select
  jul=real(99+mod(c,5)*26); xt=real(mod(c/3,4)*360)
  gmt=-2.25; amount=17.3; dt=75.; rho=1.17; dz=37.1
  if(mod(c,16)==0)then
   xt=1439.; jul=149.999
  endif
  rain=0.; channel=123.; qr=0.00002
  if(mod(c,11)==0) rain=0.010000001
  if(mod(c,13)==0) rain=0.01
  do step=1,4
   if(option==1)call channel_irrigation(jul,frac,option,amount,sh,nh,sj,ej,freq,ph,i,j,rain,channel,gmt,xt,dt,rnd)
   if(option==2)call drip_irrigation(jul,frac,option,amount,sh,nh,sj,ej,freq,ph,i,j,rain,channel,gmt,xt,dt,rnd)
   if(option==3)then
    f=frac; r=rho; z=dz; q=qr; p=rnd
    call sprinkler_irrigation(jul,f,amount,r,z,sh,nh,sj,ej,freq,ph,q,gmt,xt,dt,p,1,2,1,2,1,1,1,1,1,1,1,1,1,1)
    qr=q(1,1,1)
   endif
   write(u,'(12(i0,1x),11(es24.16e3,1x))') option,c,step,i,j,ph,freq,sh,nh,sj,ej,rnd,frac,amount,jul,xt,gmt,dt,rho,dz,rain,channel,qr
   xt=xt+dt/60.; jul=jul+dt/86400.
  enddo
 enddo
enddo
close(u)
end program
