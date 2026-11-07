program driver
use soil_oracle
implicit none
integer,parameter::nc=64,na=6,n=3
real::tsk(nc+1,2),tmn(nc+1,2),mask(nc+1,2),sst(nc+1,2)
real::st(nc+1,na,2),sm(nc+1,na,2),sw(nc+1,na,2)
real::tslb(nc+1,n,2),smois(nc+1,n,2),sh2o(nc+1,n,2)
real::zs(n),dzs(n),depths(na)
integer::levels(4),mlevels(4),wlevels(1),c,k,u
mask=1.;sst=280.;st=0.;sm=0.;sw=0.;sh2o=0.
levels=[5,25,70,150];mlevels=levels;wlevels=0
depths=[0.,.05,.25,.7,1.5,3.]
do c=1,nc
 tsk(c,1)=278.123+float(c)/23.
 tmn(c,1)=275.375+float(c)/41.
 do k=1,4
  st(c,k+1,1)=246.125+float(mod(c*11+k*17,65))+float(c)/37.
  sm(c,k+1,1)=.05+float(mod(c*17+k*13,40))*.01
 enddo
enddo
call init_soil_depth_8(zs,dzs,n)
call init_soil_2_real(tsk,tmn,smois,sh2o,tslb,st,sm,sw,mask,sst,zs,dzs, &
 levels,mlevels,wlevels,n,4,4,0,na,na,na,0,1,0, &
 1,nc+1,1,2,1,2,1,nc+1,1,2,1,2,1,nc,1,1,1,1)
open(newunit=u,file='soil-columns.bin',access='stream',form='unformatted',status='replace')
write(u)nc,na,n
write(u)depths,zs
write(u)st(1:nc,1:na,1),sm(1:nc,1:na,1)
write(u)tslb(1:nc,1:n,1),smois(1:nc,1:n,1)
close(u)
end program
