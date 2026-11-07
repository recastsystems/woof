program run_helpers
 use module_mp_gsfcgce_4ice_nuwrf, only: consat_s,negcor,auto_conversion,refl10cm_gsfc,tnw,tns,tng,roqr,roqs,roqg,xnor,xnos,xnog
 use module_mp_radar
 implicit none
 integer,parameter:: nc=80,nz=40,na=160
 real::qv(nc,nz,1),qr(nc,nz,1),qs(nc,nz,1),qg(nc,nz,1),t(nc,nz,1),p(nc,nz,1),rho(nc,nz,1),dz(nc,nz,1),x(nc,nz,1),out(nc,nz,1)
 real::x2(40,nz,2),rho2(40,nz,2),dz2(40,nz,2)
 real::liquid(na),number(na),rate(na),radius(na),xp(nc,nz,1)
 double precision::init(269)
 integer::i,j,k,c,u,v,mode,itaobraun
 character(len=1024)::folder
 call get_command_argument(1,folder)
 itaobraun=0
 call consat_s(itaobraun)
 xnor=tnw*1.e8;xnos=tns*1.e8;xnog=tng*1.e8
 xam_r=3.14159*1000./6.;xbm_r=3.;xmu_r=0.
 xam_s=3.14159*100./6.;xbm_s=3.;xmu_s=0.
 xam_g=3.14159*(roqg*1000.)/6.;xbm_g=3.;xmu_g=0.
 call radar_init
 init(1:19)=[dble(xnor),dble(xnos),dble(xnog),dble(xam_r),dble(xam_s),dble(xam_g),dble(xcrg(3)),dble(xcrg(4)),dble(xocms),dble(xocmg),dble(xobms),dble(xobmg),pi5,lamda4,k_w,real(m_w_0,8),aimag(m_w_0),real(m_i_0,8),aimag(m_i_0)]
 init(20:69)=xxDs;init(70:119)=xdts;init(120:169)=xxDg;init(170:219)=xdtg;init(220:269)=simpson(1:50)
 open(newunit=u,file=trim(folder)//'/radar-init.expected',access='stream',form='unformatted',status='replace');write(u)init;close(u)
 do k=1,nz
  do i=1,nc
   qv(i,k,1)=0.012*real(nz-k)/real(nz)
   t(i,k,1)=295.-1.7*real(k)+0.05*real(mod(i,7))
   p(i,k,1)=100000.-1800.*real(k)
   rho(i,k,1)=1.2-0.02*real(k)
   dz(i,k,1)=80.+11.*real(mod(i+k,19))
   qr(i,k,1)=real(mod(i+2*k,17))*0.00004
   qs(i,k,1)=real(mod(i+k,11))*0.00002
   qg(i,k,1)=real(mod(2*i+k,13))*0.000015
   if(i<=10)then
    qr(i,k,1)=0.;qs(i,k,1)=0.;qg(i,k,1)=0.
   elseif(i<=20)then
    t(i,k,1)=250.+0.1*real(k)
   elseif(i<=30)then
    qs(i,k,1)=0.;qg(i,k,1)=0.
   elseif(i<=40)then
    qr(i,k,1)=0.;t(i,k,1)=260.
   endif
   if(i==nc)then
    qr(i,k,1)=1.e-9;qs(i,k,1)=nearest(1.e-9,1.);qg(i,k,1)=nearest(1.e-9,-1.)
   endif
  enddo
 enddo
 open(newunit=u,file=trim(folder)//'/radar.input',access='stream',form='unformatted',status='replace');write(u)qv,qr,qs,qg,t,p;close(u)
 do i=1,nc
  call refl10cm_gsfc(qv(i,:,1),qr(i,:,1),qs(i,:,1),qg(i,:,1),t(i,:,1),p(i,:,1),out(i,:,1),1,nz,i,1)
 enddo
 open(newunit=u,file=trim(folder)//'/radar.expected',access='stream',form='unformatted',status='replace');write(u)out;close(u)
 do i=1,na
  liquid(i)=1.e-8*10.**(real(mod(i,32))/12.)
  number(i)=10.+37.25*real(i-1)
 enddo
 liquid(1)=0.;liquid(2)=1.e-33;liquid(3)=1.e-32;liquid(4)=nearest(1.e-32,1.)
 number(5)=0.;number(6)=-1.;number(7)=76.;number(8)=77.;number(9)=1000.;number(10)=10000.
 do i=1,na
  radius(i)=-999.
  call auto_conversion(liquid(i),number(i),rate(i),radius(i))
 enddo
 open(newunit=u,file=trim(folder)//'/auto.input',access='stream',form='unformatted',status='replace');write(u)liquid,number;close(u)
 open(newunit=u,file=trim(folder)//'/auto.expected',access='stream',form='unformatted',status='replace');write(u)rate,radius;close(u)
 do mode=1,3
  do k=1,nz
   do i=1,nc
    x(i,k,1)=real(mod(i+3*k,19)-3)*0.0001
    if(mode==2)x(i,k,1)=-abs(x(i,k,1))
    if(mode==3)x(i,k,1)=0.
   enddo
  enddo
  open(newunit=u,file=trim(folder)//'/neg'//achar(48+mode)//'.input',access='stream',form='unformatted',status='replace');write(u)x,rho,dz;close(u)
  call negcor(x,rho,dz,1,nc,1,1,1,nz,1,1,1,nc,1,1,1,nz)
  open(newunit=u,file=trim(folder)//'/neg'//achar(48+mode)//'.expected',access='stream',form='unformatted',status='replace');write(u)x;close(u)
 enddo
 do mode=4,5
  do j=1,2
   do k=1,nz
    do i=1,40
     x2(i,k,j)=real(mod(i+3*k+2*j,19)-3)*0.0001
     rho2(i,k,j)=1.2-0.02*real(k)
     dz2(i,k,j)=80.+11.*real(mod(i+k+j,19))
    enddo
   enddo
  enddo
  open(newunit=u,file=trim(folder)//'/neg'//achar(48+mode)//'.input',access='stream',form='unformatted',status='replace');write(u)x2,rho2,dz2;close(u)
  if(mode==4)then
   call negcor(x2,rho2,dz2,1,40,1,2,1,nz,1,1,5,36,1,2,2,39)
  else
   call negcor(x2,rho2,dz2,1,40,1,2,1,nz,1,1,5,4,1,2,2,39)
  endif
  open(newunit=u,file=trim(folder)//'/neg'//achar(48+mode)//'.expected',access='stream',form='unformatted',status='replace');write(u)x2;close(u)
 enddo
 print *, '80 radar columns, 40 levels, 160 auto cases, five negcor tiles, including 2-D subset and empty tiles'
end program
