program run_gfdleta
 use iso_fortran_env, only: int32
 use module_mp_etanew, only: FPVS
 use module_configure
 use module_ra_gfdleta
 implicit none
 integer, parameter :: nx=8
 integer :: n,arm,i,k,j,u,v,a,b,c
 character(64) :: arg,prefix
 type(grid_config_rec_type) :: cfg
 real, allocatable :: sf(:),sh(:),p(:,:,:),pw(:,:,:),t(:,:,:),pi3(:,:,:),dz(:,:,:),rho(:,:,:)
 real, allocatable :: qv(:,:,:),qw(:,:,:),qi(:,:,:),qs(:,:,:),rt(:,:,:),lw(:,:,:),sw(:,:,:),cf(:,:,:)
 real :: em(nx,1),land(nx,1),skin(nx,1),alb(nx,1),snow(nx,1),veg(nx,1)
 real :: lat(nx,1),lon(nx,1),ht(nx,1),hb(nx,1),htr(nx,1),hbr(nx,1),cu(nx,1)
 real :: glw(nx,1),gsw(nx,1),down(nx,1),clear(nx,1),toa_l(nx,1),toa_s(nx,1),cz(nx,1)
 real :: cl(nx,1),cm(nx,1),ch(nx,1),acs(nx,1),acc(nx,1),sig,x,pm
 integer :: ns(nx,1),nc(nx,1)
 call get_command_argument(1,arg); read(arg,*) n
 call get_command_argument(2,arg); read(arg,*) arm
 call get_command_argument(3,prefix)
 if (n < 8 .or. n > 80 .or. (arm/=0 .and. arm/=1)) error stop 'bad oracle arguments'
 allocate(sf(n+1),sh(n+1),p(nx,n+1,1),pw(nx,n+1,1),t(nx,n+1,1),pi3(nx,n+1,1), &
 dz(nx,n+1,1),rho(nx,n+1,1),qv(nx,n+1,1),qw(nx,n+1,1),qi(nx,n+1,1),qs(nx,n+1,1), &
 rt(nx,n+1,1),lw(nx,n+1,1),sw(nx,n+1,1),cf(nx,n+1,1))
 do k=1,n+1
  sf(k)=(1.-real(k-1)/real(n))**1.5
 enddo
 do k=1,n
  sh(k)=0.5*(sf(k)+sf(k+1))
 enddo
 sh(n+1)=0.
 cfg%co2tf=arm
 call GFDLETAINIT(em,sf,sh,1.,2001,3,21,12.,cfg,.true., &
 1,nx+1,1,2,1,n+1, 1,nx,1,1,1,n+1, 1,nx,1,1,1,n)
 if(arm==1)call export_co2
 open(newunit=u,file=trim(prefix)//'-tables.csv',status='replace')
 write(u,'(A)')'field,index,bits'
 call tab('STEMP',stemp); call tab('GTEMP',gtemp)
 call tab('CO251',reshape(co251,[size(co251)])); call tab('CDT51',reshape(cdt51,[size(cdt51)]))
 call tab('C2D51',reshape(c2d51,[size(c2d51)])); call tab('CO258',reshape(co258,[size(co258)]))
 call tab('CDT58',reshape(cdt58,[size(cdt58)])); call tab('C2D58',reshape(c2d58,[size(c2d58)]))
 call tab('CO2M51',co2m51); call tab('CDTM51',cdtm51); call tab('C2DM51',c2dm51)
 call tab('CO2M58',co2m58); call tab('CDTM58',cdtm58); call tab('C2DM58',c2dm58)
 call tab('CO231',co231); call tab('CDT31',cdt31); call tab('C2D31',c2d31)
 call tab('CO238',co238); call tab('CDT38',cdt38); call tab('C2D38',c2d38)
 call tab('CO271',co271); call tab('CDT71',cdt71); call tab('C2D71',c2d71)
 call tab('CO278',co278); call tab('CDT78',cdt78); call tab('C2D78',c2d78)
 call tab('EM1',reshape(em1,[size(em1)])); call tab('EM1WDE',reshape(em1wde,[size(em1wde)]))
 call tab('EM3',reshape(em3,[size(em3)])); call tab('TABLE1',reshape(table1,[size(table1)]))
 call tab('TABLE2',reshape(table2,[size(table2)])); call tab('TABLE3',reshape(table3,[size(table3)]))
 call tab('SOURCE',reshape(source,[size(source)])); call tab('DSRCE',reshape(dsrce,[size(dsrce)]))
 call tab('XDUO3N',reshape(xduo3n,[size(xduo3n)])); call tab('XDO3N2',reshape(xdo3n2,[size(xdo3n2)]))
 call tab('XDO3N3',reshape(xdo3n3,[size(xdo3n3)])); call tab('XDO3N4',reshape(xdo3n4,[size(xdo3n4)]))
 call tab('PRGFDL',prgfdl)
 call tab('AXSD',[(gausin(real(k)*0.01),k=1,310)])
 call tab('SOLARD',[r1])
 call tab('TABLE_AUX',[ab15wd,skc1r,sko3r,sko2d])
 close(u)
 ! CO2PTZ/SIGP mutate the sigma work arrays under CO2TF=1.
 ! The fixture atmosphere is independent of those initialization work arrays.
 do k=1,n+1
  sf(k)=(1.-real(k-1)/real(n))**1.5
 enddo
 do k=1,n
  sh(k)=0.5*(sf(k)+sf(k+1))
 enddo
 sh(n+1)=0.
 t=220.; pi3=1.; dz=100.;rho=1.;qv=0.;qw=0.;qi=0.;qs=0.
 rt=0.;lw=0.;sw=0.;cf=0.
 land=1.;skin=290.;alb=0.15;snow=0.;veg=50.
 lat=0.;lon=0.;ht=0.;hb=real(n+1);htr=0.;hbr=0.;cu=0.
 glw=0.;gsw=0.;down=0.;clear=0.;toa_l=0.;toa_s=0.;cz=0.
 cl=0.;cm=0.;ch=0.;acs=0.;acc=0.;ns=0;nc=0
 ! Profiles 1-8: clear, low cloud, mid cloud, high cloud, convective,
 ! snow, water night, clear low sun. Angles are radians as in the WRF driver.
 land(7,1)=2.;lon(7,1)=3.1415926;lon(8,1)=1.48
 snow(6,1)=0.2;skin(6,1)=265.;alb(6,1)=0.7
 ht(5,1)=real(3*n/4);hb(5,1)=real(n/4);cu(5,1)=0.002
 do i=1,nx
  do k=1,n+1
   pw(i,k,1)=1000.+100000.*sf(k)
   pm=1000.+100000.*sh(k)
   p(i,k,1)=pm
   pi3(i,k,1)=(pm/100000.)**(287./1004.)
   t(i,k,1)=max(215.,skin(i,1)-65.*(1.-sh(k)))
   qv(i,k,1)=0.008*sh(k)**3
   rho(i,k,1)=pm/(287.*t(i,k,1))
   if(k<=n)then
    if(i==2.and.pm>75000..and.pm<90000.)qw(i,k,1)=0.0005
    if(i==3.and.pm>40000..and.pm<60000.)qw(i,k,1)=0.0004
    if(i==4.and.pm>15000..and.pm<30000.)qi(i,k,1)=0.0002
    if(i==6.and.pm>50000..and.pm<75000.)qs(i,k,1)=0.0002
   endif
   if(qw(i,k,1)+qi(i,k,1)+qs(i,k,1)>0.)qv(i,k,1)=0.622*1000.*FPVS(t(i,k,1))/(pm-1000.*FPVS(t(i,k,1)))
  enddo
 enddo
 ! WRF calls ETARA first for LW with GFDL_SW false, then for SW with both true.
 call ETARA(30.,rt,lw,sw,cf,pi3,land,pw,dz,rho,p,t,qv,qw,qi,qs,skin,glw,down,gsw,clear, &
 toa_s,toa_l,cz,lat,lon,ht,hb,htr,hbr,alb,cu,veg,snow,9.81,12.,20,1,1,0.,80.,2001,80,.true.,.false., &
 cl,cm,ch,acs,ns,acc,nc, 1,nx+1,1,2,1,n+1, 1,nx,1,1,1,n+1, 1,nx,1,1,1,n)
 call ETARA(30.,rt,lw,sw,cf,pi3,land,pw,dz,rho,p,t,qv,qw,qi,qs,skin,glw,down,gsw,clear, &
 toa_s,toa_l,cz,lat,lon,ht,hb,htr,hbr,alb,cu,veg,snow,9.81,12.,20,1,1,0.,80.,2001,80,.true.,.true., &
 cl,cm,ch,acs,ns,acc,nc, 1,nx+1,1,2,1,n+1, 1,nx,1,1,1,n+1, 1,nx,1,1,1,n)
 open(newunit=u,file=trim(prefix)//'-fields.csv',status='replace')
 write(u,'(A)')'column,level,field,bits'
 do i=1,nx
  do k=1,n
   call row('RTHRATEN',rt(i,k,1));call row('RTHRATENLW',lw(i,k,1));call row('RTHRATENSW',sw(i,k,1))
   call row('CLDFRA',cf(i,k,1))
  enddo
  k=0
  call row('GLW',glw(i,1));call row('GSW',gsw(i,1));call row('SWDOWN',down(i,1));call row('SWDOWNC',clear(i,1))
  call row('RLWTOA',toa_l(i,1));call row('RSWTOA',toa_s(i,1))
  call row('CFRACL',cl(i,1));call row('CFRACM',cm(i,1));call row('CFRACH',ch(i,1))
  call row('CZMEAN',cz(i,1));call row('ACFRST',acs(i,1));call row('ACFRCV',acc(i,1))
  call row('HTOPR',htr(i,1));call row('HBOTR',hbr(i,1));call row('CUPPT',cu(i,1))
 enddo
 close(u)
contains
 subroutine tab(name,values)
  character(*)::name
  real::values(:)
  integer::q
  do q=1,size(values)
   write(u,'(A,",",I0,",",Z8.8)')name,q,transfer(values(q),0_int32)
  enddo
 end subroutine
 subroutine row(name,value)
  character(*)::name
  real::value
  write(u,'(I0,",",I0,",",A,",",Z8.8)')i,k,name,transfer(value,0_int32)
 end subroutine
 subroutine export_co2
  integer::o,x,y
  open(newunit=o,file='co2_trans',form='unformatted',status='replace')
  write(o)stemp;write(o)gtemp
  write(o)cdtm51;write(o)co2m51;write(o)c2dm51;write(o)cdtm58;write(o)co2m58;write(o)c2dm58
  write(o)((cdt51(x,y),y=1,n+1),x=1,n+1)
  write(o)((co251(x,y),y=1,n+1),x=1,n+1)
  write(o)((c2d51(x,y),y=1,n+1),x=1,n+1)
  write(o)((cdt58(x,y),y=1,n+1),x=1,n+1)
  write(o)((co258(x,y),y=1,n+1),x=1,n+1)
  write(o)((c2d58(x,y),y=1,n+1),x=1,n+1)
  write(o)cdt31;write(o)co231;write(o)c2d31;write(o)cdt38;write(o)co238;write(o)c2d38
  write(o)cdt71;write(o)co271;write(o)c2d71;write(o)cdt78;write(o)co278;write(o)c2d78
  close(o)
 end subroutine
end program
