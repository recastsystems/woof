! Replay one captured GFL call (PRE file) through the WRF v4.8.0 strict referee's own compiled
! objects (phys/module_cu_gfl_{common,deep,sh,wrf}.o, ccpp_kind_types.o) and write POST.
! usage: gfl_replay_native PRE POST
program gfl_replay_native
 use ccpp_kind_types, only: kind_phys
 use module_cu_gfl, only: cu_grell_freitas_li
 implicit none
 integer:: ims,ime,jms,jme,kms,kme,ids,ide,jds,jde,kds,kde,its,ite,jts,jte,kts,kte
 integer:: itimestep,ichoice_deep,ishallow_g3, has_thf,has_qvf,has_gdc,has_gdc2, unit
 real(kind=kind_phys):: dt,dxcell,xlv,cp,g,r_v
 real(kind=kind_phys),allocatable,dimension(:,:,:):: u,v,w,t,q,rho,p,pi,p8w,dz8w,rqvften,rthften, &
      rthblten,rqvblten,rthraten,gdc,gdc2,rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten
 real(kind=kind_phys),allocatable,dimension(:,:):: ht,xland,hfx,qfx,gsw,htop,hbot,xmb_shallow,raincv,pratec
 integer,allocatable,dimension(:,:):: kpbl,k22_shallow,kbcon_shallow,ktop_shallow
 character(len=1024):: pre, post

 call get_command_argument(1, pre); call get_command_argument(2, post)
 open(newunit=unit, file=trim(pre), access='stream', form='unformatted', status='old', convert='little_endian')
 read(unit) ims,ime,jms,jme,kms,kme,ids,ide,jds,jde,kds,kde,its,ite,jts,jte,kts,kte, itimestep,ichoice_deep,ishallow_g3
 read(unit) has_thf,has_qvf,has_gdc,has_gdc2
 read(unit) dt, dxcell, xlv, cp, g, r_v
 allocate(u(ims:ime,kms:kme,jms:jme),v,w,t,q,rho,p,pi,p8w,dz8w,rqvften,rthften,rthblten,rqvblten,rthraten, &
          gdc,gdc2,rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten, mold=u)
 allocate(ht(ims:ime,jms:jme)); allocate(xland,hfx,qfx,gsw,htop,hbot,xmb_shallow,raincv,pratec, mold=ht)
 allocate(kpbl(ims:ime,jms:jme)); allocate(k22_shallow,kbcon_shallow,ktop_shallow, mold=kpbl)
 read(unit) u,v,w,t,q,rho,p,pi,p8w,dz8w
 if (has_qvf == 1) read(unit) rqvften
 if (has_thf == 1) read(unit) rthften
 read(unit) rthblten,rqvblten,rthraten
 read(unit) ht,xland,hfx,qfx,gsw
 read(unit) kpbl
 read(unit) htop,hbot,xmb_shallow,raincv,pratec
 read(unit) k22_shallow,kbcon_shallow,ktop_shallow
 if (has_gdc == 1) read(unit) gdc
 if (has_gdc2 == 1) read(unit) gdc2
 read(unit) rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten
 close(unit)
 if (has_thf /= 1 .or. has_qvf /= 1 .or. has_gdc /= 1 .or. has_gdc2 /= 1) stop 'replay expects all optionals present'
 call cu_grell_freitas_li(itimestep=itimestep,dt=dt,dxcell=dxcell,u=u,v=v,w=w,t=t,q=q,rho=rho,p=p,pi=pi,  &
      p8w=p8w,dz8w=dz8w,ht=ht,xland=xland,hfx=hfx,qfx=qfx,gsw=gsw,rqvften=rqvften,rthften=rthften,     &
      rthblten=rthblten,rqvblten=rqvblten,rthraten=rthraten,kpbl=kpbl,xlv=xlv,cp=cp,g=g,r_v=r_v,        &
      ichoice_deep=ichoice_deep,ishallow_g3=ishallow_g3,htop=htop,hbot=hbot,k22_shallow=k22_shallow,    &
      kbcon_shallow=kbcon_shallow,ktop_shallow=ktop_shallow,xmb_shallow=xmb_shallow,raincv=raincv,      &
      pratec=pratec,gdc=gdc,gdc2=gdc2,rthcuten=rthcuten,rqvcuten=rqvcuten,rqccuten=rqccuten,            &
      rqicuten=rqicuten,rucuten=rucuten,rvcuten=rvcuten,                                                &
      ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme,ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde,  &
      its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)
 open(newunit=unit, file=trim(post), access='stream', form='unformatted', status='replace', convert='little_endian')
 write(unit) htop,hbot,xmb_shallow,raincv,pratec
 write(unit) k22_shallow,kbcon_shallow,ktop_shallow
 write(unit) gdc
 write(unit) gdc2
 write(unit) rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten
 close(unit)
end program gfl_replay_native
