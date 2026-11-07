! Input is a kept real shallow-driver capture. The witness-only module adds
! a seed before each native nscv2d call. Native nscv2d remains byte unchanged.
! Two seeds show exactly which INTENT(OUT) diagnostic words it leaves unwritten.
program witness
 use module_shcu_nscv
 implicit none
 integer :: header(8),mp,pbl,nx,ny,nz,u,v,ios
 real :: c(16)
 real,allocatable :: znu(:),a(:,:,:,:),s(:,:,:),w(:,:,:,:),o(:,:,:,:),d(:,:,:)
 character(1024) :: input,output,seed
 call get_command_argument(1,input)
 call get_command_argument(2,output)
 call get_command_argument(3,seed)
 read(seed,*) witness_seed
 open(newunit=u,file=trim(input),access='stream',form='unformatted',convert='little_endian')
 read(u)header,c,mp,pbl
 nx=header(4)-header(3)+1;ny=header(6)-header(5)+1;nz=header(8)-header(7)+1
 allocate(znu(nz),a(nx,nz+1,ny,10),s(nx,ny,5),w(nx,nz+1,ny,3),o(nx,nz+1,ny,6),d(nx,ny,3))
 a=0.;o=0.;d=0.
 read(u)znu
 do v=1,10
  read(u)a(:,1:nz,:,v)
 enddo
 do v=1,3
  read(u)w(:,:,:,v)
 enddo
 do v=1,5
  read(u)s(:,:,v)
 enddo
 close(u)
 call shcu_nscv(dt=c(1),p3di=w(:,:,:,1),p3d=a(:,:,:,5),pi3d=a(:,:,:,6), &
   qc3d=a(:,:,:,3),qi3d=a(:,:,:,4),rho3d=a(:,:,:,7),qv3d=a(:,:,:,2),t3d=a(:,:,:,1), &
   raincv=s(:,:,1),xland=s(:,:,2),dz8w=a(:,:,:,10),w=w(:,:,:,3),u3d=a(:,:,:,8),v3d=a(:,:,:,9), &
   hpbl=s(:,:,3),hfx=s(:,:,4),qfx=s(:,:,5),mp_physics=mp,pgcon=.55, &
   cp=c(9),cliq=c(10),cpv=c(11),g=c(3),xlv=c(4),r_d=c(5),r_v=c(6),ep_1=c(12),ep_2=c(13), &
   cice=c(14),xls=c(15),psat=c(16),f_qi=.true.,f_qc=.true., &
   rthshten=o(:,:,:,1),rqvshten=o(:,:,:,2),rqcshten=o(:,:,:,3),rqishten=o(:,:,:,4), &
   rushten=o(:,:,:,5),rvshten=o(:,:,:,6),pratesh=d(:,:,1),hbot=d(:,:,2),htop=d(:,:,3), &
   ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1, &
   ims=1,ime=nx,jms=1,jme=ny,kms=1,kme=nz+1, &
   its=1,ite=nx,jts=1,jte=ny,kts=1,kte=nz)
 open(newunit=u,file=trim(output),access='stream',form='unformatted',convert='little_endian',status='replace')
 do v=1,6
  write(u)o(:,1:nz,:,v)
 enddo
 write(u)d
 close(u)
end program
