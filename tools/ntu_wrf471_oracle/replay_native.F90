! Replay the actual MP_NTU arguments captured by capture.py. No NTU_INIT
! is called: the real forecast's initialized aerosol fields are inputs.
program replay_native
  use module_mp_ntu
  implicit none
  integer :: header(8), nx,ny,nz,nc,f,u,k,i
  real :: dt
  real, allocatable :: m(:,:,:,:),w(:,:,:),s(:,:,:)
  character(len=1024) :: pin,pout
  call get_command_argument(1,pin)
  call get_command_argument(2,pout)
  open(newunit=u,file=trim(pin),access='stream',form='unformatted',status='old',convert='little_endian')
  read(u) header
  if(header(1)/=int(z'4E545532')) error stop 1
  nx=header(2); ny=header(3); nz=header(4); nc=nx*ny
  read(u) dt
  allocate(m(nc,nz+1,1,32),w(nc,nz+1,1),s(nc,1,9))
  m=0.
  do f=1,32
    read(u) m(:,1:nz,1,f)
  enddo
  read(u) w
  read(u) s
  close(u)
  call aero_const(header(8))
  call mp_ntu(header(5),m(:,:,:,1),m(:,:,:,2),m(:,:,:,3),w,m(:,:,:,4),dt,s(:,:,9), &
       m(:,:,:,5),m(:,:,:,6),m(:,:,:,7),m(:,:,:,8),m(:,:,:,9),m(:,:,:,10),m(:,:,:,11), &
       m(:,:,:,12),m(:,:,:,13),m(:,:,:,14),m(:,:,:,15),m(:,:,:,16),m(:,:,:,17), &
       m(:,:,:,18),m(:,:,:,19),m(:,:,:,20),m(:,:,:,21),m(:,:,:,22), &
       m(:,:,:,23),m(:,:,:,24),m(:,:,:,25),m(:,:,:,26),m(:,:,:,27), &
       m(:,:,:,28),m(:,:,:,29),m(:,:,:,30),m(:,:,:,31),m(:,:,:,32), &
       s(:,:,1),s(:,:,2),s(:,:,3),s(:,:,4),s(:,:,5),s(:,:,6),s(:,:,7),s(:,:,8), &
       1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz)
  open(newunit=u,file=trim(pout),access='stream',form='unformatted',status='replace',convert='little_endian')
  write(u) header,dt
  write(u) m(:,1:nz,1,1)
  do f=5,32
    write(u) m(:,1:nz,1,f)
  enddo
  write(u) s
  close(u)
end program replay_native
