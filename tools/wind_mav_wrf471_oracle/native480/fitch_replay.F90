! Module replay driver for the native WRF Fitch wind-farm module.
! It links the module_wind_fitch object exactly as the pinned strict referee build compiled it
! (taken from that build's libwrflib.a; this file does no arithmetic of its own) and feeds it the
! real inputs captured at the first dragforce call of a whole native forecast
! (tools/wind_mav_wrf471_oracle/capture.patch stream layout).
!   W480 defined: WRF v4.8.0 module (THRUST formal, induction_correction module variable)
!   W480 undefined: WRF v4.7.1 module (no THRUST, no induction)
! usage: fitch_replay IN OUT [induction 0|1]
program fitch_replay
  use module_wind_fitch
  use module_driver_constants, only : max_domains
  implicit none
  integer :: nx, ny, nz, ntc, ind_cap, ind, uin, uout
  real :: dxv, dtv, cf
  integer, allocatable :: iv(:), jv(:)
  real, allocatable, dimension(:,:,:) :: u, v, dz, zw, du, dv, qke
  real, allocatable, dimension(:,:) :: power, thrust
  character(len=512) :: fin, fout, arg
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', status='old')
  read(uin) nx, ny, nz, ntc, ind_cap
  read(uin) dxv, dtv, cf
  nt = ntc
  allocate(nkind(nt), ival(nt,max_domains), jval(nt,max_domains), iv(nt), jv(nt))
  allocate(hubheight(nt), stc(nt), stc2(nt), cutin(nt), cutout(nt), diameter(nt), npower(nt))
  ival = -9999
  jval = -9999
  read(uin) diameter, hubheight, iv, jv, nkind, stc, stc2, cutin, cutout, npower
  ival(:,1) = iv
  jval(:,1) = jv
  read(uin) turbws, turbtc, turbpw
  correction_factor = cf
  ind = ind_cap
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg)
    read(arg, *) ind
  end if
#ifdef W480
  induction_correction = ind
#else
  if (ind /= 0) stop 'the v4.7.1 module has no induction correction'
#endif
  allocate(u(nx+1,nz+1,ny+1), v(nx+1,nz+1,ny+1), dz(nx+1,nz+1,ny+1), zw(nx+1,nz+1,ny+1))
  allocate(du(nx+1,nz+1,ny+1), dv(nx+1,nz+1,ny+1), qke(nx+1,nz+1,ny+1))
  allocate(power(nx+1,ny+1), thrust(nx+1,ny+1))
  u = 0.; v = 0.; dz = 0.; zw = 0.; du = 0.; dv = 0.; qke = 0.; power = 0.; thrust = 0.
  read(uin) u(1:nx,1:nz,1:ny), v(1:nx,1:nz,1:ny)
  read(uin) dz(1:nx,1:nz,1:ny), zw(1:nx,1:nz+1,1:ny)
  read(uin) du(1:nx,1:nz,1:ny), dv(1:nx,1:nz,1:ny), qke(1:nx,1:nz,1:ny)
  close(uin)
  ! Serial single tile, as the capture: ids=ims=its=1, ide=ime=ite=n+1, so itf=nx, jtf=ny, ktf=nz.
  call dragforce(1, zw, u, v, dxv, dz, dtv, qke, du, dv, 1, power,      &
#ifdef W480
                 thrust,                                              &
#endif
                 1, nx+1, 1, ny+1, 1, nz+1,                            &
                 1, nx+1, 1, ny+1, 1, nz+1,                            &
                 1, nx+1, 1, ny+1, 1, nz+1)
  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(uout) du(1:nx,1:nz,1:ny), dv(1:nx,1:nz,1:ny), qke(1:nx,1:nz,1:ny), power(1:nx,1:ny), thrust(1:nx,1:ny)
  close(uout)
  print '(a,5i6,a,i2)', 'fitch_replay nx ny nz nt ind_cap', nx, ny, nz, nt, ind_cap, ' induction', ind
end program fitch_replay
