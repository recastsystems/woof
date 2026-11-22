! Calls the referee build's own radconst and calc_coszen (module_radiation_driver.o
! from the strict wrf.exe build, linked from its libwrflib.a) the way
! radiation_driver calls them:
!   radiation call: radconst(XTIME,DECLIN,SOLCON,JULIAN,DEGRAD,DPD), then
!                   calc_coszen(...,julian,xtime+radt*0.5,gmt,declin,degrad,...)
!   swint_opt call: the same, with xtime unshifted (coszen_loc).
! Input (native stream): int32 n, int32 ncal, real lat(n), real lon(n),
!                        ncal records of real (julian, xtime, gmt, radt).
! Output (native stream): per record: declin, solcon, coszen(n), hrang(n),
!                        coszen_loc(n), hrang_loc(n).
program native_solar_replay
  use module_model_constants, only: degrad, dpd
  use module_radiation_driver, only: radconst, calc_coszen
  implicit none
  integer :: n, ncal, k, u, v
  real, allocatable :: lat(:,:), lon(:,:), cz(:,:), ha(:,:), czl(:,:), hal(:,:)
  real :: cal(4), declin, solcon, declin2, solcon2
  character(len=4096) :: fin, fout
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) n, ncal
  allocate(lat(n,1), lon(n,1), cz(n,1), ha(n,1), czl(n,1), hal(n,1))
  read(u) lat, lon
  open(newunit=v, file=trim(fout), access='stream', form='unformatted', status='replace')
  do k = 1, ncal
     read(u) cal
     call radconst(cal(2), declin, solcon, cal(1), degrad, dpd)
     call calc_coszen(1, n, 1, 1, 1, n, 1, 1, cal(1), cal(2)+cal(4)*0.5, cal(3), &
                      declin, degrad, lon, lat, cz, ha)
     call radconst(cal(2), declin2, solcon2, cal(1), degrad, dpd)
     call calc_coszen(1, n, 1, 1, 1, n, 1, 1, cal(1), cal(2), cal(3), &
                      declin2, degrad, lon, lat, czl, hal)
     write(v) declin, solcon, cz, ha, czl, hal
  end do
  close(u)
  close(v)
end program native_solar_replay
