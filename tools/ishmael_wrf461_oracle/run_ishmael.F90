program run_ishmael_oracle
  ! Drive the public WRF ISHMAEL wrapper (MP_JENSEN_ISHMAEL) of WRF v4.6.1
  ! phys/module_mp_jensen_ishmael.F over a batch of columns for N steps.
  !
  ! Every output value written here was returned by MP_JENSEN_ISHMAEL; this
  ! program reads inputs written by make_columns.py and never reimplements
  ! scheme arithmetic.  Columns are laid side by side along i (j = 1), the
  ! scheme treats every (i, j) column independently.
  !
  ! usage: run_ishmael INPUTS.bin OUTPUTS.bin NSTEPS DT
  !
  ! The main program is compiled with WRF's BYTESWAPIO flags
  ! (-fconvert=big-endian -frecord-marker=4) so jensen_ishmael_init reads
  ! WRF's big-endian ishmael-*.bin tables exactly as wrf.exe does; this
  ! driver's own stream files name convert='little_endian' explicitly.
  use module_mp_jensen_ishmael, only: jensen_ishmael_init, mp_jensen_ishmael
  implicit none

  integer, parameter :: magic_in = 1229146957   ! 'ISHM'
  integer, parameter :: magic_out = 1229146959
  integer, parameter :: nin3 = 19, nout3 = 35, nout2 = 4

  character(len=1024) :: inpath, outpath, arg
  integer :: uin, uout, ncol, nz, nsteps, istep, magic, ios
  real :: dt

  real, allocatable, dimension(:,:,:) :: p, dz, th, qv, qc, qr, nr,        &
       qi1, ni1, ai1, ci1, qi2, ni2, ai2, ci2, qi3, ni3, ai3, ci3,         &
       effc, effi, dbz, vmi1, di1, rhopo1, phii1, vmi2, di2, rhopo2, phii2, &
       vmi3, di3, rhopo3, phii3, it1, it2, it3
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg)
  read(arg, *) nsteps
  call get_command_argument(4, arg)
  read(arg, *) dt
  if (storage_size(1.0) /= 32) error stop 3

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  read(uin) magic, ncol, nz
  if (magic /= magic_in) error stop 4

  allocate(p(ncol,nz,1), dz(ncol,nz,1), th(ncol,nz,1), qv(ncol,nz,1),     &
       qc(ncol,nz,1), qr(ncol,nz,1), nr(ncol,nz,1),                        &
       qi1(ncol,nz,1), ni1(ncol,nz,1), ai1(ncol,nz,1), ci1(ncol,nz,1),     &
       qi2(ncol,nz,1), ni2(ncol,nz,1), ai2(ncol,nz,1), ci2(ncol,nz,1),     &
       qi3(ncol,nz,1), ni3(ncol,nz,1), ai3(ncol,nz,1), ci3(ncol,nz,1),     &
       effc(ncol,nz,1), effi(ncol,nz,1), dbz(ncol,nz,1),                   &
       vmi1(ncol,nz,1), di1(ncol,nz,1), rhopo1(ncol,nz,1), phii1(ncol,nz,1), &
       vmi2(ncol,nz,1), di2(ncol,nz,1), rhopo2(ncol,nz,1), phii2(ncol,nz,1), &
       vmi3(ncol,nz,1), di3(ncol,nz,1), rhopo3(ncol,nz,1), phii3(ncol,nz,1), &
       it1(ncol,nz,1), it2(ncol,nz,1), it3(ncol,nz,1),                     &
       rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), snowncv(ncol,1))

  read(uin) p, dz, th, qv, qc, qr, nr, qi1, ni1, ai1, ci1,                &
       qi2, ni2, ai2, ci2, qi3, ni3, ai3, ci3
  read(uin) rainnc, snownc
  close(uin)
  rainncv = 0.; snowncv = 0.
  effc = 0.; effi = 0.; dbz = 0.
  vmi1 = 0.; di1 = 0.; rhopo1 = 0.; phii1 = 0.
  vmi2 = 0.; di2 = 0.; rhopo2 = 0.; phii2 = 0.
  vmi3 = 0.; di3 = 0.; rhopo3 = 0.; phii3 = 0.
  it1 = 0.; it2 = 0.; it3 = 0.

  call jensen_ishmael_init()

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', action='write', convert='little_endian', iostat=ios)
  if (ios /= 0) error stop 5
  write(uout) magic_out, ncol, nz, nsteps, dt

  do istep = 1, nsteps
     call mp_jensen_ishmael(istep, dt, p, dz,                              &
          th, qv, qc, qr, nr, qi1, ni1, ai1, ci1,                          &
          qi2, ni2, ai2, ci2,                                              &
          qi3, ni3, ai3, ci3,                                              &
          1, ncol, 1, 1, 1, nz,                                            &
          1, ncol, 1, 1, 1, nz,                                            &
          1, ncol, 1, 1, 1, nz,                                            &
          rainnc, rainncv, snownc, snowncv,                                &
          effc, effi, dbz,                                                 &
          vmi1, di1, rhopo1, phii1,                                        &
          vmi2, di2, rhopo2, phii2,                                        &
          vmi3, di3, rhopo3, phii3,                                        &
          it1, it2, it3)
     write(uout) th, qv, qc, qr, nr, qi1, ni1, ai1, ci1,                   &
          qi2, ni2, ai2, ci2, qi3, ni3, ai3, ci3,                          &
          effc, effi, dbz, vmi1, di1, rhopo1, phii1,                       &
          vmi2, di2, rhopo2, phii2, vmi3, di3, rhopo3, phii3,              &
          it1, it2, it3
     write(uout) rainnc, rainncv, snownc, snowncv
  end do
  close(uout)
end program run_ishmael_oracle
