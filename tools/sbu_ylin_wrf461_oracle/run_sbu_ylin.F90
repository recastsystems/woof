! Column driver for the standalone SBU-YLin (mp_physics = 13) oracle.
!
! Reads the hex-word case file written by make_cases.py, calls WRF's own
! public SBU_YLIN wrapper once per column (one column per call, because the
! long-dt edge cases carry their own DT_IN), and writes every output word as
! an 8-digit hex IEEE pattern so nothing is lost to decimal formatting.
!
! Input file layout (all values are float32 bit patterns in hex):
!   line 1:            ncol nz
!   per column:        dt ht rainnc0
!                      nz lines: th qv ql qr qi qs rho pii p z dz8w
! Output file layout:
!   line 1:            ncol nz
!   per column:        rainnc rainncv
!                      nz lines: th qv ql qr qi qs rimi
program run_sbu_ylin
  use module_mp_sbu_ylin, only: sbu_ylin
  implicit none
  integer :: ncol, nz, icol, k, nin, nout
  character(len=512) :: inpath, outpath
  integer(4) :: w(11), s(3)
  real, allocatable :: th(:,:,:), qv(:,:,:), ql(:,:,:), qr(:,:,:)
  real, allocatable :: qi(:,:,:), qs(:,:,:), ri3d(:,:,:)
  real, allocatable :: rho(:,:,:), pii(:,:,:), p(:,:,:), z(:,:,:)
  real, allocatable :: dz8w(:,:,:)
  real :: ht(1,1), rainnc(1,1), rainncv(1,1), dt

  if (command_argument_count() /= 2) then
    write(0,*) 'usage: run_sbu_ylin INPUT OUTPUT'
    stop 2
  end if
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)

  open(newunit=nin, file=trim(inpath), status='old', action='read')
  open(newunit=nout, file=trim(outpath), status='replace', action='write')
  read(nin, *) ncol, nz
  write(nout, '(I0,1X,I0)') ncol, nz

  allocate(th(1,nz,1), qv(1,nz,1), ql(1,nz,1), qr(1,nz,1), qi(1,nz,1))
  allocate(qs(1,nz,1), ri3d(1,nz,1), rho(1,nz,1), pii(1,nz,1))
  allocate(p(1,nz,1), z(1,nz,1), dz8w(1,nz,1))

  do icol = 1, ncol
    read(nin, '(3(Z8,1X))') s
    dt = transfer(s(1), dt)
    ht(1,1) = transfer(s(2), dt)
    rainnc(1,1) = transfer(s(3), dt)
    ! RAINNCV is INTENT(INOUT) and fully overwritten; seed a sentinel so an
    ! unwritten word would show.
    rainncv(1,1) = -999.0
    do k = 1, nz
      read(nin, '(11(Z8,1X))') w
      th(1,k,1)   = transfer(w(1), dt)
      qv(1,k,1)   = transfer(w(2), dt)
      ql(1,k,1)   = transfer(w(3), dt)
      qr(1,k,1)   = transfer(w(4), dt)
      qi(1,k,1)   = transfer(w(5), dt)
      qs(1,k,1)   = transfer(w(6), dt)
      rho(1,k,1)  = transfer(w(7), dt)
      pii(1,k,1)  = transfer(w(8), dt)
      p(1,k,1)    = transfer(w(9), dt)
      z(1,k,1)    = transfer(w(10), dt)
      dz8w(1,k,1) = transfer(w(11), dt)
      ! RI3D is written for every level and never read by the scheme.
      ri3d(1,k,1) = -999.0
    end do

    call sbu_ylin(th, qv, ql, qr, qi, qs, ri3d, rho, pii, p, dt,     &
                  z, ht, dz8w, rainnc, rainncv,                      &
                  1, 1, 1, 1, 1, nz + 1,                             &
                  1, 1, 1, 1, 1, nz,                                 &
                  1, 1, 1, 1, 1, nz)

    write(nout, '(2(Z8.8,1X))') transfer(rainnc(1,1), 0_4),          &
                                transfer(rainncv(1,1), 0_4)
    do k = 1, nz
      write(nout, '(7(Z8.8,1X))') transfer(th(1,k,1), 0_4),          &
          transfer(qv(1,k,1), 0_4), transfer(ql(1,k,1), 0_4),        &
          transfer(qr(1,k,1), 0_4), transfer(qi(1,k,1), 0_4),        &
          transfer(qs(1,k,1), 0_4), transfer(ri3d(1,k,1), 0_4)
    end do
  end do
  close(nin)
  close(nout)
end program run_sbu_ylin
