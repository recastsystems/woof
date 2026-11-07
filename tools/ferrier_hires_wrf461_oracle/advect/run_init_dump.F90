! For each time step in the oracle sets (6, 20, 36, 72, 90, 120, 150, 180,
! 300 s): fer_hires_init (GSMDT=0, so DTPH=DT), dump ("init"), one
! FER_HIRES_ADVECT call on a dry one-level column (which recomputes CIACW,
! CIACR, CRACW, BRAUT and MY_GROWTH from DT), dump ("step").
! Compile the main program with -fconvert=big-endian.
program run_init_dump
  use module_mp_fer_hires, only: fer_hires_init, fer_hires_advect, fer_dump_state
  implicit none
  real, parameter :: dts(9) = (/ 6., 20., 36., 72., 90., 120., 150., 180., 300. /)
  integer :: n, iu
  character(len=16) :: tag
  real :: p(1,2,1), pi(1,2,1), th(1,2,1), qv(1,2,1), qc(1,2,1), qr(1,2,1), &
       qi(1,2,1), qrimef(1,2,1), dz(1,2,1), rho(1,2,1)
  real :: rainnc(1,1), rainncv(1,1), sr(1,1)
  integer :: lowlyr(1,1)
  open(newunit=iu, file='init_dump.csv', status='replace', action='write')
  write(iu, '(A)') 'tag,name,index,hex'
  do n = 1, size(dts)
     call fer_hires_init(0.0, dts(n), 3000., 3000., lowlyr, .false., .true., &
          1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1)
     write(tag, '(A,I0)') 'init', nint(dts(n))
     call fer_dump_state(iu, trim(tag))
     p = 50000.; pi = 0.8; th = 300.; qv = 1.e-6; qc = 0.; qr = 0.; qi = 0.
     qrimef = 0.; dz = 100.; rho = 1.; rainnc = 0.; rainncv = 0.
     call fer_hires_advect(1, dts(n), 3000., 3000., 1, rainnc, rainncv, &
          dz, rho, p, pi, th, qv, lowlyr, sr, qc, qr, qi, qrimef, &
          1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1)
     write(tag, '(A,I0)') 'step', nint(dts(n))
     call fer_dump_state(iu, trim(tag))
  end do
  close(iu)
end program run_init_dump
