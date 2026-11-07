program pow_fold_probe
  ! How gfortran lowers a REAL exponent of 2.0, at the optimisation level it
  ! is built with.  MFSHCONVPBL spells ZKIC**2. and (1.-ZKIC)**2.
  ! (module_bl_mfshconvpbl.F:985-986, :1001-1002).  Two million arguments
  ! in [0, 3): count where x**2. and x*x disagree in any bit.  build.sh runs
  ! it at -O0 and at WRF's -O2 flags and records both counts.
  implicit none
  real :: x, a, b, c, d
  integer :: i, n2, n3
  integer(8) :: s
  n2 = 0; n3 = 0
  s = 12345
  do i = 1, 2000000
    s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
    x = real(s) / 2147483648.0 * 3.0
    a = x**2.
    b = x*x
    if (transfer(a, 0) /= transfer(b, 0)) n2 = n2 + 1
    c = (1.-x)**2.
    d = (1.-x)*(1.-x)
    if (transfer(c, 0) /= transfer(d, 0)) n3 = n3 + 1
  end do
  write(*, '(A,I0,A,I0,A)') 'x**2. /= x*x at ', n2, &
       ' of 2000000 arguments; (1.-x)**2. /= (1.-x)*(1.-x) at ', n3, ''
end program pow_fold_probe
