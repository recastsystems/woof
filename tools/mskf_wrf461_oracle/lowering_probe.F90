! How gfortran lowers the operations MSKF leans on, measured at WRF's own
! scalar-referee flags (build.sh compiles this with wrf_fflags), so the port
! spells each one the way the reference executes it rather than the way it
! reads:
!   * constant integer powers x**n, n = 2..6, single and double precision:
!     which multiplication chain reproduces gfortran's word on every probe;
!   * MAX/MIN/AMAX1/AMIN1 on signed zeros, which a port's fmaxf would decide
!     differently if gfortran's lowering does not follow IEEE maxNum;
!   * NINT at halfway points.
! Inputs come from a runtime array so nothing is constant-folded.
program lowering_probe
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: n = 20000
  real :: xs(n), x, x2, x3, x4, got
  real(r8) :: xd(n), y, y2, y3, y4, gotd
  integer :: i, miss(12)
  real :: a, b
  real(r8) :: ad, bd
  integer :: seed

  seed = 12345
  do i = 1, n
    seed = mod(seed * 1103515245 + 12345, 2147483647)
    xs(i) = 1.0e-6 + abs(real(seed)) / 2147483647.0 * 3.0e2
    xd(i) = 1.0e-12_r8 + abs(real(seed, r8)) / 2147483647.0_r8 * 7.3e6_r8
  end do
  if (command_argument_count() > 99) xs(1) = 0.  ! defeat whole-array folding

  miss = 0
  do i = 1, n
    x = xs(i); x2 = x*x; x3 = x2*x; x4 = x2*x2
    if (bits(x**3) /= bits(x2*x)) miss(1) = miss(1) + 1
    if (bits(x**4) /= bits(x2*x2)) miss(2) = miss(2) + 1
    if (bits(x**4) /= bits(x3*x)) miss(3) = miss(3) + 1
    if (bits(x**5) /= bits(x3*x2)) miss(4) = miss(4) + 1
    if (bits(x**5) /= bits(x4*x)) miss(5) = miss(5) + 1
    y = xd(i); y2 = y*y; y3 = y2*y; y4 = y2*y2
    if (bitsd(y**3) /= bitsd(y2*y)) miss(6) = miss(6) + 1
    if (bitsd(y**4) /= bitsd(y2*y2)) miss(7) = miss(7) + 1
    if (bitsd(y**5) /= bitsd(y3*y2)) miss(8) = miss(8) + 1
    if (bitsd(y**5) /= bitsd(y4*y)) miss(9) = miss(9) + 1
    if (bitsd(y**6) /= bitsd(y3*y3)) miss(10) = miss(10) + 1
    if (bitsd(y**6) /= bitsd(y4*y2)) miss(11) = miss(11) + 1
    if (bitsd(y**2) /= bitsd(y*y)) miss(12) = miss(12) + 1
  end do
  write(*, '(A,I0,A)') 'integer powers over ', n, ' probes (misses):'
  write(*, '(A,I0)') '  r4 x**3 vs x2*x         ', miss(1)
  write(*, '(A,I0)') '  r4 x**4 vs x2*x2        ', miss(2)
  write(*, '(A,I0)') '  r4 x**4 vs (x2*x)*x     ', miss(3)
  write(*, '(A,I0)') '  r4 x**5 vs x3*x2        ', miss(4)
  write(*, '(A,I0)') '  r4 x**5 vs x4*x         ', miss(5)
  write(*, '(A,I0)') '  r8 y**3 vs y2*y         ', miss(6)
  write(*, '(A,I0)') '  r8 y**4 vs y2*y2        ', miss(7)
  write(*, '(A,I0)') '  r8 y**5 vs y3*y2        ', miss(8)
  write(*, '(A,I0)') '  r8 y**5 vs y4*y         ', miss(9)
  write(*, '(A,I0)') '  r8 y**6 vs y3*y3        ', miss(10)
  write(*, '(A,I0)') '  r8 y**6 vs y4*y2        ', miss(11)
  write(*, '(A,I0)') '  r8 y**2 vs y*y          ', miss(12)

  ! signed zeros through MAX/MIN (operands from memory)
  xs(1) = 0.0; xs(2) = -0.0
  a = xs(1); b = xs(2)
  write(*, '(A,Z8.8)') 'MAX( 0.,-0.) = ', transfer(max(a, b), 0)
  write(*, '(A,Z8.8)') 'MAX(-0., 0.) = ', transfer(max(b, a), 0)
  write(*, '(A,Z8.8)') 'MIN( 0.,-0.) = ', transfer(min(a, b), 0)
  write(*, '(A,Z8.8)') 'MIN(-0., 0.) = ', transfer(min(b, a), 0)
  write(*, '(A,Z8.8)') 'AMAX1(0.,-0.) = ', transfer(amax1(a, b), 0)
  write(*, '(A,Z8.8)') 'AMAX1(-0.,0.) = ', transfer(amax1(b, a), 0)
  write(*, '(A,Z8.8)') 'AMIN1(0.,-0.) = ', transfer(amin1(a, b), 0)
  write(*, '(A,Z8.8)') 'AMIN1(-0.,0.) = ', transfer(amin1(b, a), 0)
  xd(1) = 0.0_r8; xd(2) = -0.0_r8
  ad = xd(1); bd = xd(2)
  write(*, '(A,Z16.16)') 'MAX(0._r8,-0.) = ', transfer(max(ad, bd), 0_8)
  write(*, '(A,Z16.16)') 'MAX(-0.,0._r8) = ', transfer(max(bd, ad), 0_8)
  write(*, '(A,Z16.16)') 'MIN(0._r8,-0.) = ', transfer(min(ad, bd), 0_8)
  write(*, '(A,Z16.16)') 'MIN(-0.,0._r8) = ', transfer(min(bd, ad), 0_8)
  ! NaN operands
  xs(3) = transfer(int(z'7FC00000'), 1.0)
  a = xs(3); b = xs(1)
  write(*, '(A,Z8.8)') 'MAX(NaN, 0.) = ', transfer(max(a, b), 0)
  write(*, '(A,Z8.8)') 'MAX(0., NaN) = ', transfer(max(b, a), 0)
  write(*, '(A,Z8.8)') 'MIN(NaN, 0.) = ', transfer(min(a, b), 0)
  write(*, '(A,Z8.8)') 'MIN(0., NaN) = ', transfer(min(b, a), 0)
  ! Constant intrinsics gfortran folds at compile time (MPFR, correctly
  ! rounded) rather than calling the library: the port must use these words.
  write(*, '(A,Z8.8)') 'fold EXP(-4.5)            = ', transfer(exp(-4.5), 0)
  write(*, '(A,Z8.8)') 'fold 2./3.                = ', transfer(2./3., 0)
  write(*, '(A,Z16.16)') 'fold log(2._r8)         = ', transfer(log(2._r8), 0_8)
  write(*, '(A,Z16.16)') 'fold log(3._r8)         = ', transfer(log(3._r8), 0_8)
  write(*, '(A,Z16.16)') 'fold sqrt(2._r8)        = ', transfer(sqrt(2._r8), 0_8)
  write(*, '(A,Z16.16)') 'fold 4*atan(1._r8)      = ', &
       transfer(4._r8*atan(1.0_r8), 0_8)
  write(*, '(A,Z16.16)') 'fold log10(6.1071_r8)   = ', &
       transfer(log10(6.1071_r8), 0_8)
  write(*, '(A,Z16.16)') 'fold log10(1013.246_r8) = ', &
       transfer(log10(1013.246_r8), 0_8)
  write(*, '(A,Z16.16)') 'fold (25.e-6_r8)**3     = ', &
       transfer((25.e-6_r8)**3, 0_8)
  write(*, '(A,Z16.16)') 'fold 11.e-6_r8**3       = ', &
       transfer(11.e-6_r8**3, 0_8)
  write(*, '(A,Z16.16)') 'fold 25.e-6_r8**3       = ', &
       transfer(25.e-6_r8**3, 0_8)
  write(*, '(A,Z16.16)') 'fold 1./3._r8           = ', transfer(1./3._r8, 0_8)
  write(*, '(A,Z16.16)') 'fold 1./6._r8           = ', transfer(1./6._r8, 0_8)
  write(*, '(A,Z16.16)') 'fold 4._r8/3._r8        = ', &
       transfer(4._r8/3._r8, 0_8)
  ! the same cubes through a runtime chain, for comparison
  xd(5) = 25.e-6_r8; xd(6) = 11.e-6_r8
  write(*, '(A,Z16.16)') 'chain 25e-6 cubed       = ', &
       transfer(xd(5)*xd(5)*xd(5), 0_8)
  write(*, '(A,Z16.16)') 'chain 11e-6 cubed       = ', &
       transfer(xd(6)*xd(6)*xd(6), 0_8)
  ! NINT / INT / AINT at halves and negatives
  xs(4) = 2.5; xs(5) = -2.5; xs(6) = 3.5; xs(7) = -0.4; xs(8) = 1.4999999
  do i = 4, 8
    write(*, '(A,ES14.6,A,I0,A,I0,A,Z8.8)') 'x=', xs(i), ' NINT=', &
         nint(xs(i)), ' INT=', int(xs(i)), ' AINT=', transfer(aint(xs(i)), 0)
  end do

contains

  integer function bits(v)
    real, intent(in) :: v
    bits = transfer(v, 0)
  end function bits

  integer(8) function bitsd(v)
    real(r8), intent(in) :: v
    bitsd = transfer(v, 0_8)
  end function bitsd

end program lowering_probe
