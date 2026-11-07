! Positive control for build.sh's libmvec guard.
!
! EXP is one of the scheme's transcendentals.  This loop is deliberately
! vectorisable at WRF's physics flags, so gfortran/glibc must leave a _ZGV*
! undefined symbol; if it does not, silence from nm on the -O0 reference
! object proves nothing.
subroutine sbu_ylin_libmvec_positive_control(x, y, n)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: x(n)
  real, intent(out) :: y(n)
  integer :: k

  do k = 1, n
    y(k) = exp(x(k))
  end do
end subroutine sbu_ylin_libmvec_positive_control
