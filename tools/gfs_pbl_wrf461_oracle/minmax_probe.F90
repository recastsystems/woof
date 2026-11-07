! minmax_probe.F90 -- how gfortran lowers REAL(8) MAX/MIN at the oracle's
! flags, for the ties and NaN operands the kernel's GFS_MAX/GFS_MIN must
! reproduce.  Operands come through a non-inlined read so nothing folds.
module mm_probe_in
  implicit none
contains
  subroutine load(a)
    real(8), intent(out) :: a(4)
    real(8) :: z
    z = 0.d0
    a(1) = z
    a(2) = -z
    a(3) = z / z
    a(4) = 1.d0
  end subroutine
end module

program minmax_probe
  use mm_probe_in
  implicit none
  real(8) :: a(4), r
  integer :: i, j
  call load(a)
  do i = 1, 4
    do j = 1, 4
      r = max(a(i), a(j))
      write(*, '(a,i1,i1,1x,z16.16)') 'max', i, j, r
      r = min(a(i), a(j))
      write(*, '(a,i1,i1,1x,z16.16)') 'min', i, j, r
    end do
  end do
end program
