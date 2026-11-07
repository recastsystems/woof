! Service-only stub for the standalone Morrison oracle.
!
! The two byte-unmodified WRF physics modules built by build.sh USE
! module_wrf_error, but the paths exercised by this harness do not call a WRF
! logging or fatal-error routine.  The empty module supplies only the compile-
! time service dependency; it contains no physical constants or arithmetic.
module module_wrf_error
  implicit none
end module module_wrf_error

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug
