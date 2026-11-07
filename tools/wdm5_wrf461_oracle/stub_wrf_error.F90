! Service-only stub for the standalone WDM5 oracle.
!
! module_mp_radar.F (which module_mp_wdm5.F uses for its 10 cm reflectivity)
! USEs module_wrf_error and calls wrf_debug.  The empty module supplies only
! the compile-time service dependency and wrf_debug discards its message; it
! contains no physical constants or arithmetic.
module module_wrf_error
  implicit none
end module module_wrf_error

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug
