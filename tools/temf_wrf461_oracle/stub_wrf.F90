! The one WRF framework routine the TEMF pair calls: temfsfclayinit's
! CALL wrf_debug(100, 'in temfsfclayinit').  A debug message only.
subroutine wrf_debug(level, str)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug
