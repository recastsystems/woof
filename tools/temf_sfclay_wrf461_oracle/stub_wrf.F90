! The one external module_sf_temfsfclay.F calls: TEMFSFCLAYINIT's
! CALL wrf_debug(100, ...).  A debug print at level 100, silent here as it
! is in a default WRF build (debug_level = 0).
subroutine wrf_debug(level, str)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug
