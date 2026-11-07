! Service-only stubs for the standalone Goddard 4-ice oracle.
!
! module_mp_radar.F USEs module_wrf_error, and module_mp_gsfcgce_4ice_nuwrf.F
! reaches wrf_err_message through it (auto_conversion's WRITE, compiled but
! unreachable without WRF-Chem).  The scheme also calls the LOGICAL function
! wrf_dm_on_monitor only to decide whether to PRINT, and wrf_error_fatal only
! when XLAND is neither 1 nor 2.  None of these touch a physical value: the
! stubs carry no constants and no arithmetic.  wrf_error_fatal3 stops the
! oracle, so a case that reaches it fails loudly instead of writing output.
module module_wrf_error
  implicit none
  character(len=256) :: wrf_err_message
end module module_wrf_error

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .false.
end function wrf_dm_on_monitor

subroutine wrf_error_fatal3(file_str, line, str)
  implicit none
  character(len=*), intent(in) :: file_str
  integer, intent(in) :: line
  character(len=*), intent(in) :: str
  write(0, '(A)') 'wrf_error_fatal reached: ' // trim(str)
  error stop 7
end subroutine wrf_error_fatal3

! The same service under its unrewritten name: WRF's build rewrites
! CALL wrf_error_fatal( into wrf_error_fatal3 with arch/standard.sed; a
! standalone compile of the pristine file does not.
subroutine wrf_error_fatal(str)
  implicit none
  character(len=*), intent(in) :: str
  write(0, '(A)') 'wrf_error_fatal reached: ' // trim(str)
  error stop 7
end subroutine wrf_error_fatal
