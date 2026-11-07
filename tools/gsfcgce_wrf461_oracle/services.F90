! WRF logging services for the standalone Goddard GCE oracle.
!
! The referee objects (module_wrf_error.o and module_mp_radar.o of the
! scalar wrf.exe build) reference two external service routines that live in
! WRF's frame library (wrf_error_fatal3 itself is in module_wrf_error.o and
! ends in wrf_abort).  These supply them; neither contains arithmetic.  An
! abort stops the oracle with a nonzero status, so a fixture that reaches
! one cannot pass silently.
subroutine wrf_debug(level, str)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_abort
  implicit none
  write(0, '(a)') 'wrf_abort called'
  stop 3
end subroutine wrf_abort

! A dmpar native module_wrf_error asks whether this standalone process
! owns monitor logging. This service does not enter the column arithmetic.
logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
