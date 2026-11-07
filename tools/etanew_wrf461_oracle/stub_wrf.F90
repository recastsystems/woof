! Minimal stand-ins for the WRF framework routines module_mp_etanew.F
! calls.  A single-process column driver is always the monitor task and has
! nothing to broadcast; a fatal error stops the oracle with WRF's message.
! wrf_error_fatal3 is the spelling WRF's own preprocessed build references.
subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(0, '(A)') 'wrf_error_fatal: ' // trim(message)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(file_str, line, message)
  implicit none
  character(len=*), intent(in) :: file_str, message
  integer, intent(in) :: line
  write(0, '(A,1X,I0,1X,A)') trim(file_str), line, trim(message)
  error stop 1
end subroutine wrf_error_fatal3

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(buf, nbytes)
  implicit none
  integer, intent(in) :: nbytes
  integer :: buf(*)
end subroutine wrf_dm_bcast_bytes
