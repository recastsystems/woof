! Service stubs for the SSiB column oracle.  module_sf_ssib.F CALLs these
! as external procedures (it USEs no WRF module).  Nothing here computes a
! physical quantity.  A fatal stops the process with a non-zero status, so
! a fixture can never be written past a WRF refusal.
subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0, '(a,a)') 'WRF FATAL: ', trim(msg)
  error stop 3
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(file, line, msg)
  character(len=*), intent(in) :: file, msg
  integer, intent(in) :: line
  write(0, '(a,a,a,i0,a,a)') 'WRF FATAL at ', trim(file), ':', line, ': ', trim(msg)
  error stop 3
end subroutine wrf_error_fatal3

subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0, '(a)') trim(msg)
end subroutine wrf_message
