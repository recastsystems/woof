! Service-only stub for the standalone NTU oracle.  module_mp_ntu USEs
! module_wrf_error and calls wrf_error_fatal on its fatal branches; the
! oracle stops with a nonzero code if one is ever reached.
module module_wrf_error
  implicit none
end module module_wrf_error

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(0, '(A,A)') 'wrf_error_fatal: ', trim(message)
  error stop 7
end subroutine wrf_error_fatal
