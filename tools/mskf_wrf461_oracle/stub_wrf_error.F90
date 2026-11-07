! Service-only stub for the standalone MSKF (cu_physics = 11) oracle.
!
! module_cu_mskf.F USEs module_wrf_error and calls wrf_message and
! wrf_error_fatal.  Neither carries arithmetic.  wrf_message is silent here;
! wrf_error_fatal records the message and stops the harness with a nonzero
! status, because WRF itself stops the model there and a fixture written
! after a fatal error would be a column WRF never produces.  build.sh fails
! if the harness stops, so no case in the table reaches a fatal path.
module module_wrf_error
  implicit none
end module module_wrf_error

subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
end subroutine wrf_message

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(0, '(A,A)') 'WRF FATAL (oracle stub): ', trim(message)
  stop 97
end subroutine wrf_error_fatal

! WRF's tools/standard.exe rewrites every wrf_error_fatal(msg) call into
! wrf_error_fatal3(__FILE__, __LINE__, msg) (frame/module_wrf_error.F).
subroutine wrf_error_fatal3(file_str, line, message)
  implicit none
  character(len=*), intent(in) :: file_str
  integer, intent(in) :: line
  character(len=*), intent(in) :: message
  write(0, '(A,A,A,I0,A,A)') 'WRF FATAL (oracle stub) ', trim(file_str), &
       ':', line, ': ', trim(message)
  stop 97
end subroutine wrf_error_fatal3
