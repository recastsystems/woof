! Service-only stand-ins for the WRF framework symbols phys/module_cu_kf.F
! references.  Nothing here computes anything the scheme reads: the module
! only declares the message buffer KFPARA writes into on its mass-budget
! error path, and the subroutines are the framework's logging and abort
! surface.  frame/module_wrf_error.F itself needs ESMF and the WRF
! registry, so it is stood in for exactly as tools/advect_wrf_legacy_oracle
! does.
!
! A fatal call ends the process with exit code 7 after printing the WRF
! message on a line that starts "WRF-FATAL: ".  build.sh runs one process
! per case, so a fatal column is recorded as that column's outcome instead
! of ending the whole oracle.
module module_wrf_error
  implicit none
  character(len=256) :: wrf_err_message
end module module_wrf_error

subroutine wrf_error_fatal(message)
  implicit none
  character(*), intent(in) :: message
  write(*, '(a,a)') 'WRF-FATAL: ', trim(message)
  call flush(6)
  error stop 7
end subroutine wrf_error_fatal

! WRF's own build reaches the abort through the cpp macro
! wrf_error_fatal(A) -> wrf_error_fatal3(__FILE__,__LINE__,A); the
! production object (build.sh arm B) therefore references this name.
subroutine wrf_error_fatal3(file_str, line, message)
  implicit none
  character(*), intent(in) :: file_str
  integer, intent(in) :: line
  character(*), intent(in) :: message
  write(*, '(a,a)') 'WRF-FATAL: ', trim(message)
  call flush(6)
  error stop 7
end subroutine wrf_error_fatal3

subroutine wrf_message(message)
  implicit none
  character(*), intent(in) :: message
end subroutine wrf_message

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(*), intent(in) :: message
end subroutine wrf_debug
