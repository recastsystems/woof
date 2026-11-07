! The two WRF error entry points module_sf_fogdes.F can reach.  WRF's own
! build turns CALL wrf_error_fatal(...) into wrf_error_fatal3(__FILE__,
! __LINE__, ...); a plain gfortran -cpp build leaves the first form.  Both
! stop the oracle, which no case reaches (every case passes FGDP).
subroutine wrf_error_fatal(message)
  character(len=*), intent(in) :: message
  write(0, '(A)') trim(message)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(file, line, message)
  character(len=*), intent(in) :: file, message
  integer, intent(in) :: line
  write(0, '(A,1X,I0,1X,A)') trim(file), line, trim(message)
  error stop 1
end subroutine wrf_error_fatal3
