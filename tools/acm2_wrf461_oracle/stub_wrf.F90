! The one external symbol phys/module_bl_acm.F references outside libm and
! libgfortran (nm -u, build.sh receipt): WRF's fatal-error entry, which the
! module reaches only when the bulk Richardson number never exceeds RIC in
! a column (module_bl_acm.F:464-470).  WRF stops the model there; so does
! the oracle, with exit status 3 and the column's message on stderr, so a
! fixture that reaches it can never be mistaken for one that did not.
subroutine wrf_error_fatal3(file_str, line, str)
  implicit none
  character(len=*), intent(in) :: file_str
  integer, intent(in) :: line
  character(len=*), intent(in) :: str
  write(0, '(a)') 'ORACLE-FATAL ' // trim(file_str) // ': ' // trim(adjustl(str))
  write(0, '(a,i0)') 'ORACLE-FATAL line ', line
  stop 3
end subroutine wrf_error_fatal3
