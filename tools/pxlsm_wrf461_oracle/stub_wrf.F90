! Service-only stubs for the standalone Pleim-Xiu LSM WRF v4.6.1 oracle.
!
! Nothing here computes a physical quantity.  share/module_model_constants.F
! (g, r_d, SVP1..SVPT0, EP_2, KARMAN, STBOLT: the constants the whole
! reference is measured in) and frame/module_wrf_error.F (wrf_message,
! wrf_error_fatal, which pxlsm calls) are compiled from the pinned tree by
! build.sh, byte-unmodified.  What is left is what the link step reports
! as undefined outside libm/libgfortran: module_wrf_error.o's wrf_abort and
! wrf_debug.  wrf_abort stops with a failure, so a case that reached
! pxlsm's unknown-land-use fatal would fail the build instead of writing a
! fixture.

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
