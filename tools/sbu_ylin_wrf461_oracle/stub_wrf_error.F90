! Service-only stub for the standalone SBU-YLin oracle.
!
! module_mp_sbu_ylin.F USEs module_wrf_error, but no path the oracle drives
! calls a WRF logging or fatal-error routine (every wrf_debug call in the
! scheme is commented out).  The empty module supplies only the compile-time
! dependency; it holds no constants and no arithmetic.
module module_wrf_error
  implicit none
end module module_wrf_error
