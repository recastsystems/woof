! The one framework name module_sf_qnsesfc.F imports outside its constants:
! it USEs MODULE_DM, ONLY : WRF_DM_MAXVAL unconditionally (:8) and calls it
! only under NMM_CORE == 1 with DM_PARALLEL (:1076-1087), which the ARW
! serial oracle build never compiles.  This stub satisfies the USE; if the
! routine were ever called the run stops rather than reduce nothing.
module module_dm
  implicit none
contains
  subroutine wrf_dm_maxval(val, idex, jdex)
    integer, intent(inout) :: val, idex, jdex
    stop 'wrf_dm_maxval stub reached: the ARW build never calls it'
  end subroutine wrf_dm_maxval
end module module_dm
