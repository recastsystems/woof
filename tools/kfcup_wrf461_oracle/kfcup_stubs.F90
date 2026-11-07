! Link-time stand-ins for the three WRF framework entry points that
! module_cu_kfcup reaches, plus the WRF-Chem activation routine that it
! names but cannot reach at WRF_CHEM = 0.
!
! WRF's own object (phys/module_cu_kfcup.o of the strict build) references
! wrf_message_, wrf_debug_ and wrf_error_fatal3_ as EXTERNAL subroutines
! (nm -u), not as module procedures, so plain external subroutines with the
! same names satisfy both the WRF object and the -O0 recompilation.
!
! wrf_error_fatal3 stops with a distinctive code so the build script can
! tell "WRF would have aborted this column" from any other failure; a case
! that reaches it is a case WRF itself cannot run, and the fixture must not
! contain one.

module module_mixactivate
  implicit none
contains
  subroutine activate(wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, msectional, &
       maxd_atype, ntype_aer, maxd_asize, nsize_aer, na, volc, dlo_sect, dhi_sect, &
       sigman, hygro, fn, fs, fm, fluxn, fluxs, fluxm, flux_fullact, grid_id, ktau, &
       ii, jj, kk)
    ! The argument list of module_mixactivate's activate, as
    ! activate_cldbase_kfcup calls it (module_cu_kfcup.F:5068-5072).
    real :: wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, flux_fullact
    integer :: msectional, maxd_atype, ntype_aer, maxd_asize, grid_id, ktau, ii, jj, kk
    integer :: nsize_aer(*)
    real :: na(*), volc(*), dlo_sect(*), dhi_sect(*), sigman(*), hygro(*)
    real :: fn(*), fs(*), fm(*), fluxn(*), fluxs(*), fluxm(*)
    ! module_cu_kfcup calls activate only from activate_cldbase_kfcup, which
    ! runs only under flag_chem, which is .false. whenever WRF_CHEM /= 1
    ! (module_cu_kfcup.F:415-430).  Reaching here means the harness is not
    ! the WRF_CHEM = 0 build it claims to be.
    write(0,'(a)') 'FATAL: module_mixactivate%activate reached at WRF_CHEM=0'
    error stop 78
  end subroutine activate
end module module_mixactivate

subroutine wrf_message(str)
  implicit none
  character(len=*) :: str
  write(0,'(a)') 'wrf_message: '//trim(str)
end subroutine wrf_message

subroutine wrf_debug(level, str)
  implicit none
  integer :: level
  character(len=*) :: str
  write(0,'(a,i0,a)') 'wrf_debug(', level, '): '//trim(str)
end subroutine wrf_debug

subroutine wrf_error_fatal3(file_str, line, str)
  implicit none
  character(len=*) :: file_str, str
  integer :: line
  write(0,'(a,i0,a)') 'wrf_error_fatal3 at line ', line, ': '//trim(str)
  error stop 77
end subroutine wrf_error_fatal3
