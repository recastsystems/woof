! The binary64 words gfortran's front end folds out of the constant
! sub-expressions the UW shallow cumulus port transcribes, printed as hex
! so the CUDA header carries the oracle's own bits rather than a decimal
! re-derivation.  Each expression is written exactly as at its source line
! (same operands, same association), compiled with the reference flags.
!
!   module_shcu_camuwshcu_driver.F:394  (4._r8*3.14159_r8*(10.e-6_r8**3)*997._r8)
!   module_shcu_camuwshcu_driver.F:395  (4._r8*3.14159_r8*(50.e-6_r8**3)*500._r8)
!   module_shcu_camuwshcu.F:5281        log(10._r8)        (a1 = 7.5_r8*log(10._r8))
!   module_shcu_camuwshcu.F:4992        sqrt(3.141592_r8)
!   module_shcu_camuwshcu.F:2169        0.25_r8/rmaxfrac**2 (rmaxfrac = 0.10_r8 PARAMETER)
!   module_shcu_camuwshcu.F:2192        (2._r8/2.5066_r8)
program probe_constants
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: rmaxfrac = 0.10_r8
  call show('den_liq', (4._r8*3.14159_r8*(10.e-6_r8**3)*997._r8))
  call show('den_ice', (4._r8*3.14159_r8*(50.e-6_r8**3)*500._r8))
  call show('log10', log(10._r8))
  call show('a1', 7.5_r8*log(10._r8))
  call show('sqrtpi6', sqrt(3.141592_r8))
  call show('quarter_over_rmaxfrac2', 0.25_r8/rmaxfrac**2)
  call show('two_over_25066', (2._r8/2.5066_r8))
contains
  subroutine show(name, x)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: x
    integer(8) :: ix
    ix = transfer(x, ix)
    write(*, '(a,1x,z16.16,1x,es25.17)') name, ix, x
  end subroutine show
end program probe_constants
