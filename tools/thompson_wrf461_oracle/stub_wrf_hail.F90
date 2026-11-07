! Hail-aware sibling of stub_wrf.F90: one namelist switch differs.
module module_wrf_error
  implicit none
  character(len=512) :: wrf_err_message
contains
  subroutine wrf_debug(level, message)
    integer, intent(in) :: level
    character(len=*), intent(in) :: message
  end subroutine wrf_debug

  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
    print '(A)', trim(message)
  end subroutine wrf_message

  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    print '(A)', trim(message)
    error stop 1
  end subroutine wrf_error_fatal
end module module_wrf_error

module module_model_constants
  implicit none
  real, parameter :: RE_QC_BG = 2.49e-6
  real, parameter :: RE_QI_BG = 4.99e-6
  real, parameter :: RE_QS_BG = 9.99e-6
end module module_model_constants

module module_timing
  implicit none
contains
  subroutine start_timing()
    implicit none
  end subroutine start_timing

  subroutine end_timing(label)
    implicit none
    character(len=*), intent(in) :: label
  end subroutine end_timing
end module module_timing

module module_domain
  implicit none
end module module_domain

module module_dm
  implicit none
end module module_dm

subroutine nl_get_force_read_thompson(domain, value)
  implicit none
  integer, intent(in) :: domain
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_force_read_thompson

subroutine nl_get_write_thompson_tables(domain, value)
  implicit none
  integer, intent(in) :: domain
  logical, intent(out) :: value
  value = .true.
end subroutine nl_get_write_thompson_tables

subroutine nl_get_write_thompson_mp38table(domain, value)
  implicit none
  integer, intent(in) :: domain
  logical, intent(out) :: value
  ! The hail-aware (mp_physics=38) harness: WRF refuses to compute
  ! qr_acr_qg_mp38V1.dat unless this namelist switch is true
  ! (module_mp_thompson.F:4163-4165), so this sibling of stub_wrf.F90 says
  ! yes.  stub_wrf.F90 itself is pinned by the mp=8/mp=28 receipts.
  value = .true.
end subroutine nl_get_write_thompson_mp38table

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
