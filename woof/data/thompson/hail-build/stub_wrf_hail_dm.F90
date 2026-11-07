! WOOF's slice harness for WRF v4.6.1's own table generation.
!
! The hail-aware (mp_physics=38) rain-graupel table qr_acr_qg_mp38V1.dat
! integrates 9 graupel densities x 37 x 37 graupel bins x 37 x 37 rain bins,
! each over 100 x 100 size bins: about three quarters of an hour on one CPU.
! WRF itself spreads that loop over its MPI ranks: under DM_PARALLEL,
! qr_acr_qg asks wrf_dm_decomp1d for a [km_s, km_e] share of the
! ntb_r*ntb_r1 rain index pairs, computes only that share and gathers the
! rest with wrf_dm_gatherv (module_mp_thompson.F:4201-4285).  Every table
! element depends only on its own five indices, so a share computed in a
! separate process is bit for bit the element a single process computes.
!
! This file provides that decomposition WITHOUT MPI: wrf_dm_decomp1d reads
! the share from the environment (THOMPSON_TABLE_SLICE of
! THOMPSON_TABLE_SLICES), the gather is a no-op, every process writes the
! full arrays with only its share filled, and
! gpuwm/thompson_density_tables.py splices the shares back together.  It is
! otherwise stub_wrf_hail.F90: WRF's error/model-constant/timing modules,
! write_thompson_tables and write_thompson_mp38table true.
module module_wrf_error
  implicit none
  integer, parameter :: IWORDSIZE=4
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
contains
  real function wrf_dm_max_real(value)
    real, intent(in) :: value
    wrf_dm_max_real = value
  end function wrf_dm_max_real
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
  value = .true.
end subroutine nl_get_write_thompson_mp38table

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_decomp1d(nt, km_s, km_e)
  implicit none
  integer, intent(in) :: nt
  integer, intent(out) :: km_s, km_e
  character(len=32) :: text
  integer :: slice, slices, chunk, status
  call get_environment_variable('THOMPSON_TABLE_SLICE', text, status=status)
  if (status /= 0) error stop 'THOMPSON_TABLE_SLICE is not set'
  read(text, *) slice
  call get_environment_variable('THOMPSON_TABLE_SLICES', text, status=status)
  if (status /= 0) error stop 'THOMPSON_TABLE_SLICES is not set'
  read(text, *) slices
  if (slices < 1 .or. slice < 0 .or. slice >= slices) then
     error stop 'THOMPSON_TABLE_SLICE out of range'
  endif
  chunk = (nt + slices - 1) / slices
  km_s = slice * chunk
  km_e = min(nt, (slice + 1) * chunk) - 1
  print '(A,I0,A,I0,A,I0)', 'THOMPSON_TABLE_SHARE ', km_s, ' ', km_e, ' OF ', nt
end subroutine wrf_dm_decomp1d

subroutine wrf_dm_gatherv(values, elements, km_s, km_e, typesize)
  implicit none
  double precision :: values(*)
  integer, intent(in) :: elements, km_s, km_e, typesize
end subroutine wrf_dm_gatherv

subroutine wrf_dm_bcast_integer(values, n)
  implicit none
  integer :: values(*)
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_double(values, n)
  implicit none
  double precision :: values(*)
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_double

subroutine wrf_dm_bcast_bytes(values, n)
  implicit none
  integer :: values(*)
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_bytes
