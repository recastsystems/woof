!==========================================================================!
! p3_ref_driver_2cat -- executable reference harness around WRF's P3 v4.5.2
! in the mp_physics = 52 call shape (P3_2CATEGORY).
!
! module_microphysics_driver.F:1654-1706 calls mp_p3_wrapper_wrf_2cat with
! N_ICECAT=2, the eight category fields QI1/QIR1/QNI1/QIB1 and
! QI2/QIR2/QNI2/QIB2, and nc_3d=qnc_curr (so log_predictNc = .true.,
! module_mp_p3.F:1064-1065).  module_physics_init.F:4572-4573 calls
! p3_init('.', 2, .false., ...).  This driver makes exactly those two calls
! against the UNMODIFIED module_mp_p3.F (sha256 716950a3...f6a3c6); nothing
! in the module is stubbed, patched or replaced.
!
! I/O: raw little-endian IEEE-754 binary32, stream access, Fortran element
! order (first index fastest), no record markers -- the layout of
! evidence/p3-fortran-oracle-20260828/FORMAT.md widened to the 2-category
! field list (tools/p3_wrf461_oracle/README.md, "mp_physics = 52"):
!
!   fixture:  20 (ncol,nk) arrays  th qv qc qr nr nc
!                                  qi1 ni1 qir1 qib1 qi2 ni2 qir2 qib2
!                                  th_old qv_old pii p dz w
!             then 5 (ncol,) arrays rainnc rainncv sr snownc snowncv
!   frame:    25 (ncol,nk) arrays  th qv qc qr nr nc
!                                  qi1 ni1 qir1 qib1 qi2 ni2 qir2 qib2
!                                  th_old qv_old
!                                  zdbz effc effi vmi di rhopo vmi2 di2 rhopo2
!             then the 5 surface arrays.
!
! Frame 0 is the state as read, with the nine diagnostics at the poison
! value -8.888e30, so a field the scheme never writes is visible.
!==========================================================================!
program p3_ref_driver_2cat

  use microphy_p3, only: p3_init, mp_p3_wrapper_wrf_2cat

  implicit none

  character(len=1024) :: ctl_path, table_dir, fixture_path, out_path
  integer :: ncol, nk, nsteps, stat_init, u, ios, istep
  real    :: dt

  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qnr, nc
  real, allocatable, dimension(:,:,:) :: qi1, qni1, qir1, qib1
  real, allocatable, dimension(:,:,:) :: qi2, qni2, qir2, qib2
  real, allocatable, dimension(:,:,:) :: th_old, qv_old
  real, allocatable, dimension(:,:,:) :: pii, p, dz, w
  real, allocatable, dimension(:,:,:) :: zdbz, effc, effi, vmi, di, rhopo
  real, allocatable, dimension(:,:,:) :: vmi2, di2, rhopo2
  real, allocatable, dimension(:,:)   :: rainnc, rainncv, sr, snownc, snowncv

  if (command_argument_count() /= 1) then
     write(*,*) 'usage: p3_ref_driver_2cat <control-file>'
     stop 2
  end if
  call get_command_argument(1, ctl_path)

  open(newunit=u, file=trim(ctl_path), status='old', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,*) 'FATAL: cannot open control file ', trim(ctl_path)
     stop 3
  end if
  read(u,'(A)') table_dir
  read(u,'(A)') fixture_path
  read(u,'(A)') out_path
  read(u,*) ncol, nk, nsteps
  read(u,*) dt
  close(u)

  table_dir    = adjustl(table_dir)
  fixture_path = adjustl(fixture_path)
  out_path     = adjustl(out_path)

  write(*,*) 'P3REF2 ncol=', ncol, ' nk=', nk, ' nsteps=', nsteps, ' dt=', dt

  allocate(th(ncol,nk,1), qv(ncol,nk,1), qc(ncol,nk,1), qr(ncol,nk,1))
  allocate(qnr(ncol,nk,1), nc(ncol,nk,1))
  allocate(qi1(ncol,nk,1), qni1(ncol,nk,1), qir1(ncol,nk,1), qib1(ncol,nk,1))
  allocate(qi2(ncol,nk,1), qni2(ncol,nk,1), qir2(ncol,nk,1), qib2(ncol,nk,1))
  allocate(th_old(ncol,nk,1), qv_old(ncol,nk,1))
  allocate(pii(ncol,nk,1), p(ncol,nk,1), dz(ncol,nk,1), w(ncol,nk,1))
  allocate(zdbz(ncol,nk,1), effc(ncol,nk,1), effi(ncol,nk,1))
  allocate(vmi(ncol,nk,1), di(ncol,nk,1), rhopo(ncol,nk,1))
  allocate(vmi2(ncol,nk,1), di2(ncol,nk,1), rhopo2(ncol,nk,1))
  allocate(rainnc(ncol,1), rainncv(ncol,1), sr(ncol,1), snownc(ncol,1), snowncv(ncol,1))

  zdbz = -8.888e30 ; effc = -8.888e30 ; effi = -8.888e30
  vmi  = -8.888e30 ; di   = -8.888e30 ; rhopo = -8.888e30
  vmi2 = -8.888e30 ; di2  = -8.888e30 ; rhopo2 = -8.888e30

  open(newunit=u, file=trim(fixture_path), status='old', access='stream', &
       form='unformatted', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,*) 'FATAL: cannot open fixture ', trim(fixture_path)
     stop 4
  end if
  read(u) th, qv, qc, qr, qnr, nc
  read(u) qi1, qni1, qir1, qib1
  read(u) qi2, qni2, qir2, qib2
  read(u) th_old, qv_old
  read(u) pii, p, dz, w
  read(u) rainnc, rainncv, sr, snownc, snowncv
  close(u)

  ! module_physics_init.F:4572-4573, CASE (P3_2CATEGORY)
  call p3_init(trim(table_dir), 2, .false., 'WRF', stat_init, .true.)
  write(*,*) 'P3REF2 p3_init stat=', stat_init
  if (stat_init /= 0) then
     write(*,*) 'FATAL: p3_init returned non-OK'
     stop 5
  end if

  open(newunit=u, file=trim(out_path), status='replace', access='stream', &
       form='unformatted', action='write')

  call dump(u)

  do istep = 1, nsteps
     call mp_p3_wrapper_wrf_2cat(th, qv, qc, qr, qnr,                     &
                            th_old, qv_old,                               &
                            pii, p, dz, w, dt, istep,                     &
                            rainnc, rainncv, sr, snownc, snowncv, 2,      &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            zdbz, effc, effi,                             &
                            vmi, di, rhopo,                               &
                            vmi2, di2, rhopo2,                            &
                            qi1, qni1, qir1, qib1,                        &
                            qi2, qni2, qir2, qib2, nc)
     call dump(u)
  end do

  close(u)
  write(*,*) 'P3REF2 DONE steps=', nsteps

contains

  subroutine dump(unit)
    integer, intent(in) :: unit
    write(unit) th, qv, qc, qr, qnr, nc
    write(unit) qi1, qni1, qir1, qib1
    write(unit) qi2, qni2, qir2, qib2
    write(unit) th_old, qv_old
    write(unit) zdbz, effc, effi, vmi, di, rhopo, vmi2, di2, rhopo2
    write(unit) rainnc, rainncv, sr, snownc, snowncv
  end subroutine dump

end program p3_ref_driver_2cat
