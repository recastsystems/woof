!==========================================================================!
! p3_nc_ref_driver -- the mp_physics = 51 call shape of WRF 4.6.1's P3.
!
! The same harness as evidence/p3-fortran-oracle-20260828/harness/
! p3_ref_driver.f90 (the mp = 50 driver, left untouched), with the one
! difference WRF's own driver makes for P3_1CATEGORY_NC
! (module_microphysics_driver.F:1607-1647): mp_p3_wrapper_wrf receives
! nc_3d, so its present(nc_3d) test (module_mp_p3.F:819-820) switches
! log_predictNc on -- droplet number becomes a prognostic, CCN activation
! (:3313-3339) and the two-moment cloud sedimentation (:4009-4066) run.
! p3_init is called exactly as module_physics_init.F:4570-4571 calls it for
! mp = 51 (nCat 1, two-moment ice).
!
! module_mp_p3.F is compiled UNMODIFIED; nothing is stubbed or replaced.
!
! File layout (FORMAT.md, extended by ONE array): the fixture holds the
! fifteen (ncol, nk) inputs of the mp = 50 format, then nc (# kg-1) as a
! sixteenth (ncol, nk) array, then the five (ncol,) arrays.  Each dump
! frame holds the seventeen (ncol, nk) outputs of the mp = 50 format, then
! nc as an eighteenth, then the five (ncol,) arrays.  Raw little-endian
! binary32, stream access, Fortran element order.
!==========================================================================!
program p3_nc_ref_driver

  use microphy_p3, only: p3_init, mp_p3_wrapper_wrf

  implicit none

  character(len=1024) :: ctl_path, table_dir, fixture_path, out_path
  integer :: ncol, nk, nsteps, stat_init, u, ios, istep
  real    :: dt

  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qnr
  real, allocatable, dimension(:,:,:) :: qi1, qni1, qir1, qib1
  real, allocatable, dimension(:,:,:) :: th_old, qv_old
  real, allocatable, dimension(:,:,:) :: pii, p, dz, w
  real, allocatable, dimension(:,:,:) :: nc
  real, allocatable, dimension(:,:,:) :: zdbz, effc, effi, vmi, di, rhopo
  real, allocatable, dimension(:,:)   :: rainnc, rainncv, sr, snownc, snowncv

  if (command_argument_count() /= 1) then
     write(*,*) 'usage: p3_nc_ref_driver <control-file>'
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

  write(*,*) 'P3NCREF ncol=', ncol, ' nk=', nk, ' nsteps=', nsteps, ' dt=', dt

  allocate(th(ncol,nk,1), qv(ncol,nk,1), qc(ncol,nk,1), qr(ncol,nk,1), qnr(ncol,nk,1))
  allocate(qi1(ncol,nk,1), qni1(ncol,nk,1), qir1(ncol,nk,1), qib1(ncol,nk,1))
  allocate(th_old(ncol,nk,1), qv_old(ncol,nk,1))
  allocate(pii(ncol,nk,1), p(ncol,nk,1), dz(ncol,nk,1), w(ncol,nk,1))
  allocate(nc(ncol,nk,1))
  allocate(zdbz(ncol,nk,1), effc(ncol,nk,1), effi(ncol,nk,1))
  allocate(vmi(ncol,nk,1), di(ncol,nk,1), rhopo(ncol,nk,1))
  allocate(rainnc(ncol,1), rainncv(ncol,1), sr(ncol,1), snownc(ncol,1), snowncv(ncol,1))

  zdbz = -8.888e30 ; effc = -8.888e30 ; effi = -8.888e30
  vmi  = -8.888e30 ; di   = -8.888e30 ; rhopo = -8.888e30

  open(newunit=u, file=trim(fixture_path), status='old', access='stream', &
       form='unformatted', action='read', iostat=ios)
  if (ios /= 0) then
     write(*,*) 'FATAL: cannot open fixture ', trim(fixture_path)
     stop 4
  end if
  read(u) th, qv, qc, qr, qnr
  read(u) qi1, qni1, qir1, qib1
  read(u) th_old, qv_old
  read(u) pii, p, dz, w
  read(u) nc
  read(u) rainnc, rainncv, sr, snownc, snowncv
  close(u)

  ! module_physics_init.F:4570-4571, CASE (P3_1CATEGORY_NC)
  call p3_init(trim(table_dir), 1, .false., 'WRF', stat_init, .true.)
  write(*,*) 'P3NCREF p3_init stat=', stat_init
  if (stat_init /= 0) then
     write(*,*) 'FATAL: p3_init returned non-OK'
     stop 5
  end if

  open(newunit=u, file=trim(out_path), status='replace', access='stream', &
       form='unformatted', action='write')

  call dump(u)

  do istep = 1, nsteps
     ! module_microphysics_driver.F:1619-1647: the mp = 50 argument list
     ! plus nc_3d = qnc_curr.
     call mp_p3_wrapper_wrf(th, qv, qc, qr, qnr,                          &
                            th_old, qv_old,                               &
                            pii, p, dz, w, dt, istep,                     &
                            rainnc, rainncv, sr, snownc, snowncv, 1,      &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            zdbz, effc, effi,                             &
                            vmi, di, rhopo,                               &
                            qi1, qni1, qir1, qib1, nc_3d=nc)
     call dump(u)
  end do

  close(u)
  write(*,*) 'P3NCREF DONE steps=', nsteps

contains

  subroutine dump(unit)
    integer, intent(in) :: unit
    write(unit) th, qv, qc, qr, qnr
    write(unit) qi1, qni1, qir1, qib1
    write(unit) th_old, qv_old
    write(unit) zdbz, effc, effi, vmi, di, rhopo
    write(unit) nc
    write(unit) rainnc, rainncv, sr, snownc, snowncv
  end subroutine dump

end program p3_nc_ref_driver
