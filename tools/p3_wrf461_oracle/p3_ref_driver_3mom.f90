!==========================================================================!
! p3_ref_driver_3mom -- WRF 4.6.1 P3 column oracle, mp_physics = 53 shape.
!
! The mp_physics = 53 sibling of
! evidence/p3-fortran-oracle-20260828/harness/p3_ref_driver.f90.  It calls
! the UNMODIFIED phys/module_mp_p3.F (sha256 716950a3...f6a3c6) exactly as
! WRF's driver does for P3_1CAT_3MOM (module_microphysics_driver.F:1710-1746):
!
!   * p3_init(dir, nCat=1, trplMomI=.true., 'WRF')       (module_physics_init.F:4574-4575)
!   * mp_p3_wrapper_wrf(..., N_ICECAT=1, nc_3d=qnc, qzi1_3d=qzi)
!     so log_predictNc = .true. (:819-820) and log_3momentIce = .true.
!     (:822-823).
!
! The fixture carries the mp=50 fifteen (ncol, nk) inputs, then nc and qzi
! (the ADVECTED form, (N*Z)**0.5, which the wrapper converts at :826-838 and
! back at :912-924), then the five (ncol,) surface fields.  Each dump frame
! is the mp=50 frame with nc and qzi appended after the 17 3-D fields.
!
! I/O is raw little-endian binary32, stream access, Fortran element order.
!==========================================================================!
program p3_ref_driver_3mom

  use microphy_p3, only: p3_init, mp_p3_wrapper_wrf

  implicit none

  character(len=1024) :: ctl_path, table_dir, fixture_path, out_path
  integer :: ncol, nk, nsteps, stat_init, u, ios, istep
  real    :: dt

  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qnr
  real, allocatable, dimension(:,:,:) :: qi1, qni1, qir1, qib1
  real, allocatable, dimension(:,:,:) :: th_old, qv_old
  real, allocatable, dimension(:,:,:) :: pii, p, dz, w
  real, allocatable, dimension(:,:,:) :: nc, qzi
  real, allocatable, dimension(:,:,:) :: zdbz, effc, effi, vmi, di, rhopo
  real, allocatable, dimension(:,:)   :: rainnc, rainncv, sr, snownc, snowncv

  if (command_argument_count() /= 1) then
     write(*,*) 'usage: p3_ref_driver_3mom <control-file>'
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

  write(*,*) 'P3REF3 ncol=', ncol, ' nk=', nk, ' nsteps=', nsteps, ' dt=', dt

  allocate(th(ncol,nk,1), qv(ncol,nk,1), qc(ncol,nk,1), qr(ncol,nk,1), qnr(ncol,nk,1))
  allocate(qi1(ncol,nk,1), qni1(ncol,nk,1), qir1(ncol,nk,1), qib1(ncol,nk,1))
  allocate(th_old(ncol,nk,1), qv_old(ncol,nk,1))
  allocate(pii(ncol,nk,1), p(ncol,nk,1), dz(ncol,nk,1), w(ncol,nk,1))
  allocate(nc(ncol,nk,1), qzi(ncol,nk,1))
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
  read(u) nc, qzi
  read(u) rainnc, rainncv, sr, snownc, snowncv
  close(u)

  call p3_init(trim(table_dir), 1, .true., 'WRF', stat_init, .true.)
  write(*,*) 'P3REF3 p3_init stat=', stat_init
  if (stat_init /= 0) then
     write(*,*) 'FATAL: p3_init returned non-OK'
     stop 5
  end if

  open(newunit=u, file=trim(out_path), status='replace', access='stream', &
       form='unformatted', action='write')
  call dump(u)

  do istep = 1, nsteps
     call mp_p3_wrapper_wrf(th, qv, qc, qr, qnr,                          &
                            th_old, qv_old,                               &
                            pii, p, dz, w, dt, istep,                     &
                            rainnc, rainncv, sr, snownc, snowncv, 1,      &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            1, ncol, 1, 1, 1, nk,                         &
                            zdbz, effc, effi,                             &
                            vmi, di, rhopo,                               &
                            qi1, qni1, qir1, qib1, nc_3d=nc, qzi1_3d=qzi)
     call dump(u)
  end do

  close(u)
  write(*,*) 'P3REF3 DONE steps=', nsteps

contains

  subroutine dump(unit)
    integer, intent(in) :: unit
    ! The mp=50 order (17 three-dimensional, then 5 two-dimensional fields)
    ! with nc and qzi inserted after the 17: the reader depends on this order.
    write(unit) th, qv, qc, qr, qnr
    write(unit) qi1, qni1, qir1, qib1
    write(unit) th_old, qv_old
    write(unit) zdbz, effc, effi, vmi, di, rhopo
    write(unit) nc, qzi
    write(unit) rainnc, rainncv, sr, snownc, snowncv
  end subroutine dump

end program p3_ref_driver_3mom
