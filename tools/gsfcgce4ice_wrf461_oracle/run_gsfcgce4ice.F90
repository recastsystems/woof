program run_gsfcgce4ice
  ! Drive the public WRF v4.6.1 Goddard 4-ice (mp_physics = 7) entry point,
  ! gsfcgce_4ice_nuwrf, linked from a byte-unmodified module.  Every output
  ! word this program writes was returned by WRF; it constructs inputs and
  ! carries state between calls, and contains no physics.
  !
  ! Input and output are raw little-endian stream files; the layout is owned
  ! by tools/gsfcgce4ice_wrf461_oracle/layout.py:
  !   header  int32 ncol, nz, nstep, do_radar_ref
  !           float32 dt, dx
  !           int32 itimestep(nstep), diagflag(nstep)
  !   column  float32 ht(ncol), xland(ncol)
  !   levels  NLEV float32 (ncol, nz) fields, then w as (ncol, nz+1)
  !   surface NSFC float32 (ncol) fields
  ! After every call the output receives the NLEV level fields, the NPREC
  ! sedimentation-flux fields and the NSFC surface fields, in layout order.
  use module_mp_gsfcgce_4ice_nuwrf, only: gsfcgce_4ice_nuwrf
  implicit none

  integer, parameter :: nlev = 32
  integer, parameter :: nprec = 5
  integer, parameter :: nsfc = 9
  integer :: ncol, nz, nstep, do_radar_ref, istep, f, ui, uo
  real :: dt, dx
  integer, allocatable :: itimestep(:), diagflag(:)
  real, allocatable :: ht(:, :), xland(:, :)
  real, allocatable :: lev(:, :, :, :)    ! (ncol, nz+1, 1, nlev)
  real, allocatable :: prc(:, :, :, :)    ! (ncol, nz+1, 1, nprec)
  real, allocatable :: w(:, :, :)
  real, allocatable :: sfc(:, :, :)       ! (ncol, 1, nsfc)
  character(len=1024) :: in_path, out_path
  logical :: due

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_gsfcgce4ice INPUT.bin OUTPUT.bin'
    error stop 2
  end if
  ! The scheme declares default REAL; the reference must be binary32.
  if (storage_size(1.0) /= 32) error stop 3

  open(newunit=ui, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(ui) ncol, nz, nstep, do_radar_ref
  read(ui) dt, dx
  allocate(itimestep(nstep), diagflag(nstep))
  read(ui) itimestep
  read(ui) diagflag
  allocate(ht(ncol, 1), xland(ncol, 1))
  read(ui) ht(:, 1)
  read(ui) xland(:, 1)
  ! The memory window is kms:kme = 1:nz+1 exactly as WRF allocates a mass
  ! field; level nz+1 is never read by the scheme and stays zero.
  allocate(lev(ncol, nz + 1, 1, nlev), prc(ncol, nz + 1, 1, nprec))
  allocate(w(ncol, nz + 1, 1), sfc(ncol, 1, nsfc))
  lev = 0.0
  prc = 0.0
  do f = 1, nlev
    read(ui) lev(:, 1:nz, 1, f)
  end do
  read(ui) w(:, 1:nz + 1, 1)
  do f = 1, nsfc
    read(ui) sfc(:, 1, f)
  end do
  close(ui)

  open(newunit=uo, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')

  do istep = 1, nstep
    due = diagflag(istep) /= 0
    ! Index order is layout.LEVEL_FIELDS / PREC_FIELDS / SURFACE_FIELDS.
    call gsfcgce_4ice_nuwrf(                                             &
         th=lev(:, :, :, 1), qv=lev(:, :, :, 2), ql=lev(:, :, :, 3),     &
         qr=lev(:, :, :, 4), qi=lev(:, :, :, 5), qs=lev(:, :, :, 6),     &
         qh=lev(:, :, :, 8),                                             &
         rho=lev(:, :, :, 9), pii=lev(:, :, :, 10), p=lev(:, :, :, 11),  &
         dt_in=dt, z=lev(:, :, :, 12),                                   &
         ht=ht, dz8w=lev(:, :, :, 13), grav=9.81, w=w,                   &
         rhowater=1000., rhosnow=100.,                                   &
         itimestep=itimestep(istep), xland=xland, dx=dx,                 &
         ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nz + 1,           &
         ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz + 1,               &
         its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz,                   &
         rainnc=sfc(:, :, 1), rainncv=sfc(:, :, 2),                      &
         snownc=sfc(:, :, 3), snowncv=sfc(:, :, 4), sr=sfc(:, :, 5),     &
         graupelnc=sfc(:, :, 6), graupelncv=sfc(:, :, 7),                &
         refl_10cm=lev(:, :, :, 14), diagflag=due,                       &
         do_radar_ref=do_radar_ref,                                      &
         hailnc=sfc(:, :, 8), hailncv=sfc(:, :, 9),                      &
         f_qg=.true., qg=lev(:, :, :, 7),                                &
         physc=lev(:, :, :, 15), physe=lev(:, :, :, 16),                 &
         physd=lev(:, :, :, 17), physs=lev(:, :, :, 18),                 &
         physm=lev(:, :, :, 19), physf=lev(:, :, :, 20),                 &
         acphysc=lev(:, :, :, 21), acphyse=lev(:, :, :, 22),             &
         acphysd=lev(:, :, :, 23), acphyss=lev(:, :, :, 24),             &
         acphysm=lev(:, :, :, 25), acphysf=lev(:, :, :, 26),             &
         re_cloud_gsfc=lev(:, :, :, 27), re_rain_gsfc=lev(:, :, :, 28),  &
         re_ice_gsfc=lev(:, :, :, 29), re_snow_gsfc=lev(:, :, :, 30),    &
         re_graupel_gsfc=lev(:, :, :, 31), re_hail_gsfc=lev(:, :, :, 32), &
         preci3d=prc(:, :, :, 1), precs3d=prc(:, :, :, 2),               &
         precg3d=prc(:, :, :, 3), prech3d=prc(:, :, :, 4),               &
         precr3d=prc(:, :, :, 5))
    do f = 1, nlev
      write(uo) lev(:, 1:nz, 1, f)
    end do
    do f = 1, nprec
      write(uo) prc(:, 1:nz, 1, f)
    end do
    do f = 1, nsfc
      write(uo) sfc(:, 1, f)
    end do
  end do
  close(uo)
end program run_gsfcgce4ice
