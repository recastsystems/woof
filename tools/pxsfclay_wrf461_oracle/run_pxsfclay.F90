program run_pxsfclay_oracle
  ! Drive WRF v4.6.1's Pleim-Xiu surface layer over a set of columns for
  ! several steps and dump every word it writes.
  !
  ! mode 0 calls PXSFCLAY (phys/module_sf_pxsfclay.F, unmodified) exactly
  ! as module_surface_driver.F:2163 does; mode 1 calls
  ! PXSFCLAY_SEAICE_WRAPPER (module_surface_driver.F:6687, copied verbatim
  ! by extract.py) exactly as :2149 does under FRACTIONAL_SEAICE = 1.
  ! The physical constants are WRF's own PARAMETERs from
  ! share/module_model_constants.F, compiled from the same pinned tree.
  !
  ! One j row, i = 1..ncol, one model level (the scheme reads k = 1 only).
  ! Between steps the INOUT state carries over untouched, except that an
  ! optional per-column "LSM" override replaces QSFC, HFX, QFX and ZNT, the
  ! four fields a land model writes between two surface-layer calls.
  !
  ! File formats (little-endian, stream):
  !   input : int32 magic,version,ncol,nstep,mode,itstart,tice2tsk,pad
  !           real32 dx, xice_threshold
  !           real32 state0(ncol, NSTATE)
  !           per step: real32 forcing(ncol, NFORCE), override(ncol, NOVR)
  !   output: int32 magic,ncol,nstep,NOUT then per step real32 out(ncol,NOUT)
  use module_model_constants, only: cp, g, rcp, r_d, xlv, svp1, svp2, &
                                    svp3, svpt0, ep_1, ep_2, karman
  use module_sf_pxsfclay, only: pxsfclay
  use px_seaice_oracle, only: pxsfclay_seaice_wrapper
  implicit none

  integer, parameter :: NSTATE = 23, NFORCE = 13, NOVR = 5, NSEA = 12
  integer, parameter :: NOUT = NSTATE + 2 + NSEA
  integer, parameter :: MAGIC_IN = 1347965766   ! 'PXSF'
  integer, parameter :: MAGIC_OUT = 1347965775

  integer :: hdr(8), ncol, nstep, mode, itstart, step, itimestep, isfflx, k
  logical :: tice2tsk
  real :: hdrf(2), dx, xice_threshold
  character(len=1024) :: in_path, out_path

  real, allocatable :: u3d(:,:,:), v3d(:,:,:), t3d(:,:,:), th3d(:,:,:)
  real, allocatable :: qv3d(:,:,:), p3d(:,:,:), dz8w(:,:,:)
  real, allocatable :: psfc(:,:), tsk(:,:), pblh(:,:), mavail(:,:)
  real, allocatable :: xland(:,:), xice(:,:)
  real, allocatable :: regime(:,:), hfx(:,:), qfx(:,:), lh(:,:), mol(:,:)
  real, allocatable :: rmol(:,:), qsfc(:,:), gz1oz0(:,:), wspd(:,:), br(:,:)
  real, allocatable :: psim(:,:), psih(:,:), znt(:,:), zol(:,:), ust(:,:)
  real, allocatable :: cpm(:,:), chs2(:,:), cqs2(:,:), chs(:,:), flhc(:,:)
  real, allocatable :: flqc(:,:), qgh(:,:), sst(:,:), u10(:,:), v10(:,:)
  real, allocatable :: chs_sea(:,:), chs2_sea(:,:), cpm_sea(:,:)
  real, allocatable :: cqs2_sea(:,:), flhc_sea(:,:), flqc_sea(:,:)
  real, allocatable :: hfx_sea(:,:), lh_sea(:,:), qfx_sea(:,:)
  real, allocatable :: qgh_sea(:,:), qsfc_sea(:,:), tsk_sea(:,:)
  real, allocatable :: state(:,:), forcing(:,:), ovr(:,:), out(:,:)

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_pxsfclay INPUT.bin OUTPUT.bin'
    error stop 2
  end if

  open(10, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(10) hdr
  read(10) hdrf
  if (hdr(1) /= MAGIC_IN .or. hdr(2) /= 1) error stop 3
  ncol = hdr(3); nstep = hdr(4); mode = hdr(5); itstart = hdr(6)
  tice2tsk = hdr(7) /= 0
  dx = hdrf(1); xice_threshold = hdrf(2)
  isfflx = 1

  allocate(u3d(ncol,1,1), v3d(ncol,1,1), t3d(ncol,1,1), th3d(ncol,1,1))
  allocate(qv3d(ncol,1,1), p3d(ncol,1,1), dz8w(ncol,1,1))
  allocate(psfc(ncol,1), tsk(ncol,1), pblh(ncol,1), mavail(ncol,1))
  allocate(xland(ncol,1), xice(ncol,1))
  allocate(regime(ncol,1), hfx(ncol,1), qfx(ncol,1), lh(ncol,1), mol(ncol,1))
  allocate(rmol(ncol,1), qsfc(ncol,1), gz1oz0(ncol,1), wspd(ncol,1), br(ncol,1))
  allocate(psim(ncol,1), psih(ncol,1), znt(ncol,1), zol(ncol,1), ust(ncol,1))
  allocate(cpm(ncol,1), chs2(ncol,1), cqs2(ncol,1), chs(ncol,1), flhc(ncol,1))
  allocate(flqc(ncol,1), qgh(ncol,1), sst(ncol,1), u10(ncol,1), v10(ncol,1))
  allocate(chs_sea(ncol,1), chs2_sea(ncol,1), cpm_sea(ncol,1))
  allocate(cqs2_sea(ncol,1), flhc_sea(ncol,1), flqc_sea(ncol,1))
  allocate(hfx_sea(ncol,1), lh_sea(ncol,1), qfx_sea(ncol,1))
  allocate(qgh_sea(ncol,1), qsfc_sea(ncol,1), tsk_sea(ncol,1))
  allocate(state(ncol, NSTATE), forcing(ncol, NFORCE), ovr(ncol, NOVR))
  allocate(out(ncol, NOUT))

  read(10) state
  regime(:,1) = state(:,1);  hfx(:,1) = state(:,2);    qfx(:,1) = state(:,3)
  lh(:,1) = state(:,4);      mol(:,1) = state(:,5);    rmol(:,1) = state(:,6)
  qsfc(:,1) = state(:,7);    gz1oz0(:,1) = state(:,8); wspd(:,1) = state(:,9)
  br(:,1) = state(:,10);     psim(:,1) = state(:,11);  psih(:,1) = state(:,12)
  znt(:,1) = state(:,13);    zol(:,1) = state(:,14);   ust(:,1) = state(:,15)
  cpm(:,1) = state(:,16);    chs2(:,1) = state(:,17);  cqs2(:,1) = state(:,18)
  chs(:,1) = state(:,19);    flhc(:,1) = state(:,20);  flqc(:,1) = state(:,21)
  qgh(:,1) = state(:,22);    sst(:,1) = state(:,23)

  open(20, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  write(20) MAGIC_OUT, ncol, nstep, NOUT

  do step = 1, nstep
    itimestep = itstart + step - 1
    read(10) forcing
    read(10) ovr
    u3d(:,1,1) = forcing(:,1);  v3d(:,1,1) = forcing(:,2)
    t3d(:,1,1) = forcing(:,3);  th3d(:,1,1) = forcing(:,4)
    qv3d(:,1,1) = forcing(:,5); p3d(:,1,1) = forcing(:,6)
    dz8w(:,1,1) = forcing(:,7); psfc(:,1) = forcing(:,8)
    tsk(:,1) = forcing(:,9);    pblh(:,1) = forcing(:,10)
    mavail(:,1) = forcing(:,11); xland(:,1) = forcing(:,12)
    xice(:,1) = forcing(:,13)
    if (step > 1) then
      ! The land model's writes between two surface-layer calls.
      where (ovr(:,1) > 0.5)
        qsfc(:,1) = ovr(:,2)
        hfx(:,1) = ovr(:,3)
        qfx(:,1) = ovr(:,4)
        znt(:,1) = ovr(:,5)
      end where
    end if

    u10 = 0.; v10 = 0.
    chs_sea = 0.; chs2_sea = 0.; cpm_sea = 0.; cqs2_sea = 0.
    flhc_sea = 0.; flqc_sea = 0.; hfx_sea = 0.; lh_sea = 0.
    qfx_sea = 0.; qgh_sea = 0.; qsfc_sea = 0.; tsk_sea = 0.

    if (mode == 0) then
      call pxsfclay(u3d, v3d, t3d, th3d, qv3d, p3d, dz8w,              &
                    cp, g, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,  &
                    znt, ust, pblh, mavail, zol, mol, regime, psim,    &
                    psih, xland, hfx, qfx, lh, tsk, flhc, flqc, qgh,   &
                    qsfc, rmol, u10, v10, gz1oz0, wspd, br, isfflx,    &
                    dx, svp1, svp2, svp3, svpt0, ep_1, ep_2, karman,   &
                    itimestep,                                         &
                    1, ncol, 1, 1, 1, 1,                               &
                    1, ncol, 1, 1, 1, 1,                               &
                    1, ncol, 1, 1, 1, 1)
    else
      call pxsfclay_seaice_wrapper(u3d, v3d, t3d, th3d, qv3d, p3d,     &
                    dz8w, cp, g, rcp, r_d, xlv, psfc, chs, chs2, cqs2, &
                    cpm, znt, ust, pblh, mavail, zol, mol, regime,     &
                    psim, psih, xland, hfx, qfx, lh, tsk, flhc, flqc,  &
                    qgh, qsfc, rmol, u10, v10, gz1oz0, wspd, br,       &
                    isfflx, dx, svp1, svp2, svp3, svpt0, ep_1, ep_2,   &
                    karman, xice, sst, itimestep, tice2tsk,            &
                    xice_threshold, chs_sea, chs2_sea, cpm_sea,        &
                    cqs2_sea, flhc_sea, flqc_sea, hfx_sea, lh_sea,     &
                    qfx_sea, qgh_sea, qsfc_sea, tsk_sea,               &
                    1, ncol, 1, 1, 1, 1,                               &
                    1, ncol, 1, 1, 1, 1,                               &
                    1, ncol, 1, 1, 1, 1)
    end if

    out(:,1) = regime(:,1);  out(:,2) = hfx(:,1);    out(:,3) = qfx(:,1)
    out(:,4) = lh(:,1);      out(:,5) = mol(:,1);    out(:,6) = rmol(:,1)
    out(:,7) = qsfc(:,1);    out(:,8) = gz1oz0(:,1); out(:,9) = wspd(:,1)
    out(:,10) = br(:,1);     out(:,11) = psim(:,1);  out(:,12) = psih(:,1)
    out(:,13) = znt(:,1);    out(:,14) = zol(:,1);   out(:,15) = ust(:,1)
    out(:,16) = cpm(:,1);    out(:,17) = chs2(:,1);  out(:,18) = cqs2(:,1)
    out(:,19) = chs(:,1);    out(:,20) = flhc(:,1);  out(:,21) = flqc(:,1)
    out(:,22) = qgh(:,1);    out(:,23) = sst(:,1)
    out(:,24) = u10(:,1);    out(:,25) = v10(:,1)
    out(:,26) = chs_sea(:,1);  out(:,27) = chs2_sea(:,1)
    out(:,28) = cpm_sea(:,1);  out(:,29) = cqs2_sea(:,1)
    out(:,30) = flhc_sea(:,1); out(:,31) = flqc_sea(:,1)
    out(:,32) = hfx_sea(:,1);  out(:,33) = lh_sea(:,1)
    out(:,34) = qfx_sea(:,1);  out(:,35) = qgh_sea(:,1)
    out(:,36) = qsfc_sea(:,1); out(:,37) = tsk_sea(:,1)
    write(20) out
  end do
  close(10)
  close(20)
  k = 0
end program run_pxsfclay_oracle
