program run_shinhong480_columns
  ! Drive the native WRF v4.8.0 Shin-Hong objects (phys/module_bl_shinhong.o
  ! and phys/physics_mmm/bl_shinhong.o, linked unmodified from the strict
  ! referee build) through the wrapper module_pbl_driver.F calls, one column
  ! per call, over every record of a v2 column file, and dump every output
  ! as raw float32 (no text round trip: NaN payloads and signed zeros
  ! survive).
  !
  ! The call is module_pbl_driver.F's (WRF v4.8.0, :1328-1365):
  !   * t3d is t_phy = th_phy*pi_phy, the float32 product phy_prep forms;
  !     the record carries it;
  !   * psfc = p8w(kms); dx = dx2d = dx (compute_2d_dx_area);
  !   * flag_qc = flag_qi = .true.; flag_bep = .false., idiff = 0 and the
  !     BEP arrays absent, as for every run without sf_urban_physics = 2/3;
  !   * ctopo/ctopo2 = 1 unless the record carries topo_wind coefficients;
  !   * uoce/voce from the record (zero for an uncoupled run);
  !   * rthraten from the record (read only by shinhong_scu_mixing).
  !
  ! Input record (little endian, stream):
  !   int32[8]      nz, flags, family, base, topo, case, 0, 0
  !                 flags: 1 nonlocal_flux, 2 scu_mixing, 4 ke_dissipation
  !   float32[20]   dt dx dy psfc znt ust hfx qfx wspd br psim psih xland
  !                 u10 v10 corf ctopo ctopo2 uoce voce
  !   float32[nz]   u v theta pi t qv qc qi p dz tke rthraten
  !   float32[nz+1] p_interface
  ! Output record:
  !   int32[4]      nz, kpbl, family, base
  !   float32[nz]   rublten rvblten rthblten rqvblten rqcblten rqiblten
  !                 exch_h exch_m tke el
  !   float32[5]    hpbl wstar delta u10 v10
  use module_bl_shinhong, only: shinhong
  implicit none

  ! WRF module_model_constants, default single-precision build.
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.0
  real, parameter :: r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0
  real, parameter :: rcp = r_d / cp
  real, parameter :: rovg = r_d / g
  real, parameter :: xlv = 2.5e6
  real, parameter :: ep_1 = r_v / r_d - 1.0
  real, parameter :: ep_2 = r_d / r_v
  real, parameter :: karman = 0.4

  integer :: hdr(8), nz, flags, ios, uin, uout, k, ncol
  real :: sc(20)
  real, allocatable :: lv(:, :), pint(:)
  real, allocatable, dimension(:, :, :) :: u3d, v3d, t3d, qv3d, qc3d, &
       qi3d, p3d, p3di, pi3d, dz8w, rthraten, rublten, rvblten, rthblten, &
       rqvblten, rqcblten, rqiblten, exch_h, exch_m, tke, el
  real, dimension(1, 1) :: psfc, znt, ust, hpbl, psim, psih, xland, hfx, &
       qfx, wspd, br, u10, v10, uoce, voce, ctopo, ctopo2, corf, dx2d, &
       wstar, delta
  integer, dimension(1, 1) :: kpbl2d
  real :: dtstep
  logical :: nonlocal, scu, dissip
  character(len=1024) :: in_path, out_path
  character(len=256) :: errmsg
  integer :: errflg

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_shinhong480_columns IN.bin OUT.bin'
    error stop 2
  end if
  open(newunit=uin, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  open(newunit=uout, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write')

  ncol = 0
  do
    read(uin, iostat=ios) hdr
    if (ios /= 0) exit
    nz = hdr(1)
    flags = hdr(2)
    nonlocal = iand(flags, 1) /= 0
    scu = iand(flags, 2) /= 0
    dissip = iand(flags, 4) /= 0
    read(uin) sc
    allocate(lv(nz, 12), pint(nz + 1))
    read(uin) lv
    read(uin) pint

    allocate(u3d(1, nz + 1, 1), v3d(1, nz + 1, 1), t3d(1, nz + 1, 1), &
             qv3d(1, nz + 1, 1), qc3d(1, nz + 1, 1), qi3d(1, nz + 1, 1), &
             p3d(1, nz + 1, 1), p3di(1, nz + 1, 1), pi3d(1, nz + 1, 1), &
             dz8w(1, nz + 1, 1), rthraten(1, nz + 1, 1), &
             rublten(1, nz + 1, 1), rvblten(1, nz + 1, 1), &
             rthblten(1, nz + 1, 1), rqvblten(1, nz + 1, 1), &
             rqcblten(1, nz + 1, 1), rqiblten(1, nz + 1, 1), &
             exch_h(1, nz + 1, 1), exch_m(1, nz + 1, 1), &
             tke(1, nz + 1, 1), el(1, nz + 1, 1))

    ! level nz+1 of every mass array is outside kts:kte and never read
    u3d = 0.0; v3d = 0.0; t3d = 0.0; qv3d = 0.0; qc3d = 0.0; qi3d = 0.0
    p3d = 0.0; pi3d = 1.0; dz8w = 0.0; rthraten = 0.0
    do k = 1, nz
      u3d(1, k, 1) = lv(k, 1)
      v3d(1, k, 1) = lv(k, 2)
      pi3d(1, k, 1) = lv(k, 4)
      t3d(1, k, 1) = lv(k, 5)
      qv3d(1, k, 1) = lv(k, 6)
      qc3d(1, k, 1) = lv(k, 7)
      qi3d(1, k, 1) = lv(k, 8)
      p3d(1, k, 1) = lv(k, 9)
      dz8w(1, k, 1) = lv(k, 10)
      rthraten(1, k, 1) = lv(k, 12)
    end do
    do k = 1, nz + 1
      p3di(1, k, 1) = pint(k)
    end do

    rublten = 0.0; rvblten = 0.0; rthblten = 0.0
    rqvblten = 0.0; rqcblten = 0.0; rqiblten = 0.0
    exch_h = 0.0; exch_m = 0.0; el = 0.0; tke = 0.0
    do k = 1, nz
      tke(1, k, 1) = lv(k, 11)
    end do

    dtstep = sc(1); dx2d = sc(2)
    psfc = sc(4); znt = sc(5); ust = sc(6); hfx = sc(7); qfx = sc(8)
    wspd = sc(9); br = sc(10); psim = sc(11); psih = sc(12)
    xland = sc(13); u10 = sc(14); v10 = sc(15); corf = sc(16)
    ctopo = sc(17); ctopo2 = sc(18); uoce = sc(19); voce = sc(20)
    hpbl = 0.0; kpbl2d = 0; wstar = 0.0; delta = 0.0

    call shinhong(u3d=u3d, v3d=v3d, t3d=t3d,                                 &
                  qv3d=qv3d, qc3d=qc3d, qi3d=qi3d,                            &
                  p3d=p3d, p3di=p3di, pi3d=pi3d,                              &
                  rublten=rublten, rvblten=rvblten, rthblten=rthblten,        &
                  rqvblten=rqvblten, rqcblten=rqcblten, rqiblten=rqiblten,    &
                  flag_qc=.true., flag_qi=.true.,                             &
                  cp=cp, g=g, rovcp=rcp, rd=r_d, rovg=rovg,                   &
                  ep1=ep_1, ep2=ep_2, karman=karman, xlv=xlv, rv=r_v,         &
                  dz8w=dz8w, psfc=psfc,                                       &
                  znt=znt, ust=ust, hpbl=hpbl, psim=psim, psih=psih,          &
                  xland=xland, hfx=hfx, qfx=qfx, wspd=wspd, br=br,            &
                  dt=dtstep, kpbl2d=kpbl2d,                                   &
                  exch_h=exch_h, exch_m=exch_m,                               &
                  wstar=wstar, delta=delta,                                   &
                  shinhong_nonlocal_flux=nonlocal,                            &
                  tke=tke, el=el, corf=corf,                                  &
                  u10=u10, v10=v10, uoce=uoce, voce=voce,                     &
                  rthraten=rthraten, shinhong_scu_mixing=scu,                 &
                  shinhong_dissi_heating=dissip,                              &
                  ctopo=ctopo, ctopo2=ctopo2, dx=dx2d,                        &
                  idiff=0, flag_bep=.false.,                                  &
                  ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz + 1,              &
                  ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1,              &
                  its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                  &
                  errmsg=errmsg, errflg=errflg)

    write(uout) nz, kpbl2d(1, 1), hdr(3), hdr(4)
    write(uout) rublten(1, 1:nz, 1)
    write(uout) rvblten(1, 1:nz, 1)
    write(uout) rthblten(1, 1:nz, 1)
    write(uout) rqvblten(1, 1:nz, 1)
    write(uout) rqcblten(1, 1:nz, 1)
    write(uout) rqiblten(1, 1:nz, 1)
    write(uout) exch_h(1, 1:nz, 1)
    write(uout) exch_m(1, 1:nz, 1)
    write(uout) tke(1, 1:nz, 1)
    write(uout) el(1, 1:nz, 1)
    write(uout) hpbl(1, 1), wstar(1, 1), delta(1, 1), u10(1, 1), v10(1, 1)

    deallocate(lv, pint, u3d, v3d, t3d, qv3d, qc3d, qi3d, p3d, p3di, pi3d, &
               dz8w, rthraten, rublten, rvblten, rthblten, rqvblten, &
               rqcblten, rqiblten, exch_h, exch_m, tke, el)
    ncol = ncol + 1
  end do
  close(uin)
  close(uout)
  write(*, '(A,I0,A)') 'ran ', ncol, ' columns'
end program run_shinhong480_columns
