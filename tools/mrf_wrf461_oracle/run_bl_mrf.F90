! Column oracle for WRF v4.6.1 phys/module_bl_mrf.F (bl_pbl_physics = 99).
!
! Reads the case file written by gen_mrf_cases.py (stream, little-endian
! float32/int32), drives WRF's own 3-D entry MRF once per column and per step
! with a single-column tile (its = ite = the column), and writes every input
! and every output word of every step to a stream file that pack_mrf_fixture.py
! turns into the packaged CSV fixture.
!
! The scheme is NOT recompiled here: build.sh links this driver against the
! module_bl_mrf.o that WRF's own build produced (the scalar referee build,
! FCOPTIM -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), and, as a
! second arm, against a rebuild at the same scalar flags.  Both arms
! must write identical bytes.  The separate -O0 arm is diagnostic only
! (build.sh records its different words).
!
! Memory layout mirrors WRF's driver: ims:ime = 1:ncase, kms:kme = 1:nz+1,
! jms:jme = 1:1, kts:kte = 1:nz (module_first_rk_step_part1.F passes
! kte = min(k_end, kde-1)), so MRF's NK = kme-K maps K = 1..nz onto the
! mass levels nz..1 exactly as in a real run.
!
! Between steps the driver advances its own copy of the state with the
! tendencies it just received (float32, explicit, no contraction: the driver
! is compiled with -ffp-contract=off), so later steps see columns the scheme
! itself has mixed.  The fixture records each step's INPUTS as well as its
! outputs, so the GPU harness never has to reproduce this update.
program run_bl_mrf
  use module_bl_mrf, only: mrf
  implicit none

  integer :: ncase, nz, nstep, magic
  integer :: c, k, s, kme
  real, allocatable :: u(:,:,:), v(:,:,:), th(:,:,:), t(:,:,:), qv(:,:,:)
  real, allocatable :: qc(:,:,:), qi(:,:,:), p(:,:,:), pii(:,:,:)
  real, allocatable :: dz8w(:,:,:), z(:,:,:)
  real, allocatable :: ru(:,:,:), rv(:,:,:), rth(:,:,:), rqv(:,:,:)
  real, allocatable :: rqc(:,:,:), rqi(:,:,:)
  real, allocatable :: dt(:), xland(:,:), psfc(:,:), znt(:,:), ust(:,:)
  real, allocatable :: psim(:,:), psih(:,:), hfx(:,:), qfx(:,:), tsk(:,:)
  real, allocatable :: gz1oz0(:,:), wspd(:,:), br(:,:), zol(:,:)
  real, allocatable :: regime(:,:), hol(:,:), pbl(:,:)
  integer, allocatable :: flagqi(:), kpbl(:,:)
  logical :: flag_qi
  real, parameter :: sentinel = 12345.0
  ! WRF share/module_model_constants.F, the values module_pbl_driver.F
  ! passes (cp, g, rcp, r_d, rovg, xlv, r_v, svp*, ep_1, ep_2, karman,
  ! eomeg, stbolt, p1000mb), spelled as WRF spells them so gfortran folds
  ! them the same way.
  real, parameter :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  real, parameter :: rcp = r_d/cp, rovg = r_d/g, xlv = 2.5e6
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, ep_1 = r_v/r_d-1., ep_2 = r_d/r_v
  real, parameter :: karman = 0.4, eomeg = 7.2921e-5, stbolt = 5.67051e-8
  real, parameter :: p1000mb = 100000.
  real :: dtmin

  open(10, file='mrf-cases.bin', access='stream', form='unformatted', &
       status='old', action='read')
  read(10) magic, ncase, nz, nstep
  if (magic /= 1297237572) stop 'bad case-file magic'
  kme = nz + 1
  allocate(u(ncase,kme,1), v(ncase,kme,1), th(ncase,kme,1), t(ncase,kme,1))
  allocate(qv(ncase,kme,1), qc(ncase,kme,1), qi(ncase,kme,1))
  allocate(p(ncase,kme,1), pii(ncase,kme,1), dz8w(ncase,kme,1), z(ncase,kme,1))
  allocate(ru(ncase,kme,1), rv(ncase,kme,1), rth(ncase,kme,1))
  allocate(rqv(ncase,kme,1), rqc(ncase,kme,1), rqi(ncase,kme,1))
  allocate(dt(ncase), flagqi(ncase), kpbl(ncase,1))
  allocate(xland(ncase,1), psfc(ncase,1), znt(ncase,1), ust(ncase,1))
  allocate(psim(ncase,1), psih(ncase,1), hfx(ncase,1), qfx(ncase,1))
  allocate(tsk(ncase,1), gz1oz0(ncase,1), wspd(ncase,1), br(ncase,1))
  allocate(zol(ncase,1), regime(ncase,1), hol(ncase,1), pbl(ncase,1))

  ! Level kme is memory WRF never reads for a mass field; fill it with the
  ! sentinel so a read of it would show.
  u = sentinel; v = sentinel; t = sentinel; qv = sentinel; qc = sentinel
  qi = sentinel; p = sentinel; pii = sentinel; dz8w = sentinel; z = sentinel
  do c = 1, ncase
    read(10) dt(c), flagqi(c), xland(c,1), psfc(c,1), znt(c,1), ust(c,1), &
             psim(c,1), psih(c,1), hfx(c,1), qfx(c,1), tsk(c,1),          &
             gz1oz0(c,1), wspd(c,1), br(c,1), zol(c,1), regime(c,1),      &
             hol(c,1), pbl(c,1)
    read(10) (u(c,k,1), k=1,nz)
    read(10) (v(c,k,1), k=1,nz)
    read(10) (t(c,k,1), k=1,nz)
    read(10) (qv(c,k,1), k=1,nz)
    read(10) (qc(c,k,1), k=1,nz)
    read(10) (qi(c,k,1), k=1,nz)
    read(10) (p(c,k,1), k=1,nz)
    read(10) (pii(c,k,1), k=1,nz)
    read(10) (dz8w(c,k,1), k=1,nz)
    read(10) (z(c,k,1), k=1,nz)
  end do
  close(10)
  kpbl = -999

  open(20, file='mrf-oracle-out.bin', access='stream', form='unformatted', &
       status='replace', action='write')
  write(20) magic, ncase, nz, nstep
  do s = 1, nstep
    do c = 1, ncase
      ! th is MRF's TH3D dummy; MRF2D never reads it (module_bl_mrf.F:209
      ! does not even pass it on), but WRF passes th_phy, so build it.
      do k = 1, nz
        th(c,k,1) = t(c,k,1)/pii(c,k,1)
      end do
      ! The record of this step's inputs, before the call can touch any
      ! INOUT argument.
      write(20) s, c, dt(c), flagqi(c), xland(c,1), psfc(c,1), znt(c,1),  &
                ust(c,1), psim(c,1), psih(c,1), hfx(c,1), qfx(c,1),       &
                tsk(c,1), gz1oz0(c,1), wspd(c,1), br(c,1), zol(c,1),      &
                regime(c,1), hol(c,1), pbl(c,1)
      write(20) (u(c,k,1), k=1,nz), (v(c,k,1), k=1,nz), (t(c,k,1), k=1,nz), &
                (qv(c,k,1), k=1,nz), (qc(c,k,1), k=1,nz),                  &
                (qi(c,k,1), k=1,nz), (p(c,k,1), k=1,nz),                   &
                (pii(c,k,1), k=1,nz), (dz8w(c,k,1), k=1,nz),               &
                (z(c,k,1), k=1,nz)
      ru(c,:,1) = sentinel; rv(c,:,1) = sentinel; rth(c,:,1) = sentinel
      rqv(c,:,1) = sentinel; rqc(c,:,1) = sentinel; rqi(c,:,1) = sentinel
      flag_qi = flagqi(c) /= 0
      dtmin = dt(c)/60.
      call mrf(U3D=u, V3D=v, TH3D=th, T3D=t, QV3D=qv, QC3D=qc, QI3D=qi,    &
               P3D=p, PI3D=pii,                                            &
               RUBLTEN=ru, RVBLTEN=rv, RTHBLTEN=rth, RQVBLTEN=rqv,         &
               RQCBLTEN=rqc, RQIBLTEN=rqi,                                 &
               CP=cp, G=g, ROVCP=rcp, R=r_d, ROVG=rovg,                    &
               DZ8W=dz8w, Z=z, XLV=xlv, RV=r_v, PSFC=psfc,                 &
               P1000MB=p1000mb,                                            &
               ZNT=znt, UST=ust, ZOL=zol, HOL=hol, PBL=pbl,                &
               PSIM=psim, PSIH=psih, XLAND=xland, HFX=hfx, QFX=qfx,        &
               TSK=tsk, GZ1OZ0=gz1oz0, WSPD=wspd, BR=br,                   &
               DT=dt(c), DTMIN=dtmin, KPBL2D=kpbl,                         &
               SVP1=svp1, SVP2=svp2, SVP3=svp3, SVPT0=svpt0, EP1=ep_1,     &
               EP2=ep_2, KARMAN=karman, EOMEG=eomeg, STBOLT=stbolt,        &
               REGIME=regime, FLAG_QI=flag_qi,                             &
               IDS=1, IDE=ncase+1, JDS=1, JDE=2, KDS=1, KDE=kme,           &
               IMS=1, IME=ncase, JMS=1, JME=1, KMS=1, KME=kme,             &
               ITS=c, ITE=c, JTS=1, JTE=1, KTS=1, KTE=nz)
      write(20) (ru(c,k,1), k=1,kme), (rv(c,k,1), k=1,kme),               &
                (rth(c,k,1), k=1,kme), (rqv(c,k,1), k=1,kme),             &
                (rqc(c,k,1), k=1,kme), (rqi(c,k,1), k=1,kme)
      write(20) ust(c,1), hol(c,1), pbl(c,1), kpbl(c,1), zol(c,1),        &
                regime(c,1), wspd(c,1)
      ! Advance the driver's own state (explicit, float32).  qc/qi are kept
      ! non-negative the way WRF's moisture update leaves them after its
      ! positive-definite step; nothing else is clipped.
      do k = 1, nz
        u(c,k,1) = u(c,k,1) + dt(c)*ru(c,k,1)
        v(c,k,1) = v(c,k,1) + dt(c)*rv(c,k,1)
        t(c,k,1) = t(c,k,1) + dt(c)*(rth(c,k,1)*pii(c,k,1))
        qv(c,k,1) = max(qv(c,k,1) + dt(c)*rqv(c,k,1), 0.)
        qc(c,k,1) = max(qc(c,k,1) + dt(c)*rqc(c,k,1), 0.)
        if (flag_qi) qi(c,k,1) = max(qi(c,k,1) + dt(c)*rqi(c,k,1), 0.)
      end do
    end do
  end do
  close(20)
end program run_bl_mrf
