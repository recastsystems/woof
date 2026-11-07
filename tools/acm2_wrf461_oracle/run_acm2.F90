! Column driver for the ACM2 (Pleim) PBL oracle, WRF v4.6.1.
!
! Drives ACMPBL from the byte-pinned phys/module_bl_acm.F, compiled through
! WRF's own preprocessing pipeline and WRF's own FCOPTIM (build.sh), over a
! fixture of independent columns laid out as one WRF tile row (i = column,
! j = 1, k = 1..nz with kme = nz+1 as WRF's staggered memory dimension).
!
! Every step records the evolving state the scheme was handed and every
! output it returned, then advances the state with the tendencies (float32,
! the same arithmetic a forward Euler step would do).  The GPU test replays
! each recorded step from its recorded inputs, so the fixture's own state
! update is never part of the comparison.
!
! Input file (stream, little endian):
!   int32  ncol, nz, nsteps, xtime0
!   real32 dtpbl, ep1, g, rovcp, rd, cpd
!   real32 [nfield3=11][ncol][nz]  u v p dz8w th t qv qc qi rho pi
!   real32 [nfield2=12][ncol]      ust hfx qfx tsk psfc wspd psim gz1oz0
!                                  mut pblh regime rmol
! Output file (stream, little endian):
!   int32  ncol, nz, nsteps, xtime0
!   per step:
!     int32  xtime
!     real32 [7][ncol][nz]  u v th t qv qc qi        (state handed in)
!     real32 [3][ncol]      pblh regime rmol         (carried 2-D handed in)
!     real32 [8][ncol][nz]  rublten rvblten rthblten rqvblten rqcblten
!                           rqiblten exch_h exch_m   (returned)
!     real32 [3][ncol]      pblh regime rmol         (returned)
!     int32  [ncol]         kpbl                     (returned)
program run_acm2
  use module_bl_acm, only: acmpbl
  implicit none

  integer :: ncol, nz, nsteps, xtime0, s, xt, i, k
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  real :: dtpbl, ep1, g, rovcp, rd, cpd
  character(len=1024) :: inpath, outpath
  real, allocatable :: u(:,:,:), v(:,:,:), pp(:,:,:), dz8w(:,:,:), th(:,:,:)
  real, allocatable :: t(:,:,:), qv(:,:,:), qc(:,:,:), qi(:,:,:), rho(:,:,:)
  real, allocatable :: pii(:,:,:)
  real, allocatable :: rublten(:,:,:), rvblten(:,:,:), rthblten(:,:,:)
  real, allocatable :: rqvblten(:,:,:), rqcblten(:,:,:), rqiblten(:,:,:)
  real, allocatable :: exch_h(:,:,:), exch_m(:,:,:)
  real, allocatable :: ust(:,:), hfx(:,:), qfx(:,:), tsk(:,:), psfc(:,:)
  real, allocatable :: wspd(:,:), psim(:,:), gz1oz0(:,:), mut(:,:)
  real, allocatable :: pblh(:,:), regime(:,:), rmol(:,:)
  integer, allocatable :: kpbl(:,:)
  real, allocatable :: tmp(:,:), tmp2(:)

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)

  open(10, file=trim(inpath), access='stream', form='unformatted', &
       status='old', action='read')
  read(10) ncol, nz, nsteps, xtime0
  read(10) dtpbl, ep1, g, rovcp, rd, cpd

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

  allocate(u(ims:ime,kms:kme,jms:jme), v(ims:ime,kms:kme,jms:jme))
  allocate(pp(ims:ime,kms:kme,jms:jme), dz8w(ims:ime,kms:kme,jms:jme))
  allocate(th(ims:ime,kms:kme,jms:jme), t(ims:ime,kms:kme,jms:jme))
  allocate(qv(ims:ime,kms:kme,jms:jme), qc(ims:ime,kms:kme,jms:jme))
  allocate(qi(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme))
  allocate(pii(ims:ime,kms:kme,jms:jme))
  allocate(rublten(ims:ime,kms:kme,jms:jme), rvblten(ims:ime,kms:kme,jms:jme))
  allocate(rthblten(ims:ime,kms:kme,jms:jme), rqvblten(ims:ime,kms:kme,jms:jme))
  allocate(rqcblten(ims:ime,kms:kme,jms:jme), rqiblten(ims:ime,kms:kme,jms:jme))
  allocate(exch_h(ims:ime,kms:kme,jms:jme), exch_m(ims:ime,kms:kme,jms:jme))
  allocate(ust(ims:ime,jms:jme), hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme))
  allocate(tsk(ims:ime,jms:jme), psfc(ims:ime,jms:jme), wspd(ims:ime,jms:jme))
  allocate(psim(ims:ime,jms:jme), gz1oz0(ims:ime,jms:jme), mut(ims:ime,jms:jme))
  allocate(pblh(ims:ime,jms:jme), regime(ims:ime,jms:jme), rmol(ims:ime,jms:jme))
  allocate(kpbl(ims:ime,jms:jme))
  allocate(tmp(nz,ncol), tmp2(ncol))

  ! Memory beyond kte (WRF's staggered top) is never read by the scheme;
  ! it is zeroed so the fixture is deterministic.
  u = 0.; v = 0.; pp = 0.; dz8w = 0.; th = 0.; t = 0.; qv = 0.; qc = 0.
  qi = 0.; rho = 0.; pii = 0.
  rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.; rqcblten = 0.
  rqiblten = 0.; exch_h = 0.; exch_m = 0.; kpbl = 0

  call read3(u); call read3(v); call read3(pp); call read3(dz8w)
  call read3(th); call read3(t); call read3(qv); call read3(qc)
  call read3(qi); call read3(rho); call read3(pii)
  call read2(ust); call read2(hfx); call read2(qfx); call read2(tsk)
  call read2(psfc); call read2(wspd); call read2(psim); call read2(gz1oz0)
  call read2(mut); call read2(pblh); call read2(regime); call read2(rmol)
  close(10)

  open(20, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', action='write')
  write(20) ncol, nz, nsteps, xtime0

  do s = 1, nsteps
    xt = xtime0 + s - 1
    write(20) xt
    call write3(u); call write3(v); call write3(th); call write3(t)
    call write3(qv); call write3(qc); call write3(qi)
    call write2(pblh); call write2(regime); call write2(rmol)

    call acmpbl(XTIME=xt, DTPBL=dtpbl, U3D=u, V3D=v,                    &
                PP3D=pp, DZ8W=dz8w, TH3D=th, T3D=t,                     &
                QV3D=qv, QC3D=qc, QI3D=qi, RR3D=rho,                    &
                UST=ust, HFX=hfx, QFX=qfx, TSK=tsk,                     &
                PSFC=psfc, EP1=ep1, G=g,                                &
                ROVCP=rovcp, RD=rd, CPD=cpd,                            &
                PBLH=pblh, KPBL2D=kpbl, EXCH_H=exch_h, EXCH_M=exch_m,   &
                REGIME=regime,                                          &
                GZ1OZ0=gz1oz0, WSPD=wspd, PSIM=psim, MUT=mut, RMOL=rmol,&
                RUBLTEN=rublten, RVBLTEN=rvblten, RTHBLTEN=rthblten,    &
                RQVBLTEN=rqvblten, RQCBLTEN=rqcblten, RQIBLTEN=rqiblten,&
                ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,   &
                ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,   &
                its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

    call write3(rublten); call write3(rvblten); call write3(rthblten)
    call write3(rqvblten); call write3(rqcblten); call write3(rqiblten)
    call write3(exch_h); call write3(exch_m)
    call write2(pblh); call write2(regime); call write2(rmol)
    write(20) (kpbl(i,1), i = 1, ncol)

    ! Advance the fixture state (not part of any comparison).
    do k = kts, kte
      do i = its, ite
        u(i,k,1)  = u(i,k,1)  + rublten(i,k,1)  * dtpbl
        v(i,k,1)  = v(i,k,1)  + rvblten(i,k,1)  * dtpbl
        th(i,k,1) = th(i,k,1) + rthblten(i,k,1) * dtpbl
        t(i,k,1)  = th(i,k,1) * pii(i,k,1)
        qv(i,k,1) = max(qv(i,k,1) + rqvblten(i,k,1) * dtpbl, 0.)
        qc(i,k,1) = max(qc(i,k,1) + rqcblten(i,k,1) * dtpbl, 0.)
        qi(i,k,1) = max(qi(i,k,1) + rqiblten(i,k,1) * dtpbl, 0.)
      end do
    end do
  end do
  close(20)

contains

  subroutine read3(a)
    real, intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    integer :: ii, kk
    read(10) tmp
    do ii = 1, ncol
      do kk = 1, nz
        a(ii,kk,1) = tmp(kk,ii)
      end do
    end do
  end subroutine read3

  subroutine read2(a)
    real, intent(inout) :: a(ims:ime,jms:jme)
    integer :: ii
    read(10) tmp2
    do ii = 1, ncol
      a(ii,1) = tmp2(ii)
    end do
  end subroutine read2

  subroutine write3(a)
    real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
    integer :: ii, kk
    do ii = 1, ncol
      do kk = 1, nz
        tmp(kk,ii) = a(ii,kk,1)
      end do
    end do
    write(20) tmp
  end subroutine write3

  subroutine write2(a)
    real, intent(in) :: a(ims:ime,jms:jme)
    write(20) a(:,1)
  end subroutine write2

end program run_acm2
