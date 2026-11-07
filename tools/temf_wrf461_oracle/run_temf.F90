! Column driver for the WRF v4.6.1 TEMF pair: the surface layer
! (module_sf_temfsfclay.F, sf_sfclay_physics = 10) followed by the PBL
! (module_bl_temf.F, bl_pbl_physics = 10), stepped in WRF's order and
! through WRF's own entry points (temfsfclay, temfpbl), with WRF's own
! cold start (temfinit, temfsfclayinit).
!
!   run_temf <cases.bin> <out.bin>
!
! The carried TEMF state lives in this program's arrays across steps
! exactly as it lives on WRF's grid: every *_temf field, WM_TEMF, PBLH,
! KPBL, HFX/QFX, UST and the 10 m / 2 m diagnostics are passed back in
! unchanged from the previous call, so values a branch does not write
! (the updraft fields in a stable column, PBLH and KPBL outside the
! convective branch, MF_TEMF at the top level) keep last step's words as
! they do in WRF.
!
! Between steps the column is advanced by the PBL tendencies alone,
! x = x + dt*rx with WRF's float32 arithmetic, and a negative qv or qc
! is set to zero.  T is TH*PI at the top of every step.
! The WOOF harness (run_woof.py) does exactly the same words.
!
! File layout (little-endian stream, all REAL are float32):
!   header : 8 int32  (magic 'TEMF' = 1413827910, version 1, ncol, nz,
!                      nsteps, 3 spare) then dt (REAL)
!   inputs : ht xland mavail znt fcor psfc hfx0 qfx0 hpbl0   (ncol each)
!            tsk                                    (ncol x nsteps)
!            u v th qv qc qi p pi rho               (ncol x nz each)
!            z p8w                                  (ncol x (nz+1) each)
!   output : per step, the SFC2D, PBL3D, PBL2D and STATE blocks named in
!            run_woof.py (FIELDS_*), each field ncol (2-D) or ncol x nz.
program run_temf
  use module_bl_temf
  use module_sf_temfsfclay
  implicit none

  ! share/module_model_constants.F, as WRF's drivers pass them.
  real, parameter :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  real, parameter :: cpv = 4.*r_v, rcp = r_d/cp, xlv = 2.5E6
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, ep_1 = r_v/r_d-1., ep_2 = r_d/r_v
  real, parameter :: karman = 0.4, eomeg = 7.2921E-5, stbolt = 5.67051E-8

  integer :: hdr(8), ncol, nz, nsteps, n, i, k, ui, uo
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  real :: dt, dtmin, p_top
  character(len=512) :: fin, fout

  real, allocatable, dimension(:,:,:) :: u, v, th, t, qv, qc, qi, p, pii, &
       rho, z, p8w
  real, allocatable, dimension(:,:,:) :: rublten, rvblten, rthblten, &
       rqvblten, rqcblten, rqiblten
  real, allocatable, dimension(:,:,:) :: te_temf, kh_temf, km_temf, &
       shf_temf, qf_temf, uw_temf, vw_temf, wupd_temf, mf_temf, thup_temf, &
       qtup_temf, qlup_temf, cf3d_temf
  real, allocatable, dimension(:,:) :: ht, xland, mavail, znt, fcor, psfc, &
       hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, ust, chs, chs2, cqs2, cpm, &
       u10, v10, th2, t2, q2, exch_temf, wm_temf, hd_temf, lcl_temf, &
       hct_temf, cfm_temf, zol, hol, hpbl, psim, psih, gz1oz0, wspd, br
  real, allocatable :: tsk_steps(:,:)
  integer, allocatable :: kpbl(:,:)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=ui, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(ui) hdr
  if (hdr(1) /= 1413827910 .or. hdr(2) /= 1) stop 'bad cases header'
  ncol = hdr(3); nz = hdr(4); nsteps = hdr(5)
  read(ui) dt

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

  allocate(u(ncol,nz+1,1), v(ncol,nz+1,1), th(ncol,nz+1,1), &
       t(ncol,nz+1,1), qv(ncol,nz+1,1), qc(ncol,nz+1,1), qi(ncol,nz+1,1), &
       p(ncol,nz+1,1), pii(ncol,nz+1,1), rho(ncol,nz+1,1), z(ncol,nz+1,1), &
       p8w(ncol,nz+1,1))
  allocate(rublten(ncol,nz+1,1), rvblten(ncol,nz+1,1), &
       rthblten(ncol,nz+1,1), rqvblten(ncol,nz+1,1), rqcblten(ncol,nz+1,1), &
       rqiblten(ncol,nz+1,1))
  allocate(te_temf(ncol,nz+1,1), kh_temf(ncol,nz+1,1), km_temf(ncol,nz+1,1),&
       shf_temf(ncol,nz+1,1), qf_temf(ncol,nz+1,1), uw_temf(ncol,nz+1,1), &
       vw_temf(ncol,nz+1,1), wupd_temf(ncol,nz+1,1), mf_temf(ncol,nz+1,1), &
       thup_temf(ncol,nz+1,1), qtup_temf(ncol,nz+1,1), &
       qlup_temf(ncol,nz+1,1), cf3d_temf(ncol,nz+1,1))
  allocate(ht(ncol,1), xland(ncol,1), mavail(ncol,1), znt(ncol,1), &
       fcor(ncol,1), psfc(ncol,1), hfx(ncol,1), qfx(ncol,1), lh(ncol,1), &
       tsk(ncol,1), flhc(ncol,1), flqc(ncol,1), qgh(ncol,1), qsfc(ncol,1), &
       ust(ncol,1), chs(ncol,1), chs2(ncol,1), cqs2(ncol,1), cpm(ncol,1), &
       u10(ncol,1), v10(ncol,1), th2(ncol,1), t2(ncol,1), q2(ncol,1), &
       exch_temf(ncol,1), wm_temf(ncol,1), hd_temf(ncol,1), &
       lcl_temf(ncol,1), hct_temf(ncol,1), cfm_temf(ncol,1), zol(ncol,1), &
       hol(ncol,1), hpbl(ncol,1), psim(ncol,1), psih(ncol,1), &
       gz1oz0(ncol,1), wspd(ncol,1), br(ncol,1), kpbl(ncol,1))
  allocate(tsk_steps(ncol, nsteps))

  ! Registry cold start: every state field zero unless an init writes it.
  u = 0.; v = 0.; th = 0.; t = 0.; qv = 0.; qc = 0.; qi = 0.; p = 0.
  pii = 0.; rho = 0.; z = 0.; p8w = 0.
  rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.; rqcblten = 0.
  rqiblten = 0.
  te_temf = 0.; kh_temf = 0.; km_temf = 0.; shf_temf = 0.; qf_temf = 0.
  uw_temf = 0.; vw_temf = 0.; wupd_temf = 0.; mf_temf = 0.; thup_temf = 0.
  qtup_temf = 0.; qlup_temf = 0.; cf3d_temf = 0.
  lh = 0.; flhc = 0.; flqc = 0.; qgh = 0.; qsfc = 0.; ust = 0.; chs = 0.
  chs2 = 0.; cqs2 = 0.; cpm = 0.; u10 = 0.; v10 = 0.; th2 = 0.; t2 = 0.
  q2 = 0.; exch_temf = 0.; wm_temf = 0.; hd_temf = 0.; lcl_temf = 0.
  hct_temf = 0.; cfm_temf = 0.; zol = 0.; hol = 0.; psim = 0.; psih = 0.
  gz1oz0 = 0.; wspd = 0.; br = 0.; kpbl = 0

  read(ui) ht(:,1), xland(:,1), mavail(:,1), znt(:,1), fcor(:,1), &
       psfc(:,1), hfx(:,1), qfx(:,1), hpbl(:,1)
  read(ui) tsk_steps
  read(ui) u(:,1:nz,1), v(:,1:nz,1), th(:,1:nz,1), qv(:,1:nz,1), &
       qc(:,1:nz,1), qi(:,1:nz,1), p(:,1:nz,1), pii(:,1:nz,1), rho(:,1:nz,1)
  read(ui) z(:,:,1), p8w(:,:,1)
  close(ui)

  p_top = 0.
  dtmin = dt/60.

  ! WRF's cold start: module_physics_init.F CASE (TEMFPBLSCHEME) and
  ! CASE (TEMFSFCSCHEME), restart = .false.
  call temfinit(rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, &
       2, 2, .false., .true., te_temf, cf3d_temf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte)
  call temfsfclayinit(.false., .true., wm_temf, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte)

  open(newunit=uo, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')

  do n = 1, nsteps
    do k = 1, nz
      do i = 1, ncol
        t(i,k,1) = th(i,k,1)*pii(i,k,1)
      end do
    end do
    tsk(:,1) = tsk_steps(:,n)

    ! module_surface_driver.F:2407-2433, CASE (TEMFSFCSCHEME).
    call temfsfclay(u3d=u, v3d=v, th3d=th, qv3d=qv, p3d=p, pi3d=pii, &
         rho=rho, z=z, ht=ht, cp=cp, g=g, rovcp=rcp, r=r_d, xlv=xlv, &
         psfc=psfc, chs=chs, chs2=chs2, cqs2=cqs2, cpm=cpm, znt=znt, &
         ust=ust, mavail=mavail, xland=xland, hfx=hfx, qfx=qfx, lh=lh, &
         tsk=tsk, flhc=flhc, flqc=flqc, qgh=qgh, qsfc=qsfc, u10=u10, &
         v10=v10, th2=th2, t2=t2, q2=q2, svp1=svp1, svp2=svp2, svp3=svp3, &
         svpt0=svpt0, ep1=ep_1, ep2=ep_2, karman=karman, fCor=fcor, &
         te_temf=te_temf, hd_temf=hd_temf, exch_temf=exch_temf, &
         wm_temf=wm_temf, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

    ! SFC2D block, as the surface layer left it.
    write(uo) hfx(:,1), qfx(:,1), lh(:,1), flhc(:,1), flqc(:,1), qgh(:,1), &
         qsfc(:,1), ust(:,1), chs(:,1), chs2(:,1), cqs2(:,1), cpm(:,1), &
         u10(:,1), v10(:,1), th2(:,1), t2(:,1), q2(:,1), exch_temf(:,1), &
         wm_temf(:,1)

    ! module_pbl_driver.F:1161-1207, CASE (TEMFPBLSCHEME).
    call temfpbl(u3d=u, v3d=v, th3d=th, t3d=t, qv3d=qv, qc3d=qc, qi3d=qi, &
         p3d=p, p3di=p8w, pi3d=pii, rho=rho, rublten=rublten, &
         rvblten=rvblten, rthblten=rthblten, rqvblten=rqvblten, &
         rqcblten=rqcblten, rqiblten=rqiblten, flag_qi=.true., &
         g=g, cp=cp, rcp=rcp, r_d=r_d, r_v=r_v, cpv=cpv, z=z, xlv=xlv, &
         psfc=psfc, p_top=p_top, znt=znt, ht=ht, ust=ust, zol=zol, &
         hol=hol, hpbl=hpbl, psim=psim, psih=psih, xland=xland, hfx=hfx, &
         qfx=qfx, tsk=tsk, qsfc=qsfc, gz1oz0=gz1oz0, wspd=wspd, br=br, &
         dt=dt, dtmin=dtmin, kpbl2d=kpbl, svp1=svp1, svp2=svp2, svp3=svp3, &
         svpt0=svpt0, ep1=ep_1, ep2=ep_2, karman=karman, eomeg=eomeg, &
         stbolt=stbolt, kh_temf=kh_temf, km_temf=km_temf, u10=u10, &
         v10=v10, t2=t2, te_temf=te_temf, shf_temf=shf_temf, &
         qf_temf=qf_temf, uw_temf=uw_temf, vw_temf=vw_temf, &
         wupd_temf=wupd_temf, mf_temf=mf_temf, thup_temf=thup_temf, &
         qtup_temf=qtup_temf, qlup_temf=qlup_temf, cf3d_temf=cf3d_temf, &
         cfm_temf=cfm_temf, hd_temf=hd_temf, lcl_temf=lcl_temf, &
         hct_temf=hct_temf, flhc=flhc, flqc=flqc, exch_temf=exch_temf, &
         fCor=fcor, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

    ! PBL3D block.
    write(uo) rublten(:,1:nz,1), rvblten(:,1:nz,1), rthblten(:,1:nz,1), &
         rqvblten(:,1:nz,1), rqcblten(:,1:nz,1), te_temf(:,1:nz,1), &
         kh_temf(:,1:nz,1), km_temf(:,1:nz,1), shf_temf(:,1:nz,1), &
         qf_temf(:,1:nz,1), uw_temf(:,1:nz,1), vw_temf(:,1:nz,1), &
         wupd_temf(:,1:nz,1), mf_temf(:,1:nz,1), thup_temf(:,1:nz,1), &
         qtup_temf(:,1:nz,1), qlup_temf(:,1:nz,1), cf3d_temf(:,1:nz,1)
    ! PBL2D block (KPBL as its int32 word).
    write(uo) hpbl(:,1), kpbl(:,1), u10(:,1), v10(:,1), t2(:,1), ust(:,1), &
         hd_temf(:,1), lcl_temf(:,1), hct_temf(:,1), cfm_temf(:,1)

    ! Advance the column by the PBL tendencies alone.
    do k = 1, nz
      do i = 1, ncol
        u(i,k,1) = u(i,k,1) + dt*rublten(i,k,1)
        v(i,k,1) = v(i,k,1) + dt*rvblten(i,k,1)
        th(i,k,1) = th(i,k,1) + dt*rthblten(i,k,1)
        qv(i,k,1) = qv(i,k,1) + dt*rqvblten(i,k,1)
        if (qv(i,k,1) < 0.) qv(i,k,1) = 0.
        qc(i,k,1) = qc(i,k,1) + dt*rqcblten(i,k,1)
        if (qc(i,k,1) < 0.) qc(i,k,1) = 0.
      end do
    end do
    ! STATE block.
    write(uo) u(:,1:nz,1), v(:,1:nz,1), th(:,1:nz,1), qv(:,1:nz,1), &
         qc(:,1:nz,1)
  end do
  close(uo)
end program run_temf
