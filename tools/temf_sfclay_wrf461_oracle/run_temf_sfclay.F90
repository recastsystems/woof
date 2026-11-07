! Column oracle for WRF v4.6.1 phys/module_sf_temfsfclay.F (sf_sfclay_physics=10).
!
! Drives the byte-unmodified module through its own 3-D entry TEMFSFCLAY,
! exactly as CASE (TEMFSFCSCHEME) in phys/module_surface_driver.F:2407-2434
! calls it, over NC columns and NSTEP steps.  TEMFSFCLAYINIT runs first, as
! module_physics_init.F does on a cold start, so WM_TEMF starts at 0.
!
! Carried state: the scheme reads HFX (the w* term) and WM_TEMF (the
! two-step average) from the previous step, and floors TE_TEMF(:,1) in
! place.  Between steps this driver plays the parts the scheme does not
! own: the TEMF PBL rewrites TE_TEMF(:,1) and HD_TEMF, and the atmosphere
! and skin move.  Those updates are deterministic float32 formulas below;
! every word they produce is RECORDED as the next step's input, so the
! port replays the same feed while carrying HFX and WM_TEMF itself.
!
! Output: one line per (step, column), every input word before the call and
! every output word after it, as 8-digit hex IEEE-754 binary32 patterns.
program run_temf_sfclay
  use module_sf_temfsfclay, only: temfsfclay, temfsfclayinit
  implicit none
  integer, parameter :: NC = 78, NSTEP = 8, NK = 2
  integer, parameter :: ids = 1, ide = NC + 1, jds = 1, jde = 2, kds = 1, kde = NK + 1
  integer, parameter :: ims = 1, ime = NC, jms = 1, jme = 1, kms = 1, kme = NK
  integer, parameter :: its = 1, ite = NC, jts = 1, jte = 1, kts = 1, kte = NK
  ! share/module_model_constants.F, as the surface driver receives them
  real, parameter :: r_d = 287., r_v = 461.6, cp = 7.*r_d/2., g = 9.81
  real, parameter :: rcp = r_d/cp, xlv = 2.5E6, karman = 0.4
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65, svpt0 = 273.15
  real, parameter :: ep_1 = r_v/r_d - 1., ep_2 = r_d/r_v
  real, dimension(ims:ime, kms:kme, jms:jme) :: u3d, v3d, th3d, qv3d, p3d, pi3d, rho, z, te_temf
  real, dimension(ims:ime, jms:jme) :: ht, psfc, chs, chs2, cqs2, cpm, znt, ust, mavail, xland, &
       hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc, u10, v10, th2, t2, q2, fcor, hd_temf, &
       exch_temf, wm_temf
  integer :: i, k, step, cat, var, u
  ! cat 13: z (ht = 0) and znt words; zm/znt = 10.00001, 100.0001,
  ! 1000.001, 10000.2, 100000.05, 3000.005
  integer, parameter :: z13(0:5) = (/ int(z'41A0000D'), int(z'41C80012'), &
       int(z'41FA000D'), int(z'419C40D0'), int(z'41C35007'), int(z'41BB8014') /)
  integer, parameter :: znt13(0:5) = (/ int(z'40000000'), int(z'3E800000'), &
       int(z'3D000000'), int(z'3B000000'), int(z'39800000'), int(z'3C000000') /)
  character(len=256) :: infile, outfile
  integer :: uo
  real :: base_ws, wdir, dtheta, tair

  if (command_argument_count() /= 2) stop 'usage: run_temf_sfclay INPUTS.csv OUTPUTS.csv'
  call get_command_argument(1, infile)
  call get_command_argument(2, outfile)

  ! ---------------- column catalogue -------------------------------------
  ! 12 regimes x 6 variants.  cat: 1 convective land, 2 stable night land,
  ! 3 neutral windy, 4 calm (u=v=0: the S<1e-15 branch, both signs of N2),
  ! 5 sea, 6 rough (forest/urban z0), 7 very stable (Ri clamp at 0.2),
  ! 8 dry (mavail=0), 9 TE below TEmin and deep convective hd_temf,
  ! 10 near-calm, 11 cold/high-latitude and equatorial, 12 hot desert with
  ! the first mass level below 10 m (u10 extrapolates), 13 log10 words:
  ! zm/znt lands exactly on a binary32 word where the C library's log10f is
  ! not the correctly rounded value (log10-probe.txt), with a power-of-two
  ! znt so the quotient is exact.  These six columns are where arm A and
  ! arm C must differ.
  do i = 1, NC
     cat = (i - 1) / 6 + 1
     var = mod(i - 1, 6)
     ht(i,1) = 100. + 37.*real(var)
     z(i,1,1) = ht(i,1) + 25. + 3.*real(var)
     z(i,2,1) = ht(i,1) + 80. + 7.*real(var)
     psfc(i,1) = 98000. - 450.*real(var)
     p3d(i,1,1) = psfc(i,1) - 300.
     p3d(i,2,1) = psfc(i,1) - 950.
     do k = 1, NK
        pi3d(i,k,1) = (p3d(i,k,1)/1.e5)**rcp
     end do
     tair = 288. + 1.5*real(var)
     th3d(i,1,1) = tair / pi3d(i,1,1)
     th3d(i,2,1) = th3d(i,1,1) + 0.4
     qv3d(i,1,1) = 0.006 + 0.0009*real(var)
     qv3d(i,2,1) = qv3d(i,1,1) - 0.0005
     rho(i,1,1) = p3d(i,1,1) / (r_d * tair * (1. + 0.608*qv3d(i,1,1)))
     rho(i,2,1) = rho(i,1,1) * 0.99
     base_ws = 4. + 0.7*real(var)
     wdir = 0.6 + 0.9*real(var)
     u3d(i,1,1) = base_ws * cos(wdir)
     v3d(i,1,1) = base_ws * sin(wdir)
     u3d(i,2,1) = 1.2*u3d(i,1,1)
     v3d(i,2,1) = 1.2*v3d(i,1,1)
     znt(i,1) = 0.08 + 0.02*real(var)
     mavail(i,1) = 0.3 + 0.1*real(var)
     xland(i,1) = 1.
     dtheta = 0.
     tsk(i,1) = tair
     fcor(i,1) = 1.0e-4
     hd_temf(i,1) = 0.
     te_temf(i,1,1) = 0.4 + 0.15*real(var)
     te_temf(i,2,1) = 0.3
     hfx(i,1) = 0.
     select case (cat)
     case (1)   ! convective land
        tsk(i,1) = tair + 6. + 1.2*real(var)
        hfx(i,1) = 120. + 40.*real(var)
        hd_temf(i,1) = 800. + 150.*real(var)
     case (2)   ! stable night land
        tsk(i,1) = tair - 3. - 0.8*real(var)
        hfx(i,1) = -20. - 5.*real(var)
        u3d(i,1,1) = 0.6*u3d(i,1,1); v3d(i,1,1) = 0.6*v3d(i,1,1)
     case (3)   ! neutral windy
        tsk(i,1) = tair + 0.05*real(var)
        u3d(i,1,1) = (14. + 1.5*real(var)) * cos(wdir)
        v3d(i,1,1) = (14. + 1.5*real(var)) * sin(wdir)
        te_temf(i,1,1) = 2. + 0.5*real(var)
     case (4)   ! calm
        u3d(i,1,1) = 0.; v3d(i,1,1) = 0.
        if (var < 2) then
           tsk(i,1) = tair + 3.          ! N2 < 0
           hfx(i,1) = 50.
           hd_temf(i,1) = 400.
        else if (var < 4) then
           tsk(i,1) = tair - 2.          ! N2 > 0
        else
           tsk(i,1) = th3d(i,1,1)*pi3d(i,1,1)   ! N2 from qv only
        end if
     case (5)   ! sea
        xland(i,1) = 2.
        mavail(i,1) = 1.
        znt(i,1) = 1.5e-4 + 5.e-5*real(var)
        tsk(i,1) = tair + 1. - 0.6*real(var)
        ht(i,1) = 0.
        z(i,1,1) = 22. + 2.*real(var)
        z(i,2,1) = 70. + 4.*real(var)
        psfc(i,1) = 101300. - 100.*real(var)
        p3d(i,1,1) = psfc(i,1) - 260.
        p3d(i,2,1) = psfc(i,1) - 850.
        pi3d(i,1,1) = (p3d(i,1,1)/1.e5)**rcp
        pi3d(i,2,1) = (p3d(i,2,1)/1.e5)**rcp
        th3d(i,1,1) = tair / pi3d(i,1,1)
        qv3d(i,1,1) = 0.011 + 0.0006*real(var)
        if (var < 3) hfx(i,1) = 15.
     case (6)   ! rough
        znt(i,1) = 0.8 + 0.25*real(var)
        z(i,1,1) = ht(i,1) + 40. + 5.*real(var)
        tsk(i,1) = tair + 2. - 0.5*real(var)
        if (var < 3) then
           hfx(i,1) = 60.
           hd_temf(i,1) = 600.
        end if
     case (7)   ! very stable: Ri clamps at 0.2
        tsk(i,1) = tair - 9. - real(var)
        u3d(i,1,1) = 0.8 + 0.1*real(var); v3d(i,1,1) = 0.3
        hfx(i,1) = -5.
        te_temf(i,1,1) = 0.05 + 0.01*real(var)
     case (8)   ! dry: mavail = 0
        mavail(i,1) = 0.
        tsk(i,1) = tair + 4. - 1.5*real(var)
        if (var < 3) then
           hfx(i,1) = 200.
           hd_temf(i,1) = 1500.
        end if
     case (9)   ! TE below TEmin, deep convective hd_temf
        te_temf(i,1,1) = 1.e-5 * real(var)
        tsk(i,1) = tair + 8.
        hfx(i,1) = 300.
        hd_temf(i,1) = 2500. + 200.*real(var)
     case (10)  ! near calm
        u3d(i,1,1) = 1.e-3 * real(var + 1); v3d(i,1,1) = -2.e-4 * real(var)
        tsk(i,1) = tair + 0.7 - 0.3*real(var)
     case (11)  ! cold, high latitude and equator
        tair = 250. + 2.*real(var)
        th3d(i,1,1) = tair / pi3d(i,1,1)
        qv3d(i,1,1) = 0.0004 + 0.0001*real(var)
        tsk(i,1) = tair - 1. + 0.5*real(var)
        if (var < 3) then
           fcor(i,1) = 1.4e-4
        else
           fcor(i,1) = 0.
        end if
     case (12)  ! hot desert, first mass level below 10 m
        mavail(i,1) = 0.02
        z(i,1,1) = ht(i,1) + 6. + 0.5*real(var)
        tair = 310. + real(var)
        th3d(i,1,1) = tair / pi3d(i,1,1)
        qv3d(i,1,1) = 0.003
        tsk(i,1) = tair + 15.
        hfx(i,1) = 400.
        hd_temf(i,1) = 3000.
        znt(i,1) = 0.01 + 0.005*real(var)
     case (13)  ! log10 words
        ht(i,1) = 0.
        z(i,1,1) = transfer(z13(var), 1.0)
        z(i,2,1) = z(i,1,1) + 50.
        znt(i,1) = transfer(znt13(var), 1.0)
        tsk(i,1) = tair + 2. - 0.8*real(var)
        if (var < 3) then
           hfx(i,1) = 80.
           hd_temf(i,1) = 900.
        end if
     end select
     ! southern-hemisphere sign on odd variants: TEMF reads abs(fCor)
     if (mod(var, 2) == 1) fcor(i,1) = -fcor(i,1)
  end do

  call temfsfclayinit(.false., .true., wm_temf, ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
  ! INTENT(INOUT)/(OUT) fields the scheme overwrites: start them at zero, as
  ! module_physics_init.F leaves them on a cold start.
  ust = 0.; chs = 0.; chs2 = 0.; cqs2 = 0.; cpm = 0.; qgh = 0.; qsfc = 0.
  qfx = 0.; lh = 0.; flhc = 0.; flqc = 0.; u10 = 0.; v10 = 0.; th2 = 0.
  t2 = 0.; q2 = 0.; exch_temf = 0.

  open(newunit=u, file=trim(infile), status='replace', action='write')
  open(newunit=uo, file=trim(outfile), status='replace', action='write')
  write(u, '(a)') 'step,col,u,v,th,qv,p,pi,rho,z,ht,psfc,znt,mavail,xland,tsk,fcor,' // &
       'te,hd,hfx,wm'
  write(uo, '(a)') 'step,col,ust,hfx,qfx,lh,flhc,flqc,chs,chs2,cqs2,cpm,qgh,qsfc,' // &
       'u10,v10,th2,t2,q2,exch,wm,te,znt'
  do step = 1, NSTEP
     if (step > 1) call feed(step)
     do i = 1, NC
        write(u, '(i0,",",i0,19(",",z8.8))') step, i, &
             bits(u3d(i,1,1)), bits(v3d(i,1,1)), bits(th3d(i,1,1)), bits(qv3d(i,1,1)), &
             bits(p3d(i,1,1)), bits(pi3d(i,1,1)), bits(rho(i,1,1)), bits(z(i,1,1)), &
             bits(ht(i,1)), bits(psfc(i,1)), bits(znt(i,1)), bits(mavail(i,1)), &
             bits(xland(i,1)), bits(tsk(i,1)), bits(fcor(i,1)), bits(te_temf(i,1,1)), &
             bits(hd_temf(i,1)), bits(hfx(i,1)), bits(wm_temf(i,1))
     end do
     call temfsfclay(u3d, v3d, th3d, qv3d, p3d, pi3d, rho, z, ht,          &
          cp, g, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,                 &
          znt, ust, mavail, xland,                                          &
          hfx, qfx, lh, tsk, flhc, flqc, qgh, qsfc,                         &
          u10, v10, th2, t2, q2,                                            &
          svp1, svp2, svp3, svpt0, ep_1, ep_2,                              &
          karman, fcor, te_temf,                                            &
          hd_temf, exch_temf, wm_temf,                                      &
          ids, ide, jds, jde, kds, kde,                                     &
          ims, ime, jms, jme, kms, kme,                                     &
          its, ite, jts, jte, kts, kte)
     do i = 1, NC
        write(uo, '(i0,",",i0,21(",",z8.8))') step, i, &
             bits(ust(i,1)), bits(hfx(i,1)), bits(qfx(i,1)), bits(lh(i,1)), &
             bits(flhc(i,1)), bits(flqc(i,1)), bits(chs(i,1)), bits(chs2(i,1)), &
             bits(cqs2(i,1)), bits(cpm(i,1)), bits(qgh(i,1)), bits(qsfc(i,1)), &
             bits(u10(i,1)), bits(v10(i,1)), bits(th2(i,1)), bits(t2(i,1)), &
             bits(q2(i,1)), bits(exch_temf(i,1)), bits(wm_temf(i,1)), &
             bits(te_temf(i,1,1)), bits(znt(i,1))
     end do
  end do
  close(u)
  close(uo)

contains

  integer function bits(x)
    real, intent(in) :: x
    bits = transfer(x, 0)
  end function bits

  ! The parts of a model step the surface layer does not own, as float32
  ! formulas.  HFX and WM_TEMF are NOT touched: they carry.
  subroutine feed(step)
    integer, intent(in) :: step
    integer :: i, cat, var
    real :: s
    s = real(step)
    do i = 1, NC
       cat = (i - 1) / 6 + 1
       var = mod(i - 1, 6)
       ! the PBL's new surface-level TE and dry thermal top
       if (cat == 9 .and. mod(step, 2) == 0) then
          te_temf(i,1,1) = 2.e-4 * real(var)          ! below TEmin again
       else
          te_temf(i,1,1) = te_temf(i,1,1) * (0.85 + 0.05*real(mod(i + step, 7)))
       end if
       if (hd_temf(i,1) > 0.) hd_temf(i,1) = hd_temf(i,1) * (1. + 0.04*s)
       ! the atmosphere and skin move
       if (cat /= 4) then
          u3d(i,1,1) = u3d(i,1,1) * (1. + 0.013*s) + 0.05
          v3d(i,1,1) = v3d(i,1,1) * (1. - 0.011*s) - 0.03
       end if
       th3d(i,1,1) = th3d(i,1,1) + 0.03*s - 0.02*real(var)
       qv3d(i,1,1) = qv3d(i,1,1) * (1. + 0.002*s)
       select case (cat)
       case (1, 9, 12)
          tsk(i,1) = tsk(i,1) + 0.4
       case (2, 7)
          tsk(i,1) = tsk(i,1) - 0.3
       case default
          tsk(i,1) = tsk(i,1) + 0.1*real(mod(i, 3) - 1)
       end select
    end do
  end subroutine feed

end program run_temf_sfclay
