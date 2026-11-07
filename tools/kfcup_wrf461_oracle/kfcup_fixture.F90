module kfcup_fixture
  ! The KF-CuP (cu_physics = 10) fixture: the case table, the column
  ! builder and the hex writers shared by every harness in this directory,
  ! so the end-to-end driver capture and the stage captures see
  ! byte-identical inputs.
  !
  ! Everything is in WRF order: k = 1 is the lowest mass level, w-level
  ! arrays have nz + 1 entries.  Soundings are built in double precision and
  ! rounded ONCE to REAL; the rounded words are what the capture records as
  ! the inputs, so the port is graded on the same words WRF saw.
  !
  ! The case table is tuned to reach branches, not to be climatology: the
  ! CuP trigger (cupflag and the perturbation PDF), deep and shallow
  ! selection, the standard-KF fallback for stable or weak-flux columns,
  ! the NCA hold, and the edge clamps in cupSlopeSigma and cup_jfd.
  use iso_fortran_env, only: int32
  implicit none
  public

  integer, parameter :: nz = 40
  integer, parameter :: ncase_max = 128

  type kcase
     character(len=32) :: name = ''
     real(8) :: psfc = 100000.d0, ht = 0.d0
     real(8) :: thml = 300.d0, qml = 0.015d0, pblh = 1000.d0
     real(8) :: invdth = 1.d0, invdz = 200.d0
     real(8) :: gamft = 3.5d0, rhft = 0.6d0, ztrop = 14000.d0
     real(8) :: superad = 0.5d0
     real(8) :: u0 = 3.d0, ushear = 2.d0, v0 = 1.d0, vshear = 0.5d0
     real(8) :: wamp = 0.02d0, w0frac = 0.8d0
     real(8) :: hfx = 200.d0, qfx = 1.2d-4, xland = 1.d0, mavail = 0.5d0
     real(8) :: regime = 4.d0, br = -0.5d0, tskoff = 3.d0
     integer :: kpbl = 0              ! 0: first level whose AGL height >= pblh
     real(8) :: nca0 = -100.d0        ! initial NCA (s); >= 0.5*DT holds the column
     real(8) :: qconst = 0.d0         ! > 0: q held at qml up to this AGL height
  end type kcase

  type(kcase) :: cases(ncase_max)
  integer :: ncase = 0

contains

  subroutine add(c)
    type(kcase), intent(in) :: c
    ncase = ncase + 1
    if (ncase > ncase_max) error stop 'kfcup_fixture: ncase_max'
    cases(ncase) = c
  end subroutine add

  subroutine build_table()
    type(kcase) :: c
    integer :: v
    ncase = 0
    ! Each regime is entered with two variants, A and B, so every arm is
    ! reached by more than one rounding of the same physics.
    do v = 0, 1
       ! 1 tropical ocean, deep
       c = kcase(); c%name = 'trop_ocean'; c%psfc = 101000d0; c%thml = 299.5d0 + v
       c%qml = 0.0185d0; c%pblh = 600d0; c%invdth = 0.3d0; c%gamft = 3.2d0; c%rhft = 0.85d0
       c%xland = 2d0; c%hfx = 15d0 + 10*v; c%qfx = 1.2d-4; c%mavail = 1d0; c%tskoff = 1.5d0
       c%wamp = 0.03d0; call add(c)
       ! 2 tropical land afternoon, deep
       c = kcase(); c%name = 'trop_land'; c%psfc = 100500d0; c%thml = 302.5d0 + 0.5d0*v
       c%qml = 0.0170d0; c%pblh = 1300d0; c%invdth = 0.5d0; c%gamft = 3.4d0; c%rhft = 0.75d0
       c%hfx = 280d0; c%qfx = 1.6d-4; c%wamp = 0.04d0 - 0.06d0*v; call add(c)
       ! 3 plains, strong cap (loaded gun)
       c = kcase(); c%name = 'plains_cap'; c%psfc = 96000d0; c%ht = 400d0; c%thml = 306d0 + v
       c%qml = 0.0160d0; c%pblh = 1500d0; c%invdth = 4d0; c%invdz = 300d0; c%gamft = 3.0d0
       c%rhft = 0.5d0; c%hfx = 320d0; c%qfx = 1.4d-4; c%ushear = 4d0; call add(c)
       ! 4 plains, weak cap
       c = kcase(); c%name = 'plains_weak'; c%psfc = 96000d0; c%ht = 400d0; c%thml = 305d0 + v
       c%qml = 0.0155d0; c%pblh = 1700d0; c%invdth = 0.8d0; c%gamft = 3.0d0; c%rhft = 0.55d0
       c%hfx = 300d0; c%qfx = 1.5d-4; c%ushear = 5d0; call add(c)
       ! 5 trade cumulus, shallow
       c = kcase(); c%name = 'trade_cu'; c%psfc = 101500d0; c%thml = 298d0 + 0.5d0*v
       c%qml = 0.0160d0; c%pblh = 700d0; c%invdth = 1.0d0; c%gamft = 4.5d0; c%rhft = 0.35d0
       c%xland = 2d0; c%hfx = 12d0; c%qfx = 1.0d-4; c%mavail = 1d0; c%tskoff = 1d0; call add(c)
       ! 6 continental fair-weather cumulus, shallow
       c = kcase(); c%name = 'fair_cu'; c%thml = 296d0 + v; c%qml = 0.0105d0
       c%pblh = 1500d0; c%invdth = 2.0d0; c%gamft = 5.0d0; c%rhft = 0.40d0
       c%hfx = 220d0; c%qfx = 1.0d-4; call add(c)
       ! 7 dry desert, deep mixed layer
       c = kcase(); c%name = 'desert'; c%psfc = 94000d0; c%ht = 600d0; c%thml = 313d0 + v
       c%qml = 0.0060d0; c%pblh = 3500d0; c%invdth = 1d0; c%gamft = 4d0; c%rhft = 0.2d0
       c%hfx = 450d0; c%qfx = 2d-5; c%mavail = 0.05d0; c%tskoff = 8d0; call add(c)
       ! 8 high terrain, moist
       c = kcase(); c%name = 'mtn_moist'; c%psfc = 75000d0; c%ht = 2500d0; c%thml = 314d0 + v
       c%qml = 0.0100d0; c%pblh = 1200d0; c%invdth = 0.6d0; c%gamft = 3.6d0; c%rhft = 0.6d0
       c%hfx = 260d0; c%qfx = 1.3d-4; call add(c)
       ! 9 high terrain, dry
       c = kcase(); c%name = 'mtn_dry'; c%psfc = 70000d0; c%ht = 3000d0; c%thml = 318d0 + v
       c%qml = 0.0030d0; c%pblh = 2000d0; c%invdth = 1d0; c%gamft = 4d0; c%rhft = 0.15d0
       c%hfx = 380d0; c%qfx = 2d-5; c%tskoff = 10d0; call add(c)
       ! 10 nocturnal stable land: cupflag false, standard-KF fallback
       c = kcase(); c%name = 'night_land'; c%thml = 300d0 + v; c%qml = 0.0160d0
       c%pblh = 150d0; c%superad = -2d0; c%invdth = 3d0; c%gamft = 3.2d0; c%rhft = 0.8d0
       c%hfx = -30d0; c%qfx = -1d-6; c%regime = 1d0; c%br = 0.4d0; c%tskoff = -3d0
       c%wamp = 0.08d0; call add(c)
       ! 11 weak-flux land, unstable aloft, ascent: standard KF deep
       c = kcase(); c%name = 'weakflux_deep'; c%thml = 301d0 + v; c%qml = 0.0175d0
       c%pblh = 600d0; c%invdth = 0.2d0; c%gamft = 3.0d0; c%rhft = 0.85d0
       c%hfx = 60d0; c%qfx = 0.8d-4; c%wamp = 0.12d0; c%w0frac = 1d0; call add(c)
       ! 12 elevated nocturnal convection over land
       c = kcase(); c%name = 'elevated'; c%thml = 297d0 + v; c%qml = 0.0140d0
       c%pblh = 200d0; c%superad = -1d0; c%invdth = 2d0; c%invdz = 400d0; c%gamft = 3.0d0
       c%rhft = 0.9d0; c%hfx = 5d0; c%qfx = 1d-6; c%regime = 2d0; c%br = 0.1d0
       c%wamp = 0.10d0; call add(c)
       ! 13 marine stratocumulus under a hard inversion
       c = kcase(); c%name = 'stratocu'; c%psfc = 101800d0; c%thml = 290d0 + v
       c%qml = 0.0090d0; c%pblh = 800d0; c%invdth = 10d0; c%invdz = 150d0; c%gamft = 5d0
       c%rhft = 0.2d0; c%xland = 2d0; c%hfx = 8d0; c%qfx = 0.5d-4; c%mavail = 1d0
       c%tskoff = 0.5d0; call add(c)
       ! 14 polar cold continental
       c = kcase(); c%name = 'polar'; c%psfc = 100000d0; c%thml = 266d0 + v
       c%qml = 0.0021d0; c%pblh = 800d0; c%invdth = 1d0; c%gamft = 4d0; c%rhft = 0.7d0
       c%ztrop = 9000d0; c%hfx = 130d0; c%qfx = 3d-5; call add(c)
       ! 15 cold air over warm water (lake-effect like)
       c = kcase(); c%name = 'cold_over_water'; c%psfc = 100500d0; c%thml = 271d0 + v
       c%qml = 0.0040d0; c%pblh = 1500d0; c%invdth = 0.5d0; c%gamft = 2.5d0; c%rhft = 0.85d0
       c%ztrop = 9500d0; c%xland = 2d0; c%hfx = 250d0; c%qfx = 1.2d-4; c%mavail = 1d0
       c%tskoff = 12d0; call add(c)
       ! 16 kpbl = 2 edge: cupflag false regardless of flux
       c = kcase(); c%name = 'kpbl2'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1200d0; c%gamft = 3.3d0; c%rhft = 0.75d0; c%hfx = 300d0; c%kpbl = 2
       c%wamp = 0.06d0; call add(c)
       ! 17 qfx below 1/xlv: forced to sign(1/xlv, qfx)
       c = kcase(); c%name = 'qfx_tiny'; c%thml = 303d0 + v; c%qml = 0.0160d0
       c%pblh = 1400d0; c%gamft = 3.4d0; c%rhft = 0.6d0; c%hfx = 250d0
       c%qfx = (2*v - 1)*1d-8; call add(c)
       ! 18 dew: qfx negative with strong sensible flux
       c = kcase(); c%name = 'qfx_neg'; c%thml = 302d0 + v; c%qml = 0.0165d0
       c%pblh = 1300d0; c%gamft = 3.3d0; c%rhft = 0.7d0; c%hfx = 160d0; c%qfx = -3d-5
       call add(c)
       ! 19 q constant through kpbl+2: qdiff inside reallysmall
       c = kcase(); c%name = 'qdiff_small'; c%thml = 301d0 + v; c%qml = 0.0150d0
       c%pblh = 1000d0; c%gamft = 3.5d0; c%rhft = 0.9d0; c%qconst = 2500d0; c%hfx = 220d0
       call add(c)
       ! 20 strong moisture jump above the PBL: |slopeEZ| small, clamped to 10
       c = kcase(); c%name = 'slopeez_small'; c%thml = 300d0 + v; c%qml = 0.0120d0
       c%pblh = 900d0; c%invdth = 0.05d0; c%invdz = 100d0; c%gamft = 1.5d0; c%rhft = 0.95d0
       c%hfx = 200d0; call add(c)
       ! 21 very shallow PBL, kpbl 3
       c = kcase(); c%name = 'shallow_pbl'; c%thml = 300d0 + v; c%qml = 0.0160d0
       c%pblh = 150d0; c%invdth = 1.5d0; c%gamft = 3.3d0; c%rhft = 0.75d0; c%kpbl = 3
       c%hfx = 150d0; call add(c)
       ! 22 monsoon, deep moist PBL
       c = kcase(); c%name = 'monsoon'; c%psfc = 99500d0; c%thml = 304d0 + v
       c%qml = 0.0190d0; c%pblh = 2500d0; c%invdth = 0.4d0; c%gamft = 3.3d0; c%rhft = 0.8d0
       c%hfx = 200d0; c%qfx = 1.8d-4; call add(c)
       ! 23 strong shear tropical
       c = kcase(); c%name = 'trop_shear'; c%thml = 301d0 + v; c%qml = 0.0180d0
       c%pblh = 900d0; c%invdth = 0.3d0; c%gamft = 3.2d0; c%rhft = 0.8d0; c%ushear = 9d0
       c%vshear = 4d0; c%hfx = 180d0; c%qfx = 1.5d-4; call add(c)
       ! 24 tropical cyclone core-like, saturated aloft
       c = kcase(); c%name = 'tc_core'; c%psfc = 98000d0; c%thml = 301.5d0 + v
       c%qml = 0.0200d0; c%pblh = 500d0; c%invdth = 0.1d0; c%gamft = 3.0d0; c%rhft = 0.97d0
       c%xland = 2d0; c%hfx = 60d0; c%qfx = 3d-4; c%mavail = 1d0; c%u0 = 25d0; c%wamp = 0.2d0
       call add(c)
       ! 25 mid-latitude squall-line environment
       c = kcase(); c%name = 'squall'; c%psfc = 98500d0; c%thml = 300d0 + v
       c%qml = 0.0140d0; c%pblh = 1100d0; c%invdth = 0.7d0; c%gamft = 3.8d0; c%rhft = 0.55d0
       c%ushear = 6d0; c%hfx = 230d0; call add(c)
       ! 26 subtropical subsidence over water, weak flux
       c = kcase(); c%name = 'subsidence'; c%psfc = 101800d0; c%thml = 295d0 + v
       c%qml = 0.0130d0; c%pblh = 900d0; c%invdth = 6d0; c%gamft = 5d0; c%rhft = 0.25d0
       c%xland = 2d0; c%hfx = 5d0; c%qfx = 0.6d-4; c%mavail = 1d0; c%wamp = -0.03d0
       call add(c)
       ! 27 water, hfx below 1: cupflag false
       c = kcase(); c%name = 'water_hfx_lt1'; c%thml = 300d0 + v; c%qml = 0.0180d0
       c%pblh = 600d0; c%gamft = 3.2d0; c%rhft = 0.85d0; c%xland = 2d0; c%hfx = 0.5d0
       c%mavail = 1d0; c%wamp = 0.05d0; call add(c)
       ! 28 land, hfx exactly 100: cupflag true (the test is hfx < 100)
       c = kcase(); c%name = 'land_hfx100'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1000d0; c%gamft = 3.3d0; c%rhft = 0.75d0; c%hfx = 100d0; call add(c)
       ! 29 land, hfx just under 100: cupflag false
       c = kcase(); c%name = 'land_hfx99'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1000d0; c%gamft = 3.3d0; c%rhft = 0.75d0; c%hfx = 99.99d0
       c%wamp = 0.07d0; call add(c)
       ! 30 NCA hold: well above 0.5*DT
       c = kcase(); c%name = 'nca_held'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1300d0; c%gamft = 3.4d0; c%hfx = 280d0; c%nca0 = 900d0; call add(c)
       ! 31 NCA at the hold boundary (set relative to DT by the driver)
       c = kcase(); c%name = 'nca_edge'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1300d0; c%gamft = 3.4d0; c%hfx = 280d0; c%nca0 = -1000d0 - v; call add(c)
       ! 32 very warm, very moist: TLCL > 293 K so CHMIN = 4 km
       c = kcase(); c%name = 'hot_humid'; c%psfc = 100800d0; c%thml = 305d0 + v
       c%qml = 0.0215d0; c%pblh = 800d0; c%invdth = 0.3d0; c%gamft = 3.3d0; c%rhft = 0.8d0
       c%hfx = 150d0; c%qfx = 2d-4; c%xland = 2d0 - v; call add(c)
       ! 33 cool moist: TLCL between 273 and 293 K
       c = kcase(); c%name = 'cool_moist'; c%thml = 288d0 + v; c%qml = 0.0085d0
       c%pblh = 900d0; c%invdth = 0.5d0; c%gamft = 3.5d0; c%rhft = 0.8d0; c%ztrop = 11000d0
       c%hfx = 160d0; call add(c)
       ! 34 cold LCL below 273 K: CHMIN = 2 km
       c = kcase(); c%name = 'cold_lcl'; c%psfc = 90000d0; c%ht = 1000d0; c%thml = 283d0 + v
       c%qml = 0.0035d0; c%pblh = 1200d0; c%invdth = 0.4d0; c%gamft = 3.0d0; c%rhft = 0.85d0
       c%ztrop = 10000d0; c%hfx = 180d0; c%qfx = 5d-5; call add(c)
       ! 35 descent: negative w through the column
       c = kcase(); c%name = 'descent'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1200d0; c%gamft = 3.4d0; c%rhft = 0.7d0; c%hfx = 250d0; c%wamp = -0.15d0
       call add(c)
       ! 36 strong ascent: wkl > 0.1, RAD = 2000
       c = kcase(); c%name = 'ascent'; c%thml = 302d0 + v; c%qml = 0.0170d0
       c%pblh = 1200d0; c%gamft = 3.4d0; c%rhft = 0.7d0; c%hfx = 250d0; c%wamp = 0.5d0
       c%w0frac = 1d0; call add(c)
    end do
  end subroutine build_table

  ! --------------------------------------------------------------------
  ! Column builder
  ! --------------------------------------------------------------------
  subroutine build_column(c, dt, u, v, th, t, w, dz8w, p, pii, rho, qv, z, zw, &
       w0avg, psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, kpbl, t2, q2)
    type(kcase), intent(in) :: c
    real, intent(in) :: dt
    real, intent(out) :: u(nz), v(nz), th(nz), t(nz), w(nz+1), dz8w(nz), p(nz)
    real, intent(out) :: pii(nz), rho(nz), qv(nz), z(nz), zw(nz+1), w0avg(nz)
    real, intent(out) :: psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, t2, q2
    integer, intent(out) :: kpbl
    real(8), parameter :: rd = 287.d0, cpd = 1004.5d0, grav = 9.81d0
    real(8) :: dzd(nz), zwd(nz+1), zmd(nz), thd(nz), qd(nz), pw(nz+1), pm(nz), td(nz)
    real(8) :: zagl, tv, es, qs, wd(nz+1)
    integer :: k, it

    zwd(1) = c%ht
    do k = 1, nz
       dzd(k) = min(60.d0*1.1d0**(k-1), 800.d0)
       zwd(k+1) = zwd(k) + dzd(k)
       zmd(k) = 0.5d0*(zwd(k) + zwd(k+1))
    end do
    do k = 1, nz
       zagl = zmd(k) - c%ht
       thd(k) = theta_of(c, zagl)
    end do
    ! Hydrostatic pressure, with q from relative humidity above the mixed
    ! layer, iterated so q and T agree with the pressure they are formed at.
    qd = c%qml
    do it = 1, 4
       pw(1) = c%psfc
       do k = 1, nz
          pm(k) = pw(k)*exp(-grav*0.5d0*dzd(k)/(rd*thd(k)*(pw(k)/1d5)**(rd/cpd)*(1d0+0.608d0*qd(k))))
          td(k) = thd(k)*(pm(k)/1d5)**(rd/cpd)
          tv = td(k)*(1d0 + 0.608d0*qd(k))
          pw(k+1) = pw(k)*exp(-grav*dzd(k)/(rd*tv))
       end do
       do k = 1, nz
          zagl = zmd(k) - c%ht
          es = 611.2d0*exp(17.67d0*(td(k)-273.15d0)/(td(k)-29.65d0))
          qs = 0.622d0*es/(pm(k)-es)
          if (zagl < c%pblh .or. zagl < c%qconst) then
             qd(k) = min(c%qml*(1d0 - 0.05d0*zagl/max(c%pblh,1d0)), 0.995d0*qs)
             if (zagl < c%qconst) qd(k) = min(c%qml, 0.995d0*qs)
          else if (zmd(k) < c%ztrop) then
             qd(k) = min(c%qml, c%rhft*qs)
          else
             qd(k) = 3d-6
          end if
          qd(k) = max(qd(k), 3d-6)
       end do
    end do
    do k = 1, nz+1
       zagl = zwd(k) - c%ht
       if (k == 1) then
          wd(k) = 0d0
       else
          wd(k) = c%wamp*sin(3.14159265358979d0*min(zagl, 12000d0)/12000d0)
       end if
    end do
    do k = 1, nz
       zagl = zmd(k) - c%ht
       u(k) = real(c%u0 + c%ushear*zagl/1000d0)
       v(k) = real(c%v0 + c%vshear*zagl/1000d0)
       th(k) = real(thd(k))
       t(k) = real(td(k))
       p(k) = real(pm(k))
       pii(k) = real((pm(k)/1d5)**(rd/cpd))
       qv(k) = real(qd(k))
       rho(k) = real(pm(k)/(rd*td(k)*(1d0+0.608d0*qd(k))))
       z(k) = real(zmd(k))
       dz8w(k) = real(dzd(k))
       w0avg(k) = real(c%w0frac*0.5d0*(wd(k)+wd(k+1)))
    end do
    do k = 1, nz+1
       zw(k) = real(zwd(k))
       w(k) = real(wd(k))
    end do
    psfc = real(c%psfc); ht = real(c%ht)
    tsk = real(td(1) + c%tskoff); t2 = real(td(1) + 0.5d0*c%tskoff)
    q2 = real(qd(1)); hfx = real(c%hfx); qfx = real(c%qfx)
    xland = real(c%xland); mavail = real(c%mavail)
    br = real(c%br); regime = real(c%regime); pblh = real(c%pblh)
    if (c%kpbl > 0) then
       kpbl = c%kpbl
    else
       kpbl = nz
       do k = 1, nz
          if (zmd(k) - c%ht >= c%pblh) then
             kpbl = k
             exit
          end if
       end do
    end if
    if (dt < 0.) error stop 'dt'
  end subroutine build_column

  real(8) function theta_of(c, zagl) result(th)
    type(kcase), intent(in) :: c
    real(8), intent(in) :: zagl
    real(8) :: ztop_inv, zasl
    zasl = zagl + c%ht
    ztop_inv = c%pblh + c%invdz
    if (zagl < c%pblh) then
       th = c%thml + c%superad*max(0d0, 1d0 - zagl/100d0)
    else if (zagl < ztop_inv) then
       th = c%thml + c%invdth*(zagl - c%pblh)/c%invdz
    else
       th = c%thml + c%invdth + c%gamft*(min(zasl, c%ztrop) - (ztop_inv + c%ht))/1000d0
       if (zasl > c%ztrop) th = th + 15d0*(zasl - c%ztrop)/1000d0
    end if
  end function theta_of

  ! --------------------------------------------------------------------
  ! Hex writers: every float is its raw IEEE-754 word.
  ! --------------------------------------------------------------------
  character(len=8) function hx(x)
    real, intent(in) :: x
    write(hx, '(z8.8)') transfer(x, 0_int32)
  end function hx

  character(len=8) function hxi(i)
    integer, intent(in) :: i
    write(hxi, '(z8.8)') i
  end function hxi

end module kfcup_fixture
