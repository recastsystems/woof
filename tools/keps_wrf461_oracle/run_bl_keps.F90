program run_bl_keps_oracle
  ! Drive WRF v4.6.1 phys/module_bl_keps.F (bl_pbl_physics = 17) through its
  ! own 3-D entry point `keps` and write every input and every output word as
  ! raw float32, so the GPU port can be graded word for word.
  !
  ! The module has no USE statement and every CALL it makes lives inside it
  ! (keps1D, shear, c_nsquare, surface_bl_pra_ri, buoy, cdtur_ke,
  ! length_bougeault, thetaphi_alphabeta, diff, invert, terms_tpe,
  ! calc_countergradient, GET_PBLH), so there is no stub: `nm -u` on the
  ! object is the receipt.
  !
  ! The whole tile goes through ONE call per step (its..ite = all columns),
  ! as WRF's own driver does, because `keps` keeps column arrays (z1D, exch1D,
  ! pr1D, ...) across the columns of one call; a port that only matched one
  ! column per call could miss a cross-column carry.
  !
  ! A fixture SET is one grid, one flag_bep value, one start step and NSTEP
  ! consecutive calls.  Between calls the program advances the state the way
  ! WRF's tendency update would, in float32: x = x + xBLTEN*dt for u, v, th,
  ! qv and qc, and the scheme's own INOUT state (tke_pbl, diss_pbl, tpe_pbl,
  ! pr_pbl, the three *_adv arrays and pblh) carries from one call to the
  ! next untouched.  Only the initial state and every call's outputs are
  ! written: the reader replays the same float32 update to rebuild each
  ! step's inputs, so every input of every call is still a word this
  ! program fed to WRF.
  !
  ! usage: run_bl_keps SET OUT.bin MANIFEST.txt
  !   SET = Acold | Adev | Abep | Bcold | Bdev
  !     A* : 40 levels to ~17 km;  B* : 20 levels to ~2.2 km (PBL reaches
  !          the model top);  *cold : itimestep 1..NSTEP from WRF's
  !          xtime <= 2 reset;  *dev : itimestep 3.. from a developed
  !          TKE / dissipation / TPE state;  Abep : Adev's columns with
  !          flag_bep = .true., an urban fraction and nonzero BEP source
  !          terms, which keps receives and (by the pinned source) never
  !          reads.
  use module_bl_keps, only: keps
  implicit none

  integer, parameter :: ncol = 72
  integer, parameter :: nstep = 8
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.0
  real, parameter :: cp = 7.0 * r_d / 2.0
  real, parameter :: rcp = r_d / cp

  character(len=16) :: setname
  character(len=1024) :: out_path, man_path
  integer :: nz, kme, istep, itime0, ustream, uman, ix, k
  logical :: flag_bep, developed
  real :: dt
  integer(kind=8) :: offset

  real, allocatable, dimension(:,:,:) :: dz8w, u, v, th, pii, raten, p8w, rho
  real, allocatable, dimension(:,:,:) :: qv, qc
  real, allocatable, dimension(:,:,:) :: rub, rvb, rthb, rqvb, rqcb
  real, allocatable, dimension(:,:,:) :: tke, diss, tpe, tkea, dissa, tpea, pr
  real, allocatable, dimension(:,:,:) :: wu, wv, wt, wq, exch_h, exch_m
  real, allocatable, dimension(:,:,:) :: aub, avb, atb, aqb, bub, bvb, btb
  real, allocatable, dimension(:,:,:) :: bqb, beb, sfb, vlb
  real, dimension(ncol, 1) :: ust, hfx, qfx, tsk, mol, br, znt, psim, psih
  real, dimension(ncol, 1) :: frc, pblh

  call get_command_argument(1, setname)
  call get_command_argument(2, out_path)
  call get_command_argument(3, man_path)
  if (len_trim(man_path) == 0) then
    write(*, '(A)') 'usage: run_bl_keps SET OUT.bin MANIFEST.txt'
    error stop 2
  end if

  select case (trim(setname))
  case ('Acold')
    nz = 40; itime0 = 1; flag_bep = .false.; developed = .false.; dt = 30.0
  case ('Adev')
    nz = 40; itime0 = 3; flag_bep = .false.; developed = .true.; dt = 30.0
  case ('Abep')
    nz = 40; itime0 = 3; flag_bep = .true.; developed = .true.; dt = 30.0
  case ('Bcold')
    nz = 20; itime0 = 1; flag_bep = .false.; developed = .false.; dt = 10.0
  case ('Bdev')
    nz = 20; itime0 = 3; flag_bep = .false.; developed = .true.; dt = 10.0
  case default
    write(*, '(A)') 'unknown set ' // trim(setname)
    error stop 2
  end select
  kme = nz + 1

  allocate(dz8w(ncol, kme, 1), u(ncol, kme, 1), v(ncol, kme, 1))
  allocate(th(ncol, kme, 1), pii(ncol, kme, 1), raten(ncol, kme, 1))
  allocate(p8w(ncol, kme, 1), rho(ncol, kme, 1), qv(ncol, kme, 1))
  allocate(qc(ncol, kme, 1), rub(ncol, kme, 1), rvb(ncol, kme, 1))
  allocate(rthb(ncol, kme, 1), rqvb(ncol, kme, 1), rqcb(ncol, kme, 1))
  allocate(tke(ncol, kme, 1), diss(ncol, kme, 1), tpe(ncol, kme, 1))
  allocate(tkea(ncol, kme, 1), dissa(ncol, kme, 1), tpea(ncol, kme, 1))
  allocate(pr(ncol, kme, 1), wu(ncol, kme, 1), wv(ncol, kme, 1))
  allocate(wt(ncol, kme, 1), wq(ncol, kme, 1), exch_h(ncol, kme, 1))
  allocate(exch_m(ncol, kme, 1))
  allocate(aub(ncol, kme, 1), avb(ncol, kme, 1), atb(ncol, kme, 1))
  allocate(aqb(ncol, kme, 1), bub(ncol, kme, 1), bvb(ncol, kme, 1))
  allocate(btb(ncol, kme, 1), bqb(ncol, kme, 1), beb(ncol, kme, 1))
  allocate(sfb(ncol, kme, 1), vlb(ncol, kme, 1))

  call build_columns()

  open(newunit=ustream, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write')
  open(newunit=uman, file=trim(man_path), status='replace', action='write')
  write(uman, '(A,A)') 'set ', trim(setname)
  write(uman, '(A,I0,A,I0,A,I0,A,I0)') 'ncol ', ncol, ' nz ', nz, &
       ' nstep ', nstep, ' itime0 ', itime0
  write(uman, '(A,L1)') 'flag_bep ', flag_bep
  write(uman, '(A,Z8.8)') 'dt ', transfer(dt, 1)
  offset = 0

  ! ---- static inputs ---------------------------------------------------------
  call put3('dz8w', dz8w, nz); call put3('rho', rho, nz)
  call put3('pi_phy', pii, nz); call put3('rthraten', raten, nz)
  call put3('p8w', p8w, nz + 1)
  call put2('ust', ust); call put2('hfx', hfx); call put2('qfx', qfx)
  call put2('tsk', tsk); call put2('mol', mol); call put2('br', br)
  call put2('znt', znt); call put2('psim', psim); call put2('psih', psih)
  call put2('frc_urb2d', frc)
  ! ---- initial evolving state -----------------------------------------------
  call put3('u0', u, nz); call put3('v0', v, nz); call put3('th0', th, nz)
  call put3('qv0', qv, nz); call put3('qc0', qc, nz)
  call put3('tke0', tke, nz); call put3('diss0', diss, nz)
  call put3('tpe0', tpe, nz); call put3('pr0', pr, nz)
  call put3('tke_adv0', tkea, nz); call put3('diss_adv0', dissa, nz)
  call put3('tpe_adv0', tpea, nz); call put2('pblh0', pblh)

  do istep = 1, nstep
    ! INTENT(OUT) arrays the scheme never writes keep this sentinel, so the
    ! fixture records which words keps leaves alone (wq, and the kme level).
    wq = -999.0; wu = -999.0; wv = -999.0; wt = -999.0
    exch_h = -999.0; exch_m = -999.0
    call keps(mol, tsk, itime0 + istep - 1, frc, flag_bep, dz8w, dt, u, v, th, &
              pii, raten, p8w, rho, qv, qc, hfx, qfx, ust, cp, g,             &
              rub, rvb, rthb, rqvb, rqcb, tke, diss, tpe, tkea, dissa, tpea,  &
              pr, wu, wv, wt, wq, exch_h, exch_m, pblh,                       &
              aub, avb, atb, aqb, bub, bvb, btb, bqb, beb, sfb, vlb,          &
              br, znt, psim, psih,                                            &
              1, ncol + 1, 1, 2, 1, kme,                                      &
              1, ncol, 1, 1, 1, kme,                                          &
              1, ncol, 1, 1, 1, nz)
    call put3(step_name('rublten', istep), rub, nz)
    call put3(step_name('rvblten', istep), rvb, nz)
    call put3(step_name('rthblten', istep), rthb, nz)
    call put3(step_name('rqvblten', istep), rqvb, nz)
    call put3(step_name('rqcblten', istep), rqcb, nz)
    call put3(step_name('tke_pbl', istep), tke, nz)
    call put3(step_name('diss_pbl', istep), diss, nz)
    call put3(step_name('tpe_pbl', istep), tpe, nz)
    call put3(step_name('pr_pbl', istep), pr, nz)
    call put3(step_name('tke_adv', istep), tkea, nz)
    call put3(step_name('diss_adv', istep), dissa, nz)
    call put3(step_name('tpe_adv', istep), tpea, nz)
    call put3(step_name('wu_tur', istep), wu, nz)
    call put3(step_name('wv_tur', istep), wv, nz)
    call put3(step_name('wt_tur', istep), wt, nz)
    call put3(step_name('wq_tur', istep), wq, nz)
    call put3(step_name('exch_h', istep), exch_h, nz)
    call put3(step_name('exch_m', istep), exch_m, nz)
    call put2(step_name('pblh', istep), pblh)
    ! WRF's tendency update, float32, one rounding per operator.
    do k = 1, nz
      do ix = 1, ncol
        u(ix, k, 1) = u(ix, k, 1) + rub(ix, k, 1) * dt
        v(ix, k, 1) = v(ix, k, 1) + rvb(ix, k, 1) * dt
        th(ix, k, 1) = th(ix, k, 1) + rthb(ix, k, 1) * dt
        qv(ix, k, 1) = qv(ix, k, 1) + rqvb(ix, k, 1) * dt
        qc(ix, k, 1) = qc(ix, k, 1) + rqcb(ix, k, 1) * dt
      end do
    end do
  end do
  write(uman, '(A,I0)') 'end_words ', offset
  close(ustream)
  close(uman)

contains

  function step_name(base, istep) result(name)
    character(len=*), intent(in) :: base
    integer, intent(in) :: istep
    character(len=64) :: name
    write(name, '(A,A,I0)') trim(base), '@', istep
  end function step_name

  subroutine put3(name, a, nlev)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :, :)
    integer, intent(in) :: nlev
    integer :: kk
    write(uman, '(A,1X,I0,1X,I0,1X,I0)') trim(name), nlev, ncol, offset
    do kk = 1, nlev
      write(ustream) a(:, kk, 1)
    end do
    offset = offset + int(nlev, 8) * ncol
  end subroutine put3

  subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :)
    write(uman, '(A,1X,I0,1X,I0,1X,I0)') trim(name), 0, ncol, offset
    write(ustream) a(:, 1)
    offset = offset + ncol
  end subroutine put2

  ! Deterministic 0..1 sequence, no library random number generator, so a
  ! rebuild on another toolchain makes the same columns.
  real function frac01(i, j)
    integer, intent(in) :: i, j
    integer(kind=8) :: h
    h = mod(int(i, 8) * 7919_8 + int(j, 8) * 104729_8 + 12345_8, 1000003_8)
    h = mod(h * h + 17_8, 1000003_8)
    frac01 = real(h) / 1000003.0
  end function frac01

  subroutine build_columns()
    integer :: fam, var, kk
    real :: z, zc, dzk, zi, dthinv, ths, gam, pk, tk, us0, vs0, ztop
    real :: q0, wsp, z0, a, phase, rh, cloudtop
    real :: zface(kme + 1)
    do ix = 1, ncol
      ! family: 1 convective day, 2 stable night, 3 neutral windy,
      ! 4 cloud-topped, 5 urban-blended surface fluxes, 6 branch probes
      fam = 1 + (ix - 1) / 12
      var = mod(ix - 1, 12)
      ! --- vertical grid --------------------------------------------------
      zface(1) = 0.0
      do kk = 1, nz
        if (nz == 40) then
          dzk = min(900.0, 40.0 * 1.12**(kk - 1))
        else
          dzk = 60.0 * 1.06**(kk - 1)
        end if
        if (fam == 6 .and. var == 7) dzk = dzk * 0.5   ! thin layers
        dz8w(ix, kk, 1) = dzk
        zface(kk + 1) = zface(kk) + dzk
      end do
      dz8w(ix, kme, 1) = 0.0
      ztop = zface(nz + 1)
      ! --- thermodynamic profile -------------------------------------------
      ths = 290.0 + 15.0 * frac01(ix, 1)
      zi = 400.0 + 1800.0 * frac01(ix, 2)
      ! Grid B: the PBL top sits in the last levels GET_PBLH can reach
      ! (zw(kte-3) .. zw(kte-2)), and variant 0 is mixed to the model top so
      ! the search falls through to its exit safeguard.
      if (nz /= 40 .and. (fam == 1 .or. fam == 4) .and. var < 6) &
        zi = 1300.0 + 300.0 * frac01(ix, 2)
      if (nz /= 40 .and. (fam == 1 .or. fam == 4) .and. var == 0) zi = ztop * 1.2
      dthinv = 1.0 + 6.0 * frac01(ix, 3)
      gam = 0.003 + 0.004 * frac01(ix, 4)
      q0 = 0.004 + 0.012 * frac01(ix, 5)
      us0 = -6.0 + 20.0 * frac01(ix, 6)
      vs0 = -8.0 + 16.0 * frac01(ix, 7)
      z0 = 0.01 + 0.8 * frac01(ix, 8)
      cloudtop = zi
      do kk = 1, nz
        zc = 0.5 * (zface(kk) + zface(kk + 1))
        select case (fam)
        case (1, 4, 5)       ! superadiabatic surface layer, mixed layer, cap
          if (zc < 100.0) then
            th(ix, kk, 1) = ths + 1.5 * (1.0 - zc / 100.0)
          else if (zc < zi) then
            th(ix, kk, 1) = ths
          else if (zc < zi + 200.0) then
            th(ix, kk, 1) = ths + dthinv * (zc - zi) / 200.0
          else
            th(ix, kk, 1) = ths + dthinv + gam * (zc - zi - 200.0)
          end if
        case (2)             ! surface inversion
          th(ix, kk, 1) = ths + (2.0 + 8.0 * frac01(ix, 9)) * &
               (1.0 - exp(-zc / (80.0 + 200.0 * frac01(ix, 10)))) + gam * zc
        case (3)             ! near neutral
          th(ix, kk, 1) = ths + 0.0005 * min(zc, 1500.0) + &
               gam * max(zc - 1500.0, 0.0)
        case default         ! probes: mixed shapes
          th(ix, kk, 1) = ths + gam * zc + 0.5 * sin(0.003 * zc + var)
        end select
        qv(ix, kk, 1) = q0 * exp(-zc / 2500.0)
        qc(ix, kk, 1) = 0.0
        if (fam == 4 .and. zc > cloudtop - 250.0 .and. zc < cloudtop) &
          qc(ix, kk, 1) = 2.0e-4 + 4.0e-4 * frac01(ix, kk)
        if (fam == 6 .and. var == 3 .and. kk == 4) qc(ix, kk, 1) = 1.0e-12
        ! winds: log layer then turning with height
        wsp = (0.4 + 0.5 * frac01(ix, 11)) / 0.4 * log((zc + z0) / z0)
        if (fam == 3) wsp = wsp * 2.5
        phase = 0.0006 * zc * (1.0 + frac01(ix, 12))
        if (fam == 2) wsp = wsp * 0.4 + 8.0 * exp(-((zc - 250.0) / 150.0)**2)
        u(ix, kk, 1) = us0 * 0.3 + min(wsp, 25.0) * cos(phase)
        v(ix, kk, 1) = vs0 * 0.3 + min(wsp, 25.0) * sin(phase)
        if (fam == 6 .and. var == 0) then   ! uniform wind: shear exactly 0
          u(ix, kk, 1) = 7.0; v(ix, kk, 1) = -3.0
        end if
        ! hydrostatic-ish pressure and density
        pk = 1.0e5 * exp(-zc / 8400.0)
        pii(ix, kk, 1) = (pk / 1.0e5)**rcp
        tk = th(ix, kk, 1) * pii(ix, kk, 1)
        rho(ix, kk, 1) = pk / (r_d * tk * (1.0 + 0.61 * qv(ix, kk, 1)))
        raten(ix, kk, 1) = -2.0e-5 * (1.0 + frac01(ix, kk + 20))
        if (fam == 4 .and. zc > cloudtop - 150.0 .and. zc < cloudtop + 50.0) &
          raten(ix, kk, 1) = -6.0e-4 * (1.0 + frac01(ix, kk))
      end do
      do kk = 1, nz + 1
        p8w(ix, kk, 1) = 1.0e5 * exp(-zface(kk) / 8400.0)
      end do
      ! --- surface -----------------------------------------------------------
      znt(ix, 1) = z0
      tsk(ix, 1) = th(ix, 1, 1) + 2.0
      select case (fam)
      case (1)
        hfx(ix, 1) = 60.0 + 400.0 * frac01(ix, 13)
        br(ix, 1) = -(0.01 + 1.5 * frac01(ix, 14))
      case (2)
        hfx(ix, 1) = -(5.0 + 60.0 * frac01(ix, 13))
        br(ix, 1) = 0.01 + 0.8 * frac01(ix, 14)
        tsk(ix, 1) = th(ix, 1, 1) - 3.0
      case (3)
        hfx(ix, 1) = 10.0 * frac01(ix, 13) - 3.0
        br(ix, 1) = 0.02 * frac01(ix, 14) - 0.01
      case (4)
        hfx(ix, 1) = 20.0 + 80.0 * frac01(ix, 13)
        br(ix, 1) = -(0.005 + 0.3 * frac01(ix, 14))
      case (5)                ! BEP-blended: HFX_URB+(1-f)*HFX_RURAL, large
        hfx(ix, 1) = 150.0 + 450.0 * frac01(ix, 13)
        br(ix, 1) = -(0.05 + 2.0 * frac01(ix, 14))
        znt(ix, 1) = 0.5 + 1.5 * frac01(ix, 15)
      case default
        hfx(ix, 1) = 200.0 * frac01(ix, 13) - 50.0
        br(ix, 1) = frac01(ix, 14) - 0.5
      end select
      ust(ix, 1) = 0.1 + 0.6 * frac01(ix, 16)
      if (fam == 3) ust(ix, 1) = 0.5 + 0.6 * frac01(ix, 16)
      if (fam == 5) ust(ix, 1) = 0.4 + 0.8 * frac01(ix, 16)
      qfx(ix, 1) = 1.0e-5 + 2.0e-4 * frac01(ix, 17)
      if (br(ix, 1) < 0.0) then
        psim(ix, 1) = 0.2 + 2.0 * frac01(ix, 18)
        psih(ix, 1) = 0.3 + 2.5 * frac01(ix, 19)
      else
        psim(ix, 1) = -(0.1 + 5.0 * frac01(ix, 18))
        psih(ix, 1) = -(0.1 + 6.0 * frac01(ix, 19))
      end if
      mol(ix, 1) = -hfx(ix, 1) / 1200.0 / ust(ix, 1)
      ! --- branch probes (family 6) -------------------------------------------
      if (fam == 6) then
        select case (var)
        case (1)              ! WRF sfclayrev's neutral regime: br = 0, psim = psih = 0
          br(ix, 1) = 0.0; psim(ix, 1) = 0.0; psih(ix, 1) = 0.0
        case (2)              ! negative zero bulk Richardson number
          br(ix, 1) = sign(0.0, -1.0)
        case (4)              ! tiny friction velocity
          ust(ix, 1) = 1.0e-3
        case (5)              ! zero surface fluxes
          hfx(ix, 1) = 0.0; qfx(ix, 1) = 0.0
        case (6)              ! very strong heating
          hfx(ix, 1) = 900.0; br(ix, 1) = -8.0
        case (8)              ! stable with positive psi (other sign convention)
          br(ix, 1) = 0.3; psim(ix, 1) = 1.2; psih(ix, 1) = 1.5
        case (9)              ! calm lowest level
          u(ix, 1, 1) = 0.0; v(ix, 1, 1) = 0.0
        case (10)             ! moist, cloud from the surface layer up
          qc(ix, 1:6, 1) = 5.0e-4
        case (11)             ! roughness close to half the first layer
          znt(ix, 1) = 0.45 * dz8w(ix, 1, 1)
        end select
      end if
    end do
    rho(:, kme, 1) = rho(:, nz, 1)
    u(:, kme, 1) = 0.0; v(:, kme, 1) = 0.0; th(:, kme, 1) = 0.0
    qv(:, kme, 1) = 0.0; qc(:, kme, 1) = 0.0; pii(:, kme, 1) = 0.0
    raten(:, kme, 1) = 0.0
    frc = 0.0
    pblh = 0.0
    ! --- the scheme's own state -----------------------------------------------
    do ix = 1, ncol
      zc = 0.0
      do kk = 1, kme
        if (kk <= nz) zc = zc + dz8w(ix, kk, 1)
        if (developed) then
          a = exp(-zc / (300.0 + 900.0 * frac01(ix, 30)))
          tke(ix, kk, 1) = 1.0e-3 + 2.5 * a * frac01(ix, kk + 40)
          diss(ix, kk, 1) = 1.0e-6 + 8.0e-3 * a * frac01(ix, kk + 80)
          tpe(ix, kk, 1) = 1.0e-5 + 0.3 * a * frac01(ix, kk + 120)
          pr(ix, kk, 1) = 1.0 + 2.0 * frac01(ix, kk + 160)
          ! te/d above 1e4 with d at the floor: thetaphi_alphabeta's
          ! length-scale reset of the dissipation (module_bl_keps.F:914)
          if (mod(ix + kk, 9) == 0) diss(ix, kk, 1) = 1.0e-7
        else
          tke(ix, kk, 1) = 0.0; diss(ix, kk, 1) = 0.0
          tpe(ix, kk, 1) = 0.0; pr(ix, kk, 1) = 0.0
        end if
        tkea(ix, kk, 1) = tke(ix, kk, 1)
        dissa(ix, kk, 1) = diss(ix, kk, 1)
        tpea(ix, kk, 1) = tpe(ix, kk, 1)
      end do
    end do
    ! --- BEP source terms: zero unless flag_bep --------------------------------
    aub = 0.0; avb = 0.0; atb = 0.0; aqb = 0.0; bub = 0.0; bvb = 0.0
    btb = 0.0; bqb = 0.0; beb = 0.0; sfb = 1.0; vlb = 1.0
    if (flag_bep) then
      do ix = 1, ncol
        frc(ix, 1) = frac01(ix, 200)
        do kk = 1, kme
          aub(ix, kk, 1) = -1.0e-3 * frac01(ix, kk + 201)
          avb(ix, kk, 1) = -1.0e-3 * frac01(ix, kk + 202)
          atb(ix, kk, 1) = -1.0e-4 * frac01(ix, kk + 203)
          bub(ix, kk, 1) = 1.0e-3 * (frac01(ix, kk + 204) - 0.5)
          bvb(ix, kk, 1) = 1.0e-3 * (frac01(ix, kk + 205) - 0.5)
          btb(ix, kk, 1) = 5.0e-3 * frac01(ix, kk + 206)
          bqb(ix, kk, 1) = 1.0e-6 * frac01(ix, kk + 207)
          beb(ix, kk, 1) = 1.0e-3 * frac01(ix, kk + 208)
          sfb(ix, kk, 1) = 1.0 - 0.6 * frac01(ix, kk + 209)
          vlb(ix, kk, 1) = 1.0 - 0.4 * frac01(ix, kk + 210)
        end do
      end do
    end if
  end subroutine build_columns

end program run_bl_keps_oracle
