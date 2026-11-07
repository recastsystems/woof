program run_morrison_aero_oracle
  ! Drive the public WRF v4.6.1 MP_MORR_TWO_MOMENT_AERO wrapper (mp_physics=40)
  ! from the byte-unmodified module.  Every expected value written here is a
  ! value returned by WRF; this program constructs inputs and never
  ! reimplements Morrison arithmetic.  Values are written as FP32 bit patterns
  ! (hex) so the comparison is word for word.
  !
  ! usage: run_morrison_aero RIMED AERCU_OPT DT LAYOUT LEVELS.csv SURFACE.csv
  !   RIMED      morr_rimed_ice (0 graupel, 1 hail)
  !   AERCU_OPT  0, 1 or 2 (2 = prescribed CESM aerosol activation/nucleation)
  !   DT         model time step (s)
  !   LAYOUT     batch  : one wrapper call over all columns (WRF's tile loop)
  !              single : one wrapper call per column (its=ite)
  use module_model_constants, only: cp, ep_2, p1000mb, r_d
  use module_mp_morr_two_moment_aero, only: morr_two_moment_init_aero, &
                                             mp_morr_two_moment_aero, polysvp
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: ncol = 84
  integer, parameter :: nsrc = 10
  integer, parameter :: kme = nz + 1

  real, dimension(ncol, kme, 1) :: th, qv, qc, qr, qi, qs, qg
  real, dimension(ncol, kme, 1) :: ni, ns, nr, ng, nc
  real, dimension(ncol, kme, 1) :: rho, pii, pres, dz, w, kzh
  real, dimension(ncol, kme, 1) :: qrcuten, qscuten, qicuten
  real, dimension(ncol, kme, 1) :: refl, mskf_refl
  real, dimension(ncol, kme, 1) :: efcg, efig, efsg, wact
  real, dimension(ncol, kme, 1) :: ccn1, ccn2, ccn3, ccn4, ccn5, ccn6, ccn7
  real, dimension(ncol, kme, 1) :: nr_cu, qr_cu, ns_cu, qs_cu
  real, dimension(ncol, kme, 1, nsrc) :: aerocu
  real, dimension(ncol, 1) :: ht, cu_uaf
  real, dimension(ncol, 1) :: rainnc, rainncv, sr, snownc, snowncv
  real, dimension(ncol, 1) :: graupelnc, graupelncv

  ! input snapshots
  real, dimension(ncol, kme, 1) :: th0, qv0, qc0, qr0, qi0, qs0, qg0
  real, dimension(ncol, kme, 1) :: ni0, ns0, nr0, ng0, nc0
  real, dimension(ncol, kme, 1) :: rho0, pii0, pres0, dz0, w0, kzh0
  real, dimension(ncol, kme, 1) :: qrcu0, qscu0, qicu0
  real, dimension(ncol, kme, 1, nsrc) :: aero0
  real, dimension(ncol, 1) :: rainnc0, rainncv0, sr0, snownc0, snowncv0
  real, dimension(ncol, 1) :: graupelnc0, graupelncv0

  character(len=1024) :: arg, level_path, surface_path, layout
  integer :: rimed, aercu_opt, i, k, l, ulev, usfc, ios
  real :: dt, aercu_fct

  call get_command_argument(1, arg); read(arg, *, iostat=ios) rimed
  if (ios /= 0) call usage()
  call get_command_argument(2, arg); read(arg, *, iostat=ios) aercu_opt
  if (ios /= 0) call usage()
  call get_command_argument(3, arg); read(arg, *, iostat=ios) dt
  if (ios /= 0) call usage()
  call get_command_argument(4, layout)
  call get_command_argument(5, level_path)
  call get_command_argument(6, surface_path)
  if (len_trim(surface_path) == 0) call usage()
  if (storage_size(1.0) /= 32) error stop 3

  aercu_fct = 1.0
  call build_inputs(rimed, aercu_opt)
  call snapshot()

  call morr_two_moment_init_aero(rimed)

  if (trim(layout) == 'batch') then
    call one_call(1, ncol)
  else if (trim(layout) == 'single') then
    do i = 1, ncol
      call one_call(i, i)
    end do
  else
    call usage()
  end if

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'col,k,rimed,aercu_opt,dt,' // &
    'th_in,qv_in,qc_in,qr_in,qi_in,qs_in,qg_in,ni_in,ns_in,nr_in,ng_in,nc_in,' // &
    'pii_in,p_in,dz_in,w_in,kzh_up_in,qrcuten_in,qscuten_in,qicuten_in,' // &
    'aero1_in,aero2_in,aero3_in,aero4_in,aero5_in,aero6_in,aero7_in,' // &
    'aero8_in,aero9_in,aero10_in,' // &
    'th_out,qv_out,qc_out,qr_out,qi_out,qs_out,qg_out,ni_out,ns_out,' // &
    'nr_out,ng_out,nc_out,refl_10cm_out,mskf_refl_10cm_out,efcg_out,' // &
    'efig_out,efsg_out,wact_out,ccn1_out,ccn2_out,ccn3_out,ccn4_out,' // &
    'ccn5_out,ccn6_out,ccn7_out'
  do i = 1, ncol
    do k = 1, nz
      write(ulev, '(I0,",",I0,",",I0,",",I0,",",Z8.8)', advance='no') &
        i, k, rimed, aercu_opt, transfer(dt, 0)
      call hx(ulev, [th0(i,k,1), qv0(i,k,1), qc0(i,k,1), qr0(i,k,1), &
        qi0(i,k,1), qs0(i,k,1), qg0(i,k,1), ni0(i,k,1), ns0(i,k,1), &
        nr0(i,k,1), ng0(i,k,1), nc0(i,k,1), pii0(i,k,1), pres0(i,k,1), &
        dz0(i,k,1), w0(i,k,1), kzh0(i,k+1,1), qrcu0(i,k,1), qscu0(i,k,1), &
        qicu0(i,k,1), (aero0(i,k,1,l), l = 1, nsrc), &
        th(i,k,1), qv(i,k,1), qc(i,k,1), qr(i,k,1), qi(i,k,1), qs(i,k,1), &
        qg(i,k,1), ni(i,k,1), ns(i,k,1), nr(i,k,1), ng(i,k,1), nc(i,k,1), &
        refl(i,k,1), mskf_refl(i,k,1), efcg(i,k,1), efig(i,k,1), &
        efsg(i,k,1), wact(i,k,1), ccn1(i,k,1), ccn2(i,k,1), ccn3(i,k,1), &
        ccn4(i,k,1), ccn5(i,k,1), ccn6(i,k,1), ccn7(i,k,1)])
      write(ulev, '(A)') ''
    end do
  end do
  close(ulev)

  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'col,rimed,aercu_opt,dt,' // &
    'rainnc_in,rainncv_in,snownc_in,snowncv_in,graupelnc_in,' // &
    'graupelncv_in,sr_in,rainnc_out,rainncv_out,snownc_out,snowncv_out,' // &
    'graupelnc_out,graupelncv_out,sr_out'
  do i = 1, ncol
    write(usfc, '(I0,",",I0,",",I0,",",Z8.8)', advance='no') &
      i, rimed, aercu_opt, transfer(dt, 0)
    call hx(usfc, [rainnc0(i,1), rainncv0(i,1), snownc0(i,1), &
      snowncv0(i,1), graupelnc0(i,1), graupelncv0(i,1), sr0(i,1), &
      rainnc(i,1), rainncv(i,1), snownc(i,1), snowncv(i,1), &
      graupelnc(i,1), graupelncv(i,1), sr(i,1)])
    write(usfc, '(A)') ''
  end do
  close(usfc)

contains

  subroutine usage()
    write(*, '(A)') 'usage: run_morrison_aero RIMED AERCU_OPT DT batch|single LEVELS.csv SURFACE.csv'
    error stop 2
  end subroutine usage

  subroutine hx(u, values)
    integer, intent(in) :: u
    real, intent(in) :: values(:)
    integer :: n
    do n = 1, size(values)
      write(u, '(",",Z8.8)', advance='no') transfer(values(n), 0)
    end do
  end subroutine hx

  subroutine one_call(i0, i1)
    integer, intent(in) :: i0, i1
    call mp_morr_two_moment_aero(                                          &
      itimestep=1, th=th, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,        &
      ni=ni, ns=ns, nr=nr, ng=ng, nc=nc, rho=rho, pii=pii, p=pres,         &
      dt_in=dt, dz=dz, ht=ht, w=w, kzh=kzh,                                &
      rainnc=rainnc, rainncv=rainncv, sr=sr, snownc=snownc,                &
      snowncv=snowncv, graupelnc=graupelnc, graupelncv=graupelncv,         &
      refl_10cm=refl, mskf_refl_10cm=mskf_refl, diagflag=.true.,           &
      do_radar_ref=1, qrcuten=qrcuten, qscuten=qscuten, qicuten=qicuten,   &
      ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=kme,                       &
      ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=kme,                       &
      its=i0, ite=i1, jts=1, jte=1, kts=1, kte=nz,                         &
      pbl=1, aerocu=aerocu, aercu_opt=aercu_opt, aercu_fct=aercu_fct,      &
      no_src_types_cu=nsrc, efcg=efcg, efig=efig, efsg=efsg, wact=wact,    &
      ccn1_gs=ccn1, ccn2_gs=ccn2, ccn3_gs=ccn3, ccn4_gs=ccn4,              &
      ccn5_gs=ccn5, ccn6_gs=ccn6, ccn7_gs=ccn7, nr_cu=nr_cu, qr_cu=qr_cu,  &
      ns_cu=ns_cu, qs_cu=qs_cu, cu_uaf=cu_uaf)
  end subroutine one_call

  subroutine snapshot()
    th0 = th; qv0 = qv; qc0 = qc; qr0 = qr; qi0 = qi; qs0 = qs; qg0 = qg
    ni0 = ni; ns0 = ns; nr0 = nr; ng0 = ng; nc0 = nc
    rho0 = rho; pii0 = pii; pres0 = pres; dz0 = dz; w0 = w; kzh0 = kzh
    qrcu0 = qrcuten; qscu0 = qscuten; qicu0 = qicuten; aero0 = aerocu
    rainnc0 = rainnc; rainncv0 = rainncv; sr0 = sr; snownc0 = snownc
    snowncv0 = snowncv; graupelnc0 = graupelnc; graupelncv0 = graupelncv
  end subroutine snapshot

  real function gauss(z, zc, width)
    real, intent(in) :: z, zc, width
    gauss = exp(-((z - zc) / width) ** 2)
  end function gauss

  subroutine build_inputs(rimed_mode, aero_mode)
    integer, intent(in) :: rimed_mode, aero_mode
    integer :: c, kk, sel
    real :: z, zb, dzk, psfc, tsfc, t, ew, ei, qsat, qisat, rh, sc, sp, sa
    real :: nzero, subn, minnorm, thr, wpk, wz, clean, dust, salt, soot
    real :: zmid, rhok, tk

    nzero = sign(0.0, -1.0)
    subn = transfer(1, 0.0)
    minnorm = transfer(8388608, 0.0)

    th = 0.0; qv = 0.0; qc = 0.0; qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0
    ni = 0.0; ns = 0.0; nr = 0.0; ng = 0.0; nc = 0.0
    rho = 0.0; pii = 0.0; pres = 0.0; dz = 0.0; w = 0.0; kzh = 0.0
    qrcuten = 0.0; qscuten = 0.0; qicuten = 0.0
    refl = 0.0; mskf_refl = 0.0
    efcg = 0.0; efig = 0.0; efsg = 0.0; wact = 0.0
    ccn1 = 0.0; ccn2 = 0.0; ccn3 = 0.0; ccn4 = 0.0; ccn5 = 0.0
    ccn6 = 0.0; ccn7 = 0.0
    nr_cu = 0.0; qr_cu = 0.0; ns_cu = 0.0; qs_cu = 0.0; cu_uaf = 0.0
    aerocu = 0.0; ht = 0.0

    do c = 1, 28
      ! column-specific surface state
      psfc = 100000.0
      tsfc = 300.0
      select case (c)
      case (1, 2, 3)                ! deep convection
        tsfc = 303.0 - real(c - 1)
      case (4, 5)                   ! stratiform with melting layer
        tsfc = 281.0 + 2.0 * real(c - 4)
      case (6)                      ! warm rain shallow cumulus
        tsfc = 299.0
      case (7)                      ! stratocumulus drizzle
        tsfc = 289.0
      case (8)                      ! arctic
        tsfc = 248.0
      case (9)                      ! tropical tropopause cirrus, < 195.8 K
        tsfc = 302.0
      case (10)                     ! high terrain
        psfc = 61000.0
        tsfc = 279.0
      case (11)                     ! dry subsaturated, small condensate
        tsfc = 306.0
      case (12)                     ! clear dry column: no hydrometeors
        tsfc = 295.0
      case (13)                     ! clear supersaturated column
        tsfc = 288.0
      case (14)                     ! cumulus tendencies seed numbers
        tsfc = 297.0
      case (15)                     ! threshold neighbours
        tsfc = 285.0
      case (16)                     ! signed zeros and subnormals
        tsfc = 284.0
      case (17)                     ! slope clamps
        tsfc = 290.0
      case (18)                     ! Hallett-Mossop riming
        tsfc = 278.0
      case (19)                     ! homogeneous freezing
        tsfc = 262.0
      case (20)                     ! heavy sedimentation, thin layers
        tsfc = 296.0
      case (21)                     ! large ice to snow
        tsfc = 270.0
      case (22)                     ! evaporating rain under cloud base
        tsfc = 309.0
      case (23)                     ! downdraft, cold cloud
        tsfc = 268.0
      case (24)                     ! melting at exactly 273.15
        tsfc = 276.0
      case (25)                     ! polluted continental cloud
        tsfc = 295.0
      case (26)                     ! clean marine cloud
        tsfc = 293.0
      case (27)                     ! dusty cold cloud
        tsfc = 272.0
      case (28)                     ! cirrus outflow with updraft, cold
        tsfc = 290.0
      end select

      zb = 0.0
      do kk = 1, nz
        dzk = 180.0 + 14.0 * real(kk)
        if (c == 20) dzk = 60.0 + 6.0 * real(kk)
        if (c == 10) dzk = 150.0 + 10.0 * real(kk)
        zmid = zb + 0.5 * dzk
        z = zmid
        dz(c,kk,1) = dzk
        pres(c,kk,1) = psfc * exp(-z / 7600.0)

        ! temperature: tropospheric lapse with a tropopause
        t = tsfc - 0.0068 * z
        select case (c)
        case (8)
          t = tsfc + 0.004 * min(z, 800.0) - 0.0055 * max(z - 800.0, 0.0)
        case (9)
          t = tsfc - 0.0072 * z
        case (19)
          t = tsfc - 0.0075 * z
        case (24)
          if (kk == 6) t = 273.15
        end select
        t = max(t, merge(183.0, 205.0, c == 9 .or. c == 28))
        if (c == 18 .and. kk >= 4 .and. kk <= 14) &
          t = 265.0 + 0.55 * real(kk - 4)
        pii(c,kk,1) = (pres(c,kk,1) / p1000mb) ** (r_d / cp)
        th(c,kk,1) = t / pii(c,kk,1)
        rhok = pres(c,kk,1) / (r_d * t)
        rho(c,kk,1) = rhok
        tk = t

        ew = min(0.99 * pres(c,kk,1), polysvp(tk, 0))
        ei = min(0.99 * pres(c,kk,1), polysvp(tk, 1))
        if (ei > ew) ei = ew
        qsat = ep_2 * ew / (pres(c,kk,1) - ew)
        qisat = ep_2 * ei / (pres(c,kk,1) - ei)

        sc = gauss(z, 3000.0, 1800.0)
        sp = gauss(z, 2200.0, 2600.0)
        sa = gauss(z, 7500.0, 2500.0)

        ! updraft / turbulence
        wpk = 0.0
        select case (c)
        case (1, 2, 3)
          wpk = 6.0 + 4.0 * real(c)
        case (14, 25, 26, 27)
          wpk = 2.5
        case (23)
          wpk = -4.0
        case (28)
          wpk = 1.5
        case (4, 5, 10)
          wpk = 0.3
        case default
          wpk = 0.05 * real(mod(c, 5)) - 0.1
        end select
        wz = wpk * sin(3.14159265 * min(z, 12000.0) / 12000.0)
        w(c,kk,1) = wz
        kzh(c,kk,1) = 60.0 * exp(-z / 1200.0) + 0.5 * real(mod(kk + c, 3))

        rh = 0.80
        select case (c)
        case (1, 2, 3)
          rh = merge(1.02, 1.005, t < 273.15)
          qc(c,kk,1) = (1.0e-3 + 4.0e-4 * real(c)) * sc
          qr(c,kk,1) = (1.5e-3 + 1.5e-3 * real(c)) * sp
          nr(c,kk,1) = (2.0e5 + 1.0e5 * real(c)) / rhok * sp
          if (t < 273.15) then
            qi(c,kk,1) = 3.0e-4 * sa
            ni(c,kk,1) = 1.5e5 / rhok * sa
            qs(c,kk,1) = 1.2e-3 * sa
            ns(c,kk,1) = 2.0e5 / rhok * sa
            qg(c,kk,1) = (1.0e-3 * real(c)) * gauss(z, 5500.0, 2200.0)
            ng(c,kk,1) = 2.0e4 / rhok * gauss(z, 5500.0, 2200.0)
            qc(c,kk,1) = qc(c,kk,1) + 6.0e-4 * gauss(z, 6500.0, 1500.0)
          end if
          if (c == 3) qr(c,kk,1) = qr(c,kk,1) + 2.0e-3 * gauss(z, 6000.0, 1500.0)
          if (c == 3) nr(c,kk,1) = nr(c,kk,1) + 3.0e5 / rhok * gauss(z, 6000.0, 1500.0)
        case (4, 5)
          rh = merge(1.05, 0.97, t < 273.15)
          if (t < 273.15) then
            qs(c,kk,1) = 6.0e-4 * gauss(z, 3000.0, 2000.0)
            ns(c,kk,1) = 1.0e5 / rhok * gauss(z, 3000.0, 2000.0)
            qi(c,kk,1) = 1.0e-4 * sa
            ni(c,kk,1) = 6.0e4 / rhok * sa
            qg(c,kk,1) = 1.0e-4 * real(c - 3) * gauss(z, 2500.0, 1500.0)
            ng(c,kk,1) = 1.0e4 / rhok * gauss(z, 2500.0, 1500.0)
          else
            qr(c,kk,1) = 4.0e-4 * gauss(z, 400.0, 1500.0)
            nr(c,kk,1) = 5.0e4 / rhok * gauss(z, 400.0, 1500.0)
            qs(c,kk,1) = 5.0e-6 * real(c - 3)
            ns(c,kk,1) = 1.0e3 / rhok
            qg(c,kk,1) = 2.0e-6 * real(c - 3)
            ng(c,kk,1) = 5.0e2 / rhok
          end if
        case (6)
          rh = merge(1.01, 0.9, z > 600.0 .and. z < 2500.0)
          qc(c,kk,1) = 1.2e-3 * gauss(z, 1500.0, 700.0)
          qr(c,kk,1) = 3.0e-4 * gauss(z, 900.0, 900.0)
          nr(c,kk,1) = 8.0e4 / rhok * gauss(z, 900.0, 900.0)
        case (7)
          rh = merge(1.003, 0.7, z > 300.0 .and. z < 1100.0)
          qc(c,kk,1) = 4.0e-4 * gauss(z, 800.0, 300.0)
          qr(c,kk,1) = 2.0e-6 * gauss(z, 500.0, 500.0)
          nr(c,kk,1) = 2.0e5 / rhok * gauss(z, 500.0, 500.0)
        case (8)
          rh = 1.0
          qv(c,kk,1) = 1.03 * qisat
          qi(c,kk,1) = 4.0e-5 * gauss(z, 1500.0, 1200.0)
          ni(c,kk,1) = 2.0e4 / rhok * gauss(z, 1500.0, 1200.0)
          qs(c,kk,1) = 1.5e-4 * gauss(z, 1000.0, 1500.0)
          ns(c,kk,1) = 3.0e4 / rhok * gauss(z, 1000.0, 1500.0)
          qc(c,kk,1) = 2.0e-5 * gauss(z, 700.0, 300.0)
        case (9, 28)
          rh = merge(1.15, 0.6, t < 230.0)
          qi(c,kk,1) = 1.5e-5 * merge(1.0, 0.0, t < 230.0)
          ni(c,kk,1) = 2.0e5 / rhok * merge(1.0, 0.0, t < 230.0)
          qs(c,kk,1) = 3.0e-5 * merge(1.0, 0.0, t < 215.0)
          ns(c,kk,1) = 1.0e4 / rhok * merge(1.0, 0.0, t < 215.0)
          if (c == 28) qc(c,kk,1) = 1.0e-5 * merge(1.0, 0.0, t > 236.0 .and. t < 250.0)
        case (10)
          rh = 1.02
          qc(c,kk,1) = 5.0e-4 * sc
          qs(c,kk,1) = 4.0e-4 * sp * merge(1.0, 0.0, t < 273.15)
          ns(c,kk,1) = 8.0e4 / rhok * sp
          qr(c,kk,1) = 2.0e-4 * sp * merge(1.0, 0.0, t >= 273.15)
          nr(c,kk,1) = 4.0e4 / rhok * sp
          qg(c,kk,1) = 2.0e-4 * sp * merge(1.0, 0.0, t < 270.0)
          ng(c,kk,1) = 5.0e3 / rhok * sp
        case (11)
          rh = 0.35
          sel = mod(kk, 4)
          qc(c,kk,1) = merge(5.0e-9, 2.0e-8, sel == 0)
          qr(c,kk,1) = merge(7.0e-9, 3.0e-7, sel == 1)
          nr(c,kk,1) = 1.0e3
          qs(c,kk,1) = merge(6.0e-9, 4.0e-7, sel == 2)
          ns(c,kk,1) = 2.0e3
          qi(c,kk,1) = merge(9.0e-9, 5.0e-8, sel == 3)
          ni(c,kk,1) = 3.0e3
          qg(c,kk,1) = 4.0e-9
          ng(c,kk,1) = 10.0
        case (12)
          rh = 0.4
        case (13)
          rh = merge(1.012, 0.995, mod(kk, 2) == 0)
        case (14)
          rh = 0.98
          qc(c,kk,1) = 2.0e-4 * sc
          qr(c,kk,1) = 1.0e-4 * sp
          nr(c,kk,1) = 2.0e4 / rhok * sp
          qi(c,kk,1) = 5.0e-5 * sa * merge(1.0, 0.0, t < 273.15)
          ni(c,kk,1) = 1.0e4 / rhok * sa
          qs(c,kk,1) = 1.0e-4 * sa * merge(1.0, 0.0, t < 273.15)
          ns(c,kk,1) = 1.0e4 / rhok * sa
          sel = mod(kk - 1, 7)
          select case (sel)
          case (0)
            qrcuten(c,kk,1) = 0.0
            qscuten(c,kk,1) = nzero
            qicuten(c,kk,1) = subn
          case (1)
            qrcuten(c,kk,1) = nearest(1.0e-10, -1.0)
            qscuten(c,kk,1) = 1.0e-10
            qicuten(c,kk,1) = nearest(1.0e-10, 1.0)
          case (2)
            qrcuten(c,kk,1) = 2.0e-7
            qscuten(c,kk,1) = 3.0e-8
            qicuten(c,kk,1) = 4.0e-9
          case default
            qrcuten(c,kk,1) = 2.0e-8 * real(sel)
            qscuten(c,kk,1) = 1.0e-8 * real(sel)
            qicuten(c,kk,1) = 0.5e-8 * real(sel)
          end select
        case (15)
          rh = 0.93
          sel = mod(kk - 1, 10)
          select case (sel)
          case (0); thr = 0.0
          case (1); thr = nearest(1.0e-14, -1.0)
          case (2); thr = 1.0e-14
          case (3); thr = nearest(1.0e-14, 1.0)
          case (4); thr = nearest(1.0e-8, -1.0)
          case (5); thr = 1.0e-8
          case (6); thr = nearest(1.0e-6, -1.0)
          case (7); thr = 1.0e-6
          case (8); thr = 0.1e-3
          case default; thr = 0.5e-3
          end select
          qc(c,kk,1) = thr
          qr(c,kk,1) = thr
          qi(c,kk,1) = thr
          qs(c,kk,1) = thr
          qg(c,kk,1) = thr
          ni(c,kk,1) = merge(5.0e4 / rhok, thr, thr >= 1.0e-14)
          ns(c,kk,1) = ni(c,kk,1)
          nr(c,kk,1) = ni(c,kk,1)
          ng(c,kk,1) = ni(c,kk,1)
          if (mod(kk, 3) == 0) rh = 0.85
        case (16)
          sel = mod(kk - 1, 6)
          rh = 0.95
          select case (sel)
          case (0)
            qv(c,kk,1) = 0.0; qc(c,kk,1) = nzero; nr(c,kk,1) = -subn
          case (1)
            qv(c,kk,1) = nzero; qr(c,kk,1) = subn; ni(c,kk,1) = nzero
          case (2)
            qv(c,kk,1) = subn; qi(c,kk,1) = nzero; ns(c,kk,1) = subn
          case (3)
            qv(c,kk,1) = minnorm; qs(c,kk,1) = subn; ng(c,kk,1) = nzero
          case (4)
            qv(c,kk,1) = nearest(minnorm, -1.0); qg(c,kk,1) = -subn
          case default
            qv(c,kk,1) = nearest(minnorm, 1.0); qc(c,kk,1) = subn
            ng(c,kk,1) = subn
          end select
          if (kk > 24) then
            qv(c,kk,1) = 0.0
            qc(c,kk,1) = 3.0e-4
            qr(c,kk,1) = nzero
            nr(c,kk,1) = nzero
          end if
        case (17)
          rh = 1.0
          sel = mod(kk, 4)
          qr(c,kk,1) = 1.0e-4
          nr(c,kk,1) = merge(1.0e1, 1.0e9, sel < 2)
          qs(c,kk,1) = merge(2.0e-4, 0.0, t < 273.15)
          ns(c,kk,1) = merge(1.0, 1.0e10, sel == 0 .or. sel == 2)
          qi(c,kk,1) = merge(5.0e-5, 0.0, t < 273.15)
          ni(c,kk,1) = merge(1.0e-2, 1.0e12, sel == 1 .or. sel == 2)
          qg(c,kk,1) = merge(3.0e-4, 0.0, t < 273.15)
          ng(c,kk,1) = merge(1.0e-1, 1.0e9, sel >= 2)
          qc(c,kk,1) = 2.0e-4
        case (18)
          rh = 1.015
          qs(c,kk,1) = 6.0e-4 * gauss(z, 2000.0, 1500.0)
          ns(c,kk,1) = 2.0e5 / rhok * gauss(z, 2000.0, 1500.0)
          qc(c,kk,1) = 9.0e-4 * gauss(z, 1800.0, 1200.0)
          qr(c,kk,1) = 4.0e-4 * gauss(z, 1500.0, 1200.0)
          nr(c,kk,1) = 6.0e4 / rhok * gauss(z, 1500.0, 1200.0)
          qg(c,kk,1) = 5.0e-4 * gauss(z, 2100.0, 1300.0)
          ng(c,kk,1) = 2.0e4 / rhok * gauss(z, 2100.0, 1300.0)
          qi(c,kk,1) = 2.0e-5 * sc
          ni(c,kk,1) = 2.0e4 / rhok * sc
        case (19)
          rh = 1.01
          qc(c,kk,1) = 6.0e-4 * gauss(z, 3500.0, 2000.0)
          qr(c,kk,1) = 5.0e-4 * gauss(z, 3500.0, 2000.0)
          nr(c,kk,1) = 5.0e4 / rhok * gauss(z, 3500.0, 2000.0)
          qi(c,kk,1) = 1.0e-4 * sc
          ni(c,kk,1) = 1.0e5 / rhok * sc
        case (20)
          rh = 0.9
          qr(c,kk,1) = 7.0e-3 * gauss(z, 1200.0, 900.0)
          nr(c,kk,1) = 3.0e3 / rhok * gauss(z, 1200.0, 900.0) + 1.0
          qg(c,kk,1) = 4.0e-3 * gauss(z, 1800.0, 600.0) * merge(1.0, 0.0, t < 280.0)
          ng(c,kk,1) = 2.0e3 / rhok * gauss(z, 1800.0, 600.0)
          qc(c,kk,1) = 5.0e-4 * gauss(z, 1500.0, 500.0)
        case (21)
          rh = 1.06
          qi(c,kk,1) = 2.0e-3 * gauss(z, 2500.0, 1500.0)
          ni(c,kk,1) = 2.0e2 / rhok * gauss(z, 2500.0, 1500.0) + 1.0e-3
          qs(c,kk,1) = 1.0e-4 * sp
          ns(c,kk,1) = 1.0e4 / rhok * sp
        case (22)
          rh = merge(1.01, 0.3, z > 3500.0 .and. z < 5000.0)
          qc(c,kk,1) = 6.0e-4 * gauss(z, 4200.0, 600.0)
          qr(c,kk,1) = 9.0e-4 * gauss(z, 2500.0, 1500.0)
          nr(c,kk,1) = 2.0e4 / rhok * gauss(z, 2500.0, 1500.0)
        case (23)
          rh = 1.01
          qc(c,kk,1) = 4.0e-4 * sc
          qi(c,kk,1) = 2.0e-4 * sc
          ni(c,kk,1) = 5.0e4 / rhok * sc
          qs(c,kk,1) = 6.0e-4 * sp
          ns(c,kk,1) = 9.0e4 / rhok * sp
          qg(c,kk,1) = 3.0e-4 * sp
          ng(c,kk,1) = 1.0e4 / rhok * sp
        case (24)
          rh = 1.0
          qs(c,kk,1) = 3.0e-4 * gauss(z, 1200.0, 900.0)
          ns(c,kk,1) = 6.0e4 / rhok * gauss(z, 1200.0, 900.0)
          qg(c,kk,1) = 2.0e-4 * gauss(z, 1200.0, 900.0)
          ng(c,kk,1) = 8.0e3 / rhok * gauss(z, 1200.0, 900.0)
          qr(c,kk,1) = 3.0e-4 * gauss(z, 600.0, 700.0)
          nr(c,kk,1) = 3.0e4 / rhok * gauss(z, 600.0, 700.0)
          qi(c,kk,1) = 2.0e-5 * gauss(z, 1200.0, 900.0)
          ni(c,kk,1) = 1.0e4 / rhok * gauss(z, 1200.0, 900.0)
        case (25, 26, 27)
          rh = merge(1.01, 0.92, z > 800.0 .and. z < 4500.0)
          qc(c,kk,1) = 8.0e-4 * gauss(z, 2500.0, 1200.0)
          qr(c,kk,1) = 2.0e-4 * gauss(z, 1500.0, 1200.0)
          nr(c,kk,1) = 3.0e4 / rhok * gauss(z, 1500.0, 1200.0)
          if (t < 273.15) then
            qi(c,kk,1) = 5.0e-5 * sc
            ni(c,kk,1) = 2.0e4 / rhok * sc
            qs(c,kk,1) = 2.0e-4 * sc
            ns(c,kk,1) = 3.0e4 / rhok * sc
          end if
        end select

        if (c /= 8 .and. c /= 16) then
          qv(c,kk,1) = rh * qsat
        end if
        if (c == 16 .and. kk > 24) qv(c,kk,1) = 0.97 * qsat

        ! prognostic droplet number (read only under aercu_opt = 2)
        nc(c,kk,1) = merge(1.0e8, 0.0, qc(c,kk,1) > 1.0e-6) + 1.0e3 * real(mod(kk, 3))

        ! prescribed CESM aerosol (ug m-3), species order of the Registry
        ! aerocu package: dust1..4, seasalt, sulfate, phobcar, phibcar,
        ! phoocar, phiocar.
        clean = 1.0
        dust = 0.0
        salt = 0.0
        soot = 1.0
        select case (c)
        case (25, 1, 22, 11)
          clean = 25.0       ! polluted
          soot = 6.0
        case (26, 6, 7)
          clean = 0.2        ! clean marine
          salt = 12.0
          soot = 0.05
        case (27, 23, 28, 9)
          dust = 40.0        ! dusty
          clean = 2.0
        case default
          clean = 1.0 + 0.1 * real(c)
          dust = 0.5
          salt = 0.5
        end select
        sa = exp(-z / 2200.0)
        aerocu(c,kk,1,1) = 0.30 * dust * sa
        aerocu(c,kk,1,2) = 0.35 * dust * sa
        aerocu(c,kk,1,3) = 0.25 * dust * sa
        aerocu(c,kk,1,4) = 0.10 * dust * sa
        aerocu(c,kk,1,5) = salt * exp(-z / 900.0)
        aerocu(c,kk,1,6) = 2.5 * clean * sa + 0.05
        aerocu(c,kk,1,7) = 0.08 * soot * sa
        aerocu(c,kk,1,8) = 0.12 * soot * sa
        aerocu(c,kk,1,9) = 0.6 * clean * sa
        aerocu(c,kk,1,10) = 0.9 * clean * sa

        zb = zb + dzk
      end do
      kzh(c,kme,1) = 0.25

      rainnc(c,1) = 0.125 * real(c) + 0.5 * real(rimed_mode)
      rainncv(c,1) = nzero
      snownc(c,1) = 0.25 * real(c) + 0.75 * real(rimed_mode)
      snowncv(c,1) = 0.0
      graupelnc(c,1) = 0.375 * real(c) + real(aero_mode)
      graupelncv(c,1) = nzero
      sr(c,1) = nzero
    end do
    ! 28 thermodynamic seeds, each with three aerosol and updraft states.
    do c = 29, ncol
      sel = 1 + mod(c-1, 28)
      th(c,:,1) = th(sel,:,1)
      qv(c,:,1) = qv(sel,:,1)
      qc(c,:,1) = qc(sel,:,1)
      qr(c,:,1) = qr(sel,:,1)
      qi(c,:,1) = qi(sel,:,1)
      qs(c,:,1) = qs(sel,:,1)
      qg(c,:,1) = qg(sel,:,1)
      ni(c,:,1) = ni(sel,:,1)
      ns(c,:,1) = ns(sel,:,1)
      nr(c,:,1) = nr(sel,:,1)
      ng(c,:,1) = ng(sel,:,1)
      nc(c,:,1) = nc(sel,:,1)
      rho(c,:,1) = rho(sel,:,1)
      pii(c,:,1) = pii(sel,:,1)
      pres(c,:,1) = pres(sel,:,1)
      dz(c,:,1) = dz(sel,:,1)
      w(c,:,1) = w(sel,:,1)
      kzh(c,:,1) = kzh(sel,:,1)
      qrcuten(c,:,1) = qrcuten(sel,:,1)
      qscuten(c,:,1) = qscuten(sel,:,1)
      qicuten(c,:,1) = qicuten(sel,:,1)
      ht(c,1) = ht(sel,1)
      aerocu(c,:,1,:) = aerocu(sel,:,1,:)
      if (c <= 56) then
        aerocu(c,:,1,:) = aerocu(c,:,1,:) * 0.5
        w(c,:,1) = w(c,:,1) * 0.75
        qv(c,:,1) = qv(c,:,1) * 0.995
      else
        aerocu(c,:,1,:) = aerocu(c,:,1,:) * 2.0
        w(c,:,1) = w(c,:,1) * 1.25
        qv(c,:,1) = qv(c,:,1) * 1.005
      endif
      rainnc(c,1) = rainnc(sel,1)
      rainncv(c,1) = rainncv(sel,1)
      snownc(c,1) = snownc(sel,1)
      snowncv(c,1) = snowncv(sel,1)
      graupelnc(c,1) = graupelnc(sel,1)
      graupelncv(c,1) = graupelncv(sel,1)
      sr(c,1) = sr(sel,1)
    enddo
  end subroutine build_inputs

end program run_morrison_aero_oracle
