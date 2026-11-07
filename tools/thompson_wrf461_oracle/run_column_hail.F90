! Hail-aware Thompson (mp_physics=38) column oracle.
!
! A SEPARATE program from run_column_aero.F90 for the same reason that one
! is separate from run_column.F90: WRF's is_hail_aware is a module-SAVEd
! LOGICAL set purely by the presence of the optional graupel number at
! thompson_init (module_mp_thompson.F:459-466), and it also fixes the
! graupel density table to nine slabs (dimNRHG) and keeps the
! Heymsfield fall-speed pair av_g(5)/bv_g(5) that mp=8/28 overwrite with
! 442/0.89 (:463-464).  A program that ran both families would silently mix
! two schemes.
!
! The argument lists mirror WRF's own CASE(THOMPSONGH) exactly
! (phys/module_microphysics_driver.F:1132-1200): mp_gt_driver receives
! QB=qvolg and NG=qng on top of the aerosol-aware set mp=28 passes, and
! thompson_init receives ng (module_physics_init.F, the THOMPSONGH arm) on
! top of the aerosol arguments.  wif_input_opt = 0 and aer_init_opt = 0, as
! for the committed mp=28 family.
!
! Environment skeleton as run_column_aero.F90: nx=2, ny=2, nz=24,
! z(k)=(k-0.5)*500, p=p0*exp(-z/8000), WRF's own Exner function.  Every
! seeded number is a per-cubic-metre target divided by the local density
! estimate, because WRF's state arrays are per kilogram.  The graupel volume
! mixing ratio qb (m3 kg-1) is seeded as qg/rho_target, so rho_target is the
! bulk graupel density mp_thompson diagnoses at :1920-1923.
!
! The "before" snapshot is taken AFTER thompson_init (which may fill the
! aerosol profile), exactly as run_column_aero.F90 does.

program run_thompson_column_hail
  use module_mp_thompson, only: thompson_init, mp_gt_driver, RSLF, RSIF
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 24
  integer :: i, j, k, scenario_id, ios, free_unit
  logical :: unit_opened
  real, parameter :: p0 = 100000.0
  real, parameter :: r_d_wrf = 287.0, cp_wrf = 7.0 * r_d_wrf / 2.0
  real, parameter :: rd_over_cp = r_d_wrf / cp_wrf
  real, parameter :: r_dry = 287.04
  real, parameter :: re_qc_bg = 2.49e-6
  real, parameter :: re_qi_bg = 4.99e-6
  real, parameter :: re_qs_bg = 9.99e-6
  real :: z(nz), temperature(nz), dt
  real :: hgt(nx,nz,ny)
  real :: qv(nx,nz,ny), qc(nx,nz,ny), qr(nx,nz,ny)
  real :: qi(nx,nz,ny), qs(nx,nz,ny), qg(nx,nz,ny)
  real :: ni(nx,nz,ny), nr(nx,nz,ny), th(nx,nz,ny)
  real :: nc(nx,nz,ny), nwfa(nx,nz,ny), nifa(nx,nz,ny)
  real :: ng(nx,nz,ny), qb(nx,nz,ny)
  real :: nwfa2d(nx,ny), nifa2d(nx,ny)
  real :: nbca(nx,nz,ny), nbca2d(nx,ny)
  real :: pii(nx,nz,ny), p(nx,nz,ny), w(nx,nz,ny), dz(nx,nz,ny)
  real :: refl(nx,nz,ny), re_cloud(nx,nz,ny)
  real :: re_ice(nx,nz,ny), re_snow(nx,nz,ny)
  real :: rainnc(nx,ny), rainncv(nx,ny), snownc(nx,ny)
  real :: snowncv(nx,ny), graupelnc(nx,ny), graupelncv(nx,ny)
  real :: sr(nx,ny)
  real :: qv_before(nz), qc_before(nz), qr_before(nz)
  real :: qi_before(nz), qs_before(nz), qg_before(nz)
  real :: ni_before(nz), nr_before(nz), th_before(nz)
  real :: nc_before(nz), nwfa_before(nz), nifa_before(nz)
  real :: ng_before(nz), qb_before(nz)
  real :: temp_before(nz), temp_after(nz)
  real :: nwfa2d_before, nifa2d_before
  character(len=48) :: scenario
  character(len=512) :: output_dir, column_path, surface_path

  call get_command_argument(1, scenario)
  call get_command_argument(2, output_dir)
  if (len_trim(scenario) == 0 .or. len_trim(output_dir) == 0) then
     error stop 'usage: run_column_hail SCENARIO OUTPUT_DIRECTORY'
  endif

  ! Scenario identifiers start at 201: the classic space is 1..46 and the
  ! aerosol-aware space 101..122.
  select case (trim(scenario))
  case ('gh-convective')
     scenario_id = 201
  case ('gh-stratiform')
     scenario_id = 202
  case ('gh-hail')
     scenario_id = 203
  case ('gh-density-sweep')
     scenario_id = 204
  case ('gh-warm-melt')
     scenario_id = 205
  case default
     error stop 'unknown hail-aware Thompson column scenario'
  end select

  ! CCN_ACTIVATE.BIN is big-endian and is read on the lowest free unit in
  ! 20..99 (module_mp_thompson.F:5123-5133); build_hail.sh requests the
  ! conversion for unit 20 alone, exactly as build_aero.sh does.
  free_unit = -1
  do i = 20, 99
     inquire(unit=i, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = i
        exit
     endif
  enddo
  if (free_unit /= 20) then
     print '(A,I0)', 'FATAL: lowest free Fortran unit is not 20 but ', free_unit
     error stop 'CCN_ACTIVATE.BIN endian conversion assumption violated'
  endif

  select case (scenario_id)
  case (204)
     dt = 10.0
  case (205)
     dt = 20.0
  case default
     dt = 15.0
  end select

  do k = 1, nz
     z(k) = (real(k) - 0.5) * 500.0
     select case (scenario_id)
     case (201)
        temperature(k) = max(210.0, 300.0 - 0.0065 * z(k))
     case (202)
        temperature(k) = max(210.0, 285.0 - 0.0062 * z(k))
     case (203)
        temperature(k) = max(208.0, 303.0 - 0.0068 * z(k))
     case (204)
        ! -5 C everywhere: Hallett-Mossop splintering, riming with the
        ! Cober-List rime density, the frozen rain-graupel branch and the
        ! rimed-snow conversion all run on every level, so the only thing
        ! that varies down the column is the graupel density and size.
        temperature(k) = 268.15
     case (205)
        ! A ladder from just above 0 C to 12 C: melting with the
        ! density-scaled volume loss at :2823 everywhere, and on the dry
        ! upper half the melting rate goes negative near 0 C so graupel
        ! sublimates instead (:2826-2833).
        temperature(k) = 273.4 + 0.5 * real(nz - k)
     end select
  enddo

  qv = 0.0
  qc = 0.0
  qr = 0.0
  qi = 0.0
  qs = 0.0
  qg = 0.0
  ni = 0.0
  nr = 0.0
  nc = 0.0
  ng = 0.0
  qb = 0.0
  nwfa = 0.0
  nifa = 0.0
  nwfa2d = 0.0
  nifa2d = 0.0
  nbca = 0.0
  nbca2d = 0.0
  th = 0.0
  pii = 0.0
  p = 0.0
  w = 0.0
  dz = 500.0
  refl = -35.0
  re_cloud = re_qc_bg
  re_ice = re_qi_bg
  re_snow = re_qs_bg
  rainnc = 0.0
  rainncv = 0.0
  snownc = 0.0
  snowncv = 0.0
  graupelnc = 0.0
  graupelncv = 0.0
  sr = 0.0

  do j = 1, ny
     do i = 1, nx
        do k = 1, nz
           hgt(i,k,j) = z(k)
           p(i,k,j) = p0 * exp(-z(k) / 8000.0)
           pii(i,k,j) = (p(i,k,j) / p0) ** rd_over_cp
           th(i,k,j) = temperature(k) / pii(i,k,j)
           select case (scenario_id)
           case (201)
              ! Water saturated through the updraft core, ice-subsaturated
              ! anvil above it.
              if (z(k) <= 9250.0) then
                 qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = 0.9 * RSIF(p(i,k,j), temperature(k))
              endif
           case (202)
              if (z(k) <= 6250.0) then
                 qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = 0.85 * RSIF(p(i,k,j), temperature(k))
              endif
           case (203)
              if (z(k) <= 1250.0) then
                 ! Sub-cloud evaporative layer under the hail shaft.
                 qv(i,k,j) = 0.7 * RSLF(p(i,k,j), temperature(k))
              elseif (z(k) <= 10250.0) then
                 qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
              endif
           case (204)
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           case (205)
              if (k > nz / 2) then
                 qv(i,k,j) = 0.6 * RSLF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
              endif
           end select
        enddo
     enddo
  enddo

  call initialize_case(scenario_id, z, p, temperature, qv, qc, qr, qi,  &
                       qs, qg, ni, nr, nc, ng, qb, nwfa, nifa, w)

  ! thompson_init: aerosol-aware because nwfa2d/nwfa/nifa are PRESENT,
  ! hail-aware because ng is PRESENT (:459-466).  The hail-aware build of
  ! the rain-graupel tables is the nine-slab qr_acr_qg_mp38V1.dat, which
  ! build_hail.sh generates once and copies into the working directory.
  call thompson_init(                                                 &
       hgt=hgt,                                                       &
       nwfa2d=nwfa2d, nbca2d=nbca2d, ng=ng,                           &
       nwfa=nwfa, nifa=nifa, nbca=nbca,                               &
       wif_input_opt=0,                                               &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  qv_before = qv(1,:,1)
  qc_before = qc(1,:,1)
  qr_before = qr(1,:,1)
  qi_before = qi(1,:,1)
  qs_before = qs(1,:,1)
  qg_before = qg(1,:,1)
  ni_before = ni(1,:,1)
  nr_before = nr(1,:,1)
  nc_before = nc(1,:,1)
  ng_before = ng(1,:,1)
  qb_before = qb(1,:,1)
  nwfa_before = nwfa(1,:,1)
  nifa_before = nifa(1,:,1)
  th_before = th(1,:,1)
  nwfa2d_before = nwfa2d(1,1)
  nifa2d_before = nifa2d(1,1)
  do k = 1, nz
     temp_before(k) = th_before(k) * pii(1,k,1)
  enddo

  ! module_microphysics_driver.F CASE(THOMPSONGH), argument for argument.
  call mp_gt_driver(                                                  &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, qb=qb,              &
       ni=ni, nr=nr, nc=nc, ng=ng,                                   &
       nwfa=nwfa, nifa=nifa, nbca=nbca,                              &
       nwfa2d=nwfa2d, nifa2d=nifa2d, nbca2d=nbca2d,                 &
       aer_init_opt=0, wif_input_opt=0,                              &
       th=th, pii=pii, p=p, w=w, dz=dz,                             &
       dt_in=dt, itimestep=1,                                        &
       RAINNC=rainnc, RAINNCV=rainncv,                              &
       SNOWNC=snownc, SNOWNCV=snowncv,                              &
       GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr,           &
       refl_10cm=refl, diagflag=.true., ke_diag=nz, do_radar_ref=1, &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,           &
       has_reqc=1, has_reqi=1, has_reqs=1,                          &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  ! As in run_column_aero.F90: the exit temperature a pristine caller can
  ! see is th*pii, one float32 round trip away from WRF's local t1d at a
  ! few percent of rows.  The GPU gate compares theta for that reason.
  do k = 1, nz
     temp_after(k) = th(1,k,1) * pii(1,k,1)
  enddo

  column_path = trim(output_dir) // '/' // trim(scenario) // '-column.csv'
  surface_path = trim(output_dir) // '/' // trim(scenario) // '-surface.csv'
  open(newunit=ios, file=trim(column_path), status='replace', action='write')
  write(ios,'(A)') 'phase,k,z_m,p_pa,pii,w_m_s,dz_m,theta_k,temp_k,qv,qc,qr,qi,qs,qg,ni_per_kg,nr_per_kg,nc_per_kg,nwfa_per_kg,nifa_per_kg,ng_per_kg,qb_m3_per_kg,effc_m,effi_m,effs_m,refl_dbz'
  do k = 1, nz
     call write_row(ios, 'before', k, z(k), p(1,k,1), pii(1,k,1),      &
          w(1,k,1), dz(1,k,1), th_before(k), temp_before(k),          &
          qv_before(k), qc_before(k), qr_before(k), qi_before(k),     &
          qs_before(k), qg_before(k), ni_before(k), nr_before(k),     &
          nc_before(k), nwfa_before(k), nifa_before(k), ng_before(k), &
          qb_before(k), re_qc_bg, re_qi_bg, re_qs_bg, -35.0)
  enddo
  do k = 1, nz
     call write_row(ios, 'after', k, z(k), p(1,k,1), pii(1,k,1),       &
          w(1,k,1), dz(1,k,1), th(1,k,1), temp_after(k),             &
          qv(1,k,1), qc(1,k,1), qr(1,k,1), qi(1,k,1), qs(1,k,1),     &
          qg(1,k,1), ni(1,k,1), nr(1,k,1), nc(1,k,1), nwfa(1,k,1),   &
          nifa(1,k,1), ng(1,k,1), qb(1,k,1),                          &
          re_cloud(1,k,1), re_ice(1,k,1), re_snow(1,k,1), refl(1,k,1))
  enddo
  close(ios)

  open(newunit=ios, file=trim(surface_path), status='replace', action='write')
  write(ios,'(A)') 'scenario,dt_s,rainnc_mm,rainncv_mm,snownc_mm,snowncv_mm,graupelnc_mm,graupelncv_mm,sr,nwfa2d_kg_s,nifa2d_kg_s'
  write(ios,'(A,10(",",ES24.16E3))') trim(scenario), dt, rainnc(1,1), &
       rainncv(1,1), snownc(1,1), snowncv(1,1), graupelnc(1,1),       &
       graupelncv(1,1), sr(1,1), nwfa2d_before, nifa2d_before
  close(ios)

  print '(A,1X,A)', 'THOMPSON_HAIL_COLUMN_ORACLE_COMPLETE', trim(scenario)

contains

  real function per_kg(target_per_m3, pressure, temp, vapor)
    real, intent(in) :: target_per_m3, pressure, temp, vapor
    real :: rho_local
    rho_local = 0.622 * pressure / (r_dry * temp * (vapor + 0.622))
    per_kg = target_per_m3 / rho_local
  end function per_kg

  ! Graupel number per kg for an exponential distribution of mixing ratio
  ! mass_mr, median volume diameter mvd and bulk density rho_bulk:
  ! lamg = 3.672/mvd (:1930 with mu_g = 0) and ng = qg*lamg**3/(pi*rho_g)
  ! (:1929 solved for ng with cgg(3,1) = 6, ogg2 = 1).
  real function graupel_number(mass_mr, mvd, rho_bulk)
    real, intent(in) :: mass_mr, mvd, rho_bulk
    real :: lam
    lam = 3.672 / mvd
    graupel_number = mass_mr * lam * lam * lam / (3.1415926536 * rho_bulk)
  end function graupel_number

  subroutine initialize_case(which, height, pres3, temp1, qv3, qc3, qr3, &
                             qi3, qs3, qg3, ni3, nr3, nc3, ng3, qb3,     &
                             nwfa3, nifa3, w3)
    integer, intent(in) :: which
    real, intent(in) :: height(nz), temp1(nz)
    real, intent(in) :: pres3(nx,nz,ny), qv3(nx,nz,ny)
    real, intent(inout) :: qc3(nx,nz,ny), qr3(nx,nz,ny), qi3(nx,nz,ny)
    real, intent(inout) :: qs3(nx,nz,ny), qg3(nx,nz,ny)
    real, intent(inout) :: ni3(nx,nz,ny), nr3(nx,nz,ny)
    real, intent(inout) :: nc3(nx,nz,ny), ng3(nx,nz,ny), qb3(nx,nz,ny)
    real, intent(inout) :: nwfa3(nx,nz,ny), nifa3(nx,nz,ny)
    real, intent(inout) :: w3(nx,nz,ny)
    real :: cloud, rain, ice, snow, graupel, velocity
    real :: ccn_m3, in_m3, ncloud_m3, rho_bulk, mvd_g, rain_mvd
    integer :: ii, jj, kk, slot
    logical :: no_number, no_volume
    ! Bulk densities either side of every NINT(rho/100)+1 bin edge of
    ! :1923, both clamps (rho_g(1) = 50 and rho_g(9) = 800) and beyond them.
    real, parameter :: rho_ladder(24) = (/30.0, 50.0, 60.0, 75.0, 99.0,  &
         100.0, 149.0, 150.0, 151.0, 249.0, 250.0, 251.0, 350.0, 449.0,  &
         450.0, 549.0, 550.0, 649.0, 650.0, 749.0, 750.0, 800.0, 850.0, &
         1000.0/)
    ! Median volume diameters below D0r, inside, and above the 25.4 mm
    ! clamp of :1931-1939.
    real, parameter :: mvd_ladder(5) = (/0.03e-3, 0.5e-3, 2.0e-3,        &
                                         8.0e-3, 30.0e-3/)
    real, parameter :: warm_rho(8) = (/100.0, 200.0, 300.0, 400.0,       &
                                       500.0, 600.0, 700.0, 800.0/)

    do jj = 1, ny
       do ii = 1, nx
          do kk = 1, nz
             cloud = 0.0
             rain = 0.0
             ice = 0.0
             snow = 0.0
             graupel = 0.0
             velocity = 0.0
             ccn_m3 = 3.0e8
             in_m3 = 1.0e6
             ncloud_m3 = 0.0
             rho_bulk = 400.0
             mvd_g = 2.0e-3
             rain_mvd = 1.0e-3
             no_number = .false.
             no_volume = .false.

             select case (which)
             case (201)
                ! CONVECTIVE.  An 8 m/s updraft core, liquid cloud from
                ! cloud base through the mixed-phase layer, supercooled
                ! rain above the 4.1 km freezing level, moderate graupel
                ! whose density rises with height (rimed in the core), snow
                ! and ice in the anvil.
                velocity = 8.0 * exp(-((height(kk) - 5000.0) / 3000.0)**2)
                if (height(kk) >= 1000.0 .and. height(kk) <= 9250.0) then
                   cloud = 1.5e-3 * exp(-((height(kk) - 4000.0) / 2500.0)**2)
                endif
                if (height(kk) <= 7250.0) then
                   rain = 2.0e-3 * exp(-((height(kk) - 2000.0) / 2000.0)**2)
                endif
                if (height(kk) >= 1750.0 .and. height(kk) <= 11250.0) then
                   graupel = 3.0e-3 * exp(-((height(kk) - 6000.0) / 2000.0)**2)
                endif
                if (height(kk) >= 6000.0) then
                   snow = 1.0e-3 * exp(-((height(kk) - 8500.0) / 1800.0)**2)
                endif
                if (height(kk) >= 8000.0) then
                   ice = 1.0e-4 * exp(-((height(kk) - 9500.0) / 1500.0)**2)
                endif
                rho_bulk = 250.0 + 40.0 * real(mod(kk, 8))
                mvd_g = 3.0e-3
                ncloud_m3 = 300.0e6
             case (202)
                ! STRATIFORM.  A deep snow layer melting through the 1.8 km
                ! freezing level, light low-density graupel embedded in it,
                ! thin liquid cloud at the melting layer, weak ascent.
                velocity = 0.3
                if (height(kk) >= 1500.0 .and. height(kk) <= 8250.0) then
                   snow = 1.5e-3 * exp(-((height(kk) - 3500.0) / 1500.0)**2)
                   graupel = 3.0e-4 * exp(-((height(kk) - 3000.0) / 1500.0)**2)
                endif
                if (height(kk) <= 2750.0) then
                   rain = 4.0e-4 * exp(-((height(kk) - 1000.0) / 1200.0)**2)
                endif
                if (height(kk) >= 1750.0 .and. height(kk) <= 4250.0) then
                   cloud = 1.0e-4
                endif
                if (height(kk) >= 6000.0) then
                   ice = 3.0e-5 * exp(-((height(kk) - 7500.0) / 1500.0)**2)
                endif
                rho_bulk = 120.0 + 15.0 * real(mod(kk, 5))
                mvd_g = 1.0e-3
                ncloud_m3 = 100.0e6
             case (203)
                ! HAIL.  A 20 m/s updraft, heavy supercooled liquid water,
                ! large dense graupel-hail (bulk density 700-800, MVD 8 mm)
                ! aloft falling through the 4.6 km freezing level into rain
                ! and a dry sub-cloud layer.
                velocity = 20.0 * exp(-((height(kk) - 6000.0) / 3000.0)**2)
                if (height(kk) >= 1500.0 .and. height(kk) <= 10250.0) then
                   cloud = 3.0e-3 * exp(-((height(kk) - 6000.0) / 2500.0)**2)
                endif
                if (height(kk) <= 8250.0) then
                   rain = 3.0e-3 * exp(-((height(kk) - 2500.0) / 2200.0)**2)
                endif
                if (height(kk) >= 1000.0 .and. height(kk) <= 12000.0) then
                   graupel = 8.0e-3 * exp(-((height(kk) - 6000.0) / 2500.0)**2)
                endif
                if (height(kk) >= 7000.0) then
                   snow = 5.0e-4 * exp(-((height(kk) - 9000.0) / 1500.0)**2)
                   ice = 1.0e-4 * exp(-((height(kk) - 10000.0) / 1200.0)**2)
                endif
                rho_bulk = 700.0 + 15.0 * real(mod(kk, 7))
                mvd_g = 8.0e-3
                rain_mvd = 2.0e-3
                ncloud_m3 = 500.0e6
             case (204)
                ! DENSITY SWEEP.  The same graupel mass on every level,
                ! with the bulk density walking every bin edge and both
                ! clamps, the size walking both MVD clamps, no number on
                ! two levels (the :1924-1928 1.5 mm seed) and no volume on
                ! one (qb = 0, which :1920 lifts to rho_g(9)).
                cloud = 5.0e-4
                rain = 2.0e-4
                snow = 3.0e-4
                graupel = 1.0e-3
                rho_bulk = rho_ladder(kk)
                mvd_g = mvd_ladder(mod(kk - 1, 5) + 1)
                no_number = (kk == 3 .or. kk == 17)
                no_volume = (kk == 5)
                ncloud_m3 = 200.0e6
             case (205)
                ! WARM MELT.  Graupel of every density above 0 C with rain
                ! collecting it (the twet > T_0 branch of :2537-2545).
                rain = 5.0e-4
                graupel = 1.5e-3
                rho_bulk = warm_rho(mod(kk - 1, 8) + 1)
                mvd_g = 3.0e-3
                ncloud_m3 = 0.0
             case default
                error stop 'invalid hail-aware Thompson scenario id'
             end select

             qc3(ii,kk,jj) = cloud
             qr3(ii,kk,jj) = rain
             qi3(ii,kk,jj) = ice
             qs3(ii,kk,jj) = snow
             qg3(ii,kk,jj) = graupel

             if (ice > 1.0e-12) then
                ni3(ii,kk,jj) = ice * (4.0/200.0e-6)**3                 &
                     / (3.1415926536*890.0)
             else
                ni3(ii,kk,jj) = 0.0
             endif
             if (rain > 1.0e-12) then
                nr3(ii,kk,jj) = rain * (3.672/rain_mvd)**3              &
                     / (3.1415926536*1000.0)
             else
                nr3(ii,kk,jj) = 0.0
             endif
             if (graupel > 1.0e-12) then
                if (no_number) then
                   ng3(ii,kk,jj) = 0.0
                else
                   ng3(ii,kk,jj) = graupel_number(graupel, mvd_g, rho_bulk)
                endif
                if (no_volume) then
                   qb3(ii,kk,jj) = 0.0
                else
                   qb3(ii,kk,jj) = graupel / rho_bulk
                endif
             else
                ng3(ii,kk,jj) = 0.0
                qb3(ii,kk,jj) = 0.0
             endif
             if (ncloud_m3 > 0.0) then
                nc3(ii,kk,jj) = per_kg(ncloud_m3, pres3(ii,kk,jj),      &
                                       temp1(kk), qv3(ii,kk,jj))
             else
                nc3(ii,kk,jj) = 0.0
             endif
             nwfa3(ii,kk,jj) = per_kg(ccn_m3, pres3(ii,kk,jj),          &
                                      temp1(kk), qv3(ii,kk,jj))
             nifa3(ii,kk,jj) = per_kg(in_m3, pres3(ii,kk,jj),           &
                                      temp1(kk), qv3(ii,kk,jj))
             w3(ii,kk,jj) = velocity
          enddo
       enddo
    enddo
  end subroutine initialize_case

  subroutine write_row(unit, phase, level, height, pressure, exner,     &
                       velocity, layer_dz, theta, temp, vapor, cloud,  &
                       rain, ice, snow, graupel, ice_number,           &
                       rain_number, cloud_number, ccn_number,          &
                       in_number, graupel_number_kg, graupel_volume,   &
                       reqc, reqi, reqs, dbz)
    integer, intent(in) :: unit, level
    character(len=*), intent(in) :: phase
    real, intent(in) :: height, pressure, exner, velocity, layer_dz
    real, intent(in) :: theta, temp
    real, intent(in) :: vapor, cloud, rain, ice, snow, graupel
    real, intent(in) :: ice_number, rain_number, cloud_number
    real, intent(in) :: ccn_number, in_number
    real, intent(in) :: graupel_number_kg, graupel_volume
    real, intent(in) :: reqc, reqi, reqs, dbz

    write(unit,'(A,",",I0,24(",",ES24.16E3))') trim(phase), level,    &
         height, pressure, exner, velocity, layer_dz, theta, temp,     &
         vapor, cloud, rain, ice, snow, graupel, ice_number,           &
         rain_number, cloud_number, ccn_number, in_number,             &
         graupel_number_kg, graupel_volume, reqc, reqi, reqs, dbz
  end subroutine write_row

end program run_thompson_column_hail
