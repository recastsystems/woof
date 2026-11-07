program run_noah_lsm_oracle
  ! Drive the WRF v4.6.1 Noah LSM driver `lsm` (phys/module_sf_noahdrv.F:38)
  ! over a 1-D row of independent land columns and dump every driver-level
  ! input and every driver-level output as CSV.
  !
  ! WHY THE DRIVER AND NOT SFLX.  gpuwm's port is one fused device function --
  ! gpuwm/core/kernels/noah.cu `noah_column` -- whose scope is exactly
  ! `lsm`'s: the per-column input preparation (Exner ratios, the ice/water
  ! saturation blend, the mm->m conversions, the soil-type-14 reset), then
  ! SFLX, then the post-SFLX state and flux updates and the WRF driver
  ! diagnostics.  Oracling SFLX alone would leave gpuwm computing SFLX's
  ! inputs with its own transcription of the prep and then measuring itself
  ! against a reference fed those same numbers -- the mirror problem this
  ! project already lost months to on YSU.  Calling `lsm` means every number
  ! in the fixture is a driver input the port also takes, or a driver output
  ! the port also produces.
  !
  ! Nothing here invents an expected value.  Every column in the CSV is either
  ! a word this program wrote into an input array before the call, or a word
  ! `lsm` wrote into an output array during it.  The two exceptions are
  ! labelled: `sfcprs` and `zlvl` are echoes of the two derived quantities the
  ! driver forms from p8w3d and dz8w
  !     SFCPRS = (P8W3D(I,KTS+1,J) + P8W3D(I,KTS,J)) * 0.5   (:794)
  !     ZLVL   = 0.5 * DZ8W(I,1,J)                           (:129 of the loop)
  ! recomputed here in the same float32 arithmetic, because the port takes
  ! them as inputs where WRF derives them.  They are inputs, not answers.
  !
  ! The row is ims:ime = 1:ncase with jms=jme=1, so every case is its own
  ! column and no case can borrow a neighbour's scratch: Noah is column-local
  ! (module_sf_noahdrv.F's ILOOP carries no i-dependence).
  !
  ! Parameter tables come from WRF's own SOIL_VEG_GEN_PARM
  ! (module_sf_noahdrv.F:1999) reading the run/ directory's VEGPARM.TBL,
  ! SOILPARM.TBL and GENPARM.TBL, which build.sh copies from the pinned tree.
  ! That makes the fixture an independent check of gpuwm's own transcription
  ! of that READ sequence as well as of the physics.

  use module_sf_noahdrv, only: lsm_mosaic, soil_veg_gen_parm
  use module_sf_sfcdiags, only: sfcdiags
  implicit none

  integer, parameter :: ncase = 80
  integer, parameter :: nsoil = 4
  real, parameter :: dt = 60.0
  real, parameter :: r_d_over_cp = 287.0 / (7.0 * 287.0 / 2.0)   ! WRF ROVCP

  ! --- driver argument arrays, one column per case ---------------------------
  real, dimension(ncase, 2, 1) :: dz8w, qv3d, p8w3d, t3d
  real, dimension(ncase, 1) :: tsk, hfx, qfx, lh, grdflx, qgh, gsw, swdown
  real, dimension(ncase, 1) :: swddir, swddif, glw, smstav, smstot
  real, dimension(ncase, 1) :: sfcrunoff, udrunoff, vegfra, albedo, albbck
  real, dimension(ncase, 1) :: znt, z0, tmn, xland, xice, emiss, embck
  real, dimension(ncase, 1) :: snowc, qsfc, rainbl, snow, canwat
  real, dimension(ncase, 1) :: chs, chs2, cqs2, cpm, sr, chklowq, lai, qz0
  real, dimension(ncase, 1) :: snowh, snoalb, shdmin, shdmax, snotime
  real, dimension(ncase, 1) :: acsnom, acsnow, snopcx, potevp, rib, noahres
  real, dimension(ncase, 1) :: flx4_2d, fvb_2d, fbur_2d, fgsn_2d, ust_urb2d
  real, dimension(ncase, 1) :: frc_urb2d
  real, dimension(ncase, 1) :: sda_hfx, sda_qfx, hfx_both, qfx_both, qnorm
  real, dimension(ncase, nsoil, 1) :: smois, tslb, sh2o, smcrel
  integer, dimension(ncase, 1) :: ivgtyp, isltyp, utype_urb2d
  real, dimension(nsoil) :: dzs

  ! --- saved copies of everything the driver overwrites ----------------------
  real, dimension(ncase, 1) :: tsk0, hfx0, qfx0, lh0, grdflx0, qsfc0
  real, dimension(ncase, 1) :: canwat0, snow0, snowc0, snowh0, albedo0
  real, dimension(ncase, 1) :: albbck0, emiss0, znt0, z00, snotime0, lai0
  real, dimension(ncase, 1) :: smstav0, smstot0, sfcrunoff0, udrunoff0
  real, dimension(ncase, 1) :: acsnom0, acsnow0, snopcx0, potevp0, rib0
  real, dimension(ncase, 1) :: chs0, chs20, cqs20
  real, dimension(ncase, nsoil, 1) :: smois0, tslb0, sh2o0
  real, dimension(ncase, 1) :: sfcprs, zlvl


  integer, parameter :: ncat=50, ntile=2
  real, dimension(ncase,ncat,1) :: landusef,landusef2
  integer, dimension(ncase,ncat,1) :: cat_index
  real, dimension(ncase,1) :: shdavg, rc2,xlai2
  real, dimension(ncase,ntile,1) :: tsk_tile,qsfc_tile,canwat_tile,snow_tile,snowh_tile,snowc_tile,albedo_tile,albbck_tile,emiss_tile,embck_tile,znt_tile,z0_tile,hfx_tile,qfx_tile,lh_tile,grdflx_tile,snotime_tile,rc_tile,lai_tile
  real, dimension(ncase,nsoil*ntile,1) :: tslb_tile,smois_tile,sh2o_tile

  character(len=1024) :: out_path
  character(len=32) :: arg
  integer :: icase, ns, u, opt_thcnd, itimestep
  logical :: frpcpn, usemonalb, rdlai2d, ua_phys
  real, dimension(ncase,1) :: t2, th2_diag, q2
  real :: subn, nzero

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_lsm OUT.csv [opt_thcnd] [frpcpn] [usemonalb] [rdlai2d]'
    error stop 2
  end if
  opt_thcnd = 1
  frpcpn = .false.
  usemonalb = .false.
  rdlai2d = .false.
  itimestep = 2
  ua_phys = .false.
  call get_command_argument(6, arg)
  if (len_trim(arg) > 0) ua_phys = (trim(arg) == 'T')
  call get_command_argument(2, arg)
  if (len_trim(arg) > 0) read(arg, *) opt_thcnd
  call get_command_argument(3, arg)
  if (len_trim(arg) > 0) frpcpn = (trim(arg) == 'T')
  call get_command_argument(4, arg)
  if (len_trim(arg) > 0) usemonalb = (trim(arg) == 'T')
  call get_command_argument(5, arg)
  if (len_trim(arg) > 0) rdlai2d = (trim(arg) == 'T')

  ! The smallest positive float32 subnormal and a negative zero, by bit
  ! pattern rather than by a decimal literal the compiler is free to fold.
  subn = transfer(1, 1.0)
  nzero = sign(0.0, -1.0)

  call soil_veg_gen_parm('MODIFIED_IGBP_MODIS_NOAH', 'STAS')

  dzs = (/ 0.10, 0.30, 0.60, 1.00 /)

  call build_fixture()

  ! --- echoes of the two quantities the driver derives from its 3-D inputs ---
  do icase = 1, ncase
    sfcprs(icase, 1) = (p8w3d(icase, 2, 1) + p8w3d(icase, 1, 1)) * 0.5
    zlvl(icase, 1) = 0.5 * dz8w(icase, 1, 1)
  end do

  rc_tile=0.;lai_tile=0.
  landusef=0.;landusef2=0.;cat_index=0;shdavg=vegfra;rc2=0.;xlai2=0.
  do icase=1,ncase
    landusef(icase,ivgtyp(icase,1),1)=1.
    landusef2(icase,1,1)=0.7
    landusef2(icase,2,1)=0.3
    cat_index(icase,2,1)=1+mod(ivgtyp(icase,1),12)
    cat_index(icase,1,1)=ivgtyp(icase,1)
  enddo
  tsk_tile(:,1,1)=tsk(:,1)
  tsk_tile(:,2,1)=tsk(:,1)
  qsfc_tile(:,1,1)=qsfc(:,1)
  qsfc_tile(:,2,1)=qsfc(:,1)
  canwat_tile(:,1,1)=canwat(:,1)
  canwat_tile(:,2,1)=canwat(:,1)
  snow_tile(:,1,1)=snow(:,1)
  snow_tile(:,2,1)=snow(:,1)
  snowh_tile(:,1,1)=snowh(:,1)
  snowh_tile(:,2,1)=snowh(:,1)
  snowc_tile(:,1,1)=snowc(:,1)
  snowc_tile(:,2,1)=snowc(:,1)
  albedo_tile(:,1,1)=albedo(:,1)
  albedo_tile(:,2,1)=albedo(:,1)
  albbck_tile(:,1,1)=albbck(:,1)
  albbck_tile(:,2,1)=albbck(:,1)
  emiss_tile(:,1,1)=emiss(:,1)
  emiss_tile(:,2,1)=emiss(:,1)
  embck_tile(:,1,1)=embck(:,1)
  embck_tile(:,2,1)=embck(:,1)
  znt_tile(:,1,1)=znt(:,1)
  znt_tile(:,2,1)=znt(:,1)
  z0_tile(:,1,1)=z0(:,1)
  z0_tile(:,2,1)=z0(:,1)
  hfx_tile(:,1,1)=hfx(:,1)
  hfx_tile(:,2,1)=hfx(:,1)
  qfx_tile(:,1,1)=qfx(:,1)
  qfx_tile(:,2,1)=qfx(:,1)
  lh_tile(:,1,1)=lh(:,1)
  lh_tile(:,2,1)=lh(:,1)
  grdflx_tile(:,1,1)=grdflx(:,1)
  grdflx_tile(:,2,1)=grdflx(:,1)
  snotime_tile(:,1,1)=snotime(:,1)
  snotime_tile(:,2,1)=snotime(:,1)
  tsk_tile(:,2,1)=tsk(:,1)+1.
  snow_tile(:,2,1)=snow(:,1)*0.5
  snowh_tile(:,2,1)=snowh(:,1)*0.5
  tslb_tile(:,1:4,:)=tslb;tslb_tile(:,5:8,:)=tslb
  smois_tile(:,1:4,:)=smois;smois_tile(:,5:8,:)=smois
  sh2o_tile(:,1:4,:)=sh2o;sh2o_tile(:,5:8,:)=sh2o

  call save_state()

  call lsm_mosaic(dz8w, qv3d, p8w3d, t3d, tsk,                              &
           hfx, qfx, lh, grdflx, qgh, gsw, swdown,   &
           glw, smstav, smstot,                                      &
           sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
           albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
           snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
           nsoil, dt, dzs, itimestep,                                &
           smois, tslb, snow, canwat,                                &
           chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
           .false., frpcpn,                                          &
           sh2o, snowh,                                              &
           snoalb = snoalb, shdmin = shdmin, shdmax = shdmax, shdavg=shdavg, &
           u_phy=t3d, v_phy=t3d,        &
           snotime = snotime,                                        &
           acsnom = acsnom, acsnow = acsnow,                         &
           snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
           xice_threshold = 0.5,                                     &
           rdlai2d = rdlai2d, usemonalb = usemonalb,                 &
           rib = rib, noahres = noahres, opt_thcnd = opt_thcnd,      &
           nlcat=ncat,landusef=landusef,landusef2=landusef2, &
           sf_surface_mosaic=1,mosaic_cat=ntile,mosaic_cat_index=cat_index, &
           rc2=rc2,xlai2=xlai2,rc_mosaic=rc_tile,lai_mosaic=lai_tile, omg_urb2d=shdavg,cosz_urb2d=shdavg, &
           xlat_urb2d=shdavg,declin_urb=0., &
           tsk_mosaic=tsk_tile, &
           qsfc_mosaic=qsfc_tile, &
           canwat_mosaic=canwat_tile, &
           snow_mosaic=snow_tile, &
           snowh_mosaic=snowh_tile, &
           snowc_mosaic=snowc_tile, &
           albedo_mosaic=albedo_tile, &
           albbck_mosaic=albbck_tile, &
           emiss_mosaic=emiss_tile, &
           embck_mosaic=embck_tile, &
           znt_mosaic=znt_tile, &
           z0_mosaic=z0_tile, &
           hfx_mosaic=hfx_tile, &
           qfx_mosaic=qfx_tile, &
           lh_mosaic=lh_tile, &
           grdflx_mosaic=grdflx_tile, &
           snotime_mosaic=snotime_tile, &
           tslb_mosaic=tslb_tile, &
           smois_mosaic=smois_tile, &
           sh2o_mosaic=sh2o_tile, &
           ua_phys = ua_phys, flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
           fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
           ids = 1, ide = ncase, jds = 1, jde = 1, kds = 1, kde = 2, &
           ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = 2, &
           its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = 1, &
           sf_urban_physics = 0, ust_urb2d = ust_urb2d,              &
           num_roof_layers = 1, num_wall_layers = 1,                 &
           num_road_layers = 1, julian = 1, julyr = 1974,            &
           frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
           num_urban_ndm = 1, urban_map_zrd = 1, urban_map_zwd = 1,  &
           urban_map_gd = 1, urban_map_zd = 1, urban_map_zdf = 1,    &
           urban_map_bd = 1, urban_map_wd = 1, urban_map_gbd = 1,    &
           urban_map_fbd = 1, urban_map_zgrd = 1, num_urban_hi = 1,  &
           sda_hfx = sda_hfx, sda_qfx = sda_qfx,                     &
           hfx_both = hfx_both, qfx_both = qfx_both, qnorm = qnorm,  &
           use_wudapt_lcz=0, slucm_distributed_drag=.false., fasdas = 0)

  t2=0.; th2_diag=0.; q2=0.
  call sfcdiags(hfx,qfx,tsk,qsfc,chs2,cqs2,t2,th2_diag,q2,p8w3d(:,1,:), &
       7.*287./2.,287.,r_d_over_cp,chs,t3d,qv3d,ua_phys, &
       1,ncase,1,1,1,2,1,ncase,1,1,1,2,1,ncase,1,1,1,1)
  call write_csv()

contains

  subroutine build_fixture()
    integer :: i

    flx4_2d=0.; fvb_2d=0.; fbur_2d=0.; fgsn_2d=0.
    ! ---- the base column: warm, vegetated, unfrozen, snow-free land --------
    do i = 1, ncase
      ivgtyp(i, 1) = 10          ! MODIS grassland
      isltyp(i, 1) = 8           ! silty clay loam
      p8w3d(i, 1, 1) = 98000.0   ! surface pressure
      p8w3d(i, 2, 1) = 97000.0   ! -> SFCPRS = 97500 exactly
      t3d(i, 1, 1) = 290.0
      t3d(i, 2, 1) = 289.0
      qv3d(i, 1, 1) = 0.008
      qv3d(i, 2, 1) = 0.008
      dz8w(i, 1, 1) = 60.0
      dz8w(i, 2, 1) = 60.0
      qgh(i, 1) = 0.012
      gsw(i, 1) = 400.0
      swdown(i, 1) = 500.0
      swddir(i, 1) = 400.0
      swddif(i, 1) = 100.0
      glw(i, 1) = 330.0
      rainbl(i, 1) = 0.0
      sr(i, 1) = 0.0
      chs(i, 1) = 0.02
      chs2(i, 1) = 0.03
      cqs2(i, 1) = 0.03
      cpm(i, 1) = 1010.0
      qz0(i, 1) = 0.005
      rib(i, 1) = -0.1
      vegfra(i, 1) = 60.0
      shdmin(i, 1) = 10.0
      shdmax(i, 1) = 80.0
      tmn(i, 1) = 285.0
      xland(i, 1) = 1.0
      xice(i, 1) = 0.0
      snoalb(i, 1) = 0.7
      embck(i, 1) = 0.95
      tsk(i, 1) = 291.0
      hfx(i, 1) = 0.0
      qfx(i, 1) = 0.0
      lh(i, 1) = 0.0
      grdflx(i, 1) = 0.0
      qsfc(i, 1) = 0.010
      canwat(i, 1) = 0.3
      snow(i, 1) = 0.0
      snowc(i, 1) = 0.0
      snowh(i, 1) = 0.0
      albedo(i, 1) = 0.2
      albbck(i, 1) = 0.2
      emiss(i, 1) = 0.95
      znt(i, 1) = 0.1
      z0(i, 1) = 0.1
      snotime(i, 1) = 0.0
      lai(i, 1) = 3.0
      smstav(i, 1) = 0.0
      smstot(i, 1) = 0.0
      sfcrunoff(i, 1) = 0.0
      udrunoff(i, 1) = 0.0
      acsnom(i, 1) = 0.0
      acsnow(i, 1) = 0.0
      snopcx(i, 1) = 0.0
      potevp(i, 1) = 0.0
      frc_urb2d(i, 1) = 0.0
      utype_urb2d(i, 1) = 1
      sda_hfx(i, 1) = 0.0
      sda_qfx(i, 1) = 0.0
      hfx_both(i, 1) = 0.0
      qfx_both(i, 1) = 0.0
      qnorm(i, 1) = 0.0
      do ns = 1, nsoil
        smois(i, ns, 1) = 0.28
        sh2o(i, ns, 1) = 0.28
      end do
      tslb(i, 1, 1) = 289.0
      tslb(i, 2, 1) = 288.0
      tslb(i, 3, 1) = 287.0
      tslb(i, 4, 1) = 286.0
    end do

    ! ---- per-case overrides -------------------------------------------------
    ! 1  base warm land (NOPAC, no snow)                       -- unchanged

    ! 2  dry soil close to the wilting point
    smois(2, :, 1) = 0.06
    sh2o(2, :, 1) = 0.06
    vegfra(2, 1) = 25.0

    ! 3  saturated soil under heavy rain (runoff, SSTEP clamp)
    smois(3, :, 1) = 0.46
    sh2o(3, :, 1) = 0.46
    rainbl(3, 1) = 6.0

    ! 4  bare soil
    vegfra(4, 1) = 0.0
    shdmin(4, 1) = 0.0
    lai(4, 1) = 0.0
    canwat(4, 1) = 0.0

    ! 5  closed canopy holding its maximum interception
    vegfra(5, 1) = 100.0
    shdmin(5, 1) = 90.0
    shdmax(5, 1) = 100.0
    canwat(5, 1) = 0.5
    lai(5, 1) = 5.0
    ivgtyp(5, 1) = 1           ! evergreen needleleaf

    ! 6  cold snowpack, sub-freezing (SNOPAC, no melt)
    t3d(6, 1, 1) = 265.0
    tsk(6, 1) = 263.0
    snow(6, 1) = 20.0
    snowh(6, 1) = 0.08
    snowc(6, 1) = 0.9
    tslb(6, :, 1) = 268.0
    sh2o(6, :, 1) = 0.10
    glw(6, 1) = 250.0
    swdown(6, 1) = 200.0

    ! 7  melting snowpack
    t3d(7, 1, 1) = 274.0
    tsk(7, 1) = 273.5
    snow(7, 1) = 10.0
    snowh(7, 1) = 0.04
    snowc(7, 1) = 0.7
    tslb(7, :, 1) = 272.5
    sh2o(7, :, 1) = 0.20

    ! 8  deep snow, complete cover, strongly sub-freezing
    t3d(8, 1, 1) = 255.0
    tsk(8, 1) = 250.0
    snow(8, 1) = 200.0
    snowh(8, 1) = 0.60
    snowc(8, 1) = 1.0
    tslb(8, :, 1) = 260.0
    sh2o(8, :, 1) = 0.05
    snotime(8, 1) = 172800.0
    swdown(8, 1) = 120.0
    glw(8, 1) = 200.0

    ! 9  frozen soil profile: FRH2O's Newton iteration in every layer
    tslb(9, 1, 1) = 265.0
    tslb(9, 2, 1) = 268.0
    tslb(9, 3, 1) = 270.0
    tslb(9, 4, 1) = 272.0
    sh2o(9, :, 1) = 0.05
    t3d(9, 1, 1) = 270.0
    tsk(9, 1) = 268.0

    ! 10 SFCTMP exactly at the FFROZP threshold, with precipitation
    t3d(10, 1, 1) = 273.15
    rainbl(10, 1) = 1.0
    tsk(10, 1) = 272.0

    ! 11 SFCTMP one float32 step above the threshold, same precipitation
    t3d(11, 1, 1) = nearest(273.15, 1.0)
    rainbl(11, 1) = 1.0
    tsk(11, 1) = 272.0

    ! 12 T1 exactly 273.14: the ice/water Q2SAT blend boundary
    tsk(12, 1) = 273.14
    snow(12, 1) = 5.0
    snowh(12, 1) = 0.02
    snowc(12, 1) = 0.5
    t3d(12, 1, 1) = 272.0
    tslb(12, :, 1) = 272.0

    ! 13 T1 exactly 273.0 and SWDOWN exactly 10.0: the DQSDT2 damping test
    tsk(13, 1) = 273.0
    swdown(13, 1) = 10.0
    snow(13, 1) = 5.0
    snowh(13, 1) = 0.02
    snowc(13, 1) = 0.5
    t3d(13, 1, 1) = 272.0
    tslb(13, :, 1) = 272.0

    ! 14 one float32 step the other side of both
    tsk(14, 1) = nearest(273.0, 1.0)
    swdown(14, 1) = nearest(10.0, 1.0)
    snow(14, 1) = 5.0
    snowh(14, 1) = 0.02
    snowc(14, 1) = 0.5
    t3d(14, 1, 1) = 272.0
    tslb(14, :, 1) = 272.0

    ! 15 RAINBL = +0.0
    rainbl(15, 1) = 0.0

    ! 16 RAINBL = -0.0.  PRCP = RAINBL/DT keeps the sign; SNOWNG/FRZGRA test
    !    PRCP > 0., and -0.0 > 0. is false in both IEEE and CUDA -- but CuPy
    !    appends -ftz=true and this project has lost three days to exactly
    !    that class of probe.
    rainbl(16, 1) = nzero

    ! 17 RAINBL subnormal: PRCP = subn/60 underflows to +0 in float32
    rainbl(17, 1) = subn

    ! 18 SNOW subnormal: SNEQV = SNOW*0.001 underflows, and the
    !    (SNEQV /= 0 .and. SNOWHK == 0) guard then decides SNOWHK
    snow(18, 1) = subn
    snowc(18, 1) = 0.1

    ! 19 QV3D subnormal at the lowest level
    qv3d(19, 1, 1) = subn

    ! 20 QV3D exactly +0.0
    qv3d(20, 1, 1) = 0.0

    ! 21 QV3D exactly -0.0
    qv3d(21, 1, 1) = nzero

    ! 22 VEGFRA = +0.0  (SHDFAC = 0 exactly)
    vegfra(22, 1) = 0.0
    shdmin(22, 1) = 0.0

    ! 23 VEGFRA = -0.0
    vegfra(23, 1) = nzero
    shdmin(23, 1) = nzero

    ! 24 CANWAT subnormal -> CMC = subn/1000 underflows to +0
    canwat(24, 1) = subn

    ! 25 CHS subnormal: a vanishing exchange coefficient in PENMAN's RCH
    chs(25, 1) = subn
    chs2(25, 1) = subn
    cqs2(25, 1) = subn

    ! 26 open water: the driver's XLAND >= 1.5 skip
    xland(26, 1) = 2.0

    ! 27 sea ice: ICE = 1
    xice(27, 1) = 0.6
    t3d(27, 1, 1) = 260.0
    tsk(27, 1) = 258.0
    tslb(27, :, 1) = 262.0
    snow(27, 1) = 15.0
    snowh(27, 1) = 0.06
    snowc(27, 1) = 0.8

    ! 28 land ice: ICE = -1, SFLX_GLACIAL.  gpuwm SKIPS this column by
    !    documented restriction, so the row measures the size of that
    !    divergence rather than a port error.
    ivgtyp(28, 1) = 15
    t3d(28, 1, 1) = 258.0
    tsk(28, 1) = 255.0
    tslb(28, :, 1) = 258.0
    snow(28, 1) = 100.0
    snowh(28, 1) = 0.4
    snowc(28, 1) = 1.0

    ! 29 soil type 14 (water) at a land point with XICE = 0 -> reset to 7
    isltyp(29, 1) = 14

    ! 30 the urban category, with sf_urban_physics = 0 (parameter overrides
    !    inside SFLX/HRT only)
    ivgtyp(30, 1) = 13
    vegfra(30, 1) = 5.0

    ! 31 sand
    isltyp(31, 1) = 1
    smois(31, :, 1) = 0.15
    sh2o(31, :, 1) = 0.15

    ! 32 clay
    isltyp(32, 1) = 12
    smois(32, :, 1) = 0.40
    sh2o(32, :, 1) = 0.40

    ! 33 organic material
    isltyp(33, 1) = 13
    smois(33, :, 1) = 0.35
    sh2o(33, :, 1) = 0.35

    ! 34 bedrock: the smallest porosity in the table
    isltyp(34, 1) = 15
    smois(34, :, 1) = 0.08
    sh2o(34, :, 1) = 0.08

    ! 35 stable nocturnal: no shortwave, weak exchange, positive Richardson
    swdown(35, 1) = 0.0
    gsw(35, 1) = 0.0
    swddir(35, 1) = 0.0
    swddif(35, 1) = 0.0
    glw(35, 1) = 280.0
    t3d(35, 1, 1) = 280.0
    tsk(35, 1) = 278.0
    chs(35, 1) = 0.002
    chs2(35, 1) = 0.003
    cqs2(35, 1) = 0.003
    rib(35, 1) = 0.3

    ! 36 strong daytime evaporative demand
    qgh(36, 1) = 0.030
    qv3d(36, 1, 1) = 0.002
    swdown(36, 1) = 900.0
    gsw(36, 1) = 750.0
    t3d(36, 1, 1) = 305.0
    tsk(36, 1) = 312.0

    ! 37 mixed-phase precipitation: SR = 0.5 matters only when frpcpn is on,
    !    so this row discriminates the flag
    sr(37, 1) = 0.5
    rainbl(37, 1) = 2.0
    t3d(37, 1, 1) = 272.5
    tsk(37, 1) = 272.0
    snow(37, 1) = 3.0
    snowh(37, 1) = 0.012
    snowc(37, 1) = 0.4

    ! 38 aged snow: SNOTIME drives the albedo decay
    snotime(38, 1) = 86400.0
    snow(38, 1) = 8.0
    snowh(38, 1) = 0.03
    snowc(38, 1) = 0.6
    t3d(38, 1, 1) = 268.0
    tsk(38, 1) = 266.0
    tslb(38, :, 1) = 270.0

    ! 39 soil moisture above porosity on entry (the SSTEP/SRT clamps)
    smois(39, :, 1) = 0.60
    sh2o(39, :, 1) = 0.60

    ! 40 deep cold with a snowpack and frozen soil together
    t3d(40, 1, 1) = 250.0
    tsk(40, 1) = 248.0
    tslb(40, :, 1) = 250.0
    sh2o(40, :, 1) = 0.02
    snow(40, 1) = 50.0
    snowh(40, 1) = 0.2
    snowc(40, 1) = 1.0
    swdown(40, 1) = 80.0
    glw(40, 1) = 180.0

    ! 41, 42  the ONLY two soil types opt_thcnd can change.
    !    TDFCND (module_sf_noahlsm.F:4173) takes the McCumber-Pielke arm only
    !    when opt_thcnd == 2 AND SOILTYP is 3 or 4; for every other soil type
    !    the two options are the same function.  Without these two rows the
    !    opt_thcnd=2 fixture came out BYTE-IDENTICAL to the opt_thcnd=1 one,
    !    i.e. it could not have discriminated a port that ignored the switch
    !    entirely.  41 is unfrozen so TDFCND is reached through the ordinary
    !    HRT path; 42 carries a snowpack and frozen soil so it is reached
    !    through SNOPAC's as well.
    isltyp(41, 1) = 3          ! sandy loam
    smois(41, :, 1) = 0.22
    sh2o(41, :, 1) = 0.22

    isltyp(42, 1) = 4          ! silt loam
    smois(42, :, 1) = 0.30
    sh2o(42, :, 1) = 0.12
    tslb(42, :, 1) = 269.0
    t3d(42, 1, 1) = 266.0
    tsk(42, 1) = 264.0
    snow(42, 1) = 12.0
    snowh(42, 1) = 0.05
    snowc(42, 1) = 0.8
    swdown(42, 1) = 150.0
    glw(42, 1) = 230.0
    ! Additional UA columns span compressed short canopy, tall canopy,
    ! complete burial, shallow melting packs and cover limits.
    do i=43,ncase
      ivgtyp(i,1)=1+mod(i-43,12)
      snowh(i,1)=0.01*real(1+mod(i-43,10))
      if (i>=63) snowh(i,1)=0.5*real(1+mod(i-63,10))
      snow(i,1)=200.*snowh(i,1)
      snowc(i,1)=0.5
      t3d(i,1,1)=269.+real(mod(i-43,7))
      tsk(i,1)=267.+real(mod(i-43,7))
      tslb(i,:,1)=271.+real(mod(i-43,5))
      if(mod(i,3)==0) sh2o(i,:,1)=0.12
      vegfra(i,1)=real(mod(i*17,101))
    enddo
    snow(78,1)=0.00011; snowh(78,1)=0.000001
    snow(79,1)=200.; snowh(79,1)=1.0
    snow(80,1)=0.; snowh(80,1)=0.; vegfra(80,1)=0.
  end subroutine build_fixture

  subroutine save_state()
    tsk0 = tsk; hfx0 = hfx; qfx0 = qfx; lh0 = lh; grdflx0 = grdflx
    qsfc0 = qsfc; canwat0 = canwat; snow0 = snow; snowc0 = snowc
    snowh0 = snowh; albedo0 = albedo; albbck0 = albbck; emiss0 = emiss
    znt0 = znt; z00 = z0; snotime0 = snotime; lai0 = lai
    smstav0 = smstav; smstot0 = smstot
    sfcrunoff0 = sfcrunoff; udrunoff0 = udrunoff
    acsnom0 = acsnom; acsnow0 = acsnow; snopcx0 = snopcx; potevp0 = potevp
    rib0 = rib; chs0 = chs; chs20 = chs2; cqs20 = cqs2
    smois0 = smois; tslb0 = tslb; sh2o0 = sh2o
  end subroutine save_state

  subroutine write_csv()
    integer :: i
    open(newunit=u, file=trim(out_path), status='replace', action='write')
    write(u, '(A)') 'case,ivgtyp,isltyp,psfc,sfcprs,zlvl,sfctmp,qv1,qgh,' //   &
      'dz8w1,glw,swdown,rainbl,sr,chs_in,chs2_in,cqs2_in,cpm,qz0,rib_in,' //   &
      'vegfra,shdmin,shdmax,tmn,xland,xice,snoalb,embck,' //                   &
      'tsk_in,hfx_in,qfx_in,lh_in,grdflx_in,qsfc_in,canwat_in,snow_in,' //     &
      'snowc_in,snowh_in,albedo_in,albbck_in,emiss_in,znt_in,z0_in,' //        &
      'snotime_in,lai_in,smstav_in,smstot_in,sfcrunoff_in,udrunoff_in,' //     &
      'acsnom_in,acsnow_in,snopcx_in,potevp_in,' //                            &
      'smois1_in,smois2_in,smois3_in,smois4_in,' //                            &
      'tslb1_in,tslb2_in,tslb3_in,tslb4_in,' //                                &
      'sh2o1_in,sh2o2_in,sh2o3_in,sh2o4_in,' //                                &
      'tsk,hfx,qfx,lh,grdflx,qsfc,canwat,snow,snowc,snowh,albedo,albbck,' //   &
      'emiss,znt,z0,snotime,lai,smstav,smstot,sfcrunoff,udrunoff,' //          &
      'acsnom,acsnow,snopcx,potevp,noahres,chklowq,rib_out,' //                &
      'smois1,smois2,smois3,smois4,tslb1,tslb2,tslb3,tslb4,' //                &
      'sh2o1,sh2o2,sh2o3,sh2o4,smcrel1,smcrel2,smcrel3,smcrel4,' // &
      'flx4,fvb,fbur,fgsn,t2,th2,q2,tile2_category,tile2_tsk,tile2_snow,tile2_snowh,w1,w2'
    do i = 1, ncase
      write(u, '(I0,",",I0,",",I0)', advance='no') i, ivgtyp(i, 1), isltyp(i, 1)
      write(u, '(25(",",ES24.16E3))', advance='no')                            &
        p8w3d(i, 1, 1), sfcprs(i, 1), zlvl(i, 1), t3d(i, 1, 1),                &
        qv3d(i, 1, 1), qgh(i, 1), dz8w(i, 1, 1), glw(i, 1), swdown(i, 1),      &
        rainbl(i, 1), sr(i, 1), chs0(i, 1), chs20(i, 1), cqs20(i, 1),          &
        cpm(i, 1), qz0(i, 1), rib0(i, 1), vegfra(i, 1), shdmin(i, 1),          &
        shdmax(i, 1), tmn(i, 1), xland(i, 1), xice(i, 1), snoalb(i, 1),        &
        embck(i, 1)
      write(u, '(25(",",ES24.16E3))', advance='no')                            &
        tsk0(i, 1), hfx0(i, 1), qfx0(i, 1), lh0(i, 1), grdflx0(i, 1),          &
        qsfc0(i, 1), canwat0(i, 1), snow0(i, 1), snowc0(i, 1), snowh0(i, 1),   &
        albedo0(i, 1), albbck0(i, 1), emiss0(i, 1), znt0(i, 1), z00(i, 1),     &
        snotime0(i, 1), lai0(i, 1), smstav0(i, 1), smstot0(i, 1),              &
        sfcrunoff0(i, 1), udrunoff0(i, 1), acsnom0(i, 1), acsnow0(i, 1),       &
        snopcx0(i, 1), potevp0(i, 1)
      write(u, '(12(",",ES24.16E3))', advance='no')                            &
        smois0(i, 1, 1), smois0(i, 2, 1), smois0(i, 3, 1), smois0(i, 4, 1),    &
        tslb0(i, 1, 1), tslb0(i, 2, 1), tslb0(i, 3, 1), tslb0(i, 4, 1),        &
        sh2o0(i, 1, 1), sh2o0(i, 2, 1), sh2o0(i, 3, 1), sh2o0(i, 4, 1)
      write(u, '(28(",",ES24.16E3))', advance='no')                            &
        tsk(i, 1), hfx(i, 1), qfx(i, 1), lh(i, 1), grdflx(i, 1), qsfc(i, 1),   &
        canwat(i, 1), snow(i, 1), snowc(i, 1), snowh(i, 1), albedo(i, 1),      &
        albbck(i, 1), emiss(i, 1), znt(i, 1), z0(i, 1), snotime(i, 1),         &
        lai(i, 1), smstav(i, 1), smstot(i, 1), sfcrunoff(i, 1),                &
        udrunoff(i, 1), acsnom(i, 1), acsnow(i, 1), snopcx(i, 1),              &
        potevp(i, 1), noahres(i, 1), chklowq(i, 1), rib(i, 1)
      write(u, '(16(",",ES24.16E3))', advance='no')                                          &
        smois(i, 1, 1), smois(i, 2, 1), smois(i, 3, 1), smois(i, 4, 1),        &
        tslb(i, 1, 1), tslb(i, 2, 1), tslb(i, 3, 1), tslb(i, 4, 1),            &
        sh2o(i, 1, 1), sh2o(i, 2, 1), sh2o(i, 3, 1), sh2o(i, 4, 1),            &
        smcrel(i, 1, 1), smcrel(i, 2, 1), smcrel(i, 3, 1), smcrel(i, 4, 1)
      write(u, '(7(",",ES24.16E3))', advance='no') flx4_2d(i,1),fvb_2d(i,1), &
        fbur_2d(i,1),fgsn_2d(i,1),t2(i,1),th2_diag(i,1),q2(i,1)
      write(u, '(6(",",ES24.16E3))') real(1+mod(ivgtyp(i,1),12)), &
        tsk0(i,1)+1.,snow0(i,1)*0.5,snowh0(i,1)*0.5,0.7,0.3
    end do
    close(u)
  end subroutine write_csv

end program run_noah_lsm_oracle
