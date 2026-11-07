! WRF v4.6.1 NSSL option-18 variant column oracle.
!
! Runs the official nssl_2mom_init + nssl_2mom_driver (unchanged source) for
! one resolved variant mode on a slab of independent columns, for a given
! number of steps, and writes every state array, precipitation total,
! reflectivity and effective radius after the last step.
!
! Input (text):
!   line 1: nx nz nsteps itimestep0 dt
!   line 2: ipconc hail_on density_on ccn_on icdx icdxhl
!   line 3: alphah alphahl cccn cnoh cnohl cnor cnos rho_qh rho_qhl rho_qs
!   then nx*nz lines (column-major: ix outer, k inner):
!     theta qv qc qr qi qs qg qh qndrop qnr qni qns qng qnh qnn qvolg qvolh
!     qzr qzg qzh pressure exner rho dz w
!   (w is the WRF interface field w(k), k=1..nz, as the driver receives it)
!
! The module reads namelist.input for an optional nssl_mp_params group; the
! build directory holds an empty namelist.input, so every internal parameter
! keeps its compiled WRF v4.6.1 default.
program nssl2_variant_column
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_driver
  implicit none

  integer :: iu, ou, nx, nz, nsteps, it0, ix, k, n
  integer :: ipconc, hail_on, density_on, ccn_on, icdx, icdxhl
  integer :: ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: has_req
  real :: dt_s, nssl_params(20)
  real :: alphah, alphahl, cccn, cnoh, cnohl, cnor, cnos
  real :: rho_qh, rho_qhl, rho_qs
  real, allocatable, dimension(:,:,:) :: theta, qv, qc, qr, qi, qs, qg, qh
  real, allocatable, dimension(:,:,:) :: qndrop, qnr, qni, qns, qng, qnh, qnn
  real, allocatable, dimension(:,:,:) :: qvolg, qvolh, qzr, qzg, qzh
  real, allocatable, dimension(:,:,:) :: pressure, exner, rho, dz, w
  real, allocatable, dimension(:,:,:) :: dbz, re_cloud, re_ice, re_snow
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv
  real, allocatable, dimension(:,:) :: grplnc, grplncv, hailnc, hailncv, sr
  character(len=512) :: input_path, output_path

  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  if (len_trim(input_path) == 0 .or. len_trim(output_path) == 0) then
     error stop 'usage: nssl2_variant_column INPUT.txt OUTPUT.csv'
  endif

  open(newunit=iu, file=trim(input_path), status='old', action='read')
  read(iu,*) nx, nz, nsteps, it0, dt_s
  read(iu,*) ipconc, hail_on, density_on, ccn_on, icdx, icdxhl
  read(iu,*) alphah, alphahl, cccn, cnoh, cnohl, cnor, cnos, &
       rho_qh, rho_qhl, rho_qs
  if (nx < 1 .or. nz < 3 .or. nsteps < 1 .or. dt_s <= 0.0) &
       error stop 'invalid variant-column header'

  allocate(theta(nx,nz,1), qv(nx,nz,1), qc(nx,nz,1), qr(nx,nz,1))
  allocate(qi(nx,nz,1), qs(nx,nz,1), qg(nx,nz,1), qh(nx,nz,1))
  allocate(qndrop(nx,nz,1), qnr(nx,nz,1), qni(nx,nz,1), qns(nx,nz,1))
  allocate(qng(nx,nz,1), qnh(nx,nz,1), qnn(nx,nz,1))
  allocate(qvolg(nx,nz,1), qvolh(nx,nz,1))
  allocate(qzr(nx,nz,1), qzg(nx,nz,1), qzh(nx,nz,1))
  allocate(pressure(nx,nz,1), exner(nx,nz,1), rho(nx,nz,1))
  allocate(dz(nx,nz,1), w(nx,nz,1))
  allocate(dbz(nx,nz,1), re_cloud(nx,nz,1), re_ice(nx,nz,1), re_snow(nx,nz,1))
  allocate(rainnc(nx,1), rainncv(nx,1), snownc(nx,1), snowncv(nx,1))
  allocate(grplnc(nx,1), grplncv(nx,1), hailnc(nx,1), hailncv(nx,1), sr(nx,1))

  do ix = 1, nx
     do k = 1, nz
        read(iu,*) theta(ix,k,1), qv(ix,k,1), qc(ix,k,1), qr(ix,k,1), &
             qi(ix,k,1), qs(ix,k,1), qg(ix,k,1), qh(ix,k,1), &
             qndrop(ix,k,1), qnr(ix,k,1), qni(ix,k,1), qns(ix,k,1), &
             qng(ix,k,1), qnh(ix,k,1), qnn(ix,k,1), &
             qvolg(ix,k,1), qvolh(ix,k,1), &
             qzr(ix,k,1), qzg(ix,k,1), qzh(ix,k,1), &
             pressure(ix,k,1), exner(ix,k,1), rho(ix,k,1), &
             dz(ix,k,1), w(ix,k,1)
     enddo
  enddo
  close(iu)

  ! module_physics_init.F:4616-4651, with the variant flags as WRF passes
  ! them (LOGICALs from the resolved integer selectors).
  nssl_params = 0.0
  nssl_params(1) = cccn
  nssl_params(2) = alphah
  nssl_params(3) = alphahl
  nssl_params(4) = cnoh
  nssl_params(5) = cnohl
  nssl_params(6) = cnor
  nssl_params(7) = cnos
  nssl_params(8) = rho_qh
  nssl_params(9) = rho_qhl
  nssl_params(10) = rho_qs
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=ipconc, mixphase=0, &
       nssl_density_on=(density_on > 0), nssl_hail_on=(hail_on > 0),      &
       nssl_ccn_on=(ccn_on > 0), nssl_icdx=icdx, nssl_icdxhl=icdxhl)

  rainnc = 0.0; rainncv = 0.0; snownc = 0.0; snowncv = 0.0
  grplnc = 0.0; grplncv = 0.0; hailnc = 0.0; hailncv = 0.0; sr = 0.0
  dbz = -999.0
  re_cloud = -999.0; re_ice = -999.0; re_snow = -999.0
  ! module_physics_init.F:1007: radii only for nssl_2moment_on == 1.
  has_req = 0
  if (ipconc > 0) has_req = 1

  ids = 1; ide = nx; jds = 1; jde = 1; kds = 1; kde = nz
  ims = 1; ime = nx; jms = 1; jme = 1; kms = 1; kme = nz
  its = 1; ite = nx; jts = 1; jte = 1; kts = 1; kte = nz

  open(newunit=ou, file=trim(output_path), status='replace', &
       action='write', form='formatted')
  write(ou,'(A)') &
       'engine,step,ix,k,theta,qv,qc,qr,qi,qs,qg,qh,qndrop,qnr,qni,qns,' // &
       'qng,qnh,qnn,qvolg,qvolh,qzr,qzg,qzh,refl_10cm,effc_m,effi_m,' // &
       'effs_m,rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,' // &
       'hailnc,hailncv,sr'

  do n = 1, nsteps
     call nssl_2mom_driver( &
          qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qh=qg, qhl=qh, &
          ccw=qndrop, crw=qnr, cci=qni, csw=qns, chw=qng, chl=qnh, &
          vhw=qvolg, f_vhw=(density_on >= 1), &
          vhl=qvolh, f_vhl=(density_on == 2), &
          zrw=qzr, f_zrw=(ipconc >= 6), &
          zhw=qzg, f_zhw=(ipconc >= 6), &
          zhl=qzh, f_zhl=(ipconc >= 6 .and. hail_on == 1), &
          cn=qnn, f_cn=(ccn_on > 0), &
          th=theta, pii=exner, p=pressure, w=w, dn=rho, dz=dz, &
          dtp=dt_s, itimestep=it0 + n - 1, &
          rainnc=rainnc, rainncv=rainncv, &
          snownc=snownc, snowncv=snowncv, &
          grplnc=grplnc, grplncv=grplncv, &
          hailnc=hailnc, hailncv=hailncv, sr=sr, &
          dx=500.0, dy=500.0, dbz=dbz, diagflag=.true., &
          re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
          has_reqc=has_req, has_reqi=has_req, has_reqs=has_req, &
          nssl_progn=.false., &
          ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
          ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
          its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
  do ix = 1, nx
     do k = 1, nz
        ! Z0 hexadecimal float images keep every bit; the comparator reads
        ! them back exactly (no decimal round trip).
        write(ou,'(A,",",I0,",",I0,",",I0,33(",",Z8.8))') &
             'wrf', n, ix, k, theta(ix,k,1), qv(ix,k,1), qc(ix,k,1), &
             qr(ix,k,1), qi(ix,k,1), qs(ix,k,1), qg(ix,k,1), qh(ix,k,1), &
             qndrop(ix,k,1), qnr(ix,k,1), qni(ix,k,1), qns(ix,k,1), &
             qng(ix,k,1), qnh(ix,k,1), qnn(ix,k,1), &
             qvolg(ix,k,1), qvolh(ix,k,1), &
             qzr(ix,k,1), qzg(ix,k,1), qzh(ix,k,1), dbz(ix,k,1), &
             re_cloud(ix,k,1), re_ice(ix,k,1), re_snow(ix,k,1), &
             rainnc(ix,1), rainncv(ix,1), snownc(ix,1), snowncv(ix,1), &
             grplnc(ix,1), grplncv(ix,1), hailnc(ix,1), hailncv(ix,1), &
             sr(ix,1)
     enddo
  enddo
  enddo
  close(ou)

  print '(A,1X,A)', 'NSSL2_VARIANT_WRF_COMPLETE', trim(output_path)
end program nssl2_variant_column
