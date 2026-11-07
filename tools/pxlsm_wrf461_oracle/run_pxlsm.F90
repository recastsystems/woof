program run_pxlsm_oracle
  ! Drive WRF v4.6.1's Pleim-Xiu LSM driver `pxlsm` (phys/module_sf_pxlsm.F:27)
  ! over one case file written by gen_cases.py and dump the driver's state
  ! after every call.
  !
  ! THE DRIVER, NOT SURFPX.  gpuwm's port is one device function per column
  ! whose scope is exactly pxlsm's column body: VEGELAND's land-use weighting,
  ! SOILPROP, the nudging-analysis interpolation, PXSNOW, the sea-ice
  ! conversion, the seasonal LAI/vegetation fraction, the PX sub-step loop
  ! of SURFPX (with QFLUX and SMASS inside it) and the closing TSK/CANWAT/
  ! QSFC updates.  Calling pxlsm means every word in the fixture is a driver
  ! input the port also takes or a driver output the port also produces.
  !
  ! The tile is ids:ide = 1:NX+1, jds:jde = 1:NY+1 (WRF's staggered domain
  ! ends), memory = tile = 1:NX x 1:NY, so the edge-ring rule that switches
  ! soil nudging off at I<=2, I>=IDE-1, J<=2, J>=JDE-1 (:601-602) acts on
  ! the outer two rows and columns exactly as it does in a real domain.
  !
  ! Between calls the arrays are left exactly as pxlsm left them -- the
  ! surface driver hands the same grid arrays back next step -- and
  ! ITIMESTEP and CURR_SECS advance as the solver's do.  Nothing else
  ! changes, so the trajectory is pxlsm's own.
  !
  ! Usage: run_pxlsm CASE.bin OUT.bin

  use module_sf_pxlsm, only: pxlsm
  implicit none

  integer :: hi(16), nx, ny, nlcat, nscat, nsteps, smois_init, soil_nudge
  integer :: modis_veg, anal_interval, itimestep0, itimestep, step, u, v
  real :: hf(8), dt, dtbl, curr_secs
  real :: zs(2), dzs(2)
  character(len=512) :: case_path, out_path
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte

  ! IN_2D
  real, allocatable, dimension(:,:) :: psfc, gsw, glw, rainbl, emiss, albbck
  real, allocatable, dimension(:,:) :: shdmin, shdmax, pblh, rmol, snowncv
  real, allocatable, dimension(:,:) :: ust, mavail, sst
  real, allocatable, dimension(:,:) :: t2_ndg_old, t2_ndg_new, q2_ndg_old
  real, allocatable, dimension(:,:) :: q2_ndg_new, sn_ndg_old, sn_ndg_new
  ! INOUT_2D
  real, allocatable, dimension(:,:) :: ra, rs, lai, znt, qsfc, capg, canwat
  real, allocatable, dimension(:,:) :: qfx, hfx, lh, psih, vegfra, vegf_px
  real, allocatable, dimension(:,:) :: snow, snoalb, snowh, snowc, albedo
  real, allocatable, dimension(:,:) :: xland, xice, imperv, canfra
  ! OUT_2D
  real, allocatable, dimension(:,:) :: grdflx, tsk, ta2, qa2, t2obs, q2obs
  real, allocatable, dimension(:,:) :: lai_px, wwlt_px, wfc_px, wsat_px
  real, allocatable, dimension(:,:) :: clay_px, csand_px, fmsand_px
  integer, allocatable, dimension(:,:) :: isltyp
  ! ATM_3D (ims:ime, kms:kme, jms:jme)
  real, allocatable, dimension(:,:,:) :: u3d, v3d, dz8w, qv3d, t3d, th3d, rho
  ! SOIL_3D
  real, allocatable, dimension(:,:,:) :: smois, tslb
  real, allocatable, dimension(:,:,:) :: landusef, soilctop, soilcbot

  call get_command_argument(1, case_path)
  call get_command_argument(2, out_path)

  open(newunit=u, file=trim(case_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) hi
  read(u) hf
  nx = hi(1); ny = hi(2); nlcat = hi(3); nscat = hi(4); nsteps = hi(5)
  smois_init = hi(6); soil_nudge = hi(7); modis_veg = hi(8)
  anal_interval = hi(9); itimestep0 = hi(10)
  dt = hf(1); dtbl = hf(2); curr_secs = hf(3)
  dzs = (/ hf(4), hf(5) /)
  zs = (/ hf(6), hf(7) /)

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = 3
  ims = 1; ime = nx; jms = 1; jme = ny; kms = 1; kme = 2
  its = 1; ite = nx; jts = 1; jte = ny; kts = 1; kte = 1

  allocate(psfc(nx,ny), gsw(nx,ny), glw(nx,ny), rainbl(nx,ny), emiss(nx,ny))
  allocate(albbck(nx,ny), shdmin(nx,ny), shdmax(nx,ny), pblh(nx,ny))
  allocate(rmol(nx,ny), snowncv(nx,ny), ust(nx,ny), mavail(nx,ny), sst(nx,ny))
  allocate(t2_ndg_old(nx,ny), t2_ndg_new(nx,ny), q2_ndg_old(nx,ny))
  allocate(q2_ndg_new(nx,ny), sn_ndg_old(nx,ny), sn_ndg_new(nx,ny))
  allocate(ra(nx,ny), rs(nx,ny), lai(nx,ny), znt(nx,ny), qsfc(nx,ny))
  allocate(capg(nx,ny), canwat(nx,ny), qfx(nx,ny), hfx(nx,ny), lh(nx,ny))
  allocate(psih(nx,ny), vegfra(nx,ny), vegf_px(nx,ny), snow(nx,ny))
  allocate(snoalb(nx,ny), snowh(nx,ny), snowc(nx,ny), albedo(nx,ny))
  allocate(xland(nx,ny), xice(nx,ny), imperv(nx,ny), canfra(nx,ny))
  allocate(grdflx(nx,ny), tsk(nx,ny), ta2(nx,ny), qa2(nx,ny), t2obs(nx,ny))
  allocate(q2obs(nx,ny), lai_px(nx,ny), wwlt_px(nx,ny), wfc_px(nx,ny))
  allocate(wsat_px(nx,ny), clay_px(nx,ny), csand_px(nx,ny), fmsand_px(nx,ny))
  allocate(isltyp(nx,ny))
  allocate(u3d(nx,2,ny), v3d(nx,2,ny), dz8w(nx,2,ny), qv3d(nx,2,ny))
  allocate(t3d(nx,2,ny), th3d(nx,2,ny), rho(nx,2,ny))
  allocate(smois(nx,2,ny), tslb(nx,2,ny))
  allocate(landusef(nx,nlcat,ny), soilctop(nx,nscat,ny), soilcbot(nx,nscat,ny))

  ! IN_2D, in schema.py order
  read(u) psfc, gsw, glw, rainbl, emiss, albbck, shdmin, shdmax
  read(u) pblh, rmol, snowncv, ust, mavail, sst
  read(u) t2_ndg_old, t2_ndg_new, q2_ndg_old, q2_ndg_new
  read(u) sn_ndg_old, sn_ndg_new
  ! INOUT_2D
  read(u) ra, rs, lai, znt, qsfc, capg, canwat, qfx, hfx, lh
  read(u) psih, vegfra, vegf_px, snow, snoalb, snowh, snowc
  read(u) albedo, xland, xice, imperv, canfra
  ! OUT_2D
  read(u) grdflx, tsk, ta2, qa2, t2obs, q2obs
  read(u) lai_px, wwlt_px, wfc_px, wsat_px, clay_px, csand_px
  read(u) fmsand_px
  read(u) isltyp
  read(u) u3d, v3d, dz8w, qv3d, t3d, th3d, rho
  read(u) smois, tslb
  read(u) landusef, soilctop, soilcbot
  close(u)

  open(newunit=v, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  do step = 1, nsteps
    itimestep = itimestep0 + step - 1
    call pxlsm(u3d, v3d, dz8w, qv3d, t3d, th3d, rho,                 &
               psfc, gsw, glw, rainbl, emiss,                        &
               itimestep, curr_secs, 2, dt, anal_interval,           &
               xland, xice, albbck, albedo,                          &
               snoalb, smois, tslb, mavail, ta2,                     &
               qa2, qsfc, zs, dzs, psih,                             &
               landusef, soilctop, soilcbot, vegfra, vegf_px,        &
               isltyp, ra, rs, lai, imperv, canfra, nlcat, nscat,    &
               hfx, qfx, lh, tsk, sst, znt, canwat,                  &
               grdflx, shdmin, shdmax,                               &
               snowc, pblh, rmol, ust, capg, dtbl,                   &
               t2_ndg_old, t2_ndg_new,                               &
               q2_ndg_old, q2_ndg_new,                               &
               sn_ndg_old, sn_ndg_new, snow, snowh, snowncv,         &
               t2obs, q2obs, smois_init,                             &
               soil_nudge,                                           &
               modis_veg,                                            &
               lai_px,                                               &
               wwlt_px, wfc_px, wsat_px,                             &
               clay_px, csand_px, fmsand_px,                         &
               ids, ide, jds, jde, kds, kde,                         &
               ims, ime, jms, jme, kms, kme,                         &
               its, ite, jts, jte, kts, kte)
    write(v) ra, rs, lai, znt, qsfc, capg, canwat, qfx, hfx, lh
    write(v) psih, vegfra, vegf_px, snow, snoalb, snowh, snowc
    write(v) albedo, xland, xice, imperv, canfra
    write(v) grdflx, tsk, ta2, qa2, t2obs, q2obs
    write(v) lai_px, wwlt_px, wfc_px, wsat_px, clay_px, csand_px
    write(v) fmsand_px
    write(v) isltyp
    write(v) smois, tslb
    curr_secs = curr_secs + dt
  end do
  close(v)
end program run_pxlsm_oracle
