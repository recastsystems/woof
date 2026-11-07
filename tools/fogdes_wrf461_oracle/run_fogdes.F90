! FogDES column oracle: WRF v4.6.1 grav_settling = 1 and 2, byte-unmodified.
!
! Each case is one call shape of WRF's own two routines, in the order the
! drivers run them on a surface/PBL step:
!
!   module_surface_driver.F:2499-2516  DFGDP = 0, then sf_fogdes (VDFG, the
!                                      deposition DFGDP and its accumulator
!                                      FGDP), after the surface layer
!   module_pbl_driver.F:2195-2205      bl_fogdes adds the settling flux
!                                      divergence to the RQCBLTEN the PBL
!                                      scheme just wrote
!
! for NSTEPS consecutive steps.  Between steps the driver advances the
! column with the tendency it was handed (qc <- max(0, qc + dqc*dtbl)), so
! every step after the first starts from a state WRF's own arithmetic made.
! The synthetic RQCBLTEN a PBL scheme would have written is set before every
! bl_fogdes call, so the "add to what is there" half is graded too.
!
! Arrays are WRF's (ims:ime, kms:kme, jms:jme) with one j row, kme = NZ+1
! and kte = NZ = min(k_end, kde-1), the top MASS level the first-RK driver
! passes (dyn_em/module_first_rk_step_part1.F:1006).  Columns are i.
!
! Every input and every output of every step is written per case under
! <root>/<case>/ by oracle_io (../urban_wrf471_oracle/oracle_io.F90).
program run_fogdes
  use module_bl_fogdes, only: bl_fogdes
  use module_sf_fogdes, only: sf_fogdes
  use oracle_io
  implicit none
  integer, parameter :: nz = 20
  integer, parameter :: nk = nz + 1
  integer, parameter :: nsteps = 6
  character(len=1024) :: root

  call oracle_root(root)

  ! USGS 24 categories, FogDES (grav_settling = 2).  Every vegetation class
  ! of sf_fogdes:116-136 and its OTHERS arm (urban 1, water 16, barren 19,
  ! snow 24), dense radiation fog, thin fog, no cloud water, LAI = 0 (the
  ! LAD**-0.5 = Inf arm), and a strong-wind column that reaches the
  ! half-layer cap of :212.
  call run_case('usgs24_gs2', 2, 24, 60.0, &
       ivg  = [ 7,   2,   6,   8,  11,  15,  14,  16,   1,  19,   7,  11,  16,   7,  13,  10,  24,   9], &
       lai0 = [1.2, 0.5, 2.0, 3.5, 6.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 2.5, 0.0, 6.0, 0.0, 0.8], &
       wsp0 = [1.5, 0.8, 2.2, 3.0, 1.0, 0.6, 2.5, 4.0, 1.1, 2.0, 0.4, 1.7, 3.0, 1.0, 1.2, 40., 0.7, 2.9], &
       prof = [ 1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   3,   3,   1,   7,   2,   4], &
       dz1  = [ 8., 12., 20., 10., 15., 30., 9.,  25., 18., 11., 14., 22., 16., 8.,  13., 9.,  10., 12.], &
       pbl  = [ 0,   1,   2,   0,   1,   2,   0,   1,   0,   2,   1,   0,   1,   0,   2,   1,   0,   2])

  ! USGS 24 categories, Duynkerke settling (grav_settling = 1).  Land use is
  ! not read, so the columns vary the cloud water instead: dense, thin, the
  ! qcgmin = 1e-8 edges, an elevated stratus deck, cloud at the model top,
  ! a deep fog under a strong wind.
  call run_case('usgs24_gs1', 1, 24, 180.0, &
       ivg  = [ 7,   7,  16,   1,   7,  11,   7,   7,  16,  10], &
       lai0 = [1.2, 1.2, 0.0, 0.0, 2.0, 5.0, 1.0, 0.5, 0.0, 3.0], &
       wsp0 = [1.5, 0.8, 4.0, 1.1, 2.0, 0.6, 3.0, 0.2, 9.0, 40.], &
       prof = [ 1,   2,   1,   4,   5,   6,   3,   7,   2,   7], &
       dz1  = [ 8., 12., 20., 10., 15., 30., 9.,  25., 18., 6.], &
       pbl  = [ 0,   1,   2,   0,   1,   2,   0,   1,   0,   2])

  ! MODIS 21 categories (with lake), FogDES.  The MODIS arm of :139-156:
  ! grass 10, crop 12, mosaic 14, shrubs 6/7, broadleaf 2/4, mixed 5/19,
  ! needleleaf 1/3/8/9/18, and OTHERS (wetland 11, urban 13, snow 15,
  ! barren 16, water 17, lake 21), one column with cloud at the model top.
  call run_case('modis21_gs2', 2, 21, 30.0, &
       ivg  = [10,  12,  14,   6,   7,   2,   4,   5,  19,   1,   3,   8,   9,  18,  11,  13,  15,  16,  17,  21,  10,   2], &
       lai0 = [1.0, 2.5, 1.5, 0.7, 1.3, 5.5, 4.5, 3.8, 2.2, 6.0, 4.0, 1.1, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.5], &
       wsp0 = [2.0, 1.0, 0.5, 3.3, 1.8, 0.9, 2.4, 1.2, 0.7, 3.9, 1.6, 2.1, 1.4, 0.3, 2.2, 1.0, 0.6, 5.0, 7.0, 3.0, 2.0, 35.], &
       prof = [ 1,   1,   1,   1,   2,   1,   1,   1,   2,   1,   1,   2,   1,   1,   1,   1,   2,   1,   1,   2,   3,   6], &
       dz1  = [ 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 9.,  11., 13., 15., 17., 19., 21., 23., 25., 7.], &
       pbl  = [ 1,   0,   2,   1,   0,   2,   1,   0,   2,   1,   0,   2,   1,   0,   2,   1,   0,   2,   1,   0,   2,   1])

  ! MODIS 20 categories (no lake), Duynkerke settling.
  call run_case('modis20_gs1', 1, 20, 60.0, &
       ivg  = [10,  17,  13,   2,  16,  12], &
       lai0 = [1.0, 0.0, 0.0, 5.0, 0.0, 2.0], &
       wsp0 = [2.0, 6.0, 1.0, 0.5, 3.0, 1.5], &
       prof = [ 1,   1,   4,   5,   6,   7], &
       dz1  = [ 8., 20., 12., 30., 15., 10.], &
       pbl  = [ 0,   1,   2,   0,   1,   2])

  ! USGS 28 categories (with lake), FogDES: categories 25-28 are OTHERS.
  call run_case('usgs28_gs2', 2, 28, 90.0, &
       ivg  = [25,  26,  27,  28,   7,  22], &
       lai0 = [0.0, 0.0, 0.0, 0.0, 2.0, 3.0], &
       wsp0 = [2.0, 1.0, 3.0, 0.5, 1.2, 2.6], &
       prof = [ 1,   2,   1,   7,   1,   1], &
       dz1  = [10., 14., 18., 22., 12., 16.], &
       pbl  = [ 1,   0,   2,   1,   0,   2])

contains

  subroutine run_case(name, gs, nlcat, dtbl0, ivg, lai0, wsp0, prof, dz1, pbl)
    character(len=*), intent(in) :: name
    integer, intent(in) :: gs, nlcat
    real, intent(in) :: dtbl0
    integer, intent(in) :: ivg(:), prof(:), pbl(:)
    real, intent(in) :: lai0(:), wsp0(:), dz1(:)
    integer :: ncol, i, k, s
    real :: dtbl
    real, allocatable :: qc(:,:,:), rho(:,:,:), dz8w(:,:,:), dqc(:,:,:)
    real, allocatable :: vdfg(:,:), fgdp(:,:), dfgdp(:,:), lai(:,:), wspd(:,:)
    integer, allocatable :: ivgtyp(:,:)
    real, allocatable :: qc_in(:,:,:), dqc_in(:,:,:), dqc_out(:,:,:)
    real, allocatable :: fgdp_in(:,:), vdfg_out(:,:), dfgdp_out(:,:), fgdp_out(:,:)

    ncol = size(ivg)
    if (size(lai0) /= ncol .or. size(wsp0) /= ncol .or. size(prof) /= ncol &
        .or. size(dz1) /= ncol .or. size(pbl) /= ncol) &
        error stop 'run_fogdes: column table lengths differ'
    allocate(qc(ncol, nk, 1), rho(ncol, nk, 1), dz8w(ncol, nk, 1), dqc(ncol, nk, 1))
    allocate(vdfg(ncol, 1), fgdp(ncol, 1), dfgdp(ncol, 1), lai(ncol, 1), &
             wspd(ncol, 1), ivgtyp(ncol, 1))
    allocate(qc_in(ncol, nk, nsteps), dqc_in(ncol, nk, nsteps), dqc_out(ncol, nk, nsteps))
    allocate(fgdp_in(ncol, nsteps), vdfg_out(ncol, nsteps), dfgdp_out(ncol, nsteps), &
             fgdp_out(ncol, nsteps))

    dtbl = dtbl0
    do i = 1, ncol
      ivgtyp(i, 1) = ivg(i)
      lai(i, 1) = lai0(i)
      wspd(i, 1) = wsp0(i)
      do k = 1, nk
        qc(i, k, 1) = profile(prof(i), k)
        rho(i, k, 1) = 1.25 - 0.01 * real(k - 1)
        dz8w(i, k, 1) = dz1(i) + 4.0 * real(k - 1)
      end do
      qc(i, nk, 1) = 0.
    end do
    vdfg = 0.
    fgdp = 0.

    do s = 1, nsteps
      do i = 1, ncol
        do k = 1, nk
          dqc(i, k, 1) = pbl_tendency(pbl(i), k, s)
        end do
      end do
      qc_in(:, :, s) = qc(:, :, 1)
      dqc_in(:, :, s) = dqc(:, :, 1)
      fgdp_in(:, s) = fgdp(:, 1)
      dfgdp = 0.
      call sf_fogdes(vdfg, fgdp, dfgdp, ivgtyp, lai, wspd, qc,          &
                     dtbl, rho, dz8w, gs, nlcat,                      &
                     1, ncol, 1, 1, 1, nk,                            &
                     1, ncol, 1, 1, 1, nk,                            &
                     1, ncol, 1, 1, 1, nz)
      call bl_fogdes(vdfg, qc, dtbl, rho, dz8w, gs, dqc,                &
                     1, ncol, 1, 1, 1, nk,                            &
                     1, ncol, 1, 1, 1, nk,                            &
                     1, ncol, 1, 1, 1, nz)
      vdfg_out(:, s) = vdfg(:, 1)
      dfgdp_out(:, s) = dfgdp(:, 1)
      fgdp_out(:, s) = fgdp(:, 1)
      dqc_out(:, :, s) = dqc(:, :, 1)
      do i = 1, ncol
        do k = 1, nz
          qc(i, k, 1) = max(0., qc(i, k, 1) + dqc(i, k, 1) * dtbl)
        end do
      end do
    end do

    call oracle_open(name)
    call oracle_put('grav_settling', gs)
    call oracle_put('nlcat', nlcat)
    call oracle_put('kte', nz)
    call oracle_put('dtbl', dtbl)
    call oracle_put('ivgtyp', ivgtyp(:, 1))
    call oracle_put('lai', lai(:, 1))
    call oracle_put('wspd', wspd(:, 1))
    call oracle_put('rho', rho(:, :, 1))
    call oracle_put('dz8w', dz8w(:, :, 1))
    call oracle_put('qc_in', qc_in)
    call oracle_put('dqc_in', dqc_in)
    call oracle_put('fgdp_in', fgdp_in)
    call oracle_put('vdfg', vdfg_out)
    call oracle_put('dfgdp', dfgdp_out)
    call oracle_put('fgdp', fgdp_out)
    call oracle_put('rqcblten', dqc_out)
    call oracle_close()
  end subroutine run_case

  ! Cloud water profiles (kg/kg), k = 1 at the surface.
  real function profile(kind, k)
    integer, intent(in) :: kind, k
    real, parameter :: dense(8) = [4.2e-4, 3.6e-4, 2.9e-4, 2.1e-4, 1.2e-4, &
                                   5.0e-5, 8.0e-6, 3.0e-7]
    real, parameter :: thin(6) = [3.0e-5, 1.8e-5, 6.0e-6, 1.5e-6, 2.0e-7, 1.1e-8]
    real, parameter :: edge(6) = [1.0e-8, 1.2e-8, 2.0e-8, 0.9e-8, 1.0e-8, 1.1e-8]
    real, parameter :: stratus(5) = [1.0e-4, 2.2e-4, 2.5e-4, 1.5e-4, 5.0e-5]
    real, parameter :: deep(10) = [6.5e-4, 6.0e-4, 5.4e-4, 4.6e-4, 3.8e-4, &
                                   2.9e-4, 1.8e-4, 9.0e-5, 2.0e-5, 1.0e-6]
    profile = 0.
    select case (kind)
    case (1)            ! dense radiation fog
      if (k <= 8) profile = dense(k)
    case (2)            ! thin fog
      if (k <= 6) profile = thin(k)
    case (3)            ! no cloud water
      profile = 0.
    case (4)            ! the qcgmin edges: 1e-8 is not > 1e-8
      if (k <= 6) profile = edge(k)
    case (5)            ! elevated stratus, clear at the surface
      if (k >= 7 .and. k <= 11) profile = stratus(k - 6)
    case (6)            ! dense fog plus cloud in the top two mass levels
      if (k <= 8) profile = dense(k)
      if (k == nz - 1) profile = 5.0e-5
      if (k == nz) profile = 8.0e-5
    case (7)            ! deep fog
      if (k <= 10) profile = deep(k)
    case default
      error stop 'run_fogdes: unknown profile'
    end select
  end function profile

  ! The RQCBLTEN a PBL scheme hands bl_fogdes (kg/kg/s).
  real function pbl_tendency(kind, k, s)
    integer, intent(in) :: kind, k, s
    pbl_tendency = 0.
    select case (kind)
    case (0)
      pbl_tendency = 0.
    case (1)
      pbl_tendency = 1.0e-9 * real(mod(k + s, 5) - 2)
    case (2)
      if (k <= 3) then
        pbl_tendency = -3.0e-8
      else if (k <= 6) then
        pbl_tendency = 1.0e-8
      end if
    case default
      error stop 'run_fogdes: unknown PBL tendency'
    end select
  end function pbl_tendency

end program run_fogdes
