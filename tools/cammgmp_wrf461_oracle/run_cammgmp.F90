! WRF v4.6.1 CAMMGMP (mp_physics = 11) column oracle driver.
!
! Links the SCALAR WRF build's own libwrflib.a (the 0 ULP referee build:
! -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize, gfortran 13.3,
! glibc 2.39) and calls, unchanged, the same entry points WRF calls:
!   module_physics_init::CAM_INIT          (constituents, esinti, modal aerosol)
!   module_mp_cammgmp_driver::CAMMGMP_INIT (ini_micro, ini_microp_aero,
!                                           activate_init)
!   module_mp_cammgmp_driver::CAMMGMP      (one call per step over a tile)
! No WRF source is extracted or reimplemented here.  The driver only reads
! the input deck, calls WRF, writes what WRF returned, and between steps
! applies the deck's float32 forcing (th, qv) and t_phy = th*pi_phy, which
! the GPU harness reproduces with the same float32 operations.
!
! Usage: run_cammgmp <input.bin> <output.bin> <helpers.bin> <words.txt>
!
! Input layout (little-endian stream, see make_cases.py): a header
! (magic, version, ngroups) then per group an integer block
! (ncol, nz, nstep, itimestep0, cu_physics, shcu_physics), a real block
! (dt, accum_mode, aitken_mode, coarse_mode) and the float32 arrays in the
! order of the READ statements below, Fortran order (column fastest).
program run_cammgmp
  use module_configure,          only: grid_config_rec_type
  use module_physics_init,       only: cam_init
  use module_mp_cammgmp_driver,  only: cammgmp_init, cammgmp
  use module_cam_support,        only: pver
  use wv_saturation,             only: estblf, polysvp
  use error_function,            only: erf
  use cldwat2m_micro,            only: gamma
  use modal_aero_data,           only: ntot_amode_mp, sigmag_amode_mp,        &
       dgnum_amode_mp, voltonumb_amode_mp, voltonumblo_amode_mp,              &
       voltonumbhi_amode_mp, alnsg_amode_mp, specdens_amode_mp, spechygro_mp, &
       numptr_amode_mp, nspec_amode_mp, lmassptr_amode_mp, lspectype_amode_mp
  use shr_kind_mod,              only: r8 => shr_kind_r8
  use iso_c_binding,             only: c_char, c_null_char
  implicit none

  interface
     subroutine cammgmp_dump_words(path) bind(C, name='cammgmp_dump_words')
       import :: c_char
       character(kind=c_char), dimension(*), intent(in) :: path
     end subroutine cammgmp_dump_words
  end interface

  type(grid_config_rec_type) :: cf
  character(len=1024) :: fin, fout, fhelp, fwords
  integer :: magic, version, ngroups, g, ncol, nz, nstep, it0, cu, shcu, istep
  integer :: ixcldliq, ixcldice, ixnumliq, ixnumice
  real :: dt, accum, aitken, coarse
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, i, k, n
  logical :: initialised

  real, allocatable, dimension(:,:,:) :: p8w, p_hyd, t_phy, pi_phy, z_at_w, &
       tke_pbl, turbtype3d, smaw3d, dlf3d, dlf2_3d, z_sea_level, kvh3d, alt, &
       icwmrsh3d, icwmrdp3d, shfrc3d, cmfmc3d, cmfmc2_3d, f_ice_phy,        &
       f_rain_phy, th, cldfra_old_mp, cldfra_mp, cldfra_mp_all, lradius,    &
       iradius, cldfrai, cldfral, cldfra_conv, wsedl3d, qv, qc, qi, qs, qr,  &
       nc3d, ni3d, ns3d, nr3d, qndrop, rh_old_mp, lcd_old_mp, dthdt, dqvdt
  real, allocatable, dimension(:,:) :: qfx, rliq2d, ht, xland, snowh, &
       rainnc, rainncv, snownc, snowncv, sr

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, fhelp)
  call get_command_argument(4, fwords)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', &
       convert='little_endian')
  open(11, file=trim(fout), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  read(10) magic, version, ngroups
  if (magic /= 1297040707 .or. version /= 1) stop 'bad input header'
  write(11) magic, version, ngroups

  initialised = .false.
  do g = 1, ngroups
     read(10) ncol, nz, nstep, it0, cu, shcu
     read(10) dt, accum, aitken, coarse
     write(11) ncol, nz, nstep, it0, cu, shcu
     write(11) dt, accum, aitken, coarse

     ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
     ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
     its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

     call alloc3(p8w); call alloc3(p_hyd); call alloc3(t_phy); call alloc3(pi_phy)
     call alloc3(z_at_w); call alloc3(tke_pbl); call alloc3(turbtype3d)
     call alloc3(smaw3d); call alloc3(dlf3d); call alloc3(dlf2_3d)
     call alloc3(z_sea_level); call alloc3(kvh3d); call alloc3(alt)
     call alloc3(icwmrsh3d); call alloc3(icwmrdp3d); call alloc3(shfrc3d)
     call alloc3(cmfmc3d); call alloc3(cmfmc2_3d); call alloc3(f_ice_phy)
     call alloc3(f_rain_phy); call alloc3(th); call alloc3(cldfra_old_mp)
     call alloc3(cldfra_mp); call alloc3(cldfra_mp_all); call alloc3(lradius)
     call alloc3(iradius); call alloc3(cldfrai); call alloc3(cldfral)
     call alloc3(cldfra_conv); call alloc3(wsedl3d); call alloc3(qv)
     call alloc3(qc); call alloc3(qi); call alloc3(qs); call alloc3(qr)
     call alloc3(nc3d); call alloc3(ni3d); call alloc3(ns3d); call alloc3(nr3d)
     call alloc3(qndrop); call alloc3(rh_old_mp); call alloc3(lcd_old_mp)
     call alloc3(dthdt); call alloc3(dqvdt)
     call alloc2(qfx); call alloc2(rliq2d); call alloc2(ht); call alloc2(xland)
     call alloc2(snowh); call alloc2(rainnc); call alloc2(rainncv)
     call alloc2(snownc); call alloc2(snowncv); call alloc2(sr)

     ! Interface-level (kme) inputs are read over all nz+1 levels, mass-level
     ! inputs over the same storage (level nz+1 is carried but unused).
     read(10) p8w, p_hyd, t_phy, pi_phy, z_at_w, tke_pbl, turbtype3d, smaw3d, &
          dlf3d, dlf2_3d, z_sea_level, kvh3d, alt, icwmrsh3d, icwmrdp3d,      &
          shfrc3d, cmfmc3d, cmfmc2_3d, f_ice_phy, f_rain_phy, th,             &
          cldfra_old_mp, cldfra_mp, cldfra_mp_all, lradius, iradius, cldfrai, &
          cldfral, cldfra_conv, wsedl3d, qv, qc, qi, qs, qr, nc3d, ni3d, ns3d,&
          nr3d, qndrop, rh_old_mp, lcd_old_mp, dthdt, dqvdt
     read(10) qfx, rliq2d, ht, xland, snowh, rainnc, rainncv, snownc, snowncv, sr

     cf%mp_physics     = 11
     cf%cu_physics     = cu
     cf%shcu_physics   = shcu
     cf%bl_pbl_physics = 9
     cf%chem_opt       = 0
     if (.not. initialised) then
        call cam_init(ixcldliq, ixcldice, ixnumliq, ixnumice, cf)
        initialised = .true.
     end if
     ! CAMMGMP_INIT sets pver from kde and (re)derives the module constants;
     ! WRF calls it once per domain at phy_init.
     call cammgmp_init(ixcldliq, ixcldice, ixnumliq, ixnumice, 0, &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte)
     if (pver /= nz) stop 'pver mismatch'

     do istep = 1, nstep
        call cammgmp(it0 + istep - 1, dt, p8w, p_hyd, t_phy, pi_phy, z_at_w, qfx, &
             tke_pbl, turbtype3d, smaw3d, dlf3d, dlf2_3d, rliq2d, z_sea_level,  &
             kvh3d, ht, alt, accum, aitken, coarse, icwmrsh3d, icwmrdp3d,       &
             shfrc3d, cmfmc3d, cmfmc2_3d, cf, f_ice_phy, f_rain_phy,            &
             ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme,        &
             its, ite, jts, jte, kts, kte,                                      &
             th, cldfra_old_mp, cldfra_mp, cldfra_mp_all, lradius, iradius,     &
             cldfrai, cldfral, cldfra_conv, wsedl3d, rainnc, rainncv, snownc,   &
             snowncv, sr, qv, qc, qi, qs, qr, nc3d, ni3d, ns3d, nr3d, qndrop,   &
             rh_old_mp, lcd_old_mp, xland, snowh)
        write(11) th(:,1:nz,:), qv(:,1:nz,:), qc(:,1:nz,:), qi(:,1:nz,:),     &
             qs(:,1:nz,:), qr(:,1:nz,:), nc3d(:,1:nz,:), ni3d(:,1:nz,:),       &
             ns3d(:,1:nz,:), nr3d(:,1:nz,:), qndrop(:,1:nz,:),                 &
             cldfra_old_mp(:,1:nz,:), cldfra_mp(:,1:nz,:),                     &
             cldfra_mp_all(:,1:nz,:), lradius(:,1:nz,:), iradius(:,1:nz,:),    &
             cldfrai(:,1:nz,:), cldfral(:,1:nz,:), cldfra_conv(:,1:nz,:),      &
             wsedl3d(:,1:nz,:), rh_old_mp(:,1:nz,:), lcd_old_mp(:,1:nz,:)
        write(11) rainnc, rainncv, snownc, snowncv, sr
        ! Between-step forcing, float32, in this order (the GPU harness
        ! repeats it operation for operation).
        do k = 1, nz
           do i = 1, ncol
              th(i,k,1) = th(i,k,1) + dthdt(i,k,1)*dt
              qv(i,k,1) = qv(i,k,1) + dqvdt(i,k,1)*dt
              t_phy(i,k,1) = th(i,k,1)*pi_phy(i,k,1)
           end do
        end do
     end do

     deallocate(p8w, p_hyd, t_phy, pi_phy, z_at_w, tke_pbl, turbtype3d,      &
          smaw3d, dlf3d, dlf2_3d, z_sea_level, kvh3d, alt, icwmrsh3d,          &
          icwmrdp3d, shfrc3d, cmfmc3d, cmfmc2_3d, f_ice_phy, f_rain_phy, th,   &
          cldfra_old_mp, cldfra_mp, cldfra_mp_all, lradius, iradius, cldfrai,  &
          cldfral, cldfra_conv, wsedl3d, qv, qc, qi, qs, qr, nc3d, ni3d, ns3d, &
          nr3d, qndrop, rh_old_mp, lcd_old_mp, dthdt, dqvdt)
     deallocate(qfx, rliq2d, ht, xland, snowh, rainnc, rainncv, snownc, &
          snowncv, sr)
  end do
  close(10)
  close(11)
  call write_helpers(trim(fhelp))
  call cammgmp_dump_words(trim(fwords)//c_null_char)

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme))
    a = 0.0
  end subroutine alloc3

  subroutine alloc2(a)
    real, allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime, jms:jme))
    a = 0.0
  end subroutine alloc2

  ! Helper words for the device library's unit checks: WRF's own estblf,
  ! polysvp, the CAM error function and MG's GAMMA, evaluated by WRF on a
  ! fixed argument grid, plus the modal-aerosol tables CAM_INIT published.
  subroutine write_helpers(path)
    character(len=*), intent(in) :: path
    integer, parameter :: nt = 20001, ne = 20001, ng = 4001
    real(r8) :: t, x, y
    integer :: j
    open(12, file=path, access='stream', form='unformatted', status='replace', &
         convert='little_endian')
    write(12) nt, ne, ng
    do j = 1, nt
       t = 150._r8 + real(j - 1, r8)*0.0115_r8
       write(12) t, estblf(t), polysvp(t, 0), polysvp(t, 1)
    end do
    do j = 1, ne
       x = -7._r8 + real(j - 1, r8)*0.0007_r8
       write(12) x, erf(x)
    end do
    do j = 1, ng
       y = 0.05_r8 + real(j - 1, r8)*0.00525_r8
       write(12) y, gamma(y)
    end do
    write(12) ntot_amode_mp
    write(12) sigmag_amode_mp(1:ntot_amode_mp), dgnum_amode_mp(1:ntot_amode_mp), &
         voltonumb_amode_mp(1:ntot_amode_mp), voltonumblo_amode_mp(1:ntot_amode_mp), &
         voltonumbhi_amode_mp(1:ntot_amode_mp), alnsg_amode_mp(1:ntot_amode_mp)
    write(12) specdens_amode_mp(1:3), spechygro_mp(1:3)
    write(12) numptr_amode_mp(1:ntot_amode_mp), nspec_amode_mp(1:ntot_amode_mp)
    write(12) lmassptr_amode_mp(1:2, 1:ntot_amode_mp), lspectype_amode_mp(1:2, 1:ntot_amode_mp)
    close(12)
  end subroutine write_helpers

end program run_cammgmp
