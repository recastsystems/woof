! Column driver for the WRF v4.6.1 QNSE surface layer
! (phys/module_sf_qnsesfc.F, sf_sfclay_physics = 4), byte-unmodified.
!
!   run_qnsesfc CASE.bin OUT.bin
!
! CASE.bin (little-endian stream, written by make_cases.py):
!   int32  ncol, nz, nsteps, it0
!   real32 HT(ncol)
!   real32 DZ, PMID, TH, T, QV, QC, U, V, Q2   each (ncol, nz), column fastest
!   real32 PINT(ncol, nz+1)
!   real32 TSK, XLAND, MAVAIL, Z0BASE          each (ncol)
!   real32 QSFC, THZ0, QZ0, UZ0, VZ0, USTAR, ZNT, AKHS, AKMS   each (ncol)
!   real32 HFX_LSM(ncol, nsteps), QFX_LSM(ncol, nsteps)
!
! The domain is one row of ncol columns (J = 1), KTE = nz, KME = nz + 1,
! LOWLYR = 1, as WRF's surface driver hands QNSESFC a tile.  QNSESFCINIT
! runs once with RESTART = .TRUE. (so it builds the PSIM/PSIH tables and sets
! LOWLYR without touching the seeded USTAR or ZNT); then QNSESFC runs nsteps
! times at ITIMESTEP = it0, it0+1, ..., carrying its own INOUT state between
! calls exactly as WRF's surface driver does, with SCM_FORCE_FLUX present
! and 0 (module_first_rk_step_part1.F:654 passes grid%scm_force_flux, whose
! Registry default is 0), so SFCDIF computes HFX and QFX.
!
! HFX and QFX are INOUT in QNSESFC and the land branch READS the incoming
! HFX (THZ0=HFX/(AKHS*RLOW*CP)+THLOW, module_sf_qnsesfc.F:744).  In WRF the
! land-surface model overwrites both over land after every surface-layer
! call, so before each call the driver writes HFX_LSM/QFX_LSM(:, step) into
! the land columns (XLAND < 1.5), standing in for that LSM.  Water columns
! keep the scheme's own values, as they do under Noah.
!
! OUT.bin (little-endian stream):
!   int32  ncol, nz, nsteps, it0, KZTM, NFIELD, NPRE
!   real32 DZETA1, DZETA2, ZTMIN1, ZTMAX1, ZTMIN2, ZTMAX2, FH01, FH02
!   real32 PSIM1, PSIH1, PSIM2, PSIH2          each (KZTM)
!   then per step: NPRE records of ncol words (the pre-call INOUT state, in
!   the order of PRE below), then NFIELD records of ncol words (post-call,
!   the launcher's MYJ_SFCLAY_INOUT then MYJ_SFCLAY_OUTPUTS names).
program run_qnsesfc
  use module_sf_qnsesfc
  implicit none

  integer, parameter :: NFIELD = 35, NPRE = 11
  character(len=256) :: case_path, out_path
  integer :: ncol, nz, nsteps, it0, step, unit_in, unit_out
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte

  real, allocatable :: ht(:,:), tsk(:,:), xland(:,:), mavail(:,:), z0base(:,:)
  real, allocatable :: dz(:,:,:), pmid(:,:,:), pint(:,:,:), th(:,:,:), t(:,:,:)
  real, allocatable :: qv(:,:,:), qc(:,:,:), u(:,:,:), v(:,:,:), q2(:,:,:)
  real, allocatable :: qsfc(:,:), thz0(:,:), qz0(:,:), uz0(:,:), vz0(:,:)
  real, allocatable :: ustar(:,:), znt(:,:), akhs(:,:), akms(:,:)
  real, allocatable :: pblh(:,:), rmol(:,:), rib(:,:), chs(:,:), chs2(:,:)
  real, allocatable :: cqs2(:,:), hfx(:,:), qfx(:,:), flx_lh(:,:), flhc(:,:)
  real, allocatable :: flqc(:,:), qgh(:,:), cpm(:,:), ct(:,:), u10(:,:)
  real, allocatable :: v10(:,:), t02(:,:), th02(:,:), tshltr(:,:), th10(:,:)
  real, allocatable :: q02(:,:), qshltr(:,:), q10(:,:), pshltr(:,:)
  real, allocatable :: u10e(:,:), v10e(:,:), seamask(:,:), xice(:,:)
  real, allocatable :: hfx_lsm(:,:), qfx_lsm(:,:)
  integer, allocatable :: lowlyr(:,:), ivgtyp(:,:)
  real, allocatable :: buf(:)
  integer :: k, i

  call get_command_argument(1, case_path)
  call get_command_argument(2, out_path)

  open(newunit=unit_in, file=trim(case_path), access='stream', &
       form='unformatted', status='old', convert='little_endian')
  read(unit_in) ncol, nz, nsteps, it0

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

  allocate(ht(ims:ime,jms:jme), tsk(ims:ime,jms:jme), xland(ims:ime,jms:jme))
  allocate(mavail(ims:ime,jms:jme), z0base(ims:ime,jms:jme))
  allocate(dz(ims:ime,kms:kme,jms:jme), pmid(ims:ime,kms:kme,jms:jme))
  allocate(pint(ims:ime,kms:kme,jms:jme), th(ims:ime,kms:kme,jms:jme))
  allocate(t(ims:ime,kms:kme,jms:jme), qv(ims:ime,kms:kme,jms:jme))
  allocate(qc(ims:ime,kms:kme,jms:jme), u(ims:ime,kms:kme,jms:jme))
  allocate(v(ims:ime,kms:kme,jms:jme), q2(ims:ime,kms:kme,jms:jme))
  allocate(qsfc(ims:ime,jms:jme), thz0(ims:ime,jms:jme), qz0(ims:ime,jms:jme))
  allocate(uz0(ims:ime,jms:jme), vz0(ims:ime,jms:jme), ustar(ims:ime,jms:jme))
  allocate(znt(ims:ime,jms:jme), akhs(ims:ime,jms:jme), akms(ims:ime,jms:jme))
  allocate(pblh(ims:ime,jms:jme), rmol(ims:ime,jms:jme), rib(ims:ime,jms:jme))
  allocate(chs(ims:ime,jms:jme), chs2(ims:ime,jms:jme), cqs2(ims:ime,jms:jme))
  allocate(hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme), flx_lh(ims:ime,jms:jme))
  allocate(flhc(ims:ime,jms:jme), flqc(ims:ime,jms:jme), qgh(ims:ime,jms:jme))
  allocate(cpm(ims:ime,jms:jme), ct(ims:ime,jms:jme), u10(ims:ime,jms:jme))
  allocate(v10(ims:ime,jms:jme), t02(ims:ime,jms:jme), th02(ims:ime,jms:jme))
  allocate(tshltr(ims:ime,jms:jme), th10(ims:ime,jms:jme), q02(ims:ime,jms:jme))
  allocate(qshltr(ims:ime,jms:jme), q10(ims:ime,jms:jme), pshltr(ims:ime,jms:jme))
  allocate(u10e(ims:ime,jms:jme), v10e(ims:ime,jms:jme))
  allocate(seamask(ims:ime,jms:jme), xice(ims:ime,jms:jme))
  allocate(lowlyr(ims:ime,jms:jme), ivgtyp(ims:ime,jms:jme))
  allocate(buf(ncol), hfx_lsm(ncol, nsteps), qfx_lsm(ncol, nsteps))

  ! Everything a column does not set is zero, never uninitialised memory.
  dz = 0.; pmid = 0.; pint = 0.; th = 0.; t = 0.; qv = 0.; qc = 0.
  u = 0.; v = 0.; q2 = 0.
  pblh = 0.; rmol = 0.; rib = 0.; chs = 0.; chs2 = 0.; cqs2 = 0.; hfx = 0.
  qfx = 0.; flx_lh = 0.; flhc = 0.; flqc = 0.; qgh = 0.; cpm = 0.; ct = 0.
  u10 = 0.; v10 = 0.; t02 = 0.; th02 = 0.; tshltr = 0.; th10 = 0.; q02 = 0.
  qshltr = 0.; q10 = 0.; pshltr = 0.; u10e = 0.; v10e = 0.
  xice = 0.; ivgtyp = 0; lowlyr = 1

  call read_2d(ht)
  call read_3d(dz, nz)
  call read_3d(pmid, nz)
  call read_3d(th, nz)
  call read_3d(t, nz)
  call read_3d(qv, nz)
  call read_3d(qc, nz)
  call read_3d(u, nz)
  call read_3d(v, nz)
  call read_3d(q2, nz)
  call read_3d(pint, nz + 1)
  call read_2d(tsk)
  call read_2d(xland)
  call read_2d(mavail)
  call read_2d(z0base)
  call read_2d(qsfc)
  call read_2d(thz0)
  call read_2d(qz0)
  call read_2d(uz0)
  call read_2d(vz0)
  call read_2d(ustar)
  call read_2d(znt)
  call read_2d(akhs)
  call read_2d(akms)
  do step = 1, nsteps
    read(unit_in) buf
    hfx_lsm(:, step) = buf
  end do
  do step = 1, nsteps
    read(unit_in) buf
    qfx_lsm(:, step) = buf
  end do
  close(unit_in)

  seamask = xland
  call QNSESFCINIT(LOWLYR, USTAR, ZNT, SEAMASK, XICE, IVGTYP, .TRUE.,   &
                   .TRUE., ids, ide, jds, jde, kds, kde,                 &
                   ims, ime, jms, jme, kms, kme,                         &
                   its, ite, jts, jte, kts, kte)

  open(newunit=unit_out, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', convert='little_endian')
  write(unit_out) ncol, nz, nsteps, it0, KZTM, NFIELD, NPRE
  write(unit_out) DZETA1, DZETA2, ZTMIN1, ZTMAX1, ZTMIN2, ZTMAX2, FH01, FH02
  write(unit_out) PSIM1, PSIH1, PSIM2, PSIH2

  do step = 0, nsteps - 1
    ! The land-surface model's write-back over land (see the header).
    do i = its, ite
      if (xland(i,1) < 1.5) then
        hfx(i,1) = hfx_lsm(i, step + 1)
        qfx(i,1) = qfx_lsm(i, step + 1)
      end if
    end do
    ! PRE: the INOUT words this call starts from.
    write(unit_out) ustar(:,1), znt(:,1), thz0(:,1), qz0(:,1), uz0(:,1), &
                    vz0(:,1), qsfc(:,1), akhs(:,1), akms(:,1), hfx(:,1),  &
                    qfx(:,1)
    call QNSESFC(it0 + step, HT, DZ                                      &
               ,PMID, PINT, TH, T, QV, QC, U, V, Q2                     &
               ,TSK, QSFC, THZ0, QZ0, UZ0, VZ0                          &
               ,LOWLYR, XLAND                                           &
               ,USTAR, ZNT, Z0BASE, PBLH, MAVAIL, RMOL                  &
               ,AKHS, AKMS                                              &
               ,RIB                                                     &
               ,CHS, CHS2, CQS2, HFX, QFX, FLX_LH, FLHC, FLQC           &
               ,QGH, CPM, CT                                            &
               ,U10, V10, T02, TH02, TSHLTR, TH10, Q02, QSHLTR, Q10, PSHLTR &
               ,U10E, V10E                                              &
               ,ids, ide, jds, jde, kds, kde                            &
               ,ims, ime, jms, jme, kms, kme                            &
               ,its, ite, jts, jte, kts, kte, 0)
    ! MYJ_SFCLAY_INOUT order
    write(unit_out) ustar(:,1), znt(:,1), thz0(:,1), qz0(:,1), uz0(:,1), &
                    vz0(:,1), qsfc(:,1), akhs(:,1), akms(:,1)
    ! MYJ_SFCLAY_OUTPUTS order
    write(unit_out) rmol(:,1), ct(:,1), pblh(:,1), rib(:,1), chs(:,1),    &
                    chs2(:,1), cqs2(:,1), hfx(:,1), qfx(:,1), flx_lh(:,1), &
                    flhc(:,1), flqc(:,1), qgh(:,1), cpm(:,1), u10(:,1),   &
                    v10(:,1), t02(:,1), th02(:,1), tshltr(:,1), th10(:,1), &
                    q02(:,1), qshltr(:,1), q10(:,1), pshltr(:,1),         &
                    u10e(:,1), v10e(:,1)
  end do
  close(unit_out)

contains

  subroutine read_2d(a)
    real, intent(inout) :: a(ims:ime,jms:jme)
    read(unit_in) buf
    a(:,1) = buf
  end subroutine read_2d

  subroutine read_3d(a, nk)
    real, intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    integer, intent(in) :: nk
    do k = 1, nk
      read(unit_in) buf
      a(:,k,1) = buf
    end do
  end subroutine read_3d

end program run_qnsesfc
