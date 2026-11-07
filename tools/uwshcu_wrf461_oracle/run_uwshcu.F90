! Full-driver column oracle for WRF v4.6.1 shcu_physics = 2 (CAMUWSHCU).
!
! Reads a driver case file written by make_cases.py, initializes exactly as
! WRF does for this scheme, and calls the byte-unmodified camuwshcu_driver
! for NSTEPS consecutive steps on every column, writing each step's inputs
! and outputs through uwio.  Usage:  run_uwshcu CASES.bin OUTSTEM
!
! Initialization, in WRF's order:
!   module_physics_init.F CAM_INIT: pcnst_runtime = 5 and pcnst_mp = 5 (EM
!   core, no chemistry, mp_physics /= CAMMGMP), esinti (the saturation
!   table), then the constituent arrays and the five cnst_add calls (Q,
!   CLDLIQ, CLDICE, NUMLIQ, NUMICE).
!   module_physics_init.F:4312-4321: init_uwshcu(r8, latvap, cpair, latice,
!   zvir, rair, gravit, mwh2o/mwdry, ..., bl_pbl_physics, ..., restart =
!   .false., full-level tile), which sets the module constants, rpen = 10
!   (uwshcu_readnl under WRF_PORT) and pver.
! Step: module_shallowcu_driver.F:517-553 with KTE = min(k_end, kde-1) =
!   nk and is_CAMMGMP_used = .false. (ArWen carries no CAMMGMP), so the
!   driver reads CLDFRA/CLDFRA_OLD; both are dead inside compute_uwshcu
!   (copied in, never read), which the port's tests rely on.
!
! Between steps the harness advances the state with the step's own
! tendencies in single precision (t, qv, qc, qi, qnc, qni, u, v) and
! carries every INOUT field (CUSH, HTOP/HBOT, the deep-plus-shallow
! accumulators) as WRF does.  That update is harness bookkeeping, not WRF:
! every step's exact inputs are written beside its outputs, so a port is
! graded step by step and the update rule never enters a comparison.
program run_uwshcu
  use shr_kind_mod, only: r8 => shr_kind_r8
  use module_cam_support, only: pcnst_runtime, pcnst_mp
  use module_cam_esinti, only: esinti
  use module_cam_gffgch, only: gffgch
  use physconst, only: mwh2o, cpwv, epsilo, latvap, latice, rh2o, cpair, &
       tmelt, mwdry, gravit, rair, zvir
  use constituents, only: cnst_add, cnst_name, cnst_longname, cnst_cp, &
       cnst_cv, cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
       cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, dcconnam, &
       fixcnam, tendnam, ptendnam, dmetendnam, sflxnam, tottnam
  use module_state_description, only: p_qc, p_qr, p_qi, p_qs, p_qg, &
       p_qnc, p_qni, param_first_scalar, CAMUWPBLSCHEME
  use uwshcu, only: init_uwshcu
  use module_shcu_camuwshcu_driver, only: camuwshcu_driver
  use uwio
  implicit none

  integer, parameter :: num_moist = 7
  character(len=512) :: casefile, outstem
  integer :: u_in, magic, version, ncol, nk, nsteps, has_qnc, has_qni
  real(4) :: dt
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: i, k, step, mm, ixcldliq, ixcldice, ixnumliq, ixnumice, n, itype
  real(8) :: t8, es8, table(250)
  character(len=16) :: tag

  real(4), allocatable, dimension(:,:,:) :: u, v, t, qv, qc, qi, qnc, qni, &
       p, z, pi, cldfra, cldfra_old, cldfra_old_mp, cldfra_conv, dz8w, &
       dlf, evapcsh, icwmrsh, rprdsh, shfrc3d, dlf2, cldfrash, &
       p8w, z_at_w, tke_pbl, cmfmc, cmfmc2, cmfsl, cmflq, &
       rushten, rvshten, rthshten, rqvshten, rqcshten, rqrshten, rqishten, &
       rqsshten, rqgshten, rqcnshten, rqinshten
  real(4), allocatable, dimension(:,:,:,:) :: moist
  real(4), allocatable, dimension(:,:) :: pblh, ht, cush, cutop, cubot, &
       rliq, rliq2, pratesh, snowsh, cbmf
  real(4), allocatable :: buf(:,:)

  call get_command_argument(1, casefile)
  call get_command_argument(2, outstem)

  open(newunit=u_in, file=trim(casefile), access='stream', form='unformatted', &
       status='old', action='read')
  read(u_in) magic, version, ncol, nk, nsteps, has_qnc, has_qni
  read(u_in) dt
  if (magic /= 1431786307 .or. version /= 1) error stop 'bad case file'  ! 'UWSC'

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nk + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nk + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1

  call alloc3(u); call alloc3(v); call alloc3(t); call alloc3(qv)
  call alloc3(qc); call alloc3(qi); call alloc3(qnc); call alloc3(qni)
  call alloc3(p); call alloc3(z); call alloc3(pi); call alloc3(cldfra)
  call alloc3(cldfra_old); call alloc3(cldfra_old_mp); call alloc3(cldfra_conv)
  call alloc3(dz8w); call alloc3(dlf); call alloc3(evapcsh); call alloc3(icwmrsh)
  call alloc3(rprdsh); call alloc3(shfrc3d); call alloc3(dlf2); call alloc3(cldfrash)
  call alloc3(p8w); call alloc3(z_at_w); call alloc3(tke_pbl); call alloc3(cmfmc)
  call alloc3(cmfmc2); call alloc3(cmfsl); call alloc3(cmflq)
  call alloc3(rushten); call alloc3(rvshten); call alloc3(rthshten)
  call alloc3(rqvshten); call alloc3(rqcshten); call alloc3(rqrshten)
  call alloc3(rqishten); call alloc3(rqsshten); call alloc3(rqgshten)
  call alloc3(rqcnshten); call alloc3(rqinshten)
  allocate(moist(ims:ime,kms:kme,jms:jme,num_moist)); moist = 0.
  call alloc2(pblh); call alloc2(ht); call alloc2(cush); call alloc2(cutop)
  call alloc2(cubot); call alloc2(rliq); call alloc2(rliq2); call alloc2(pratesh)
  call alloc2(snowsh); call alloc2(cbmf)

  call read_mass(u); call read_mass(v); call read_mass(t); call read_mass(qv)
  call read_mass(qc); call read_mass(qi); call read_mass(qnc); call read_mass(qni)
  call read_mass(p); call read_mass(z); call read_mass(pi); call read_mass(cldfra)
  call read_mass(cldfra_old); call read_mass(dlf); call read_mass(evapcsh)
  call read_mass(icwmrsh); call read_mass(rprdsh); call read_mass(shfrc3d)
  call read_full(p8w); call read_full(z_at_w); call read_full(tke_pbl)
  call read_full(cmfmc); call read_full(cmfmc2); call read_full(cmfsl)
  call read_full(cmflq)
  call read_sfc(pblh); call read_sfc(ht); call read_sfc(cush); call read_sfc(cutop)
  call read_sfc(cubot); call read_sfc(rliq); call read_sfc(rliq2); call read_sfc(pratesh)
  close(u_in)

  ! WRF's species indices: QNC/QNI exist only for microphysics that carry
  ! them; absent ones are WRF's dummy index 1 (< param_first_scalar).
  p_qnc = 1; p_qni = 1
  if (has_qnc /= 0) p_qnc = 7
  if (has_qni /= 0) p_qni = 8

  ! ---- CAM_INIT (module_physics_init.F), EM core, no chemistry ----
  pcnst_runtime = 5
  pcnst_mp = pcnst_runtime
  call esinti(epsilo, latvap, latice, rh2o, cpair, tmelt)
  call cam_constituents()

  ! ---- phy_init -> init_uwshcu (module_physics_init.F:4312), full tile ----
  call init_uwshcu(r8, latvap, cpair, latice, zvir, rair, gravit, mwh2o/mwdry, &
       rushten, rvshten, rthshten, rqvshten, &
       rqcshten, rqrshten, rqishten, rqsshten, rqgshten, &
       p_qc, p_qr, p_qi, p_qs, p_qg, &
       CAMUWPBLSCHEME, param_first_scalar, .false., &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kde)
  kte = nk

  call uwio_open(trim(outstem))
  uwio_prefix = 'const/'
  call uwio_put1_r4('dt', dt)
  call uwio_put1_i4('ncol', ncol); call uwio_put1_i4('nk', nk)
  call uwio_put1_i4('nsteps', nsteps)
  call uwio_put1_i4('has_qnc', has_qnc); call uwio_put1_i4('has_qni', has_qni)
  call uwio_put1_i4('ixnumliq', ixnumliq); call uwio_put1_i4('ixnumice', ixnumice)
  call uwio_put1_r8('qmin1', qmin(1)); call uwio_put1_r8('qmin2', qmin(2))
  call uwio_put1_r8('qmin3', qmin(3))
  ! gestbl's own table (module_cam_wv_saturation.F gestbl from esinti's
  ! arguments tmn=173.16, tmx=375.16, trice=20, ip=.true.), replayed through
  ! the public gffgch with gestbl's arithmetic, so the port's UW_ESTBL
  ! upload is graded against this host's own words.
  t8 = 173.16d0 - 1.0d0
  table = -99999.0d0
  do n = 1, int(375.16d0 - 173.16d0 + 2.000001d0)
     t8 = t8 + 1.0d0
     itype = -20
     call gffgch(t8, es8, itype)
     table(n) = es8
  end do
  call uwio_put_r8('estbl', table, 250)

  do step = 1, nsteps
     write(tag, '(a,i0,a)') 's', step, '/'
     uwio_prefix = tag
     call put_mass('u', u); call put_mass('v', v); call put_mass('t', t)
     call put_mass('qv', qv); call put_mass('qc', qc); call put_mass('qi', qi)
     call put_mass('qnc', qnc); call put_mass('qni', qni); call put_mass('p', p)
     call put_mass('z', z); call put_mass('pi', pi); call put_mass('cldfra', cldfra)
     call put_mass('cldfra_old', cldfra_old)
     call put_mass('dlf_in', dlf); call put_full('cmfmc_in', cmfmc)
     call put_full('p8w', p8w); call put_full('z_at_w', z_at_w)
     call put_full('tke_pbl', tke_pbl)
     call put_sfc('pblh', pblh); call put_sfc('ht', ht)
     call put_sfc('cush_in', cush); call put_sfc('cutop_in', cutop)
     call put_sfc('cubot_in', cubot); call put_sfc('rliq_in', rliq)
     call uwio_put1_i4('itimestep', step)

     ! module_first_rk_step_part1.F:1569 binds QV_CURR..QNI_CURR to the
     ! moist/scalar slices; the harness passes the same arrays.
     call camuwshcu_driver( &
          ids, ide, jds, jde, kds, kde, &
          ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte, &
          num_moist, dt, &
          p, p8w, pi, z, z_at_w, dz8w, &
          t, u, v, &
          moist, qv, qc, qi, qnc, qni, &
          pblh, tke_pbl, cldfra, cldfra_old, &
          cldfra_old_mp, cldfra_conv, .false., &
          cldfrash, &
          cush, pratesh, snowsh, icwmrsh, &
          cmfmc, cmfmc2, rprdsh, cbmf, &
          cmfsl, cmflq, dlf, dlf2, evapcsh, &
          rliq, rliq2, cubot, cutop, &
          rushten, rvshten, rthshten, &
          rqvshten, rqcshten, rqrshten, &
          rqishten, rqsshten, rqgshten, &
          rqcnshten, rqinshten, &
          ht, shfrc3d, step)

     call put_mass('rushten', rushten); call put_mass('rvshten', rvshten)
     call put_mass('rthshten', rthshten); call put_mass('rqvshten', rqvshten)
     call put_mass('rqcshten', rqcshten); call put_mass('rqishten', rqishten)
     call put_mass('rqcnshten', rqcnshten); call put_mass('rqinshten', rqinshten)
     call put_mass('dlf', dlf); call put_mass('dlf2', dlf2)
     call put_mass('shfrc3d', shfrc3d); call put_mass('evapcsh', evapcsh)
     call put_mass('icwmrsh', icwmrsh); call put_mass('rprdsh', rprdsh)
     call put_full('cmfmc', cmfmc); call put_full('cmfmc2', cmfmc2)
     call put_full('cmfsl', cmfsl); call put_full('cmflq', cmflq)
     call put_sfc('cush', cush); call put_sfc('pratesh', pratesh)
     call put_sfc('cutop', cutop); call put_sfc('cubot', cubot)
     call put_sfc('rliq', rliq); call put_sfc('rliq2', rliq2)

     ! harness bookkeeping only (see header)
     do k = kts, kte
        do i = its, ite
           t(i,k,1) = t(i,k,1) + dt * rthshten(i,k,1) * pi(i,k,1)
           qv(i,k,1) = max(0., qv(i,k,1) + dt * rqvshten(i,k,1))
           qc(i,k,1) = max(0., qc(i,k,1) + dt * rqcshten(i,k,1))
           qi(i,k,1) = max(0., qi(i,k,1) + dt * rqishten(i,k,1))
           qnc(i,k,1) = max(0., qnc(i,k,1) + dt * rqcnshten(i,k,1))
           qni(i,k,1) = max(0., qni(i,k,1) + dt * rqinshten(i,k,1))
           u(i,k,1) = u(i,k,1) + dt * rushten(i,k,1)
           v(i,k,1) = v(i,k,1) + dt * rvshten(i,k,1)
        end do
     end do
  end do
  call uwio_close()

contains

  subroutine cam_constituents()
    allocate(cnst_name(pcnst_runtime), cnst_longname(pcnst_runtime), &
         cnst_cp(pcnst_runtime), cnst_cv(pcnst_runtime), cnst_mw(pcnst_runtime), &
         cnst_type(pcnst_runtime), cnst_rgas(pcnst_runtime), qmin(pcnst_runtime), &
         qmincg(pcnst_runtime), cnst_fixed_ubc(pcnst_runtime), apcnst(pcnst_runtime), &
         bpcnst(pcnst_runtime), hadvnam(pcnst_runtime), vadvnam(pcnst_runtime), &
         dcconnam(pcnst_runtime), fixcnam(pcnst_runtime), tendnam(pcnst_runtime), &
         ptendnam(pcnst_runtime), dmetendnam(pcnst_runtime), sflxnam(pcnst_runtime), &
         tottnam(pcnst_runtime))
    cnst_fixed_ubc(:) = .false.
    call cnst_add('Q', mwh2o, cpwv, 1.E-12_r8, mm, longname='Specific humidity', readiv=.true.)
    call cnst_add('CLDLIQ', mwdry, cpair, 0._r8, ixcldliq, longname='Grid box averaged cloud liquid amount')
    call cnst_add('CLDICE', mwdry, cpair, 0._r8, ixcldice, longname='Grid box averaged cloud ice amount')
    call cnst_add('NUMLIQ', mwdry, cpair, 0._r8, ixnumliq, longname='Grid box averaged cloud liquid number')
    call cnst_add('NUMICE', mwdry, cpair, 0._r8, ixnumice, longname='Grid box averaged cloud ice number')
  end subroutine cam_constituents

  subroutine alloc3(a)
    real(4), allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime,kms:kme,jms:jme))
    a = 0.
  end subroutine alloc3

  subroutine alloc2(a)
    real(4), allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime,jms:jme))
    a = 0.
  end subroutine alloc2

  subroutine uwio_put1_r4(name, x)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: x
    real(4) :: a(1)
    a(1) = x
    call uwio_put_r4(name, a, 1)
  end subroutine uwio_put1_r4

  subroutine read_mass(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    allocate(buf(nk, ncol))
    read(u_in) buf
    do i = 1, ncol
       a(i, 1:nk, 1) = buf(:, i)
    end do
    deallocate(buf)
  end subroutine read_mass

  subroutine read_full(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    allocate(buf(nk + 1, ncol))
    read(u_in) buf
    do i = 1, ncol
       a(i, 1:nk + 1, 1) = buf(:, i)
    end do
    deallocate(buf)
  end subroutine read_full

  subroutine read_sfc(a)
    real(4), intent(inout) :: a(ims:ime,jms:jme)
    allocate(buf(1, ncol))
    read(u_in) buf
    a(1:ncol, 1) = buf(1, :)
    deallocate(buf)
  end subroutine read_sfc

  subroutine put_mass(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,kms:kme,jms:jme)
    real(4) :: o(nk, ncol)
    do i = 1, ncol
       o(:, i) = a(i, 1:nk, 1)
    end do
    call uwio_put_r4(name, o, nk * ncol)
  end subroutine put_mass

  subroutine put_full(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,kms:kme,jms:jme)
    real(4) :: o(nk + 1, ncol)
    do i = 1, ncol
       o(:, i) = a(i, 1:nk + 1, 1)
    end do
    call uwio_put_r4(name, o, (nk + 1) * ncol)
  end subroutine put_full

  subroutine put_sfc(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,jms:jme)
    call uwio_put_r4(name, a(1:ncol, 1), ncol)
  end subroutine put_sfc
end program run_uwshcu
