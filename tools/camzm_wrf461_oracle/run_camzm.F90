! Full-driver column oracle for WRF v4.6.1 cu_physics = 7 (CAMZMSCHEME).
!
! Reads a case file written by make_cases.py, initializes exactly as WRF
! does for this scheme, and calls camzm_driver for NSTEPS consecutive steps
! over every column, writing each step's inputs and outputs through zmio.
! Usage:  run_camzm CASES.bin OUTSTEM
!
! Initialization, in WRF's order:
!   module_physics_init.F CAM_INIT (body at SUBROUTINE CAM_INIT, :5011):
!   pcnst_runtime = 5 (EM core, no chemistry, mp_physics /= CAMMGMP),
!   esinti (the saturation table), the constituent arrays
!   ALLOCATE_CAM_ARRAYS allocates, and the five cnst_add calls (Q, CLDLIQ,
!   CLDICE, NUMLIQ, NUMICE).
!   module_physics_init.F:4162-4173: zm_conv_init(DT, DX, ...,
!   restart=.false.), which sets pver/pverp, limcnv = 2, the hard-wired
!   zmconv namelist (c0_lnd, c0_ocn, ke) and zm_convi's tau(dx).
! Step: module_cumulus_driver.F:1332-1388, KTE = min(k_end, kde-1) = nk,
!   is_CAMMGMP_used = .false. (WOOF has no CAMMGMP microphysics), so
!   CLDFRA is the cloud fraction the scheme reads.  ADAPT_STEP_FLAG is
!   passed .false. and CUDTACTTIME 0.
!
! Between steps the harness advances the state with the step's own
! tendencies in single precision (th, t = th*pi, qv, qc, qi, u, v).  That
! update is harness bookkeeping, not WRF: every step's exact inputs are
! written beside its outputs, so a port is graded step by step and the
! update rule never enters a comparison.
!
! jt2d and ideep2d are copied by camzm_driver from local arrays
! that zm_convr leaves unassigned when no column convects (lengath = 0,
! zm_convr returns at :677).  Their words are then stack contents, not
! scheme output, so the harness writes -1 for them wherever lengath2d = 0.
! maxg is initialized to 1 before that return and must remain unmodified.
! That is the one post-processing step applied to any output.
program run_camzm
  use shr_kind_mod, only: r8 => shr_kind_r8
  use module_cam_support, only: pcnst_runtime, pcnst_mp, pver, pverp
  use module_cam_esinti, only: esinti
  use physconst, only: mwh2o, cpwv, epsilo, latvap, latice, rh2o, cpair, &
       tmelt, mwdry, gravit, rair, cpliq
  use constituents, only: cnst_add, cnst_name, cnst_longname, cnst_cp, &
       cnst_cv, cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
       cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, dcconnam, &
       fixcnam, tendnam, ptendnam, dmetendnam, sflxnam, tottnam
  use wv_saturation, only: estblf
  use module_cu_camzm_driver, only: camzm_driver, zm_conv_init
  use zmio
  implicit none

  character(len=512) :: casefile, outstem, mode
  logical :: swapx
  real(4), allocatable :: xland_call(:,:)
  integer :: u_in, magic, version, ncol, nk, nsteps, blpbl, sfclay, itime0
  real(4) :: dt, dx, cudt, cudtacttime, curr_secs
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: i, k, n, step, mm, ixcldliq, ixcldice, ixnumliq, ixnumice
  integer :: stepcu, itimestep
  character(len=16) :: tag
  real(r8) :: table(250)

  ! inputs
  real(4), allocatable, dimension(:,:,:) :: th, t, p, pi, qv, qc, qi, z, dz8w, &
       u, v, cldfra, cldfra_mp_all, p8w, z_at_w, tke_pbl
  real(4), allocatable, dimension(:,:) :: tsk, ust, mavail, pblh, xland, ht, &
       psfc, hfx, qfx, tpert
  integer, allocatable, dimension(:,:) :: kpbl
  ! outputs
  real(4), allocatable, dimension(:,:,:) :: mu_out, md_out, zmdt, zmdq, dlf_out, &
       rucuten, rvcuten, rthcuten, rqvcuten, rqccuten, rqicuten, rqcncuten, &
       rqincuten, evaptzm, fzsntzm, evsntzm, evapqzm, zmflxprc, zmflxsnw, &
       zmntprpd, zmntsnpd, zmeiheat, cmfmc, cmfmcdzm, zmmtu, zmmtv, zmupgu, &
       zmupgd, zmvpgu, zmvpgd, zmicuu, zmicud, zmicvu, zmicvd, zmdice, zmdliq, &
       evapcdp3d, icwmrdp3d, rprddp3d, dp3d, du3d, ed3d, eu3d, md3d, mu3d
  real(4), allocatable, dimension(:,:) :: cape_out, rliq_out, pconvb, pconvt, &
       cubot, cutop, raincv, pratec, preccdzm, precz, dsubcld2d
  integer, allocatable, dimension(:,:) :: ideep2d, jt2d, maxg2d, lengath2d

  call get_command_argument(1, casefile)
  call get_command_argument(2, outstem)
  ! Optional third argument "swap-xland": hand camzm_driver XLAND with 1 and
  ! 2 exchanged (the recorded inputs keep the true XLAND).  Run against the
  ! stock sources it must reproduce the fixed build word for word, which
  ! proves the landfrac fix is exactly an inversion of that one test and
  ! touches nothing else (camzm_driver reads XLAND nowhere else).
  mode = ''
  if (command_argument_count() >= 3) call get_command_argument(3, mode)
  swapx = trim(mode) == "swap-xland"

  open(newunit=u_in, file=trim(casefile), access='stream', form='unformatted', &
       status='old', action='read')
  read(u_in) magic, version, ncol, nk, nsteps, blpbl, sfclay, itime0, stepcu
  read(u_in) dt, dx, cudt
  if (magic /= 1129991479 .or. version /= 1) error stop 'bad case file'  ! 'CZM7'

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nk + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nk + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nk

  call alloc3(th); call alloc3(t); call alloc3(p); call alloc3(pi)
  call alloc3(qv); call alloc3(qc); call alloc3(qi); call alloc3(z)
  call alloc3(dz8w); call alloc3(u); call alloc3(v); call alloc3(cldfra)
  call alloc3(cldfra_mp_all); call alloc3(p8w); call alloc3(z_at_w)
  call alloc3(tke_pbl)
  call alloc2(tsk); call alloc2(ust); call alloc2(mavail); call alloc2(pblh)
  call alloc2(xland); call alloc2(ht); call alloc2(psfc); call alloc2(hfx)
  call alloc2(qfx); call alloc2(tpert); call alloc2(xland_call)
  allocate(kpbl(ims:ime,jms:jme)); kpbl = 0

  call alloc3(mu_out); call alloc3(md_out); call alloc3(zmdt); call alloc3(zmdq)
  call alloc3(dlf_out); call alloc3(rucuten); call alloc3(rvcuten)
  call alloc3(rthcuten); call alloc3(rqvcuten); call alloc3(rqccuten)
  call alloc3(rqicuten); call alloc3(rqcncuten); call alloc3(rqincuten)
  call alloc3(evaptzm); call alloc3(fzsntzm); call alloc3(evsntzm)
  call alloc3(evapqzm); call alloc3(zmflxprc); call alloc3(zmflxsnw)
  call alloc3(zmntprpd); call alloc3(zmntsnpd); call alloc3(zmeiheat)
  call alloc3(cmfmc); call alloc3(cmfmcdzm); call alloc3(zmmtu); call alloc3(zmmtv)
  call alloc3(zmupgu); call alloc3(zmupgd); call alloc3(zmvpgu); call alloc3(zmvpgd)
  call alloc3(zmicuu); call alloc3(zmicud); call alloc3(zmicvu); call alloc3(zmicvd)
  call alloc3(zmdice); call alloc3(zmdliq); call alloc3(evapcdp3d)
  call alloc3(icwmrdp3d); call alloc3(rprddp3d); call alloc3(dp3d); call alloc3(du3d)
  call alloc3(ed3d); call alloc3(eu3d); call alloc3(md3d); call alloc3(mu3d)
  call alloc2(cape_out); call alloc2(rliq_out); call alloc2(pconvb)
  call alloc2(pconvt); call alloc2(cubot); call alloc2(cutop); call alloc2(raincv)
  call alloc2(pratec); call alloc2(preccdzm); call alloc2(precz); call alloc2(dsubcld2d)
  allocate(ideep2d(ims:ime,jms:jme), jt2d(ims:ime,jms:jme), &
       maxg2d(ims:ime,jms:jme), lengath2d(ims:ime,jms:jme))
  ideep2d = 0; jt2d = 0; maxg2d = 0; lengath2d = 0

  call read_mass(th); call read_mass(t); call read_mass(p); call read_mass(pi)
  call read_mass(qv); call read_mass(qc); call read_mass(qi); call read_mass(z)
  call read_mass(dz8w); call read_mass(u); call read_mass(v); call read_mass(cldfra)
  call read_full(p8w); call read_full(z_at_w); call read_full(tke_pbl)
  call read_sfc(tsk); call read_sfc(ust); call read_sfc(mavail); call read_sfc(pblh)
  call read_sfc(xland); call read_sfc(ht); call read_sfc(psfc); call read_sfc(hfx)
  call read_sfc(qfx); call read_sfc(tpert)
  read(u_in) kpbl(1:ncol,1)
  close(u_in)
  cldfra_mp_all = 0.

  ! ---- CAM_INIT (module_physics_init.F:5011), EM core, no chemistry ----
  pcnst_runtime = 5
  pcnst_mp = pcnst_runtime
  call esinti(epsilo, latvap, latice, rh2o, cpair, tmelt)
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

  ! ---- phy_init -> cu_init -> zm_conv_init (module_physics_init.F:4162) ----
  call zm_conv_init(dt, dx, rucuten, rvcuten, rthcuten, rqvcuten, &
       rqccuten, rqicuten, rqcncuten, rqincuten, &
       2, 3, 4, 5, param_first_scalar_value(), .false., &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte)

  call zmio_open(trim(outstem))
  zmio_prefix = 'const/'
  call zmio_put1_r8('cpair', cpair); call zmio_put1_r8('gravit', gravit)
  call zmio_put1_r8('rair', rair); call zmio_put1_r8('latvap', latvap)
  call zmio_put1_r8('latice', latice); call zmio_put1_r8('epsilo', epsilo)
  call zmio_put1_r8('rh2o', rh2o); call zmio_put1_r8('tmelt', tmelt)
  call zmio_put1_r8('cpwv', cpwv); call zmio_put1_r8('cpliq', cpliq)
  call zmio_put1_r4('dt', dt); call zmio_put1_r4('dx', dx)
  call zmio_put1_r4('cudt', cudt)
  call zmio_put1_i4('ncol', ncol); call zmio_put1_i4('nk', nk)
  call zmio_put1_i4('nsteps', nsteps); call zmio_put1_i4('bl_pbl_physics', blpbl)
  call zmio_put1_i4('sf_sfclay_physics', sfclay); call zmio_put1_i4('itimestep0', itime0)
  call zmio_put1_i4('stepcu', stepcu)
  call zmio_put1_i4('pver', pver); call zmio_put1_i4('pverp', pverp)
  ! camzm_driver's detrained-number denominators (module_cu_camzm_driver.F
  ! :713-714), spelled exactly as there so the compiler folds them the same
  ! way; and zmconv_readnl's ke, a default-REAL literal stored in real(r8).
  call zmio_put1_r8('qcn_den', (4._r8*3.14159_r8*(8.e-6_r8**3)*997._r8))
  call zmio_put1_r8('qin_den', (4._r8*3.14159_r8*(25.e-6_r8**3)*500._r8))
  call zmio_put1_r8('ke', real(1.0E-6, r8))
  call put_model_constants()
  ! wv_saturation's estbl, read back through the public estblf at the
  ! table's own nodes (tmin + n - 1, n = 1..203): a function fixture of the
  ! saturation lookup aqsat uses.
  table = -99999.0_r8
  do n = 1, 203
     table(n) = estblf(173.16_r8 + real(n - 1, r8))
  end do
  call zmio_put_r8('estblf_probe', table, 203)

  cudtacttime = 0.
  do step = 1, nsteps
     itimestep = itime0 + step - 1
     curr_secs = real(itimestep - 1) * dt
     write(tag, '(a,i0,a)') 's', step, '/'
     zmio_prefix = trim(tag)//'in/'
     call zmio_put1_i4('itimestep', itimestep)
     ! camzm_driver's own run decision (module_cu_camzm_driver.F:369-399,
     ! non-adaptive arm), restated so a fixture says which steps computed;
     ! on a held step the driver returns before touching any output.
     if (itimestep == 1 .or. cudt == 0. .or. stepcu == 1 .or. &
         mod(itimestep, stepcu) == 0) then
        call zmio_put1_i4('ran', 1)
     else
        call zmio_put1_i4('ran', 0)
     end if
     call put3('th', th, nk); call put3('t', t, nk); call put3('p', p, nk)
     call put3('pi', pi, nk); call put3('qv', qv, nk); call put3('qc', qc, nk)
     call put3('qi', qi, nk); call put3('z', z, nk); call put3('dz8w', dz8w, nk)
     call put3('u', u, nk); call put3('v', v, nk); call put3('cldfra', cldfra, nk)
     call put3('p8w', p8w, nk + 1); call put3('z_at_w', z_at_w, nk + 1)
     call put3('tke_pbl', tke_pbl, nk + 1)
     call put2('tsk', tsk); call put2('ust', ust); call put2('mavail', mavail)
     call put2('pblh', pblh); call put2('xland', xland); call put2('ht', ht)
     call put2('psfc', psfc); call put2('hfx', hfx); call put2('qfx', qfx)
     call put2('tpert', tpert)
     call zmio_put_i4('kpbl', kpbl(1:ncol,1), ncol)

     xland_call = xland
     if (swapx) then
        where (xland == 2.) xland_call = 1.
        where (xland == 1.) xland_call = 2.
     end if
     call camzm_driver( &
          ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, &
          its,ite, jts,jte, kts,kte, &
          itimestep, blpbl, sfclay, &
          th, t, tsk, tke_pbl, ust, qv, qc, qi, &
          mavail, kpbl, pblh, xland_call, z, &
          z_at_w, dz8w, ht, p, p8w, pi, psfc, &
          u, v, hfx, qfx, cldfra, cldfra_mp_all, &
          .false., tpert, &
          dx, dt, stepcu, cudt, curr_secs, .false., &
          cudtacttime, &
          cape_out, mu_out, md_out, zmdt, zmdq, &
          rliq_out, dlf_out, &
          pconvb, pconvt, cubot, cutop, raincv, pratec, &
          rucuten, rvcuten, &
          rthcuten, rqvcuten, rqccuten, rqicuten, &
          rqcncuten, rqincuten, &
          evaptzm, fzsntzm, evsntzm, evapqzm, zmflxprc, &
          zmflxsnw, zmntprpd, zmntsnpd, zmeiheat, &
          cmfmc, cmfmcdzm, preccdzm, precz, &
          zmmtu, zmmtv, zmupgu, zmupgd, zmvpgu, zmvpgd, &
          zmicuu, zmicud, zmicvu, zmicvd, &
          zmdice, zmdliq, &
          evapcdp3d, icwmrdp3d, rprddp3d, &
          dp3d, du3d, ed3d, eu3d, md3d, mu3d, dsubcld2d, &
          ideep2d, jt2d, maxg2d, lengath2d)

     do i = 1, ncol
        if (lengath2d(i,1) == 0) then
           jt2d(i,1) = -1; ideep2d(i,1) = -1
        end if
     end do

     zmio_prefix = trim(tag)//'out/'
     call put3('rthcuten', rthcuten, nk); call put3('rqvcuten', rqvcuten, nk)
     call put3('rqccuten', rqccuten, nk); call put3('rqicuten', rqicuten, nk)
     call put3('rqcncuten', rqcncuten, nk); call put3('rqincuten', rqincuten, nk)
     call put3('rucuten', rucuten, nk); call put3('rvcuten', rvcuten, nk)
     call put2('raincv', raincv); call put2('pratec', pratec)
     call put2('cape', cape_out); call put2('rliq', rliq_out)
     call put2('pconvb', pconvb); call put2('pconvt', pconvt)
     call put2('cubot', cubot); call put2('cutop', cutop)
     call put2('preccdzm', preccdzm); call put2('precz', precz)
     call put2('dsubcld2d', dsubcld2d)
     call put3('zmmu', mu_out, nk); call put3('zmmd', md_out, nk)
     call put3('zmdt', zmdt, nk); call put3('zmdq', zmdq, nk)
     call put3('dlf', dlf_out, nk)
     call put3('evaptzm', evaptzm, nk); call put3('fzsntzm', fzsntzm, nk)
     call put3('evsntzm', evsntzm, nk); call put3('evapqzm', evapqzm, nk)
     call put3('zmflxprc', zmflxprc, nk); call put3('zmflxsnw', zmflxsnw, nk)
     call put3('zmntprpd', zmntprpd, nk); call put3('zmntsnpd', zmntsnpd, nk)
     call put3('zmeiheat', zmeiheat, nk)
     call put3('cmfmc', cmfmc, nk + 1); call put3('cmfmcdzm', cmfmcdzm, nk + 1)
     call put3('zmmtu', zmmtu, nk); call put3('zmmtv', zmmtv, nk)
     call put3('zmupgu', zmupgu, nk); call put3('zmupgd', zmupgd, nk)
     call put3('zmvpgu', zmvpgu, nk); call put3('zmvpgd', zmvpgd, nk)
     call put3('zmicuu', zmicuu, nk); call put3('zmicud', zmicud, nk)
     call put3('zmicvu', zmicvu, nk); call put3('zmicvd', zmicvd, nk)
     call put3('zmdice', zmdice, nk); call put3('zmdliq', zmdliq, nk)
     call put3('evapcdp3d', evapcdp3d, nk); call put3('icwmrdp3d', icwmrdp3d, nk)
     call put3('rprddp3d', rprddp3d, nk)
     call put3('dp3d', dp3d, nk); call put3('du3d', du3d, nk)
     call put3('ed3d', ed3d, nk); call put3('eu3d', eu3d, nk)
     call put3('md3d', md3d, nk); call put3('mu3d', mu3d, nk)
     call zmio_put_i4('ideep2d', ideep2d(1:ncol,1), ncol)
     call zmio_put_i4('jt2d', jt2d(1:ncol,1), ncol)
     call zmio_put_i4('maxg2d', maxg2d(1:ncol,1), ncol)
     call zmio_put_i4('lengath2d', lengath2d(1:ncol,1), ncol)

     ! harness bookkeeping: advance the state with the step's tendencies
     do k = 1, nk
        do i = 1, ncol
           th(i,k,1) = th(i,k,1) + dt*rthcuten(i,k,1)
           t(i,k,1)  = th(i,k,1)*pi(i,k,1)
           qv(i,k,1) = max(qv(i,k,1) + dt*rqvcuten(i,k,1), 0.)
           qc(i,k,1) = max(qc(i,k,1) + dt*rqccuten(i,k,1), 0.)
           qi(i,k,1) = max(qi(i,k,1) + dt*rqicuten(i,k,1), 0.)
           u(i,k,1)  = u(i,k,1) + dt*rucuten(i,k,1)
           v(i,k,1)  = v(i,k,1) + dt*rvcuten(i,k,1)
        end do
     end do
  end do
  call zmio_close()

contains

  subroutine put_model_constants()
    ! module_model_constants words get_tpert reads (default REAL).
    use module_model_constants, only: cp, ep_1, ep_2, g, r_d, rcp, &
         svp1, svp2, svp3, svpt0
    call zmio_put1_r4('mc_cp', cp); call zmio_put1_r4('mc_ep_1', ep_1)
    call zmio_put1_r4('mc_ep_2', ep_2); call zmio_put1_r4('mc_g', g)
    call zmio_put1_r4('mc_r_d', r_d); call zmio_put1_r4('mc_rcp', rcp)
    call zmio_put1_r4('mc_svp1', svp1); call zmio_put1_r4('mc_svp2', svp2)
    call zmio_put1_r4('mc_svp3', svp3); call zmio_put1_r4('mc_svpt0', svpt0)
  end subroutine put_model_constants

  integer function param_first_scalar_value()
    use module_state_description, only: param_first_scalar
    param_first_scalar_value = param_first_scalar
  end function param_first_scalar_value

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

  subroutine read_mass(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    read(u_in) a(1:ncol,1:nk,1)
  end subroutine read_mass

  subroutine read_full(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    read(u_in) a(1:ncol,1:nk+1,1)
  end subroutine read_full

  subroutine read_sfc(a)
    real(4), intent(inout) :: a(ims:ime,jms:jme)
    read(u_in) a(1:ncol,1)
  end subroutine read_sfc

  subroutine put3(name, a, nlev)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,kms:kme,jms:jme)
    integer, intent(in) :: nlev
    real(4), allocatable :: buf(:)
    integer :: ii, kk
    allocate(buf(ncol*nlev))
    do kk = 1, nlev
       do ii = 1, ncol
          buf((kk-1)*ncol + ii) = a(ii,kk,1)
       end do
    end do
    call zmio_put_r4(name, buf, ncol*nlev)
  end subroutine put3

  subroutine put2(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,jms:jme)
    call zmio_put_r4(name, a(1:ncol,1), ncol)
  end subroutine put2

end program run_camzm
