! Core column oracle for WRF v4.6.1 shcu_physics = 2: compute_uwshcu_inv
! (and through it compute_uwshcu, the scheme's whole r8 column) called
! alone on binary64 inputs that no float32 field carries, one column per
! call exactly as camuwshcu_driver calls it (mix = pcols = 1, iend = 1,
! ncnst = 5, qsat = fqsatd).  Usage:  run_uwshcu_core CORE.bin OUTSTEM
!
! Initialization is run_uwshcu.F90's (CAM_INIT then init_uwshcu).  Every
! output of compute_uwshcu_inv is written, in its own (CAM, top-first)
! orientation.
program run_uwshcu_core
  use shr_kind_mod, only: r8 => shr_kind_r8
  use module_cam_support, only: pcnst_runtime, pcnst_mp
  use module_cam_esinti, only: esinti
  use physconst, only: mwh2o, cpwv, epsilo, latvap, latice, rh2o, cpair, &
       tmelt, mwdry, gravit, rair, zvir
  use constituents, only: cnst_add, cnst_name, cnst_longname, cnst_cp, &
       cnst_cv, cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
       cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, dcconnam, &
       fixcnam, tendnam, ptendnam, dmetendnam, sflxnam, tottnam
  use module_state_description, only: p_qc, p_qr, p_qi, p_qs, p_qg, &
       param_first_scalar, CAMUWPBLSCHEME
  use uwshcu, only: init_uwshcu, compute_uwshcu_inv
  use wv_saturation, only: fqsatd
  use uwio
  implicit none

  character(len=512) :: casefile, outstem
  integer :: u_in, magic, version, ncol, nk, i, mm, n
  integer :: ixcldliq, ixcldice, ixnumliq, ixnumice
  real(r8) :: dt
  real(r8), allocatable, dimension(:,:) :: p0, z0, dp0, dpdry0, u0, v0, qv0, &
       ql0, qi0, t0, s0, cldfrct, concldfrct, ps0, zs0, tke
  real(r8), allocatable :: tr0(:,:,:), pblh(:), cush(:)
  ! outputs, one column
  real(r8), allocatable, dimension(:,:) :: umf, slflx, qtflx, flxprc1, &
       flxsnow1, qvten, qlten, qiten, sten, uten, vten, qrten, qsten, evapc, &
       cufrc, qcu, qlu, qiu, qcout
  real(r8), allocatable :: trten(:,:,:)
  real(r8) :: precip(1), snow(1), cbmf(1), rliq(1), cnt(1), cnb(1), cush1(1), &
       pblh1(1)
  ! all columns, for the fixture
  real(r8), allocatable, dimension(:,:) :: a_umf, a_slflx, a_qtflx, &
       a_flxprc1, a_flxsnow1, a_qvten, a_qlten, a_qiten, a_sten, a_uten, &
       a_vten, a_qrten, a_qsten, a_evapc, a_cufrc, a_qcu, a_qlu, a_qiu, a_qc, &
       a_trten
  real(r8), allocatable, dimension(:) :: a_precip, a_snow, a_cbmf, a_rliq, &
       a_cnt, a_cnb, a_cush

  call get_command_argument(1, casefile)
  call get_command_argument(2, outstem)
  open(newunit=u_in, file=trim(casefile), access='stream', form='unformatted', &
       status='old', action='read')
  read(u_in) magic, version, ncol, nk
  if (magic /= 1431786296 .or. version /= 1) error stop 'bad core case file'  ! 'UWS8'

  allocate(p0(nk,ncol), z0(nk,ncol), dp0(nk,ncol), dpdry0(nk,ncol), &
       u0(nk,ncol), v0(nk,ncol), qv0(nk,ncol), ql0(nk,ncol), qi0(nk,ncol), &
       t0(nk,ncol), s0(nk,ncol), cldfrct(nk,ncol), concldfrct(nk,ncol), &
       ps0(nk+1,ncol), zs0(nk+1,ncol), tke(nk+1,ncol), tr0(nk,5,ncol), &
       pblh(ncol), cush(ncol))
  read(u_in) p0; read(u_in) z0; read(u_in) dp0; read(u_in) dpdry0
  read(u_in) u0; read(u_in) v0; read(u_in) qv0; read(u_in) ql0; read(u_in) qi0
  read(u_in) t0; read(u_in) s0; read(u_in) cldfrct; read(u_in) concldfrct
  read(u_in) ps0; read(u_in) zs0; read(u_in) tke
  read(u_in) tr0
  read(u_in) pblh; read(u_in) cush
  read(u_in) dt
  close(u_in)

  pcnst_runtime = 5
  pcnst_mp = pcnst_runtime
  call esinti(epsilo, latvap, latice, rh2o, cpair, tmelt)
  allocate(cnst_name(5), cnst_longname(5), cnst_cp(5), cnst_cv(5), cnst_mw(5), &
       cnst_type(5), cnst_rgas(5), qmin(5), qmincg(5), cnst_fixed_ubc(5), &
       apcnst(5), bpcnst(5), hadvnam(5), vadvnam(5), dcconnam(5), fixcnam(5), &
       tendnam(5), ptendnam(5), dmetendnam(5), sflxnam(5), tottnam(5))
  cnst_fixed_ubc(:) = .false.
  call cnst_add('Q', mwh2o, cpwv, 1.E-12_r8, mm, longname='Specific humidity', readiv=.true.)
  call cnst_add('CLDLIQ', mwdry, cpair, 0._r8, ixcldliq, longname='Grid box averaged cloud liquid amount')
  call cnst_add('CLDICE', mwdry, cpair, 0._r8, ixcldice, longname='Grid box averaged cloud ice amount')
  call cnst_add('NUMLIQ', mwdry, cpair, 0._r8, ixnumliq, longname='Grid box averaged cloud liquid number')
  call cnst_add('NUMICE', mwdry, cpair, 0._r8, ixnumice, longname='Grid box averaged cloud ice number')
  ! init_uwshcu needs WRF-dimensioned tendency arrays only to zero them;
  ! a 1x1x1 tile with kte = nk + 1 sets pver = nk exactly as the driver run.
  call init_core()

  allocate(umf(1,nk+1), slflx(1,nk+1), qtflx(1,nk+1), flxprc1(1,nk+1), &
       flxsnow1(1,nk+1), qvten(1,nk), qlten(1,nk), qiten(1,nk), sten(1,nk), &
       uten(1,nk), vten(1,nk), qrten(1,nk), qsten(1,nk), evapc(1,nk), &
       cufrc(1,nk), qcu(1,nk), qlu(1,nk), qiu(1,nk), qcout(1,nk), trten(1,nk,5))
  allocate(a_umf(nk+1,ncol), a_slflx(nk+1,ncol), a_qtflx(nk+1,ncol), &
       a_flxprc1(nk+1,ncol), a_flxsnow1(nk+1,ncol), a_qvten(nk,ncol), &
       a_qlten(nk,ncol), a_qiten(nk,ncol), a_sten(nk,ncol), a_uten(nk,ncol), &
       a_vten(nk,ncol), a_qrten(nk,ncol), a_qsten(nk,ncol), a_evapc(nk,ncol), &
       a_cufrc(nk,ncol), a_qcu(nk,ncol), a_qlu(nk,ncol), a_qiu(nk,ncol), &
       a_qc(nk,ncol), a_trten(nk*5,ncol), a_precip(ncol), a_snow(ncol), &
       a_cbmf(ncol), a_rliq(ncol), a_cnt(ncol), a_cnb(ncol), a_cush(ncol))

  do i = 1, ncol
     cush1(1) = cush(i)
     pblh1(1) = pblh(i)
     call compute_uwshcu_inv(1, nk, 1, 5, dt, &
          reshape(ps0(:,i), (/1,nk+1/)), reshape(zs0(:,i), (/1,nk+1/)), &
          reshape(p0(:,i), (/1,nk/)), reshape(z0(:,i), (/1,nk/)), &
          reshape(dp0(:,i), (/1,nk/)), &
          reshape(u0(:,i), (/1,nk/)), reshape(v0(:,i), (/1,nk/)), &
          reshape(qv0(:,i), (/1,nk/)), reshape(ql0(:,i), (/1,nk/)), &
          reshape(qi0(:,i), (/1,nk/)), &
          reshape(t0(:,i), (/1,nk/)), reshape(s0(:,i), (/1,nk/)), &
          reshape(tr0(:,:,i), (/1,nk,5/)), &
          reshape(tke(:,i), (/1,nk+1/)), reshape(cldfrct(:,i), (/1,nk/)), &
          reshape(concldfrct(:,i), (/1,nk/)), pblh1, cush1, &
          umf, slflx, qtflx, flxprc1, flxsnow1, &
          qvten, qlten, qiten, sten, uten, vten, trten, &
          qrten, qsten, precip, snow, evapc, &
          cufrc, qcu, qlu, qiu, cbmf, qcout, rliq, &
          cnt, cnb, fqsatd, i, reshape(dpdry0(:,i), (/1,nk/)))
     a_umf(:,i) = umf(1,:); a_slflx(:,i) = slflx(1,:); a_qtflx(:,i) = qtflx(1,:)
     a_flxprc1(:,i) = flxprc1(1,:); a_flxsnow1(:,i) = flxsnow1(1,:)
     a_qvten(:,i) = qvten(1,:); a_qlten(:,i) = qlten(1,:); a_qiten(:,i) = qiten(1,:)
     a_sten(:,i) = sten(1,:); a_uten(:,i) = uten(1,:); a_vten(:,i) = vten(1,:)
     a_qrten(:,i) = qrten(1,:); a_qsten(:,i) = qsten(1,:); a_evapc(:,i) = evapc(1,:)
     a_cufrc(:,i) = cufrc(1,:); a_qcu(:,i) = qcu(1,:); a_qlu(:,i) = qlu(1,:)
     a_qiu(:,i) = qiu(1,:); a_qc(:,i) = qcout(1,:)
     do n = 1, 5
        a_trten((n-1)*nk+1:n*nk,i) = trten(1,:,n)
     end do
     a_precip(i) = precip(1); a_snow(i) = snow(1); a_cbmf(i) = cbmf(1)
     a_rliq(i) = rliq(1); a_cnt(i) = cnt(1); a_cnb(i) = cnb(1); a_cush(i) = cush1(1)
  end do

  call uwio_open(trim(outstem))
  uwio_prefix = 'in/'
  call uwio_put1_i4('ncol', ncol); call uwio_put1_i4('nk', nk)
  call uwio_put1_r8('dt', dt)
  call uwio_put_r8('p0', p0, nk*ncol); call uwio_put_r8('z0', z0, nk*ncol)
  call uwio_put_r8('dp0', dp0, nk*ncol); call uwio_put_r8('dpdry0', dpdry0, nk*ncol)
  call uwio_put_r8('u0', u0, nk*ncol); call uwio_put_r8('v0', v0, nk*ncol)
  call uwio_put_r8('qv0', qv0, nk*ncol); call uwio_put_r8('ql0', ql0, nk*ncol)
  call uwio_put_r8('qi0', qi0, nk*ncol); call uwio_put_r8('t0', t0, nk*ncol)
  call uwio_put_r8('s0', s0, nk*ncol)
  call uwio_put_r8('ps0', ps0, (nk+1)*ncol); call uwio_put_r8('zs0', zs0, (nk+1)*ncol)
  call uwio_put_r8('tke', tke, (nk+1)*ncol)
  call uwio_put_r8('tr0', tr0, nk*5*ncol)
  call uwio_put_r8('pblh', pblh, ncol); call uwio_put_r8('cush_in', cush, ncol)
  uwio_prefix = 'out/'
  call uwio_put_r8('umf', a_umf, (nk+1)*ncol); call uwio_put_r8('slflx', a_slflx, (nk+1)*ncol)
  call uwio_put_r8('qtflx', a_qtflx, (nk+1)*ncol)
  call uwio_put_r8('flxprc1', a_flxprc1, (nk+1)*ncol)
  call uwio_put_r8('flxsnow1', a_flxsnow1, (nk+1)*ncol)
  call uwio_put_r8('qvten', a_qvten, nk*ncol); call uwio_put_r8('qlten', a_qlten, nk*ncol)
  call uwio_put_r8('qiten', a_qiten, nk*ncol); call uwio_put_r8('sten', a_sten, nk*ncol)
  call uwio_put_r8('uten', a_uten, nk*ncol); call uwio_put_r8('vten', a_vten, nk*ncol)
  call uwio_put_r8('qrten', a_qrten, nk*ncol); call uwio_put_r8('qsten', a_qsten, nk*ncol)
  call uwio_put_r8('evapc', a_evapc, nk*ncol); call uwio_put_r8('cufrc', a_cufrc, nk*ncol)
  call uwio_put_r8('qcu', a_qcu, nk*ncol); call uwio_put_r8('qlu', a_qlu, nk*ncol)
  call uwio_put_r8('qiu', a_qiu, nk*ncol); call uwio_put_r8('qc', a_qc, nk*ncol)
  call uwio_put_r8('trten', a_trten, nk*5*ncol)
  call uwio_put_r8('precip', a_precip, ncol); call uwio_put_r8('snow', a_snow, ncol)
  call uwio_put_r8('cbmf', a_cbmf, ncol); call uwio_put_r8('rliq', a_rliq, ncol)
  call uwio_put_r8('cnt', a_cnt, ncol); call uwio_put_r8('cnb', a_cnb, ncol)
  call uwio_put_r8('cush', a_cush, ncol)
  call uwio_close()

contains

  subroutine init_core()
    real(4), allocatable :: w(:,:,:)
    allocate(w(1:1,1:nk+1,1:1)); w = 0.
    call init_uwshcu(r8, latvap, cpair, latice, zvir, rair, gravit, mwh2o/mwdry, &
         w, w, w, w, w, w, w, w, w, &
         p_qc, p_qr, p_qi, p_qs, p_qg, &
         CAMUWPBLSCHEME, param_first_scalar, .false., &
         1, 2, 1, 2, 1, nk+1, 1, 1, 1, 1, 1, nk+1, &
         1, 1, 1, 1, 1, nk+1)
  end subroutine init_core
end program run_uwshcu_core
