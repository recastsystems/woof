module qnse_oracle_io
  ! Self-describing stream records: name(16), rank, dims(rank), kind, data.
  ! kind 0 = real(4), 1 = integer(4).  pack_fixture.py reads them back.
  implicit none
contains
  subroutine put_r2(u, name, a)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :)
    character(len=16) :: tag
    tag = name
    write(u) tag, 2, size(a, 1), size(a, 2), 0, a
  end subroutine put_r2

  subroutine put_r1(u, name, a)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    character(len=16) :: tag
    tag = name
    write(u) tag, 1, size(a), 0, a
  end subroutine put_r1

  subroutine put_i1(u, name, a)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    character(len=16) :: tag
    tag = name
    write(u) tag, 1, size(a), 1, a
  end subroutine put_i1
end module qnse_oracle_io


program run_qnse_oracle
  ! Drive the byte-unmodified WRF v4.6.1 phys/module_bl_qnsepbl.F and
  ! phys/module_bl_mfshconvpbl.F the way module_pbl_driver.F's
  ! CASE (QNSEPBLSCHEME) does (:1515-1557): the five BL tendencies and the
  ! u_phytmp/v_phytmp scratch (WTHV_MF, LM_BL89) zeroed first (:1028-1068),
  ! MFSHCONVPBL when mfshconv = 1, then QNSEPBL adding its own tendencies.
  !
  ! One call per column (its = ite = i).  MFSHCONVPBL flattens its whole
  ! tile into one updraft vector and steps the vertical loop while ANY
  ! column of the tile is still rising, so in WRF a finished column's
  ! RV_UP/RC_UP above its top and its DETR_EDKF depend on its neighbours
  ! (COMPUTE_UPDRAFT, module_bl_mfshconvpbl.F:1452-1594, and the unmasked
  ! statements of COMPUTE_ENTR_DETR :969-971).  A one-column tile is the
  ! historical fixture decomposition. The fixture launcher explicitly
  ! uses native_tile=False; forecasts preserve whole-tile COUNT(GTEST).
  !
  ! Four steps per arm.  Between steps the harness advances the
  ! atmosphere by the tendencies (pressure, Exner, density and dz held),
  ! and the scheme's carried state (TKE, RC_MF, CT, THZ0, QZ0, QSFC,
  ! EXCH_H/M) carries exactly as WRF carries it.  Every step's inputs and
  ! outputs are written, so the port can be graded step by step.
  use qnse_oracle_io
  use module_model_constants
  use module_bl_qnsepbl
  use module_bl_mfshconvpbl
  implicit none

  integer :: ncol, nz, nstep, mfsh, stepbl, istep, i, k, uin, uout, hdr(5)
  real :: dt, dtturbl
  character(len=512) :: inpath, outprefix, fname
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: jts, jte, kts, kte

  real, allocatable, dimension(:, :) :: ht, tsk, qsfc, chklowq, thz0, qz0, &
       uz0, vz0, corf, xland, sice, snow, ustar, znt, ct, akhs, akms, lh, &
       hfx, qfx, pblh
  integer, allocatable :: lowlyr(:, :), kpbl(:, :)
  real, allocatable, dimension(:, :, :) :: dz, pmid, pint, th, t, exner, qv, &
       qc, u, v, rho, tke, exch_h, exch_m, el_pbl, rublten, rvblten, &
       rthblten, rqvblten, rqcblten, u_phytmp, v_phytmp, massflux, entr, &
       detr, thl_up, thv_up, rt_up, rv_up, rc_up, u_up, v_up, frac_up, rc_mf

  if (command_argument_count() /= 2) then
    write(0, *) 'usage: run_qnse CASES.bin OUT_PREFIX'
    stop 2
  end if
  call get_command_argument(1, inpath)
  call get_command_argument(2, outprefix)

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old')
  read(uin) hdr
  ncol = hdr(1); nz = hdr(2); nstep = hdr(3); mfsh = hdr(4); stepbl = hdr(5)
  read(uin) dt
  dtturbl = dt * stepbl

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol; jms = 1; jme = 1; kms = 1; kme = nz + 1
  jts = 1; jte = 1; kts = 1; kte = nz

  allocate(ht(ncol, 1), tsk(ncol, 1), qsfc(ncol, 1), chklowq(ncol, 1), &
       thz0(ncol, 1), qz0(ncol, 1), uz0(ncol, 1), vz0(ncol, 1), &
       corf(ncol, 1), xland(ncol, 1), sice(ncol, 1), snow(ncol, 1), &
       ustar(ncol, 1), znt(ncol, 1), ct(ncol, 1), akhs(ncol, 1), &
       akms(ncol, 1), lh(ncol, 1), hfx(ncol, 1), qfx(ncol, 1), pblh(ncol, 1), &
       lowlyr(ncol, 1), kpbl(ncol, 1))
  read(uin) ht, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0, corf, xland, sice, &
       snow, ustar, znt, ct, akhs, akms, lh, hfx, qfx
  lowlyr = 1

  call alloc3(dz); call alloc3(pmid); call alloc3(pint); call alloc3(th)
  call alloc3(t); call alloc3(exner); call alloc3(qv); call alloc3(qc)
  call alloc3(u); call alloc3(v); call alloc3(rho); call alloc3(tke)
  call alloc3(exch_h); call alloc3(exch_m); call alloc3(el_pbl)
  call alloc3(rublten); call alloc3(rvblten); call alloc3(rthblten)
  call alloc3(rqvblten); call alloc3(rqcblten); call alloc3(u_phytmp)
  call alloc3(v_phytmp); call alloc3(massflux); call alloc3(entr)
  call alloc3(detr); call alloc3(thl_up); call alloc3(thv_up)
  call alloc3(rt_up); call alloc3(rv_up); call alloc3(rc_up)
  call alloc3(u_up); call alloc3(v_up); call alloc3(frac_up)
  call alloc3(rc_mf)

  read(uin) dz(:, 1:nz, 1), pmid(:, 1:nz, 1), th(:, 1:nz, 1), &
       t(:, 1:nz, 1), exner(:, 1:nz, 1), qv(:, 1:nz, 1), qc(:, 1:nz, 1), &
       u(:, 1:nz, 1), v(:, 1:nz, 1), rho(:, 1:nz, 1), tke(:, 1:nz, 1)
  read(uin) pint(:, 1:nz + 1, 1)
  close(uin)

  ! Registry cold start: rc_mf and the EDKF diagnostics are zero;
  ! EXCH_H/EXCH_M zero (QNSEPBLINIT zeroes EXCH_H, :1362).
  rc_mf = 0.; exch_h = 0.; exch_m = 0.; pblh = 0.; kpbl = 0

  fname = trim(outprefix) // '-static.bin'
  open(newunit=uout, file=trim(fname), access='stream', form='unformatted', &
       status='replace')
  write(uout) ncol, nz, nstep, mfsh, stepbl, dt
  call put_r1(uout, 'ht', ht(:, 1)); call put_r1(uout, 'tsk', tsk(:, 1))
  call put_r1(uout, 'chklowq', chklowq(:, 1)); call put_r1(uout, 'uz0', uz0(:, 1))
  call put_r1(uout, 'vz0', vz0(:, 1)); call put_r1(uout, 'corf', corf(:, 1))
  call put_r1(uout, 'xland', xland(:, 1)); call put_r1(uout, 'sice', sice(:, 1))
  call put_r1(uout, 'snow', snow(:, 1)); call put_r1(uout, 'ustar', ustar(:, 1))
  call put_r1(uout, 'znt', znt(:, 1)); call put_r1(uout, 'akhs', akhs(:, 1))
  call put_r1(uout, 'akms', akms(:, 1)); call put_r1(uout, 'lh', lh(:, 1))
  call put_r1(uout, 'hfx', hfx(:, 1)); call put_r1(uout, 'qfx', qfx(:, 1))
  call put_r2(uout, 'dz', dz(:, 1:nz, 1)); call put_r2(uout, 'pmid', pmid(:, 1:nz, 1))
  call put_r2(uout, 'pint', pint(:, 1:nz + 1, 1))
  call put_r2(uout, 'exner', exner(:, 1:nz, 1)); call put_r2(uout, 'rho', rho(:, 1:nz, 1))
  close(uout)

  do istep = 1, nstep
    write(fname, '(A,A,I0,A)') trim(outprefix), '-s', istep, '.bin'
    open(newunit=uout, file=trim(fname), access='stream', &
         form='unformatted', status='replace')
    ! ---- this step's inputs (atmosphere and carried state) ----------------
    call put_r2(uout, 'in_th', th(:, 1:nz, 1)); call put_r2(uout, 'in_t', t(:, 1:nz, 1))
    call put_r2(uout, 'in_qv', qv(:, 1:nz, 1)); call put_r2(uout, 'in_qc', qc(:, 1:nz, 1))
    call put_r2(uout, 'in_u', u(:, 1:nz, 1)); call put_r2(uout, 'in_v', v(:, 1:nz, 1))
    call put_r2(uout, 'in_tke', tke(:, 1:nz, 1))
    call put_r2(uout, 'in_rc_mf', rc_mf(:, 1:nz, 1))
    call put_r2(uout, 'in_exch_h', exch_h(:, 1:nz, 1))
    call put_r2(uout, 'in_exch_m', exch_m(:, 1:nz, 1))
    call put_r1(uout, 'in_ct', ct(:, 1)); call put_r1(uout, 'in_thz0', thz0(:, 1))
    call put_r1(uout, 'in_qz0', qz0(:, 1)); call put_r1(uout, 'in_qsfc', qsfc(:, 1))

    ! ---- module_pbl_driver.F:1028-1068: zero the tendencies and scratch ---
    rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.; rqcblten = 0.
    u_phytmp = 0.; v_phytmp = 0.

    do i = 1, ncol
      if (mfsh == 1) then
        call mfshconvpbl(DT=dt, STEPBL=stepbl, HT=ht, DZ=dz,             &
             RHO=rho, PMID=pmid, PINT=pint, TH=th, EXNER=exner,          &
             QV=qv, QC=qc, U=u, V=v, HFX=hfx, QFX=qfx, TKE=tke,          &
             RUBLTEN=rublten, RVBLTEN=rvblten, RTHBLTEN=rthblten,        &
             RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,                       &
             IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde,       &
             IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme,       &
             ITS=i, ITE=i, JTS=jts, JTE=jte, KTS=kts, KTE=kte,           &
             KRR=2, MASSFLUX_EDKF=massflux, ENTR_EDKF=entr,              &
             DETR_EDKF=detr, THL_UP=thl_up, THV_UP=thv_up, RT_UP=rt_up,  &
             RV_UP=rv_up, RC_UP=rc_up, U_UP=u_up, V_UP=v_up,             &
             FRAC_UP=frac_up, RC_MF=rc_mf, WTHV=u_phytmp,                &
             PLM_BL89=v_phytmp)
      end if
      call qnsepbl(DT=dt, STEPBL=stepbl, HT=ht, DZ=dz,                   &
           PMID=pmid, PINT=pint, TH=th, T=t, EXNER=exner,                &
           QV=qv, CWM=qc, U=u, V=v, RHO=rho,                             &
           TSK=tsk, QSFC=qsfc, CHKLOWQ=chklowq, THZ0=thz0,               &
           QZ0=qz0, UZ0=uz0, VZ0=vz0, CORF=corf, LOWLYR=lowlyr,          &
           XLAND=xland, SICE=sice, SNOW=snow,                            &
           TKE=tke, EXCH_H=exch_h, EXCH_M=exch_m, USTAR=ustar, ZNT=znt,  &
           EL_MYJ=el_pbl, PBLH=pblh, KPBL=kpbl, CT=ct,                   &
           AKHS=akhs, AKMS=akms, ELFLX=lh, HFX=hfx,                      &
           WTHV_MF=u_phytmp, LM_BL89=v_phytmp,                           &
           RUBLTEN=rublten, RVBLTEN=rvblten, RTHBLTEN=rthblten,          &
           RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,                         &
           IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde,         &
           IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme,         &
           ITS=i, ITE=i, JTS=jts, JTE=jte, KTS=kts, KTE=kte)
    end do

    ! ---- this step's outputs ----------------------------------------------
    call put_r2(uout, 'rublten', rublten(:, 1:nz, 1))
    call put_r2(uout, 'rvblten', rvblten(:, 1:nz, 1))
    call put_r2(uout, 'rthblten', rthblten(:, 1:nz, 1))
    call put_r2(uout, 'rqvblten', rqvblten(:, 1:nz, 1))
    call put_r2(uout, 'rqcblten', rqcblten(:, 1:nz, 1))
    call put_r2(uout, 'tke', tke(:, 1:nz, 1))
    call put_r2(uout, 'el_pbl', el_pbl(:, 1:nz, 1))
    call put_r2(uout, 'exch_h', exch_h(:, 1:nz, 1))
    call put_r2(uout, 'exch_m', exch_m(:, 1:nz, 1))
    call put_r1(uout, 'pblh', pblh(:, 1)); call put_i1(uout, 'kpbl', kpbl(:, 1))
    call put_r1(uout, 'ct', ct(:, 1)); call put_r1(uout, 'thz0', thz0(:, 1))
    call put_r1(uout, 'qz0', qz0(:, 1)); call put_r1(uout, 'qsfc', qsfc(:, 1))
    call put_r1(uout, 'akhs', akhs(:, 1)); call put_r1(uout, 'akms', akms(:, 1))
    call put_r1(uout, 'ustar', ustar(:, 1)); call put_r1(uout, 'uz0', uz0(:, 1))
    call put_r1(uout, 'vz0', vz0(:, 1))
    if (mfsh == 1) then
      call put_r2(uout, 'massflux_edkf', massflux(:, 1:nz, 1))
      call put_r2(uout, 'entr_edkf', entr(:, 1:nz, 1))
      call put_r2(uout, 'detr_edkf', detr(:, 1:nz, 1))
      call put_r2(uout, 'thl_up', thl_up(:, 1:nz, 1))
      call put_r2(uout, 'thv_up', thv_up(:, 1:nz, 1))
      call put_r2(uout, 'rt_up', rt_up(:, 1:nz, 1))
      call put_r2(uout, 'rv_up', rv_up(:, 1:nz, 1))
      call put_r2(uout, 'rc_up', rc_up(:, 1:nz, 1))
      call put_r2(uout, 'u_up', u_up(:, 1:nz, 1))
      call put_r2(uout, 'v_up', v_up(:, 1:nz, 1))
      call put_r2(uout, 'frac_up', frac_up(:, 1:nz, 1))
      call put_r2(uout, 'rc_mf', rc_mf(:, 1:nz, 1))
      call put_r2(uout, 'wthv_mf', u_phytmp(:, 1:nz, 1))
      call put_r2(uout, 'lm_bl89', v_phytmp(:, 1:nz, 1))
    end if
    close(uout)

    ! ---- advance the atmosphere by the PBL tendencies (harness rule) -------
    do k = 1, nz
      do i = 1, ncol
        th(i, k, 1) = th(i, k, 1) + dtturbl * rthblten(i, k, 1)
        qv(i, k, 1) = max(0., qv(i, k, 1) + dtturbl * rqvblten(i, k, 1))
        qc(i, k, 1) = max(0., qc(i, k, 1) + dtturbl * rqcblten(i, k, 1))
        u(i, k, 1) = u(i, k, 1) + dtturbl * rublten(i, k, 1)
        v(i, k, 1) = v(i, k, 1) + dtturbl * rvblten(i, k, 1)
        t(i, k, 1) = th(i, k, 1) * exner(i, k, 1)
      end do
    end do
  end do

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:, :, :)
    allocate(a(ims:ime, kms:kme, jms:jme))
    a = 0.
  end subroutine alloc3

end program run_qnse_oracle
