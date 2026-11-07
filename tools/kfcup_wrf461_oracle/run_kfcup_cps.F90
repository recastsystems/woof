program run_kfcup_cps
  ! End-to-end capture of WRF v4.6.1 KF_cup_CPS (cu_physics = 10), driven
  ! through the same argument list module_cumulus_driver.F:1515-1574 hands
  ! it, over every fixture column, under five namelist configurations.
  !
  ! The whole fixture is ONE tile (ims:ime = 1:ncase, one j row), exactly as
  ! WRF calls the scheme.  The domain is wider than 3 points so the
  ! single-column diagnostics branch (module_cu_kfcup.F:433-436, :480-485)
  ! stays off; that branch also changes the KTAU gate and writes to unit 98.
  !
  ! LNTERMS IS NOT CAPTURED, because WRF does not define it.  KF_cup_CPS
  ! passes the non-contiguous section lnterms(i,:,j) to cupCloudFraction's
  ! INTENT(OUT) explicit-shape dummy (module_cu_kfcup.F:1075, :4339), so
  ! gfortran hands the callee a temporary and copies it back; the callee
  ! writes it only for shallow columns (:4379, all zeros), and for every
  ! other column the copy-back is uninitialised stack.  MEASURED: WRF's own
  ! -O2 object returns different garbage per tile layout while the -O0
  ! recompilation returns the caller's words -- 10165 differing words on
  ! the first capture, and the only field that differed.  Where WRF does
  ! define it, it is identically zero.
  !
  ! Column independence is measured, not assumed: config 1 is rerun with
  ! every case alone in a four-wide tile of copies of itself, and the
  ! differing-word count is written to kfcup-isolation.csv.
  use iso_fortran_env, only: int32
  use module_model_constants, only: xlv0, xlv1, xls0, xls1, cp, r_d, g, ep_1, &
       ep_2, svp1, svp2, svp3, svpt0
  use module_cu_kfcup, only: kf_cup_cps, kf_cup_init
  use kfcup_fixture
  implicit none

  type kcfg
     character(len=24) :: name
     integer :: numbins
     real :: thbin, rbin, mindeep, minshallow, dt, dx
     integer :: stepcu, ktau, sfclay
     logical :: warm_rain, f_qr, f_qi, f_qs
  end type kcfg

  integer, parameter :: ncfg = 5
  type(kcfg) :: cfgs(ncfg)
  integer :: ic, n, ucfg, uinl, uins, uoutl, uouts, uiso
  integer, parameter :: nlev_out = 21, nsca_out = 21

  call build_table()
  ! KF_LUTAB through the scheme's own initialiser, as module_physics_init.F
  ! does for KFCUPSCHEME; restart=.true. leaves the dummy arrays alone and
  ! runs only the table build (module_cu_kfcup.F:4104-4158).
  call init_tables()
  cfgs(1) = kcfg('recommended',      21, 0.1,  1.0e-4, 0.333, 1.0e-2,  60., 12000.,  5,  1,  1, .false., .true.,  .true.,  .true.)
  cfgs(2) = kcfg('readme_th001',     21, 0.01, 1.0e-4, 0.333, 1.0e-2,  20.,  4000., 15, 15, 91, .false., .true.,  .false., .false.)
  cfgs(3) = kcfg('registry_default',  1, 1.0,  1.0,    1.0,   1.0,    120., 27000.,  3,  1,  3, .true.,  .true.,  .false., .false.)
  cfgs(4) = kcfg('deep_friendly',    11, 0.25, 2.0e-4, 0.1,   1.0e-3,  60., 12000.,  5, 10,  2, .false., .false., .false., .true.)
  cfgs(5) = kcfg('fine_pdf',         41, 0.05, 5.0e-5, 0.5,   1.0e-2,  45.,  9000.,  6, 12,  1, .false., .true.,  .true.,  .true.)

  open(newunit=ucfg, file='kfcup-config.csv', status='replace')
  write(ucfg,'(a)') 'cfg,name,numbins,thbinsize,rbinsize,mindeepfreq,minshallowfreq,dt,dx,stepcu,ktau,sf_sfclay_physics,warm_rain,f_qr,f_qi,f_qs'
  do ic = 1, ncfg
     write(ucfg,'(i0,",",a,",",i0,6(",",a),3(",",i0),4(",",l1))') ic, trim(cfgs(ic)%name), cfgs(ic)%numbins, &
          hx(cfgs(ic)%thbin), hx(cfgs(ic)%rbin), hx(cfgs(ic)%mindeep), hx(cfgs(ic)%minshallow), &
          hx(cfgs(ic)%dt), hx(cfgs(ic)%dx), cfgs(ic)%stepcu, cfgs(ic)%ktau, cfgs(ic)%sfclay, &
          cfgs(ic)%warm_rain, cfgs(ic)%f_qr, cfgs(ic)%f_qi, cfgs(ic)%f_qs
  end do
  close(ucfg)

  open(newunit=uinl, file='kfcup-in-levels.csv', status='replace')
  write(uinl,'(a)') 'cfg,case,k,u,v,th,t,w,dz8w,p,pi,rho,qv,z,z_at_w,w0avg,cldfra_cup,rthcuten,rqvcuten'
  open(newunit=uins, file='kfcup-in-surface.csv', status='replace')
  write(uins,'(a)') 'cfg,case,name,psfc,ht,tsk,hfx,qfx,xland,mavail,br,regime,pblh,kpbl,t2,q2,nca,raincv,shall,cupflag'
  open(newunit=uoutl, file='kfcup-levels.csv', status='replace')
  write(uoutl,'(a)') 'cfg,case,k,rthcuten,rqvcuten,rqccuten,rqrcuten,rqicuten,rqscuten,w0avg,cldfra_cup,'// &
       'cldfratend_cup,wup_cup,qc_ic_cup,qndrop_ic_cup,qc_iu_cup,fcvt_qc_to_pr_cup,fcvt_qc_to_qi_cup,'// &
       'fcvt_qi_to_pr_cup,mfup_cup,mfup_ent_cup,mfdn_cup,mfdn_ent_cup,updfra_cup'
  open(newunit=uouts, file='kfcup-surface.csv', status='replace')
  write(uouts,'(a)') 'cfg,case,raincv,nca,cutop,cubot,shall,cupflag,cu_act_flag,slopesfc,slopeez,sigmasfc,'// &
       'sigmaez,taucloud,tactive,tstar,lnint,activefrac,wcloudbase,wact_cup,wulcl_cup,tcloud_cup'
  open(newunit=uiso, file='kfcup-isolation.csv', status='replace')
  write(uiso,'(a)') 'cfg,case,differing_words'

  do ic = 1, ncfg
     call run_config(ic, .true., 0)
  end do
  do n = 1, ncase
     call run_config(1, .false., n)
  end do
  close(uinl); close(uins); close(uoutl); close(uouts); close(uiso)

contains

  subroutine init_tables()
    real :: a3(1,1,1), b2(1,1)
    logical :: l2(1,1)
    a3 = 0.; b2 = 0.; l2 = .false.
    call kf_cup_init(a3, a3, a3, a3, a3, a3, b2, a3, 1, 1, svp1, svp2, svp3, svpt0, &
         l2, a3, a3, b2, b2, 1, .true., .true., &
         1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  end subroutine init_tables

  subroutine run_config(ic, packed, alone)
    integer, intent(in) :: ic, alone
    logical, intent(in) :: packed
    integer :: nc, i, k, kk, ndiff, src
    integer :: ims, ime, kms, kme
    real, allocatable, dimension(:,:,:) :: u3, v3, th3, t3, w3, dz3, p3, pi3, rho3, qv3, z3, zw3
    real, allocatable, dimension(:,:,:) :: w0avg, cldfra_cup, cldfratend_cup, lnterms, wup_cup, &
         qc_ic_cup, qndrop_ic_cup, qc_iu_cup, fpr, fqi, fipr, mfup, mfupe, mfdn, mfdne, updfra, &
         rth, rqv, rqc, rqr, rqi, rqs
    real, allocatable, dimension(:,:) :: psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, t2, q2, &
         raincv, nca, shall, cubot, cutop, slopesfc, slopeez, sigmasfc, sigmaez, taucloud, tactive, &
         tstar, lnint, activefrac, wcloudbase, wact_cup, wulcl_cup, tcloud_cup
    integer, allocatable :: kpbl(:,:)
    logical, allocatable :: cu_act_flag(:,:), cupflag(:,:)
    real :: col_u(nz), col_v(nz), col_th(nz), col_t(nz), col_w(nz+1), col_dz(nz), col_p(nz), col_pi(nz)
    real :: col_rho(nz), col_qv(nz), col_z(nz), col_zw(nz+1), col_w0(nz)
    real :: s_psfc, s_ht, s_tsk, s_hfx, s_qfx, s_xland, s_mavail, s_br, s_regime, s_pblh, s_t2, s_q2
    integer :: s_kpbl
    type(kcfg) :: c
    real :: dt
    integer, save :: ref_lev_words(ncase_max, nz+1, 21)
    integer, save :: ref_sca_words(ncase_max, 21)
    logical, save :: have_ref = .false.
    integer :: lw(nz+1, 21), sw(21)

    c = cfgs(ic)
    dt = c%dt
    if (packed) then
       nc = ncase
    else
       nc = 4
    end if
    ims = 1; ime = nc; kms = 1; kme = nz + 1
    allocate(u3(ims:ime,kms:kme,1:1))
    allocate(v3(ims:ime,kms:kme,1:1))
    allocate(th3(ims:ime,kms:kme,1:1))
    allocate(t3(ims:ime,kms:kme,1:1))
    allocate(w3(ims:ime,kms:kme,1:1))
    allocate(dz3(ims:ime,kms:kme,1:1))
    allocate(p3(ims:ime,kms:kme,1:1))
    allocate(pi3(ims:ime,kms:kme,1:1))
    allocate(rho3(ims:ime,kms:kme,1:1))
    allocate(qv3(ims:ime,kms:kme,1:1))
    allocate(z3(ims:ime,kms:kme,1:1))
    allocate(zw3(ims:ime,kms:kme,1:1))
    allocate(w0avg(ims:ime,kms:kme,1:1))
    allocate(cldfra_cup(ims:ime,kms:kme,1:1))
    allocate(cldfratend_cup(ims:ime,kms:kme,1:1))
    allocate(lnterms(ims:ime,kms:kme,1:1))
    allocate(wup_cup(ims:ime,kms:kme,1:1))
    allocate(qc_ic_cup(ims:ime,kms:kme,1:1))
    allocate(qndrop_ic_cup(ims:ime,kms:kme,1:1))
    allocate(qc_iu_cup(ims:ime,kms:kme,1:1))
    allocate(fpr(ims:ime,kms:kme,1:1))
    allocate(fqi(ims:ime,kms:kme,1:1))
    allocate(fipr(ims:ime,kms:kme,1:1))
    allocate(mfup(ims:ime,kms:kme,1:1))
    allocate(mfupe(ims:ime,kms:kme,1:1))
    allocate(mfdn(ims:ime,kms:kme,1:1))
    allocate(mfdne(ims:ime,kms:kme,1:1))
    allocate(updfra(ims:ime,kms:kme,1:1))
    allocate(rth(ims:ime,kms:kme,1:1))
    allocate(rqv(ims:ime,kms:kme,1:1))
    allocate(rqc(ims:ime,kms:kme,1:1))
    allocate(rqr(ims:ime,kms:kme,1:1))
    allocate(rqi(ims:ime,kms:kme,1:1))
    allocate(rqs(ims:ime,kms:kme,1:1))
    allocate(psfc(ims:ime,1:1))
    allocate(ht(ims:ime,1:1))
    allocate(tsk(ims:ime,1:1))
    allocate(hfx(ims:ime,1:1))
    allocate(qfx(ims:ime,1:1))
    allocate(xland(ims:ime,1:1))
    allocate(mavail(ims:ime,1:1))
    allocate(br(ims:ime,1:1))
    allocate(regime(ims:ime,1:1))
    allocate(pblh(ims:ime,1:1))
    allocate(t2(ims:ime,1:1))
    allocate(q2(ims:ime,1:1))
    allocate(raincv(ims:ime,1:1))
    allocate(nca(ims:ime,1:1))
    allocate(shall(ims:ime,1:1))
    allocate(cubot(ims:ime,1:1))
    allocate(cutop(ims:ime,1:1))
    allocate(slopesfc(ims:ime,1:1))
    allocate(slopeez(ims:ime,1:1))
    allocate(sigmasfc(ims:ime,1:1))
    allocate(sigmaez(ims:ime,1:1))
    allocate(taucloud(ims:ime,1:1))
    allocate(tactive(ims:ime,1:1))
    allocate(tstar(ims:ime,1:1))
    allocate(lnint(ims:ime,1:1))
    allocate(activefrac(ims:ime,1:1))
    allocate(wcloudbase(ims:ime,1:1))
    allocate(wact_cup(ims:ime,1:1))
    allocate(wulcl_cup(ims:ime,1:1))
    allocate(tcloud_cup(ims:ime,1:1))
    allocate(kpbl(ims:ime,1), cu_act_flag(ims:ime,1), cupflag(ims:ime,1))
    u3 = 0.; v3 = 0.; th3 = 0.; t3 = 0.; w3 = 0.; dz3 = 0.; p3 = 0.; pi3 = 0.; rho3 = 0.; qv3 = 0.
    z3 = 0.; zw3 = 0.

    do i = 1, nc
       if (packed) then
          src = i
       else
          src = alone
       end if
       call build_column(cases(src), dt, col_u, col_v, col_th, col_t, col_w, col_dz, col_p, col_pi, &
            col_rho, col_qv, col_z, col_zw, col_w0, s_psfc, s_ht, s_tsk, s_hfx, s_qfx, s_xland, &
            s_mavail, s_br, s_regime, s_pblh, s_kpbl, s_t2, s_q2)
       u3(i,1:nz,1) = col_u; v3(i,1:nz,1) = col_v; th3(i,1:nz,1) = col_th; t3(i,1:nz,1) = col_t
       w3(i,1:nz+1,1) = col_w; dz3(i,1:nz,1) = col_dz; p3(i,1:nz,1) = col_p; pi3(i,1:nz,1) = col_pi
       rho3(i,1:nz,1) = col_rho; qv3(i,1:nz,1) = col_qv; z3(i,1:nz,1) = col_z; zw3(i,1:nz+1,1) = col_zw
       w0avg(i,:,1) = 0.; w0avg(i,1:nz,1) = col_w0
       psfc(i,1) = s_psfc; ht(i,1) = s_ht; tsk(i,1) = s_tsk; hfx(i,1) = s_hfx; qfx(i,1) = s_qfx
       xland(i,1) = s_xland; mavail(i,1) = s_mavail; br(i,1) = s_br; regime(i,1) = s_regime
       pblh(i,1) = s_pblh; kpbl(i,1) = s_kpbl; t2(i,1) = s_t2; q2(i,1) = s_q2
       ! NCA: the case's own value, or the hold boundary 0.5*DT itself (v = 0)
       ! and the float just below it (v = 1) -- KF_cup_CPS:516 holds on .ge.
       if (cases(src)%nca0 <= -1000.d0) then
          nca(i,1) = 0.5*dt
          if (cases(src)%nca0 < -1000.5d0) nca(i,1) = nearest(0.5*dt, -1.0)
       else
          nca(i,1) = real(cases(src)%nca0)
       end if
    end do
    ! Every INOUT/OUT array starts at a recognisable value, so a column the
    ! scheme does not write (the NCA hold) carries its input word through.
    rth = 1.25e-5; rqv = -2.5e-9; rqc = 3.5e-9; rqr = 4.5e-9; rqi = 5.5e-9; rqs = 6.5e-9
    cldfra_cup = 0.125; cldfratend_cup = 0.0625; lnterms = -7.75; wup_cup = 0.375
    qc_ic_cup = 1.5e-4; qndrop_ic_cup = 2.5e6; qc_iu_cup = 2.5e-4; fpr = 0.25; fqi = 0.5; fipr = 0.75
    mfup = 0.0175; mfupe = 0.0275; mfdn = -0.0125; mfdne = 0.0225; updfra = 0.03125
    raincv = 7.75; shall = 2.; cubot = -3.; cutop = -4.; slopesfc = -5.; slopeez = -6.
    sigmasfc = -7.; sigmaez = -8.; taucloud = -9.; tactive = -10.; tstar = -11.; lnint = -12.
    activefrac = -13.; wcloudbase = -14.; wact_cup = -15.; wulcl_cup = -16.; tcloud_cup = -17.
    cu_act_flag = .false.; cupflag = .true.

    if (packed) then
       do i = 1, nc
          ! The level inputs do not depend on the configuration (only NCA,
          ! a surface field, does), so they are written once, under cfg 1.
          if (ic == 1) then
          do k = 1, nz + 1
             write(uinl,'(i0,",",i0,",",i0,16(",",a))') ic, i, k, hx(u3(i,k,1)), hx(v3(i,k,1)), &
                  hx(th3(i,k,1)), hx(t3(i,k,1)), hx(w3(i,k,1)), hx(dz3(i,k,1)), hx(p3(i,k,1)), &
                  hx(pi3(i,k,1)), hx(rho3(i,k,1)), hx(qv3(i,k,1)), hx(z3(i,k,1)), hx(zw3(i,k,1)), &
                  hx(w0avg(i,k,1)), hx(cldfra_cup(i,k,1)), hx(rth(i,k,1)), hx(rqv(i,k,1))
          end do
          end if
          write(uins,'(i0,",",i0,",",a,10(",",a),",",i0,5(",",a),",",l1)') ic, i, trim(cases(i)%name), &
               hx(psfc(i,1)), hx(ht(i,1)), hx(tsk(i,1)), hx(hfx(i,1)), hx(qfx(i,1)), hx(xland(i,1)), &
               hx(mavail(i,1)), hx(br(i,1)), hx(regime(i,1)), hx(pblh(i,1)), kpbl(i,1), hx(t2(i,1)), &
               hx(q2(i,1)), hx(nca(i,1)), hx(raincv(i,1)), hx(shall(i,1)), cupflag(i,1)
       end do
    end if

    call kf_cup_cps(grid_id=1, &
         ids=1, ide=nc+1, jds=1, jde=2, kds=1, kde=nz+1, &
         ims=ims, ime=ime, jms=1, jme=1, kms=kms, kme=kme, &
         its=1, ite=nc, jts=1, jte=1, kts=1, kte=nz, &
         dt=dt, ktau=c%ktau, dx=c%dx, &
         rho=rho3, raincv=raincv, nca=nca, &
         u=u3, v=v3, th=th3, t=t3, w=w3, dz8w=dz3, pcps=p3, pi=pi3, &
         w0avg=w0avg, xlv0=xlv0, xlv1=xlv1, xls0=xls0, xls1=xls1, cp=cp, r=r_d, g=g, ep1=ep_1, &
         ep2=ep_2, svp1=svp1, svp2=svp2, svp3=svp3, svpt0=svpt0, &
         stepcu=c%stepcu, cu_act_flag=cu_act_flag, warm_rain=c%warm_rain, cutop=cutop, cubot=cubot, &
         qv=qv3, xland=xland, &
         psfc=psfc, z=z3, z_at_w=zw3, ht=ht, tsk=tsk, hfx=hfx, qfx=qfx, mavail=mavail, &
         sf_sfclay_physics=c%sfclay, br=br, regime=regime, pblh=pblh, kpbl=kpbl, t2=t2, q2=q2, &
         slopesfc=slopesfc, slopeez=slopeez, sigmasfc=sigmasfc, sigmaez=sigmaez, &
         cupflag=cupflag, cldfra_cup=cldfra_cup, cldfratend_cup=cldfratend_cup, &
         shall=shall, taucloud=taucloud, tactive=tactive, activefrac=activefrac, &
         tstar=tstar, lnterms=lnterms, lnint=lnint, &
         numbins=c%numbins, thbinsize=c%thbin, rbinsize=c%rbin, &
         mindeepfreq=c%mindeep, minshallowfreq=c%minshallow, &
         wcloudbase=wcloudbase, wact_cup=wact_cup, wulcl_cup=wulcl_cup, wup_cup=wup_cup, &
         qc_ic_cup=qc_ic_cup, qndrop_ic_cup=qndrop_ic_cup, qc_iu_cup=qc_iu_cup, &
         fcvt_qc_to_pr_cup=fpr, fcvt_qc_to_qi_cup=fqi, fcvt_qi_to_pr_cup=fipr, &
         mfup_cup=mfup, mfup_ent_cup=mfupe, mfdn_cup=mfdn, mfdn_ent_cup=mfdne, &
         updfra_cup=updfra, tcloud_cup=tcloud_cup, shcu_aerosols_opt=0, &
         rthcuten=rth, rqvcuten=rqv, rqccuten=rqc, rqrcuten=rqr, rqicuten=rqi, rqscuten=rqs, &
         f_qv=.true., f_qc=.true., f_qr=c%f_qr, f_qi=c%f_qi, f_qs=c%f_qs)

    do i = 1, nc
       do k = 1, nz + 1
          lw(k,:) = [ transfer(rth(i,k,1),0_int32), transfer(rqv(i,k,1),0_int32), transfer(rqc(i,k,1),0_int32), &
               transfer(rqr(i,k,1),0_int32), transfer(rqi(i,k,1),0_int32), transfer(rqs(i,k,1),0_int32), &
               transfer(w0avg(i,k,1),0_int32), transfer(cldfra_cup(i,k,1),0_int32), &
               transfer(cldfratend_cup(i,k,1),0_int32), &
               transfer(wup_cup(i,k,1),0_int32), transfer(qc_ic_cup(i,k,1),0_int32), &
               transfer(qndrop_ic_cup(i,k,1),0_int32), transfer(qc_iu_cup(i,k,1),0_int32), &
               transfer(fpr(i,k,1),0_int32), transfer(fqi(i,k,1),0_int32), transfer(fipr(i,k,1),0_int32), &
               transfer(mfup(i,k,1),0_int32), transfer(mfupe(i,k,1),0_int32), transfer(mfdn(i,k,1),0_int32), &
               transfer(mfdne(i,k,1),0_int32), transfer(updfra(i,k,1),0_int32) ]
       end do
       sw = [ transfer(raincv(i,1),0_int32), transfer(nca(i,1),0_int32), transfer(cutop(i,1),0_int32), &
            transfer(cubot(i,1),0_int32), transfer(shall(i,1),0_int32), merge(1,0,cupflag(i,1)), &
            merge(1,0,cu_act_flag(i,1)), transfer(slopesfc(i,1),0_int32), transfer(slopeez(i,1),0_int32), &
            transfer(sigmasfc(i,1),0_int32), transfer(sigmaez(i,1),0_int32), transfer(taucloud(i,1),0_int32), &
            transfer(tactive(i,1),0_int32), transfer(tstar(i,1),0_int32), transfer(lnint(i,1),0_int32), &
            transfer(activefrac(i,1),0_int32), transfer(wcloudbase(i,1),0_int32), &
            transfer(wact_cup(i,1),0_int32), transfer(wulcl_cup(i,1),0_int32), transfer(tcloud_cup(i,1),0_int32), 0 ]
       if (packed) then
          do k = 1, nz + 1
             write(uoutl,'(i0,",",i0,",",i0,21(",",z8.8))') ic, i, k, (lw(k,kk), kk = 1, nlev_out)
          end do
          write(uouts,'(i0,",",i0,5(",",z8.8),2(",",i0),13(",",z8.8))') ic, i, (sw(kk), kk = 1, 20)
          if (ic == 1) then
             ref_lev_words(i,:,:) = lw
             ref_sca_words(i,:) = sw
          end if
       else if (i == 1) then
          ndiff = count(lw /= ref_lev_words(alone,:,:)) + count(sw /= ref_sca_words(alone,:))
          write(uiso,'(i0,",",i0,",",i0)') ic, alone, ndiff
       end if
    end do
    have_ref = .true.
  end subroutine run_config

end program run_kfcup_cps
