program run_kfcup_stages
  ! Stage captures inside KF_cup_CPS, for a port that has to find where it
  ! went wrong rather than only that it did.
  !
  !   kfcup-slope.csv   cupSlopeSigma: the trigger potential (cupflag), the
  !                     PDF slopes and sigmas and wStar, per case and per
  !                     sf_sfclay_physics (91 = the old MM5 layer, 1 and 3
  !                     take other CASE arms)
  !   kfcup-jfd.csv     cup_jfd: the clamped slopes/sigmas it hands back and
  !                     the normalised joint frequency distribution, per
  !                     case and per PDF configuration
  !   kfcup-para-*.csv  KF_cup_PARA called directly for a fixed set of
  !                     perturbations per case (centre, its four neighbours,
  !                     two off-diagonal corners) plus the standard-KF call
  !                     the driver makes for a stable column
  !
  ! The perturbation subset is the same index set for every case so the
  ! files are rectangular; KF_cup_CPS itself is graded end to end by
  ! run_kfcup_cps over every perturbation the driver visits.
  use iso_fortran_env, only: int32
  use module_model_constants, only: xlv0, xlv1, xls0, xls1, cp, r_d, g, ep_1, &
       ep_2, svp1, svp2, svp3, svpt0
  use module_cu_kfcup, only: kf_cup_init, cupslopesigma, cup_jfd, kf_cup_para
  use kfcup_fixture
  implicit none

  integer, parameter :: npdf = 4
  integer :: pdf_nb(npdf) = [21, 11, 41, 1]
  real :: pdf_th(npdf) = [0.1, 0.25, 0.05, 1.0]
  real :: pdf_r(npdf) = [1.0e-4, 2.0e-4, 5.0e-5, 1.0]
  integer :: sfc_opts(3) = [1, 91, 3]
  integer, parameter :: npert = 8
  integer :: n, is, ip, k, kk, nb, jp
  integer :: uslope, ujfd, upl, ups
  real :: u(nz), v(nz), th(nz), t(nz), w(nz+1), dz(nz), p(nz), pii(nz), rho(nz), qv(nz)
  real :: z(nz), zw(nz+1), w0avg(nz)
  real :: psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, t2, q2
  integer :: kpbl
  real :: slopesfc, slopeez, sigmasfc, sigmaez, wstar, shall
  real :: ss0, se0, gs0, ge0
  logical :: cupflag
  real, allocatable :: thp(:), rp(:), jfd(:,:)
  ! perturbation offsets from the centre bin, (di, dj); the last entry is the
  ! standard-KF call (cupflag false, th_perturb(1), r_perturb(1), freq 0.01)
  integer :: pdi(npert) = [0, -1, 1, 0, 0, -3, 3, 0]
  integer :: pdj(npert) = [0, 0, 0, -1, 1, 3, -3, 0]
  real, parameter :: dt = 60., dx = 12000.
  integer, parameter :: ntst = 5

  call build_table()
  call init_tables()

  open(newunit=uslope, file='kfcup-slope.csv', status='replace')
  write(uslope,'(a)') 'case,sf_sfclay_physics,cupflag,shall,slopesfc,slopeez,sigmasfc,sigmaez,wstar'
  open(newunit=ujfd, file='kfcup-jfd.csv', status='replace')
  write(ujfd,'(a)') 'case,pdf,i,j,slopesfc,slopeez,sigmasfc,sigmaez,th_perturb,r_perturb,jfd'
  open(newunit=upl, file='kfcup-para-levels.csv', status='replace')
  write(upl,'(a)') 'case,pert,k,w0avg,dqdt,dqidt,dqcdt,dqrdt,dqsdt,dtdt,qlg,qig,umfout,uerout,udrout,'// &
       'dmfout,derout,ddrout,wup,qc1d,qi1d,fcvt_qc_to_qi,fcvt_qc_to_pr,fcvt_qi_to_pr,qndrop1d'
  open(newunit=ups, file='kfcup-para-surface.csv', status='replace')
  write(ups,'(a)') 'case,pert,cupflag,th_perturb,r_perturb,freq,ishall,raincv,nca,cutop,cubot,wlcl,'// &
       'updfra,wulcl,wact'

  do n = 1, ncase
     call build_column(cases(n), dt, u, v, th, t, w, dz, p, pii, rho, qv, z, zw, w0avg, &
          psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, kpbl, t2, q2)
     ! ---- cupSlopeSigma ----
     do is = 1, 3
        shall = 7.
        slopesfc = -1.; slopeez = -2.; sigmasfc = -3.; sigmaez = -4.; wstar = -5.
        call cupslopesigma(dx, psfc, p, rho, dz, z, ht, t, th, tsk, u, v, qv, hfx, xland, qfx, &
             mavail, sfc_opts(is), br, regime, pblh, kpbl, t2, q2, slopesfc, slopeez, sigmasfc, &
             sigmaez, wstar, cupflag, shall, 1, nz, 1, nz)
        write(uslope,'(i0,",",i0,",",i0,6(",",a))') n, sfc_opts(is), merge(1,0,cupflag), hx(shall), &
             hx(slopesfc), hx(slopeez), hx(sigmasfc), hx(sigmaez), hx(wstar)
     end do
     ! sf_sfclay_physics = 1 is the arm the stage captures below continue from
     shall = 7.
     call cupslopesigma(dx, psfc, p, rho, dz, z, ht, t, th, tsk, u, v, qv, hfx, xland, qfx, &
          mavail, 1, br, regime, pblh, kpbl, t2, q2, slopesfc, slopeez, sigmasfc, &
          sigmaez, wstar, cupflag, shall, 1, nz, 1, nz)
     ss0 = slopesfc; se0 = slopeez; gs0 = sigmasfc; ge0 = sigmaez
     ! ---- cup_jfd ----
     if (cupflag) then
        do ip = 1, npdf
           ! The 41-bin PDF is 1681 words per column; six columns of it
           ! grade the large-grid indexing without a 10 MB file.
           if (ip == 3 .and. n > 6) cycle
           nb = pdf_nb(ip)
           allocate(thp(nb), rp(nb), jfd(nb, nb))
           slopesfc = ss0; slopeez = se0; sigmasfc = gs0; sigmaez = ge0
           call cup_jfd(slopesfc, slopeez, sigmasfc, sigmaez, nb, pdf_th(ip), pdf_r(ip), thp, rp, jfd)
           do jp = 1, nb
              do k = 1, nb
                 write(ujfd,'(i0,",",i0,",",i0,",",i0,7(",",a))') n, ip, k, jp, hx(slopesfc), &
                      hx(slopeez), hx(sigmasfc), hx(sigmaez), hx(thp(k)), hx(rp(jp)), hx(jfd(k,jp))
              end do
           end do
           deallocate(thp, rp, jfd)
        end do
     end if
     ! ---- KF_cup_PARA, config 1's PDF ----
     nb = pdf_nb(1)
     allocate(thp(nb), rp(nb), jfd(nb, nb))
     slopesfc = ss0; slopeez = se0; sigmasfc = gs0; sigmaez = ge0
     if (cupflag) then
        call cup_jfd(slopesfc, slopeez, sigmasfc, sigmaez, nb, pdf_th(1), pdf_r(1), thp, rp, jfd)
     else
        thp = 0.; rp = 0.; jfd = 0.
        ! The stable-column driver call hands KF_cup_PARA th_perturb(1) and
        ! r_perturb(1) from arrays cup_jfd never filled (KF_cup_CPS:1139);
        ! the standard-KF trigger does not read them (:1780-1787).
     end if
     ! The driver's W0AVG update precedes the column (KF_cup_CPS:439-451).
     call update_w0avg(w, w0avg)
     do jp = 1, npert
        call one_para(n, jp, jp < npert .and. cupflag, nb, thp, rp, jfd)
     end do
     deallocate(thp, rp, jfd)
  end do
  close(uslope); close(ujfd); close(upl); close(ups)

contains

  subroutine init_tables()
    real :: a3(1,1,1), b2(1,1)
    logical :: l2(1,1)
    a3 = 0.; b2 = 0.; l2 = .false.
    call kf_cup_init(a3, a3, a3, a3, a3, a3, b2, a3, 1, 1, svp1, svp2, svp3, svpt0, &
         l2, a3, a3, b2, b2, 1, .true., .true., &
         1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  end subroutine init_tables

  subroutine update_w0avg(w, w0avg)
    real, intent(in) :: w(nz+1)
    real, intent(inout) :: w0avg(nz)
    real :: tst, w0
    integer :: k
    ! KF_cup_CPS:406 and :446-447, operation for operation.
    tst = float(ntst*2)
    do k = 1, nz
       w0 = 0.5*(w(k)+w(k+1))
       w0avg(k) = (w0avg(k)*(tst-1.)+w0)/tst
    end do
  end subroutine update_w0avg

  subroutine one_para(n, jp, cup, nb, thp, rp, jfd)
    integer, intent(in) :: n, jp, nb
    logical, intent(in) :: cup
    real, intent(in) :: thp(nb), rp(nb), jfd(nb, nb)
    integer :: ishall, i0, j0, k
    real :: thpert, rpert, freq, wlcl, updfra, wulcl, wact
    real :: dqdt(nz), dqidt(nz), dqcdt(nz), dqrdt(nz), dqsdt(nz), dtdt(nz), qlg(nz), qig(nz)
    real :: umf(nz), uer(nz), udr(nz), dmf(nz), der(nz), ddr(nz), wup(nz)
    real :: qndrop(nz), qc1d(nz), qi1d(nz), fqi(nz), fpr(nz), fipr(nz), chem1d(nz, 1)
    real :: raincv(1,1), nca(1,1), cutop(1,1), cubot(1,1)

    if (cup) then
       i0 = nb/2 + 1 + pdi(jp); j0 = nb/2 + 1 + pdj(jp)
       thpert = thp(i0); rpert = rp(j0); freq = jfd(i0, j0)
       ! Only perturbations the driver can visit: KF_cup_CPS:700 skips any
       ! bin below minFreq = minShallowFreq*jfd(centre) (config 1: 1e-2).
       ! A skipped bin is not a WRF state -- its freq can underflow to 0,
       ! and the shallow closure then forms 0/0 in FRC2 (:3197).
       if (freq < 1.0e-2*jfd(nb/2+1, nb/2+1)) return
    else
       thpert = 0.; rpert = 0.; freq = 0.01
    end if
    dqdt = -1.; dqidt = -2.; dqcdt = -3.; dqrdt = -4.; dqsdt = -5.; dtdt = -6.
    qlg = -7.; qig = -8.; raincv = -9.; nca = -10.; cutop = -11.; cubot = -12.
    wlcl = -13.; chem1d = 0.
    call kf_cup_para(1, 1, 1, 1, u, v, t, qv, p, dz, w0avg, dt, dx, dx*dx, rho, &
         xlv0, xlv1, xls0, xls1, cp, r_d, g, ep_2, svp1, svp2, svp3, svpt0, &
         pblh, zw(1:nz), cup, thpert, rpert, freq, ishall, qlg, qig, &
         dqdt, dqidt, dqcdt, dqrdt, dqsdt, dtdt, raincv, nca, ntst, &
         .true., .true., .false., cutop, cubot, wlcl, &
         1, 2, 1, 2, 1, nz+1, 1, 1, 1, 1, 1, nz+1, 1, 1, 1, 1, 1, nz, &
         0, updfra, wulcl, wup, umf, uer, udr, dmf, der, ddr, 0, .false., 1, &
         wact, qndrop, qc1d, qi1d, fqi, fpr, fipr, chem1d, 1, 1, 1, 1)
    do k = 1, nz
       write(upl,'(i0,",",i0,",",i0,22(",",a))') n, jp, k, hx(w0avg(k)), hx(dqdt(k)), hx(dqidt(k)), &
            hx(dqcdt(k)), hx(dqrdt(k)), hx(dqsdt(k)), hx(dtdt(k)), hx(qlg(k)), hx(qig(k)), hx(umf(k)), &
            hx(uer(k)), hx(udr(k)), hx(dmf(k)), hx(der(k)), hx(ddr(k)), hx(wup(k)), hx(qc1d(k)), &
            hx(qi1d(k)), hx(fqi(k)), hx(fpr(k)), hx(fipr(k)), hx(qndrop(k))
    end do
    write(ups,'(i0,",",i0,",",i0,3(",",a),",",i0,8(",",a))') n, jp, merge(1,0,cup), hx(thpert), &
         hx(rpert), hx(freq), ishall, hx(raincv(1,1)), hx(nca(1,1)), hx(cutop(1,1)), hx(cubot(1,1)), &
         hx(wlcl), hx(updfra), hx(wulcl), hx(wact)
  end subroutine one_para

end program run_kfcup_stages
