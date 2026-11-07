! WDM5 (mp_physics = 14) column oracle driver for WRF v4.6.1.
!
! Calls the public `wdm5` entry of phys/module_mp_wdm5.F exactly as
! module_microphysics_driver.F:2423-2462 does, with WRF's own
! module_model_constants for every physical constant, after `wdm5init` as
! module_physics_init.F:4580-4581 calls it.  The columns are the i dimension
! (ims=its=1 .. ime=ite=ncol), j is one row, k is 1..nz.
!
! usage: run_wdm5 INPUT OUTPUT CONSTANTS
!
! INPUT (little-endian stream): int32 ncol, nz, nsteps, itimestep0;
!   float32 delt, ccn0; then 13 float32 (ncol, nz) arrays in Fortran order
!   (i fastest): th qv qc qr qi qs nn nc nr den pii p delz; then two float32
!   (ncol) arrays: rain snow.
! OUTPUT: for every step, 13 (ncol, nz) arrays: th qv qc qr qi qs nn nc nr
!   re_cloud re_ice re_snow refl_10cm, then 5 (ncol) arrays: rain rainncv
!   snow snowncv sr.
! CONSTANTS: the bit patterns wdm5init and radar_init left in the modules'
!   saved variables, and the constant expressions gfortran folds at compile
!   time inside wdm52D (written here with the module's own PARAMETER values).
!
! Step s is called with itimestep = itimestep0 + s - 1, so itimestep0 = 1
! exercises the cold-start CCN fill (:216-224) and any larger value does not.
program run_wdm5
  use module_mp_wdm5
  use module_mp_radar, only: xam_r, xam_s, xobmr, xorg2, xosg2, &
       xoams, xobms, xocms, xcre, xcrg, xcse, xcsg, &
       K_w, PI5, lamda4, m_w_0, m_i_0, nrbins, xxDs, xdts, simpson
  use module_model_constants, only: g, cp, cpv, r_d, r_v, svpt0, ep_1, ep_2, &
       epsilon, xls, xlv, xlf, rhoair0, rhowater, rhosnow, cliq, cice, psat
  implicit none
  character(len=1024) :: inpath, outpath, cpath
  integer :: ncol, nz, nsteps, itimestep0, step
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte, iu, ou
  real :: delt, ccn0
  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qi, qs, nn, nc, nr
  real, allocatable, dimension(:,:,:) :: den, pii, p, delz
  real, allocatable, dimension(:,:,:) :: re_cloud, re_ice, re_snow, refl
  real, allocatable, dimension(:,:) :: rain, rainncv, snow, snowncv, sr
  logical :: diagflag

  if (command_argument_count() /= 3) then
    write(*,*) 'usage: run_wdm5 INPUT OUTPUT CONSTANTS'
    error stop 2
  end if
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, cpath)

  ! WRF's -DRWORDSIZE=4 configuration: default REAL is binary32.
  if (kind(1.0) /= 4 .or. digits(1.0) /= 24) error stop 3

  open(newunit=iu, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(iu) ncol, nz, nsteps, itimestep0
  read(iu) delt, ccn0
  ids = 1; ide = ncol; jds = 1; jde = 1; kds = 1; kde = nz
  ims = ids; ime = ide; jms = jds; jme = jde; kms = kds; kme = kde
  its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde
  allocate(th(ims:ime,kms:kme,jms:jme), qv(ims:ime,kms:kme,jms:jme))
  allocate(qc(ims:ime,kms:kme,jms:jme), qr(ims:ime,kms:kme,jms:jme))
  allocate(qi(ims:ime,kms:kme,jms:jme), qs(ims:ime,kms:kme,jms:jme))
  allocate(nn(ims:ime,kms:kme,jms:jme))
  allocate(nc(ims:ime,kms:kme,jms:jme), nr(ims:ime,kms:kme,jms:jme))
  allocate(den(ims:ime,kms:kme,jms:jme), pii(ims:ime,kms:kme,jms:jme))
  allocate(p(ims:ime,kms:kme,jms:jme), delz(ims:ime,kms:kme,jms:jme))
  allocate(re_cloud(ims:ime,kms:kme,jms:jme), re_ice(ims:ime,kms:kme,jms:jme))
  allocate(re_snow(ims:ime,kms:kme,jms:jme), refl(ims:ime,kms:kme,jms:jme))
  allocate(rain(ims:ime,jms:jme), rainncv(ims:ime,jms:jme))
  allocate(snow(ims:ime,jms:jme), snowncv(ims:ime,jms:jme), sr(ims:ime,jms:jme))
  read(iu) th, qv, qc, qr, qi, qs, nn, nc, nr, den, pii, p, delz
  read(iu) rain, snow
  close(iu)
  rainncv = 0.; snowncv = 0.; sr = 0.
  re_cloud = 0.; re_ice = 0.; re_snow = 0.; refl = 0.
  diagflag = .true.

  call wdm5init(rhoair0, rhowater, rhosnow, cliq, cpv, ccn0, .true.)
  call dump_constants(trim(cpath))

  open(newunit=ou, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  do step = 1, nsteps
    call wdm5(th=th, q=qv, qc=qc, qr=qr, qi=qi, qs=qs,                      &
              nn=nn, nc=nc, nr=nr,                                           &
              den=den, pii=pii, p=p, delz=delz,                              &
              delt=delt, g=g, cpd=cp, cpv=cpv, ccn0=ccn0,                    &
              rd=r_d, rv=r_v, t0c=svpt0,                                     &
              ep1=ep_1, ep2=ep_2, qmin=epsilon,                              &
              xls=xls, xlv0=xlv, xlf0=xlf,                                   &
              den0=rhoair0, denr=rhowater,                                   &
              cliq=cliq, cice=cice, psat=psat,                               &
              rain=rain, rainncv=rainncv,                                    &
              snow=snow, snowncv=snowncv,                                    &
              sr=sr,                                                         &
              refl_10cm=refl, diagflag=diagflag, do_radar_ref=1,             &
              has_reqc=1, has_reqi=1, has_reqs=1,                            &
              re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,             &
              itimestep=itimestep0 + step - 1,                               &
              ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,          &
              ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,          &
              its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
    write(ou) th, qv, qc, qr, qi, qs, nn, nc, nr, re_cloud, re_ice, &
              re_snow, refl
    write(ou) rain, rainncv, snow, snowncv, sr
  end do
  close(ou)

contains

  subroutine put(u, name, x)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    write(u, '(a, 1x, z8.8, 1x, es16.9)') name, transfer(x, 0), x
  end subroutine put

  subroutine putd(u, name, x)
    integer, intent(in) :: u
    character(len=*), intent(in) :: name
    double precision, intent(in) :: x
    write(u, '(a, 1x, z16.16, 1x, es24.17)') name, transfer(x, 0_8), x
  end subroutine putd

  subroutine dump_constants(path)
    character(len=*), intent(in) :: path
    ! The module's own PARAMETER values (module_mp_wdm5.F:15-53), restated
    ! so the compile-time folds below are the folds gfortran performs on
    ! the same constant expressions inside wdm52D.
    real, parameter :: actr = 1.5, n0smax = 1.e11, n0s = 2.e6
    integer :: u, i4
    character(len=1) :: tag
    character(len=2) :: tag2
    open(newunit=u, file=path, status='replace', action='write')
    call put(u, 'pi', pi); call put(u, 'xlv1', xlv1)
    call put(u, 'qc0', qc0); call put(u, 'qck1', qck1)
    call put(u, 'pidnc', pidnc)
    call put(u, 'bvtr1', bvtr1); call put(u, 'bvtr2', bvtr2)
    call put(u, 'bvtr3', bvtr3); call put(u, 'bvtr4', bvtr4)
    call put(u, 'bvtr5', bvtr5); call put(u, 'bvtr7', bvtr7)
    call put(u, 'bvtr2o5', bvtr2o5); call put(u, 'bvtr3o5', bvtr3o5)
    call put(u, 'g1pbr', g1pbr); call put(u, 'g2pbr', g2pbr)
    call put(u, 'g3pbr', g3pbr); call put(u, 'g4pbr', g4pbr)
    call put(u, 'g5pbr', g5pbr); call put(u, 'g7pbr', g7pbr)
    call put(u, 'g5pbro2', g5pbro2); call put(u, 'g7pbro2', g7pbro2)
    call put(u, 'pvtr', pvtr); call put(u, 'pvtrn', pvtrn)
    call put(u, 'eacrr', eacrr); call put(u, 'pacrr', pacrr)
    call put(u, 'precr1', precr1); call put(u, 'precr2', precr2)
    call put(u, 'pidn0r', pidn0r); call put(u, 'pidnr', pidnr)
    call put(u, 'xmmax', xmmax); call put(u, 'roqimax', roqimax)
    call put(u, 'bvts1', bvts1); call put(u, 'bvts2', bvts2)
    call put(u, 'bvts3', bvts3); call put(u, 'bvts4', bvts4)
    call put(u, 'g1pbs', g1pbs); call put(u, 'g3pbs', g3pbs)
    call put(u, 'g4pbs', g4pbs); call put(u, 'g5pbso2', g5pbso2)
    call put(u, 'pvts', pvts); call put(u, 'pacrs', pacrs)
    call put(u, 'precs1', precs1); call put(u, 'precs2', precs2)
    call put(u, 'pidn0s', pidn0s); call put(u, 'pacrc', pacrc)
    call put(u, 'rslopecmax', rslopecmax); call put(u, 'rslopec2max', rslopec2max)
    call put(u, 'rslopec3max', rslopec3max)
    call put(u, 'rslopermax', rslopermax); call put(u, 'rslopesmax', rslopesmax)
    call put(u, 'rsloperbmax', rsloperbmax); call put(u, 'rslopesbmax', rslopesbmax)
    call put(u, 'rsloper2max', rsloper2max); call put(u, 'rslopes2max', rslopes2max)
    call put(u, 'rsloper3max', rsloper3max); call put(u, 'rslopes3max', rslopes3max)
    ! effectRad_wdm5 evaluates rgmma(5./3.) on every call (:2351).
    call put(u, 'cdm2', rgmma(5./3.))
    ! Constant expressions gfortran folds at compile time in wdm52D.
    call put(u, 'fold_actr3', (actr*1.E-6)**3)          ! :1536
    call put(u, 'fold_24cbrt', (24.)**(.3333333))       ! :995, :1507
    call put(u, 'fold_1e20o16', 1.e20/16.)              ! :1055
    call put(u, 'fold_n0sratio', n0smax/n0s)            ! :796, :1167
    call put(u, 'fold_0176o093', 0.176/0.93)            ! :2252
    ! radar_init products after wdm5init's xam/xbm/xmu (:1730-1740); the
    ! REAL ones as float32 words, the DOUBLE PRECISION ones as binary64.
    call put(u, 'xam_r', xam_r); call put(u, 'xam_s', xam_s)
    call put(u, 'xobmr', xobmr); call put(u, 'xorg2', xorg2)
    call put(u, 'xosg2', xosg2)
    call put(u, 'xoams', xoams); call put(u, 'xobms', xobms)
    call put(u, 'xocms', xocms)
    do i4 = 1, 4
      write(tag, '(i1)') i4
      call put(u, 'xcre'//tag, xcre(i4)); call put(u, 'xcrg'//tag, xcrg(i4))
      call put(u, 'xcse'//tag, xcse(i4)); call put(u, 'xcsg'//tag, xcsg(i4))
    end do
    call putd(u, 'K_w', K_w); call putd(u, 'PI5', PI5)
    call putd(u, 'lamda4', lamda4)
    call putd(u, 'm_w_0_re', dble(m_w_0)); call putd(u, 'm_w_0_im', dimag(m_w_0))
    call putd(u, 'm_i_0_re', dble(m_i_0)); call putd(u, 'm_i_0_im', dimag(m_i_0))
    do i4 = 1, nrbins
      write(tag2, '(i2.2)') i4
      call putd(u, 'xxDs'//tag2, xxDs(i4)); call putd(u, 'xdts'//tag2, xdts(i4))
      call putd(u, 'simpson'//tag2, simpson(i4))
    end do
    close(u)
  end subroutine dump_constants
end program run_wdm5
