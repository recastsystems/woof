program run_gd_oracle
  ! Drive WRF v4.6.1's Grell-Devenyi ensemble cumulus scheme (cu_physics = 93)
  ! through its own WRF entry point GRELLDRV and dump every output word.
  !
  ! The object linked here is chosen by build.sh: the module_cu_gd.o of the
  ! WRF build tree itself (the object wrf.exe links), the same source rebuilt
  ! with the strict or the scalar flag set, or the patched source carrying
  ! WOOF's declared closure-count fix (patches/gd-closure-count.patch).
  ! Nothing in this program changes what the scheme computes.
  !
  ! Input file (stream, little endian), written by gd_cases.py:
  !   int32  nx, ny, nz, nsteps, stepcu, periodic_x, periodic_y, f_qi
  !   real32 dt, dx, xlv, cp, g, r_v
  !   per step s = 1..nsteps, every array in WRF memory order (i fastest,
  !   then k, then j), levels 1..nz (the kte = nz+1 face is never read):
  !     u v w t q p pi rho dz8w p8w rthften rqvften rthraten rthblten rqvblten
  !     ht xland gsw                        (2-D)
  !   then the initial accumulators (2-D): htop hbot apr_gr apr_w apr_mc
  !     apr_st apr_as apr_capma apr_capme apr_capmi
  !
  ! The cumulus cadence is WRF's own (module_cumulus_driver.F:832-849 for a
  ! fixed step): GRELLDRV runs when itimestep == 1, stepcu == 1 or
  ! mod(itimestep, stepcu) == 0; otherwise RTHCUTEN/PRATEC stay as they are.
  ! RAINC then accumulates PRATEC*DT every step exactly as
  ! module_physics_addtendc.F:2141 does.  RTHFTEN/RQVFTEN are re-read every
  ! step because GRELLDRV folds the radiative and boundary-layer forcing
  ! into them in place (module_cu_gd.F:219-224) and WRF's dycore rewrites
  ! them every step.
  !
  ! Output file (stream), per step:
  !   int32  due
  !   real32 rthcuten rqvcuten rqccuten rqicuten gdc gdc2   (nx,nz,ny)
  !   real32 raincv pratec rainc htop hbot apr_gr apr_w apr_mc apr_st apr_as
  !          apr_capma apr_capme apr_capmi mass_flux        (nx,ny)
  !   int32  ktop_deep                                      (nx,ny)
  !   real32 xf_ens pr_ens                                  (nx,ny,144)
  use module_cu_gd, only: grelldrv
  implicit none

  integer, parameter :: maxiens = 1, maxens = 3, maxens2 = 3, maxens3 = 16
  integer, parameter :: ensdim = maxiens*maxens*maxens2*maxens3
  integer :: nx, ny, nz, nsteps, stepcu, iperx, ipery, ifqi
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: istep, due, i, j
  real :: dt, dx, xlv, cp, g, r_v
  logical :: periodic_x, periodic_y, f_qi
  real, allocatable, dimension(:,:,:) :: u, v, w, t, q, p, pi, rho, dz8w, p8w
  real, allocatable, dimension(:,:,:) :: rthften, rqvften, rthraten, rthblten, rqvblten
  real, allocatable, dimension(:,:,:) :: rthcuten, rqvcuten, rqccuten, rqicuten, gdc, gdc2
  real, allocatable, dimension(:,:) :: ht, xland, gsw, raincv, pratec, rainc, mass_flux
  real, allocatable, dimension(:,:) :: htop, hbot, apr_gr, apr_w, apr_mc, apr_st, apr_as
  real, allocatable, dimension(:,:) :: apr_capma, apr_capme, apr_capmi
  real, allocatable, dimension(:,:,:) :: xf_ens, pr_ens
  real, allocatable, dimension(:,:,:) :: cfu1, cfd1, dfu1, efu1, dfd1, efd1
  integer, allocatable :: ktop_deep(:,:)
  logical, allocatable :: cu_act_flag(:,:)
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old')
  open(11, file=trim(fout), access='stream', form='unformatted', status='replace')
  read(10) nx, ny, nz, nsteps, stepcu, iperx, ipery, ifqi
  read(10) dt, dx, xlv, cp, g, r_v
  periodic_x = iperx /= 0
  periodic_y = ipery /= 0
  f_qi = ifqi /= 0

  ! One tile covering the whole mass grid.  ite = ide-1 (the last mass
  ! column) rather than ide: CUP_enss and massflx_stats loop to ite, and a
  ! tile ending at ide would make them process an uninitialised column that
  ! no output reads.  kte = kde = nz+1 as WRF's physics tiles are.
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  its = 1; ite = nx; jts = 1; jte = ny; kts = 1; kte = nz + 1
  ims = its; ime = ite; jms = jts; jme = jte; kms = kts; kme = kte

  allocate(u(ims:ime,kms:kme,jms:jme), v(ims:ime,kms:kme,jms:jme), &
           w(ims:ime,kms:kme,jms:jme), t(ims:ime,kms:kme,jms:jme), &
           q(ims:ime,kms:kme,jms:jme), p(ims:ime,kms:kme,jms:jme), &
           pi(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme), &
           dz8w(ims:ime,kms:kme,jms:jme), p8w(ims:ime,kms:kme,jms:jme), &
           rthften(ims:ime,kms:kme,jms:jme), rqvften(ims:ime,kms:kme,jms:jme), &
           rthraten(ims:ime,kms:kme,jms:jme), rthblten(ims:ime,kms:kme,jms:jme), &
           rqvblten(ims:ime,kms:kme,jms:jme))
  allocate(rthcuten(ims:ime,kms:kme,jms:jme), rqvcuten(ims:ime,kms:kme,jms:jme), &
           rqccuten(ims:ime,kms:kme,jms:jme), rqicuten(ims:ime,kms:kme,jms:jme), &
           gdc(ims:ime,kms:kme,jms:jme), gdc2(ims:ime,kms:kme,jms:jme))
  allocate(cfu1(ims:ime,kms:kme,jms:jme), cfd1(ims:ime,kms:kme,jms:jme), &
           dfu1(ims:ime,kms:kme,jms:jme), efu1(ims:ime,kms:kme,jms:jme), &
           dfd1(ims:ime,kms:kme,jms:jme), efd1(ims:ime,kms:kme,jms:jme))
  allocate(ht(ims:ime,jms:jme), xland(ims:ime,jms:jme), gsw(ims:ime,jms:jme), &
           raincv(ims:ime,jms:jme), pratec(ims:ime,jms:jme), rainc(ims:ime,jms:jme), &
           mass_flux(ims:ime,jms:jme), htop(ims:ime,jms:jme), hbot(ims:ime,jms:jme), &
           apr_gr(ims:ime,jms:jme), apr_w(ims:ime,jms:jme), apr_mc(ims:ime,jms:jme), &
           apr_st(ims:ime,jms:jme), apr_as(ims:ime,jms:jme), apr_capma(ims:ime,jms:jme), &
           apr_capme(ims:ime,jms:jme), apr_capmi(ims:ime,jms:jme), &
           ktop_deep(ims:ime,jms:jme), cu_act_flag(ims:ime,jms:jme))
  allocate(xf_ens(ims:ime,jms:jme,ensdim), pr_ens(ims:ime,jms:jme,ensdim))

  ! gdinit (module_cu_gd.F:4144-4244, restart = .false.): every tendency,
  ! the forcing pair, MASS_FLUX and the APR_* accumulators start at zero.
  rthcuten = 0.; rqvcuten = 0.; rqccuten = 0.; rqicuten = 0.
  gdc = 0.; gdc2 = 0.; raincv = 0.; pratec = 0.; rainc = 0.; mass_flux = 0.
  xf_ens = 0.; pr_ens = 0.; ktop_deep = 0
  cfu1 = 0.; cfd1 = 0.; dfu1 = 0.; efu1 = 0.; dfd1 = 0.; efd1 = 0.
  u = 0.; v = 0.; w = 0.; t = 0.; q = 0.; p = 0.; pi = 0.; rho = 0.
  dz8w = 0.; p8w = 0.; rthften = 0.; rqvften = 0.; rthraten = 0.
  rthblten = 0.; rqvblten = 0.

  do istep = 1, nsteps
     call read3(u); call read3(v); call read3(w); call read3(t); call read3(q)
     call read3(p); call read3(pi); call read3(rho); call read3(dz8w)
     call read3(p8w); call read3(rthften); call read3(rqvften)
     call read3(rthraten); call read3(rthblten); call read3(rqvblten)
     read(10) ht, xland, gsw
     if (istep == 1) then
        read(10) htop, hbot, apr_gr, apr_w, apr_mc, apr_st, apr_as, &
                 apr_capma, apr_capme, apr_capmi
     end if

     due = 0
     if (istep == 1 .or. stepcu == 1 .or. mod(istep, stepcu) == 0) due = 1
     if (due == 1) then
        call grelldrv(DT=dt, ITIMESTEP=istep, DX=dx, &
             U=u, V=v, T=t, W=w, RHO=rho, P=p, PI=pi, Q=q, RAINCV=raincv, &
             DZ8W=dz8w, P8W=p8w, XLV=xlv, CP=cp, G=g, R_V=r_v, &
             PRATEC=pratec, &
             APR_GR=apr_gr, APR_W=apr_w, APR_MC=apr_mc, APR_ST=apr_st, &
             APR_AS=apr_as, APR_CAPMA=apr_capma, APR_CAPME=apr_capme, &
             APR_CAPMI=apr_capmi, MASS_FLUX=mass_flux, &
             XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht, XLAND=xland, GSW=gsw, &
             GDC=gdc, GDC2=gdc2, &
             ENSDIM=ensdim, MAXIENS=maxiens, MAXENS=maxens, &
             MAXENS2=maxens2, MAXENS3=maxens3, &
             HTOP=htop, HBOT=hbot, KTOP_DEEP=ktop_deep, &
             CU_ACT_FLAG=cu_act_flag, WARM_RAIN=.not. f_qi, &
             IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
             IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, &
             ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
             PERIODIC_X=periodic_x, PERIODIC_Y=periodic_y, &
             RTHCUTEN=rthcuten, RTHFTEN=rthften, &
             RQICUTEN=rqicuten, RQVFTEN=rqvften, &
             RTHRATEN=rthraten, RTHBLTEN=rthblten, &
             RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, RQVBLTEN=rqvblten, &
             F_QV=.true., F_QC=.true., F_QR=.true., F_QI=f_qi, F_QS=f_qi, &
             CFU1=cfu1, CFD1=cfd1, DFU1=dfu1, EFU1=efu1, DFD1=dfd1, &
             EFD1=efd1, F_FLUX=.false.)
     end if
     ! module_physics_addtendc.F:2141, every step.
     do j = jts, min(jte, jde-1)
        do i = its, min(ite, ide-1)
           rainc(i,j) = rainc(i,j) + pratec(i,j)*dt
        end do
     end do

     write(11) due
     call write3(rthcuten); call write3(rqvcuten); call write3(rqccuten)
     call write3(rqicuten); call write3(gdc); call write3(gdc2)
     write(11) raincv, pratec, rainc, htop, hbot, apr_gr, apr_w, apr_mc, &
               apr_st, apr_as, apr_capma, apr_capme, apr_capmi, mass_flux
     write(11) ktop_deep
     write(11) xf_ens, pr_ens
  end do
  close(10)
  close(11)

contains

  subroutine read3(a)
    real, intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    integer :: jj
    do jj = jms, jme
       read(10) a(ims:ime, kms:kme-1, jj)
    end do
  end subroutine read3

  subroutine write3(a)
    real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
    integer :: jj
    do jj = jms, jme
       write(11) a(ims:ime, kms:kme-1, jj)
    end do
  end subroutine write3

end program run_gd_oracle
