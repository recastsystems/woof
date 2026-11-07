! One column of WRF v4.6.1 old Kain-Fritsch (cu_physics = 99) through the
! scheme's own public entry point, KFCPS, exactly as
! phys/module_cumulus_driver.F:958-989 calls it.
!
! usage: run_kf_old CASES.bin INDEX OUT.bin
!
! The column is a 1 x nz x 1 tile: ims=ime=its=ite=1, jms=jme=jts=jte=1,
! kms=kts=1, kte=nz (the driver hands KFCPS kte = min(kte, kde-1), the
! mass-level count) and kme=kde=nz+1, so W(I,K+1,J) is in bounds for
! K = kte as it is in WRF.  The physical constants come from WRF's own
! share/module_model_constants.F, compiled from the pinned tree.
!
! Output (stream, little-endian):
!   int32 status (0), nz, cu_act_flag (0/1)
!   float32 raincv, pratec, nca, timec_kf
!   float32 rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten,
!           w0avg, udr_kf, ddr_kf, uer_kf, der_kf     (nz each)
! A column that reaches wrf_error_fatal writes nothing here; the stub
! prints the WRF message and exits 7, and build.sh records it.
program run_kf_old
  use module_model_constants, only: g, r_d, cp, ep_1, ep_2, xlv0, xlv1, &
                                    xls0, xls1, svp1, svp2, svp3, svpt0
  use module_cu_kf, only: kfcps
  implicit none
  integer, parameter :: magic_expected = 1262899020   ! 0x4B464F4C
  integer :: u_in, u_out, magic, version, ncase, want, i, k
  integer :: nz, stepcu, adapt_i, warm_i, fqi_i, fqs_i, fqr_i, edr
  real :: dt, dx, cudt, nca_in
  real, allocatable :: u1(:), v1(:), w1(:), t1(:), th1(:), qv1(:), &
                       dz1(:), p1(:), rho1(:), pi1(:), w0avg1(:)
  real, allocatable :: u3(:,:,:), v3(:,:,:), w3(:,:,:), t3(:,:,:), &
                       th3(:,:,:), qv3(:,:,:), dz3(:,:,:), p3(:,:,:), &
                       rho3(:,:,:), pi3(:,:,:), w0avg3(:,:,:)
  real, allocatable :: rthcuten(:,:,:), rqvcuten(:,:,:), rqccuten(:,:,:), &
                       rqrcuten(:,:,:), rqicuten(:,:,:), rqscuten(:,:,:), &
                       udr(:,:,:), ddr(:,:,:), uer(:,:,:), der(:,:,:)
  real :: raincv(1,1), pratec(1,1), nca(1,1), timec(1,1)
  logical :: act(1,1), adapt, warm_rain, f_qv, f_qc, f_qr, f_qi, f_qs
  character(len=1024) :: arg_in, arg_index, arg_out

  if (kind(1.0) /= 4) error stop 'default REAL is not 4 bytes'
  call get_command_argument(1, arg_in)
  call get_command_argument(2, arg_index)
  call get_command_argument(3, arg_out)
  read(arg_index, *) want

  open(newunit=u_in, file=trim(arg_in), access='stream', &
       form='unformatted', status='old', action='read')
  read(u_in) magic, version, ncase
  if (magic /= magic_expected .or. version /= 1) error stop 'bad case file'
  if (want < 0 .or. want >= ncase) error stop 'case index out of range'

  do i = 0, want
    read(u_in) nz, stepcu, adapt_i, warm_i, fqi_i, fqs_i, fqr_i, edr
    read(u_in) dt, dx, cudt, nca_in
    if (allocated(u1)) deallocate(u1, v1, w1, t1, th1, qv1, dz1, p1, &
                                  rho1, pi1, w0avg1)
    allocate(u1(nz), v1(nz), w1(nz+1), t1(nz), th1(nz), qv1(nz), &
             dz1(nz), p1(nz), rho1(nz), pi1(nz), w0avg1(nz))
    read(u_in) u1, v1, w1, t1, th1, qv1, dz1, p1, rho1, pi1, w0avg1
  end do
  close(u_in)

  allocate(u3(1,nz+1,1), v3(1,nz+1,1), w3(1,nz+1,1), t3(1,nz+1,1), &
           th3(1,nz+1,1), qv3(1,nz+1,1), dz3(1,nz+1,1), p3(1,nz+1,1), &
           rho3(1,nz+1,1), pi3(1,nz+1,1), w0avg3(1,nz+1,1))
  allocate(rthcuten(1,nz+1,1), rqvcuten(1,nz+1,1), rqccuten(1,nz+1,1), &
           rqrcuten(1,nz+1,1), rqicuten(1,nz+1,1), rqscuten(1,nz+1,1), &
           udr(1,nz+1,1), ddr(1,nz+1,1), uer(1,nz+1,1), der(1,nz+1,1))
  ! The level above the top mass level exists in WRF's memory extent and
  ! is never read by KFCPS for mass fields; give it a finite value.
  u3 = 0.; v3 = 0.; w3 = 0.; t3 = 250.; th3 = 300.; qv3 = 0.
  dz3 = 500.; p3 = 1000.; rho3 = 0.1; pi3 = 0.5; w0avg3 = 0.
  do k = 1, nz
    u3(1,k,1) = u1(k); v3(1,k,1) = v1(k); t3(1,k,1) = t1(k)
    th3(1,k,1) = th1(k); qv3(1,k,1) = qv1(k); dz3(1,k,1) = dz1(k)
    p3(1,k,1) = p1(k); rho3(1,k,1) = rho1(k); pi3(1,k,1) = pi1(k)
    w0avg3(1,k,1) = w0avg1(k)
  end do
  do k = 1, nz + 1
    w3(1,k,1) = w1(k)
  end do
  rthcuten = 0.; rqvcuten = 0.; rqccuten = 0.; rqrcuten = 0.
  rqicuten = 0.; rqscuten = 0.
  udr = 0.; ddr = 0.; uer = 0.; der = 0.
  raincv = 0.; pratec = 0.; timec = 0.
  nca(1,1) = nca_in
  act = .false.
  adapt = adapt_i /= 0
  warm_rain = warm_i /= 0
  f_qv = .true.
  f_qc = fqr_i /= 0
  f_qr = fqr_i /= 0
  f_qi = fqi_i /= 0
  f_qs = fqs_i /= 0

  call kfcps(                                                     &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1,               &
       ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1,               &
       its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                 &
       dt=dt, ktau=1, dx=dx, cudt=cudt, adapt_step_flag=adapt,    &
       rho=rho3, raincv=raincv, pratec=pratec, nca=nca,           &
       u=u3, v=v3, th=th3, t=t3, w=w3, qv=qv3, dz8w=dz3,          &
       pcps=p3, pi=pi3, w0avg=w0avg3,                             &
       xlv0=xlv0, xlv1=xlv1, xls0=xls0, xls1=xls1,                &
       cp=cp, r=r_d, g=g, ep1=ep_1, ep2=ep_2,                     &
       svp1=svp1, svp2=svp2, svp3=svp3, svpt0=svpt0,              &
       stepcu=stepcu, cu_act_flag=act, warm_rain=warm_rain,       &
       f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs,     &
       rqvcuten=rqvcuten, rqccuten=rqccuten, rqrcuten=rqrcuten,   &
       rqicuten=rqicuten, rqscuten=rqscuten, rthcuten=rthcuten,   &
       udr_kf=udr, ddr_kf=ddr, uer_kf=uer, der_kf=der,            &
       timec_kf=timec, kf_edrates=edr)

  open(newunit=u_out, file=trim(arg_out), access='stream', &
       form='unformatted', status='replace', action='write')
  write(u_out) 0, nz, merge(1, 0, act(1,1))
  write(u_out) raincv(1,1), pratec(1,1), nca(1,1), timec(1,1)
  write(u_out) rthcuten(1,1:nz,1), rqvcuten(1,1:nz,1), rqccuten(1,1:nz,1), &
               rqrcuten(1,1:nz,1), rqicuten(1,1:nz,1), rqscuten(1,1:nz,1), &
               w0avg3(1,1:nz,1), udr(1,1:nz,1), ddr(1,1:nz,1),             &
               uer(1,1:nz,1), der(1,1:nz,1)
  close(u_out)
end program run_kf_old
