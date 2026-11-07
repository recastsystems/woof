! KIAPS SAS (cu_physics = 14) column oracle: drives WRF v4.6.1's own
! CU_KSAS, exactly as module_cumulus_driver.F:1444-1474 calls it, over the
! call groups gen_cases.py wrote.
!
! Linked against the module_cu_ksas.o of the scalar WRF build (the 0 ULP
! referee: -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize), not a
! re-compilation of it, so the arithmetic graded is the arithmetic wrf.exe
! runs.  build.sh also links a second copy compiled here from the pristine
! source at the same flags and refuses if the two disagree on any word.
!
! The constants are WRF's own PARAMETERs from share/module_model_constants.F,
! restated here with the same expressions (gfortran folds each at float
! precision, operation by operation, exactly as the WRF compile does).
program run_ksas
  use module_cu_ksas, only : cu_ksas
  implicit none
  real, parameter :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  real, parameter :: cpv = 4.*r_v, cliq = 4190., cice = 2106., psat = 610.78
  real, parameter :: xls = 2.85e6, xlv = 2.5e6
  real, parameter :: ep_1 = r_v/r_d-1., ep_2 = r_d/r_v
  real, parameter :: pgcon = 0.55          ! Registry sas_pgcon default
  integer :: ngroups, ig, ncol, nz, stepcu, mp, fqc, fqi, dxf, itimestep
  integer :: hdr(8), u_in, u_out
  real :: rhdr(2), dt, dx
  character(len=512) :: fin, fout
  real, allocatable :: znu(:), buf(:, :), bufi(:, :), sfc(:)
  real, allocatable, dimension(:, :, :) :: p3d, pi3d, t3d, qv3d, qc3d, qi3d, &
       rho3d, dz8w, u3d, v3d, p3di, w, rthcuten, rqvcuten, rqccuten, &
       rqicuten, rucuten, rvcuten
  real, allocatable, dimension(:, :) :: xland, hpbl, hpbl_hold, hfx, qfx, &
       raincv, pratec, hbot, htop
  logical, allocatable :: cu_act_flag(:, :)
  logical :: f_qc, f_qi

  if (kind(1.0) /= 4) stop 'run_ksas: default REAL must be 4 bytes'
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u_in, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  open(newunit=u_out, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  read(u_in) ngroups
  do ig = 1, ngroups
    read(u_in) hdr
    ncol = hdr(1); nz = hdr(2); stepcu = hdr(3); mp = hdr(4)
    fqc = hdr(5); fqi = hdr(6); dxf = hdr(7); itimestep = hdr(8)
    read(u_in) rhdr
    dt = rhdr(1); dx = rhdr(2)
    f_qc = fqc /= 0
    f_qi = fqi /= 0
    allocate(znu(nz + 1), buf(ncol, nz), bufi(ncol, nz + 1), sfc(ncol))
    znu = 0.
    read(u_in) znu(1:nz)
    ! memory  ims:ime = 1:ncol, kms:kme = 1:nz+1, jms:jme = 1:1
    ! tile    its:ite = 1:ncol, kts:kte = 1:nz,   jts:jte = 1:1
    call lev(p3d); call lev(pi3d); call lev(t3d); call lev(qv3d)
    call lev(qc3d); call lev(qi3d); call lev(rho3d); call lev(dz8w)
    call lev(u3d); call lev(v3d)
    call ifc(p3di); call ifc(w)
    call srf(xland); call srf(hpbl); call srf(hpbl_hold); call srf(hfx)
    call srf(qfx)
    call sentinel(rthcuten); call sentinel(rqvcuten); call sentinel(rqccuten)
    call sentinel(rqicuten); call sentinel(rucuten); call sentinel(rvcuten)
    allocate(raincv(ncol, 1), pratec(ncol, 1), hbot(ncol, 1), htop(ncol, 1))
    allocate(cu_act_flag(ncol, 1))
    raincv = -999.; pratec = -999.; hbot = -999.; htop = -999.
    cu_act_flag = .false.

    call cu_ksas(dt=dt, dx=dx, p3di=p3di, p3d=p3d, pi3d=pi3d, qc3d=qc3d, &
         qi3d=qi3d, rho3d=rho3d, itimestep=itimestep, stepcu=stepcu, &
         hbot=hbot, htop=htop, cu_act_flag=cu_act_flag, &
         rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqccuten, &
         rqicuten=rqicuten, rucuten=rucuten, rvcuten=rvcuten, &
         qv3d=qv3d, t3d=t3d, raincv=raincv, pratec=pratec, xland=xland, &
         dz8w=dz8w, w=w, u3d=u3d, v3d=v3d, hpbl=hpbl, hfx=hfx, qfx=qfx, &
         hpbl_hold=hpbl_hold, znu=znu, mp_physics=mp, &
         dx_factor_nsas=dxf, p_qc=2, p_qi=3, p_first_scalar=2, &
         pgcon=pgcon, cp=cp, cliq=cliq, cpv=cpv, g=g, xlv=xlv, r_d=r_d, &
         r_v=r_v, ep_1=ep_1, ep_2=ep_2, cice=cice, xls=xls, psat=psat, &
         f_qi=f_qi, f_qc=f_qc, &
         ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nz + 1, &
         ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz + 1, &
         its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)

    call wlev(rthcuten); call wlev(rqvcuten); call wlev(rqccuten)
    call wlev(rqicuten); call wlev(rucuten); call wlev(rvcuten)
    write(u_out) raincv(:, 1)
    write(u_out) pratec(:, 1)
    write(u_out) hbot(:, 1)
    write(u_out) htop(:, 1)
    deallocate(znu, buf, bufi, sfc, p3d, pi3d, t3d, qv3d, qc3d, qi3d, rho3d, &
         dz8w, u3d, v3d, p3di, w, rthcuten, rqvcuten, rqccuten, rqicuten, &
         rucuten, rvcuten, xland, hpbl, hpbl_hold, hfx, qfx, raincv, pratec, &
         hbot, htop, cu_act_flag)
  end do
  close(u_in)
  close(u_out)

contains

  subroutine lev(a)
    real, allocatable, intent(inout) :: a(:, :, :)
    integer :: k
    allocate(a(ncol, nz + 1, 1))
    read(u_in) buf
    do k = 1, nz
      a(:, k, 1) = buf(:, k)
    end do
    a(:, nz + 1, 1) = 0.
  end subroutine lev

  subroutine ifc(a)
    real, allocatable, intent(inout) :: a(:, :, :)
    integer :: k
    allocate(a(ncol, nz + 1, 1))
    read(u_in) bufi
    do k = 1, nz + 1
      a(:, k, 1) = bufi(:, k)
    end do
  end subroutine ifc

  subroutine srf(a)
    real, allocatable, intent(inout) :: a(:, :)
    allocate(a(ncol, 1))
    read(u_in) sfc
    a(:, 1) = sfc
  end subroutine srf

  subroutine sentinel(a)
    real, allocatable, intent(inout) :: a(:, :, :)
    allocate(a(ncol, nz + 1, 1))
    a = -999.
  end subroutine sentinel

  subroutine wlev(a)
    real, intent(in) :: a(:, :, :)
    integer :: k
    do k = 1, nz
      buf(:, k) = a(:, k, 1)
    end do
    write(u_out) buf
  end subroutine wlev

end program run_ksas
