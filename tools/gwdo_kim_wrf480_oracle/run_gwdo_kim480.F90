program run_gwdo_kim480_oracle
  ! gwd_opt = 1, WRF v4.8.0: drive SUBROUTINE gwdo of WRF v4.8.0's
  ! phys/module_bl_gwdo.F (which calls bl_gwdo_run in
  ! phys/physics_mmm/bl_gwdo.F90, MMM-physics 550b5b49) exactly as
  ! module_pbl_driver.F:2176-2196 calls it: WRF's module_model_constants CP,
  ! G, R_D, R_V, EP_1, the literal PI=3.141592653, DT=dtbl, DX=dx2d, ELVMAX,
  ! and dx_factor / if_nonhyd from config_flags%gwd_dx_factor and
  ! config_flags%gwd_if_nonhyd (dyn_em/module_first_rk_step_part1.F:1310-1311).
  ! build.sh links this program against the REFEREE'S OWN object files
  ! (WRF-4.8.0-serial-pristine: phys/module_bl_gwdo.o,
  ! phys/physics_mmm/bl_gwdo.o, phys/ccpp_kind_types.o,
  ! share/module_model_constants.o), so every output word was computed by
  ! the referee's machine code.  This program reads inputs and copies arrays;
  ! it does no drag arithmetic.
  !
  ! usage: run_gwdo_kim480 INPUTS.bin OUTPUTS.bin
  ! INPUTS.bin (stream, little-endian):
  !   int32 ncol, nz, ncase
  !   float32 dx(ncase), dt(ncase), dx_factor(ncase); int32 if_nonhyd(ncase)
  !   float32 u, v, t, qv, p, pi, z, dz, rublten0, rvblten0  (ncol, nz) each
  !   float32 p_interface (ncol, nz+1)
  !   float32 var, oc1, sina, cosa, elvmax (ncol) each
  !   float32 oa, ol (ncol, 4) each
  ! For each case it restarts from the inputs (RUBLTEN/RVBLTEN as the PBL
  ! scheme left them), calls gwdo once and writes RUBLTEN, RVBLTEN, DTAUX3D,
  ! DTAUY3D (each ncol x nz, i fastest) then DUSFCG, DVSFCG (ncol).
  use module_model_constants, only: cp, g, r_d, r_v, ep_1
  use module_bl_gwdo, only: gwdo
  implicit none

  integer :: ncol, nz, ncase, kme, ic, uin, uout, m, errflg
  real, allocatable :: dxs(:), dts(:), dxfs(:)
  integer, allocatable :: nonhyd(:)
  real, allocatable, dimension(:, :, :) :: u3d, v3d, t3d, qv3d, p3d, pi3d, &
       z, dz, rub0, rvb0, p3di, rublten, rvblten, dtaux3d, dtauy3d
  real, allocatable, dimension(:, :) :: var2d, oc12d, sina, cosa, elvmax, &
       dusfcg, dvsfcg, dx2d
  real, allocatable :: oa(:, :, :), ol(:, :, :), buf(:, :)
  integer, allocatable :: kpbl(:, :)
  real, allocatable :: znu(:), znw(:)
  real, parameter :: p_top = 5000.0
  logical :: if_nonhyd
  character(len=1024) :: fin, fout
  character(len=256) :: errmsg

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  if (len_trim(fin) == 0 .or. len_trim(fout) == 0) then
    write(*, '(A)') 'usage: run_gwdo_kim480 INPUTS.bin OUTPUTS.bin'
    error stop 2
  end if
  if (storage_size(1.0) /= 32) error stop 3

  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  read(uin) ncol, nz, ncase
  allocate(dxs(ncase), dts(ncase), dxfs(ncase), nonhyd(ncase))
  read(uin) dxs
  read(uin) dts
  read(uin) dxfs
  read(uin) nonhyd
  ! WRF memory dims: kme = kde = nz + 1 (the staggered top); the scheme's
  ! tile is kts..kte = 1..nz mass levels, p3di reads 1..nz+1.
  kme = nz + 1
  allocate(u3d(ncol, kme, 1), v3d(ncol, kme, 1), t3d(ncol, kme, 1), &
           qv3d(ncol, kme, 1), p3d(ncol, kme, 1), pi3d(ncol, kme, 1), &
           z(ncol, kme, 1), dz(ncol, kme, 1), rub0(ncol, kme, 1), &
           rvb0(ncol, kme, 1), p3di(ncol, kme, 1), rublten(ncol, kme, 1), &
           rvblten(ncol, kme, 1), dtaux3d(ncol, kme, 1), dtauy3d(ncol, kme, 1))
  allocate(var2d(ncol, 1), oc12d(ncol, 1), sina(ncol, 1), cosa(ncol, 1), &
           elvmax(ncol, 1), dusfcg(ncol, 1), dvsfcg(ncol, 1), dx2d(ncol, 1), &
           kpbl(ncol, 1))
  allocate(oa(ncol, 1, 4), ol(ncol, 1, 4), buf(ncol, 4), znu(kme), znw(kme))
  ! the padding level nz+1 of the mass-level arrays is never read; define it
  u3d = 0.; v3d = 0.; t3d = 0.; qv3d = 0.; p3d = 0.; pi3d = 0.
  z = 0.; dz = 0.; rub0 = 0.; rvb0 = 0.; p3di = 0.
  read(uin) u3d(:, 1:nz, 1)
  read(uin) v3d(:, 1:nz, 1)
  read(uin) t3d(:, 1:nz, 1)
  read(uin) qv3d(:, 1:nz, 1)
  read(uin) p3d(:, 1:nz, 1)
  read(uin) pi3d(:, 1:nz, 1)
  read(uin) z(:, 1:nz, 1)
  read(uin) dz(:, 1:nz, 1)
  read(uin) rub0(:, 1:nz, 1)
  read(uin) rvb0(:, 1:nz, 1)
  read(uin) p3di(:, 1:kme, 1)
  read(uin) var2d(:, 1)
  read(uin) oc12d(:, 1)
  read(uin) sina(:, 1)
  read(uin) cosa(:, 1)
  read(uin) elvmax(:, 1)
  read(uin) buf
  do m = 1, 4
    oa(:, 1, m) = buf(:, m)
  end do
  read(uin) buf
  do m = 1, 4
    ol(:, 1, m) = buf(:, m)
  end do
  close(uin)
  ! gwd_opt = 1 ignores KPBL2D, ZNU, ZNW and P_TOP (bl_gwdo_run does not
  ! take them); they are passed because the interface requires them.
  kpbl = 1; znu = 0.; znw = 0.

  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write', convert='little_endian')
  do ic = 1, ncase
    rublten = rub0; rvblten = rvb0
    dtaux3d = 0.; dtauy3d = 0.; dusfcg = 0.; dvsfcg = 0.
    dx2d = dxs(ic)
    if_nonhyd = nonhyd(ic) /= 0
    errmsg = ''; errflg = -1
    call gwdo(u3d=u3d, v3d=v3d, t3d=t3d, qv3d=qv3d,                         &
              p3d=p3d, p3di=p3di, pi3d=pi3d, z=z,                           &
              rublten=rublten, rvblten=rvblten,                             &
              dtaux3d=dtaux3d, dtauy3d=dtauy3d,                             &
              dusfcg=dusfcg, dvsfcg=dvsfcg,                                 &
              var2d=var2d, oc12d=oc12d,                                     &
              oa2d1=oa(:, :, 1), oa2d2=oa(:, :, 2),                         &
              oa2d3=oa(:, :, 3), oa2d4=oa(:, :, 4),                         &
              ol2d1=ol(:, :, 1), ol2d2=ol(:, :, 2),                         &
              ol2d3=ol(:, :, 3), ol2d4=ol(:, :, 4),                         &
              elvmax=elvmax,                                                &
              sina=sina, cosa=cosa, znu=znu, znw=znw, p_top=p_top,          &
              cp=cp, g=g, rd=r_d, rv=r_v, ep1=ep_1, pi=3.141592653,         &
              dt=dts(ic), dx=dx2d, kpbl2d=kpbl, itimestep=1,                &
              dx_factor=dxfs(ic), if_nonhyd=if_nonhyd,                      &
              ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=kme,            &
              ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=kme,                &
              its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz,                 &
              errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,A)') 'gwdo errmsg: ', trim(errmsg)
      error stop 4
    end if
    write(uout) rublten(:, 1:nz, 1)
    write(uout) rvblten(:, 1:nz, 1)
    write(uout) dtaux3d(:, 1:nz, 1)
    write(uout) dtauy3d(:, 1:nz, 1)
    write(uout) dusfcg(:, 1)
    write(uout) dvsfcg(:, 1)
  end do
  close(uout)
end program run_gwdo_kim480_oracle
