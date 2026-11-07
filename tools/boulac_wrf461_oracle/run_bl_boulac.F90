! BouLac column oracle driver (WRF v4.6.1 phys/module_bl_boulac.F, unmodified).
!
! Reads one grid written by make_cases.py, calls WRF's 3-D BOULAC entry for
! nsteps consecutive steps on a (ncol, nz+1, 1) memory shape -- ims=its=1,
! ime=ite=ncol, jms=jme=jts=jte=1, kms=kts=1, kme=nz+1, kte=nz, the shape
! module_pbl_driver.F hands every PBL scheme (kte = kde-1 mass levels, kme
! one face above) -- and records every step's inputs and outputs.
!
! Carried by the scheme itself, exactly as WRF carries them: TKE (INOUT),
! DLG_BEP (INOUT under flag_bep), PBLH (INOUT, rewritten every call), WU..WQ
! (INTENT(OUT) but written only when idiff /= 1; WRF's grid arrays keep their
! previous values otherwise).  The RUBLTEN..RQCBLTEN family is zeroed before
! every call, as module_pbl_driver.F:1036-1042 does.  Between steps the state
! advances by its own tendencies in float32, x = x + dt*xten, so the next
! step sees a state BouLac itself moved.
!
! Usage: run_bl_boulac GRID.in.bin GRID.rec.bin
program run_bl_boulac
  use module_bl_boulac, only: boulac
  implicit none
  integer, parameter :: magic_expected = 11579816   ! 0xB0B1A8
  character(len=1024) :: inpath, outpath
  integer :: hdr(6), nz, nc, nsteps, iflag_bep, idiff, step, k
  real :: rh(3), dt, cp, g
  logical :: flag_bep
  real, allocatable, dimension(:,:,:) :: dz8w, u, v, th, qv, qc, rho, tke
  real, allocatable, dimension(:,:,:) :: a_u, a_v, a_t, a_q, a_e, b_u, b_v, &
       b_t, b_q, b_e, dlg, dl_u, vl, sf
  real, allocatable, dimension(:,:,:) :: rublten, rvblten, rthblten, &
       rqvblten, rqcblten, dlk, wu, wv, wt, wq, exch_h, exch_m
  real, allocatable, dimension(:,:) :: hfx, qfx, ust, frc, pblh
  real, allocatable :: tmp(:,:), tmp1(:), tmpf(:,:)

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(10, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(10) hdr
  read(10) rh
  if (hdr(1) /= magic_expected) stop 'bad magic'
  nz = hdr(2); nc = hdr(3); nsteps = hdr(4); iflag_bep = hdr(5)
  idiff = hdr(6)
  dt = rh(1); cp = rh(2); g = rh(3)
  flag_bep = iflag_bep /= 0

  call alloc3(dz8w); call alloc3(u); call alloc3(v); call alloc3(th)
  call alloc3(qv); call alloc3(qc); call alloc3(rho); call alloc3(tke)
  call alloc3(a_u); call alloc3(a_v); call alloc3(a_t); call alloc3(a_q)
  call alloc3(a_e); call alloc3(b_u); call alloc3(b_v); call alloc3(b_t)
  call alloc3(b_q); call alloc3(b_e); call alloc3(dlg); call alloc3(dl_u)
  call alloc3(vl); call alloc3(sf)
  call alloc3(rublten); call alloc3(rvblten); call alloc3(rthblten)
  call alloc3(rqvblten); call alloc3(rqcblten); call alloc3(dlk)
  call alloc3(wu); call alloc3(wv); call alloc3(wt); call alloc3(wq)
  call alloc3(exch_h); call alloc3(exch_m)
  allocate(hfx(nc,1), qfx(nc,1), ust(nc,1), frc(nc,1), pblh(nc,1))
  allocate(tmp(nc,nz), tmp1(nc), tmpf(nc,nz+1))

  call read3(dz8w); call read3(u); call read3(v); call read3(th)
  call read3(qv); call read3(qc); call read3(rho); call read3(tke)
  read(10) tmp1; hfx(:,1) = tmp1
  read(10) tmp1; qfx(:,1) = tmp1
  read(10) tmp1; ust(:,1) = tmp1
  read(10) tmp1; frc(:,1) = tmp1
  read(10) tmp1; pblh(:,1) = tmp1
  call read3(a_u); call read3(a_v); call read3(a_t); call read3(a_q)
  call read3(a_e); call read3(b_u); call read3(b_v); call read3(b_t)
  call read3(b_q); call read3(b_e); call read3(dlg); call read3(dl_u)
  call read3(vl)
  read(10) tmpf
  sf(:,:,1) = tmpf
  close(10)

  open(20, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  write(20) hdr
  write(20) rh
  do step = 1, nsteps
     ! inputs of this step
     call write3(u); call write3(v); call write3(th); call write3(qv)
     call write3(qc); call write3(tke); call write3(dlg)
     write(20) pblh(:,1)
     rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.
     rqcblten = 0.
     call boulac(frc, idiff, flag_bep, dz8w, dt, u, v, th, rho, qv, qc, &
          hfx, qfx, ust, cp, g, rublten, rvblten, rthblten, rqvblten,    &
          rqcblten, tke, dlk, wu, wv, wt, wq, exch_h, exch_m, pblh,       &
          a_u, a_v, a_t, a_q, a_e, b_u, b_v, b_t, b_q, b_e, dlg, dl_u,    &
          sf, vl,                                                         &
          1, nc, 1, 1, 1, nz+1,                                           &
          1, nc, 1, 1, 1, nz+1,                                           &
          1, nc, 1, 1, 1, nz)
     ! outputs of this step
     call write3(rublten); call write3(rvblten); call write3(rthblten)
     call write3(rqvblten); call write3(rqcblten); call write3(tke)
     call write3(dlk); call write3(exch_h); call write3(exch_m)
     call write3(wu); call write3(wv); call write3(wt); call write3(wq)
     call write3(dlg)
     write(20) pblh(:,1)
     ! advance the state by its own tendencies (float32, mult then add)
     do k = 1, nz
        u(:,k,1) = u(:,k,1) + dt*rublten(:,k,1)
        v(:,k,1) = v(:,k,1) + dt*rvblten(:,k,1)
        th(:,k,1) = th(:,k,1) + dt*rthblten(:,k,1)
        qv(:,k,1) = qv(:,k,1) + dt*rqvblten(:,k,1)
        qc(:,k,1) = qc(:,k,1) + dt*rqcblten(:,k,1)
     end do
  end do
  close(20)

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(nc, nz+1, 1))
    a = 0.
  end subroutine alloc3

  subroutine read3(a)
    real, intent(inout) :: a(:,:,:)
    read(10) tmp
    a(:, 1:nz, 1) = tmp
  end subroutine read3

  subroutine write3(a)
    real, intent(in) :: a(:,:,:)
    tmp = a(:, 1:nz, 1)
    write(20) tmp
  end subroutine write3

end program run_bl_boulac
