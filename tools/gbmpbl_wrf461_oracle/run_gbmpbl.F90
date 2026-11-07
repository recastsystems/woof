program run_gbmpbl_oracle
  ! Drive WRF v4.6.1's own GBM PBL (phys/module_bl_gbmpbl.F, the
  ! bl_pbl_physics=12 entry `gbmpbl`) on the columns make_cases.py wrote, for
  ! several steps, and dump every input the scheme read and every word it
  ! wrote.  Linked against either WRF's compiled object from the scalar
  ! build (the referee) or a -O0 compile of the pristine source; build.sh
  ! builds both and requires them to agree word for word.
  !
  ! One call per step covers a whole column set as one tile: its..ite are
  ! the columns, j is 1, kte = nz and kme = kde = nz + 1, exactly as WRF's
  ! PBL driver passes them (kte = min(kte, kde-1)).
  !
  ! Between steps the harness advances the state the way WRF's solver would
  ! see it after the PBL alone: u, v, theta and the three moisture species
  ! by dt times the coupled tendency (species clipped at zero), T = theta *
  ! pi, and TKE_PBL carried as GBMPBL left it.  The surface forcing is held.
  ! The GPU test repeats this float32 update in the same order.
  !
  ! Output layout (little-endian stream), per set:
  !   int32 ncol, nz, nstep ; float32 dt
  !   per step: 3-D (nz+1, ncol) u v th t qv qc qi tke_in
  !             rublten rvblten rthblten rqvblten rqcblten rqiblten
  !             exch_m exch_h exch_tke el_pbl tke_out
  !             2-D (ncol) pblh ust regime ; int32 kpbl
  use module_bl_gbmpbl, only: gbmpbl
  implicit none

  integer :: nset, iset, ncol, nz, nstep, istep, k
  real :: dt
  real, allocatable, dimension(:,:,:) :: u, v, th, t, qv, qc, qi, p, pi, &
       dz8w, z, rthraten, tke, tke_in, rublten, rvblten, rthblten, &
       rqvblten, rqcblten, rqiblten, exch_m, exch_h, exch_tke, el_pbl
  real, allocatable, dimension(:,:) :: psfc, znt, ust, zol, hol, pbl, &
       regime, psim, psih, xland, hfx, qfx, tsk, gz1oz0, wspd, br
  integer, allocatable, dimension(:,:) :: kpbl
  character(len=512) :: inpath, outpath
  integer :: uin, uout

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old', action='read')
  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', action='write')

  read(uin) nset
  do iset = 1, nset
     read(uin) ncol, nz, nstep
     read(uin) dt
     allocate(u(ncol,nz+1,1), v(ncol,nz+1,1), th(ncol,nz+1,1), &
          t(ncol,nz+1,1), qv(ncol,nz+1,1), qc(ncol,nz+1,1), qi(ncol,nz+1,1), &
          p(ncol,nz+1,1), pi(ncol,nz+1,1), dz8w(ncol,nz+1,1), z(ncol,nz+1,1), &
          rthraten(ncol,nz+1,1), tke(ncol,nz+1,1), tke_in(ncol,nz+1,1), &
          rublten(ncol,nz+1,1), rvblten(ncol,nz+1,1), rthblten(ncol,nz+1,1), &
          rqvblten(ncol,nz+1,1), rqcblten(ncol,nz+1,1), rqiblten(ncol,nz+1,1), &
          exch_m(ncol,nz+1,1), exch_h(ncol,nz+1,1), exch_tke(ncol,nz+1,1), &
          el_pbl(ncol,nz+1,1))
     allocate(psfc(ncol,1), znt(ncol,1), ust(ncol,1), zol(ncol,1), &
          hol(ncol,1), pbl(ncol,1), regime(ncol,1), psim(ncol,1), &
          psih(ncol,1), xland(ncol,1), hfx(ncol,1), qfx(ncol,1), tsk(ncol,1), &
          gz1oz0(ncol,1), wspd(ncol,1), br(ncol,1), kpbl(ncol,1))
     read(uin) u, v, th, qv, qc, qi, p, pi, dz8w, z, rthraten, tke
     read(uin) psfc, znt, ust, zol, hol, pbl, regime, psim, psih, xland, &
          hfx, qfx, tsk, gz1oz0, wspd, br
     read(uin) kpbl
     t = th * pi
     write(uout) ncol, nz, nstep
     write(uout) dt
     do istep = 1, nstep
        tke_in = tke
        ! INTENT(INOUT)/(OUT) arrays start from a recognisable fill so a word
        ! the scheme does not write cannot pass for one it did.
        rublten = -999.0; rvblten = -999.0; rthblten = -999.0
        rqvblten = -999.0; rqcblten = -999.0; rqiblten = -999.0
        exch_m = -999.0; exch_h = -999.0; exch_tke = -999.0; el_pbl = -999.0
        call gbmpbl(u, v, th, t, qv, qc, qi, p, pi,                       &
             rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten,     &
             exch_m, exch_h, exch_tke, el_pbl,                             &
             dz8w, z, psfc, tke, rthraten,                                 &
             znt, ust, zol, hol, pbl, kpbl, regime, psim, psih,            &
             xland, hfx, qfx, tsk, gz1oz0, wspd, br,                       &
             dt, dt / 60.0,                                                &
             1, ncol + 1, 1, 2, 1, nz + 1,                                 &
             1, ncol, 1, 1, 1, nz + 1,                                     &
             1, ncol, 1, 1, 1, nz)
        write(uout) u, v, th, t, qv, qc, qi, tke_in
        write(uout) rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten
        write(uout) exch_m, exch_h, exch_tke, el_pbl, tke
        write(uout) pbl, ust, regime
        write(uout) kpbl
        do k = 1, nz
           u(:,k,1) = u(:,k,1) + dt * rublten(:,k,1)
           v(:,k,1) = v(:,k,1) + dt * rvblten(:,k,1)
           th(:,k,1) = th(:,k,1) + dt * rthblten(:,k,1)
           t(:,k,1) = th(:,k,1) * pi(:,k,1)
           qv(:,k,1) = max(qv(:,k,1) + dt * rqvblten(:,k,1), 0.0)
           qc(:,k,1) = max(qc(:,k,1) + dt * rqcblten(:,k,1), 0.0)
           qi(:,k,1) = max(qi(:,k,1) + dt * rqiblten(:,k,1), 0.0)
        end do
     end do
     deallocate(u, v, th, t, qv, qc, qi, p, pi, dz8w, z, rthraten, tke, &
          tke_in, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, &
          exch_m, exch_h, exch_tke, el_pbl)
     deallocate(psfc, znt, ust, zol, hol, pbl, regime, psim, psih, xland, &
          hfx, qfx, tsk, gz1oz0, wspd, br, kpbl)
  end do
  close(uin)
  close(uout)
end program run_gbmpbl_oracle
