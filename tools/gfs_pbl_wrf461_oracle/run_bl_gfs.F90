! run_bl_gfs.F90 -- column driver for the WRF v4.6.1 GFS PBL oracle.
!
! Calls the byte-unmodified BL_GFS (phys/module_bl_gfs.F, bl_pbl_physics=3)
! exactly as WRF's PBL driver binds it (phys/module_pbl_driver.F:1406-1424),
! with WRF's own module_model_constants for CP, G, ROVCP, R, ROVG, EP1 and
! KARMAN.  One j row of NCOL columns, NSTEP calls.  Between calls the driver
! advances the state with the returned tendencies (forward Euler over DT,
! single precision, negative moisture clipped at zero) so later steps see a
! state the scheme itself produced; WSPD and UST are the scheme's INOUT
! words and carry from one call to the next exactly as Fortran leaves them.
!
! Input (stream, native little-endian): int32 ncol, nz, nstep, flqi; real32
! dt; then the 3-D fields (ncol, nz+1) in Fortran order u, v, th, t, qv, qc,
! qi, p, pi, dz8w, z and the 2-D fields (ncol) psfc, ust, pbl, psim, psih,
! hfx, qfx, tsk, gz1oz0, wspd, br.
!
! Output (stream): the same int32/real32 header, then per step the step's
! complete input record (the 11 3-D and 11 2-D fields, as above) followed by
! the outputs rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten
! (ncol, nz+1), ust, wspd, pbl (ncol) and kpbl2d (int32, ncol).
program run_bl_gfs
   use module_model_constants, only: cp, g, rcp, r_d, rovg, ep_1, karman
   use module_bl_gfs, only: bl_gfs
   implicit none
   integer :: ncol, nz, nstep, flqi, istep, k, i, p_qi
   integer, parameter :: p_first_scalar = 2
   real :: dt
   real, allocatable, dimension(:,:,:) :: u, v, th, t, qv, qc, qi, p, pi, &
        dz8w, z, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten
   real, allocatable, dimension(:,:) :: psfc, ust, pbl, psim, psih, hfx, &
        qfx, tsk, gz1oz0, wspd, br
   integer, allocatable, dimension(:,:) :: kpbl2d
   character(len=512) :: fin, fout
   integer :: kme

   call get_command_argument(1, fin)
   call get_command_argument(2, fout)
   open(10, file=trim(fin), access='stream', form='unformatted', status='old')
   read(10) ncol, nz, nstep, flqi, dt
   kme = nz + 1
   allocate(u(ncol,kme,1), v(ncol,kme,1), th(ncol,kme,1), t(ncol,kme,1), &
            qv(ncol,kme,1), qc(ncol,kme,1), qi(ncol,kme,1), p(ncol,kme,1), &
            pi(ncol,kme,1), dz8w(ncol,kme,1), z(ncol,kme,1))
   allocate(rublten(ncol,kme,1), rvblten(ncol,kme,1), rthblten(ncol,kme,1), &
            rqvblten(ncol,kme,1), rqcblten(ncol,kme,1), rqiblten(ncol,kme,1))
   allocate(psfc(ncol,1), ust(ncol,1), pbl(ncol,1), psim(ncol,1), &
            psih(ncol,1), hfx(ncol,1), qfx(ncol,1), tsk(ncol,1), &
            gz1oz0(ncol,1), wspd(ncol,1), br(ncol,1), kpbl2d(ncol,1))
   read(10) u, v, th, t, qv, qc, qi, p, pi, dz8w, z
   read(10) psfc, ust, pbl, psim, psih, hfx, qfx, tsk, gz1oz0, wspd, br
   close(10)
   ! WRF's P_QI is >= PARAM_FIRST_SCALAR exactly when the moist set carries
   ! cloud ice (module_state_description); 1 stands for "not present".
   p_qi = 1
   if (flqi /= 0) p_qi = 4

   open(20, file=trim(fout), access='stream', form='unformatted', &
        status='replace')
   write(20) ncol, nz, nstep, flqi, dt
   do istep = 1, nstep
      write(20) u, v, th, t, qv, qc, qi, p, pi, dz8w, z
      write(20) psfc, ust, pbl, psim, psih, hfx, qfx, tsk, gz1oz0, wspd, br
      ! gfsinit's zeroing (module_bl_gfs.F:530-562); the scheme overwrites
      ! kts..kte and RQIBLTEN only with ice.
      rublten = 0.; rvblten = 0.; rthblten = 0.
      rqvblten = 0.; rqcblten = 0.; rqiblten = 0.
      kpbl2d = -999
      call bl_gfs(U3D=u, V3D=v, TH3D=th, T3D=t,                         &
                  QV3D=qv, QC3D=qc, QI3D=qi, P3D=p, PI3D=pi,             &
                  RUBLTEN=rublten, RVBLTEN=rvblten, RTHBLTEN=rthblten,   &
                  RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,                  &
                  RQIBLTEN=rqiblten,                                     &
                  CP=cp, G=g, ROVCP=rcp, R=r_d, ROVG=rovg,               &
                  DZ8W=dz8w, z=z, PSFC=psfc,                             &
                  UST=ust, PBL=pbl, PSIM=psim, PSIH=psih,                &
                  HFX=hfx, QFX=qfx, TSK=tsk, GZ1OZ0=gz1oz0,              &
                  WSPD=wspd, BR=br,                                      &
                  DT=dt, KPBL2D=kpbl2d, EP1=ep_1, KARMAN=karman,         &
                  P_QI=p_qi, P_FIRST_SCALAR=p_first_scalar,              &
                  IDS=1, IDE=ncol+1, JDS=1, JDE=2, KDS=1, KDE=kme,       &
                  IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=kme,         &
                  ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nz)
      write(20) rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten
      write(20) ust, wspd, pbl, kpbl2d
      ! Advance the state the next call sees.
      do k = 1, nz
         do i = 1, ncol
            u(i,k,1) = u(i,k,1) + rublten(i,k,1)*dt
            v(i,k,1) = v(i,k,1) + rvblten(i,k,1)*dt
            th(i,k,1) = th(i,k,1) + rthblten(i,k,1)*dt
            t(i,k,1) = th(i,k,1)*pi(i,k,1)
            qv(i,k,1) = max(qv(i,k,1) + rqvblten(i,k,1)*dt, 0.)
            qc(i,k,1) = max(qc(i,k,1) + rqcblten(i,k,1)*dt, 0.)
            qi(i,k,1) = max(qi(i,k,1) + rqiblten(i,k,1)*dt, 0.)
         end do
      end do
   end do
   close(20)
end program run_bl_gfs
