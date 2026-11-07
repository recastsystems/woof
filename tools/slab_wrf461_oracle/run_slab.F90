! Column driver for the byte-unmodified WRF v4.6.1 phys/module_sf_slab.F.
!
! It calls the module's own 3-D entry, SLAB, exactly the way the WRF surface
! driver does (module_surface_driver.F:2556-2565): one column per call
! (ims=ime=its=ite=1, jms=jme=jts=jte=1, kms=kme=kts=kte=1), the physical
! constants taken from WRF's own module_model_constants (svp1, svp2, svp3,
! svpt0, ep_2, karman, eomeg, stbolt, xlv, rcp, p1000mb), ROVCP=rcp,
! DELTSM=dtbl, DTMIN=dtbl/60 and RADIATION=(ra_lw_physics > 0).
!
! Input (stream, native little-endian):
!   int32 magic (= 5150), int32 ncase
!   per case: int32 ifsnow, int32 radiation(0/1), int32 nstep,
!             float32 deltsm,
!             float32 column(21) = t1 qv1 p1 flhc flqc psfc xland tmn hfx
!                                  qfx lh tsk qsfc chklowq gsw glw capg thc
!                                  snowc emiss mavail,
!             float32 tslb(5), zs(5), dzs(5)
! Output (stream):
!   int32 magic (= 5152), int32 ncase,
!   float32 constants(11) = svp1 svp2 svp3 svpt0 ep_2 karman eomeg stbolt
!                           xlv rcp p1000mb
!   per case, per step: float32 hfx qfx lh capg tsk qsfc chklowq tslb(1:5) flhc flqc
!
! The forcing (T/QV/P of the lowest level, FLHC, FLQC, PSFC, GSW, GLW, ...)
! is held for all steps of a case; the INOUT state (HFX, QFX, LH, CAPG, TSK,
! QSFC, CHKLOWQ, TSLB) carries from step to step exactly as WRF carries it.
program run_slab
  use module_model_constants, only: svp1, svp2, svp3, svpt0, ep_2, &
                                    karman, eomeg, stbolt, xlv, rcp, &
                                    p1000mb
  use module_sf_slab, only: slab
  implicit none
  integer, parameter :: nsl = 5
  integer :: magic, ncase, icase, istep, nstep, ifsnow, irad
  integer :: u_in, u_out
  real :: deltsm, dtmin
  real :: colv(21), tslb0(nsl), zs(nsl), dzs(nsl)
  real, dimension(1, 1, 1) :: t3d, qv3d, p3d
  real, dimension(1, 1) :: flhc, flqc, psfc, xland, tmn, hfx, qfx, lh, &
                           tsk, qsfc, chklowq, gsw, glw, capg, thc, &
                           snowc, emiss, mavail
  real :: tslb(1, nsl, 1)
  logical :: radiation
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u_in, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  open(newunit=u_out, file=trim(fout), access='stream', &
       form='unformatted', status='replace', action='write')
  read(u_in) magic, ncase
  if (magic /= 5150) stop 'bad input magic'
  write(u_out) 5152, ncase
  write(u_out) svp1, svp2, svp3, svpt0, ep_2, karman, eomeg, stbolt, &
               xlv, rcp, p1000mb

  do icase = 1, ncase
    read(u_in) ifsnow, irad, nstep, deltsm, colv, tslb0, zs, dzs
    radiation = (irad /= 0)
    dtmin = deltsm / 60.
    t3d(1, 1, 1) = colv(1)
    qv3d(1, 1, 1) = colv(2)
    p3d(1, 1, 1) = colv(3)
    flhc = colv(4)
    flqc = colv(5)
    psfc = colv(6)
    xland = colv(7)
    tmn = colv(8)
    hfx = colv(9)
    qfx = colv(10)
    lh = colv(11)
    tsk = colv(12)
    qsfc = colv(13)
    chklowq = colv(14)
    gsw = colv(15)
    glw = colv(16)
    capg = colv(17)
    thc = colv(18)
    snowc = colv(19)
    emiss = colv(20)
    mavail = colv(21)
    tslb(1, :, 1) = tslb0
    do istep = 1, nstep
      call slab(t3d, qv3d, p3d, flhc, flqc,                       &
                psfc, xland, tmn, hfx, qfx, lh, tsk, qsfc, chklowq, &
                gsw, glw, capg, thc, snowc, emiss, mavail,         &
                deltsm, rcp, xlv, dtmin, ifsnow,                   &
                svp1, svp2, svp3, svpt0, ep_2, karman, eomeg, stbolt, &
                tslb, zs, dzs, nsl, radiation,                     &
                p1000mb,                                           &
                1, 2, 1, 2, 1, 2,                                  &
                1, 1, 1, 1, 1, 1,                                  &
                1, 1, 1, 1, 1, 1)
      write(u_out) hfx(1, 1), qfx(1, 1), lh(1, 1), capg(1, 1), &
                   tsk(1, 1), qsfc(1, 1), chklowq(1, 1), tslb(1, :, 1), flhc(1, 1), flqc(1, 1)
    end do
  end do
  close(u_in)
  close(u_out)
end program run_slab
