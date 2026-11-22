! Module replay of WRF v4.8.0 MYNN-SFC (b66187f5) at its two surface-driver
! call sites, linked against the referee build's own objects (libwrflib.a of
! WRF-4.8.0-serial-pristine, strict recipe: GNU Fortran 13.3.0, -O2
! -fno-fast-math -ffp-contract=off -fno-tree-vectorize).  Nothing here
! computes physics: it reads captured actual arguments, calls
! mynnsfc_init + mynnsfc_driver (fractional_seaice = 0,
! module_surface_driver.F:2462-2486) or mynn_seaice_wrapper
! (fractional_seaice = 1, :2430-2460), and writes every INOUT/OUT field.
!
! File layout (stream, little-endian, float32 unless noted):
!   int32 header(12): nx ny mode itimestep isfflx spp_pbl land water shal
!                     flag_lsm restart tice2tsk_if2cold ; float32 xice_threshold
!   3-D fields (nx,2,ny) in N3 order, then 2-D fields (nx,ny) in N2 order.
! Output: the 2-D fields again, in N2 order, after the call.
program replay_mynnsfc
  use module_sf_mynnsfc_driver, only: mynnsfc_driver, mynnsfc_init
  use module_surface_driver, only: mynn_seaice_wrapper
  implicit none
  integer, parameter :: n3 = 9, n2 = 57
  integer :: hdr(12), nx, ny, mode, itimestep, isfflx, spp_pbl, land, water
  integer :: shal, flag_lsm, errflg, k
  logical :: restart, tice2tsk
  real :: xice_threshold
  real, allocatable :: a3(:,:,:,:), a2(:,:,:)
  character(len=256) :: errmsg, fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', &
       convert='little_endian', status='old')
  read(10) hdr, xice_threshold
  nx = hdr(1); ny = hdr(2); mode = hdr(3); itimestep = hdr(4)
  isfflx = hdr(5); spp_pbl = hdr(6); land = hdr(7); water = hdr(8)
  shal = hdr(9); flag_lsm = hdr(10); restart = hdr(11) /= 0
  tice2tsk = hdr(12) /= 0
  allocate(a3(nx,2,ny,n3), a2(nx,ny,n2))
  do k = 1, n3
     read(10) a3(:,:,:,k)
  end do
  do k = 1, n2
     read(10) a2(:,:,k)
  end do
  close(10)

  errmsg = ' '
  errflg = 0
  call mynnsfc_init(.true., errmsg, errflg)
  if (errflg /= 0) stop 'mynnsfc_init'

! N3: 1 u 2 v 3 t 4 qv 5 p 6 dz8w 7 th 8 rho 9 pattern_spp_pbl
! N2: 1 psfc 2 mavail 3 xland 4 tsk 5 snowh 6 xice 7 dx 8 water_depth
!     9 lakemask 10 sst 11 chs 12 chs2 13 cqs 14 cqs2 15 cpm 16 znt 17 ust
!     18 pblh 19 zol 20 mol 21 psim 22 psih 23 qgh 24 hfx 25 qfx 26 lh
!     27 flhc 28 flqc 29 qsfc 30 rmol 31 gz1oz0 32 wspd 33 br 34 ch 35 ustm
!     36 ck 37 cka 38 cd 39 cda 40 u10 41 v10 42 th2 43 t2 44 q2
!     45 tsk_sea 46 qgh_sea 47 chs2_sea 48 chs_sea 49 cpm_sea 50 cqs2_sea
!     51 flhc_sea 52 flqc_sea 53 hfx_sea 54 lh_sea 55 qfx_sea 56 qsfc_sea
!     57 znt_sea
  if (mode == 0) then
     call mynnsfc_driver(u3d=a3(:,:,:,1), v3d=a3(:,:,:,2), t3d=a3(:,:,:,3), &
          qv3d=a3(:,:,:,4), p3d=a3(:,:,:,5), dz8w=a3(:,:,:,6),             &
          th3d=a3(:,:,:,7), rho3d=a3(:,:,:,8), psfcpa=a2(:,:,1),           &
          chs=a2(:,:,11), chs2=a2(:,:,12), cqs=a2(:,:,13), cqs2=a2(:,:,14), &
          cpm=a2(:,:,15), znt=a2(:,:,16), ust=a2(:,:,17), pblh=a2(:,:,18), &
          mavail=a2(:,:,2), zol=a2(:,:,19), mol=a2(:,:,20),                &
          psim=a2(:,:,21), psih=a2(:,:,22), qgh=a2(:,:,23),                &
          xland=a2(:,:,3), hfx=a2(:,:,24), qfx=a2(:,:,25), lh=a2(:,:,26),  &
          tsk=a2(:,:,4), flhc=a2(:,:,27), flqc=a2(:,:,28), qsfc=a2(:,:,29),&
          rmol=a2(:,:,30), u10=a2(:,:,40), v10=a2(:,:,41), th2=a2(:,:,42), &
          t2=a2(:,:,43), q2=a2(:,:,44), snowh=a2(:,:,5),                   &
          gz1oz0=a2(:,:,31), wspd=a2(:,:,32), br=a2(:,:,33), dx=a2(:,:,7), &
          itimestep=itimestep, ch=a2(:,:,34),                              &
          spp_pbl=spp_pbl, pattern_spp_pbl=a3(:,:,:,9),                    &
          ids=1, ide=nx+1, jds=1, jde=ny+1, kds=1, kde=3,                  &
          ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=2,                      &
          its=1, ite=nx, jts=1, jte=ny, kts=1, kte=1,                      &
          ustm=a2(:,:,35), ck=a2(:,:,36), cka=a2(:,:,37), cd=a2(:,:,38),   &
          cda=a2(:,:,39), xice=a2(:,:,6), xice_threshold=xice_threshold,   &
          sf_mynn_sfcflux_land=land, sf_mynn_sfcflux_water=water,          &
          isfflx=isfflx, shalwater_z0=shal, water_depth=a2(:,:,8),         &
          lakemask=a2(:,:,9), flag_lsm=flag_lsm, restart=restart,          &
          errmsg=errmsg, errflg=errflg)
  else
     call mynn_seaice_wrapper(                                             &
          u3d=a3(:,:,:,1), v3d=a3(:,:,:,2), t3d=a3(:,:,:,3),               &
          qv3d=a3(:,:,:,4), p3d=a3(:,:,:,5), dz8w=a3(:,:,:,6),             &
          th3d=a3(:,:,:,7), rho=a3(:,:,:,8), psfc=a2(:,:,1),               &
          chs=a2(:,:,11), chs2=a2(:,:,12), cqs=a2(:,:,13), cqs2=a2(:,:,14), &
          cpm=a2(:,:,15), znt=a2(:,:,16), ust=a2(:,:,17), pblh=a2(:,:,18), &
          mavail=a2(:,:,2), zol=a2(:,:,19), mol=a2(:,:,20),                &
          psim=a2(:,:,21), psih=a2(:,:,22), qgh=a2(:,:,23),                &
          xland=a2(:,:,3), hfx=a2(:,:,24), qfx=a2(:,:,25), lh=a2(:,:,26),  &
          tsk=a2(:,:,4), flhc=a2(:,:,27), flqc=a2(:,:,28), qsfc=a2(:,:,29),&
          rmol=a2(:,:,30), u10=a2(:,:,40), v10=a2(:,:,41), th2=a2(:,:,42), &
          t2=a2(:,:,43), q2=a2(:,:,44), snowh=a2(:,:,5),                   &
          gz1oz0=a2(:,:,31), wspd=a2(:,:,32), br=a2(:,:,33), dx=a2(:,:,7), &
          itimestep=itimestep, ch=a2(:,:,34),                              &
          spp_pbl=spp_pbl, pattern_spp_pbl=a3(:,:,:,9),                    &
          xice=a2(:,:,6), sst=a2(:,:,10), tsk_sea=a2(:,:,45),              &
          qgh_sea=a2(:,:,46), chs2_sea=a2(:,:,47), chs_sea=a2(:,:,48),     &
          cpm_sea=a2(:,:,49), cqs2_sea=a2(:,:,50), flhc_sea=a2(:,:,51),    &
          flqc_sea=a2(:,:,52), hfx_sea=a2(:,:,53), lh_sea=a2(:,:,54),      &
          qfx_sea=a2(:,:,55), qsfc_sea=a2(:,:,56), znt_sea=a2(:,:,57),     &
          tice2tsk_if2cold=tice2tsk, xice_threshold=xice_threshold,        &
          ids=1, ide=nx+1, jds=1, jde=ny+1, kds=1, kde=3,                  &
          ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=2,                      &
          its=1, ite=nx, jts=1, jte=ny, kts=1, kte=1,                      &
          ustm=a2(:,:,35), ck=a2(:,:,36), cka=a2(:,:,37), cd=a2(:,:,38),   &
          cda=a2(:,:,39), sf_mynn_sfcflux_land=land,                       &
          sf_mynn_sfcflux_water=water, isfflx=isfflx, shalwater_z0=shal,   &
          water_depth=a2(:,:,8), lakemask=a2(:,:,9), flag_lsm=flag_lsm,    &
          restart=restart, errmsg=errmsg, errflg=errflg)
  end if
  if (errflg /= 0) then
     print *, 'errflg=', errflg, trim(errmsg)
  end if

  open(11, file=trim(fout), access='stream', form='unformatted', &
       convert='little_endian', status='replace')
  do k = 1, n2
     write(11) a2(:,:,k)
  end do
  close(11)
end program replay_mynnsfc
