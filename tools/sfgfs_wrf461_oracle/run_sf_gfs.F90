! Column driver for the WRF v4.6.1 GFS surface layer (sf_sfclay_physics=3).
!
!   run_sf_gfs CASES_BIN OUT_DIR
!
! Reads the cases make_cases.py wrote, then drives the byte-unmodified
! phys/module_sf_gfs.F through four arms, carrying the scheme's state (ZNT,
! UST and, for the wrapper arms, SST and the *_SEA carriers) from step to
! step exactly as WRF's grid arrays carry it:
!
!   armA.bin  SF_GFS called directly (FRACTIONAL_SEAICE = 0), ISFFLX = 1
!   armB.bin  sf_gfs_seaice_wrapper as WRF v4.6.1 ships it (XLAND to the
!             open-water call), XICE_THRESHOLD = 0.02
!   armC.bin  the same wrapper with the open-water call given XLAND_SEA
!   armD.bin  SF_GFS called directly with ISFFLX = 0
!
! Each arm file is float32 [nstep][nfield][ncol] in the field order written
! to fields.txt.  ITIMESTEP runs 1..nstep (WRF's one-based NTSD), so the
! wrapper's ITIMESTEP <= 3 SST ladder is reached on the first three steps.
!
! It also writes the saturation-pressure function GFS_SFC reads:
!   fpvs_table.bin   the 7501 binary64 words of tbpvs plus c1xpvs, c2xpvs,
!                    copied out of module_gfs_funcphys's own storage after
!                    GFUNCPHYS ran (read_tbpvs.c)
!   fpvs_sweep.bin   int32 n, float32 t[n], float64 fpvs(t)[n] over 150 K to
!                    350 K, both table clamps included
program run_sf_gfs
  use, intrinsic :: iso_c_binding, only: c_double
  use module_model_constants, only: cp, rcp, r_d, xlv, ep_1, ep_2, karman
  use module_sf_gfs
  use module_gfs_funcphys, only: gfuncphys, fpvs
  use sfgfs_oracle_wrapper
  implicit none

  interface
     subroutine sfgfs_read_tbpvs(table, c1, c2) bind(C, name="sfgfs_read_tbpvs")
       import :: c_double
       real(c_double), intent(out) :: table(7501), c1, c2
     end subroutine
  end interface

  character(len=1024) :: cases_path, out_dir
  integer :: header(5), magic, ncol, nstep, nin, ninit
  real, allocatable :: per_step(:,:,:), initial(:,:)
  integer :: arm, step, i, u
  real(c_double) :: table(7501), c1, c2
  integer, parameter :: nsweep = 20001
  real :: tsweep(nsweep)
  real(c_double) :: psweep(nsweep)

  integer, parameter :: ndirect = 20, nwrap = 35

  if (command_argument_count() /= 2) then
     write(0,*) 'usage: run_sf_gfs CASES_BIN OUT_DIR'
     stop 2
  end if
  call get_command_argument(1, cases_path)
  call get_command_argument(2, out_dir)

  open(newunit=u, file=trim(cases_path), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(u) header
  magic = header(1); ncol = header(2); nstep = header(3)
  nin = header(4); ninit = header(5)
  if (magic /= int(z'47465331') .or. nin /= 9 .or. ninit /= 3) then
     write(0,*) 'cases file has an unexpected header', header
     stop 3
  end if
  allocate(per_step(ncol, nin, nstep), initial(ncol, ninit))
  read(u) per_step
  read(u) initial
  close(u)

  call write_fields_txt()
  do arm = 1, 4
     call run_arm(arm)
  end do

  ! The table GFS_SFC interpolates, exactly as GFUNCPHYS left it.
  call gfuncphys()
  call sfgfs_read_tbpvs(table, c1, c2)
  open(newunit=u, file=trim(out_dir)//'/fpvs_table.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')
  write(u) table, c1, c2
  close(u)
  do i = 1, nsweep
     tsweep(i) = real(150.0d0 + 0.01d0 * dble(i - 1))
     psweep(i) = fpvs(dble(tsweep(i)))
  end do
  open(newunit=u, file=trim(out_dir)//'/fpvs_sweep.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')
  write(u) nsweep, tsweep, psweep
  close(u)

contains

  subroutine write_fields_txt()
    integer :: v
    open(newunit=v, file=trim(out_dir)//'/fields.txt', status='replace')
    write(v,'(a)') 'direct br chs chs2 cpm cqs2 flhc flqc gz1oz0 hfx lh psim psih qfx qgh qsfc u10 v10 wspd znt ust'
    write(v,'(a)') 'wrapper br chs chs2 cpm cqs2 flhc flqc gz1oz0 hfx lh psim psih qfx qgh qsfc u10 v10 wspd znt ust &
         &chs_sea chs2_sea cpm_sea cqs2_sea flhc_sea flqc_sea hfx_sea lh_sea qfx_sea qgh_sea qsfc_sea ust_sea znt_sea sst tsk_sea'
    close(v)
  end subroutine

  subroutine run_arm(arm)
    integer, intent(in) :: arm
    integer, parameter :: kms = 1, kme = 1, kts = 1, kte = 1
    real, dimension(ncol, kms:kme, 1) :: u3d, v3d, t3d, qv3d, p3d
    real, dimension(ncol, 1) :: psfc, tsk, xland, xice, znt, ust, sst
    real, dimension(ncol, 1) :: br, chs, chs2, cpm, cqs2, flhc, flqc, gz1oz0, &
         hfx, lh, psim, psih, qfx, qgh, qsfc, u10, v10, wspd
    real, dimension(ncol, 1) :: chs_sea, chs2_sea, cpm_sea, cqs2_sea, flhc_sea, &
         flqc_sea, hfx_sea, lh_sea, qfx_sea, qgh_sea, qsfc_sea, ust_sea, &
         znt_sea, tsk_sea
    real, allocatable :: outbuf(:,:)
    character(len=8) :: name
    integer :: v, isfflx, nout

    znt(:,1) = initial(:,1)
    ust(:,1) = initial(:,2)
    sst(:,1) = initial(:,3)
    chs_sea = 0.; chs2_sea = 0.; cpm_sea = 0.; cqs2_sea = 0.
    flhc_sea = 0.; flqc_sea = 0.; hfx_sea = 0.; lh_sea = 0.
    qfx_sea = 0.; qgh_sea = 0.; qsfc_sea = 0.; ust_sea = 0.; znt_sea = 0.
    tsk_sea = 0.
    isfflx = 1
    if (arm == 4) isfflx = 0
    nout = ndirect
    if (arm == 2 .or. arm == 3) nout = nwrap
    allocate(outbuf(ncol, nout))

    write(name,'(a,a1,a)') 'arm', achar(iachar('A') + arm - 1), ''
    open(newunit=v, file=trim(out_dir)//'/'//trim(name)//'.bin', access='stream', &
         form='unformatted', status='replace', convert='little_endian')

    do step = 1, nstep
       u3d(:,1,1)  = per_step(:,1,step)
       v3d(:,1,1)  = per_step(:,2,step)
       t3d(:,1,1)  = per_step(:,3,step)
       qv3d(:,1,1) = per_step(:,4,step)
       p3d(:,1,1)  = per_step(:,5,step)
       psfc(:,1)   = per_step(:,6,step)
       tsk(:,1)    = per_step(:,7,step)
       xland(:,1)  = per_step(:,8,step)
       xice(:,1)   = per_step(:,9,step)
       if (arm == 1 .or. arm == 4) then
          call SF_GFS(u3d, v3d, t3d, qv3d, p3d,                           &
               cp, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,             &
               znt, ust, psim, psih,                                      &
               xland, hfx, qfx, lh, tsk, flhc, flqc,                      &
               qgh, qsfc, u10, v10,                                       &
               gz1oz0, wspd, br, isfflx,                                  &
               ep_1, ep_2, karman, step,                                  &
               1, ncol, 1, 1, kms, kme,                                   &
               1, ncol, 1, 1, kms, kme,                                   &
               1, ncol, 1, 1, kts, kte)
       else
          call sf_gfs_seaice_wrapper(u3d, v3d, t3d, qv3d, p3d,            &
               cp, rcp, r_d, xlv, psfc, chs, chs2, cqs2, cpm,             &
               znt, ust, psim, psih,                                      &
               xland, hfx, qfx, lh, tsk, flhc, flqc,                      &
               qgh, qsfc, u10, v10,                                       &
               gz1oz0, wspd, br, isfflx,                                  &
               ep_1, ep_2, karman, step,                                  &
               .false., 0.02,                                             &
               chs_sea, chs2_sea, cpm_sea, cqs2_sea,                      &
               flhc_sea, flqc_sea,                                        &
               hfx_sea, lh_sea, qfx_sea, qgh_sea, qsfc_sea,               &
               ust_sea, znt_sea, sst, xice,                               &
               1, ncol, 1, 1, kms, kme,                                   &
               1, ncol, 1, 1, kms, kme,                                   &
               1, ncol, 1, 1, kts, kte,                                   &
               arm == 3, tsk_sea)
       end if
       outbuf(:,1) = br(:,1);    outbuf(:,2) = chs(:,1)
       outbuf(:,3) = chs2(:,1);  outbuf(:,4) = cpm(:,1)
       outbuf(:,5) = cqs2(:,1);  outbuf(:,6) = flhc(:,1)
       outbuf(:,7) = flqc(:,1);  outbuf(:,8) = gz1oz0(:,1)
       outbuf(:,9) = hfx(:,1);   outbuf(:,10) = lh(:,1)
       outbuf(:,11) = psim(:,1); outbuf(:,12) = psih(:,1)
       outbuf(:,13) = qfx(:,1);  outbuf(:,14) = qgh(:,1)
       outbuf(:,15) = qsfc(:,1); outbuf(:,16) = u10(:,1)
       outbuf(:,17) = v10(:,1);  outbuf(:,18) = wspd(:,1)
       outbuf(:,19) = znt(:,1);  outbuf(:,20) = ust(:,1)
       if (nout == nwrap) then
          outbuf(:,21) = chs_sea(:,1);  outbuf(:,22) = chs2_sea(:,1)
          outbuf(:,23) = cpm_sea(:,1);  outbuf(:,24) = cqs2_sea(:,1)
          outbuf(:,25) = flhc_sea(:,1); outbuf(:,26) = flqc_sea(:,1)
          outbuf(:,27) = hfx_sea(:,1);  outbuf(:,28) = lh_sea(:,1)
          outbuf(:,29) = qfx_sea(:,1);  outbuf(:,30) = qgh_sea(:,1)
          outbuf(:,31) = qsfc_sea(:,1); outbuf(:,32) = ust_sea(:,1)
          outbuf(:,33) = znt_sea(:,1);  outbuf(:,34) = sst(:,1)
          outbuf(:,35) = tsk_sea(:,1)
       end if
       write(v) outbuf
    end do
    close(v)
    deallocate(outbuf)
  end subroutine

end program run_sf_gfs
