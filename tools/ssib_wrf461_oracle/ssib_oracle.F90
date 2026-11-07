! WRF v4.6.1 SSiB column oracle driver.
!
! Calls the byte-unmodified module_sf_ssib.F (SSIB and SSIB_SEAICE) through a
! transcription of WRF's own CASE (SSIBSCHEME) arm of
! phys/module_surface_driver.F (lines 4156-4345 at d66e442), with
! FRACTIONAL_SEAICE = 1 (WRF refuses SSiB without it, :1483-1486), so
! XICE_THRESHOLD = 0.02 (:1490-1494) and isisfc = .TRUE. under
! sf_sfclay_physics = 1 (:1905-1915).
!
! Loop order is WRF's: every column at step n before any column at step
! n+1, so module-level SAVEd state (TEMRS2's DATA-initialised DELTH) sees
! exactly the call sequence a WRF tile does.
!
! I/O: little-endian float32 stream files.
!   columns.bin : int32 ncol, nstep, nz; then static(NC,ncol);
!                 state0(NS,ncol); then per step forcing(NF+nz,ncol)
!   out.bin     : per step state(NS,ncol) after the step
program ssib_oracle
  use module_sf_ssib, only : ssib, ssib_seaice
  implicit none
#include "layout.inc"
  character(len=512) :: inpath, outpath
  integer :: ncol, nstep, nz, icol, istep, k, iu, ou, ios
  real, allocatable :: cst(:,:), st(:,:), frc(:,:)
  character(len=32), parameter :: maps(4) = (/ 'SSIB                            ', &
       'USGS                            ', 'MODIS                           ', &
       'MODIFIED_IGBP_MODIS_NOAH        ' /)

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(newunit=iu, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian', iostat=ios)
  if (ios /= 0) stop 'cannot open input'
  read(iu) ncol, nstep, nz
  allocate(cst(NC, ncol), st(NS, ncol), frc(NF + nz, ncol))
  read(iu) cst
  read(iu) st
  open(newunit=ou, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  do istep = 1, nstep
     read(iu) frc
     do icol = 1, ncol
        call one_column(cst(:, icol), st(:, icol), frc(:, icol), nz, istep)
     end do
     write(ou) st
  end do
  close(iu)
  close(ou)

contains

  subroutine one_column(c, s, f, nz, itimestep)
    real, intent(in) :: c(NC), f(NF + nz)
    real, intent(inout) :: s(NS)
    integer, intent(in) :: nz, itimestep
    real :: xland, xlat, dtbl, julian_in, day, cloudfrac, uv10, zdiff
    real :: xice, xice_save, xice_threshold
    integer :: ivgtyp, isnow, ra_sw_physics, mminlu
    real :: u1, v1
    ! Driver-level scalars bound to SSIB's INOUT dummies.  Each is a
    ! distinct variable, exactly as each grid array element is in WRF.
    real :: smois1, smois2, smois3, tslb1, tslb2, tslb3, snow, sfcrunoff
    real :: canwat, hfx, lh, grdflx, qfx, tsk, ust, q2, t2, albedo
    real :: br, fm, fh, cm, lhf, shf, ghf, egs, eci, ect, egi, egt
    real :: sdn, sup, ldn, lup, wat, shc, shg, lai, vcf, z00, veg, xdd
    real :: swe, snowden, snowdepth, tkair
    real :: sl(12, 4)
    real :: alvd, alvf, alnd, alnf, svd, svf, snd, snf

    xland  = c(C_XLAND)
    ivgtyp = nint(c(C_IVGTYP))
    xlat   = c(C_XLAT)
    mminlu = nint(c(C_MMINLU))
    ra_sw_physics = nint(c(C_RA_SW))
    dtbl   = c(C_DTBL)
    xice   = f(F_XICE)
    xice_save = f(F_XICE_SAVE)
    julian_in = f(F_JULIAN)
    u1 = f(F_U1)
    v1 = f(F_V1)
    xice_threshold = 0.02          ! fractional_seaice == 1, :1492-1494

    ! The radiation-written SW components: RRTMG's values this step.
    s(S_SWVISDIR) = f(F_SWVISDIR)
    s(S_SWVISDIF) = f(F_SWVISDIF)
    s(S_SWNIRDIR) = f(F_SWNIRDIR)
    s(S_SWNIRDIF) = f(F_SWNIRDIF)

    smois1 = s(S_SMOIS1); smois2 = s(S_SMOIS2); smois3 = s(S_SMOIS3)
    tslb1 = s(S_TSLB1); tslb2 = s(S_TSLB2); tslb3 = s(S_TSLB3)
    snow = s(S_SNOW); sfcrunoff = s(S_SFCRUNOFF); canwat = s(S_CANWAT)
    hfx = s(S_HFX); lh = s(S_LH); grdflx = s(S_GRDFLX); qfx = s(S_QFX)
    tsk = s(S_TSK); ust = s(S_UST); q2 = s(S_Q2)
    t2 = f(F_T2_SFCLAY)        ! the surface layer's T2 of this call
    albedo = s(S_ALBEDO)
    br = s(S_SSIB_BR); fm = s(S_SSIB_FM); fh = s(S_SSIB_FH); cm = s(S_SSIB_CM)
    lhf = s(S_SSIB_LHF); shf = s(S_SSIB_SHF); ghf = s(S_SSIB_GHF)
    egs = s(S_SSIB_EGS); eci = s(S_SSIB_ECI); ect = s(S_SSIB_ECT)
    egi = s(S_SSIB_EGI); egt = s(S_SSIB_EGT)
    sdn = s(S_SSIB_SDN); sup = s(S_SSIB_SUP); ldn = s(S_SSIB_LDN)
    lup = s(S_SSIB_LUP); wat = s(S_SSIB_WAT); shc = s(S_SSIB_SHC)
    shg = s(S_SSIB_SHG); lai = s(S_SSIB_LAI); vcf = s(S_SSIB_VCF)
    z00 = s(S_SSIB_Z00); veg = s(S_SSIB_VEG); xdd = s(S_SSIBXDD)
    alvd = s(S_ALSWVISDIR); alvf = s(S_ALSWVISDIF)
    alnd = s(S_ALSWNIRDIR); alnf = s(S_ALSWNIRDIF)
    svd = s(S_SWVISDIR); svf = s(S_SWVISDIF)
    snd = s(S_SWNIRDIR); snf = s(S_SWNIRDIF)
    isnow = nint(s(S_ISNOW)); swe = s(S_SWE); snowden = s(S_SNOWDEN)
    snowdepth = s(S_SNOWDEPTH); tkair = s(S_TKAIR)
    do k = 1, 4
       sl(:, k) = s(S_DZO1 + 12*(k-1) : S_DZO1 + 12*(k-1) + 11)
    end do

    ! :4162-4172 -- ALBEDO from the blended value to the ice portion.
    IF ( ( XICE .GE. XICE_THRESHOLD ) .AND. ( XICE .LE. 1 ) ) THEN
       ALBEDO = (ALBEDO-(1.-XICE)*0.08)/XICE
    ENDIF

    ! :4208
    day = float(int(julian_in+0.01))+1.

    IF (XLAND .LT. 1.5) THEN
       ! :4224-4229 -- CLDFRA from the column (KMS..KME in WRF).
       CLOUDFRAC = 0.
       DO K = 1, nz
          CLOUDFRAC = AMAX1(CLOUDFRAC, AMIN1(f(NF + K), 1.0))
       ENDDO

       IF (XICE .GE. XICE_THRESHOLD) THEN
          CALL ssib_seaice                                                 &
               ( 1, 1, DTBL, itimestep, xlat, f(F_COSZEN),                 &
                 f(F_RAINNCV), f(F_RAINCV), f(F_GLW), f(F_DZ8W1),           &
                 smois1, smois2, smois3,                                   &
                 tslb1, tslb2, tslb3,                                      &
                 snow, sfcrunoff, xice_save,                               &
                 u1, v1, f(F_QV1), f(F_T1),                                &
                 f(F_P1), f(F_PSFC),                                       &
                 f(F_SWDOWN), canwat,                                      &
                 alvd, alvf, alnd, alnf,                                   &
                 svd, svf, snd, snf,                                       &
                 hfx, lh, grdflx, qfx, tsk,                                &
                 ust, br, fm, fh, cm,                                      &
                 lhf, shf, ghf,                                            &
                 sdn, sup, ldn, lup,                                       &
                 wat,                                                      &
                 z00, veg,                                                 &
                 day, cloudfrac, q2, t2, albedo, uv10,                     &
                 ra_sw_physics, xice_threshold )
       ELSE
          CALL ssib( 1, 1, DTBL, itimestep, xlat, f(F_COSZEN),             &
                 f(F_RAINNCV), f(F_RAINCV), f(F_GLW), f(F_DZ8W1),           &
                 smois1, smois2, smois3,                                   &
                 tslb1, tslb2, tslb3,                                      &
                 snow, sfcrunoff,                                          &
                 u1, v1, f(F_QV1), f(F_T1),                                &
                 f(F_P1), f(F_PSFC), ivgtyp,                               &
                 f(F_SWDOWN), canwat,                                      &
                 alvd, alvf, alnd, alnf,                                   &
                 svd, svf, snd, snf,                                       &
                 hfx, lh, grdflx, qfx, tsk,                                &
                 ust, br, fm, fh, cm,                                      &
                 lhf, shf, ghf, egs,                                       &
                 eci, ect, egi, egt,                                       &
                 sdn, sup, ldn, lup,                                       &
                 wat, shc, shg, lai,                                       &
                 vcf, z00, veg, xdd,                                       &
                 isnow, swe, snowden, snowdepth, tkair,                    &
                 sl(1,1), sl(2,1), sl(3,1), sl(4,1), sl(5,1), sl(6,1),     &
                 sl(7,1), sl(8,1), sl(9,1), sl(10,1), sl(11,1), sl(12,1),  &
                 sl(1,2), sl(2,2), sl(3,2), sl(4,2), sl(5,2), sl(6,2),     &
                 sl(7,2), sl(8,2), sl(9,2), sl(10,2), sl(11,2), sl(12,2),  &
                 sl(1,3), sl(2,3), sl(3,3), sl(4,3), sl(5,3), sl(6,3),     &
                 sl(7,3), sl(8,3), sl(9,3), sl(10,3), sl(11,3), sl(12,3),  &
                 sl(1,4), sl(2,4), sl(3,4), sl(4,4), sl(5,4), sl(6,4),     &
                 sl(7,4), sl(8,4), sl(9,4), sl(10,4), sl(11,4), sl(12,4),  &
                 day, cloudfrac, q2, t2, albedo, uv10,                     &
                 ra_sw_physics, trim(maps(mminlu)) )
       ENDIF

       ! :4286-4301
       s(S_BR) = br
       s(S_ZNT) = z00
       s(S_SFCEVP) = s(S_SFCEVP) + QFX*DTBL
       t2 = tsk
       IF (itimestep .ne. 1) THEN
          ZDIFF = (0.5*f(F_DZ8W1)) - xdd
          IF (ZDIFF .LE. s(S_ZNT)) ZDIFF = s(S_ZNT)+0.2
          s(S_GZ1OZ0) = ALOG(ZDIFF/s(S_ZNT))
       ENDIF
       IF (XICE .GE. XICE_THRESHOLD) THEN
          s(S_SNOWH) = 0.0
       ELSE
          s(S_SNOWH) = snowdepth
       ENDIF
       s(S_U10) = UV10*u1/SQRT(u1**2+v1**2)
       s(S_V10) = UV10*v1/SQRT(u1**2+v1**2)
    ENDIF

    ! :4311-4345 (FRACTIONAL_SEAICE == 1, isisfc true)
    IF ( ( XICE .GE. XICE_THRESHOLD) .AND. ( XICE .LE. 1.0 ) ) THEN
       albedo = ( albedo * XICE ) + ( (1.0-XICE) * 0.08 )
    ENDIF
    IF ( ( XICE .GE. XICE_THRESHOLD) .AND. ( XICE .LE. 1.0 ) ) THEN
       hfx = ( hfx * XICE ) + ( (1.0-XICE) * f(F_HFX_SEA) )
       qfx = ( qfx * XICE ) + ( (1.0-XICE) * f(F_QFX_SEA) )
       lh  = ( lh  * XICE ) + ( (1.0-XICE) * f(F_LH_SEA) )
       s(S_TSK_SAVE) = tsk
       tsk = ( tsk * XICE ) + ( (1.0-XICE) * f(F_TSK_SEA) )
    ENDIF

    s(S_SMOIS1) = smois1; s(S_SMOIS2) = smois2; s(S_SMOIS3) = smois3
    s(S_TSLB1) = tslb1; s(S_TSLB2) = tslb2; s(S_TSLB3) = tslb3
    s(S_SNOW) = snow; s(S_SFCRUNOFF) = sfcrunoff; s(S_CANWAT) = canwat
    s(S_HFX) = hfx; s(S_LH) = lh; s(S_GRDFLX) = grdflx; s(S_QFX) = qfx
    s(S_TSK) = tsk; s(S_UST) = ust; s(S_Q2) = q2; s(S_T2) = t2
    s(S_ALBEDO) = albedo
    s(S_SSIB_BR) = br; s(S_SSIB_FM) = fm; s(S_SSIB_FH) = fh; s(S_SSIB_CM) = cm
    s(S_SSIB_LHF) = lhf; s(S_SSIB_SHF) = shf; s(S_SSIB_GHF) = ghf
    s(S_SSIB_EGS) = egs; s(S_SSIB_ECI) = eci; s(S_SSIB_ECT) = ect
    s(S_SSIB_EGI) = egi; s(S_SSIB_EGT) = egt
    s(S_SSIB_SDN) = sdn; s(S_SSIB_SUP) = sup; s(S_SSIB_LDN) = ldn
    s(S_SSIB_LUP) = lup; s(S_SSIB_WAT) = wat; s(S_SSIB_SHC) = shc
    s(S_SSIB_SHG) = shg; s(S_SSIB_LAI) = lai; s(S_SSIB_VCF) = vcf
    s(S_SSIB_Z00) = z00; s(S_SSIB_VEG) = veg; s(S_SSIBXDD) = xdd
    s(S_ALSWVISDIR) = alvd; s(S_ALSWVISDIF) = alvf
    s(S_ALSWNIRDIR) = alnd; s(S_ALSWNIRDIF) = alnf
    s(S_SWVISDIR) = svd; s(S_SWVISDIF) = svf
    s(S_SWNIRDIR) = snd; s(S_SWNIRDIF) = snf
    s(S_ISNOW) = real(isnow); s(S_SWE) = swe; s(S_SNOWDEN) = snowden
    s(S_SNOWDEPTH) = snowdepth; s(S_TKAIR) = tkair
    do k = 1, 4
       s(S_DZO1 + 12*(k-1) : S_DZO1 + 12*(k-1) + 11) = sl(:, k)
    end do
  end subroutine one_column

end program ssib_oracle
