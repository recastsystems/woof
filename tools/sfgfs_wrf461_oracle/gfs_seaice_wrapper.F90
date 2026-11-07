! The FRACTIONAL_SEAICE=1 arm of WRF v4.6.1's GFS surface layer, for the
! column oracle.
!
! sf_gfs_seaice_wrapper and get_local_ice_tsk live in
! phys/module_surface_driver.F (:5564-5815 and :7117-7208), a module that
! drags in the whole WRF frame, so the oracle carries their bodies here
! rather than linking the driver.  Both bodies are WRF's statements in WRF's
! order; build.sh pins module_surface_driver.F's sha256 so a reader can diff
! these lines against the pinned file.  Two additions, both marked ORACLE:
!
!  * xland_sea_fix selects the mask of the open-water call.  WRF v4.6.1
!    builds XLAND_SEA (=2 on sea-ice cells, :5751) and then hands the
!    open-water SF_GFS call XLAND instead (:5772), so on a sea-ice cell
!    (XLAND=1 after real.exe) the "open water" component is computed as
!    land: no Charnock roughness, no Zeng thermal roughness, and the land
!    -250 W m-2 HFX floor.  Every other *_seaice_wrapper in the same file
!    passes XLAND_SEA.  .false. reproduces the shipped statement; .true.
!    is the repaired call WOOF runs, and the oracle records both so the
!    divergence is measured, not asserted.
!  * tsk_sea_out returns the wrapper-local TSK_SEA.  WRF's GFS wrapper keeps
!    TSK_SEA local (:5670) and the surface driver's own TSK_SEA, which the
!    post-RUC blend reads (:3561), is never written under this scheme --
!    every other wrapper takes TSK_SEA as an argument.  The oracle exports
!    the value the repaired wrapper publishes.

module sfgfs_oracle_wrapper
  use module_sf_gfs
  implicit none
contains

   SUBROUTINE get_local_ice_tsk ( ims, ime, jms, jme,     &
                                  its, ite, jts, jte,     &
                                  itimestep,              &
                                  sfc_layer_values,       &
                                  tice2tsk_if2cold,       &
                                  XICE, XICE_THRESHOLD,   &
                                  SST, TSK, TSK_SEA, TSK_ICE )
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: ims, ime, jms, jme
      INTEGER, INTENT(IN) :: its, ite, jts, jte
      INTEGER, INTENT(IN) :: itimestep
      LOGICAL, INTENT(IN) :: sfc_layer_values
      LOGICAL, INTENT(IN) :: tice2tsk_if2cold
      REAL, DIMENSION( ims:ime , jms:jme ), INTENT(IN)    :: XICE
      REAL                                , INTENT(IN)    :: XICE_THRESHOLD
      REAL, DIMENSION( ims:ime , jms:jme ), INTENT(IN)    :: TSK
      REAL, DIMENSION( ims:ime , jms:jme ), INTENT(INOUT) :: SST
      REAL, DIMENSION( ims:ime , jms:jme ), INTENT(OUT)   :: TSK_SEA
      REAL, DIMENSION( ims:ime , jms:jme ), INTENT(OUT)   :: TSK_ICE
      INTEGER :: i,j
      REAL    :: TICE_MIN

      TICE_MIN = 221.4

      DO j = JTS , JTE
         DO i = ITS , ITE
            IF ( ( XICE(i,j) >= XICE_THRESHOLD ) .AND. ( XICE(I,J) <= 1.0 ) ) THEN
               IF ( SST(i,j) < 271.4 ) THEN
                  SST(i,j) = 271.4
               ENDIF
               IF (sfc_layer_values) THEN
                  IF ( SST(i,j) > 273. .AND. itimestep <= 3) then
                     IF ( XICE(i,j) >= 0.6 ) THEN
                        SST(i,j) = 271.4
                     ELSEIF ( XICE(i,j) >= 0.4 ) THEN
                        SST(i,j) = 273.
                     ELSEIF (XICE(i,j) >= 0.2 .AND. SST(i,j) > 275.) THEN
                        SST(i,j) = 275.
                     ELSEIF (SST(i,j) > 278.) THEN
                        SST(i,j) = 278.
                     ENDIF
                  ENDIF
               ENDIF
               TSK_SEA(i,j) = SST(i,j)
               IF ( tice2tsk_if2cold ) THEN
                  TSK_ICE(i,j) = MIN( TSK(i,j), 273.15 )
               ELSE
                  TSK_ICE(i,j) = ( TSK(i,j) - (1.0-XICE(i,j)) * SST(i,j) ) / XICE(i,j)
                  IF ( TSK_ICE(i,j) < TICE_MIN ) TSK_ICE(i,j) = TICE_MIN
               ENDIF
               IF ( ( XICE(i,j) < 0.2 ) .AND. ( TSK(i,j) < 253.15 ) ) THEN
                  TSK_ICE(i,j) = 253.15
               ENDIF
               IF ( ( XICE(i,j) < 0.1 ) .AND. ( TSK(i,j) < 263.15 ) ) THEN
                  TSK_ICE(i,j) = 263.15
               ENDIF
            ELSE
               TSK_SEA(i,j) = TSK(i,j)
               TSK_ICE(i,j) = TSK(i,j)
            ENDIF
         ENDDO
      ENDDO
   END SUBROUTINE get_local_ice_tsk

   SUBROUTINE sf_gfs_seaice_wrapper(U3D,V3D,T3D,QV3D,P3D,        &
                     CP,ROVCP,R,XLV,PSFC,CHS,CHS2,CQS2,CPM,          &
                     ZNT,UST,PSIM,PSIH,                          &
                     XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,             &
                     QGH,QSFC,U10,V10,                           &
                     GZ1OZ0,WSPD,BR,ISFFLX,                      &
                     EP1,EP2,KARMAN,itimestep,                   &
                     TICE2TSK_IF2COLD,                           &
                     XICE_THRESHOLD,                             &
                     CHS_SEA, CHS2_SEA, CPM_SEA, CQS2_SEA,       &
                     FLHC_SEA, FLQC_SEA,                         &
                     HFX_SEA, LH_SEA, QFX_SEA, QGH_SEA, QSFC_SEA,&
                     UST_SEA, ZNT_SEA, SST, XICE,                &
                     ids,ide, jds,jde, kds,kde,                  &
                     ims,ime, jms,jme, kms,kme,                  &
                     its,ite, jts,jte, kts,kte,                  &
                     xland_sea_fix, tsk_sea_out                  )
     implicit none
     INTEGER, INTENT(IN) ::             ids,ide, jds,jde, kds,kde,      &
                                        ims,ime, jms,jme, kms,kme,      &
                                        its,ite, jts,jte, kts,kte,      &
                                        ISFFLX,itimestep
      REAL,    INTENT(IN) :: CP, EP1, EP2, KARMAN, R, ROVCP, XLV
      REAL,    DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(IN) ::      &
                                        P3D, QV3D, T3D, U3D, V3D
      REAL,    DIMENSION(ims:ime, jms:jme), INTENT(IN) :: TSK, PSFC, XLAND
      REAL,    DIMENSION(ims:ime, jms:jme), INTENT(INOUT) :: UST, ZNT
      REAL, DIMENSION(ims:ime, jms:jme), INTENT(OUT) ::                 &
                                        BR, CHS, CHS2, CPM, CQS2, FLHC, &
                                        FLQC, GZ1OZ0, HFX, LH, PSIM,    &
                                        PSIH, QFX, QGH, QSFC, U10, V10, &
                                        WSPD
      REAL, DIMENSION(ims:ime, jms:jme), INTENT(IN) :: XICE
      REAL, DIMENSION(ims:ime, jms:jme), INTENT(OUT) ::                 &
                                        CHS_SEA, CHS2_SEA, CPM_SEA,     &
                                        CQS2_SEA, FLHC_SEA, FLQC_SEA,   &
                                        HFX_SEA, LH_SEA, QFX_SEA,       &
                                        QGH_SEA, QSFC_SEA, UST_SEA,     &
                                        ZNT_SEA
      REAL, DIMENSION(ims:ime, jms:jme), INTENT(INOUT) :: SST
      REAL,                              INTENT(IN)    :: XICE_THRESHOLD
      LOGICAL,                          INTENT(IN)     :: TICE2TSK_IF2COLD
      LOGICAL,                          INTENT(IN)     :: xland_sea_fix  ! ORACLE
      REAL, DIMENSION(ims:ime, jms:jme), INTENT(OUT)   :: tsk_sea_out   ! ORACLE

      INTEGER :: I
      INTEGER :: J
      REAL, DIMENSION(ims:ime, jms:jme) ::                              &
                                        BR_SEA, GZ1OZ0_SEA, PSIM_SEA,   &
                                        PSIH_SEA, U10_SEA, V10_SEA,     &
                                        WSPD_SEA, XLAND_SEA, TSK_SEA,   &
                                        UST_HOLD, ZNT_HOLD, TSK_LOCAL
      REAL, DIMENSION(ims:ime, jms:jme) :: XLAND_CALL2                   ! ORACLE

      CALL get_local_ice_tsk( ims, ime, jms, jme, its, ite, jts, jte,  &
                              itimestep, .true., tice2tsk_if2cold,     &
                              XICE, XICE_THRESHOLD,                    &
                              SST, TSK, TSK_SEA, TSK_LOCAL )

     ZNT_HOLD = ZNT
     UST_HOLD = UST

     CALL SF_GFS(U3D,V3D,T3D,QV3D,P3D,                  &
          CP,ROVCP,R,XLV,PSFC,CHS,CHS2,CQS2,CPM_SEA,    &
          ZNT,UST,PSIM,PSIH,                            &
          XLAND,HFX,QFX,LH,TSK_LOCAL,FLHC,FLQC,         &
          QGH,QSFC,U10,V10,                             &
          GZ1OZ0,WSPD,BR,ISFFLX,                        &
          EP1,EP2,KARMAN,ITIMESTEP,                     &
          ids,ide, jds,jde, kds,kde,                    &
          ims,ime, jms,jme, kms,kme,                    &
          its,ite, jts,jte, kts,kte                     )

     DO j = JTS , JTE
        DO i = ITS , ITE
           IF ( ( XICE(I,J) .GE. XICE_THRESHOLD ) .and. ( XICE(i,j) .LE. 1.0 ) ) THEN
              XLAND_SEA(i,j)=2.
              ZNT_SEA(I,J) = 0.0001
              IF ( SST(i,j) .LT. 271.4 ) THEN
                 SST(i,j) = 271.4
              ENDIF
              TSK_SEA(i,j) = SST(i,j)
           ELSE
              XLAND_SEA(i,j)=xland(i,j)
              ZNT_SEA(I,J) = ZNT_HOLD(I,J)
              UST_SEA(i,j) = UST_HOLD(i,j)
              TSK_SEA(i,j) = TSK(i,j)
           ENDIF
        ENDDO
     ENDDO

     IF (xland_sea_fix) THEN                                    ! ORACLE
        XLAND_CALL2 = XLAND_SEA                                 ! ORACLE
     ELSE                                                       ! ORACLE
        XLAND_CALL2 = XLAND                                     ! ORACLE
     ENDIF                                                      ! ORACLE

     CALL SF_GFS(U3D,V3D,T3D,QV3D,P3D,                  &
          CP,ROVCP,R,XLV,PSFC,CHS_SEA,CHS2_SEA,CQS2_SEA,CPM,        &
          ZNT_SEA,UST_SEA,PSIM_SEA,PSIH_SEA,                        &
          XLAND_CALL2,HFX_SEA,QFX_SEA,LH_SEA,TSK_SEA,FLHC_SEA,FLQC_SEA,   &
          QGH_SEA,QSFC_SEA,U10_SEA,V10_SEA,                         &
          GZ1OZ0_SEA,WSPD_SEA,BR_SEA,ISFFLX,                        &
          EP1,EP2,KARMAN,ITIMESTEP,                     &
          ids,ide, jds,jde, kds,kde,                    &
          ims,ime, jms,jme, kms,kme,                    &
          its,ite, jts,jte, kts,kte                     )

     DO j = JTS , JTE
        DO i = ITS , ITE
           IF ( ( XICE(I,J) .GE. XICE_THRESHOLD ) .and. ( XICE(i,j) .LE. 1.0 ) ) THEN
              BR(i,j)     = ( BR(i,j)     * XICE(i,j) ) + ( (1.0-XICE(i,j)) * BR_SEA(i,j)     )
              GZ1OZ0(i,j) = ( GZ1OZ0(i,j) * XICE(i,j) ) + ( (1.0-XICE(i,j)) * GZ1OZ0_SEA(i,j) )
              PSIM(i,j)   = ( PSIM(i,j)   * XICE(i,j) ) + ( (1.0-XICE(i,j)) * PSIM_SEA(i,j)   )
              PSIH(i,j)   = ( PSIH(i,j)   * XICE(i,j) ) + ( (1.0-XICE(i,j)) * PSIH_SEA(i,j)   )
              U10(i,j)    = ( U10(i,j)    * XICE(i,j) ) + ( (1.0-XICE(i,j)) * U10_SEA(i,j)    )
              V10(i,j)    = ( V10(i,j)    * XICE(i,j) ) + ( (1.0-XICE(i,j)) * V10_SEA(i,j)    )
              WSPD(i,j)   = ( WSPD(i,j)   * XICE(i,j) ) + ( (1.0-XICE(i,j)) * WSPD_SEA(i,j)   )
           ENDIF
        ENDDO
     ENDDO

     tsk_sea_out = TSK_SEA                                      ! ORACLE

   END SUBROUTINE sf_gfs_seaice_wrapper

end module sfgfs_oracle_wrapper
