!-- Appended to a COPY of module_mp_fer_hires.F (before END MODULE) by
!-- build_init_dump.sh: writes every SAVE constant and table fer_hires_init
!-- and FER_HIRES set, as binary32 hex, so a port's init can be graded
!-- word for word.  The module's own code is not edited.
      SUBROUTINE FER_DUMP_STATE(IU, TAG)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: IU
      CHARACTER(LEN=*), INTENT(IN) :: TAG
      INTEGER :: I, J
      CALL DS('ABFR', ABFR); CALL DS('CBFR', CBFR); CALL DS('CIACW', CIACW)
      CALL DS('CIACR', CIACR); CALL DS('C_N0r0', C_N0r0); CALL DS('C_NR', C_NR)
      CALL DS('CRAIN', CRAIN); CALL DS('ARAUT', ARAUT); CALL DS('BRAUT', BRAUT)
      CALL DS('CN0r0', CN0r0); CALL DS('CN0r_DMRmin', CN0r_DMRmin)
      CALL DS('CN0r_DMRmax', CN0r_DMRmax); CALL DS('CRACW', CRACW)
      CALL DS('ESW0', ESW0); CALL DS('RFmax', RFmax); CALL DS('RQR_DRmin', RQR_DRmin)
      CALL DS('RQR_DRmax', RQR_DRmax); CALL DS('RR_DRmin', RR_DRmin)
      CALL DS('RR_DR1', RR_DR1); CALL DS('RR_DR2', RR_DR2); CALL DS('RR_DR3', RR_DR3)
      CALL DS('RR_DR4', RR_DR4); CALL DS('RR_DR5', RR_DR5); CALL DS('RR_DRmax', RR_DRmax)
      CALL DS('BETA6', BETA6); CALL DS('PI_E', PI_E); CALL DS('RFmx1', RFmx1)
      CALL DS('ARcw', ARcw); CALL DS('RH_NgC', RH_NgC); CALL DS('RH_NgT', RH_NgT)
      CALL DS('RQhail', RQhail); CALL DS('AVhail', AVhail); CALL DS('BVhail', BVhail)
      CALL DS('QAUT0', QAUT0)
      CALL DS('C1XPVS0', C1XPVS0); CALL DS('C2XPVS0', C2XPVS0)
      CALL DS('C1XPVS', C1XPVS); CALL DS('C2XPVS', C2XPVS)
      CALL DS('K1E5C1', (1.E5)**C1)
      CALL DS('EPS', EPS); CALL DS('EPS1', EPS1); CALL DS('EPSQ1', EPSQ1)
      CALL DS('RCP', RCP); CALL DS('RCPRV', RCPRV); CALL DS('RRHOL', RRHOL)
      CALL DS('XLS', XLS); CALL DS('XLV1', XLV1); CALL DS('XLF1', XLF1)
      CALL DS('XLS1', XLS1); CALL DS('XLV2', XLV2); CALL DS('XLS2', XLS2)
      CALL DS('XLV3', XLV3); CALL DS('XLS3', XLS3); CALL DS('CLIMIT', CLIMIT)
      CALL DS('T_ICEK', T_ICEK); CALL DS('C1', C1); CALL DS('XMIexp', XMIexp)
      CALL DS('XMImax', XMImax); CALL DS('PI', PI); CALL DS('Cdry', Cdry)
      CALL DS('K1EM3CUBE', (1.E-3)**3)
      DO I = MY_T1, MY_T2
        CALL DA('MY_GROWTH', I, MY_GROWTH(I))
      ENDDO
      DO I = 1, NX
        CALL DA('TBPVS', I, TBPVS(I)); CALL DA('TBPVS0', I, TBPVS0(I))
      ENDDO
      DO I = MDRmin, MDRmax
        CALL DA('VENTR1', I, VENTR1(I)); CALL DA('VENTR2', I, VENTR2(I))
        CALL DA('ACCRR', I, ACCRR(I)); CALL DA('MASSR', I, MASSR(I))
        CALL DA('VRAIN', I, VRAIN(I)); CALL DA('RRATE', I, RRATE(I))
      ENDDO
      DO I = MDImin, MDImax
        CALL DA('VENTI1', I, VENTI1(I)); CALL DA('VENTI2', I, VENTI2(I))
        CALL DA('ACCRI', I, ACCRI(I)); CALL DA('MASSI', I, MASSI(I))
        CALL DA('VSNOWI', I, VSNOWI(I)); CALL DA('SDENS', I, SDENS(I))
      ENDDO
      DO J = 0, Nrime
        DO I = 2, 9
          CALL DA('VEL_RF', (I - 2) + 8 * J, VEL_RF(I, J))
        ENDDO
      ENDDO
      CONTAINS
      SUBROUTINE DS(NAME, V)
        CHARACTER(LEN=*), INTENT(IN) :: NAME
        REAL, INTENT(IN) :: V
        WRITE(IU, '(A,",",A,",0,",Z8.8)') TAG, NAME, V
      END SUBROUTINE DS
      SUBROUTINE DA(NAME, K, V)
        CHARACTER(LEN=*), INTENT(IN) :: NAME
        INTEGER, INTENT(IN) :: K
        REAL, INTENT(IN) :: V
        WRITE(IU, '(A,",",A,",",I0,",",Z8.8)') TAG, NAME, K, V
      END SUBROUTINE DA
      END SUBROUTINE FER_DUMP_STATE
