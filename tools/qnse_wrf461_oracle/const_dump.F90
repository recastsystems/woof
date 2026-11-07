program qnse_const_dump
  ! Print the float32 bit pattern of every PARAMETER the QNSE PBL and the
  ! EDMF shallow mass flux read, AS gfortran FOLDED THEM.  The port pins
  ! these words (kernels/qnsepbl.cu) instead of re-deriving them in C:
  ! gfortran folds an integer power such as C0**3 with one rounding, which
  ! a C multiply chain does not reproduce.  test_qnse_wrf461_parity.py
  ! compares the kernel's pinned words against this listing.
  use module_model_constants, only: R_D, R_V, CP, G, XLV, XLS, P608, EP_1, &
       PQ0, A2, A3, A4
  use module_bl_qnsepbl, only: EPSQ2L, C0, CEPS, BLCKDR, CN, AM1, AM2, AM3, &
       AH1, AH2, AH3, AH4, AH5, ARIMIN, BM1, BM2, BH1, BH2, BH3, EPSKM, &
       CAPA, RLIVWV, ELOCP, EPS1, EPSL, EPSRU, FH, ALPH, EL0MAX, EL0MIN, &
       ELFC, ESQ, ELZ0, PQ0SEA, REQU, EPSGH, EPSGM, AUBH, AUBM, BUBH, &
       BUBM, CUBR, RCUBR, ADNH, ADNM, BDNH, BDNM, VKARMAN, BTG
  use module_bl_mfshconvpbl, only: XG, XP00, XRD, XRV, XCPD, XCPV, XCL, &
       XCI, XTT, XLVTT, XLSTT, XGAMW, XBETAW, XALPW, XGAMI, XBETAI, XALPI, &
       XLINI, XALP_PERT, XABUO, XBENTR, XBDETR, XCMF, XENTR_DRY, XDETR_DRY, &
       XDETR_LUP, XENTR_MF, XCRAD_MF, XKCF_MF, XKRC_MF, XTAUSIGMF, &
       XPRES_UV, XFRAC_UP_MAX
  implicit none
  call e('R_D', R_D); call e('R_V', R_V); call e('CP', CP); call e('G', G)
  call e('XLV', XLV); call e('XLS', XLS); call e('P608', P608)
  call e('EP_1', EP_1); call e('PQ0', PQ0); call e('A2', A2)
  call e('A3', A3); call e('A4', A4)
  call e('EPSQ2L', EPSQ2L); call e('C0', C0); call e('CEPS', CEPS)
  call e('BLCKDR', BLCKDR); call e('CN', CN); call e('AM1', AM1)
  call e('AM2', AM2); call e('AM3', AM3); call e('AH1', AH1)
  call e('AH2', AH2); call e('AH3', AH3); call e('AH4', AH4)
  call e('AH5', AH5); call e('ARIMIN', ARIMIN); call e('BM1', BM1)
  call e('BM2', BM2); call e('BH1', BH1); call e('BH2', BH2)
  call e('BH3', BH3); call e('EPSKM', EPSKM); call e('CAPA', CAPA)
  call e('RLIVWV', RLIVWV); call e('ELOCP', ELOCP); call e('EPS1', EPS1)
  call e('EPSL', EPSL); call e('EPSRU', EPSRU); call e('FH', FH)
  call e('ALPH', ALPH); call e('EL0MAX', EL0MAX); call e('EL0MIN', EL0MIN)
  call e('ELFC', ELFC); call e('ESQ', ESQ); call e('ELZ0', ELZ0)
  call e('PQ0SEA', PQ0SEA); call e('REQU', REQU); call e('EPSGH', EPSGH)
  call e('EPSGM', EPSGM); call e('AUBH', AUBH); call e('AUBM', AUBM)
  call e('BUBH', BUBH); call e('BUBM', BUBM); call e('CUBR', CUBR)
  call e('RCUBR', RCUBR); call e('ADNH', ADNH); call e('ADNM', ADNM)
  call e('BDNH', BDNH); call e('BDNM', BDNM); call e('VKARMAN', VKARMAN)
  call e('BTG', BTG)
  call e('RC02', 2.0 / (C0 * C0))
  call e('XG', XG); call e('XP00', XP00); call e('XRD', XRD)
  call e('XRV', XRV); call e('XCPD', XCPD); call e('XCPV', XCPV)
  call e('XCL', XCL); call e('XCI', XCI); call e('XTT', XTT)
  call e('XLVTT', XLVTT); call e('XLSTT', XLSTT); call e('XGAMW', XGAMW)
  call e('XBETAW', XBETAW); call e('XALPW', XALPW); call e('XGAMI', XGAMI)
  call e('XBETAI', XBETAI); call e('XALPI', XALPI); call e('XLINI', XLINI)
  call e('XALP_PERT', XALP_PERT); call e('XABUO', XABUO)
  call e('XBENTR', XBENTR); call e('XBDETR', XBDETR); call e('XCMF', XCMF)
  call e('XENTR_DRY', XENTR_DRY); call e('XDETR_DRY', XDETR_DRY)
  call e('XDETR_LUP', XDETR_LUP); call e('XENTR_MF', XENTR_MF)
  call e('XCRAD_MF', XCRAD_MF); call e('XKCF_MF', XKCF_MF)
  call e('XKRC_MF', XKRC_MF); call e('XTAUSIGMF', XTAUSIGMF)
  call e('XPRES_UV', XPRES_UV); call e('XFRAC_UP_MAX', XFRAC_UP_MAX)
  ! Folded constant sub-expressions the module spells inline.
  call e('XRD/XCPD', XRD / XCPD)
  call e('XRD/XRV', XRD / XRV)
  call e('XRV/XRD', XRV / XRD)
  call e('XRD/XRV-1', XRD / XRV - 1.)
  call e('XCPV-XCL', XCPV - XCL)
  call e('XCPV-XCI', XCPV - XCI)
  call e('2*(XABUO-XBENTR*XENTR_DRY)', 2. * (XABUO - XBENTR * XENTR_DRY))
  call e('0.5/(XABUO-XBENTR*XENTR_DRY)', 0.5 / (XABUO - XBENTR * XENTR_DRY))
  call e('0.5/XABUO', 0.5 / (XABUO))
  call e('XENTR_MF*XG/XCRAD_MF', XENTR_MF * XG / XCRAD_MF)
  call e('0.5*XKCF_MF', 0.5 * XKCF_MF)
  call e('2./3.', 2. / 3.)
  call e('1./3.', 1. / 3.)
  call e('-2./3.', -2. / 3.)
  call e('0.5*P608', 0.5 * P608)
contains
  subroutine e(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    write(*, '(A,1X,Z8.8,1X,ES17.9E3)') name, transfer(x, 0), x
  end subroutine e
end program qnse_const_dump
