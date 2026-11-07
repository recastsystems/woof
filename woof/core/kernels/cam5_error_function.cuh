// WRF 4.7.1 phys/module_cam_error_function.F:168-319, CALERF_r8.
// Public-domain WRF release; see tools/cam5_wrf471_oracle/LICENSE-WRF.txt.
// CUDA transcription for JINT=0 (erf) and JINT=1 (erfc). Keep native
// rational coefficients, evaluation order and the two-exponential split.
#ifndef CAM5_ERROR_FUNCTION_CUH
#define CAM5_ERROR_FUNCTION_CUH
__device__ R8 cam5_calerf_r8(R8 arg, bool complementary) {
    const R8 A[5] = {R8(3.16112374387056560E00), R8(1.13864154151050156E02),
                     R8(3.77485237685302021E02), R8(3.20937758913846947E03),
                     R8(1.85777706184603153E-1)};
    const R8 B[4] = {R8(2.36012909523441209E01), R8(2.44024637934444173E02),
                     R8(1.28261652607737228E03), R8(2.84423683343917062E03)};
    const R8 C[9] = {R8(5.64188496988670089E-1), R8(8.88314979438837594E00),
                     R8(6.61191906371416295E01), R8(2.98635138197400131E02),
                     R8(8.81952221241769090E02), R8(1.71204761263407058E03),
                     R8(2.05107837782607147E03), R8(1.23033935479799725E03),
                     R8(2.15311535474403846E-8)};
    const R8 D[8] = {R8(1.57449261107098347E01), R8(1.17693950891312499E02),
                     R8(5.37181101862009858E02), R8(1.62138957456669019E03),
                     R8(3.29079923573345963E03), R8(4.36261909014324716E03),
                     R8(3.43936767414372164E03), R8(1.23033935480374942E03)};
    const R8 P[6] = {R8(3.05326634961232344E-1), R8(3.60344899949804439E-1),
                     R8(1.25781726111229246E-1), R8(1.60837851487422766E-2),
                     R8(6.58749161529837803E-4), R8(1.63153871373020978E-2)};
    const R8 Q[5] = {R8(2.56852019228982242E00), R8(1.87295284992346047E00),
                     R8(5.27905102951428412E-1), R8(6.05183413124413191E-2),
                     R8(2.33520497626869185E-3)};
    const R8 ZERO(0.0), FOUR(4.0), ONE(1.0), TWO(2.0);
    const R8 SQRPI(5.6418958354775628695E-1), THRESH(0.46875E0), SIXTEN(16.0);
    const R8 XSMALL(1.11E-16), XBIG(26.543E0);
    R8 x = arg, y = uw_abs(x), ysq, xnum, xden, del, result;
    if (y <= THRESH) {
        ysq = ZERO;
        if (y > XSMALL) ysq = y * y;
        xnum = A[4] * ysq;
        xden = ysq;
        for (int i = 0; i < 3; ++i) {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        result = x * (xnum + A[3]) / (xden + B[3]);
        if (complementary) result = ONE - result;
        return result;                               // GO TO 80
    } else if (y <= FOUR) {
        xnum = C[8] * y;
        xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        result = (xnum + C[7]) / (xden + D[7]);
        ysq = uw_aint(y * SIXTEN) / SIXTEN;          // JINT /= 2
        del = (y - ysq) * (y + ysq);
        result = uw_exp(-(ysq * ysq)) * uw_exp(-del) * result;
    } else {
        result = ZERO;
        if (y >= XBIG) goto l30;                     // JINT /= 2
        ysq = ONE / (y * y);
        xnum = P[5] * ysq;
        xden = ysq;
        for (int i = 0; i < 4; ++i) {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        result = ysq * (xnum + P[4]) / (xden + Q[4]);
        result = (SQRPI - result) / y;
        ysq = uw_aint(y * SIXTEN) / SIXTEN;
        del = (y - ysq) * (y + ysq);
        result = uw_exp(-(ysq * ysq)) * uw_exp(-del) * result;
    }
l30:
    if (complementary) {
        if (x < ZERO) result = TWO - result;
    } else {
        result = (R8(0.5) - result) + R8(0.5);
        if (x < ZERO) result = -result;
    }
    return result;
}

#endif
