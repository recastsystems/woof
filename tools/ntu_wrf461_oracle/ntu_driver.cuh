// ---------------------------------------------------------------------------
// Column drivers: WRF MP_NTU (module_mp_ntu.F:1689-1811) and NTU_INIT /
// INIT_AEROSOL (:255-314, :1569-1685), one GPU thread per column.
//
// Field layout is the engine's: (nz, ny, nx) contiguous, k = 0 at the
// surface; a column is c = j*nx + i and level k of it lives at k*ncol + c.
// WRF's MP_NTU turns the column over (K = 1 is the model top); the gather
// below does the same flip, NK = KTE-K+1.
// ---------------------------------------------------------------------------
enum NtuField3 {
    NTU_F_TH = 0, NTU_F_P, NTU_F_DZ, NTU_F_PII,
    NTU_F_QV, NTU_F_QC, NTU_F_QR, NTU_F_QI, NTU_F_QS, NTU_F_QG, NTU_F_QH,
    NTU_F_NC, NTU_F_NR, NTU_F_NI, NTU_F_NS, NTU_F_NG, NTU_F_NH,
    NTU_F_QDCN, NTU_F_QTCN, NTU_F_QCCN, NTU_F_QRCN, NTU_F_QNIN,
    NTU_F_FI, NTU_F_FS, NTU_F_VI, NTU_F_VS, NTU_F_VG,
    NTU_F_AI, NTU_F_AS, NTU_F_AG, NTU_F_AH, NTU_F_I3M,
    NTU_F_W,                      // (nz+1) w levels
    NTU_NFIELD3
};
enum NtuField2 {
    NTU_S_RAINNC = 0, NTU_S_RAINNCV, NTU_S_SNOWNC, NTU_S_SNOWNCV,
    NTU_S_GRAPNC, NTU_S_GRAPNCV, NTU_S_HAILNC, NTU_S_HAILNCV, NTU_S_SR,
    NTU_NFIELD2
};

// MP_NTU for one column.
NTU_DEV void ntu_mp_column(NtuAeroConst* __restrict__ AC,
                           float* const* __restrict__ f3,
                           float* const* __restrict__ f2,
                           size_t c, size_t ncol, int nz, float DT_MP)
{
    const int KTS = 1;
    const int KTE = nz;
    float TK1D[NTU_KMAX], P1D[NTU_KMAX], W1D[NTU_KMAX], S1D[NTU_KMAX],
          DZ1D[NTU_KMAX], QV1D[NTU_KMAX], QC1D[NTU_KMAX], QR1D[NTU_KMAX],
          QI1D[NTU_KMAX], QS1D[NTU_KMAX], QG1D[NTU_KMAX], QH1D[NTU_KMAX],
          NC1D[NTU_KMAX], NR1D[NTU_KMAX], NI1D[NTU_KMAX], NS1D[NTU_KMAX],
          NG1D[NTU_KMAX], NH1D[NTU_KMAX], VI1D[NTU_KMAX], VS1D[NTU_KMAX],
          VG1D[NTU_KMAX], FI1D[NTU_KMAX], FS1D[NTU_KMAX], AI1D[NTU_KMAX],
          AS1D[NTU_KMAX], AG1D[NTU_KMAX], AH1D[NTU_KMAX], I3M1D[NTU_KMAX],
          QDCN1D[NTU_KMAX], QTCN1D[NTU_KMAX], QCCN1D[NTU_KMAX],
          QRCN1D[NTU_KMAX], QNIN1D[NTU_KMAX];
    float* TH = f3[NTU_F_TH];
    const float* PII = f3[NTU_F_PII];
    const float* W = f3[NTU_F_W];
    float DT = DT_MP;
    float DTMN = DT_MP / 60.0f;
    float CLODNCV = 0.0f, ICENCV = 0.0f, VINCV = 0.0f, VSNCV = 0.0f,
          VGNCV = 0.0f, FINCV = 0.0f, FSNCV = 0.0f, AINCV = 0.0f,
          ASNCV = 0.0f, AGNCV = 0.0f, AHNCV = 0.0f, I3MNCV = 0.0f;
    float& RAINNC = f2[NTU_S_RAINNC][c];
    float& RAINNCV = f2[NTU_S_RAINNCV][c];
    float& SNOWNC = f2[NTU_S_SNOWNC][c];
    float& SNOWNCV = f2[NTU_S_SNOWNCV][c];
    float& GRAPNC = f2[NTU_S_GRAPNC][c];
    float& GRAPNCV = f2[NTU_S_GRAPNCV][c];
    float& HAILNC = f2[NTU_S_HAILNC][c];
    float& HAILNCV = f2[NTU_S_HAILNCV][c];
    float& SR = f2[NTU_S_SR][c];
    RAINNCV = 0.0f; SNOWNCV = 0.0f; GRAPNCV = 0.0f; HAILNCV = 0.0f;
    SR = 0.0f;
    for (int K = KTS; K <= KTE; ++K) {
        const int NK = KTE - K + 1;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        const size_t above = (size_t)NK * ncol + c;
        const int k = K - 1;
        S1D[k] = 0.0f;
        TK1D[k] = TH[at] * PII[at];
        W1D[k] = 0.5f * (W[at] + W[above]);
        QV1D[k] = f3[NTU_F_QV][at];
        DZ1D[k] = f3[NTU_F_DZ][at];
        P1D[k] = f3[NTU_F_P][at];
        QC1D[k] = f3[NTU_F_QC][at];
        QR1D[k] = f3[NTU_F_QR][at];
        QI1D[k] = f3[NTU_F_QI][at];
        QS1D[k] = f3[NTU_F_QS][at];
        QG1D[k] = f3[NTU_F_QG][at];
        QH1D[k] = f3[NTU_F_QH][at];
        NC1D[k] = f3[NTU_F_NC][at];
        NR1D[k] = f3[NTU_F_NR][at];
        NI1D[k] = f3[NTU_F_NI][at];
        NS1D[k] = f3[NTU_F_NS][at];
        NG1D[k] = f3[NTU_F_NG][at];
        NH1D[k] = f3[NTU_F_NH][at];
        VI1D[k] = f3[NTU_F_VI][at];
        VS1D[k] = f3[NTU_F_VS][at];
        VG1D[k] = f3[NTU_F_VG][at];
        FI1D[k] = f3[NTU_F_FI][at];
        FS1D[k] = f3[NTU_F_FS][at];
        AI1D[k] = f3[NTU_F_AI][at];
        AS1D[k] = f3[NTU_F_AS][at];
        AG1D[k] = f3[NTU_F_AG][at];
        AH1D[k] = f3[NTU_F_AH][at];
        I3M1D[k] = f3[NTU_F_I3M][at];
        QDCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QDCN][at]);
        QTCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QTCN][at]);
        QCCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QCCN][at]);
        QRCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QRCN][at]);
        QNIN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QNIN][at]);
    }
    ntu_ntu_micro(AC, TK1D, QV1D, DZ1D, P1D, S1D, W1D, SR, ICENCV, VINCV,
                  VSNCV, VGNCV, FINCV, FSNCV, AINCV, ASNCV, AGNCV, AHNCV,
                  I3MNCV, CLODNCV, RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAPNC,
                  GRAPNCV, HAILNC, HAILNCV, DT, DTMN, QC1D, QR1D, QI1D, QS1D,
                  QG1D, QH1D, NC1D, NR1D, NI1D, NS1D, NG1D, NH1D, VI1D, VS1D,
                  VG1D, FI1D, FS1D, AI1D, AS1D, AG1D, AH1D, I3M1D, QDCN1D,
                  QTCN1D, QCCN1D, QRCN1D, QNIN1D, KTS, KTE);
    for (int K = KTS; K <= KTE; ++K) {
        const int NK = KTE - K + 1;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        const int k = K - 1;
        TH[at] = TK1D[k] / PII[at];
        f3[NTU_F_QV][at] = ntu_fmax(0.0f, QV1D[k]);
        f3[NTU_F_QC][at] = ntu_fmax(0.0f, QC1D[k]);
        f3[NTU_F_QR][at] = ntu_fmax(0.0f, QR1D[k]);
        f3[NTU_F_QI][at] = ntu_fmax(0.0f, QI1D[k]);
        f3[NTU_F_QS][at] = ntu_fmax(0.0f, QS1D[k]);
        f3[NTU_F_QG][at] = ntu_fmax(0.0f, QG1D[k]);
        f3[NTU_F_QH][at] = ntu_fmax(0.0f, QH1D[k]);
        f3[NTU_F_NC][at] = ntu_fmax(0.0f, NC1D[k]);
        f3[NTU_F_NR][at] = ntu_fmax(0.0f, NR1D[k]);
        f3[NTU_F_NI][at] = ntu_fmax(0.0f, NI1D[k]);
        f3[NTU_F_NS][at] = ntu_fmax(0.0f, NS1D[k]);
        f3[NTU_F_NG][at] = ntu_fmax(0.0f, NG1D[k]);
        f3[NTU_F_NH][at] = ntu_fmax(0.0f, NH1D[k]);
        f3[NTU_F_VI][at] = ntu_fmax(0.0f, VI1D[k]);
        f3[NTU_F_VS][at] = ntu_fmax(0.0f, VS1D[k]);
        f3[NTU_F_VG][at] = ntu_fmax(0.0f, VG1D[k]);
        f3[NTU_F_FI][at] = ntu_fmax(0.0f, FI1D[k]);
        f3[NTU_F_FS][at] = ntu_fmax(0.0f, FS1D[k]);
        f3[NTU_F_AI][at] = ntu_fmax(0.0f, AI1D[k]);
        f3[NTU_F_AS][at] = ntu_fmax(0.0f, AS1D[k]);
        f3[NTU_F_AG][at] = ntu_fmax(0.0f, AG1D[k]);
        f3[NTU_F_AH][at] = ntu_fmax(0.0f, AH1D[k]);
        f3[NTU_F_I3M][at] = ntu_fmax(0.0f, I3M1D[k]);
        f3[NTU_F_QDCN][at] = ntu_fmax(0.0f, QDCN1D[k]);
        f3[NTU_F_QTCN][at] = ntu_fmax(0.0f, QTCN1D[k]);
        f3[NTU_F_QCCN][at] = ntu_fmax(0.0f, QCCN1D[k]);
        f3[NTU_F_QRCN][at] = ntu_fmax(0.0f, QRCN1D[k]);
        f3[NTU_F_QNIN][at] = ntu_fmax(0.0f, QNIN1D[k]);
    }
}

// NTU_INIT + INIT_AEROSOL for one column.  phb/ph are on the nz+1 w levels,
// p/pb/alt/qv on the nz mass levels; the five aerosol fields are written in
// place (left as given when restart != 0, exactly as WRF).
//
// Declared divergence (fix): INIT_AEROSOL's first level loop runs
// K = KTS..KTE with KTE = kde, one level past the nz mass levels NTU_INIT
// filled, so WRF compares an uninitialised pressure word against 850 hPa and
// can move the boundary-layer top K_PBL to KTE-2.  The loop here stops at the
// last mass level; every other statement is WRF's.
NTU_DEV void ntu_init_column(NtuAeroConst* __restrict__ AC,
                             const float* __restrict__ phb,
                             const float* __restrict__ ph,
                             const float* __restrict__ p,
                             const float* __restrict__ pb,
                             const float* __restrict__ alt,
                             const float* __restrict__ qv,
                             float* const* __restrict__ aero,
                             size_t c, size_t ncol, int nz, int restart)
{
    const int KTS = 1;
    const int KTE = nz + 1;            // NTU_INIT receives kte = kde
    float DZ[NTU_KMAX + 1], DZ8W[NTU_KMAX + 1], RHO[NTU_KMAX + 1],
          P_PHY[NTU_KMAX + 1];
    float QAERO[NTU_KMAX + 1][5];
    for (int K = KTS; K <= KTE; ++K) {
        const size_t at = (size_t)(K - 1) * ncol + c;
        DZ[K - 1] = (phb[at] + ph[at]) / NTU_G;
    }
    for (int K = KTS; K <= KTE - 1; ++K) {
        const int NK = KTE - K;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        DZ8W[K - 1] = DZ[NK] - DZ[NK - 1];
        RHO[K - 1] = (1.0f / alt[at]) * (1.0f + qv[at]);
        P_PHY[K - 1] = p[at] + pb[at];
        for (int s = 0; s < 5; ++s) QAERO[K - 1][s] = aero[s][at];
    }
    if (!restart) {
        const float P_PBL = 85000.0f;
        const float BACKIN = 4.0E5f;
        float XMASS[NTU_NAER];
        float RASH[NTU_NAER];
        for (int IA = 1; IA <= NTU_NAERT; ++IA)
            for (int K = KTS; K <= KTE - 1; ++K) QAERO[K - 1][IA - 1] = 0.0f;
        for (int J = 1; J <= NTU_NAER; ++J) {
            XMASS[J - 1] = 0.0f;
            for (int IM = 1; IM <= NTU_NCCN; ++IM) {
                const int ij = (IM - 1) + 3 * (J - 1);
                float XMAS = (AC->ZCCN[ij] * NTU_POWF(AC->CNMOD[ij], 3.0f))
                             * NTU_EXPF(4.5f * ntu_sq(AC->CNSTD[ij]));
                XMASS[J - 1] = XMASS[J - 1] + XMAS;
            }
            XMASS[J - 1] = ((XMASS[J - 1] * NTU_C4PI3) * AC->DNAS[J - 1])
                           * AC->SENS[J - 1];
        }
        // ID_DUST*ID_IN == 0: no dust, background ice nuclei.
        float DUST_IN0 = BACKIN;
        float DUST_IN = 0.0f, DAIR = 0.0f;
        int K_PBL = 1;
        for (int K = KTS; K <= KTE - 1; ++K) {
            int IV = 1;
            for (int IA = 1; IA <= NTU_NAER; ++IA) {
                QAERO[K - 1][IV - 1] = XMASS[IA - 1] / RHO[K - 1];
                RASH[IA - 1] = 1.0f / AC->ASH[IA - 1];
                DUST_IN = DUST_IN0;
                if (IA == NTU_ID_IN) {
                    // ID_DUST == 0: background value, doubled scale height.
                    DUST_IN = BACKIN;
                    RASH[IA - 1] = RASH[IA - 1] * 0.5f;
                    DAIR = RHO[K - 1];
                    DUST_IN = ntu_fmax(BACKIN, DUST_IN);
                    QAERO[K - 1][IV - 1] = DUST_IN / DAIR;
                }
                IV = IV + NTU_NAERN[IA - 1];
            }
            if (P_PHY[K - 1] < P_PBL) K_PBL = K;
        }
        K_PBL = ntu_imin(K_PBL, KTE - 3 + 1);
        float ZH = 0.0f, DZ0 = 0.0f;
        for (int K = K_PBL - 1; K >= 1; --K) {
            DAIR = RHO[K - 1];
            ZH = ZH + (DZ0 + DZ8W[K - 1]) * 0.5f;
            DZ0 = DZ8W[K - 1];
            int IV = 1;
            for (int IA = 1; IA <= NTU_NAER; ++IA) {
                QAERO[K - 1][IV - 1] = QAERO[K - 1][IV - 1]
                                       * NTU_EXPF(-(RASH[IA - 1] * ZH));
                IV = IV + NTU_NAERN[IA - 1];
            }
        }
        int IV = 0;
        for (int IA = 1; IA <= NTU_NAER; ++IA) {
            const int top = ntu_imin(2, NTU_NAERN[IA - 1]);
            for (int IB = 2; IB <= top; ++IB)
                for (int K = KTS; K <= KTE - 1; ++K)
                    QAERO[K - 1][IV + IB - 1] = QAERO[K - 1][IV];
            IV = IV + NTU_NAERN[IA - 1];
        }
        (void)DAIR;
    }
    for (int K = KTS; K <= KTE - 1; ++K) {
        const int NK = KTE - K;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        for (int s = 0; s < 5; ++s) aero[s][at] = QAERO[K - 1][s];
    }
}

// ---------------------------------------------------------------------------
// 10 cm radar reflectivity from NTU's own predicted size distributions.
//
// WRF's NTU arm writes no REFL_10CM; this diagnostic is WOOF's.  Each
// category's gamma distribution n(D) ~ D^AFA exp(-LAM D) comes from the
// scheme's own SOLVE_AFA* routines (run on copies, so the state is not
// touched), and contributes its Rayleigh sixth moment
//   M6 = N Gamma(AFA+7) / (Gamma(AFA+1) LAM^6)        [m^6 per kg of air].
// Frozen categories scale by |K_ice|^2/|K_w|^2 (0.176/0.93) and by the
// square of their predicted bulk density over solid ice (917 kg m-3), the
// mass-squared form that holds for any particle shape.  The floor is the
// -35 dBZ the reflectivity family uses.
NTU_DEV double ntu_m6(float N, float AFA, float LAM) {
    return (double)N * exp((double)(ntu_gamln(AFA + 7.0f) - ntu_gamln(AFA + 1.0f))
                           - 6.0 * log((double)LAM));
}

NTU_DEV float ntu_refl_cell(NtuAeroConst* __restrict__ AC,
                            float* const* __restrict__ f3, size_t at)
{
    const double KICE = 0.176 / 0.93;
    const float T = f3[NTU_F_TH][at] * f3[NTU_F_PII][at];
    const float P = f3[NTU_F_P][at];
    const float QV = f3[NTU_F_QV][at];
    const float RHO = P / (T * (1.0f + 0.61f * QV)) / NTU_R;
    double z = 0.0;
    float q, n, lam = 0.0f, mvd = 0.0f, afa = 0.0f;
    q = f3[NTU_F_QC][at]; n = f3[NTU_F_NC][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        ntu_solve_afac(AC, T, q, n, lam, mvd, afa);
        if (lam > 0.0f) z += ntu_m6(n, afa, lam);
    }
    q = f3[NTU_F_QR][at]; n = f3[NTU_F_NR][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        ntu_solve_afar(T, q, n, lam, mvd, afa);
        if (lam > 0.0f) z += ntu_m6(n, afa, lam);
    }
    q = f3[NTU_F_QI][at]; n = f3[NTU_F_NI][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vi = f3[NTU_F_VI][at], fi = f3[NTU_F_FI][at];
        float ai = f3[NTU_F_AI][at], i3 = f3[NTU_F_I3M][at];
        float adagr = 1.0f, zeta = 0.0f, rhoi = 0.0f, ami = 0.0f, bmi = 0.0f,
              avi = 0.0f, bvi = 0.0f, best = 0.0f;
        ntu_solve_afai(T, P, RHO, QV, q, n, vi, fi, ai, i3, adagr, zeta, lam,
                       afa, mvd, rhoi, ami, bmi, avi, bvi, best);
        if (lam > 0.0f) {
            const double r = (double)rhoi / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QS][at]; n = f3[NTU_F_NS][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vs = f3[NTU_F_VS][at], fs = f3[NTU_F_FS][at],
              as2 = f3[NTU_F_AS][at];
        float rhos = 0.0f, saspr = 1.0f, ams = 0.0f, avs = 0.0f, bvs = 0.0f;
        ntu_solve_afas(T, RHO, q, f3[NTU_F_QC][at], n, vs, fs, as2, afa, lam,
                       mvd, rhos, saspr, ams, avs, bvs);
        if (lam > 0.0f) {
            const double r = (double)rhos / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QG][at]; n = f3[NTU_F_NG][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vg = f3[NTU_F_VG][at], ag = f3[NTU_F_AG][at];
        float rhog = 0.0f, amg = 0.0f, avg = 0.0f, bvg = 0.0f;
        ntu_solve_afag(T, RHO, q, f3[NTU_F_QC][at], n, vg, ag, lam, afa, mvd,
                       rhog, amg, avg, bvg);
        if (lam > 0.0f) {
            const double r = (double)rhog / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QH][at]; n = f3[NTU_F_NH][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float ah = f3[NTU_F_AH][at], avh = 0.0f, bvh = 0.0f;
        ntu_solve_afah(T, RHO, q, n, ah, lam, afa, mvd, avh, bvh);
        if (lam > 0.0f) {
            const double r = (double)NTU_RHOH / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    const double zmm = z * (double)RHO * 1.0e18;   // mm^6 m^-3
    return zmm > 3.1622776601683794e-4 ? (float)(10.0 * log10(zmm)) : -35.0f;
}

#ifndef NTU_HOST
extern "C" __global__ void ntu_aero_const_kernel(NtuAeroConst* AC, int ccnty)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) ntu_aero_const(AC, ccnty);
}

extern "C" __global__ void ntu_init_kernel(
    NtuAeroConst* AC, const float* phb, const float* ph, const float* p,
    const float* pb, const float* alt, const float* qv,
    float* const* aero, long long ncol, int nz, int restart,
    unsigned int* status)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    ntu_status_word = status;
    if (nz + 1 > NTU_KMAX + 1) { NTU_FATAL(NTU_ERR_KMAX); return; }
    ntu_init_column(AC, phb, ph, p, pb, alt, qv, aero, (size_t)c,
                    (size_t)ncol, nz, restart);
}

extern "C" __global__ void ntu_refl_kernel(
    NtuAeroConst* AC, float* const* f3, long long ncell, float* refl)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    refl[i] = ntu_refl_cell(AC, f3, (size_t)i);
}

extern "C" __global__ void ntu_mp_kernel(
    NtuAeroConst* AC, float* const* f3, float* const* f2, long long ncol,
    int nz, float dt_mp, unsigned int* status)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    ntu_status_word = status;
    if (nz > NTU_KMAX) { NTU_FATAL(NTU_ERR_KMAX); return; }
    ntu_mp_column(AC, f3, f2, (size_t)c, (size_t)ncol, nz, dt_mp);
}
#endif
