// gpuwm/core/kernels/ishmael.cu
//
// ISHMAEL (Ice-Spheroids Habit Model with Aspect-ratio Evolution), WRF
// mp_physics = 55: Jensen et al. 2017 (JAS), 2018 (MWR), 2018 (JAS).
//
// Transcription authority: WRF v4.7.1 phys/module_mp_jensen_ishmael.F
// (f52c197e, 4711 lines; public domain).  Line numbers below are that
// file's. Independent cell physics (:696-2727) runs across all grid cells.
// One thread then owns each column for Reisner sedimentation on split steps
// (:2734-2836) and the post-sedimentation check (:2841-2960). The last active
// cell's effective density is carried into that ordered column pass.
//
// ARITHMETIC.  Every value is float32 as in the Fortran (default REAL),
// except gammln and AVINT, which are REAL*8 there and double here.  The
// default arm permits fused arithmetic and CUDA elementary functions. The
// strict replay arm disables contraction and uses the scalar reference
// elementary functions. Both preserve subnormal habit moments (--ftz=false).
// Strict real-input replay: 213,156 output words, zero differences,
// 2026-11-01; tools/ishmael_wrf461_oracle/source-pin.json records the scope.
//
// MIN/MAX.  gfortran -O2 lowers REAL MAX(a,b) to the x86 maxss pair with a
// constant operand second, i.e. (a > b ? a : b); MIN likewise.  ish_max/
// ish_min are those words; a NaN or a signed zero in the variable operand
// resolves exactly as the WRF build resolves it.
//
// DECLARED DIVERGENCE D1, default arm only: :1105 bounds ice-RAIN riming
// by the ice-CLOUD riming sum (tmpsum). The default uses tmpsumr. Strict
// replay preserves the pristine WRF limiter so it can be compared directly.

// Fast arithmetic is the production arm. Preserve subnormal habit moments.
#ifdef ISHMAEL_FAST_NUMERICS
#define gfk_exp expf
#define gfk_log logf
#define gfk_log10 log10f
#define gfk_tanh tanhf
#define gfk_pow powf
#define glibc_sinf sinf
#endif

__device__ __forceinline__ double ish_log64(double x) {
#ifdef ISHMAEL_FAST_NUMERICS
    return log(x);
#else
    return g64_log::log(x);
#endif
}

#define ISH_CAT 3
#define ISH_NGAMMA 505001
#define ISH_NDN 60

// Module constants (:31-43).
#define ISH_PI 3.14159265f
#define ISH_G 9.8f
#define ISH_RD 287.15f
#define ISH_RV 461.5f
#define ISH_CP 1004.0f
#define ISH_CPW 4218.0f
#define ISH_RCP 0.285856574f
#define ISH_RHOW 1000.0f
#define ISH_T0 273.15f
#define ISH_RHOI 920.0f
#define ISH_R0 1.27494f

// me_ishmael parameters (:506-518).
#define ISH_NU 4.0f
#define ISH_QSMALL 1.e-12f
#define ISH_QNSMALL 1.25e-7f
#define ISH_QASMALL 1.e-24f
#define ISH_AO 0.1e-6f
#define ISH_AR 149.1f
#define ISH_BR 0.5f
#define ISH_F1R 0.78f
#define ISH_F2R 0.308f

// Values gfortran evaluates at COMPILE time, correctly rounded (MPFR), are
// the generated words of ishmael_folded.cuh (listed before this unit by
// gpuwm/core/ishmael.py): the cubes of a REAL literal (a multiply chain
// would round twice), log(ao), and the -O2-folded log/exp/pow of constant
// arguments.  folded_constants.py also shows that every folded log/exp/pow
// word equals the runtime float32 word, so where GCC's folding is not
// certain the choice cannot move a result.
#define ISH_LAMMAXR (1.f / 20.E-6f)
#define ISH_LAMMINR (1.f / 2800.E-6f)
#define ISH_F_C1 (4.0f / (5.83f * 5.83f * sqrtf(0.6f)))
#define ISH_SIX_OVER_PI_SQ ((6.f / ISH_PI) * (6.f / ISH_PI))

__device__ __forceinline__ float ish_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float ish_min(float a, float b) { return a < b ? a : b; }
// NINT on default REAL: round half away from zero (lroundf).
__device__ __forceinline__ int ish_nint(float x) { return (int)roundf(x); }

// gamma_tab index (:1120 and its 50 copies), 1-based like the Fortran.
__device__ __forceinline__ int ish_gi(float gamma_arg)
{
    int n = ish_nint((gamma_arg * 100000.f) - 355000.f + 1.f);
    n = n > 1 ? n : 1;
    return n < ISH_NGAMMA ? n : ISH_NGAMMA;
}

struct IshTables {
    const float* itab;     // (51,51,51,11,2), file order: i4 fastest, value last
    const float* itabr;    // (51,51,51,11,6)
    const float* gtab;     // 505001
    const float* coltab;   // [7 pairs][60 idny][60 idnx]
    const float* coltabn;
};

// Inherent growth ratio data, -1 C to -60 C (:175-184).
__device__ const float ISH_IGRDATA[60] = {
    0.910547f, 0.81807f, 0.6874f, 0.60127f, 1.59767f, 2.32423f, 2.08818f,
    1.61921f, 1.15865f, 0.863071f, 0.617586f, 0.453917f, 0.351975f, 0.28794f,
    0.269298f, 0.28794f, 0.333623f, 0.418883f, 0.56992f, 0.796458f, 1.14325f,
    1.64103f, 1.90138f, 1.82653f, 1.61921f, 1.47436f, 1.32463f, 1.25556f,
    1.22239f, 1.206f, 1.11522f, 1.10751f, 1.10738f, 1.11484f, 1.12234f,
    1.12221f, 1.14529f, 1.16884f, 1.20104f, 1.22573f, 1.25094f, 1.27666f,
    1.31183f, 1.3388f, 1.35704f, 1.37553f, 1.38479f, 1.39411f, 1.40349f,
    1.41294f, 1.42245f, 1.43202f, 1.44166f, 1.45137f, 1.46114f, 1.47097f,
    1.48087f, 1.50105f, 1.50087f, 1.51098f };

// ---------------------------------------------------------------------------
// GAMMA (:3583-3815), Cody and Stoltz, in the module's default REAL.
// ---------------------------------------------------------------------------
__device__ float ish_gamma(float X)
{
    const float ONE = 1.0f, HALF = 0.5f, TWELVE = 12.0f, TWO = 2.0f, ZERO = 0.0f;
    const float XBIG = 35.040f, XMININ = 1.18E-38f, EPS = 1.19E-7f, XINF = 3.4E38f;
    const float P[8] = {
        -1.71618513886549492533811E+0f, 2.47656508055759199108314E+1f,
        -3.79804256470945635097577E+2f, 6.29331155312818442661052E+2f,
        8.66966202790413211295064E+2f, -3.14512729688483675254357E+4f,
        -3.61444134186911729807069E+4f, 6.64561438202405440627855E+4f };
    const float Q[8] = {
        -3.08402300119738975254353E+1f, 3.15350626979604161529144E+2f,
        -1.01515636749021914166146E+3f, -3.10777167157231109440444E+3f,
        2.25381184209801510330112E+4f, 4.75584627752788110767815E+3f,
        -1.34659959864969306392456E+5f, -1.15132259675553483497211E+5f };
    const float C[7] = {
        -1.910444077728E-03f, 8.4171387781295E-04f,
        -5.952379913043012E-04f, 7.93650793500350248E-04f,
        -2.777777777777681622553E-03f, 8.333333333333333331554247E-02f,
        5.7083835261E-03f };
    const float xxx = 0.9189385332046727417803297f;
    bool PARITY = false;
    float FACT = ONE;
    int N = 0;
    float Y = X, Y1, RES, Z, XNUM, XDEN, YSQ, SUM;
    if (Y <= ZERO) {
        Y = -X;
        Y1 = truncf(Y);
        RES = Y - Y1;
        if (RES != ZERO) {
            if (Y1 != truncf(Y1 * HALF) * TWO) PARITY = true;
            FACT = -ISH_PI / glibc_sinf(ISH_PI * RES);
            Y = Y + ONE;
        } else {
            return XINF;
        }
    }
    if (Y < EPS) {
        if (Y >= XMININ) {
            RES = ONE / Y;
        } else {
            return XINF;
        }
    } else if (Y < TWELVE) {
        Y1 = Y;
        if (Y < ONE) {
            Z = Y;
            Y = Y + ONE;
        } else {
            N = (int)Y - 1;
            Y = Y - (float)N;
            Z = Y - ONE;
        }
        XNUM = ZERO;
        XDEN = ONE;
        for (int I = 0; I < 8; ++I) {
            XNUM = (XNUM + P[I]) * Z;
            XDEN = XDEN * Z + Q[I];
        }
        RES = XNUM / XDEN + ONE;
        if (Y1 < Y) {
            RES = RES / Y1;
        } else if (Y1 > Y) {
            for (int I = 1; I <= N; ++I) {
                RES = RES * Y;
                Y = Y + ONE;
            }
        }
    } else {
        if (Y <= XBIG) {
            YSQ = Y * Y;
            SUM = C[6];
            for (int I = 0; I < 6; ++I) SUM = SUM / YSQ + C[I];
            SUM = SUM / Y - Y + xxx;
            SUM = SUM + (Y - HALF) * gfk_log(Y);
            RES = gfk_exp(SUM);
        } else {
            return XINF;
        }
    }
    if (PARITY) RES = -RES;
    if (FACT != ONE) RES = FACT / RES;
    return RES;
}

// polysvp (:3427-3482), Flatau et al. 1992, Pa.
__device__ float ish_polysvp(float T, int TYPE)
{
    float dt, r = 0.0f;
    if (TYPE == 1) {
        const float a0i = 6.11147274f, a1i = 0.503160820f, a2i = 0.188439774e-1f,
            a3i = 0.420895665e-3f, a4i = 0.615021634e-5f, a5i = 0.602588177e-7f,
            a6i = 0.385852041e-9f, a7i = 0.146898966e-11f, a8i = 0.252751365e-14f;
        dt = ish_max(T - 273.16f, -80.f);
        r = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i + dt * (a5i + dt * (a6i + dt * (a7i + a8i * dt)))))));
        r = r * 100.f;
    }
    if (TYPE == 0) {
        const float a0 = 6.11239921f, a1 = 0.443987641f, a2 = 0.142986287e-1f,
            a3 = 0.264847430e-3f, a4 = 0.302950461e-5f, a5 = 0.206739458e-7f,
            a6 = 0.640689451e-10f, a7 = -0.952447341e-13f, a8 = -0.976195544e-15f;
        dt = ish_max(T - 273.16f, -80.f);
        r = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4 + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
        r = r * 100.f;
    }
    return r;
}

// get_igr (:3820-3849).
__device__ float ish_get_igr(float temp)
{
    float tc = temp - ISH_T0;
    float dum, igr1, igr2;
    if (tc >= -59.f && tc <= -1.f) {
        dum = (fabsf((float)(int)(temp - ISH_T0)) + 1.f) - fabsf(temp - ISH_T0);
        int i1 = (int)(temp - ISH_T0) * (-1);
        i1 = i1 > 1 ? i1 : 1;
        int i2 = ((int)(temp - ISH_T0) * (-1)) + 1;
        i2 = i2 < 60 ? i2 : 60;
        igr1 = ISH_IGRDATA[i1 - 1];
        igr2 = ISH_IGRDATA[i2 - 1];
        return dum * igr1 + (1.f - dum) * igr2;
    } else if (tc > -1.f && tc <= 0.f) {
        dum = 1.0f - fabsf(temp - ISH_T0);
        igr1 = 1.0f;
        igr2 = ISH_IGRDATA[0];
        return dum * igr1 + (1.f - dum) * igr2;
    } else if (tc >= -60.f && tc < -59.f) {
        dum = 60.0f - fabsf(temp - ISH_T0);
        igr1 = ISH_IGRDATA[58];
        igr2 = ISH_IGRDATA[59];
        return dum * igr1 + (1.f - dum) * igr2;
    } else if (tc < -60.f) {
        return ISH_IGRDATA[59];
    }
    return 1.f;
}

// access_lookup_table (:3487-3547).  tab is (51,51,51,11,nidx) in file order
// (i1 slowest ... i4, then the value index fastest).
__device__ __forceinline__ float ish_tab(const float* tab, int nidx, int jj, int ii,
                                         int i, int k, int idx)
{
    size_t off = ((((size_t)(jj - 1) * 51 + (ii - 1)) * 51 + (i - 1)) * 11 + (k - 1));
    return tab[off * nidx + (idx - 1)];
}

__device__ float ish_access_lookup_table(const float* tab, int nidx, int dumjj,
        int dumii, int dumi, int dumk, int index, float dum1, float dum2,
        float dum4, float dum5)
{
    float dproc1, dproc2, iproc1, gproc1, tmp1, tmp2;
    dproc1 = ish_tab(tab, nidx, dumjj, dumii, dumi, dumk, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj, dumii, dumi + 1, dumk, index) - ish_tab(tab, nidx, dumjj, dumii, dumi, dumk, index));
    dproc2 = ish_tab(tab, nidx, dumjj, dumii, dumi, dumk + 1, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj, dumii, dumi + 1, dumk + 1, index) - ish_tab(tab, nidx, dumjj, dumii, dumi, dumk + 1, index));
    iproc1 = dproc1 + (dum2 - (float)dumk) * (dproc2 - dproc1);
    dproc1 = ish_tab(tab, nidx, dumjj, dumii + 1, dumi, dumk, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj, dumii + 1, dumi + 1, dumk, index) - ish_tab(tab, nidx, dumjj, dumii + 1, dumi, dumk, index));
    dproc2 = ish_tab(tab, nidx, dumjj, dumii + 1, dumi, dumk + 1, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj, dumii + 1, dumi + 1, dumk + 1, index) - ish_tab(tab, nidx, dumjj, dumii + 1, dumi, dumk + 1, index));
    gproc1 = dproc1 + (dum2 - (float)dumk) * (dproc2 - dproc1);
    tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    dproc1 = ish_tab(tab, nidx, dumjj + 1, dumii, dumi, dumk, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj + 1, dumii, dumi + 1, dumk, index) - ish_tab(tab, nidx, dumjj + 1, dumii, dumi, dumk, index));
    dproc2 = ish_tab(tab, nidx, dumjj + 1, dumii, dumi, dumk + 1, index) +
        (dum1 - (float)dumi) * (ish_tab(tab, nidx, dumjj + 1, dumii, dumi + 1, dumk + 1, index) -
        ish_tab(tab, nidx, dumjj + 1, dumii, dumi, dumk + 1, index));
    iproc1 = dproc1 + (dum2 - (float)dumk) * (dproc2 - dproc1);
    dproc1 = ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi, dumk, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi + 1, dumk, index) - ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi, dumk, index));
    dproc2 = ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi, dumk + 1, index) + (dum1 - (float)dumi) *
        (ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi + 1, dumk + 1, index) - ish_tab(tab, nidx, dumjj + 1, dumii + 1, dumi, dumk + 1, index));
    gproc1 = dproc1 + (dum2 - (float)dumk) * (dproc2 - dproc1);
    tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

// var_check (:2968-3063).
__device__ void ish_var_check(const float* gtab, float gammnu, float fourthirdspi,
        float qidum, float& dsdum, float& ani, float& cni, float& rni,
        float& rbdum, float& nidum, float& aidum, float& cidum,
        float& alphstr, float& alphv, float& betam)
{
    const float ao = ISH_AO;
    float maxsize, gamma_arg;
    int gi;
    if (dsdum < 0.55f) {
        dsdum = 0.55f;
        ani = gfk_pow(cni / ISH_AO_POW_045, 1.f / dsdum);
        aidum = ani * ani * cni * nidum;
        cidum = cni * cni * ani * nidum;
    } else if (dsdum > 1.3f) {
        dsdum = 1.3f;
        cni = ISH_AO_POW_M03 * gfk_pow(ani, dsdum);
        aidum = ani * ani * cni * nidum;
        cidum = cni * cni * ani * nidum;
    }
    alphstr = gfk_pow(ao, 1.f - dsdum);
    alphv = fourthirdspi * alphstr;
    betam = 2.0f + dsdum;
    gamma_arg = ISH_NU + 2.f + dsdum;
    gi = ish_gi(gamma_arg);
    if (ani > 2.e-6f) {
        rbdum = qidum * gammnu / (nidum * alphv * gfk_pow(ani, betam) * gtab[gi - 1]);
    } else {
        rbdum = ISH_RHOI;
    }
    if (rbdum > ISH_RHOI) {
        rbdum = ISH_RHOI;
        ani = gfk_pow((qidum * gammnu) / (rbdum * nidum * alphv * gtab[gi - 1]), 1.f / betam);
        cni = gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, dsdum);
        aidum = ani * ani * cni * nidum;
        cidum = cni * cni * ani * nidum;
    } else if (rbdum < 50.f) {
        rbdum = 50.f;
        ani = gfk_pow((qidum * gammnu) / (rbdum * nidum * alphv * gtab[gi - 1]), 1.f / betam);
        cni = gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, dsdum);
        aidum = ani * ani * cni * nidum;
        cidum = cni * cni * ani * nidum;
    }
    rni = gfk_pow(qidum * 3.f / (nidum * rbdum * 4.f * ISH_PI * (gtab[gi - 1] / gammnu)), 0.333333333333f);
    if (rni < 2.e-6f) {
        rni = 2.e-6f;
        nidum = 3.f * qidum * gammnu / (4.f * ISH_PI * rbdum * (rni * rni * rni) * (gtab[gi - 1]));
        ani = gfk_pow((qidum * gammnu) / (rbdum * nidum * alphv * gtab[gi - 1]), 1.f / betam);
        cni = gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, dsdum);
        aidum = ani * ani * cni * nidum;
        cidum = cni * cni * ani * nidum;
    }
    maxsize = ish_max(ani, cni);
    if (maxsize > 1.e-3f) {
        if (ani >= cni) {
            ani = 1.e-3f;
            nidum = qidum * gammnu / (fourthirdspi * rbdum * gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, 2.f + dsdum) * (gtab[gi - 1]));
            cni = gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, dsdum);
            aidum = ani * ani * cni * nidum;
            cidum = cni * cni * ani * nidum;
        } else {
            cni = 1.e-3f;
            ani = gfk_pow(cni / (gfk_pow(ao, 1.f - dsdum)), 1.f / dsdum);
            nidum = qidum * gammnu / (fourthirdspi * rbdum * gfk_pow(ao, 1.f - dsdum) * gfk_pow(ani, 2.f + dsdum) * (gtab[gi - 1]));
            aidum = ani * ani * cni * nidum;
            cidum = cni * cni * ani * nidum;
        }
        rni = gfk_pow(qidum / (nidum * rbdum * fourthirdspi * (gtab[gi - 1] / gammnu)), 0.333333333333f);
    }
}

// capacitance_gamma (:3355-3400).
__device__ float ish_capacitance_gamma(float ani, float dsdum, float alphstr, float i_gammnu)
{
    float a1, a2, b1, b2, c1 = 0.f, c2 = 0.f, d1 = 0.f, d2 = 0.f, gammad1, gammad2;
    float cap = 0.f;
    const float nu = ISH_NU;
    if (dsdum <= 1.0f) {
        a1 = 0.6369427f;
        a2 = 0.57f * a1;
        b1 = 0.0f;
        b2 = 0.95f;
        c1 = a1 * gfk_pow(alphstr, b1);
        c2 = a2 * gfk_pow(alphstr, b2);
        d1 = b1 * (dsdum - 1.0f) + 1.0f;
        d2 = b2 * (dsdum - 1.0f) + 1.0f;
    } else if (dsdum > 1.f) {
        a1 = 0.5714285f;
        a2 = 0.75f * a1;
        b1 = -1.0f;
        b2 = -0.18f;
        c1 = a1 * gfk_pow(alphstr, b1 + 1.0f);
        c2 = a2 * gfk_pow(alphstr, b2 + 1.0f);
        d1 = b1 * (dsdum - 1.0f) + dsdum;
        d2 = b2 * (dsdum - 1.0f) + dsdum;
    }
    if (dsdum == 1.f && nu == 1.f) {
        gammad1 = ish_gamma(nu + 1.0f);
        cap = ani * gammad1 * i_gammnu;
    } else if (dsdum <= 1.f) {
        gammad1 = ish_gamma(nu + d1);
        gammad2 = ish_gamma(nu + d2);
        cap = c1 * gfk_pow(ani, d1) * gammad1 * i_gammnu +
              c2 * gfk_pow(ani, d2) * gammad2 * i_gammnu;
    } else if (dsdum > 1.f) {
        gammad1 = ish_gamma(nu + d1);
        gammad2 = ish_gamma(nu + d2);
        cap = c1 * gfk_pow(ani, d1) * gammad1 * i_gammnu +
              c2 * gfk_pow(ani, d2) * gammad2 * i_gammnu;
    }
    return cap;
}

// The Best-number fall speed and ventilation block shared by vaporgrow
// (:3104-3163) and the final fall speeds (:2501-2556).
struct IshFall {
    float xn, bx, am, bm, Nre, xm;
};

__device__ IshFall ish_best(float phii, float rb, float rhoair, float mu,
                            float alphstr, float dsdum, float fourthirdspi,
                            float ani, float i_gammnu)
{
    float bl = 0.f, al = 0.f, aa = 0.f, ba = 0.f, qe = 0.f;
    if (phii < 1.0f) {
        bl = 1.f; al = 2.f; aa = ISH_PI; ba = 2.f;
        qe = (1.f - phii) * (rb / ISH_RHOI) + phii;
    } else if (phii > 1.0f) {
        al = 2.0f; bl = 1.f; aa = ISH_PI * alphstr; ba = dsdum + 1.f; qe = 1.0f;
    } else if (phii == 1.0f) {
        bl = 1.f; al = 2.f; aa = ISH_PI; ba = 2.f; qe = 1.0f;
    }
    qe = ish_min(qe, 1.0f);
    IshFall f;
    f.xn = 2.f / rb * (rb - rhoair) * ISH_G * rhoair / (mu * mu) *
           (fourthirdspi * rb) * alphstr * (al * al) / (aa * qe) * gfk_pow(qe, 3.f / 4.f);
    f.bx = dsdum + 2.f + 2.f * bl - ba;
    f.xm = f.xn * gfk_pow(ani, f.bx) * (ish_gamma(ISH_NU + f.bx)) * i_gammnu;
    const float f_c1 = ISH_F_C1;
    const float f_c2 = (5.83f * 5.83f) / 4.0f;
    float s1 = sqrtf(1.0f + f_c1 * sqrtf(f.xm));
    f.bm = ((f_c1 * sqrtf(f.xm)) / (2.0f * (s1 - 1.0f) * (s1))) -
           (1.0e-5f * f.xm) / (f_c2 * ((s1 - 1.0f) * (s1 - 1.0f)));
    f.am = ((f_c2 * ((s1 - 1.0f) * (s1 - 1.0f))) - (1.0e-5f * f.xm)) / gfk_pow(f.xm, f.bm);
    if (f.xm > 1.e8f) {
        f.am = 1.0865f;
        f.bm = 0.499f;
    }
    f.Nre = f.am * gfk_pow(f.xm, f.bm);
    return f;
}

// vaporgrow (:3070-3348).  vtbarbz (the reflectivity-weighted speed) is not
// formed: me_ishmael overwrites vtrzi1 for every species before reading it
// (:2635-2643), so it never reaches an output.  alphstr is the calling
// level's ao**(1.-deltastr) (:1128), passed as the Fortran passes it.
__device__ void ish_vaporgrow(const float* gtab, float dt, float ani, float cni,
        float rni, float igr, float nidum, float temp, float rimesum,
        float alphstr, float sui, float sup, float qvs, float qvi, float mu,
        float iwci, float rhodum, float dv, float kt, float nsch, float npr,
        float gammnu, float i_gammnu, float fourthirdspi, float svpi,
        float xxls, float xxlf, float capgam, float& vtbarb, float& vtbarbm,
        float& anf, float& cnf, float& rnf, float& iwcf, float& fvdum,
        float& fhdum, float& rbdum, float dsdum, float& rdout,
        float& dsdumout)
{
    const float QASMALLV = 1.e-19f;
    const float nu = ISH_NU;
    float gammnubet, phii, fs, alphanr;
    float maxsui, rhodep, vtbranch, alpha, del1, del2, del, afn;
    float xvent, ntherm, bv1, bv2, gv, bt1, bt2, gt;
    float videp, vmin, betavol, vi, vf, phif, gamma_arg, rbdumtmp;
    int gi;
    fvdum = 1.0f;
    fhdum = 1.0f;
    dsdumout = dsdum;
    gamma_arg = ISH_NU - 1.f + dsdum;
    gi = ish_gi(gamma_arg);
    phii = cni / ani * gtab[gi - 1] * i_gammnu;
    fs = capgam / rni;
    alphanr = ani / gfk_pow(rni, 3.f / (2.f + igr));

    IshFall f = ish_best(phii, rbdum, rhodum, mu, alphstr, dsdum, fourthirdspi,
                         ani, i_gammnu);
    vtbarb = mu / rhodum * 0.5f * f.am * gfk_pow(f.xn, f.bm) * gfk_pow(ani, f.bx * f.bm - 1.f) *
             (ish_gamma(nu + f.bx * f.bm - 1.f)) * i_gammnu;
    vtbarbm = mu / rhodum * 0.5f * f.am * gfk_pow(f.xn, f.bm) * gfk_pow(ani, f.bx * f.bm - 1.f) *
              (ish_gamma(nu + f.bx * f.bm - 1.f + 2.f + dsdum)) / (ish_gamma(nu + 2.f + dsdum));

    xvent = gfk_pow(nsch, 0.333333333f) * gfk_pow(f.Nre, 0.5f);
    ntherm = gfk_pow(f.Nre, 0.5f) * gfk_pow(npr, 0.333333333f);
    if (xvent <= 1.0f) {
        bv1 = 1.0f; bv2 = 0.14f; gv = 2.f;
    } else {
        bv1 = 0.86f; bv2 = 0.28f; gv = 1.f;
    }
    if (ntherm < 1.4f) {
        bt1 = 1.0f; bt2 = 0.108f; gt = 2.0f;
    } else {
        bt1 = 0.78f; bt2 = 0.308f; gt = 1.0f;
    }
    fvdum = bv1 + bv2 * gfk_pow(xvent, gv);
    fhdum = bt1 + bt2 * gfk_pow(ntherm, gt);

    if (temp <= ISH_T0) {
        vtbranch = vtbarbm;
        if (sup >= 0.f) {
            maxsui = 1.f;
        } else if (sui >= 0.f && qvi < qvs) {
            maxsui = ((sui + 1.f) * qvi) / (qvs - qvi) - qvi / (qvs - qvi);
            maxsui = ish_min(maxsui, 1.f);
            maxsui = ish_max(maxsui, 0.f);
        } else {
            maxsui = 0.f;
        }
        if (igr <= 1.f) {
            if (vtbranch > 0.f) {
                if (ani > sqrtf((dv * ISH_PI * 2.f * cni) / (vtbranch * nu))) {
                    rhodep = (ISH_RHOI * igr) * maxsui + ISH_RHOI * (1.f - maxsui);
                } else {
                    rhodep = ISH_RHOI;
                }
            } else {
                rhodep = ISH_RHOI;
            }
        } else {
            rhodep = (ISH_RHOI / igr) * maxsui + ISH_RHOI * (1.f - maxsui);
        }
        rhodep = ish_min(rhodep, 700.f);

        alpha = (dv * fvdum * svpi * xxls) / (ISH_RV * kt * fhdum * temp);
        if (nidum > 0.0f && rimesum > 0.0f) {
            del1 = (xxlf * (rimesum / nidum) / (4.f * ISH_PI * kt * fhdum * capgam)) *
                   (1.0f / (temp + alpha * ((xxls / (ISH_RV * temp)) - 1.f)));
        } else {
            del1 = 0.0f;
        }
        del2 = sui * (1.0f / ((temp / alpha) + ((xxls / (ISH_RV * temp)) - 1.f)));
        del = del1 + del2;
        afn = ((dv * fvdum * ish_polysvp(temp, 1)) / (ISH_RV * temp)) *
              (sui - del * ((xxls / (ISH_RV * temp)) - 1.f));
        if (afn < 0.0f) {
            rhodep = rbdum;
            videp = rni * rni * rni;
            vmin = ISH_VMIN;
            if (vmin < videp) {
                betavol = gfk_log(ISH_RHOI / rbdum) * 1.f / (gfk_log(vmin / videp));
                rhodep = rbdum * (1.f + betavol);
            } else {
                rhodep = rbdum;
            }
        }
        rhodep = ish_max(rhodep, 50.f);
        rhodep = ish_min(rhodep, ISH_RHOI);

        gamma_arg = ISH_NU + 2.f + dsdum;
        gi = ish_gi(gamma_arg);
        gammnubet = gtab[gi - 1];

        rnf = gfk_pow(ish_max((rni * rni + 2.f * afn * fs / rhodep * gammnu / gammnubet * dt),
                              QASMALLV), 0.5f);
        anf = alphanr * gfk_pow(rnf, 3.f / (2.f + igr));

        phif = phii * gfk_pow((rnf * rnf * rnf) / (rni * rni * rni), (igr - 1.f) / (igr + 2.f));
        if (sui < 0.0f || afn < 0.0f) {
            if (phii > 1.0f && phif < 1.0f) {
                phif = phii; alphanr = ani / rni; anf = alphanr * rnf;
            }
            if (phii < 1.0f && phif > 1.0f) {
                phif = phii; alphanr = ani / rni; anf = alphanr * rnf;
            }
            if (phii > 1.f) {
                if (phif > phii) {
                    phif = phii; alphanr = ani / rni; anf = alphanr * rnf;
                }
            } else {
                if (phif < phii) {
                    phif = phii; alphanr = ani / rni; anf = alphanr * rnf;
                }
            }
        }

        vi = fourthirdspi * (rni * rni * rni) * gtab[gi - 1] * i_gammnu;
        vf = fourthirdspi * (rnf * rnf * rnf) * gtab[gi - 1] * i_gammnu;
        rdout = rhodep;
        rbdumtmp = rbdum * (vi / vf) + rhodep * (1.f - vi / vf);
        rbdumtmp = ish_min(rbdumtmp, ISH_RHOI);
        iwcf = nidum * rbdumtmp * vf;

        if (igr != 1.0f) {
            if (anf > (1.1f * ISH_AO)) {
                dsdumout = (3.f * gfk_log(rnf) - 2.f * gfk_log(anf) - ISH_LOG_AO) /
                           (gfk_log(anf) - ISH_LOG_AO);
            } else {
                dsdumout = 1.f;
            }
        }
        if (afn < 0.f && rnf < 1.e-6f) {
            rbdum = ISH_RHOI; rdout = ISH_RHOI; dsdumout = 1.f; phif = 1.f;
            alphanr = ani / rni; anf = alphanr * rnf;
        }
        if (afn < 0.f && anf < 1.e-6f) {
            rbdum = ISH_RHOI; rdout = ISH_RHOI; dsdumout = 1.f; phif = 1.f;
            alphanr = ani / rni; anf = alphanr * rnf;
        }
        if (afn < 0.f && dsdumout <= 0.f) {
            dsdumout = 1.f;
            anf = rnf;
        }
        gamma_arg = ISH_NU - 1.f + dsdumout;
        gi = ish_gi(gamma_arg);
        cnf = phif * anf * gammnu / gtab[gi - 1];
    } else {
        anf = ani;
        cnf = cni;
        rnf = rni;
        iwcf = iwci;
    }
}

// wet_growth_check (:3404-3423): returns the updated dry-growth flag.
__device__ bool ish_wet_growth_dry(float temp, float rhodum, float xxlv, float xxlf,
        float qvdum, float dv, float kt, float qs0, float fvdum, float fhdum,
        float rimedum, float rni, float nidum, bool dgflag)
{
    float dum = nidum * ISH_NU * 2.f * rni;
    float wetg = 2.f * ISH_PI * dum * (kt * fhdum * (ISH_T0 - temp) +
                 rhodum * xxlv * dv * fvdum * (qs0 - qvdum)) / (xxlf + (ISH_CPW * (temp - ISH_T0)));
    if (rimedum / rhodum > wetg) dgflag = false;
    return dgflag;
}

// ---------------------------------------------------------------------------
// Aggregation (:3855-4341) and col1 (:4346-4408) for the one level the
// caller passes (nz = 1).  Only categories 3 (planar), 4 (columnar) and 5
// (aggregates) take part; col1's pair index selects one of the seven
// collection tables ishk_coltab builds (slot order below).
// ---------------------------------------------------------------------------
#define ISH_SLOT_33 0
#define ISH_SLOT_43 1
#define ISH_SLOT_34 2
#define ISH_SLOT_44 3
#define ISH_SLOT_35 4
#define ISH_SLOT_45 5
#define ISH_SLOT_55 6

struct IshAggTab {
    int ict[3];
    float wct1[3], wct2[3];
};

__device__ __forceinline__ float ish_coltab(const float* tab, int slot, int idnx, int idny)
{
    return tab[((size_t)slot * ISH_NDN + (idny - 1)) * ISH_NDN + (idnx - 1)];
}

// One col1 call (icf = 5): collector mx collects my into category 5; mx and
// my are 0..2 for categories 3..5.
__device__ void ish_col1(const IshTables& T, int slot, int mx, int my,
        float dtlt, const float* en, const float* r, const float* t,
        const IshAggTab& g, float rhoa, float denfac, float efdum,
        float& rx, float& enx)
{
    const float cr = 1.f;
    float tmax = ish_max(t[mx], t[my]);
    float eff = ish_min(gfk_pow(10.f, 0.035f * tmax - 0.7f), 0.2f) * efdum;
    if (fabsf(t[mx] + 14.f) <= 2.f) eff = 1.4f * efdum;   // mz is 5 on every call
    int ix = g.ict[mx], iy = g.ict[my];
    float colamt = dtlt * cr * 0.785f * eff * denfac / rhoa * en[mx] * en[my] * (
        g.wct1[mx] * g.wct1[my] * ish_coltab(T.coltab, slot, ix, iy) +
        g.wct2[mx] * g.wct1[my] * ish_coltab(T.coltab, slot, ix + 1, iy) +
        g.wct1[mx] * g.wct2[my] * ish_coltab(T.coltab, slot, ix, iy + 1) +
        g.wct2[mx] * g.wct2[my] * ish_coltab(T.coltab, slot, ix + 1, iy + 1));
    colamt = ish_min(colamt, r[mx]);
    rx = rx + colamt;
    float colamtn = dtlt * cr * 0.785f * eff * denfac / rhoa * en[mx] * en[my] * (
        g.wct1[mx] * g.wct1[my] * ish_coltab(T.coltabn, slot, ix, iy) +
        g.wct2[mx] * g.wct1[my] * ish_coltab(T.coltabn, slot, ix + 1, iy) +
        g.wct1[mx] * g.wct2[my] * ish_coltab(T.coltabn, slot, ix, iy + 1) +
        g.wct2[mx] * g.wct2[my] * ish_coltab(T.coltabn, slot, ix + 1, iy + 1));
    colamtn = ish_min(colamtn, en[mx]);
    float wght = ish_min(ish_max(100.f * fabsf(colamt) / ish_max(r[mx], 1.e-20f), 0.f), 1.f);
    float deltan = colamtn * (1.f - wght) +
                   wght * en[mx] * colamt / ish_max(r[mx], 1.e-20f);
    enx = enx + deltan;
}

// The density/shape collection-efficiency factor (:4081-4109 and copies).
__device__ float ish_efffact(float rhoeffmax, float phieffmax)
{
    float rhoeff, phieff;
    if (rhoeffmax <= 400.f) {
        rhoeff = 1.f;
    } else {
        rhoeff = -0.001923f * rhoeffmax + 1.76916f;
    }
    rhoeff = ish_max(rhoeff, 0.f);
    rhoeff = ish_min(rhoeff, 1.f);
    if (phieffmax <= 0.03f) {
        phieff = 1.f;
    } else if (phieffmax > 0.03f && phieffmax < 0.5f) {
        phieff = 0.0001f * gfk_pow(phieffmax + 0.07f, -4.f);
    } else {
        phieff = 0.f;
    }
    if (phieff <= 0.001f) phieff = 0.f;
    phieff = ish_max(phieff, 0.01f);
    phieff = ish_min(phieff, 1.f);
    return phieff * rhoeff;
}

__device__ void ish_aggregation(const IshTables& T, float dtdum, float rhoairdum,
        float tempdum, float qdum1, float ndum1, float ddum1, float qdum2,
        float ndum2, float ddum2, float qdum3, float ndum3, float& ddum3,
        float rhodum1, float rhodum2, float phidum1, float phidum2,
        float& qagg1, float& qagg2, float& qagg3, float& nagg1, float& nagg2,
        float& nagg3)
{
    const float QNS = 1.25e-7f;
    const float tabloln = ISH_LOG_1EM6, tabhiln = ISH_LOG_1EM2;
    const float dict = (float)(ISH_NDN - 1) / (tabhiln - tabloln);
    const float rictmin = 1.0001f;
    const float rictmax = 0.9999f * (float)ISH_NDN;
    const float cfmas5 = 3.14159f / 6.f * 50.f * 0.2f;
    const float pwmas5 = 3.0f;
    const float gnu5 = 4.0f;

    float rhoa = rhoairdum;
    float denfac = sqrtf(1.f / rhoa);
    float dn[3] = {ddum1, ddum2, ddum3};
    float en[3] = {ndum1 * rhoa, ndum2 * rhoa, ndum3 * rhoa};
    float r[3] = {qdum1, qdum2, qdum3};
    float tc = tempdum - 273.15f;
    float t[3] = {tc, tc, tc};
    IshAggTab g;
    for (int m = 0; m < 3; ++m) {
        float rict = dict * (gfk_log(ish_max(dn[m], 1.e-10f)) - tabloln) + 1.f;
        float rictmm = ish_max(ish_min(rict, rictmax), rictmin);
        g.ict[m] = (int)rictmm;
        g.wct2[m] = rictmm - (float)g.ict[m];
        g.wct1[m] = 1.0f - g.wct2[m];
    }
    float rxfer3 = 0.f, rxfer4 = 0.f, enxfer3 = 0.f, enxfer4 = 0.f;
    float parxfer3 = 0.f, parxfer4 = 0.f, paenxfer3 = 0.f, paenxfer4 = 0.f;
    float pprxfer3 = 0.f, pprxfer4 = 0.f, pprxfer5 = 0.f;
    float ppenxfer3 = 0.f, ppenxfer4 = 0.f, ppenxfer5 = 0.f;
    float efffact;

    efffact = ish_efffact(ish_max(rhodum1, rhodum2),
                          ish_max(ish_min(phidum1, 1.f / phidum1), ish_min(phidum2, 1.f / phidum2)));
    ish_col1(T, ISH_SLOT_34, 0, 1, dtdum, en, r, t, g, rhoa, denfac, efffact, rxfer3, enxfer3);
    ish_col1(T, ISH_SLOT_43, 1, 0, dtdum, en, r, t, g, rhoa, denfac, efffact, rxfer4, enxfer4);
    efffact = ish_efffact(rhodum1, ish_min(phidum1, 1.f / phidum1));
    ish_col1(T, ISH_SLOT_35, 0, 2, dtdum, en, r, t, g, rhoa, denfac, efffact, parxfer3, paenxfer3);
    efffact = ish_efffact(rhodum2, ish_min(phidum2, 1.f / phidum2));
    ish_col1(T, ISH_SLOT_45, 1, 2, dtdum, en, r, t, g, rhoa, denfac, efffact, parxfer4, paenxfer4);
    efffact = ish_efffact(rhodum1, ish_min(phidum1, 1.f / phidum1));
    ish_col1(T, ISH_SLOT_33, 0, 0, dtdum, en, r, t, g, rhoa, denfac, efffact, pprxfer3, ppenxfer3);
    efffact = ish_efffact(rhodum2, ish_min(phidum2, 1.f / phidum2));
    ish_col1(T, ISH_SLOT_44, 1, 1, dtdum, en, r, t, g, rhoa, denfac, efffact, pprxfer4, ppenxfer4);
    ish_col1(T, ISH_SLOT_55, 2, 2, dtdum, en, r, t, g, rhoa, denfac, 1.f, pprxfer5, ppenxfer5);

    const float rcrossloss = 1.0f, rselfloss = 1.f, rcrossgain = 1.0f, rselfgain = 1.0f;
    const float ncrossloss = 1.f, nselfloss = 1.f, ncrossgain = 1.f / 2.f, nselfgain = 1.f / 2.f;
    float sink3 = rcrossloss * rxfer3 + rcrossloss * parxfer3 + rselfloss * pprxfer3;
    float source3 = r[0];
    float sink4 = rcrossloss * rxfer4 + rcrossloss * parxfer4 + rselfloss * pprxfer4;
    float source4 = r[1];
    if (sink3 > source3 && source3 > 1.e-8f) {
        float ratioagg = source3 / sink3;
        rxfer3 = rxfer3 * ratioagg;
        parxfer3 = parxfer3 * ratioagg;
        pprxfer3 = pprxfer3 * ratioagg;
    }
    if (sink4 > source4 && source4 > 1.e-8f) {
        float ratioagg = source4 / sink4;
        rxfer4 = rxfer4 * ratioagg;
        parxfer4 = parxfer4 * ratioagg;
        pprxfer4 = pprxfer4 * ratioagg;
    }
    qagg1 = -rcrossloss * rxfer3 - rcrossloss * parxfer3 - rselfloss * pprxfer3;
    nagg1 = -ncrossloss * enxfer3 - ncrossloss * paenxfer3 - nselfloss * ppenxfer3;
    qagg2 = -rcrossloss * rxfer4 - rcrossloss * parxfer4 - rselfloss * pprxfer4;
    nagg2 = -ncrossloss * enxfer4 - ncrossloss * paenxfer4 - nselfloss * ppenxfer4;
    qagg3 = rcrossgain * rxfer3 + rcrossgain * rxfer4 + rcrossgain * parxfer3 +
            rcrossgain * parxfer4 + rselfgain * pprxfer3 + rselfgain * pprxfer4;
    nagg3 = ncrossgain * enxfer3 + ncrossgain * enxfer4 + nselfgain * ppenxfer3 +
            nselfgain * ppenxfer4 - 0.5f * ppenxfer5;
    qagg1 = ish_min(qagg1, 0.f);
    qagg2 = ish_min(qagg2, 0.f);
    qagg3 = ish_max(qagg3, 0.f);
    dn[2] = gfk_pow((r[2] + qagg3) * rhoa / (ish_max(en[2] + nagg3, QNS) * cfmas5 *
                    (ish_gamma(gnu5 + pwmas5)) / (ish_gamma(gnu5))), 1.f / pwmas5);
    ddum3 = dn[2];
    nagg1 = nagg1 / rhoa;
    nagg2 = nagg2 / rhoa;
    nagg3 = nagg3 / rhoa;
}

// ---------------------------------------------------------------------------
// The aggregation collection tables (mkcoltb, :4413-4464, with xjnum,
// gammap/gammq, lowgseries/highgcontfrac, gammln and AVINT), built once per
// process on the device by ishk_coltab: one thread per (pair, idny, idnx).
// Only the seven pairs aggregation reads are built; the other 28 of WRF's
// 35 are never indexed.
// ---------------------------------------------------------------------------
// dstprms columns (:88-103), categories 1..7.
__device__ const float ISH_CFMAS[7] = {524.f, 524.f, .333f, .333f, .496f, 157.f, 471.f};
__device__ const float ISH_PWMAS[7] = {3.f, 3.f, 2.4f, 2.4f, 2.4f, 3.f, 3.f};
__device__ const float ISH_CFVT[7] = {3173.f, 149.f, 4.836f, 4.836f, 3.084f, 93.3f, 161.f};
__device__ const float ISH_PWVT[7] = {2.f, .5f, 0.25f, .25f, .2f, .5f, .5f};
// slot -> (ix, iy), 1-based categories, in ISH_SLOT_* order.
__device__ const int ISH_PAIR_IX[7] = {3, 4, 3, 4, 3, 4, 5};
__device__ const int ISH_PAIR_IY[7] = {3, 3, 4, 4, 5, 5, 5};

__device__ float ish_gammln(float xx)
{
    const double cof[6] = {76.18009172947146, -86.50532032941677,
                           24.01409824083091, -1.231739572450155,
                           0.1208650973866179e-2, -0.5395239384953e-5};
    const double stp = 2.5066282746310005;
    double x = (double)xx, y = x;
    double tmp = x + 5.5;
    tmp = (x + 0.5) * ish_log64(tmp) - tmp;
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; ++j) {
        y = y + 1.0;
        ser = ser + cof[j] / y;
    }
    return (float)(tmp + ish_log64(stp * ser / x));
}

__device__ float ish_lowgseries(float a, float x)
{
    const float EPS = 3.0e-7f;
    float gln = ish_gammln(a);
    if (x <= 0.f) return 0.0f;
    float ap = a, sum = 1.f / a, del = sum;
    for (int n = 1; n <= 100; ++n) {
        ap = ap + 1.f;
        del = del * x / ap;
        sum = sum + del;
        if (fabsf(del) < fabsf(sum) * EPS) break;
    }
    return sum * gfk_exp(-x + a * gfk_log(x) - gln);
}

__device__ float ish_highgcontfrac(float a, float x)
{
    const float EPS = 3.0e-7f, FPMIN = 1.0e-30f;
    float gln = ish_gammln(a);
    float b = x + 1.f - a;
    float c = 1.f / FPMIN;
    float d = 1.f / b;
    float h = d, an, del;
    for (int n = 1; n <= 100; ++n) {
        an = -((float)n * ((float)n - a));
        b = b + 2.f;
        d = an * d + b;
        if (fabsf(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (fabsf(c) < FPMIN) c = FPMIN;
        d = 1.f / d;
        del = d * c;
        h = h * del;
        if (fabsf(del - 1.f) < EPS) break;
    }
    return gfk_exp(-x + a * gfk_log(x) - gln) * h;
}

__device__ float ish_gammap(float a, float x)
{
    if (x < (a + 1.0f)) return ish_lowgseries(a, x);
    return 1.0f - ish_highgcontfrac(a, x);
}

__device__ float ish_gammq(float a, float x)
{
    if (x < a + 1.f) return 1.f - ish_lowgseries(a, x);
    return ish_highgcontfrac(a, x);
}

__device__ float ish_xjnum(float dx, float cvx, float pvx, float cvy, float pvy,
        float vny, float dnx, float dny, float xnu, float ynu, float gyn1,
        float gyn2, float gynp, float gynp1, float gynp2)
{
    float dnxi = 1.f / dnx;
    float rdx = dx * dnxi;
    float vx = ish_max((cvx * gfk_pow(dx, pvx)), 1.e-6f);
    float dxy = gfk_pow(vx / cvy, 1.f / pvy) / dny;
    dxy = ish_max(dxy, 1.e-5f);
    dxy = ish_min(dxy, 70.f);
    float ynup = ynu + pvy;
    if (rdx < 38.f) {
        return gfk_exp(-rdx - ish_gammln(xnu) - ish_gammln(ynu)) * gfk_pow(rdx, xnu - 1.f) * dnxi * (
            vx * (dx * dx * (ish_gammap(ynu, dxy) - ish_gammq(ynu, dxy))
                  + 2.f * dx * dny * gyn1 * (ish_gammap(ynu + 1.f, dxy) - ish_gammq(ynu + 1.f, dxy))
                  + dny * dny * gyn2 * (ish_gammap(ynu + 2.f, dxy) - ish_gammq(ynu + 2.f, dxy)))
            - vny * (dx * dx * gynp * (ish_gammap(ynup, dxy) - ish_gammq(ynup, dxy))
                     + 2.f * dx * dny * gynp1 * (ish_gammap(ynup + 1.f, dxy) - ish_gammq(ynup + 1.f, dxy))
                     + dny * dny * gynp2 * (ish_gammap(ynup + 2.f, dxy) - ish_gammq(ynup + 2.f, dxy))));
    }
    return 0.f;
}

// AVINT (:4612-4704), REAL*8 inside, REAL in and out.  Returns NaN where
// the Fortran would STOP (it never does for the abscissas mkcoltb builds).
__device__ float ish_avint(const float* X, const float* Y, int N, float XLO, float XUP)
{
    double R3, RP5, SUM, SYL, SYL2, SYL3, SYU, SYU2, SYU3, X1, X2, X3, X12, X13, X23;
    double TERM1, TERM2, TERM3, A, B, C, CA = 0.0, CB = 0.0, CC = 0.0;
    float ANS = 0.0f;
    float d = XLO - XUP;
    if (d == 0.f) return ANS;
    if (!(d < 0.f)) return __int_as_float(0x7fc00000);
    if (N < 2) return __int_as_float(0x7fc00000);
    for (int I = 2; I <= N; ++I) {
        if (X[I - 1] <= X[I - 2]) return __int_as_float(0x7fc00000);
        if (X[I - 1] > XUP) break;
    }
    if (X[N - 3] < XLO) return __int_as_float(0x7fc00000);
    if (X[2] > XUP) return __int_as_float(0x7fc00000);
    int I = 1;
    while (X[I - 1] < XLO) I = I + 1;
    int INLFT = I;
    I = N;
    while (X[I - 1] > XUP) I = I - 1;
    int INRT = I;
    if ((INRT - INLFT) < 2) return __int_as_float(0x7fc00000);
    int ISTART = INLFT;
    if (INLFT == 1) ISTART = 2;
    int ISTOP = INRT;
    if (INRT == N) ISTOP = N - 1;
    R3 = 3.0;
    RP5 = 0.5;
    SUM = 0.0;
    SYL = (double)XLO;
    SYL2 = SYL * SYL;
    SYL3 = SYL2 * SYL;
    for (I = ISTART; I <= ISTOP; ++I) {
        X1 = (double)X[I - 2];
        X2 = (double)X[I - 1];
        X3 = (double)X[I];
        X12 = X1 - X2;
        X13 = X1 - X3;
        X23 = X2 - X3;
        TERM1 = (double)Y[I - 2] / (X12 * X13);
        TERM2 = -((double)Y[I - 1] / (X12 * X23));
        TERM3 = (double)Y[I] / (X13 * X23);
        A = TERM1 + TERM2 + TERM3;
        B = -((X2 + X3) * TERM1) - (X1 + X3) * TERM2 - (X1 + X2) * TERM3;
        C = X2 * X3 * TERM1 + X1 * X3 * TERM2 + X1 * X2 * TERM3;
        if (I - ISTART <= 0) {
            CA = A; CB = B; CC = C;
        } else {
            CA = 0.5 * (A + CA);
            CB = 0.5 * (B + CB);
            CC = 0.5 * (C + CC);
        }
        SYU = X2;
        SYU2 = SYU * SYU;
        SYU3 = SYU2 * SYU;
        SUM = SUM + CA * (SYU3 - SYL3) / R3 + CB * RP5 * (SYU2 - SYL2) + CC * (SYU - SYL);
        CA = A; CB = B; CC = C;
        SYL = SYU; SYL2 = SYU2; SYL3 = SYU3;
    }
    SYU = (double)XUP;
    ANS = (float)(SUM + CA * ((SYU * SYU) * SYU - SYL3) / R3 + CB * RP5 * ((SYU * SYU) - SYL2) +
                  CC * (SYU - SYL));
    return ANS;
}

extern "C" __global__ void ishk_coltab(float* coltab, float* coltabn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 7 * ISH_NDN * ISH_NDN) return;
    int slot = tid / (ISH_NDN * ISH_NDN);
    int rem = tid - slot * ISH_NDN * ISH_NDN;
    int idny = rem / ISH_NDN + 1;
    int idnx = rem - (idny - 1) * ISH_NDN + 1;
    const int ndn = ISH_NDN;
    const int ndx = 20;
    int ix = ISH_PAIR_IX[slot] - 1, iy = ISH_PAIR_IY[slot] - 1;
    const float gnux = 4.0f, gnuy = 4.0f;
    const float tablo = 1.e-6f, tabhi = 1.e-2f;
    float gyn = (ish_gamma(gnuy));
    float gyn1 = (ish_gamma(gnuy + 1.f)) / gyn;
    float gyn2 = (ish_gamma(gnuy + 2.f)) / gyn;
    float gynp = (ish_gamma(gnuy + ISH_PWVT[iy])) / gyn;
    float gynp1 = (ish_gamma(gnuy + ISH_PWVT[iy] + 1.f)) / gyn;
    float gynp2 = (ish_gamma(gnuy + ISH_PWVT[iy] + 2.f)) / gyn;
    float dxlo = tablo * 0.01f;
    float dxhi = tabhi * 10.0f;
    float dny = tablo * gfk_pow(tabhi / tablo, (float)(idny - 1) / (float)(ndn - 1));
    float vny = ISH_CFVT[iy] * gfk_pow(dny, ISH_PWVT[iy]);
    float dnx = tablo * gfk_pow(tabhi / tablo, (float)(idnx - 1) / (float)(ndn - 1));
    float dx[ndx], fx[ndx], gx[ndx];
    for (int idx = 1; idx <= ndx; ++idx) {
        dx[idx - 1] = dxlo * gfk_pow(dxhi / dxlo, (float)(idx - 1) / (float)(ndx - 1));
        fx[idx - 1] = ish_xjnum(dx[idx - 1], ISH_CFVT[ix], ISH_PWVT[ix], ISH_CFVT[iy],
                                ISH_PWVT[iy], vny, dnx, dny, gnux, gnuy,
                                gyn1, gyn2, gynp, gynp1, gynp2);
        if (fx[idx - 1] < 1.e-15f) fx[idx - 1] = 0.f;
        gx[idx - 1] = fx[idx - 1] * ISH_CFMAS[ix] * gfk_pow(dx[idx - 1], ISH_PWMAS[ix]);
    }
    float ans = ish_avint(dx, gx, ndx, dxlo, dxhi);
    coltab[tid] = ish_max(ans, 0.f);
    ans = ish_avint(dx, fx, ndx, dxlo, dxhi);
    coltabn[tid] = ish_max(ans, 0.f);
}

// ---------------------------------------------------------------------------
// The column: MP_JENSEN_ISHMAEL's per-column body (:351-451) around
// me_ishmael (:461-2962).  Fields are level-major (k * ncol + i), k = 0 at
// the surface (kts).
// ---------------------------------------------------------------------------
// Field slots in the pointer table F (the order gpuwm/core/ishmael.py
// FIELD_SLOTS names).
#define IF_P 0
#define IF_DZ 1
#define IF_TH 2
#define IF_QV 3
#define IF_QC 4
#define IF_QR 5
#define IF_NR 6
#define IF_QI 7      // + cc        (qi1, qi2, qi3)
#define IF_NI 10     // + cc
#define IF_AI 13     // + cc
#define IF_CI 16     // + cc
#define IF_NFIELDS 19
// Diagnostic slots in D.
#define ID_EFFC 0
#define ID_EFFI 1
#define ID_DBZ 2
#define ID_VMI 3     // + cc
#define ID_DI 6      // + cc
#define ID_RHOPO 9   // + cc
#define ID_PHII 12   // + cc
#define ID_ITYPE 15  // + cc
#define ID_NDIAG 18
// Scratch slots in W (level values carried from the level loop into
// sedimentation).
#define IW_RHO 0
#define IW_VTRM 1
#define IW_VTRN 2
#define IW_VTRMI 3   // + cc
#define IW_VTRNI 6   // + cc
#define IW_SEDI 9
#define IW_RHOBAR 10   // + cc, negative means inactive
#define IW_NSCRATCH 13
// Surface slots in S.
#define IS_RAINNC 0
#define IS_RAINNCV 1
#define IS_SNOWNC 2
#define IS_SNOWNCV 3

struct IshCtx {
    float gammnu, i_gammnu, i_dt, i_cp, fourthirdspi, dt;
};

// Rain slope with WRF's clamps (:987-997 and its four copies).
__device__ __forceinline__ void ish_rain_dsd(float qr, float& nr, float& lamr, float& n0rr)
{
    lamr = gfk_pow(ISH_PI * ISH_RHOW * nr / qr, 0.333333333f);
    n0rr = nr * lamr;
    if (lamr < ISH_LAMMINR) {
        lamr = ISH_LAMMINR;
        n0rr = (lamr * lamr) * (lamr * lamr) * qr / (ISH_PI * ISH_RHOW);
        nr = n0rr / lamr;
    } else if (lamr > ISH_LAMMAXR) {
        lamr = ISH_LAMMAXR;
        n0rr = (lamr * lamr) * (lamr * lamr) * qr / (ISH_PI * ISH_RHOW);
        nr = n0rr / lamr;
    }
}

// Macklin rime-density coefficient (:1184-1201, :1229-1246).
__device__ __forceinline__ float ish_rimec1(float temp)
{
    float rimec1 = 0.0066f, dum;
    float tc = temp - ISH_T0;
    if (tc < -30.f) {
        rimec1 = 0.0036f;
    } else if (tc < -20.f && tc >= -30.f) {
        dum = (fabsf(tc) - 20.f) / 10.f;
        rimec1 = dum * (0.0036f) + (1.f - dum) * (0.004f);
    } else if (tc < -15.f && tc >= -20.f) {
        dum = (fabsf(tc) - 15.f) / 5.f;
        rimec1 = dum * (0.004f) + (1.f - dum) * (0.005f);
    } else if (tc < -10.f && tc >= -15.f) {
        dum = (fabsf(tc) - 10.f) / 5.f;
        rimec1 = dum * (0.005f) + (1.f - dum) * (0.0066f);
    } else if (tc < -5.f && tc >= -10.f) {
        dum = (fabsf(tc) - 5.f) / 5.f;
        rimec1 = dum * (0.0066f) + (1.f - dum) * (0.012f);
    } else if (tc >= -5.f) {
        rimec1 = 0.012f;
    }
    return rimec1;
}

// Rime density from a normalized density/rate pair (:1202-1214).
__device__ __forceinline__ float ish_rime_density(float temp, float rimec1, float nrd,
                                                  float nrm, bool dry)
{
    float g = (1000.f * (0.8f * gfk_tanh(rimec1 * nrd / nrm) + 0.1f));
    float tc = temp - ISH_T0;
    if (tc > -5.f && tc <= 0.f) {
        float dum = (fabsf(tc) - 0.f) / 5.f;
        g = dum * g + (1.f - dum) * 900.f;
    }
    if (tc > 0.f || !dry) g = 900.f;
    g = ish_max(g, 50.f);
    g = ish_min(g, 900.f);
    return g;
}

// The bulk shape rediagnosis that recurs after every update (:1914-1920 and
// copies): ani/cni from the moments, moments from ani/cni, delta*.
__device__ __forceinline__ void ish_axes(float& ai, float& ci, float ni, float& ani,
                                         float& cni, float& ds)
{
    ani = ish_max(gfk_pow(((ai * ai) / (ci * ni)), 0.333333333333f), 2.e-6f);
    cni = ish_max(gfk_pow(((ci * ci) / (ai * ni)), 0.333333333333f), 2.e-6f);
    ci = cni * cni * ani * ni;
    ai = ani * ani * cni * ni;
    ds = (gfk_log(cni) - ISH_LOG_AO) / (gfk_log(ani) - ISH_LOG_AO);
}

__device__ __forceinline__ float ish_delta_or_one(float ani, float cni)
{
    if (ani > (1.1f * ISH_AO) && cni > (1.1f * ISH_AO))
        return (gfk_log(cni) - ISH_LOG_AO) / (gfk_log(ani) - ISH_LOG_AO);
    return 1.f;
}

// Melting and nucleation only make the distribution more spherical
// (:1969-1983, :2176-2190).
__device__ __forceinline__ float ish_toward_sphere(float ds, float dsold)
{
    if (dsold <= 1.f) {
        if (ds < dsold) ds = dsold;
        if (ds > 1.f) ds = 1.f;
    } else {
        if (ds > dsold) ds = dsold;
        if (ds < 1.f) ds = 1.f;
    }
    return ds;
}

// Aggregate final shape and implicit breakup (:2441-2486, :2877-2924).
// Above freezing the post-sedimentation copy recomputes ai/ci from the
// same ani/cni first; that rewrite is the identity, so one routine serves
// both sites.
__device__ void ish_aggregate_shape(const float* gtab, const IshCtx& C, float temp,
        float qi, float& ni, float& ai, float& ci, float& ani, float& cni,
        float& rni, float& rhobar, float& ds, float& alphstr, float& alphv,
        float& betam)
{
    if (temp <= ISH_T0) {
        rhobar = 50.f;
        float phiagg3 = 0.2f;
        int gi = ish_gi(ISH_NU - 1.f + ds);
        cni = phiagg3 * ani * C.gammnu / gtab[gi - 1];
        ai = ani * ani * cni * ni;
        ci = cni * cni * ani * ni;
    }
    ds = ish_delta_or_one(ani, cni);
    ds = ish_min(ds, 1.f);
    alphstr = gfk_pow(ISH_AO, 1.f - ds);
    int gi = ish_gi(ISH_NU + 2.f + ds);
    ani = gfk_pow((qi) / (ni * rhobar * C.fourthirdspi * alphstr * gtab[gi - 1] / C.gammnu),
                  1.f / (2.f + ds));
    ani = ish_max(ani, 2.e-6f);
    cni = alphstr * gfk_pow(ani, ds);
    if (ani > 0.5e-3f) {
        ani = 0.5e-3f;
        cni = gfk_pow(ISH_AO, 1.f - ds) * gfk_pow(ani, ds);
        gi = ish_gi(ISH_NU + 2.f + ds);
        ni = (qi) / (rhobar * C.fourthirdspi * gfk_pow(ISH_AO, 1.f - ds) *
                     gfk_pow(ani, 2.f + ds) * gtab[gi - 1] / C.gammnu);
        ni = ish_max(ni, ISH_QNSMALL);
    }
    ai = ani * ani * cni * ni;
    ci = cni * cni * ani * ni;
    ish_var_check(gtab, C.gammnu, C.fourthirdspi, qi, ds, ani, cni, rni, rhobar,
                  ni, ai, ci, alphstr, alphv, betam);
}

// One level of the me_ishmael k loop (:696-2727).  Prognostics in/out by
// reference; diagnostics written to D; fall speeds and density to W.
// Returns true when this level sets the column's sedimentation flag.
__device__ bool ish_level(const IshTables& T, const IshCtx& C, float pres, float& theta,
        float& qv, float& qc, float& qr, float& nr, float* qi, float* ni,
        float* ai, float* ci, float& effc1d, float& effi1d, float& dbz1d,
        float* vmi1d, float* di1d, float* rhopo1d, float* phii1d,
        float* icetype1d, float& qrpre1d, float* qipre1d, float* rhobar_c,
        float& rhoair_out,
        float& vtrm_out, float& vtrn_out, float* vtrmi_out, float* vtrni_out)
{
    const float* gtab = T.gtab;
    const float dt = C.dt, i_dt = C.i_dt, i_cp = C.i_cp;
    const float gammnu = C.gammnu, i_gammnu = C.i_gammnu, fourthirdspi = C.fourthirdspi;
    const float ao = ISH_AO;
    const int ICE1 = 0, ICE2 = 1, ICE3 = 2;
    bool sedi = false;

    float temp = theta / gfk_pow(100000.f / pres, ISH_RCP);
    float svpl = ish_polysvp(temp, 0);
    float qvs = 0.622f * svpl / (pres - svpl);
    float sup = qv / qvs - 1.f;
    float svpi = ish_polysvp(temp, 1);
    float qvi = 0.622f * svpi / (pres - svpi);
    if (temp > ISH_T0) qvi = qvs;
    float sui = qv / qvi - 1.f;

    bool domicro = (qc > ISH_QSMALL || qr > ISH_QSMALL || qi[ICE1] > ISH_QSMALL ||
                    qi[ICE2] > ISH_QSMALL || qi[ICE3] > ISH_QSMALL || sup >= 0.f);
    float rhoair, i_rhoair, i_temp, xxls, xxlv;
    float vtrm = 0.f, vtrn = 0.f;
    float vtrmi1[ISH_CAT], vtrni1[ISH_CAT], vtrzi1[ISH_CAT];

    if (domicro) {
        bool has_ice[ISH_CAT];
        int icearray[ISH_CAT];
        int numice = 0;
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            has_ice[cc] = false;
            if (qi[cc] > ISH_QSMALL) {
                has_ice[cc] = true;
                numice = numice + 1;
            }
        }
        {
            int cur = 0;
            for (int cc = 0; cc < ISH_CAT; ++cc)
                if (has_ice[cc]) icearray[cur++] = cc;
        }

        bool dry_growth[ISH_CAT];
        float qcrimefrac[ISH_CAT], rimetotal[ISH_CAT], nmult[ISH_CAT], qmult[ISH_CAT];
        float ani[ISH_CAT], cni[ISH_CAT], rni[ISH_CAT], qagg[ISH_CAT], nagg[ISH_CAT];
        float prd[ISH_CAT], nrd[ISH_CAT], ard[ISH_CAT], crd[ISH_CAT];
        float prdr[ISH_CAT], ardr[ISH_CAT], crdr[ISH_CAT];
        float qmlt[ISH_CAT], nmlt[ISH_CAT], amlt[ISH_CAT], cmlt[ISH_CAT];
        float deltastr[ISH_CAT], dsold[ISH_CAT], dsdepout[ISH_CAT];
        float rhobar[ISH_CAT], rhodepout[ISH_CAT], rhorimeout[ISH_CAT];
        float fv[ISH_CAT], fh[ISH_CAT], rimesum[ISH_CAT], rimesumr[ISH_CAT];
        float numrateri[ISH_CAT], rainrateri[ISH_CAT], icerateri[ISH_CAT];
        float dQRfzri[ISH_CAT], dQIfzri[ISH_CAT], dNfzri[ISH_CAT];
        float dQImltri[ISH_CAT], dNmltri[ISH_CAT];
        float gdenavg[ISH_CAT], gdenavgr[ISH_CAT], totalnuc[ISH_CAT], totalnucn[ISH_CAT];
        float nibnuc[ISH_CAT], aniold[ISH_CAT], cniold[ISH_CAT];
        float qi_qc_nrm[ISH_CAT], qi_qc_nrd[ISH_CAT];
        float qi_qr_nrm[ISH_CAT], qi_qr_nrd[ISH_CAT], qi_qr_nrn[ISH_CAT];
        float effi[ISH_CAT];
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            dry_growth[cc] = true;
            qcrimefrac[cc] = 0.f; rimetotal[cc] = 0.f; nmult[cc] = 0.f; qmult[cc] = 0.f;
            ani[cc] = 0.f; cni[cc] = 0.f; rni[cc] = 0.f; qagg[cc] = 0.f; nagg[cc] = 0.f;
            prd[cc] = 0.f; nrd[cc] = 0.f; ard[cc] = 0.f; crd[cc] = 0.f;
            prdr[cc] = 0.f; ardr[cc] = 0.f; crdr[cc] = 0.f;
            qmlt[cc] = 0.f; nmlt[cc] = 0.f; amlt[cc] = 0.f; cmlt[cc] = 0.f;
            deltastr[cc] = 1.f; dsold[cc] = 1.f; dsdepout[cc] = 1.f;
            rhobar[cc] = ISH_RHOI; rhodepout[cc] = ISH_RHOI; rhorimeout[cc] = ISH_RHOI;
            if (cc == ICE3) rhobar[cc] = 50.f;
            fv[cc] = 1.f; fh[cc] = 1.f; rimesum[cc] = 0.f; rimesumr[cc] = 0.f;
            numrateri[cc] = 0.f; rainrateri[cc] = 0.f; icerateri[cc] = 0.f;
            dQRfzri[cc] = 0.f; dQIfzri[cc] = 0.f; dNfzri[cc] = 0.f;
            dQImltri[cc] = 0.f; dNmltri[cc] = 0.f;
            gdenavg[cc] = 400.f; gdenavgr[cc] = 400.f; totalnuc[cc] = 0.f; totalnucn[cc] = 0.f;
            nibnuc[cc] = 0.f; aniold[cc] = 0.f; cniold[cc] = 0.f;
            qi_qc_nrm[cc] = 0.f; qi_qc_nrd[cc] = 0.f;
            qi_qr_nrm[cc] = 0.f; qi_qr_nrn[cc] = 0.f; qi_qr_nrd[cc] = 0.f;
            di1d[cc] = 0.f; vmi1d[cc] = 0.f; vtrmi1[cc] = 0.f; vtrni1[cc] = 0.f;
            vtrzi1[cc] = 0.f; rhopo1d[cc] = 0.f; phii1d[cc] = 0.f; qipre1d[cc] = 0.f;
            effi[cc] = 0.f; icetype1d[cc] = 0.f;
        }
        dbz1d = -35.f;
        qrpre1d = 0.f;
        effc1d = 0.f;
        effi1d = 0.f;
        float nprc = 0.f, npre = 0.f, pra = 0.f, prc = 0.f, nprc1 = 0.f, nragg = 0.f;
        float nim = 0.f, mim = 0.f, nimr = 0.f, mimr = 0.f, mbiggr = 0.f, nbiggr = 0.f;
        float mnuccd = 0.f, nnuccd = 0.f, pcc = 0.f, fmult = 0.f;
        float gdentotal = 400.f, rimedr = 0.f, rimedrr = 0.f, dbzr = 0.f, dbzsum = 0.f;
        float lamr = 0.f, n0rr = 0.f, pre = 0.f, epsr, dum, dum1;
        float alphstr = 0.f, alphv = 0.f, betam = 0.f;
        (void)nprc; (void)rimedr; (void)rimedrr;

        // Temperature and temperature-dependent variables (:847-876).
        i_temp = 1.f / temp;
        rhoair = pres / (ISH_RD * temp);
        i_rhoair = 1.f / rhoair;
        svpi = ish_polysvp(temp, 1);
        qvi = 0.622f * svpi / (pres - svpi);
        if (temp > ISH_T0) qvi = qvs;
        float qs0 = 0.622f * ish_polysvp(ISH_T0, 0) / (pres - ish_polysvp(ISH_T0, 0));
        sui = qv / qvi - 1.f;
        xxls = 3.15e6f - 2370.f * temp + 0.3337e6f;
        xxlv = 3.1484E6f - 2370.f * temp;
        float xxlf = xxls - xxlv;
        float ab = 1.f + (xxlv * qvs / (ISH_RV * (temp * temp))) * xxlv * i_cp;
        float abi = 1.f + (xxls * qvi / (ISH_RV * (temp * temp))) * xxls * i_cp;
        float mu = 1.496E-6f * gfk_pow(temp, 1.5f) / (temp + 120.f);
        float arn = ISH_AR * gfk_pow(ISH_R0 * i_rhoair, 0.5f);
        float dv = 8.794E-5f * gfk_pow(temp, 1.81f) / pres;
        float kt = 2.3823e-2f + 7.1177e-5f * (temp - ISH_T0);
        float nsch = mu * i_rhoair / dv;
        float npr = mu * i_rhoair / kt;
        (void)ab;
        float igr = ish_get_igr(temp);
        if ((temp - ISH_T0) < -20.f) igr = 0.7f;
        float nc = 200.e6f * i_rhoair;

        // ------------------------------------------------------------------
        // Ice microphysics (:882-1409).
        // ------------------------------------------------------------------
        if (numice > 0) {
            float tmpsum = 0.f, tmpsumr = 0.f;
            for (int ccvar = 0; ccvar < numice; ++ccvar) {
                int cc = icearray[ccvar];
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                ani[cc] = ish_max(gfk_pow(((ai[cc] * ai[cc]) / (ci[cc] * ni[cc])), 0.333333333333f), 2.e-6f);
                cni[cc] = ish_max(gfk_pow(((ci[cc] * ci[cc]) / (ai[cc] * ni[cc])), 0.333333333333f), 2.e-6f);
                if (ai[cc] < 1.e-12f || ci[cc] < 1.e-12f) {
                    ai[cc] = ish_min(ai[cc], ci[cc]);
                    ci[cc] = ai[cc];
                    ani[cc] = gfk_pow(ai[cc] / ni[cc], 0.3333333333f);
                    cni[cc] = gfk_pow(ci[cc] / ni[cc], 0.3333333333f);
                }
                if (ani[cc] < 2.e-6f || cni[cc] < 2.e-6f) {
                    ani[cc] = 2.e-6f;
                    cni[cc] = 2.e-6f;
                }
                ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
                ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                deltastr[cc] = (gfk_log(cni[cc]) - ISH_LOG_AO) / (gfk_log(ani[cc]) - ISH_LOG_AO);
                ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                              cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                              alphstr, alphv, betam);
                dsold[cc] = deltastr[cc];

                // Ice-cloud riming rates from the lookup table (:925-978).
                if (qc > 1.e-7f) {
                    float rrni = 13.498f * gfk_log10(0.5e6f * rni[cc]);
                    float rqci = 8.776f * gfk_log10(1.0e7f * (gfk_exp(qc) - 1.f));
                    float rdsi = 50.f * (deltastr[cc] - 0.5f);
                    float rrho = 7.888f * gfk_log10(0.02f * rhobar[cc]);
                    int irni = (int)rrni, iqci = (int)rqci, idsi = (int)rdsi, irho = (int)rrho;
                    rrni = ish_max(rrni, 1.f); rqci = ish_max(rqci, 1.f);
                    rdsi = ish_max(rdsi, 1.f); rrho = ish_max(rrho, 1.f);
                    irni = irni > 1 ? irni : 1; iqci = iqci > 1 ? iqci : 1;
                    idsi = idsi > 1 ? idsi : 1; irho = irho > 1 ? irho : 1;
                    rrni = ish_min(rrni, 51.f - 1.f); rqci = ish_min(rqci, 51.f - 1.f);
                    rdsi = ish_min(rdsi, 51.f - 1.f); rrho = ish_min(rrho, 11.f - 1.f);
                    irni = irni < 50 ? irni : 50; iqci = iqci < 50 ? iqci : 50;
                    idsi = idsi < 50 ? idsi : 50; irho = irho < 10 ? irho : 10;
                    float proc1 = ish_access_lookup_table(T.itab, 2, irni, iqci, idsi, irho, 1,
                                                          rdsi, rrho, rqci, rrni);
                    float proc2 = ish_access_lookup_table(T.itab, 2, irni, iqci, idsi, irho, 2,
                                                          rdsi, rrho, rqci, rrni);
                    rimesum[cc] = ish_max((proc1 * ni[cc] * nc * (rhoair * rhoair)), 0.f);
                    qi_qc_nrm[cc] = proc1;
                    qi_qc_nrd[cc] = proc2;
                    if ((rimesum[cc] * i_rhoair) < ISH_QSMALL) {
                        rimesum[cc] = 0.f;
                        qi_qc_nrm[cc] = 0.f;
                        qi_qc_nrd[cc] = 0.f;
                    }
                }

                // Ice-rain riming rates (:983-1086).
                if (qr > ISH_QSMALL) {
                    nr = ish_max(nr, ISH_QNSMALL);
                    ish_rain_dsd(qr, nr, lamr, n0rr);
                    float rrr = 0.5f * (1.f / lamr);
                    float rrni = 13.498f * gfk_log10(0.5e6f * rni[cc]);
                    float rrri = 23.273f * gfk_log10(1.e5f * rrr);
                    float rdsi = 50.f * (deltastr[cc] - 0.5f);
                    float rrho = 7.888f * gfk_log10(0.02f * rhobar[cc]);
                    int irni = (int)rrni, irri = (int)rrri, idsi = (int)rdsi, irho = (int)rrho;
                    rrni = ish_max(rrni, 1.f); rrri = ish_max(rrri, 1.f);
                    rdsi = ish_max(rdsi, 1.f); rrho = ish_max(rrho, 1.f);
                    irni = irni > 1 ? irni : 1; irri = irri > 1 ? irri : 1;
                    idsi = idsi > 1 ? idsi : 1; irho = irho > 1 ? irho : 1;
                    rrni = ish_min(rrni, 51.f - 1.f); rrri = ish_min(rrri, 51.f - 1.f);
                    rdsi = ish_min(rdsi, 51.f - 1.f); rrho = ish_min(rrho, 11.f - 1.f);
                    irni = irni < 50 ? irni : 50; irri = irri < 50 ? irri : 50;
                    idsi = idsi < 50 ? idsi : 50; irho = irho < 10 ? irho : 10;
                    float procr[6];
                    for (int iti = 1; iti <= 6; ++iti)
                        procr[iti - 1] = ish_access_lookup_table(T.itabr, 6, irni, irri, idsi,
                                                                 irho, iti, rdsi, rrho, rrri, rrni);
                    numrateri[cc] = ish_max((procr[3] * ni[cc] * nr * rhoair), 0.f);
                    rainrateri[cc] = ish_max((procr[4] * ni[cc] * nr * rhoair), 0.f);
                    icerateri[cc] = ish_max((procr[5] * ni[cc] * nr * rhoair), 0.f);
                    if (rainrateri[cc] < ISH_QSMALL || icerateri[cc] < ISH_QSMALL) {
                        rainrateri[cc] = 0.f;
                        icerateri[cc] = 0.f;
                        numrateri[cc] = 0.f;
                    }
                    if (temp <= ISH_T0) {
                        if (qr > 0.1e-3f && qi[cc] > 0.1e-3f) {
                            dQRfzri[cc] = rainrateri[cc];
                            dQIfzri[cc] = icerateri[cc];
                            dNfzri[cc] = numrateri[cc];
                        }
                    } else {
                        dQImltri[cc] = icerateri[cc];
                        dNmltri[cc] = numrateri[cc];
                    }
                    rimesumr[cc] = ish_max((procr[0] * ni[cc] * nr * (rhoair * rhoair)), 0.f);
                    qi_qr_nrm[cc] = procr[0];
                    qi_qr_nrd[cc] = procr[1];
                    qi_qr_nrn[cc] = procr[2];
                    if ((rimesumr[cc] * i_rhoair) < ISH_QSMALL) {
                        rimesumr[cc] = 0.f;
                        qi_qr_nrm[cc] = 0.f;
                        qi_qr_nrd[cc] = 0.f;
                        qi_qr_nrn[cc] = 0.f;
                    }
                }

                tmpsum = tmpsum + rimesum[cc];
                tmpsumr = tmpsumr + rimesumr[cc];
                if (ccvar == numice - 1) {
                    float sink = tmpsum * i_rhoair;
                    float source = qc * i_dt;
                    if (sink > source && qc > ISH_QSMALL) {
                        float ratio = source / sink;
                        for (int c2 = 0; c2 < ISH_CAT; ++c2) {
                            rimesum[c2] = rimesum[c2] * ratio;
                            qi_qc_nrm[c2] = qi_qc_nrm[c2] * ratio;
                            qi_qc_nrd[c2] = qi_qc_nrd[c2] * ratio;
                        }
                    }
                    // D1: WRF :1105 bounds this by tmpsum (the cloud sum).
#ifdef ISHMAEL_FAST_NUMERICS
                    sink = tmpsumr * i_rhoair;
#else
                    // Strict replay retains the pristine WRF limiter.
                    sink = tmpsum * i_rhoair;
#endif
                    source = qr * i_dt;
                    if (sink > source && qr > ISH_QSMALL) {
                        float ratio = source / sink;
                        for (int c2 = 0; c2 < ISH_CAT; ++c2) {
                            rimesumr[c2] = rimesumr[c2] * ratio;
                            qi_qr_nrm[c2] = qi_qr_nrm[c2] * ratio;
                            qi_qr_nrd[c2] = qi_qr_nrd[c2] * ratio;
                            qi_qr_nrn[c2] = qi_qr_nrn[c2] * ratio;
                        }
                    }
                }

                // Initial volume, iwc, #/m^3 for vapor growth (:1119-1138).
                int gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                float vi = fourthirdspi * (rni[cc] * rni[cc] * rni[cc]) * gtab[gi - 1] * i_gammnu;
                float iwci = ni[cc] * rhobar[cc] * vi * rhoair;
                float nim3dum = ni[cc] * rhoair;
                rimetotal[cc] = rimesum[cc] + rimesumr[cc];
                alphstr = gfk_pow(ao, 1.f - deltastr[cc]);
                float capgam = ish_capacitance_gamma(ani[cc], deltastr[cc], alphstr, i_gammnu);
                float anf, cnf, rnf, iwcf;
                ish_vaporgrow(gtab, dt, ani[cc], cni[cc], rni[cc], igr, nim3dum, temp,
                              rimetotal[cc], alphstr, sui, sup, qvs, qvi, mu, iwci, rhoair,
                              dv, kt, nsch, npr, gammnu, i_gammnu, fourthirdspi, svpi,
                              xxls, xxlf, capgam, vtrni1[cc], vtrmi1[cc], anf, cnf, rnf,
                              iwcf, fv[cc], fh[cc], rhobar[cc], deltastr[cc],
                              rhodepout[cc], dsdepout[cc]);
                (void)rnf;
                prd[cc] = (iwcf - iwci) * i_rhoair * i_dt;
                prd[cc] = ish_max(prd[cc], -qi[cc] * i_dt);
                ard[cc] = (2.f * (anf - ani[cc]) * cni[cc] + (cnf - cni[cc]) * ani[cc]) * ani[cc] * ni[cc] * i_dt;
                crd[cc] = (2.f * (cnf - cni[cc]) * ani[cc] + (anf - ani[cc]) * cni[cc]) * cni[cc] * ni[cc] * i_dt;
                if (prd[cc] < 0.f && qi[cc] > ISH_QSMALL) {
                    nrd[cc] = prd[cc] * ni[cc] / qi[cc];
                }
                if (fabsf(prd[cc] * dt) < (ISH_QSMALL * 0.01f)) {
                    prd[cc] = 0.f; nrd[cc] = 0.f; ard[cc] = 0.f; crd[cc] = 0.f;
                }

                // Wet-growth check (:1165-1174).
                rimetotal[cc] = rimesum[cc] + rimesumr[cc];
                dry_growth[cc] = ish_wet_growth_dry(temp, rhoair, xxlv, xxlf, qv, dv, kt, qs0,
                                                    fv[cc], fh[cc], rimetotal[cc], rni[cc],
                                                    ni[cc], dry_growth[cc]);
                if (temp > ISH_T0) dry_growth[cc] = false;

                float vir = vi;
                float rnfr = rni[cc];
                (void)vir;
                if (qc > ISH_QSMALL && qi_qc_nrm[cc] > 0.f) {
                    float rimec1 = ish_rimec1(temp);
                    gdenavg[cc] = ish_rime_density(temp, rimec1, qi_qc_nrd[cc], qi_qc_nrm[cc],
                                                   dry_growth[cc]);
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    float rdr = ish_max(((qi_qc_nrm[cc] / gdenavg[cc]) / ((gtab[gi - 1] * i_gammnu) *
                                         4.f * ISH_PI * (rni[cc] * rni[cc]))), 0.f);
                    rnfr = rnfr + ish_max((rdr * nc * rhoair), 0.f) * dt;
                }
                if (qr > ISH_QSMALL && qi_qr_nrm[cc] > 0.f) {
                    float rimec1 = ish_rimec1(temp);
                    gdenavgr[cc] = ish_rime_density(temp, rimec1, qi_qr_nrd[cc], qi_qr_nrm[cc],
                                                    dry_growth[cc]);
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    float rdrr = ish_max(((qi_qr_nrm[cc] / gdenavgr[cc]) / ((gtab[gi - 1] * i_gammnu) *
                                          4.f * ISH_PI * (rni[cc] * rni[cc]))), 0.f);
                    rnfr = rnfr + ish_max((rdrr * nr * rhoair), 0.f) * dt;
                }

                // Volume after riming (:1272-1294).
                gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                float vfr = fourthirdspi * (rnfr * rnfr * rnfr) * gtab[gi - 1] * i_gammnu;
                vfr = ish_max(vfr, vi);
                rnfr = ish_max(rnfr, rni[cc]);
                float iwcfr;
                if (rimetotal[cc] > 0.f && vfr > vi) {
                    qcrimefrac[cc] = rimesum[cc] / rimetotal[cc];
                    qcrimefrac[cc] = ish_max(qcrimefrac[cc], 0.f);
                    qcrimefrac[cc] = ish_min(qcrimefrac[cc], 1.f);
                    gdentotal = qcrimefrac[cc] * gdenavg[cc] + (1.f - qcrimefrac[cc]) * gdenavgr[cc];
                    rhorimeout[cc] = (rhobar[cc] * (vi / vfr)) + (gdentotal * (1.f - (vi / vfr)));
                    rhorimeout[cc] = ish_min(rhorimeout[cc], ISH_RHOI);
                    iwcfr = rhorimeout[cc] * vfr * nim3dum;
                    rhorimeout[cc] = gdentotal;
                    rhorimeout[cc] = ish_min(rhorimeout[cc], ISH_RHOI);
                } else {
                    iwcfr = iwci;
                    rnfr = rni[cc];
                }

                if (dry_growth[cc]) {
                    float cnfr = cni[cc], anfr = ani[cc], phibr = 1.f, phifr = 0.f;
                    if (rimetotal[cc] > 0.f && vfr > vi) {
                        gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                        phibr = cni[cc] / ani[cc] * gtab[gi - 1] * i_gammnu;
                        if (phibr == 1.f) {
                            phifr = 1.f;
                            anfr = rnfr;
                            cnfr = rnfr;
                        } else if (phibr > 1.25f) {
                            float rr = rnfr / rni[cc];
                            phifr = phibr * gfk_pow((rr * rr * rr), -0.5f);
                            anfr = (ani[cc] / gfk_pow(rni[cc], 1.5f)) * gfk_pow(rnfr, 1.5f);
                        } else if (phibr < 0.8f) {
                            float rr = rnfr / rni[cc];
                            phifr = phibr * (rr * rr * rr);
                            gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                            cnfr = phifr * anfr * gammnu / gtab[gi - 1];
                        } else {
                            phifr = phibr;
                            gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                            int gi2 = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                            anfr = gfk_pow(((vfr * gtab[gi - 1]) / (fourthirdspi * phifr * gtab[gi2 - 1])),
                                           0.3333333333f);
                            cnfr = phifr * anfr * gammnu / gtab[gi - 1];
                        }
                        if (phibr <= 1.f && phifr > 1.f) {
                            phifr = 0.99f;
                            gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                            int gi2 = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                            anfr = gfk_pow(((vfr * gtab[gi - 1]) / (fourthirdspi * phifr * gtab[gi2 - 1])),
                                           0.3333333333f);
                            cnfr = phifr * anfr * gammnu / gtab[gi - 1];
                        }
                        if (phibr >= 1.f && phifr < 1.f) {
                            phifr = 1.01f;
                            gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                            int gi2 = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                            anfr = gfk_pow(((vfr * gtab[gi - 1]) / (fourthirdspi * phifr * gtab[gi2 - 1])),
                                           0.3333333333f);
                            cnfr = phifr * anfr * gammnu / gtab[gi - 1];
                        }
                    }
                    cnfr = ish_max(cnfr, cni[cc]);
                    anfr = ish_max(anfr, ani[cc]);
                    prdr[cc] = (iwcfr - iwci) * i_rhoair * i_dt;
                    ardr[cc] = (2.f * (anfr - ani[cc]) * cni[cc] + (cnfr - cni[cc]) * ani[cc]) * ani[cc] * ni[cc] * i_dt;
                    crdr[cc] = (2.f * (cnfr - cni[cc]) * ani[cc] + (anfr - ani[cc]) * cni[cc]) * cni[cc] * ni[cc] * i_dt;
                    prdr[cc] = ish_max(prdr[cc], 0.f);
                    ardr[cc] = ish_max(ardr[cc], 0.f);
                    crdr[cc] = ish_max(crdr[cc], 0.f);
                    if (prdr[cc] == 0.0f) {
                        ardr[cc] = 0.f;
                        crdr[cc] = 0.f;
                    }
                } else {
                    prdr[cc] = (iwcfr - iwci) * i_rhoair * i_dt;
                    ardr[cc] = 0.f;
                    crdr[cc] = 0.f;
                }

                if (temp > ISH_T0) {
                    qmlt[cc] = 2.f * ISH_PI * (kt * fh[cc] * (ISH_T0 - temp) + rhoair * xxlv * dv * fv[cc] * (qs0 - qv)) /
                               xxlf * (ni[cc] * ISH_NU * ish_max(ani[cc], cni[cc])) -
                               (ISH_CPW / xxlf * (temp - ISH_T0) * (rimetotal[cc] / rhoair + dQImltri[cc]));
                    qmlt[cc] = ish_min(qmlt[cc], 0.f);
                    qmlt[cc] = ish_max(qmlt[cc], (-qi[cc] * i_dt));
                    if (qmlt[cc] < 0.f) {
                        if (ai[cc] < 1.e-12f || ci[cc] < 1.e-12f) qmlt[cc] = -qi[cc] * i_dt;
                    }
                    nmlt[cc] = ish_max((-ni[cc] * i_dt), (ni[cc] * qmlt[cc] / qi[cc] - dNmltri[cc]));
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    float tmpmelt = fourthirdspi * alphstr * gtab[gi - 1] * i_gammnu;
                    amlt[cc] = (1.f / tmpmelt) * (1.f / rhobar[cc]) * qmlt[cc] + ai[cc] * nmlt[cc] / ni[cc];
                    cmlt[cc] = amlt[cc] * cni[cc] / ani[cc] * (1.f + (2.f * deltastr[cc])) / (2.f + deltastr[cc]);
                }
            }
        }

        // ------------------------------------------------------------------
        // Nucleation, freezing, splinters, aggregation (:1417-1546).
        // ------------------------------------------------------------------
        if (qc > ISH_QSMALL && temp < (ISH_T0 - 35.f)) {
            mim = qc * i_dt;
            nim = nc * i_dt;
        }
        if (qr > ISH_QSMALL && temp < (ISH_T0 - 35.f)) {
            mimr = qr * i_dt;
            nimr = nr * i_dt;
        }
        if (qr > ISH_QSMALL && temp < (ISH_T0 - 4.f)) {
            nr = ish_max(nr, ISH_QNSMALL);
            ish_rain_dsd(qr, nr, lamr, n0rr);
            float l3 = lamr * lamr * lamr;
            mbiggr = 20.f * ISH_PI * ISH_PI * ISH_RHOW * 100.f * nr * (gfk_exp(0.66f * (ISH_T0 - temp)) - 1.f) / l3 / l3;
            nbiggr = ISH_PI * 100.f * nr * (gfk_exp(0.66f * (ISH_T0 - temp)) - 1.f) / l3;
            mbiggr = ish_min(mbiggr, qr * i_dt);
            nbiggr = ish_min(nbiggr, nr * i_dt);
        }
        if (temp < ISH_T0 && sup >= 0.f) {
            const float a_demott = 0.0000594f, b_demott = 3.33f, c_demott = 0.0264f, d_demott = 0.0033f;
            float inrate = a_demott * gfk_pow(273.16f - temp, b_demott) *
                           gfk_pow(0.03f, (c_demott * (273.16f - temp)) + d_demott);
            inrate = inrate * 1000.0f;
            inrate = inrate * i_rhoair;
            if ((((ni[ICE1] + ni[ICE2] + inrate) * rhoair) / 1000.f) <= 10000.f) {
                mnuccd = (inrate) * (fourthirdspi * ISH_RHOI * ISH_CUBE_2EM6) * i_dt;
                nnuccd = inrate * i_dt;
            } else {
                float curnum = (ni[ICE1] + ni[ICE2]) * rhoair / 1000.f;
                float ratel = ish_max((10000.f - curnum), 0.f);
                float ratekg = ratel * 1000.f * i_rhoair;
                mnuccd = (ratekg) * (fourthirdspi * ISH_RHOI * ISH_CUBE_2EM6) * i_dt;
                nnuccd = ratekg * i_dt;
            }
        }
        // Rime splinters (:1476-1497).
        if (temp < 270.16f && temp > 265.16f) {
            if (temp > 270.16f) {
                fmult = 0.f;
            } else if (temp <= 270.16f && temp > 268.16f) {
                fmult = (270.16f - temp) / 2.f;
            } else if (temp >= 265.16f && temp <= 268.16f) {
                fmult = (temp - 265.16f) / 3.f;
            } else if (temp < 265.16f) {
                fmult = 0.f;
            }
        }
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if (prdr[cc] > 0.f) {
                nmult[cc] = 35.E4f * prdr[cc] * fmult * 1000.f;
                qmult[cc] = nmult[cc] * fourthirdspi * ISH_RHOI * ISH_CUBE_5EM6;
                qmult[cc] = ish_min(qmult[cc], prdr[cc]);
                prdr[cc] = prdr[cc] - qmult[cc];
            }
        }
        float dn3 = 0.f;
        if (temp <= ISH_T0) {
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
            }
            float dn1 = 2.f * gfk_pow(((ai[ICE1] * ai[ICE1]) / (ci[ICE1] * ni[ICE1])), 0.333333333333f);
            float dn2 = 2.f * gfk_pow(((ci[ICE2] * ci[ICE2]) / (ai[ICE2] * ni[ICE2])), 0.333333333333f);
            dn3 = 2.f * gfk_pow(((ai[ICE3] * ai[ICE3]) / (ci[ICE3] * ni[ICE3])), 0.333333333333f);
            dn1 = ish_min(dn1, 1.e-2f); dn2 = ish_min(dn2, 1.e-2f); dn3 = ish_min(dn3, 1.e-2f);
            dn1 = ish_max(dn1, 1.e-6f); dn2 = ish_max(dn2, 1.e-6f); dn3 = ish_max(dn3, 1.e-6f);
            int gi = ish_gi(ISH_NU - 1.f + deltastr[0]);
            float phiagg1 = ci[0] / ai[0] * gtab[gi - 1] * i_gammnu;
            gi = ish_gi(ISH_NU - 1.f + deltastr[1]);
            float phiagg2 = ci[1] / ai[1] * gtab[gi - 1] * i_gammnu;
            phiagg1 = ish_min(phiagg1, 100.f);
            phiagg1 = ish_max(phiagg1, 0.01f);
            phiagg2 = ish_min(phiagg2, 100.f);
            phiagg2 = ish_max(phiagg2, 0.01f);
            if (qi[ICE1] > 1.e-8f || qi[ICE2] > 1.e-8f || qi[ICE3] > 1.e-8f) {
                ish_aggregation(T, dt, rhoair, temp, qi[ICE1], ni[ICE1], dn1, qi[ICE2], ni[ICE2],
                                dn2, qi[ICE3], ni[ICE3], dn3, rhobar[ICE1], rhobar[ICE2],
                                phiagg1, phiagg2, qagg[ICE1], qagg[ICE2], qagg[ICE3],
                                nagg[ICE1], nagg[ICE2], nagg[ICE3]);
            }
        }

        // ------------------------------------------------------------------
        // Warm microphysics (:1553-1630).
        // ------------------------------------------------------------------
        if (qr > ISH_QSMALL) {
            nr = ish_max(nr, ISH_QNSMALL);
            ish_rain_dsd(qr, nr, lamr, n0rr);
        }
        if (qc >= 1.E-6f) {
            prc = 1350.f * gfk_pow(qc, 2.47f) * gfk_pow((nc / 1.e6f * rhoair), -1.79f);
            nprc1 = prc / (4.f / 3.f * ISH_PI * ISH_RHOW * ISH_CUBE_25EM6);
            float nprc_ = prc / (qc / nc);
            nprc_ = ish_min(nprc_, nc * i_dt);
            nprc1 = ish_min(nprc1, nprc_);
        }
        if (qr >= 1.e-8f && qc >= 1.e-8f) {
            dum = (qc * qr);
            pra = 67.f * gfk_pow(dum, 1.15f);
        }
        if (qr >= 1.e-8f) {
            dum1 = 300.e-6f;
            if (1.f / lamr < dum1) {
                dum = 1.f;
            } else if (1.f / lamr >= dum1) {
                dum = 2.f - gfk_exp(2300.f * (1.f / lamr - dum1));
            }
            nragg = -5.78f * dum * nr * qr * rhoair;
        }
        if (qr > ISH_QSMALL) {
            epsr = 2.f * ISH_PI * n0rr * rhoair * dv *
                   (ISH_F1R / (lamr * lamr) +
                    ISH_F2R * gfk_pow(arn * rhoair / mu, 0.5f) *
                    gfk_pow(nsch, 0.333333333f) * (ish_gamma(5.f / 2.f + ISH_BR / 2.f)) /
                    (gfk_pow(lamr, 5.f / 2.f + ISH_BR / 2.f)));
        } else {
            epsr = 0.f;
        }
        if (qv < qvs) {
            pre = epsr * (qv - qvs) / ab;
            pre = ish_min(pre, 0.f);
            float dumt = temp + xxlv * i_cp * pre * dt;
            float dumqv = qv - pre * dt;
            float dumqvs = 0.622f * ish_polysvp(dumt, 0) / (pres - ish_polysvp(dumt, 0));
            if (pre < 0.f && dumqv / dumqvs > 1.f) {
                pre = (qv - qvs) * i_dt / ab;
            }
        } else {
            pre = 0.f;
        }
        if (pre < 0.f) {
            dum = pre * dt / qr;
            dum = ish_max(dum, -1.f);
            npre = dum * nr / dt;
        }

        // ------------------------------------------------------------------
        // Do not over-deplete (:1647-1848).
        // ------------------------------------------------------------------
        {
            float qcrimesum = 0.f;
            for (int cc = 0; cc < ISH_CAT; ++cc) qcrimesum = qcrimesum + prdr[cc] * qcrimefrac[cc];
            float sink = (prc + pra + qcrimesum + mim) * dt;
            float source = qc;
            if (sink > source && qc > ISH_QSMALL) {
                float ratio = source / sink;
                prc = prc * ratio;
                pra = pra * ratio;
                mim = mim * ratio;
                for (int cc = 0; cc < ISH_CAT; ++cc) {
                    prdr[cc] = prdr[cc] * ratio;
                    ardr[cc] = ardr[cc] * ratio;
                    crdr[cc] = crdr[cc] * ratio;
                }
            }
            float qcrimesumr = 0.f;
            for (int cc = 0; cc < ISH_CAT; ++cc) qcrimesumr = qcrimesumr + prdr[cc] * (1.f - qcrimefrac[cc]);
            sink = (-pre + qcrimesumr + mimr + mbiggr) * dt;
            source = qr + (pra + prc) * dt;
            for (int cc = 0; cc < ISH_CAT; ++cc) source = source + (-qmlt[cc] * dt);
            for (int cc = 0; cc < ISH_CAT; ++cc) sink = sink + (dQRfzri[cc]) * dt;
            if (sink > source && qr > ISH_QSMALL) {
                float ratio = source / sink;
                for (int cc = 0; cc < ISH_CAT; ++cc) {
                    prdr[cc] = prdr[cc] * ratio;
                    ardr[cc] = ardr[cc] * ratio;
                    crdr[cc] = crdr[cc] * ratio;
                    qi_qr_nrn[cc] = qi_qr_nrn[cc] * ratio;
                    dQRfzri[cc] = dQRfzri[cc] * ratio;
                    dQIfzri[cc] = dQIfzri[cc] * ratio;
                    dNfzri[cc] = dNfzri[cc] * ratio;
                }
                mimr = mimr * ratio;
                nimr = nimr * ratio;
                mbiggr = mbiggr * ratio;
                nbiggr = nbiggr * ratio;
                pre = pre * ratio;
            }

            // Ice: the nucleating species takes all initiation sources.
            int nuc = (igr <= 1.f) ? ICE1 : ICE2;
            int oth = (igr <= 1.f) ? ICE2 : ICE1;
            // nucleating species
            sink = (-qmlt[nuc] - qagg[nuc]) * dt;
            source = qi[nuc] + (prdr[nuc]) * dt;
            if (prd[nuc] > 0.f) source = source + prd[nuc] * dt;
            else sink = sink + (-prd[nuc]) * dt;
            source = source + (mnuccd + mim + mimr + mbiggr + dQIfzri[oth] + dQIfzri[ICE3]) * dt;
            for (int cc = 0; cc < ISH_CAT; ++cc) source = source + (qmult[cc] + dQRfzri[cc]) * dt;
            if (sink > source && qi[nuc] > ISH_QSMALL) {
                float ratio = source / sink;
                qmlt[nuc] = qmlt[nuc] * ratio;
                qagg[nuc] = qagg[nuc] * ratio;
                nagg[nuc] = nagg[nuc] * ratio;
                if (prd[nuc] < 0.f) prd[nuc] = prd[nuc] * ratio;
            }
            // the other planar/columnar species
            sink = (dQIfzri[oth] - qmlt[oth] - qagg[oth]) * dt;
            source = qi[oth] + (prdr[oth]) * dt;
            if (prd[oth] > 0.f) source = source + prd[oth] * dt;
            else sink = sink + (-prd[oth]) * dt;
            if (sink > source && qi[oth] > ISH_QSMALL) {
                float ratio = source / sink;
                dQIfzri[oth] = dQIfzri[oth] * ratio;
                dQRfzri[oth] = dQRfzri[oth] * ratio;
                dNfzri[oth] = dNfzri[oth] * ratio;
                qmlt[oth] = qmlt[oth] * ratio;
                qagg[oth] = qagg[oth] * ratio;
                nagg[oth] = nagg[oth] * ratio;
                if (prd[oth] < 0.f) prd[oth] = prd[oth] * ratio;
            }
            // aggregates
            sink = (dQIfzri[ICE3] - qmlt[ICE3]) * dt;
            source = qi[ICE3] + (prdr[ICE3] + qagg[ICE3]) * dt;
            if (prd[ICE3] > 0.f) source = source + prd[ICE3] * dt;
            else sink = sink + (-prd[ICE3]) * dt;
            if (sink > source && qi[ICE3] > ISH_QSMALL) {
                float ratio = source / sink;
                dQIfzri[ICE3] = dQIfzri[ICE3] * ratio;
                dQRfzri[ICE3] = dQRfzri[ICE3] * ratio;
                dNfzri[ICE3] = dNfzri[ICE3] * ratio;
                qmlt[ICE3] = qmlt[ICE3] * ratio;
                if (prd[ICE3] < 0.f) prd[ICE3] = prd[ICE3] * ratio;
            }
        }

        // Do not let vapor growth create subsaturated conditions (:1835-1848).
        float prdsum = 0.f;
        for (int cc = 0; cc < ISH_CAT; ++cc) prdsum = prdsum + prd[cc];
        bool vgflag = false;
        {
            float sink = prdsum;
            float source = 0.99f * sui * qvi / abi * i_dt;
            if (sink > source && sui > 0.f) {
                vgflag = true;
                float ratio = source / sink;
                for (int cc = 0; cc < ISH_CAT; ++cc) prd[cc] = prd[cc] * ratio;
            }
        }

        // Vapor growth update (:1851-1884).
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if ((qi[cc] + (prd[cc]) * dt) > ISH_QSMALL && qi[cc] > ISH_QSMALL) {
                rhobar[cc] = rhobar[cc] * (qi[cc] / (qi[cc] + (prd[cc]) * dt)) +
                             rhodepout[cc] * (1.f - (qi[cc] / (qi[cc] + (prd[cc]) * dt)));
            } else {
                rhobar[cc] = ISH_RHOI;
            }
            qi[cc] = qi[cc] + (prd[cc]) * dt;
            ni[cc] = ni[cc] + (nrd[cc]) * dt;
            ai[cc] = ai[cc] + (ard[cc]) * dt;
            ci[cc] = ci[cc] + (crd[cc]) * dt;
            ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
            ai[cc] = ish_max(ai[cc], ISH_QASMALL);
            ci[cc] = ish_max(ci[cc], ISH_QASMALL);
            if (vgflag && qi[cc] > ISH_QSMALL) {
                alphstr = gfk_pow(ao, 1.f - deltastr[cc]);
                alphv = fourthirdspi * alphstr;
                betam = 2.f + deltastr[cc];
                int gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                ani[cc] = gfk_pow(((qi[cc] * gammnu) / (rhobar[cc] * ni[cc] * alphv * gtab[gi - 1])),
                                  1.f / betam);
                cni[cc] = gfk_pow(ao, 1.f - deltastr[cc]) * gfk_pow(ani[cc], deltastr[cc]);
                ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
            }
        }
        // Riming update (:1887-1906).
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if ((qi[cc] + (prdr[cc]) * dt) > ISH_QSMALL) {
                if (dry_growth[cc]) {
                    rhobar[cc] = rhobar[cc] * (qi[cc] / (qi[cc] + (prdr[cc]) * dt)) +
                                 rhorimeout[cc] * (1.f - (qi[cc] / (qi[cc] + (prdr[cc]) * dt)));
                } else {
                    rhobar[cc] = rhobar[cc] * (qi[cc] / (qi[cc] + (prdr[cc]) * dt)) +
                                 ISH_RHOW * (1.f - (qi[cc] / (qi[cc] + (prdr[cc]) * dt)));
                    rhobar[cc] = ish_max(rhobar[cc], ISH_RHOI);
                }
            } else {
                rhobar[cc] = ISH_RHOI;
            }
            qi[cc] = qi[cc] + (prdr[cc]) * dt;
            ai[cc] = ai[cc] + (ardr[cc]) * dt;
            ci[cc] = ci[cc] + (crdr[cc]) * dt;
        }
        // Bounds after vapor growth and riming (:1909-1928).
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if (qi[cc] > ISH_QSMALL) {
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                ish_axes(ai[cc], ci[cc], ni[cc], ani[cc], cni[cc], deltastr[cc]);
                ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                              cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                              alphstr, alphv, betam);
                dsold[cc] = deltastr[cc];
            }
        }

        // Update to ice after melting (:1933-1995).
        if (temp > ISH_T0) {
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                if (qi[cc] > ISH_QSMALL) {
                    rhobar[cc] = rhobar[cc] * ((qi[cc] + (qmlt[cc]) * dt) / qi[cc]) +
                                 ISH_RHOW * ((-qmlt[cc] * dt) / qi[cc]);
                    rhobar[cc] = ish_min(rhobar[cc], ISH_RHOI);
                } else {
                    rhobar[cc] = ISH_RHOI;
                }
                qi[cc] = qi[cc] + qmlt[cc] * dt;
                ni[cc] = ni[cc] + nmlt[cc] * dt;
                ai[cc] = ai[cc] + amlt[cc] * dt;
                ci[cc] = ci[cc] + cmlt[cc] * dt;
                if (qi[cc] <= ISH_QSMALL) {
                    qr = qr + qi[cc];
                    qi[cc] = 0.f;
                }
                if (qi[cc] > ISH_QSMALL) {
                    ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                    ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                    ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                    ish_axes(ai[cc], ci[cc], ni[cc], ani[cc], cni[cc], deltastr[cc]);
                    deltastr[cc] = ish_toward_sphere(deltastr[cc], dsold[cc]);
                    cni[cc] = gfk_pow(ao, 1.f - deltastr[cc]) * gfk_pow(ani[cc], deltastr[cc]);
                    ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                    ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
                    ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                                  cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                                  alphstr, alphv, betam);
                    dsold[cc] = deltastr[cc];
                }
            }
        }

        // Vapor, cloud, rain updates (:2000-2017).
        qv = qv + (-pre - mnuccd) * dt;
        for (int cc = 0; cc < ISH_CAT; ++cc) qv = qv + (-prd[cc]) * dt;
        qc = qc + (-prc - pra - mim) * dt;
        for (int cc = 0; cc < ISH_CAT; ++cc) qc = qc - (prdr[cc] * qcrimefrac[cc] * dt);
        qr = qr + (pre + prc + pra - mimr - mbiggr) * dt;
        nr = nr + (nprc1 + nragg + npre - nimr - nbiggr) * dt;
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            qr = qr + ((-prdr[cc] * (1.f - qcrimefrac[cc])) - qmlt[cc] - dQRfzri[cc]) * dt;
            nr = ish_max((nr + ((-qi_qr_nrn[cc] * ni[cc] * nr * rhoair) - nmlt[cc] - dNfzri[cc]) * dt), 0.f);
        }

        // Nucleation transfer (:2022-2068).
        for (int cc = 0; cc < ISH_CAT; ++cc) nibnuc[cc] = ni[cc];
        {
            int nuc = (igr <= 1.f) ? ICE1 : ICE2;
            int oth = (igr <= 1.f) ? ICE2 : ICE1;
            totalnuc[nuc] = (mnuccd + mim + mimr + mbiggr + dQIfzri[oth] + dQIfzri[ICE3]) * dt;
            totalnucn[nuc] = (nnuccd + nim + nimr + nbiggr + dNfzri[oth] + dNfzri[ICE3]) * dt;
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                totalnuc[nuc] = totalnuc[nuc] + (qmult[cc] + dQRfzri[cc]) * dt;
                totalnucn[nuc] = totalnucn[nuc] + (nmult[cc]) * dt;
            }
            qi[nuc] = qi[nuc] + totalnuc[nuc];
            ni[nuc] = ni[nuc] + totalnucn[nuc];
            qi[oth] = qi[oth] - (dQIfzri[oth]) * dt;
            ni[oth] = ni[oth] - (dNfzri[oth]) * dt;
            qi[ICE3] = qi[ICE3] - (dQIfzri[ICE3]) * dt;
            ni[ICE3] = ni[ICE3] - (dNfzri[ICE3]) * dt;
        }
        for (int cc = 0; cc < ISH_CAT; ++cc) ni[cc] = ish_min(ni[cc], (1000.f * 1000.f * i_rhoair));

        // Final check on rain and its fall speed (:2076-2105).
        if (qr > ISH_QSMALL) {
            nr = ish_max(nr, ISH_QNSMALL);
            ish_rain_dsd(qr, nr, lamr, n0rr);
            vtrn = arn * (ish_gamma(1.f + ISH_BR)) / gfk_pow(lamr, ISH_BR);
            vtrm = arn * (ish_gamma(4.f + ISH_BR)) / 6.f / gfk_pow(lamr, ISH_BR);
            vtrn = ish_min(vtrn, 9.1f);
            vtrm = ish_min(vtrm, 9.1f);
            float l3 = lamr * lamr * lamr;
            dbzr = rhoair * nr / l3 / l3 * 720.f;
            dbzr = 1.e18f * dbzr;
        } else {
            qv = qv + qr;
            theta = theta + theta * i_temp * (qr * xxlv) * i_cp;
            qr = 0.f;
            nr = 0.f;
            vtrm = 0.f;
            vtrn = 0.f;
        }

        // Post-nucleation distribution update (:2109-2255).
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if (qi[cc] >= ISH_QSMALL) {
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                float nucfrac = totalnuc[cc] / qi[cc];
                nucfrac = ish_min(nucfrac, 1.0f);
                nucfrac = ish_max(nucfrac, 0.0f);
                if (cc == ICE3) nucfrac = 0.f;
                rhobar[cc] = ISH_RHOI * nucfrac + (1.f - nucfrac) * rhobar[cc];
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                nibnuc[cc] = ish_max(nibnuc[cc], ISH_QNSMALL);
                ani[cc] = ish_max(gfk_pow(((ai[cc] * ai[cc]) / (ci[cc] * nibnuc[cc])), 0.333333333333f), 2.e-6f);
                cni[cc] = ish_max(gfk_pow(((ci[cc] * ci[cc]) / (ai[cc] * nibnuc[cc])), 0.333333333333f), 2.e-6f);
                ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
                ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                aniold[cc] = ani[cc];
                cniold[cc] = gfk_pow(ao, 1.f - dsold[cc]) * gfk_pow(aniold[cc], dsold[cc]);
                if (nucfrac > 1.e-5f && totalnucn[cc] > 0.f) {
                    int gi = ish_gi(ISH_NU + 3.f);
                    int gi2 = ish_gi(ISH_NU + 4.f);
                    float anuc = (gfk_pow(((totalnuc[cc] * gammnu) / (ISH_RHOI * totalnucn[cc] * fourthirdspi *
                                  gtab[gi - 1])), 0.3333333f)) * gtab[gi2 - 1] / gtab[gi - 1];
                    anuc = ish_max(anuc, 2.e-6f);
                    anuc = ish_min(anuc, 3.e-3f);
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    gi2 = ish_gi(ISH_NU + 3.f + deltastr[cc]);
                    int gi3 = ish_gi(ISH_NU + 2.f + (2.f * deltastr[cc]));
                    float amass = ani[cc] * gtab[gi2 - 1] / gtab[gi - 1];
                    float cmass = cni[cc] * gtab[gi3 - 1] / gtab[gi - 1];
                    amass = amass * (1.f - nucfrac) + anuc * nucfrac;
                    cmass = cmass * (1.f - nucfrac) + anuc * nucfrac;
                    ani[cc] = ish_max((amass * gtab[gi - 1] / gtab[gi2 - 1]), 2.e-6f);
                    cni[cc] = ish_max((cmass * gtab[gi - 1] / gtab[gi3 - 1]), 2.e-6f);
                    deltastr[cc] = ish_delta_or_one(ani[cc], cni[cc]);
                    deltastr[cc] = ish_toward_sphere(deltastr[cc], dsold[cc]);
                }
                alphstr = gfk_pow(ao, 1.f - deltastr[cc]);
                alphv = fourthirdspi * alphstr;
                betam = 2.f + deltastr[cc];
                int gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                ani[cc] = gfk_pow(((qi[cc] * gammnu) / (rhobar[cc] * ni[cc] * alphv * gtab[gi - 1])),
                                  1.f / betam);
                cni[cc] = gfk_pow(ao, 1.f - deltastr[cc]) * gfk_pow(ani[cc], deltastr[cc]);
                ai[cc] = ai[cc] + cniold[cc] * nibnuc[cc] * 2.f * aniold[cc] * (ani[cc] - aniold[cc]) +
                         nibnuc[cc] * (aniold[cc] * aniold[cc]) * (cni[cc] - cniold[cc]) +
                         (aniold[cc] * aniold[cc]) * cniold[cc] * (ni[cc] - nibnuc[cc]);
                ci[cc] = ci[cc] + aniold[cc] * nibnuc[cc] * 2.f * cniold[cc] * (cni[cc] - cniold[cc]) +
                         nibnuc[cc] * (cniold[cc] * cniold[cc]) * (ani[cc] - aniold[cc]) +
                         (cniold[cc] * cniold[cc]) * aniold[cc] * (ni[cc] - nibnuc[cc]);
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                deltastr[cc] = ish_delta_or_one(ani[cc], cni[cc]);
                ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                              cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                              alphstr, alphv, betam);
                dsold[cc] = deltastr[cc];
                vtrzi1[cc] = ish_min(vtrzi1[cc], 25.f);
                vtrmi1[cc] = ish_min(vtrmi1[cc], 25.f);
                vtrni1[cc] = ish_min(vtrni1[cc], 25.f);
            } else {
                qv = qv + qi[cc];
                theta = theta + theta * i_temp * (qi[cc] * xxls) * i_cp;
                qi[cc] = 0.f; ai[cc] = 0.f; ni[cc] = 0.f; ci[cc] = 0.f;
                vtrzi1[cc] = 0.f; vtrmi1[cc] = 0.f; vtrni1[cc] = 0.f;
            }
        }

        // Aggregation update (:2259-2351).
        if (temp <= ISH_T0) {
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                dsold[cc] = deltastr[cc];
                nibnuc[cc] = ish_max(ni[cc], ISH_QNSMALL);
                qi[cc] = qi[cc] + qagg[cc];
                ni[cc] = ni[cc] + nagg[cc];
                if (qi[cc] > ISH_QSMALL) {
                    if (cc == ICE3) {
                        rhobar[cc] = 50.f;
                        float phiagg3 = 0.2f;
                        ani[cc] = .5f * dn3;
                        int gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                        cni[cc] = phiagg3 * ani[cc] * gammnu / gtab[gi - 1];
                        ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                        ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
                        ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                        ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                        ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                        deltastr[cc] = ish_delta_or_one(ani[cc], cni[cc]);
                        ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                                      cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                                      alphstr, alphv, betam);
                    } else {
                        ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                        ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                        ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                        ani[cc] = ish_max(gfk_pow(((ai[cc] * ai[cc]) / (ci[cc] * nibnuc[cc])), 0.333333333333f), 2.e-6f);
                        cni[cc] = ish_max(gfk_pow(((ci[cc] * ci[cc]) / (ai[cc] * nibnuc[cc])), 0.333333333333f), 2.e-6f);
                        aniold[cc] = ani[cc];
                        cniold[cc] = cni[cc];
                        deltastr[cc] = dsold[cc];
                        alphstr = gfk_pow(ao, 1.f - deltastr[cc]);
                        int gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                        ani[cc] = gfk_pow(((qi[cc]) / (ni[cc] * rhobar[cc] * fourthirdspi * alphstr *
                                           gtab[gi - 1] * i_gammnu)), (1.f / (2.f + deltastr[cc])));
                        ani[cc] = ish_max(ani[cc], 2.e-6f);
                        cni[cc] = alphstr * gfk_pow(ani[cc], deltastr[cc]);
                        ai[cc] = ai[cc] + cniold[cc] * nibnuc[cc] * 2.f * aniold[cc] * (ani[cc] - aniold[cc]) +
                                 nibnuc[cc] * (aniold[cc] * aniold[cc]) * (cni[cc] - cniold[cc]) +
                                 (aniold[cc] * aniold[cc]) * cniold[cc] * (ni[cc] - nibnuc[cc]);
                        ci[cc] = ci[cc] + aniold[cc] * nibnuc[cc] * 2.f * cniold[cc] * (cni[cc] - cniold[cc]) +
                                 nibnuc[cc] * (cniold[cc] * cniold[cc]) * (ani[cc] - aniold[cc]) +
                                 (cniold[cc] * cniold[cc]) * aniold[cc] * (ni[cc] - nibnuc[cc]);
                        ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                        ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                        ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                        deltastr[cc] = ish_delta_or_one(ani[cc], cni[cc]);
                        ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                                      cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                                      alphstr, alphv, betam);
                    }
                }
            }
        }

        // Thermodynamics (:2356-2377).
        theta = theta + theta * i_temp * ((mnuccd) * xxls * i_cp * dt);
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            theta = theta + theta * i_temp * ((prd[cc]) * xxls * i_cp * dt);
            theta = theta + theta * i_temp * ((qmlt[cc]) * xxlf * i_cp * dt);
        }
        theta = theta + theta * i_temp * (pre * xxlv) * i_cp * dt;
        if (temp <= ISH_T0) {
            theta = theta + theta * i_temp * (mim + mimr + mbiggr) * xxlf * i_cp * dt;
            for (int cc = 0; cc < ISH_CAT; ++cc)
                theta = theta + theta * i_temp * (prdr[cc] + dQRfzri[cc]) * xxlf * i_cp * dt;
        }
        temp = theta / gfk_pow(100000.f / pres, ISH_RCP);
        i_temp = 1.f / temp;
        qvs = 0.622f * ish_polysvp(temp, 0) / (pres - ish_polysvp(temp, 0));
        xxlv = 3.1484E6f - 2370.f * temp;
        rhoair = pres / (ISH_RD * temp);
        i_rhoair = 1.f / rhoair;

        // Saturation adjustment for qc (:2382-2395).
        pcc = (qv - qvs) / (1.0f + (xxlv * xxlv) * qvs / (ISH_CP * ISH_RV * (temp * temp))) * i_dt;
        if (pcc * dt + qc < 0.f) pcc = -qc / dt;
        theta = theta + theta * i_temp * pcc * xxlv * i_cp * dt;
        qv = qv - pcc * dt;
        qc = qc + pcc * dt;
        if (qc <= ISH_QSMALL) {
            qv = qv + qc;
            theta = theta + theta * i_temp * (qc * xxlv) * i_cp;
            qc = 0.f;
            nc = 0.f;
        }

        // Cloud effective radius (:2400-2413).
        {
            const float sig = ISH_SIG;
            float lwc = qc * rhoair;
            float ncm3dum = nc * rhoair;
            float r_n;
            if (ncm3dum > 0.f) {
                r_n = gfk_pow(((lwc) / (fourthirdspi * ncm3dum * ISH_RHOW * ISH_EXP_45SIG2)), 0.33333333333f);
            } else {
                r_n = 0.f;
            }
            effc1d = r_n * (5.f * ISH_EXP_SIG2_HALF);
            effc1d = ish_min(effc1d, 50.e-6f);
            effc1d = ish_max(effc1d, 2.51e-6f);
        }

        // Final check on ice before sedimentation, fall speeds, radiative
        // sizes (:2422-2606).
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            if (qi[cc] > ISH_QSMALL) {
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                ish_axes(ai[cc], ci[cc], ni[cc], ani[cc], cni[cc], deltastr[cc]);
                ish_var_check(gtab, gammnu, fourthirdspi, qi[cc], deltastr[cc], ani[cc],
                              cni[cc], rni[cc], rhobar[cc], ni[cc], ai[cc], ci[cc],
                              alphstr, alphv, betam);
                if (cc == ICE3) {
                    ish_aggregate_shape(gtab, C, temp, qi[cc], ni[cc], ai[cc], ci[cc], ani[cc],
                                        cni[cc], rni[cc], rhobar[cc], deltastr[cc], alphstr,
                                        alphv, betam);
                }
                fv[cc] = 1.0f;
                fh[cc] = 1.0f;
                int gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                float phiivt = cni[cc] / ani[cc] * gtab[gi - 1] * i_gammnu;
                alphstr = gfk_pow(ao, 1.f - deltastr[cc]);
                IshFall f = ish_best(phiivt, rhobar[cc], rhoair, mu, alphstr, deltastr[cc],
                                     fourthirdspi, ani[cc], i_gammnu);
                vtrni1[cc] = ish_min((mu / rhoair * 0.5f * f.am * gfk_pow(f.xn, f.bm) *
                                      gfk_pow(ani[cc], f.bx * f.bm - 1.f) *
                                      (ish_gamma(ISH_NU + f.bx * f.bm - 1.f)) * i_gammnu), 25.f);
                vtrmi1[cc] = ish_min((mu / rhoair * 0.5f * f.am * gfk_pow(f.xn, f.bm) *
                                      gfk_pow(ani[cc], f.bx * f.bm - 1.f) *
                                      (ish_gamma(ISH_NU + f.bx * f.bm - 1.f + 2.f + deltastr[cc])) /
                                      (ish_gamma(ISH_NU + 2.f + deltastr[cc]))), 25.f);
                if (temp > ISH_T0 && qmlt[cc] < 0.f) vtrni1[cc] = vtrmi1[cc];

                gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
                float phirad = cni[cc] / ani[cc] * gtab[gi - 1] * i_gammnu;
                if (phirad <= 1.f) {
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    int gi2 = ish_gi(ISH_NU + 2.f);
                    effi[cc] = cni[cc] * gtab[gi - 1] / gtab[gi2 - 1];
                } else {
                    gi = ish_gi(ISH_NU + 2.f + deltastr[cc]);
                    int gi2 = ish_gi(ISH_NU + 1.f + deltastr[cc]);
                    effi[cc] = ani[cc] * gtab[gi - 1] / gtab[gi2 - 1];
                }
                ci[cc] = cni[cc] * cni[cc] * ani[cc] * ni[cc];
                ai[cc] = ani[cc] * ani[cc] * cni[cc] * ni[cc];
                ai[cc] = ish_max(ai[cc], ISH_QASMALL);
                ci[cc] = ish_max(ci[cc], ISH_QASMALL);
                ni[cc] = ish_max(ni[cc], ISH_QNSMALL);
            } else {
                qv = qv + qi[cc];
                theta = theta + theta * i_temp * (qi[cc] * xxls) * i_cp;
                qi[cc] = 0.f; ai[cc] = 0.f; ni[cc] = 0.f; ci[cc] = 0.f;
                vtrzi1[cc] = 0.f; vtrmi1[cc] = 0.f; vtrni1[cc] = 0.f;
            }

            // Output diagnostics (:2611-2643).
            int gi = ish_gi(ISH_NU - 1.f + deltastr[cc]);
            if (ai[cc] > 1.e-12f && ci[cc] > 1.e-12f && ani[cc] > 2.e-6f && cni[cc] > 2.e-6f &&
                qi[cc] > 1.e-9f) {
                phii1d[cc] = cni[cc] / ani[cc] * gtab[gi - 1] * i_gammnu;
                rhopo1d[cc] = rhobar[cc];
            }
            gi = ish_gi(ISH_NU + 3.f + deltastr[cc]);
            int gi2 = ish_gi(ISH_NU + 2.f + deltastr[cc]);
            int gi3 = ish_gi(ISH_NU + 2.f + (2.f * deltastr[cc]));
            if (ai[cc] > 1.e-12f && ci[cc] > 1.e-12f && ani[cc] > 2.e-6f && cni[cc] > 2.e-6f &&
                qi[cc] > 1.e-9f) {
                di1d[cc] = ish_max((2.f * ani[cc] * gtab[gi - 1] / gtab[gi2 - 1]),
                                   (2.f * cni[cc] * gtab[gi3 - 1] / gtab[gi2 - 1]));
                vmi1d[cc] = vtrmi1[cc];
            }
            if (ni[cc] > 0.f && ani[cc] > 0.f) {
                float b = (fourthirdspi * rhobar[cc] * gfk_pow(ao, 1.f - deltastr[cc])) /
                          (gfk_pow(2.f, 2.f + deltastr[cc]));
                vtrzi1[cc] = 0.176f / 0.93f * ISH_SIX_OVER_PI_SQ * (b * b) *
                             rhoair / 900.f / 900.f * ni[cc] *
                             gfk_pow(2.f * ani[cc], 2.f * (2.f + deltastr[cc])) *
                             ish_gamma(ISH_NU + 2.f * (2.f + deltastr[cc])) * i_gammnu;
                vtrzi1[cc] = 1.e18f * vtrzi1[cc];
            } else {
                vtrzi1[cc] = 0.f;
            }
        }
        dbzsum = dbzr + vtrzi1[ICE1] + vtrzi1[ICE2] + vtrzi1[ICE3];
        if (dbzsum != 0.f) {
            dbz1d = 10.f * gfk_log10(dbzsum);
            dbz1d = ish_max(dbz1d, -35.f);
        }

        // Ice effective radius (:2655-2667).
        float totaliwc = 0.f;
        for (int cc = 0; cc < ISH_CAT; ++cc) totaliwc = totaliwc + qi[cc];
        if (totaliwc > 0.f) {
            float radw1 = ish_max((ish_min((qi[ICE1] / totaliwc), 1.f)), 0.f);
            float radw2 = ish_max((ish_min((qi[ICE2] / totaliwc), 1.f)), 0.f);
            float radw3 = ish_max((ish_min((qi[ICE3] / totaliwc), 1.f)), 0.f);
            effi1d = ish_max((ish_min((effi[ICE1] * radw1 + effi[ICE2] * radw2 +
                                       effi[ICE3] * radw3), 999.e-6f)), 5.e-6f);
        } else {
            effi1d = 0.f;
        }

        if (qr > ISH_QSMALL) sedi = true;
        for (int cc = 0; cc < ISH_CAT; ++cc)
            if (qi[cc] > ISH_QSMALL) sedi = true;
        for (int cc = 0; cc < ISH_CAT; ++cc) rhobar_c[cc] = rhobar[cc];
    } else {
        // Not domicro: push to vapor (:2689-2724).
        xxls = 3.15e6f - 2370.f * temp + 0.3337e6f;
        xxlv = 3.1484E6f - 2370.f * temp;
        i_temp = 1.f / temp;
        rhoair = pres / (ISH_RD * temp);
        i_rhoair = 1.f / rhoair;
        qv = qv + qc + qr;
        theta = theta + theta * i_temp * ((qc + qr) * xxlv) * i_cp;
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            qv = qv + qi[cc];
            theta = theta + theta * i_temp * ((qi[cc]) * xxls) * i_cp;
            qi[cc] = 0.f; ni[cc] = 0.f; ai[cc] = 0.f; ci[cc] = 0.f;
            vtrmi1[cc] = 0.f; vtrni1[cc] = 0.f; vtrzi1[cc] = 0.f;
            vmi1d[cc] = 0.f; di1d[cc] = 0.f; phii1d[cc] = 0.f; rhopo1d[cc] = 0.f;
            qipre1d[cc] = 0.f; icetype1d[cc] = 0.f;
        }
        effc1d = 0.f;
        effi1d = 0.f;
        dbz1d = -35.f;
        qc = 0.f;
        qr = 0.f;
        nr = 0.f;
        vtrm = 0.f;
        vtrn = 0.f;
        qrpre1d = 0.f;
    }
    rhoair_out = rhoair;
    vtrm_out = vtrm;
    vtrn_out = vtrn;
    for (int cc = 0; cc < ISH_CAT; ++cc) {
        vtrmi_out[cc] = vtrmi1[cc];
        vtrni_out[cc] = vtrni1[cc];
    }
    return sedi;
}

// Independent cell physics runs in parallel across levels and columns.
// Sedimentation and its density carry remain in the native column order.
extern "C" __global__ void ishk_levels(
        float* F0, float* F1, float* F2, float* F3,
        float* F4, float* F5, float* F6, float* F7,
        float* F8, float* F9, float* F10, float* F11,
        float* F12, float* F13, float* F14, float* F15,
        float* F16, float* F17, float* F18, float* D0,
        float* D1, float* D2, float* D3, float* D4,
        float* D5, float* D6, float* D7, float* D8,
        float* D9, float* D10, float* D11, float* D12,
        float* D13, float* D14, float* D15, float* D16,
        float* D17, float* W0, float* W1, float* W2,
        float* W3, float* W4, float* W5, float* W6,
        float* W7, float* W8, float* W9, float* W10,
        float* W11, float* W12, float* S0, float* S1,
        float* S2, float* S3,
        const float* itab, const float* itabr, const float* gtab,
        const float* coltab, const float* coltabn, int ncol, int nz, float dt_in)
{
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= (size_t)ncol * nz) return;
    float* const F[] = {F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18};
    float* const D[] = {D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17};
    float* const W[] = {W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12};
    float* const S[] = {S0, S1, S2, S3};
    IshTables T;
    T.itab = itab; T.itabr = itabr; T.gtab = gtab; T.coltab = coltab; T.coltabn = coltabn;

    IshCtx C;
    C.dt = dt_in;
    C.gammnu = ish_gamma(ISH_NU);
    C.i_gammnu = 1.f / C.gammnu;
    C.i_dt = 1.f / dt_in;
    C.i_cp = 1.f / ISH_CP;
    C.fourthirdspi = 4.f / 3.f * ISH_PI;
    const float dt = dt_in;

    float qrpre1d = 0.f;
    float qipre1d[ISH_CAT] = {0.f, 0.f, 0.f};
    float rhobar_c[ISH_CAT] = {-1.f, -1.f, -1.f};

    float pres = F[IF_P][o];
    float th = F[IF_TH][o], qv = F[IF_QV][o], qc = F[IF_QC][o];
    float qr = F[IF_QR][o], nr = F[IF_NR][o];
    float qi[ISH_CAT], ni[ISH_CAT], ai[ISH_CAT], ci[ISH_CAT];
    for (int cc = 0; cc < ISH_CAT; ++cc) {
        qi[cc] = F[IF_QI + cc][o];
        ni[cc] = F[IF_NI + cc][o];
        ai[cc] = F[IF_AI + cc][o];
        ci[cc] = F[IF_CI + cc][o];
    }
    // The wrapper's per-level resets (:361-393).
    float effc1d = 0.f, effi1d = 0.f, dbz1d = -35.f;
    float vmi1d[ISH_CAT] = {0.f, 0.f, 0.f}, di1d[ISH_CAT] = {0.f, 0.f, 0.f};
    float rhopo1d[ISH_CAT] = {0.f, 0.f, 0.f}, phii1d[ISH_CAT] = {0.f, 0.f, 0.f};
    float icetype1d[ISH_CAT] = {0.f, 0.f, 0.f};
    float rhoair, vtrm, vtrn, vtrmi[ISH_CAT], vtrni[ISH_CAT];
    bool s = ish_level(T, C, pres, th, qv, qc, qr, nr, qi, ni, ai, ci, effc1d,
                       effi1d, dbz1d, vmi1d, di1d, rhopo1d, phii1d, icetype1d,
                       qrpre1d, qipre1d, rhobar_c, rhoair, vtrm, vtrn, vtrmi,
                       vtrni);
    W[IW_SEDI][o] = s ? 1.f : 0.f;
    for (int cc = 0; cc < ISH_CAT; ++cc)
        W[IW_RHOBAR + cc][o] = rhobar_c[cc];
    F[IF_TH][o] = th; F[IF_QV][o] = qv; F[IF_QC][o] = qc;
    F[IF_QR][o] = qr; F[IF_NR][o] = nr;
    for (int cc = 0; cc < ISH_CAT; ++cc) {
        F[IF_QI + cc][o] = qi[cc];
        F[IF_NI + cc][o] = ni[cc];
        F[IF_AI + cc][o] = ai[cc];
        F[IF_CI + cc][o] = ci[cc];
        D[ID_VMI + cc][o] = vmi1d[cc];
        D[ID_DI + cc][o] = di1d[cc];
        D[ID_RHOPO + cc][o] = rhopo1d[cc];
        D[ID_PHII + cc][o] = phii1d[cc];
        D[ID_ITYPE + cc][o] = icetype1d[cc];
        W[IW_VTRMI + cc][o] = vtrmi[cc];
        W[IW_VTRNI + cc][o] = vtrni[cc];
    }
    // The wrapper's clamps on the way out (:409-411).
    D[ID_EFFC][o] = ish_max((ish_min(effc1d, 50.e-6f)), 2.51e-6f);
    D[ID_EFFI][o] = ish_max((ish_min(effi1d, 999.e-6f)), 5.e-6f);
    D[ID_DBZ][o] = ish_max(dbz1d, -35.f);
    W[IW_RHO][o] = rhoair;
    W[IW_VTRM][o] = vtrm;
    W[IW_VTRN][o] = vtrn;
}

extern "C" __global__ void ishk_column(
        float* F0, float* F1, float* F2, float* F3,
        float* F4, float* F5, float* F6, float* F7,
        float* F8, float* F9, float* F10, float* F11,
        float* F12, float* F13, float* F14, float* F15,
        float* F16, float* F17, float* F18, float* D0,
        float* D1, float* D2, float* D3, float* D4,
        float* D5, float* D6, float* D7, float* D8,
        float* D9, float* D10, float* D11, float* D12,
        float* D13, float* D14, float* D15, float* D16,
        float* D17, float* W0, float* W1, float* W2,
        float* W3, float* W4, float* W5, float* W6,
        float* W7, float* W8, float* W9, float* W10,
        float* W11, float* W12, float* S0, float* S1,
        float* S2, float* S3,
        const float* itab, const float* itabr, const float* gtab,
        const float* coltab, const float* coltabn, int ncol, int nz, float dt_in)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncol) return;
    float* const F[] = {F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18};
    float* const D[] = {D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17};
    float* const W[] = {W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12};
    float* const S[] = {S0, S1, S2, S3};
    IshTables T;
    T.itab = itab; T.itabr = itabr; T.gtab = gtab; T.coltab = coltab; T.coltabn = coltabn;

    IshCtx C;
    C.dt = dt_in;
    C.gammnu = ish_gamma(ISH_NU);
    C.i_gammnu = 1.f / C.gammnu;
    C.i_dt = 1.f / dt_in;
    C.i_cp = 1.f / ISH_CP;
    C.fourthirdspi = 4.f / 3.f * ISH_PI;
    const float dt = dt_in;

    bool sedi = false;
    float qrpre1d = 0.f;
    float qipre1d[ISH_CAT] = {0.f, 0.f, 0.f};
    float rhobar_c[ISH_CAT] = {ISH_RHOI, ISH_RHOI, 50.f};

    // Preserve the native last-active-level density carried into its
    // post-sedimentation loop. A negative marker denotes an inactive level.
    for (int k = 0; k < nz; ++k) {
        size_t o = (size_t)k * ncol + i;
        sedi = sedi || W[IW_SEDI][o] != 0.f;
        if (W[IW_RHOBAR][o] >= 0.f)
            for (int cc = 0; cc < ISH_CAT; ++cc)
                rhobar_c[cc] = W[IW_RHOBAR + cc][o];
    }

    // Sedimentation, Reisner et al. 1998, on split steps (:2734-2836).
    if (sedi) {
        int nstep = 1;
        for (int k = nz - 1; k >= 0; --k) {
            size_t o = (size_t)k * ncol + i;
            float maxfall = W[IW_VTRMI + 0][o];
            maxfall = ish_max(maxfall, W[IW_VTRNI + 0][o]);
            maxfall = ish_max(maxfall, W[IW_VTRMI + 1][o]);
            maxfall = ish_max(maxfall, W[IW_VTRNI + 1][o]);
            maxfall = ish_max(maxfall, W[IW_VTRMI + 2][o]);
            maxfall = ish_max(maxfall, W[IW_VTRNI + 2][o]);
            maxfall = ish_max(maxfall, W[IW_VTRM][o]);
            maxfall = ish_max(maxfall, W[IW_VTRN][o]);
            int n = (int)(maxfall * dt / F[IF_DZ][o] + 1.f);
            nstep = n > nstep ? n : nstep;
            float rho = W[IW_RHO][o];
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                F[IF_QI + cc][o] = F[IF_QI + cc][o] * rho;
                F[IF_NI + cc][o] = F[IF_NI + cc][o] * rho;
                F[IF_AI + cc][o] = F[IF_AI + cc][o] * rho;
                F[IF_CI + cc][o] = F[IF_CI + cc][o] * rho;
            }
            F[IF_QR][o] = F[IF_QR][o] * rho;
            F[IF_NR][o] = F[IF_NR][o] * rho;
        }
        const float fn = (float)nstep;
        for (int nn = 1; nn <= nstep; ++nn) {
            // Top level: everything that falls in leaves through the bottom.
            int k = nz - 1;
            size_t o = (size_t)k * ncol + i;
            float dzk = F[IF_DZ][o];
            float fa_q[ISH_CAT], fa_n[ISH_CAT], fa_a[ISH_CAT], fa_c[ISH_CAT], fa_qr, fa_nr;
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                float vm = W[IW_VTRMI + cc][o], vn = W[IW_VTRNI + cc][o];
                fa_q[cc] = vm * F[IF_QI + cc][o];
                fa_n[cc] = vn * F[IF_NI + cc][o];
                fa_a[cc] = vm * F[IF_AI + cc][o];
                fa_c[cc] = vm * F[IF_CI + cc][o];
                F[IF_QI + cc][o] = F[IF_QI + cc][o] - (fa_q[cc] / dzk) * dt / fn;
                F[IF_NI + cc][o] = F[IF_NI + cc][o] - (fa_n[cc] / dzk) * dt / fn;
                F[IF_AI + cc][o] = F[IF_AI + cc][o] - (fa_a[cc] / dzk) * dt / fn;
                F[IF_CI + cc][o] = F[IF_CI + cc][o] - (fa_c[cc] / dzk) * dt / fn;
            }
            fa_qr = W[IW_VTRM][o] * F[IF_QR][o];
            fa_nr = W[IW_VTRN][o] * F[IF_NR][o];
            F[IF_QR][o] = F[IF_QR][o] - (fa_qr / dzk) * dt / fn;
            F[IF_NR][o] = F[IF_NR][o] - (fa_nr / dzk) * dt / fn;
            for (k = nz - 2; k >= 0; --k) {
                o = (size_t)k * ncol + i;
                dzk = F[IF_DZ][o];
                for (int cc = 0; cc < ISH_CAT; ++cc) {
                    float vm = W[IW_VTRMI + cc][o], vn = W[IW_VTRNI + cc][o];
                    float fq = vm * F[IF_QI + cc][o];
                    float fnn = vn * F[IF_NI + cc][o];
                    float fa = vm * F[IF_AI + cc][o];
                    float fc = vm * F[IF_CI + cc][o];
                    F[IF_QI + cc][o] = F[IF_QI + cc][o] + ((fa_q[cc] - fq) / dzk) * dt / fn;
                    F[IF_NI + cc][o] = F[IF_NI + cc][o] + ((fa_n[cc] - fnn) / dzk) * dt / fn;
                    F[IF_AI + cc][o] = F[IF_AI + cc][o] + ((fa_a[cc] - fa) / dzk) * dt / fn;
                    F[IF_CI + cc][o] = F[IF_CI + cc][o] + ((fa_c[cc] - fc) / dzk) * dt / fn;
                    if (k == 0 && F[IF_QI + cc][o] > 1.e-9f)
                        qipre1d[cc] = qipre1d[cc] + fq * dt / fn;
                    fa_q[cc] = fq; fa_n[cc] = fnn; fa_a[cc] = fa; fa_c[cc] = fc;
                }
                float fqr = W[IW_VTRM][o] * F[IF_QR][o];
                float fnr = W[IW_VTRN][o] * F[IF_NR][o];
                F[IF_QR][o] = F[IF_QR][o] + ((fa_qr - fqr) / dzk) * dt / fn;
                F[IF_NR][o] = F[IF_NR][o] + ((fa_nr - fnr) / dzk) * dt / fn;
                if (k == 0 && F[IF_QR][o] > 1.e-9f) qrpre1d = qrpre1d + fqr * dt / fn;
                fa_qr = fqr; fa_nr = fnr;
            }
        }
        for (int k = 0; k < nz; ++k) {
            size_t o = (size_t)k * ncol + i;
            float irho = 1.f / W[IW_RHO][o];
            for (int cc = 0; cc < ISH_CAT; ++cc) {
                F[IF_QI + cc][o] = F[IF_QI + cc][o] * irho;
                F[IF_NI + cc][o] = F[IF_NI + cc][o] * irho;
                F[IF_AI + cc][o] = F[IF_AI + cc][o] * irho;
                F[IF_CI + cc][o] = F[IF_CI + cc][o] * irho;
            }
            F[IF_QR][o] = F[IF_QR][o] * irho;
            F[IF_NR][o] = F[IF_NR][o] * irho;
        }
    }

    // Check after sedimentation (:2841-2960).
    for (int k = 0; k < nz; ++k) {
        size_t o = (size_t)k * ncol + i;
        float pres = F[IF_P][o];
        float theta = F[IF_TH][o];
        float qv = F[IF_QV][o];
        float temp = theta / gfk_pow(100000.f / pres, ISH_RCP);
        float i_temp = 1.f / temp;
        float xxls = 3.15e6f - 2370.f * temp + 0.3337e6f;
        float xxlv = 3.1484E6f - 2370.f * temp;
        for (int cc = 0; cc < ISH_CAT; ++cc) {
            float qi = F[IF_QI + cc][o], ni = F[IF_NI + cc][o];
            float ai = F[IF_AI + cc][o], ci = F[IF_CI + cc][o];
            if (qi > ISH_QSMALL) {
                float ani, cni, rni, ds, alphstr, alphv, betam;
                ni = ish_max(ni, ISH_QNSMALL);
                ai = ish_max(ai, ISH_QASMALL);
                ci = ish_max(ci, ISH_QASMALL);
                ani = ish_max(gfk_pow(((ai * ai) / (ci * ni)), 0.333333333333f), 2.e-6f);
                cni = ish_max(gfk_pow(((ci * ci) / (ai * ni)), 0.333333333333f), 2.e-6f);
                if (ai < 1.e-12f || ci < 1.e-12f) {
                    ai = ish_min(ai, ci);
                    ci = ai;
                    ani = gfk_pow(ai / ni, 0.3333333333f);
                    cni = gfk_pow(ci / ni, 0.3333333333f);
                }
                if (ani < 2.e-6f || cni < 2.e-6f) {
                    ani = 2.e-6f;
                    cni = 2.e-6f;
                }
                ci = cni * cni * ani * ni;
                ai = ani * ani * cni * ni;
                ds = (gfk_log(cni) - ISH_LOG_AO) / (gfk_log(ani) - ISH_LOG_AO);
                if (cc == 2) {
                    ish_aggregate_shape(gtab, C, temp, qi, ni, ai, ci, ani, cni, rni,
                                        rhobar_c[cc], ds, alphstr, alphv, betam);
                }
                ish_var_check(gtab, C.gammnu, C.fourthirdspi, qi, ds, ani, cni, rni,
                              rhobar_c[cc], ni, ai, ci, alphstr, alphv, betam);
            } else {
                qv = qv + qi;
                theta = theta + theta * i_temp * (qi * xxls) * C.i_cp;
                qi = 0.f; ni = 0.f; ai = 0.f; ci = 0.f;
            }
            F[IF_QI + cc][o] = qi; F[IF_NI + cc][o] = ni;
            F[IF_AI + cc][o] = ai; F[IF_CI + cc][o] = ci;
        }
        float qr = F[IF_QR][o], nr = F[IF_NR][o];
        if (qr > ISH_QSMALL) {
            float lamr, n0rr;
            nr = ish_max(nr, ISH_QNSMALL);
            ish_rain_dsd(qr, nr, lamr, n0rr);
        } else {
            qv = qv + qr;
            theta = theta + theta * i_temp * (qr * xxlv) * C.i_cp;
            qr = 0.f;
            nr = 0.f;
        }
        F[IF_QR][o] = qr; F[IF_NR][o] = nr;
        F[IF_QV][o] = qv; F[IF_TH][o] = theta;
    }

    // Surface accumulation (:446-451).
    float snow = qipre1d[0] + qipre1d[1] + qipre1d[2];
    S[IS_SNOWNC][i] = S[IS_SNOWNC][i] + qipre1d[0] + qipre1d[1] + qipre1d[2];
    S[IS_SNOWNCV][i] = snow;
    S[IS_RAINNC][i] = S[IS_RAINNC][i] + qrpre1d + qipre1d[0] + qipre1d[1] + qipre1d[2];
    S[IS_RAINNCV][i] = qrpre1d + qipre1d[0] + qipre1d[1] + qipre1d[2];
}

// Between-step radar operator, using me_ishmael's own PSD constraints and
// diagnostics (:2076-2105, :2611-2651). Prognostics are read-only. This is
// used for analysis history, before any microphysics call has produced dBZ.
extern "C" __global__ void ishk_reflectivity(
    const float* p, const float* th, const float* qr, const float* nr,
    const float* qi1, const float* qi2, const float* qi3,
    const float* ni1, const float* ni2, const float* ni3,
    const float* ai1, const float* ai2, const float* ai3,
    const float* ci1, const float* ci2, const float* ci3,
    const float* gtab, float* out, unsigned long long cells)
{
    unsigned long long o = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= cells) return;
    float temp = th[o] / gfk_pow(100000.f / p[o], ISH_RCP);
    float rho = p[o] / (ISH_RD * temp);
    float sum = 0.f;
    float q = qr[o], n = nr[o];
    if (q > ISH_QSMALL) {
        n = ish_max(n, ISH_QNSMALL);
        float lambda, intercept;
        ish_rain_dsd(q, n, lambda, intercept);
        float l3 = lambda * lambda * lambda;
        sum = 1.e18f * (rho * n / l3 / l3 * 720.f);
    }
    const float* masses[3] = {qi1,qi2,qi3};
    const float* numbers[3] = {ni1,ni2,ni3};
    const float* axes_a[3] = {ai1,ai2,ai3};
    const float* axes_c[3] = {ci1,ci2,ci3};
    float gammnu = ish_gamma(ISH_NU), fourthirdspi = (4.f / 3.f) * ISH_PI;
    for (int cc=0;cc<3;++cc) {
        q=masses[cc][o];
        if (q <= ISH_QSMALL) continue;
        n=ish_max(numbers[cc][o],ISH_QNSMALL);
        float a=ish_max(axes_a[cc][o],ISH_QASMALL);
        float c=ish_max(axes_c[cc][o],ISH_QASMALL);
        float ani=ish_max(gfk_pow(a*a/(c*n),0.333333333333f),2.e-6f);
        float cni=ish_max(gfk_pow(c*c/(a*n),0.333333333333f),2.e-6f);
        float delta=ish_delta_or_one(ani,cni);
        float rni,rb,alphstr,alphv,betam;
        ish_var_check(gtab,gammnu,fourthirdspi,q,delta,ani,cni,rni,rb,n,a,c,alphstr,alphv,betam);
        float b=(fourthirdspi*rb*gfk_pow(ISH_AO,1.f-delta))/gfk_pow(2.f,2.f+delta);
        float z=0.176f/0.93f*ISH_SIX_OVER_PI_SQ*(b*b)*rho/900.f/900.f*n*
                gfk_pow(2.f*ani,2.f*(2.f+delta))*ish_gamma(ISH_NU+2.f*(2.f+delta))/gammnu;
        sum += 1.e18f*z;
    }
    out[o]=sum!=0.f ? ish_max(10.f*gfk_log10(sum),-35.f) : -35.f;
}
