// gpuwm/core/kernels/gd.cu
//
// Grell-Devenyi ensemble cumulus (cu_physics = 93), WRF v4.6.1
// phys/module_cu_gd.F -- GRELLDRV, CUP_enss and every routine under them, one
// CUDA thread per column, transcribed statement for statement in WRF's
// operation order.  Graded word for word against WRF's own compiled
// module_cu_gd.o by tests/test_gd_cu93.py (tools/gd_wrf461_oracle).
//
// Arithmetic rules, the same three every bitwise kernel in this tree keeps:
//
// 1. Every float32 operation is FADD/FSUB/FMUL/FDIV/FSQRT (the _rn
//    intrinsics), so no multiply and add the Fortran rounded twice can be
//    contracted into one FMA, whatever --fmad says.  Fortran evaluates a
//    chain of equal-precedence operators left to right; every chain below is
//    nested the same way, and the comment beside a non-obvious one spells the
//    Fortran.  Integer powers are what gfortran emits: x**2 = x*x,
//    x**3 = (x*x)*x, x**4 = (x*x)*(x*x).
// 2. No constant-constant arithmetic reaches the compiler's folder: the few
//    float32 constants WRF's front end (or its constant propagation) folds
//    are pinned words in GDK_* below, computed once in float32
//    round-to-nearest (tools/gd_wrf461_oracle/README.md lists them).
// 3. EXP and ALOG are WOOF's own float32 exp and log (gfk_exp, gfk_log in
//    the shared math header); SQRT is the IEEE square root.  Nothing else
//    transcendental is reached: GRELLDRV's one ATAN2 feeds `direction`,
//    which only the commented-out upstream-convection test
//    (cup_direction2) would read, so no output depends on it and it is not
//    computed.
//
// Dead code not transcribed (no output word depends on it, measured by the
// oracle): the steering-flow direction above; massfln (the next-step
// downdraft mass flux store, local to GRELLDRV and never read back);
// CUP_enss's non-forced downdraft (cup_dd_he / cup_dd_moisture on the
// unforced state -- only cdd from that cup_dd_nms call is used) and the
// perturbed-state downdraft and updraft-moisture calls inside the iedt loop
// (their outputs feed nothing; xaa0 reads only xz/xzu/xdby/xt_cup);
// massflx_stats's per-closure std/skewness/kurtosis (read by nothing; the
// column total xt_std IS computed because it gates the APR_* update).  None
// of those routines writes ierr, so dropping them cannot change a branch.
//
// Stock WRF closure counting is the forecast default (wrf_closure_count=1).
// The explicit zero arm exists for the correction experiment in
// tools/gd_wrf461_oracle/patches/gd-closure-count.patch. It is not selected
// by cu_physics=93. The correction counts closures once and removes closure
// 16 over water; stock WRF decrements the count in all nine ensemble passes.
// Both arms are graded separately against their native Fortran objects.
//
// Layout.  Column arrays live in a caller-sized global workspace, not in the
// local frame (the GF lesson: a local frame is priced at the card's whole
// resident-thread capacity).  Region per block, lanes interleaved:
//     ws[blockIdx.x * GD_WS_FLOATS(nz) * blockDim.x
//        + (slot * (nz+2) + k) * blockDim.x + threadIdx.x]
// with k the Fortran 1-based level (0 unused, nz+1 = kte), followed by the
// 2 x 144 ensemble words.

#define GD_NSLOT 76
#define GD_ENS 144
#define GD_WS_FLOATS(nz) (GD_NSLOT * ((nz) + 2) + 2 * GD_ENS)

// Input level fields, gd_grelldrv's lvin[col][field][k] order.
#define GDI_U 0
#define GDI_V 1
#define GDI_W 2          // w on the face BELOW mass level k (WRF passes w_2)
#define GDI_T 3
#define GDI_Q 4
#define GDI_P 5          // hydrostatic p (WRF passes p_hyd), Pa
#define GDI_PI 6
#define GDI_RHO 7
#define GDI_RTHFTEN 8
#define GDI_RQVFTEN 9
#define GDI_RTHRATEN 10
#define GDI_RTHBLTEN 11
#define GDI_RQVBLTEN 12
#define GDI_NLEV 13
// Input scalars scin[col][..]
#define GDS_P8W1 0       // p_hyd_w at the surface face, Pa
#define GDS_HT 1
#define GDS_XLAND 2
#define GDS_GSW 3
#define GDS_DT 4
#define GDS_HTOP 5
#define GDS_HBOT 6
#define GDS_APR 7        // 8 accumulators: GR W MC ST AS CAPMA CAPME CAPMI
#define GDS_NSCA 15
// Input ints iin[col][..]
#define GDN_BAND 0       // 2 interior, 1 i-edge band (j inside), 0 j-edge band
#define GDN_FQI 1
#define GDN_NINT 2
// Output levels lev[col][..][k]
#define GDO_RTHCUTEN 0
#define GDO_RQVCUTEN 1
#define GDO_RQCCUTEN 2
#define GDO_RQICUTEN 3
#define GDO_GDC 4
#define GDO_GDC2 5
#define GDO_NLEV 6
// Output scalars sca[col][..]
#define GDR_RAINCV 0
#define GDR_PRATEC 1
#define GDR_HTOP 2
#define GDR_HBOT 3
#define GDR_APR 4        // 8 accumulators
#define GDR_PRET 12      // pret after neg_check, before the edge band
#define GDR_NSCA 13
// Output ints isc[col][..]
#define GDK_KTOP_DEEP 0
#define GDK_KBCON 1
#define GDK_KTOP 2
#define GDK_IERR 3
#define GDK_K22 4
#define GDK_NINT 5

// Pinned float32 words (rule 2).  entr_rate = .2/radius with radius = 12000
// (iens = 1: 14000 - float(1)*2000); the rest as named.
#define GDK_ENTR      __uint_as_float(0x378bcf65u)   // .2/12000.
#define GDK_CD0       __uint_as_float(0x35dfb23bu)   // 0.1*entr_rate
#define GDK_CD15      __uint_as_float(0x37d1b718u)   // 1.5*entr_rate
#define GDK_CD10      __uint_as_float(0x392ec33eu)   // 10.0*entr_rate
#define GDK_RADIUS_UP __uint_as_float(0x463b8000u)   // .2/mentr_rate
#define GDK_A_DD      __uint_as_float(0x3f7851ecu)   // 1.-perc, perc=.03
#define GDK_C01G      __uint_as_float(0x3dc8e8a7u)   // .01*9.81 (front-end fold)
#define GDK_LOG61071  __uint_as_float(0x40cd4496u)   // ALOG(610.71) (folded)

__constant__ float GD_PCRIT[15] = {850.f, 800.f, 750.f, 700.f, 650.f, 600.f,
    550.f, 500.f, 450.f, 400.f, 350.f, 300.f, 250.f, 200.f, 150.f};
__constant__ float GD_ACRIT[15] = {.0633f, .0445f, .0553f, .0664f, .075f,
    .1082f, .1521f, .2216f, .3151f, .3677f, .41f, .5255f, .7663f, 1.1686f,
    1.6851f};
__constant__ float GD_ACRITT[15] = {.203f, .515f, .521f, .566f, .625f, .665f,
    .659f, .688f, .743f, .813f, .886f, .947f, 1.138f, 1.377f, 1.896f};

struct GdCol {
    float* p;
    int s;
    __device__ __forceinline__ float& operator[](int k) const { return p[k * s]; }
};

struct GdEns {
    float* p;
    int s;
    // 1-based ensemble index, as Fortran's xf_ens(i,j,n)
    __device__ __forceinline__ float& operator()(int n) const { return p[(n - 1) * s]; }
};

__device__ __forceinline__ float gd_max(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float gd_min(float a, float b) { return (a < b) ? a : b; }

// ---------------------------------------------------------------------------
// cup_env (module_cu_gd.F:2345-2480).  itest 0: heights from the hypsometric
// stack and he from (z,t,q); itest 2: heights from the given he.
// ---------------------------------------------------------------------------
__device__ void gd_cup_env(GdCol z, GdCol qes, GdCol he, GdCol hes, GdCol t,
                           GdCol q, GdCol p, GdCol tv, float z1, float psur,
                           int ierr, float tcrit, int itest, float xl, float cp,
                           int ktf)
{
    float ht1 = FDIV(xl, cp);
    float ht2 = FDIV(2.834E6f, cp);
    float be1 = FDIV(FMUL(.622f, ht1), .286f);
    float ae1 = FADD(FDIV(be1, 273.f), GDK_LOG61071);
    float be2 = FDIV(FMUL(.622f, ht2), .286f);
    float ae2 = FADD(FDIV(be2, 273.f), GDK_LOG61071);
    if (ierr != 0) return;
    for (int k = 1; k <= ktf; ++k) {
        float ae = ae1, be = be1;
        if (t[k] <= tcrit) { ae = ae2; be = be2; }
        float e = gfk_exp(FSUB(ae, FDIV(be, t[k])));
        float qs = FDIV(FMUL(.622f, e), FSUB(FMUL(100.f, p[k]), e));
        if (qs <= 1.E-08f) qs = 1.E-08f;
        qes[k] = qs;
        if (q[k] > qs) q[k] = qs;
        tv[k] = FADD(t[k], FMUL(FMUL(.608f, q[k]), t[k]));
    }
    if (itest != 2) {
        // Z(1)=max(0.,Z1)-(ALOG(P(1))-ALOG(PSUR))*287.*TV(1)/9.81
        z[1] = FSUB(gd_max(0.f, z1),
                    FDIV(FMUL(FMUL(FSUB(gfk_log(p[1]), gfk_log(psur)), 287.f),
                              tv[1]), 9.81f));
        for (int k = 2; k <= ktf; ++k) {
            float tvbar = FADD(FMUL(.5f, tv[k]), FMUL(.5f, tv[k - 1]));
            z[k] = FSUB(z[k - 1],
                        FDIV(FMUL(FMUL(FSUB(gfk_log(p[k]), gfk_log(p[k - 1])),
                                       287.f), tvbar), 9.81f));
        }
    } else {
        for (int k = 1; k <= ktf; ++k) {
            // z=(he-1004.*t-2.5e6*q)/9.81
            float zz = FDIV(FSUB(FSUB(he[k], FMUL(1004.f, t[k])),
                                 FMUL(2.5e6f, q[k])), 9.81f);
            z[k] = gd_max(1.e-3f, zz);
        }
    }
    for (int k = 1; k <= ktf; ++k) {
        if (itest == 0)
            he[k] = FADD(FADD(FMUL(9.81f, z[k]), FMUL(1004.f, t[k])),
                         FMUL(2.5E06f, q[k]));
        hes[k] = FADD(FADD(FMUL(9.81f, z[k]), FMUL(1004.f, t[k])),
                      FMUL(2.5E06f, qes[k]));
        if (he[k] >= hes[k]) he[k] = hes[k];
    }
}

// cup_env_clev (2483-2575)
__device__ void gd_cup_env_clev(GdCol t, GdCol qes, GdCol q, GdCol he,
                                GdCol hes, GdCol z, GdCol p, GdCol qes_cup,
                                GdCol q_cup, GdCol he_cup, GdCol hes_cup,
                                GdCol z_cup, GdCol p_cup, GdCol gamma_cup,
                                GdCol t_cup, float psur, int ierr, float z1,
                                float xl, float rv, float cp, int ktf)
{
    if (ierr != 0) return;
    float xlcp = FDIV(xl, cp);
    for (int k = 2; k <= ktf; ++k) {
        qes_cup[k] = FMUL(.5f, FADD(qes[k - 1], qes[k]));
        q_cup[k] = FMUL(.5f, FADD(q[k - 1], q[k]));
        hes_cup[k] = FMUL(.5f, FADD(hes[k - 1], hes[k]));
        he_cup[k] = FMUL(.5f, FADD(he[k - 1], he[k]));
        if (he_cup[k] > hes_cup[k]) he_cup[k] = hes_cup[k];
        z_cup[k] = FMUL(.5f, FADD(z[k - 1], z[k]));
        p_cup[k] = FMUL(.5f, FADD(p[k - 1], p[k]));
        t_cup[k] = FMUL(.5f, FADD(t[k - 1], t[k]));
        // (xl/cp)*(xl/(rv*t_cup*t_cup))*qes_cup
        gamma_cup[k] = FMUL(FMUL(xlcp, FDIV(xl, FMUL(FMUL(rv, t_cup[k]),
                                                     t_cup[k]))), qes_cup[k]);
    }
    qes_cup[1] = qes[1];
    q_cup[1] = q[1];
    hes_cup[1] = hes[1];
    he_cup[1] = he[1];
    z_cup[1] = FMUL(.5f, FADD(z[1], z1));
    p_cup[1] = FMUL(.5f, FADD(p[1], psur));
    t_cup[1] = t[1];
    gamma_cup[1] = FMUL(FMUL(xlcp, FDIV(xl, FMUL(FMUL(rv, t_cup[1]),
                                                 t_cup[1]))), qes_cup[1]);
}

// cup_MAXIMI (3336-3394)
__device__ int gd_maximi(GdCol a, int ks, int ke, int ierr)
{
    int maxx = ks;
    if (ierr == 0) {
        float x = a[ks];
        for (int k = ks; k <= ke; ++k) {
            float xar = a[k];
            if (xar >= x) { x = xar; maxx = k; }
        }
    }
    return maxx;
}

// cup_minimi (3397-3451)
__device__ int gd_minimi(GdCol a, int ks, int kend, int ierr)
{
    int kt = ks;
    if (ierr == 0) {
        float x = a[ks];
        int kstop = (ks + 1 > kend) ? ks + 1 : kend;
        for (int k = ks + 1; k <= kstop; ++k)
            if (a[k] < x) { x = a[k]; kt = k; }
    }
    return kt;
}

// cup_kbcon (3084-3162); k22 and ierr are inout.
__device__ int gd_cup_kbcon(float cap_inc, int iloop, int& k22, GdCol he_cup,
                            GdCol hes_cup, int& ierr, int kbmax, GdCol p_cup,
                            float cap_max)
{
    int kbcon = 1;
    if (ierr != 0) return kbcon;
    kbcon = k22;
    for (;;) {
        // label 32
        while (he_cup[k22] < hes_cup[kbcon]) {
            // label 31
            kbcon = kbcon + 1;
            if (kbcon > kbmax + 2) {
                if (iloop < 4) ierr = 3;
                return kbcon;
            }
        }
        if (kbcon - k22 == 1) return kbcon;
        float pbcdif = FADD(-p_cup[kbcon], p_cup[k22]);
        float plus = gd_max(25.f, FSUB(cap_max, FMUL((float)(iloop - 1), cap_inc)));
        if (iloop == 4) plus = cap_max;
        if (pbcdif > plus) {
            k22 = k22 + 1;
            kbcon = k22;
            continue;
        }
        return kbcon;
    }
}

// cup_up_he (3810-3908)
__device__ void gd_cup_up_he(int k22, float& hkb, GdCol z_cup, GdCol cd,
                             float entr, GdCol he_cup, GdCol hc, int kbcon,
                             int ierr, GdCol dby, GdCol he, GdCol hes_cup,
                             int ktf)
{
    if (ierr != 0) return;
    hkb = he_cup[k22];
    for (int k = 1; k <= k22; ++k) { hc[k] = he_cup[k]; dby[k] = 0.f; }
    for (int k = k22; k <= kbcon - 1; ++k) { hc[k] = hkb; dby[k] = 0.f; }
    hc[kbcon] = hkb;
    dby[kbcon] = FSUB(hkb, hes_cup[kbcon]);
    for (int k = 2; k <= ktf; ++k) {
        if (k > kbcon) {
            float dz = FSUB(z_cup[k], z_cup[k - 1]);
            float hcd = FMUL(FMUL(.5f, cd[k]), dz);
            float num = FADD(FMUL(hc[k - 1], FSUB(1.f, hcd)),
                             FMUL(FMUL(entr, dz), he[k - 1]));
            float den = FSUB(FADD(1.f, FMUL(entr, dz)), hcd);
            hc[k] = FDIV(num, den);
            dby[k] = FSUB(hc[k], hes_cup[k]);
        }
    }
}

// cup_ktop (3264-3333)
__device__ int gd_cup_ktop(int ilo, GdCol dby, int kbcon, int& ierr, int ktf)
{
    int ktop = 1;
    if (ierr != 0) return ktop;
    for (int k = kbcon + 1; k <= ktf - 1; ++k) {
        if (dby[k] <= 0.f) {
            ktop = k - 1;
            for (int kk = ktop + 1; kk <= ktf; ++kk) dby[kk] = 0.f;
            return ktop;
        }
    }
    if (ilo == 1) ierr = 5;
    return ktop;
}

// cup_up_nms (4063-4141)
__device__ void gd_cup_up_nms(GdCol zu, GdCol z_cup, float entr, GdCol cd,
                              int kbcon, int ktop, int ierr, int k22, int ktf)
{
    for (int k = 1; k <= ktf; ++k) zu[k] = 0.f;
    if (ierr != 0) return;
    for (int k = k22; k <= kbcon; ++k) zu[k] = 1.f;
    for (int k = kbcon + 1; k <= ktop; ++k) {
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        zu[k] = FMUL(zu[k - 1], FADD(1.f, FMUL(FSUB(entr, cd[k]), dz)));
    }
}

// cup_dd_nms (1826-1933)
__device__ void gd_cup_dd_nms(GdCol zd, GdCol z_cup, GdCol cdd, float entr,
                              int jmin, int ierr, int itest, int kdet,
                              float z1, int ktf)
{
    const float perc = .03f;
    for (int k = 1; k <= ktf; ++k) {
        zd[k] = 0.f;
        if (itest == 0) cdd[k] = 0.f;
    }
    const float a = GDK_A_DD;
    if (ierr != 0) return;
    zd[jmin] = 1.f;
    for (int ki = jmin - 1; ki >= 1; --ki) {
        float dz = FSUB(z_cup[ki + 1], z_cup[ki]);
        if (ki <= kdet && itest == 0) {
            float b = FMUL(perc, FSUB(z_cup[kdet], z1));
            float num = FADD(FMUL(a, FSUB(z_cup[ki], z1)), b);
            float den = FADD(FMUL(a, FSUB(z_cup[ki + 1], z1)), b);
            cdd[ki] = FADD(entr, FDIV(FSUB(1.f, FDIV(num, den)), dz));
        }
        zd[ki] = FMUL(zd[ki + 1], FADD(1.f, FMUL(FSUB(entr, cdd[ki]), dz)));
    }
}

// cup_dd_he (1600-1693)
__device__ void gd_cup_dd_he(GdCol hes_cup, GdCol hcd, GdCol z_cup, GdCol cdd,
                             float entr, int jmin, int ierr, GdCol he,
                             int ktf)
{
    if (ierr != 0) return;
    for (int k = 2; k <= ktf; ++k) hcd[k] = hes_cup[k];
    hcd[jmin] = hes_cup[jmin];
    for (int ki = jmin - 1; ki >= 1; --ki) {
        float dz = FSUB(z_cup[ki + 1], z_cup[ki]);
        float hc = FMUL(FMUL(.5f, cdd[ki]), dz);
        hcd[ki] = FDIV(FADD(FMUL(hcd[ki + 1], FSUB(1.f, hc)),
                            FMUL(FMUL(entr, dz), he[ki])),
                       FSUB(FADD(1.f, FMUL(entr, dz)), hc));
    }
}

// cup_dd_moisture (1696-1823), iloop 1 (the forced state; may set ierr 7)
__device__ void gd_cup_dd_moisture(GdCol zd, GdCol hcd, GdCol hes_cup,
                                   GdCol qcd, GdCol qes_cup, GdCol pwd,
                                   GdCol q_cup, GdCol z_cup, GdCol cdd,
                                   float entr, int jmin, int& ierr,
                                   GdCol gamma_cup, float& pwev, GdCol qrcd,
                                   GdCol q, float xl, int ktf)
{
    float bu = 0.f;
    pwev = 0.f;
    for (int k = 1; k <= ktf; ++k) { qcd[k] = 0.f; qrcd[k] = 0.f; pwd[k] = 0.f; }
    if (ierr != 0) return;
    int k = jmin;
    float dz = FSUB(z_cup[k + 1], z_cup[k]);
    qcd[k] = q_cup[k];
    qrcd[k] = qes_cup[k];
    pwd[jmin] = gd_min(0.f, FSUB(qcd[k], qrcd[k]));
    pwev = FADD(pwev, pwd[jmin]);
    qcd[k] = qes_cup[k];
    float dh = FSUB(hcd[k], hes_cup[k]);
    bu = FMUL(dz, dh);
    float rxl = FDIV(1.f, xl);
    for (int ki = jmin - 1; ki >= 1; --ki) {
        dz = FSUB(z_cup[ki + 1], z_cup[ki]);
        float hc = FMUL(FMUL(.5f, cdd[ki]), dz);
        qcd[ki] = FDIV(FADD(FMUL(qcd[ki + 1], FSUB(1.f, hc)),
                            FMUL(FMUL(entr, dz), q[ki])),
                       FSUB(FADD(1.f, FMUL(entr, dz)), hc));
        dh = FSUB(hcd[ki], hes_cup[ki]);
        bu = FADD(bu, FMUL(dz, dh));
        qrcd[ki] = FADD(qes_cup[ki],
                        FMUL(FMUL(rxl, FDIV(gamma_cup[ki],
                                            FADD(1.f, gamma_cup[ki]))), dh));
        float dqeva = FSUB(qcd[ki], qrcd[ki]);
        if (dqeva > 0.f) dqeva = 0.f;
        pwd[ki] = FMUL(zd[ki], dqeva);
        qcd[ki] = qrcd[ki];
        pwev = FADD(pwev, pwd[ki]);
    }
    if (pwev == 0.f) ierr = 7;
    if (bu >= 0.f) ierr = 7;
}

// cup_up_moisture (3911-4060).  pw/pwav only where the caller needs them.
__device__ void gd_cup_up_moisture(int ierr, GdCol z_cup, GdCol qc, GdCol qrc,
                                   GdCol pw, float* pwav, int kbcon, int ktop,
                                   GdCol cd, GdCol dby, float mentr_rate,
                                   GdCol q, GdCol gamma_cup, GdCol zu,
                                   GdCol qes_cup, int k22, GdCol qe_cup,
                                   float xl, int ktf)
{
    float c0 = .002f;
    if (mentr_rate > 0.f) {
        float radius = GDK_RADIUS_UP;
        if (radius < 900.f) c0 = 0.f;
    }
    float pwav_l = 0.f;
    for (int k = 1; k <= ktf; ++k) {
        pw[k] = 0.f;
        if (ierr == 0) qc[k] = qes_cup[k];
        qrc[k] = 0.f;
    }
    if (ierr == 0) {
        for (int k = k22; k <= kbcon - 1; ++k) qc[k] = qe_cup[k22];
        float rxl = FDIV(1.f, xl);
        for (int k = 2; k <= ktf; ++k) {
            if (k < kbcon || k > ktop) continue;
            float dz = FSUB(z_cup[k], z_cup[k - 1]);
            float hc = FMUL(FMUL(.5f, cd[k]), dz);
            qc[k] = FDIV(FADD(FMUL(qc[k - 1], FSUB(1.f, hc)),
                              FMUL(FMUL(mentr_rate, dz), q[k - 1])),
                         FSUB(FADD(1.f, FMUL(mentr_rate, dz)), hc));
            float qrch = FADD(qes_cup[k],
                              FMUL(FMUL(rxl, FDIV(gamma_cup[k],
                                                  FADD(1.f, gamma_cup[k]))),
                                   dby[k]));
            float r = FDIV(FSUB(qc[k], qrch),
                           FADD(1.f, FMUL(FMUL(c0, dz), zu[k])));
            if (r < 0.f) r = 0.f;
            qrc[k] = r;
            pw[k] = FMUL(FMUL(FMUL(c0, dz), r), zu[k]);
            qc[k] = FADD(r, qrch);
            pwav_l = FADD(pwav_l, pw[k]);
        }
    }
    if (pwav) *pwav = pwav_l;
}

// cup_up_aa0 (3733-3807)
__device__ float gd_cup_up_aa0(GdCol z, GdCol zu, GdCol dby, GdCol gamma_cup,
                               GdCol t_cup, int kbcon, int ktop, int ierr,
                               int ktf)
{
    float aa0 = 0.f;
    if (ierr != 0) return aa0;
    for (int k = 2; k <= ktf; ++k) {
        if (k <= kbcon || k > ktop) continue;
        float dz = FSUB(z[k], z[k - 1]);
        // zu*DZ*(9.81/(1004.*T_cup))*DBY(K-1)/(1.+GAMMA_CUP)
        float da = FDIV(FMUL(FMUL(FMUL(zu[k], dz),
                                  FDIV(9.81f, FMUL(1004.f, t_cup[k]))),
                             dby[k - 1]),
                        FADD(1.f, gamma_cup[k]));
        if (k == ktop && da <= 0.f) continue;
        aa0 = FADD(aa0, da);
        if (aa0 < 0.f) aa0 = 0.f;
    }
    return aa0;
}

// cup_dd_edt (1479-1597) -> edtc[0..2]
__device__ void gd_cup_dd_edt(int ierr, GdCol us, GdCol vs, GdCol z, int ktop,
                              int kbcon, GdCol p, float pwav, float pwev,
                              float edtmax, float edtmin, float* edtc, int ktf)
{
    if (ierr != 0) return;
    float vws = 0.f, sdp = 0.f, vshear = 0.f;
    int ktopc = (ktop < ktf) ? ktop : ktf;
    for (int kk = 1; kk <= ktf - 1; ++kk) {
        if (kk <= ktopc && kk >= kbcon) {
            float dzz = FSUB(z[kk + 1], z[kk]);
            float a = fabsf(FDIV(FSUB(us[kk + 1], us[kk]), dzz));
            float b = fabsf(FDIV(FSUB(vs[kk + 1], vs[kk]), dzz));
            vws = FADD(vws, FMUL(FADD(a, b), FSUB(p[kk], p[kk + 1])));
            sdp = FSUB(FADD(sdp, p[kk]), p[kk + 1]);
        }
        if (kk == ktf - 1) vshear = FDIV(FMUL(1.e3f, vws), sdp);
    }
    float v2 = FMUL(vshear, vshear);
    float v3 = FMUL(v2, vshear);
    float pef = FSUB(FADD(FSUB(1.591f, FMUL(.639f, vshear)), FMUL(.0953f, v2)),
                     FMUL(.00496f, v3));
    if (pef > edtmax) pef = edtmax;
    if (pef < edtmin) pef = edtmin;
    float zkbc = FMUL(z[kbcon], 3.281e-3f);
    float prezk = .02f;
    if (zkbc > 3.f) {
        float r = FSUB(4.2772E-4f, FMUL(zkbc, 5.44E-6f));
        r = FADD(-1.2569798E-2f, FMUL(zkbc, r));
        r = FADD(.162179896f, FMUL(zkbc, r));
        r = FADD(-.70034167f, FMUL(zkbc, r));
        prezk = FADD(.96729352f, FMUL(zkbc, r));
    }
    if (zkbc > 25.f) prezk = 2.4f;
    float pefb = FDIV(1.f, FADD(1.f, prezk));
    if (pefb > edtmax) pefb = edtmax;
    if (pefb < edtmin) pefb = edtmin;
    float edt = FSUB(1.f, FMUL(.5f, FADD(pefb, pef)));
    float einc = FMUL(.2f, edt);
    for (int k = 1; k <= 3; ++k)
        edtc[k - 1] = FADD(edt, FMUL((float)(k - 2), einc));
    for (int k = 1; k <= 3; ++k) {
        float e = -FDIV(FMUL(edtc[k - 1], pwav), pwev);
        if (e > edtmax) e = edtmax;
        if (e < edtmin) e = edtmin;
        edtc[k - 1] = e;
    }
}

// cup_dellabot (1936-2020), the della(1) slot only (no flux arrays).
__device__ float gd_cup_dellabot(GdCol he_cup, int ierr, GdCol z_cup,
                                 GdCol p_cup, GdCol hcd, float edt, GdCol zd,
                                 GdCol cdd, GdCol he, float mentrd_rate,
                                 float g)
{
    if (ierr != 0) return 0.f;
    float dz = FSUB(z_cup[2], z_cup[1]);
    float dp = FMUL(100.f, FSUB(p_cup[1], p_cup[2]));
    float detdo1 = FMUL(FMUL(FMUL(edt, zd[2]), cdd[1]), dz);
    float detdo2 = FMUL(edt, zd[1]);
    float entdo = FMUL(FMUL(FMUL(edt, zd[2]), mentrd_rate), dz);
    float subin = -FMUL(edt, zd[2]);
    float s = FADD(FMUL(FMUL(detdo1, .5f), FADD(hcd[1], hcd[2])),
                   FMUL(detdo2, hcd[1]));
    s = FADD(s, FMUL(subin, he_cup[2]));
    s = FSUB(s, FMUL(entdo, he[1]));
    return FDIV(FMUL(s, g), dp);
}

// cup_dellas (2023-2208), della(2..ktf).  della(1) is the caller's.
__device__ void gd_cup_dellas(int ierr, GdCol z_cup, GdCol p_cup, GdCol hcd,
                              float edt, GdCol zd, GdCol cdd, GdCol he,
                              GdCol della, float mentrd_rate, GdCol zu,
                              float g, GdCol cd, GdCol hc, int ktop, int k22,
                              int kbcon, float mentr_rate, int jmin,
                              GdCol he_cup, int kdet, int kpbl, int ktf)
{
    for (int k = 2; k <= ktf; ++k) della[k] = 0.f;
    if (ierr != 0) return;
    for (int k = 2; k <= ktf - 1; ++k) {
        if (k > ktop) continue;
        float dz = FSUB(z_cup[k + 1], z_cup[k]);
        float detdo = FMUL(FMUL(FMUL(edt, cdd[k]), dz), zd[k + 1]);
        float entdo = FMUL(FMUL(FMUL(edt, mentrd_rate), dz), zd[k + 1]);
        float subin = FSUB(zu[k + 1], FMUL(zd[k + 1], edt));
        float entup = 0.f, detup = 0.f;
        if (k >= kbcon && k < ktop) {
            entup = FMUL(FMUL(mentr_rate, dz), zu[k]);
            detup = FMUL(FMUL(cd[k + 1], dz), zu[k]);
        }
        float subdown = FSUB(zu[k], FMUL(zd[k], edt));
        float entdoj = 0.f, entupk = 0.f, detupk = 0.f;
        if (k == jmin) entdoj = FMUL(edt, zd[k]);
        if (k == k22 - 1) entupk = zu[kpbl];
        if (k > kdet) detdo = 0.f;
        if (k == ktop) { detupk = zu[ktop]; subin = 0.f; }
        if (k < kbcon) detup = 0.f;
        float dp = FMUL(100.f, FSUB(p_cup[k - 1], p_cup[k]));
        float s = FSUB(FMUL(subin, he_cup[k + 1]), FMUL(subdown, he_cup[k]));
        s = FADD(s, FMUL(FMUL(detup, .5f), FADD(hc[k + 1], hc[k])));
        s = FADD(s, FMUL(FMUL(detdo, .5f), FADD(hcd[k + 1], hcd[k])));
        s = FSUB(s, FMUL(entup, he[k]));
        s = FSUB(s, FMUL(entdo, he[k]));
        s = FSUB(s, FMUL(entupk, he_cup[k22]));
        s = FSUB(s, FMUL(entdoj, he_cup[jmin]));
        s = FADD(s, FMUL(detupk, hc[ktop]));
        della[k] = FDIV(FMUL(s, g), dp);
    }
}

// massflx_stats (4247-4476) for one column.  Returns x_ave of closures in
// xave[16], x_ave_cap in xcap[3]; *xt_ave, *xt_std; *ok = xt_std > 0 gate.
__device__ void gd_massflx_stats(GdEns xf, int ierr, float* xave, float* xcap,
                                 float* xt_ave_o, float* xt_std_o)
{
    float x_ave[16], x_ave_cap[3];
    for (int k = 0; k < 3; ++k) x_ave_cap[k] = 0.f;
    for (int k = 0; k < 16; ++k) x_ave[k] = 0.f;
    float xt_ave = 0.f, xt_std = 0.f;
    if (ierr == 0) {
        for (int kk = 1; kk <= 9; ++kk)
            for (int k = 1; k <= 16; ++k)
                x_ave[k - 1] = FADD(x_ave[k - 1], xf(16 * (kk - 1) + k));
        for (int iedt = 1; iedt <= 3; ++iedt)
            for (int k = 1; k <= 3; ++k)
                for (int kk = 1; kk <= 16; ++kk)
                    x_ave_cap[k - 1] = FADD(x_ave_cap[k - 1],
                        xf(16 * (k - 1) + (iedt - 1) * 48 + kk));
        for (int k = 0; k < 3; ++k) x_ave_cap[k] = FDIV(x_ave_cap[k], 48.f);
        for (int k = 0; k < 16; ++k) x_ave[k] = FDIV(x_ave[k], 9.f);
        for (int k = 0; k < 16; ++k) xt_ave = FADD(xt_ave, x_ave[k]);
        xt_ave = FDIV(xt_ave, 16.f);
        if (xt_ave > 0.f) {
            for (int k = 0; k < 16; ++k) {
                float d = FSUB(x_ave[k], xt_ave);
                xt_std = FADD(xt_std, FMUL(d, d));
            }
        }
        if (xt_std > 0.f) {
            xt_std = FDIV(xt_std, 16.f);
            float a3 = gd_max(1.e-6f, xt_std);
            xt_std = FSQRT(a3);
        }
    }
    for (int k = 0; k < 16; ++k) xave[k] = x_ave[k];
    for (int k = 0; k < 3; ++k) xcap[k] = x_ave_cap[k];
    *xt_ave_o = xt_ave;
    *xt_std_o = xt_std;
}

__device__ __forceinline__ float gd_third(float a, float b, float c)
{   // .333*(a+b+c)
    return FMUL(.333f, FADD(FADD(a, b), c));
}

// ---------------------------------------------------------------------------
// The driver: GRELLDRV (9-384) and CUP_enss (387-1396) for one column.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64)
gd_grelldrv(const float* __restrict__ lvin, const float* __restrict__ scin,
            const int* __restrict__ iin, float* __restrict__ lev,
            float* __restrict__ sca, int* __restrict__ isc,
            float* __restrict__ ens_out, float* __restrict__ ws,
            const float xlv, const float cp, const float g, const float rv,
            const int wrf_closure_count, const int ncol, const int nz)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int L = blockDim.x;
    const int kp = nz + 2;
    float* wbase = ws + (size_t)blockIdx.x * GD_WS_FLOATS(nz) * L + threadIdx.x;
    int slot = 0;
#define GD_SLOT(name) GdCol name = {wbase + (size_t)(slot++) * kp * L, L}
    GD_SLOT(t2d); GD_SLOT(q2d); GD_SLOT(tn); GD_SLOT(qo); GD_SLOT(po);
    GD_SLOT(us); GD_SLOT(vs); GD_SLOT(omeg); GD_SLOT(cupclw); GD_SLOT(outt);
    GD_SLOT(outq); GD_SLOT(outqc); GD_SLOT(tv);
    // unforced state, later reused (aliased) for the mbdt-perturbed state
    GD_SLOT(z); GD_SLOT(qes); GD_SLOT(he); GD_SLOT(hes);
    GD_SLOT(qes_cup); GD_SLOT(q_cup); GD_SLOT(he_cup); GD_SLOT(hes_cup);
    GD_SLOT(z_cup); GD_SLOT(p_cup); GD_SLOT(gamma_cup); GD_SLOT(t_cup);
    GD_SLOT(dby); GD_SLOT(hc); GD_SLOT(zu); GD_SLOT(qc); GD_SLOT(qrc);
    GD_SLOT(pwscr); GD_SLOT(zd);
    // forced state
    GD_SLOT(zo); GD_SLOT(qeso); GD_SLOT(heo); GD_SLOT(heso);
    GD_SLOT(qeso_cup); GD_SLOT(qo_cup); GD_SLOT(heo_cup); GD_SLOT(heso_cup);
    GD_SLOT(zo_cup); GD_SLOT(po_cup); GD_SLOT(gammao_cup); GD_SLOT(tn_cup);
    GD_SLOT(dbyo); GD_SLOT(hco); GD_SLOT(zuo); GD_SLOT(zdo); GD_SLOT(hcdo);
    GD_SLOT(qcdo); GD_SLOT(qrcdo); GD_SLOT(pwdo); GD_SLOT(qco); GD_SLOT(qrco);
    GD_SLOT(pwo);
    GD_SLOT(cd); GD_SLOT(cdd); GD_SLOT(scr1); GD_SLOT(dellah); GD_SLOT(dellaq);
    GD_SLOT(dellat); GD_SLOT(dellaqc); GD_SLOT(xt); GD_SLOT(xq);
    GD_SLOT(dt_e0); GD_SLOT(dt_e1); GD_SLOT(dt_e2);
    GD_SLOT(dq_e0); GD_SLOT(dq_e1); GD_SLOT(dq_e2);
    GD_SLOT(dc_e0); GD_SLOT(dc_e1); GD_SLOT(dc_e2);
    GD_SLOT(pw_e0); GD_SLOT(pw_e1); GD_SLOT(pw_e2);
#undef GD_SLOT
    // slot == GD_NSLOT here (76); tests/test_gd_cu93.py counts the GD_SLOT lines
    GdEns xf_ens = {wbase + (size_t)GD_NSLOT * kp * L, L};
    GdEns pr_ens = {wbase + ((size_t)GD_NSLOT * kp + GD_ENS) * L, L};
    GdCol dellat_ens[3] = {dt_e0, dt_e1, dt_e2};
    GdCol dellaq_ens[3] = {dq_e0, dq_e1, dq_e2};
    GdCol dellaqc_ens[3] = {dc_e0, dc_e1, dc_e2};
    GdCol pwo_ens[3] = {pw_e0, pw_e1, pw_e2};
    // the perturbed state lives in the unforced state's slots
    GdCol xz = z, xqes = qes, xhe = he, xhes = hes;
    GdCol xqes_cup = qes_cup, xq_cup = q_cup, xhe_cup = he_cup,
          xhes_cup = hes_cup, xz_cup = z_cup, xt_cup = t_cup;
    GdCol xdby = dby, xhc = hc, xzu = zu;

    const float* in = lvin + (size_t)col * GDI_NLEV * nz;
    const float* sc = scin + (size_t)col * GDS_NSCA;
    const int band = iin[(size_t)col * GDN_NINT + GDN_BAND];
    const int f_qi = iin[(size_t)col * GDN_NINT + GDN_FQI];
#define IN(f, k) in[(f) * nz + (k) - 1]
    const int ktf = nz;
    const float dt = sc[GDS_DT];
    const float tcrit = 258.f;

    // Every slot starts at zero so a word WRF leaves undefined but never
    // reads (dellaqc(ktf), the kte face) has a defined value here too.
    for (int s = 0; s < GD_NSLOT; ++s)
        for (int k = 0; k < kp; ++k)
            wbase[((size_t)s * kp + k) * L] = 0.f;

    // ---- GRELLDRV column preparation (203-292) ----
    float pret = 0.f;
    float psur = FMUL(sc[GDS_P8W1], .01f);
    float ter11 = sc[GDS_HT];
    float mconv = 0.f;
    for (int k = 1; k <= ktf; ++k) {
        // RTHFTEN=(RTHFTEN+RTHRATEN+RTHBLTEN)*pi ; RQVFTEN=RQVFTEN+RQVBLTEN
        float rthf = FMUL(FADD(FADD(IN(GDI_RTHFTEN, k), IN(GDI_RTHRATEN, k)),
                               IN(GDI_RTHBLTEN, k)), IN(GDI_PI, k));
        float rqvf = FADD(IN(GDI_RQVFTEN, k), IN(GDI_RQVBLTEN, k));
        po[k] = FMUL(IN(GDI_P, k), .01f);
        us[k] = IN(GDI_U, k);
        vs[k] = IN(GDI_V, k);
        t2d[k] = IN(GDI_T, k);
        q2d[k] = IN(GDI_Q, k);
        omeg[k] = -FMUL(FMUL(g, IN(GDI_RHO, k)), IN(GDI_W, k));
        tn[k] = FADD(t2d[k], FMUL(rthf, dt));
        if (tn[k] < 200.f) tn[k] = t2d[k];
        qo[k] = FADD(q2d[k], FMUL(rqvf, dt));
        if (q2d[k] < 1.E-08f) q2d[k] = 1.E-08f;
        if (qo[k] < 1.E-08f) qo[k] = 1.E-08f;
    }
    for (int k = 1; k <= ktf - 1; ++k) {
        float dq = FSUB(q2d[k + 1], q2d[k]);
        mconv = FADD(mconv, FDIV(FMUL(omeg[k], dq), g));
    }
    if (mconv < 0.f) mconv = 0.f;

    // ---- CUP_enss (387-1396) ----
    float closure_n = 16.f;
    float xland1 = 1.f;
    if (sc[GDS_XLAND] > 1.5f) xland1 = 0.f;
    const float cap_max_increment = 25.f;
    const float entr_rate = GDK_ENTR;
    const float mentrd_rate = 0.f;
    const float mentr_rate = entr_rate;
    for (int k = 1; k <= ktf; ++k) {
        cupclw[k] = 0.f;
        cd[k] = GDK_CD0;
        cdd[k] = 0.f;
    }
    const float edtmax = .8f, edtmin = .2f, depth_min = 500.f;
    int kbmax = 1;
    float aa0 = 0.f, aa1 = 0.f;
    int kstabm = ktf - 1;
    int ierr = 0, ierr2 = 0, ierr3 = 0;
    float cap_max = 75.f;
    if (sc[GDS_GSW] < 1.f) cap_max = 25.f;
    const float zkbmax = 4000.f, zcutdown = 3000.f, z_detr = 1250.f;
    float mbdt_ens[3];
    for (int n = 1; n <= 3; ++n)
        mbdt_ens[n - 1] = FADD(FMUL(FMUL(FSUB((float)n, 3.f), dt), 1.e-3f),
                               FMUL(dt, 5.E-03f));
    for (int n = 1; n <= GD_ENS; ++n) { xf_ens(n) = 0.f; pr_ens(n) = 0.f; }

    gd_cup_env(z, qes, he, hes, t2d, q2d, po, tv, ter11, psur, ierr, tcrit,
               0, xlv, cp, ktf);
    gd_cup_env(zo, qeso, heo, heso, tn, qo, po, tv, ter11, psur, ierr, tcrit,
               0, xlv, cp, ktf);
    gd_cup_env_clev(t2d, qes, q2d, he, hes, z, po, qes_cup, q_cup, he_cup,
                    hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur, ierr,
                    ter11, xlv, rv, cp, ktf);
    gd_cup_env_clev(tn, qeso, qo, heo, heso, zo, po, qeso_cup, qo_cup,
                    heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup,
                    psur, ierr, ter11, xlv, rv, cp, ktf);
    int kdet = 0;
    if (ierr == 0) {
        for (int k = 1; k <= ktf - 2; ++k)
            if (zo_cup[k] > FADD(zkbmax, ter11)) { kbmax = k; break; }
        for (int k = 1; k <= ktf; ++k)
            if (zo_cup[k] > FADD(z_detr, ter11)) { kdet = k; break; }
    }
    int k22 = gd_maximi(heo_cup, 3, kbmax, ierr);
    if (ierr == 0 && k22 >= kbmax) ierr = 2;
    int kbcon = gd_cup_kbcon(cap_max_increment, 1, k22, heo_cup, heso_cup,
                             ierr, kbmax, po_cup, cap_max);
    int kstabi = gd_minimi(heso_cup, kbcon, kstabm, ierr);
    if (ierr == 0 && kstabm - 1 > kstabi) {
        for (int k = kstabi; k <= kstabm - 1; ++k) {
            cd[k] = FADD(cd[k - 1], GDK_CD15);
            if (cd[k] > GDK_CD10) cd[k] = GDK_CD10;
        }
    }
    float hkb = 0.f, hkbo = 0.f;
    gd_cup_up_he(k22, hkb, z_cup, cd, mentr_rate, he_cup, hc, kbcon, ierr,
                 dby, he, hes_cup, ktf);
    gd_cup_up_he(k22, hkbo, zo_cup, cd, mentr_rate, heo_cup, hco, kbcon, ierr,
                 dbyo, heo, heso_cup, ktf);
    int ktop = gd_cup_ktop(1, dbyo, kbcon, ierr, ktf);
    int kzdown = 0;
    if (ierr == 0) {
        float zktop = FMUL(FSUB(zo_cup[ktop], ter11), .6f);
        zktop = gd_min(FADD(zktop, ter11), FADD(zcutdown, ter11));
        for (int k = 1; k <= ktf + 1; ++k)
            if (zo_cup[k] > zktop) { kzdown = k; break; }
    }
    int jmin = gd_minimi(heso_cup, k22, kzdown, ierr);
    if (ierr == 0) {
        int jmini = jmin;
        bool keep_going = true;
        while (keep_going) {
            keep_going = false;
            if (jmini - 1 < kdet) kdet = jmini - 1;
            if (jmini >= ktop - 1) jmini = ktop - 2;
            int ki = jmini;
            hcdo[ki] = heso_cup[ki];
            float dh = 0.f;
            for (int k = ki - 1; k >= 1; --k) {
                hcdo[k] = heso_cup[jmini];
                float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
                dh = FADD(dh, FMUL(dz, FSUB(hcdo[k], heso_cup[k])));
                if (dh > 0.f) {
                    jmini = jmini - 1;
                    if (jmini > 3) {
                        keep_going = true;
                    } else {
                        ierr = 9;
                        break;
                    }
                }
            }
        }
        jmin = jmini;
        if (jmini <= 3) ierr = 4;
    }
    if (ierr == 0 && FADD(-zo_cup[kbcon], zo_cup[ktop]) < depth_min) ierr = 6;

    gd_cup_up_nms(zu, z_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
    gd_cup_up_nms(zuo, zo_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
    gd_cup_dd_nms(zd, z_cup, cdd, mentrd_rate, jmin, ierr, 0, kdet, ter11, ktf);
    gd_cup_dd_nms(zdo, zo_cup, cdd, mentrd_rate, jmin, ierr, 1, kdet, ter11, ktf);
    gd_cup_dd_he(heso_cup, hcdo, zo_cup, cdd, mentrd_rate, jmin, ierr, heo, ktf);
    float pwevo = 0.f;
    gd_cup_dd_moisture(zdo, hcdo, heso_cup, qcdo, qeso_cup, pwdo, qo_cup,
                       zo_cup, cdd, mentrd_rate, jmin, ierr, gammao_cup,
                       pwevo, qrcdo, qo, xlv, ktf);
    gd_cup_up_moisture(ierr, z_cup, qc, qrc, pwscr, nullptr, kbcon, ktop, cd,
                       dby, mentr_rate, q2d, gamma_cup, zu, qes_cup, k22,
                       q_cup, xlv, ktf);
    for (int k = 1; k <= ktf; ++k) cupclw[k] = qrc[k];
    float pwavo = 0.f;
    gd_cup_up_moisture(ierr, zo_cup, qco, qrco, pwo, &pwavo, kbcon, ktop, cd,
                       dbyo, mentr_rate, qo, gammao_cup, zuo, qeso_cup, k22,
                       qo_cup, xlv, ktf);
    aa0 = gd_cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf);
    aa1 = gd_cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, ktf);
    if (ierr == 0 && aa1 == 0.f) ierr = 17;
    float edtc[3] = {0.f, 0.f, 0.f};
    gd_cup_dd_edt(ierr, us, vs, zo, ktop, kbcon, po, pwavo, pwevo, edtmax,
                  edtmin, edtc, ktf);

    float edt = 0.f, edto = 0.f;
    float xaa0_ens[3];
    int k22x = 3, kbconx = 1;
    for (int iedt = 1; iedt <= 3; ++iedt) {
        if (ierr == 0) { edt = edtc[iedt - 1]; edto = edtc[iedt - 1]; }
        for (int k = 1; k <= ktf; ++k) {
            dellat_ens[iedt - 1][k] = 0.f;
            dellaq_ens[iedt - 1][k] = 0.f;
            dellaqc_ens[iedt - 1][k] = 0.f;
            pwo_ens[iedt - 1][k] = 0.f;
        }
        // della in the bottom layer, then everywhere else
        dellah[1] = gd_cup_dellabot(heo_cup, ierr, zo_cup, po, hcdo, edto,
                                    zdo, cdd, heo, mentrd_rate, g);
        dellaq[1] = gd_cup_dellabot(qo_cup, ierr, zo_cup, po, qrcdo, edto,
                                    zdo, cdd, qo, mentrd_rate, g);
        gd_cup_dellas(ierr, zo_cup, po_cup, hcdo, edto, zdo, cdd, heo, dellah,
                      mentrd_rate, zuo, g, cd, hco, ktop, k22, kbcon,
                      mentr_rate, jmin, heo_cup, kdet, k22, ktf);
        for (int k = 1; k <= ktf - 1; ++k) {
            scr1[k] = 0.f;
            dellaqc[k] = 0.f;
            if (ierr == 0) {
                scr1[k] = FSUB(qco[k], qrco[k]);
                if (k == ktop)
                    dellaqc[k] = FDIV(FMUL(FMUL(FMUL(.01f, zuo[ktop]),
                                                qrco[ktop]), 9.81f),
                                      FSUB(po_cup[k], po_cup[k + 1]));
                if (k < ktop && k > kbcon) {
                    float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
                    float a = FMUL(FMUL(FMUL(FMUL(GDK_C01G, cd[k]), dz),
                                        zuo[k]), .5f);
                    dellaqc[k] = FDIV(FMUL(a, FADD(qrco[k], qrco[k + 1])),
                                      FSUB(po_cup[k], po_cup[k + 1]));
                }
            }
        }
        gd_cup_dellas(ierr, zo_cup, po_cup, qrcdo, edto, zdo, cdd, qo, dellaq,
                      mentrd_rate, zuo, g, cd, scr1, ktop, k22, kbcon,
                      mentr_rate, jmin, qo_cup, kdet, k22, ktf);

        // the mbdt-perturbed environment
        const float mbdt = mbdt_ens[1];
        xaa0_ens[0] = 0.f; xaa0_ens[1] = 0.f; xaa0_ens[2] = 0.f;
        float rcp = FDIV(1.f, cp);
        for (int k = 1; k <= ktf; ++k) {
            dellat[k] = 0.f;
            if (ierr == 0) {
                xhe[k] = FADD(FMUL(dellah[k], mbdt), heo[k]);
                xq[k] = FADD(FMUL(dellaq[k], mbdt), qo[k]);
                dellat[k] = FMUL(rcp, FSUB(dellah[k], FMUL(xlv, dellaq[k])));
                xt[k] = FADD(FMUL(dellat[k], mbdt), tn[k]);
                if (xq[k] <= 0.f) xq[k] = 1.E-08f;
            }
        }
        if (ierr == 0) {
            xhe[ktf] = heo[ktf];
            xq[ktf] = qo[ktf];
            xt[ktf] = tn[ktf];
            if (xq[ktf] <= 0.f) xq[ktf] = 1.E-08f;
        }
        gd_cup_env(xz, xqes, xhe, xhes, xt, xq, po, tv, ter11, psur, ierr,
                   tcrit, 2, xlv, cp, ktf);
        // WRF hands po_cup and gamma_cup as this call's p_cup/gamma_cup
        // outputs: po_cup is rewritten with the same words, gamma_cup with
        // the perturbed state's (module_cu_gd.F:1175-1180).
        gd_cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup,
                        xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup,
                        psur, ierr, ter11, xlv, rv, cp, ktf);
        float xhkb = 0.f;
        gd_cup_up_he(k22, xhkb, xz_cup, cd, mentr_rate, xhe_cup, xhc, kbcon,
                     ierr, xdby, xhe, xhes_cup, ktf);
        gd_cup_up_nms(xzu, xz_cup, mentr_rate, cd, kbcon, ktop, ierr, k22, ktf);
        float xaa0 = gd_cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon,
                                   ktop, ierr, ktf);
        for (int nens = 1; nens <= 3; ++nens) {
            if (ierr != 0) continue;
            xaa0_ens[nens - 1] = xaa0;
            int nall = (iedt - 1) * 48 + (nens - 1) * 16;
            for (int k = 1; k <= ktf; ++k) {
                if (k > ktop) continue;
                for (int n3 = 1; n3 <= 16; ++n3) {
                    if (n3 == 7 || n3 == 8 || n3 == 9)
                        pr_ens(nall + n3) = FADD(pr_ens(nall + n3), pwo[k]);
                    else   // pr_ens+pwo(k)+edto*pwdo(k), left to right
                        pr_ens(nall + n3) = FADD(FADD(pr_ens(nall + n3), pwo[k]),
                                                 FMUL(edto, pwdo[k]));
                }
            }
            if (pr_ens(nall + 7) < 1.e-6f) {
                ierr = 18;
                for (int n3 = 1; n3 <= 16; ++n3) pr_ens(nall + n3) = 0.f;
            }
            for (int n3 = 1; n3 <= 16; ++n3)
                if (pr_ens(nall + n3) < 1.e-4f) pr_ens(nall + n3) = 0.f;
        }

        // ---- large-scale forcing ----
        k22x = gd_maximi(heo_cup, 3, kbmax, ierr);
        ierr2 = ierr;
        ierr3 = ierr;
        kbconx = gd_cup_kbcon(cap_max_increment, 2, k22x, heo_cup, heso_cup,
                              ierr2, kbmax, po_cup, cap_max);
        kbconx = gd_cup_kbcon(cap_max_increment, 3, k22x, heo_cup, heso_cup,
                              ierr3, kbmax, po_cup, cap_max);

        // cup_forcing_ens (2578-3081)
        if (ierr == 0) {
            int kclim = 1;
            bool found = false;
            for (int k = 15; k >= 1; --k)
                if (po_cup[ktop] < GD_PCRIT[k - 1]) { kclim = k; found = true; break; }
            if (!found && po_cup[ktop] >= GD_PCRIT[0]) kclim = 1;
            if (kclim < 1) kclim = 1;
            int kk = (kclim - 1 > 1) ? kclim - 1 : 1;
            float aclim1 = FMUL(GD_ACRIT[kclim - 1], 1.e3f);
            float aclim2 = FMUL(GD_ACRIT[kk - 1], 1.e3f);
            float aclim3 = FMUL(GD_ACRITT[kclim - 1], 1.e3f);
            float aclim4 = FMUL(GD_ACRITT[kk - 1], 1.e3f);
            float xff_ens3[17];
            float xff0 = FDIV(FSUB(aa1, aa0), dt);
            xff_ens3[1] = FDIV(FSUB(aa1, aa0), dt);
            xff_ens3[2] = FMUL(.9f, xff_ens3[1]);
            xff_ens3[3] = FMUL(1.1f, xff_ens3[1]);
            xff_ens3[4] = -FDIV(omeg[k22], 9.81f);
            xff_ens3[5] = -FDIV(omeg[kbcon], 9.81f);
            xff_ens3[6] = -FDIV(omeg[1], 9.81f);
            for (int k = 2; k <= kbcon - 1; ++k) {
                float xomg = -FDIV(omeg[k], 9.81f);
                if (xomg > xff_ens3[6]) xff_ens3[6] = xomg;
            }
            xff_ens3[7] = mconv;
            xff_ens3[8] = mconv;
            xff_ens3[9] = mconv;
            xff_ens3[10] = FDIV(aa1, 1200.f);
            xff_ens3[11] = FDIV(aa1, 1800.f);
            xff_ens3[12] = FDIV(aa1, 2400.f);
            xff_ens3[13] = gd_max(0.f, FDIV(FSUB(aa1, aclim1), dt));
            xff_ens3[14] = gd_max(0.f, FDIV(FSUB(aa1, aclim2), dt));
            xff_ens3[15] = gd_max(0.f, FDIV(FSUB(aa1, aclim3), dt));
            xff_ens3[16] = gd_max(0.f, FDIV(FSUB(aa1, aclim4), dt));
            float xk[3];
            for (int n = 1; n <= 3; ++n) {
                float x = FDIV(FSUB(xaa0_ens[n - 1], aa1), mbdt_ens[1]);
                if (x <= 0.f && x > -1.e-6f) x = -1.e-6f;
                if (x > 0.f && x < 1.e-6f) x = 1.e-6f;
                xk[n - 1] = x;
            }
            const float massfld = 0.f;
            for (int ne = 1; ne <= 3; ++ne) {
                int nall = (iedt - 1) * 48 + (ne - 1) * 16;
                // the count once per column (see the header), or WRF's
                const bool count = wrf_closure_count ? true
                                                     : (iedt == 1 && ne == 1);
                if (xland1 < .1f) {
                    if (ierr2 > 0 || ierr3 > 0) {
                        xff_ens3[1] = 0.f; xff_ens3[2] = 0.f; xff_ens3[3] = 0.f;
                        if (count)
                            closure_n = FSUB(closure_n, wrf_closure_count ? 1.f : 3.f);
                        xff_ens3[7] = 0.f; xff_ens3[8] = 0.f; xff_ens3[9] = 0.f;
                        if (count)
                            closure_n = FSUB(closure_n, wrf_closure_count ? 1.f : 3.f);
                    }
                    xff_ens3[4] = 0.f; xff_ens3[5] = 0.f; xff_ens3[6] = 0.f;
                    if (count) closure_n = FSUB(closure_n, 3.f);
                    xff_ens3[10] = 0.f; xff_ens3[11] = 0.f; xff_ens3[12] = 0.f;
                    if (ne == 1 && (wrf_closure_count || iedt == 1))
                        closure_n = FSUB(closure_n, 3.f);
                    xff_ens3[13] = 0.f; xff_ens3[14] = 0.f; xff_ens3[15] = 0.f;
                    if (!wrf_closure_count) xff_ens3[16] = 0.f;
                    if (ne == 1 && (wrf_closure_count || iedt == 1))
                        closure_n = FSUB(closure_n, 4.f);
                }
                const float xkn = xk[ne - 1];
                if (xff0 > 0.f) {
                    xf_ens(nall + 1) = FADD(gd_max(0.f, -FDIV(xff_ens3[1], xkn)), massfld);
                    xf_ens(nall + 2) = FADD(gd_max(0.f, -FDIV(xff_ens3[2], xkn)), massfld);
                    xf_ens(nall + 3) = FADD(gd_max(0.f, -FDIV(xff_ens3[3], xkn)), massfld);
                    xf_ens(nall + 13) = FADD(gd_max(0.f, -FDIV(xff_ens3[13], xkn)), massfld);
                    xf_ens(nall + 14) = FADD(gd_max(0.f, -FDIV(xff_ens3[14], xkn)), massfld);
                    xf_ens(nall + 15) = FADD(gd_max(0.f, -FDIV(xff_ens3[15], xkn)), massfld);
                    xf_ens(nall + 16) = FADD(gd_max(0.f, -FDIV(xff_ens3[16], xkn)), massfld);
                } else {
                    xf_ens(nall + 1) = massfld; xf_ens(nall + 2) = massfld;
                    xf_ens(nall + 3) = massfld; xf_ens(nall + 13) = massfld;
                    xf_ens(nall + 14) = massfld; xf_ens(nall + 15) = massfld;
                    xf_ens(nall + 16) = massfld;
                }
                xf_ens(nall + 4) = gd_max(0.f, FADD(xff_ens3[4], massfld));
                xf_ens(nall + 5) = gd_max(0.f, FADD(xff_ens3[5], massfld));
                xf_ens(nall + 6) = gd_max(0.f, FADD(xff_ens3[6], massfld));
                for (int n3 = 7; n3 <= 9; ++n3) {
                    float a1 = gd_max(1.e-3f, pr_ens(nall + n3));
                    xf_ens(nall + n3) = gd_max(0.f, FDIV(xff_ens3[n3], a1));
                }
                if (xkn < 0.f) {
                    xf_ens(nall + 10) = FADD(gd_max(0.f, -FDIV(xff_ens3[10], xkn)), massfld);
                    xf_ens(nall + 11) = FADD(gd_max(0.f, -FDIV(xff_ens3[11], xkn)), massfld);
                    xf_ens(nall + 12) = FADD(gd_max(0.f, -FDIV(xff_ens3[12], xkn)), massfld);
                } else {
                    xf_ens(nall + 10) = massfld;
                    xf_ens(nall + 11) = massfld;
                    xf_ens(nall + 12) = massfld;
                }
                // icoic == 0
                xf_ens(nall + 14) = xf_ens(nall + 13);
                if ((ne == 2 && ierr2 > 0) || (ne == 3 && ierr3 > 0))
                    for (int n3 = 1; n3 <= 16; ++n3) xf_ens(nall + n3) = 0.f;
            }
            int nall = (iedt - 1) * 48;
            int nall2 = nall + 16;
            for (int n3 = 4; n3 <= 12; ++n3) xf_ens(nall + n3) = xf_ens(nall2 + n3);
        } else if (ierr != 20) {
            for (int n = 1; n <= GD_ENS; ++n) xf_ens(n) = 0.f;
        }

        for (int k = 1; k <= ktf; ++k) {
            if (ierr == 0) {
                dellat_ens[iedt - 1][k] = dellat[k];
                dellaq_ens[iedt - 1][k] = dellaq[k];
                dellaqc_ens[iedt - 1][k] = dellaqc[k];
                pwo_ens[iedt - 1][k] = FADD(pwo[k], FMUL(edt, pwdo[k]));
            } else {
                dellat_ens[iedt - 1][k] = 0.f;
                dellaq_ens[iedt - 1][k] = 0.f;
                dellaqc_ens[iedt - 1][k] = 0.f;
                pwo_ens[iedt - 1][k] = 0.f;
            }
        }
    }

    // ---- cup_output_ens (3454-3730) ----
    float apr[8];
    for (int a = 0; a < 8; ++a) apr[a] = sc[GDS_APR + a];
    for (int k = 1; k <= ktf; ++k) { outt[k] = 0.f; outq[k] = 0.f; outqc[k] = 0.f; }
    float pre = 0.f, xmb = 0.f;
    if (ierr == 0)
        for (int n = 1; n <= GD_ENS; ++n)
            if (pr_ens(n) <= 0.f) xf_ens(n) = 0.f;
    float xave[16], xcap[3], xmb_ave, xmb_std, prave[16], prcap[3], pr_ave, pr_std;
    gd_massflx_stats(xf_ens, ierr, xave, xcap, &xmb_ave, &xmb_std);
    float pr_gr = 0.f, pr_w = 0.f, pr_mc = 0.f, pr_st = 0.f, pr_as = 0.f,
          pr_capma = 0.f, pr_capme = 0.f, pr_capmi = 0.f;
    if (ierr == 0 && xmb_std > 0.f) {
        pr_gr = gd_third(xave[0], xave[1], xave[2]);
        pr_w = gd_third(xave[3], xave[4], xave[5]);
        pr_mc = gd_third(xave[6], xave[7], xave[8]);
        pr_st = gd_third(xave[9], xave[10], xave[11]);
        pr_as = FMUL(.25f, FADD(FADD(FADD(xave[12], xave[13]), xave[14]), xave[15]));
        pr_capma = xcap[0]; pr_capme = xcap[1]; pr_capmi = xcap[2];
    }
    gd_massflx_stats(pr_ens, ierr, prave, prcap, &pr_ave, &pr_std);
    if (ierr == 0 && pr_std > 0.f) {
        apr[0] = FADD(FMUL(FMUL(gd_third(prave[0], prave[1], prave[2]), 3600.f), pr_gr), apr[0]);
        apr[1] = FADD(FMUL(FMUL(gd_third(prave[3], prave[4], prave[5]), 3600.f), pr_w), apr[1]);
        apr[2] = FADD(FMUL(FMUL(gd_third(prave[6], prave[7], prave[8]), 3600.f), pr_mc), apr[2]);
        apr[3] = FADD(FMUL(FMUL(gd_third(prave[9], prave[10], prave[11]), 3600.f), pr_st), apr[3]);
        apr[4] = FADD(FMUL(FMUL(FMUL(.25f, FADD(FADD(FADD(prave[12], prave[13]), prave[14]),
                                               prave[15])), 3600.f), pr_as), apr[4]);
        apr[5] = FADD(FMUL(FMUL(prcap[0], 3600.f), pr_capma), apr[5]);
        apr[6] = FADD(FMUL(FMUL(prcap[1], 3600.f), pr_capme), apr[6]);
        apr[7] = FADD(FMUL(FMUL(prcap[2], 3600.f), pr_capmi), apr[7]);
    }
    const float ddtes = 200.f, tuning = 0.f;
    float xfac1 = 1.f;
    if (ierr == 0) {
        if (xmb_ave <= 0.f) { ierr = 13; xmb_ave = 0.f; }
        xmb = gd_max(FMUL(.1f, xmb_ave), FSUB(xmb_ave, FMUL(tuning, xmb_std)));
        float clos_wei = FDIV(16.f, gd_max(1.f, closure_n));
        if (xland1 < .5f) xmb = FMUL(xmb, clos_wei);
        if (xmb == 0.f) ierr = 19;
        if (xmb > 100.f) ierr = 19;
    }
    xfac1 = xmb_ave;
    for (int k = 1; k <= ktf; ++k) {
        float dtt = 0.f, dtq = 0.f, dtqc = 0.f, dtpw = 0.f;
        if (ierr == 0 && k <= ktop) {
            for (int n = 0; n < 3; ++n) {
                dtt = FADD(dtt, dellat_ens[n][k]);
                dtq = FADD(dtq, dellaq_ens[n][k]);
                dtqc = FADD(dtqc, dellaqc_ens[n][k]);
                dtpw = FADD(dtpw, pwo_ens[n][k]);
            }
            float outtes = FDIV(FMUL(FMUL(dtt, xmb), 86400.f), 3.f);
            if (outtes > FMUL(2.f, ddtes) && k > 2) {
                xmb = FMUL(FDIV(FMUL(2.f, ddtes), outtes), xmb);
                outtes = FMUL(1.f, ddtes);
            }
            if (outtes < -ddtes) {
                xmb = -FMUL(FDIV(ddtes, outtes), xmb);
                outtes = -ddtes;
            }
            if (outtes > FMUL(.5f, ddtes) && k <= 2) {
                xmb = FMUL(FDIV(ddtes, outtes), xmb);
                outtes = FMUL(.5f, ddtes);
            }
            outt[k] = FDIV(FMUL(xmb, dtt), 3.f);
            outq[k] = FDIV(FMUL(xmb, dtq), 3.f);
            outqc[k] = FDIV(FMUL(xmb, dtqc), 3.f);
            pre = FADD(pre, FDIV(FMUL(xmb, dtpw), 3.f));
        }
    }
    bool zero_ens = false;
    if (ierr == 0) {
        float prerate = FMUL(pre, 3600.f);
        if (prerate < .1f && (ierr2 > 0 || ierr3 > 0)) {
            pre = 0.f;
            ierr = 221;
            for (int k = 1; k <= ktf; ++k) { outt[k] = 0.f; outq[k] = 0.f; outqc[k] = 0.f; }
            zero_ens = true;
        }
    }
    if (zero_ens)
        for (int n = 1; n <= GD_ENS; ++n) xf_ens(n) = 0.f;
    if (ierr == 0) {
        xfac1 = FDIV(xmb, xfac1);
        for (int n = 1; n <= GD_ENS; ++n) xf_ens(n) = FMUL(xf_ens(n), xfac1);
    }
    pre = gd_max(pre, 0.f);
    pret = pre;

    // ---- neg_check (4479-4557) on GRELLDRV's q2d ----
    {
        float thresh = 200.01f;
        float qmemf = 1.f;
        for (int k = 1; k <= ktf; ++k) {
            float qmem = FMUL(outt[k], 86400.f);
            if (qmem > FMUL(2.f, thresh)) qmemf = gd_min(qmemf, FDIV(FMUL(2.f, thresh), qmem));
            if (qmem < -thresh) qmemf = gd_min(qmemf, -FDIV(thresh, qmem));
        }
        for (int k = 1; k <= ktf; ++k) {
            outq[k] = FMUL(outq[k], qmemf);
            outt[k] = FMUL(outt[k], qmemf);
            outqc[k] = FMUL(outqc[k], qmemf);
        }
        pret = FMUL(pret, qmemf);
        thresh = 1.e-10f;
        qmemf = 1.f;
        for (int k = 1; k <= ktf; ++k) {
            float qmem = outq[k];
            if (fabsf(qmem) > 0.f) {
                float qtest = FADD(q2d[k], FMUL(outq[k], dt));
                if (qtest < thresh) {
                    float qmem1 = outq[k];
                    float qmem2 = FDIV(FSUB(thresh, q2d[k]), dt);
                    qmemf = gd_min(qmemf, FDIV(qmem2, qmem1));
                }
            }
        }
        for (int k = 1; k <= ktf; ++k) {
            outq[k] = FMUL(outq[k], qmemf);
            outt[k] = FMUL(outt[k], qmemf);
            outqc[k] = FMUL(outqc[k], qmemf);
        }
        pret = FMUL(pret, qmemf);
    }

    // ---- GRELLDRV outputs (310-380) ----
    float* lo = lev + (size_t)col * GDO_NLEV * nz;
    float* so = sca + (size_t)col * GDR_NSCA;
    int* io = isc + (size_t)col * GDK_NINT;
#define OUT(f, k) lo[(f) * nz + (k) - 1]
    float htop = sc[GDS_HTOP], hbot = sc[GDS_HBOT];
    float pratec = 0.f, raincv = 0.f, cuten = 0.f;
    so[GDR_PRET] = pret;
    int ktop_deep = 0;
    if (band >= 1) {
        ktop_deep = ktop;
        if (band == 2) {
            if (pret > 0.f) {
                pratec = pret;
                raincv = FMUL(pret, dt);
                cuten = 1.f;
                if ((float)ktop > htop) htop = FADD((float)ktop, .001f);
                if ((float)kbcon < hbot) hbot = FADD((float)kbcon, .001f);
            }
        } else {
            pret = 0.f;
        }
        for (int k = 1; k <= ktf; ++k) {
            OUT(GDO_RTHCUTEN, k) = FDIV(FMUL(outt[k], cuten), IN(GDI_PI, k));
            OUT(GDO_RQVCUTEN, k) = FMUL(outq[k], cuten);
            float qcten = FMUL(outqc[k], cuten);
            float clw = FMUL(cupclw[k], cuten);
            OUT(GDO_RQCCUTEN, k) = qcten;
            OUT(GDO_GDC, k) = clw;
            OUT(GDO_GDC2, k) = 0.f;
            OUT(GDO_RQICUTEN, k) = 0.f;
            if (f_qi) {
                if (t2d[k] < 258.f) {
                    OUT(GDO_RQICUTEN, k) = qcten;
                    OUT(GDO_RQCCUTEN, k) = 0.f;
                    OUT(GDO_GDC2, k) = clw;
                } else {
                    OUT(GDO_RQICUTEN, k) = 0.f;
                    OUT(GDO_RQCCUTEN, k) = qcten;
                    OUT(GDO_GDC, k) = clw;
                }
            }
        }
    } else {
        // outside the j band GRELLDRV writes no tendency (they stay at their
        // gdinit zero) and leaves HTOP/HBOT alone
        for (int f = 0; f < GDO_NLEV; ++f)
            for (int k = 1; k <= ktf; ++k) OUT(f, k) = 0.f;
    }
    so[GDR_RAINCV] = raincv;
    so[GDR_PRATEC] = pratec;
    so[GDR_HTOP] = htop;
    so[GDR_HBOT] = hbot;
    for (int a = 0; a < 8; ++a) so[GDR_APR + a] = apr[a];
    io[GDK_KTOP_DEEP] = ktop_deep;
    io[GDK_KBCON] = kbcon;
    io[GDK_KTOP] = ktop;
    io[GDK_IERR] = ierr;
    io[GDK_K22] = k22;
    if (ens_out) {
        float* e = ens_out + (size_t)col * 2 * GD_ENS;
        for (int n = 1; n <= GD_ENS; ++n) { e[n - 1] = xf_ens(n); e[GD_ENS + n - 1] = pr_ens(n); }
    }
#undef OUT
#undef IN
}
