// noahmp_vegeflux.cu -- CUDA transcription of the Noah-MP VEGE_FLUX subtree
// (WRF v4.6.1 phys/module_sf_noahmplsm.F, commit
//  d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
//
// Gate: max_ulp 0 against the gfortran+glibc oracle fixture
// gpuwm/data/noahmp/oracle/noahmp-vegeflux.csv.
//
// Two things make that reachable and both are load-bearing:
//
//  1. Every FP32 operation goes through __fadd_rn / __fsub_rn / __fmul_rn /
//     __fdiv_rn / __fsqrt_rn.  Those pin the *hardware* rounding mode and, just
//     as importantly, stop nvcc contracting a*b+c into an FMA that the SSE
//     baseline gfortran targets cannot emit.
//
//  2. The transcendentals are glibc 2.39's own logf/expf/powf/atanf, not CUDA's
//     device libm.  CUDA's __logf/logf/powf are different functions with
//     different error, and no amount of rounding control makes them agree.
//     glibc's kernels compute in binary64 and round once; the binary64 steps
//     here use __dadd_rn/__dsub_rn/__dmul_rn and __fma_rn, with __fma_rn used
//     at exactly the sites where the -fma multiarch build of glibc contracts
//     (read off the disassembly of the installed libm.so.6, not guessed).
//
// The coefficient tables live in __constant__ memory and are uploaded from the
// host.  They are deliberately *not* literal arrays inside the kernels: ptxas
// 12.8's constant folder does not honour round-to-nearest-even, so a literal
// FP table can have its differences mis-folded at compile time, and __fsub_rn
// pins the hardware mode, not the compiler's folder.  Uploading them means the
// compiler never sees the values.

// nvrtc has no <stdint.h>, so the fixed-width types are spelled out.  They are
// the only integer widths this file relies on and both are exact on every
// target CUDA supports.
typedef unsigned int uint32_t;
typedef int int32_t;

// ---------------------------------------------------------------------------
// rounding-pinned arithmetic
// ---------------------------------------------------------------------------
#define FA(a, b) __fadd_rn((a), (b))
#define FS(a, b) __fsub_rn((a), (b))
#define FM(a, b) __fmul_rn((a), (b))
#define FD(a, b) __fdiv_rn((a), (b))
#define FSQ(a)   __fsqrt_rn(a)

#define DA(a, b) __dadd_rn((a), (b))
#define DS(a, b) __dsub_rn((a), (b))
#define DM(a, b) __dmul_rn((a), (b))
#define DFMA(a, b, c) __fma_rn((a), (b), (c))

__device__ __forceinline__ float fmnf(float a, float b) { return a < b ? a : b; }
__device__ __forceinline__ float fmxf(float a, float b) { return a > b ? a : b; }

// gfortran's expansion of integer powers.  x**3 -> x*(x*x), x**4 -> (x*x)*(x*x).
__device__ __forceinline__ float p3f(float x) { return FM(x, FM(x, x)); }
__device__ __forceinline__ float p4f(float x) { float t = FM(x, x); return FM(t, t); }

// ---------------------------------------------------------------------------
// glibc 2.39 libm tables (uploaded from the host; never literals)
// ---------------------------------------------------------------------------
// Declared inside extern "C" so the symbol names survive unmangled and the
// host can cuModuleGetGlobal them by name to upload the values.
extern "C" {
__constant__ unsigned long long c_exp2f_tab[32];
__constant__ double c_exp2f_poly[3];         // unscaled  (powf's exp2_inline)
__constant__ double c_exp2f_poly_scaled[3];  // scaled    (expf)
__constant__ double c_exp2f_shift;           // 0x1.8p52
__constant__ double c_exp2f_shift_scaled;    // 0x1.8p52 / 32
__constant__ double c_exp2f_invln2_scaled;   // (1/ln2) * 32

__constant__ double c_logf_tab[32];          // invc, logc interleaved
__constant__ double c_logf_ln2;
__constant__ double c_logf_poly[3];

__constant__ double c_powf_tab[32];          // invc, logc interleaved
__constant__ double c_powf_poly[5];

__constant__ float c_atanhi[4];
__constant__ float c_atanlo[4];
__constant__ float c_atan_aT[11];
}

// ---------------------------------------------------------------------------
// glibc logf  (sysdeps/ieee754/flt-32/e_logf.c, __logf_fma)
// ---------------------------------------------------------------------------
__device__ float glibc_logf(float x)
{
#ifdef USE_DEVICE_LIBM
    return logf(x);   // negative control: CUDA device libm, a different function
#else
    uint32_t ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if ((ix - 0x00800000u) >= (0x7f800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return -__int_as_float(0x7f800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(FM(x, __int_as_float(0x4b000000)));  // x * 0x1p23
        ix -= (23u << 23);
    }

    uint32_t tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = ((int32_t)tmp) >> 23;
    uint32_t iz = ix - (tmp & 0xff800000u);
    double invc = c_logf_tab[2 * i];
    double logc = c_logf_tab[2 * i + 1];
    double z = (double)__uint_as_float(iz);

    double r = DFMA(z, invc, -1.0);
    double y0 = DFMA((double)k, c_logf_ln2, logc);

    double r2 = DM(r, r);
    double y = DFMA(c_logf_poly[1], r, c_logf_poly[2]);
    y = DFMA(c_logf_poly[0], r2, y);
    y = DFMA(r2, y, DA(y0, r));
    return (float)y;
#endif
}

// ---------------------------------------------------------------------------
// glibc expf  (sysdeps/ieee754/flt-32/e_expf.c, __expf_fma)
// ---------------------------------------------------------------------------
__device__ float glibc_expf(float x)
{
#ifdef USE_DEVICE_LIBM
    return expf(x);   // negative control
#else
    uint32_t ux = __float_as_uint(x);
    uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return FA(x, x);
        if (x > 88.72283935546875f) return __int_as_float(0x7f800000);
        if (x < -103.97207641601562f) return 0.0f;
    }

    double xd = (double)x;
    double kd = DFMA(c_exp2f_invln2_scaled, xd, c_exp2f_shift);
    unsigned long long ki = __double_as_longlong(kd);
    kd = DS(kd, c_exp2f_shift);
    double r = DFMA(c_exp2f_invln2_scaled, xd, -kd);

    unsigned long long t = c_exp2f_tab[ki & 31ull];
    t += ki << 47;
    double s = __longlong_as_double((long long)t);

    double z = DFMA(c_exp2f_poly_scaled[0], r, c_exp2f_poly_scaled[1]);
    double r2 = DM(r, r);
    double y = DFMA(r, c_exp2f_poly_scaled[2], 1.0);
    y = DFMA(z, r2, y);
    y = DM(y, s);
    return (float)y;
#endif
}

// ---------------------------------------------------------------------------
// glibc powf  (sysdeps/ieee754/flt-32/e_powf.c, __powf_fma)
// Only the finite, positive-base domain the Noah-MP leaves reach is
// transcribed; anything else returns NaN rather than a value nothing verified.
// ---------------------------------------------------------------------------
__device__ double powf_log2_inline(uint32_t ix)
{
    uint32_t tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    uint32_t top = tmp & 0xff800000u;
    uint32_t iz = ix - top;
    int k = ((int32_t)top) >> 23;
    double invc = c_powf_tab[2 * i];
    double logc = c_powf_tab[2 * i + 1];
    double z = (double)__uint_as_float(iz);

    double r = DFMA(z, invc, -1.0);
    double y0 = DA(logc, (double)k);

    double r2 = DM(r, r);
    double y = DFMA(c_powf_poly[0], r, c_powf_poly[1]);
    double p = DFMA(c_powf_poly[2], r, c_powf_poly[3]);
    double r4 = DM(r2, r2);
    double q = DFMA(c_powf_poly[4], r, y0);
    q = DFMA(p, r2, q);
    y = DFMA(y, r4, q);
    return y;
}

__device__ double powf_exp2_inline(double xd)
{
    double kd = DA(xd, c_exp2f_shift_scaled);
    unsigned long long ki = __double_as_longlong(kd);
    kd = DS(kd, c_exp2f_shift_scaled);
    double r = DS(xd, kd);

    unsigned long long t = c_exp2f_tab[ki & 31ull];
    t += ki << 47;
    double s = __longlong_as_double((long long)t);

    double z = DFMA(c_exp2f_poly[0], r, c_exp2f_poly[1]);
    double r2 = DM(r, r);
    double y = DFMA(r, c_exp2f_poly[2], 1.0);
    y = DFMA(z, r2, y);
    return DM(y, s);
}

__device__ float glibc_powf(float x, float y)
{
#ifdef USE_DEVICE_LIBM
    return powf(x, y);   // negative control
#else
    uint32_t ix = __float_as_uint(x);
    uint32_t iy = __float_as_uint(y);

    // zeroinfnan(iy) or x subnormal/inf/nan/negative
    bool iy_special = ((2u * iy - 1u) >= (2u * 0x7f800000u - 1u));
    if ((ix - 0x00800000u) >= (0x7f800000u - 0x00800000u) || iy_special) {
        if (iy_special) {
            if ((2u * iy) == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
        }
        return __int_as_float(0x7fc00000);   // outside the transcribed domain
    }

    double logx = powf_log2_inline(ix);
    double ylogx = DM((double)y, logx);
    unsigned long long uy = __double_as_longlong(ylogx);
    if (((uy >> 47) & 0xffffull) >= (__double_as_longlong(126.0) >> 47)) {
        if (ylogx > 127.99999995223258)  return __int_as_float(0x7f800000);
        if (ylogx <= -150.0)             return 0.0f;
    }
    return (float)powf_exp2_inline(ylogx);
#endif
}

// ---------------------------------------------------------------------------
// glibc atanf  (sysdeps/ieee754/flt-32/s_atanf.c -- plain binary32, no FMA)
// aT[0] is 0x3EAAAAAB: the decimal literal, not the stale hex comment.
// ---------------------------------------------------------------------------
__device__ float glibc_atanf(float x)
{
#ifdef USE_DEVICE_LIBM
    return atanf(x);   // negative control
#elif defined(GPUWM_NOAHMP_ATAN_GLIBC243)
    return noahmp_glibc243_atanf(x);
#else
    int32_t hx = (int32_t)__float_as_uint(x);
    int32_t ix = hx & 0x7fffffff;
    int id;

    if (ix >= 0x4c000000) {
        if (ix > 0x7f800000) return FA(x, x);
        return hx > 0 ? FA(c_atanhi[3], c_atanlo[3])
                      : FS(-c_atanhi[3], c_atanlo[3]);
    }
    if (ix < 0x3ee00000) {
        if (ix < 0x31000000) return x;
        id = -1;
    } else {
        x = fabsf(x);
        if (ix < 0x3f980000) {
            if (ix < 0x3f300000) {
                id = 0;
                x = FD(FS(FM(2.0f, x), 1.0f), FA(2.0f, x));
            } else {
                id = 1;
                x = FD(FS(x, 1.0f), FA(x, 1.0f));
            }
        } else if (ix < 0x401c0000) {
            id = 2;
            x = FD(FS(x, 1.5f), FA(1.0f, FM(1.5f, x)));
        } else {
            id = 3;
            x = FD(-1.0f, x);
        }
    }

    float z = FM(x, x);
    float w = FM(z, z);
    float s1 = FA(c_atan_aT[8], FM(w, c_atan_aT[10]));
    s1 = FA(c_atan_aT[6], FM(w, s1));
    s1 = FA(c_atan_aT[4], FM(w, s1));
    s1 = FA(c_atan_aT[2], FM(w, s1));
    s1 = FA(c_atan_aT[0], FM(w, s1));
    s1 = FM(z, s1);
    float s2 = FA(c_atan_aT[7], FM(w, c_atan_aT[9]));
    s2 = FA(c_atan_aT[5], FM(w, s2));
    s2 = FA(c_atan_aT[3], FM(w, s2));
    s2 = FA(c_atan_aT[1], FM(w, s2));
    s2 = FM(w, s2);

    if (id < 0) return FS(x, FM(x, FA(s1, s2)));
    float r = FS(c_atanhi[id], FS(FS(FM(x, FA(s1, s2)), c_atanlo[id]), x));
    return hx < 0 ? -r : r;
#endif
}

// ---------------------------------------------------------------------------
// module constants (module_sf_noahmplsm.F lines 204-220)
// ---------------------------------------------------------------------------
extern "C" {
__constant__ float c_phys[9];
__constant__ float c_esat_a[7];
__constant__ float c_esat_b[7];
__constant__ float c_esat_c[7];
__constant__ float c_esat_d[7];
}
#define PH_GRAV   c_phys[0]
#define PH_SB     c_phys[1]
#define PH_VKC    c_phys[2]
#define PH_TFRZ   c_phys[3]
#define PH_CPAIR  c_phys[4]
#define PH_CWAT   c_phys[5]
#define PH_CICE   c_phys[6]
#define PH_DENH2O c_phys[7]
#define PH_DENICE c_phys[8]

__device__ __forceinline__ float esat_horner(float t, const float *c)
{
    float acc = FA(c[5], FM(t, c[6]));
    acc = FA(c[4], FM(t, acc));
    acc = FA(c[3], FM(t, acc));
    acc = FA(c[2], FM(t, acc));
    acc = FA(c[1], FM(t, acc));
    return FA(c[0], FM(t, acc));
}

__device__ void dev_esat(float t, float *esw, float *esi, float *desw, float *desi)
{
    *esw = FM(100.0f, esat_horner(t, c_esat_a));
    *esi = FM(100.0f, esat_horner(t, c_esat_b));
    *desw = FM(100.0f, esat_horner(t, c_esat_c));
    *desi = FM(100.0f, esat_horner(t, c_esat_d));
}

// ---------------------------------------------------------------------------
// RAGRB
// ---------------------------------------------------------------------------
__device__ void dev_ragrb(int iter, float vai, float rhoair, float hg, float tah,
                          float zpd, float z0mg, float z0hg, float hcan, float uc,
                          float z0h, float fv, float cwp, float mpe, float dleaf,
                          float *mozg_io, float *fhg_io,
                          float *ramg, float *rahg, float *rawg, float *rb)
{
    float mozg = 0.0f;
    float molg = 0.0f;
    float fhg = *fhg_io;

    if (iter > 1) {
        float tmp1 = FD(FM(FM(PH_VKC, FD(PH_GRAV, tah)), hg), FM(rhoair, PH_CPAIR));
        if (fabsf(tmp1) <= mpe) tmp1 = mpe;
        molg = FD(FM(-1.0f, p3f(fv)), tmp1);
        mozg = fmnf(FD(FS(zpd, z0mg), molg), 1.0f);
    }

    float fhgnew;
    if (mozg < 0.0f)
        fhgnew = glibc_powf(FS(1.0f, FM(15.0f, mozg)), -0.25f);
    else
        fhgnew = FA(1.0f, FM(4.7f, mozg));

    fhg = (iter == 1) ? fhgnew : FM(0.5f, FA(fhg, fhgnew));

    float cwpc = glibc_powf(FM(FM(FM(cwp, vai), hcan), fhg), 0.5f);

    float tmp1 = glibc_expf(-FD(FM(cwpc, z0hg), hcan));
    float tmp2 = glibc_expf(-FD(FM(cwpc, FA(z0h, zpd)), hcan));
    float tmprah2 = FM(FD(FM(hcan, glibc_expf(cwpc)), cwpc), FS(tmp1, tmp2));

    float kh = fmxf(FM(FM(PH_VKC, fv), FS(hcan, zpd)), mpe);
    *ramg = 0.0f;
    *rahg = FD(tmprah2, kh);
    *rawg = *rahg;

    float tmprb = FD(FM(cwpc, 50.0f), FS(1.0f, glibc_expf(-FD(cwpc, 2.0f))));
    float r = FM(tmprb, FSQ(FD(dleaf, uc)));
    *rb = fmnf(fmxf(r, 5.0f), 50.0f);

    *mozg_io = mozg;
    *fhg_io = fhg;
}

// ---------------------------------------------------------------------------
// SFCDIF1
// ---------------------------------------------------------------------------
__device__ void dev_sfcdif1(int iter, float sfctmp, float rhoair, float h,
                            float qair, float zlvl, float zpd, float z0m,
                            float z0h, float ur, float mpe,
                            float *moz_io, int *mozsgn_io, float *fm_io,
                            float *fh_io, float *fm2_io, float *fh2_io,
                            float *fv_io, float *cm_o, float *ch_o, float *ch2_o)
{
    float moz = *moz_io, fm = *fm_io, fh = *fh_io, fm2 = *fm2_io, fh2 = *fh2_io;
    float fv = *fv_io;
    int mozsgn = *mozsgn_io;
    float mozold = moz;

    float tmpcm = glibc_logf(FD(FS(zlvl, zpd), z0m));
    float tmpch = glibc_logf(FD(FS(zlvl, zpd), z0h));
    float tmpcm2 = glibc_logf(FD(FA(2.0f, z0m), z0m));
    float tmpch2 = glibc_logf(FD(FA(2.0f, z0h), z0h));

    float moz2;
    if (iter == 1) {
        fv = 0.0f;
        moz = 0.0f;
        moz2 = 0.0f;
    } else {
        float tvir = FM(FA(1.0f, FM(0.61f, qair)), sfctmp);
        float tmp1 = FD(FM(FM(PH_VKC, FD(PH_GRAV, tvir)), h), FM(rhoair, PH_CPAIR));
        if (fabsf(tmp1) <= mpe) tmp1 = mpe;
        float mol = FD(FM(-1.0f, p3f(fv)), tmp1);
        moz = fmnf(FD(FS(zlvl, zpd), mol), 1.0f);
        moz2 = fmnf(FD(FA(2.0f, z0h), mol), 1.0f);
    }

    if (FM(mozold, moz) < 0.0f) mozsgn += 1;
    if (mozsgn >= 2) {
        moz = 0.0f; fm = 0.0f; fh = 0.0f;
        moz2 = 0.0f; fm2 = 0.0f; fh2 = 0.0f;
    }

    float fmnew, fhnew, fm2new, fh2new;
    if (moz < 0.0f) {
        float t1 = glibc_powf(FS(1.0f, FM(16.0f, moz)), 0.25f);
        float t2 = glibc_logf(FD(FA(1.0f, FM(t1, t1)), 2.0f));
        float t3 = glibc_logf(FD(FA(1.0f, t1), 2.0f));
        fmnew = FA(FS(FA(FM(2.0f, t3), t2), FM(2.0f, glibc_atanf(t1))), 1.5707963f);
        fhnew = FM(2.0f, t2);

        float t12 = glibc_powf(FS(1.0f, FM(16.0f, moz2)), 0.25f);
        float t22 = glibc_logf(FD(FA(1.0f, FM(t12, t12)), 2.0f));
        float t32 = glibc_logf(FD(FA(1.0f, t12), 2.0f));
        fm2new = FA(FS(FA(FM(2.0f, t32), t22), FM(2.0f, glibc_atanf(t12))), 1.5707963f);
        fh2new = FM(2.0f, t22);
    } else {
        fmnew = FM(-5.0f, moz);
        fhnew = fmnew;
        fm2new = FM(-5.0f, moz2);
        fh2new = fm2new;
    }

    if (iter == 1) {
        fm = fmnew; fh = fhnew; fm2 = fm2new; fh2 = fh2new;
    } else {
        fm = FM(0.5f, FA(fm, fmnew));
        fh = FM(0.5f, FA(fh, fhnew));
        fm2 = FM(0.5f, FA(fm2, fm2new));
        fh2 = FM(0.5f, FA(fh2, fh2new));
    }

    fh = fmnf(fh, FM(0.9f, tmpch));
    fm = fmnf(fm, FM(0.9f, tmpcm));
    fh2 = fmnf(fh2, FM(0.9f, tmpch2));
    fm2 = fmnf(fm2, FM(0.9f, tmpcm2));

    float cmfm = FS(tmpcm, fm);
    float chfh = FS(tmpch, fh);
    float cm2fm2 = FS(tmpcm2, fm2);
    float ch2fh2 = FS(tmpch2, fh2);
    if (fabsf(cmfm) <= mpe) cmfm = mpe;
    if (fabsf(chfh) <= mpe) chfh = mpe;
    if (fabsf(cm2fm2) <= mpe) cm2fm2 = mpe;
    if (fabsf(ch2fh2) <= mpe) ch2fh2 = mpe;

    float cm = FD(FM(PH_VKC, PH_VKC), FM(cmfm, cmfm));
    float ch = FD(FM(PH_VKC, PH_VKC), FM(cmfm, chfh));
    fv = FM(ur, FSQ(cm));
    float ch2 = FD(FM(PH_VKC, fv), ch2fh2);

    *moz_io = moz; *mozsgn_io = mozsgn;
    *fm_io = fm; *fh_io = fh; *fm2_io = fm2; *fh2_io = fh2;
    *fv_io = fv; *cm_o = cm; *ch_o = ch; *ch2_o = ch2;
}

// ---------------------------------------------------------------------------
// STOMATA
// ---------------------------------------------------------------------------
// parameter slots shared by STOMATA and VEGE_FLUX
#define PP_DLEAF  0
#define PP_HVT    1
#define PP_CBIOM  2
#define PP_C3PSN  3
#define PP_KC25   4
#define PP_AKC    5
#define PP_KO25   6
#define PP_AKO    7
#define PP_AVCMX  8
#define PP_VCMX25 9
#define PP_BP     10
#define PP_MP     11
#define PP_QE25   12
#define PP_FOLNMX 13

__device__ __forceinline__ float sf1(float ab, float bc)
{
    return glibc_powf(ab, FD(FS(bc, 25.0f), 10.0f));
}

__device__ __forceinline__ float sf2(float ab)
{
    float t = FA(ab, 273.16f);
    return FA(1.0f, glibc_expf(FD(FA(-2.2e05f, FM(710.0f, t)), FM(8.314f, t))));
}

__device__ void dev_stomata(const float *p, float mpe, float apar, float foln,
                            float tv, float ei, float ea, float sfctmp,
                            float sfcprs, float fveg, float o2, float co2,
                            float igs, float btran, float rb,
                            float *rs_o, float *psn_o)
{
    float c3 = p[PP_C3PSN];
    float apar_scale = FD(apar, fmxf(fveg, 1.0e-6f));
    float cf = FM(FD(sfcprs, FM(8.314f, sfctmp)), 1.0e06f);
    float rs = FM(FD(1.0f, p[PP_BP]), cf);
    float psn = 0.0f;

    if (apar_scale <= 0.0f) { *rs_o = rs; *psn_o = psn; return; }

    float fnf = fmnf(FD(foln, fmxf(mpe, p[PP_FOLNMX])), 1.0f);
    float tc = FS(tv, PH_TFRZ);
    float ppf = FM(4.6f, apar_scale);
    float j = FM(ppf, p[PP_QE25]);
    float kc = FM(p[PP_KC25], sf1(p[PP_AKC], tc));
    float ko = FM(p[PP_KO25], sf1(p[PP_AKO], tc));
    float awc = FM(kc, FA(1.0f, FD(o2, ko)));
    float cp = FM(FM(FD(FM(0.5f, kc), ko), o2), 0.21f);
    float vcmx = FM(FM(FM(FD(p[PP_VCMX25], sf2(tc)), fnf), btran), sf1(p[PP_AVCMX], tc));

    float ci = FA(FM(FM(0.7f, co2), c3), FM(FM(0.4f, co2), FS(1.0f, c3)));
    float rlb = FD(rb, cf);
    float cea = fmxf(FA(FM(FM(0.25f, ei), c3), FM(FM(0.40f, ei), FS(1.0f, c3))),
                     fmnf(ea, ei));

    for (int it = 0; it < 3; ++it) {
        float wj = FA(FM(FD(FM(fmxf(FS(ci, cp), 0.0f), j), FA(ci, FM(2.0f, cp))), c3),
                      FM(j, FS(1.0f, c3)));
        float wc = FA(FM(FD(FM(fmxf(FS(ci, cp), 0.0f), vcmx), FA(ci, awc)), c3),
                      FM(vcmx, FS(1.0f, c3)));
        float we = FA(FM(FM(0.5f, vcmx), c3),
                      FM(FD(FM(FM(4000.0f, vcmx), ci), sfcprs), FS(1.0f, c3)));
        psn = FM(fmnf(fmnf(wj, wc), we), igs);

        float cs = fmxf(FS(co2, FM(FM(FM(1.37f, rlb), sfcprs), psn)), mpe);
        float a = FA(FD(FM(FM(FM(p[PP_MP], psn), sfcprs), cea), FM(cs, ei)), p[PP_BP]);
        float b = FS(FM(FA(FD(FM(FM(p[PP_MP], psn), sfcprs), cs), p[PP_BP]), rlb), 1.0f);
        float c = -rlb;
        float disc = FSQ(FS(FM(b, b), FM(FM(4.0f, a), c)));
        float q = (b >= 0.0f) ? FM(-0.5f, FA(b, disc)) : FM(-0.5f, FS(b, disc));
        float r1 = FD(q, a);
        float r2 = FD(c, q);
        rs = fmxf(r1, r2);
        ci = fmxf(FS(cs, FM(FM(FM(psn, sfcprs), 1.65f), rs)), 0.0f);
    }

    *rs_o = FM(rs, cf);
    *psn_o = psn;
}

// ---------------------------------------------------------------------------
// VEGE_FLUX
// ---------------------------------------------------------------------------
// Float input slots (see gpuwm/core/kernels/README-vegeflux ordering mirrored in
// tools/noahmp_wrf461_oracle/validate_vegeflux_cuda.py::VF_IN).
enum {
    VI_DT = 0, VI_SAV, VI_SAG, VI_LWDN, VI_UR, VI_UU, VI_VV, VI_SFCTMP,
    VI_QAIR, VI_EAIR, VI_RHOAIR, VI_SNOWH, VI_VAI, VI_GAMMAV, VI_GAMMAG,
    VI_FWET, VI_LAISUN, VI_LAISHA, VI_CWP, VI_ZLVL, VI_ZPD, VI_Z0M, VI_FVEG,
    VI_Z0MG, VI_EMV, VI_EMG, VI_CANLIQ, VI_CANICE, VI_RSURF, VI_LATHEAV,
    VI_PARSUN, VI_PARSHA, VI_IGS, VI_FOLN, VI_CO2AIR, VI_O2AIR, VI_BTRAN,
    VI_SFCPRS, VI_RHSUR, VI_PAHV, VI_PAHG, VI_EAH, VI_TAH, VI_TV, VI_TG,
    VI_CM, VI_CH, VI_QSFC, VI_PSFC, VI_FSR, VI_NFLOAT
};
#define VF_NLAYER 7          /* indices -2 .. 4 for NSNOW=3, NSOIL=4 */
#define VF_LOFF   2          /* layer k lives at slot k + VF_LOFF     */

enum {
    VO_EAH = 0, VO_TAH, VO_TV, VO_TG, VO_CM, VO_CH, VO_TAUXV, VO_TAUYV,
    VO_IRG, VO_IRC, VO_SHG, VO_SHC, VO_EVG, VO_EVC, VO_TR, VO_GH, VO_T2MV,
    VO_PSNSUN, VO_PSNSHA, VO_CANHS, VO_QSFC, VO_Q2V, VO_CAH2, VO_CHLEAF,
    VO_CHUC, VO_RSSUN, VO_RSSHA, VO_SAV, VO_SAG, VO_FSR, VO_NOUT
};

__device__ __forceinline__ float tdc(float t)
{
    return fmnf(50.0f, fmxf(-50.0f, FS(t, PH_TFRZ)));
}

__device__ void dev_vege_flux(const float *in, const float *p, int isnow,
                              const float *dzsnso, const float *stc,
                              const float *df, float *out,
                              int opt_stc=1, float fsno=0.0f)
{
    const float dt = in[VI_DT], sav = in[VI_SAV], sag = in[VI_SAG];
    const float lwdn = in[VI_LWDN], ur = in[VI_UR], uu = in[VI_UU], vv = in[VI_VV];
    const float sfctmp = in[VI_SFCTMP], qair = in[VI_QAIR], eair = in[VI_EAIR];
    const float rhoair = in[VI_RHOAIR], snowh = in[VI_SNOWH], vai = in[VI_VAI];
    const float gammav = in[VI_GAMMAV], gammag = in[VI_GAMMAG], fwet = in[VI_FWET];
    const float laisun = in[VI_LAISUN], laisha = in[VI_LAISHA], cwp = in[VI_CWP];
    const float zlvl = in[VI_ZLVL], zpd = in[VI_ZPD], z0m = in[VI_Z0M];
    const float fveg = in[VI_FVEG], z0mg = in[VI_Z0MG], emv = in[VI_EMV];
    const float emg = in[VI_EMG], canliq = in[VI_CANLIQ], canice = in[VI_CANICE];
    const float rsurf = in[VI_RSURF], latheav = in[VI_LATHEAV];
    const float parsun = in[VI_PARSUN], parsha = in[VI_PARSHA], igs = in[VI_IGS];
    const float foln = in[VI_FOLN], co2air = in[VI_CO2AIR], o2air = in[VI_O2AIR];
    const float btran = in[VI_BTRAN], sfcprs = in[VI_SFCPRS], rhsur = in[VI_RHSUR];
    const float pahv = in[VI_PAHV], pahg = in[VI_PAHG];
    const float psfc = in[VI_PSFC];

    float eah = in[VI_EAH], tah = in[VI_TAH], tv = in[VI_TV], tg = in[VI_TG];
    float cm = in[VI_CM], ch = in[VI_CH], qsfc = in[VI_QSFC];

    const float mpe = 1.0e-6f;
    int liter = 0;
    float fv = 0.1f;
    float dtv = 0.0f, dtg = 0.0f;
    float moz = 0.0f, fm = 0.0f, fh = 0.0f, fm2 = 0.0f, fh2 = 0.0f;
    int mozsgn = 0;
    float hg = 0.0f, h = 0.0f;
    float mozg = 0.0f, fhg = 0.0f;
    float rssun = 0.0f, rssha = 0.0f, psnsun = 0.0f, psnsha = 0.0f;

    float vaie = fmnf(6.0f, vai);
    float laisune = fmnf(6.0f, laisun);
    float laishae = fmnf(6.0f, laisha);

    float esatw, esati, dsatw, dsati;
    float t = tdc(tg);
    dev_esat(t, &esatw, &esati, &dsatw, &dsati);
    float estg = (t > 0.0f) ? esatw : esati;

    qsfc = FD(FM(0.622f, eair), FS(psfc, FM(0.378f, eair)));

    float hcan = p[PP_HVT];
    float uc = FD(FM(ur, glibc_logf(FD(FA(FS(hcan, zpd), z0m), z0m))),
                  glibc_logf(FD(zlvl, z0m)));

    float air = FS(-FM(FM(emv, FA(1.0f, FM(FS(1.0f, emv), FS(1.0f, emg)))), lwdn),
                   FM(FM(FM(emv, emg), PH_SB), p4f(tg)));
    float cir = FM(FM(FS(2.0f, FM(emv, FS(1.0f, emg))), emv), PH_SB);

    float cah = 0.0f, cvh = 0.0f, rahg = 1.0f, rawg = 1.0f, z0h = z0m;
    float shc = 0.0f, evc = 0.0f, tr = 0.0f, irc = 0.0f, canhs = 0.0f;

    for (int it = 1; it <= 20; ++it) {
        z0h = z0m;
        float z0hg = z0mg;
        float ch2;
        dev_sfcdif1(it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur, mpe,
                    &moz, &mozsgn, &fm, &fh, &fm2, &fh2, &fv, &cm, &ch, &ch2);

        float rahc = fmxf(1.0f, FD(1.0f, FM(ch, ur)));
        float rawc = rahc;

        float ramg, rb;
        dev_ragrb(it, vaie, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc,
                  z0h, fv, cwp, mpe, p[PP_DLEAF], &mozg, &fhg,
                  &ramg, &rahg, &rawg, &rb);

        t = tdc(tv);
        dev_esat(t, &esatw, &esati, &dsatw, &dsati);
        float estv = (t > 0.0f) ? esatw : esati;
        float destv = (t > 0.0f) ? dsatw : dsati;

        if (it == 1) {
            dev_stomata(p, mpe, parsun, foln, tv, estv, eah, sfctmp, sfcprs,
                        fveg, o2air, co2air, igs, btran, rb, &rssun, &psnsun);
            dev_stomata(p, mpe, parsha, foln, tv, estv, eah, sfctmp, sfcprs,
                        fveg, o2air, co2air, igs, btran, rb, &rssha, &psnsha);
        }

        cah = FD(1.0f, rahc);
        cvh = FD(FM(2.0f, vaie), rb);
        float cgh = FD(1.0f, rahg);
        float cond = FA(FA(cah, cvh), cgh);
        float ata = FD(FA(FM(sfctmp, cah), FM(tg, cgh)), cond);
        float bta = FD(cvh, cond);
        float csh = FM(FM(FM(FS(1.0f, bta), rhoair), PH_CPAIR), cvh);

        float caw = FD(1.0f, rawc);
        float cew = FD(FM(fwet, vaie), rb);
        float ctw = FM(FS(1.0f, fwet),
                       FA(FD(laisune, FA(rb, rssun)), FD(laishae, FA(rb, rssha))));
        float cgw = FD(1.0f, FA(rawg, rsurf));
        cond = FA(FA(FA(caw, cew), ctw), cgw);
        float aea = FD(FA(FM(eair, caw), FM(estg, cgw)), cond);
        float bea = FD(FA(cew, ctw), cond);
        float cev = FD(FM(FM(FM(FS(1.0f, bea), cew), rhoair), PH_CPAIR), gammav);
        float ctr = FD(FM(FM(FM(FS(1.0f, bea), ctw), rhoair), PH_CPAIR), gammav);

        tah = FA(ata, FM(bta, tv));
        eah = FA(aea, FM(bea, estv));

        irc = FM(fveg, FA(air, FM(cir, p4f(tv))));
        shc = FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), cvh), FS(tv, tah));
        evc = FD(FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), cew), FS(estv, eah)), gammav);
        tr = FD(FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), ctw), FS(estv, eah)), gammav);
        if (tv > PH_TFRZ) evc = fmnf(FD(FM(canliq, latheav), dt), evc);
        else              evc = fmnf(FD(FM(canice, latheav), dt), evc);

        float hcv = FA(FA(FM(FM(p[PP_CBIOM], vaie), PH_CWAT),
                          FD(FM(canliq, PH_CWAT), PH_DENH2O)),
                       FD(FM(canice, PH_CICE), PH_DENICE));

        float b = FA(FS(FS(FS(FS(sav, irc), shc), evc), tr), pahv);
        float a = FM(fveg, FA(FA(FA(FM(FM(4.0f, cir), p3f(tv)), csh),
                                 FM(FA(cev, ctr), destv)), FD(hcv, dt)));
        dtv = FD(b, a);

        irc = FA(irc, FM(FM(FM(FM(fveg, 4.0f), cir), p3f(tv)), dtv));
        shc = FA(shc, FM(FM(fveg, csh), dtv));
        evc = FA(evc, FM(FM(FM(fveg, cev), destv), dtv));
        tr = FA(tr, FM(FM(FM(fveg, ctr), destv), dtv));
        canhs = FD(FM(FM(dtv, fveg), hcv), dt);

        tv = FA(tv, dtv);

        h = FD(FM(FM(rhoair, PH_CPAIR), FS(tah, sfctmp)), rahc);
        hg = FD(FM(FM(rhoair, PH_CPAIR), FS(tg, tah)), rahg);

        qsfc = FD(FM(0.622f, eah), FS(sfcprs, FM(0.378f, eah)));

        if (liter == 1) break;
        if (it >= 5 && fabsf(dtv) <= 0.01f && liter == 0) liter = 1;
    }

    air = FS(-FM(FM(emg, FS(1.0f, emv)), lwdn), FM(FM(FM(emg, emv), PH_SB), p4f(tv)));
    cir = FM(emg, PH_SB);
    float csh = FD(FM(rhoair, PH_CPAIR), rahg);
    float cev = FD(FM(rhoair, PH_CPAIR), FM(gammag, FA(rawg, rsurf)));
    float cgh = FD(FM(2.0f, df[isnow + 1 + VF_LOFF]), dzsnso[isnow + 1 + VF_LOFF]);

    float irg = 0.0f, shg = 0.0f, evg = 0.0f, gh = 0.0f, destg = 0.0f;
    estg = 0.0f;
    for (int it = 0; it < 5; ++it) {
        t = tdc(tg);
        dev_esat(t, &esatw, &esati, &dsatw, &dsati);
        estg = (t > 0.0f) ? esatw : esati;
        destg = (t > 0.0f) ? dsatw : dsati;

        irg = FA(FM(cir, p4f(tg)), air);
        shg = FM(csh, FS(tg, tah));
        evg = FM(cev, FS(FM(estg, rhsur), eah));
        gh = FM(cgh, FS(tg, stc[isnow + 1 + VF_LOFF]));

        float b = FA(FS(FS(FS(FS(sag, irg), shg), evg), gh), pahg);
        float a = FA(FA(FA(FM(FM(4.0f, cir), p3f(tg)), csh), FM(cev, destg)), cgh);
        dtg = FD(b, a);

        irg = FA(irg, FM(FM(FM(4.0f, cir), p3f(tg)), dtg));
        shg = FA(shg, FM(csh, dtg));
        evg = FA(evg, FM(FM(cev, destg), dtg));
        gh = FA(gh, FM(cgh, dtg));
        tg = FA(tg, dtg);
    }

    // WRF-lineage Noah-MP 848f54ad :4125-4136, STC=1 reset / STC=3 blend.
    // Original notice: licenses/LICENSE-WRF-public-domain.txt. No Noah-MP 5.
    if (snowh > 0.05f && tg > PH_TFRZ) {
        tg = opt_stc == 3
            ? FA(FM(FS(1.0f, fsno), tg), FM(fsno, PH_TFRZ)) : PH_TFRZ;
        // Correct-mode SN1 (gpuwm/correct_mode.py): EVG at the reset ground
        // temperature. The WRF land reset keeps ESTG from the discarded,
        // warmer TG, so a sunlit melting snowpack evaporates as if warm and
        // its 2 m mixing ratio goes to zero or below. WRF's own glacier
        // reset (:1092-1105, kernels/noahmp_glacier.cu) re-evaluates it.
#ifdef GPUWM_NOAHMP_MELT_RESET_ESAT
        t = tdc(tg);
        dev_esat(t, &esatw, &esati, &dsatw, &dsati);
        estg = (t > 0.0f) ? esatw : esati;
#endif
        irg = FS(FS(FM(cir, p4f(tg)), FM(FM(emg, FS(1.0f, emv)), lwdn)),
                 FM(FM(FM(emg, emv), PH_SB), p4f(tv)));
        shg = FM(csh, FS(tg, tah));
        evg = FM(cev, FS(FM(estg, rhsur), eah));
        gh = FS(FA(sag, pahg), FA(FA(irg, shg), evg));
    }

    float tauxv = -FM(FM(FM(rhoair, cm), ur), uu);
    float tauyv = -FM(FM(FM(rhoair, cm), ur), vv);

    float cah2 = FD(FM(fv, PH_VKC), FS(glibc_logf(FD(FA(2.0f, z0h), z0h)), fh2));
    float cq2v = cah2;
    float t2mv, q2v;
    if (cah2 < 1.0e-5f) {
        t2mv = tah;
        q2v = qsfc;
    } else {
        t2mv = FS(tah, FD(FM(FD(FA(shg, FD(shc, fveg)), FM(rhoair, PH_CPAIR)), 1.0f),
                          cah2));
        q2v = FS(qsfc, FD(FM(FD(FA(FD(FA(evc, tr), fveg), evg),
                                FM(latheav, rhoair)), 1.0f), cq2v));
    }

    out[VO_EAH] = eah;   out[VO_TAH] = tah;   out[VO_TV] = tv;   out[VO_TG] = tg;
    out[VO_CM] = cm;     out[VO_CH] = cah;
    out[VO_TAUXV] = tauxv; out[VO_TAUYV] = tauyv;
    out[VO_IRG] = irg;   out[VO_IRC] = irc;   out[VO_SHG] = shg; out[VO_SHC] = shc;
    out[VO_EVG] = evg;   out[VO_EVC] = evc;   out[VO_TR] = tr;   out[VO_GH] = gh;
    out[VO_T2MV] = t2mv; out[VO_PSNSUN] = psnsun; out[VO_PSNSHA] = psnsha;
    out[VO_CANHS] = canhs; out[VO_QSFC] = qsfc; out[VO_Q2V] = q2v;
    out[VO_CAH2] = cah2; out[VO_CHLEAF] = cvh; out[VO_CHUC] = FD(1.0f, rahg);
    out[VO_RSSUN] = rssun; out[VO_RSSHA] = rssha;
    out[VO_SAV] = sav;   out[VO_SAG] = sag;   out[VO_FSR] = in[VI_FSR];
    (void)dtg;
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
extern "C" __global__
void k_esat(const float *tin, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a, b, c, d;
    dev_esat(tin[i], &a, &b, &c, &d);
    out[4 * i + 0] = a; out[4 * i + 1] = b; out[4 * i + 2] = c; out[4 * i + 3] = d;
}

// in stride 17: VAI RHOAIR HG TAH ZPD Z0MG Z0HG HCAN UC Z0H FV CWP MPE TV MOZG FHG DLEAF
extern "C" __global__
void k_ragrb(const float *in, const int *iter, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *v = in + 17 * i;
    float mozg = v[14], fhg = v[15], ramg, rahg, rawg, rb;
    dev_ragrb(iter[i], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
              v[9], v[10], v[11], v[12], v[16], &mozg, &fhg,
              &ramg, &rahg, &rawg, &rb);
    float *o = out + 6 * i;
    o[0] = mozg; o[1] = fhg; o[2] = ramg; o[3] = rahg; o[4] = rawg; o[5] = rb;
}

// in stride 16: SFCTMP RHOAIR H QAIR ZLVL ZPD Z0M Z0H UR MPE MOZ FM FH FM2 FH2 FV
// iin stride 2: ITER MOZSGN
extern "C" __global__
void k_sfcdif1(const float *in, const int *iin, float *out, int *iout, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *v = in + 16 * i;
    float moz = v[10], fm = v[11], fh = v[12], fm2 = v[13], fh2 = v[14], fv = v[15];
    int mozsgn = iin[2 * i + 1];
    float cm, ch, ch2;
    dev_sfcdif1(iin[2 * i], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                v[9], &moz, &mozsgn, &fm, &fh, &fm2, &fh2, &fv, &cm, &ch, &ch2);
    float *o = out + 9 * i;
    o[0] = moz; o[1] = fm; o[2] = fh; o[3] = fm2; o[4] = fh2;
    o[5] = cm; o[6] = ch; o[7] = fv; o[8] = ch2;
    iout[i] = mozsgn;
}

// in stride 14: MPE APAR FOLN TV EI EA SFCTMP SFCPRS FVEG O2 CO2 IGS BTRAN RB
// p  stride 14: the PP_* slots
extern "C" __global__
void k_stomata(const float *in, const float *pin, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *v = in + 14 * i;
    float rs, psn;
    dev_stomata(pin + 14 * i, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                v[8], v[9], v[10], v[11], v[12], v[13], &rs, &psn);
    out[2 * i + 0] = rs; out[2 * i + 1] = psn;
}

extern "C" __global__
void k_vegeflux(const float *in, const float *pin, const int *isnow,
                const float *dzsnso, const float *stc, const float *df,
                float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dev_vege_flux(in + VI_NFLOAT * i, pin + 14 * i, isnow[i],
                  dzsnso + VF_NLAYER * i, stc + VF_NLAYER * i,
                  df + VF_NLAYER * i, out + VO_NOUT * i);
}

// Existing row layout plus actual FSNO[n], omitted by the old STC=1 packing.
extern "C" __global__
void k_vegeflux_stc3(const float *in, const float *pin, const int *isnow,
                     const float *dzsnso, const float *stc, const float *df,
                     const float *fsno, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dev_vege_flux(in + VI_NFLOAT * i, pin + 14 * i, isnow[i],
                  dzsnso + VF_NLAYER * i, stc + VF_NLAYER * i,
                  df + VF_NLAYER * i, out + VO_NOUT * i, 3, fsno[i]);
}

// Fixed-width dispatch. Inactive columns never enter the physics leaf.
extern "C" __global__
void k_vegeflux_masked(const float *in, const float *pin, const int *isnow,
                       const float *dzsnso, const float *stc, const float *df,
                       float *out, int n, const unsigned char *active)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!active[i]) {
        for (int j = 0; j < VO_NOUT; ++j) out[VO_NOUT * i + j] = 0.0f;
        return;
    }
    dev_vege_flux(in + VI_NFLOAT * i, pin + 14 * i, isnow[i],
                  dzsnso + VF_NLAYER * i, stc + VF_NLAYER * i,
                  df + VF_NLAYER * i, out + VO_NOUT * i);
}
